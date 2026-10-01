"""Leakage-resistant evaluation orchestration and benchmark metrics.

System callbacks receive only the runtime claim and corpus-backed retriever.
Scoring annotations are joined after every system has produced its output.
"""
from __future__ import annotations

import json
import math
import statistics
import time
import traceback as traceback_module
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from evaluation.experiments.phase10e_infrastructure import OutputState

from evaluation.scifact.metrics import retrieval_metrics_for_claim
from evaluation.scifact.scifact_adapter import (
    ClaimAnnotations, ScifactClaimRecord, load_scifact_dataset, runtime_claim_payload,
)
from .experiment_config import ExperimentConfig

LABELS = ("SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE")
SYSTEM_NAMES = ("A", "B", "C")


@dataclass
class SystemOutput:
    """System response; fields absent from a system stay null/empty."""
    predicted_label: str | None = None
    retrieved_chunks: list[dict[str, Any]] = field(default_factory=list)
    selected_evidence_doc_ids: list[int] = field(default_factory=list)
    assessments: list[dict[str, Any]] = field(default_factory=list)
    unsupported_claims: int | None = None
    verification_calls: int | None = None
    llm_calls: int | None = None
    retrieval_calls: int | None = None
    controller_cycles: int | None = None
    raw_output: Any = None
    error: str | None = None
    status: str | None = None
    error_type: str | None = None
    error_stage: str | None = None
    error_traceback: str | None = None
    retrieval_output_status: str | None = None
    evidence_output_status: str | None = None


class SystemExecutionError(RuntimeError):
    """A callback failure that still carries safely observed partial output."""
    def __init__(self, message: str, partial_output: SystemOutput):
        super().__init__(message)
        self.partial_output = partial_output


System = Callable[[dict[str, Any], Any, ExperimentConfig], SystemOutput]


def gold_claim_label(annotations: ClaimAnnotations) -> tuple[str | None, str]:
    """Return an unambiguous label only when annotations justify it.

    SciFact's split files store evidence-level rationale stances rather than a
    top-level claim label. Empty evidence and opposing rationale stances are
    deliberately left unscored instead of silently mapped to NEI.
    """
    if not annotations.labels_available or annotations.evidence is None:
        return None, "gold_label_unavailable"
    labels = {rationale.label for document in annotations.evidence
              for rationale in document.rationales}
    if not labels:
        return None, "no_rationale_label; NEI mapping not asserted"
    if len(labels) != 1:
        return None, "conflicting_rationale_labels"
    label = next(iter(labels))
    if label == "SUPPORT":
        return "SUPPORTED", "unanimous gold rationale stance"
    if label == "CONTRADICT":
        return "CONTRADICTED", "unanimous gold rationale stance"
    return None, "unsupported gold rationale label"


def _safe_call(system: System, runtime: dict[str, Any], provider: Any,
               config: ExperimentConfig) -> tuple[SystemOutput, float]:
    start = time.perf_counter()
    try:
        output = system(dict(runtime), provider, config)
        if not isinstance(output, SystemOutput):
            raise TypeError("system callback must return SystemOutput")
    except Exception as exc:  # Keep failures visible in paired results.
        partial = getattr(exc, "partial_output", None)
        if isinstance(partial, SystemOutput):
            output = partial
            output.error = f"{type(exc).__name__}: {exc}"
            output.status = "TIMEOUT" if isinstance(exc, TimeoutError) else "FAILED"
            output.error_type = type(exc).__name__
            output.error_stage = output.error_stage or "system_callback"
            output.error_traceback = traceback_module.format_exc()
        else:
            output = SystemOutput(
                error=f"{type(exc).__name__}: {exc}",
                status="TIMEOUT" if isinstance(exc, TimeoutError) else "FAILED",
                error_type=type(exc).__name__, error_stage="system_callback",
                error_traceback=traceback_module.format_exc(),
            )
    if output.error and output.status is None:
        output.error_type = output.error_type or output.error.split(":", 1)[0]
        output.status = "TIMEOUT" if output.error_type == "TimeoutError" else "FAILED"
        output.error_stage = output.error_stage or "system_callback"
    if not output.error and output.status is None:
        output.status = "SUCCESS"
    # Adapters should set these fields when they know whether the stage ran.
    # The fallback deliberately says unavailable rather than inferring empty.
    if output.retrieval_output_status is None:
        output.retrieval_output_status = (
            OutputState.EXECUTION_FAILED.value if output.status != "SUCCESS"
            else OutputState.VALID_NONEMPTY.value if output.retrieved_chunks
            else OutputState.OUTPUT_UNAVAILABLE.value
        )
    if output.evidence_output_status is None and output.status != "SUCCESS":
        output.evidence_output_status = OutputState.EXECUTION_FAILED.value
    return output, time.perf_counter() - start


def _prf(tp: int, fp: int, fn: int) -> dict[str, float | None]:
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall is not None and precision + recall else None)
    return {"precision": precision, "recall": recall, "f1": f1}


def _attribution_metrics(annotations: ClaimAnnotations, selected_doc_ids: Sequence[int],
                        retrieved_chunks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Score citations at document level and retrieval at rationale-sentence level."""
    if not annotations.labels_available or annotations.evidence is None:
        return {"status": "gold_annotations_unavailable", "citation": None,
                "retrieved_rationale_sentence_recall": None}
    gold_docs = {document.doc_id for document in annotations.evidence}
    if not gold_docs:
        return {"status": "no_annotated_gold_evidence", "citation": None,
                "retrieved_rationale_sentence_recall": None}
    selected = set(selected_doc_ids)
    citation = _prf(len(selected & gold_docs), len(selected - gold_docs), len(gold_docs - selected))
    gold_sentences = {(document.doc_id, sentence)
                      for document in annotations.evidence
                      for rationale in document.rationales
                      for sentence in rationale.sentence_indices}
    retrieved_sentences = set()
    for chunk in retrieved_chunks:
        try:
            doc_id = int(chunk.get("scifact_doc_id", chunk.get("document_id")))
        except (TypeError, ValueError):
            continue
        for sentence in chunk.get("scifact_sentence_indices", []):
            if isinstance(sentence, int) and not isinstance(sentence, bool):
                retrieved_sentences.add((doc_id, sentence))
    overlap = gold_sentences & retrieved_sentences
    return {"status": "scored", "citation": citation,
            "retrieved_rationale_sentence_recall": len(overlap) / len(gold_sentences) if gold_sentences else None,
            "gold_rationale_sentence_count": len(gold_sentences),
            "retrieved_gold_rationale_sentence_count": len(overlap)}


def _classification(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    scored = [r for r in records if r["gold_label"] and r["predicted_label"] in LABELS]
    correct = sum(r["gold_label"] == r["predicted_label"] for r in scored)
    per_class = {}
    for label in LABELS:
        tp = sum(r["gold_label"] == label and r["predicted_label"] == label for r in scored)
        fp = sum(r["gold_label"] != label and r["predicted_label"] == label for r in scored)
        fn = sum(r["gold_label"] == label and r["predicted_label"] != label for r in scored)
        per_class[label] = _prf(tp, fp, fn)
    f1s = [v["f1"] for v in per_class.values() if v["f1"] is not None]
    return {
        "scorable_examples": len(scored),
        "accuracy": correct / len(scored) if scored else None,
        "macro_f1": sum(f1s) / len(f1s) if f1s else None,
        "per_class": per_class,
        "confusion_matrix": {gold: {pred: sum(r["gold_label"] == gold and r["predicted_label"] == pred
                                                   for r in scored) for pred in LABELS}
                             for gold in LABELS},
    }


def _aggregate(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    retrieval_rows = [r["retrieval"] for r in records
                      if r["retrieval"].get("retrieval_scoring_status") == "scored"]
    gold_total = sum(r["gold_evidence_document_count"] for r in retrieval_rows)
    recall = {}
    precision = {}
    for k in (1, 5, 10, 20):
        recall_hits = sum(round(r["recall_at_k"].get(str(k), 0) * r["gold_evidence_document_count"])
                          for r in retrieval_rows)
        recall[str(k)] = recall_hits / gold_total if gold_total else None
        # Doc-level precision@K uses unique docs and is undefined when no docs returned.
        vals = []
        for record in records:
            gold = set(record["gold_evidence_doc_ids"] or [])
            chunks = record["retrieved_chunks"][:k]
            doc_ids = []
            for chunk in chunks:
                value = chunk.get("scifact_doc_id", chunk.get("document_id"))
                try:
                    if value is not None:
                        doc_ids.append(int(value))
                except (TypeError, ValueError):
                    pass
            unique = set(doc_ids)
            if gold and unique:
                vals.append(len(gold & unique) / len(unique))
        precision[str(k)] = sum(vals) / len(vals) if vals else None
    elapsed = [r["latency_seconds"] for r in records if r["latency_seconds"] is not None]
    supported = [r for r in records if r["predicted_label"] == "SUPPORTED"]
    unsupported = [r["unsupported_claims"] for r in supported
                   if r["unsupported_claims"] is not None]
    citation_scores = [r["citation"] for r in records if r["citation"] is not None]
    sentence_recalls = [r["attribution"]["retrieved_rationale_sentence_recall"] for r in records
                        if r["attribution"]["retrieved_rationale_sentence_recall"] is not None]
    return {
        **_classification(records),
        "examples": len(records), "failures": sum(bool(r["error"]) for r in records),
        "retrieval_scored_examples": len(retrieval_rows), "gold_document_recall_at_k": recall,
        "mean_reciprocal_rank": (statistics.mean(
            row["reciprocal_rank"] for row in retrieval_rows if row.get("reciprocal_rank") is not None
        ) if any(row.get("reciprocal_rank") is not None for row in retrieval_rows) else None),
        "gold_document_precision_at_k_macro": precision,
        "unsupported_claim_rate": (sum(unsupported) / len(unsupported) if unsupported else None),
        "unsupported_claim_rate_denominator": len(unsupported),
        "selected_evidence_citation_macro": {
            key: (statistics.mean(item[key] for item in citation_scores if item[key] is not None)
                  if any(item[key] is not None for item in citation_scores) else None)
            for key in ("precision", "recall", "f1")
        },
        "retrieved_rationale_sentence_recall_macro": statistics.mean(sentence_recalls) if sentence_recalls else None,
        "mean_latency_seconds": statistics.mean(elapsed) if elapsed else None,
        "median_latency_seconds": statistics.median(elapsed) if elapsed else None,
        "retrieval_calls": sum(r["retrieval_calls"] or 0 for r in records),
        "verification_calls": sum(r["verification_calls"] or 0 for r in records),
        "llm_calls": sum(r["llm_calls"] or 0 for r in records),
        "controller_cycles": sum(r["controller_cycles"] or 0 for r in records),
    }


def run_experiment(config: ExperimentConfig, provider: Any, systems: Mapping[str, System],
                   output_dir: str | Path) -> dict[str, Any]:
    """Run A/B/C on the same ordered examples and write JSON/JSONL artifacts.

    `provider` must be corpus-only (e.g. ExistingRAGProvider). The callbacks
    cannot receive the ScifactClaimRecord or its scoring annotations.
    """
    if set(systems) != set(SYSTEM_NAMES):
        raise ValueError("systems must contain exactly A, B, and C")
    dataset = load_scifact_dataset(config.dataset_root, config.split)
    claims = list(dataset.claims)
    if config.example_ids is not None:
        requested = set(config.example_ids)
        by_id = {claim.runtime.claim_id: claim for claim in claims}
        missing = requested - set(by_id)
        if missing:
            raise ValueError(f"Configured example IDs are not in split {config.split}: {sorted(missing)}")
        claims = [by_id[item] for item in config.example_ids]
    elif config.example_limit is not None:
        claims = claims[:config.example_limit]
    out = Path(output_dir).expanduser().resolve()
    paths = [out / "run_manifest.json", out / "per_example.jsonl", out / "summary.json"]
    if any(path.exists() for path in paths):
        raise FileExistsError("Refusing to overwrite Phase 10 run files")
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "phase": 10, "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": config.to_dict(), "dataset_split": config.split,
        "selected_example_ids": [claim.runtime.claim_id for claim in claims],
        "dataset_claim_count": len(dataset.claims), "evaluated_claim_count": len(claims),
        "systems": ["A", "B", "C"], "same_claim_order_and_corpus": True,
        "gold_annotation_data_passed_to_runtime": False,
        "test_threshold_tuning": False,
        "label_mapping": "unanimous rationale stances only; empty/conflicting rationales unscored",
        "model": {"name": config.model, "temperature": config.temperature,
                  "version": "NOT AVAILABLE from Ollama tags"},
        "retrieval": {"top_k": config.top_k, "hybrid": config.hybrid_retrieval,
                      "reranking": config.reranking},
        "threshold_source": config.threshold_source,
        "seed": config.seed,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    all_records: list[dict[str, Any]] = []
    aggregates: dict[str, list[dict[str, Any]]] = {name: [] for name in SYSTEM_NAMES}
    with (out / "per_example.jsonl").open("w", encoding="utf-8") as stream:
        for claim in claims:
            # Runtime data shape is deliberately constructed without annotations.
            runtime = runtime_claim_payload(claim.runtime)
            per_system = {}
            for name in SYSTEM_NAMES:
                print(f"Phase 10: claim {claim.runtime.claim_id}, system {name} started", flush=True)
                output, elapsed = _safe_call(systems[name], runtime, provider, config)
                print(f"Phase 10: claim {claim.runtime.claim_id}, system {name} finished in {elapsed:.2f}s", flush=True)
                gold_label, label_status = gold_claim_label(claim.annotations)
                annotation = claim.annotations
                gold_docs = sorted({doc.doc_id for doc in annotation.evidence or ()}) if annotation.evidence is not None else None
                retrieval_valid = output.retrieval_output_status in {
                    OutputState.VALID_EMPTY.value, OutputState.VALID_NONEMPTY.value}
                retrieval = (retrieval_metrics_for_claim(annotation, output.retrieved_chunks)
                    if retrieval_valid else {"retrieval_scoring_status": output.retrieval_output_status,
                        "reciprocal_rank": None, "recall_at_k": {},
                        "gold_evidence_document_count": len(gold_docs or []),
                        "gold_evidence_documents_retrieved": None})
                attribution = _attribution_metrics(annotation, output.selected_evidence_doc_ids,
                                                   output.retrieved_chunks)
                if output.evidence_output_status not in {
                        OutputState.VALID_EMPTY.value, OutputState.VALID_NONEMPTY.value}:
                    attribution["citation"] = None
                    attribution["status"] = "evidence_output_" + str(output.evidence_output_status or "unavailable").lower()
                # Deterministic unsupportedness definition: a supported
                # prediction that cites no gold rationale document.
                unsupported = None
                if output.predicted_label == "SUPPORTED" and gold_docs:
                    unsupported = int(not (set(output.selected_evidence_doc_ids) & set(gold_docs)))
                record = {
                    "example_id": claim.runtime.claim_id, "split": config.split,
                    "system": name, "runtime_input": runtime,
                    "gold_label": gold_label, "gold_label_status": label_status,
                    "gold_evidence_doc_ids": gold_docs,
                    "retrieval_output_status": output.retrieval_output_status,
                    "evidence_output_status": output.evidence_output_status,
                    "retrieved_chunks": output.retrieved_chunks,
                    "selected_evidence_doc_ids": output.selected_evidence_doc_ids,
                    "predicted_label": output.predicted_label,
                    "assessments": output.assessments, "retrieval": retrieval,
                    "attribution": attribution, "citation": attribution["citation"],
                    "unsupported_claims": unsupported,
                    "latency_seconds": elapsed, "retrieval_calls": output.retrieval_calls,
                    "verification_calls": output.verification_calls, "llm_calls": output.llm_calls,
                    "controller_cycles": output.controller_cycles,
                    "status": output.status, "error": output.error,
                    "error_type": output.error_type, "error_stage": output.error_stage,
                    "error_traceback": output.error_traceback,
                    "raw_output": output.raw_output,
                }
                all_records.append(record)
                aggregates[name].append(record)
                per_system[name] = {key: record[key] for key in (
                    "predicted_label", "retrieved_chunks", "selected_evidence_doc_ids", "assessments",
                    "attribution", "citation", "unsupported_claims", "latency_seconds", "retrieval_calls", "verification_calls",
                    "llm_calls", "controller_cycles", "status", "error", "error_type",
                    "error_stage", "error_traceback", "raw_output", "retrieval")}
            # One paired row per example retains shared gold only in scoring output.
            row = {"example_id": claim.runtime.claim_id, "split": config.split,
                   "runtime_input": runtime, "scoring": {"gold_label": gold_claim_label(claim.annotations)[0],
                   "gold_label_status": gold_claim_label(claim.annotations)[1],
                   "gold_evidence_doc_ids": sorted({d.doc_id for d in claim.annotations.evidence or ()})
                   if claim.annotations.evidence is not None else None}, "systems": per_system}
            for name, system_record in per_system.items():
                system_record.update({
                    "example_id": claim.runtime.claim_id,
                    "input_claim": runtime["claim"],
                    "gold_label": row["scoring"]["gold_label"],
                    "evidence": system_record["retrieved_chunks"],
                    "verification_status": (system_record["predicted_label"]
                                             if system_record["verification_calls"] else "NOT_APPLICABLE"),
                    "final_result": (system_record["predicted_label"]
                                     if system_record["verification_calls"] else
                                     (system_record["raw_output"] or {}).get("raw_output")
                                     if isinstance(system_record["raw_output"], dict) else
                                     system_record["raw_output"]),
                    "provenance": {
                        "retrieved_documents": [{key: item.get(key) for key in
                            ("rank", "evidence_id", "chunk_id", "scifact_doc_id")}
                            for item in system_record["retrieved_chunks"]],
                        "selected_document_ids": system_record["selected_evidence_doc_ids"],
                        "controller_evidence": ((system_record["raw_output"] or {})
                            .get("research_state", {}).get("evidence", [])
                            if isinstance(system_record["raw_output"], dict) else []),
                    },
                })
            stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            stream.flush()
    summary = {name: _aggregate(aggregates[name]) for name in SYSTEM_NAMES}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"manifest": manifest, "summary": summary, "records": len(claims),
            "full_benchmark_completed": config.split == "test" and len(claims) == 300}
