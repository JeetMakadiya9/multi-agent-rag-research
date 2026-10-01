"""Phase 10C SciFact DEV evaluation with durable gold-free execution records."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import statistics
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.experiments.evaluator import (  # noqa: E402
    LABELS, SYSTEM_NAMES, SystemOutput, _attribution_metrics, _safe_call, gold_claim_label,
)
from evaluation.experiments.experiment_config import ExperimentConfig  # noqa: E402
from evaluation.experiments.phase10e_infrastructure import (  # noqa: E402
    OutputState, append_jsonl, atomic_write_json, attempt_id_key, exclusive_run_lock,
    claim_system_key, make_configuration_lock, recover_jsonl, retrieval_availability_summary,
    interrupted_attempt_record, load_resume_manifest, resume_pending_pairs,
    validate_attempt_record, validate_resume_configuration,
    validate_scored_record,
)
from evaluation.experiments.system_a_rag import run_system_a  # noqa: E402
from evaluation.experiments.system_b_verified_rag import run_system_b  # noqa: E402
from evaluation.experiments.system_c_multi_agent import run_system_c  # noqa: E402
from evaluation.scifact.metrics import retrieval_metrics_for_claim  # noqa: E402
from evaluation.scifact.scifact_adapter import (  # noqa: E402
    corpus_to_rag_chunks, load_scifact_dataset, runtime_claim_payload,
)

FROZEN_FILES = (
    "rag.py", "claim_verification.py", "experiment_b_verification.py",
    "research_capability_dispatch.py", "research_continuation.py",
    "research_autonomous_controller.py", "research_autonomous_execution.py",
    "research_verification_capability.py",
)
SYSTEMS = {"A": run_system_a, "B": run_system_b, "C": run_system_c}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_stratified_dev_ids(dataset: Any, count: int) -> list[int]:
    """Choose a proportional, deterministic sample from annotated and empty strata."""
    if count < 1:
        raise ValueError("count must be positive")
    if count > len(dataset.claims):
        raise ValueError("count exceeds the dev split")
    by_label: dict[str, list[int]] = {"SUPPORTED": [], "CONTRADICTED": [], "UNLABELED": []}
    for row in dataset.claims:
        label, _ = gold_claim_label(row.annotations)
        by_label[label or "UNLABELED"].append(row.runtime.claim_id)
    labels = ("SUPPORTED", "CONTRADICTED", "UNLABELED")
    exact = {label: count * len(by_label[label]) / len(dataset.claims) for label in labels}
    quotas = {label: int(exact[label]) for label in labels}
    remainder = count - sum(quotas.values())
    for label in sorted(labels, key=lambda value: (-(exact[value] - quotas[value]), labels.index(value)))[:remainder]:
        quotas[label] += 1
    chosen: set[int] = set()
    for label in labels:
        population = by_label[label]
        sample_count = quotas[label]
        if sample_count == 0:
            continue
        # Midpoint spacing spreads each stratum across its dev-file order.
        positions = [min(len(population) - 1, int((i + 0.5) * len(population) / sample_count))
                     for i in range(sample_count)]
        chosen.update(population[position] for position in positions)
    return [row.runtime.claim_id for row in dataset.claims if row.runtime.claim_id in chosen]


def _json_safe(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _json_safe(value.to_dict())
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _classification(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    labeled = [r for r in records if r["gold_label"] in LABELS]
    valid = [r for r in labeled if r["predicted_label"] in LABELS]
    correct = sum(r["gold_label"] == r["predicted_label"] for r in valid)
    # Invalid or failed outputs remain in the primary labeled denominator and
    # are counted as incorrect; valid-output accuracy is reported separately.
    matrix = {gold: {pred: 0 for pred in (*LABELS, "INVALID_OR_FAILED")} for gold in LABELS}
    for row in labeled:
        pred = row["predicted_label"] if row["predicted_label"] in LABELS else "INVALID_OR_FAILED"
        matrix[row["gold_label"]][pred] += 1
    per_class: dict[str, Any] = {}
    for label in LABELS:
        tp = matrix[label][label]
        fp = sum(matrix[g][label] for g in LABELS if g != label)
        fn = sum(matrix[label][p] for p in (*LABELS, "INVALID_OR_FAILED") if p != label)
        p_den, r_den = tp + fp, tp + fn
        precision = tp / p_den if p_den else None
        recall = tp / r_den if r_den else None
        f1 = (2 * precision * recall / (precision + recall)
              if precision is not None and recall is not None and precision + recall else
              0.0 if precision == 0 or recall == 0 else None)
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1,
                            "gold_support": r_den, "predicted_support": p_den}
    supported_f1 = [per_class[label]["f1"] for label in LABELS
                    if per_class[label]["gold_support"] > 0]
    return {
        "status": "scored" if labeled else "NOT COMPUTED",
        "accuracy": correct / len(labeled) if labeled else None,
        "valid_prediction_accuracy": correct / len(valid) if valid else None,
        "correct": correct, "denominator": len(labeled),
        "valid_prediction_denominator": len(valid),
        "invalid_or_failed_predictions": len(labeled) - len(valid),
        "macro_f1_over_gold_supported_classes": (
            sum(supported_f1) / len(supported_f1) if supported_f1 else None),
        "macro_f1_scope": "mean over SUPPORT and CONTRADICT classes present in gold denominator; NEI has no SciFact gold class",
        "per_class": per_class, "confusion_matrix": matrix,
        "unscorable_gold_examples": sum(r["gold_label"] is None for r in records),
    }


def _system_metrics(records: Sequence[dict[str, Any]], system: str) -> dict[str, Any]:
    availability = retrieval_availability_summary(records)
    retrieval = [r["retrieval"] for r in records if r["retrieval"]["retrieval_scoring_status"] == "scored"]
    gold_docs = sum(r["retrieval"]["gold_evidence_document_count"] for r in records
                    if r["retrieval"]["retrieval_scoring_status"] == "scored")
    retrieval_k: dict[str, Any] = {}
    for k in (1, 5, 10, 20):
        hits = sum(r["retrieval"].get("gold_evidence_documents_retrieved", 0) if k >= 20 else
                   round(r["retrieval"]["recall_at_k"].get(str(k), 0) *
                         r["retrieval"]["gold_evidence_document_count"])
                   for r in records if r["retrieval"]["retrieval_scoring_status"] == "scored")
        retrieval_k[str(k)] = {"hits": hits, "gold_document_denominator": gold_docs,
                               "recall": hits / gold_docs if gold_docs else None}
    mrr_vals = [r["retrieval"]["reciprocal_rank"] for r in records
                if r["retrieval"].get("reciprocal_rank") is not None]
    precision_k: dict[str, Any] = {}
    for k in (1, 5, 10, 20):
        vals = [
            len(set(x for c in r["retrieved_chunks"][:k]
                    if (x := c.get("scifact_doc_id")) is not None) & set(r["gold_evidence_doc_ids"] or [])) /
            len(set(c.get("scifact_doc_id") for c in r["retrieved_chunks"][:k]
                     if c.get("scifact_doc_id") is not None))
            for r in records if r["retrieval"]["retrieval_scoring_status"] == "scored" and
            any(c.get("scifact_doc_id") is not None for c in r["retrieved_chunks"][:k])]
        precision_k[str(k)] = {"macro_precision": statistics.mean(vals) if vals else None,
                               "claim_denominator": len(vals)}
    citation_rows = [r["citation"] for r in records if r["citation"] is not None]
    if system == "A":
        evidence = {"status": "NOT APPLICABLE",
                    "reason": "System A emits no selected evidence/citations"}
    else:
        evidence = {"status": "scored" if citation_rows else "NOT COMPUTED",
                    "claim_denominator": len(citation_rows),
                    **{key: (statistics.mean(v[key] for v in citation_rows if v[key] is not None)
                             if any(v[key] is not None for v in citation_rows) else None)
                       for key in ("precision", "recall", "f1")}}
    elapsed = [r["latency_seconds"] for r in records if r["latency_seconds"] is not None]
    verdict_counts = Counter(r["predicted_label"] or "MISSING_OR_INVALID" for r in records)
    rationale_coverage = [r["attribution"].get("retrieved_rationale_sentence_recall") for r in records
                          if r.get("attribution") and
                          r["attribution"].get("retrieved_rationale_sentence_recall") is not None]
    supported_rows = [r for r in records if r["predicted_label"] == "SUPPORTED" and
                      r.get("gold_evidence_doc_ids")]
    unsupported_supported = [int(not (set(r["selected_evidence_doc_ids"] or []) &
                                      set(r["gold_evidence_doc_ids"]))) for r in supported_rows]
    return {
        "attempted": len(records),
        "completed": sum(r["status"] == "SUCCESS" for r in records),
        "failed": sum(r["status"] != "SUCCESS" for r in records),
        "classification": ("NOT APPLICABLE" if system == "A" else _classification(records)),
        "retrieval": {"scorable_claim_denominator": len(retrieval),
                      "output_availability": availability,
                      "Recall@k": retrieval_k,
                      "MRR": {"value": statistics.mean(mrr_vals) if mrr_vals else None,
                              "claim_denominator": len(mrr_vals)},
                      "document_precision@k_macro": precision_k,
                      "interpretation": "retrieval of annotated evidence documents; not entailment"},
        "selected_evidence_document_metrics": evidence,
        "retrieved_rationale_sentence_recall": {
            "value": statistics.mean(rationale_coverage) if rationale_coverage else None,
            "claim_denominator": len(rationale_coverage),
            "interpretation": "retrieval contains exact annotated rationale sentences; relevance is not entailment"},
        "reliability": {
            "verdict_counts": dict(verdict_counts),
            "insufficient_evidence_rate": {
                "numerator": verdict_counts.get("INSUFFICIENT_EVIDENCE", 0),
                "denominator": len(records),
                "value": verdict_counts.get("INSUFFICIENT_EVIDENCE", 0) / len(records) if records else None},
            "unsupported_supported_claim_rate": {
                "numerator": sum(unsupported_supported), "denominator": len(unsupported_supported),
                "value": sum(unsupported_supported) / len(unsupported_supported) if unsupported_supported else None,
                "definition": "among SUPPORTED predictions with annotated gold evidence, selected no gold-rationale document"},
            "evidence_coverage": {"status": "NOT COMPUTED",
                "reason": "no preregistered coverage definition in the existing experiment evaluator"}},
        "efficiency": {
            "latency_seconds": {"mean": statistics.mean(elapsed) if elapsed else None,
                                "median": statistics.median(elapsed) if elapsed else None,
                                "denominator": len(elapsed)},
            "llm_calls": sum(r["llm_calls"] or 0 for r in records),
            "retrieval_calls": sum(r["retrieval_calls"] or 0 for r in records),
            "verification_calls": sum(r["verification_calls"] or 0 for r in records),
            "controller_cycles": sum(r["controller_cycles"] or 0 for r in records),
        },
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records, _ = recover_jsonl(path)
    return records


def aggregate_stored_records(attempts: Sequence[dict[str, Any]],
                             scored: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {system: _system_metrics([r for r in scored if r["system"] == system], system)
            for system in SYSTEM_NAMES}


def paired_accuracy_bootstrap(records: Sequence[dict[str, Any]], *, seed: int = 0,
                              replicates: int = 10_000) -> dict[str, Any]:
    """Paired percentile interval for C minus B accuracy on common labeled claims."""
    by_pair: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in records:
        if row.get("system") in {"B", "C"}:
            by_pair[row["example_id"]][row["system"]] = row
    paired = [(pair["B"], pair["C"]) for pair in by_pair.values()
              if set(pair) == {"B", "C"} and pair["B"].get("gold_label") in LABELS and
              pair["C"].get("gold_label") == pair["B"].get("gold_label")]
    if len(paired) < 10:
        return {"status": "NOT COMPUTED", "reason": "fewer than 10 paired labeled examples",
                "paired_denominator": len(paired)}
    differences = [int(c.get("predicted_label") == c["gold_label"]) -
                   int(b.get("predicted_label") == b["gold_label"]) for b, c in paired]
    rng = random.Random(seed)
    estimates = [sum(rng.choice(differences) for _ in differences) / len(differences)
                 for _ in range(replicates)]
    estimates.sort()
    return {"status": "descriptive paired bootstrap", "comparison": "System C minus System B accuracy",
            "paired_denominator": len(paired), "accuracy_difference": statistics.mean(differences),
            "percentile_95_ci": {"low": estimates[int(0.025 * replicates)],
                                 "high": estimates[min(replicates - 1, int(0.975 * replicates))]},
            "replicates": replicates, "seed": seed,
            "interpretation": "uncertainty interval only; no hypothesis test or ranking"}


def _run_phase10_dev_unlocked(*, dataset_root: str | Path | None = None,
                    output_root: str | Path | None = None,
                    count: int = 30, preflight: bool = False,
                    evidence_limit: int = 2,
                    resume: bool = False, provider: Any = None,
                    model_preflight: Mapping[str, Any] | None = None) -> dict[str, Any]:
    root = Path(dataset_root or ROOT / "data" / "scifact" / "data").resolve()
    results_root = Path(output_root or ROOT / "evaluation" / "experiments" / "results").resolve()
    run_dir = results_root / (f"phase10_dev_preflight_{evidence_limit}e" if preflight else "phase10_dev_run")
    combined_path = results_root / (f"phase10_dev_preflight_{evidence_limit}e.json" if preflight else "phase10_dev.json")
    if evidence_limit < 1:
        raise ValueError("evidence_limit must be positive")
    if not resume and (run_dir.exists() or combined_path.exists()):
        raise FileExistsError("Phase 10C output exists; choose a new output root or pass --resume")
    run_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_scifact_dataset(root, "dev")
    selection = ([1, 3, 42] if preflight else select_stratified_dev_ids(dataset, count))
    by_id = {row.runtime.claim_id: row for row in dataset.claims}
    if any(i not in by_id for i in selection):
        raise ValueError("Phase 10C selection contains claim IDs outside dev")
    config = ExperimentConfig(
        dataset_root=str(root), split="dev", example_ids=tuple(selection), top_k=20,
        verification_evidence_limit=evidence_limit, model="qwen3:4b", temperature=0.0,
        maximum_controller_cycles=1, maximum_retrieval_actions=1,
        maximum_verification_actions=1,
        notes={"phase": "10C", "selection_rule": "fixed 1,3,42 preflight" if preflight else
               "proportional stratified sample across SUPPORT, CONTRADICT, and unlabeled DEV strata; midpoint-spread within each original file-order stratum",
               "threshold_tuning": "none"},
    )
    attempt_path, scored_path = run_dir / "runtime_attempts.jsonl", run_dir / "per_example.jsonl"
    inflight_path = run_dir / "inflight_attempt.json"
    existing: list[dict[str, Any]] = []
    recovery_reports = []
    scored_existing: list[dict[str, Any]] = []
    if resume:
        existing, recovery = recover_jsonl(attempt_path, validator=validate_attempt_record,
                                           unique_key=attempt_id_key)
        recovery_reports.append(recovery.__dict__)
        pair_keys = [(r.get("example_id"), r.get("system")) for r in existing]
        if len(pair_keys) != len(set(pair_keys)):
            raise ValueError("Duplicate claim/system attempts prevent safe resume")
        scored_existing, recovery = recover_jsonl(scored_path, validator=validate_scored_record,
                                                   unique_key=claim_system_key)
        recovery_reports.append(recovery.__dict__)
        if inflight_path.exists():
            inflight = json.loads(inflight_path.read_text(encoding="utf-8"))
            pair = (inflight["example_id"], inflight["system"])
            if pair not in set(pair_keys):
                interrupted = interrupted_attempt_record(inflight)
                append_jsonl(attempt_path, interrupted)
                existing.append(interrupted)
            inflight_path.unlink()
    expected_pairs = [(claim_id, system) for claim_id in selection for system in SYSTEM_NAMES]
    pending_pairs = set(resume_pending_pairs(expected_pairs, existing))
    done = set(expected_pairs) - pending_pairs
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists() and not resume:
        raise FileExistsError(manifest_path)
    if resume and not manifest_path.exists():
        raise ValueError("Incompatible resume configuration (run_manifest.json: <missing>)")
    corpus_hash = sha256_file(dataset.corpus_path)
    claims_hash = sha256_file(dataset.claims_path)
    hashes = {name: sha256_file(ROOT / name) for name in FROZEN_FILES}
    if provider is None:
        provider = build_provider(dataset)
    source_files = ["evaluation/experiments/run_phase10_dev.py", "evaluation/experiments/evaluator.py",
        "evaluation/experiments/experiment_config.py", "evaluation/experiments/system_a_rag.py",
        "evaluation/experiments/system_b_verified_rag.py", "evaluation/experiments/system_c_multi_agent.py",
        "evaluation/experiments/local_model.py", "evaluation/experiments/phase10e_infrastructure.py",
        "src/baseline_rag.py", "research_capability_dispatch.py", "research_continuation.py",
        "research_autonomous_controller.py", "research_autonomous_execution.py",
        "research_verification_capability.py"]
    source_hashes = {name: sha256_file(ROOT / name) for name in source_files}
    preflight = dict(model_preflight or {})
    config_material = {
        "experiment": "Controlled A/B/C SciFact development evaluation", "phase": "10C",
        "run_id": None,
        "dataset": {"identity": "SciFact-Orig/dev", "claims_sha256": claims_hash,
            "corpus_sha256": corpus_hash, "claim_count": len(dataset.claims),
            "corpus_documents": len(dataset.corpus)},
        "selection": {"claim_ids": selection, "seed": config.seed,
            "algorithm_version": "proportional-midpoint-stratified-v1",
            "rule": config.notes["selection_rule"]},
        "systems": {"A": "Strong RAG; free text; no classification",
            "B": "Strong RAG plus frozen Experiment B verification",
            "C": "real Phase 8A/8B/8C controller; 7B dispatch; Phase 6; explicit 7C continuation"},
        "model": {"provider": "Ollama", "endpoint": "http://127.0.0.1:11434",
            "name": config.model, "digest": preflight.get("model_digest"),
            "digest_status": preflight.get("digest_status", "unavailable"),
            "temperature": config.temperature, "generation": {"num_gpu": 0, "think": False}},
        "prompt_and_code_hashes": source_hashes,
        "frozen_core_sha256": {name: sha256_file(ROOT / name) for name in (
            "rag.py", "research_pipeline.py", "claim_verification.py", "verification_units.py",
            "experiment_b_verification.py", "evaluation/scifact/scifact_adapter.py",
            "evaluation/scifact/metrics.py", "evaluation/scifact/run_scifact_evaluation.py")},
        "retrieval": {"top_k": config.top_k, "hybrid": config.hybrid_retrieval,
            "reranking": config.reranking, "evidence_limit": evidence_limit,
            "index_identity": getattr(provider, "phase10_model_snapshots", None)},
        "verification": {"architecture": "frozen Experiment B", "evidence_limit": evidence_limit},
        "schema_version": "phase10c-per-example-v2",
        "failure_semantics": {"retry": "none", "unavailable_is_not_miss": True},
    }
    preflight_attempts = _read_jsonl(results_root / "phase10_dev_preflight_2e" / "runtime_attempts.jsonl")
    preflight_means = {}
    for system in SYSTEM_NAMES:
        durations = [row["latency_seconds"] for row in preflight_attempts
                     if row.get("system") == system and row.get("status") == "SUCCESS"]
        if durations:
            preflight_means[system] = statistics.mean(durations)
    planning_runtime = {
        "estimate_type": "planning projection; not an experimental result",
        "matched_preflight_attempts": str(results_root / "phase10_dev_preflight_2e" / "runtime_attempts.jsonl"),
        "mean_seconds_per_system_from_matched_preflight": preflight_means or
            {"A": 34.8, "B": 65.0, "C": 63.7},
        "source": "Phase 10C two-evidence preflight" if preflight_means else "reported Phase 10B smoke",
        "projected_total_seconds": (sum(preflight_means.values()) * len(selection)
            if preflight_means else 163.5 * len(selection)),
        "projected_total_minutes": ((sum(preflight_means.values()) * len(selection) / 60)
            if preflight_means else 163.5 * len(selection) / 60),
    }
    label_counts = Counter(gold_claim_label(row.annotations)[0] or "UNLABELED"
                           for row in dataset.claims)
    annotation_counts = Counter(
        "unavailable" if not row.annotations.labels_available else
        "nonempty_rationale_evidence" if row.annotations.evidence else "explicitly_empty_evidence"
        for row in dataset.claims)
    manifest = {
        "experiment": "Controlled A/B/C SciFact development evaluation",
        "phase": "10C", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "RUNNING", "dataset": {"name": "SciFact-Orig", "split": "dev",
            "claims_file": str(dataset.claims_path), "corpus_file": str(dataset.corpus_path),
            "claims_sha256": claims_hash, "corpus_sha256": corpus_hash,
            "claim_count": len(dataset.claims), "corpus_document_count": len(dataset.corpus),
            "gold_label_counts": dict(label_counts),
            "evidence_annotation_counts": dict(annotation_counts)},
        "selection": {"rule": config.notes["selection_rule"], "claim_ids": selection,
                      "selected_count": len(selection), "full_dev": len(selection) == len(dataset.claims)},
        "config": config.to_dict(), "systems": {"A": "Strong RAG free-text generation; no classification",
            "B": "same top-20 retrieval plus frozen Experiment B verification",
            "C": "real 8C controller; Phase 8A deterministic scoped action binding; 8B, 7B, Phase 6; explicit continuation"},
        "gold_leakage": {"execution_inputs": ["claim_id", "claim"],
            "gold_join_stage": "after all system callbacks for a claim have completed",
            "gold_annotations_passed_to_execution": False},
        "failure_handling": {"automatic_retries": False,
            "resume_policy": "existing system/claim attempts, including failures, are never rerun; resume executes only missing pairs",
            "failure_status_is_preserved": True},
        "retrieval": {"top_k": 20, "hybrid": True, "reranking": True,
            "verification_evidence_limit": evidence_limit, "same_provider_shared": True,
            "index_identity": getattr(provider, "phase10_model_snapshots", None)},
        "model": {"tag": config.model, "temperature": config.temperature,
            "thinking": False, "device": "CPU (Ollama num_gpu=0)",
            "ollama": _ollama_model_details(config.model)},
        "runtime": {"python": sys.version, "platform": platform.platform(), "seed": config.seed,
            "planning_estimate": planning_runtime},
        "frozen_sha256": hashes,
        "source_sha256": source_hashes,
        "configuration_lock": None,
        "jsonl_recovery": recovery_reports,
        "metrics_definitions": {
            "Recall@k": "unique annotated evidence documents retrieved in top k chunks divided by all annotated evidence documents; pooled document-hit micro denominator",
            "MRR": "mean reciprocal rank of first retrieved annotated evidence document over claims with nonempty rationale evidence",
            "document_precision@k": "macro claim-level unique retrieved document precision against annotated evidence document IDs",
            "citation P/R/F1": "selected evidence document IDs versus annotated evidence document IDs; not claim entailment; A not applicable",
            "classification": "SUPPORT/CONTRADICT mapped to SUPPORTED/CONTRADICTED only for unanimous rationale stance; invalid or failed predictions count incorrect in primary labeled denominator; A not applicable",
            "latency": "wall-clock seconds around one system callback, includes its retrieval/model/controller work",
            "calls": "adapter-reported calls; C includes provenance preflight retrieval and controller work"},
        "limitations": ["DEV-only evaluation; no test split loaded", "No prompt or threshold tuning on dev labels"],
    }
    if resume and manifest_path.exists():
        saved_manifest = load_resume_manifest(manifest_path)
        if (saved_manifest.get("attempted_system_claim_pairs", 0) > 0 and
                not attempt_path.exists()):
            raise ValueError("Resume artifact missing: runtime_attempts.jsonl")
        stored = (saved_manifest.get("configuration_lock") or {}).get("configuration")
        if stored is None:
            raise ValueError("Incompatible resume configuration (configuration_lock: <missing>)")
        config_material["run_id"] = stored.get("run_id")
        validate_resume_configuration(stored, config_material)
        normalized_config = json.loads(json.dumps(config.to_dict()))
        if (saved_manifest.get("selection", {}).get("claim_ids") != selection or
                saved_manifest.get("config") != normalized_config):
            raise ValueError("Incompatible resume configuration (selection/config differs)")
        manifest = saved_manifest
        manifest["dataset"].setdefault("gold_label_counts", dict(label_counts))
        manifest["dataset"].setdefault("evidence_annotation_counts", dict(annotation_counts))
        manifest.setdefault("failure_handling", {
            "automatic_retries": False,
            "resume_policy": "existing system/claim attempts, including failures, are never rerun; resume executes only missing pairs",
            "failure_status_is_preserved": True,
        })
    if not manifest_path.exists():
        config_material["run_id"] = hashlib.sha256(
            f"{datetime.now(timezone.utc).isoformat()}:{os.getpid()}".encode()).hexdigest()[:24]
        manifest["configuration_lock"] = make_configuration_lock(config_material)
        atomic_write_json(manifest_path, manifest)
    if provider is None:
        raise RuntimeError("retrieval provider unavailable")
    with attempt_path.open("a", encoding="utf-8") as attempt_stream:
        for claim_id in selection:
            record = by_id[claim_id]
            runtime = runtime_claim_payload(record.runtime)
            outputs: dict[str, SystemOutput] = {}
            for system, callback in SYSTEMS.items():
                old = next((row for row in existing if row["example_id"] == claim_id and row["system"] == system), None)
                if old is not None:
                    outputs[system] = _output_from_record(old)
                    continue
                if (claim_id, system) in done:
                    continue
                inflight = {"attempt_id": hashlib.sha256(
                    f"{claim_id}:{system}:{time.time_ns()}:{os.getpid()}".encode()).hexdigest(),
                    "example_id": claim_id, "system": system, "runtime_input": runtime,
                    "started_at_utc": datetime.now(timezone.utc).isoformat()}
                atomic_write_json(inflight_path, inflight)
                output, elapsed = _safe_call(callback, runtime, provider, config)
                output_record = _output_to_record(output)
                # Persist execution output before any annotation object is joined.
                attempt = {"attempt_id": inflight["attempt_id"],
                           "example_id": claim_id, "system": system, "split": "dev",
                           "runtime_input": runtime, "status": output.status,
                           "latency_seconds": elapsed, **output_record}
                append_jsonl(attempt_path, attempt)
                inflight_path.unlink(missing_ok=True)
                existing.append(attempt)
                done.add((claim_id, system))
                outputs[system] = output
                print(f"Phase 10C {claim_id} {system}: {output.status} ({elapsed:.1f}s)", flush=True)
            if len(outputs) != 3:
                continue
            gold_label, gold_status = gold_claim_label(record.annotations)
            gold_docs = sorted({d.doc_id for d in record.annotations.evidence or ()}) if record.annotations.evidence is not None else None
            with scored_path.open("a", encoding="utf-8") as score_stream:
                for system, output in outputs.items():
                    valid_retrieval = output.retrieval_output_status in {
                        OutputState.VALID_EMPTY.value, OutputState.VALID_NONEMPTY.value}
                    retrieval = (retrieval_metrics_for_claim(record.annotations, output.retrieved_chunks)
                        if valid_retrieval else {"retrieval_scoring_status": output.retrieval_output_status,
                            "reciprocal_rank": None, "recall_at_k": {},
                            "gold_evidence_document_count": len(gold_docs or []),
                            "gold_evidence_documents_retrieved": None})
                    attribution = _attribution_metrics(record.annotations,
                        output.selected_evidence_doc_ids, output.retrieved_chunks)
                    citation = None
                    if (system != "A" and record.annotations.evidence and output.evidence_output_status in {
                            OutputState.VALID_EMPTY.value, OutputState.VALID_NONEMPTY.value}):
                        selected_docs = set(output.selected_evidence_doc_ids)
                        gold_doc_set = set(gold_docs or [])
                        tp = len(selected_docs & gold_doc_set)
                        p = tp / len(selected_docs) if selected_docs else None
                        r = tp / len(gold_doc_set) if gold_doc_set else None
                        f1 = (2*p*r/(p+r) if p is not None and r is not None and p+r else
                              0.0 if p == 0 or r == 0 else None)
                        citation = {"precision": p, "recall": r, "f1": f1}
                    result = {"example_id": claim_id, "system": system, "split": "dev",
                        "runtime_input": runtime, "status": output.status,
                        "output": _output_to_record(output), "predicted_label": output.predicted_label,
                        "retrieval_output_status": output.retrieval_output_status,
                        "evidence_output_status": output.evidence_output_status,
                        "gold_label": gold_label, "gold_label_status": gold_status,
                        "gold_evidence_doc_ids": gold_docs,
                        "retrieved_chunks": output.retrieved_chunks,
                        "retrieved_evidence_ids": [c.get("evidence_id") for c in output.retrieved_chunks],
                        "selected_evidence_doc_ids": output.selected_evidence_doc_ids,
                        "latency_seconds": next(r["latency_seconds"] for r in existing
                            if r["example_id"] == claim_id and r["system"] == system),
                        "llm_calls": output.llm_calls, "retrieval_calls": output.retrieval_calls,
                        "verification_calls": output.verification_calls,
                        "controller_cycles": output.controller_cycles,
                        "provenance": _provenance(output.raw_output),
                        "error": output.error, "error_type": output.error_type,
                        "error_stage": output.error_stage, "scorable": {
                            "classification": gold_label is not None,
                            "reason": None if gold_label is not None else gold_status,
                            "retrieval": retrieval["retrieval_scoring_status"] == "scored",
                            "retrieval_reason": retrieval["retrieval_scoring_status"]},
                        "retrieval": retrieval, "citation": citation}
                    result["attribution"] = attribution
                    # Idempotent append for resumes.
                    current = scored_existing
                    if not any(r["example_id"] == claim_id and r["system"] == system for r in current):
                        append_jsonl(scored_path, result)
                        current.append(result)
        scored = _read_jsonl(scored_path)
    attempts = _read_jsonl(attempt_path)
    summary = aggregate_stored_records(attempts, scored)
    subset_completed = len(scored) == 3 * len(selection)
    manifest["execution_status"] = "COMPLETED" if subset_completed else "PARTIAL"
    manifest["status"] = ("COMPLETED" if subset_completed and len(selection) == len(dataset.claims)
                          else "PARTIAL")
    manifest["attempted_system_claim_pairs"] = len(attempts)
    manifest["scored_system_claim_pairs"] = len(scored)
    manifest["execution_counts"] = {system: {
        "attempted": sum(r["system"] == system for r in attempts),
        "completed": sum(r["system"] == system and r["status"] == "SUCCESS" for r in attempts),
        "failed": sum(r["system"] == system and r["status"] != "SUCCESS" for r in attempts)}
        for system in SYSTEM_NAMES}
    manifest["runtime"]["measured_callback_seconds"] = {
        system: sum(row["latency_seconds"] or 0 for row in attempts if row["system"] == system)
        for system in SYSTEM_NAMES}
    manifest["runtime"]["measured_total_callback_seconds"] = sum(
        manifest["runtime"]["measured_callback_seconds"].values())
    manifest["runtime"]["measured_mean_seconds_per_selected_claim"] = (
        manifest["runtime"]["measured_total_callback_seconds"] / len(selection) if selection else None)
    manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(run_dir / "summary.json", summary)
    atomic_write_json(manifest_path, manifest)
    combined = {"experiment": manifest["experiment"], "phase": "10C", "status": manifest["status"],
        "execution_status": manifest["execution_status"],
        "dataset": manifest["dataset"], "selection": manifest["selection"],
        "systems": summary, "runtime": manifest["runtime"], "gold_leakage": manifest["gold_leakage"],
        "statistical_analysis": paired_accuracy_bootstrap(scored, seed=config.seed),
        "metrics_definitions": manifest["metrics_definitions"],
        "attempted_system_claim_pairs": len(attempts), "scored_system_claim_pairs": len(scored),
        "per_example_results_path": str(scored_path), "runtime_attempts_path": str(attempt_path),
        "manifest_path": str(manifest_path), "limitations": manifest["limitations"],
        "per_example_results": scored}
    atomic_write_json(combined_path, combined)
    return combined


def run_phase10_dev(*, dataset_root: str | Path | None = None,
                    output_root: str | Path | None = None,
                    count: int = 30, preflight: bool = False,
                    evidence_limit: int = 2, resume: bool = False,
                    provider: Any = None,
                    model_preflight: Mapping[str, Any] | None = None) -> dict[str, Any]:
    results_root = Path(output_root or ROOT / "evaluation" / "experiments" / "results").resolve()
    run_dir = results_root / (f"phase10_dev_preflight_{evidence_limit}e" if preflight else "phase10_dev_run")
    with exclusive_run_lock(results_root / f".{run_dir.name}.phase10e.writer.lock"):
        return _run_phase10_dev_unlocked(dataset_root=dataset_root, output_root=output_root,
            count=count, preflight=preflight, evidence_limit=evidence_limit,
            resume=resume, provider=provider, model_preflight=model_preflight)


def _output_to_record(output: SystemOutput) -> dict[str, Any]:
    return {"predicted_label": output.predicted_label,
        "retrieved_chunks": _json_safe(output.retrieved_chunks),
        "selected_evidence_doc_ids": _json_safe(output.selected_evidence_doc_ids),
        "assessments": _json_safe(output.assessments),
        "unsupported_claims": output.unsupported_claims,
        "verification_calls": output.verification_calls, "llm_calls": output.llm_calls,
        "retrieval_calls": output.retrieval_calls, "controller_cycles": output.controller_cycles,
        "raw_output": _json_safe(output.raw_output), "error": output.error,
        "status": output.status, "error_type": output.error_type,
        "error_stage": output.error_stage, "error_traceback": output.error_traceback,
        "retrieval_output_status": output.retrieval_output_status,
        "evidence_output_status": output.evidence_output_status}


def _output_from_record(row: Mapping[str, Any]) -> SystemOutput:
    return SystemOutput(**{k: row.get(k) for k in SystemOutput.__dataclass_fields__})


def _provenance(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    state = raw.get("research_state", {})
    return {"controller_decisions": raw.get("controller_decisions", []),
            "controller_traces": raw.get("controller_traces", []),
            "document_versions": state.get("document_versions", []) if isinstance(state, dict) else [],
            "evidence": state.get("evidence", []) if isinstance(state, dict) else []}


def _ollama_model_details(model: str) -> dict[str, Any]:
    req = urllib.request.Request("http://127.0.0.1:11434/api/show",
        data=json.dumps({"name": model}).encode(), headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
        return {"details": data.get("details", {}), "model_info": data.get("model_info", {})}
    except Exception as exc:
        return {"status": f"unavailable: {type(exc).__name__}: {exc}"}


def build_provider(dataset: Any):
    import numpy as np
    import rag
    import torch
    from research_retrieval import ExistingRAGProvider
    from sentence_transformers import CrossEncoder, SentenceTransformer

    torch.set_num_threads(4)
    def cached_snapshot(model_id: str) -> Path:
        hub = Path.home() / ".cache" / "huggingface" / "hub"
        folder = "models--" + model_id.replace("/", "--")
        paths = [hub / folder]
        if "/" not in model_id:
            paths.extend(hub.glob(f"models--*--{model_id}"))
        snapshots = sorted((s for p in paths if (p / "snapshots").is_dir()
            for s in (p / "snapshots").iterdir() if (s / "config.json").is_file()), key=lambda p: p.name)
        if not snapshots:
            raise FileNotFoundError(f"Local model cache unavailable: {model_id}")
        return snapshots[-1]
    chunks = corpus_to_rag_chunks(dataset.corpus, rag)
    embedding_path, reranker_path = cached_snapshot(rag.EMBEDDING_MODEL_NAME), cached_snapshot(rag.RERANKER_MODEL_NAME)
    embedding = SentenceTransformer(str(embedding_path), device="cpu")
    digest = hashlib.sha256()
    digest.update(embedding_path.name.encode())
    for chunk in chunks:
        digest.update(str(chunk.get("chunk_id", "")).encode())
        digest.update(str(chunk.get("text", "")).encode())
    cache_path = Path(os.getenv("TEMP", tempfile_dir())) / f"phase10_scifact_dev_embeddings_{digest.hexdigest()[:20]}.npy"
    if cache_path.is_file():
        embeddings = np.load(cache_path, mmap_mode="r")
        if len(embeddings) != len(chunks):
            raise ValueError("Cached embedding count does not match corpus chunk count")
        faiss_index, bm25_index = rag.build_faiss_index(embeddings), rag.build_bm25_index(chunks)
    else:
        embeddings, faiss_index, bm25_index = rag.build_search_database(chunks, embedding)
        np.save(cache_path, embeddings)
    reranker = CrossEncoder(str(reranker_path), device="cpu")
    provider = ExistingRAGProvider(chunks=chunks, embedding_model=embedding, reranker=reranker,
        faiss_index=faiss_index, bm25_index=bm25_index, parents=None, retrieval_module=rag)
    provider.phase10_model_snapshots = {"embedding": {"model": rag.EMBEDDING_MODEL_NAME,
        "snapshot": embedding_path.name}, "reranker": {"model": rag.RERANKER_MODEL_NAME,
        "snapshot": reranker_path.name}, "corpus_chunks": len(chunks),
        "embedding_cache_file": cache_path.name, "embedding_cache_sha256": sha256_file(cache_path),
        "embedding_cache_shape": list(np.load(cache_path, mmap_mode="r").shape),
        "retrieval_config": {key: getattr(rag, key) for key in ("CHILD_CHUNK_SIZE", "CHILD_CHUNK_OVERLAP",
            "VECTOR_TOP_K", "BM25_TOP_K", "RRF_K", "RERANK_TOP_K", "FINAL_TOP_K") if hasattr(rag, key)}}
    return provider


def tempfile_dir() -> str:
    import tempfile
    return tempfile.gettempdir()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--evidence-limit", type=int, default=2)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dataset-root", default=str(ROOT / "data" / "scifact" / "data"))
    parser.add_argument("--output-root", default=str(ROOT / "evaluation" / "experiments" / "results"))
    parser.add_argument("--model-preflight-json", default=None,
                        help="Successful local model health report required for a full-dev run")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    model_preflight = None
    if args.model_preflight_json:
        model_preflight = json.loads(Path(args.model_preflight_json).read_text(encoding="utf-8"))
    if args.count == 300 and (not model_preflight or model_preflight.get("status") != "PASS"):
        raise SystemExit("Full DEV evaluation refused: pass a successful --model-preflight-json first")
    result = run_phase10_dev(dataset_root=args.dataset_root, output_root=args.output_root,
        count=args.count, evidence_limit=args.evidence_limit,
        preflight=args.preflight, resume=args.resume, model_preflight=model_preflight)
    print(json.dumps({key: value for key, value in result.items() if key != "per_example_results"},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
