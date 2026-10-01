"""Metrics for SciFact retrieval coverage and evidence-level Experiment B checks.

No overall three-class SciFact gold label is inferred in this module.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:  # Support both direct script imports and ``evaluation.scifact`` imports.
    from .scifact_adapter import ClaimAnnotations
except ImportError:  # pragma: no cover - used when the runner is executed by path.
    from scifact_adapter import ClaimAnnotations


LABEL_TO_STANCE = {"SUPPORT": "SUPPORTS", "CONTRADICT": "CONTRADICTS"}
DEFAULT_TOP_K = (1, 5, 10, 20)


def _doc_id(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _chunk_doc_id(chunk: Mapping[str, Any]) -> int | None:
    value = chunk.get("scifact_doc_id", chunk.get("document_id"))
    if value is None:
        filename = str(chunk.get("filename", ""))
        prefix = "scifact_doc_"
        if filename.startswith(prefix):
            value = filename[len(prefix):]
    return _doc_id(value)


def retrieval_metrics_for_claim(
    annotations: ClaimAnnotations,
    retrieved_chunks: Sequence[Mapping[str, Any]],
    top_k: Sequence[int] = DEFAULT_TOP_K,
) -> dict[str, Any]:
    """Calculate unique-gold-document Recall@K and first-gold rank.

    Duplicate chunks from one SciFact abstract never increase the numerator.
    An unavailable or explicitly empty gold evidence mapping is unscorable and
    returns null recall rather than an invented zero/NEI label.
    """
    if any(isinstance(k, bool) or not isinstance(k, int) or k < 1 for k in top_k):
        raise ValueError("top_k values must be positive integers.")
    if not annotations.labels_available or annotations.evidence is None:
        return {
            "retrieval_scoring_status": "gold_annotations_unavailable",
            "gold_evidence_document_count": None,
            "unique_retrieved_document_count": len({x for chunk in retrieved_chunks if (x := _chunk_doc_id(chunk)) is not None}),
            "recall_at_k": {str(k): None for k in top_k},
            "first_gold_rank": None,
            "reciprocal_rank": None,
        }
    gold_doc_ids = {doc.doc_id for doc in annotations.evidence}
    unique_retrieved = {
        doc_id for chunk in retrieved_chunks if (doc_id := _chunk_doc_id(chunk)) is not None
    }
    if not gold_doc_ids:
        return {
            "retrieval_scoring_status": "no_annotated_gold_evidence",
            "gold_evidence_document_count": 0,
            "unique_retrieved_document_count": len(unique_retrieved),
            "recall_at_k": {str(k): None for k in top_k},
            "first_gold_rank": None,
            "reciprocal_rank": None,
        }
    first_gold_rank = next(
        (rank for rank, chunk in enumerate(retrieved_chunks, 1)
         if _chunk_doc_id(chunk) in gold_doc_ids),
        None,
    )
    recall_at_k: dict[str, float] = {}
    for k in top_k:
        found = {
            doc_id for chunk in retrieved_chunks[:k]
            if (doc_id := _chunk_doc_id(chunk)) is not None and doc_id in gold_doc_ids
        }
        recall_at_k[str(k)] = len(found) / len(gold_doc_ids)
    return {
        "retrieval_scoring_status": "scored",
        "gold_evidence_document_count": len(gold_doc_ids),
        "unique_retrieved_document_count": len(unique_retrieved),
        "gold_evidence_documents_retrieved": len(gold_doc_ids & unique_retrieved),
        "recall_at_k": recall_at_k,
        "first_gold_rank": first_gold_rank,
        "reciprocal_rank": 1.0 / first_gold_rank if first_gold_rank else 0.0,
    }


def _assessment_trace_stats(report: Mapping[str, Any]) -> tuple[int, int, int]:
    """Return expected pair count, invalid pair count, and retry count."""
    expected_pairs = invalid_pairs = retries = 0
    for unit_trace in report.get("trace", []) or []:
        for judge_attempt in unit_trace.get("judge_attempts", []) or []:
            pair_calls = judge_attempt.get("pair_assessment_calls", []) or []
            expected_pairs += len(pair_calls)
            invalid_pairs += sum(bool(pair.get("error")) for pair in pair_calls)
            retries += sum(max(0, len(pair.get("attempts", []) or []) - 1) for pair in pair_calls)
        retrieval_attempts = unit_trace.get("retrieval_attempts", []) or []
        retries += sum(int(attempt.get("retry", 0) or 0) > 0 for attempt in retrieval_attempts)
    return expected_pairs, invalid_pairs, retries


def verification_metrics_for_claim(
    annotations: ClaimAnnotations,
    retrieved_chunks: Sequence[Mapping[str, Any]],
    verification_report: Mapping[str, Any] | None,
    *,
    evidence_limit: int = 8,
) -> dict[str, Any]:
    """Compare pairwise stances only with retrieved gold-rationale sentences.

    A claim-level verdict is recorded as produced, but is never compared to an
    invented three-class SciFact label. Multi-component and opposing-gold-label
    cases are explicitly marked ambiguous for stance alignment.
    """
    if evidence_limit < 1:
        raise ValueError("evidence_limit must be >= 1.")
    report = verification_report or {}
    verifications = report.get("verifications", []) or []
    verification = verifications[0] if verifications else {}
    verdict = verification.get("verdict")
    assessment_complete = verification.get("assessment_complete")
    missing_assessments = list(verification.get("missing_assessments") or [])
    assessments = verification.get("evidence_assessments", []) or []
    expected_pairs, invalid_pairs, retry_count = _assessment_trace_stats(report)

    if not annotations.labels_available or annotations.evidence is None:
        status = "gold_annotations_unavailable"
        gold_docs: dict[int, Any] = {}
    else:
        gold_docs = {doc.doc_id: doc for doc in annotations.evidence}
        if not gold_docs:
            status = "no_annotated_gold_evidence"
        else:
            status = "gold_evidence_not_retrieved"

    retrieved_doc_ids = {
        doc_id for chunk in retrieved_chunks if (doc_id := _chunk_doc_id(chunk)) is not None
    }
    retrieved_gold_doc_ids = retrieved_doc_ids & set(gold_docs)
    input_evidence = list(retrieved_chunks[:evidence_limit])
    verification_doc_ids = {
        doc_id for chunk in input_evidence if (doc_id := _chunk_doc_id(chunk)) is not None
    }
    verification_gold_doc_ids = verification_doc_ids & set(gold_docs)
    rationale_sentence_labels: dict[tuple[int, int], set[str]] = {}
    retrieved_gold_labels: set[str] = set()
    verification_gold_labels: set[str] = set()
    if annotations.labels_available and annotations.evidence is not None:
        for doc_id in retrieved_gold_doc_ids:
            gold_doc = gold_docs[doc_id]
            for rationale in gold_doc.rationales:
                retrieved_gold_labels.add(rationale.label)
                if doc_id in verification_gold_doc_ids:
                    verification_gold_labels.add(rationale.label)
                if doc_id not in verification_gold_doc_ids:
                    continue
                for sentence_index in rationale.sentence_indices:
                    rationale_sentence_labels.setdefault((doc_id, sentence_index), set()).add(rationale.label)

    evidence_assessment_rows: list[dict[str, Any]] = []
    rationale_hits = 0
    component_ids = {
        str(component.get("component_index"))
        for component in (verification.get("coverage_matrix") or [])
        if component.get("component_index") is not None
    }
    for evidence_number, chunk in enumerate(input_evidence, start=1):
        doc_id = _chunk_doc_id(chunk)
        indices = chunk.get("scifact_sentence_indices", [])
        indices = indices if isinstance(indices, list) else []
        source_sentences = chunk.get("scifact_sentence_texts", [])
        source_sentences = source_sentences if isinstance(source_sentences, list) else []
        retrieved_text = " ".join(str(chunk.get(key, "")) for key in ("text", "chunk_text"))
        normalized_retrieved_text = " ".join(retrieved_text.casefold().split())
        for sentence_index in indices:
            if isinstance(sentence_index, bool) or not isinstance(sentence_index, int):
                continue
            try:
                sentence_position = indices.index(sentence_index)
                source_sentence = str(source_sentences[sentence_position])
            except (ValueError, IndexError):
                # Mapping alone is insufficient to assert that a long source
                # sentence was fully present in a possibly split RAG chunk.
                continue
            normalized_source_sentence = " ".join(source_sentence.casefold().split())
            if not normalized_source_sentence or normalized_source_sentence not in normalized_retrieved_text:
                continue
            labels = rationale_sentence_labels.get((doc_id, sentence_index), set())
            if not labels:
                continue
            rationale_hits += 1
            evidence_index = f"E{evidence_number}"
            observed_rows = [
                row for row in assessments
                if isinstance(row, dict) and str(row.get("evidence_index")) == evidence_index
            ]
            observed_stances = {str(row.get("stance", "")).upper() for row in observed_rows}
            if len(labels) != 1:
                alignment = "ambiguous_conflicting_gold_labels"
                expected = sorted(labels)
            elif len(component_ids) > 1:
                alignment = "ambiguous_multiple_components"
                expected = [next(iter(labels))]
            elif not observed_stances:
                alignment = "missing_assessment"
                expected = [next(iter(labels))]
            elif len(observed_stances) != 1:
                alignment = "ambiguous_multiple_assessments"
                expected = [next(iter(labels))]
            else:
                expected_label = next(iter(labels))
                expected_stance = LABEL_TO_STANCE[expected_label]
                observed_stance = next(iter(observed_stances))
                alignment = "aligned" if observed_stance == expected_stance else "misaligned"
                expected = [expected_label]
            evidence_assessment_rows.append({
                "evidence_index": evidence_index,
                "doc_id": doc_id,
                "sentence_indices": [sentence_index],
                "gold_labels": expected,
                "observed_stances": sorted(observed_stances),
                "alignment": alignment,
            })

    retrieved_gold_conflict = {"SUPPORT", "CONTRADICT"}.issubset(retrieved_gold_labels)
    conflicting_gold = {"SUPPORT", "CONTRADICT"}.issubset(verification_gold_labels)
    if retrieved_gold_conflict:
        status = "conflicting_retrieved_gold_annotations"
    elif status == "gold_evidence_not_retrieved" and retrieved_gold_doc_ids:
        status = "gold_evidence_document_retrieved"
    if rationale_hits and not conflicting_gold:
        status = "gold_rationale_sentence_retrieved"

    if annotations.labels_available and annotations.evidence and not conflicting_gold:
        if not retrieved_gold_doc_ids:
            alignment_status = "gold_evidence_not_retrieved"
        elif rationale_hits == 0:
            alignment_status = "gold_rationale_sentence_not_retrieved"
        elif retrieved_gold_conflict:
            alignment_status = "ambiguous_conflicting_retrieved_gold"
        elif len(component_ids) > 1:
            alignment_status = "ambiguous_multiple_components"
        elif any(row["alignment"].startswith("ambiguous") for row in evidence_assessment_rows):
            alignment_status = "ambiguous"
        elif any(row["alignment"] == "missing_assessment" for row in evidence_assessment_rows):
            alignment_status = "missing_assessment"
        elif any(row["alignment"] == "misaligned" for row in evidence_assessment_rows):
            alignment_status = "misaligned"
        else:
            alignment_status = "aligned"
    else:
        alignment_status = status

    return {
        "predicted_verdict": verdict,
        "assessment_complete": assessment_complete,
        "missing_assessments": missing_assessments,
        "invalid_assessment_pairs": invalid_pairs,
        "expected_assessment_pairs": expected_pairs,
        "assessment_retry_count": retry_count,
        "gold_evidence_documents_retrieved": len(retrieved_gold_doc_ids),
        "gold_rationale_sentence_hits_in_verification_context": rationale_hits,
        "retrieved_gold_annotations_conflict": retrieved_gold_conflict,
        "verification_context_gold_annotations_conflict": conflicting_gold,
        "gold_labels_on_retrieved_evidence": sorted(retrieved_gold_labels),
        "gold_labels_in_verification_context": sorted(verification_gold_labels),
        "evidence_stance_alignment_status": alignment_status,
        "evidence_stance_assessments": evidence_assessment_rows,
        "unable_to_establish_sufficient_evidence": verdict == "INSUFFICIENT_EVIDENCE",
        "verification_ambiguous": alignment_status.startswith("ambiguous") or conflicting_gold or retrieved_gold_conflict,
    }


def _mean(values: Iterable[float | int | None]) -> float | None:
    selected = [float(value) for value in values if value is not None]
    return sum(selected) / len(selected) if selected else None


def summarize_records(records: Sequence[Mapping[str, Any]], top_k: Sequence[int] = DEFAULT_TOP_K) -> dict[str, Any]:
    """Aggregate supported retrieval, verification, and efficiency counts."""
    scored = [row for row in records if row.get("retrieval", {}).get("retrieval_scoring_status") == "scored"]
    recalls: dict[str, float | None] = {}
    for k in top_k:
        denominator = sum(int(row["retrieval"]["gold_evidence_document_count"]) for row in scored)
        numerator = sum(
            int(round(row["retrieval"]["recall_at_k"][str(k)] * row["retrieval"]["gold_evidence_document_count"]))
            for row in scored
        )
        recalls[str(k)] = numerator / denominator if denominator else None
    mrr_values = [row["retrieval"].get("reciprocal_rank") for row in scored]
    verdicts = [row.get("experiment_b", {}).get("verification", {}).get("predicted_verdict") for row in records]
    verification_rows = [row.get("experiment_b", {}).get("verification", {}) for row in records]
    aligned = sum(row.get("evidence_stance_alignment_status") == "aligned" for row in verification_rows)
    misaligned = sum(row.get("evidence_stance_alignment_status") == "misaligned" for row in verification_rows)
    ambiguous = sum(bool(row.get("verification_ambiguous")) for row in verification_rows)
    incomplete = sum(row.get("assessment_complete") is False for row in verification_rows)
    invalid_pairs = sum(int(row.get("invalid_assessment_pairs", 0) or 0) for row in verification_rows)
    expected_pairs = sum(int(row.get("expected_assessment_pairs", 0) or 0) for row in verification_rows)
    missing_pairs = sum(len(row.get("missing_assessments", []) or []) for row in verification_rows)
    coverage_candidates = [
        row for row in verification_rows
        if row.get("gold_rationale_sentence_hits_in_verification_context", 0) > 0
        and not row.get("retrieved_gold_annotations_conflict")
        and not row.get("verification_ambiguous")
    ]
    covered = sum(row.get("assessment_complete") is True and row.get("predicted_verdict") != "INSUFFICIENT_EVIDENCE" for row in coverage_candidates)
    total_rag_calls = sum(int(row.get("efficiency", {}).get("rag_calls", 0) or 0) for row in records)
    total_llm_calls = sum(int(row.get("efficiency", {}).get("llm_calls", 0) or 0) for row in records)
    total_retries = sum(int(row.get("efficiency", {}).get("retries", 0) or 0) for row in records)
    return {
        "split": records[0].get("split") if records else None,
        "claims_processed": len(records),
        "retrieval_scored_claims": len(scored),
        "claims_with_gold_evidence_retrieved": sum(row["retrieval"].get("gold_evidence_documents_retrieved", 0) > 0 for row in scored),
        "retrieval_recall_at_k": recalls,
        "mean_reciprocal_rank": _mean(mrr_values),
        "mean_first_gold_rank": _mean(row["retrieval"].get("first_gold_rank") for row in scored if row["retrieval"].get("first_gold_rank") is not None),
        "experiment_b_verdict_counts": {
            verdict: sum(value == verdict for value in verdicts)
            for verdict in ("SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE")
        },
        "correctly_aligned_gold_evidence_claims": aligned,
        "misaligned_gold_evidence_claims": misaligned,
        "ambiguous_or_conflicting_claims": ambiguous,
        "unable_to_establish_sufficient_evidence": sum(row.get("unable_to_establish_sufficient_evidence", False) for row in verification_rows),
        "verification_coverage": covered / len(coverage_candidates) if coverage_candidates else None,
        "verification_coverage_denominator": len(coverage_candidates),
        "assessment_incomplete_claims": incomplete,
        "invalid_assessment_pairs": invalid_pairs,
        "missing_assessment_pairs": missing_pairs,
        "invalid_or_missing_assessment_rate": (invalid_pairs + missing_pairs) / expected_pairs if expected_pairs else None,
        "efficiency": {
            "rag_calls": total_rag_calls,
            "llm_calls": total_llm_calls,
            "retries": total_retries,
            "mean_retrieval_latency_seconds": _mean(row.get("efficiency", {}).get("retrieval_latency_seconds") for row in records),
            "mean_verification_latency_seconds": _mean(row.get("efficiency", {}).get("verification_latency_seconds") for row in records),
            "mean_end_to_end_latency_seconds": _mean(row.get("efficiency", {}).get("end_to_end_latency_seconds") for row in records),
        },
        "label_mapping_note": "No three-class SciFact claim gold label or accuracy/F1 is inferred.",
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_revision(project_root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=project_root,
            check=True, capture_output=True, text=True, timeout=5,
        ).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def build_run_manifest(
    *,
    dataset_root: str | Path,
    split: str,
    corpus_path: str | Path,
    claims_path: str | Path,
    output_dir: str | Path,
    project_root: str | Path,
    rag_module: Any,
    verification_config: Mapping[str, Any],
    top_k: Sequence[int],
    run_experiment_a: bool,
    run_experiment_b: bool,
) -> dict[str, Any]:
    """Create a reproducibility manifest without reading gold into runtime."""
    root = Path(project_root).resolve()
    corpus_file = Path(corpus_path).resolve()
    claims_file = Path(claims_path).resolve()
    rag_file = root / "rag.py"
    prompt_files = {
        "experiment_b_verification.py": root / "experiment_b_verification.py",
        "baseline_rag.py": root / "src" / "baseline_rag.py",
    }
    try:
        retrieval_configuration = {
            key: getattr(rag_module, key)
            for key in (
                "EMBEDDING_MODEL_NAME", "RERANKER_MODEL_NAME", "CHILD_CHUNK_SIZE",
                "CHILD_CHUNK_OVERLAP", "VECTOR_TOP_K", "BM25_TOP_K", "EXACT_TOP_K",
                "RRF_K", "RERANK_TOP_K", "FINAL_TOP_K", "VECTOR_RELEVANCE_THRESHOLD",
                "RERANK_MIN_SCORE", "MMR_LAMBDA",
            ) if hasattr(rag, key)
        }
    except Exception:
        retrieval_configuration = {}
    packages = {
        "numpy": _package_version("numpy"),
        "sentence-transformers": _package_version("sentence-transformers"),
        "rank-bm25": _package_version("rank-bm25"),
        "faiss-cpu": _package_version("faiss-cpu"),
        "faiss-gpu": _package_version("faiss-gpu"),
        "langchain-core": _package_version("langchain-core"),
        "langchain-ollama": _package_version("langchain-ollama"),
    }
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "SciFact-Orig",
            "root": str(Path(dataset_root).resolve()),
            "split": split,
            "files": {
                "corpus": {"path": str(corpus_file), "sha256": _sha256(corpus_file)},
                "claims": {"path": str(claims_file), "sha256": _sha256(claims_file)},
            },
        },
        "project": {"root": str(root), "git_commit": _git_revision(root)},
        "frozen_components": {"rag_py_sha256": _sha256(rag_file)},
        "models": {
            "verification_model": verification_config.get("model"),
            "embedding_model": retrieval_configuration.get("EMBEDDING_MODEL_NAME"),
            "reranker_model": retrieval_configuration.get("RERANKER_MODEL_NAME"),
        },
        "retrieval_configuration": retrieval_configuration,
        "verification_configuration": dict(verification_config),
        "prompt_identifiers": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in prompt_files.items() if path.is_file()
        },
        "evaluation": {
            "top_k": list(top_k),
            "run_experiment_a": bool(run_experiment_a),
            "run_experiment_b": bool(run_experiment_b),
            "gold_annotation_data_passed_to_runtime": False,
            "cited_doc_ids_used_for_retrieval": False,
            "three_class_gold_mapping": None,
        },
        "environment": {
            "python": platform.python_version(),
            "packages": packages,
        },
        "output_dir": str(Path(output_dir).resolve()),
    }
