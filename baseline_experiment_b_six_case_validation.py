"""One-pass six-case live Experiment B validation through the real RAG provider.

This is a controlled validation artifact, not an evaluation benchmark. It uses
only local Ollama and in-memory FAISS/BM25 indexes, and stops at the first case
error or verdict mismatch. No production modules are edited by this script.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

import experiment_b_verification as verification_module
import rag
from experiment_b_verification import ExperimentBConfig
from llm import OllamaProvider
from research_pipeline import run_experiment_b
from research_retrieval import ExistingRAGProvider


_ORIGINAL_ANALYSIS = verification_module.analyze_verification_units
_ORIGINAL_VALIDATION = verification_module.validate_judgement
_ORIGINAL_REVISION = verification_module.revise_answer


ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "baseline_experiment_b_six_case_validation_results.jsonl"
SUMMARY_PATH = ROOT / "baseline_experiment_b_six_case_validation_summary.json"

CASES: list[dict[str, Any]] = [
    {
        "case_id": "case_01_direct_support",
        "question": "What does RAG combine?",
        "answer": "RAG combines information retrieval with language generation.",
        "expected_verdict": "SUPPORTED",
        "evidence": [{"source_id": "rag_definition.txt", "page": 1,
                       "text": "Retrieval-Augmented Generation (RAG) combines information retrieval with language generation."}],
    },
    {
        "case_id": "case_02_direct_contradiction",
        "question": "Does BM25 handle image generation?",
        "answer": "The system uses BM25 for image generation.",
        "expected_verdict": "CONTRADICTED",
        "evidence": [{"source_id": "bm25_image_generation.txt", "page": 1,
                       "text": "BM25 is not used for image generation; it is used for lexical document retrieval."}],
    },
    {
        "case_id": "case_03_relevant_insufficient",
        "question": "Did Company A acquire Company B in 2024?",
        "answer": "Company A acquired Company B in 2024.",
        "expected_verdict": "INSUFFICIENT_EVIDENCE",
        "evidence": [{"source_id": "company_partnership.txt", "page": 1,
                       "text": "Company A and Company B announced a strategic partnership in 2024."}],
    },
    {
        "case_id": "case_04_partial_support",
        "question": "Which retrieval methods does the pipeline use?",
        "answer": "The pipeline uses FAISS and BM25 for retrieval.",
        "expected_verdict": "INSUFFICIENT_EVIDENCE",
        "evidence": [{"source_id": "pipeline_faiss.txt", "page": 1,
                       "text": "The pipeline uses FAISS for dense vector retrieval."}],
    },
    {
        "case_id": "case_05_conflicting_evidence",
        "question": "Does System X use BM25 for lexical retrieval?",
        "answer": "System X uses BM25 for lexical retrieval.",
        "expected_verdict": "INSUFFICIENT_EVIDENCE",
        "evidence": [
            {"source_id": "system_x_record_a.txt", "page": 1,
             "text": "System X uses BM25 for lexical retrieval."},
            {"source_id": "system_x_record_b.txt", "page": 1,
             "text": "System X does not use BM25; lexical retrieval is performed with TF-IDF."},
        ],
    },
    {
        "case_id": "case_06_technical_multi_component",
        "question": "How does the retrieval pipeline rank documents?",
        "answer": "The retrieval pipeline combines dense retrieval, BM25 lexical retrieval, and cross-encoder reranking.",
        "expected_verdict": "SUPPORTED",
        "evidence": [{"source_id": "hybrid_pipeline.txt", "page": 1,
                       "text": "The retrieval pipeline combines dense retrieval, BM25 lexical retrieval, and cross-encoder reranking."}],
    },
]


def timed(bucket: list[float], function: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            bucket.append(time.perf_counter() - started)
    return wrapper


def jsonl_append(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def make_provider(case: dict[str, Any], embedding_model: Any, reranker: Any) -> ExistingRAGProvider:
    chunks: list[dict[str, Any]] = []
    for evidence in case["evidence"]:
        chunks.extend(rag.create_chunks(
            [{"page": evidence["page"], "text": evidence["text"]}], evidence["source_id"]
        ))
    texts = [str(chunk.get("text", "")) for chunk in chunks]
    vectors = embedding_model.encode(
        texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    bm25 = BM25Okapi([rag.tokenize_text(text) for text in texts])
    return ExistingRAGProvider(
        chunks=chunks, embedding_model=embedding_model, reranker=reranker,
        faiss_index=index, bm25_index=bm25, retrieval_module=rag,
    )


def run_case(case: dict[str, Any], embedding_model: Any, reranker: Any) -> dict[str, Any]:
    stage_times: dict[str, list[float]] = {
        "verification_unit_analysis": [], "judge": [], "validation": [], "revision": [],
    }
    verification_module.analyze_verification_units = timed(stage_times["verification_unit_analysis"], _ORIGINAL_ANALYSIS)
    verification_module.validate_judgement = timed(stage_times["validation"], _ORIGINAL_VALIDATION)
    verification_module.revise_answer = timed(stage_times["revision"], _ORIGINAL_REVISION)

    provider = make_provider(case, embedding_model, reranker)
    rag_retrieval_times: list[float] = []
    original_retrieve = provider.retrieve

    def retrieve_timed(query: str, limit: int = 8):
        started = time.perf_counter()
        try:
            return original_retrieve(query, limit)
        finally:
            rag_retrieval_times.append(time.perf_counter() - started)

    provider.retrieve = retrieve_timed
    initial_started = time.perf_counter()
    initial_response = provider.retrieve(case["question"], limit=8)
    initial_retrieval_seconds = time.perf_counter() - initial_started

    config = ExperimentBConfig()  # current qwen3:4b, think=false, JSON, temp=0, bounded retries
    config.timeout_seconds = 60  # hard request bound for this one-pass live validation
    judge = OllamaProvider(
        model=config.model, base_url=config.ollama_url,
        think=config.think, format=config.response_format,
    )

    class TimedJudge:
        def generate(self, messages, *, temperature=0.0, timeout=120):
            started = time.perf_counter()
            try:
                return judge.generate(messages, temperature=temperature, timeout=timeout)
            finally:
                stage_times["judge"].append(time.perf_counter() - started)

    started = time.perf_counter()
    result = run_experiment_b(
        case["question"], case["answer"], initial_response.evidence,
        retrieval_provider=provider, llm_provider=TimedJudge(),
        verification_config=config, revise_answer=True,
    )
    experiment_b_seconds = time.perf_counter() - started
    report = result.report
    unit_traces = [row for row in (report.trace if report else []) if row.get("unit")]
    verdict_rows = []
    if report:
        for item in report.verifications:
            verdict_rows.append({
                "unit_id": item.claim_id, "unit_text": item.claim,
                "judge_verdict": item.llm_verdict, "final_verdict": item.verdict,
                "supporting_evidence_ids": [x.evidence_id for x in item.supporting_evidence],
                "contradicting_evidence_ids": [x.evidence_id for x in item.contradicting_evidence],
                "supported_components": item.supported_components,
                "unsupported_components": item.unsupported_components,
                "evidence_assessments": item.evidence_assessments,
                "contradicted_components": [
                    component for row in item.evidence_assessments
                    if str(row.get("interpretation", "")).upper() in {"REFUTES", "CONTRADICTS"}
                    for component in row.get("contradicted_components", [])
                ],
                "validator_overrides": item.validator_overrides,
                "retry_events": item.retry_events,
            })

    retrieval_attempts = [attempt for row in unit_traces for attempt in row.get("retrieval_attempts", [])]
    retries = [attempt for attempt in retrieval_attempts if int(attempt.get("retry", 0)) > 0]
    expected_texts = [item["text"] for item in case["evidence"]]
    retrieved_rows = [item for attempt in retrieval_attempts for item in attempt.get("evidence", [])]
    retrieved_initial_rows = [item.to_dict() for item in initial_response.evidence]
    all_retrieved = retrieved_initial_rows + retrieved_rows
    conflict_status = "not_applicable"
    if case["case_id"] == "case_05_conflicting_evidence":
        interpretations = {
            str(assessment.get("interpretation", "NEUTRAL")).upper()
            for verdict in verdict_rows for assessment in verdict.get("evidence_assessments", [])
        }
        conflict_status = "both_sides_assessed" if (
            interpretations & {"SUPPORTS", "SUPPORT"}
            and interpretations & {"REFUTES", "CONTRADICTS", "CONTRADICT"}
        ) else "not_both_sides_assessed"

    final_verdicts = [row["final_verdict"] for row in verdict_rows]
    observed = final_verdicts[0] if len(final_verdicts) == 1 else final_verdicts
    matched = bool(report) and observed == case["expected_verdict"]
    row = {
        "case_id": case["case_id"], "question": case["question"],
        "original_answer": case["answer"], "expected_verdict": case["expected_verdict"],
        "verification_units": report.verification_units if report else [],
        "evidence_expected": expected_texts,
        "initial_retrieved_evidence": retrieved_initial_rows,
        "unit_retrieval_attempts": retrieval_attempts,
        "all_retrieved_evidence": all_retrieved,
        "retrieved_evidence_ids": sorted({str(item.get("evidence_id", "")) for item in all_retrieved}),
        "intended_evidence_retrieved": all(any(expected == item.get("text") for item in all_retrieved) for expected in expected_texts),
        "verdicts": verdict_rows,
        "validator_results": [trace.get("validator_result") for trace in unit_traces],
        "conflict_status": conflict_status,
        "judge_retry_occurred": any(
            event.get("stage") == "semantic_judge"
            for item in (report.verifications if report else []) for event in item.retry_events
        ),
        "targeted_retrieval_retry_occurred": bool(retries),
        "targeted_retrieval_retry_count": len(retries),
        "retry_reasons": [reason for trace in unit_traces for reason in trace.get("retry_reasons", [])],
        "revision_occurred": bool(report and report.revised_answer != case["answer"]),
        "revised_answer": report.revised_answer if report else None,
        "final_answer": result.final_answer,
        "errors": ([result.error] if result.error else []) + (report.errors if report else []),
        "timings_seconds": {
            "verification_unit_analysis": sum(stage_times["verification_unit_analysis"]),
            "initial_retrieval": initial_retrieval_seconds,
            "unit_specific_retrieval": sum(rag_retrieval_times[1:len(unit_traces) + 1]),
            "judge": sum(stage_times["judge"]),
            "deterministic_validation": sum(stage_times["validation"]),
            "targeted_retrieval_retry": sum(rag_retrieval_times[len(unit_traces) + 1:]) if retries else 0.0,
            "revision": sum(stage_times["revision"]),
            "total_experiment_b": experiment_b_seconds,
            "judge_calls": len(stage_times["judge"]),
            "rag_retrieval_calls_including_initial": len(rag_retrieval_times),
            "rag_retrieval_calls_in_pipeline": max(0, len(rag_retrieval_times) - 1),
        },
        "ollama_configuration": {
            "model": config.model, "think": config.think,
            "format": config.response_format, "temperature": config.temperature,
            "timeout_seconds": config.timeout_seconds,
            "max_judge_retries": config.max_judge_retries,
            "max_targeted_retrieval_retries": config.max_targeted_retrieval_retries,
        },
        "error": result.error,
        "matches_expected_verdict": matched,
    }
    return row


def main() -> int:
    RESULTS_PATH.write_text("", encoding="utf-8")
    started_at = datetime.now(timezone.utc).isoformat()
    print(f"Started UTC: {started_at}", flush=True)
    print("Loading the existing local embedding and reranker models...", flush=True)
    embedding_model = SentenceTransformer(rag.EMBEDDING_MODEL_NAME)
    reranker = CrossEncoder(rag.RERANKER_MODEL_NAME)

    results: list[dict[str, Any]] = []
    stop_reason = None
    for case in CASES:
        print(f"Running {case['case_id']} through ExistingRAGProvider...", flush=True)
        try:
            row = run_case(case, embedding_model, reranker)
        except Exception as exc:
            row = {
                "case_id": case["case_id"], "question": case["question"],
                "original_answer": case["answer"], "expected_verdict": case["expected_verdict"],
                "error": f"{type(exc).__name__}: {exc}", "matches_expected_verdict": False,
            }
        results.append(row)
        jsonl_append(RESULTS_PATH, row)
        print(json.dumps({
            "case_id": row["case_id"], "expected": row["expected_verdict"],
            "observed": [v["final_verdict"] for v in row.get("verdicts", [])],
            "matched": row["matches_expected_verdict"], "error": row.get("error"),
            "timings_seconds": row.get("timings_seconds"),
        }, ensure_ascii=False), flush=True)
        if not row["matches_expected_verdict"]:
            stop_reason = "Stopped at first case error or verdict mismatch; no implementation tuning performed."
            break

    runtimes = [row.get("timings_seconds", {}).get("total_experiment_b") for row in results]
    runtimes = [float(value) for value in runtimes if value is not None]
    summary = {
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "cases_requested": len(CASES), "cases_completed": len(results),
        "matches": sum(bool(row.get("matches_expected_verdict")) for row in results),
        "mismatches_or_errors": sum(not bool(row.get("matches_expected_verdict")) for row in results),
        "stop_reason": stop_reason,
        "average_experiment_b_seconds": statistics.mean(runtimes) if runtimes else None,
        "median_experiment_b_seconds": statistics.median(runtimes) if runtimes else None,
        "total_rag_calls": sum(int(row.get("timings_seconds", {}).get("rag_retrieval_calls_including_initial", 0)) for row in results),
        "total_llm_calls": sum(int(row.get("timings_seconds", {}).get("judge_calls", 0)) for row in results),
        "results": [{
            "case_id": row["case_id"], "expected_verdict": row["expected_verdict"],
            "observed_verdicts": [item["final_verdict"] for item in row.get("verdicts", [])],
            "matches_expected_verdict": row["matches_expected_verdict"],
            "retry": row.get("judge_retry_occurred", False) or row.get("targeted_retrieval_retry_occurred", False),
            "errors": row.get("errors", [row.get("error")] if row.get("error") else []),
        } for row in results],
        "artifacts": {"raw_jsonl": RESULTS_PATH.name, "summary_json": SUMMARY_PATH.name},
        "production_source_modified": False,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if not summary["mismatches_or_errors"] else 1


if __name__ == "__main__":
    sys.exit(main())
