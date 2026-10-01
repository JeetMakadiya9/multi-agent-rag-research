"""Isolated positional-evidence judge validation; does not change production code."""
from __future__ import annotations

import ast
import hashlib
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

import rag
from claim_verification import (
    CONTRADICTED, INSUFFICIENT_EVIDENCE, SUPPORTED, ClaimVerification,
    EvidenceItem, evidence_from_retrieval_results,
)
from experiment_b_verification import (
    ExperimentBConfig, build_unit_retrieval_query, revise_answer, validate_judgement,
)
from research_retrieval import ExistingRAGProvider
from verification_units import analyze_verification_units

ROOT = Path(__file__).resolve().parent
PREVIOUS_CASES = ROOT / "baseline_experiment_b_six_case_validation.py"
RESULTS_PATH = ROOT / "baseline_experiment_b_judge_v2_validation_results.jsonl"
SUMMARY_PATH = ROOT / "baseline_experiment_b_judge_v2_validation_summary.json"
JUDGE_URL = "http://127.0.0.1:11434/api/chat"
MODEL = "qwen3:4b"
TIMEOUT = 60
MAX_EVIDENCE = 8

PROTECTED_FILES = [
    "rag.py", "claim_verification.py", "research_pipeline.py", "llm.py",
    "research_state.py", "research_tools.py", "research_retrieval.py",
    "experiment_b_verification.py", "verification_units.py",
    "tests/test_experiment_b.py", "tests/test_experiment_b_verification.py",
    "baseline_rag_test.py", "baseline_rag_audit.py",
    "baseline_experiment_b_six_case_validation.py",
    "baseline_experiment_b_six_case_validation_results.jsonl",
    "baseline_experiment_b_six_case_validation_summary.json",
    "baseline_experiment_b_case2_judge_diagnostic.py",
    "baseline_experiment_b_case2_judge_diagnostic.json",
    "baseline_case2_judge_interface_benchmark.py",
    "baseline_case2_judge_interface_results.jsonl",
    "baseline_case2_judge_interface_summary.json",
    "evaluation/experiment_b/development.jsonl",
    "evaluation/experiment_b/validation.jsonl",
    "evaluation/experiment_b/heldout.jsonl",
]


def sha256_snapshot() -> dict[str, str | None]:
    snapshot: dict[str, str | None] = {}
    for relative in PROTECTED_FILES:
        path = ROOT / relative
        snapshot[relative] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    return snapshot


def load_exact_cases() -> list[dict[str, Any]]:
    tree = ast.parse(PREVIOUS_CASES.read_text(encoding="utf-8"), filename=str(PREVIOUS_CASES))
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "CASES":
            cases = ast.literal_eval(node.value)
            if len(cases) != 6:
                raise ValueError(f"Expected six existing cases, found {len(cases)}")
            return cases
    raise ValueError("Could not locate literal CASES assignment in previous validation script")


def make_provider(case: dict[str, Any], embedder: Any, reranker: Any) -> ExistingRAGProvider:
    chunks: list[dict[str, Any]] = []
    for evidence in case["evidence"]:
        chunks.extend(rag.create_chunks(
            [{"page": evidence["page"], "text": evidence["text"]}], evidence["source_id"]
        ))
    vectors = embedder.encode(
        [str(chunk.get("text", "")) for chunk in chunks], show_progress_bar=False,
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype("float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    bm25 = BM25Okapi([rag.tokenize_text(str(chunk.get("text", ""))) for chunk in chunks])
    return ExistingRAGProvider(
        chunks=chunks, embedding_model=embedder, reranker=reranker,
        faiss_index=index, bm25_index=bm25, retrieval_module=rag,
    )


def call_ollama(messages: list[dict[str, str]], temperature: float) -> dict[str, Any]:
    payload = {
        "model": MODEL, "messages": messages, "stream": True,
        "think": False, "format": "json", "options": {"temperature": temperature},
    }
    request = urllib.request.Request(
        JUDGE_URL, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    started = time.perf_counter()
    raw_lines: list[str] = []
    content_parts: list[str] = []
    status: int | None = None
    completed = False
    exception: str | None = None
    stream_events: list[Any] = []
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            status = int(response.status)
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace")
                raw_lines.append(line)
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                    stream_events.append(event)
                except json.JSONDecodeError:
                    continue
                if event.get("error"):
                    exception = f"Ollama stream error: {event['error']}"
                message = event.get("message", {})
                if isinstance(message, dict) and message.get("content"):
                    content_parts.append(str(message["content"]))
                if event.get("done") is True:
                    completed = True
                    break
    except urllib.error.HTTPError as exc:
        status = int(exc.code)
        try:
            raw_lines.append(exc.read().decode("utf-8", errors="replace"))
        except Exception:
            pass
        exception = f"HTTPError: {exc}"
    except Exception as exc:  # Record the actual transport/stream exception.
        exception = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - started
    raw_text = "".join(content_parts)
    parsed = None
    parse_exception = None
    try:
        parsed = json.loads(raw_text)
    except Exception as exc:
        parse_exception = f"{type(exc).__name__}: {exc}"
    return {
        "http_status": status, "stream_completed": completed,
        "raw_response_text": raw_text, "raw_stream_payload_lines": raw_lines,
        "raw_stream_payload_exact_join": "".join(raw_lines),
        "parsed_json": parsed, "json_parse_success": isinstance(parsed, dict),
        "parse_exception": parse_exception, "transport_exception": exception,
        "elapsed_seconds": elapsed, "stream_events": stream_events,
        "request": payload,
    }


def judge_messages(question: str, unit_text: str, evidence: list[EvidenceItem]) -> tuple[list[dict[str, str]], dict[str, str]]:
    mapping = {f"E{i}": item.evidence_id for i, item in enumerate(evidence, start=1)}
    evidence_text = "\n\n".join(
        f"Evidence {position}:\n{item.text}" for position, item in zip(mapping, evidence)
    )
    system = (
        "You are an evidence-grounded verification judge. Judge only the supplied verification unit and evidence. "
        "Relevance is not entailment. Missing evidence is not contradiction. Preserve all qualifiers and dependencies. "
        "Partial support is INSUFFICIENT_EVIDENCE. Do not invent facts or evidence references. "
        "The verdict field must contain exactly one of these values: SUPPORTED, CONTRADICTED, INSUFFICIENT_EVIDENCE. "
        "Use only positional evidence references (E1, E2, ...) that appear below. "
        "Return a JSON object with exactly these fields: verdict, evidence_indices, reason. "
        "evidence_indices must be an array of positional references; reason must be a string. "
        "Example format for an unrelated hypothetical: {\"verdict\":\"SUPPORTED\",\"evidence_indices\":[\"E1\"],\"reason\":\"Evidence establishes the proposition.\"}. "
        "For the current case, independently decide from its evidence."
    )
    user = (
        f"Original question: {question}\nVerification unit: {unit_text}\n\n"
        f"Supplied evidence:\n{evidence_text}\n\n"
        "Return only the required JSON object. Cite evidence using its E-number, never a source identifier."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}], mapping


def validate_response(parsed: Any, mapping: dict[str, str]) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(parsed, dict):
        return {"required_fields_valid": False, "evidence_references_valid": False,
                "errors": ["JSON root must be an object"], "verdict": None, "mapped_ids": []}
    required = {"verdict", "evidence_indices", "reason"}
    missing = sorted(required - set(parsed))
    extras = sorted(set(parsed) - required)
    if missing:
        errors.append("Missing required fields: " + ", ".join(missing))
    if extras:
        errors.append("Unexpected fields: " + ", ".join(extras))
    verdict = parsed.get("verdict")
    if not isinstance(verdict, str) or verdict not in {SUPPORTED, CONTRADICTED, INSUFFICIENT_EVIDENCE}:
        errors.append("Invalid verdict value")
    refs = parsed.get("evidence_indices")
    if not isinstance(refs, list) or not all(isinstance(ref, str) for ref in refs):
        errors.append("evidence_indices must be an array of strings")
        refs = []
    invalid_refs = [ref for ref in refs if ref not in mapping]
    if invalid_refs:
        errors.append("Invalid evidence reference(s): " + ", ".join(invalid_refs))
    if not isinstance(parsed.get("reason"), str):
        errors.append("reason must be a string")
    return {
        "required_fields_valid": not missing and not extras and isinstance(parsed.get("reason"), str),
        "evidence_references_valid": not invalid_refs and isinstance(parsed.get("evidence_indices"), list),
        "errors": errors, "verdict": verdict if isinstance(verdict, str) else None,
        "mapped_ids": [mapping[ref] for ref in refs if ref in mapping],
        "invalid_references": invalid_refs,
    }


def retrieve(provider: ExistingRAGProvider, query: str, timings: dict[str, float], calls: list[dict[str, Any]], kind: str) -> list[EvidenceItem]:
    start = time.perf_counter()
    response = provider.retrieve(query, limit=MAX_EVIDENCE)
    elapsed = time.perf_counter() - start
    timings[kind] = timings.get(kind, 0.0) + elapsed
    calls.append({"kind": kind, "query": query, "elapsed_seconds": elapsed, "provider": response.provider,
                  "diagnostics": response.diagnostics, "evidence": [item.to_dict() for item in response.evidence]})
    return evidence_from_retrieval_results(response.evidence, max_items=MAX_EVIDENCE)


def dedupe(groups: list[list[EvidenceItem]]) -> list[EvidenceItem]:
    merged: list[EvidenceItem] = []
    ids: set[str] = set()
    for group in groups:
        for item in group:
            if item.evidence_id not in ids:
                ids.add(item.evidence_id)
                merged.append(item)
    return merged[:MAX_EVIDENCE]


def run_judge(unit: Any, question: str, evidence: list[EvidenceItem], retry_index: int,
              timings: dict[str, float], raw_attempts: list[dict[str, Any]]) -> dict[str, Any]:
    messages, mapping = judge_messages(question, unit.original_text, evidence)
    config = ExperimentBConfig(timeout_seconds=TIMEOUT)
    # Match the existing bounded policy: one additional judge attempt only for parse/schema failure.
    last_validation: dict[str, Any] = {}
    for attempt in range(config.max_judge_retries + 1):
        call = call_ollama(messages, temperature=config.temperature)
        timings["judge"] = timings.get("judge", 0.0) + call["elapsed_seconds"]
        parsed = call["parsed_json"]
        validation = validate_response(parsed, mapping)
        call.update({
            "case_stage": "judge_retry" if retry_index else "initial_judge",
            "attempt_number": attempt + 1, "targeted_retrieval_retry_number": retry_index,
            "evidence_position_to_real_id": mapping, "output_validation": validation,
        })
        raw_attempts.append(call)
        last_validation = validation
        retryable = not call["json_parse_success"] or not validation["required_fields_valid"]
        if not retryable or attempt >= config.max_judge_retries:
            return {"call": call, "parsed": parsed, "validation": validation, "mapping": mapping}
    raise RuntimeError("unreachable judge retry state")


def run_case(case: dict[str, Any], embedder: Any, reranker: Any) -> dict[str, Any]:
    total_start = time.perf_counter()
    timings: dict[str, float] = {}
    retrieval_calls: list[dict[str, Any]] = []
    judge_attempts: list[dict[str, Any]] = []
    provider = make_provider(case, embedder, reranker)
    initial = retrieve(provider, case["question"], timings, retrieval_calls, "initial_retrieval")
    analysis_start = time.perf_counter()
    units = analyze_verification_units(case["answer"])
    timings["verification_unit_analysis"] = time.perf_counter() - analysis_start
    final_units: list[dict[str, Any]] = []
    retry_count = 0
    errors: list[str] = []
    for unit in units:
        query = build_unit_retrieval_query(case["question"], unit)
        targeted = retrieve(provider, query, timings, retrieval_calls, "unit_specific_retrieval")
        evidence = dedupe([targeted, initial])
        unit_result: dict[str, Any] = {"unit": unit.to_dict(), "retrieval_query": query,
                                       "retrieved_evidence": [asdict(item) for item in evidence],
                                       "targeted_retrieval_attempts": []}
        judgement = run_judge(unit, case["question"], evidence, 0, timings, judge_attempts)
        verdict_data = judgement["parsed"] if isinstance(judgement["parsed"], dict) else {}
        output_validation = judgement["validation"]
        mapped_ids = output_validation.get("mapped_ids", [])
        final_verdict = output_validation.get("verdict")
        validator_details = None
        if output_validation["required_fields_valid"] and output_validation["evidence_references_valid"] and final_verdict in {SUPPORTED, CONTRADICTED, INSUFFICIENT_EVIDENCE}:
            normalized = {
                "verdict": final_verdict,
                "supporting_evidence_ids": mapped_ids if final_verdict == SUPPORTED else [],
                "contradicting_evidence_ids": mapped_ids if final_verdict == CONTRADICTED else [],
                "supported_components": [], "unsupported_components": [],
                "evidence_assessments": [],
            }
            validation_start = time.perf_counter()
            validator_result = validate_judgement(unit, normalized, evidence, enabled=True)
            timings["deterministic_validation"] = timings.get("deterministic_validation", 0.0) + time.perf_counter() - validation_start
            validator_details = {"llm_verdict": final_verdict, "verdict": validator_result.verdict,
                                 "overrides": validator_result.overrides,
                                 "conflict_resolution": validator_result.conflict_resolution}
            final_verdict = validator_result.verdict
        else:
            errors.extend(output_validation.get("errors", []))
            validator_details = {"reached": False, "reason": "judge output failed contract/reference validation"}

        unit_result["initial_judgement"] = {
            "raw_verdict": verdict_data.get("verdict"), "normalized_verdict": final_verdict,
            "reason": verdict_data.get("reason"), "evidence_indices": verdict_data.get("evidence_indices"),
            "mapped_real_evidence_ids": mapped_ids, "output_validation": output_validation,
            "validator_result": validator_details,
        }

        if final_verdict == INSUFFICIENT_EVIDENCE and retry_count < 1:
            retry_count += 1
            missing = unit.required_evidence or ["claim relationship and qualifiers"]
            retry_query = build_unit_retrieval_query(case["question"], unit, missing_components=missing)
            if retry_query.casefold().strip() == query.casefold().strip():
                retry_query = "Missing evidence detail: " + " ".join(missing) + " Context: " + case["question"]
            additional = retrieve(provider, retry_query, timings, retrieval_calls, "targeted_retrieval_retry")
            evidence_after_retry = dedupe([additional, evidence])
            unit_result["targeted_retrieval_attempts"].append({"query": retry_query,
                                                               "evidence": [asdict(item) for item in additional]})
            if additional:
                retry_judgement = run_judge(unit, case["question"], evidence_after_retry, 1, timings, judge_attempts)
                retry_data = retry_judgement["parsed"] if isinstance(retry_judgement["parsed"], dict) else {}
                retry_validation = retry_judgement["validation"]
                retry_ids = retry_validation.get("mapped_ids", [])
                retry_verdict = retry_validation.get("verdict")
                retry_validator = None
                if retry_validation["required_fields_valid"] and retry_validation["evidence_references_valid"] and retry_verdict in {SUPPORTED, CONTRADICTED, INSUFFICIENT_EVIDENCE}:
                    normalized_retry = {
                        "verdict": retry_verdict,
                        "supporting_evidence_ids": retry_ids if retry_verdict == SUPPORTED else [],
                        "contradicting_evidence_ids": retry_ids if retry_verdict == CONTRADICTED else [],
                        "supported_components": [], "unsupported_components": [], "evidence_assessments": [],
                    }
                    val_start = time.perf_counter()
                    val = validate_judgement(unit, normalized_retry, evidence_after_retry, enabled=True)
                    timings["deterministic_validation"] = timings.get("deterministic_validation", 0.0) + time.perf_counter() - val_start
                    retry_validator = {"llm_verdict": retry_verdict, "verdict": val.verdict,
                                       "overrides": val.overrides, "conflict_resolution": val.conflict_resolution}
                    retry_verdict = val.verdict
                else:
                    errors.extend(retry_validation.get("errors", []))
                    retry_validator = {"reached": False, "reason": "judge output failed contract/reference validation"}
                unit_result["retry_judgement"] = {
                    "raw_verdict": retry_data.get("verdict"), "normalized_verdict": retry_verdict,
                    "reason": retry_data.get("reason"), "evidence_indices": retry_data.get("evidence_indices"),
                    "mapped_real_evidence_ids": retry_ids, "output_validation": retry_validation,
                    "validator_result": retry_validator,
                }
                final_verdict = retry_verdict
                evidence = evidence_after_retry
            else:
                unit_result["retry_judgement"] = None

        unit_result["final_verdict"] = final_verdict or INSUFFICIENT_EVIDENCE
        unit_result["validator_result"] = unit_result.get("retry_judgement", {}).get("validator_result") if unit_result.get("retry_judgement") else validator_details
        unit_result["evidence_after_retry"] = [asdict(item) for item in evidence]
        final_units.append(unit_result)

    verification_objects: list[ClaimVerification] = []
    for unit_result in final_units:
        row = unit_result.get("retry_judgement") or unit_result.get("initial_judgement", {})
        verdict = unit_result["final_verdict"]
        ids = row.get("mapped_real_evidence_ids", [])
        evidence = [EvidenceItem(**{k: v for k, v in item.items() if k in EvidenceItem.__dataclass_fields__})
                    for item in unit_result["evidence_after_retry"]]
        refs = [item for item in evidence if item.evidence_id in ids]
        verification_objects.append(ClaimVerification(
            claim_id=unit_result["unit"]["id"], claim=unit_result["unit"]["original_text"],
            verdict=verdict, confidence=0.0,
            supporting_evidence=refs if verdict == SUPPORTED else [],
            contradicting_evidence=refs if verdict == CONTRADICTED else [],
            explanation=str(row.get("reason", "")), llm_verdict=row.get("raw_verdict"),
        ))
    revise_start = time.perf_counter()
    final_answer = revise_answer(case["answer"], verification_objects)
    timings["answer_revision"] = time.perf_counter() - revise_start
    timings["total_runtime"] = time.perf_counter() - total_start
    observed = [unit["final_verdict"] for unit in final_units]
    matches = observed == [case["expected_verdict"]] * len(observed) and len(observed) > 0
    return {
        "case_id": case["case_id"], "question": case["question"],
        "original_answer": case["answer"], "expected_verdict": case["expected_verdict"],
        "verification_units": [unit.to_dict() for unit in units],
        "expected_evidence_slice": case["evidence"],
        "initial_retrieval": retrieval_calls[0] if retrieval_calls else None,
        "unit_specific_retrievals": [row for row in retrieval_calls if row["kind"] == "unit_specific_retrieval"],
        "retry_retrievals": [row for row in retrieval_calls if row["kind"] == "targeted_retrieval_retry"],
        "all_retrieval_calls": retrieval_calls,
        "judge_attempts": judge_attempts,
        "final_unit_results": final_units, "observed_verdicts": observed,
        "matches_expected_verdict": matches, "revision_occurred": final_answer != case["answer"],
        "final_answer": final_answer, "errors": errors,
        "retry_count": retry_count, "rag_call_count": len(retrieval_calls),
        "llm_call_count": len(judge_attempts), "timings_seconds": timings,
        "configuration": {"model": MODEL, "think": False, "temperature": 0.0,
                           "format": "json", "timeout_seconds": TIMEOUT,
                           "max_judge_retries": 1, "max_targeted_retrieval_retries": 1},
    }


def main() -> int:
    # These exact v2 paths are the artifacts created by this requested run;
    # the prior six-case validation uses different filenames and is protected.
    before = sha256_snapshot()
    started = datetime.now(timezone.utc).isoformat()
    cases = load_exact_cases()
    RESULTS_PATH.write_text("", encoding="utf-8")
    print(f"Started UTC: {started}", flush=True)
    print(f"Interpreter: {sys.executable}", flush=True)
    print("Loading existing local embedding and cross-encoder models...", flush=True)
    embedder = SentenceTransformer(rag.EMBEDDING_MODEL_NAME)
    reranker = CrossEncoder(rag.RERANKER_MODEL_NAME)
    rows: list[dict[str, Any]] = []
    stop_reason = None
    for case in cases:
        print(f"Running {case['case_id']} with isolated evidence slice", flush=True)
        try:
            row = run_case(case, embedder, reranker)
        except Exception as exc:
            row = {"case_id": case["case_id"], "question": case["question"],
                   "expected_verdict": case["expected_verdict"], "matches_expected_verdict": False,
                   "errors": [f"{type(exc).__name__}: {exc}"], "judge_attempts": [],
                   "all_retrieval_calls": []}
        rows.append(row)
        with RESULTS_PATH.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        print(json.dumps({"case_id": row["case_id"], "expected": row["expected_verdict"],
                          "observed": row.get("observed_verdicts"),
                          "matches": row["matches_expected_verdict"], "errors": row.get("errors"),
                          "timings_seconds": row.get("timings_seconds")}, ensure_ascii=False), flush=True)
        if not row["matches_expected_verdict"]:
            stop_reason = f"Stopped on first case mismatch or error: {case['case_id']}"
            break
    after = sha256_snapshot()
    changed = [name for name in before if before[name] != after[name]]
    latencies = [attempt["elapsed_seconds"] for row in rows for attempt in row.get("judge_attempts", [])]
    summary = {
        "started_at_utc": started, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "interpreter": sys.executable, "cases_requested": len(cases), "cases_completed": len(rows),
        "cases_matching_expected": sum(bool(row.get("matches_expected_verdict")) for row in rows),
        "first_failure": next((row["case_id"] for row in rows if not row.get("matches_expected_verdict")), None),
        "stop_reason": stop_reason, "placeholder_echo_observed": any(
            "SUPPORTED |" in attempt.get("raw_response_text", "") for row in rows for attempt in row.get("judge_attempts", [])),
        "invalid_evidence_references": [
            {"case_id": row["case_id"], "attempt_number": attempt.get("attempt_number"),
             "invalid_references": attempt.get("output_validation", {}).get("invalid_references", [])}
            for row in rows for attempt in row.get("judge_attempts", [])
            if attempt.get("output_validation", {}).get("invalid_references")
        ],
        "positional_reference_validity": "all valid" if not any(
            attempt.get("output_validation", {}).get("invalid_references") for row in rows for attempt in row.get("judge_attempts", [])) else "invalid reference(s) observed",
        "mean_judge_latency_seconds": statistics.mean(latencies) if latencies else None,
        "median_judge_latency_seconds": statistics.median(latencies) if latencies else None,
        "min_judge_latency_seconds": min(latencies) if latencies else None,
        "max_judge_latency_seconds": max(latencies) if latencies else None,
        "total_rag_calls": sum(int(row.get("rag_call_count", 0)) for row in rows),
        "total_llm_calls": sum(int(row.get("llm_call_count", 0)) for row in rows),
        "case_results": [{"case_id": row["case_id"], "expected": row["expected_verdict"],
                          "observed": row.get("observed_verdicts"), "match": row.get("matches_expected_verdict"),
                          "retry_count": row.get("retry_count"), "errors": row.get("errors", [])} for row in rows],
        "protected_sha256_before": before, "protected_sha256_after": after,
        "protected_hash_changes": changed, "production_files_modified": bool(changed),
        "artifacts": [RESULTS_PATH.name, SUMMARY_PATH.name],
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if not changed and not stop_reason else 1


if __name__ == "__main__":
    raise SystemExit(main())
