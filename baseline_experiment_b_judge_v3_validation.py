"""Isolated six-case per-evidence stance validation; no production changes."""
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
from claim_verification import EvidenceItem, evidence_from_retrieval_results
from experiment_b_verification import ExperimentBConfig, build_unit_retrieval_query
from research_retrieval import ExistingRAGProvider
from verification_units import VerificationUnit, analyze_verification_units

ROOT = Path(__file__).resolve().parent
CASES_SOURCE = ROOT / "baseline_experiment_b_six_case_validation.py"
RESULTS_PATH = ROOT / "baseline_experiment_b_judge_v3_validation_results.jsonl"
SUMMARY_PATH = ROOT / "baseline_experiment_b_judge_v3_validation_summary.json"
MODEL = "qwen3:4b"
URL = "http://127.0.0.1:11434/api/chat"
TIMEOUT_SECONDS = 60
MAX_EVIDENCE = 8
MAX_TARGETED_RETRIES = 1
MAX_JUDGE_RETRIES = 1
ALLOWED_STANCES = {"SUPPORTS", "CONTRADICTS", "NEUTRAL"}

PROTECTED_PRODUCTION = [
    "rag.py", "claim_verification.py", "research_pipeline.py", "llm.py",
    "research_state.py", "research_tools.py", "research_retrieval.py",
    "experiment_b_verification.py", "verification_units.py",
]
OWN_ARTIFACT_NAMES = {
    "baseline_experiment_b_judge_v3_validation.py",
    "baseline_experiment_b_judge_v3_validation_results.jsonl",
    "baseline_experiment_b_judge_v3_validation_summary.json",
}


def integrity_inventory() -> list[Path]:
    files = [ROOT / name for name in PROTECTED_PRODUCTION]
    tests = ROOT / "tests"
    if tests.exists():
        files.extend(path for path in tests.rglob("*") if path.is_file())
    # All earlier root-level baseline artifacts are protected during this run.
    files.extend(path for path in ROOT.glob("baseline_*")
                 if path.is_file() and path.name not in OWN_ARTIFACT_NAMES)
    evaluation = ROOT / "evaluation"
    if evaluation.exists():
        files.extend(path for path in evaluation.rglob("*") if path.is_file())
    return sorted(set(path for path in files if path.is_file()))


def hash_snapshot() -> dict[str, str]:
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in integrity_inventory()
    }


def load_cases_exactly() -> list[dict[str, Any]]:
    tree = ast.parse(CASES_SOURCE.read_text(encoding="utf-8"), filename=str(CASES_SOURCE))
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "CASES":
            cases = ast.literal_eval(node.value)
            if len(cases) != 6:
                raise ValueError(f"Expected six prior controlled cases, found {len(cases)}")
            return cases
    raise ValueError("Literal CASES assignment not found in the existing validation script")


def build_provider(case: dict[str, Any], embedder: Any, reranker: Any) -> ExistingRAGProvider:
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
    lexical = BM25Okapi([rag.tokenize_text(str(chunk.get("text", ""))) for chunk in chunks])
    return ExistingRAGProvider(
        chunks=chunks, embedding_model=embedder, reranker=reranker,
        faiss_index=index, bm25_index=lexical, retrieval_module=rag,
    )


def timed_retrieve(provider: ExistingRAGProvider, query: str, stage: str,
                   timing: dict[str, float], calls: list[dict[str, Any]]) -> list[EvidenceItem]:
    start = time.perf_counter()
    response = provider.retrieve(query, limit=MAX_EVIDENCE)
    elapsed = time.perf_counter() - start
    timing[stage] = timing.get(stage, 0.0) + elapsed
    calls.append({
        "stage": stage, "query": query, "provider": response.provider,
        "elapsed_seconds": elapsed, "diagnostics": response.diagnostics,
        "evidence": [item.to_dict() for item in response.evidence],
    })
    return evidence_from_retrieval_results(response.evidence, max_items=MAX_EVIDENCE)


def merge_evidence(*groups: list[EvidenceItem]) -> list[EvidenceItem]:
    merged: list[EvidenceItem] = []
    seen: set[str] = set()
    for group in groups:
        for item in group:
            if item.evidence_id not in seen:
                seen.add(item.evidence_id)
                merged.append(item)
    return merged[:MAX_EVIDENCE]


def build_messages(question: str, unit: VerificationUnit,
                   evidence: list[EvidenceItem]) -> tuple[list[dict[str, str]], dict[str, str]]:
    mapping = {f"E{i}": item.evidence_id for i, item in enumerate(evidence, start=1)}
    blocks = [f"{key}:\n{item.text}" for key, item in zip(mapping, evidence)]
    system = (
        "Assess each supplied evidence item independently against the complete verification unit. "
        "Do not produce an aggregate or final verdict. "
        "Use SUPPORTS only when this evidence item supports the complete verification unit, including its material components and qualifiers. "
        "Use CONTRADICTS only when this evidence item directly conflicts with a material part of the unit. "
        "Use NEUTRAL when the item is irrelevant, ambiguous, or only partially related without establishing or directly conflicting with the complete unit. "
        "Allowed stance values are SUPPORTS, CONTRADICTS, and NEUTRAL. "
        "Use only the positional evidence IDs supplied below. Do not invent or output source IDs. "
        "Return only JSON with exactly two top-level fields: evidence_assessments and reason. "
        "Include exactly one assessment for each supplied evidence ID. Each assessment has exactly evidence_index and stance. "
        "The reason is explanatory only; do not include a final verdict. "
        'Example structure: {"evidence_assessments":[{"evidence_index":"E1","stance":"NEUTRAL"}],'
        '"reason":"This evidence does not establish or conflict with the complete unit."}'
    )
    user = (
        f"Original question: {question}\nVerification unit: {unit.original_text}\n\n"
        "Evidence items:\n" + "\n\n".join(blocks) +
        "\n\nReturn individual evidence assessments only, following the required JSON structure."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}], mapping


def call_ollama(messages: list[dict[str, str]], temperature: float) -> dict[str, Any]:
    request_body = {
        "model": MODEL, "messages": messages, "stream": True, "think": False,
        "format": "json", "options": {"temperature": temperature},
    }
    request = urllib.request.Request(
        URL, data=json.dumps(request_body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    started = time.perf_counter()
    status: int | None = None
    raw_lines: list[str] = []
    response_parts: list[str] = []
    events: list[Any] = []
    completed = False
    transport_error: str | None = None
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            status = int(response.status)
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace")
                raw_lines.append(line)
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                    events.append(event)
                except json.JSONDecodeError:
                    continue
                if event.get("error"):
                    transport_error = f"Ollama stream error: {event['error']}"
                message = event.get("message")
                if isinstance(message, dict) and message.get("content"):
                    response_parts.append(str(message["content"]))
                if event.get("done") is True:
                    completed = True
                    break
    except urllib.error.HTTPError as exc:
        status = int(exc.code)
        try:
            raw_lines.append(exc.read().decode("utf-8", errors="replace"))
        except Exception:
            pass
        transport_error = f"HTTPError: {exc}"
    except Exception as exc:
        transport_error = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - started
    raw_text = "".join(response_parts)
    parsed: Any = None
    parse_exception = None
    try:
        parsed = json.loads(raw_text)
    except Exception as exc:
        parse_exception = f"{type(exc).__name__}: {exc}"
    return {
        "http_status": status, "stream_completed": completed,
        "elapsed_seconds": elapsed, "raw_response_text": raw_text,
        "raw_stream_payload_lines": raw_lines,
        "raw_stream_payload_exact_join": "".join(raw_lines),
        "parsed_json": parsed, "json_parse_success": isinstance(parsed, dict),
        "parse_exception": parse_exception, "transport_exception": transport_error,
        "stream_events": events, "request": request_body,
    }


def validate_assessments(parsed: Any, positional_ids: list[str]) -> dict[str, Any]:
    errors: list[str] = []
    invalid_indices: list[str] = []
    invalid_stances: list[str] = []
    assessments: list[dict[str, Any]] = []
    if not isinstance(parsed, dict):
        return {"schema_valid": False, "validation_errors": ["JSON root must be an object"],
                "evidence_assessments": [], "evidence_indices": [], "stance_values": [],
                "invalid_evidence_indices": [], "invalid_stance_values": [],
                "deterministic_verdict": "INSUFFICIENT_EVIDENCE"}
    required_top = {"evidence_assessments", "reason"}
    missing = sorted(required_top - set(parsed))
    extras = sorted(set(parsed) - required_top)
    if missing:
        errors.append("Missing required top-level field(s): " + ", ".join(missing))
    if extras:
        errors.append("Unexpected top-level field(s): " + ", ".join(extras))
    if not isinstance(parsed.get("reason"), str):
        errors.append("reason must be a string")
    rows = parsed.get("evidence_assessments")
    if not isinstance(rows, list):
        errors.append("evidence_assessments must be an array")
        rows = []
    seen: list[str] = []
    valid_stances: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"Assessment {index} must be an object")
            continue
        if set(row) != {"evidence_index", "stance"}:
            errors.append(f"Assessment {index} must contain exactly evidence_index and stance")
        ref = row.get("evidence_index")
        stance = row.get("stance")
        if not isinstance(ref, str) or ref not in positional_ids:
            invalid_indices.append(str(ref))
            errors.append(f"Invalid evidence index: {ref!r}")
        else:
            seen.append(ref)
        if not isinstance(stance, str) or stance not in ALLOWED_STANCES:
            invalid_stances.append(str(stance))
            errors.append(f"Invalid stance value: {stance!r}")
        elif isinstance(ref, str) and ref in positional_ids:
            valid_stances.append(stance)
        assessments.append({"evidence_index": ref, "stance": stance})
    if sorted(seen) != sorted(positional_ids):
        errors.append("Assessment coverage does not include each supplied positional evidence ID exactly once")
    if len(seen) != len(set(seen)):
        errors.append("Duplicate positional evidence index")
    coverage_valid = (
        not invalid_indices and not invalid_stances and sorted(seen) == sorted(positional_ids)
        and len(seen) == len(set(seen))
        and all(isinstance(row, dict) and set(row) == {"evidence_index", "stance"} for row in rows)
    )

    # Apply only the user's listed stance aggregation rules. Invalid output is
    # recorded, never repaired; valid returned assessments remain inspectable.
    has_support = "SUPPORTS" in valid_stances
    has_contradiction = "CONTRADICTS" in valid_stances
    if has_support and has_contradiction:
        verdict = "INSUFFICIENT_EVIDENCE"
    elif has_support:
        verdict = "SUPPORTED"
    elif has_contradiction:
        verdict = "CONTRADICTED"
    else:
        verdict = "INSUFFICIENT_EVIDENCE"
    return {
        "schema_valid": not errors, "validation_errors": errors,
        "evidence_assessments": assessments,
        "evidence_indices": [row.get("evidence_index") for row in assessments],
        "stance_values": [row.get("stance") for row in assessments],
        "valid_stances_used_for_aggregation": valid_stances,
        "invalid_evidence_indices": invalid_indices,
        "invalid_stance_values": invalid_stances,
        "assessment_coverage_valid": coverage_valid,
        "deterministic_verdict": verdict,
    }


def judge_unit(question: str, unit: VerificationUnit, evidence: list[EvidenceItem],
               timing: dict[str, float], attempts: list[dict[str, Any]],
               retrieval_round: int) -> tuple[dict[str, Any], dict[str, str]]:
    messages, mapping = build_messages(question, unit, evidence)
    position_ids = list(mapping)
    latest: dict[str, Any] = {}
    for retry in range(MAX_JUDGE_RETRIES + 1):
        response = call_ollama(messages, temperature=0.0)
        timing["judge"] = timing.get("judge", 0.0) + response["elapsed_seconds"]
        validation = validate_assessments(response["parsed_json"], position_ids)
        attempt = {
            **response, "retrieval_round": retrieval_round,
            "judge_attempt_number": retry + 1,
            "position_to_real_evidence_id": mapping,
            "validation": validation,
            "placeholder_echo_detected": any(token in response["raw_response_text"] for token in (
                "SUPPORTED |", "CONTRADICTED |", "INSUFFICIENT_EVIDENCE |", "SUPPORTS |", "CONTRADICTS |", "NEUTRAL |")),
        }
        attempts.append(attempt)
        latest = {"response": response, "validation": validation}
        retryable = not response["json_parse_success"] or not validation["schema_valid"]
        if not retryable or retry >= MAX_JUDGE_RETRIES:
            return latest, mapping
    return latest, mapping


def run_case(case: dict[str, Any], embedder: Any, reranker: Any) -> dict[str, Any]:
    total_start = time.perf_counter()
    timing: dict[str, float] = {}
    rag_calls: list[dict[str, Any]] = []
    judge_attempts: list[dict[str, Any]] = []
    provider = build_provider(case, embedder, reranker)
    initial_evidence = timed_retrieve(provider, case["question"], "initial_retrieval", timing, rag_calls)
    analysis_start = time.perf_counter()
    units = analyze_verification_units(case["answer"])
    timing["verification_unit_analysis"] = time.perf_counter() - analysis_start
    unit_results: list[dict[str, Any]] = []
    total_targeted_retries = 0
    for unit in units:
        unit_query = build_unit_retrieval_query(case["question"], unit)
        unit_evidence = timed_retrieve(provider, unit_query, "unit_specific_retrieval", timing, rag_calls)
        evidence = merge_evidence(unit_evidence, initial_evidence)
        all_rounds: list[dict[str, Any]] = []
        result, position_map = judge_unit(case["question"], unit, evidence, timing, judge_attempts, retrieval_round=0)
        validation = result["validation"]
        rounds = [{"round": 0, "query": unit_query,
                   "evidence": [asdict(item) for item in evidence],
                   "position_to_real_evidence_id": position_map,
                   "deterministic_verdict": validation["deterministic_verdict"],
                   "validation_errors": validation["validation_errors"]}]
        aggregate = validation["deterministic_verdict"]
        if aggregate == "INSUFFICIENT_EVIDENCE" and total_targeted_retries < MAX_TARGETED_RETRIES:
            total_targeted_retries += 1
            missing = unit.required_evidence or ["claim relationship and qualifiers"]
            retry_query = build_unit_retrieval_query(case["question"], unit, missing_components=missing)
            if " ".join(retry_query.lower().split()) == " ".join(unit_query.lower().split()):
                retry_query = "Missing evidence detail: " + " ".join(missing) + " Context: " + case["question"]
            additional = timed_retrieve(provider, retry_query, "targeted_retrieval_retry", timing, rag_calls)
            evidence = merge_evidence(additional, evidence)
            retry_result, retry_map = judge_unit(case["question"], unit, evidence, timing, judge_attempts, retrieval_round=1)
            retry_validation = retry_result["validation"]
            position_map = retry_map
            rounds.append({"round": 1, "query": retry_query,
                           "evidence": [asdict(item) for item in evidence],
                           "position_to_real_evidence_id": retry_map,
                           "deterministic_verdict": retry_validation["deterministic_verdict"],
                           "validation_errors": retry_validation["validation_errors"]})
            validation = retry_validation
            aggregate = retry_validation["deterministic_verdict"]
        all_rounds.extend(rounds)
        positional_assessments = []
        for assessment in validation.get("evidence_assessments", []):
            ref = assessment.get("evidence_index")
            positional_assessments.append({
                "evidence_index": ref, "stance": assessment.get("stance"),
                "evidence_id": position_map.get(ref) if isinstance(ref, str) else None,
            })
        unit_results.append({
            "unit": unit.to_dict(), "expected_verdict": case["expected_verdict"],
            "deterministic_verdict": aggregate,
            "evidence_assessments": positional_assessments,
            "validation_errors": validation["validation_errors"],
            "schema_valid": validation["schema_valid"],
            "assessment_coverage_valid": validation.get("assessment_coverage_valid", False),
            "rounds": all_rounds,
        })
    observed = [result["deterministic_verdict"] for result in unit_results]
    schema_valid_all = bool(unit_results) and all(result["schema_valid"] for result in unit_results)
    matched = schema_valid_all and observed == [case["expected_verdict"]] * len(observed)
    timing["total_runtime"] = time.perf_counter() - total_start
    return {
        "case_id": case["case_id"], "question": case["question"],
        "original_answer": case["answer"], "expected_verdict": case["expected_verdict"],
        "expected_evidence_slice": case["evidence"],
        "verification_units": [unit.to_dict() for unit in units],
        "initial_retrieval": rag_calls[0] if rag_calls else None,
        "rag_calls": rag_calls, "rag_call_count": len(rag_calls),
        "unit_results": unit_results, "judge_attempts": judge_attempts,
        "llm_call_count": len(judge_attempts),
        "targeted_retrieval_retry_count": total_targeted_retries,
        "observed_deterministic_verdicts": observed,
        "matches_expected": matched,
        "aggregation_reached": bool(unit_results),
        "timings_seconds": timing,
    }


def main() -> int:
    for target in (RESULTS_PATH, SUMMARY_PATH):
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite previous artifact: {target.name}")
    before = hash_snapshot()
    started_at = datetime.now(timezone.utc).isoformat()
    cases = load_cases_exactly()
    print(f"Interpreter: {sys.executable}", flush=True)
    print(f"Started UTC: {started_at}; exact source cases: {CASES_SOURCE.name}", flush=True)
    print("Loading local embedding and cross-encoder models in offline mode...", flush=True)
    embedder = SentenceTransformer(rag.EMBEDDING_MODEL_NAME)
    reranker = CrossEncoder(rag.RERANKER_MODEL_NAME)
    RESULTS_PATH.write_text("", encoding="utf-8")
    results: list[dict[str, Any]] = []
    stop_reason = None
    for case in cases:
        print(f"Running {case['case_id']} through ExistingRAGProvider", flush=True)
        try:
            row = run_case(case, embedder, reranker)
        except Exception as exc:
            row = {
                "case_id": case["case_id"], "question": case["question"],
                "original_answer": case["answer"], "expected_verdict": case["expected_verdict"],
                "unit_results": [], "rag_calls": [], "judge_attempts": [],
                "rag_call_count": 0, "llm_call_count": 0,
                "aggregation_reached": False, "matches_expected": False,
                "errors": [f"{type(exc).__name__}: {exc}"],
            }
        results.append(row)
        with RESULTS_PATH.open("a", encoding="utf-8") as out:
            out.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        print(json.dumps({
            "case_id": row["case_id"], "expected": row["expected_verdict"],
            "deterministic": row.get("observed_deterministic_verdicts"),
            "assessments": [unit.get("evidence_assessments") for unit in row.get("unit_results", [])],
            "matched": row["matches_expected"], "rag_calls": row["rag_call_count"],
            "llm_calls": row["llm_call_count"], "errors": row.get("errors", []),
            "timings_seconds": row.get("timings_seconds"),
        }, ensure_ascii=False), flush=True)
        if not row["matches_expected"]:
            stop_reason = f"Stopped immediately after first mismatch/error: {case['case_id']}"
            break
    after = hash_snapshot()
    changed = sorted(path for path in set(before) | set(after) if before.get(path) != after.get(path))
    judge_times = [float(attempt["elapsed_seconds"]) for row in results
                   for attempt in row.get("judge_attempts", [])]
    all_attempts = [attempt for row in results for attempt in row.get("judge_attempts", [])]
    invalid_refs = sum(len(attempt.get("validation", {}).get("invalid_evidence_indices", [])) for attempt in all_attempts)
    invalid_stances = sum(len(attempt.get("validation", {}).get("invalid_stance_values", [])) for attempt in all_attempts)
    summary = {
        "started_at_utc": started_at, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "interpreter": sys.executable,
        "configuration": {"model": MODEL, "think": False, "temperature": 0,
                          "format": "json", "streaming": True, "timeout_seconds": TIMEOUT_SECONDS,
                          "max_judge_retries": MAX_JUDGE_RETRIES,
                          "max_targeted_retrieval_retries": MAX_TARGETED_RETRIES},
        "case_source": CASES_SOURCE.name, "cases_requested": len(cases),
        "cases_completed": len(results),
        "cases_matched": sum(bool(row.get("matches_expected")) for row in results),
        "first_mismatch": next((row["case_id"] for row in results if not row.get("matches_expected")), None),
        "stop_reason": stop_reason,
        "deterministic_results": [{
            "case_id": row["case_id"], "expected_verdict": row["expected_verdict"],
            "deterministic_verdicts": row.get("observed_deterministic_verdicts", []),
            "evidence_assessments": [u.get("evidence_assessments", []) for u in row.get("unit_results", [])],
            "matches_expected": row.get("matches_expected"),
            "aggregation_reached": row.get("aggregation_reached", False),
        } for row in results],
        "invalid_evidence_reference_count": invalid_refs,
        "invalid_stance_count": invalid_stances,
        "placeholder_echo_count": sum(bool(a.get("placeholder_echo_detected")) for a in all_attempts),
        "judge_latency_seconds": {
            "mean": statistics.mean(judge_times) if judge_times else None,
            "median": statistics.median(judge_times) if judge_times else None,
            "min": min(judge_times) if judge_times else None,
            "max": max(judge_times) if judge_times else None,
        },
        "total_rag_calls": sum(int(row.get("rag_call_count", 0)) for row in results),
        "total_llm_calls": sum(int(row.get("llm_call_count", 0)) for row in results),
        "targeted_retrieval_retry_count": sum(int(row.get("targeted_retrieval_retry_count", 0)) for row in results),
        "aggregation_reached_for_all_completed_cases": all(bool(row.get("aggregation_reached")) for row in results),
        "integrity_hashes_before": before, "integrity_hashes_after": after,
        "changed_protected_tests_or_prior_artifacts": changed,
        "production_files_modified": any(
            path.replace("/", "\\") in {name.replace("/", "\\") for name in PROTECTED_PRODUCTION}
            for path in changed
        ),
        "artifacts": [RESULTS_PATH.name, SUMMARY_PATH.name],
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)
    print("PRODUCTION FILES MODIFIED: " + ("YES" if summary["production_files_modified"] else "NO"), flush=True)
    return 0 if not changed else 1


if __name__ == "__main__":
    raise SystemExit(main())
