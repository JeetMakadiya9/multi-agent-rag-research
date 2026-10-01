"""Three-repeat isolated stance-output diagnostic for saved Experiment B Case 5."""
from __future__ import annotations

import hashlib
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
SOURCE_RESULTS = ROOT / "baseline_experiment_b_judge_v2_validation_results.jsonl"
RESULTS_PATH = ROOT / "baseline_case5_evidence_stance_results.jsonl"
SUMMARY_PATH = ROOT / "baseline_case5_evidence_stance_summary.json"
MODEL = "qwen3:4b"
URL = "http://127.0.0.1:11434/api/chat"
TIMEOUT_SECONDS = 60
EXPECTED = "INSUFFICIENT_EVIDENCE"
ALLOWED_VERDICTS = {"SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"}
ALLOWED_STANCES = {"SUPPORTS", "CONTRADICTS", "NEUTRAL"}

PROTECTED_PRODUCTION = [
    "rag.py", "claim_verification.py", "research_pipeline.py", "llm.py",
    "research_state.py", "research_tools.py", "research_retrieval.py",
    "experiment_b_verification.py", "verification_units.py",
]
NEW_NAMES = {p.name for p in (ROOT / "baseline_case5_evidence_stance_benchmark.py",
                               ROOT / "baseline_case5_evidence_stance_results.jsonl",
                               ROOT / "baseline_case5_evidence_stance_summary.json")}


def file_inventory() -> list[Path]:
    paths = [ROOT / rel for rel in PROTECTED_PRODUCTION]
    tests_dir = ROOT / "tests"
    if tests_dir.exists():
        paths.extend(p for p in tests_dir.rglob("*") if p.is_file())
    # Hash the earlier experiment artifacts, without including this experiment's
    # own output paths or touching any of them.
    paths.extend(p for p in ROOT.glob("baseline_*") if p.is_file() and p.name not in NEW_NAMES)
    return sorted(set(p for p in paths if p.is_file()))


def hashes() -> dict[str, str]:
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in file_inventory()}


def load_case5() -> dict[str, Any]:
    found = None
    for line in SOURCE_RESULTS.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("case_id") == "case_05_conflicting_evidence":
            found = row
            break
    if found is None:
        raise RuntimeError(f"Case 5 not present in {SOURCE_RESULTS.name}")
    evidence = found.get("expected_evidence_slice")
    if not isinstance(evidence, list) or len(evidence) != 2:
        raise ValueError("Saved Case 5 must contain exactly two evidence items")
    return {
        "case_id": found["case_id"], "question": found["question"],
        "claim": found["original_answer"], "evidence": evidence,
        "expected_verdict": found["expected_verdict"],
        "saved_from": SOURCE_RESULTS.name,
    }


def make_messages(case: dict[str, Any]) -> list[dict[str, str]]:
    evidence_lines = []
    for index, item in enumerate(case["evidence"], start=1):
        evidence_lines.append(f"E{index}:\n{item['text']}")
    system = (
        "Assess each supplied evidence item independently against the claim. "
        "Allowed verdict values are SUPPORTED, CONTRADICTED, and INSUFFICIENT_EVIDENCE. "
        "Allowed evidence stance values are SUPPORTS, CONTRADICTS, and NEUTRAL. "
        "Use only positional evidence IDs shown below. Never generate or guess source IDs. "
        "Return only a JSON object with exactly three top-level fields: verdict, evidence_assessments, reason. "
        "evidence_assessments must contain one assessment for every supplied evidence item; "
        "each assessment has exactly evidence_index and stance. "
        "Example of the required structure for this task: "
        '{"verdict":"INSUFFICIENT_EVIDENCE","evidence_assessments":'
        '[{"evidence_index":"E1","stance":"SUPPORTS"},'
        '{"evidence_index":"E2","stance":"CONTRADICTS"}],"reason":"The evidence conflicts."}'
    )
    user = (
        f"Question: {case['question']}\nClaim to assess: {case['claim']}\n\n"
        "Evidence items:\n" + "\n\n".join(evidence_lines) +
        "\n\nAssess both evidence items individually and return the required JSON only."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def validate_and_derive(parsed: Any, evidence_ids: list[str]) -> dict[str, Any]:
    errors: list[str] = []
    invalid_indices: list[str] = []
    invalid_stances: list[str] = []
    assessments: list[dict[str, Any]] = []
    if not isinstance(parsed, dict):
        return {
            "schema_valid": False, "validation_errors": ["JSON root must be an object"],
            "evidence_indices": [], "stance_values": [], "invalid_evidence_indices": [],
            "invalid_stance_values": [], "validator_verdict": "INSUFFICIENT_EVIDENCE",
        }
    required = {"verdict", "evidence_assessments", "reason"}
    missing = sorted(required - set(parsed))
    extra = sorted(set(parsed) - required)
    if missing:
        errors.append("Missing required fields: " + ", ".join(missing))
    if extra:
        errors.append("Unexpected top-level fields: " + ", ".join(extra))
    model_verdict = parsed.get("verdict")
    if not isinstance(model_verdict, str) or model_verdict not in ALLOWED_VERDICTS:
        errors.append("Invalid model verdict")
    if not isinstance(parsed.get("reason"), str):
        errors.append("reason must be a string")
    raw_assessments = parsed.get("evidence_assessments")
    if not isinstance(raw_assessments, list):
        errors.append("evidence_assessments must be an array")
        raw_assessments = []
    seen: list[str] = []
    for position, assessment in enumerate(raw_assessments):
        if not isinstance(assessment, dict):
            errors.append(f"Assessment {position} must be an object")
            continue
        if set(assessment) != {"evidence_index", "stance"}:
            errors.append(f"Assessment {position} must contain exactly evidence_index and stance")
        evidence_index = assessment.get("evidence_index")
        stance = assessment.get("stance")
        if not isinstance(evidence_index, str) or evidence_index not in evidence_ids:
            invalid_indices.append(str(evidence_index))
            errors.append(f"Invalid evidence index: {evidence_index!r}")
        else:
            seen.append(evidence_index)
        if not isinstance(stance, str) or stance not in ALLOWED_STANCES:
            invalid_stances.append(str(stance))
            errors.append(f"Invalid stance value: {stance!r}")
        assessments.append({"evidence_index": evidence_index, "stance": stance})
    if sorted(seen) != sorted(evidence_ids):
        errors.append("Evidence assessment coverage does not match supplied evidence IDs exactly")
    if len(seen) != len(set(seen)):
        errors.append("Duplicate evidence index in assessments")

    # Invalid/partial assessments cannot safely resolve a conflict: preserve the
    # invalidity and conservatively derive INSUFFICIENT_EVIDENCE.
    assessment_valid = (
        not invalid_indices and not invalid_stances and sorted(seen) == sorted(evidence_ids)
        and len(seen) == len(set(seen))
        and all(isinstance(a, dict) and set(a) == {"evidence_index", "stance"}
                for a in raw_assessments)
    )
    stances = {a["stance"] for a in assessments if a.get("stance") in ALLOWED_STANCES}
    if not assessment_valid:
        derived = "INSUFFICIENT_EVIDENCE"
    elif "SUPPORTS" in stances and "CONTRADICTS" in stances:
        derived = "INSUFFICIENT_EVIDENCE"
    elif "SUPPORTS" in stances:
        derived = "SUPPORTED"
    elif "CONTRADICTS" in stances:
        derived = "CONTRADICTED"
    else:
        derived = "INSUFFICIENT_EVIDENCE"
    schema_valid = not errors
    return {
        "schema_valid": schema_valid, "validation_errors": errors,
        "evidence_indices": [a.get("evidence_index") for a in assessments],
        "stance_values": [a.get("stance") for a in assessments],
        "invalid_evidence_indices": invalid_indices, "invalid_stance_values": invalid_stances,
        "evidence_assessments": assessments, "assessment_coverage_valid": assessment_valid,
        "validator_verdict": derived,
    }


def call_ollama(messages: list[dict[str, str]]) -> dict[str, Any]:
    body = {
        "model": MODEL, "messages": messages, "stream": True, "think": False,
        "format": "json", "options": {"temperature": 0},
    }
    req = urllib.request.Request(
        URL, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    started = time.perf_counter()
    status: int | None = None
    raw_lines: list[str] = []
    content_parts: list[str] = []
    completed = False
    transport_error: str | None = None
    stream_events: list[Any] = []
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as response:
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
                    transport_error = f"Ollama stream error: {event['error']}"
                message = event.get("message")
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
        transport_error = f"HTTPError: {exc}"
    except Exception as exc:
        transport_error = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - started
    raw_text = "".join(content_parts)
    parsed: Any = None
    parse_error = None
    try:
        parsed = json.loads(raw_text)
    except Exception as exc:
        parse_error = f"{type(exc).__name__}: {exc}"
    return {
        "http_status": status, "stream_completed": completed,
        "elapsed_seconds": elapsed, "raw_response_text": raw_text,
        "raw_stream_payload_lines": raw_lines,
        "raw_stream_payload_exact_join": "".join(raw_lines),
        "parsed_json": parsed, "parse_success": isinstance(parsed, dict),
        "parse_exception": parse_error, "transport_exception": transport_error,
        "stream_events": stream_events, "request_configuration": body,
    }


def main() -> int:
    if RESULTS_PATH.exists() or SUMMARY_PATH.exists():
        raise FileExistsError("Refusing to overwrite an existing Case 5 stance artifact")
    before = hashes()
    started_at = datetime.now(timezone.utc).isoformat()
    case = load_case5()
    evidence_indices = [f"E{i}" for i in range(1, len(case["evidence"]) + 1)]
    evidence_real_ids = [str(item.get("evidence_id", "")) for item in case["evidence"]]
    messages = make_messages(case)
    print(f"Interpreter: {sys.executable}", flush=True)
    print(f"Started UTC: {started_at}; running 3 Case 5 calls", flush=True)
    rows = []
    RESULTS_PATH.write_text("", encoding="utf-8")
    for repetition in range(1, 4):
        call = call_ollama(messages)
        validation = validate_and_derive(call["parsed_json"], evidence_indices)
        raw_parsed = call["parsed_json"] if isinstance(call["parsed_json"], dict) else {}
        model_verdict = raw_parsed.get("verdict")
        raw_text = call["raw_response_text"]
        row = {
            "case_id": case["case_id"], "repetition": repetition,
            "question": case["question"], "claim": case["claim"],
            "evidence": [{"evidence_index": key, "evidence_id": evidence_real_ids[i],
                          "text": case["evidence"][i]["text"]}
                         for i, key in enumerate(evidence_indices)],
            "expected_verdict": case["expected_verdict"],
            "http_status": call["http_status"], "stream_completed": call["stream_completed"],
            "elapsed_seconds": call["elapsed_seconds"],
            "raw_response_text": raw_text,
            "raw_stream_payload_lines": call["raw_stream_payload_lines"],
            "raw_stream_payload_exact_join": call["raw_stream_payload_exact_join"],
            "parsed_json": call["parsed_json"], "parse_success": call["parse_success"],
            "parse_exception": call["parse_exception"],
            "transport_exception": call["transport_exception"],
            "schema_valid": validation["schema_valid"],
            "validation_errors": validation["validation_errors"],
            "evidence_indices": validation["evidence_indices"],
            "stance_values": validation["stance_values"],
            "invalid_evidence_indices": validation["invalid_evidence_indices"],
            "invalid_stance_values": validation["invalid_stance_values"],
            "assessment_coverage_valid": validation.get("assessment_coverage_valid", False),
            "model_verdict": model_verdict,
            "validator_verdict": validation["validator_verdict"],
            "placeholder_echo_detected": ("SUPPORTED |" in raw_text or
                                           "CONTRADICTED |" in raw_text or
                                           "INSUFFICIENT_EVIDENCE |" in raw_text or
                                           "SUPPORTS |" in raw_text or
                                           "CONTRADICTS |" in raw_text or
                                           "NEUTRAL |" in raw_text),
            "model_verdict_correct": model_verdict == case["expected_verdict"],
            "validator_verdict_correct": validation["validator_verdict"] == case["expected_verdict"],
            "assessment_outcome": (
                "both evidence stances correct" if validation["evidence_assessments"] == [
                    {"evidence_index": "E1", "stance": "SUPPORTS"},
                    {"evidence_index": "E2", "stance": "CONTRADICTS"},
                ] else "E2 omitted" if "E2" not in validation["evidence_indices"]
                else "stance classification differs from expected"
            ),
        }
        rows.append(row)
        with RESULTS_PATH.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps({"repetition": repetition, "http_status": row["http_status"],
                          "completed": row["stream_completed"], "seconds": row["elapsed_seconds"],
                          "model_verdict": model_verdict,
                          "stances": row["stance_values"],
                          "validator_verdict": row["validator_verdict"],
                          "schema_valid": row["schema_valid"],
                          "errors": row["validation_errors"]}, ensure_ascii=False), flush=True)
    after = hashes()
    changed = sorted(path for path in set(before) | set(after) if before.get(path) != after.get(path))
    latencies = [float(row["elapsed_seconds"]) for row in rows]
    all_success = len(rows) == 3 and all(
        row["parse_success"] and row["schema_valid"]
        and row["evidence_indices"] == ["E1", "E2"]
        and row["stance_values"] == ["SUPPORTS", "CONTRADICTS"]
        and row["validator_verdict"] == EXPECTED
        for row in rows
    )
    summary = {
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "interpreter": sys.executable,
        "model_configuration": {"model": MODEL, "think": False, "temperature": 0,
                                "format": "json", "stream": True,
                                "timeout_seconds": TIMEOUT_SECONDS},
        "source_case_artifact": case["saved_from"],
        "case_id": case["case_id"], "question": case["question"], "claim": case["claim"],
        "expected_verdict": EXPECTED, "repetitions": len(rows),
        "success_criteria_met_all_repetitions": all_success,
        "model_verdicts": [row["model_verdict"] for row in rows],
        "validator_verdicts": [row["validator_verdict"] for row in rows],
        "expected_verdicts": [row["expected_verdict"] for row in rows],
        "schema_valid_count": sum(bool(row["schema_valid"]) for row in rows),
        "parse_success_count": sum(bool(row["parse_success"]) for row in rows),
        "correct_per_evidence_assessment_count": sum(
            row["assessment_outcome"] == "both evidence stances correct" for row in rows),
        "E2_omitted_count": sum(row["assessment_outcome"] == "E2 omitted" for row in rows),
        "stance_classification_failure_count": sum(
            row["assessment_outcome"] == "stance classification differs from expected" for row in rows),
        "model_verdict_correct_count": sum(bool(row["model_verdict_correct"]) for row in rows),
        "validator_verdict_correct_count": sum(bool(row["validator_verdict_correct"]) for row in rows),
        "placeholder_echo_count": sum(bool(row["placeholder_echo_detected"]) for row in rows),
        "latency_seconds": {"mean": statistics.mean(latencies) if latencies else None,
                            "median": statistics.median(latencies) if latencies else None,
                            "min": min(latencies) if latencies else None,
                            "max": max(latencies) if latencies else None},
        "per_repetition": [{"repetition": row["repetition"],
                            "model_verdict": row["model_verdict"],
                            "validator_verdict": row["validator_verdict"],
                            "expected_verdict": row["expected_verdict"],
                            "assessment_outcome": row["assessment_outcome"],
                            "schema_valid": row["schema_valid"],
                            "elapsed_seconds": row["elapsed_seconds"]} for row in rows],
        "hashes_before": before, "hashes_after": after,
        "changed_protected_or_prior_files": changed,
        "production_files_modified": any(p in changed for p in PROTECTED_PRODUCTION),
        "artifacts": [RESULTS_PATH.name, SUMMARY_PATH.name],
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if not changed and len(rows) == 3 else 1


if __name__ == "__main__":
    raise SystemExit(main())
