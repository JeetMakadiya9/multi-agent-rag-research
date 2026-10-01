"""Controlled contradiction/evidence-stance diagnostic for Experiment B.

This script is diagnostic-only. It does not change production code, prompts,
retrieval, verdict logic, tests, or the existing six-case validation artifacts.
Test A uses Experiment B's current production prompt unchanged. Tests B-F use a
small one-stance JSON prompt to isolate per-evidence classification.
"""
from __future__ import annotations

import hashlib
import json
import statistics
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiment_b_verification import ExperimentBConfig, _judge_prompt, _parse_judgement
from claim_verification import EvidenceItem
from verification_units import analyze_verification_units


ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "experiment_b_contradiction_diagnostic_results.jsonl"
SUMMARY_PATH = ROOT / "experiment_b_contradiction_diagnostic_summary.json"
REPETITIONS = 5
TIMEOUT_SECONDS = 120
ALLOWED_STANCES = {"SUPPORTS", "CONTRADICTS", "NEUTRAL"}
CLAIM = "System X uses BM25 for lexical retrieval."
E1 = "System X uses BM25 for lexical retrieval."
E2 = "System X does not use BM25; lexical retrieval is performed with TF-IDF."
SIX_CASE_RESULTS = ROOT / "experiment_b_component_coverage_validation_results.jsonl"

PROTECTED_FILES = (
    "rag.py",
    "claim_verification.py",
    "experiment_b_verification.py",
    "verification_units.py",
    "experiment_b_component_coverage_validation.py",
    "tests/test_component_coverage.py",
    "tests/test_experiment_b.py",
    "tests/test_experiment_b_verification.py",
)


def _hashes() -> dict[str, str]:
    return {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in PROTECTED_FILES if (ROOT / name).is_file()
    }


def _original_question() -> str:
    if not SIX_CASE_RESULTS.exists():
        return "Can the evidence verify the claim?"
    for line in SIX_CASE_RESULTS.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("case_id") == "case_05_conflicting_evidence":
            return str(row.get("question") or "Can the evidence verify the claim?")
    return "Can the evidence verify the claim?"


def _cases() -> list[dict[str, Any]]:
    # A is built with _judge_prompt, the exact current production prompt.
    unit = analyze_verification_units(CLAIM)[0]
    production_evidence = [
        EvidenceItem(evidence_id="E1", text=E1),
        EvidenceItem(evidence_id="E2", text=E2),
    ]
    test_a_messages = _judge_prompt(_original_question(), unit, production_evidence)
    cases: list[dict[str, Any]] = [{
        "case_id": "A_current_multi_evidence",
        "test_arm": "A",
        "description": "Current production prompt with E1 and E2 assessed against C1 together.",
        "expected_assessments": {"E1-C1": "SUPPORTS", "E2-C1": "CONTRADICTS"},
        "messages": test_a_messages,
        "output_schema": "production_assessments",
    }]

    def one_stance(case_id: str, arm: str, description: str, claim: str,
                   evidence: str, expected: str) -> dict[str, Any]:
        system = (
            "Classify the supplied evidence statement against the claim. "
            "Choose exactly one stance: SUPPORTS, CONTRADICTS, or NEUTRAL. "
            "SUPPORTS means the evidence affirms the claim. CONTRADICTS means "
            "the evidence negates or gives an incompatible value for the claim. "
            "NEUTRAL means it neither affirms nor contradicts the claim. "
            'Return JSON only with exactly this shape: {"stance":"SUPPORTS | CONTRADICTS | NEUTRAL"}. '
            "Do not return a final claim verdict or explanation."
        )
        user = f"Claim: {claim}\nEvidence: {evidence}"
        return {
            "case_id": case_id, "test_arm": arm, "description": description,
            "claim": claim, "evidence": evidence,
            "expected_stance": expected,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "output_schema": "single_stance",
        }

    cases.append(one_stance(
        "B_isolated_case5_contradiction", "B",
        "Exact Case 5 E2 presented alone for a single stance decision.", CLAIM, E2, "CONTRADICTS"))
    cases.append(one_stance(
        "C_per_evidence_E1", "C",
        "E1 independently classified; E2 is not shown to the model.", CLAIM, E1, "SUPPORTS"))
    cases.append(one_stance(
        "C_per_evidence_E2", "C",
        "E2 independently classified; E1 is not shown to the model.", CLAIM, E2, "CONTRADICTS"))

    variants = [
        ("D01_bm25_negation", "System X uses BM25.", "System X does not use BM25."),
        ("D02_tf_idf_instead", "System X uses BM25 for lexical retrieval.",
         "System X uses TF-IDF rather than BM25 for lexical retrieval."),
        ("D03_acquisition_negation", "Company A acquired Company B in 2024.",
         "Company A did not acquire Company B in 2024."),
        ("D04_numeric_mismatch", "The model achieved 95% accuracy.",
         "The reported accuracy was 82%."),
        ("D05_gpu_negation", "System X supports GPU acceleration.",
         "System X does not support GPU acceleration."),
    ]
    for case_id, claim, evidence in variants:
        cases.append(one_stance(case_id, "D", "Controlled contradiction variant.",
                                claim, evidence, "CONTRADICTS"))

    positives = [
        ("E01_bm25_support", "System X uses BM25.",
         "System X uses BM25 for lexical retrieval."),
        ("E02_acquisition_support", "Company A acquired Company B in 2024.",
         "Company A completed its acquisition of Company B in 2024."),
    ]
    for case_id, claim, evidence in positives:
        cases.append(one_stance(case_id, "E", "Explicit positive/support control.",
                                claim, evidence, "SUPPORTS"))

    neutrals = [
        ("F01_python_neutral", "System X uses BM25.", "System X was developed in Python."),
        ("F02_gpu_neutral", "System X supports GPU acceleration.",
         "System X was released in 2021."),
    ]
    for case_id, claim, evidence in neutrals:
        cases.append(one_stance(case_id, "F", "Relevant-topic but non-entailing neutral control.",
                                claim, evidence, "NEUTRAL"))
    return cases


def _call_ollama(messages: list[dict[str, str]], config: ExperimentBConfig) -> dict[str, Any]:
    body = {
        "model": config.model,
        "messages": messages,
        "stream": True,
        "think": config.think,
        "format": config.response_format,
        "options": {"temperature": config.temperature},
    }
    request = urllib.request.Request(
        config.ollama_url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    started = time.perf_counter()
    status: int | None = None
    lines: list[str] = []
    events: list[dict[str, Any]] = []
    content_parts: list[str] = []
    stream_complete = False
    error: str | None = None
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            status = int(response.status)
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace")
                lines.append(line)
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    error = f"Invalid stream event JSON: {exc}"
                    continue
                events.append(event)
                if event.get("error"):
                    error = f"Ollama stream error: {event['error']}"
                message = event.get("message")
                if isinstance(message, dict) and message.get("content"):
                    content_parts.append(str(message["content"]))
                if event.get("done") is True:
                    stream_complete = True
    except urllib.error.HTTPError as exc:
        status = int(exc.code)
        try:
            lines.append(exc.read().decode("utf-8", errors="replace"))
        except Exception:
            pass
        error = f"HTTPError: {exc}"
    except Exception as exc:  # Keep the failed repetition in the artifact.
        error = f"{type(exc).__name__}: {exc}"

    latency = time.perf_counter() - started
    raw_text = "".join(content_parts)
    parsed: Any = None
    json_parse_error: str | None = None
    try:
        parsed = json.loads(raw_text)
    except Exception as exc:
        json_parse_error = f"{type(exc).__name__}: {exc}"
    return {
        "http_status": status,
        "stream_complete": stream_complete,
        "latency_seconds": latency,
        "raw_output": raw_text,
        "raw_stream_lines": lines,
        "raw_stream_events": events,
        "parsed_output": parsed,
        "json_parse_success": isinstance(parsed, dict),
        "json_parse_error": json_parse_error,
        "transport_or_stream_error": error,
        "request": body,
    }


def _assess_call(case: dict[str, Any], call: dict[str, Any]) -> dict[str, Any]:
    parsed = call["parsed_output"]
    if case["output_schema"] == "production_assessments":
        parse_error = None
        normalized: dict[str, Any] | None = None
        if isinstance(parsed, dict):
            try:
                # Use the same parser as the production verification path.
                normalized = _parse_judgement(call["raw_output"])
            except Exception as exc:
                parse_error = f"{type(exc).__name__}: {exc}"
        assessments: list[dict[str, Any]] = []
        returned_pairs: dict[str, str] = {}
        if normalized is not None and isinstance(normalized.get("assessments"), list):
            for row in normalized["assessments"]:
                if not isinstance(row, dict):
                    continue
                eidx = row.get("evidence_index")
                stance = row.get("stance")
                cindices = row.get("component_indices")
                assessments.append({
                    "evidence_index": eidx,
                    "component_indices": cindices,
                    "stance": stance,
                })
                if isinstance(eidx, str) and isinstance(cindices, list) and isinstance(stance, str):
                    for cidx in cindices:
                        if isinstance(cidx, str):
                            returned_pairs[f"{eidx}-{cidx}"] = stance.upper()
        expected = case["expected_assessments"]
        missing = sorted(set(expected) - set(returned_pairs))
        extra = sorted(set(returned_pairs) - set(expected))
        duplicates: list[str] = []
        if normalized is not None:
            seen: set[str] = set()
            for row in assessments:
                eidx, cindices = row.get("evidence_index"), row.get("component_indices")
                if isinstance(eidx, str) and isinstance(cindices, list):
                    for cidx in cindices:
                        key = f"{eidx}-{cidx}"
                        if key in seen:
                            duplicates.append(key)
                        seen.add(key)
        valid_stances = all(value in ALLOWED_STANCES for value in returned_pairs.values())
        complete = not missing and not extra and not duplicates and valid_stances
        correct = {
            pair: returned_pairs.get(pair) == stance and pair in returned_pairs
            for pair, stance in expected.items()
        }
        return {
            "schema_parse_success": normalized is not None,
            "schema_parse_error": parse_error,
            "parsed_assessments": assessments,
            "returned_pair_stances": returned_pairs,
            "missing_pairs": missing,
            "extra_pairs": extra,
            "duplicate_pairs": duplicates,
            "complete_matrix": complete,
            "exact_assessment_matches": correct,
        }

    stance = parsed.get("stance") if isinstance(parsed, dict) else None
    schema_ok = isinstance(stance, str) and stance.upper() in ALLOWED_STANCES
    stance = stance.upper() if isinstance(stance, str) else stance
    return {
        "schema_parse_success": schema_ok,
        "schema_parse_error": None if schema_ok else "Expected JSON object with allowed stance field.",
        "parsed_stance": stance,
        "exact_stance_match": stance == case["expected_stance"],
    }


def _mean_median(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return statistics.mean(values), statistics.median(values)


def main() -> int:
    if RESULTS_PATH.exists() or SUMMARY_PATH.exists():
        raise FileExistsError("Refusing to overwrite existing contradiction diagnostic artifacts.")

    config = ExperimentBConfig()
    before = _hashes()
    started_at = datetime.now(timezone.utc).isoformat()
    cases = _cases()
    rows: list[dict[str, Any]] = []
    print(f"Model={config.model} think={config.think} temperature={config.temperature} format={config.response_format}", flush=True)
    print(f"Cases={len(cases)} repetitions={REPETITIONS} total_calls={len(cases) * REPETITIONS}", flush=True)

    # Create only the new diagnostic output. Each attempted call is flushed so
    # failures are preserved if a later call or process is interrupted.
    with RESULTS_PATH.open("x", encoding="utf-8") as results_file:
        call_number = 0
        for case in cases:
            for repetition in range(1, REPETITIONS + 1):
                call_number += 1
                call = _call_ollama(case["messages"], config)
                assessed = _assess_call(case, call)
                row = {
                    "run_number": call_number,
                    "case_id": case["case_id"],
                    "test_arm": case["test_arm"],
                    "description": case["description"],
                    "repetition": repetition,
                    "expected_assessments": case.get("expected_assessments"),
                    "expected_stance": case.get("expected_stance"),
                    "claim": case.get("claim", CLAIM if case["test_arm"] == "A" else None),
                    "evidence": case.get("evidence"),
                    "messages": case["messages"],
                    "model": config.model,
                    "think": config.think,
                    "format": config.response_format,
                    "temperature": config.temperature,
                    **call,
                    **assessed,
                }
                rows.append(row)
                results_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                results_file.flush()
                print(
                    f"[{call_number}/{len(cases) * REPETITIONS}] {case['case_id']} rep={repetition} "
                    f"HTTP={call['http_status']} complete={call['stream_complete']} "
                    f"parse={call['json_parse_success']} latency={call['latency_seconds']:.2f}s",
                    flush=True,
                )

    # Scores count missing/malformed assessments as incorrect. The multi-case
    # contributes one expected stance for each required evidence/component pair.
    expected_decisions: list[tuple[str, str, bool]] = []
    for row in rows:
        if row["test_arm"] == "A":
            for pair, expected in (row.get("expected_assessments") or {}).items():
                expected_decisions.append((expected, (row.get("returned_pair_stances") or {}).get(pair, ""),
                                           (row.get("exact_assessment_matches") or {}).get(pair, False)))
        else:
            expected_decisions.append((row.get("expected_stance") or "", row.get("parsed_stance") or "",
                                       bool(row.get("exact_stance_match"))))

    total_calls = len(rows)
    json_successes = sum(bool(row["json_parse_success"]) for row in rows)
    schema_successes = sum(bool(row["schema_parse_success"]) for row in rows)
    stance_correct = sum(correct for _, _, correct in expected_decisions)
    by_stance: dict[str, dict[str, Any]] = {}
    for stance in sorted(ALLOWED_STANCES):
        relevant = [(expected, correct) for expected, _observed, correct in expected_decisions if expected == stance]
        by_stance[stance.lower()] = {
            "correct": sum(correct for _expected, correct in relevant),
            "total": len(relevant),
            "accuracy": sum(correct for _expected, correct in relevant) / len(relevant) if relevant else None,
        }
    a_rows = [row for row in rows if row["test_arm"] == "A"]
    contradictions = by_stance["contradicts"]
    supports = by_stance["supports"]
    neutrals = by_stance["neutral"]
    latencies = [float(row["latency_seconds"]) for row in rows]
    mean_latency, median_latency = _mean_median(latencies)
    failures = [row for row in rows if not row.get("json_parse_success") or not row.get("schema_parse_success")
                or (row["test_arm"] == "A" and (not row.get("complete_matrix")
                    or not all((row.get("exact_assessment_matches") or {}).values())))
                or (row["test_arm"] != "A" and not row.get("exact_stance_match"))]
    failure_examples = [{
        "run_number": row["run_number"], "case_id": row["case_id"],
        "test_arm": row["test_arm"], "repetition": row["repetition"],
        "expected_assessments": row.get("expected_assessments"),
        "expected_stance": row.get("expected_stance"),
        "parsed_output": row.get("parsed_output"), "raw_output": row.get("raw_output"),
        "missing_pairs": row.get("missing_pairs"),
        "transport_or_stream_error": row.get("transport_or_stream_error"),
        "json_parse_error": row.get("json_parse_error"),
        "schema_parse_error": row.get("schema_parse_error"),
    } for row in failures[:12]]

    b_c_exact = [row for row in rows if row["case_id"] in {
        "B_isolated_case5_contradiction", "C_per_evidence_E2"}]
    b_c_correct = sum(bool(row.get("exact_stance_match")) for row in b_c_exact)
    b_c_total = len(b_c_exact)
    b_c_accuracy = b_c_correct / b_c_total if b_c_total else None
    a_complete_rate = sum(bool(row.get("complete_matrix")) for row in a_rows) / len(a_rows) if a_rows else None
    if a_complete_rate is not None and a_complete_rate < 1.0 and b_c_accuracy is not None and b_c_accuracy >= 0.8:
        classification = "A — multi-evidence output completeness failure is primary; isolated Case 5 contradiction classification usually succeeds."
    elif a_complete_rate is not None and a_complete_rate < 1.0 and b_c_accuracy is not None and b_c_accuracy < 0.8:
        classification = "C — both multi-evidence output completeness and isolated Case 5 contradiction reasoning failed often."
    elif a_complete_rate == 1.0 and b_c_accuracy is not None and b_c_accuracy < 0.8:
        classification = "B — isolated Case 5 contradiction reasoning failed often; multi-evidence output was complete in these repetitions."
    else:
        classification = "D — the exact Case 5 issue was not consistently reproduced by the paired tests; inspect per-arm outcomes for another interaction."

    after = _hashes()
    summary = {
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": config.model,
        "ollama_url": config.ollama_url,
        "request_settings": {"think": config.think, "temperature": config.temperature,
                             "format": config.response_format, "stream": True},
        "diagnostic_case_count": len(cases),
        "repetitions_per_case": REPETITIONS,
        "total_calls": total_calls,
        "attempted_calls": total_calls,
        "http_200_calls": sum(row["http_status"] == 200 for row in rows),
        "completed_streams": sum(bool(row["stream_complete"]) for row in rows),
        "json_parse_successes": json_successes,
        "parse_success_rate": json_successes / total_calls if total_calls else None,
        "schema_parse_successes": schema_successes,
        "schema_parse_success_rate": schema_successes / total_calls if total_calls else None,
        "expected_stance_decisions": len(expected_decisions),
        "exact_stance_correct": stance_correct,
        "exact_stance_accuracy": stance_correct / len(expected_decisions) if expected_decisions else None,
        "contradiction_detection": contradictions,
        "contradiction_detection_accuracy": contradictions["accuracy"],
        "support_detection": supports,
        "support_detection_accuracy": supports["accuracy"],
        "neutral_detection": neutrals,
        "neutral_detection_accuracy": neutrals["accuracy"],
        "multi_evidence": {
            "case_repetitions": len(a_rows),
            "complete_matrices": sum(bool(row.get("complete_matrix")) for row in a_rows),
            "complete_matrix_rate": a_complete_rate,
            "both_expected_stances_correct": sum(
                bool(row.get("complete_matrix")) and all((row.get("exact_assessment_matches") or {}).values())
                for row in a_rows),
            "exact_complete_and_correct_rate": sum(
                bool(row.get("complete_matrix")) and all((row.get("exact_assessment_matches") or {}).values())
                for row in a_rows) / len(a_rows) if a_rows else None,
        },
        "case5_isolated_pairwise": {
            "calls": b_c_total, "correct": b_c_correct, "accuracy": b_c_accuracy,
        },
        "mean_latency_seconds": mean_latency,
        "median_latency_seconds": median_latency,
        "failure_count": len(failures),
        "raw_failure_examples": failure_examples,
        "observed_failure_classification": classification,
        "arm_case_ids": {arm: [case["case_id"] for case in cases if case["test_arm"] == arm]
                         for arm in ("A", "B", "C", "D", "E", "F")},
        "diagnostic_prompt_note": "Test A used experiment_b_verification._judge_prompt unchanged. Tests B-F used the diagnostic-only single-stance prompt stored per call in the JSONL.",
        "production_files_modified": False,
        "protected_sha256_before": before,
        "protected_sha256_after": after,
        "protected_hashes_unchanged": before == after,
        "artifacts": {"results_jsonl": RESULTS_PATH.name, "summary_json": SUMMARY_PATH.name},
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in (
        "total_calls", "parse_success_rate", "exact_stance_accuracy",
        "contradiction_detection_accuracy", "support_detection_accuracy",
        "neutral_detection_accuracy", "multi_evidence", "mean_latency_seconds",
        "median_latency_seconds", "failure_count", "observed_failure_classification",
        "protected_hashes_unchanged",
    )}, ensure_ascii=False, indent=2), flush=True)
    return 0 if summary["protected_hashes_unchanged"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
