"""Isolated 12-call Ollama judge-interface diagnostic for captured Case 2."""
from __future__ import annotations

import hashlib
import json
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
CASE_ARTIFACT = ROOT / "baseline_experiment_b_case2_judge_diagnostic.json"
RESULTS_PATH = ROOT / "baseline_case2_judge_interface_results.jsonl"
SUMMARY_PATH = ROOT / "baseline_case2_judge_interface_summary.json"
MODEL = "qwen3:4b"
ENDPOINT = "http://127.0.0.1:11434/api/chat"
TEMPERATURE = 0.0
THINK = False
TIMEOUT_SECONDS = 60
ALLOWED_VERDICTS = {"SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"}
LABELS = {0: "SUPPORTED", 1: "INSUFFICIENT_EVIDENCE", 2: "CONTRADICTED"}

PROTECTED_PATHS = [
    "rag.py", "claim_verification.py", "research_pipeline.py", "llm.py",
    "research_state.py", "research_tools.py", "research_retrieval.py",
    "tests/test_experiment_b.py", "tests/test_experiment_b_verification.py",
    "baseline_rag_test.py", "baseline_rag_audit.py",
    "baseline_experiment_b_six_case_validation.py",
    "baseline_experiment_b_six_case_validation_results.jsonl",
    "baseline_experiment_b_six_case_validation_summary.json",
    "baseline_experiment_b_case2_judge_diagnostic.py",
    "baseline_experiment_b_case2_judge_diagnostic.json",
]
PROTECTED_PATHS.extend(
    str(path.relative_to(ROOT))
    for path in sorted((ROOT / "evaluation" / "experiment_b").glob("**/*"))
    if path.is_file()
)


def load_case() -> tuple[str, str, str, str]:
    data = json.loads(CASE_ARTIFACT.read_text(encoding="utf-8"))
    claim = str(data["original_answer"])
    evidence = str(data["evidence_corpus"][0]["text"])
    evidence_id = str(data["initial_retrieval_evidence"][0]["evidence_id"])
    question = str(data["question"])
    return question, claim, evidence, evidence_id


def variants(question: str, claim: str, evidence: str, evidence_id: str) -> list[dict[str, Any]]:
    facts = (
        f"Question: {question}\nClaim to verify: {claim}\n"
        f"Only supplied evidence: {evidence}\n"
        "The evidence explicitly says BM25 is not used for image generation. "
        "Decide whether the evidence supports, contradicts, or is insufficient for the claim."
    )
    return [
        {
            "variant": "A_minimal_verdict",
            "system": (
                "Verify the claim using only the supplied evidence. The allowed verdict values are "
                "SUPPORTED, CONTRADICTED, and INSUFFICIENT_EVIDENCE. Select exactly one value. "
                "The evidence explicitly negates the claim. Return exactly this JSON object and no other text: "
                '{"verdict":"CONTRADICTED"}'
            ),
            "user": facts,
        },
        {
            "variant": "B_verdict_and_reason",
            "system": (
                "Verify the claim using only the supplied evidence. The allowed verdict values are "
                "SUPPORTED, CONTRADICTED, and INSUFFICIENT_EVIDENCE. Select exactly one value. "
                "Return only one JSON object with exactly two keys: verdict and reason. The verdict must be "
                "CONTRADICTED and the reason must be a short non-empty sentence."
            ),
            "user": facts,
        },
        {
            "variant": "C_numeric_label",
            "system": (
                "Verify the claim using only the supplied evidence. Use this numeric mapping: "
                "0 means SUPPORTED; 1 means INSUFFICIENT_EVIDENCE; 2 means CONTRADICTED. "
                "The evidence explicitly negates the claim, so select 2. "
                "Return one JSON object with exactly one key, label, whose value is the selected integer."
            ),
            "user": facts,
        },
        {
            "variant": "D_verdict_and_evidence_id",
            "system": (
                "Verify the claim using only the supplied evidence. The allowed verdict values are "
                "SUPPORTED, CONTRADICTED, and INSUFFICIENT_EVIDENCE. Select exactly one value. "
                "Return only one JSON object with exactly two keys: verdict and evidence_id. For this case, "
                "the verdict must be CONTRADICTED and evidence_id must equal the supplied exact ID."
            ),
            "user": facts + f"\nThe exact evidence_id is: {evidence_id}",
        },
    ]


def hash_snapshot() -> dict[str, str | None]:
    values: dict[str, str | None] = {}
    for relative in PROTECTED_PATHS:
        path = ROOT / relative
        values[relative] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    return values


def git_status() -> dict[str, Any]:
    try:
        root = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(ROOT), "status", "--short"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout
        return {"available": True, "root": root, "status_before_artifacts": status.splitlines()}
    except Exception as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}


def classify_output(variant: dict[str, Any], raw: str, exception: str | None) -> dict[str, Any]:
    parsed = None
    parse_exception = None
    try:
        parsed = json.loads(raw)
    except Exception as exc:
        parse_exception = f"{type(exc).__name__}: {exc}"

    name = variant["variant"]
    required = {
        "A_minimal_verdict": ["verdict"],
        "B_verdict_and_reason": ["verdict", "reason"],
        "C_numeric_label": ["label"],
        "D_verdict_and_evidence_id": ["verdict", "evidence_id"],
    }[name]
    fields_exist = isinstance(parsed, dict) and all(field in parsed for field in required)
    normalized = None
    allowed = False
    evidence_id_matches = None
    if isinstance(parsed, dict):
        if name == "C_numeric_label":
            label = parsed.get("label")
            allowed = isinstance(label, int) and not isinstance(label, bool) and label in LABELS
            normalized = LABELS[label] if allowed else None
        else:
            verdict = parsed.get("verdict")
            allowed = isinstance(verdict, str) and verdict in ALLOWED_VERDICTS
            normalized = verdict if allowed else None
        if name == "B_verdict_and_reason":
            fields_exist = fields_exist and isinstance(parsed.get("reason"), str)
        if name == "D_verdict_and_evidence_id":
            evidence_id_matches = parsed.get("evidence_id") == CASE_EVIDENCE_ID
            fields_exist = fields_exist and isinstance(parsed.get("evidence_id"), str) and evidence_id_matches

    exact_keys = isinstance(parsed, dict) and set(parsed) == set(required)
    valid_structure = bool(parse_exception is None and fields_exist and allowed and exact_keys)
    placeholder = (
        "|" in raw
        or "SUPPORTED | CONTRADICTED | INSUFFICIENT_EVIDENCE" in raw
        or "SUPPORTS | REFUTES | NEUTRAL" in raw
    )
    return {
        "parsed_json": parsed,
        "parse_success": parse_exception is None,
        "parse_exception": parse_exception,
        "required_fields": required,
        "required_fields_exist": bool(fields_exist),
        "returned_value_allowed": bool(allowed),
        "normalized_verdict": normalized,
        "decision_correct": normalized == "CONTRADICTED",
        "valid_structured_output": valid_structure,
        "exact_required_keys": bool(exact_keys),
        "evidence_id_matches": evidence_id_matches,
        "placeholder_echo": placeholder,
        "contains_pipe_character": "|" in raw,
        "contains_full_verdict_alternatives": "SUPPORTED | CONTRADICTED | INSUFFICIENT_EVIDENCE" in raw,
        "contains_full_interpretation_alternatives": "SUPPORTS | REFUTES | NEUTRAL" in raw,
        "exception": exception,
    }


def request_once(variant: dict[str, Any], repetition: int) -> dict[str, Any]:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": variant["system"]},
            {"role": "user", "content": variant["user"]},
        ],
        "stream": True,
        "think": THINK,
        "format": "json",
        "options": {"temperature": TEMPERATURE},
    }
    request = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    raw_chunks: list[bytes] = []
    raw_lines: list[str] = []
    events: list[dict[str, Any]] = []
    content_fragments: list[str] = []
    done_event = None
    status = None
    headers: dict[str, str] = {}
    exception = None
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            status = response.status
            headers = dict(response.headers.items())
            for line in response:
                raw_line = bytes(line)
                raw_chunks.append(raw_line)
                raw_lines.append(raw_line.decode("utf-8", errors="replace"))
                event = json.loads(raw_line.decode("utf-8"))
                events.append(event)
                message = event.get("message", {})
                content = message.get("content", "") if isinstance(message, dict) else ""
                if content:
                    content_fragments.append(str(content))
                if event.get("done"):
                    done_event = event
                    break
    except urllib.error.HTTPError as exc:
        status = exc.code
        headers = dict(exc.headers.items()) if exc.headers else {}
        exception = f"HTTPError: {exc.code} {exc.reason}; body={exc.read().decode('utf-8', errors='replace')}"
    except Exception as exc:
        exception = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - started
    raw_text = "".join(content_fragments)
    classified = classify_output(variant, raw_text, exception)
    return {
        "variant": variant["variant"], "repetition": repetition,
        "model": MODEL, "think": THINK, "temperature": TEMPERATURE,
        "format": "json", "timeout_seconds": TIMEOUT_SECONDS,
        "endpoint": ENDPOINT, "http_status": status,
        "response_headers": headers,
        "stream_completed": bool(done_event and done_event.get("done")),
        "streamed_successfully": bool(status == 200 and done_event and not exception),
        "elapsed_seconds": elapsed,
        "raw_response_text": raw_text,
        "raw_http_stream_text": "".join(raw_lines),
        "raw_http_stream_chunks_base64": [
            __import__("base64").b64encode(chunk).decode("ascii") for chunk in raw_chunks
        ],
        "response_length_characters": len(raw_text),
        "stream_events": events,
        "done_event": done_event,
        **classified,
    }


def main() -> int:
    global CASE_EVIDENCE_ID
    started_at = datetime.now(timezone.utc).isoformat()
    before_hashes = hash_snapshot()
    git = git_status()
    question, claim, evidence, CASE_EVIDENCE_ID = load_case()
    all_results: list[dict[str, Any]] = []
    RESULTS_PATH.write_text("", encoding="utf-8")

    print(f"Started UTC: {started_at}", flush=True)
    print(f"Using captured evidence ID: {CASE_EVIDENCE_ID}", flush=True)
    for variant in variants(question, claim, evidence, CASE_EVIDENCE_ID):
        for repetition in range(1, 4):
            print(f"Calling {variant['variant']} repetition {repetition}/3 ...", flush=True)
            row = request_once(variant, repetition)
            all_results.append(row)
            with RESULTS_PATH.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            print(json.dumps({
                key: row.get(key) for key in (
                    "variant", "repetition", "http_status", "stream_completed",
                    "elapsed_seconds", "response_length_characters", "parse_success",
                    "normalized_verdict", "valid_structured_output", "placeholder_echo", "exception",
                )
            }, ensure_ascii=False), flush=True)

    after_hashes = hash_snapshot()
    hash_changes = {
        path: {"before": before_hashes[path], "after": after_hashes[path]}
        for path in before_hashes if before_hashes[path] != after_hashes[path]
    }
    groups = {}
    for variant in variants(question, claim, evidence, CASE_EVIDENCE_ID):
        rows = [row for row in all_results if row["variant"] == variant["variant"]]
        latencies = [row["elapsed_seconds"] for row in rows]
        failures = [row for row in rows if not row["valid_structured_output"] or not row["decision_correct"] or row["exception"]]
        decisions = [row["normalized_verdict"] for row in rows]
        groups[variant["variant"]] = {
            "calls": len(rows),
            "valid_structured_output_rate": sum(row["valid_structured_output"] for row in rows) / len(rows),
            "correct_decision_rate": sum(row["decision_correct"] for row in rows) / len(rows),
            "placeholder_echo_rate": sum(row["placeholder_echo"] for row in rows) / len(rows),
            "parse_success_rate": sum(row["parse_success"] for row in rows) / len(rows),
            "mean_latency_seconds": statistics.mean(latencies),
            "median_latency_seconds": statistics.median(latencies),
            "min_latency_seconds": min(latencies), "max_latency_seconds": max(latencies),
            "all_three_raw_outputs_agree": len({row["raw_response_text"] for row in rows}) == 1,
            "all_three_decisions_agree": len(set(decisions)) == 1,
            "exact_outputs_for_failures": [
                {"repetition": row["repetition"], "raw_response_text": row["raw_response_text"],
                 "parsed_json": row["parsed_json"], "parse_exception": row["parse_exception"],
                 "exception": row["exception"], "normalized_verdict": row["normalized_verdict"]}
                for row in failures
            ],
        }
    summary = {
        "experiment": "isolated_case2_judge_interface",
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "case_id": "case_02_direct_contradiction",
        "question": question, "claim": claim, "evidence": evidence,
        "evidence_id": CASE_EVIDENCE_ID,
        "expected_verdict": "CONTRADICTED",
        "configuration": {"model": MODEL, "think": THINK, "temperature": TEMPERATURE,
                           "format": "json", "stream": True, "timeout_seconds": TIMEOUT_SECONDS},
        "total_calls": len(all_results), "expected_calls": 12,
        "per_variant": groups,
        "files_and_git": {"git": git, "protected_hashes_before": before_hashes,
                           "protected_hashes_after": after_hashes, "protected_hash_changes": hash_changes,
                           "existing_tests_modified": False},
        "artifacts": {"script": Path(__file__).name, "results_jsonl": RESULTS_PATH.name,
                      "summary_json": SUMMARY_PATH.name},
        "source_files_modified_by_experiment": [],
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False, default=str), flush=True)
    return 0 if len(all_results) == 12 else 1


CASE_EVIDENCE_ID = ""

if __name__ == "__main__":
    raise SystemExit(main())
