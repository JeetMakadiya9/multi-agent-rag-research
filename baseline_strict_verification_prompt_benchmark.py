"""Paired prompt-quality benchmark for constrained qwen3:4b verification.

Calls local Ollama directly. It never imports or modifies production code.
Full raw responses are written to baseline_strict_verification_results.jsonl.
"""

from __future__ import annotations

import json
import statistics
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASE_URL = "http://127.0.0.1:11434"
MODEL = "qwen3:4b"
TEMPERATURE = 0.0
TIMEOUT_SECONDS = 120
REPETITIONS = 3
RESULTS_PATH = Path(__file__).with_name("baseline_strict_verification_results.jsonl")
SUMMARY_PATH = Path(__file__).with_name("baseline_strict_verification_summary.json")

SUPPORTED = "SUPPORTED"
CONTRADICTED = "CONTRADICTED"
INSUFFICIENT = "INSUFFICIENT_EVIDENCE"
VALID_VERDICTS = {SUPPORTED, CONTRADICTED, INSUFFICIENT}

SYSTEM_PROMPT = "Verify claims strictly from evidence. Return JSON only."

# Copied from the current claim_verification.verify_claim() instruction text.
EXISTING_PROMPT = '''\
You are a strict claim-level fact verifier.

Evaluate the CLAIM using ONLY the supplied EVIDENCE.

Important distinction:
- Evidence being topically relevant is NOT enough to support a claim.
- SUPPORTED requires that the evidence entails the claim.
- CONTRADICTED requires evidence that conflicts with the claim.
- If the evidence is relevant but does not establish or contradict the claim,
  use INSUFFICIENT_EVIDENCE.
- Do not use outside knowledge.
- Do not infer missing facts.
- Confidence must describe confidence in the verdict, not source quality.

Return JSON only with this schema:
{
  "verdict": "SUPPORTED | CONTRADICTED | INSUFFICIENT_EVIDENCE",
  "confidence": 0.0,
  "evidence_ids": ["E1"],
  "explanation": "short evidence-grounded explanation"
}
'''

STRICT_PROMPT = '''\
You are a strict claim-level fact verifier. Base the verdict only on the
supplied evidence. Do not use outside knowledge.

VERDICT DEFINITIONS:

SUPPORTED:
The evidence establishes the complete claim, including all important
entities, relationships, quantities, dates, and conditions stated by the
claim.

CONTRADICTED:
The evidence directly conflicts with an important part of the claim.

INSUFFICIENT_EVIDENCE:
The evidence is relevant or partially related but does not establish the
complete claim, or there is not enough information to determine whether the
claim is true. If the supplied evidence conflicts with itself and does not
resolve the claim, use INSUFFICIENT_EVIDENCE.

REQUIRED RULES:
- Do not infer missing facts.
- Do not treat partial support as full support.
- Every major component of a compound claim must be supported.
- If one important component is unsupported and there is no contradiction,
  return INSUFFICIENT_EVIDENCE.
- Base the verdict only on supplied evidence.
- Select only evidence IDs that support or contradict the chosen verdict.
- Confidence describes confidence in the verdict, not source quality.

CALIBRATION EXAMPLE:
Claim:
"The system uses FAISS and BM25 for hybrid retrieval."

Evidence:
"The system uses FAISS for dense vector retrieval."

Correct verdict:
INSUFFICIENT_EVIDENCE

Return JSON only with this schema:
{
  "verdict": "SUPPORTED | CONTRADICTED | INSUFFICIENT_EVIDENCE",
  "confidence": 0.0,
  "evidence_ids": ["E1"],
  "explanation": "short evidence-grounded explanation"
}
'''

CASES = [
    {
        "case_id": "CASE1_DIRECT_SUPPORT",
        "claim": "Water freezes at 0 degrees Celsius at standard atmospheric pressure.",
        "evidence": [("E1", "At standard atmospheric pressure, pure water freezes at 0 degrees Celsius.")],
        "expected": SUPPORTED,
    },
    {
        "case_id": "CASE2_DIRECT_CONTRADICTION",
        "claim": "Water freezes at 50 degrees Celsius at standard atmospheric pressure.",
        "evidence": [("E1", "At standard atmospheric pressure, pure water freezes at 0 degrees Celsius.")],
        "expected": CONTRADICTED,
    },
    {
        "case_id": "CASE3_INSUFFICIENT_ACQUISITION",
        "claim": "Company X acquired Company Y in 2024.",
        "evidence": [("E1", "Company X announced a partnership with Company Y in 2024.")],
        "expected": INSUFFICIENT,
    },
    {
        "case_id": "CASE4_PARAPHRASE_SUPPORT",
        "claim": "Earth revolves around the Sun.",
        "evidence": [("E1", "The Earth orbits the Sun.")],
        "expected": SUPPORTED,
    },
    {
        "case_id": "CASE5_IRRELEVANT_EVIDENCE",
        "claim": "Python was first released in 1991.",
        "evidence": [("E1", "Water freezes at 0 degrees Celsius at standard atmospheric pressure.")],
        "expected": INSUFFICIENT,
    },
    {
        "case_id": "CASE6_KNOWN_PARTIAL_SUPPORT",
        "claim": "The system uses FAISS and BM25 for hybrid retrieval.",
        "evidence": [("E1", "The system uses FAISS for dense vector retrieval.")],
        "expected": INSUFFICIENT,
    },
    {
        "case_id": "CASE7_COMPOUND_ONE_UNSUPPORTED",
        "claim": "The system uses FAISS for dense retrieval and PostgreSQL to store documents.",
        "evidence": [("E1", "FAISS performs dense vector retrieval for the system.")],
        "expected": INSUFFICIENT,
    },
    {
        "case_id": "CASE8_DATE",
        "claim": "The Apollo 11 Moon landing occurred in 1969.",
        "evidence": [("E1", "Apollo 11 landed on the Moon on 20 July 1969.")],
        "expected": SUPPORTED,
    },
    {
        "case_id": "CASE9_QUANTITY",
        "claim": "The archive contains 48 records.",
        "evidence": [("E1", "The archive contains 48 records.")],
        "expected": SUPPORTED,
    },
    {
        "case_id": "CASE10_CONFLICTING_EVIDENCE",
        "claim": "The library has 10 maintainers.",
        "evidence": [
            ("E1", "The library has 10 maintainers."),
            ("E2", "The library has 12 maintainers."),
        ],
        "expected": INSUFFICIENT,
    },
]


def api_json(path: str, payload: dict | None = None, timeout: int = 10) -> Any:
    if payload is None:
        request = urllib.request.Request(BASE_URL + path, method="GET")
    else:
        request = urllib.request.Request(
            BASE_URL + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def make_messages(case: dict[str, Any], prompt_template: str) -> list[dict[str, str]]:
    blocks = []
    for evidence_id, text in case["evidence"]:
        blocks.append(
            f"[{evidence_id}] synthetic-evidence.txt, page 1\n"
            f"Retrieval score: 1.0000\n{text}"
        )
    prompt = (
        prompt_template
        + f"\nCLAIM:\n{case['claim']}\n\nEVIDENCE:\n"
        + "\n\n".join(blocks)
        + "\n"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def run_call(case: dict[str, Any], prompt_key: str, repetition: int) -> dict[str, Any]:
    prompt_template = EXISTING_PROMPT if prompt_key == "existing" else STRICT_PROMPT
    messages = make_messages(case, prompt_template)
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": True,
        "keep_alive": "30m",
        "think": False,
        "format": "json",
        "options": {"temperature": TEMPERATURE},
    }
    start_wall = datetime.now(timezone.utc)
    start = time.perf_counter()
    first_content = None
    content_parts: list[str] = []
    final_event = None
    error = None
    request = urllib.request.Request(
        BASE_URL + "/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            for line in response:
                if not line.strip():
                    continue
                event = json.loads(line.decode("utf-8"))
                content = str((event.get("message") or {}).get("content") or "")
                if content and first_content is None:
                    first_content = time.perf_counter() - start
                content_parts.append(content)
                if event.get("error"):
                    error = str(event["error"])
                    break
                if event.get("done"):
                    final_event = event
                    break
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    end_wall = datetime.now(timezone.utc)
    raw_response = "".join(content_parts)
    parsed = None
    parse_error = None
    try:
        parsed = json.loads(raw_response.strip())
        if not isinstance(parsed, dict):
            raise ValueError("JSON response was not an object.")
    except (json.JSONDecodeError, ValueError) as exc:
        parse_error = f"{type(exc).__name__}: {exc}"

    raw_verdict = str(parsed.get("verdict", "")).strip().upper() if parsed else None
    valid_verdict = raw_verdict in VALID_VERDICTS
    actual_verdict = raw_verdict if valid_verdict else None
    evidence_ids = parsed.get("evidence_ids", []) if parsed else []
    if not isinstance(evidence_ids, list):
        evidence_ids = []

    record = {
        "case_id": case["case_id"],
        "expected_verdict": case["expected"],
        "prompt_configuration": prompt_key,
        "repetition": repetition,
        "model": MODEL,
        "thinking_configuration": False,
        "format_configuration": "json",
        "temperature": TEMPERATURE,
        "start_utc": start_wall.isoformat(),
        "end_utc": end_wall.isoformat(),
        "prompt_characters": sum(len(message["content"]) for message in messages),
        "first_content_latency_seconds": first_content,
        "total_latency_seconds": time.perf_counter() - start,
        "response_characters": len(raw_response),
        "completed": final_event is not None and error is None,
        "parse_success": parsed is not None,
        "parse_error": parse_error,
        "valid_verdict": valid_verdict,
        "actual_verdict": actual_verdict,
        "raw_verdict": raw_verdict,
        "correct": actual_verdict == case["expected"],
        "evidence_ids": [str(value) for value in evidence_ids],
        "ollama_metadata": {
            key: final_event.get(key)
            for key in (
                "total_duration", "load_duration", "prompt_eval_count",
                "prompt_eval_duration", "eval_count", "eval_duration",
                "done_reason", "created_at",
            )
            if final_event is not None and key in final_event
        },
        "error": error,
        "claim": case["claim"],
        "evidence": case["evidence"],
        "raw_response": raw_response,
    }
    with RESULTS_PATH.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, ensure_ascii=False) + "\n")
        output.flush()

    print(json.dumps({
        key: record[key]
        for key in (
            "case_id", "prompt_configuration", "repetition", "start_utc", "end_utc",
            "first_content_latency_seconds", "total_latency_seconds", "response_characters",
            "completed", "parse_success", "valid_verdict", "actual_verdict", "correct",
            "evidence_ids", "ollama_metadata", "error",
        )
    }, ensure_ascii=False), flush=True)
    return record


def config_summary(records: list[dict[str, Any]], prompt_key: str) -> dict[str, Any]:
    group = [item for item in records if item["prompt_configuration"] == prompt_key]
    completed = [item for item in group if item["completed"]]
    parsed = [item for item in completed if item["parse_success"]]
    latencies = [item["total_latency_seconds"] for item in completed]
    classes = {}
    for label in sorted({item["expected_verdict"] for item in group}):
        subset = [item for item in completed if item["expected_verdict"] == label]
        classes[label] = {
            "calls": len(subset),
            "correct": sum(item["correct"] for item in subset),
            "accuracy": sum(item["correct"] for item in subset) / len(subset) if subset else None,
        }
    repeated = []
    for case in CASES:
        subset = [item for item in completed if item["case_id"] == case["case_id"]]
        if len(subset) > 1:
            verdicts = sorted({item["actual_verdict"] for item in subset})
            repeated.append({
                "case_id": case["case_id"],
                "runs": len(subset),
                "verdicts": verdicts,
                "consistent": len(verdicts) == 1,
            })
    return {
        "calls": len(group),
        "completed_calls": len(completed),
        "parse_success_rate": len(parsed) / len(completed) if completed else None,
        "verdict_accuracy": sum(item["correct"] for item in completed) / len(completed) if completed else None,
        "per_class_accuracy": classes,
        "latency_seconds": {
            "mean": statistics.mean(latencies) if latencies else None,
            "median": statistics.median(latencies) if latencies else None,
            "min": min(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
        },
        "repeated_run_consistency": repeated,
    }


def main() -> None:
    if RESULTS_PATH.exists():
        raise SystemExit(f"Refusing to overwrite existing raw-results file: {RESULTS_PATH}")
    version = api_json("/api/version")
    tags = api_json("/api/tags")
    models = [item.get("name") for item in tags.get("models", [])]
    if MODEL not in models:
        raise SystemExit(f"Required model {MODEL!r} is unavailable. Found: {models}")
    show = api_json("/api/show", {"model": MODEL})
    print(json.dumps({
        "benchmark_start_utc": datetime.now(timezone.utc).isoformat(),
        "interpreter": __import__("sys").executable,
        "ollama_version": version.get("version"),
        "model": MODEL,
        "think": False,
        "format": "json",
        "temperature": TEMPERATURE,
        "repetitions_per_case_and_prompt": REPETITIONS,
        "model_thinking_metadata": show.get("thinking"),
        "model_capabilities": show.get("capabilities"),
        "initial_ollama_ps": api_json("/api/ps"),
        "results_path": str(RESULTS_PATH),
    }, ensure_ascii=False), flush=True)

    records = []
    for case in CASES:
        for prompt_key in ("existing", "strict"):
            for repetition in range(1, REPETITIONS + 1):
                print(
                    f"START {case['case_id']} prompt={prompt_key} "
                    f"rep={repetition}/{REPETITIONS} at {datetime.now(timezone.utc).isoformat()}",
                    flush=True,
                )
                records.append(run_call(case, prompt_key, repetition))

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "ollama_version": version.get("version"),
        "calls": len(records),
        "cases": [{
            "case_id": case["case_id"],
            "expected": case["expected"],
            "verdicts_by_prompt": {
                key: [
                    item["actual_verdict"]
                    for item in records
                    if item["case_id"] == case["case_id"] and item["prompt_configuration"] == key
                ]
                for key in ("existing", "strict")
            },
        } for case in CASES],
        "configuration_summaries": {
            "existing": config_summary(records, "existing"),
            "strict": config_summary(records, "strict"),
        },
        "prompt_disagreements": [
            {
                "case_id": case["case_id"],
                "existing_verdicts": sorted({
                    item["actual_verdict"] for item in records
                    if item["case_id"] == case["case_id"] and item["prompt_configuration"] == "existing"
                }),
                "strict_verdicts": sorted({
                    item["actual_verdict"] for item in records
                    if item["case_id"] == case["case_id"] and item["prompt_configuration"] == "strict"
                }),
            }
            for case in CASES
            if {
                item["actual_verdict"] for item in records
                if item["case_id"] == case["case_id"] and item["prompt_configuration"] == "existing"
            }
            != {
                item["actual_verdict"] for item in records
                if item["case_id"] == case["case_id"] and item["prompt_configuration"] == "strict"
            }
        ],
        "parse_failures": [
            {"case_id": item["case_id"], "prompt": item["prompt_configuration"], "rep": item["repetition"], "error": item["parse_error"]}
            for item in records if not item["parse_success"]
        ],
        "invalid_verdicts": [
            {"case_id": item["case_id"], "prompt": item["prompt_configuration"], "rep": item["repetition"], "raw_verdict": item["raw_verdict"]}
            for item in records if item["parse_success"] and not item["valid_verdict"]
        ],
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("SUMMARY=" + json.dumps(summary, ensure_ascii=False), flush=True)
    print(f"SUMMARY_PATH={SUMMARY_PATH}", flush=True)


if __name__ == "__main__":
    main()
