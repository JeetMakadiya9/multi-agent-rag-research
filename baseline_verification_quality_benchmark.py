"""Isolated qwen3:4b claim-verification quality/latency benchmark.

No production pipeline functions are called or edited. Each request is sent
directly to local Ollama, and its complete content/thinking stream is appended
to baseline_verification_quality_results.jsonl beside this script.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASE_URL = "http://127.0.0.1:11434"
MODEL = "qwen3:4b"
TIMEOUT_SECONDS = 120
RESULTS_PATH = Path(__file__).with_name("baseline_verification_quality_results.jsonl")

SUPPORTED = "SUPPORTED"
CONTRADICTED = "CONTRADICTED"
INSUFFICIENT = "INSUFFICIENT_EVIDENCE"
VALID_VERDICTS = {SUPPORTED, CONTRADICTED, INSUFFICIENT}

SYSTEM_PROMPT = "Verify claims strictly from evidence. Return JSON only."
VERIFICATION_INSTRUCTIONS = '''\
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

CASES = [
    {
        "case_id": "CASE1_SUPPORTED",
        "claim": "Water freezes at 0 degrees Celsius at standard atmospheric pressure.",
        "evidence": "At standard atmospheric pressure, pure water freezes at 0 degrees Celsius.",
        "expected": SUPPORTED,
    },
    {
        "case_id": "CASE2_CONTRADICTED",
        "claim": "Water freezes at 50 degrees Celsius at standard atmospheric pressure.",
        "evidence": "At standard atmospheric pressure, pure water freezes at 0 degrees Celsius.",
        "expected": CONTRADICTED,
    },
    {
        "case_id": "CASE3_INSUFFICIENT",
        "claim": "Company X acquired Company Y in 2024.",
        "evidence": "Company X announced a partnership with Company Y in 2024.",
        "expected": INSUFFICIENT,
    },
    {
        "case_id": "CASE4_SUPPORTED_PARAPHRASE",
        "claim": "Earth revolves around the Sun.",
        "evidence": "The Earth orbits the Sun.",
        "expected": SUPPORTED,
    },
    {
        "case_id": "CASE5_IRRELEVANT",
        "claim": "Python was first released in 1991.",
        "evidence": "Water freezes at 0 degrees Celsius at standard atmospheric pressure.",
        "expected": INSUFFICIENT,
    },
    {
        "case_id": "CASE6_PARTIAL_SUPPORT",
        "claim": "The system uses FAISS and BM25 for hybrid retrieval.",
        "evidence": "The system uses FAISS for dense vector retrieval.",
        "expected": INSUFFICIENT,
    },
    {
        "case_id": "CASE7_CONTRADICTION",
        "claim": "The model uses only BM25 retrieval.",
        "evidence": "The system combines dense FAISS retrieval with BM25 lexical retrieval.",
        "expected": CONTRADICTED,
    },
    {
        "case_id": "CASE8_DIRECT_SUPPORT",
        "claim": "FAISS is used for dense vector retrieval.",
        "evidence": "FAISS performs dense vector retrieval using vector similarity search.",
        "expected": SUPPORTED,
    },
]

CONFIGS = {
    "A": {"name": "default_thinking_no_format", "think": "omitted", "format": "omitted"},
    "B": {"name": "think_false_no_format", "think": False, "format": "omitted"},
    "C": {"name": "think_false_json_format", "think": False, "format": "json"},
}


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


def json_objects(text: str) -> list[dict[str, Any]]:
    """Return valid JSON objects embedded in a possibly verbose response."""
    found: list[dict[str, Any]] = []
    for start, char in enumerate(text):
        if char != "{":
            continue
        depth = 0
        in_string = False
        escaped = False
        for end in range(start, len(text)):
            current = text[end]
            if in_string:
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == '"':
                    in_string = False
                continue
            if current == '"':
                in_string = True
            elif current == "{":
                depth += 1
            elif current == "}":
                depth -= 1
                if depth == 0:
                    try:
                        candidate = json.loads(text[start : end + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(candidate, dict):
                        found.append(candidate)
                    break
    return found


def build_messages(case: dict[str, str]) -> list[dict[str, str]]:
    evidence_text = (
        f'[E1] synthetic-evidence.txt, page 1\n'
        f'Retrieval score: 1.0000\n{case["evidence"]}'
    )
    prompt = (
        VERIFICATION_INSTRUCTIONS
        + f'\n\nCLAIM:\n{case["claim"]}\n\nEVIDENCE:\n{evidence_text}\n'
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def run_call(case: dict[str, str], config_key: str, repetition: int) -> dict[str, Any]:
    config = CONFIGS[config_key]
    messages = build_messages(case)
    payload: dict[str, Any] = {
        "model": MODEL,
        "messages": messages,
        "stream": True,
        "keep_alive": "30m",
        "options": {"temperature": 0.0},
    }
    if config["think"] != "omitted":
        payload["think"] = config["think"]
    if config["format"] != "omitted":
        payload["format"] = config["format"]

    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    first_content = None
    first_thinking = None
    content_parts: list[str] = []
    thinking_parts: list[str] = []
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
                message = event.get("message") or {}
                content = str(message.get("content") or "")
                thinking = str(message.get("thinking") or "")
                now = time.perf_counter()
                if content and first_content is None:
                    first_content = now - started
                if thinking and first_thinking is None:
                    first_thinking = now - started
                content_parts.append(content)
                thinking_parts.append(thinking)
                if event.get("error"):
                    error = str(event["error"])
                    break
                if event.get("done"):
                    final_event = event
                    break
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    ended_at = datetime.now(timezone.utc)
    content_text = "".join(content_parts)
    thinking_text = "".join(thinking_parts)
    objects = json_objects(content_text)
    production_object = objects[0] if objects else None
    last_object = objects[-1] if objects else None
    raw_verdict = (
        str(production_object.get("verdict", "")).strip().upper()
        if production_object
        else None
    )
    production_parsed = production_object is not None
    verdict_valid = raw_verdict in VALID_VERDICTS
    # Match verify_claim(): an absent or unrecognized verdict is normalized to
    # INSUFFICIENT_EVIDENCE, while separately retaining that it was invalid.
    actual_verdict = raw_verdict if verdict_valid else INSUFFICIENT
    evidence_ids = production_object.get("evidence_ids") if production_object else None
    if not isinstance(evidence_ids, list):
        evidence_ids = []
    evidence_ids = [str(value) for value in evidence_ids]

    record = {
        "case_id": case["case_id"],
        "expected_verdict": case["expected"],
        "configuration": config_key,
        "configuration_name": config["name"],
        "repetition": repetition,
        "model": MODEL,
        "thinking_configuration": config["think"],
        "format_configuration": config["format"],
        "temperature": 0.0,
        "start_utc": started_at.isoformat(),
        "end_utc": ended_at.isoformat(),
        "prompt_characters": sum(len(message["content"]) for message in messages),
        "first_content_latency_seconds": first_content,
        "first_thinking_latency_seconds": first_thinking,
        "total_latency_seconds": time.perf_counter() - started,
        "response_characters": len(content_text),
        "thinking_characters": len(thinking_text),
        "completed": final_event is not None and error is None,
        "error": error,
        "production_parser_success": production_parsed,
        "valid_verdict": verdict_valid,
        "raw_verdict": raw_verdict,
        "actual_verdict_after_pipeline_normalization": actual_verdict,
        "last_json_verdict": (
            str(last_object.get("verdict", "")).strip().upper()
            if last_object
            else None
        ),
        "evidence_ids_from_first_json": evidence_ids,
        "ollama_metadata": {
            key: final_event.get(key)
            for key in (
                "total_duration", "load_duration", "prompt_eval_count",
                "prompt_eval_duration", "eval_count", "eval_duration",
                "done_reason", "created_at",
            )
            if final_event is not None and key in final_event
        },
        "claim": case["claim"],
        "evidence": case["evidence"],
        "raw_content": content_text,
        "raw_thinking": thinking_text,
    }
    with RESULTS_PATH.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, ensure_ascii=False) + "\n")
        output.flush()
    print(json.dumps({
        key: record[key]
        for key in (
            "case_id", "configuration", "repetition", "start_utc", "end_utc",
            "first_content_latency_seconds", "first_thinking_latency_seconds",
            "total_latency_seconds", "response_characters", "thinking_characters",
            "completed", "production_parser_success", "valid_verdict",
            "raw_verdict", "actual_verdict_after_pipeline_normalization",
            "evidence_ids_from_first_json", "ollama_metadata", "error",
        )
    }, ensure_ascii=False), flush=True)
    return record


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def load_records() -> list[dict[str, Any]]:
    if not RESULTS_PATH.exists():
        return []
    return [json.loads(line) for line in RESULTS_PATH.read_text(encoding="utf-8").splitlines() if line]


def summarize() -> dict[str, Any]:
    records = load_records()
    summaries: dict[str, Any] = {}
    for key, config in CONFIGS.items():
        subset = [item for item in records if item["configuration"] == key]
        completed = [item for item in subset if item["completed"]]
        latencies = [item["total_latency_seconds"] for item in completed]
        parsed = sum(item["production_parser_success"] for item in completed)
        valid = sum(item["valid_verdict"] for item in completed)
        correct = sum(
            item["actual_verdict_after_pipeline_normalization"] == item["expected_verdict"]
            for item in completed
        )
        per_class = {}
        for expected in sorted({item["expected_verdict"] for item in subset}):
            group = [item for item in completed if item["expected_verdict"] == expected]
            per_class[expected] = {
                "count": len(group),
                "correct": sum(
                    item["actual_verdict_after_pipeline_normalization"] == expected
                    for item in group
                ),
                "accuracy": (
                    sum(item["actual_verdict_after_pipeline_normalization"] == expected for item in group)
                    / len(group)
                    if group else None
                ),
            }
        summaries[key] = {
            "configuration_name": config["name"],
            "calls": len(subset),
            "completed_calls": len(completed),
            "parse_success_rate": parsed / len(completed) if completed else None,
            "valid_verdict_rate": valid / len(completed) if completed else None,
            "verdict_accuracy": correct / len(completed) if completed else None,
            "per_class_accuracy": per_class,
            "latency_seconds": {
                "mean": statistics.mean(latencies) if latencies else None,
                "median": statistics.median(latencies) if latencies else None,
                "min": min(latencies) if latencies else None,
                "max": max(latencies) if latencies else None,
            },
        }

    repeated_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    by_case_config: dict[tuple[str, str], list[dict[str, Any]]] = {}
    by_case: dict[str, dict[str, list[str]]] = {}
    for item in records:
        key = (item["case_id"], item["configuration"])
        repeated_groups.setdefault(key, []).append(item)
        by_case_config.setdefault(key, []).append(item)
        by_case.setdefault(item["case_id"], {}).setdefault(item["configuration"], []).append(
            item["actual_verdict_after_pipeline_normalization"]
        )

    repeat_disagreements = []
    for (case_id, config_key), group in sorted(repeated_groups.items()):
        observed = sorted({item["actual_verdict_after_pipeline_normalization"] for item in group})
        if len(group) > 1 and len(observed) > 1:
            repeat_disagreements.append({
                "case_id": case_id,
                "configuration": config_key,
                "verdicts": observed,
                "repetitions": len(group),
            })

    configuration_disagreements = []
    evidence_selection_differences = []
    for case in CASES:
        observed = {
            config_key: sorted(set(verdicts))
            for config_key, verdicts in by_case.get(case["case_id"], {}).items()
        }
        union = {value for values in observed.values() for value in values}
        if len(union) > 1:
            configuration_disagreements.append({
                "case_id": case["case_id"],
                "expected": case["expected"],
                "actual_by_configuration": observed,
            })
        selections = {}
        for config_key in CONFIGS:
            selections[config_key] = sorted({
                tuple(item["evidence_ids_from_first_json"])
                for item in by_case_config.get((case["case_id"], config_key), [])
            })
        if len({tuple(value) for values in selections.values() for value in values}) > 1:
            evidence_selection_differences.append({
                "case_id": case["case_id"],
                "evidence_ids_by_configuration": selections,
            })

    invalid_outputs = [
        {
            "case_id": item["case_id"],
            "configuration": item["configuration"],
            "repetition": item["repetition"],
            "raw_verdict": item["raw_verdict"],
            "parse_success": item["production_parser_success"],
        }
        for item in records
        if item["completed"] and (not item["production_parser_success"] or not item["valid_verdict"])
    ]

    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "ollama_version": api_json("/api/version").get("version"),
        "results_file": str(RESULTS_PATH),
        "record_count": len(records),
        "configuration_summaries": summaries,
        "configuration_disagreements": configuration_disagreements,
        "repeated_run_disagreements": repeat_disagreements,
        "invalid_or_unparseable_outputs": invalid_outputs,
        "evidence_selection_differences": evidence_selection_differences,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=CONFIGS, help="Run only one configuration for targeted repeats.")
    parser.add_argument("--case", action="append", help="Case id; repeatable for targeted runs.")
    parser.add_argument("--repeats", type=int, help="Repetition count; defaults to the initial schedule.")
    parser.add_argument("--append", action="store_true", help="Append to existing JSONL instead of starting fresh.")
    args = parser.parse_args()

    version = api_json("/api/version")
    tags = api_json("/api/tags")
    available = [item.get("name") for item in tags.get("models", [])]
    if MODEL not in available:
        raise SystemExit(f"Required model {MODEL!r} not found. Available models: {available}")
    show = api_json("/api/show", {"model": MODEL})
    print(json.dumps({
        "benchmark_start_utc": datetime.now(timezone.utc).isoformat(),
        "interpreter": __import__("sys").executable,
        "ollama_version": version.get("version"),
        "model": MODEL,
        "model_thinking_metadata": show.get("model_info", {}).get("general.finetune"),
        "thinking_capability": show.get("capabilities"),
        "thinking_configuration_metadata": show.get("thinking"),
        "initial_ollama_ps": api_json("/api/ps"),
        "results_file": str(RESULTS_PATH),
    }, ensure_ascii=False), flush=True)

    if not args.append and not args.config:
        RESULTS_PATH.write_text("", encoding="utf-8")
    selected_cases = [case for case in CASES if not args.case or case["case_id"] in args.case]
    if args.case and {case["case_id"] for case in selected_cases} != set(args.case):
        unknown = set(args.case) - {case["case_id"] for case in CASES}
        raise SystemExit(f"Unknown case id(s): {sorted(unknown)}")

    if args.config:
        schedule = {args.config: args.repeats or 1}
    else:
        schedule = {key: (args.repeats or (3 if key == "C" else 1)) for key in CONFIGS}

    for case in selected_cases:
        for config_key, repetitions in schedule.items():
            existing_count = 0
            if args.append:
                existing_count = sum(
                    record["case_id"] == case["case_id"]
                    and record["configuration"] == config_key
                    for record in load_records()
                )
            for repetition in range(existing_count + 1, existing_count + repetitions + 1):
                print(
                    f"START {case['case_id']} config={config_key} "
                    f"repetition={repetition} at {datetime.now(timezone.utc).isoformat()}",
                    flush=True,
                )
                run_call(case, config_key, repetition)

    summary = summarize()
    summary_path = RESULTS_PATH.with_name("baseline_verification_quality_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("SUMMARY=" + json.dumps(summary, ensure_ascii=False), flush=True)
    print(f"SUMMARY_FILE={summary_path}", flush=True)


if __name__ == "__main__":
    main()
