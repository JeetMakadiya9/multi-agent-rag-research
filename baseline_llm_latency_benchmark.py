"""Controlled local Ollama latency benchmark for Experiment B prompts.

This is an analysis script. It does not import or modify the RAG pipeline.
It sends read-only chat requests to the configured local Ollama endpoint and
prints per-call timing, streamed response metadata, and Ollama/GPU snapshots.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASE_URL = "http://127.0.0.1:11434"
MODEL = "qwen3:4b"
TIMEOUT_SECONDS = 120

ANSWER = "Water freezes at 0 degrees Celsius."
EXTRACTION_SYSTEM = "Extract factual claims. Return JSON only."
EXTRACTION_PROMPT = f'''\
You are the claim extraction component of a research fact-verification system.

Extract only atomic factual claims from the answer below.

Rules:
- Split compound statements into separate claims when needed.
- Keep claims specific and independently verifiable.
- Do not create claims that are not stated or clearly implied by the answer.
- Exclude greetings, opinions, recommendations, instructions, and meta-text.
- Preserve important names, dates, numbers, and relationships exactly.
- Return JSON only.

Required JSON schema:
{{
  "claims": [
    {{"claim_id": "C1", "text": "..."}},
    {{"claim_id": "C2", "text": "..."}}
  ]
}}

ANSWER:
{ANSWER}
'''
VERIFY_SYSTEM = "Verify claims strictly from evidence. Return JSON only."
VERIFY_PROMPT = '''\
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

CLAIM:
Water freezes at 0 degrees Celsius.

EVIDENCE:
[E1] synthetic.txt, page 1
Retrieval score: 0.9000
Pure water freezes at 0 degrees Celsius at standard atmospheric pressure.
'''


def request_json(path: str, payload: dict | None = None, timeout: int = 10) -> Any:
    url = BASE_URL + path
    if payload is None:
        request = urllib.request.Request(url, method="GET")
    else:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def ollama_ps() -> dict:
    try:
        return request_json("/api/ps")
    except Exception as exc:  # diagnostic data; preserve exact failure in result
        return {"diagnostic_error": f"{type(exc).__name__}: {exc}"}


def gpu_snapshot() -> dict:
    result: dict[str, Any] = {}
    commands = {
        "gpu": [
            "nvidia-smi",
            "--query-gpu=name,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        "compute_processes": [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
    }
    for label, command in commands.items():
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True, timeout=5, check=False
            )
            result[label] = {
                "returncode": completed.returncode,
                "stdout": completed.stdout.strip(),
                "stderr": completed.stderr.strip(),
            }
        except Exception as exc:
            result[label] = {"error": f"{type(exc).__name__}: {exc}"}
    return result


def call_benchmark(name: str, messages: list[dict[str, str]], *, think: Any = "omitted",
                   output_format: Any = "omitted", cold_label: bool = False) -> dict:
    payload: dict[str, Any] = {
        "model": MODEL,
        "messages": messages,
        "stream": True,
        "keep_alive": "30m",
        "options": {"temperature": 0.0},
    }
    if think != "omitted":
        payload["think"] = think
    if output_format != "omitted":
        payload["format"] = output_format

    result: dict[str, Any] = {
        "name": name,
        "model": MODEL,
        "thinking_configuration": think,
        "temperature": payload["options"]["temperature"],
        "format_configuration": output_format,
        "prompt_characters": sum(len(message["content"]) for message in messages),
        "cold_candidate": cold_label,
        "request_payload_config": {
            "stream": True,
            "think": think,
            "temperature": payload["options"]["temperature"],
            "format": output_format,
        },
        "ollama_ps_before": ollama_ps(),
        "gpu_before": gpu_snapshot(),
    }
    start_wall = datetime.now(timezone.utc)
    start = time.perf_counter()
    result["start_utc"] = start_wall.isoformat()
    first_content_seconds = None
    first_any_generated_seconds = None
    chunks: list[str] = []
    thinking_chunks: list[str] = []
    final_event: dict[str, Any] | None = None
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
                now = time.perf_counter()
                message = event.get("message") or {}
                content = message.get("content") or ""
                thinking_text = message.get("thinking") or ""
                if (content or thinking_text) and first_any_generated_seconds is None:
                    first_any_generated_seconds = now - start
                if content and first_content_seconds is None:
                    first_content_seconds = now - start
                if content:
                    chunks.append(str(content))
                if thinking_text:
                    thinking_chunks.append(str(thinking_text))
                if event.get("error"):
                    error = str(event["error"])
                    break
                if event.get("done"):
                    final_event = event
                    break
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    end = time.perf_counter()
    result.update({
        "end_utc": datetime.now(timezone.utc).isoformat(),
        "time_to_first_content_seconds": first_content_seconds,
        "time_to_first_generated_content_seconds": first_any_generated_seconds,
        "total_generation_seconds": end - start,
        "response": "".join(chunks),
        "response_characters": len("".join(chunks)),
        "thinking_characters": len("".join(thinking_chunks)),
        "completed": final_event is not None and error is None,
        "error": error,
        "ollama_final_event_metadata": {
            key: final_event.get(key)
            for key in (
                "total_duration", "load_duration", "prompt_eval_count",
                "prompt_eval_duration", "eval_count", "eval_duration",
                "done_reason", "created_at",
            )
            if final_event is not None and key in final_event
        },
        "ollama_ps_after": ollama_ps(),
        "gpu_after": gpu_snapshot(),
    })
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main() -> None:
    try:
        version = request_json("/api/version")
        tags = request_json("/api/tags")
    except Exception as exc:
        raise SystemExit(f"Local Ollama preflight failed: {type(exc).__name__}: {exc}")

    initial_ps = ollama_ps()
    initial_gpu = gpu_snapshot()
    available = [item.get("name") for item in tags.get("models", [])]
    if MODEL not in available:
        raise SystemExit(f"Model {MODEL!r} is not available. Ollama tags: {available}")

    print(json.dumps({
        "benchmark_start_utc": datetime.now(timezone.utc).isoformat(),
        "interpreter": str(Path(__import__("sys").executable)),
        "ollama_version": version.get("version"),
        "model": MODEL,
        "model_details": request_json("/api/show", {"model": MODEL}),
        "initial_ollama_ps": initial_ps,
        "initial_gpu": initial_gpu,
        "requested_timeout_seconds": TIMEOUT_SECONDS,
    }, ensure_ascii=False, default=str), flush=True)

    results = []
    results.append(call_benchmark(
        "tiny_cold_candidate",
        [{"role": "user", "content": "Reply with OK."}],
        cold_label=True,
    ))
    results.append(call_benchmark(
        "tiny_warm_repeat",
        [{"role": "user", "content": "Reply with OK."}],
    ))
    extraction_messages = [
        {"role": "system", "content": EXTRACTION_SYSTEM},
        {"role": "user", "content": EXTRACTION_PROMPT},
    ]
    results.append(call_benchmark(
        "extraction_current_default_thinking_no_format",
        extraction_messages,
    ))
    results.append(call_benchmark(
        "extraction_think_false_no_format",
        extraction_messages,
        think=False,
    ))
    results.append(call_benchmark(
        "extraction_think_false_json_format",
        extraction_messages,
        think=False,
        output_format="json",
    ))
    verification_messages = [
        {"role": "system", "content": VERIFY_SYSTEM},
        {"role": "user", "content": VERIFY_PROMPT},
    ]
    results.append(call_benchmark(
        "verification_current_default_thinking_no_format",
        verification_messages,
    ))
    results.append(call_benchmark(
        "verification_think_false_no_format",
        verification_messages,
        think=False,
    ))
    results.append(call_benchmark(
        "verification_think_false_json_format",
        verification_messages,
        think=False,
        output_format="json",
    ))
    results.append(call_benchmark(
        "verification_think_true_json_format",
        verification_messages,
        think=True,
        output_format="json",
    ))

    completed = sum(item["completed"] for item in results)
    print(json.dumps({
        "benchmark_end_utc": datetime.now(timezone.utc).isoformat(),
        "calls": len(results),
        "completed_calls": completed,
        "failed_calls": len(results) - completed,
        "final_ollama_ps": ollama_ps(),
        "final_gpu": gpu_snapshot(),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
