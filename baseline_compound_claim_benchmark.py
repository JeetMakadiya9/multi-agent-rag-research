"""Isolated compound-claim prompt benchmark; does not import production code."""

from __future__ import annotations

import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASE_URL = "http://127.0.0.1:11434"
MODEL = "qwen3:4b"
TEMPERATURE = 0.0
TIMEOUT_SECONDS = 120
REPETITIONS = 3
ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "baseline_compound_claim_results.jsonl"
SUMMARY_PATH = ROOT / "baseline_compound_claim_summary.json"

SUPPORTED = "SUPPORTED"
CONTRADICTED = "CONTRADICTED"
INSUFFICIENT = "INSUFFICIENT_EVIDENCE"
VALID_VERDICTS = {SUPPORTED, CONTRADICTED, INSUFFICIENT}

SYSTEM_PROMPT = "Verify the claim using only the supplied evidence. Return one JSON object."

# This is the strict prompt used in the preceding prompt-quality benchmark.
CURRENT_STRICT_PROMPT = '''\
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

STRUCTURED_DECISION_PROMPT = '''\
You are a strict claim verifier. Use ONLY the supplied evidence; do not use
outside knowledge or infer missing facts.

First decompose the claim into its material factual components. Decide the
verdict from those components BEFORE drafting the reason. Then output the
component analysis and the already-decided verdict in the JSON object.

COMPONENT RULES:
- SUPPORTED only if every material component is established by evidence.
- CONTRADICTED if any material component is directly contradicted by evidence.
- Otherwise use INSUFFICIENT_EVIDENCE.
- Never infer an unsupported component.
- For a compound claim, every component must be supported for SUPPORTED.
- The final verdict MUST be logically consistent with the component fields.
- If unsupported_components is non-empty and contradicted_components is
  empty, verdict MUST be INSUFFICIENT_EVIDENCE.
- If contradicted_components is non-empty, verdict MUST be CONTRADICTED.
- If every material component is supported and none is contradicted, verdict
  may be SUPPORTED.
- If evidence items conflict (one supports and another contradicts a material
  component) and the evidence does not resolve that conflict, classify that
  component as unsupported_components, leave contradicted_components empty,
  and use INSUFFICIENT_EVIDENCE for the unresolved overall verdict. Identify
  the conflict in reason and cite both evidence IDs. Reserve
  contradicted_components and CONTRADICTED for a component that the evidence
  establishes as false without an unresolved opposing evidence item.
- evidence_ids must cite only supplied items relevant to the component
  analysis. Do not invent IDs.

Return exactly one JSON object with these required fields. Keep the verdict
field first. Component fields must be arrays of short strings; use [] when
empty. Use the exact verdict labels shown:
{
  "verdict": "SUPPORTED | CONTRADICTED | INSUFFICIENT_EVIDENCE",
  "supported_components": ["component established by cited evidence"],
  "unsupported_components": ["material component not established"],
  "contradicted_components": ["material component directly contradicted"],
  "evidence_ids": ["E1"],
  "reason": "brief explanation consistent with verdict and component fields"
}
'''

CASES = [
    {
        "case_id": "CASE1_FULL_COMPOUND_SUPPORT",
        "category": "compound",
        "claim": "The system uses FAISS for dense retrieval and BM25 for lexical retrieval.",
        "evidence": [("E1", "The system uses FAISS for dense retrieval."), ("E2", "The system uses BM25 for lexical retrieval.")],
        "expected": SUPPORTED,
    },
    {
        "case_id": "CASE2_PARTIAL_COMPOUND",
        "category": "compound",
        "claim": "The system uses FAISS for dense retrieval and BM25 for lexical retrieval.",
        "evidence": [("E1", "The system uses FAISS for dense retrieval.")],
        "expected": INSUFFICIENT,
    },
    {
        "case_id": "CASE3_EXPLICIT_MISSING_COMPONENT",
        "category": "compound",
        "claim": "The system uses FAISS for dense retrieval and BM25 for lexical retrieval.",
        "evidence": [("E1", "The system uses FAISS for dense retrieval; this document gives no information about BM25.")],
        "expected": INSUFFICIENT,
    },
    {
        "case_id": "CASE4_CONTRADICTED_COMPONENT",
        "category": "compound",
        "claim": "The system uses FAISS for dense retrieval and BM25 for lexical retrieval.",
        "evidence": [("E1", "The system uses FAISS for dense retrieval."), ("E2", "The system does not use BM25 for lexical retrieval.")],
        "expected": CONTRADICTED,
    },
    {
        "case_id": "CASE5_THREE_COMPONENTS_PARTIAL",
        "category": "compound",
        "claim": "The system uses FAISS for dense retrieval, BM25 for lexical retrieval, and PostgreSQL to store documents.",
        "evidence": [("E1", "The system uses FAISS for dense retrieval."), ("E2", "The system uses BM25 for lexical retrieval.")],
        "expected": INSUFFICIENT,
    },
    {
        "case_id": "CASE6_RELATIONSHIP_PARTNER_ACQUISITION",
        "category": "compound",
        "claim": "Company A acquired Company B in 2024.",
        "evidence": [("E1", "Company A announced a partnership with Company B in 2024.")],
        "expected": INSUFFICIENT,
    },
    {
        "case_id": "CASE7_DATE_MISSING",
        "category": "compound",
        "claim": "The Orion data center opened in 2024.",
        "evidence": [("E1", "The Orion data center opened; the announcement gives no opening date.")],
        "expected": INSUFFICIENT,
    },
    {
        "case_id": "CASE8_QUANTITY_MISSING",
        "category": "compound",
        "claim": "The Orion data center's capacity increased by 30% in 2024.",
        "evidence": [("E1", "The Orion data center's capacity increased in 2024, but no percentage is reported.")],
        "expected": INSUFFICIENT,
    },
    {
        "case_id": "CASE9_CONFLICTING_EVIDENCE",
        "category": "compound",
        "claim": "The Orion data center opened in 2024.",
        "evidence": [("E1", "The Orion data center opened in 2024."), ("E2", "The Orion data center opened in 2025.")],
        "expected": INSUFFICIENT,
    },
    {
        "case_id": "CASE10_SINGLE_DIRECT_SUPPORT",
        "category": "single",
        "claim": "The archive contains 48 records.",
        "evidence": [("E1", "The archive contains 48 records.")],
        "expected": SUPPORTED,
    },
]


def api_json(path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        BASE_URL + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode("utf-8"))


def extract_json_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("response JSON is not an object")
    return value


def validator(result: dict[str, Any], variant: str) -> dict[str, Any]:
    required = ["verdict", "supported_components", "unsupported_components", "contradicted_components", "evidence_ids", "reason"]
    if variant == "existing_strict":
        required = ["verdict", "explanation"]
    missing = [field for field in required if field not in result]
    verdict = result.get("verdict")
    issues: list[str] = []
    if missing:
        issues.append("missing required fields: " + ", ".join(missing))
    if verdict not in VALID_VERDICTS:
        issues.append(f"invalid verdict: {verdict!r}")
    if variant == "structured_decision" and not missing:
        unsupported = result.get("unsupported_components")
        contradicted = result.get("contradicted_components")
        supported = result.get("supported_components")
        for name, value in (("supported_components", supported), ("unsupported_components", unsupported), ("contradicted_components", contradicted), ("evidence_ids", result.get("evidence_ids"))):
            if not isinstance(value, list):
                issues.append(f"{name} is not an array")
        if all(isinstance(v, list) for v in (unsupported, contradicted, supported)):
            if unsupported and not contradicted and verdict == SUPPORTED:
                issues.append("unsupported components present without contradiction, but verdict is SUPPORTED")
            if contradicted and verdict != CONTRADICTED:
                issues.append("contradicted components present, but verdict is not CONTRADICTED")
            if supported and not unsupported and not contradicted and verdict not in (SUPPORTED,):
                # Not a violation: a response may still conservatively abstain.
                pass
    override = None
    if variant == "structured_decision" and not missing:
        unsupported = result.get("unsupported_components")
        contradicted = result.get("contradicted_components")
        if isinstance(unsupported, list) and isinstance(contradicted, list):
            if contradicted and verdict != CONTRADICTED:
                override = CONTRADICTED
            elif unsupported and not contradicted and verdict == SUPPORTED:
                override = INSUFFICIENT
    return {
        "required_fields_present": not missing,
        "missing_fields": missing,
        "verdict_valid": verdict in VALID_VERDICTS,
        "component_consistency_valid": (not issues) if variant == "structured_decision" else None,
        "consistency_issues": issues,
        "validator_override_verdict": override,
        "validator_would_override": override is not None and override != verdict,
    }


def run_call(case: dict[str, Any], variant: str, repetition: int) -> dict[str, Any]:
    prompt = CURRENT_STRICT_PROMPT if variant == "existing_strict" else STRUCTURED_DECISION_PROMPT
    evidence = "\n".join(f"[{eid}] {text}" for eid, text in case["evidence"])
    user_prompt = f"CLAIM:\n{case['claim']}\n\nEVIDENCE:\n{evidence}"
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": True,
        "think": False,
        "format": "json",
        "keep_alive": "30m",
        "options": {"temperature": TEMPERATURE},
    }
    start = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    first_content = None
    chunks: list[str] = []
    metadata: dict[str, Any] = {}
    error = None
    completed = False
    try:
        request = urllib.request.Request(
            BASE_URL + "/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            for raw_line in response:
                if not raw_line.strip():
                    continue
                chunk = json.loads(raw_line.decode("utf-8"))
                content = chunk.get("message", {}).get("content", "")
                if content:
                    if first_content is None:
                        first_content = time.perf_counter() - t0
                    chunks.append(content)
                if chunk.get("done"):
                    metadata = {key: chunk.get(key) for key in (
                        "total_duration", "load_duration", "prompt_eval_count",
                        "prompt_eval_duration", "eval_count", "eval_duration",
                        "done_reason", "created_at",
                    )}
                    completed = True
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - t0
    raw_response = "".join(chunks)
    parse_error = None
    parsed: dict[str, Any] = {}
    try:
        parsed = extract_json_object(raw_response)
    except (json.JSONDecodeError, ValueError) as exc:
        parse_error = f"{type(exc).__name__}: {exc}"
    validation = validator(parsed, variant)
    verdict = parsed.get("verdict")
    return {
        "case_id": case["case_id"],
        "category": case["category"],
        "expected_verdict": case["expected"],
        "prompt_variant": variant,
        "repetition": repetition,
        "model": MODEL,
        "thinking_configuration": False,
        "format_configuration": "json",
        "temperature": TEMPERATURE,
        "start_utc": start.isoformat(),
        "end_utc": datetime.now(timezone.utc).isoformat(),
        "prompt_characters": len(prompt) + len(user_prompt),
        "first_content_latency_seconds": first_content,
        "total_latency_seconds": elapsed,
        "response_characters": len(raw_response),
        "completed": completed,
        "json_parse_success": parse_error is None,
        "parse_error": parse_error,
        "actual_verdict": verdict,
        "verdict_valid": validation["verdict_valid"],
        "correct": verdict == case["expected"],
        "parsed_response": parsed,
        "validation": validation,
        "ollama_metadata": metadata,
        "error": error,
        "claim": case["claim"],
        "evidence": case["evidence"],
        "raw_response": raw_response,
    }


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def summarize(records: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    rows = [r for r in records if r["prompt_variant"] == variant]
    parsed = [r for r in rows if r["json_parse_success"]]
    classes: dict[str, Any] = {}
    for label in sorted({r["expected_verdict"] for r in rows}):
        subset = [r for r in rows if r["expected_verdict"] == label]
        classes[label] = {"calls": len(subset), "correct": sum(r["correct"] for r in subset), "accuracy": sum(r["correct"] for r in subset) / len(subset) if subset else None}
    comp = [r for r in rows if r["category"] == "compound"]
    repeats = []
    for case in CASES:
        subset = [r for r in rows if r["case_id"] == case["case_id"]]
        verdicts = [r["actual_verdict"] for r in subset]
        repeats.append({"case_id": case["case_id"], "runs": len(subset), "verdicts": verdicts, "consistent": len(set(verdicts)) == 1 if verdicts else False})
    overrides = [r for r in rows if r["validation"].get("validator_would_override")]
    return {
        "calls": len(rows),
        "completed_calls": sum(r["completed"] for r in rows),
        "json_parse_success_count": len(parsed),
        "json_parse_success_rate": len(parsed) / len(rows) if rows else None,
        "required_fields_success_count": sum(r["validation"]["required_fields_present"] for r in rows),
        "valid_verdict_count": sum(r["verdict_valid"] for r in rows),
        "accuracy": sum(r["correct"] for r in rows) / len(rows) if rows else None,
        "per_class_accuracy": classes,
        "compound_claim_accuracy": sum(r["correct"] for r in comp) / len(comp) if comp else None,
        "compound_claim_correct": sum(r["correct"] for r in comp),
        "compound_claim_calls": len(comp),
        "component_verdict_consistency_applicable_calls": sum(r["validation"]["component_consistency_valid"] is not None for r in rows),
        "component_verdict_consistency_count": sum(r["validation"]["component_consistency_valid"] is True for r in rows) if variant == "structured_decision" else None,
        "component_verdict_consistency_rate": (sum(r["validation"]["component_consistency_valid"] is True for r in rows) / sum(r["validation"]["component_consistency_valid"] is not None for r in rows)) if variant == "structured_decision" and rows else None,
        "latency_seconds": quantiles([r["total_latency_seconds"] for r in rows]),
        "completed_response_latency_seconds": quantiles([r["total_latency_seconds"] for r in rows if r["completed"]]),
        "three_run_consistency": repeats,
        "validator_override_count": len(overrides),
        "validator_overrides": [{"case_id": r["case_id"], "repetition": r["repetition"], "llm_verdict": r["actual_verdict"], "override_verdict": r["validation"]["validator_override_verdict"]} for r in overrides],
    }


def main() -> None:
    for path in (RESULTS_PATH, SUMMARY_PATH):
        if path.exists():
            raise SystemExit(f"Refusing to overwrite existing benchmark artifact: {path}")
    version = api_json("/api/version")
    tags = api_json("/api/tags")
    models = [item.get("name") for item in tags.get("models", [])]
    if MODEL not in models:
        raise SystemExit(f"Required model {MODEL!r} unavailable. Found: {models}")
    show = api_json("/api/show", {"model": MODEL})
    print(json.dumps({
        "benchmark_start_utc": datetime.now(timezone.utc).isoformat(),
        "interpreter": sys.executable,
        "ollama_version": version.get("version"),
        "model": MODEL,
        "think": False,
        "format": "json",
        "temperature": TEMPERATURE,
        "repetitions": REPETITIONS,
        "model_capabilities": show.get("capabilities"),
        "initial_ollama_ps": api_json("/api/ps"),
        "results_path": str(RESULTS_PATH),
    }, ensure_ascii=False), flush=True)
    records: list[dict[str, Any]] = []
    for case in CASES:
        for variant in ("existing_strict", "structured_decision"):
            for repetition in range(1, REPETITIONS + 1):
                print(f"START {case['case_id']} prompt={variant} rep={repetition}/{REPETITIONS} {datetime.now(timezone.utc).isoformat()}", flush=True)
                result = run_call(case, variant, repetition)
                records.append(result)
                with RESULTS_PATH.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                print(json.dumps({"case_id": case["case_id"], "prompt": variant, "rep": repetition, "verdict": result["actual_verdict"], "expected": case["expected"], "latency_s": round(result["total_latency_seconds"], 3), "json_ok": result["json_parse_success"], "consistency": result["validation"]["component_consistency_valid"], "override": result["validation"]["validator_override_verdict"]}, ensure_ascii=False), flush=True)
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "ollama_version": version.get("version"),
        "query_case_count": len(CASES),
        "repetitions_per_case_prompt": REPETITIONS,
        "total_calls": len(records),
        "cases": [{"case_id": c["case_id"], "category": c["category"], "claim": c["claim"], "evidence": c["evidence"], "expected": c["expected"], "verdicts": {p: [r["actual_verdict"] for r in records if r["case_id"] == c["case_id"] and r["prompt_variant"] == p] for p in ("existing_strict", "structured_decision")}} for c in CASES],
        "configuration_summaries": {p: summarize(records, p) for p in ("existing_strict", "structured_decision")},
        "component_consistency_failures": [{"case_id": r["case_id"], "prompt": r["prompt_variant"], "repetition": r["repetition"], "verdict": r["actual_verdict"], "validation": r["validation"], "parsed_response": r["parsed_response"]} for r in records if not r["validation"]["component_consistency_valid"]],
        "validator_overrides": [{"case_id": r["case_id"], "prompt": r["prompt_variant"], "repetition": r["repetition"], "llm_verdict": r["actual_verdict"], "override_verdict": r["validation"]["validator_override_verdict"], "parsed_response": r["parsed_response"]} for r in records if r["validation"]["validator_would_override"]],
        "parse_or_completion_failures": [{"case_id": r["case_id"], "prompt": r["prompt_variant"], "repetition": r["repetition"], "completed": r["completed"], "parse_error": r["parse_error"], "error": r["error"]} for r in records if not r["completed"] or not r["json_parse_success"]],
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("SUMMARY=" + json.dumps(summary, ensure_ascii=False), flush=True)
    print(f"SUMMARY_PATH={SUMMARY_PATH}", flush=True)


if __name__ == "__main__":
    main()
