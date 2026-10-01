"""Isolated qwen3:4b atomic-claim decomposition benchmark.

This script calls Ollama directly and does not import production code. Ground
truth and matching rules are manually specified below; outputs are retained
verbatim for human review.
"""

from __future__ import annotations

import json
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASE_URL = "http://127.0.0.1:11434"
MODEL = "qwen3:4b"
TEMPERATURE = 0.0
THINK = False
FORMAT = "json"
TIMEOUT_SECONDS = 120
REPETITIONS = 3
ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "baseline_atomic_claim_decomposition_results.jsonl"
SUMMARY_PATH = ROOT / "baseline_atomic_claim_decomposition_summary.json"

SYSTEM_PROMPT = "You decompose claims only. Do not verify whether they are true. Return JSON only."
DECOMPOSITION_PROMPT = '''\
Decompose the original factual claim into the smallest independently
verifiable factual claims. Return ONLY valid JSON in exactly this shape:
{
  "claims": [
    {"claim_id": "C1", "text": "..."}
  ]
}

Rules:
1. Each claim must be independently verifiable.
2. Preserve the meaning of the original claim.
3. Do not add facts absent from the original claim.
4. Do not omit any material factual component.
5. Do not infer missing dates, quantities, relationships, causes, conditions,
   or entities.
6. Do not combine independently verifiable facts into one claim.
7. Do not split a single indivisible fact into meaningless fragments.
8. Preserve dates, quantities, conditions, comparisons, relationships,
   negations, causal language, and named entities.
9. Every material factual assertion must appear in at least one atomic claim.
10. The union of the claims must preserve the factual content of the original.
11. Do not create claims from grammatical fragments.
12. Do not perform truth verification. Only decompose.
Use sequential IDs C1, C2, ... and no additional JSON fields.
'''


def component(text: str, core: list[list[str]], qualifiers: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Each core group is required; alternatives within a group are synonyms."""
    return {"text": text, "core_anchor_groups": core, "qualifiers": qualifiers or []}


def q(name: str, *alternatives: str) -> dict[str, Any]:
    return {"name": name, "alternatives": list(alternatives)}


CASES: list[dict[str, Any]] = [
    {"case_id": "CASE01_FAISS_ATOMIC", "category": "simple_atomic", "original": "FAISS performs vector similarity search.", "expected": [component("FAISS performs vector similarity search.", [["faiss"], ["performs", "provides", "supports"], ["vector similarity search", "vector search", "similarity search"]])]},
    {"case_id": "CASE02_BM25_ATOMIC", "category": "simple_atomic", "original": "BM25 performs lexical retrieval.", "expected": [component("BM25 performs lexical retrieval.", [["bm25"], ["lexical retrieval", "lexical search"]])]},
    {"case_id": "CASE03_TWO_RETRIEVAL_FACTS", "category": "two_independent_facts", "original": "FAISS performs vector similarity search and BM25 performs lexical retrieval.", "expected": [component("FAISS performs vector similarity search.", [["faiss"], ["vector similarity search", "vector search", "similarity search"]]), component("BM25 performs lexical retrieval.", [["bm25"], ["lexical retrieval", "lexical search"]])]},
    {"case_id": "CASE04_LANGUAGE_TYPES", "category": "two_independent_facts", "original": "Python is dynamically typed and Java is statically typed.", "expected": [component("Python is dynamically typed.", [["python"], ["dynamically typed", "dynamic typing"]]), component("Java is statically typed.", [["java"], ["statically typed", "static typing"]])]},
    {"case_id": "CASE05_THREE_COMPONENTS", "category": "three_component", "original": "FAISS performs vector search, BM25 performs lexical retrieval, and PostgreSQL stores relational data.", "expected": [component("FAISS performs vector search.", [["faiss"], ["vector search", "vector retrieval"]]), component("BM25 performs lexical retrieval.", [["bm25"], ["lexical retrieval", "lexical search"]]), component("PostgreSQL stores relational data.", [["postgresql"], ["stores", "stores data", "stores information"], ["relational data", "relational information"]])]},
    {"case_id": "CASE06_ACQUISITION_RELATIONSHIP", "category": "relationship", "original": "Company A acquired Company B in 2024.", "expected": [component("Company A acquired Company B in 2024.", [["company a", "a"], ["acquired", "acquisition"], ["company b", "b"]], [q("year 2024", "2024")])]},
    {"case_id": "CASE07_TWO_RELATIONSHIPS", "category": "relationship_multiple_facts", "original": "Company A acquired Company B in 2024 and Company C partnered with Company D in 2025.", "expected": [component("Company A acquired Company B in 2024.", [["company a"], ["acquired", "acquisition"], ["company b"]], [q("year 2024", "2024")]), component("Company C partnered with Company D in 2025.", [["company c"], ["partnered", "partnership"], ["company d"]], [q("year 2025", "2025")])]},
    {"case_id": "CASE08_QUANTITIES", "category": "quantity", "original": "The model achieved 92% accuracy and reduced latency by 30%.", "expected": [component("The model achieved 92% accuracy.", [["model"], ["achieved", "reached"], ["92%", "92 percent"], ["accuracy"]]), component("The model reduced latency by 30%.", [["model"], ["reduced", "decreased"], ["latency"], ["30%", "30 percent"]])]},
    {"case_id": "CASE09_DATES", "category": "date", "original": "The system was developed in 2023 and deployed in January 2024.", "expected": [component("The system was developed in 2023.", [["system"], ["developed", "development"], ["2023"]]), component("The system was deployed in January 2024.", [["system"], ["deployed", "deployment"], ["january 2024", "2024-01", "january", "2024"]])]},
    {"case_id": "CASE10_CONDITION", "category": "condition", "original": "The model achieves 95% accuracy when the input images are normalized.", "expected": [component("The model achieves 95% accuracy when the input images are normalized.", [["model"], ["achieves", "achieved", "reaches"], ["95%", "95 percent"], ["accuracy"]], [q("normalized-image condition", "when", "if"), q("input images normalized", "input images are normalized", "input images normalized", "images are normalized", "images normalized")])]},
    {"case_id": "CASE11_NEGATION", "category": "negation", "original": "The system does not use PostgreSQL and does not require an external API.", "expected": [component("The system does not use PostgreSQL.", [["system"], ["postgresql"], ["does not use", "doesn't use", "never uses", "does not utilize"]]), component("The system does not require an external API.", [["system"], ["external api"], ["does not require", "doesn't require", "does not need", "doesn't need"]])]},
    {"case_id": "CASE12_CAUSAL_CLAIM", "category": "causal", "original": "Data augmentation improves model robustness because it exposes the model to more varied training examples.", "expected": [component("Data augmentation improves model robustness because it exposes the model to more varied training examples.", [["data augmentation", "augmentation"], ["improves", "increases", "enhances"], ["model robustness", "robustness"], ["because", "by", "through"], ["exposes", "exposure"], ["more varied training examples", "varied training examples", "diverse training examples"]])]},
    {"case_id": "CASE13_COMPARISONS", "category": "comparison", "original": "Model A is faster than Model B and uses less memory.", "expected": [component("Model A is faster than Model B.", [["model a"], ["faster than", "quicker than"], ["model b"]]), component("Model A uses less memory than Model B.", [["model a"], ["less memory", "lower memory", "uses less memory"], ["model b", "than model b"]])]},
    {"case_id": "CASE14_CONDITIONAL_MULTI_PART", "category": "conditional_multi_part", "original": "When GPU acceleration is enabled, the system processes images faster and consumes less CPU.", "expected": [component("When GPU acceleration is enabled, the system processes images faster.", [["system"], ["processes", "handles"], ["images"], ["faster", "more quickly"]], [q("GPU-enabled condition", "when", "if"), q("GPU acceleration enabled", "gpu acceleration is enabled", "gpu acceleration enabled", "gpu is enabled")]), component("When GPU acceleration is enabled, the system consumes less CPU.", [["system"], ["consumes", "uses"], ["less cpu", "lower cpu", "less cpu resources"]], [q("GPU-enabled condition", "when", "if"), q("GPU acceleration enabled", "gpu acceleration is enabled", "gpu acceleration enabled", "gpu is enabled")])]},
    {"case_id": "CASE15_THREE_RETRIEVAL_RELATIONS", "category": "multi_entity_relationship", "original": "FAISS retrieves vector-similar documents, BM25 retrieves lexically similar documents, and a cross-encoder reranks the retrieved candidates.", "expected": [component("FAISS retrieves vector-similar documents.", [["faiss"], ["retrieves", "retrieval"], ["vector-similar", "vector similar", "similar vectors"]]), component("BM25 retrieves lexically similar documents.", [["bm25"], ["retrieves", "retrieval"], ["lexically similar", "lexical similarity"]]), component("A cross-encoder reranks the retrieved candidates.", [["cross-encoder", "cross encoder"], ["reranks", "reranking", "re-ranks"], ["retrieved candidates", "candidates"]])]},
    {"case_id": "CASE16_FAISS_BM25_OMISSION_REGRESSION", "category": "two_independent_facts_regression", "original": "FAISS performs vector search and BM25 performs lexical retrieval.", "expected": [component("FAISS performs vector search.", [["faiss"], ["vector search", "vector retrieval"]]), component("BM25 performs lexical retrieval.", [["bm25"], ["lexical retrieval", "lexical search"]])]},
]

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "by", "for", "from", "in", "is", "it", "its", "of", "on", "or", "that", "the", "to", "was", "were", "when", "with", "this", "these", "those", "their", "they", "than", "then", "because", "more", "less", "only", "all", "each", "must", "should", "claim", "claims", "fact", "facts", "component", "components", "statement", "stated", "states", "state", "according", "based", "directly", "explicitly", "entity", "relationship", "date", "quantity", "condition", "information", "document", "system", "model"
}


def api_json(path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(BASE_URL + path, data=data, headers={"Content-Type": "application/json"}, method="GET" if payload is None else "POST")
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode("utf-8"))


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9%]+", " ", text.lower()).strip()


def contains_phrase(text: str, phrase: str) -> bool:
    source, needle = f" {norm(text)} ", f" {norm(phrase)} "
    return needle in source if needle.strip() else False


def core_match(text: str, expected: dict[str, Any]) -> bool:
    return all(any(contains_phrase(text, alternative) for alternative in group) for group in expected["core_anchor_groups"])


def missing_qualifiers(text: str, expected: dict[str, Any]) -> list[str]:
    return [item["name"] for item in expected["qualifiers"] if not any(contains_phrase(text, phrase) for phrase in item["alternatives"])]


def output_claims(parsed: Any) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    if not isinstance(parsed, dict):
        return [], ["top-level JSON value must be an object"]
    claims = parsed.get("claims")
    if not isinstance(claims, list):
        return [], ["required 'claims' field must be an array"]
    valid: list[dict[str, Any]] = []
    ids: set[str] = set()
    for index, item in enumerate(claims):
        if not isinstance(item, dict) or not isinstance(item.get("claim_id"), str) or not isinstance(item.get("text"), str):
            errors.append(f"claim entry {index} must contain string claim_id and text")
            continue
        if item["claim_id"] in ids:
            errors.append(f"duplicate claim_id: {item['claim_id']}")
        ids.add(item["claim_id"])
        if not item["text"].strip():
            errors.append(f"claim entry {index} has empty text")
        valid.append(item)
    return valid, errors


def evaluate(case: dict[str, Any], parsed: Any, parse_success: bool) -> dict[str, Any]:
    claims, schema_errors = output_claims(parsed)
    expected = case["expected"]
    mappings: list[dict[str, Any]] = []
    covered: set[int] = set()
    missing_qualifier_events: list[dict[str, Any]] = []
    merge_events: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []

    for output in claims:
        core_indices = [i for i, item in enumerate(expected) if core_match(output["text"], item)]
        fully_indices = [i for i in core_indices if not missing_qualifiers(output["text"], expected[i])]
        if len(core_indices) > 1:
            merge_events.append({"claim_id": output["claim_id"], "text": output["text"], "expected_component_indices": core_indices})
        if not core_indices:
            unmatched.append(output)
        for i in fully_indices:
            covered.add(i)
        for i in core_indices:
            absent = missing_qualifiers(output["text"], expected[i])
            if absent:
                missing_qualifier_events.append({"claim_id": output["claim_id"], "expected_component_index": i, "missing_qualifiers": absent, "text": output["text"]})
        mappings.append({"claim_id": output["claim_id"], "text": output["text"], "core_component_indices": core_indices, "fully_preserved_component_indices": fully_indices})

    omitted_indices = [i for i in range(len(expected)) if i not in covered]
    output_count = len(claims)
    expected_count = len(expected)
    over_split_excess = max(0, output_count - expected_count)
    # Unmapped factual sentences are treated as unsupported/invented assertions;
    # this is a deterministic flag for review, not a semantic oracle.
    invented = [{**item, "reason": "does not map to any manually specified expected component"} for item in unmatched]
    all_components_preserved = len(covered) == expected_count and not merge_events and not missing_qualifier_events
    case_accurate = bool(parse_success and not schema_errors and all_components_preserved and not invented and output_count == expected_count)
    return {
        "schema_valid": parse_success and not schema_errors,
        "schema_errors": schema_errors,
        "expected_component_count": expected_count,
        "output_claim_count": output_count,
        "covered_component_indices": sorted(covered),
        "omitted_component_indices": omitted_indices,
        "component_coverage": len(covered) / expected_count if expected_count else 1.0,
        "omission_count": len(omitted_indices),
        "qualifier_or_meaning_failures": missing_qualifier_events,
        "merge_errors": merge_events,
        "merge_error_count": len(merge_events),
        "unmatched_outputs": unmatched,
        "invented_or_unsupported_output_count": len(invented),
        "invented_or_unsupported_outputs": invented,
        "over_split_excess_claim_count": over_split_excess,
        "over_split_case": over_split_excess > 0,
        "meaning_preserved": all_components_preserved,
        "decomposition_accurate": case_accurate,
        "output_to_expected_mapping": mappings,
    }


def parse_response(text: str) -> tuple[Any, bool, str | None]:
    try:
        return json.loads(text), True, None
    except json.JSONDecodeError as exc:
        return None, False, f"{type(exc).__name__}: {exc}"


def run_call(case: dict[str, Any], repetition: int) -> dict[str, Any]:
    user_prompt = f"ORIGINAL CLAIM:\n{case['original']}"
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + DECOMPOSITION_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "stream": True,
        "think": THINK,
        "format": FORMAT,
        "keep_alive": "30m",
        "options": {"temperature": TEMPERATURE},
    }
    start_utc = datetime.now(timezone.utc)
    started = time.perf_counter()
    first_content_latency = None
    chunks: list[str] = []
    metadata: dict[str, Any] = {}
    completed = False
    error = None
    try:
        request = urllib.request.Request(BASE_URL + "/api/chat", data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            for line in response:
                if not line.strip():
                    continue
                part = json.loads(line.decode("utf-8"))
                text = part.get("message", {}).get("content", "")
                if text:
                    if first_content_latency is None:
                        first_content_latency = time.perf_counter() - started
                    chunks.append(text)
                if part.get("done"):
                    completed = True
                    metadata = {key: part.get(key) for key in ("total_duration", "load_duration", "prompt_eval_count", "prompt_eval_duration", "eval_count", "eval_duration", "done_reason", "created_at")}
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    total_latency = time.perf_counter() - started
    raw = "".join(chunks)
    parsed, parse_success, parse_error = parse_response(raw)
    evaluation = evaluate(case, parsed, parse_success)
    return {
        "case_id": case["case_id"],
        "category": case["category"],
        "original_claim": case["original"],
        "expected_atomic_claims": case["expected"],
        "repetition": repetition,
        "model": MODEL,
        "think": THINK,
        "temperature": TEMPERATURE,
        "format": FORMAT,
        "start_utc": start_utc.isoformat(),
        "end_utc": datetime.now(timezone.utc).isoformat(),
        "prompt_characters": len(SYSTEM_PROMPT) + len(DECOMPOSITION_PROMPT) + len(user_prompt),
        "first_content_latency_seconds": first_content_latency,
        "total_latency_seconds": total_latency,
        "response_characters": len(raw),
        "completed": completed,
        "json_parse_success": parse_success,
        "parse_error": parse_error,
        "error": error,
        "parsed_output": parsed,
        "raw_output": raw,
        "ollama_metadata": metadata,
        "evaluation": evaluation,
    }


def stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {"mean": statistics.mean(values), "median": statistics.median(values), "min": min(values), "max": max(values)}


def normalized_decomposition(record: dict[str, Any]) -> tuple[str, ...]:
    claims = record.get("parsed_output", {}).get("claims", []) if isinstance(record.get("parsed_output"), dict) else []
    if not isinstance(claims, list):
        return ("<invalid>",)
    return tuple(norm(item.get("text", "")) for item in claims if isinstance(item, dict))


def summarize_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    accuracies = [r["evaluation"]["decomposition_accurate"] for r in records]
    expected_total = sum(r["evaluation"]["expected_component_count"] for r in records)
    omitted_total = sum(r["evaluation"]["omission_count"] for r in records)
    output_total = sum(r["evaluation"]["output_claim_count"] for r in records)
    invented_total = sum(r["evaluation"]["invented_or_unsupported_output_count"] for r in records)
    merge_total = sum(r["evaluation"]["merge_error_count"] for r in records)
    over_split_cases = sum(r["evaluation"]["over_split_case"] for r in records)
    feature_cases = sum(bool(r["evaluation"]["qualifier_or_meaning_failures"]) for r in records)
    return {
        "calls": len(records),
        "accurate_calls": sum(accuracies),
        "overall_decomposition_accuracy": sum(accuracies) / len(records) if records else None,
        "expected_component_count": expected_total,
        "covered_component_count": expected_total - omitted_total,
        "component_coverage": (expected_total - omitted_total) / expected_total if expected_total else None,
        "omitted_component_count": omitted_total,
        "omission_rate": omitted_total / expected_total if expected_total else None,
        "invented_or_unsupported_output_count": invented_total,
        "invention_rate_per_output_claim": invented_total / output_total if output_total else None,
        "output_claim_count": output_total,
        "merge_error_count": merge_total,
        "merge_error_rate_per_call": merge_total / len(records) if records else None,
        "over_split_call_count": over_split_cases,
        "over_split_rate_per_call": over_split_cases / len(records) if records else None,
        "meaning_preservation_failure_calls": feature_cases + sum(not r["evaluation"]["meaning_preserved"] and not r["evaluation"]["qualifier_or_meaning_failures"] for r in records),
        "json_parse_success_count": sum(r["json_parse_success"] for r in records),
        "json_parse_success_rate": sum(r["json_parse_success"] for r in records) / len(records) if records else None,
        "mean_latency_seconds": stats([r["total_latency_seconds"] for r in records])["mean"],
        "median_latency_seconds": stats([r["total_latency_seconds"] for r in records])["median"],
        "min_latency_seconds": stats([r["total_latency_seconds"] for r in records])["min"],
        "max_latency_seconds": stats([r["total_latency_seconds"] for r in records])["max"],
    }


def main() -> None:
    if RESULTS_PATH.exists() or SUMMARY_PATH.exists():
        raise SystemExit(f"Refusing to overwrite prior benchmark artifact(s): {RESULTS_PATH} / {SUMMARY_PATH}")
    version = api_json("/api/version")
    tags = api_json("/api/tags")
    model_names = [entry.get("name") for entry in tags.get("models", [])]
    if MODEL not in model_names:
        raise SystemExit(f"Model {MODEL!r} is not available; Ollama reports {model_names}")
    print(json.dumps({
        "benchmark_start_utc": datetime.now(timezone.utc).isoformat(),
        "interpreter": sys.executable,
        "ollama_version": version.get("version"),
        "model": MODEL,
        "think": THINK,
        "temperature": TEMPERATURE,
        "format": FORMAT,
        "case_count": len(CASES),
        "repetitions": REPETITIONS,
        "total_calls_planned": len(CASES) * REPETITIONS,
        "initial_ollama_ps": api_json("/api/ps"),
        "results_path": str(RESULTS_PATH),
    }, ensure_ascii=False), flush=True)

    records: list[dict[str, Any]] = []
    for case in CASES:
        for repetition in range(1, REPETITIONS + 1):
            print(f"START {case['case_id']} run={repetition}/{REPETITIONS} {datetime.now(timezone.utc).isoformat()}", flush=True)
            result = run_call(case, repetition)
            records.append(result)
            with RESULTS_PATH.open("a", encoding="utf-8") as output:
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(json.dumps({"case_id": case["case_id"], "run": repetition, "parsed": result["json_parse_success"], "accurate": result["evaluation"]["decomposition_accurate"], "coverage": result["evaluation"]["component_coverage"], "omissions": result["evaluation"]["omission_count"], "merges": result["evaluation"]["merge_error_count"], "unmapped": result["evaluation"]["invented_or_unsupported_output_count"], "latency_s": round(result["total_latency_seconds"], 3), "error": result["error"]}, ensure_ascii=False), flush=True)

    cases_summary = []
    for case in CASES:
        subset = [r for r in records if r["case_id"] == case["case_id"]]
        signatures = {normalized_decomposition(r) for r in subset}
        cases_summary.append({
            "case_id": case["case_id"],
            "category": case["category"],
            "original_claim": case["original"],
            "expected_atomic_claims": [item["text"] for item in case["expected"]],
            "calls": len(subset),
            "accurate_calls": sum(r["evaluation"]["decomposition_accurate"] for r in subset),
            "mean_component_coverage": statistics.mean(r["evaluation"]["component_coverage"] for r in subset) if subset else None,
            "omission_count": sum(r["evaluation"]["omission_count"] for r in subset),
            "invention_or_unsupported_output_count": sum(r["evaluation"]["invented_or_unsupported_output_count"] for r in subset),
            "over_split_excess_claims": sum(r["evaluation"]["over_split_excess_claim_count"] for r in subset),
            "merge_error_count": sum(r["evaluation"]["merge_error_count"] for r in subset),
            "meaning_preservation_failures": [r["evaluation"]["qualifier_or_meaning_failures"] for r in subset if r["evaluation"]["qualifier_or_meaning_failures"]],
            "distinct_decompositions": len(signatures),
            "exact_output_consistent": len(signatures) == 1 if subset else False,
            "outputs": [{"repetition": r["repetition"], "claims": (r["parsed_output"].get("claims") if isinstance(r["parsed_output"], dict) else None), "evaluation": r["evaluation"]} for r in subset],
        })

    categories: dict[str, Any] = {}
    for category in sorted({case["category"] for case in CASES}):
        case_ids = {case["case_id"] for case in CASES if case["category"] == category}
        categories[category] = summarize_group([r for r in records if r["case_id"] in case_ids])

    failure_records = []
    for record in records:
        ev = record["evaluation"]
        if not ev["decomposition_accurate"]:
            case = next(item for item in CASES if item["case_id"] == record["case_id"])
            failure_types = []
            if ev["omission_count"]:
                failure_types.append("OMISSION")
            if ev["invented_or_unsupported_output_count"]:
                failure_types.append("INVENTION")
            if ev["merge_error_count"]:
                failure_types.append("MERGE")
            if ev["over_split_case"]:
                failure_types.append("OVER-SPLIT")
            if ev["qualifier_or_meaning_failures"]:
                failure_types.append("MEANING-CHANGE")
            if not record["json_parse_success"]:
                failure_types.append("OTHER: JSON PARSE")
            failure_records.append({
                "case_id": record["case_id"],
                "repetition": record["repetition"],
                "original_claim": case["original"],
                "expected_atomic_claims": [item["text"] for item in case["expected"]],
                "model_output": record["parsed_output"] if record["json_parse_success"] else record["raw_output"],
                "failure_types": failure_types or ["OTHER: unmatched decomposition"],
                "evaluation": ev,
            })

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "ollama_version": version.get("version"),
        "think": THINK,
        "temperature": TEMPERATURE,
        "format": FORMAT,
        "total_calls": len(records),
        "total_cases": len(CASES),
        "repetitions_per_case": REPETITIONS,
        "ground_truth_method": "Manually defined expected atomic claims and deterministic phrase/qualifier anchors; no LLM evaluator.",
        "evaluation_limitations": "Lexical anchor matching is a deterministic aid, not a semantic oracle. Unmatched outputs are flagged as unsupported/invented for manual review; paraphrase edge cases may be false positives/negatives.",
        "overall": summarize_group(records),
        "per_category": categories,
        "per_case": cases_summary,
        "hardest_and_failing_calls": failure_records,
        "three_run_consistency": [{"case_id": case["case_id"], "distinct_decompositions": next(row["distinct_decompositions"] for row in cases_summary if row["case_id"] == case["case_id"]), "exact_output_consistent": next(row["exact_output_consistent"] for row in cases_summary if row["case_id"] == case["case_id"]), "repetition_signatures": [list(normalized_decomposition(r)) for r in records if r["case_id"] == case["case_id"]]} for case in CASES],
        "records_path": str(RESULTS_PATH),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("SUMMARY=" + json.dumps({"overall": summary["overall"], "per_category": categories, "summary_path": str(SUMMARY_PATH)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
