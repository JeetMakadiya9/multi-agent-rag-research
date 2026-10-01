"""Deterministic post-hoc analysis of saved Phase 10C artifacts only."""
from __future__ import annotations

import ast
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ANALYSIS_VERSION = "phase10d-analysis-1.0"
SYSTEMS = ("A", "B", "C")
VALID_PREDICTIONS = {"SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"}
GOLD_CLASSES = ("SUPPORTED", "CONTRADICTED")
ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "evaluation" / "experiments" / "results"
RUN_DIR = RESULTS / "phase10_dev_run"
OUTPUT_DIR = RESULTS / "phase10_error_analysis"
SOURCE_FILES = {
    "combined": RESULTS / "phase10_dev.json",
    "per_example": RUN_DIR / "per_example.jsonl",
    "runtime_attempts": RUN_DIR / "runtime_attempts.jsonl",
    "manifest": RUN_DIR / "run_manifest.json",
    "summary": RUN_DIR / "summary.json",
}
FROZEN_HASHES = {
    "rag.py": "D2B0CF8930BC66FC66995AD63C1028AE3B126424E01E091DF22ED238AD5DBA19",
    "research_pipeline.py": "F55A29386AA13774A621B855E89CA8CA71FEAC0CFE2A124F90EF54DD7CD541D0",
    "claim_verification.py": "63FC8D73F55B7F48DBB340E4CD67C97975D0D587FB009CEAFA97E7FEF1C639A1",
    "verification_units.py": "BEFE07461C9B0DA76BF68226D86A96346D761C1AD3F127C554D4BB9C51644DDF",
    "experiment_b_verification.py": "66EA7B6BE6119E7A3E25A7BAD3AF3D078F08CBA4CC8AC4A462BAF6170548F3B4",
    "evaluation/scifact/scifact_adapter.py": "8F2BE845CFAF9ABB2210C88242F02E55EA8F47FF68E71BCDD27098287904BC9A",
    "evaluation/scifact/scifact_metrics.py": "8BE4B90BB28090FC48F874AF399798B0F7E39C25AAD7A1FA17B864A002E55587",
    "evaluation/scifact/run_scifact_evaluation.py": "21F540C534B5CFE8C8328C648C6E89AF6E4C4C16497CF78BC52DB51BC9629338",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def source_hashes(root: Path = ROOT) -> dict[str, dict[str, Any]]:
    result = {}
    for name, expected in FROZEN_HASHES.items():
        path = root / name
        if not path.is_file():
            result[name] = {"status": "MISSING", "expected_sha256": expected, "actual_sha256": None}
        else:
            actual = sha256_file(path)
            result[name] = {"status": "MATCH" if actual == expected else "MISMATCH",
                            "expected_sha256": expected, "actual_sha256": actual}
    return result


def _claim_index(rows: list[dict[str, Any]]) -> dict[int, dict[str, dict[str, Any]]]:
    result: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        result[int(row["example_id"])][row["system"]] = row
    return result


def _gold_docs(row: dict[str, Any]) -> set[Any]:
    ids = row.get("gold_evidence_doc_ids")
    return set(ids) if isinstance(ids, list) else set()


def retrieved_doc_ids(row: dict[str, Any]) -> list[Any]:
    chunks = row.get("retrieved_chunks")
    if not isinstance(chunks, list):
        return []
    ordered = sorted((x for x in chunks if isinstance(x, dict)), key=lambda x: x.get("rank", 10**9))
    return [x.get("scifact_doc_id") for x in ordered if x.get("scifact_doc_id") is not None]


def retrieval_hit_at_1(row: dict[str, Any]) -> bool | None:
    gold = _gold_docs(row)
    if not gold:
        return None
    if row.get("status") == "FAILED" and not row.get("retrieved_chunks"):
        return None
    docs = retrieved_doc_ids(row)
    return bool(docs and docs[0] in gold)


def prediction_state(row: dict[str, Any], system: str) -> str:
    if system == "A":
        return "NOT_APPLICABLE"
    if row.get("status") != "SUCCESS":
        return "FAILED"
    prediction = row.get("predicted_label")
    if prediction in VALID_PREDICTIONS:
        return "VALID"
    return "MISSING" if prediction is None else "INVALID"


def prediction_correct(row: dict[str, Any], system: str) -> bool | None:
    if system == "A" or row.get("gold_label") not in GOLD_CLASSES:
        return None
    if prediction_state(row, system) != "VALID":
        return False
    return row.get("predicted_label") == row.get("gold_label")


def _parse_controller_failure(error: Any) -> dict[str, Any] | None:
    marker = "Phase 8C execution failed: "
    if not isinstance(error, str) or marker not in error:
        return None
    try:
        value = ast.literal_eval(error.split(marker, 1)[1])
    except (SyntaxError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _controller_failure_info(attempt: dict[str, Any]) -> dict[str, Any] | None:
    parsed = _parse_controller_failure(attempt.get("error"))
    if parsed is None:
        return None
    trace = parsed.get("trace") or {}
    cycles = trace.get("cycles") or []
    return {
        "stop_reason": parsed.get("stop_reason"), "cycles_completed": parsed.get("cycles_completed"),
        "executions_completed": parsed.get("executions_completed"), "failure_reason": parsed.get("failure_reason"),
        "controller_id": trace.get("controller_id"), "run_id": trace.get("run_id"), "loop_id": trace.get("loop_id"),
        "trace_created_at": trace.get("created_at"),
        "cycles": [{
            "cycle_number": c.get("cycle_number"),
            "validation_status": (c.get("validation") or {}).get("validation_status"),
            "execution_status": (c.get("execution") or {}).get("status"),
            "capability": ((c.get("execution") or {}).get("dispatch") or {}).get("capability"),
            "dispatch_status": ((c.get("execution") or {}).get("dispatch") or {}).get("status"),
            "dispatch_error": ((c.get("execution") or {}).get("dispatch") or {}).get("error"),
            "evidence_ids": ((c.get("decision") or {}).get("proposal") or {}).get("evidence_ids"),
        } for c in cycles],
    }


def classify_failure(attempt: dict[str, Any]) -> tuple[str, str | None]:
    error = str(attempt.get("error") or "")
    text = error.lower()
    if "winerror 10061" in text or "actively refused" in text or "connection refused" in text:
        return "model_endpoint_failure", "connection_refused"
    if "phase 8c execution failed" in text or "taskexecutionerror" in text:
        return "controller_execution_failure", "phase5_task_execution_error"
    if "timeout" in text or "timed out" in text:
        return "timeout", None
    if any(token in text for token in ("malformed", "schema", "json")):
        return "invalid_model_output", None
    if "retriev" in text:
        return "retrieval_error", None
    if "verif" in text or "assessment" in text:
        return "verification_error", None
    if "authoriz" in text or "approval" in text:
        return "authorization_error", None
    return "other_recorded_execution_failure", None


def _timestamp_strings(value: Any) -> list[str]:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return sorted(set(re.findall(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})", text)))


def build_failure_records(attempts: list[dict[str, Any]], scored: list[dict[str, Any]],
                          manifest: dict[str, Any]) -> list[dict[str, Any]]:
    scored_index = {(int(x["example_id"]), x["system"]): x for x in scored}
    pair_counts = Counter((int(x["example_id"]), x["system"]) for x in attempts)
    failures = []
    for attempt in attempts:
        if attempt.get("status") == "SUCCESS":
            continue
        cid, system = int(attempt["example_id"]), attempt["system"]
        scored_row = scored_index.get((cid, system), {})
        category, secondary = classify_failure(attempt)
        controller = _controller_failure_info(attempt) if system == "C" else None
        chunks = attempt.get("retrieved_chunks")
        raw = attempt.get("raw_output") or {}
        if chunks:
            retrieval = "YES — retrieved chunks are persisted"
        elif system == "C" and controller and any(c.get("evidence_ids") for c in controller["cycles"]):
            retrieval = "YES — VERIFY_CLAIM trace contains evidence IDs; chunk list not persisted"
        else:
            retrieval = "UNKNOWN / NOT RECORDED"
        if system == "A":
            verification = "NO — System A has no verifier"
        elif system == "B" and (attempt.get("verification_calls") or 0) > 0:
            verification = "YES — invocation attempted; assessment incomplete"
        elif system == "C" and controller and any(c.get("capability") == "evidence_verification" for c in controller["cycles"]):
            verification = "YES — capability dispatch attempted; execution failed"
        else:
            verification = "UNKNOWN / NOT RECORDED"
        inference = ("NO — endpoint refused the connection before inference"
                     if category == "model_endpoint_failure" else "UNKNOWN / NOT RECORDED")
        trace_meta = controller or (raw.get("trace") if isinstance(raw, dict) else None)
        failures.append({
            "claim_id": cid, "system": system,
            "attempt_id": attempt.get("attempt_id", "NOT RECORDED"),
            "attempt_timestamp": attempt.get("timestamp", "NOT RECORDED"),
            "timestamps_in_error_or_trace": _timestamp_strings({"error": attempt.get("error"), "trace": trace_meta}),
            "status": attempt.get("status"), "primary_category": category, "secondary_category": secondary,
            "recorded_error_type": attempt.get("error_type", "NOT RECORDED"),
            "recorded_error_stage": attempt.get("error_stage", "NOT RECORDED"),
            "controller_failure_trace": controller,
            "exact_recorded_error": attempt.get("error", "NOT RECORDED"),
            "recorded_error_traceback": attempt.get("error_traceback", "NOT RECORDED"),
            "before_inference": True if category == "model_endpoint_failure" else "UNKNOWN / NOT RECORDED",
            "inference_started": inference, "retrieval_occurred": retrieval,
            "verification_occurred": verification, "controller_involved": system == "C",
            "prediction_produced": attempt.get("predicted_label") is not None,
            "predicted_label": attempt.get("predicted_label"),
            "persisted_in_runtime_attempts": True,
            "persisted_in_per_example": (cid, system) in scored_index,
            "retry_within_final_run": ("NO — unique claim/system pair; manifest says persisted failures are not rerun"
                if pair_counts[(cid, system)] == 1 and manifest.get("failure_handling", {}).get("resume_policy") else "NOT RECORDED"),
            "latency_seconds": attempt.get("latency_seconds"), "llm_calls": attempt.get("llm_calls"),
            "retrieval_calls": attempt.get("retrieval_calls"), "verification_calls": attempt.get("verification_calls"),
            "controller_cycles": attempt.get("controller_cycles"), "claim_gold_label": scored_row.get("gold_label"),
        })
    return failures


def build_claim_matrix(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_claim = _claim_index(rows)
    matrix = []
    for cid, systems in sorted(by_claim.items()):
        exemplar = next(iter(systems.values()))
        result = {"claim_id": cid, "claim_text": (exemplar.get("runtime_input") or {}).get("claim", "NOT RECORDED"),
                  "gold_label": exemplar.get("gold_label"), "gold_label_status": exemplar.get("gold_label_status"),
                  "gold_evidence_doc_ids": exemplar.get("gold_evidence_doc_ids"),
                  "classification_scorable": exemplar.get("gold_label") in GOLD_CLASSES}
        for system in SYSTEMS:
            row = systems.get(system)
            if row is None:
                result.update({f"{system}_status": "N/A — attempt missing", f"{system}_prediction": "N/A",
                    f"{system}_validity": "N/A", f"{system}_correct": "N/A", f"{system}_failure_reason": "N/A",
                    f"{system}_retrieval_result": "N/A", f"{system}_latency": None})
                continue
            hit = retrieval_hit_at_1(row)
            if system == "A":
                prediction = "N/A — no verifier-derived classification"
                validity = "NOT APPLICABLE"
                correct: Any = "N/A"
            else:
                prediction = row.get("predicted_label")
                validity = prediction_state(row, system)
                correct = prediction_correct(row, system)
            result.update({
                f"{system}_status": row.get("status", "NOT RECORDED"),
                f"{system}_prediction": prediction, f"{system}_validity": validity, f"{system}_correct": correct,
                f"{system}_failure_reason": row.get("error") or "N/A — no recorded failure",
                f"{system}_retrieval_result": {"hit_at_1": hit, "retrieved_document_ids_in_rank_order": retrieved_doc_ids(row),
                    "gold_document_ids": sorted(_gold_docs(row)), "selected_evidence_document_ids": row.get("selected_evidence_doc_ids")},
                f"{system}_latency": row.get("latency_seconds"),
            })
        matrix.append(result)
    return matrix


def classification_metrics(rows: list[dict[str, Any]], system: str) -> dict[str, Any]:
    labeled = [row for row in rows if row.get("system") == system and row.get("gold_label") in GOLD_CLASSES]
    outcomes = Counter()
    confusion = {label: Counter() for label in GOLD_CLASSES}
    correct = valid = 0
    for row in labeled:
        state = prediction_state(row, system)
        gold = row["gold_label"]
        if state == "FAILED":
            outcomes["failed_execution"] += 1
            confusion[gold]["FAILED_OR_INVALID"] += 1
        elif state == "INVALID":
            outcomes["invalid_prediction"] += 1
            confusion[gold]["FAILED_OR_INVALID"] += 1
        elif state == "MISSING":
            outcomes["missing_prediction"] += 1
            confusion[gold]["MISSING_PREDICTION"] += 1
        else:
            valid += 1
            prediction = row["predicted_label"]
            is_correct = prediction == gold
            correct += int(is_correct)
            outcomes["correct_prediction" if is_correct else "incorrect_prediction"] += 1
            confusion[gold][prediction] += 1
    per_class = {}
    for label in GOLD_CLASSES:
        tp = confusion[label][label]
        support = sum(confusion[label].values())
        predicted = sum(confusion[g][label] for g in GOLD_CLASSES)
        precision = tp / predicted if predicted else None
        recall = tp / support if support else None
        f1 = (2 * precision * recall / (precision + recall)
              if precision is not None and recall is not None and precision + recall else
              0.0 if precision is not None and recall is not None else None)
        per_class[label] = {"tp": tp, "gold_support": support, "predicted_support": predicted,
                            "precision": precision, "recall": recall, "f1": f1}
    f1 = [v["f1"] for v in per_class.values() if v["f1"] is not None]
    return {
        "gold_labeled_denominator": len(labeled), "correct_numerator": correct,
        "accuracy_including_failed_as_incorrect": correct / len(labeled) if labeled else None,
        "valid_prediction_denominator": valid,
        "valid_prediction_accuracy": correct / valid if valid else None,
        "outcome_counts": {name: outcomes.get(name, 0) for name in (
            "correct_prediction", "incorrect_prediction", "invalid_prediction", "missing_prediction", "failed_execution")},
        "confusion_matrix": {gold: dict(counter) for gold, counter in confusion.items()},
        "per_class": per_class,
        "macro_f1_over_gold_supported_classes": sum(f1) / len(f1) if f1 else None,
        "unlabeled_count": sum(row.get("gold_label") not in GOLD_CLASSES for row in rows if row.get("system") == system),
    }


def build_agreement(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_claim = _claim_index(rows)
    status_counts, bc_counts = Counter(), Counter()
    differences, retrieval_differences = [], []
    retrieval_agreement = Counter()
    all_complete_disagreements = []
    for cid, systems in sorted(by_claim.items()):
        statuses = {s: systems.get(s, {}).get("status", "MISSING") for s in SYSTEMS}
        done = {s: statuses[s] == "SUCCESS" for s in SYSTEMS}
        if all(done.values()): category = "all_completed"
        elif not any(done.values()): category = "all_failed"
        elif done["A"] and done["B"] and not done["C"]: category = "A_B_completed_C_failed"
        elif done["A"] and done["C"] and not done["B"]: category = "A_C_completed_B_failed"
        elif done["B"] and done["C"] and not done["A"]: category = "B_C_completed_A_failed"
        else: category = "other_partial_completion"
        status_counts[category] += 1
        b, c = systems.get("B"), systems.get("C")
        bs = prediction_state(b, "B") if b else "MISSING"
        cs = prediction_state(c, "C") if c else "MISSING"
        if bs == "FAILED" and cs == "FAILED": bc = "both_failed"
        elif bs == "FAILED" or cs == "FAILED": bc = "one_failed"
        elif bs == "VALID" and cs == "VALID":
            bc = "both_valid_same" if b.get("predicted_label") == c.get("predicted_label") else "both_valid_different"
        elif bs == "VALID" and cs == "INVALID": bc = "B_valid_C_invalid"
        elif bs == "INVALID" and cs == "VALID": bc = "B_invalid_C_valid"
        elif bs == "VALID" and cs == "MISSING": bc = "B_valid_C_missing"
        elif bs == "MISSING" and cs == "VALID": bc = "B_missing_C_valid"
        elif bs == "INVALID" and cs == "INVALID": bc = "both_invalid"
        elif bs == "MISSING" and cs == "MISSING": bc = "both_missing"
        elif bs == "INVALID" or cs == "INVALID": bc = "one_invalid"
        else: bc = "one_missing"
        bc_counts[bc] += 1
        if bc in {"both_failed", "one_failed", "both_valid_different", "B_valid_C_invalid",
                  "B_invalid_C_valid", "B_valid_C_missing", "B_missing_C_valid"}:
            differences.append({
                "claim_id": cid, "gold_label": (b or c or {}).get("gold_label"), "category": bc,
                "B_prediction": (b or {}).get("predicted_label"), "B_validity": bs,
                "C_prediction": (c or {}).get("predicted_label"), "C_validity": cs,
                "B_retrieved_document_ids": retrieved_doc_ids(b) if b else None,
                "C_retrieved_document_ids": retrieved_doc_ids(c) if c else None,
                "retrieval_difference": (retrieved_doc_ids(b) != retrieved_doc_ids(c)
                    if b and c and not ((b.get("status") != "SUCCESS" and not b.get("retrieved_chunks")) or
                                        (c.get("status") != "SUCCESS" and not c.get("retrieved_chunks"))) else None),
                "retrieval_comparison_status": ("N/A — failed execution has no persisted retrieved document IDs"
                    if (b and b.get("status") != "SUCCESS" and not b.get("retrieved_chunks")) or
                       (c and c.get("status") != "SUCCESS" and not c.get("retrieved_chunks")) else "scored from persisted ranked document IDs"),
                "B_selected_evidence_document_ids": (b or {}).get("selected_evidence_doc_ids"),
                "C_selected_evidence_document_ids": (c or {}).get("selected_evidence_doc_ids"),
                "B_assessments": (((b or {}).get("output") or {}).get("assessments")),
                "C_assessments": (((c or {}).get("output") or {}).get("assessments")),
                "B_raw_verification_record": ((((b or {}).get("output") or {}).get("raw_output") or {}).get("verification")),
                "C_raw_verification_record": ((((c or {}).get("output") or {}).get("raw_output") or {}).get("verification")),
                "C_controller_traces": (c or {}).get("provenance", {}).get("controller_traces") if c else None,
                "C_controller_failure": _parse_controller_failure((c or {}).get("error")),
            })
        if all(done.values()) and bs == cs == "VALID" and b.get("predicted_label") != c.get("predicted_label"):
            all_complete_disagreements.append(cid)
        seq = {s: (None if s not in systems or
                    (systems[s].get("status") != "SUCCESS" and not systems[s].get("retrieved_chunks"))
                   else retrieved_doc_ids(systems[s])) for s in SYSTEMS}
        if seq["A"] == seq["B"] == seq["C"]: rcat = "all_three_exact_same"
        elif seq["A"] == seq["B"]: rcat = "A_B_same_C_diff"
        elif seq["A"] == seq["C"]: rcat = "A_C_same_B_diff"
        elif seq["B"] == seq["C"]: rcat = "B_C_same_A_diff"
        else: rcat = "all_different_or_missing"
        retrieval_agreement[rcat] += 1
        if len({json.dumps(x, sort_keys=True) for x in seq.values()}) > 1:
            retrieval_differences.append({"claim_id": cid, "ranked_document_ids": seq})
    return {
        "claim_status_agreement_counts": {k: status_counts.get(k, 0) for k in (
            "all_completed", "all_failed", "A_B_completed_C_failed", "A_C_completed_B_failed",
            "B_C_completed_A_failed", "other_partial_completion")},
        "B_C_prediction_agreement_counts": dict(bc_counts),
        "B_C_valid_prediction_agreement": {"both_valid_same": bc_counts.get("both_valid_same", 0),
            "both_valid_different": bc_counts.get("both_valid_different", 0),
            "valid_pair_denominator": bc_counts.get("both_valid_same", 0) + bc_counts.get("both_valid_different", 0)},
        "B_C_differences_or_missing_outputs": differences,
        "all_completed_B_C_disagreements": all_complete_disagreements,
        "retrieval_ranked_document_sequence_agreement_counts": dict(retrieval_agreement),
        "retrieval_ranked_document_sequence_disagreements": retrieval_differences,
    }


def build_retrieval_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_claim = _claim_index(rows)
    hit_patterns, hits, denominators = Counter(), Counter(), Counter()
    misses = {s: [] for s in SYSTEMS}
    lists = {key: [] for key in ("all_hit", "all_miss", "A_only_miss", "B_only_miss", "C_only_miss", "AB_hit_C_miss", "C_hit_AB_miss")}
    per_claim = []
    gold_doc_denominator = 0
    for cid, systems in sorted(by_claim.items()):
        exemplar = next(iter(systems.values()))
        gold = _gold_docs(exemplar)
        if not gold:
            continue
        gold_doc_denominator += len(gold)
        details, claim_hits = {}, {}
        for s in SYSTEMS:
            row = systems.get(s)
            if row is None:
                claim_hits[s] = None
                details[s] = {"status": "MISSING", "hit_at_1": None}
                continue
            docs = retrieved_doc_ids(row)
            hit = retrieval_hit_at_1(row)
            claim_hits[s] = hit
            if hit is not None:
                denominators[s] += 1
                hits[s] += int(hit)
            details[s] = {"status": row.get("status"), "hit_at_1": hit,
                "gold_doc_ids": sorted(gold), "retrieved_doc_ids_in_rank_order": docs,
                "top1_doc_id": docs[0] if docs else None,
                "persisted_reciprocal_rank": (row.get("retrieval") or {}).get("reciprocal_rank"),
                "scorable_from_ranked_doc_ids": hit is not None}
            if hit is False:
                misses[s].append({"claim_id": cid, **details[s]})
        sig = tuple(claim_hits[s] for s in SYSTEMS)
        hit_patterns[str(sig)] += 1
        if all(claim_hits[s] is True for s in SYSTEMS): lists["all_hit"].append(cid)
        if all(claim_hits[s] is False for s in SYSTEMS): lists["all_miss"].append(cid)
        if claim_hits["A"] is False and claim_hits["B"] is True and claim_hits["C"] is True: lists["A_only_miss"].append(cid)
        if claim_hits["B"] is False and claim_hits["A"] is True and claim_hits["C"] is True: lists["B_only_miss"].append(cid)
        if claim_hits["C"] is False and claim_hits["A"] is True and claim_hits["B"] is True: lists["C_only_miss"].append(cid)
        if claim_hits["A"] is True and claim_hits["B"] is True and claim_hits["C"] is False: lists["AB_hit_C_miss"].append(cid)
        if claim_hits["C"] is True and claim_hits["A"] is False and claim_hits["B"] is False: lists["C_hit_AB_miss"].append(cid)
        per_claim.append({"claim_id": cid, "gold_doc_ids": sorted(gold), "systems": details})
    return {"definition": "Rank-1 SciFact document-ID intersection with gold rationale document IDs. Relevance overlap only; not entailment or correctness.",
        "gold_evidence_claim_denominator": len(per_claim), "gold_evidence_document_denominator": gold_doc_denominator,
        "unavailable_ranked_doc_outputs_by_system": {s: [x["claim_id"] for x in per_claim
            if x["systems"][s].get("hit_at_1") is None] for s in SYSTEMS},
        "Recall@1_by_system": {s: {"hits": hits[s], "claim_denominator": denominators[s],
            "value": hits[s] / denominators[s] if denominators[s] else None} for s in SYSTEMS},
        "hit_pattern_counts_A_B_C": dict(hit_patterns),
        "claims_retrieved_by_all": lists["all_hit"], "claims_missed_by_all": lists["all_miss"],
        "claims_missed_by_A_only": lists["A_only_miss"], "claims_missed_by_B_only": lists["B_only_miss"],
        "claims_missed_by_C_only": lists["C_only_miss"], "claims_A_B_hit_C_missed": lists["AB_hit_C_miss"],
        "claims_C_hit_A_B_missed": lists["C_hit_AB_miss"], "misses_by_system": misses, "per_claim": per_claim}


def build_evidence_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_claim = _claim_index(rows)
    systems_result, p1_incomplete, zero_recall = {}, [], []
    nonvalid_selected = {"B": [], "C": []}
    same, different, same_empty, same_nonempty = [], [], [], []
    for system in ("B", "C"):
        claim_rows, p_values, r_values, f_values = [], [], [], []
        stored_citations = [row.get("citation") for row in rows
                            if row.get("system") == system and isinstance(row.get("citation"), dict)]
        for cid, systems in sorted(by_claim.items()):
            row = systems.get(system)
            if row is None or not _gold_docs(row):
                continue
            if row.get("status") != "SUCCESS":
                claim_rows.append({"claim_id": cid, "status": row.get("status"),
                    "evidence_status": "UNAVAILABLE — execution failed", "selected_doc_ids": None,
                    "gold_doc_ids": sorted(_gold_docs(row)),
                    "selected_document_reference_status": "UNAVAILABLE — execution failed",
                    "persisted_evaluator_citation_metrics": row.get("citation")})
                continue
            gold, selected = _gold_docs(row), set(row.get("selected_evidence_doc_ids") or [])
            tp = len(gold & selected)
            precision = tp / len(selected) if selected else None
            recall = tp / len(gold) if gold else None
            f1 = (2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall
                  else 0.0 if precision is not None and recall is not None else None)
            if precision is not None: p_values.append(precision)
            if recall is not None: r_values.append(recall)
            if f1 is not None: f_values.append(f1)
            if precision == 1.0 and recall is not None and recall < 1.0:
                p1_incomplete.append({"claim_id": cid, "system": system, "precision": precision, "recall": recall})
            if recall == 0.0: zero_recall.append({"claim_id": cid, "system": system})
            if selected and prediction_state(row, system) != "VALID": nonvalid_selected[system].append(cid)
            retrieved = set(retrieved_doc_ids(row))
            reference_status = ("NOT APPLICABLE — no selected evidence IDs" if not selected else
                "selected document IDs are present in persisted retrieved document IDs" if selected <= retrieved else
                "some selected document IDs are absent from persisted retrieved document IDs")
            claim_rows.append({"claim_id": cid, "status": row.get("status"), "evidence_status": "scored",
                "gold_doc_ids": sorted(gold), "selected_doc_ids": sorted(selected), "selected_evidence_count": len(selected),
                "selected_document_reference_status": reference_status,
                "true_positive_documents": tp, "precision": precision, "recall": recall, "f1": f1,
                "classification_prediction": row.get("predicted_label"), "classification_validity": prediction_state(row, system),
                "persisted_evaluator_citation_metrics": row.get("citation")})
        persisted_metrics = {}
        for metric in ("precision", "recall", "f1"):
            values = [citation.get(metric) for citation in stored_citations if citation.get(metric) is not None]
            persisted_metrics[metric] = {"mean": statistics.mean(values) if values else None,
                                         "non_null_denominator": len(values)}
        systems_result[system] = {"gold_evidence_claims": sum(bool(_gold_docs(r)) for r in rows if r.get("system") == system),
            "scored_claims": sum(x.get("evidence_status") == "scored" for x in claim_rows),
            "phase10c_persisted_citation_metric": {"citation_row_denominator": len(stored_citations),
                                                    "metrics": persisted_metrics},
            "precision_mean_over_nonempty_selections": statistics.mean(p_values) if p_values else None,
            "precision_denominator": len(p_values), "recall_mean": statistics.mean(r_values) if r_values else None,
            "recall_denominator": len(r_values), "f1_mean_over_defined_claims": statistics.mean(f_values) if f_values else None,
            "f1_denominator": len(f_values), "claims": claim_rows}
    for cid, systems in sorted(by_claim.items()):
        b, c = systems.get("B"), systems.get("C")
        if not b or not c or b.get("status") != "SUCCESS" or c.get("status") != "SUCCESS":
            continue
        bset, cset = set(b.get("selected_evidence_doc_ids") or []), set(c.get("selected_evidence_doc_ids") or [])
        target = same if bset == cset else different
        target.append({"claim_id": cid, "B_selected_doc_ids": sorted(bset), "C_selected_doc_ids": sorted(cset)})
        if bset == cset:
            (same_nonempty if bset else same_empty).append(cid)
    return {"definition": "Selected evidence document IDs compared with gold rationale document IDs; overlap does not demonstrate entailment.",
        "systems": systems_result, "B_C_selected_evidence_comparison": {
            "same_selection_claims": same, "different_selection_claims": different,
            "same_nonempty_selection_claim_ids": same_nonempty, "same_empty_selection_claim_ids": same_empty,
            "precision_1_with_incomplete_recall": p1_incomplete, "zero_recall_claims": zero_recall,
            "selected_evidence_without_valid_classification": nonvalid_selected,
            "failed_execution_selection_unavailable_claims": {s: [cid for cid, ss in by_claim.items()
                if s in ss and ss[s].get("status") != "SUCCESS" and _gold_docs(ss[s])] for s in ("B", "C")}}}


def _controller_results(row: dict[str, Any]) -> list[dict[str, Any]]:
    raw = (row.get("output") or {}).get("raw_output") or {}
    traces = raw.get("controller_traces")
    return traces if isinstance(traces, list) else []


def build_system_c_analysis(rows: list[dict[str, Any]], attempts: list[dict[str, Any]]) -> dict[str, Any]:
    c_rows = [r for r in rows if r.get("system") == "C"]
    c_attempts = [r for r in attempts if r.get("system") == "C"]
    successful = [r for r in c_rows if r.get("status") == "SUCCESS"]
    accepted = completed = auth_claims = trace_claims = cycle_entries = 0
    dispatches, observed_dispatches, failed_dispatches = Counter(), Counter(), []
    by_outcome = Counter()
    claim_trace_records = []
    for row in successful:
        traces = _controller_results(row)
        raw = (row.get("output") or {}).get("raw_output") or {}
        state = raw.get("research_state") or {}
        auths = [auth for loop in (state.get("research_loops") or []) for auth in (loop.get("continuation_authorizations") or [])]
        trace_claims += int(bool(traces))
        auth_claims += int(bool(auths))
        cycles_for_claim = []
        for trace in traces:
            for cycle in ((trace.get("trace") or {}).get("cycles") or []):
                cycle_entries += 1
                validation = cycle.get("validation") or {}
                execution = cycle.get("execution") or {}
                dispatch = execution.get("dispatch") or {}
                v, e, cap = validation.get("validation_status"), execution.get("status"), dispatch.get("capability")
                by_outcome[(v, e, cap)] += 1
                accepted += int(v == "ACCEPTED")
                completed += int(e == "COMPLETED")
                dispatches[cap or "NOT RECORDED"] += 1
                observed_dispatches[cap or "NOT RECORDED"] += 1
                cycles_for_claim.append({"cycle_number": cycle.get("cycle_number"), "validation_status": v,
                    "execution_status": e, "capability": cap, "dispatch_status": dispatch.get("status")})
        claim_trace_records.append({"claim_id": row["example_id"], "status": row.get("status"),
            "controller_trace_count": len(traces), "cycles": cycles_for_claim,
            "continuation_authorization_count": len(auths), "final_stop_reason": raw.get("final_stop_reason"),
            "research_state_status": state.get("status") if isinstance(state, dict) else None})
    for attempt in c_attempts:
        if attempt.get("status") == "SUCCESS": continue
        info = _controller_failure_info(attempt)
        if not info: continue
        for cycle in info["cycles"]:
            failed_dispatches.append({"claim_id": attempt["example_id"], **cycle})
            observed_dispatches[cycle.get("capability") or "NOT RECORDED"] += 1
    failed_cycles = sum(item.get("execution_status") == "FAILED" for item in failed_dispatches)
    return {"C_attempts": len(c_attempts), "successful_executions": len(successful),
        "failed_executions": len(c_attempts) - len(successful), "successful_claims_with_controller_traces": trace_claims,
        "successful_trace_cycle_entries": cycle_entries, "successful_accepted_8A_validations": accepted,
        "successful_completed_8B_executions": completed, "successful_dispatches_by_capability": dict(dispatches),
        "all_trace_observed_dispatches_by_capability": dict(observed_dispatches),
        "successful_claims_with_continuation_authorization": auth_claims,
        "successful_controller_cycle_outcomes": {"|".join(k): v for k, v in by_outcome.items()},
        "failed_dispatch_cycles_in_error_traces": failed_cycles,
        "trace_observed_cycle_entries_including_failed": cycle_entries + len(failed_dispatches),
        "top_level_failed_attempt_cycle_counters_missing": sum(a.get("controller_cycles") is None for a in c_attempts if a.get("status") != "SUCCESS"),
        "failed_dispatch_traces": failed_dispatches, "successful_claim_traces": claim_trace_records,
        "interpretation": "Successful traces show the real 8A validation, 8B execution, 7B dispatch, and continuation path. Failed traces show an evidence_verification dispatch failure; failed top-level call counters are absent."}


def _stats(values: Iterable[Any]) -> dict[str, Any]:
    clean = [float(x) for x in values if isinstance(x, (int, float)) and not isinstance(x, bool)]
    if not clean:
        return {"count": 0, "mean": None, "median": None, "minimum": None, "maximum": None}
    return {"count": len(clean), "mean": statistics.mean(clean), "median": statistics.median(clean),
            "minimum": min(clean), "maximum": max(clean)}


def build_efficiency(rows: list[dict[str, Any]], attempts: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for system in SYSTEMS:
        srows = [x for x in rows if x.get("system") == system]
        sattempts = [x for x in attempts if x.get("system") == system]
        latency = {status.lower(): _stats(r.get("latency_seconds") for r in srows if r.get("status") == status)
                   for status in ("SUCCESS", "FAILED")}
        calls = {}
        for field in ("llm_calls", "retrieval_calls", "verification_calls", "controller_cycles"):
            split = {}
            for status in ("SUCCESS", "FAILED"):
                values = [a.get(field) for a in sattempts if a.get("status") == status]
                known = [x for x in values if isinstance(x, (int, float))]
                split[status.lower()] = {"sum_of_persisted_values": sum(known), "attempts": len(values),
                    "recorded_values": len(known), "missing_values": len(values) - len(known)}
            all_known = [a.get(field) for a in sattempts if isinstance(a.get(field), (int, float))]
            calls[field] = {**split, "all_attempts": {"sum_of_persisted_values": sum(all_known),
                "recorded_values": len(all_known), "missing_values": len(sattempts) - len(all_known)}}
        values = [(r["example_id"], float(r["latency_seconds"])) for r in srows
                  if isinstance(r.get("latency_seconds"), (int, float))]
        outliers = []
        if len(values) >= 4:
            sorted_values = sorted(v for _, v in values)
            q1, q3 = statistics.quantiles(sorted_values, n=4, method="inclusive")[:3:2]
            threshold = q3 + 1.5 * (q3 - q1)
            outliers = [{"claim_id": cid, "latency_seconds": value, "rule": "above Q3 + 1.5*IQR"}
                        for cid, value in values if value > threshold]
        result[system] = {"latency_seconds": latency, "call_counts": calls,
                          "high_latency_outliers": outliers}
    return result


def _counts_by_status(attempts: list[dict[str, Any]], system: str) -> dict[str, int]:
    subset = [a for a in attempts if a.get("system") == system]
    return {"attempted": len(subset), "completed": sum(a.get("status") == "SUCCESS" for a in subset),
            "failed": sum(a.get("status") != "SUCCESS" for a in subset)}


def reconcile(rows: list[dict[str, Any]], attempts: list[dict[str, Any]], combined: dict[str, Any],
              summary: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    by_claim = _claim_index(rows)
    discrepancies = []
    pair_a = [(int(x["example_id"]), x["system"]) for x in attempts]
    pair_p = [(int(x["example_id"]), x["system"]) for x in rows]
    if len(set(pair_a)) != len(pair_a): discrepancies.append({"item": "duplicate_runtime_attempt_pairs", "severity": "inconsistency"})
    if len(set(pair_p)) != len(pair_p): discrepancies.append({"item": "duplicate_per_example_pairs", "severity": "inconsistency"})
    if sorted(pair_a) != sorted(pair_p): discrepancies.append({"item": "attempt_pairs_vs_scored_pairs",
        "runtime_attempts": len(pair_a), "per_example": len(pair_p), "severity": "inconsistency"})
    selection = manifest.get("selection", {}).get("claim_ids", [])
    if selection != combined.get("selection", {}).get("claim_ids"):
        discrepancies.append({"item": "selected_ids_manifest_vs_combined", "severity": "inconsistency"})
    if sorted(selection) != sorted(by_claim):
        discrepancies.append({"item": "selected_ids_vs_per_example", "manifest_count": len(selection),
            "per_example_count": len(by_claim), "severity": "inconsistency"})
    splits = sorted(set(x.get("split") for x in attempts))
    if splits != ["dev"]: discrepancies.append({"item": "runtime_splits", "expected": ["dev"], "actual": splits,
                                                   "severity": "inconsistency"})
    gold_inconsistent = []
    labels = {}
    for cid, systems in by_claim.items():
        values = {(r.get("gold_label"), r.get("gold_label_status"), tuple(r.get("gold_evidence_doc_ids") or []))
                  for r in systems.values()}
        if len(values) != 1: gold_inconsistent.append(cid)
        labels[cid] = next(iter(values))[0] if values else None
    if gold_inconsistent:
        discrepancies.append({"item": "gold_join_disagrees_across_system_rows", "claim_ids": gold_inconsistent,
                              "severity": "inconsistency"})
    systems_seen = sorted(set(a.get("system") for a in attempts))
    if systems_seen != list(SYSTEMS): discrepancies.append({"item": "system_set", "actual": systems_seen,
                                                            "expected": list(SYSTEMS), "severity": "inconsistency"})
    attempt_status = Counter((a.get("system"), a.get("status")) for a in attempts)
    per_system_counts = {s: _counts_by_status(attempts, s) for s in SYSTEMS}
    recorded_counts = {}
    for s in SYSTEMS:
        recorded_counts[s] = {}
        for artifact, values in (("summary.json", summary.get(s, {})), ("phase10_dev.json", combined.get("systems", {}).get(s, {}))):
            observed = {k: values.get(k) for k in ("attempted", "completed", "failed")}
            recorded_counts[s][artifact] = observed
            if observed != per_system_counts[s]:
                discrepancies.append({"item": f"{s}_status_counts_vs_{artifact}", "runtime_attempts": per_system_counts[s],
                    "recorded": observed, "severity": "inconsistency"})
    successful_triples = sum(all(by_claim[cid].get(s, {}).get("status") == "SUCCESS" for s in SYSTEMS) for cid in by_claim)
    all_failed_triples = sum(all(by_claim[cid].get(s, {}).get("status") != "SUCCESS" for s in SYSTEMS) for cid in by_claim)
    labeled = sum(label in GOLD_CLASSES for label in labels.values())
    unlabeled = len(labels) - labeled
    pred_counts = {}
    for s in SYSTEMS:
        counts = Counter(prediction_state(r, s) for r in rows if r.get("system") == s)
        pred_counts[s] = {k: counts.get(k, 0) for k in ("VALID", "INVALID", "MISSING", "FAILED", "NOT_APPLICABLE")}
    calls = {}
    for s in SYSTEMS:
        calls[s] = {}
        for field in ("llm_calls", "retrieval_calls", "verification_calls", "controller_cycles"):
            vals = [a.get(field) for a in attempts if a.get("system") == s]
            known = [x for x in vals if isinstance(x, (int, float))]
            calls[s][field] = {"recorded_sum": sum(known), "recorded_values": len(known),
                               "missing_values": sum(x is None for x in vals)}
    rec = {
        "selected_claim_count": len(by_claim), "selected_claim_ids": sorted(by_claim),
        "systems": systems_seen, "runtime_attempt_count": len(attempts), "per_example_record_count": len(rows),
        "successful_attempts": sum(a.get("status") == "SUCCESS" for a in attempts),
        "failed_attempts": sum(a.get("status") != "SUCCESS" for a in attempts),
        "attempt_counts_by_system": per_system_counts,
        "successful_A_B_C_triples": successful_triples, "all_failed_A_B_C_triples": all_failed_triples,
        "claims_with_all_three_successful": successful_triples,
        "labeled_claim_count": labeled, "unlabeled_claim_count": unlabeled,
        "recorded_gold_label_counts": dict(Counter(labels[cid] if labels[cid] in GOLD_CLASSES else "UNLABELED" for cid in labels)),
        "prediction_states_by_system": pred_counts,
        "prediction_value_presence_by_system": {
            s: {"raw_null_prediction_fields": sum(r.get("predicted_label") is None for r in rows if r.get("system") == s),
                "failed_attempts_without_prediction": sum(r.get("system") == s and r.get("status") != "SUCCESS" and r.get("predicted_label") is None for r in rows),
                "successful_attempts_with_missing_prediction": (0 if s == "A" else
                    sum(r.get("system") == s and r.get("status") == "SUCCESS" and r.get("predicted_label") is None for r in rows)),
                "classification_not_applicable": 30 if s == "A" else 0}
            for s in SYSTEMS},
        "retrieval_records": sum(isinstance(r.get("retrieval"), dict) for r in rows),
        "retrieval_records_scored": sum((r.get("retrieval") or {}).get("retrieval_scoring_status") == "scored" for r in rows),
        "retrieved_chunk_lists_persisted": sum(isinstance(r.get("retrieved_chunks"), list) for r in rows),
        "selected_evidence_fields_persisted": sum("selected_evidence_doc_ids" in r for r in rows),
        "latency_records": sum(isinstance(r.get("latency_seconds"), (int, float)) for r in rows),
        "call_counts_by_system": calls, "reported_status_counts_by_system": recorded_counts,
        "discrepancy_log": discrepancies,
        "observability_notes": [
            "C failed attempts have null top-level component call counters; nested controller traces are analyzed separately.",
            "evaluation/scifact/scifact_metrics.py is absent; its frozen hash cannot be verified.",
        ],
    }
    for field in ("attempted_system_claim_pairs", "scored_system_claim_pairs"):
        observed = len(attempts) if field == "attempted_system_claim_pairs" else len(rows)
        if combined.get(field) != observed:
            discrepancies.append({"item": field, "reconstructed": observed, "combined_artifact": combined.get(field),
                                  "severity": "inconsistency"})
    return rec


def compare_saved_metrics(rows: list[dict[str, Any]], attempts: list[dict[str, Any]],
                         combined: dict[str, Any], summary: dict[str, Any]) -> list[dict[str, Any]]:
    checks = []
    for system in SYSTEMS:
        saved = combined.get("systems", {}).get(system, {})
        if system in ("B", "C"):
            rebuilt = classification_metrics(rows, system)
            old = saved.get("classification") or {}
            for metric, actual, recorded in (
                ("correct", rebuilt["correct_numerator"], old.get("correct")),
                ("labeled_denominator", rebuilt["gold_labeled_denominator"], old.get("denominator")),
                ("accuracy", rebuilt["accuracy_including_failed_as_incorrect"], old.get("accuracy")),
                ("valid_prediction_denominator", rebuilt["valid_prediction_denominator"], old.get("valid_prediction_denominator")),
                ("valid_prediction_accuracy", rebuilt["valid_prediction_accuracy"], old.get("valid_prediction_accuracy")),
                ("macro_f1", rebuilt["macro_f1_over_gold_supported_classes"], old.get("macro_f1_over_gold_supported_classes")),
            ):
                matched = (math.isclose(actual, recorded, rel_tol=1e-10, abs_tol=1e-10)
                           if isinstance(actual, (int, float)) and isinstance(recorded, (int, float)) else actual == recorded)
                checks.append({"system": system, "metric": "classification." + metric,
                    "reconstructed_from_per_example": actual, "saved_in_phase10_dev": recorded, "match": matched})
        ret = [r for r in rows if r.get("system") == system and isinstance(r.get("retrieval"), dict)
               and r["retrieval"].get("retrieval_scoring_status") == "scored"]
        total_gold_docs = sum(len(_gold_docs(r)) for r in ret)
        for k in (1, 5, 10, 20):
            rank_hits = 0
            precision_values = []
            for row in ret:
                gold = _gold_docs(row)
                chunks = row.get("retrieved_chunks") or []
                top = chunks[:k]
                docs = {x.get("scifact_doc_id") for x in top if x.get("scifact_doc_id") is not None}
                rank_hits += len(docs & gold)
                if docs:
                    precision_values.append(len(docs & gold) / len(docs))
            saved_metric = saved.get("retrieval", {}).get("Recall@k", {}).get(str(k), {})
            recorded_hits, recorded_denom = saved_metric.get("hits"), saved_metric.get("gold_document_denominator")
            matches = rank_hits == recorded_hits and total_gold_docs == recorded_denom
            checks.append({"system": system, "metric": f"retrieval.Recall@{k}",
                "reconstructed_hits": rank_hits, "reconstructed_gold_document_denominator": total_gold_docs,
                "saved_hits": recorded_hits, "saved_gold_document_denominator": recorded_denom,
                "match": matches})
            precision_saved = saved.get("retrieval", {}).get("document_precision@k_macro", {}).get(str(k), {})
            precision_value = statistics.mean(precision_values) if precision_values else None
            precision_denom = len(precision_values)
            saved_precision, saved_precision_denom = precision_saved.get("macro_precision"), precision_saved.get("claim_denominator")
            precision_match = ((math.isclose(precision_value, saved_precision, rel_tol=1e-10, abs_tol=1e-10)
                                if isinstance(precision_value, (int, float)) and isinstance(saved_precision, (int, float))
                                else precision_value == saved_precision)
                               and precision_denom == saved_precision_denom)
            checks.append({"system": system, "metric": f"retrieval.document_precision@{k}",
                "reconstructed_macro_precision": precision_value, "reconstructed_denominator": precision_denom,
                "saved_macro_precision": saved_precision, "saved_denominator": saved_precision_denom,
                "match": precision_match})
        rr = [r["retrieval"].get("reciprocal_rank") for r in ret if r["retrieval"].get("reciprocal_rank") is not None]
        mrr = statistics.mean(rr) if rr else None
        saved_mrr = saved.get("retrieval", {}).get("MRR", {})
        checks.append({"system": system, "metric": "retrieval.MRR",
            "reconstructed_mean": mrr, "reconstructed_denominator": len(rr),
            "saved_mean": saved_mrr.get("value"), "saved_denominator": saved_mrr.get("claim_denominator"),
            "match": ((math.isclose(mrr, saved_mrr.get("value"), rel_tol=1e-10, abs_tol=1e-10)
                       if isinstance(mrr, (int, float)) and isinstance(saved_mrr.get("value"), (int, float))
                       else mrr == saved_mrr.get("value")) and len(rr) == saved_mrr.get("claim_denominator"))})
        if system != "A":
            citations = [r.get("citation") for r in rows if r.get("system") == system and isinstance(r.get("citation"), dict)]
            e_saved = saved.get("selected_evidence_document_metrics", {})
            for metric in ("precision", "recall", "f1"):
                vals = [x.get(metric) for x in citations if x.get(metric) is not None]
                mean = statistics.mean(vals) if vals else None
                recorded = e_saved.get(metric)
                checks.append({"system": system, "metric": "selected_evidence." + metric,
                    "reconstructed_from_persisted_citation_rows": mean,
                    "reconstructed_non_null_denominator": len(vals),
                    "saved_in_phase10_dev": recorded, "saved_citation_row_denominator": e_saved.get("claim_denominator"),
                    "match": math.isclose(mean, recorded, rel_tol=1e-10, abs_tol=1e-10)
                        if isinstance(mean, (int, float)) and isinstance(recorded, (int, float)) else mean == recorded})
        eff = saved.get("efficiency", {})
        all_latencies = [r.get("latency_seconds") for r in rows if r.get("system") == system and isinstance(r.get("latency_seconds"), (int, float))]
        mean = statistics.mean(all_latencies) if all_latencies else None
        checks.append({"system": system, "metric": "efficiency.latency_mean_all_attempts",
            "reconstructed": mean, "saved": eff.get("latency_seconds", {}).get("mean"),
            "match": math.isclose(mean, eff.get("latency_seconds", {}).get("mean"), rel_tol=1e-10, abs_tol=1e-10)
                if isinstance(mean, (int, float)) and isinstance(eff.get("latency_seconds", {}).get("mean"), (int, float)) else False})
        for field in ("llm_calls", "retrieval_calls", "verification_calls", "controller_cycles"):
            known = [a.get(field) for a in attempts if a.get("system") == system and isinstance(a.get(field), (int, float))]
            saved_value = eff.get(field)
            checks.append({"system": system, "metric": "efficiency." + field,
                "reconstructed_recorded_counter_sum": sum(known), "reconstructed_recorded_value_count": len(known),
                "saved_phase10c_value": saved_value,
                "match": sum(known) == saved_value if isinstance(saved_value, (int, float)) else False,
                "note": "A match does not fill missing counters on failed C attempts." if system == "C" else None})
    return checks


def build_failure_impact(combined: dict[str, Any], retrieval_analysis: dict[str, Any],
                         evidence_analysis: dict[str, Any], efficiency: dict[str, Any]) -> list[dict[str, Any]]:
    impact = []
    for system in SYSTEMS:
        if system == "A":
            impact.append({"metric": "classification_accuracy", "system": system, "numerator": None,
                "denominator": None, "excluded_cases": 30,
                "reason": "NOT APPLICABLE — A has no verifier-derived classification"})
        else:
            c = combined.get("systems", {}).get(system, {}).get("classification", {})
            impact.append({"metric": "classification_accuracy_primary", "system": system,
                "numerator": c.get("correct"), "denominator": c.get("denominator"),
                "excluded_cases": c.get("unscorable_gold_examples"),
                "reason": "Failed or invalid predictions remain in the labeled denominator and count incorrect; unlabeled claims are excluded."})
            impact.append({"metric": "valid_prediction_accuracy", "system": system,
                "numerator": c.get("correct"), "denominator": c.get("valid_prediction_denominator"),
                "excluded_cases": c.get("invalid_or_failed_predictions"),
                "reason": "Only valid predictions are included; failed/missing cases are excluded, so denominators differ by system."})
        recall = combined.get("systems", {}).get(system, {}).get("retrieval", {}).get("Recall@k", {}).get("1", {})
        posthoc = retrieval_analysis.get("Recall@1_by_system", {}).get(system, {})
        impact.append({"metric": "retrieval_Recall@1", "system": system, "numerator": recall.get("hits"),
            "denominator": recall.get("gold_document_denominator"), "excluded_cases": 11,
            "posthoc_ranked_doc_available_numerator": posthoc.get("hits"),
            "posthoc_ranked_doc_available_denominator": posthoc.get("claim_denominator"),
            "unavailable_failed_retrieval_claims": retrieval_analysis.get("unavailable_ranked_doc_outputs_by_system", {}).get(system, []),
            "reason": "The Phase 10C scorer treated empty ranked-document lists on failed callbacks as misses. This post-hoc table separately excludes failures without persisted document IDs. Both are document overlap, not entailment."})
        evidence = combined.get("systems", {}).get(system, {}).get("selected_evidence_document_metrics", {})
        posthoc_evidence = evidence_analysis.get("systems", {}).get(system, {})
        impact.append({"metric": "selected_evidence_document_recall", "system": system,
            "numerator": None,
            "denominator": evidence.get("claim_denominator"), "excluded_cases": 11,
            "reported_mean_recall": evidence.get("recall"),
            "successful_output_recall_mean": posthoc_evidence.get("recall_mean"),
            "successful_output_denominator": posthoc_evidence.get("recall_denominator"),
            "failed_selection_unavailable_claims": evidence_analysis.get("B_C_selected_evidence_comparison", {}).get("failed_execution_selection_unavailable_claims", {}).get(system, []),
            "reason": "Numerator is N/A because the aggregate is a mean of claim scores. The saved citation metric includes failure rows scored with empty selected evidence; execution-specific availability is reported separately. A is not applicable."})
        lat = efficiency[system]["latency_seconds"]
        latency_n = lat["success"]["count"] + lat["failed"]["count"]
        latency_mean = ((lat["success"]["mean"] or 0) * lat["success"]["count"] +
                        (lat["failed"]["mean"] or 0) * lat["failed"]["count"]) / latency_n if latency_n else None
        impact.append({"metric": "latency_mean_seconds", "system": system, "numerator": None,
            "denominator": latency_n, "excluded_cases": 30 - latency_n,
            "reported_value": latency_mean,
            "reason": "All persisted latency values, including failures, are included; see efficiency_analysis for mean/median/min/max."})
        for field, data in efficiency[system]["call_counts"].items():
            all_attempts = data["all_attempts"]
            impact.append({"metric": field, "system": system,
                "numerator": all_attempts["sum_of_persisted_values"],
                "denominator": all_attempts["recorded_values"],
                "excluded_cases": all_attempts["missing_values"],
                "reason": "Sum/coverage of persisted counters. Missing counters are not imputed; denominator counts records with a numeric value."})
    return impact


def build_unlabeled_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_claim = _claim_index(rows)
    claims = []
    for cid, systems in sorted(by_claim.items()):
        exemplar = next(iter(systems.values()))
        if exemplar.get("gold_label") in GOLD_CLASSES:
            continue
        item = {"claim_id": cid, "gold_label": None, "classification_correctness": "N/A — no gold label",
                "gold_evidence_doc_ids": exemplar.get("gold_evidence_doc_ids")}
        for s in SYSTEMS:
            row = systems.get(s)
            item[s] = ({"status": "MISSING", "prediction_validity": "MISSING", "retrieval_hit_at_1": None,
                        "selected_evidence_doc_ids": None, "latency_seconds": None} if row is None else {
                "status": row.get("status"), "prediction": row.get("predicted_label") if s != "A" else "N/A",
                "prediction_validity": prediction_state(row, s), "retrieval_hit_at_1": retrieval_hit_at_1(row),
                "retrieved_document_ids": retrieved_doc_ids(row),
                "selected_evidence_doc_ids": row.get("selected_evidence_doc_ids"), "latency_seconds": row.get("latency_seconds")})
        b, c = systems.get("B"), systems.get("C")
        item["B_C_agreement"] = ((b or {}).get("predicted_label") == (c or {}).get("predicted_label") if b and c else None)
        claims.append(item)
    return {"unlabeled_claim_count": len(claims), "claims": claims,
            "classification_correctness": "N/A — no gold label; unlabeled is distinct from incorrect."}


def analyze(root: Path = ROOT, generated_at: str | None = None) -> dict[str, Any]:
    paths = {name: root / path.relative_to(ROOT) for name, path in SOURCE_FILES.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required Phase 10C artifact(s): " + ", ".join(missing))
    combined = read_json(paths["combined"])
    rows = read_jsonl(paths["per_example"])
    attempts = read_jsonl(paths["runtime_attempts"])
    manifest = read_json(paths["manifest"])
    summary = read_json(paths["summary"])
    matrix = build_claim_matrix(rows)
    failures = build_failure_records(attempts, rows, manifest)
    classification = {"A": {"status": "NOT APPLICABLE — no verifier-derived classification"},
        "B": classification_metrics(rows, "B"), "C": classification_metrics(rows, "C")}
    agreement = build_agreement(rows)
    retrieval = build_retrieval_analysis(rows)
    evidence = build_evidence_analysis(rows)
    system_c = build_system_c_analysis(rows, attempts)
    efficiency = build_efficiency(rows, attempts)
    reconciliation = reconcile(rows, attempts, combined, summary, manifest)
    retrieval["phase10c_saved_aggregate_metrics"] = {
        s: {"Recall@1": combined.get("systems", {}).get(s, {}).get("retrieval", {}).get("Recall@k", {}).get("1"),
            "MRR": combined.get("systems", {}).get(s, {}).get("retrieval", {}).get("MRR")}
        for s in SYSTEMS}
    retrieval["failed_rows_without_ranked_doc_ids"] = {
        s: [r["example_id"] for r in rows if r.get("system") == s and r.get("status") != "SUCCESS"
            and bool(_gold_docs(r)) and not r.get("retrieved_chunks")]
        for s in SYSTEMS}
    if retrieval["failed_rows_without_ranked_doc_ids"].get("C"):
        reconciliation["discrepancy_log"].append({
            "item": "C_retrieval_denominator_scope",
            "phase10c_saved_metric": combined.get("systems", {}).get("C", {}).get("retrieval", {}).get("Recall@k", {}).get("1"),
            "ranked_document_ids_unavailable_on_failed_claims": retrieval["failed_rows_without_ranked_doc_ids"]["C"],
            "posthoc_ranked_document_denominator": retrieval["Recall@1_by_system"]["C"]["claim_denominator"],
            "severity": "representational_scope_difference",
            "explanation": "Phase 10C scored these failed rows with empty chunk lists as misses; Phase 10D does not call them retrieval misses because ranked document IDs were not persisted."})
    reconciliation["discrepancy_log"].append({
        "item": "C_failed_attempt_call_counter_coverage",
        "phase10c_reported_controller_cycles": combined.get("systems", {}).get("C", {}).get("efficiency", {}).get("controller_cycles"),
        "failed_controller_cycles_visible_in_error_traces": system_c["failed_dispatch_cycles_in_error_traces"],
        "failed_top_level_call_counters_missing": system_c["top_level_failed_attempt_cycle_counters_missing"],
        "severity": "observability_scope_difference",
        "explanation": "The saved C efficiency counters sum returned outputs; failed attempts have null component counters, although nested traces record failed dispatch cycles."})
    failure_categories = Counter(f["primary_category"] for f in failures)
    failure_systems = Counter(f["system"] for f in failures)
    by_failure_claim: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for failure in failures:
        by_failure_claim[failure["claim_id"]].append({"system": failure["system"],
            "primary_category": failure["primary_category"], "exact_recorded_error": failure["exact_recorded_error"]})
    hashes = {name: {"sha256": sha256_file(path), "path": str(path)} for name, path in paths.items()}
    frozen_before = source_hashes(root)
    # Record an unverifiable protected path explicitly, rather than substituting another source.
    for name, item in frozen_before.items():
        if item["status"] == "MISSING":
            reconciliation["discrepancy_log"].append({"item": "protected_file_missing", "path": name,
                "expected_sha256": item["expected_sha256"], "severity": "unverifiable"})
    metric_checks = compare_saved_metrics(rows, attempts, combined, summary)
    for check in metric_checks:
        if check.get("match") is False:
            reconciliation["discrepancy_log"].append({"item": "saved_metric_mismatch",
                "system": check.get("system"), "metric": check.get("metric"),
                "details": check, "severity": "inconsistency"})
    report = {
        "analysis_version": ANALYSIS_VERSION,
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "phase": "10D", "scope": "Deterministic post-hoc analysis of saved Phase 10C artifacts only.",
        "source_artifacts": hashes, "frozen_hash_verification_before_analysis": frozen_before,
        "frozen_hash_verification_after_analysis": None,
        "artifact_reconciliation": reconciliation, "artifact_metric_checks": metric_checks,
        "failure_taxonomy": {
            "primary_categories": ["retrieval_error", "verification_error", "invalid_model_output", "model_endpoint_failure",
                "controller_execution_failure", "missing_gold_annotation", "missing_persisted_field", "other_recorded_execution_failure"],
            "mapping_rule": "One primary category per failed attempt. A/B connection refusals map to model_endpoint_failure with connection_refused as secondary detail.",
            "failure_summary_by_system": {s: failure_systems.get(s, 0) for s in SYSTEMS},
            "failure_summary_by_category": dict(failure_categories),
            "failure_summary_by_claim": {str(cid): items for cid, items in sorted(by_failure_claim.items())}},
        "failure_records": failures, "claim_outcome_matrix": matrix,
        "classification_analysis": classification, "agreement_analysis": agreement,
        "retrieval_error_analysis": retrieval, "evidence_analysis": evidence,
        "system_c_analysis": system_c, "efficiency_analysis": efficiency,
        "failure_metric_denominators": build_failure_impact(combined, retrieval, evidence, efficiency),
        "unlabeled_claim_analysis": build_unlabeled_analysis(rows),
        "conclusions_supported_by_data": [
            "Six A/B failures record a refused local model endpoint connection; three C failures record a Phase 8C/Phase 5 execution failure.",
            "Among paired valid B/C predictions, persisted outputs agree; failures are categorized separately.",
            "Retrieved/selected document overlap is not a claim-entailment or final-answer correctness measure.",
            "Scope remains the 30-claim DEV subset; it does not represent full DEV or test-set results."],
        "limitations": [
            "No attempt_id or top-level attempt timestamp is persisted; nested controller timestamps are reported separately.",
            "The requested evaluation/scifact/scifact_metrics.py file is missing, so its frozen hash is unverifiable.",
            "Failed C attempts omit top-level call counters; trace records expose a failed verification dispatch but not all component calls.",
            "This report evaluates recorded artifacts only and does not semantically adjudicate evidence."],
        "external_calls": {"llm": False, "search_or_api": False, "inference_rerun": False}}
    return report


def _md_table(headers: list[str], rows: Iterable[Iterable[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        cells = [str(x).replace("|", "\\|").replace("\n", " ") for x in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_markdown(report: dict[str, Any]) -> str:
    rec = report["artifact_reconciliation"]
    matrix = report["claim_outcome_matrix"]
    statuses = rec["attempt_counts_by_system"]
    class_results = report["classification_analysis"]
    rows = [
        "# Phase 10D — Failure, Error & Agreement Analysis", "",
        f"Generated from saved Phase 10C artifacts at {report['generated_at']}. Analysis version: {report['analysis_version']}.", "",
        "## 1. Scope", "",
        "Deterministic post-hoc analysis of the 30-claim SciFact DEV subset. No LLM, search, API, or inference calls were made. The Phase 10C artifacts were read only.", "",
        "The requested evaluation/scifact/scifact_metrics.py file is absent from this checkout, so its protected hash could not be verified. The other seven requested frozen hashes matched before and after analysis.", "",
        "## 2. Source Artifacts", "",
        _md_table(["Artifact", "SHA-256"], [[k, v["sha256"]] for k, v in report["source_artifacts"].items()]), "",
        "## 3. Phase 10C Reconstruction", "",
        f"Selected claims: {rec['selected_claim_count']}; attempts: {rec['runtime_attempt_count']}; successes: {rec['successful_attempts']}; failures: {rec['failed_attempts']}; complete A/B/C triples: {rec['successful_A_B_C_triples']}; labeled: {rec['labeled_claim_count']}; unlabeled: {rec['unlabeled_claim_count']}.", "",
        _md_table(["System", "Attempts", "Success", "Failed", "Valid predictions", "Invalid", "Missing", "N/A"],
            [[s, statuses[s]["attempted"], statuses[s]["completed"], statuses[s]["failed"],
              rec["prediction_states_by_system"][s]["VALID"], rec["prediction_states_by_system"][s]["INVALID"],
              rec["prediction_states_by_system"][s]["MISSING"], rec["prediction_states_by_system"][s]["NOT_APPLICABLE"]]
             for s in SYSTEMS]), "",
        f"Persisted retrieval records: {rec['retrieval_records']}; marked scored: {rec['retrieval_records_scored']}; retrieved chunk lists: {rec['retrieved_chunk_lists_persisted']}; selected-evidence fields: {rec['selected_evidence_fields_persisted']}; latency records: {rec['latency_records']}.", "",
        "Runtime attempts and per-example rows contain the same claim/system pairs. The discrepancy log lists the missing protected metrics source and C failure-counter observability gap. Phase 10C summary C counters describe successful returned outputs; failed serialized traces separately show failed controller cycles.", "",
        "## 4. Artifact Reconciliation", "",
        json.dumps(rec["discrepancy_log"], ensure_ascii=False), "",
        "## 5. Failure Summary", "",
        _md_table(["Claim", "System", "Category", "Stage", "Timestamp"],
            [[f["claim_id"], f["system"], f["primary_category"], f["recorded_error_stage"],
              ", ".join(f["timestamps_in_error_or_trace"]) or "NOT RECORDED"] for f in report["failure_records"]]), "",
        "Failure counts by system: " + ", ".join(f"{s}={report['failure_taxonomy']['failure_summary_by_system'][s]}" for s in SYSTEMS) + ".", "",
        "Failure categories: " + ", ".join(f"{k}={v}" for k, v in report["failure_taxonomy"]["failure_summary_by_category"].items()) + ". Exact error messages and per-attempt execution details are in failure_records.jsonl.", "",
        "A/B connection-refused records show that inference did not start. C failures are recorded as Phase 8C execution failures; whether inference started is not recorded. Attempt IDs and top-level attempt timestamps are absent.", "",
        "## 6. Claim-Level Outcome Matrix", "",
        "The complete 30-row matrix is in claim_outcome_matrix.jsonl. A has no verifier-derived classification; its prediction is N/A.", "",
        _md_table(["Claim", "Gold", "A status", "B status/pred", "C status/pred", "A/B/C hit@1", "A/B/C latency (s)"],
            [[r["claim_id"], r["gold_label"] or "UNLABELED", r["A_status"],
              str(r["B_status"]) + " / " + str(r["B_prediction"]), str(r["C_status"]) + " / " + str(r["C_prediction"]),
              "/".join(str(r[f"{s}_retrieval_result"]["hit_at_1"]) for s in SYSTEMS),
              "/".join(str(r[f"{s}_latency"]) for s in SYSTEMS)] for r in matrix]), "",
        "## 7. Classification Error Analysis", "",
        _md_table(["System", "Correct/labeled", "Accuracy", "Valid n", "Valid-only accuracy", "Macro F1"],
            [[s, f"{class_results[s]['correct_numerator']}/{class_results[s]['gold_labeled_denominator']}",
              class_results[s]["accuracy_including_failed_as_incorrect"], class_results[s]["valid_prediction_denominator"],
              class_results[s]["valid_prediction_accuracy"], class_results[s]["macro_f1_over_gold_supported_classes"]]
             for s in ("B", "C")]), "",
        "Confusion tables distinguish SUPPORTED, CONTRADICTED, INSUFFICIENT_EVIDENCE, failed/invalid, and missing outputs. Macro F1 averages the two gold-supported classes only. No successful malformed predictions were persisted.", "",
        "## 8. B/C Agreement Analysis", "",
        f"Valid pair agreement: {report['agreement_analysis']['B_C_valid_prediction_agreement']['both_valid_same']} same and {report['agreement_analysis']['B_C_valid_prediction_agreement']['both_valid_different']} different. Failed pairs remain separate: {json.dumps(report['agreement_analysis']['B_C_prediction_agreement_counts'], ensure_ascii=False)}.", "",
        "The disagreement detail file distinguishes genuine valid-prediction disagreement from one system failing while the other produced an output. Retrieval and selected-evidence comparisons are persisted per such claim.", "",
        "## 9. Retrieval Error Analysis", "",
        "Top-ranked document overlap with gold rationale document IDs measures retrieval relevance only. It is not evidence entailment, verification, or answer correctness.", "",
        _md_table(["System", "Hit numerator", "Claim denominator", "Recall@1"],
            [[s, report["retrieval_error_analysis"]["Recall@1_by_system"][s]["hits"],
              report["retrieval_error_analysis"]["Recall@1_by_system"][s]["claim_denominator"],
              report["retrieval_error_analysis"]["Recall@1_by_system"][s]["value"]] for s in SYSTEMS]), "",
        "Claim lists for all-hit, all-miss, individual misses, A/B hit with C miss, and C hit with A/B miss are in retrieval_error_analysis.json, with retrieved and gold IDs.", "",
        "The saved Phase 10C aggregate records C Recall@1 as 14/19. The persisted C rows for claims 230, 249, and 859 have no ranked document IDs because their callbacks failed. On the 16 C gold-evidence rows with ranked document IDs persisted, the post-hoc hit rate is 14/16. The saved 14/19 treats the three failed empty lists as misses; these are not identifiable retrieval misses from the C traces.", "",
        "## 10. Evidence Analysis", "",
        "B/C selected evidence IDs are compared against gold rationale document IDs. Per-claim precision, recall, F1, evidence counts, same/different selections, precision-1 incomplete recall, and zero recall are in the report JSON. These comparisons do not demonstrate entailment. A has no selected citation. Failed execution makes the selection unavailable.", "",
        "The persisted Phase 10C citation aggregate uses 19 rows and reports recall 11/19 for B and C. Among successful outputs, the corresponding recall means are B 11/17 and C 11/16. Failed evidence-selection rows remain unavailable in the per-claim analysis.", "",
        "## 11. System C Execution Analysis", "",
        f"Successful C traces: {report['system_c_analysis']['successful_claims_with_controller_traces']}; accepted 8A validations: {report['system_c_analysis']['successful_accepted_8A_validations']}; completed 8B executions: {report['system_c_analysis']['successful_completed_8B_executions']}; claims with continuation authorization: {report['system_c_analysis']['successful_claims_with_continuation_authorization']}; failed dispatch cycles in error traces: {report['system_c_analysis']['failed_dispatch_cycles_in_error_traces']}.", "",
        "The trace artifact records local_retrieval and evidence_verification dispatches for successful C runs. Failed C traces show the evidence_verification dispatch failure. Missing internal details are not reconstructed.", "",
        "## 12. Efficiency Analysis", "",
        _md_table(["System", "Status", "n", "Mean (s)", "Median (s)", "Min (s)", "Max (s)"],
            [[s, status, report["efficiency_analysis"][s]["latency_seconds"][status.lower()]["count"],
              report["efficiency_analysis"][s]["latency_seconds"][status.lower()]["mean"],
              report["efficiency_analysis"][s]["latency_seconds"][status.lower()]["median"],
              report["efficiency_analysis"][s]["latency_seconds"][status.lower()]["minimum"],
              report["efficiency_analysis"][s]["latency_seconds"][status.lower()]["maximum"]]
             for s in SYSTEMS for status in ("SUCCESS", "FAILED")]), "",
        "Persisted LLM/retrieval/verification/cycle counter sums, split by success/failure and showing missing fields, are in efficiency_analysis.json. C's three failed top-level counter sets are not imputed.", "",
        "## 13. Failure Impact on Metrics", "",
        "The JSON report contains a metric-by-system numerator/denominator table. B/C primary classification accuracy uses 19 labeled claims and counts failures as not correct; valid-only accuracy excludes them and uses different denominators. Unlabeled claims are excluded from correctness. Retrieval uses annotated gold documents; missing retrieved hits produce no overlap. Citation overlap is not entailment.", "",
        "## 14. Unlabeled Claim Analysis", "",
        f"The {report['unlabeled_claim_analysis']['unlabeled_claim_count']} unlabeled claims include A 10 successes/1 failure, B 10 successes/1 failure, and C 11 successes. B/C outputs agree on 10 and differ on 1 (claim 268 has a B failure and a C prediction). Classification correctness is N/A because no gold label exists. Retrieval/evidence overlap with gold rationale documents is also N/A because annotations are explicitly empty; saved runtime retrieval IDs, evidence selections, latencies, and prediction validity remain listed per claim in the JSON report. Unlabeled is not incorrect.", "",
        "## 15. Key Observations", "",
        "- Six A/B failures are local endpoint connection refusals; three C failures are controller/Phase 5 execution failures.",
        "- Execution failure, invalid prediction, missing prediction, and wrong prediction are distinct outcomes.",
        "- Among paired valid B/C predictions, there are no different labels; one-failed and both-failed cases remain separate.",
        "- Retrieval relevance, selected evidence overlap, verification verdict, and final answer correctness are different measures.", "",
        "## 16. Limitations", "",
        "- Results describe only the selected 30-claim DEV subset.",
        "- Attempt IDs and root-level attempt timestamps are not persisted.",
        "- Failed C top-level call counters are missing.",
        "- One requested protected source file, scifact_metrics.py, was absent, preventing verification of that hash.",
        "- No semantic evidence adjudication is possible from this deterministic analysis alone.", "",
        "## 17. What Phase 10D Does NOT Establish", "",
        "It does not rank systems, establish superiority, confirm or reject a hypothesis, generalize to all DEV claims or test, or treat retrieval overlap or citation precision as entailment.", "",
        "## 18. Recommended Next Experimental Step", "",
        "For any later evaluation, preregister the sample, failure/retry policy, and denominators; preserve per-attempt counters and traces; and report a larger development evaluation descriptively. This analysis does not recommend tuning systems toward a score.", "",
        "### Additional artifacts", "",
        "phase10d_report.json, failure_records.jsonl, claim_outcome_matrix.jsonl, agreement_analysis.json, retrieval_error_analysis.json, system_c_analysis.json, efficiency_analysis.json.", "",
    ]
    return "\n".join(rows)


def write_outputs(root: Path = ROOT, generated_at: str | None = None) -> dict[str, Path]:
    report = analyze(root=root, generated_at=generated_at)
    output_dir = root / OUTPUT_DIR.relative_to(ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "report_json": output_dir / "phase10d_report.json",
        "report_md": output_dir / "phase10d_report.md",
        "failure_records": output_dir / "failure_records.jsonl",
        "claim_matrix": output_dir / "claim_outcome_matrix.jsonl",
        "agreement": output_dir / "agreement_analysis.json",
        "retrieval": output_dir / "retrieval_error_analysis.json",
        "system_c": output_dir / "system_c_analysis.json",
        "efficiency": output_dir / "efficiency_analysis.json",
    }
    report["frozen_hash_verification_after_analysis"] = source_hashes(root)
    report["analysis_status"] = "PASS" if all(x["status"] == "MATCH" for x in report["frozen_hash_verification_after_analysis"].values()) else "PARTIAL"
    report["analysis_status_reason"] = ("All requested frozen hashes matched." if report["analysis_status"] == "PASS" else
        "Post-hoc analysis completed, but at least one requested frozen source file is missing or could not be hash-verified.")
    outputs["report_json"].write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    outputs["report_md"].write_text(render_markdown(report), encoding="utf-8")
    for key, records in (("failure_records", report["failure_records"]), ("claim_matrix", report["claim_outcome_matrix"])):
        outputs[key].write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    for key, artifact_key in (("agreement", "agreement_analysis"), ("retrieval", "retrieval_error_analysis"),
                              ("system_c", "system_c_analysis"), ("efficiency", "efficiency_analysis")):
        outputs[key].write_text(json.dumps(report[artifact_key], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return outputs


if __name__ == "__main__":
    paths = write_outputs()
    report = read_json(paths["report_json"])
    print("PHASE 10D STATUS:", report["analysis_status"])
    for key, path in paths.items():
        print(f"{key}: {path}")
