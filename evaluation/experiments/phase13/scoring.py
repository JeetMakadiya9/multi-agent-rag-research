"""Score Phase 13 variants after execution (gold joined only here).

Classification uses the frozen Phase 10F ``_classification`` function; evidence
uses the Phase 11B micro document-overlap definition and the frozen
``retrieval_metrics_for_claim``. B3 (guard v2) is a deterministic
re-aggregation of a live variant's recorded stances; see ``b_plus.reaggregate_output``.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from typing import Any, Mapping, Sequence

from evaluation.experiments.phase13.common import (
    BUCKETS, GOLD_LABELS, OUTPUT_DIR, citation_micro, classification_metrics, gold_by_claim, insufficient_rate,
    load_dataset, mean, median, read_jsonl, stance_counts,
)


def load_attempts(variant: str) -> list[dict[str, Any]]:
    path = OUTPUT_DIR / "runs" / variant / "attempts.jsonl"
    return read_jsonl(path) if path.exists() else []


def judged_texts(row: Mapping[str, Any]) -> list[str]:
    """Exact evidence texts the judge saw (from the recorded Experiment B trace)."""
    trace = (row.get("raw_output") or {}).get("trace") or []
    for stage in trace:
        if stage.get("stage") == "pipeline_input":
            return [e["text"] for e in stage["initial_evidence"]]
    return []


def apply_guard_v2(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Deterministically re-aggregate recorded stances with guard v2 (no inference)."""
    from evaluation.experiments.phase13.b_plus import reaggregate_output
    out = []
    for row in rows:
        new = dict(row)
        if row.get("status") == "SUCCESS":
            v2 = reaggregate_output(row["runtime_input"]["claim"], row.get("assessments") or [],
                                    judged_texts(row), guard_version="v2")
            v1 = reaggregate_output(row["runtime_input"]["claim"], row.get("assessments") or [],
                                    judged_texts(row), guard_version="v1")
            if v1["verdict"] != row["predicted_label"]:
                raise AssertionError(f"v1 re-aggregation mismatch for claim {row['claim_id']}")
            new["predicted_label"] = v2["verdict"]
            new["guard_v1_messages"] = v1["guard_messages"]
            new["guard_v2_messages"] = v2["guard_messages"]
            new["verdict_before_guard_v2"] = row["predicted_label"]
        out.append(new)
    return out


def join_gold(rows: Sequence[Mapping[str, Any]], gold, manifest) -> list[dict[str, Any]]:
    from evaluation.scifact.metrics import retrieval_metrics_for_claim
    meta = {c["claim_id"]: c for c in manifest["claims"]}
    dataset_records = {r.runtime.claim_id: r for r in load_dataset().claims}
    joined = []
    for row in rows:
        g = gold[int(row["claim_id"])]
        retrieval = retrieval_metrics_for_claim(dataset_records[int(row["claim_id"])].annotations,
                                                row.get("retrieved_chunks") or []) if row.get("retrieved_chunks") else {}
        joined.append({**row, "gold_label": g.label, "bucket": meta[row["claim_id"]]["bucket"],
                       "multi_sentence_rationale": meta[row["claim_id"]]["multi_sentence_rationale"],
                       "retrieval": retrieval})
    return joined


def mcnemar_exact(b_only: int, c_only: int) -> float | None:
    n = b_only + c_only
    if n == 0:
        return None
    k = min(b_only, c_only)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def paired(rows_a: Sequence[Mapping[str, Any]], rows_b: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    a = {r["claim_id"]: r for r in rows_a}
    b = {r["claim_id"]: r for r in rows_b}

    def ok(r):
        return r["status"] == "SUCCESS" and r.get("predicted_label") in (
            "SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE") and r.get("gold_label") in GOLD_LABELS
    common = [c for c in a if c in b and ok(a[c]) and ok(b[c])]
    both = sum(a[c]["predicted_label"] == a[c]["gold_label"] and b[c]["predicted_label"] == b[c]["gold_label"]
               for c in common)
    neither = sum(a[c]["predicted_label"] != a[c]["gold_label"] and b[c]["predicted_label"] != b[c]["gold_label"]
                  for c in common)
    a_only = sum(a[c]["predicted_label"] == a[c]["gold_label"] and b[c]["predicted_label"] != b[c]["gold_label"]
                 for c in common)
    b_only = sum(a[c]["predicted_label"] != a[c]["gold_label"] and b[c]["predicted_label"] == b[c]["gold_label"]
                 for c in common)
    changed = sorted(c for c in common if a[c]["predicted_label"] != b[c]["predicted_label"])
    return {"pairs": len(common), "both_correct": both, "both_wrong": neither,
            "reference_only_correct": a_only, "variant_only_correct": b_only,
            "net_gain_claims": b_only - a_only,
            "paired_accuracy_reference": (both + a_only) / len(common) if common else None,
            "paired_accuracy_variant": (both + b_only) / len(common) if common else None,
            "mcnemar_exact_p": mcnemar_exact(a_only, b_only),
            "label_changes": len(changed),
            "label_change_transitions": dict(Counter(
                f"{a[c]['predicted_label']}->{b[c]['predicted_label']} (gold {a[c]['gold_label']})"
                for c in changed)),
            "changed_claim_ids": changed}


def _group_acc(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = [r for r in rows if r["status"] == "SUCCESS" and r.get("gold_label") in GOLD_LABELS]
    correct = sum(r["predicted_label"] == r["gold_label"] for r in valid)
    sup = [r for r in valid if r["gold_label"] == "SUPPORTED"]
    con = [r for r in valid if r["gold_label"] == "CONTRADICTED"]
    return {"claims": len(rows), "valid": len(valid), "correct": correct,
            "accuracy": correct / len(valid) if valid else None,
            "supported_accuracy": sum(r["predicted_label"] == "SUPPORTED" for r in sup) / len(sup) if sup else None,
            "contradicted_accuracy": sum(r["predicted_label"] == "CONTRADICTED" for r in con) / len(con) if con else None,
            "insufficient_predictions": sum(r["predicted_label"] == "INSUFFICIENT_EVIDENCE" for r in valid)}


def summarize(rows: Sequence[Mapping[str, Any]], gold) -> dict[str, Any]:
    status = Counter(r["status"] for r in rows)
    latencies = [r["latency_seconds"] for r in rows if isinstance(r.get("latency_seconds"), (int, float))]
    success_lat = [r["latency_seconds"] for r in rows if r["status"] == "SUCCESS"]
    llm = [r["llm_calls"] for r in rows if isinstance(r.get("llm_calls"), int)]
    rr = [r["retrieval"].get("reciprocal_rank") for r in rows if r.get("retrieval", {}).get("retrieval_scoring_status") == "scored"]
    r1 = [r["retrieval"]["recall_at_k"].get("1") for r in rows
          if r.get("retrieval", {}).get("retrieval_scoring_status") == "scored"]
    cls = classification_metrics(rows)
    return {
        "attempts": len(rows), "status_counts": dict(status),
        "failure_rate": (len(rows) - status.get("SUCCESS", 0)) / len(rows) if rows else None,
        "failures": [{"claim_id": r["claim_id"], "status": r["status"], "error": (r.get("error") or "")[:300]}
                     for r in rows if r["status"] != "SUCCESS"],
        "classification": cls,
        "accuracy": cls["accuracy"]["value"], "macro_f1": cls["macro_f1"]["value"],
        "precision": {k: cls["per_class"][k]["precision"] for k in ("SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE")},
        "recall": {k: cls["per_class"][k]["recall"] for k in ("SUPPORTED", "CONTRADICTED")},
        "insufficient_rate": insufficient_rate(rows),
        "evidence_metrics": {"citation_document_micro": citation_micro(rows, gold),
                             "retrieval_recall_at_1_mean": mean(x for x in r1 if x is not None),
                             "retrieval_mrr_misses_zero": mean(0.0 if x is None else x for x in rr),
                             "note": "retrieval is replayed from the frozen run, so retrieval metrics are constant"},
        "judge_stance_counts": stance_counts(rows),
        "latency_seconds": {"mean_all": mean(latencies), "median_all": median(latencies),
                            "mean_success": mean(success_lat), "max": max(latencies) if latencies else None},
        "llm_calls": {"total": sum(llm), "mean_per_attempt": mean(llm)},
        "by_bucket": {b: _group_acc([r for r in rows if r["bucket"] == b]) for b in BUCKETS},
        "by_rationale_type": {"single_sentence": _group_acc([r for r in rows if not r["multi_sentence_rationale"]]),
                              "multi_sentence": _group_acc([r for r in rows if r["multi_sentence_rationale"]])},
    }


def score(variant_rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads((OUTPUT_DIR / "development_set_manifest.json").read_text(encoding="utf-8"))
    gold = gold_by_claim(load_dataset())
    rows = join_gold(variant_rows, gold, manifest)
    return rows, summarize(rows, gold)
