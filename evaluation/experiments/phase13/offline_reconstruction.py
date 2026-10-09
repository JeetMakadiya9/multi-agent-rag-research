"""Phase 13 offline error reconstruction from the frozen 900-attempt run.

No inference. Reads only frozen Phase 10F per-example records and SciFact DEV
annotations, and writes offline_error_reconstruction.{json,md}.

Buckets are defined from retrieval ranks (not predictions):
  A_top2         a gold rationale sentence is at retrieved rank 1-2 (judged by B v1)
  B_rank3_8      first gold rationale sentence is at rank 3-8 (retrieved, never judged)
  C_absent_top8  no gold rationale sentence among the <=8 retrieved chunks
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

from evaluation.experiments.phase13.common import (
    BUCKETS, GOLD_LABELS, OUTPUT_DIR, chunk_keys, complete_rationale_within, first_gold_rationale_rank,
    gold_by_claim, gold_doc_rank, load_dataset, load_frozen_rows, rationale_bucket, write_json, write_text,
)


def _acc(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    correct = sum(r["correct"] for r in rows)
    sup = [r for r in rows if r["gold"] == "SUPPORTED"]
    con = [r for r in rows if r["gold"] == "CONTRADICTED"]
    pred = Counter(r["pred"] for r in rows)
    return {
        "claims": n, "correct": correct, "errors": n - correct,
        "accuracy": correct / n if n else None,
        "supported_claims": len(sup),
        "supported_accuracy": sum(r["correct"] for r in sup) / len(sup) if sup else None,
        "contradicted_claims": len(con),
        "contradicted_accuracy": sum(r["correct"] for r in con) / len(con) if con else None,
        "predicted_label_counts": dict(pred),
    }


def _verdict_cause(verification: Mapping[str, Any], stances: Sequence[str]) -> str:
    """Deterministic explanation of why a verdict was INSUFFICIENT_EVIDENCE."""
    errors = [str(e) for e in verification.get("validation_errors") or verification.get("validator_overrides") or []]
    guard = any(e.startswith("Uncovered required detail") for e in errors)
    if verification.get("conflicting_components"):
        return "conflicting_stances"
    if verification.get("missing_assessments"):
        return "missing_or_invalid_assessment"
    if guard and "SUPPORTS" in stances and "CONTRADICTS" not in stances:
        return "guard_overrode_supports"
    if guard:
        return "guard_fired_no_supports"
    if stances and all(s == "NEUTRAL" for s in stances):
        return "all_neutral"
    if "SUPPORTS" in stances:
        return "partial_component_support"
    return "other"


def _rows_for(system: str, frozen: Sequence[Mapping[str, Any]], gold) -> tuple[list[dict], list[dict]]:
    valid, failed = [], []
    for row in frozen:
        if row["system"] != system or row.get("gold_label") not in GOLD_LABELS:
            continue
        g = gold[int(row["claim_id"])]
        chunks = row.get("retrieved_chunks") or []
        rank = first_gold_rationale_rank(g, chunks) if chunks else None
        base = {
            "claim_id": int(row["claim_id"]), "gold": row["gold_label"],
            "bucket": rationale_bucket(rank) if chunks else "retrieval_unavailable",
            "first_gold_rationale_rank": rank,
            "first_gold_doc_rank": gold_doc_rank(g, chunks) if chunks else None,
            "multi_sentence_rationale": g.multi_sentence,
            "complete_rationale_top2": complete_rationale_within(g, chunks, 2) if chunks else False,
            "complete_rationale_top8": complete_rationale_within(g, chunks, 8) if chunks else False,
        }
        if row["status"] != "SUCCESS" or row.get("predicted_label") not in ("SUPPORTED", "CONTRADICTED",
                                                                            "INSUFFICIENT_EVIDENCE"):
            failed.append({**base, "status": row["status"], "error_type": row.get("error_type"),
                           "error_stage": row.get("error_stage")})
            continue
        assessments = row["output"].get("assessments") or []
        if system == "B":
            verification = row["output"]["raw_output"]["verification"]
            stance_rows = assessments
        else:
            verification = assessments[0] if assessments else {}
            stance_rows = verification.get("evidence_assessments") or []
        stances = [str(a.get("stance")) for a in stance_rows]
        # Stance assigned to judged chunks that contain a gold rationale sentence.
        # B/C judge pairs in evidence-rank order (E1..En x C1..Cm); one component
        # is by far the common case. Multi-component units are flagged.
        n_judged = len({a.get("evidence_index") for a in stance_rows} - {None}) if system == "B" else len(stance_rows)
        gold_stances = []
        if system == "B":
            by_index = defaultdict(list)
            for a in stance_rows:
                by_index[a.get("evidence_index")].append(a.get("stance"))
            for chunk in chunks:
                if chunk_keys(chunk) & g.sentences and f"E{chunk['rank']}" in by_index:
                    gold_stances.extend(by_index[f"E{chunk['rank']}"])
        else:
            for chunk, a in zip(chunks, stance_rows):
                if chunk_keys(chunk) & g.sentences:
                    gold_stances.append(a.get("stance"))
        pred = row["predicted_label"]
        valid.append({**base, "pred": pred, "correct": pred == row["gold_label"],
                      "stances": stances, "judged_evidence_count": n_judged,
                      "stances_on_judged_gold_sentences": gold_stances,
                      "insufficient_cause": _verdict_cause(verification, stances)
                      if pred == "INSUFFICIENT_EVIDENCE" else None,
                      "guard_messages": [e for e in (verification.get("validation_errors")
                                                     or verification.get("validator_overrides") or [])
                                         if str(e).startswith("Uncovered required detail")]})
    return valid, failed


def _system_report(valid: list[dict], failed: list[dict]) -> dict[str, Any]:
    report: dict[str, Any] = {"overall": _acc(valid)}
    report["by_bucket"] = {b: _acc([r for r in valid if r["bucket"] == b]) for b in BUCKETS}
    report["by_rationale_type"] = {
        name: _acc([r for r in valid if r["multi_sentence_rationale"] == flag])
        for name, flag in (("single_sentence_rationale", False), ("multi_sentence_rationale", True))}
    report["by_bucket_and_rationale_type"] = {
        f"{b}|{name}": _acc([r for r in valid if r["bucket"] == b and r["multi_sentence_rationale"] == flag])
        for b in BUCKETS for name, flag in (("single", False), ("multi", True))}
    report["complete_rationale"] = {
        "complete_in_top2": _acc([r for r in valid if r["complete_rationale_top2"]]),
        "partial_or_none_in_top2_but_complete_in_top8": _acc(
            [r for r in valid if not r["complete_rationale_top2"] and r["complete_rationale_top8"]]),
        "incomplete_in_top8": _acc([r for r in valid if not r["complete_rationale_top8"]]),
    }
    errors = [r for r in valid if not r["correct"]]
    report["error_bucket_counts"] = dict(Counter(r["bucket"] for r in errors))
    report["error_transition_by_bucket"] = {
        f"{r_gold}->{r_pred}|{b}": n for (r_gold, r_pred, b), n in sorted(
            Counter((r["gold"], r["pred"], r["bucket"]) for r in errors).items())}
    report["insufficient_error_causes"] = dict(Counter(r["insufficient_cause"] for r in errors
                                                       if r["pred"] == "INSUFFICIENT_EVIDENCE"))
    report["insufficient_causes_all_valid"] = dict(Counter(r["insufficient_cause"] for r in valid
                                                           if r["pred"] == "INSUFFICIENT_EVIDENCE"))
    report["judge_stance_counts_valid_labeled"] = dict(Counter(s for r in valid for s in r["stances"]))
    report["stance_on_judged_gold_sentences"] = {
        f"{gold}->{pred}|{stance}": n for (gold, pred, stance), n in sorted(Counter(
            (r["gold"], r["pred"], s) for r in valid for s in r["stances_on_judged_gold_sentences"]).items())}
    report["guard_fired_claims"] = {
        "all_valid": sum(bool(r["guard_messages"]) for r in valid),
        "errors": sum(bool(r["guard_messages"]) for r in errors),
        "correct": sum(bool(r["guard_messages"]) for r in valid if r["correct"]),
        "error_examples": [{"claim_id": r["claim_id"], "gold": r["gold"], "pred": r["pred"],
                            "stances": r["stances"], "guards": r["guard_messages"]}
                           for r in errors if r["guard_messages"]],
    }
    report["multi_component_claims"] = sum(r["judged_evidence_count"] < len(r["stances"]) for r in valid)
    report["failed_labeled"] = {"count": len(failed),
                                "by_bucket": dict(Counter(r["bucket"] for r in failed)),
                                "by_error_type": dict(Counter(str(r["error_type"]) for r in failed)),
                                "claim_ids": sorted(r["claim_id"] for r in failed)}
    return report


def build() -> dict[str, Any]:
    gold = gold_by_claim(load_dataset())
    frozen = load_frozen_rows()
    result: dict[str, Any] = {
        "phase": "13", "artifact": "offline_error_reconstruction", "inference_performed": False,
        "source": "evaluation/experiments/results/phase10_full_dev/per_example.jsonl (read-only)",
        "bucket_definition": __doc__.split("Buckets are defined", 1)[1].strip(),
        "multi_sentence_definition": "claim has at least one gold rationale with >=2 sentence indices",
        "complete_rationale_definition": "every sentence of at least one gold rationale is within top-k retrieved chunks",
        "judged_evidence_note": "Frozen B v1 and C judge only ranks 1-2 (verification_evidence_limit=2).",
        "systems": {},
        "per_claim": {},
    }
    for system in ("B", "C"):
        valid, failed = _rows_for(system, frozen, gold)
        result["systems"][system] = _system_report(valid, failed)
        result["per_claim"][system] = valid + failed
    # Cross-check against the frozen Phase 11B / 12 headline numbers.
    b = result["systems"]["B"]["overall"]
    c = result["systems"]["C"]["overall"]
    result["reconciliation"] = {
        "B_valid_correct": [b["claims"], b["correct"]], "expected_B": [179, 87],
        "C_valid_correct": [c["claims"], c["correct"]], "expected_C": [176, 85],
        "B_errors_rationale_in_top8": result["systems"]["B"]["error_bucket_counts"].get("A_top2", 0)
        + result["systems"]["B"]["error_bucket_counts"].get("B_rank3_8", 0), "expected_phase12_B": 66,
        "C_errors_rationale_in_top8": result["systems"]["C"]["error_bucket_counts"].get("A_top2", 0)
        + result["systems"]["C"]["error_bucket_counts"].get("B_rank3_8", 0), "expected_phase12_C": 65,
    }
    result["reconciliation"]["matches"] = (
        result["reconciliation"]["B_valid_correct"] == [179, 87]
        and result["reconciliation"]["C_valid_correct"] == [176, 85]
        and result["reconciliation"]["B_errors_rationale_in_top8"] == 66
        and result["reconciliation"]["C_errors_rationale_in_top8"] == 65)
    return result


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _table(rows: Mapping[str, Mapping[str, Any]]) -> str:
    out = ["| Group | Claims | Correct | Errors | Accuracy | SUPPORTED acc (n) | CONTRADICTED acc (n) |",
           "|---|---|---|---|---|---|---|"]
    for name, m in rows.items():
        out.append(f"| {name} | {m['claims']} | {m['correct']} | {m['errors']} | {_pct(m['accuracy'])} | "
                   f"{_pct(m['supported_accuracy'])} ({m['supported_claims']}) | "
                   f"{_pct(m['contradicted_accuracy'])} ({m['contradicted_claims']}) |")
    return "\n".join(out)


def render(result: Mapping[str, Any]) -> str:
    lines = ["# Phase 13 — Offline Error Reconstruction (no inference)", "",
             "Source: frozen Phase 10F per-example records (read-only) joined with SciFact DEV gold "
             "rationales after the fact. Buckets use retrieval ranks only, never predictions.", "",
             "- **A_top2**: a gold rationale sentence is at rank 1–2 (the only ranks the frozen verifier judged).",
             "- **B_rank3_8**: the first gold rationale sentence is at rank 3–8 (retrieved but never judged).",
             "- **C_absent_top8**: no gold rationale sentence in the ≤8 retrieved chunks.", "",
             f"Reconciliation with Phase 11B/12 headline counts: **{'MATCH' if result['reconciliation']['matches'] else 'MISMATCH'}** "
             f"(`{result['reconciliation']}`).", ""]
    for system in ("B", "C"):
        s = result["systems"][system]
        lines += [f"## System {system}", "", "### By rationale-rank bucket", "",
                  _table({"ALL": s["overall"], **s["by_bucket"]}), "",
                  "### By rationale type", "", _table(s["by_rationale_type"]), "",
                  "### Bucket × rationale type", "", _table(s["by_bucket_and_rationale_type"]), "",
                  "### Complete rationale availability", "", _table(s["complete_rationale"]), "",
                  f"- Error bucket counts: `{s['error_bucket_counts']}`",
                  f"- Error transitions (gold->pred|bucket): `{s['error_transition_by_bucket']}`",
                  f"- INSUFFICIENT_EVIDENCE error causes: `{s['insufficient_error_causes']}`",
                  f"- INSUFFICIENT_EVIDENCE causes over all valid labeled outputs: `{s['insufficient_causes_all_valid']}`",
                  f"- Judge stance counts (valid labeled): `{s['judge_stance_counts_valid_labeled']}`",
                  f"- Stance on judged gold-rationale chunks (gold->pred|stance): `{s['stance_on_judged_gold_sentences']}`",
                  f"- Claims where a lexical guard fired: all={s['guard_fired_claims']['all_valid']}, "
                  f"errors={s['guard_fired_claims']['errors']}, correct={s['guard_fired_claims']['correct']}",
                  f"- Failed labeled attempts (excluded from accuracy, still visible): {s['failed_labeled']['count']} "
                  f"`{s['failed_labeled']['by_error_type']}` by bucket `{s['failed_labeled']['by_bucket']}`", ""]
    lines += ["## Interpretation limits", "",
              "- Bucket membership is an observational association, not a causal attribution. A gold sentence at "
              "rank 1–2 can still be insufficient alone (multi-sentence rationales), and a correct verdict in "
              "C_absent_top8 can come from non-annotated but valid evidence.",
              "- 'Guard' causes are deterministic re-derivations of the frozen validator messages.",
              "- C's per-pair stances are mapped to ranks by evaluation order (E1..En); C judged the same top-2 chunks.",
              "- These diagnostics motivate H1–H4 but do not establish them; the controlled experiments do.", ""]
    return "\n".join(lines)


def main() -> None:
    result = build()
    write_json(OUTPUT_DIR / "offline_error_reconstruction.json", result)
    write_text(OUTPUT_DIR / "offline_error_reconstruction.md", render(result))
    print("reconciliation:", result["reconciliation"])


if __name__ == "__main__":
    main()
