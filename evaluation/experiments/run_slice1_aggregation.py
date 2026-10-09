"""Slice 1 offline aggregation experiment (deterministic; no model, retrieval or network).

Paired comparison on the SAME reconstructed Slice 0 graph per record:
  CONTROL      LegacyAnySupportsPolicy (frozen Experiment B semantics, exact)
  PRIMARY      ConservativeComponentPolicy
  SENSITIVITY  IndependentCorroborationSensitivity (not selected as primary)

Gold metrics: ONLY the 69 Phase 13 development claims, from the Phase 13 runs
(B0/B1/B2/B4). The 119 reserved labeled DEV claims are never scored. Usage:
    python -m evaluation.experiments.run_slice1_aggregation
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evidence_core.adapters.experiment_b import AmbiguousRecord, NotReplayable, adapt_experiment_b_record  # noqa: E402
from evidence_core.aggregation_v2 import (  # noqa: E402
    ConservativeComponentPolicy, IndependentCorroborationSensitivity, SUPPORTED_C, UNVERIFIED_C, classify_failure,
)
from evidence_core.contracts import Stance, VerdictState  # noqa: E402
from evidence_core.graph import EvidenceGraph  # noqa: E402
from evidence_core.replay import network_forbidden, recorded_sources, replay_record, scifact_lookup  # noqa: E402

RESULTS = ROOT / "evaluation/experiments/results"
OUTPUT_DIR = RESULTS / "slice1_aggregation"
DEV_MANIFEST = RESULTS / "phase13_improvement/development_set_manifest.json"
GOLD_SOURCES = ("phase13_B0", "phase13_B1", "phase13_B2", "phase13_B4")
LEGACY_TO_RICH_IDENTITY = {"SUPPORTED": "SUPPORTED", "CONTRADICTED": "CONTRADICTED",
                           "INSUFFICIENT_EVIDENCE": "INSUFFICIENT_EVIDENCE"}


def legacy_label(root) -> str:
    return root.legacy.verdict.value if root.legacy.verdict else "FAILED"


def change_reason(legacy: str, primary: str, legacy_guard_fired: bool) -> str:
    """Deterministic explanation of a legacy -> primary change (legacy semantics are exact, so these are exhaustive)."""
    if LEGACY_TO_RICH_IDENTITY.get(legacy) == primary:
        return "UNCHANGED"
    if legacy == "FAILED":
        return "LEGACY_ATTEMPT_FAILED_ON_INVALID_PAIR__PRIMARY_DECIDES_FROM_VALID_JUDGEMENTS"
    if primary == "MIXED_EVIDENCE":
        return "CONFLICT_MADE_EXPLICIT"
    if legacy == "INSUFFICIENT_EVIDENCE" and primary == "SUPPORTED" and legacy_guard_fired:
        return "LEGACY_LEXICAL_GUARD_NOT_APPLIED"
    if legacy == "INSUFFICIENT_EVIDENCE" and primary == "PARTIALLY_SUPPORTED":
        return "PARTIAL_COMPONENT_SUPPORT_MADE_EXPLICIT"
    return "UNEXPLAINED"


def sensitivity_reason(primary: str, sensitivity: str) -> str:
    if primary == sensitivity:
        return "UNCHANGED_FROM_PRIMARY"
    if primary in ("SUPPORTED", "PARTIALLY_SUPPORTED"):
        return "SINGLE_INDEPENDENCE_GROUP_SUPPORT_NOT_CORROBORATED"
    return "UNEXPLAINED"


def transition_matrix(rows: Sequence[Mapping[str, Any]], a: str, b: str) -> dict[str, dict[str, int]]:
    matrix: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        matrix[row[a]][row[b]] += 1
    return {k: dict(sorted(v.items())) for k, v in sorted(matrix.items())}


# ------------------------------------------------------------------ gold metrics (dev 69 only)
def gold_metrics(pairs: Sequence[tuple[str, str]]) -> dict[str, Any]:
    """pairs = (gold in {SUPPORTED, CONTRADICTED}, predicted state). Non-S/C predictions are non-decisive (never correct)."""
    confusion: dict[str, Counter] = {g: Counter() for g in ("SUPPORTED", "CONTRADICTED")}
    for gold, pred in pairs:
        confusion[gold][pred] += 1
    out: dict[str, Any] = {"n": len(pairs), "confusion": {g: dict(sorted(c.items())) for g, c in confusion.items()},
                           "per_class": {}}
    f1s = []
    for label in ("SUPPORTED", "CONTRADICTED"):
        tp = confusion[label][label]
        predicted = sum(confusion[g][label] for g in confusion)
        gold_n = sum(confusion[label].values())
        p = tp / predicted if predicted else None
        r = tp / gold_n if gold_n else None
        f1 = (2 * p * r / (p + r)) if p and r else (0.0 if (p == 0 or r == 0) else None)
        out["per_class"][label] = {"precision": p, "recall": r, "f1": f1, "tp": tp, "predicted": predicted, "gold": gold_n}
        if gold_n:
            f1s.append(f1)
    correct = sum(confusion[g][g] for g in confusion)
    out["macro_f1"] = sum(f1s) / len(f1s) if f1s else None
    out["accuracy_secondary"] = correct / len(pairs) if pairs else None
    out["false_support"] = confusion["CONTRADICTED"]["SUPPORTED"]
    out["false_contradiction"] = confusion["SUPPORTED"]["CONTRADICTED"]
    out["non_decisive"] = sum(v for g in confusion for k, v in confusion[g].items() if k not in ("SUPPORTED", "CONTRADICTED"))
    return out


def run() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    lookup = scifact_lookup()
    primary, sensitivity = ConservativeComponentPolicy(), IndependentCorroborationSensitivity()
    rows, not_replayable, multi_unit, invalid_rows = [], [], [], []
    for source in recorded_sources():
        with open(source["path"], encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        for record in filter(source["filter"], records):
            claim_id = record["claim_id"]
            run_key = f"{source['name']}:{claim_id}"
            try:
                adapted = adapt_experiment_b_record(record, claim_key=f"scifact-dev:{claim_id}", run_key=run_key,
                                                    judge=source["judge"], lookup=lookup)
            except NotReplayable as exc:
                not_replayable.append({"source": source["name"], "claim_id": claim_id, "reason": str(exc)})
                continue
            except AmbiguousRecord as exc:
                raise SystemExit(f"BLOCKER: ambiguous record {run_key}: {exc}")
            graph = EvidenceGraph()
            graph.add_all(adapted.objects())
            graph.validate()
            legacy_root, legacy_units = replay_record(adapted, run_key)
            rec = adapted.recorded
            if legacy_root.legacy.run_status is not rec.run_status or legacy_root.legacy.verdict != rec.verdict:
                raise SystemExit(f"BLOCKER: Slice 0 legacy replay is not exact for {run_key}")
            args = (adapted.root_claim, adapted.unit_claims, adapted.component_claims, adapted.evidence,
                    adapted.judgements, graph)
            p = primary.decide(*args, run_key=run_key)
            s = sensitivity.decide(*args, run_key=run_key)
            guard = any(str(d).startswith("Uncovered required detail") for d in legacy_units[0].legacy.diagnostics)
            legacy = legacy_label(legacy_root)
            row = {"source": source["name"], "claim_id": claim_id, "legacy": legacy, "legacy_guard_fired": guard,
                   "primary": p.state.value, "sensitivity": s.state.value,
                   "primary_reason_codes": list(p.reason_codes), "sensitivity_reason_codes": list(s.reason_codes),
                   "change_reason": change_reason(legacy, p.state.value, guard),
                   "sensitivity_change_reason": sensitivity_reason(p.state.value, s.state.value),
                   "binary_changed": (legacy == "SUPPORTED") != (p.state is VerdictState.SUPPORTED),
                   "required_slots": list(p.required_slots), "units": len(adapted.unit_claims),
                   "primary_components": [c for c in p.to_dict()["components"]],
                   "sensitivity_component_states": [c.state for c in s.components]}
            rows.append(row)
            for comp in p.components:
                for jid, cat in zip(comp.invalid_judgement_ids, comp.invalid_categories):
                    invalid_rows.append({"source": source["name"], "claim_id": claim_id, "category": cat,
                                         "component_state": comp.state})
            if len(adapted.unit_claims) > 1:
                all_units = primary.decide(*args, run_key=run_key, all_units=True)
                multi_unit.append({"source": source["name"], "claim_id": claim_id, "legacy": legacy,
                                   "primary_first_unit": p.state.value, "diagnostic_all_units": all_units.state.value,
                                   "note": "diagnostic only; not used to choose the primary policy"})
    return _summarise(rows, not_replayable, multi_unit, invalid_rows), rows


def _summarise(rows, not_replayable, multi_unit, invalid_rows) -> dict[str, Any]:
    changed = [r for r in rows if r["change_reason"] != "UNCHANGED"]
    legacy_supported = [r for r in rows if r["legacy"] == "SUPPORTED"]
    primary_supported = [r for r in rows if r["primary"] == "SUPPORTED"]
    comps = [c for r in rows for c in r["primary_components"]]
    supported_comps = [c for c in comps if c["state"] == SUPPORTED_C]
    report: dict[str, Any] = {
        "artifact": "slice1_aggregation", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope_statement": ("Slice 1 does not evaluate slot-aware aggregation because the frozen judgements contain no "
                            "slot-level coverage annotations. Claim aggregation is component-level because slot-level "
                            "judgement coverage is absent from the frozen records."),
        "offline": "no model inference, no retrieval, no network (socket connections forbidden during the run)",
        "policies": {"control": "LegacyAnySupportsPolicy (Slice 0, unchanged)",
                     "primary": f"{ConservativeComponentPolicy.name}@{ConservativeComponentPolicy.version}",
                     "sensitivity": f"{IndependentCorroborationSensitivity.name} (>=2 independence groups) — "
                                    "sensitivity analysis, not selected as primary policy"},
        "records": {"replayable_evaluated_by_all_policies": len(rows), "not_replayable": not_replayable},
        "distributions": {"legacy": dict(sorted(Counter(r["legacy"] for r in rows).items())),
                          "primary": dict(sorted(Counter(r["primary"] for r in rows).items())),
                          "sensitivity": dict(sorted(Counter(r["sensitivity"] for r in rows).items()))},
        "changes_legacy_to_primary": {
            "changed": len(changed), "changed_pct": len(changed) / len(rows) if rows else None,
            "by_reason": dict(sorted(Counter(r["change_reason"] for r in changed).items())),
            "support_binary_flips": sum(r["binary_changed"] for r in rows),
            "legacy_supported_rejected_by_primary": sum(r["primary"] != "SUPPORTED" for r in legacy_supported),
            "non_supported_to_supported": sum(r["legacy"] != "SUPPORTED" and r["primary"] == "SUPPORTED" for r in rows),
            "contradicted_changes": sum((r["legacy"] == "CONTRADICTED") != (r["primary"] == "CONTRADICTED") for r in rows),
            "conflict_state_changes": sum(r["primary"] == "MIXED_EVIDENCE" for r in rows),
            "insufficient_state_changes": sum(r["legacy"] == "INSUFFICIENT_EVIDENCE" and r["primary"] != "INSUFFICIENT_EVIDENCE"
                                              for r in rows),
            "unexplained": sum(r["change_reason"] == "UNEXPLAINED" for r in rows)},
        "changes_legacy_to_sensitivity": {
            "legacy_supported_rejected": sum(r["sensitivity"] != "SUPPORTED" for r in legacy_supported),
            "primary_supported_rejected": sum(r["sensitivity"] != "SUPPORTED" for r in primary_supported),
            "by_reason_vs_primary": dict(sorted(Counter(r["sensitivity_change_reason"] for r in rows).items()))},
        "primary_supported_why_allowed": dict(Counter(
            "single independence group" if "SINGLE_INDEPENDENCE_GROUP_SUPPORT" in r["primary_reason_codes"]
            else "every supported component has >=2 independence groups" for r in primary_supported)),
        "component_statistics": {
            "components": len(comps), "states": dict(sorted(Counter(c["state"] for c in comps).items())),
            "judgement_incomplete_components": sum(c["judgement_incomplete"] for c in comps),
            "invalid_only_components": sum(c["state"] == UNVERIFIED_C for c in comps),
            "incomplete_with_valid_support": sum(c["judgement_incomplete"] and c["state"] == SUPPORTED_C for c in comps),
            "conflict_within_independence_group": sum(bool(c["groups_with_internal_conflict"]) for c in comps),
            "supported_components_by_supporting_groups": dict(sorted(Counter(
                min(len(c["supporting_groups"]), 3) for c in supported_comps).items())),
            "independence_note": ("independence key = Source.independence_group if present else source_id; SciFact "
                                  "sources carry no independence groups, so groups == documents here (UNKNOWN beyond "
                                  "document level)")},
        "claims_with_incomplete_judgements": sum("INVALID_JUDGEMENT_PRESENT" in r["primary_reason_codes"] for r in rows),
        "invalid_failure_taxonomy": {"total_invalid_judgements": len(invalid_rows),
                                     "by_category": dict(sorted(Counter(i["category"] for i in invalid_rows).items())),
                                     "by_component_state": dict(sorted(Counter(i["component_state"] for i in invalid_rows).items())),
                                     "note": "infrastructure categories are execution failures, not reasoning errors"},
        "slot_coverage": {"claims_with_required_slots": sum(bool(r["required_slots"]) for r in rows),
                          "slot_level_judgements": 0, "status": "NOT_ASSESSED"},
        "multi_unit_records": multi_unit,
        "nei_probe_descriptive": {k: dict(sorted(Counter(r[k] for r in rows if r["source"] == "phase13_nei_probe_B1").items()))
                                  for k in ("legacy", "primary", "sensitivity")},
    }
    report["gold_evaluation"] = _gold(rows)
    report["decisions_sha256"] = hashlib.sha256(json.dumps(rows, sort_keys=True).encode("utf-8")).hexdigest()
    return report


def _gold(rows) -> dict[str, Any]:
    from evaluation.experiments.phase13.common import gold_by_claim, load_dataset
    from evaluation.experiments.run_phase10f import _classification
    dev_ids = set(json.loads(DEV_MANIFEST.read_text(encoding="utf-8"))["claim_ids"])
    gold = gold_by_claim(load_dataset())
    result: dict[str, Any] = {"denominator_statement": "ONLY the 69 Phase 13 development claims (development evidence, "
                              "not generalisation evidence); the 119 reserved labeled DEV claims are not scored",
                              "per_run": {}}
    for source in GOLD_SOURCES:
        subset = [r for r in rows if r["source"] == source]
        if {r["claim_id"] for r in subset} - dev_ids:
            raise SystemExit(f"BLOCKER: {source} contains claims outside the 69-claim development set")
        labeled = [r for r in subset if gold[r["claim_id"]].label in ("SUPPORTED", "CONTRADICTED")]
        legacy_valid = [r for r in labeled if r["legacy"] != "FAILED"]
        per = {}
        for name, population in (("legacy_valid_denominator", legacy_valid), ("all_replayable_denominator", labeled)):
            per[name] = {policy: gold_metrics([(gold[r["claim_id"]].label, r[policy]) for r in population])
                         for policy in ("legacy", "primary", "sensitivity")}
        # Consistency: legacy on its valid denominator must equal the frozen Phase 10F definition.
        frozen = _classification([{"gold_label": gold[r["claim_id"]].label, "status": "SUCCESS",
                                   "predicted_label": r["legacy"]} for r in legacy_valid])
        mine = per["legacy_valid_denominator"]["legacy"]
        if abs((frozen["macro_f1"]["value"] or 0) - (mine["macro_f1"] or 0)) > 1e-12 or \
                frozen["accuracy"]["numerator"] != round(mine["accuracy_secondary"] * mine["n"]):
            raise SystemExit(f"BLOCKER: gold metric does not reproduce the frozen definition for {source}")
        per["frozen_definition_check"] = {"legacy_accuracy": f"{frozen['accuracy']['numerator']}/{frozen['accuracy']['denominator']}",
                                          "legacy_macro_f1": frozen["macro_f1"]["value"], "matches": True}
        per["labeled_claims"], per["legacy_failed_labeled"] = len(labeled), len(labeled) - len(legacy_valid)
        # Attribution: for each changed decision, did it move from wrong->right or right->wrong against gold?
        def correct(pred, g):
            return pred == g
        per["change_attribution_vs_gold"] = dict(sorted(Counter(
            f"{r['change_reason']}|{'right' if correct(r['legacy'], gold[r['claim_id']].label) else 'wrong'}->"
            f"{'right' if correct(r['primary'], gold[r['claim_id']].label) else 'wrong'}|gold={gold[r['claim_id']].label}"
            for r in labeled if r["change_reason"] != "UNCHANGED").items()))
        per["sensitivity_attribution_vs_gold"] = dict(sorted(Counter(
            f"{'right' if correct(r['primary'], gold[r['claim_id']].label) else 'wrong'}->"
            f"{'right' if correct(r['sensitivity'], gold[r['claim_id']].label) else 'wrong'}|gold={gold[r['claim_id']].label}"
            for r in labeled if r["primary"] != r["sensitivity"]).items()))
        result["per_run"][source] = per
    return result


def render(report: Mapping[str, Any]) -> str:
    def pct(x):
        return "n/a" if x is None else f"{100 * x:.1f}%"
    c = report["changes_legacy_to_primary"]
    lines = ["# Slice 1 — Offline Aggregation Experiment", "", f"> {report['scope_statement']}", "",
             f"Offline: {report['offline']}.", "", "## Policies", ""]
    lines += [f"- **{k}**: {v}" for k, v in report["policies"].items()]
    lines += ["", f"Records evaluated by all three policies: **{report['records']['replayable_evaluated_by_all_policies']}** "
              f"(not replayable: {len(report['records']['not_replayable'])}).", "", "## Distributions", "",
              "| Policy | Distribution |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in report["distributions"].items()]
    lines += ["", "## Legacy → primary", "", f"- changed: {c['changed']} ({pct(c['changed_pct'])}); support-binary flips: {c['support_binary_flips']}",
              f"- by reason: `{c['by_reason']}`",
              f"- legacy SUPPORTED rejected by primary: {c['legacy_supported_rejected_by_primary']}; non-SUPPORTED → SUPPORTED: {c['non_supported_to_supported']}",
              f"- unexplained changes: {c['unexplained']}", "", "## Legacy → sensitivity (not selected)", "",
              f"`{report['changes_legacy_to_sensitivity']}`", "", "## Components, independence, INVALID", "",
              f"`{report['component_statistics']}`", "", f"INVALID taxonomy: `{report['invalid_failure_taxonomy']}`", "",
              f"Multi-unit records: `{report['multi_unit_records']}`", "", f"NEI probe (descriptive, no gold): `{report['nei_probe_descriptive']}`",
              "", "## Gold evaluation (69 development claims only)", "", report["gold_evaluation"]["denominator_statement"], "",
              "| Run | Denominator | Policy | n | Macro-F1 | SUP P / R | CON P / R | False support | Non-decisive | Accuracy (secondary) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for run_name, per in report["gold_evaluation"]["per_run"].items():
        for denom in ("legacy_valid_denominator", "all_replayable_denominator"):
            for policy, m in per[denom].items():
                s, k = m["per_class"]["SUPPORTED"], m["per_class"]["CONTRADICTED"]
                lines.append(f"| {run_name} | {denom.split('_')[0]} | {policy} | {m['n']} | "
                             f"{'n/a' if m['macro_f1'] is None else f'{m['macro_f1']:.3f}'} | {pct(s['precision'])} / {pct(s['recall'])} | "
                             f"{pct(k['precision'])} / {pct(k['recall'])} | {m['false_support']} | {m['non_decisive']} | {pct(m['accuracy_secondary'])} |")
    lines += ["", f"Decisions digest: `{report['decisions_sha256']}`", ""]
    return "\n".join(lines)


def main() -> None:
    with network_forbidden():
        report, rows = run()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "slice1_report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    (OUTPUT_DIR / "slice1_report.md").write_text(render(report), encoding="utf-8")
    (OUTPUT_DIR / "transition_matrix_primary.json").write_text(
        json.dumps(transition_matrix(rows, "legacy", "primary"), indent=1) + "\n", encoding="utf-8")
    (OUTPUT_DIR / "transition_matrix_sensitivity.json").write_text(
        json.dumps(transition_matrix(rows, "legacy", "sensitivity"), indent=1) + "\n", encoding="utf-8")
    with open(OUTPUT_DIR / "decision_diagnostics.jsonl", "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in ("distributions", "changes_legacy_to_primary", "changes_legacy_to_sensitivity",
                                              "invalid_failure_taxonomy")}, indent=1))


if __name__ == "__main__":
    main()
