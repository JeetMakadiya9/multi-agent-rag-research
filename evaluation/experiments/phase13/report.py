"""Build Phase 13 experiment reports from recorded runs (no inference)."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from typing import Any, Mapping, Sequence

from evaluation.experiments.phase13 import b_plus
from evaluation.experiments.phase13.common import (
    GOLD_LABELS, OUTPUT_DIR, gold_by_claim, load_dataset, load_frozen_rows, write_json, write_text,
)
from evaluation.experiments.phase13.scoring import apply_guard_v2, load_attempts, paired, score

PROMPT_HASH = {"v1": hashlib.sha256(b_plus.PROMPT_V1_SYSTEM.encode()).hexdigest(),
               "v2": hashlib.sha256(b_plus.PROMPT_V2_SYSTEM.encode()).hexdigest()}
UNCHANGED_ALWAYS = ["SciFact DEV data and gold labels", "frozen Phase 10F retrieval (replayed chunks, ranks, scores)",
                    "embedding model / BM25 / RRF / reranker / MMR (not re-executed; replayed)",
                    "claim decomposition (frozen verification_units)", "frozen run_system_b adapter and label mapping",
                    "Experiment B engine, pairwise task, JSON schema, max_judge_retries=1",
                    "model qwen3:4b (digest 359d7dd4...), CPU num_gpu=0, num_predict=256, temperature 0, think=False",
                    "metric definitions (frozen Phase 10F _classification)"]

EXPERIMENTS: dict[str, dict[str, Any]] = {
    "B1": {"hypothesis": "H1: the 2-evidence cap materially suppresses verification accuracy.",
           "reference": "B0", "changed_component": "verification_evidence_limit 2 -> 8 (all retrieved chunks)",
           "prompt": "v1", "evidence_limit": 8, "context_window": 0, "guards": "v1"},
    "B2": {"hypothesis": "H2: evidence packaging / loss of abstract context suppresses accuracy.",
           "reference": "B1", "changed_component": "evidence text = retrieved sentence + same-abstract neighbours "
                                                    "(window 1), clearly marked; limit stays 8",
           "prompt": "v1", "evidence_limit": 8, "context_window": 1, "guards": "v1"},
    "B3": {"hypothesis": "H3: deterministic qualifier/coordinated-object guards create false "
                         "INSUFFICIENT_EVIDENCE decisions on biomedical language.",
           "reference": "B0", "changed_component": "lexical guards v1 -> v2 (rules G1-G5); applied by exact "
                                                    "deterministic re-aggregation of B0's recorded stances",
           "prompt": "v1", "evidence_limit": 2, "context_window": 0, "guards": "v2", "source_run": "B0"},
    "B4": {"hypothesis": "H4: the Qwen judge systematically under-detects contradiction.",
           "reference": "B0", "changed_component": "pairwise judge SYSTEM message v1 -> v2 (explicit "
                                                    "SUPPORTS/CONTRADICTS/NEUTRAL definitions); user payload unchanged",
           "prompt": "v2", "evidence_limit": 2, "context_window": 0, "guards": "v1"},
}


def _load_b5_spec() -> dict[str, Any] | None:
    path = OUTPUT_DIR / "b5_definition.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def variant_rows(name: str) -> list[dict[str, Any]]:
    if name == "B3":
        return apply_guard_v2(load_attempts("B0"))
    if name == "B5":
        spec = _load_b5_spec()
        rows = load_attempts("B5")
        return apply_guard_v2(rows) if spec and spec.get("guards") == "v2" else rows
    return load_attempts(name)


def _complete(rows: Sequence[Any], n: int) -> bool:
    return len(rows) == n


def _fmt(x: Any, pct: bool = True) -> str:
    if x is None:
        return "n/a"
    return f"{100 * x:.1f}%" if pct else f"{x:.4f}"


def experiment_record(name: str, spec: Mapping[str, Any], scored: Mapping[str, Any], rows, ref_rows,
                      manifest) -> dict[str, Any]:
    cls = scored["classification"]
    return {
        "experiment": name, "hypothesis": spec["hypothesis"],
        "baseline_version": "B v1 (frozen Phase 10F System B), re-run live on the dev subset as B0",
        "reference_variant": spec["reference"], "changed_component": spec["changed_component"],
        "unchanged_components": UNCHANGED_ALWAYS + [k for k in (
            "evidence limit" if spec["evidence_limit"] == (2 if spec["reference"] == "B0" else 8) else None,
            "evidence packaging" if spec["context_window"] == 0 or spec["reference"] == "B2" else None,
            "lexical guards v1" if spec["guards"] == "v1" else None,
            "judge prompt v1" if spec["prompt"] == "v1" else None) if k],
        "dataset": "SciFact DEV, Phase 13 development subset (engineering only; not a benchmark)",
        "claim_ids": manifest["claim_ids"], "model": "qwen3:4b (Ollama, digest 359d7dd4bcda...)",
        "prompt": {"version": spec["prompt"], "system_prompt_sha256": PROMPT_HASH[spec["prompt"]]},
        "evidence_limit": spec["evidence_limit"], "context_window": spec["context_window"],
        "retrieval_settings": "frozen Phase 10F hybrid retrieval replayed (FINAL_TOP_K=8, MiniLM-L6 + BM25 + exact, "
                              "RRF k=60, ms-marco MiniLM cross-encoder, MMR 0.72); claim text as query",
        "validator_settings": {"guards": spec["guards"], "decision_ladder": "frozen validate_judgement",
                               "max_judge_retries": 1},
        "random_seed": {"development_subset": 2026,
                        "decoding": "not applicable: greedy decoding (temperature 0); Ollama seed unset as in frozen B"},
        "accuracy": cls["accuracy"], "macro_f1": cls["macro_f1"],
        "per_class_precision": {k: v["precision"] for k, v in cls["per_class"].items()},
        "per_class_recall": {k: v["recall"] for k, v in cls["per_class"].items()},
        "confusion_matrix": cls["confusion_matrix"], "insufficient_rate": scored["insufficient_rate"],
        "evidence_metrics": scored["evidence_metrics"], "latency_seconds": scored["latency_seconds"],
        "llm_calls": scored["llm_calls"], "failure_rate": scored["failure_rate"],
        "status_counts": scored["status_counts"], "failures": scored["failures"],
        "judge_stance_counts": scored["judge_stance_counts"],
        "by_bucket": scored["by_bucket"], "by_rationale_type": scored["by_rationale_type"],
        "paired_vs_reference": paired(ref_rows, rows) if ref_rows else None,
        "paired_vs_B0": paired(variant_cache["B0"][0], rows) if "B0" in variant_cache else None,
        "full_classification": cls,
    }


variant_cache: dict[str, tuple[list[dict[str, Any]], dict[str, Any]]] = {}


def render_experiment(rec: Mapping[str, Any]) -> str:
    acc = rec["accuracy"]
    pr = rec["paired_vs_reference"] or {}
    p0 = rec["paired_vs_B0"] or {}
    cm = rec["confusion_matrix"]
    lines = [f"# Phase 13 — Experiment {rec['experiment']}", "",
             f"**Hypothesis.** {rec['hypothesis']}", "",
             f"- Baseline: {rec['baseline_version']}; reference for this single-factor comparison: **{rec['reference_variant']}**",
             f"- Changed component (only): {rec['changed_component']}",
             f"- Evidence limit {rec['evidence_limit']}, context window {rec['context_window']}, prompt "
             f"{rec['prompt']['version']} (`{rec['prompt']['system_prompt_sha256'][:12]}`), guards "
             f"{rec['validator_settings']['guards']}",
             f"- Dataset: {rec['dataset']} — {len(rec['claim_ids'])} claims", "",
             "## Results (development subset only)", "",
             "| Metric | Value |", "|---|---|",
             f"| Accuracy | {acc['numerator']}/{acc['denominator']} = {_fmt(acc['value'])} |",
             f"| Macro F1 (2 gold classes) | {_fmt(rec['macro_f1']['value'], False)} |",
             f"| SUPPORTED P / R | {_fmt(rec['per_class_precision']['SUPPORTED'])} / {_fmt(rec['per_class_recall']['SUPPORTED'])} |",
             f"| CONTRADICTED P / R | {_fmt(rec['per_class_precision']['CONTRADICTED'])} / {_fmt(rec['per_class_recall']['CONTRADICTED'])} |",
             f"| INSUFFICIENT rate (valid labeled) | {rec['insufficient_rate']['numerator']}/{rec['insufficient_rate']['denominator']} = {_fmt(rec['insufficient_rate']['value'])} |",
             f"| Failure rate | {_fmt(rec['failure_rate'])} `{rec['status_counts']}` |",
             f"| LLM calls total / mean | {rec['llm_calls']['total']} / {rec['llm_calls']['mean_per_attempt']:.2f} |",
             f"| Latency mean / median (s) | {rec['latency_seconds']['mean_all']:.1f} / {rec['latency_seconds']['median_all']:.1f} |",
             f"| Citation doc micro P / R / F1 | {_fmt(rec['evidence_metrics']['citation_document_micro']['precision'])} / "
             f"{_fmt(rec['evidence_metrics']['citation_document_micro']['recall'])} / "
             f"{_fmt(rec['evidence_metrics']['citation_document_micro']['f1'], False)} |",
             f"| Judge stances | `{rec['judge_stance_counts']}` |", "",
             "Confusion matrix (rows gold, columns predicted SUP / CON / NEI):", "",
             f"- SUPPORTED: {cm['SUPPORTED']['SUPPORTED']} / {cm['SUPPORTED']['CONTRADICTED']} / {cm['SUPPORTED']['INSUFFICIENT_EVIDENCE']}",
             f"- CONTRADICTED: {cm['CONTRADICTED']['SUPPORTED']} / {cm['CONTRADICTED']['CONTRADICTED']} / {cm['CONTRADICTED']['INSUFFICIENT_EVIDENCE']}", "",
             f"## Paired comparison vs {rec['reference_variant']}", "",
             f"`{ {k: v for k, v in pr.items() if k != 'changed_claim_ids'} }`", "",
             f"Changed claim IDs: {pr.get('changed_claim_ids')}", ""]
    if rec["reference_variant"] != "B0":
        lines += ["## Paired comparison vs B0", "", f"`{ {k: v for k, v in p0.items() if k != 'changed_claim_ids'} }`", ""]
    lines += ["## By retrieval bucket / rationale type", "", "| Group | Valid | Correct | Accuracy | SUP acc | CON acc | NEI preds |",
              "|---|---|---|---|---|---|---|"]
    for name, g in list(rec["by_bucket"].items()) + list(rec["by_rationale_type"].items()):
        lines.append(f"| {name} | {g['valid']} | {g['correct']} | {_fmt(g['accuracy'])} | {_fmt(g['supported_accuracy'])} | "
                     f"{_fmt(g['contradicted_accuracy'])} | {g['insufficient_predictions']} |")
    if rec["failures"]:
        lines += ["", "## Failures (kept visible, excluded from accuracy denominators)", ""]
        lines += [f"- claim {f['claim_id']}: {f['status']} — {f['error']}" for f in rec["failures"]]
    lines += ["", "Development-subset results inform engineering decisions only; the subset is stratified and is "
                  "not comparable to the 179-claim full-DEV accuracy and is not a benchmark.", ""]
    return "\n".join(lines)


def guard_safety_on_frozen() -> dict[str, Any]:
    """Apply guard v2 to ALL 280 frozen B outputs (incl. 101 unlabeled) — permissiveness check."""
    gold = gold_by_claim(load_dataset())
    rows = []
    for r in load_frozen_rows():
        if r["system"] != "B" or r["status"] != "SUCCESS":
            continue
        out = r["output"]
        n = len({a["evidence_index"] for a in out["assessments"]})
        texts = [c["text"] for c in r["retrieved_chunks"][:n]]
        v2 = b_plus.reaggregate_output(r["runtime_input"]["claim"], out["assessments"], texts, guard_version="v2")
        v1 = b_plus.reaggregate_output(r["runtime_input"]["claim"], out["assessments"], texts, guard_version="v1")
        rows.append({"claim_id": r["claim_id"], "gold": gold[r["claim_id"]].label, "v1": v1["verdict"],
                     "v2": v2["verdict"], "v1_guards": v1["guard_messages"], "v2_guards": v2["guard_messages"],
                     "claim": r["runtime_input"]["claim"], "stances": [a["stance"] for a in out["assessments"]]})
    changed = [r for r in rows if r["v1"] != r["v2"]]
    unlabeled_changes = [r for r in changed if r["gold"] is None]
    labeled = [r for r in rows if r["gold"] in GOLD_LABELS]
    return {
        "scope": "all 280 successful frozen Phase 10F B outputs (179 labeled + 101 unlabeled); no inference",
        "claims_where_v1_guard_fired": sum(bool(r["v1_guards"]) for r in rows),
        "claims_where_v2_guard_fires": sum(bool(r["v2_guards"]) for r in rows),
        "verdict_changes": len(changed),
        "verdict_change_transitions": dict(Counter(f"{r['v1']}->{r['v2']} (gold {r['gold']})" for r in changed)),
        "labeled_accuracy_v1": sum(r["v1"] == r["gold"] for r in labeled) / len(labeled),
        "labeled_accuracy_v2": sum(r["v2"] == r["gold"] for r in labeled) / len(labeled),
        "labeled_correct_v1_v2": [sum(r["v1"] == r["gold"] for r in labeled), sum(r["v2"] == r["gold"] for r in labeled),
                                  len(labeled)],
        "unlabeled_new_supported_or_contradicted": len([r for r in unlabeled_changes if r["v2"] != "INSUFFICIENT_EVIDENCE"]),
        "unlabeled_changes": unlabeled_changes,
        "changed_rows": changed,
        "still_firing_v2": [{"claim_id": r["claim_id"], "gold": r["gold"], "guards": r["v2_guards"]}
                            for r in rows if r["v2_guards"]],
        "interpretation": "On labeled claims a guard can only move a verdict to INSUFFICIENT_EVIDENCE, which is always "
                          "scored wrong; therefore labeled accuracy alone cannot show guard harm. Unlabeled (SciFact "
                          "NOT-ENOUGH-INFO) claims are where an over-permissive guard would create false SUPPORTED/"
                          "CONTRADICTED verdicts.",
    }


def reproducibility_b0() -> dict[str, Any]:
    frozen = {r["claim_id"]: r for r in load_frozen_rows() if r["system"] == "B"}
    b0 = load_attempts("B0")
    agree = disagree = 0
    diffs = []
    stance_same = 0
    for r in b0:
        f = frozen.get(r["claim_id"])
        if not f or f["status"] != "SUCCESS" or r["status"] != "SUCCESS":
            continue
        if f["predicted_label"] == r["predicted_label"]:
            agree += 1
        else:
            disagree += 1
            diffs.append({"claim_id": r["claim_id"], "frozen": f["predicted_label"], "rerun": r["predicted_label"]})
        stance_same += [a["stance"] for a in f["output"]["assessments"]] == [a["stance"] for a in r["assessments"]]
    return {"compared": agree + disagree, "label_agreement": agree, "label_disagreement": disagree,
            "identical_stance_sequences": stance_same, "differences": diffs,
            "frozen_failed_rerun_success": sorted(r["claim_id"] for r in b0 if r["status"] == "SUCCESS"
                                                  and frozen.get(r["claim_id"], {}).get("status") != "SUCCESS")}


def main() -> None:
    manifest = json.loads((OUTPUT_DIR / "development_set_manifest.json").read_text(encoding="utf-8"))
    n = len(manifest["claim_ids"])
    names = ["B0", "B1", "B2", "B3", "B4"] + (["B5"] if _load_b5_spec() else [])
    for name in names:
        rows = variant_rows(name)
        if _complete(rows, n):
            variant_cache[name] = score(rows)
    table = []
    if "B0" in variant_cache:
        b0_rows, b0 = variant_cache["B0"]
        write_json(OUTPUT_DIR / "runs" / "B0" / "scored_summary.json", b0)
        table.append(("B0", "B v1 live re-run (control)", b0))
    specs = dict(EXPERIMENTS)
    if _load_b5_spec():
        specs["B5"] = _load_b5_spec()
    for name, spec in specs.items():
        if name not in variant_cache:
            continue
        rows, scored = variant_cache[name]
        ref_rows = variant_cache.get(spec["reference"], (None,))[0]
        rec = experiment_record(name, spec, scored, rows, ref_rows, manifest)
        if name == "B3":
            rec["guard_safety_full_frozen_B"] = guard_safety_on_frozen()
            rec["secondary_guard_v2_on_other_runs"] = {
                other: {"accuracy": score(apply_guard_v2(variant_cache[other][0]))[1]["accuracy"],
                        "accuracy_guard_v1": variant_cache[other][1]["accuracy"]}
                for other in ("B1", "B2", "B4") if other in variant_cache}
        write_json(OUTPUT_DIR / f"experiment_{name}.json", rec)
        write_text(OUTPUT_DIR / f"experiment_{name}.md", render_experiment(rec))
        table.append((name, spec["changed_component"], scored))
    cols = ["variant", "change", "valid", "correct", "accuracy", "macro_f1", "sup_precision", "sup_recall",
            "con_precision", "con_recall", "insufficient_rate", "failure_rate", "llm_calls_total",
            "llm_calls_mean", "latency_mean_s", "citation_doc_f1", "contradicts_stances", "neutral_stances",
            "supports_stances"]
    out = []
    for name, change, s in table:
        c = s["classification"]
        out.append({"variant": name, "change": change, "valid": c["accuracy"]["denominator"],
                    "correct": c["accuracy"]["numerator"], "accuracy": s["accuracy"], "macro_f1": s["macro_f1"],
                    "sup_precision": s["precision"]["SUPPORTED"], "sup_recall": s["recall"]["SUPPORTED"],
                    "con_precision": s["precision"]["CONTRADICTED"], "con_recall": s["recall"]["CONTRADICTED"],
                    "insufficient_rate": s["insufficient_rate"]["value"], "failure_rate": s["failure_rate"],
                    "llm_calls_total": s["llm_calls"]["total"], "llm_calls_mean": s["llm_calls"]["mean_per_attempt"],
                    "latency_mean_s": s["latency_seconds"]["mean_all"],
                    "citation_doc_f1": s["evidence_metrics"]["citation_document_micro"]["f1"],
                    "contradicts_stances": s["judge_stance_counts"].get("CONTRADICTS", 0),
                    "neutral_stances": s["judge_stance_counts"].get("NEUTRAL", 0),
                    "supports_stances": s["judge_stance_counts"].get("SUPPORTS", 0)})
    write_json(OUTPUT_DIR / "comparison_table.json", {
        "note": "Development subset (69 claims, stratified); engineering evidence only. B3 and the guard component "
                "of B5 are exact deterministic re-aggregations of recorded stances.",
        "frozen_full_dev_reference": {"B_accuracy": "87/179 = 48.6%", "B_macro_f1": 0.4842},
        "rows": out, "b0_reproducibility_vs_frozen": reproducibility_b0() if "B0" in variant_cache else None})
    with open(OUTPUT_DIR / "comparison_table.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols)
        writer.writeheader()
        writer.writerows(out)
    print(json.dumps([{k: r[k] for k in ("variant", "valid", "correct", "accuracy", "macro_f1", "con_recall",
                                         "insufficient_rate", "failure_rate")} for r in out], indent=1))


if __name__ == "__main__":
    main()
