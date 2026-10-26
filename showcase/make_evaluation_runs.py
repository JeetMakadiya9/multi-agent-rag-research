"""Build showcase/data/evaluation_runs.json: run-level aggregates of the Studio verifier's evaluation runs.

Offline maintenance script (not part of the site). It reads the private run folders (records, manifests, run.json,
guard logs) and writes aggregates only: no claim ids, claim text, passages, quotes or local paths. No model is called.

    python showcase/make_evaluation_runs.py --runs <folder with the run folders and sets/> [--out FILE]

What is computed, and from which field:
* predictive quality: ``score.summarise`` (three-class, the protocol metric) and ``historical_metric`` (two-class),
  from each record's ``gold`` and ``verdict``; only for runs that cover their whole claim set;
* efficiency: ``seconds`` (wall-clock per claim, as recorded) and ``calls`` (model calls per claim);
* reliability: ``sub_state``, ``error`` and ``parse_ok``; invocations from the manifests; memory stops from the guard
  logs;
* evidence coverage: ``id_coverage.measure`` and ``passage_diagnostics``;
* reproducibility: manifests, run.json and the claim-set hashes. Fields that were never recorded stay null.

Each run's phase, role and decision come from the plans and reports named in ``RUNS``; they are not inferred.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from evaluation.studio_eval import historical_metric, id_coverage, score  # noqa: E402
from evaluation.studio_eval.records import RecordIntegrityError, load_records, summary  # noqa: E402

OUT = Path(__file__).resolve().parent / "data" / "evaluation_runs.json"
# failure kinds by the code the record gives (the error text before ":", or the sub-state when there is none)
INFRASTRUCTURE = {"unreachable", "http_error", "HARNESS_ERROR"}         # model server down or broken, out of memory
MODEL_SIDE = {"model_error", "empty", "ASSESSMENT_FAILED"}              # the model's own output was unusable
# "timeout" is kept apart: a slow model and a sleeping or overloaded machine look the same in the record


def failure_kind(record) -> str:
    error = str(record.get("error") or "")
    code = error.split(":", 1)[0].strip() if ":" in error else ""
    if record.get("sub_state") == "HARNESS_ERROR":
        return "HARNESS_ERROR"
    return code or record.get("sub_state") or "unknown"
EVIDENCE = "fixed packets: the cited SciFact abstracts (not web evidence retrieved by the Studio)"

P1 = "evaluation/studio_eval/ALT_BENCHMARKS.md"
R1 = "evaluation/studio_eval/IMPROVEMENT_PLAN.md"
R2 = "evaluation/studio_eval/IMPROVEMENT_PLAN_2.md"
R3 = "evaluation/studio_eval/IMPROVEMENT_PLAN_3.md"
R4 = "evaluation/studio_eval/IMPROVEMENT_PLAN_4_DRAFT.md"

# key: (label, phase, role, plan, status, note). status: valid | rejected | invalid | incomplete | partial | coverage
RUNS = {
    "runs/retro_arm_c_studio": ("arm_c_studio (default)", "Proxy benchmarks", "development", P1, "valid",
                                "Default verifier on the retrospective development set."),
    "runs/retro_arm_c": ("arm_c (historical Arm C rules)", "Proxy benchmarks", "development", P1, "valid",
                         "Comparison baseline on the same packets."),
    "runs/prog_arm_c_studio": ("arm_c_studio (default)", "Proxy benchmarks", "programmatic cases", P1, "valid",
                               "Rule-defined labels (swap, negate); not comparable with the SciFact sets."),
    "runs/adv_arm_c_studio": ("arm_c_studio (default)", "Proxy benchmarks", "adversarial cases", P1, "valid",
                              "Injected-instruction packets; not comparable with the SciFact sets."),
    "runs/retro_arm_c_studio_repeat": ("arm_c_studio (default)", "Proxy benchmarks", "repeatability subset", P1,
                                       "partial", "30-claim repeat of the default; a repeatability check, not scored."),
    "baselines/retro_quote_stance": ("quote_stance", "Proxy benchmarks", "development", P1, "valid",
                                     "Earlier per-passage verifier on the same packets."),
    "baselines/retro_arm_c0": ("arm_c0", "Proxy benchmarks", "development", P1, "rejected",
                               "306 rows for 300 claims: duplicate records without run identity. Rejected by the "
                               "duplicate policy; superseded by the re-run with run identity."),
    "improve/dev_arm_c_studio_qr": ("arm_c_studio_qr (quote repair)", "Round 1", "development", R1, "valid",
                                    "Retained on development; not adopted after the held-out test."),
    "improve/sub90_arm_c_studio_qwen35": ("arm_c_studio with qwen3.5", "Round 1", "screening", R1, "valid",
                                          "Rejected: more false support, 3 failures, 2.7x runtime."),
    "improve/test_arm_c_studio": ("arm_c_studio (default)", "Round 1", "held-out test (used once)", R1, "valid",
                                  "SciFact DEV split, used once for a decision; no longer held out."),
    "improve/test_arm_c_studio_qr": ("arm_c_studio_qr (quote repair)", "Round 1", "held-out test (used once)", R1,
                                     "valid", "Not adopted: no gain on held-out data and more false support."),
    "improve2/sub90_arm_c_studio_p2": ("arm_c_studio_p2 (prompt P2)", "Round 2", "screening", R2, "valid",
                                       "Rejected at screening: more false support. Ran partly on the CPU (16k context)."),
    "diag/smoke_arm_c_studio_think": ("arm_c_studio_think (reasoning mode)", "Round 2", "smoke test", R2, "partial",
                                      "3-claim feasibility smoke; rejected as infeasible (600-700 s per claim)."),
    "baselines/retro_arm_c0_rerun2": ("arm_c0 (re-run)", "arm_c0 re-run", "development", P1, "incomplete",
                                      "Stopped by an out-of-memory crash of the machine; kept as evidence only."),
    "baselines/retro_arm_c0_rerun3": ("arm_c0 (re-run)", "arm_c0 re-run", "development", P1, "invalid",
                                      "Infrastructure failure: the model runtime was missing after a cancelled update, "
                                      "so every call failed (HTTP 500). No model behaviour was measured."),
    "baselines/retro_arm_c0_rerun4": ("arm_c0 (re-run)", "arm_c0 re-run", "development", P1, "incomplete",
                                      "Stopped by the memory guard: a 16k context pushed the model onto the CPU."),
    "baselines/retro_arm_c0_rerun5": ("arm_c0 (re-run)", "arm_c0 re-run", "development", P1, "valid",
                                      "Valid re-run with run identity; replaces the rejected original."),
    "round3/screen_arm_c_studio": ("arm_c_studio (default)", "Round 3", "screening", R3, "valid",
                                   "Same-session comparator for the round-3 screen."),
    "round3/screen_arm_c_studio_all": ("arm_c_studio_all", "Round 3", "screening", R3, "valid",
                                       "Passed the operational gate. A passage-ID defect left 59% of passages after "
                                       "P1 unjudged, so the intended treatment was only partly applied."),
    "round3/confirm_arm_c_studio": ("arm_c_studio (default)", "Round 3", "confirmation (blind, now used)", R3,
                                    "valid", "Comparator for the blind confirmation."),
    "round3/confirm_arm_c_studio_all": ("arm_c_studio_all", "Round 3", "confirmation (blind, now used)", R3, "valid",
                                        "75% hypothesis not supported. One claim timed out while the machine slept "
                                        "(scored as NOT_ENOUGH_INFO). The passage-ID defect left 64% of passages after "
                                        "P1 unjudged."),
    "coverage_idfix/retro10": ("arm_c_studio_all_idfix", "Passage-ID fix", "coverage check", R4, "coverage",
                               "Coverage-only check of the fix on the first 10 development claims. Accuracy was "
                               "deliberately not computed."),
}

# score files with a paired comparison: (file, system run, baseline run)
PAIRED = [
    ("scores/retro.json", "runs/retro_arm_c_studio", "runs/retro_arm_c"),
    ("baselines/score_studio_vs_quote_stance.json", "runs/retro_arm_c_studio", "baselines/retro_quote_stance"),
    ("baselines/score_studio_vs_arm_c0_rerun5.json", "runs/retro_arm_c_studio", "baselines/retro_arm_c0_rerun5"),
    ("improve/score_dev_qr.json", "improve/dev_arm_c_studio_qr", "runs/retro_arm_c_studio"),
    ("improve/score_test.json", "improve/test_arm_c_studio_qr", "improve/test_arm_c_studio"),
    ("improve/score_sub90_qwen35.json", "improve/sub90_arm_c_studio_qwen35", "runs/retro_arm_c_studio"),
    ("improve2/score_sub90_p2.json", "improve2/sub90_arm_c_studio_p2", "runs/retro_arm_c_studio"),
    ("round3/score_screen.json", "round3/screen_arm_c_studio_all", "round3/screen_arm_c_studio"),
    ("round3/score_confirm.json", "round3/confirm_arm_c_studio_all", "round3/confirm_arm_c_studio"),
]


def _jsonl(path: Path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def claim_sets(base: Path):
    sets = {}
    for folder in sorted((base / "sets").iterdir()):
        path = folder / "claims.jsonl"
        if path.is_file():
            rows = _jsonl(path)
            sets[hashlib.sha256(path.read_bytes()).hexdigest()] = {
                "name": folder.name, "ids": [r["claim_id"] for r in rows], "gold": {r["claim_id"]: r["label"] for r in rows}}
    return sets


def manifests(folder: Path):
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("manifest*.json"))]


def guard_log(base: Path, key: str):
    log = base / (key + ".log")
    if not log.is_file():
        return {"ollama_version": None, "memory_stops": None, "guard_stops": None}
    text = log.read_text(encoding="utf-8", errors="replace")
    versions = sorted(set(re.findall(r'Ollama \{"version":"([0-9.]+)"\}', text)))
    return {"ollama_version": versions[0] if len(versions) == 1 else (versions or None),
            "memory_stops": len(re.findall(r"GUARD[^\n]*free memory fell below", text)),
            "guard_stops": len(re.findall(r"GUARD", text))}


def integrity(path: Path, run_id):
    try:
        _, diagnostics = load_records(path, run_id=run_id)
        return {"status": "ok", **{k: v for k, v in summary(diagnostics).items() if k != "conflicts"}}
    except RecordIntegrityError as exc:
        return {"status": "rejected", "reason": str(exc).split(":")[0][:160]}


def median(values):
    return round(statistics.median(values), 2) if values else None


def build(base: Path):
    sets = claim_sets(base)
    runs = []
    for key, (label, phase, role, plan, status, note) in RUNS.items():
        folder = base / key
        records_path = folder / "records.jsonl"
        if not records_path.is_file():
            continue
        rows = _jsonl(records_path)
        found = manifests(folder)
        first, last = (found[0], found[-1]) if found else ({}, {})
        run_json = json.loads((folder / "run.json").read_text(encoding="utf-8")) if (folder / "run.json").is_file() else {}
        claim_set = sets.get(first.get("claims_sha256"))
        env = last.get("environment") or {}
        by_id = {}
        for r in rows:
            by_id.setdefault(r.get("claim_id"), r)
        ids = claim_set["ids"] if claim_set else []
        covered = sum(1 for i in ids if i in by_id)
        seconds = [float(r["seconds"]) for r in by_id.values() if isinstance(r.get("seconds"), (int, float))]
        calls = [int(r["calls"]) for r in by_id.values() if isinstance(r.get("calls"), int)]
        sub = Counter(r.get("sub_state") or "decided" for r in by_id.values())
        failures = Counter(failure_kind(r) for r in by_id.values() if score.is_failure(r))
        log = guard_log(base, key)
        entry = {
            "key": key, "label": label, "verifier": (first.get("options") or {}).get("verifier"),
            "phase": phase, "role": role, "plan": plan, "status": status, "note": note,
            "set": claim_set["name"] if claim_set else None,
            "set_sha256": first.get("claims_sha256"),
            "set_size": len(ids) or None, "records": len(rows), "claims_with_records": covered or len(by_id),
            "complete_set": bool(ids) and covered == len(ids),
            "started_utc": first.get("started_utc"), "last_invocation_utc": last.get("started_utc"),
            "status_utc": run_json.get("status_utc"),
            "provenance": {
                "run_id": run_json.get("run_id"), "run_status": run_json.get("status"),
                "invocations": len(found), "git_head": first.get("git_head"),
                "working_tree_dirty": (env.get("git") or {}).get("dirty"),
                "config_fingerprint": first.get("config_fingerprint"), "model": first.get("model"),
                "model_digest": None, "context_length": None, "ollama_version": log["ollama_version"],
                "python": ((env.get("interpreter") or {}).get("version") or "").split(" ")[0] or None,
                "record_integrity": integrity(records_path, run_json.get("run_id")),
            },
            "efficiency": {
                "seconds_per_claim_median": median(seconds),
                "seconds_per_claim_mean": round(sum(seconds) / len(seconds), 2) if seconds else None,
                "seconds_total": round(sum(seconds), 1) if seconds else None,
                "claims_per_minute": round(len(seconds) / (sum(seconds) / 60), 2) if seconds and sum(seconds) else None,
                "calls_per_claim_mean": round(sum(calls) / len(calls), 2) if calls else None,
                "tokens": None, "gpu_memory": None,
            },
            "reliability": {
                "claims_attempted": len(by_id), "sub_states": dict(sub), "failure_kinds": dict(failures),
                "infrastructure_failures": sum(v for k, v in failures.items() if k in INFRASTRUCTURE),
                "model_output_failures": sum(v for k, v in failures.items() if k in MODEL_SIDE),
                "timeouts": failures.get("timeout", 0),
                "insufficient_rate": round(sub.get("NO_DECISIVE_EVIDENCE", 0) / len(by_id), 4) if by_id else None,
                "memory_stops": log["memory_stops"], "guard_stops": log["guard_stops"],
            },
            "coverage": None, "metrics": None,
        }
        if any(r.get("passage_diagnostics") for r in rows):
            measured = id_coverage.measure({k: r.get(k) for k in id_coverage.ALLOWED} for r in by_id.values())
            diags = [d for r in by_id.values() for d in (r.get("passage_diagnostics") or [])]
            entry["coverage"] = dict(
                measured, claims_with_diagnostics=sum(1 for r in by_id.values() if r.get("passage_diagnostics")),
                passages_per_claim_mean=round(len(diags) / len(by_id), 2) if by_id else None,
                quotes_demoted=sum(1 for d in diags if d.get("raw_stance") in ("SUPPORTS", "CONTRADICTS")
                                   and not d.get("grounded")),
                evidence=EVIDENCE)
        else:
            entry["coverage"] = {"evidence": EVIDENCE, "not_recorded": "passage diagnostics were added on 2026-10-25; "
                                                                       "this run predates them"}
        if status == "valid" and entry["complete_set"]:
            gold = [claim_set["gold"][i] for i in ids]
            recs = [by_id.get(i) for i in ids]
            s = score.summarise(gold, recs)
            both = historical_metric.both(gold, recs)
            entry["metrics"] = {
                "three_class_macro_f1": s["macro_f1"], "three_class_ci95": s["macro_f1_ci95"],
                "two_class_macro_f1": both["two_class"]["macro_f1"], "two_class_ci95": both["two_class"]["ci95"],
                "two_class_claims": both["two_class"]["claims"], "accuracy": s["accuracy"],
                "per_class": {c: dict(v, f1=round(2 * v["precision"] * v["recall"] / (v["precision"] + v["recall"]), 4)
                                      if v["precision"] and v["recall"] else 0.0)
                              for c, v in s["per_class"].items()},
                "confusion": s["confusion"], "false_support": s["false_support"],
                "nei_decided": both["nei_claims_decided"], "gold_distribution": s["gold_distribution"],
            }
        runs.append(entry)
    paired = []
    for file, system, baseline in PAIRED:
        path = base / file
        if path.is_file():
            p = json.loads(path.read_text(encoding="utf-8")).get("paired") or {}
            paired.append({"system": system, "baseline": baseline,
                           "delta_three_class_macro_f1": p.get("delta_macro_f1_system_minus_baseline"),
                           "ci95": p.get("delta_ci95_paired_bootstrap_seed_20261025"),
                           "permutation_p": p.get("permutation_p_seed_20261026"),
                           "mcnemar_p": p.get("accuracy_mcnemar_p"), "verdicts_changed": p.get("prediction_changed"),
                           "note": "baseline restricted to the same 90 claims" if "sub90" in file and
                                   baseline == "runs/retro_arm_c_studio" else ""})
    from webrag.model_service import STUDIO_VERIFIER
    return {
        "kind": "Evaluation run registry (aggregates only)",
        "default_verifier": STUDIO_VERIFIER,
        "about": "Run-level aggregates of the Studio verifier's evaluations on SciFact proxy sets, 2026-10-23 to "
                 "2026-10-26. Built offline from the private run folders by showcase/make_evaluation_runs.py; no claim "
                 "ids, claim text, passages or quotes. Phase 11B (the earlier research backend) is not part of it.",
        "target": {"metric": "three_class_macro_f1", "value": 0.75, "source": R1,
                   "note": "Three-class (SUPPORTED, CONTRADICTED, NOT_ENOUGH_INFO). The historical Arm C figure is "
                           "two-class and is not evidence for this target."},
        "scorer": {"three_class": "evaluation/studio_eval/score.py (summarise)",
                   "two_class": "evaluation/studio_eval/historical_metric.py",
                   "coverage": "evaluation/studio_eval/id_coverage.py",
                   "bootstrap": "95% percentile, 10,000 resamples, seed 20261024",
                   "failures": "a failed or missing record counts as NOT_ENOUGH_INFO in every metric"},
        "runs": runs, "paired": paired,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    data = build(args.runs)
    args.out.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(data['runs'])} runs, {len(data['paired'])} paired comparisons -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
