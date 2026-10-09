"""Offline legacy replay: recorded Experiment B outputs -> evidence core -> legacy policy.

For every recorded attempt with recorded judgements, the harness

  1. adapts the record into contracts (no LLM, no retrieval, no network),
  2. validates referential integrity of the resulting graph,
  3. applies LegacyAnySupportsPolicy, and
  4. requires EXACT equality with the frozen record:
       - claim-level run status (verdict vs failed-incomplete) and verdict,
       - every unit's verdict,
       - every unit's per-component coverage (status + support/refute indices).

Records without recorded judgements (interrupted attempts) are listed as
non-replayable with their reason; they are never counted as matches.

Usage: python -m evidence_core.replay   (writes the report under
evaluation/experiments/results/slice0_evidence_core/)
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import socket
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiment_b_verification as frozen_b  # noqa: E402

from evidence_core.adapters.experiment_b import (  # noqa: E402
    AdaptedExperimentB, AmbiguousRecord, NotReplayable, adapt_experiment_b_record,
)
from evidence_core.contracts import ClaimVerdict, JudgeIdentity, JudgeKind  # noqa: E402
from evidence_core.graph import EvidenceGraph  # noqa: E402
from evidence_core.policies import LegacyAnySupportsPolicy  # noqa: E402

RESULTS = ROOT / "evaluation/experiments/results"
OUTPUT_DIR = RESULTS / "slice0_evidence_core"


def v1_system_prompt_sha256() -> str:
    unit = frozen_b.analyze_verification_units("A uses B.")[0]
    messages = frozen_b._pairwise_judge_prompt("q", unit, "E1", frozen_b.EvidenceItem("x", "y"), unit.components[0])
    return hashlib.sha256(messages[0]["content"].encode("utf-8")).hexdigest()


def _judge(model: str, prompt_sha: Optional[str], prompt_label: str, settings: Mapping[str, Any]) -> JudgeIdentity:
    return JudgeIdentity(kind=JudgeKind.LLM, name="experiment_b_pairwise_judge", model=model,
                         model_version="359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7",
                         prompt_version=f"{prompt_label}:{prompt_sha}", settings=dict(settings))


def recorded_sources() -> list[dict[str, Any]]:
    """Every recorded Experiment B run in the project (all read-only)."""
    judge_settings = {"num_predict": 256, "num_gpu": 0, "temperature": 0.0, "think": False, "format": "json",
                      "max_judge_retries": 1}
    sources = [{"name": "phase10f_frozen_full_dev_B", "path": RESULTS / "phase10_full_dev/per_example.jsonl",
                "filter": lambda r: r.get("system") == "B",
                "judge": _judge("qwen3:4b", v1_system_prompt_sha256(), "pairwise_v1", judge_settings)}]
    for variant in ("B0", "B1", "B2", "B4"):
        run_dir = RESULTS / "phase13_improvement/runs" / variant
        if (run_dir / "attempts.jsonl").exists():
            cfg = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
            sources.append({"name": f"phase13_{variant}", "path": run_dir / "attempts.jsonl", "filter": lambda r: True,
                            "judge": _judge("qwen3:4b", cfg["judge"]["system_prompt_sha256"],
                                            f"pairwise_{cfg['prompt']}", judge_settings)})
    probe = RESULTS / "phase13/runs_nei_probe/B1"
    if (probe / "attempts.jsonl").exists():
        cfg = json.loads((probe / "run_config.json").read_text(encoding="utf-8"))
        sources.append({"name": "phase13_nei_probe_B1", "path": probe / "attempts.jsonl", "filter": lambda r: True,
                        "judge": _judge("qwen3:4b", cfg["system_prompt_sha256"], "pairwise_v1", judge_settings)})
    return sources


def scifact_lookup():
    corpus = {}
    with open(ROOT / "data/scifact/data/corpus.jsonl", encoding="utf-8") as handle:
        for line in handle:
            doc = json.loads(line)
            corpus[f"scifact:{doc['doc_id']}"] = (doc["title"], tuple(doc["abstract"]))
    return corpus.get


@contextlib.contextmanager
def network_forbidden() -> Iterator[None]:
    """Any socket connection attempt (Ollama, downloads) raises during replay."""
    def refuse(*_args, **_kwargs):
        raise RuntimeError("Network access attempted during offline replay")
    saved = (socket.socket.connect, socket.create_connection)
    socket.socket.connect, socket.create_connection = refuse, refuse
    try:
        yield
    finally:
        socket.socket.connect, socket.create_connection = saved


def replay_record(adapted: AdaptedExperimentB, run_key: str) -> tuple[ClaimVerdict, list[ClaimVerdict]]:
    graph = EvidenceGraph()
    graph.add_all(adapted.objects())
    graph.validate()
    policy = LegacyAnySupportsPolicy()
    unit_verdicts, unit_judgements = [], []
    for unit, comps in zip(adapted.unit_claims, adapted.component_claims):
        comp_ids = {c.claim_id for c in comps}
        judged = [j for j in adapted.judgements if j.claim_id in comp_ids]
        unit_verdicts.append(policy.aggregate_unit(unit, comps, adapted.evidence, judged, run_key=run_key))
        unit_judgements.append(judged)
    root = policy.aggregate_claim(adapted.root_claim, unit_verdicts, unit_judgements, run_key=run_key)
    graph.add_all([*unit_verdicts, root])
    graph.validate()
    return root, unit_verdicts


def _coverage_rows(verdict: ClaimVerdict, adapted: AdaptedExperimentB, unit_pos: int) -> list[dict[str, Any]]:
    comps = {c.claim_id: c.metadata["legacy_component_id"] for c in adapted.component_claims[unit_pos]}
    index = {e.evidence_id: e.admission.presentation_index for e in adapted.evidence}
    return [{"component_index": comps[cov.component_claim_id], "status": cov.status,
             "supporting_evidence_indices": [index[i] for i in cov.supporting_evidence_ids],
             "contradicting_evidence_indices": [index[i] for i in cov.contradicting_evidence_ids]}
            for cov in verdict.component_coverage]


def _recorded_rows(rows) -> list[dict[str, Any]]:
    keys = ("component_index", "status", "supporting_evidence_indices", "contradicting_evidence_indices")
    return [{k: (list(row[k]) if isinstance(row[k], (list, tuple)) else row[k]) for k in keys} for row in rows]


def run_replay() -> dict[str, Any]:
    lookup = scifact_lookup()
    results, totals = [], Counter()
    for source in recorded_sources():
        with open(source["path"], encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        for record in filter(source["filter"], records):
            claim_id = record["claim_id"]
            run_key = f"{source['name']}:{claim_id}"
            entry: dict[str, Any] = {"source": source["name"], "claim_id": claim_id,
                                     "recorded_status": record.get("status")}
            try:
                adapted = adapt_experiment_b_record(record, claim_key=f"scifact-dev:{claim_id}", run_key=run_key,
                                                    judge=source["judge"], lookup=lookup)
            except NotReplayable as exc:
                entry.update(result="NOT_REPLAYABLE", reason=str(exc))
                results.append(entry); totals["not_replayable"] += 1
                continue
            except AmbiguousRecord as exc:
                entry.update(result="BLOCKER_AMBIGUOUS_RECORD", reason=str(exc))
                results.append(entry); totals["blocker"] += 1
                continue
            root, units = replay_record(adapted, run_key)
            rec = adapted.recorded
            mismatches = []
            if root.legacy.run_status is not rec.run_status:
                mismatches.append(f"run status {rec.run_status.value} != {root.legacy.run_status.value}")
            if root.legacy.verdict != rec.verdict:
                mismatches.append(f"verdict {rec.verdict} != {root.legacy.verdict}")
            for pos, unit in enumerate(units):
                if unit.legacy.verdict.value != rec.unit_verdicts[pos]:
                    mismatches.append(f"unit {pos + 1} verdict {rec.unit_verdicts[pos]} != {unit.legacy.verdict.value}")
                if _coverage_rows(unit, adapted, pos) != _recorded_rows(rec.unit_coverage[pos]):
                    mismatches.append(f"unit {pos + 1} coverage matrix differs")
            entry.update(result="EXACT_MATCH" if not mismatches else "MISMATCH",
                         old_status=rec.run_status.value, new_status=root.legacy.run_status.value,
                         old_verdict=rec.verdict.value if rec.verdict else None,
                         new_verdict=root.legacy.verdict.value if root.legacy.verdict else None,
                         units=len(units), judgements=len(adapted.judgements),
                         invalid_judgements=sum(j.stance.value == "INVALID" for j in adapted.judgements),
                         evidence=len(adapted.evidence), candidates=len(adapted.retrieval.candidates),
                         mismatch_reasons=mismatches)
            results.append(entry)
            totals["exact" if not mismatches else "mismatch"] += 1
    per_source = {}
    for entry in results:
        per_source.setdefault(entry["source"], Counter())[entry["result"]] += 1
    return {
        "artifact": "slice0_legacy_replay", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": f"{LegacyAnySupportsPolicy.name}@{LegacyAnySupportsPolicy.version}",
        "acceptance": "every replayable record: exact equality of run status, verdict, unit verdicts and coverage",
        "network_guard": "socket connections forbidden during replay",
        "totals": {"records": len(results), "replayed": totals["exact"] + totals["mismatch"],
                   "exact_matches": totals["exact"], "mismatches": totals["mismatch"],
                   "not_replayable": totals["not_replayable"], "blockers": totals["blocker"]},
        "per_source": {k: dict(v) for k, v in per_source.items()},
        "verdict_distribution_replayed": dict(Counter(f"{e.get('new_status')}|{e.get('new_verdict')}"
                                                      for e in results if e["result"] in ("EXACT_MATCH", "MISMATCH"))),
        "mismatches": [e for e in results if e["result"] == "MISMATCH"],
        "not_replayable": [e for e in results if e["result"] == "NOT_REPLAYABLE"],
        "blockers": [e for e in results if e["result"] == "BLOCKER_AMBIGUOUS_RECORD"],
        "records": results,
    }


def render(report: Mapping[str, Any]) -> str:
    t = report["totals"]
    lines = ["# Slice 0 — Legacy Replay Report", "",
             f"Policy: `{report['policy']}`. Acceptance: {report['acceptance']}. {report['network_guard']}.", "",
             "| Records | Replayed | Exact matches | Mismatches | Not replayable | Blockers |", "|---|---|---|---|---|---|",
             f"| {t['records']} | {t['replayed']} | {t['exact_matches']} | {t['mismatches']} | {t['not_replayable']} | {t['blockers']} |",
             "", "## Per source", "", "| Source | Results |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in report["per_source"].items()]
    lines += ["", "## Replayed outcome distribution (status, verdict)", ""]
    lines += [f"- {k}: {v}" for k, v in report["verdict_distribution_replayed"].items()]
    for title, key in (("Mismatches", "mismatches"), ("Not replayable", "not_replayable"), ("Blockers", "blockers")):
        lines += ["", f"## {title}", ""]
        rows = report[key]
        lines += [f"- {r['source']} claim {r['claim_id']}: {r.get('reason') or r.get('mismatch_reasons')}" for r in rows] or ["- none"]
    return "\n".join(lines) + "\n"


def main() -> None:
    with network_forbidden():
        report = run_replay()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "legacy_replay_report.json").write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")
    (OUTPUT_DIR / "legacy_replay_report.md").write_text(render(report), encoding="utf-8")
    print(json.dumps({"totals": report["totals"], "per_source": report["per_source"]}, indent=1))
    if report["totals"]["mismatches"] or report["totals"]["blockers"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
