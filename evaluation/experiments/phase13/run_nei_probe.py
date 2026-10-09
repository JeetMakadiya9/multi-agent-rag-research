"""POST-HOC NEI false-support probe (added after observing B4; NOT preregistered).

Question: does B1 (evidence limit 8) produce more SUPPORTED/CONTRADICTED verdicts
than B0 (limit 2) on SciFact DEV claims that have no gold evidence (NEI)?
These claims have no gold label, so nothing here is "accuracy". This probe is
never merged into the preregistered B1 result.

Arms
* B0_FROZEN_REPLAY: frozen Phase 10F System B outputs (B0's exact configuration;
  reproduced 65/65 labels on the Phase 13 dev subset). No B0 inference is run.
  Claims without a usable frozen output are marked unavailable, never re-run.
* B1: the registered Phase 13 B1 variant, unchanged (``run_experiment.VARIANTS["B1"]``),
  through the frozen ``run_system_b`` adapter with frozen-retrieval replay.

Sample (frozen before inference): the 112 DEV claims whose gold evidence is empty,
ranked by (frozen top-1 retrieval score, claim_id); tercile = floor(3*rank/112);
within each tercile ordered by SHA-256("phase13-nei-probe-2026:<claim_id>");
first 10 per tercile. No prediction, outcome or failure status is consulted.

Thermal safety: ``run --batch-size N`` (max 5) executes at most N missing claims
in the foreground, fsyncs each record, verifies the durable count and exits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from evaluation.experiments.phase13.common import (
    DATA_DIR, FROZEN_RUN_DIR, OUTPUT_DIR, RESULTS_DIR, ROOT, frozen_retrieval_by_claim, gold_by_claim,
    load_dataset, load_frozen_rows, mean, median, read_jsonl, sha256_file, snapshot_protected, write_json,
)

PROBE_DIR = RESULTS_DIR / "phase13" / "runs_nei_probe"
MANIFEST_PATH = PROBE_DIR / "nei_probe_manifest.json"
B1_DIR = PROBE_DIR / "B1"
SALT = "phase13-nei-probe-2026"
PER_TERCILE = 10
MAX_BATCH = 5
MODEL_DIGEST = "359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7"
VALID = ("SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def order_key(claim_id: int) -> str:
    return hashlib.sha256(f"{SALT}:{claim_id}".encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ sample

def build_manifest(gold=None, retrieval=None) -> dict[str, Any]:
    gold = gold if gold is not None else gold_by_claim(load_dataset())
    retrieval = retrieval if retrieval is not None else frozen_retrieval_by_claim(load_frozen_rows())
    pool = sorted((float(retrieval[c]["chunks"][0]["retrieval_score"]), c)
                  for c, g in gold.items() if g.label is None and not g.rationales)
    n = len(pool)
    members: dict[int, list[dict[str, Any]]] = {0: [], 1: [], 2: []}
    for rank, (score, claim_id) in enumerate(pool):
        members[(3 * rank) // n].append({"claim_id": claim_id, "top1_retrieval_score": score,
                                         "score_rank": rank, "order_key": order_key(claim_id)})
    selected = []
    names = {0: "low", 1: "middle", 2: "high"}
    for t in (0, 1, 2):
        for row in sorted(members[t], key=lambda r: r["order_key"])[:PER_TERCILE]:
            selected.append({**row, "tercile": names[t]})
    selected.sort(key=lambda r: (["low", "middle", "high"].index(r["tercile"]), r["order_key"]))
    return {
        "artifact": "nei_probe_manifest", "status": "POST-HOC DIAGNOSTIC — added after observing B4; NOT preregistered",
        "frozen_before_inference": True, "created_at_utc": _now(),
        "rule": __doc__.split("Sample (frozen before inference):", 1)[1].split("Thermal safety", 1)[0].strip(),
        "pool": {"unlabeled_nei_claims": n, "tercile_sizes": {names[t]: len(members[t]) for t in members},
                 "tercile_score_ranges": {names[t]: [members[t][0]["top1_retrieval_score"],
                                                     members[t][-1]["top1_retrieval_score"]] for t in members}},
        "inputs_consulted": ["gold evidence emptiness (to identify NEI claims)", "frozen Phase 10F top-1 retrieval score"],
        "inputs_not_consulted": ["any prediction", "outcome", "failure status", "latency"],
        "claim_ids": [r["claim_id"] for r in selected], "claims": selected,
    }


def freeze_manifest() -> dict[str, Any]:
    manifest = build_manifest()
    if MANIFEST_PATH.exists():
        existing = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if existing["claim_ids"] != manifest["claim_ids"]:
            raise SystemExit("Refusing to overwrite a frozen NEI manifest with different claim IDs.")
        return existing
    write_json(MANIFEST_PATH, manifest)
    write_json(PROBE_DIR / "nei_probe_manifest.sha256.json",
               {"nei_probe_manifest.json": sha256_file(MANIFEST_PATH), "recorded_at_utc": _now()})
    return manifest


def load_manifest() -> dict[str, Any]:
    recorded = json.loads((PROBE_DIR / "nei_probe_manifest.sha256.json").read_text(encoding="utf-8"))
    if sha256_file(MANIFEST_PATH) != recorded["nei_probe_manifest.json"]:
        raise SystemExit("NEI manifest hash mismatch; refusing to run.")
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ B1 configuration

def b1_spec() -> dict[str, Any]:
    from evaluation.experiments.phase13.run_experiment import VARIANTS
    return dict(VARIANTS["B1"])


def b1_config(claim_ids):
    """Same behavioural fields as run_experiment.run_variant builds for B1."""
    from evaluation.experiments.experiment_config import ExperimentConfig
    spec = b1_spec()
    return ExperimentConfig(dataset_root=str(DATA_DIR), split="dev", example_ids=tuple(claim_ids),
                            seed=2026, top_k=20, verification_evidence_limit=spec["evidence_limit"],
                            model="qwen3:4b", temperature=0.0,
                            notes={"phase": "13", "variant": "B1", "retrieval": "frozen Phase 10F replay",
                                   "probe": "post-hoc NEI false-support probe"})


def model_digest() -> str | None:
    with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=10) as response:
        for model in json.load(response)["models"]:
            if model["name"] == "qwen3:4b":
                return model["digest"]
    return None


def completed_claims() -> list[int]:
    path = B1_DIR / "attempts.jsonl"
    return [r["claim_id"] for r in read_jsonl(path)] if path.exists() else []


def run_batch(batch_size: int) -> dict[str, Any]:
    if not 1 <= batch_size <= MAX_BATCH:
        raise SystemExit(f"batch size must be 1..{MAX_BATCH}")
    from evaluation.experiments.evaluator import _safe_call
    from evaluation.experiments.run_phase10_dev import _output_to_record
    from evaluation.experiments.system_b_verified_rag import run_system_b
    from evaluation.scifact.scifact_adapter import runtime_claim_payload
    from evaluation.experiments.phase13.b_plus import PROMPT_V1_SYSTEM, ReplayRetrievalProvider

    if model_digest() != MODEL_DIGEST:
        raise SystemExit("qwen3:4b digest mismatch or model unavailable; refusing to run.")
    manifest = load_manifest()
    claim_ids = manifest["claim_ids"]
    spec = b1_spec()
    if (spec["evidence_limit"], spec["context_window"], spec["prompt"]) != (8, 0, "v1"):
        raise SystemExit("Registered B1 spec is not (8, 0, v1); refusing to run.")
    dataset = load_dataset()
    records = {r.runtime.claim_id: r for r in dataset.claims}
    retrieval = frozen_retrieval_by_claim(load_frozen_rows())
    provider = ReplayRetrievalProvider({records[c].runtime.text: retrieval[c]["chunks"] for c in claim_ids},
                                       corpus={}, context_window=spec["context_window"])
    config = b1_config(claim_ids)

    B1_DIR.mkdir(parents=True, exist_ok=True)
    attempts_path = B1_DIR / "attempts.jsonl"
    inflight_path = B1_DIR / "inflight.json"
    done = completed_claims()
    if len(done) != len(set(done)):
        raise SystemExit("Duplicate attempts detected; refusing to run.")
    if inflight_path.exists():
        with open(B1_DIR / "interruptions.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps({**json.loads(inflight_path.read_text(encoding="utf-8")),
                                     "detected_at_utc": _now(),
                                     "handling": "claim re-executed; interruption kept visible"}) + "\n")
        inflight_path.unlink()
    config_path = B1_DIR / "run_config.json"
    if not config_path.exists():
        write_json(config_path, {
            "probe": "post-hoc NEI false-support probe (not preregistered)", "variant": "B1", **spec,
            "model": "qwen3:4b", "model_digest": MODEL_DIGEST, "experiment_config": config.to_dict(),
            "system_prompt_sha256": hashlib.sha256(PROMPT_V1_SYSTEM.encode("utf-8")).hexdigest(),
            "adapter": "evaluation.experiments.system_b_verified_rag.run_system_b (frozen, unmodified)",
            "retrieval": "ReplayRetrievalProvider over frozen phase10_full_dev retrieved_chunks",
            "manifest_sha256": sha256_file(MANIFEST_PATH), "created_at_utc": _now()})
    pending = [c for c in claim_ids if c not in set(done)][:batch_size]
    for claim_id in pending:
        runtime = runtime_claim_payload(records[claim_id].runtime)      # claim_id + claim text only
        inflight_path.write_text(json.dumps({"claim_id": claim_id, "started_at_utc": _now()}), encoding="utf-8")
        output, latency = _safe_call(run_system_b, runtime, provider, config)
        row = {"claim_id": claim_id, "variant": "B1", "probe": "nei", "runtime_input": runtime,
               "latency_seconds": latency, "finished_at_utc": _now(), **_output_to_record(output)}
        with open(attempts_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        inflight_path.unlink()
        print(f"[NEI-B1] claim {claim_id}: {row['status']} {row['predicted_label']} {latency:.1f}s "
              f"llm={row['llm_calls']}", flush=True)
    durable = completed_claims()
    if len(durable) != len(set(durable)):
        raise SystemExit("Duplicate attempts after batch!")
    return {"executed_this_batch": len(pending), "durable_attempts": len(durable),
            "remaining": len(claim_ids) - len(durable), "next_claims": [c for c in claim_ids if c not in set(durable)][:MAX_BATCH]}


# ------------------------------------------------------------------ report / integrity

def b0_frozen_replay(claim_ids) -> dict[int, dict[str, Any]]:
    frozen = {r["claim_id"]: r for r in load_frozen_rows() if r["system"] == "B"}
    out = {}
    for c in claim_ids:
        r = frozen.get(c)
        usable = bool(r) and r["status"] == "SUCCESS" and r.get("predicted_label") in VALID
        out[c] = {"available": usable, "status": r["status"] if r else "MISSING",
                  "predicted_label": r["predicted_label"] if usable else None,
                  "error": (r.get("error") or "")[:200] if r else None}
    return out


def _failure_kind(error: str | None) -> str:
    e = (error or "").lower()
    if any(k in e for k in ("timeouterror", "timed out", "10061", "connection refused", "urlerror", "interrupt")):
        return "infrastructure"
    return "model_output_schema"


def _rates(labels) -> dict[str, Any]:
    c = Counter(labels)
    n = sum(c.values())
    return {"valid_predictions": n, "SUPPORTED": c["SUPPORTED"], "CONTRADICTED": c["CONTRADICTED"],
            "INSUFFICIENT_EVIDENCE": c["INSUFFICIENT_EVIDENCE"],
            "false_support_rate": c["SUPPORTED"] / n if n else None,
            "false_contradiction_rate": c["CONTRADICTED"] / n if n else None,
            "abstention_rate": c["INSUFFICIENT_EVIDENCE"] / n if n else None}


def integrity(manifest) -> dict[str, Any]:
    from evaluation.experiments.run_phase10f import FROZEN_EXPECTED
    before = json.loads((OUTPUT_DIR / "protected_hashes_before.json").read_text(encoding="utf-8"))["hashes"]
    after = snapshot_protected()
    rows = read_jsonl(B1_DIR / "attempts.jsonl") if (B1_DIR / "attempts.jsonl").exists() else []
    retrieval = frozen_retrieval_by_claim(load_frozen_rows())
    gold = gold_by_claim(load_dataset())
    issues = []
    for r in rows:
        if set(r["runtime_input"]) != {"claim_id", "claim"} or r["runtime_input"]["claim"] != gold[r["claim_id"]].claim:
            issues.append(f"{r['claim_id']}: runtime payload")
        frozen_chunks = retrieval[r["claim_id"]]["chunks"]
        if [(c["scifact_doc_id"], c["scifact_sentence_indices"]) for c in r.get("retrieved_chunks") or []] != \
                [(c["scifact_doc_id"], c["scifact_sentence_indices"]) for c in frozen_chunks] and r["status"] == "SUCCESS":
            issues.append(f"{r['claim_id']}: retrieved refs differ from frozen")
        trace = ((r.get("raw_output") or {}).get("trace") or [{}])[0]
        if r["status"] == "SUCCESS" and [e["text"] for e in trace.get("initial_evidence", [])] != [c["text"] for c in frozen_chunks[:8]]:
            issues.append(f"{r['claim_id']}: judged evidence differs from frozen top-8")
        if any(m in json.dumps(r.get("raw_output") or {}) for m in ('"gold_label"', '"rationales"', "gold_rationale")):
            issues.append(f"{r['claim_id']}: gold marker in output")
    ids = [r["claim_id"] for r in rows]
    recorded = json.loads((PROBE_DIR / "nei_probe_manifest.sha256.json").read_text(encoding="utf-8"))
    return {"frozen_expected_mismatches": [k for k, v in FROZEN_EXPECTED.items() if after.get(k) != v],
            "protected_modified": [k for k in before if after.get(k) != before[k]],
            "protected_added": [k for k in after if k not in before],
            "model_digest_ok": model_digest() == MODEL_DIGEST,
            "manifest_hash_ok": sha256_file(MANIFEST_PATH) == recorded["nei_probe_manifest.json"],
            "duplicates": [c for c, n in Counter(ids).items() if n > 1],
            "attempts_outside_manifest": [c for c in ids if c not in manifest["claim_ids"]],
            "replay_and_leakage_issues": issues}


def report() -> dict[str, Any]:
    manifest = load_manifest()
    ids = manifest["claim_ids"]
    tercile = {c["claim_id"]: c["tercile"] for c in manifest["claims"]}
    b0 = b0_frozen_replay(ids)
    rows = {r["claim_id"]: r for r in read_jsonl(B1_DIR / "attempts.jsonl")} if (B1_DIR / "attempts.jsonl").exists() else {}
    b1_valid = {c: r["predicted_label"] for c, r in rows.items() if r["status"] == "SUCCESS" and r["predicted_label"] in VALID}
    b0_valid = {c: v["predicted_label"] for c, v in b0.items() if v["available"]}
    paired = [c for c in ids if c in b1_valid and c in b0_valid]
    lat = [r["latency_seconds"] for r in rows.values()]
    result = {
        "status": "POST-HOC DIAGNOSTIC (not preregistered; not merged into B1 accuracy; NEI claims have no gold label)",
        "manifest_claims": len(ids), "b1_attempted": len(rows), "b1_remaining": len(ids) - len(rows),
        "B0_FROZEN_REPLAY": {**_rates(b0_valid[c] for c in ids if c in b0_valid),
                             "unavailable": sorted(c for c in ids if not b0[c]["available"])},
        "B1": {**_rates(b1_valid.values()),
               "failures": [{"claim_id": c, "status": r["status"], "kind": _failure_kind(r.get("error")),
                             "error": (r.get("error") or "")[:200]} for c, r in rows.items() if r["status"] != "SUCCESS"],
               "latency_seconds": {"mean": mean(lat), "median": median(lat), "max": max(lat) if lat else None},
               "llm_calls_total": sum(r["llm_calls"] or 0 for r in rows.values())},
        "paired_denominator": len(paired),
        "paired_B0": _rates(b0_valid[c] for c in paired), "paired_B1": _rates(b1_valid[c] for c in paired),
        "transitions_B0_to_B1": dict(Counter(f"{b0_valid[c]}->{b1_valid[c]}" for c in paired)),
        "by_tercile": {t: {"paired": sum(tercile[c] == t for c in paired),
                           "B0_SUPPORTED": sum(tercile[c] == t and b0_valid[c] == "SUPPORTED" for c in paired),
                           "B1_SUPPORTED": sum(tercile[c] == t and b1_valid[c] == "SUPPORTED" for c in paired)}
                       for t in ("low", "middle", "high")},
        "per_claim": [{"claim_id": c, "tercile": tercile[c], "B0": b0[c]["predicted_label"] or f"unavailable ({b0[c]['status']})",
                       "B1": (rows[c]["predicted_label"] or f"{rows[c]['status']}") if c in rows else "not yet run"} for c in ids],
        "integrity": integrity(manifest),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("freeze-manifest")
    run = sub.add_parser("run")
    run.add_argument("--batch-size", type=int, default=MAX_BATCH)
    sub.add_parser("report")
    args = parser.parse_args()
    if args.cmd == "freeze-manifest":
        m = freeze_manifest()
        print(json.dumps({"claim_ids": m["claim_ids"], "pool": m["pool"],
                          "sha256": sha256_file(MANIFEST_PATH)}, indent=1))
    elif args.cmd == "run":
        print(json.dumps(run_batch(args.batch_size), indent=1))
    else:
        result = report()
        write_json(PROBE_DIR / "nei_probe_report.json", result)
        print(json.dumps({k: v for k, v in result.items() if k != "per_claim"}, indent=1))


if __name__ == "__main__":
    main()
