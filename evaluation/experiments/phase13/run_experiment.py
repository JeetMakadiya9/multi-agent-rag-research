"""Run one Phase 13 variant on the frozen development subset (live qwen3:4b).

Attempts are gold-free (runtime payload = claim ID + claim text) and appended
durably to results/phase13_improvement/runs/<variant>/attempts.jsonl. Gold is
joined only by ``scoring.py`` after the variant has finished. Attempts are
never retried: FAILED/TIMEOUT stay visible. A process interruption leaves an
inflight marker; on restart it is logged to interruptions.jsonl and that claim
is executed (the interruption itself remains recorded).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from datetime import datetime, timezone
from typing import Any

from evaluation.experiments.phase13.common import (
    DATA_DIR, FROZEN_RUN_DIR, OUTPUT_DIR, frozen_retrieval_by_claim, load_dataset, load_frozen_rows, sha256_file,
    write_json,
)

MODEL = "qwen3:4b"

VARIANTS: dict[str, dict[str, Any]] = {
    "B0": {"evidence_limit": 2, "context_window": 0, "prompt": "v1",
           "description": "B v1 re-run on the development subset (live paired control)"},
    "B1": {"evidence_limit": 8, "context_window": 0, "prompt": "v1",
           "description": "H1: B v1 + evidence limit 8"},
    "B2": {"evidence_limit": 8, "context_window": 1, "prompt": "v1",
           "description": "H2: B1 + same-abstract neighbour-sentence packaging (window 1)"},
    "B4": {"evidence_limit": 2, "context_window": 0, "prompt": "v2",
           "description": "H4: B v1 + contradiction-aware judge system prompt"},
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def register_variant(name: str, spec: dict[str, Any]) -> None:
    VARIANTS[name] = spec


def run_variant(name: str, *, limit: int | None = None) -> dict[str, Any]:
    from evaluation.experiments.evaluator import _safe_call
    from evaluation.experiments.experiment_config import ExperimentConfig
    from evaluation.experiments.run_phase10_dev import _output_to_record
    from evaluation.experiments.system_b_verified_rag import run_system_b
    from evaluation.scifact.scifact_adapter import runtime_claim_payload
    from evaluation.experiments.phase13 import b_plus

    spec = VARIANTS[name]
    manifest = json.loads((OUTPUT_DIR / "development_set_manifest.json").read_text(encoding="utf-8"))
    claim_ids = manifest["claim_ids"][:limit] if limit else manifest["claim_ids"]
    dataset = load_dataset()
    records = {r.runtime.claim_id: r for r in dataset.claims}
    corpus = {d.doc_id: d for d in dataset.corpus}
    retrieval = frozen_retrieval_by_claim(load_frozen_rows())
    by_text = {records[c].runtime.text: retrieval[c]["chunks"] for c in claim_ids}
    provider = b_plus.ReplayRetrievalProvider(by_text, corpus=corpus, context_window=spec["context_window"])
    config = ExperimentConfig(dataset_root=str(DATA_DIR), split="dev", example_ids=tuple(claim_ids),
                              seed=2026, top_k=20, verification_evidence_limit=spec["evidence_limit"],
                              model=MODEL, temperature=0.0,
                              notes={"phase": "13", "variant": name, "retrieval": "frozen Phase 10F replay"})

    run_dir = OUTPUT_DIR / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    attempts_path = run_dir / "attempts.jsonl"
    inflight_path = run_dir / "inflight.json"
    done = set()
    if attempts_path.exists():
        done = {json.loads(line)["claim_id"] for line in attempts_path.open(encoding="utf-8") if line.strip()}
    if inflight_path.exists():
        inflight = json.loads(inflight_path.read_text(encoding="utf-8"))
        with open(run_dir / "interruptions.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps({**inflight, "detected_at_utc": _now(),
                                     "handling": "re-executed; interruption kept visible here"}) + "\n")
        inflight_path.unlink()
    write_json(run_dir / "run_config.json", {
        "variant": name, **spec, "model": MODEL, "experiment_config": config.to_dict(),
        "judge": {"provider": "evaluation.experiments.local_model.OllamaCPUProvider (frozen)",
                  "num_predict": 256, "num_gpu": 0, "think": False, "format": "json", "temperature": 0.0,
                  "timeout_seconds": 120, "max_judge_retries": 1,
                  "system_prompt_sha256": __import__("hashlib").sha256(
                      (b_plus.PROMPT_V2_SYSTEM if spec["prompt"] == "v2" else b_plus.PROMPT_V1_SYSTEM)
                      .encode("utf-8")).hexdigest()},
        "adapter": "evaluation.experiments.system_b_verified_rag.run_system_b (frozen, unmodified)",
        "retrieval": "ReplayRetrievalProvider over frozen phase10_full_dev retrieved_chunks",
        "host": {"platform": platform.platform(), "python": platform.python_version()},
        "frozen_per_example_sha256": sha256_file(FROZEN_RUN_DIR / "per_example.jsonl"),
        "updated_at_utc": _now(),
    })

    executed = 0
    for claim_id in claim_ids:
        if claim_id in done:
            continue
        runtime = runtime_claim_payload(records[claim_id].runtime)   # gold-free
        inflight_path.write_text(json.dumps({"claim_id": claim_id, "variant": name, "started_at_utc": _now()}),
                                 encoding="utf-8")
        if spec["prompt"] == "v2":
            with b_plus.judge_prompt_override(b_plus.PROMPT_V2_SYSTEM) as judges:
                output, latency = _safe_call(run_system_b, runtime, provider, config)
            replacements = sum(j.replacements for j in judges)
        else:
            output, latency = _safe_call(run_system_b, runtime, provider, config)
            replacements = 0
        row = {"claim_id": claim_id, "variant": name, "runtime_input": runtime,
               "latency_seconds": latency, "prompt_substitutions": replacements,
               "finished_at_utc": _now(), **_output_to_record(output)}
        with open(attempts_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        inflight_path.unlink()
        executed += 1
        print(f"[{name}] claim {claim_id}: {row['status']} {row['predicted_label']} "
              f"{latency:.1f}s llm={row['llm_calls']}", flush=True)
    return {"variant": name, "executed": executed, "previously_done": len(done)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True)
    parser.add_argument("--limit", type=int, default=None, help="debug only: first N dev claims")
    args = parser.parse_args()
    if args.variant == "B5":
        from evaluation.experiments.phase13.b5 import register
        register()
    started = time.time()
    print(run_variant(args.variant, limit=args.limit), f"{time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
