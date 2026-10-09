"""Deterministic Phase 13 development subset of SciFact DEV.

Sampling algorithm (documented in development_set_manifest.json):

1. Universe: the 188 DEV claims with an unambiguous gold label (SUPPORTED or
   CONTRADICTED under the frozen ``gold_claim_label``). Unlabeled claims are
   kept out of accuracy work, as in Phase 11B.
2. Each claim is assigned a *retrieval* bucket from the frozen Phase 10F
   retrieval ranks (not from any prediction): A_top2 / B_rank3_8 / C_absent_top8.
3. Strata = gold label x bucket (6 strata). Within a stratum, claims are ordered
   by SHA-256("phase13-dev-2026:<claim_id>") and the first PER_STRATUM are taken.
4. Coverage top-up: if fewer than MIN_MULTI multi-sentence-rationale claims or
   fewer than MIN_COMPOUND compound claims (>=2 deterministic verification
   components or sentences) were drawn, further such claims are added in the
   same hash order until the minimum is met (or the pool is exhausted).
5. All remaining labeled claims are recorded as RESERVED for a later held-out
   validation run; Phase 13 experiments never touch them.

No prediction, correctness or failure status is consulted at any step.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Any

from evaluation.experiments.phase13.common import (
    BUCKETS, GOLD_LABELS, OUTPUT_DIR, complete_rationale_within, first_gold_rationale_rank,
    frozen_retrieval_by_claim, gold_by_claim, load_dataset, load_frozen_rows, rationale_bucket, write_json,
)

SEED = 2026
SALT = f"phase13-dev-{SEED}"
PER_STRATUM = 10
MIN_MULTI = 12
MIN_COMPOUND = 8


def order_key(claim_id: int) -> str:
    return hashlib.sha256(f"{SALT}:{claim_id}".encode("utf-8")).hexdigest()


def claim_attributes(text: str) -> dict[str, Any]:
    from verification_units import analyze_verification_units
    units = analyze_verification_units(text)
    components = sum(len(u.components) for u in units)
    return {
        "verification_units": len(units),
        "verification_components": components,
        "compound": len(units) > 1 or components > 1,
        "negation": bool(re.search(r"\b(?:not|no|never|without|neither|nor|lack|lacks|lacking|absence)\b", text, re.I)),
        "has_number": bool(re.search(r"\d", text)),
        "token_count": len(text.split()),
        "quantity_constraints": sum(len(u.quantitative_constraints) + len(u.temporal_constraints) for u in units),
    }


def build() -> dict[str, Any]:
    gold = gold_by_claim(load_dataset())
    retrieval = frozen_retrieval_by_claim(load_frozen_rows())
    pool: list[dict[str, Any]] = []
    for claim_id, g in sorted(gold.items()):
        if g.label not in GOLD_LABELS:
            continue
        chunks = retrieval[claim_id]["chunks"]
        rank = first_gold_rationale_rank(g, chunks)
        pool.append({
            "claim_id": claim_id, "gold_label": g.label, "bucket": rationale_bucket(rank),
            "first_gold_rationale_rank": rank,
            "multi_sentence_rationale": g.multi_sentence,
            "complete_rationale_top2": complete_rationale_within(g, chunks, 2),
            "complete_rationale_top8": complete_rationale_within(g, chunks, 8),
            "retrieval_source_system": retrieval[claim_id]["source_system"],
            "order_key": order_key(claim_id),
            **claim_attributes(g.claim),
        })
    pool.sort(key=lambda r: r["order_key"])
    selected: dict[int, dict[str, Any]] = {}
    strata_available = Counter((r["gold_label"], r["bucket"]) for r in pool)
    for label in GOLD_LABELS:
        for bucket in BUCKETS:
            members = [r for r in pool if r["gold_label"] == label and r["bucket"] == bucket]
            for r in members[:PER_STRATUM]:
                selected[r["claim_id"]] = {**r, "selection_reason": f"stratum {label}|{bucket}"}
    for flag, minimum in (("multi_sentence_rationale", MIN_MULTI), ("compound", MIN_COMPOUND)):
        for r in pool:
            if sum(bool(s[flag]) for s in selected.values()) >= minimum:
                break
            if r[flag] and r["claim_id"] not in selected:
                selected[r["claim_id"]] = {**r, "selection_reason": f"coverage top-up: {flag}"}
    dev = sorted(selected.values(), key=lambda r: r["claim_id"])
    reserved = sorted(r["claim_id"] for r in pool if r["claim_id"] not in selected)

    def summary(rows):
        return {
            "claims": len(rows),
            "by_label": dict(Counter(r["gold_label"] for r in rows)),
            "by_bucket": dict(Counter(r["bucket"] for r in rows)),
            "by_label_bucket": {f"{k[0]}|{k[1]}": v for k, v in sorted(Counter(
                (r["gold_label"], r["bucket"]) for r in rows).items())},
            "multi_sentence_rationale": sum(r["multi_sentence_rationale"] for r in rows),
            "compound": sum(r["compound"] for r in rows),
            "negation": sum(r["negation"] for r in rows),
            "has_number": sum(r["has_number"] for r in rows),
            "complete_rationale_top2": sum(r["complete_rationale_top2"] for r in rows),
        }

    return {
        "phase": "13", "artifact": "development_set_manifest",
        "status": "FROZEN BEFORE ANY PHASE 13 INFERENCE",
        "purpose": "engineering decisions only; NOT a benchmark and NOT held-out evidence",
        "seed": SEED, "hash_salt": SALT, "per_stratum": PER_STRATUM,
        "min_multi_sentence": MIN_MULTI, "min_compound": MIN_COMPOUND,
        "algorithm": __doc__.strip(),
        "inputs_consulted": ["SciFact DEV gold rationales (labels, sentence indices)",
                             "frozen Phase 10F retrieved chunk ranks",
                             "deterministic claim-text attributes"],
        "inputs_not_consulted": ["any system prediction", "correctness", "execution status", "latency"],
        "strata_available": {f"{k[0]}|{k[1]}": v for k, v in sorted(strata_available.items())},
        "development_set_summary": summary(dev),
        "labeled_pool_summary": summary(pool),
        "claim_ids": [r["claim_id"] for r in dev],
        "claims": dev,
        "reserved_for_later_validation_claim_ids": reserved,
        "reserved_count": len(reserved),
        "note": "Stratified to balance labels and retrieval buckets, so development-set accuracy is NOT "
                "comparable to the 179-claim full-DEV accuracy; compare variants only against B v1 on this "
                "same subset.",
    }


def main() -> None:
    manifest = build()
    path = OUTPUT_DIR / "development_set_manifest.json"
    if path.exists():
        import json
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("claim_ids") != manifest["claim_ids"]:
            raise SystemExit("Refusing to overwrite a frozen development set with different claim IDs.")
    write_json(path, manifest)
    print(manifest["development_set_summary"], manifest["reserved_count"])


if __name__ == "__main__":
    main()
