"""The agentic verification study's two-class metric, computed beside the three-class one on the same records.

The 2026-10-23 study (README, ``showcase/data/historical_agentic_verification.json``) scored "two-class macro-F1
(SUPPORTED, CONTRADICTED) over all 188 labelled claims; INSUFFICIENT and failed outputs count as errors". Claims
whose gold label is NOT_ENOUGH_INFO were not in that denominator; it reported separately how many of them were still
decided ("no evidence committed"). ``score.summarise`` (PROTOCOL.md) uses all three classes. This module reports
both, from the same records, so the two numbers are never confused.

    python -m evaluation.studio_eval.historical_metric --records RUN/records.jsonl --claims SET/claims.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import score
from .score import C, N, S


def two_class_macro_f1(gold: Sequence[str], pred: Sequence[str]) -> float:
    """Mean F1 of SUPPORTED and CONTRADICTED over claims whose gold label is one of them; any other prediction
    (NOT_ENOUGH_INFO, a failure) is an error for the claim's class and a false positive for neither."""
    pairs = [(g, p) for g, p in zip(gold, pred) if g in (S, C)]
    scores = []
    for cls in (S, C):
        tp = sum(g == cls and p == cls for g, p in pairs)
        fp = sum(g != cls and p == cls for g, p in pairs)
        fn = sum(g == cls and p != cls for g, p in pairs)
        scores.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return sum(scores) / 2


def bootstrap_two_class(gold: Sequence[str], pred: Sequence[str], n: int = 10000, seed: int = 20261024) -> List[float]:
    """95% percentile interval, resampling the SUPPORTED/CONTRADICTED claims (as ``score.bootstrap`` does)."""
    pairs = [(g, p) for g, p in zip(gold, pred) if g in (S, C)]
    rng, idx, values = random.Random(seed), range(len(pairs)), []
    for _ in range(n):
        sample = [pairs[rng.choice(idx)] for _ in idx]
        values.append(two_class_macro_f1([g for g, _ in sample], [p for _, p in sample]))
    values.sort()
    return [round(values[int(0.025 * n)], 4), round(values[int(0.975 * n) - 1], 4)]


def both(gold: List[str], records: List[Optional[Dict[str, Any]]], n: int = 10000) -> Dict[str, Any]:
    pred = [score.predicted(r) for r in records]
    nei = [p for g, p in zip(gold, pred) if g == N]
    return {
        "three_class": {"definition": "macro-F1 over SUPPORTED, CONTRADICTED, NOT_ENOUGH_INFO; all claims",
                        "claims": len(gold), "macro_f1": round(score.macro_f1(gold, pred), 4),
                        "ci95": score.bootstrap(gold, pred, n=n)},
        "two_class": {"definition": "historical: macro-F1 over SUPPORTED, CONTRADICTED; gold S/C claims only; "
                                    "NOT_ENOUGH_INFO predictions and failures count as errors",
                      "claims": sum(g in (S, C) for g in gold), "macro_f1": round(two_class_macro_f1(gold, pred), 4),
                      "ci95": bootstrap_two_class(gold, pred, n=n)},
        "nei_claims_decided": f"{sum(p != N for p in nei)}/{len(nei)}",
    }


def _load(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--claims", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)
    claims = _load(args.claims)
    by_id = {r["claim_id"]: r for r in _load(args.records)}
    result = both([c["label"] for c in claims], [by_id.get(c["claim_id"]) for c in claims])
    text = json.dumps(result, indent=1)
    print(text)
    if args.json_out:
        args.json_out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
