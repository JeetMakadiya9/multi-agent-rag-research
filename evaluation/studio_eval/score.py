"""Score Research Studio evaluation records against the frozen protocol (PROTOCOL.md Sections 5 and 6).

Standard library only. Every eligible claim is in the denominator; a failure or missing record counts as
NOT_ENOUGH_INFO for the primary endpoint and is reported separately.

Usage:
    python -m evaluation.studio_eval.score --records DIR/records.jsonl [--candidate DIR/records.jsonl]
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from math import comb
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

S, C, N = "SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO"
CLASSES = (S, C, N)
FAILURE_STATES = {"SERVICE_ERROR", "ASSESSMENT_FAILED", "HARNESS_ERROR"}


def predicted(record: Optional[Dict[str, Any]]) -> str:
    """The Studio's verdict on the protocol's labels; anything not a decided verdict counts as NOT_ENOUGH_INFO."""
    if not record:
        return N
    verdict = record.get("verdict")
    if verdict in (S, C) and record.get("sub_state") not in FAILURE_STATES and record.get("parse_ok", True):
        return verdict
    return N


def is_failure(record: Optional[Dict[str, Any]]) -> bool:
    return (not record or record.get("verdict") is None or record.get("sub_state") in FAILURE_STATES
            or not record.get("parse_ok", True))


def macro_f1(gold: Sequence[str], pred: Sequence[str]) -> float:
    scores = []
    for cls in CLASSES:
        tp = sum(g == cls and p == cls for g, p in zip(gold, pred))
        fp = sum(g != cls and p == cls for g, p in zip(gold, pred))
        fn = sum(g == cls and p != cls for g, p in zip(gold, pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores)


def bootstrap(gold, pred, other=None, n=10000, seed=20261024):
    """95% percentile interval of macro-F1, or of the paired difference when ``other`` is given."""
    rng, idx, values = random.Random(seed), range(len(gold)), []
    for _ in range(n):
        sample = [rng.choice(idx) for _ in idx]
        g = [gold[i] for i in sample]
        value = macro_f1(g, [pred[i] for i in sample])
        if other is not None:
            value -= macro_f1(g, [other[i] for i in sample])
        values.append(value)
    values.sort()
    return [round(values[int(0.025 * n)], 4), round(values[int(0.975 * n) - 1], 4)]


def mcnemar(b: int, c: int) -> float:
    m = b + c
    return min(1.0, 2 * sum(comb(m, j) for j in range(min(b, c) + 1)) / 2 ** m) if m else 1.0


def summarise(gold: List[str], records: List[Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    pred = [predicted(r) for r in records]
    confusion = {g: {p: sum(1 for a, b in zip(gold, pred) if a == g and b == p) for p in CLASSES} for g in CLASSES}
    per_class = {}
    for cls in CLASSES:
        tp = confusion[cls][cls]
        col = sum(confusion[g][cls] for g in CLASSES)
        row = sum(confusion[cls].values())
        per_class[cls] = {"precision": round(tp / col, 4) if col else None, "recall": round(tp / row, 4) if row else None,
                          "support": row}
    not_supported = [i for i, g in enumerate(gold) if g != S]
    contradicted = [i for i, g in enumerate(gold) if g == C]
    failures = Counter((r or {}).get("sub_state") or ("MISSING" if not r else "NO_VERDICT")
                       for r in records if is_failure(r))
    return {
        "claims": len(gold), "gold_distribution": dict(Counter(gold)),
        "macro_f1": round(macro_f1(gold, pred), 4), "macro_f1_ci95": bootstrap(gold, pred),
        "accuracy": round(sum(g == p for g, p in zip(gold, pred)) / len(gold), 4),
        "per_class": per_class, "confusion": confusion,
        "false_support": f"{sum(pred[i] == S for i in not_supported)}/{len(not_supported)}",
        "missed_contradiction": f"{sum(pred[i] != C for i in contradicted)}/{len(contradicted)}",
        "insufficient_rate": round(sum(p == N for p in pred) / len(pred), 4),
        "operational_failures": dict(failures),
        "adjudicated": sum(1 for r in records if r and r.get("sub_state") == "ADJUDICATED"),
        "injection_withheld_claims": sum(1 for r in records if r and r.get("injection_withheld")),
    }


def promotion_gate(gold: List[str], baseline: List[Optional[Dict[str, Any]]],
                   candidate: List[Optional[Dict[str, Any]]], adequate_set: bool) -> Dict[str, Any]:
    """PROTOCOL.md Section 6: every criterion must hold; the default is kept otherwise."""
    base, cand = [predicted(r) for r in baseline], [predicted(r) for r in candidate]
    ci = bootstrap(gold, cand, base, seed=20261025)
    false_support = lambda pred: sum(1 for g, p in zip(gold, pred) if g != S and p == S)  # noqa: E731
    fails = lambda recs: sum(1 for r in recs if is_failure(r))  # noqa: E731
    criteria = {
        "claim_set_adequate": adequate_set,
        "delta_macro_f1_ci_lower_above_0": ci[0] > 0,
        "false_support_not_higher": false_support(cand) <= false_support(base),
        "operational_failures_not_higher": fails(candidate) <= fails(baseline),
    }
    b = sum(1 for g, x, y in zip(gold, base, cand) if x == g and y != g)
    c = sum(1 for g, x, y in zip(gold, base, cand) if y == g and x != g)
    return {"delta_macro_f1": round(macro_f1(gold, cand) - macro_f1(gold, base), 4), "delta_ci95": ci,
            "accuracy_mcnemar_p": round(mcnemar(b, c), 4), "criteria": criteria,
            "promote": all(criteria.values())}


def adequate(gold: List[str]) -> bool:
    """PROTOCOL.md Section 5: at least 300 eligible claims and 100 per class."""
    counts = Counter(gold)
    return len(gold) >= 300 and all(counts[c] >= 100 for c in CLASSES)


def _load(path: Path) -> Dict[str, Dict[str, Any]]:
    return {r["claim_id"]: r for r in (json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evaluation.studio_eval.score")
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--candidate", type=Path)
    args = parser.parse_args(argv)
    base = _load(args.records)
    ids = sorted(base)
    gold = [base[i]["gold"] for i in ids]
    out: Dict[str, Any] = {"baseline": summarise(gold, [base.get(i) for i in ids]), "adequate_claim_set": adequate(gold)}
    if args.candidate:
        cand = _load(args.candidate)
        out["candidate"] = summarise(gold, [cand.get(i) for i in ids])
        out["gate"] = promotion_gate(gold, [base.get(i) for i in ids], [cand.get(i) for i in ids], adequate(gold))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
