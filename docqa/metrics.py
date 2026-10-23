"""Metric definitions. Every rate is reported as {"value", "num", "den"}; ``value`` is None when ``den`` is 0.

Classification (claim verification):
* confusion[gold][pred] counts items;
* precision(c) = TP / #predicted c (None if nothing was predicted c); recall(c) = TP / #gold c (None if no gold c);
* F1(c) = 2PR / (P + R), taken as 0 when TP = 0 and the class occurs in gold or predictions;
* macro-F1 = mean F1 over the classes present in gold; accuracy = correct / n.

Text matching (answers): text is lower-cased, NFKC-normalised, dashes unified and whitespace collapsed; a phrase
matches only on alphanumeric boundaries, so "23" does not match "123" or "23.5".
"""
from __future__ import annotations

import re
import statistics
import unicodedata
from typing import Dict, Iterable, List, Optional, Sequence

LABELS = ("SUPPORTED", "CONTRADICTED", "INSUFFICIENT")


def rate(num: int, den: int) -> Dict[str, Optional[float]]:
    return {"value": round(num / den, 4) if den else None, "num": num, "den": den}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").lower()
    text = re.sub(r"[‐-―−]", "-", text)
    return re.sub(r"\s+", " ", text).strip()


def contains_phrase(text: str, phrase: str) -> bool:
    pattern = r"(?<![0-9a-z])" + re.escape(normalize(phrase)) + r"(?![0-9a-z]|\.[0-9])"
    return re.search(pattern, normalize(text)) is not None


def facts_found(text: str, facts: Sequence[Sequence[str]]) -> List[bool]:
    """Each fact is a list of acceptable surface forms; a fact is found if any form appears."""
    return [any(contains_phrase(text, form) for form in alternatives) for alternatives in facts]


def snippet_in(text: str, snippet: str) -> bool:
    return normalize(snippet) in normalize(text)


def classification_report(gold: Sequence[str], pred: Sequence[str], labels: Sequence[str] = LABELS) -> Dict:
    if len(gold) != len(pred):
        raise ValueError("gold and pred differ in length")
    confusion = {g: {p: 0 for p in labels} for g in labels}
    for g, p in zip(gold, pred):
        confusion[g][p] += 1
    per_class, f1s = {}, []
    for c in labels:
        tp = confusion[c][c]
        support = sum(confusion[c].values())
        predicted = sum(confusion[g][c] for g in labels)
        precision = tp / predicted if predicted else None
        recall = tp / support if support else None
        if precision and recall:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0 if (support or predicted) else None
        per_class[c] = {"precision": _round(precision), "recall": _round(recall), "f1": _round(f1),
                        "tp": tp, "support": support, "predicted": predicted}
        if support:
            f1s.append(f1)
    correct = sum(confusion[c][c] for c in labels)
    return {"n": len(gold), "accuracy": rate(correct, len(gold)), "macro_f1": _round(sum(f1s) / len(f1s)) if f1s else None,
            "per_class": per_class, "confusion": confusion}


def latency_summary(seconds: Iterable[float]) -> Dict[str, Optional[float]]:
    values = sorted(float(s) for s in seconds)
    if not values:
        return {"n": 0, "mean": None, "median": None, "p90": None, "max": None, "total": 0.0}
    p90 = values[min(len(values) - 1, int(round(0.9 * (len(values) - 1))))]
    return {"n": len(values), "mean": round(statistics.fmean(values), 2), "median": round(statistics.median(values), 2),
            "p90": round(p90, 2), "max": round(values[-1], 2), "total": round(sum(values), 1)}


def counts(values: Iterable[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


def _round(x: Optional[float]) -> Optional[float]:
    return None if x is None else round(x, 4)
