"""H5 post-annotation tooling: validation, inter-annotator agreement, adjudication package, final gold.

Used only after humans have filled the annotator files. Never assigns a label
itself: the final gold label is the agreed label, else the human adjudicator's.

Pre-registered quality flag (h5_protocol.md §10): GOLD_QUALITY_WARNING if
Cohen's kappa < 0.60 or any class's specific agreement < 0.50.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any, Mapping, Sequence

LABELS = ("SUPPORT", "CONTRADICT", "NEUTRAL")
ANNOTATOR_FIELDS = ("pair_id", "claim_text", "evidence_text", "gold_label", "annotator_note")
KAPPA_WARNING_BELOW = 0.60
SPECIFIC_AGREEMENT_WARNING_BELOW = 0.50
ADJUDICATION_SEED = "h5-2026-v1-adjudication"


def validate_annotator_rows(rows: Sequence[Mapping[str, Any]], *, require_labels: bool) -> None:
    ids = [r.get("pair_id") for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate pair_id in annotator file")
    for row in rows:
        if tuple(sorted(row)) != tuple(sorted(ANNOTATOR_FIELDS)):
            raise ValueError(f"annotator row {row.get('pair_id')} has fields {sorted(row)}")
        label = row["gold_label"]
        if label is None and require_labels:
            raise ValueError(f"unlabelled pair {row['pair_id']}")
        if label is not None and label not in LABELS:
            raise ValueError(f"invalid label {label!r} for {row['pair_id']}")


def agreement(a_rows: Sequence[Mapping[str, Any]], b_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    validate_annotator_rows(a_rows, require_labels=True)
    validate_annotator_rows(b_rows, require_labels=True)
    a = {r["pair_id"]: r for r in a_rows}
    b = {r["pair_id"]: r for r in b_rows}
    if set(a) != set(b):
        raise ValueError("annotator files cover different pairs")
    for pid in a:
        if (a[pid]["claim_text"], a[pid]["evidence_text"]) != (b[pid]["claim_text"], b[pid]["evidence_text"]):
            raise ValueError(f"pair {pid} text differs between annotator files")
    n = len(a)
    confusion = {x: {y: 0 for y in LABELS} for x in LABELS}
    for pid in a:
        confusion[a[pid]["gold_label"]][b[pid]["gold_label"]] += 1
    observed = sum(confusion[x][x] for x in LABELS) / n
    row_tot = {x: sum(confusion[x].values()) for x in LABELS}
    col_tot = {y: sum(confusion[x][y] for x in LABELS) for y in LABELS}
    expected = sum(row_tot[x] * col_tot[x] for x in LABELS) / (n * n)
    kappa = (observed - expected) / (1 - expected) if expected < 1 else None
    specific = {x: (2 * confusion[x][x] / (row_tot[x] + col_tot[x]) if row_tot[x] + col_tot[x] else None) for x in LABELS}
    warning = (kappa is None or kappa < KAPPA_WARNING_BELOW or
               any(v is not None and v < SPECIFIC_AGREEMENT_WARNING_BELOW for v in specific.values()))
    return {"n": n, "raw_agreement": observed, "cohens_kappa": kappa, "per_class_specific_agreement": specific,
            "confusion_annotator_a_rows_b_cols": confusion, "disagreements": n - sum(confusion[x][x] for x in LABELS),
            "GOLD_QUALITY_WARNING": warning,
            "warning_rule": f"kappa < {KAPPA_WARNING_BELOW} or any specific agreement < {SPECIFIC_AGREEMENT_WARNING_BELOW} "
                            "(pre-registered)"}


def adjudication_rows(a_rows, b_rows) -> list[dict[str, Any]]:
    """Disagreements only; labels anonymised and order randomised per pair; nothing else revealed."""
    b = {r["pair_id"]: r for r in b_rows}
    out = []
    for row in a_rows:
        other = b[row["pair_id"]]
        if row["gold_label"] == other["gold_label"]:
            continue
        flip = int(hashlib.sha256(f"{ADJUDICATION_SEED}|{row['pair_id']}".encode()).hexdigest(), 16) % 2
        first, second = (other, row) if flip else (row, other)
        out.append({"pair_id": row["pair_id"], "claim_text": row["claim_text"], "evidence_text": row["evidence_text"],
                    "label_1": first["gold_label"], "label_2": second["gold_label"],
                    "adjudicated_label": None, "adjudication_note": None})
    return sorted(out, key=lambda r: hashlib.sha256(f"{ADJUDICATION_SEED}|order|{r['pair_id']}".encode()).hexdigest())


def final_gold(manifest_rows, a_rows, b_rows, adjudicated_rows=()) -> list[dict[str, Any]]:
    a = {r["pair_id"]: r["gold_label"] for r in a_rows}
    b = {r["pair_id"]: r["gold_label"] for r in b_rows}
    adj = {r["pair_id"]: r for r in adjudicated_rows}
    out = []
    for row in manifest_rows:
        pid = row["pair_id"]
        la, lb = a[pid], b[pid]
        needed = la != lb
        new = {**row, "annotator_a_label": la, "annotator_b_label": lb, "adjudication_required": needed}
        if not needed:
            new.update(gold_label=la, annotation_status="AGREED", adjudicated_label=None, adjudication_note=None)
        elif pid in adj and adj[pid]["adjudicated_label"] in LABELS:
            new.update(gold_label=adj[pid]["adjudicated_label"], annotation_status="ADJUDICATED",
                       adjudicated_label=adj[pid]["adjudicated_label"], adjudication_note=adj[pid]["adjudication_note"])
        else:
            new.update(gold_label=None, annotation_status="AWAITING_ADJUDICATION")
        out.append(new)
    return out


def label_distribution(rows) -> dict[str, int]:
    return dict(sorted(Counter(r["gold_label"] for r in rows).items(), key=lambda kv: str(kv[0])))


def load_jsonl(path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
