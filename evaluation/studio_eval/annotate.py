"""Annotation and freeze workflow for the Research Studio evaluation (PROTOCOL.md Sections 2-4).

Three steps, each refusing to proceed when the protocol's conditions are not met:

1. ``tasks``: from a draft packet file (claims and fixed evidence packets, **no labels**), write one blinded task file
   per annotator. Each annotator gets the claims in a different order and sees no model output and no other label.
2. ``agreement``: from the two annotators' files, report Cohen's kappa and the raw agreement rate (reported before
   adjudication, Section 4.2), and write the list of disagreements for the adjudicator.
3. ``freeze``: from the draft, both annotation files and the adjudication file, write the final claims file in the
   format ``run_eval.py`` reads, and ``FREEZE.json`` with its SHA-256. Claims the adjudicator marks ``AMBIGUOUS``
   are excluded and counted. The overlap check runs against the SciFact claim files and any further used claim sets.

Labels come only from people. Nothing here calls a model, and no label is ever filled in automatically: a missing or
invalid label stops the workflow. Data files stay private (outside the public repository).

Usage:
    python -m evaluation.studio_eval.annotate tasks --draft DRAFT.jsonl --out DIR [--seed-a N --seed-b N]
    python -m evaluation.studio_eval.annotate agreement --draft DRAFT.jsonl --a A.jsonl --b B.jsonl --out DIR
    python -m evaluation.studio_eval.annotate freeze --draft DRAFT.jsonl --a A.jsonl --b B.jsonl \
        --adjudication ADJ.jsonl --out DIR [--exclude-claims USED.jsonl ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from evaluation.studio_eval.run_eval import (GOLD_LABELS, MAX_CLAIM_CHARS, MAX_PASSAGES, ProtocolError, _normalise,
                                             check_overlap, load_claims, sha256_file)

AMBIGUOUS = "AMBIGUOUS"
MIN_TOTAL, MIN_PER_CLASS = 300, 100                       # PROTOCOL.md Section 5 (sample size)
PROVENANCE_FIELDS = ("claim_author", "written_at", "source")
PASSAGE_FIELDS = ("url", "text", "licence")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ProtocolError(f"{Path(path).name} line {number}: not JSON ({exc.msg})") from None
    return rows


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows),
                          encoding="utf-8", newline="\n")


def load_draft(path: Path) -> List[Dict[str, Any]]:
    """The draft packet file: claims with fixed packets and provenance, and no labels (annotation is blind)."""
    rows, seen = _read_jsonl(path), set()
    if not rows:
        raise ProtocolError("the draft file has no claims")
    for n, r in enumerate(rows, 1):
        problems = []
        cid, claim, passages, prov = r.get("claim_id"), r.get("claim"), r.get("passages"), r.get("provenance")
        if "label" in r or "gold" in r:
            problems.append("a draft must not carry a label (annotation is blind)")
        if not isinstance(cid, str) or not cid.strip() or cid in seen:
            problems.append("claim_id must be a unique non-empty string")
        if not isinstance(claim, str) or not claim.strip() or len(claim) > MAX_CLAIM_CHARS:
            problems.append(f"claim must be 1-{MAX_CLAIM_CHARS} characters")
        if not isinstance(passages, list) or not 1 <= len(passages) <= MAX_PASSAGES:
            problems.append(f"passages must be a list of 1-{MAX_PASSAGES} items")
        elif any(not isinstance(p, dict) or any(not str(p.get(k) or "").strip() for k in PASSAGE_FIELDS)
                 for p in passages):
            problems.append("every passage needs non-empty url, text and licence")
        if not isinstance(prov, dict) or any(not str(prov.get(k) or "").strip() for k in PROVENANCE_FIELDS):
            problems.append(f"provenance needs {', '.join(PROVENANCE_FIELDS)}")
        if problems:
            raise ProtocolError(f"draft line {n}: " + "; ".join(problems))
        seen.add(cid)
    return rows


def packet_sha256(row: Dict[str, Any]) -> str:
    """Identifies a claim's text and packet exactly, so annotators' labels can be tied to the packet they saw."""
    payload = {"claim": row["claim"], "passages": [{k: p.get(k) for k in ("url", "title", "text")}
                                                   for p in row["passages"]]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def make_tasks(draft: List[Dict[str, Any]], seed: int) -> List[Dict[str, Any]]:
    """One annotator's blinded task list: claim, packet and an empty label, in an order of its own."""
    rows = [{"claim_id": r["claim_id"], "packet_sha256": packet_sha256(r), "claim": r["claim"],
             "passages": [{"n": i, "title": p.get("title", ""), "url": p["url"], "text": p["text"]}
                          for i, p in enumerate(r["passages"], 1)],
             "label": None, "note": ""} for r in draft]
    random.Random(seed).shuffle(rows)
    return rows


def load_annotations(path: Path, draft: List[Dict[str, Any]], who: str,
                     allowed: Sequence[str] = GOLD_LABELS) -> Dict[str, str]:
    """An annotator's (or adjudicator's) labels by claim_id; every label must be valid and tied to the right packet."""
    packets = {r["claim_id"]: packet_sha256(r) for r in draft}
    labels: Dict[str, str] = {}
    for n, r in enumerate(_read_jsonl(path), 1):
        cid = r.get("claim_id")
        if cid not in packets:
            raise ProtocolError(f"{who} line {n}: unknown claim_id {cid!r}")
        if cid in labels:
            raise ProtocolError(f"{who} line {n}: duplicate claim_id {cid!r}")
        if r.get("packet_sha256") not in (None, packets[cid]):
            raise ProtocolError(f"{who} line {n}: claim {cid!r} was labelled against a different packet")
        if r.get("label") not in allowed:
            raise ProtocolError(f"{who} line {n}: label {r.get('label')!r} is not one of {', '.join(allowed)}")
        labels[cid] = r["label"]
    return labels


def require_complete(labels: Dict[str, str], draft: List[Dict[str, Any]], who: str) -> None:
    missing = [r["claim_id"] for r in draft if r["claim_id"] not in labels]
    if missing:
        raise ProtocolError(f"{who} has no label for {len(missing)} claim(s), e.g. {missing[:3]}")


def cohen_kappa(a: Sequence[str], b: Sequence[str], labels: Sequence[str] = GOLD_LABELS) -> Optional[float]:
    n = len(a)
    if not n:
        return None
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb[k] for k in labels) / (n * n)
    return None if expected == 1 else (observed - expected) / (1 - expected)


def agreement(draft, a: Dict[str, str], b: Dict[str, str]) -> Dict[str, Any]:
    ids = [r["claim_id"] for r in draft]
    la, lb = [a[i] for i in ids], [b[i] for i in ids]
    kappa = cohen_kappa(la, lb)
    return {"claims": len(ids), "raw_agreement": round(sum(x == y for x, y in zip(la, lb)) / len(ids), 4),
            "cohen_kappa": None if kappa is None else round(kappa, 4),
            "disagreements": [i for i in ids if a[i] != b[i]],
            "confusion_a_by_b": {x: {y: sum(1 for i in ids if a[i] == x and b[i] == y) for y in GOLD_LABELS}
                                 for x in GOLD_LABELS}}


def overlap(claims: List[Dict[str, Any]], extra_files: Sequence[Path]) -> List[str]:
    """Claims equal (after normalisation) to a claim in a set the project has used: SciFact plus any given files."""
    hits = set(check_overlap(claims))
    used = set()
    for path in extra_files:
        for row in _read_jsonl(path):
            text = row.get("claim") or row.get("text") or row.get("question") or ""
            if text:
                used.add(_normalise(str(text)))
    hits.update(c["claim_id"] for c in claims if _normalise(c["claim"]) in used)
    return sorted(hits)


def freeze(draft, a, b, adjudication, extra_used: Sequence[Path], files: Dict[str, Path], out: Path) -> Dict[str, Any]:
    """Final labels: agreement where the annotators agree, the adjudicator's label where they do not."""
    stats = agreement(draft, a, b)
    unresolved = [i for i in stats["disagreements"] if i not in adjudication]
    if unresolved:
        raise ProtocolError(f"{len(unresolved)} disagreement(s) have no adjudicated label, e.g. {unresolved[:3]}")
    extra = sorted(set(adjudication) - set(stats["disagreements"]))
    if extra:
        raise ProtocolError(f"the adjudication file labels claims the annotators agreed on: {extra[:3]}")
    final, ambiguous = [], []
    for r in draft:
        cid = r["claim_id"]
        label = adjudication.get(cid, a[cid])
        if label == AMBIGUOUS:
            ambiguous.append(cid)
            continue
        final.append(dict(r, label=label, annotation={"a": a[cid], "b": b[cid], "adjudicated": cid in adjudication}))
    if not final:
        raise ProtocolError("no claim is left after excluding ambiguous ones")
    hits = overlap(final, extra_used)
    if hits:
        raise ProtocolError(f"{len(hits)} claim(s) match a claim set the project has used, e.g. {hits[:3]}; remove "
                            "them from the draft and annotate again")
    out.mkdir(parents=True, exist_ok=True)
    claims_path = out / "claims.jsonl"
    _write_jsonl(claims_path, final)
    load_claims(claims_path)                              # the harness's own validation must accept it
    counts = Counter(c["label"] for c in final)
    adequate = len(final) >= MIN_TOTAL and all(counts[k] >= MIN_PER_CLASS for k in GOLD_LABELS)
    record = {
        "claims_file": claims_path.name,
        "claims_sha256": sha256_file(claims_path),
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "claims": len(final), "class_counts": {k: counts[k] for k in GOLD_LABELS},
        "adequate_for_promotion": adequate,
        "annotation": {"annotators": 2, "adjudicator": "a third person",
                       "independent_of_model_output": True,
                       "cohen_kappa": stats["cohen_kappa"], "raw_agreement": stats["raw_agreement"],
                       "disagreements": len(stats["disagreements"]), "ambiguous_excluded": len(ambiguous)},
        "input_sha256": {k: sha256_file(v) for k, v in files.items()},
        "provenance": sorted({str(r["provenance"]["source"]) for r in final}),
        "overlap_check": {"used_sets": ["SciFact train/dev/test"] + [Path(p).name for p in extra_used],
                          "matches": 0},
        "protocol": "evaluation/studio_eval/PROTOCOL.md",
    }
    (out / "FREEZE.json").write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8",
                                     newline="\n")
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evaluation.studio_eval.annotate", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="step", required=True)
    t = sub.add_parser("tasks", help="write blinded task files for two annotators")
    t.add_argument("--draft", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--seed-a", type=int, default=1)
    t.add_argument("--seed-b", type=int, default=2)
    g = sub.add_parser("agreement", help="agreement before adjudication, and the disagreements to adjudicate")
    f = sub.add_parser("freeze", help="final labels, claims file and FREEZE.json")
    for p in (g, f):
        p.add_argument("--draft", type=Path, required=True)
        p.add_argument("--a", type=Path, required=True)
        p.add_argument("--b", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
    f.add_argument("--adjudication", type=Path, required=True)
    f.add_argument("--exclude-claims", type=Path, nargs="*", default=[],
                   help="further claim sets the project has used (WebRAG sets, demonstrations, ...)")
    args = parser.parse_args(argv)
    try:
        draft = load_draft(args.draft)
        if args.step == "tasks":
            if args.seed_a == args.seed_b:
                raise ProtocolError("the two annotators need different orders (different seeds)")
            args.out.mkdir(parents=True, exist_ok=True)
            _write_jsonl(args.out / "tasks_annotator_a.jsonl", make_tasks(draft, args.seed_a))
            _write_jsonl(args.out / "tasks_annotator_b.jsonl", make_tasks(draft, args.seed_b))
            print(f"{len(draft)} claims; task files written to {args.out}")
            return 0
        a = load_annotations(args.a, draft, "annotator A")
        b = load_annotations(args.b, draft, "annotator B")
        require_complete(a, draft, "annotator A")
        require_complete(b, draft, "annotator B")
        if args.step == "agreement":
            stats = agreement(draft, a, b)
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / "agreement.json").write_text(json.dumps({k: v for k, v in stats.items() if k != "disagreements"}
                                                                | {"disagreements": len(stats["disagreements"])},
                                                                indent=1) + "\n", encoding="utf-8")
            rows = {r["claim_id"]: r for r in draft}
            _write_jsonl(args.out / "to_adjudicate.jsonl",
                         [{"claim_id": i, "packet_sha256": packet_sha256(rows[i]), "claim": rows[i]["claim"],
                           "passages": rows[i]["passages"], "label_a": a[i], "label_b": b[i], "label": None,
                           "note": ""} for i in stats["disagreements"]])
            print(json.dumps({k: stats[k] for k in ("claims", "raw_agreement", "cohen_kappa")}
                             | {"disagreements": len(stats["disagreements"])}, indent=1))
            return 0
        adjudication = load_annotations(args.adjudication, draft, "adjudicator", GOLD_LABELS + (AMBIGUOUS,))
        record = freeze(draft, a, b, adjudication, args.exclude_claims,
                        {"draft": args.draft, "annotator_a": args.a, "annotator_b": args.b,
                         "adjudication": args.adjudication}, args.out)
    except ProtocolError as exc:
        parser.error(str(exc))
    print(json.dumps({k: record[k] for k in ("claims", "class_counts", "adequate_for_promotion", "claims_sha256")},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
