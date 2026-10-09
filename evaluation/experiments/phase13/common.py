"""Shared, read-only helpers for Phase 13 analysis and experiments."""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data/scifact/data"
RESULTS_DIR = ROOT / "evaluation/experiments/results"
FROZEN_RUN_DIR = RESULTS_DIR / "phase10_full_dev"
OUTPUT_DIR = RESULTS_DIR / "phase13_improvement"

# Directories that Phase 13 must never modify (hash-verified before/after).
PROTECTED_DIRS = (
    RESULTS_DIR / "phase10_full_dev",
    RESULTS_DIR / "phase11_final",
    RESULTS_DIR / "phase11_results",
    RESULTS_DIR / "phase12_diagnosis",
    DATA_DIR,
)
# Frozen verifier/retrieval sources (Phase 10F FROZEN_EXPECTED plus adapters).
PROTECTED_SOURCES = (
    "rag.py", "research_pipeline.py", "claim_verification.py", "verification_units.py",
    "experiment_b_verification.py", "llm.py", "research_retrieval.py",
    "research_verification_capability.py",
    "evaluation/scifact/scifact_adapter.py", "evaluation/scifact/metrics.py",
    "evaluation/scifact/run_scifact_evaluation.py",
    "evaluation/experiments/system_a_rag.py", "evaluation/experiments/system_b_verified_rag.py",
    "evaluation/experiments/system_c_multi_agent.py", "evaluation/experiments/local_model.py",
    "evaluation/experiments/evaluator.py", "evaluation/experiments/run_phase10f.py",
    "evaluation/experiments/run_phase10_dev.py", "evaluation/experiments/experiment_config.py",
)

LABELS = ("SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE")
GOLD_LABELS = ("SUPPORTED", "CONTRADICTED")
BUCKETS = ("A_top2", "B_rank3_8", "C_absent_top8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def snapshot_protected() -> dict[str, str]:
    """Hash every protected file so before/after comparisons are exact."""
    hashes: dict[str, str] = {}
    for directory in PROTECTED_DIRS:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                hashes[path.relative_to(ROOT).as_posix()] = sha256_file(path)
    for rel in PROTECTED_SOURCES:
        hashes[rel] = sha256_file(ROOT / rel)
    return hashes


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=False)
        handle.write("\n")
    tmp.replace(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


@dataclass(frozen=True)
class GoldInfo:
    claim_id: int
    claim: str
    label: str | None            # SUPPORTED / CONTRADICTED / None (unscored)
    label_status: str
    rationales: tuple[tuple[int, tuple[int, ...], str], ...]   # (doc, sentences, SUPPORT|CONTRADICT)

    @property
    def sentences(self) -> set[tuple[int, int]]:
        return {(doc, s) for doc, sents, _ in self.rationales for s in sents}

    @property
    def docs(self) -> set[int]:
        return {doc for doc, _, _ in self.rationales}

    @property
    def multi_sentence(self) -> bool:
        return any(len(sents) >= 2 for _, sents, _ in self.rationales)


def load_dataset():
    from evaluation.scifact.scifact_adapter import load_scifact_dataset
    return load_scifact_dataset(DATA_DIR, "dev")


def gold_by_claim(dataset) -> dict[int, GoldInfo]:
    from evaluation.experiments.evaluator import gold_claim_label
    result: dict[int, GoldInfo] = {}
    for record in dataset.claims:
        label, status = gold_claim_label(record.annotations)
        rationales = tuple(
            (doc.doc_id, tuple(r.sentence_indices), r.label)
            for doc in (record.annotations.evidence or ())
            for r in doc.rationales
        )
        result[record.runtime.claim_id] = GoldInfo(record.runtime.claim_id, record.runtime.text,
                                                    label, status, rationales)
    return result


def chunk_keys(chunk: Mapping[str, Any]) -> set[tuple[int, int]]:
    try:
        doc = int(chunk.get("scifact_doc_id"))
    except (TypeError, ValueError):
        return set()
    return {(doc, int(i)) for i in chunk.get("scifact_sentence_indices", []) or []}


def first_gold_rationale_rank(gold: GoldInfo, chunks: Sequence[Mapping[str, Any]]) -> int | None:
    sentences = gold.sentences
    for chunk in chunks:
        if chunk_keys(chunk) & sentences:
            return int(chunk["rank"])
    return None


def rationale_bucket(rank: int | None) -> str:
    if rank is None:
        return "C_absent_top8"
    return "A_top2" if rank <= 2 else "B_rank3_8"


def complete_rationale_within(gold: GoldInfo, chunks: Sequence[Mapping[str, Any]], k: int) -> bool:
    """True when every sentence of at least one gold rationale is in the top-k chunks."""
    present: set[tuple[int, int]] = set()
    for chunk in chunks:
        if int(chunk["rank"]) <= k:
            present |= chunk_keys(chunk)
    return any(all((doc, s) in present for s in sents) for doc, sents, _ in gold.rationales)


def gold_doc_rank(gold: GoldInfo, chunks: Sequence[Mapping[str, Any]]) -> int | None:
    for chunk in chunks:
        try:
            if int(chunk.get("scifact_doc_id")) in gold.docs:
                return int(chunk["rank"])
        except (TypeError, ValueError):
            continue
    return None


def load_frozen_rows() -> list[dict[str, Any]]:
    return read_jsonl(FROZEN_RUN_DIR / "per_example.jsonl")


def frozen_retrieval_by_claim(rows: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    """Frozen 10F retrieval per claim; prefer B's own record, then A, then C.

    All three systems call the same frozen retriever with the same query.
    """
    by_claim: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_claim.setdefault(int(row["claim_id"]), {})[row["system"]] = row
    result: dict[int, dict[str, Any]] = {}
    for claim_id, systems in by_claim.items():
        for name in ("B", "A", "C"):
            row = systems.get(name)
            if row and row.get("retrieved_chunks"):
                result[claim_id] = {"source_system": name, "chunks": row["retrieved_chunks"]}
                break
    return result


# ------------------------------------------------------------------ metrics

def classification_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Exactly the frozen Phase 10F classification definition (imported, not copied)."""
    from evaluation.experiments.run_phase10f import _classification
    return _classification(rows)


def insufficient_rate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = [r for r in rows if r.get("gold_label") in GOLD_LABELS and r.get("status") == "SUCCESS"
             and r.get("predicted_label") in LABELS]
    n = sum(r["predicted_label"] == "INSUFFICIENT_EVIDENCE" for r in valid)
    return {"numerator": n, "denominator": len(valid), "value": n / len(valid) if valid else None}


def citation_micro(rows: Sequence[Mapping[str, Any]], gold: Mapping[int, GoldInfo]) -> dict[str, Any]:
    """Micro document-level selected-evidence P/R/F1 (Phase 11B definition)."""
    tp = fp = fn = 0
    empty = 0
    for r in rows:
        if r.get("status") != "SUCCESS" or r.get("gold_label") not in GOLD_LABELS:
            continue
        docs = gold[int(r["claim_id"])].docs
        selected = set(int(x) for x in r.get("selected_evidence_doc_ids") or [])
        empty += not selected
        tp += len(selected & docs)
        fp += len(selected - docs)
        fn += len(docs - selected)
    p = tp / (tp + fp) if tp + fp else None
    rc = tp / (tp + fn) if tp + fn else None
    f1 = 2 * p * rc / (p + rc) if p and rc else (0.0 if p == 0 or rc == 0 else None)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": rc, "f1": f1,
            "valid_empty_selections": empty,
            "definition": "micro document-level overlap of verifier-selected evidence docs vs gold docs "
                          "(Phase 11B definition); overlap is not entailment"}


def stance_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for r in rows:
        if r.get("status") != "SUCCESS" or r.get("gold_label") not in GOLD_LABELS:
            continue
        for a in r.get("assessments") or []:
            counts[str(a.get("stance"))] += 1
    return dict(counts)


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def median(values: Iterable[float]) -> float | None:
    values = sorted(values)
    if not values:
        return None
    mid = len(values) // 2
    return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2
