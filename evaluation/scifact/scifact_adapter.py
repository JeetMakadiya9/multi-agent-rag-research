"""Strict SciFact-Orig data adapters with runtime/gold separation.

This module only loads data and adapts the evidence corpus to the existing
``rag.create_chunks`` contract. It does not implement or change retrieval.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SPLITS = ("train", "dev", "test")
VALID_GOLD_LABELS = {"SUPPORT", "CONTRADICT"}


@dataclass(frozen=True)
class CorpusDocument:
    doc_id: int
    title: str
    abstract: tuple[str, ...]
    structured: bool


@dataclass(frozen=True)
class RuntimeClaim:
    """Only fields permitted to enter retrieval or verification."""

    claim_id: int
    text: str


@dataclass(frozen=True)
class GoldRationale:
    label: str
    sentence_indices: tuple[int, ...]


@dataclass(frozen=True)
class GoldEvidenceDocument:
    doc_id: int
    rationales: tuple[GoldRationale, ...]


@dataclass(frozen=True)
class ClaimAnnotations:
    """Scoring-only annotations. Never pass this object to a model/provider."""

    labels_available: bool
    evidence_field_present: bool
    evidence: tuple[GoldEvidenceDocument, ...] | None
    cited_doc_ids: tuple[int, ...]


@dataclass(frozen=True)
class ScifactClaimRecord:
    runtime: RuntimeClaim
    annotations: ClaimAnnotations


@dataclass(frozen=True)
class SciFactDataset:
    dataset_root: Path
    split: str
    corpus_path: Path
    claims_path: Path
    corpus: tuple[CorpusDocument, ...]
    claims: tuple[ScifactClaimRecord, ...]


def _strict_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{where} must be an integer.")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
                if not isinstance(value, dict):
                    raise ValueError(f"Expected an object at {path}:{line_number}.")
                rows.append(value)
    except OSError as exc:
        raise OSError(f"Could not read SciFact file {path}: {exc}") from exc
    return rows


def _required_files(root: Path, split: str) -> tuple[Path, Path]:
    if split not in SPLITS:
        raise ValueError(f"Unsupported SciFact split {split!r}; choose one of {SPLITS}.")
    corpus_path = root / "corpus.jsonl"
    claims_path = root / f"claims_{split}.jsonl"
    missing = [str(path) for path in (corpus_path, claims_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required SciFact-Orig file(s) are missing: " + ", ".join(missing)
            + ". Supply a dataset root containing corpus.jsonl and the selected claims split. "
            "No download is performed automatically."
        )
    return corpus_path, claims_path


def load_corpus(path: str | Path) -> tuple[CorpusDocument, ...]:
    """Load corpus records and reject malformed types/duplicate document IDs."""
    corpus_path = Path(path).expanduser().resolve()
    if not corpus_path.is_file():
        raise FileNotFoundError(f"SciFact corpus file not found: {corpus_path}")
    docs: list[CorpusDocument] = []
    seen: set[int] = set()
    for row_number, row in enumerate(_read_jsonl(corpus_path), start=1):
        where = f"{corpus_path}:record {row_number}"
        doc_id = _strict_int(row.get("doc_id"), f"{where}.doc_id")
        title = row.get("title")
        abstract = row.get("abstract")
        structured = row.get("structured")
        if not isinstance(title, str) or not title.strip():
            raise ValueError(f"{where}.title must be a non-empty string.")
        if not isinstance(abstract, list) or any(
            not isinstance(sentence, str) or not sentence.strip() for sentence in abstract
        ):
            raise ValueError(f"{where}.abstract must be a list of non-empty sentence strings.")
        if type(structured) is not bool:
            raise ValueError(f"{where}.structured must be a boolean.")
        if doc_id in seen:
            raise ValueError(f"Duplicate SciFact doc_id {doc_id} in {corpus_path}.")
        seen.add(doc_id)
        docs.append(CorpusDocument(doc_id, title, tuple(abstract), structured))
    return tuple(docs)


def load_claims(path: str | Path, corpus: Iterable[CorpusDocument] | None = None) -> tuple[ScifactClaimRecord, ...]:
    """Load claims, placing annotations only under ``.annotations``.

    Missing ``evidence`` means labels are unavailable (as in public test data).
    An explicitly present empty evidence object is preserved as an empty gold
    annotation; it is never converted to an Experiment B verdict.
    """
    claims_path = Path(path).expanduser().resolve()
    if not claims_path.is_file():
        raise FileNotFoundError(f"SciFact claims file not found: {claims_path}")
    corpus_by_id = {doc.doc_id: doc for doc in corpus} if corpus is not None else None
    records: list[ScifactClaimRecord] = []
    seen: set[int] = set()
    for row_number, row in enumerate(_read_jsonl(claims_path), start=1):
        where = f"{claims_path}:record {row_number}"
        claim_id = _strict_int(row.get("id"), f"{where}.id")
        text = row.get("claim")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{where}.claim must be a non-empty string.")
        if claim_id in seen:
            raise ValueError(f"Duplicate SciFact claim id {claim_id} in {claims_path}.")
        seen.add(claim_id)

        evidence_present = "evidence" in row
        raw_evidence = row.get("evidence")
        if evidence_present and not isinstance(raw_evidence, dict):
            raise ValueError(f"{where}.evidence must be an object when present.")
        evidence_docs: list[GoldEvidenceDocument] = []
        if evidence_present:
            for raw_doc_id, raw_rationales in raw_evidence.items():
                try:
                    doc_id = int(raw_doc_id)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{where}.evidence key {raw_doc_id!r} is not a document ID.") from exc
                if str(doc_id) != str(raw_doc_id):
                    raise ValueError(f"{where}.evidence key {raw_doc_id!r} is not a canonical integer ID.")
                if corpus_by_id is not None and doc_id not in corpus_by_id:
                    raise ValueError(f"{where} references evidence document {doc_id}, absent from corpus.")
                if not isinstance(raw_rationales, list) or not raw_rationales:
                    raise ValueError(f"{where}.evidence[{raw_doc_id!r}] must be a non-empty rationale list.")
                rationales: list[GoldRationale] = []
                for rationale_index, raw_rationale in enumerate(raw_rationales):
                    label = raw_rationale.get("label") if isinstance(raw_rationale, dict) else None
                    indices = raw_rationale.get("sentences") if isinstance(raw_rationale, dict) else None
                    if not isinstance(label, str) or label not in VALID_GOLD_LABELS:
                        raise ValueError(f"{where}.evidence[{raw_doc_id!r}][{rationale_index}].label must be SUPPORT or CONTRADICT.")
                    if not isinstance(indices, list) or not indices:
                        raise ValueError(f"{where}.evidence[{raw_doc_id!r}][{rationale_index}].sentences must be a non-empty list.")
                    sentence_indices = tuple(_strict_int(x, f"{where}.evidence sentence index") for x in indices)
                    if any(index < 0 for index in sentence_indices):
                        raise ValueError(f"{where} contains a negative gold sentence index.")
                    if corpus_by_id is not None and any(index >= len(corpus_by_id[doc_id].abstract) for index in sentence_indices):
                        raise ValueError(f"{where} contains a gold sentence index outside document {doc_id}.")
                    rationales.append(GoldRationale(label, sentence_indices))
                evidence_docs.append(GoldEvidenceDocument(doc_id, tuple(rationales)))

        raw_cited = row.get("cited_doc_ids", [])
        if not isinstance(raw_cited, list):
            raise ValueError(f"{where}.cited_doc_ids must be a list when present.")
        cited_doc_ids = tuple(_strict_int(item, f"{where}.cited_doc_ids item") for item in raw_cited)
        annotations = ClaimAnnotations(
            labels_available=evidence_present,
            evidence_field_present=evidence_present,
            evidence=tuple(evidence_docs) if evidence_present else None,
            cited_doc_ids=cited_doc_ids,
        )
        records.append(ScifactClaimRecord(RuntimeClaim(claim_id, text), annotations))
    return tuple(records)


def load_scifact_dataset(dataset_root: str | Path, split: str) -> SciFactDataset:
    root = Path(dataset_root).expanduser().resolve()
    corpus_path, claims_path = _required_files(root, split)
    corpus = load_corpus(corpus_path)
    claims = load_claims(claims_path, corpus)
    return SciFactDataset(root, split, corpus_path, claims_path, corpus, claims)


def corpus_to_rag_chunks(corpus: Iterable[CorpusDocument], rag_module: Any) -> list[dict[str, Any]]:
    """Use the frozen RAG chunker and add a scoring-safe source mapping.

    ``rag.create_chunks`` preserves filename/page/chunk IDs but drops arbitrary
    PageRecord metadata. To retain a precise mapping without touching rag.py,
    each source abstract sentence is given a synthetic one-based page number;
    the wrapper adds its original zero-based sentence index and document ID to
    the resulting chunk. The title is included in searchable text. No claim or
    gold-annotation fields are accepted by this function.
    """
    if not callable(getattr(rag_module, "create_chunks", None)):
        raise TypeError("rag_module must expose the existing create_chunks() function.")
    result: list[dict[str, Any]] = []
    next_chunk_id = 0
    for doc in corpus:
        pages = [
            {
                "page": sentence_index + 1,
                "text": f"Title: {doc.title}. Abstract sentence: {sentence}",
            }
            for sentence_index, sentence in enumerate(doc.abstract)
        ]
        if not pages:
            continue
        filename = f"scifact_doc_{doc.doc_id}"
        source_chunks = rag_module.create_chunks(pages, filename)
        local_ids = {
            chunk.get("chunk_id"): next_chunk_id + offset
            for offset, chunk in enumerate(source_chunks)
            if isinstance(chunk.get("chunk_id"), int) and not isinstance(chunk.get("chunk_id"), bool)
        }
        if len(local_ids) != len(source_chunks):
            raise ValueError(f"RAG chunker returned a missing or non-integer chunk ID for document {doc.doc_id}.")
        for chunk in source_chunks:
            page_number = chunk.get("page")
            if isinstance(page_number, bool) or not isinstance(page_number, int):
                raise ValueError(f"RAG chunk from SciFact document {doc.doc_id} has no integer source page.")
            sentence_index = page_number - 1
            if sentence_index < 0 or sentence_index >= len(doc.abstract):
                raise ValueError(f"RAG chunk from SciFact document {doc.doc_id} cannot be mapped to a source sentence.")
            mapped = dict(chunk)
            old_chunk_id = chunk.get("chunk_id")
            mapped["chunk_id"] = local_ids[old_chunk_id]
            mapped["scifact_chunk_uid"] = f"scifact-{doc.doc_id}-{old_chunk_id}"
            for link_key in ("previous_chunk_id", "next_chunk_id"):
                linked_id = chunk.get(link_key)
                if linked_id is not None:
                    if linked_id not in local_ids:
                        raise ValueError(f"RAG chunk {old_chunk_id} has an unmappable {link_key} for document {doc.doc_id}.")
                    mapped[link_key] = local_ids[linked_id]
            mapped["document_id"] = str(doc.doc_id)
            mapped["scifact_doc_id"] = doc.doc_id
            mapped["scifact_title"] = doc.title
            mapped["scifact_structured"] = doc.structured
            mapped["scifact_sentence_indices"] = [sentence_index]
            mapped["scifact_sentence_texts"] = [doc.abstract[sentence_index]]
            result.append(mapped)
        next_chunk_id += len(source_chunks)
    return result


def runtime_claim_payload(claim: RuntimeClaim) -> dict[str, Any]:
    """Return the only claim fields allowed into runtime stages."""
    return {"claim_id": claim.claim_id, "claim": claim.text}
