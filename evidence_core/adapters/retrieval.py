"""Adapt existing retrieval outputs into RetrievedCandidate contracts (read-only).

Produces Source -> DocumentVersion -> Span -> RetrievedCandidate. It never
creates Evidence: a retrieval hit is not evidence until a separate, recorded
admission step makes it a candidate evidentiary unit for a specific claim.

Two inputs are supported:
* recorded chunk references as persisted by the frozen Phase 10F / Phase 13
  runs (``retrieved_chunks`` rows), and
* live ``research_retrieval.RetrievalResponse`` objects from ExistingRAGProvider.
Retrieval behaviour itself is untouched; this module only reads outputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

from ..contracts import (
    DocumentVersion, Provenance, RetrievalProvenance, RetrievedCandidate, Source, SourceType, Span, SpanKind,
    content_hash, stable_id,
)

ADAPTER = "evidence_core.adapters.retrieval"
ADAPTER_VERSION = "slice0-1"
RELEVANCE_NOTE = "ranked by the frozen hybrid retriever (relevance, not support)"

# Optional lookup: canonical document key -> (title, ordered sentence/segment texts)
DocumentLookup = Callable[[str], Optional[Tuple[str, Sequence[str]]]]


@dataclass(frozen=True)
class AdaptedRetrieval:
    sources: Tuple[Source, ...]
    document_versions: Tuple[DocumentVersion, ...]
    spans: Tuple[Span, ...]
    candidates: Tuple[RetrievedCandidate, ...]

    def objects(self):
        return (*self.sources, *self.document_versions, *self.spans, *self.candidates)


@dataclass(frozen=True)
class _Entry:
    ref: Mapping[str, Any]
    namespace: str
    scores: Mapping[str, Optional[float]]
    method: str
    provider: str


def retrieval_run_id(query: str, run_key: str) -> str:
    return stable_id("rrun", query, run_key)


def _doc_objects(canonical: str, namespace: str, lookup: Optional[DocumentLookup]):
    source_id = Source.make_id(canonical)
    looked = lookup(canonical) if lookup else None
    title, version_key, structure = None, None, {}
    if looked:
        title, segments = looked
        version_key = content_hash("\n".join([title or "", *segments]))
        structure = {"segment_count": len(segments), "segmentation": "provided-by-lookup"}
    # Source type is UNKNOWN: retrieval outputs do not state what kind of source a document is.
    source = Source(source_id=source_id, canonical_identity=canonical, source_type=SourceType.UNKNOWN,
                    metadata={"namespace": namespace},
                    provenance=Provenance(producer=ADAPTER, producer_version=ADAPTER_VERSION))
    docv = DocumentVersion(document_version_id=DocumentVersion.make_id(source_id, version_key), source_id=source_id,
                           content_hash=version_key, title=title, structure=structure,
                           metadata={"version_known": version_key is not None})
    return source, docv


def _adapt(entries: Sequence[_Entry], *, query: str, run_key: str,
           lookup: Optional[DocumentLookup]) -> AdaptedRetrieval:
    run_id = retrieval_run_id(query, run_key)
    sources, docvs, spans, cands = {}, {}, {}, []
    for entry in entries:
        ref = entry.ref
        doc_key = ref.get("scifact_doc_id", ref.get("document_id"))
        if doc_key in (None, ""):
            raise ValueError(f"Retrieved item at rank {ref.get('rank')} has no document identity")
        source, docv = _doc_objects(f"{entry.namespace}:{doc_key}", entry.namespace, lookup)
        sources[source.source_id], docvs[docv.document_version_id] = source, docv
        indices = tuple(int(i) for i in ref.get("scifact_sentence_indices") or ())
        texts = list(ref.get("scifact_sentence_texts") or ())
        if indices:
            locator, span_text, kind = ("sentences", indices), " ".join(texts) if texts else ref["text"], SpanKind.SENTENCE
        else:
            locator, span_text, kind = ("chunk", str(ref["chunk_id"])), ref["text"], SpanKind.PASSAGE
        span = Span(span_id=Span.make_id(docv.document_version_id, locator), document_version_id=docv.document_version_id,
                    text=span_text, kind=kind, sentence_indices=indices)
        spans[span.span_id] = span
        rank = int(ref["rank"])
        cands.append(RetrievedCandidate(
            candidate_id=stable_id("cand", run_id, rank, str(ref["chunk_id"])),
            span_id=span.span_id, document_version_id=docv.document_version_id, source_id=source.source_id,
            presented_text=ref["text"],
            retrieval=RetrievalProvenance(query=query, method=entry.method, rank=rank, retrieval_run_id=run_id,
                                          scores=dict(entry.scores), provider=entry.provider, reason=RELEVANCE_NOTE),
            metadata={"chunk_id": str(ref["chunk_id"]), "legacy_evidence_id": ref.get("evidence_id")}))
    return AdaptedRetrieval(tuple(sources.values()), tuple(docvs.values()), tuple(spans.values()), tuple(cands))


def adapt_recorded_refs(refs: Sequence[Mapping[str, Any]], *, query: str, run_key: str,
                        method: str = "existing_advanced_rag", provider: str = "ExistingRAGProvider",
                        namespace: str = "scifact", lookup: Optional[DocumentLookup] = None) -> AdaptedRetrieval:
    """Recorded chunk references (frozen runs) -> candidates.

    Each ref carries rank, chunk_id, a document identifier, the emitted text
    and the recorded ``retrieval_score``. SciFact refs also carry 0-based
    sentence indices and raw sentence text, which become sentence Spans.
    """
    entries = [_Entry(ref, namespace, {"retrieval_score": ref.get("retrieval_score")}, method, provider) for ref in refs]
    return _adapt(entries, query=query, run_key=run_key, lookup=lookup)


def adapt_retrieval_response(response: Any, *, run_key: str, lookup: Optional[DocumentLookup] = None) -> AdaptedRetrieval:
    """Live research_retrieval.RetrievalResponse -> candidates (order and scores preserved as returned)."""
    entries = []
    for rank, item in enumerate(response.evidence, start=1):
        chunk = dict((getattr(item, "retrieval_metadata", {}) or {}).get("chunk", {}) or {})
        ref: dict[str, Any] = {"rank": rank, "chunk_id": getattr(item, "chunk_id", "") or getattr(item, "evidence_id", ""),
                               "text": item.text}
        if chunk.get("scifact_doc_id") is not None:
            ref.update(scifact_doc_id=chunk["scifact_doc_id"],
                       scifact_sentence_indices=chunk.get("scifact_sentence_indices") or [],
                       scifact_sentence_texts=chunk.get("scifact_sentence_texts") or [])
            namespace = "scifact"
        else:
            ref["document_id"] = (getattr(item, "document_id", "") or getattr(item, "filename", "")
                                  or getattr(item, "source_id", ""))
            namespace = "document"
        scores = {name: getattr(item, name, None)
                  for name in ("retrieval_score", "score", "reranker_score", "semantic_score", "lexical_score")}
        entries.append(_Entry(ref, namespace, scores, getattr(item, "retrieval_method", "existing_advanced_rag"),
                              getattr(response, "provider", "unknown")))
    return _adapt(entries, query=response.query, run_key=run_key, lookup=lookup)
