"""Agent C - Evidence retrieval and ranking: splits source text into traceable passages and ranks them per question
or claim with the existing hybrid engine (rag.py: dense + BM25 + exact match, RRF, cross-encoder rerank, MMR),
imported unchanged through docqa.index. Each passage keeps its source ID and page.

Diversity: rag.py returns at most 8 passages per query, and one long document (a 25-page PDF) can fill all of them,
which the first live run showed (7 of 7 passages from one source). Each source therefore gets its own index; every
source contributes its best passages, the cross-encoder relevance score orders them globally, and at most
``per_source`` passages come from one origin (copies of one report share an origin). rag.py's relevance gate still
drops sources with no relevant passage.
"""
from __future__ import annotations

import functools
import hashlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from docqa.index import Evidence, EvidenceIndex
from docqa.ingest import Page

from .researcher import Source


@functools.lru_cache(maxsize=1)
def load_models():
    """Embedding model and reranker from the local cache (HF offline mode is set by docqa.index)."""
    import rag
    from sentence_transformers import CrossEncoder, SentenceTransformer

    return SentenceTransformer(rag.EMBEDDING_MODEL_NAME), CrossEncoder(rag.RERANKER_MODEL_NAME)


@dataclass
class Passage:
    key: str                    # stable reference: <source id>:p<page>:<text hash>
    source_id: str
    page: int
    text: str
    score: float                # cross-encoder relevance to the query that retrieved it

    def as_evidence(self, label: str) -> Evidence:
        return Evidence(id=label, document=self.source_id, page=self.page, text=self.text, score=self.score)


def _pages(source: Source) -> List[Page]:
    if source.pages:
        return [Page(source.id, i, text) for i, text in enumerate(source.pages, 1) if text.strip()]
    return [Page(source.id, 1, source.text)]


class EvidenceStore:
    def __init__(self, sources: Sequence[Source], models=None):
        self.sources: Dict[str, Source] = {}
        self._models = models or load_models()
        self.indexes: Dict[str, EvidenceIndex] = {}
        self.add_sources(sources)
        if not self.indexes:
            raise ValueError("No source text to index.")

    def add_sources(self, sources: Sequence[Source]) -> None:
        """Index more sources (follow-up research); existing indexes are kept, not rebuilt."""
        embedding_model, reranker = self._models
        for source in sources:
            self.sources[source.id] = source
            pages = _pages(source) if source.usable else []
            if pages and source.id not in self.indexes:
                self.indexes[source.id] = EvidenceIndex(pages, embedding_model, reranker)

    @property
    def size(self) -> int:
        return sum(len(index.chunks) for index in self.indexes.values())

    def rank(self, query: str, k: int = 6, per_source: int = 2) -> List[Passage]:
        candidates: List[Passage] = []
        for source_id, index in self.indexes.items():
            for evidence in index.search(query, k=per_source):
                score = evidence.rerank_score if evidence.rerank_score is not None else evidence.score
                digest = hashlib.sha1(evidence.text.encode("utf-8")).hexdigest()[:8]
                candidates.append(Passage(f"{source_id}:p{evidence.page}:{digest}", source_id, evidence.page,
                                          evidence.text, float(score)))
        candidates.sort(key=lambda p: (-p.score, p.source_id))
        return cap_per_origin(candidates, self.sources, k, per_source)


def cap_per_origin(passages: Iterable[Passage], sources: Dict[str, Source], k: int, per_source: int) -> List[Passage]:
    """Keep order; at most ``per_source`` passages per origin (a source and its near-copies) and ``k`` in total."""
    kept, per_origin = [], {}
    for passage in passages:
        origin = sources[passage.source_id].origin or passage.source_id
        if per_origin.get(origin, 0) >= per_source:
            continue
        per_origin[origin] = per_origin.get(origin, 0) + 1
        kept.append(passage)
        if len(kept) == k:
            break
    return kept


def merge_passages(*groups: Iterable[Passage], limit: int) -> List[Passage]:
    """Union in order of first appearance, de-duplicated by passage text."""
    seen, merged = set(), []
    for group in groups:
        for passage in group:
            if passage.text in seen:
                continue
            seen.add(passage.text)
            merged.append(passage)
            if len(merged) == limit:
                return merged
    return merged
