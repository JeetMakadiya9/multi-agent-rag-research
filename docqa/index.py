"""Evidence index: chunking and hybrid search through the existing ``rag.py`` engine (imported, never modified).

``rag.create_chunks`` works one document at a time with local integer chunk IDs; they are remapped to global
positions (as the SciFact adapter does) so several documents can share one index.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from .ingest import Page

# Use the locally cached embedding and reranker models only; never download.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


@dataclass
class Evidence:
    id: str                     # positional label shown to the model: E1, E2, ...
    document: str
    page: int
    text: str
    score: float = 0.0          # rag.py's combined evidence score
    rerank_score: Optional[float] = None

    @property
    def citation(self) -> str:
        return f"{self.document}, p. {self.page}"


def _global_chunks(pages: Sequence[Page]) -> List[Dict[str, Any]]:
    import rag

    by_document: Dict[str, List[Page]] = {}
    for page in pages:
        by_document.setdefault(page.document, []).append(page)
    chunks: List[Dict[str, Any]] = []
    for document in sorted(by_document):
        local = rag.create_chunks([{"page": p.page, "text": p.text} for p in by_document[document]], document)
        offset = len(chunks)
        ids = {chunk["chunk_id"]: offset + i for i, chunk in enumerate(local)}
        for chunk in local:
            mapped = dict(chunk)
            mapped["chunk_id"] = ids[chunk["chunk_id"]]
            for key in ("previous_chunk_id", "next_chunk_id"):
                if chunk.get(key) is not None:
                    mapped[key] = ids.get(chunk[key])
            chunks.append(mapped)
    return chunks


class EvidenceIndex:
    """Build once per corpus; ``search`` returns at most ``k`` page-attributed passages (possibly none)."""

    def __init__(self, pages: Sequence[Page], embedding_model: Any = None, reranker: Any = None):
        import rag

        if not pages:
            raise ValueError("No document text to index.")
        self.chunks = _global_chunks(pages)
        if embedding_model is None or reranker is None:
            from sentence_transformers import CrossEncoder, SentenceTransformer
            embedding_model = embedding_model or SentenceTransformer(rag.EMBEDDING_MODEL_NAME)
            reranker = reranker or CrossEncoder(rag.RERANKER_MODEL_NAME)
        self.embedding_model, self.reranker = embedding_model, reranker
        _, self.faiss_index, self.bm25_index = rag.build_search_database(self.chunks, embedding_model)

    def search(self, query: str, k: int = 4) -> List[Evidence]:
        import rag

        results = rag.retrieve_chunks_advanced(question=query, chunks=self.chunks,
                                               embedding_model=self.embedding_model, reranker=self.reranker,
                                               faiss_index=self.faiss_index, bm25_index=self.bm25_index)
        evidence = []
        for rank, result in enumerate(results[:k], 1):
            chunk = result.chunk
            evidence.append(Evidence(id=f"E{rank}", document=str(chunk.get("filename", "unknown")),
                                     page=int(chunk.get("page") or 1), text=str(chunk.get("text", "")).strip(),
                                     score=float(result.evidence_score), rerank_score=result.rerank_score))
        return evidence
