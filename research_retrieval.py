"""
Adapter between the existing advanced RAG engine and the research layer.

The adapter deliberately does not replace or modify rag.py. It exposes the
existing retrieve_chunks_advanced() pipeline through the same provider-style
boundary used by the research system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence

from research_state import (
    DocumentVersion,
    Evidence,
    PassageReference,
    ResearchState,
    SearchAction,
    Source,
)
from research_tools import SearchResponse, SearchResult


@dataclass
class RetrievalEvidence:
    evidence_id: str
    text: str
    source_id: str
    filename: str
    page: int
    section: str = ""
    score: float = 0.0
    retrieval_metadata: Dict[str, Any] = field(default_factory=dict)
    document_id: str = ""
    chunk_id: str = ""
    retrieval_method: str = "existing_advanced_rag"
    retrieval_score: Optional[float] = None
    reranker_score: Optional[float] = None
    semantic_score: Optional[float] = None
    lexical_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "text": self.text,
            "source_id": self.source_id,
            "filename": self.filename,
            "page": self.page,
            "section": self.section,
            "score": self.score,
            "retrieval_metadata": dict(self.retrieval_metadata),
            "document_id": self.document_id,
            "chunk_id": self.chunk_id or self.evidence_id,
            "retrieval_method": self.retrieval_method,
            "retrieval_score": self.retrieval_score if self.retrieval_score is not None else self.score,
            "reranker_score": self.reranker_score,
            "semantic_score": self.semantic_score,
            "lexical_score": self.lexical_score,
        }


@dataclass
class RetrievalResponse:
    query: str
    evidence: List[RetrievalEvidence]
    provider: str = "existing_advanced_rag"
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "provider": self.provider,
            "evidence": [item.to_dict() for item in self.evidence],
            "diagnostics": dict(self.diagnostics),
        }


def passage_reference_from_retrieval_evidence(
    evidence: RetrievalEvidence,
    document_version: DocumentVersion,
) -> PassageReference:
    """Map existing RAG location fields to the generic passage locator.

    The caller supplies the source/document version explicitly; retrieval
    output alone is not treated as proof of a captured document version.
    """
    if evidence.document_id and evidence.document_id != document_version.document_id:
        raise ValueError(
            "RetrievalEvidence document_id does not match the supplied DocumentVersion."
        )

    if not isinstance(evidence.page, int) or isinstance(evidence.page, bool):
        raise ValueError("RetrievalEvidence page must be an integer.")
    if evidence.page < 0:
        raise ValueError("RetrievalEvidence page cannot be negative.")

    page = evidence.page if evidence.page >= 1 else None
    section = evidence.section or None
    chunk_id = evidence.chunk_id
    if chunk_id is not None and not isinstance(chunk_id, str):
        raise ValueError("RetrievalEvidence chunk_id must be a string when supplied.")
    if chunk_id is not None and not chunk_id.strip() and chunk_id != "":
        raise ValueError("RetrievalEvidence chunk_id cannot contain only whitespace.")
    if chunk_id == "":
        chunk_id = None

    # Page plus optional chunk is the most specific location exposed by the
    # current RAG adapter. Use a chunk-only locator when page is unavailable.
    if page is not None:
        return PassageReference(
            document_version_id=document_version.version_id,
            locator_type="page",
            page=page,
            chunk_id=chunk_id,
            section=section,
        )
    if chunk_id:
        return PassageReference(
            document_version_id=document_version.version_id,
            locator_type="chunk",
            chunk_id=chunk_id,
            section=section,
        )
    raise ValueError("RetrievalEvidence has neither a usable chunk_id nor a 1-based page.")


def register_rag_document_version(
    state: ResearchState,
    *,
    source_id: str,
    document_id: str,
    filename: str,
    version_id: Optional[str] = None,
    captured_at: Optional[str] = None,
    content_hash: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> DocumentVersion:
    """Explicitly register a local RAG source and document version.

    No file is read or hashed. A caller-provided content hash is preserved in
    DocumentVersion.metadata under ``content_hash``; generated version IDs
    are identities, not content hashes.
    """
    if not source_id.strip():
        raise ValueError("source_id is required for RAG document registration.")
    if not filename.strip():
        raise ValueError("filename is required for RAG document registration.")
    if not document_id.strip():
        raise ValueError("document_id is required for RAG document registration.")

    version_metadata = dict(metadata or {})
    if content_hash is not None:
        existing_hash = version_metadata.get("content_hash")
        if existing_hash is not None and existing_hash != content_hash:
            raise ValueError("Conflicting content_hash values were supplied.")
        version_metadata["content_hash"] = content_hash

    version_kwargs: Dict[str, Any] = {
        "document_id": document_id,
        "source_id": source_id,
        "metadata": version_metadata,
    }
    if version_id is not None:
        version_kwargs["version_id"] = version_id
    if captured_at is not None:
        version_kwargs["captured_at"] = captured_at
    version = DocumentVersion(**version_kwargs)
    if any(item.version_id == version.version_id for item in state.document_versions):
        raise ValueError(f"Duplicate document version_id {version.version_id!r}.")

    source = next((item for item in state.sources if item.source_id == source_id), None)
    if source is None:
        state.add_source(
            Source(
                source_id=source_id,
                title=filename,
                source_type="local_rag",
                metadata={"filename": filename},
            )
        )
    return state.add_document_version(version)


def adapt_retrieval_evidence_to_state(
    retrieval_evidence: RetrievalEvidence,
    document_version: DocumentVersion,
    state: ResearchState,
    *,
    search_result_id: Optional[str] = None,
) -> Evidence:
    """Register a retrieved RAG excerpt against an already registered version."""
    if not isinstance(retrieval_evidence, RetrievalEvidence):
        raise TypeError("retrieval_evidence must be a RetrievalEvidence instance.")
    if not isinstance(document_version, DocumentVersion):
        raise TypeError("document_version must be an explicitly registered DocumentVersion.")

    registered_version = next(
        (item for item in state.document_versions if item.version_id == document_version.version_id),
        None,
    )
    if registered_version is None:
        raise ValueError("DocumentVersion must be registered in ResearchState before adaptation.")
    if (
        registered_version.document_id != document_version.document_id
        or registered_version.source_id != document_version.source_id
    ):
        raise ValueError("Supplied DocumentVersion does not match the registered version record.")
    if not any(source.source_id == registered_version.source_id for source in state.sources):
        raise ValueError("Registered DocumentVersion has no matching Source in ResearchState.")
    if retrieval_evidence.source_id != document_version.source_id:
        raise ValueError(
            "RetrievalEvidence source_id does not match DocumentVersion source_id."
        )
    if (
        retrieval_evidence.document_id
        and document_version.document_id
        and retrieval_evidence.document_id != document_version.document_id
    ):
        raise ValueError(
            "RetrievalEvidence document_id does not match DocumentVersion document_id."
        )
    if search_result_id is not None:
        linked_result = next(
            (item for item in state.search_results if item.result_id == search_result_id),
            None,
        )
        if linked_result is None:
            raise ValueError(f"Unknown search_result_id {search_result_id!r}.")
        if linked_result.source_id != retrieval_evidence.source_id:
            raise ValueError("SearchResult source_id does not match RetrievalEvidence source_id.")

    candidate = passage_reference_from_retrieval_evidence(
        retrieval_evidence,
        document_version,
    )
    # Reuse a stable chunk location within this version. Every adaptation
    # still creates a new Evidence record for its retrieval context.
    reference = None
    if candidate.chunk_id:
        reference = next(
            (
                item
                for item in state.passage_references
                if item.document_version_id == candidate.document_version_id
                and item.locator_type == candidate.locator_type
                and item.page == candidate.page
                and item.chunk_id == candidate.chunk_id
                and item.section == candidate.section
                and item.start_char == candidate.start_char
                and item.end_char == candidate.end_char
            ),
            None,
        )
    if reference is None:
        reference = state.add_passage_reference(candidate)

    metadata: Dict[str, Any] = {
        "retrieval_evidence_id": retrieval_evidence.evidence_id,
        "retrieval_method": retrieval_evidence.retrieval_method,
        "retrieval_result_score": retrieval_evidence.score,
        "retrieval_metadata": dict(retrieval_evidence.retrieval_metadata),
        "retrieval_scores": {
            field_name: value
            for field_name in (
                "retrieval_score",
                "reranker_score",
                "semantic_score",
                "lexical_score",
            )
            if (value := getattr(retrieval_evidence, field_name)) is not None
        },
    }

    evidence = Evidence(
        text=retrieval_evidence.text,
        source_id=retrieval_evidence.source_id,
        page=retrieval_evidence.page if retrieval_evidence.page >= 1 else None,
        chunk_id=retrieval_evidence.chunk_id or None,
        document_version_id=document_version.version_id,
        passage_reference_id=reference.passage_reference_id,
        metadata=metadata,
        search_result_id=search_result_id,
    )
    return state.add_evidence(evidence)


def record_rag_retrieval(
    state: ResearchState,
    *,
    query: str,
    retrieval_results: Sequence[RetrievalEvidence],
    run_id: Optional[str] = None,
    iteration_id: Optional[str] = None,
    question_id: Optional[str] = None,
    provider: str = "existing_advanced_rag",
    purpose: str = "retrieval",
    evidence_document_versions: Optional[Mapping[int, DocumentVersion]] = None,
) -> SearchResponse:
    """Record an already completed RAG retrieval and optionally selected evidence.

    Retrieval results are stored in their supplied order. ``evidence_document_versions``
    is an explicit selection keyed by 1-based retrieval rank; only those ranks become
    provenance-linked Evidence, and each must name an already registered version.
    This function never invokes or alters the retrieval engine.
    """
    if not query.strip():
        raise ValueError("RAG retrieval query cannot be empty.")
    items = list(retrieval_results)
    for rank, item in enumerate(items, start=1):
        if not isinstance(item, RetrievalEvidence):
            raise TypeError("retrieval_results must contain RetrievalEvidence instances.")
        if not item.source_id:
            raise ValueError(f"RAG retrieval result at rank {rank} has no source_id.")
    versions_by_rank = dict(evidence_document_versions or {})
    for rank, version in versions_by_rank.items():
        if not isinstance(rank, int) or isinstance(rank, bool) or rank < 1 or rank > len(items):
            raise ValueError(f"Selected evidence rank {rank!r} is outside the retrieved result order.")
        if not isinstance(version, DocumentVersion):
            raise TypeError("Selected evidence versions must be DocumentVersion instances.")
        registered_version = next(
            (existing for existing in state.document_versions if existing.version_id == version.version_id),
            None,
        )
        if registered_version is None:
            raise ValueError(f"DocumentVersion {version.version_id!r} must be registered before retrieval recording.")
        if (
            registered_version.document_id != version.document_id
            or registered_version.source_id != version.source_id
        ):
            raise ValueError("Selected DocumentVersion does not match its registered state record.")
        retrieval_item = items[rank - 1]
        if retrieval_item.source_id != version.source_id:
            raise ValueError("Selected DocumentVersion source_id does not match its retrieval result.")
        if retrieval_item.document_id and retrieval_item.document_id != version.document_id:
            raise ValueError("Selected DocumentVersion document_id does not match its retrieval result.")
        passage_reference_from_retrieval_evidence(retrieval_item, version)

    searched_at = datetime.now(timezone.utc).isoformat()
    action = state.add_search(
        SearchAction(
            query=query,
            provider=provider,
            result_count=len(items),
            purpose=purpose,
            created_at=searched_at,
            run_id=run_id,
            iteration_id=iteration_id,
            question_id=question_id,
        )
    )

    persisted_results: List[SearchResult] = []
    unique_sources: set[str] = set()
    for rank, item in enumerate(items, start=1):
        if item.source_id not in unique_sources and not any(
            source.source_id == item.source_id for source in state.sources
        ):
            source_metadata: Dict[str, Any] = {}
            if item.filename:
                source_metadata["filename"] = item.filename
            state.add_source(
                Source(
                    title="",
                    source_id=item.source_id,
                    source_type="unknown",
                    metadata=source_metadata,
                )
            )
        unique_sources.add(item.source_id)

        retrieval_metadata: Dict[str, Any] = {
            "retrieval_evidence_id": item.evidence_id,
            "filename": item.filename,
            "page": item.page,
            "section": item.section,
            "document_id": item.document_id,
            "chunk_id": item.chunk_id,
            "retrieval_method": item.retrieval_method,
            "retrieval_scores": {
                name: value
                for name in ("score", "retrieval_score", "reranker_score", "semantic_score", "lexical_score")
                if (value := getattr(item, name)) is not None
            },
            "retrieval_metadata": dict(item.retrieval_metadata),
        }
        result = state.add_search_result(
            SearchResult(
                source_id=item.source_id,
                title="",
                content=item.text,
                source_type="unknown",
                metadata=retrieval_metadata,
                search_id=action.search_id,
                rank=rank,
            )
        )
        persisted_results.append(result)

        version = versions_by_rank.get(rank)
        if version is not None:
            adapt_retrieval_evidence_to_state(
                item,
                version,
                state,
                search_result_id=result.result_id,
            )

    state.log(
        "retrieval",
        f"RAG retrieval recorded: {query}",
        f"Provider: {provider}; returned {len(persisted_results)} result(s).",
    )
    return SearchResponse(
        query=query,
        results=persisted_results,
        provider=provider,
        searched_at=searched_at,
        metadata={"purpose": purpose, "recorded_result_count": len(persisted_results)},
    )


class RetrievalProvider(Protocol):
    def retrieve(self, query: str, limit: int = 8) -> RetrievalResponse:
        ...


class ExistingRAGProvider:
    """
    Provider wrapper for the existing rag.py retrieval engine.

    Dependencies are injected so the research system does not own model/index
    construction. That remains the responsibility of the existing RAG system.
    """

    name = "existing_advanced_rag"

    def __init__(
        self,
        *,
        chunks: Sequence[Dict[str, Any]],
        embedding_model: Any,
        reranker: Any,
        faiss_index: Any,
        bm25_index: Any,
        parents: Optional[Sequence[Dict[str, Any]]] = None,
        history: Optional[Sequence[Dict[str, Any]]] = None,
        retrieval_module: Any = None,
    ):
        if retrieval_module is None:
            import rag as retrieval_module

        self.chunks = chunks
        self.embedding_model = embedding_model
        self.reranker = reranker
        self.faiss_index = faiss_index
        self.bm25_index = bm25_index
        self.parents = parents
        self.history = history
        self.rag = retrieval_module

    @staticmethod
    def _convert_result(result: Any, fallback_index: int) -> RetrievalEvidence:
        if hasattr(result, "chunk"):
            chunk = result.chunk
            metadata = result.to_dict()
            score = float(
                getattr(
                    result,
                    "evidence_score",
                    getattr(result, "final_score", 0.0),
                )
                or 0.0
            )
        else:
            chunk = result.get("chunk", result)
            metadata = dict(result)
            score = float(result.get("evidence_score", result.get("final_score", result.get("score", 0.0))))

        filename = str(chunk.get("filename", "Unknown"))
        page = int(chunk.get("page", 0) or 0)
        section = str(chunk.get("section", "") or "")
        text = str(chunk.get("text", "") or "")

        evidence_id = str(
            chunk.get("chunk_id")
            or f"{filename}:page-{page}:result-{fallback_index}"
        )

        def optional_score(name: str) -> Optional[float]:
            value = getattr(result, name, None)
            if value is None:
                value = metadata.get(name)
            try:
                return float(value) if value is not None else None
            except (TypeError, ValueError):
                return None

        source_id = f"{filename}:page-{page}"

        return RetrievalEvidence(
            evidence_id=evidence_id,
            text=text,
            source_id=source_id,
            filename=filename,
            page=page,
            section=section,
            score=score,
            retrieval_metadata=metadata,
            document_id=str(chunk.get("document_id") or filename),
            chunk_id=str(chunk.get("chunk_id") or evidence_id),
            retrieval_method="existing_advanced_rag",
            retrieval_score=score,
            reranker_score=optional_score("rerank_score"),
            semantic_score=optional_score("semantic_score"),
            lexical_score=optional_score("lexical_score"),
        )

    def retrieve(self, query: str, limit: int = 8) -> RetrievalResponse:
        if not query.strip():
            raise ValueError("Retrieval query cannot be empty.")
        if limit < 1:
            raise ValueError("Retrieval limit must be >= 1.")

        diagnostics = None
        if hasattr(self.rag, "RetrievalDiagnostics"):
            diagnostics = self.rag.RetrievalDiagnostics(query=query)

        results = self.rag.retrieve_chunks_advanced(
            question=query,
            chunks=self.chunks,
            embedding_model=self.embedding_model,
            reranker=self.reranker,
            faiss_index=self.faiss_index,
            bm25_index=self.bm25_index,
            history=self.history,
            parents=self.parents,
            diagnostics=diagnostics,
        )

        converted = [
            self._convert_result(result, index)
            for index, result in enumerate(results[:limit])
        ]

        diagnostic_dict: Dict[str, Any] = {}
        if diagnostics is not None and hasattr(diagnostics, "to_dict"):
            diagnostic_dict = diagnostics.to_dict()
        elif diagnostics is not None:
            diagnostic_dict = {
                key: value
                for key, value in vars(diagnostics).items()
                if not key.startswith("_")
            }

        return RetrievalResponse(
            query=query,
            evidence=converted,
            provider=self.name,
            diagnostics=diagnostic_dict,
        )


class StaticRetrievalProvider:
    """Small deterministic provider for integration tests."""

    name = "static_retrieval"

    def __init__(self, evidence: Sequence[RetrievalEvidence]):
        self.evidence = list(evidence)

    def retrieve(self, query: str, limit: int = 8) -> RetrievalResponse:
        if not query.strip():
            raise ValueError("Retrieval query cannot be empty.")

        terms = set(query.lower().split())
        scored = []

        for index, item in enumerate(self.evidence):
            tokens = set(item.text.lower().split())
            overlap = len(terms & tokens)
            if overlap:
                scored.append((overlap, -index, item))

        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

        return RetrievalResponse(
            query=query,
            evidence=[item for _, _, item in scored[:limit]],
            provider=self.name,
            diagnostics={"candidate_count": len(self.evidence)},
        )


__all__ = [
    "RetrievalEvidence",
    "RetrievalResponse",
    "passage_reference_from_retrieval_evidence",
    "register_rag_document_version",
    "adapt_retrieval_evidence_to_state",
    "record_rag_retrieval",
    "RetrievalProvider",
    "ExistingRAGProvider",
    "StaticRetrievalProvider",
]
