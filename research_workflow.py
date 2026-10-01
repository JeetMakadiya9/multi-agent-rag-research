"""Opt-in workflow boundary for RAG retrieval with ResearchState lineage.

This module connects an existing retrieval provider to the research state
without changing the retrieval engine or the Experiment A/B pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

from research_retrieval import (
    RetrievalProvider,
    RetrievalResponse,
    record_rag_retrieval,
)
from research_state import DocumentVersion, ResearchState
from research_tools import SearchResponse


@dataclass
class ResearchRetrievalResult:
    """The unchanged provider response and its recorded lineage response."""

    retrieval_response: RetrievalResponse
    search_response: SearchResponse


class ResearchLineageRecordingError(RuntimeError):
    """Recording failed after retrieval; retain the successful raw response."""

    def __init__(self, retrieval_response: RetrievalResponse, cause: Exception):
        self.retrieval_response = retrieval_response
        self.cause = cause
        super().__init__(
            "RAG retrieval succeeded but ResearchState lineage recording failed: "
            f"{type(cause).__name__}: {cause}"
        )


def retrieve_for_research(
    provider: RetrievalProvider,
    state: ResearchState,
    *,
    query: str,
    limit: int = 8,
    run_id: Optional[str] = None,
    iteration_id: Optional[str] = None,
    question_id: Optional[str] = None,
    evidence_document_versions: Optional[Mapping[int, DocumentVersion]] = None,
    purpose: str = "retrieval",
) -> ResearchRetrievalResult:
    """Retrieve normally, then record the activity and selected evidence.

    The caller opts into this workflow boundary by supplying a ResearchState.
    Existing provider retrieval remains available directly through
    ``provider.retrieve(query, limit)``. Evidence selection is explicit by
    1-based result rank through ``evidence_document_versions``.
    """
    retrieval_response = provider.retrieve(query, limit=limit)
    try:
        search_response = record_rag_retrieval(
            state,
            query=query,
            retrieval_results=retrieval_response.evidence,
            run_id=run_id,
            iteration_id=iteration_id,
            question_id=question_id,
            provider=retrieval_response.provider,
            purpose=purpose,
            evidence_document_versions=evidence_document_versions,
        )
    except Exception as exc:
        # Preserve the successful retrieval output while making the incomplete
        # lineage explicit to the caller.
        raise ResearchLineageRecordingError(retrieval_response, exc) from exc

    return ResearchRetrievalResult(
        retrieval_response=retrieval_response,
        search_response=search_response,
    )


__all__ = [
    "ResearchRetrievalResult",
    "ResearchLineageRecordingError",
    "retrieve_for_research",
]
