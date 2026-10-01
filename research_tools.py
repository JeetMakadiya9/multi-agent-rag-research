"""
Provider-independent research/search layer.

This module deliberately contains no LLM-specific or web-provider-specific logic.
The thesis can therefore evaluate the research workflow against:
1. a deterministic/static provider for tests,
2. a local corpus provider (to be connected to the existing RAG engine),
3. a live web provider added later.

The ResearchState object is updated only through explicit, auditable records.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence
from urllib.parse import urlparse
from uuid import uuid4


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _result_id() -> str:
    return f"result_{uuid4().hex[:10]}"


@dataclass
class SearchResult:
    """One source returned by a search provider or persisted search result."""

    source_id: str
    title: str
    url: str = ""
    snippet: str = ""
    content: str = ""
    source_type: str = "unknown"
    published_at: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Populated when SearchManager persists a result. Defaults preserve
    # existing provider/corpus construction call sites.
    result_id: Optional[str] = None
    search_id: Optional[str] = None
    rank: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "source_id": self.source_id,
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "content": self.content,
            "source_type": self.source_type,
            "published_at": self.published_at,
            "authors": list(self.authors),
            "metadata": dict(self.metadata),
        }
        if self.result_id is not None:
            result["result_id"] = self.result_id
        if self.search_id is not None:
            result["search_id"] = self.search_id
        if self.rank is not None:
            result["rank"] = self.rank
        return result


@dataclass
class SearchRequest:
    """Provider-neutral search request."""

    query: str
    limit: int = 10
    source_types: Optional[List[str]] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResponse:
    """Auditable result of one search operation."""

    query: str
    results: List[SearchResult]
    provider: str
    searched_at: str = field(default_factory=_utc_now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "results": [r.to_dict() for r in self.results],
            "provider": self.provider,
            "searched_at": self.searched_at,
            "metadata": dict(self.metadata),
        }


class SearchProvider(Protocol):
    """Interface implemented by every research/search backend."""

    @property
    def name(self) -> str:
        ...

    def search(self, request: SearchRequest) -> SearchResponse:
        ...


class StaticSearchProvider:
    """
    Deterministic provider used for development and tests.

    It performs simple lexical matching over an in-memory corpus. It is not
    intended to be the thesis' final retrieval implementation.
    """

    name = "static"

    def __init__(self, documents: Sequence[SearchResult]):
        self.documents = list(documents)

    @staticmethod
    def _tokens(text: str) -> List[str]:
        return [
            token.strip(".,!?;:()[]{}\"'").lower()
            for token in text.split()
            if token.strip(".,!?;:()[]{}\"'")
        ]

    def search(self, request: SearchRequest) -> SearchResponse:
        if not request.query.strip():
            raise ValueError("Search query cannot be empty.")
        if request.limit < 1:
            raise ValueError("Search limit must be >= 1.")

        query_tokens = set(self._tokens(request.query))
        scored = []

        for position, document in enumerate(self.documents):
            haystack = " ".join(
                [
                    document.title,
                    document.snippet,
                    document.content,
                    " ".join(document.authors),
                ]
            )
            doc_tokens = set(self._tokens(haystack))
            overlap = len(query_tokens & doc_tokens)
            if overlap == 0:
                continue

            # Small deterministic tie-breaker keeps tests reproducible.
            score = overlap / max(len(query_tokens), 1)
            scored.append((score, -position, document))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)

        results: List[SearchResult] = []
        for score, _, document in scored[: request.limit]:
            copied = SearchResult(
                source_id=document.source_id,
                title=document.title,
                url=document.url,
                snippet=document.snippet,
                content=document.content,
                source_type=document.source_type,
                published_at=document.published_at,
                authors=list(document.authors),
                metadata={**document.metadata, "static_match_score": score},
            )
            results.append(copied)

        return SearchResponse(
            query=request.query,
            results=results,
            provider=self.name,
            metadata={"candidate_count": len(self.documents)},
        )


class SearchManager:
    """
    Thin orchestration layer around a SearchProvider.

    This class is intentionally not an agent. It validates requests, performs
    the provider call, deduplicates sources, and can write the result into a
    ResearchState object without making reasoning decisions.
    """

    def __init__(self, provider: SearchProvider):
        self.provider = provider

    @staticmethod
    def _normalise_url(url: str) -> str:
        if not url:
            return ""
        parsed = urlparse(url.strip())
        if parsed.scheme and parsed.netloc:
            return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}{parsed.path}".rstrip("/")
        return url.strip().lower()

    def search(
        self,
        request: SearchRequest,
        state: Optional[Any] = None,
        *,
        run_id: Optional[str] = None,
        iteration_id: Optional[str] = None,
        question_id: Optional[str] = None,
    ) -> SearchResponse:
        response = self.provider.search(request)
        response = self._deduplicate(response)

        if state is not None:
            self._record_in_state(
                state,
                request,
                response,
                run_id=run_id,
                iteration_id=iteration_id,
                question_id=question_id,
            )

        return response

    def _deduplicate(self, response: SearchResponse) -> SearchResponse:
        seen = set()
        unique: List[SearchResult] = []

        for result in response.results:
            key = (
                result.source_id.strip().lower()
                or self._normalise_url(result.url)
                or result.title.strip().lower()
            )
            if not key or key in seen:
                continue
            seen.add(key)
            unique.append(result)

        return SearchResponse(
            query=response.query,
            results=unique,
            provider=response.provider,
            searched_at=response.searched_at,
            metadata={**response.metadata, "deduplicated_count": len(unique)},
        )

    @staticmethod
    def _record_in_state(
        state: Any,
        request: SearchRequest,
        response: SearchResponse,
        *,
        run_id: Optional[str] = None,
        iteration_id: Optional[str] = None,
        question_id: Optional[str] = None,
    ) -> None:
        """
        Integrates with the current ResearchState API without importing it.

        This avoids a hard dependency and keeps the search layer independently
        testable.
        """
        records_searches = hasattr(state, "add_search")
        if records_searches and not hasattr(state, "add_search_result"):
            raise TypeError(
                "State must implement add_search_result() to record linked search results."
            )

        for result in response.results:
            source_payload = {
                "source_id": result.source_id,
                "title": result.title,
                "url": result.url,
                "source_type": result.source_type,
                "published_at": result.published_at,
                "authors": list(result.authors),
                "metadata": dict(result.metadata),
            }

            # Current ResearchState.add_source expects a Source object.
            # Import lazily so this module can still be tested independently.
            try:
                from research_state import Source
            except ImportError:
                # This fallback is useful for future state implementations.
                if hasattr(state, "sources") and isinstance(state.sources, list):
                    state.sources.append(source_payload)
            else:
                state.add_source(
                    Source(
                        source_id=result.source_id,
                        title=result.title,
                        url=result.url,
                        source_type=result.source_type,
                        published_at=result.published_at,
                        metadata=source_payload,
                    )
                )

        if records_searches:
            try:
                from research_state import SearchAction
            except ImportError as exc:
                raise RuntimeError("Cannot import SearchAction while recording search state.") from exc

            action = SearchAction(
                query=request.query,
                provider=response.provider,
                result_count=len(response.results),
                created_at=response.searched_at,
                run_id=run_id,
                iteration_id=iteration_id,
                question_id=question_id,
            )
            state.add_search(action)

            # Persist independent result instances in final post-dedup order.
            # The same source may be returned by more than one search.
            from dataclasses import replace

            persisted_results = [
                replace(
                    result,
                    result_id=_result_id(),
                    search_id=action.search_id,
                    rank=rank,
                )
                for rank, result in enumerate(response.results, start=1)
            ]
            for result in persisted_results:
                state.add_search_result(result)
            response.results = persisted_results

        if hasattr(state, "log"):
            state.log(
                "search",
                f"Search completed: {request.query}",
                f"Provider: {response.provider}; returned {len(response.results)} result(s).",
            )


def search_and_collect(
    provider: SearchProvider,
    query: str,
    *,
    limit: int = 10,
    state: Optional[Any] = None,
    run_id: Optional[str] = None,
    iteration_id: Optional[str] = None,
    question_id: Optional[str] = None,
    **metadata: Any,
) -> SearchResponse:
    """Convenience function for the Research Agent layer."""

    manager = SearchManager(provider)
    return manager.search(
        SearchRequest(query=query, limit=limit, metadata=metadata),
        state=state,
        run_id=run_id,
        iteration_id=iteration_id,
        question_id=question_id,
    )


__all__ = [
    "SearchResult",
    "SearchRequest",
    "SearchResponse",
    "SearchProvider",
    "StaticSearchProvider",
    "SearchManager",
    "search_and_collect",
]
