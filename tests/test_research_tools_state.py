from __future__ import annotations

import unittest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research_state import ResearchState
from research_tools import (
    SearchManager,
    SearchRequest,
    SearchResponse,
    SearchResult,
    StaticSearchProvider,
)


class SearchManagerResearchStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.results = [
            SearchResult(
                source_id="source-1",
                title="Hybrid retrieval study",
                url="https://example.org/study",
                snippet="BM25 and dense retrieval are combined.",
                content="The study combines BM25 and dense retrieval.",
                source_type="paper",
                metadata={"venue": "Example Proceedings"},
            ),
            SearchResult(
                source_id="source-2",
                title="Reranking study",
                url="https://example.org/reranking",
                snippet="A reranker improves ranking.",
                content="The reranker reorders retrieved documents.",
                source_type="paper",
            ),
        ]
        self.provider = StaticSearchProvider(self.results)
        self.request = SearchRequest(query="study", limit=5)

    def test_search_records_action_sources_and_event_in_research_state(self) -> None:
        state = ResearchState(user_request="Research hybrid retrieval")

        response = SearchManager(self.provider).search(self.request, state=state)

        self.assertEqual(response.query, self.request.query)
        self.assertEqual(response.provider, "static")
        self.assertEqual(len(response.results), 2)

        self.assertEqual(len(state.searches), 1)
        action = state.searches[0]
        self.assertEqual(action.query, self.request.query)
        self.assertEqual(action.provider, "static")
        self.assertEqual(action.result_count, 2)
        self.assertEqual(action.purpose, "research")
        self.assertEqual(action.created_at, response.searched_at)
        self.assertTrue(action.search_id)

        self.assertEqual(len(state.search_results), 2)
        self.assertEqual([item.rank for item in state.search_results], [1, 2])
        self.assertEqual(
            [item.search_id for item in state.search_results],
            [action.search_id, action.search_id],
        )
        self.assertEqual(
            [item.source_id for item in state.search_results],
            [source.source_id for source in state.sources],
        )
        self.assertEqual(
            [item.result_id for item in state.search_results],
            [item.result_id for item in response.results],
        )
        self.assertTrue(all(item.result_id for item in state.search_results))
        self.assertEqual(len({item.result_id for item in state.search_results}), 2)

        self.assertEqual([source.source_id for source in state.sources], ["source-1", "source-2"])
        self.assertEqual(state.sources[0].metadata["metadata"]["venue"], "Example Proceedings")
        self.assertEqual(state.sources[0].metadata["metadata"]["static_match_score"], 1.0)

        self.assertEqual(len(state.events), 1)
        event = state.events[0]
        self.assertEqual(event.agent, "search")
        self.assertEqual(event.action, f"Search completed: {self.request.query}")
        self.assertIn("Provider: static", event.message)
        self.assertIn("returned 2 result(s)", event.message)

    def test_state_recording_errors_are_not_silently_swallowed(self) -> None:
        class BrokenSearchState(ResearchState):
            def add_search(self, search) -> None:
                raise TypeError("search recording failed")

        state = BrokenSearchState(user_request="Research hybrid retrieval")
        with self.assertRaisesRegex(TypeError, "search recording failed"):
            SearchManager(self.provider).search(self.request, state=state)

    def test_search_without_state_keeps_provider_results(self) -> None:
        response = SearchManager(self.provider).search(self.request)

        self.assertEqual(response.query, self.request.query)
        self.assertEqual(response.provider, "static")
        self.assertEqual([result.source_id for result in response.results], ["source-1", "source-2"])
        self.assertEqual(response.metadata["deduplicated_count"], 2)
        self.assertTrue(all(result.search_id is None for result in response.results))
        self.assertTrue(all(result.rank is None for result in response.results))
        self.assertTrue(all("search_id" not in result.to_dict() for result in response.results))

    def test_ranks_follow_final_deduplicated_order(self) -> None:
        class DuplicateProvider:
            name = "duplicate-test"

            def search(self, request):
                return SearchResponse(
                    query=request.query,
                    provider=self.name,
                    searched_at="2026-01-01T00:00:00+00:00",
                    results=[
                        SearchResult(source_id="source-a", title="first A", content="first"),
                        SearchResult(source_id="source-a", title="duplicate A", content="duplicate"),
                        SearchResult(source_id="source-b", title="B", content="second"),
                        SearchResult(source_id="source-c", title="C", content="third"),
                    ],
                )

        state = ResearchState(user_request="Check final ordering")
        response = SearchManager(DuplicateProvider()).search(self.request, state=state)

        self.assertEqual(
            [result.source_id for result in response.results],
            ["source-a", "source-b", "source-c"],
        )
        self.assertEqual([result.rank for result in response.results], [1, 2, 3])
        self.assertEqual([result.content for result in response.results], ["first", "second", "third"])
        self.assertEqual([result.rank for result in state.search_results], [1, 2, 3])

    def test_repeated_source_across_searches_has_distinct_result_and_search_ids(self) -> None:
        state = ResearchState(user_request="Search the same source twice")
        manager = SearchManager(self.provider)

        first_response = manager.search(self.request, state=state)
        second_response = manager.search(self.request, state=state)
        first = next(item for item in first_response.results if item.source_id == "source-1")
        second = next(item for item in second_response.results if item.source_id == "source-1")

        self.assertEqual(first.source_id, second.source_id)
        self.assertNotEqual(first.result_id, second.result_id)
        self.assertNotEqual(first.search_id, second.search_id)
        self.assertEqual(first.rank, second.rank)
        self.assertEqual(first.search_id, state.searches[0].search_id)
        self.assertEqual(second.search_id, state.searches[1].search_id)

    def test_research_state_round_trip_preserves_search_results(self) -> None:
        state = ResearchState(user_request="Round-trip search results")
        response = SearchManager(self.provider).search(self.request, state=state)

        restored = ResearchState.from_dict(state.to_dict())

        self.assertEqual(len(restored.search_results), 2)
        for original, recovered in zip(response.results, restored.search_results):
            self.assertEqual(recovered.to_dict(), original.to_dict())
            self.assertEqual(recovered.result_id, original.result_id)
            self.assertEqual(recovered.search_id, original.search_id)
            self.assertEqual(recovered.rank, original.rank)
            self.assertEqual(recovered.source_id, original.source_id)
            self.assertEqual(recovered.title, original.title)
            self.assertEqual(recovered.url, original.url)
            self.assertEqual(recovered.snippet, original.snippet)
            self.assertEqual(recovered.content, original.content)
            self.assertEqual(recovered.source_type, original.source_type)
            self.assertEqual(recovered.published_at, original.published_at)
            self.assertEqual(recovered.authors, original.authors)
            self.assertEqual(recovered.metadata, original.metadata)

    def test_old_research_state_payload_without_search_results_loads(self) -> None:
        old_payload = ResearchState(user_request="Legacy payload").to_dict()
        old_payload.pop("search_results", None)

        restored = ResearchState.from_dict(old_payload)

        self.assertEqual(restored.search_results, [])

    def test_old_search_result_payload_loads_without_lineage_fields(self) -> None:
        old_result = SearchResult(
            source_id="legacy-source",
            title="Legacy result",
            url="https://example.org/legacy",
            metadata={"legacy": True},
        ).to_dict()
        restored = ResearchState.from_dict(
            {"user_request": "Legacy result payload", "search_results": [old_result]}
        )

        result = restored.search_results[0]
        self.assertEqual(result.source_id, "legacy-source")
        self.assertEqual(result.title, "Legacy result")
        self.assertEqual(result.metadata, {"legacy": True})
        self.assertIsNone(result.result_id)
        self.assertIsNone(result.search_id)
        self.assertIsNone(result.rank)

    def test_existing_search_result_constructor_remains_compatible(self) -> None:
        result = SearchResult("source-old", "Old result", "https://example.org/old")

        self.assertEqual(result.source_id, "source-old")
        self.assertEqual(result.title, "Old result")
        self.assertEqual(result.url, "https://example.org/old")
        self.assertIsNone(result.result_id)
        self.assertIsNone(result.search_id)
        self.assertIsNone(result.rank)

    def test_research_state_log_and_round_trip_remain_compatible(self) -> None:
        state = ResearchState(user_request="Compatibility check")
        event = state.log("test-agent", "test-action", "test message")

        self.assertEqual(event.agent, "test-agent")
        self.assertEqual(event.action, "test-action")
        self.assertEqual(event.message, "test message")

        restored = ResearchState.from_dict(state.to_dict())
        self.assertEqual(len(restored.events), 1)
        self.assertEqual(restored.events[0].agent, "test-agent")
        self.assertEqual(restored.events[0].action, "test-action")
        self.assertEqual(restored.events[0].message, "test message")


if __name__ == "__main__":
    unittest.main()
