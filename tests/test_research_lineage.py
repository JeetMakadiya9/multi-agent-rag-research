from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research_state import (
    DocumentVersion,
    Evidence,
    PassageReference,
    ResearchIteration,
    ResearchQuestion,
    ResearchRun,
    ResearchState,
    SearchAction,
    Source,
)
from research_tools import SearchManager, SearchRequest, SearchResult, StaticSearchProvider, search_and_collect


class ResearchLineageTests(unittest.TestCase):
    def test_run_has_unique_id_status_metadata_and_round_trips(self) -> None:
        state = ResearchState(user_request="Lineage test")
        first = state.add_run(ResearchRun(status="running", metadata={"topic": "RAG"}))
        second = state.add_run()

        self.assertTrue(first.run_id.startswith("run_"))
        self.assertNotEqual(first.run_id, second.run_id)
        self.assertEqual(first.status, "running")
        restored = ResearchState.from_dict(state.to_dict())
        self.assertEqual(restored.research_runs, [first, second])

    def test_unknown_run_and_invalid_status_are_rejected(self) -> None:
        state = ResearchState(user_request="Invalid run")
        with self.assertRaisesRegex(ValueError, "Unsupported ResearchRun status"):
            ResearchRun(status="paused")  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "unknown run_id"):
            state.add_iteration(ResearchIteration(run_id="missing", number=1))

    def test_duplicate_run_id_is_rejected(self) -> None:
        state = ResearchState(user_request="Duplicate run")
        run = state.add_run(ResearchRun(run_id="run-fixed"))
        with self.assertRaisesRegex(ValueError, "Duplicate research run_id"):
            state.add_run(ResearchRun(run_id=run.run_id))

    def test_iterations_are_numbered_per_run_and_duplicate_numbers_rejected(self) -> None:
        state = ResearchState(user_request="Iteration numbering")
        run_a = state.add_run()
        run_b = state.add_run()
        a1 = state.create_iteration(run_a.run_id)
        a2 = state.create_iteration(run_a.run_id, status="running")
        b1 = state.create_iteration(run_b.run_id)

        self.assertEqual([a1.number, a2.number, b1.number], [1, 2, 1])
        with self.assertRaisesRegex(ValueError, "Duplicate iteration number"):
            state.add_iteration(ResearchIteration(run_id=run_a.run_id, number=1))
        restored = ResearchState.from_dict(state.to_dict())
        self.assertEqual(restored.iterations, [a1, a2, b1])

    def test_iteration_requires_positive_integer(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive integer"):
            ResearchIteration(run_id="run", number=0)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            ResearchIteration(run_id="run", number=True)  # type: ignore[arg-type]

    def test_duplicate_iteration_id_is_rejected(self) -> None:
        state = ResearchState(user_request="Duplicate iteration ID")
        run = state.add_run()
        original = state.add_iteration(ResearchIteration(run_id=run.run_id, number=1, iteration_id="iteration-fixed"))
        with self.assertRaisesRegex(ValueError, "Duplicate iteration_id"):
            state.add_iteration(ResearchIteration(run_id=run.run_id, number=2, iteration_id=original.iteration_id))

    def test_question_links_run_iteration_and_parent_across_iterations(self) -> None:
        state = ResearchState(user_request="Question hierarchy")
        run = state.add_run()
        iteration1 = state.create_iteration(run.run_id)
        iteration2 = state.create_iteration(run.run_id)
        parent = state.add_question("How does method X work?", run_id=run.run_id, iteration_id=iteration1.iteration_id)
        child = state.add_question(
            "What limitation is reported?",
            run_id=run.run_id,
            iteration_id=iteration2.iteration_id,
            parent_question_id=parent.question_id,
        )

        self.assertEqual(parent.run_id, run.run_id)
        self.assertEqual(parent.iteration_id, iteration1.iteration_id)
        self.assertEqual(child.parent_question_id, parent.question_id)
        self.assertEqual(child.iteration_id, iteration2.iteration_id)

    def test_question_derives_run_from_iteration(self) -> None:
        state = ResearchState(user_request="Derived lineage")
        run = state.add_run()
        iteration = state.create_iteration(run.run_id)
        question = state.add_question("Question", iteration_id=iteration.iteration_id)
        self.assertEqual(question.run_id, run.run_id)

    def test_invalid_question_run_iteration_and_parent_references_are_rejected(self) -> None:
        state = ResearchState(user_request="Invalid question links")
        run1 = state.add_run()
        run2 = state.add_run()
        iteration1 = state.create_iteration(run1.run_id)
        with self.assertRaisesRegex(ValueError, "unknown iteration_id"):
            state.add_question("bad iteration", iteration_id="missing")
        with self.assertRaisesRegex(ValueError, "does not match"):
            state.add_question("bad run", run_id=run2.run_id, iteration_id=iteration1.iteration_id)
        parent = state.add_question("parent", run_id=run1.run_id)
        with self.assertRaisesRegex(ValueError, "same research run"):
            state.add_question("cross-run child", run_id=run2.run_id, parent_question_id=parent.question_id)
        with self.assertRaisesRegex(ValueError, "unknown parent_question_id"):
            state.add_question("orphan", parent_question_id="missing")

    def test_question_rejects_unknown_run(self) -> None:
        state = ResearchState(user_request="Unknown question run")
        with self.assertRaisesRegex(ValueError, "unknown run_id"):
            state.add_question("Q", run_id="missing")

    def test_parent_can_reference_linked_question_in_same_run(self) -> None:
        state = ResearchState(user_request="Parent linkage")
        run = state.add_run()
        parent = state.add_question("Parent", run_id=run.run_id)
        child = state.add_question("Child", run_id=run.run_id, parent_question_id=parent.question_id)
        self.assertEqual(child.parent_question_id, parent.question_id)

    def test_search_manager_records_explicit_run_iteration_question_lineage(self) -> None:
        state = ResearchState(user_request="Search lineage")
        run = state.add_run()
        iteration = state.create_iteration(run.run_id)
        question = state.add_question("Find papers", iteration_id=iteration.iteration_id)
        provider = StaticSearchProvider([
            SearchResult(source_id="s1", title="Result", url="https://example.org/r", content="papers")
        ])
        response = SearchManager(provider).search(
            SearchRequest(query="papers"),
            state,
            run_id=run.run_id,
            iteration_id=iteration.iteration_id,
            question_id=question.question_id,
        )

        action = state.searches[0]
        self.assertEqual((action.run_id, action.iteration_id, action.question_id),
                         (run.run_id, iteration.iteration_id, question.question_id))
        self.assertEqual(response.results[0].search_id, action.search_id)
        self.assertEqual(state.search_results[0].search_id, action.search_id)

    def test_search_manager_derives_run_from_iteration_or_question(self) -> None:
        state = ResearchState(user_request="Derived search lineage")
        run = state.add_run()
        iteration = state.create_iteration(run.run_id)
        question = state.add_question("Find", iteration_id=iteration.iteration_id)
        provider = StaticSearchProvider([])
        action1 = SearchManager(provider).search(
            SearchRequest(query="one"), state, iteration_id=iteration.iteration_id
        )
        action2 = SearchManager(provider).search(
            SearchRequest(query="two"), state, question_id=question.question_id
        )
        self.assertEqual(state.searches[0].run_id, run.run_id)
        self.assertEqual(state.searches[1].run_id, run.run_id)
        self.assertEqual(state.searches[1].question_id, question.question_id)

    def test_search_and_collect_accepts_lineage_without_putting_it_in_request_metadata(self) -> None:
        state = ResearchState(user_request="Convenience API lineage")
        run = state.add_run()
        iteration = state.create_iteration(run.run_id)
        question = state.add_question("Find", iteration_id=iteration.iteration_id)
        response = search_and_collect(
            StaticSearchProvider([]), "query", state=state,
            run_id=run.run_id, iteration_id=iteration.iteration_id,
            question_id=question.question_id, trace_label="test",
        )
        self.assertEqual(state.searches[0].run_id, run.run_id)
        self.assertEqual(state.searches[0].question_id, question.question_id)
        self.assertEqual(response.metadata["deduplicated_count"], 0)

    def test_unlinked_search_action_remains_compatible(self) -> None:
        state = ResearchState(user_request="Legacy search")
        action = state.add_search(SearchAction("legacy query"))
        self.assertIsNone(action.run_id)
        self.assertIsNone(action.iteration_id)
        self.assertIsNone(action.question_id)

    def test_search_rejects_unknown_or_cross_run_references(self) -> None:
        state = ResearchState(user_request="Invalid searches")
        run1 = state.add_run()
        run2 = state.add_run()
        iteration1 = state.create_iteration(run1.run_id)
        question2 = state.add_question("Q2", run_id=run2.run_id)
        with self.assertRaisesRegex(ValueError, "unknown run_id"):
            state.add_search(SearchAction("bad", run_id="missing"))
        with self.assertRaisesRegex(ValueError, "does not match its iteration"):
            state.add_search(SearchAction("bad", run_id=run2.run_id, iteration_id=iteration1.iteration_id))
        with self.assertRaisesRegex(ValueError, "different research runs"):
            state.add_search(SearchAction("bad", run_id=run1.run_id, question_id=question2.question_id))
        with self.assertRaisesRegex(ValueError, "unknown question_id"):
            state.add_search(SearchAction("bad", question_id="missing"))

    def test_search_rejects_unknown_iteration(self) -> None:
        state = ResearchState(user_request="Unknown search iteration")
        with self.assertRaisesRegex(ValueError, "unknown iteration_id"):
            state.add_search(SearchAction("q", iteration_id="missing"))

    def test_search_result_resolves_lineage_through_search_action(self) -> None:
        state = ResearchState(user_request="Result lineage")
        run = state.add_run()
        iteration = state.create_iteration(run.run_id)
        question = state.add_question("Q", iteration_id=iteration.iteration_id)
        action = state.add_search(SearchAction(
            "query", run_id=run.run_id, iteration_id=iteration.iteration_id,
            question_id=question.question_id,
        ))
        result = SearchResult(source_id="s1", title="R", result_id="res-1", search_id=action.search_id, rank=1)
        state.add_search_result(result)

        search = next(item for item in state.searches if item.search_id == result.search_id)
        self.assertEqual((search.run_id, search.iteration_id, search.question_id),
                         (run.run_id, iteration.iteration_id, question.question_id))

    def test_search_result_rejects_unknown_search_and_invalid_rank(self) -> None:
        state = ResearchState(user_request="Invalid result")
        with self.assertRaisesRegex(ValueError, "unknown search_id"):
            state.add_search_result(SearchResult(source_id="s", title="R", search_id="missing"))
        action = state.add_search(SearchAction("q"))
        with self.assertRaisesRegex(ValueError, "positive integer"):
            state.add_search_result(SearchResult(source_id="s", title="R", search_id=action.search_id, rank=0))

    def test_legacy_search_result_without_action_remains_allowed(self) -> None:
        state = ResearchState(user_request="Legacy result")
        result = state.add_search_result(SearchResult(source_id="s", title="legacy"))
        self.assertIsNone(result.search_id)
        self.assertEqual(state.validate_lineage(), [])

    def test_evidence_can_reference_search_result_and_checks_source(self) -> None:
        state = ResearchState(user_request="Evidence lineage")
        action = state.add_search(SearchAction("query"))
        result = state.add_search_result(SearchResult(
            source_id="s1", title="R", result_id="res-1", search_id=action.search_id, rank=1
        ))
        evidence = state.add_evidence(Evidence(text="excerpt", source_id="s1", search_result_id=result.result_id))
        self.assertEqual(evidence.search_result_id, result.result_id)
        with self.assertRaisesRegex(ValueError, "does not match its SearchResult"):
            state.add_evidence(Evidence(text="wrong", source_id="s2", search_result_id=result.result_id))
        with self.assertRaisesRegex(ValueError, "unknown search_result_id"):
            state.add_evidence(Evidence(text="missing", source_id="s1", search_result_id="missing"))

    def test_full_lineage_round_trip_from_run_to_evidence(self) -> None:
        state = ResearchState(user_request="End-to-end lineage")
        run = state.add_run(ResearchRun(status="running"))
        iteration = state.create_iteration(run.run_id, status="completed", metadata={"note": "follow-up"})
        question = state.add_question("Find evidence", iteration_id=iteration.iteration_id)
        provider = StaticSearchProvider([
            SearchResult(source_id="source-1", title="Paper", url="https://example.org/paper", content="evidence")
        ])
        response = SearchManager(provider).search(
            SearchRequest(query="evidence"), state,
            run_id=run.run_id,
            iteration_id=iteration.iteration_id,
            question_id=question.question_id,
        )
        result = response.results[0]
        version = state.add_document_version(DocumentVersion(
            document_id="doc-1", source_id=result.source_id, version_id="version-1"
        ))
        passage = state.add_passage_reference(PassageReference(
            document_version_id=version.version_id, locator_type="page", page=2, chunk_id="chunk-2"
        ))
        evidence = state.add_evidence(Evidence(
            text="retrieved passage", source_id=result.source_id,
            document_version_id=version.version_id,
            passage_reference_id=passage.passage_reference_id,
            search_result_id=result.result_id,
        ))

        restored = ResearchState.from_dict(state.to_dict())
        restored_evidence = restored.evidence[0]
        restored_result = next(item for item in restored.search_results if item.result_id == restored_evidence.search_result_id)
        restored_search = next(item for item in restored.searches if item.search_id == restored_result.search_id)
        restored_question = next(item for item in restored.research_questions if item.question_id == restored_search.question_id)
        restored_iteration = next(item for item in restored.iterations if item.iteration_id == restored_question.iteration_id)
        restored_run = next(item for item in restored.research_runs if item.run_id == restored_iteration.run_id)

        self.assertEqual(restored_evidence.evidence_id, evidence.evidence_id)
        self.assertEqual(restored_run.run_id, run.run_id)
        self.assertEqual(restored_iteration.number, 1)
        self.assertEqual(restored_question.question_id, question.question_id)
        self.assertEqual(restored_search.search_id, result.search_id)
        self.assertEqual(restored_result.source_id, restored_evidence.source_id)
        self.assertEqual(restored.validate_lineage(), [])

    def test_old_state_without_run_iteration_or_link_fields_loads(self) -> None:
        state = ResearchState(user_request="Legacy state")
        state.add_question("legacy question")
        state.add_search(SearchAction("legacy search"))
        payload = state.to_dict()
        payload.pop("research_runs")
        payload.pop("iterations")
        for question in payload["research_questions"]:
            question.pop("run_id")
            question.pop("iteration_id")
            question.pop("parent_question_id")
        for search in payload["searches"]:
            search.pop("run_id")
            search.pop("iteration_id")
            search.pop("question_id")

        restored = ResearchState.from_dict(payload)
        self.assertEqual(restored.research_runs, [])
        self.assertEqual(restored.iterations, [])
        self.assertIsNone(restored.searches[0].run_id)
        self.assertIsNone(restored.research_questions[0].parent_question_id)

    def test_malformed_serialized_lineage_fails_explicitly(self) -> None:
        state = ResearchState(user_request="Malformed lineage")
        state.add_search(SearchAction("legacy"))
        payload = state.to_dict()
        payload["searches"][0]["run_id"] = "missing-run"
        with self.assertRaisesRegex(ValueError, "Invalid serialized lineage"):
            ResearchState.from_dict(payload)

    def test_malformed_serialized_question_parent_fails_explicitly(self) -> None:
        state = ResearchState(user_request="Malformed parent")
        state.add_question("child", parent_question_id=None)
        payload = state.to_dict()
        payload["research_questions"][0]["parent_question_id"] = "missing-parent"
        with self.assertRaisesRegex(ValueError, "Invalid serialized lineage"):
            ResearchState.from_dict(payload)


if __name__ == "__main__":
    unittest.main()
