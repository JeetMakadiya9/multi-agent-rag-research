from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research_retrieval import ExistingRAGProvider
from research_state import ResearchState
from research_workflow import ResearchLineageRecordingError, retrieve_for_research
from research_retrieval import register_rag_document_version


def _synthetic_provider() -> ExistingRAGProvider:
    """Use the real frozen rag retrieval function with synthetic chunks."""
    import rag

    chunks = [
        {
            "chunk_id": 1,
            "document_id": "synthetic-doc-x",
            "filename": "method-x.pdf",
            "page": 1,
            "section": "Limitations",
            "text": "Method X has a limitation in low-resource settings.",
        },
        {
            "chunk_id": 2,
            "document_id": "synthetic-doc-y",
            "filename": "method-y.pdf",
            "page": 1,
            "section": "Evaluation",
            "text": "Method Y uses a benchmark dataset for evaluation.",
        },
    ]
    reranker_type = type(
        "DeterministicReranker",
        (),
        {"predict": lambda self, pairs, show_progress_bar=False: [1.0 for _ in pairs]},
    )
    return ExistingRAGProvider(
        chunks=chunks,
        embedding_model=None,
        reranker=reranker_type(),
        faiss_index=None,
        bm25_index=None,
        retrieval_module=rag,
    )


class ResearchWorkflowIntegrationTests(unittest.TestCase):
    def test_workflow_records_full_selected_evidence_lineage_after_actual_rag_retrieval(self) -> None:
        state = ResearchState(user_request="Research method X limitations")
        run = state.add_run()
        iteration = state.create_iteration(run.run_id)
        question = state.add_question(
            "What limitation is documented for method X?",
            iteration_id=iteration.iteration_id,
        )
        version = register_rag_document_version(
            state,
            source_id="method-x.pdf:page-1",
            document_id="synthetic-doc-x",
            filename="method-x.pdf",
            version_id="method-x-v1",
        )

        result = retrieve_for_research(
            _synthetic_provider(),
            state,
            query=question.text,
            run_id=run.run_id,
            iteration_id=iteration.iteration_id,
            question_id=question.question_id,
            evidence_document_versions={1: version},
        )

        retrieved = result.retrieval_response.evidence[0]
        search_result = result.search_response.results[0]
        self.assertEqual(search_result.content, retrieved.text)
        self.assertEqual(search_result.rank, 1)
        self.assertEqual(search_result.source_id, retrieved.source_id)
        self.assertEqual(len(state.searches), 1)
        search = state.searches[0]
        self.assertEqual(search.query, question.text)
        self.assertEqual(search.provider, "existing_advanced_rag")
        self.assertEqual(search.purpose, "retrieval")
        self.assertEqual(search.result_count, len(result.retrieval_response.evidence))
        self.assertEqual((search.run_id, search.iteration_id, search.question_id),
                         (run.run_id, iteration.iteration_id, question.question_id))

        evidence = state.evidence[0]
        self.assertEqual(evidence.search_result_id, search_result.result_id)
        self.assertEqual(evidence.document_version_id, version.version_id)
        self.assertTrue(evidence.passage_reference_id)
        self.assertEqual(evidence.source_id, search_result.source_id)
        self.assertTrue(evidence.text_hash)

        serialized = ResearchState.from_dict(state.to_dict())
        restored_evidence = serialized.evidence[0]
        restored_result = next(
            item for item in serialized.search_results
            if item.result_id == restored_evidence.search_result_id
        )
        restored_search = next(
            item for item in serialized.searches if item.search_id == restored_result.search_id
        )
        restored_question = next(
            item for item in serialized.research_questions
            if item.question_id == restored_search.question_id
        )
        restored_iteration = next(
            item for item in serialized.iterations
            if item.iteration_id == restored_question.iteration_id
        )
        restored_run = next(
            item for item in serialized.research_runs
            if item.run_id == restored_iteration.run_id
        )
        restored_passage = next(
            item for item in serialized.passage_references
            if item.passage_reference_id == restored_evidence.passage_reference_id
        )
        restored_version = next(
            item for item in serialized.document_versions
            if item.version_id == restored_passage.document_version_id
        )
        restored_source = next(
            item for item in serialized.sources if item.source_id == restored_version.source_id
        )

        self.assertEqual(restored_run.run_id, run.run_id)
        self.assertEqual(restored_iteration.number, 1)
        self.assertEqual(restored_question.question_id, question.question_id)
        self.assertEqual(restored_search.search_id, search.search_id)
        self.assertEqual(restored_result.result_id, search_result.result_id)
        self.assertEqual(restored_evidence.text, retrieved.text)
        self.assertEqual(restored_source.source_id, retrieved.source_id)
        self.assertEqual(serialized.validate_lineage(), [])
        self.assertEqual(serialized.validate_provenance(), [])

    def test_multiple_queries_and_iterations_get_distinct_correct_searches(self) -> None:
        state = ResearchState(user_request="Research two methods")
        run = state.add_run()
        iteration1 = state.create_iteration(run.run_id)
        iteration2 = state.create_iteration(run.run_id)
        q1 = state.add_question("Method X limitation", iteration_id=iteration1.iteration_id)
        q2 = state.add_question("Method Y benchmark dataset", iteration_id=iteration2.iteration_id)
        provider = _synthetic_provider()

        first = retrieve_for_research(
            provider, state, query=q1.text, run_id=run.run_id,
            iteration_id=iteration1.iteration_id, question_id=q1.question_id,
        )
        second = retrieve_for_research(
            provider, state, query=q2.text, run_id=run.run_id,
            iteration_id=iteration2.iteration_id, question_id=q2.question_id,
        )

        self.assertEqual(len(state.searches), 2)
        self.assertNotEqual(first.search_response.results[0].search_id, second.search_response.results[0].search_id)
        self.assertEqual(first.search_response.results[0].rank, 1)
        self.assertEqual(second.search_response.results[0].rank, 1)
        self.assertEqual(state.searches[0].question_id, q1.question_id)
        self.assertEqual(state.searches[1].question_id, q2.question_id)
        self.assertEqual(state.searches[0].iteration_id, iteration1.iteration_id)
        self.assertEqual(state.searches[1].iteration_id, iteration2.iteration_id)
        self.assertTrue(all(item.search_id == state.searches[0].search_id for item in state.search_results[:len(first.search_response.results)]))
        self.assertTrue(all(item.search_id == state.searches[1].search_id for item in state.search_results[len(first.search_response.results):]))
        self.assertEqual(state.evidence, [])  # no versions were selected

    def test_legacy_retrieval_without_state_still_works(self) -> None:
        provider = _synthetic_provider()
        response = provider.retrieve("Method X limitation")
        self.assertTrue(response.evidence)
        self.assertEqual(response.evidence[0].text, "Method X has a limitation in low-resource settings.")

    def test_workflow_can_record_unlinked_activity_without_creating_evidence(self) -> None:
        state = ResearchState(user_request="Unlinked workflow")
        result = retrieve_for_research(
            _synthetic_provider(), state, query="Method X limitation"
        )
        self.assertEqual(len(state.searches), 1)
        self.assertIsNone(state.searches[0].run_id)
        self.assertEqual(len(state.search_results), len(result.retrieval_response.evidence))
        self.assertEqual(state.evidence, [])

    def test_recording_failure_is_explicit_and_preserves_raw_retrieval_response(self) -> None:
        state = ResearchState(user_request="Recording failure")
        with self.assertRaises(ResearchLineageRecordingError) as caught:
            retrieve_for_research(
                _synthetic_provider(), state, query="Method X limitation", run_id="missing-run"
            )

        self.assertTrue(caught.exception.retrieval_response.evidence)
        self.assertIn("unknown run_id", str(caught.exception))
        self.assertEqual(state.searches, [])


if __name__ == "__main__":
    unittest.main()
