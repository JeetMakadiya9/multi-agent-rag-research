from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research_retrieval import (
    ExistingRAGProvider,
    RetrievalEvidence,
    adapt_retrieval_evidence_to_state,
    register_rag_document_version,
    record_rag_retrieval,
)
from research_state import DocumentVersion, Evidence, ResearchState


class RAGRetrievalLineageTests(unittest.TestCase):
    def _item(self, *, identifier: str, source_id: str, text: str, rank_fields: dict | None = None):
        fields = rank_fields or {}
        return RetrievalEvidence(
            evidence_id=identifier,
            text=text,
            source_id=source_id,
            filename="",
            page=fields.get("page", 1),
            section=fields.get("section", ""),
            document_id=fields.get("document_id", ""),
            chunk_id=fields.get("chunk_id", identifier),
            score=fields.get("score", 0.5),
            retrieval_metadata={"trace": "synthetic"},
        )

    def test_retrieval_creates_action_with_query_provider_count_and_purpose(self) -> None:
        state = ResearchState(user_request="Record RAG retrieval")
        results = [self._item(identifier="c1", source_id="s1", text="one")]
        response = record_rag_retrieval(
            state, query="limitation of method X", retrieval_results=results,
            provider="local_rag", purpose="retrieval",
        )

        action = state.searches[0]
        self.assertEqual(action.query, "limitation of method X")
        self.assertEqual(action.provider, "local_rag")
        self.assertEqual(action.result_count, 1)
        self.assertEqual(action.purpose, "retrieval")
        self.assertEqual(response.searched_at, action.created_at)

    def test_rag_result_order_and_identity_are_preserved(self) -> None:
        state = ResearchState(user_request="Preserve retrieval order")
        results = [
            self._item(identifier="chunk-a", source_id="s-a", text="first"),
            self._item(identifier="chunk-b", source_id="s-b", text="second"),
            self._item(identifier="chunk-c", source_id="s-c", text="third"),
        ]
        response = record_rag_retrieval(state, query="q", retrieval_results=results)

        self.assertEqual([result.rank for result in response.results], [1, 2, 3])
        self.assertEqual([result.source_id for result in response.results], ["s-a", "s-b", "s-c"])
        self.assertTrue(all(result.search_id == state.searches[0].search_id for result in response.results))
        self.assertEqual(len({result.result_id for result in response.results}), 3)
        self.assertTrue(all(result.result_id for result in response.results))

    def test_result_metadata_and_only_available_source_fields_are_mapped(self) -> None:
        state = ResearchState(user_request="Metadata mapping")
        item = self._item(
            identifier="chunk-1", source_id="s1", text="exact passage",
            rank_fields={"page": 4, "section": "Methods", "document_id": "doc-1", "score": 0.73},
        )
        result = record_rag_retrieval(state, query="q", retrieval_results=[item]).results[0]

        self.assertEqual(result.content, "exact passage")
        self.assertEqual(result.title, "")
        self.assertEqual(result.url, "")
        self.assertIsNone(result.published_at)
        self.assertEqual(result.authors, [])
        self.assertEqual(result.source_type, "unknown")
        self.assertEqual(result.metadata["retrieval_evidence_id"], "chunk-1")
        self.assertEqual(result.metadata["page"], 4)
        self.assertEqual(result.metadata["section"], "Methods")
        self.assertEqual(result.metadata["retrieval_scores"]["score"], 0.73)
        self.assertEqual(result.metadata["retrieval_metadata"], {"trace": "synthetic"})

    def test_selected_rank_becomes_linked_provenance_evidence(self) -> None:
        state = ResearchState(user_request="Select one result as evidence")
        items = [
            self._item(identifier="chunk-a", source_id="doc.pdf:page-1", text="not selected",
                       rank_fields={"document_id": "doc", "page": 1}),
            self._item(identifier="chunk-b", source_id="doc.pdf:page-2", text="selected exact text",
                       rank_fields={"document_id": "doc", "page": 2}),
        ]
        version = register_rag_document_version(
            state, source_id="doc.pdf:page-2", document_id="doc", filename="doc.pdf", version_id="ver-1"
        )
        response = record_rag_retrieval(
            state, query="claim", retrieval_results=items,
            evidence_document_versions={2: version},
        )

        self.assertEqual(len(state.searches), 1)
        self.assertEqual(len(state.search_results), 2)
        self.assertEqual(len(state.evidence), 1)
        evidence = state.evidence[0]
        selected_result = response.results[1]
        self.assertEqual(evidence.text, "selected exact text")
        self.assertEqual(evidence.search_result_id, selected_result.result_id)
        self.assertEqual(evidence.source_id, selected_result.source_id)
        self.assertEqual(evidence.document_version_id, version.version_id)
        passage = next(item for item in state.passage_references if item.passage_reference_id == evidence.passage_reference_id)
        self.assertEqual((passage.locator_type, passage.page, passage.chunk_id), ("page", 2, "chunk-b"))

    def test_run_iteration_question_lineage_reaches_recorded_rag_search(self) -> None:
        state = ResearchState(user_request="Lineaged retrieval")
        run = state.add_run()
        iteration = state.create_iteration(run.run_id)
        question = state.add_question("What is the limitation?", iteration_id=iteration.iteration_id)
        response = record_rag_retrieval(
            state, query=question.text,
            retrieval_results=[self._item(identifier="c1", source_id="s1", text="evidence")],
            run_id=run.run_id, iteration_id=iteration.iteration_id, question_id=question.question_id,
        )
        result = response.results[0]
        search = next(item for item in state.searches if item.search_id == result.search_id)
        self.assertEqual((search.run_id, search.iteration_id, search.question_id),
                         (run.run_id, iteration.iteration_id, question.question_id))

    def test_unlinked_retrieval_is_recorded_without_creating_evidence(self) -> None:
        state = ResearchState(user_request="Legacy retrieval")
        response = record_rag_retrieval(
            state, query="q", retrieval_results=[self._item(identifier="c1", source_id="s1", text="hit")]
        )
        self.assertEqual(len(state.searches), 1)
        self.assertIsNone(state.searches[0].run_id)
        self.assertEqual(len(state.search_results), 1)
        self.assertEqual(state.evidence, [])
        self.assertIsNone(state.searches[0].run_id)
        self.assertEqual(response.results[0].rank, 1)

    def test_same_source_multiple_results_have_distinct_ids_and_single_source_record(self) -> None:
        state = ResearchState(user_request="Repeated source chunks")
        items = [
            self._item(identifier="same", source_id="doc:page-1", text="chunk one"),
            self._item(identifier="same", source_id="doc:page-1", text="chunk two"),
        ]
        response = record_rag_retrieval(state, query="q", retrieval_results=items)
        self.assertEqual(len(response.results), 2)
        self.assertNotEqual(response.results[0].result_id, response.results[1].result_id)
        self.assertEqual([item.rank for item in response.results], [1, 2])
        self.assertEqual(len([source for source in state.sources if source.source_id == "doc:page-1"]), 1)

    def test_multiple_selected_evidence_link_to_their_exact_results(self) -> None:
        state = ResearchState(user_request="Map several selected results")
        items = [
            self._item(identifier="duplicate-chunk", source_id="doc:page-1", text="first", rank_fields={"document_id": "d"}),
            self._item(identifier="duplicate-chunk", source_id="doc:page-2", text="second", rank_fields={"document_id": "d", "page": 2}),
        ]
        v1 = register_rag_document_version(state, source_id="doc:page-1", document_id="d", filename="doc", version_id="v1")
        v2 = register_rag_document_version(state, source_id="doc:page-2", document_id="d", filename="doc", version_id="v2")
        response = record_rag_retrieval(
            state, query="q", retrieval_results=items,
            evidence_document_versions={1: v1, 2: v2},
        )
        by_result = {item.result_id: item for item in response.results}
        self.assertEqual(len(state.evidence), 2)
        for evidence in state.evidence:
            self.assertIn(evidence.search_result_id, by_result)
            self.assertEqual(evidence.source_id, by_result[evidence.search_result_id].source_id)
        self.assertEqual([item.text for item in state.evidence], ["first", "second"])

    def test_mismatched_document_version_source_is_rejected_before_recording(self) -> None:
        state = ResearchState(user_request="Wrong version")
        item = self._item(identifier="c1", source_id="one:page-1", text="text", rank_fields={"document_id": "d"})
        version = register_rag_document_version(
            state, source_id="other:page-1", document_id="d", filename="other", version_id="v"
        )
        with self.assertRaisesRegex(ValueError, "source_id does not match"):
            record_rag_retrieval(state, query="q", retrieval_results=[item], evidence_document_versions={1: version})
        self.assertEqual(state.searches, [])

    def test_unregistered_version_and_invalid_selection_are_rejected(self) -> None:
        state = ResearchState(user_request="Missing version")
        items = [self._item(identifier="c1", source_id="s1", text="text")]
        version = DocumentVersion(document_id="d", source_id="s1", version_id="unregistered")
        with self.assertRaisesRegex(ValueError, "must be registered"):
            record_rag_retrieval(state, query="q", retrieval_results=items, evidence_document_versions={1: version})
        with self.assertRaisesRegex(ValueError, "outside the retrieved result order"):
            record_rag_retrieval(state, query="q", retrieval_results=items, evidence_document_versions={2: version})
        self.assertEqual(state.searches, [])

    def test_duplicate_retrieval_evidence_ids_are_disambiguated_by_rank(self) -> None:
        state = ResearchState(user_request="Repeated provider IDs")
        items = [
            self._item(identifier="same-id", source_id="doc:page-1", text="first", rank_fields={"document_id": "d"}),
            self._item(identifier="same-id", source_id="doc:page-2", text="second", rank_fields={"document_id": "d", "page": 2}),
        ]
        v2 = register_rag_document_version(state, source_id="doc:page-2", document_id="d", filename="doc", version_id="v2")
        response = record_rag_retrieval(
            state, query="q", retrieval_results=items, evidence_document_versions={2: v2}
        )
        self.assertEqual(len(state.evidence), 1)
        self.assertEqual(state.evidence[0].text, "second")
        self.assertEqual(state.evidence[0].search_result_id, response.results[1].result_id)

    def test_full_run_to_source_lineage_round_trips(self) -> None:
        state = ResearchState(user_request="Full RAG lineage")
        run = state.add_run()
        iteration = state.create_iteration(run.run_id)
        question = state.add_question("Find the limitation", iteration_id=iteration.iteration_id)
        item = self._item(
            identifier="chunk-9", source_id="paper.pdf:page-9", text="The exact passage.",
            rank_fields={"document_id": "paper-doc", "page": 9, "section": "Limitations"},
        )
        version = register_rag_document_version(
            state, source_id=item.source_id, document_id="paper-doc", filename="paper.pdf", version_id="paper-v1"
        )
        response = record_rag_retrieval(
            state, query=question.text, retrieval_results=[item], run_id=run.run_id,
            iteration_id=iteration.iteration_id, question_id=question.question_id,
            evidence_document_versions={1: version},
        )

        restored = ResearchState.from_dict(state.to_dict())
        evidence = restored.evidence[0]
        result = next(item for item in restored.search_results if item.result_id == evidence.search_result_id)
        search = next(item for item in restored.searches if item.search_id == result.search_id)
        restored_question = next(item for item in restored.research_questions if item.question_id == search.question_id)
        restored_iteration = next(item for item in restored.iterations if item.iteration_id == restored_question.iteration_id)
        restored_run = next(item for item in restored.research_runs if item.run_id == restored_iteration.run_id)
        passage = next(item for item in restored.passage_references if item.passage_reference_id == evidence.passage_reference_id)
        version = next(item for item in restored.document_versions if item.version_id == passage.document_version_id)
        source = next(item for item in restored.sources if item.source_id == version.source_id)

        self.assertEqual(response.results[0].rank, 1)
        self.assertEqual(restored_run.run_id, run.run_id)
        self.assertEqual(restored_question.question_id, question.question_id)
        self.assertEqual(search.search_id, result.search_id)
        self.assertEqual(result.result_id, evidence.search_result_id)
        self.assertEqual(evidence.text, "The exact passage.")
        self.assertEqual(passage.chunk_id, "chunk-9")
        self.assertEqual(version.document_id, "paper-doc")
        self.assertEqual(source.source_id, item.source_id)
        self.assertEqual(restored.validate_lineage(), [])

    def test_actual_rag_retrieval_output_flows_through_lineage_bridge(self) -> None:
        # Uses rag.retrieve_chunks_advanced with synthetic input and a
        # deterministic reranker stub; no embedding or LLM model is invoked.
        import rag

        chunks = [{
            "chunk_id": 1,
            "document_id": "synthetic-doc",
            "filename": "synthetic.pdf",
            "page": 1,
            "section": "Limitations",
            "text": "Method X has a limitation in low-resource settings.",
        }]
        reranker_type = type(
            "DeterministicReranker",
            (),
            {"predict": lambda self, pairs, show_progress_bar=False: [1.0 for _ in pairs]},
        )
        provider = ExistingRAGProvider(
            chunks=chunks,
            embedding_model=None,
            reranker=reranker_type(),
            faiss_index=None,
            bm25_index=None,
            retrieval_module=rag,
        )
        retrieval_response = provider.retrieve("Method X limitation")
        self.assertEqual(len(retrieval_response.evidence), 1)
        retrieved = retrieval_response.evidence[0]

        state = ResearchState(user_request="Trace actual RAG output")
        run = state.add_run()
        iteration = state.create_iteration(run.run_id)
        question = state.add_question("Method X limitation", iteration_id=iteration.iteration_id)
        version = register_rag_document_version(
            state,
            source_id=retrieved.source_id,
            document_id=retrieved.document_id,
            filename=retrieved.filename,
            version_id="synthetic-version",
        )
        recorded = record_rag_retrieval(
            state,
            query=question.text,
            retrieval_results=retrieval_response.evidence,
            run_id=run.run_id,
            iteration_id=iteration.iteration_id,
            question_id=question.question_id,
            provider=retrieval_response.provider,
            evidence_document_versions={1: version},
        )

        self.assertEqual(recorded.results[0].content, retrieved.text)
        self.assertEqual(state.evidence[0].search_result_id, recorded.results[0].result_id)
        self.assertEqual(state.evidence[0].text, retrieved.text)
        self.assertEqual(state.validate_lineage(), [])

    def test_invalid_search_result_link_is_rejected_by_adapter_and_state(self) -> None:
        state = ResearchState(user_request="Invalid result link")
        item = self._item(identifier="c1", source_id="s1", text="passage", rank_fields={"document_id": "d"})
        version = register_rag_document_version(state, source_id="s1", document_id="d", filename="doc", version_id="v")
        with self.assertRaisesRegex(ValueError, "Unknown search_result_id"):
            adapt_retrieval_evidence_to_state(item, version, state, search_result_id="missing")
        self.assertEqual(state.passage_references, [])

    def test_evidence_source_must_match_linked_search_result(self) -> None:
        state = ResearchState(user_request="Search result source validation")
        record_rag_retrieval(
            state, query="q", retrieval_results=[self._item(identifier="c1", source_id="source-a", text="hit")]
        )
        result = state.search_results[0]
        with self.assertRaisesRegex(ValueError, "does not match its SearchResult"):
            state.add_evidence(Evidence(text="hit", source_id="source-b", search_result_id=result.result_id))


if __name__ == "__main__":
    unittest.main()
