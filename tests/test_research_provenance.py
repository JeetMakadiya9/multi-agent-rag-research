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
    StaticRetrievalProvider,
    adapt_retrieval_evidence_to_state,
    passage_reference_from_retrieval_evidence,
    register_rag_document_version,
)
from research_state import (
    DocumentVersion,
    Evidence,
    PassageReference,
    ResearchState,
    SearchAction,
    Source,
)
from research_tools import SearchResult


class ResearchProvenanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = ResearchState(user_request="Trace a source passage")
        self.source = self.state.add_source(
            Source(
                source_id="source-paper",
                title="Example paper",
                url="https://example.org/paper",
                source_type="paper",
            )
        )

    def test_document_versions_have_distinct_ids_and_round_trip(self) -> None:
        first = self.state.add_document_version(
            DocumentVersion(document_id="paper-1", source_id=self.source.source_id)
        )
        second = self.state.add_document_version(
            DocumentVersion(document_id="paper-1", source_id=self.source.source_id)
        )

        self.assertTrue(first.version_id.startswith("version_"))
        self.assertNotEqual(first.version_id, second.version_id)
        self.assertEqual(first.document_id, second.document_id)

        restored = ResearchState.from_dict(self.state.to_dict())
        self.assertEqual(restored.document_versions, [first, second])

    def test_pdf_page_locator_supports_optional_chunk_and_span(self) -> None:
        version = DocumentVersion(document_id="pdf-1", source_id=self.source.source_id)
        reference = PassageReference(
            document_version_id=version.version_id,
            locator_type="page",
            page=12,
            chunk_id="chunk-12a",
            section="Methods",
            start_char=15,
            end_char=42,
        )

        self.assertEqual(reference.locator_type, "page")
        self.assertEqual(reference.page, 12)
        self.assertEqual(reference.chunk_id, "chunk-12a")
        self.assertEqual((reference.start_char, reference.end_char), (15, 42))

    def test_chunk_and_sentence_locators_are_typed_and_validated(self) -> None:
        version = DocumentVersion(document_id="doc-1", source_id=self.source.source_id)
        chunk = PassageReference(
            document_version_id=version.version_id,
            locator_type="chunk",
            chunk_id="rag-chunk-9",
            page=3,
            section="Results",
        )
        sentence = PassageReference(
            document_version_id=version.version_id,
            locator_type="sentence",
            sentence_index=0,
            section="Abstract",
        )
        span = PassageReference(
            document_version_id=version.version_id,
            locator_type="span",
            start_char=5,
            end_char=15,
            section="Introduction",
        )

        self.assertEqual(chunk.chunk_id, "rag-chunk-9")
        self.assertEqual(sentence.sentence_index, 0)
        self.assertEqual((span.start_char, span.end_char), (5, 15))
        with self.assertRaisesRegex(ValueError, "zero-based sentence_index"):
            PassageReference(document_version_id=version.version_id, locator_type="sentence")
        with self.assertRaisesRegex(ValueError, "requires chunk_id"):
            PassageReference(document_version_id=version.version_id, locator_type="chunk")

    def test_retrieval_evidence_maps_page_chunk_and_section(self) -> None:
        version = DocumentVersion(document_id="doc-7", source_id=self.source.source_id)
        retrieved = RetrievalEvidence(
            evidence_id="ev-rag-1",
            text="The exact retrieved text.",
            source_id="paper.pdf:page-4",
            filename="paper.pdf",
            page=4,
            section="Evaluation",
            document_id="doc-7",
            chunk_id="chunk-4b",
        )

        reference = passage_reference_from_retrieval_evidence(retrieved, version)

        self.assertEqual(reference.document_version_id, version.version_id)
        self.assertEqual(reference.locator_type, "page")
        self.assertEqual(reference.chunk_id, "chunk-4b")
        self.assertEqual(reference.page, 4)
        self.assertEqual(reference.section, "Evaluation")

        chunk_only = RetrievalEvidence(
            evidence_id="ev-rag-2",
            text="Chunk only",
            source_id=self.source.source_id,
            filename="paper.pdf",
            page=0,
            document_id="doc-7",
            chunk_id="chunk-only",
        )
        chunk_reference = passage_reference_from_retrieval_evidence(chunk_only, version)
        self.assertEqual(chunk_reference.locator_type, "chunk")
        self.assertEqual(chunk_reference.chunk_id, "chunk-only")
        self.assertIsNone(chunk_reference.page)

    def test_retrieval_mapping_rejects_wrong_document_or_missing_locator(self) -> None:
        version = DocumentVersion(document_id="doc-7", source_id=self.source.source_id)
        wrong_document = RetrievalEvidence(
            evidence_id="ev-wrong",
            text="Text",
            source_id="source-paper",
            filename="paper.pdf",
            page=2,
            document_id="another-doc",
        )
        no_locator = RetrievalEvidence(
            evidence_id="ev-no-location",
            text="Text",
            source_id="source-paper",
            filename="paper.pdf",
            page=0,
            document_id="doc-7",
        )

        with self.assertRaisesRegex(ValueError, "does not match"):
            passage_reference_from_retrieval_evidence(wrong_document, version)
        with self.assertRaisesRegex(ValueError, "neither a usable chunk_id"):
            passage_reference_from_retrieval_evidence(no_locator, version)

        negative_page = RetrievalEvidence(
            evidence_id="ev-negative-page",
            text="Text",
            source_id="source-paper",
            filename="paper.pdf",
            page=-1,
            document_id="doc-7",
            chunk_id="chunk-1",
        )
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            passage_reference_from_retrieval_evidence(negative_page, version)

        invalid_chunk = RetrievalEvidence(
            evidence_id="ev-invalid-chunk",
            text="Text",
            source_id="source-paper",
            filename="paper.pdf",
            page=0,
            document_id="doc-7",
            chunk_id="   ",
        )
        with self.assertRaisesRegex(ValueError, "only whitespace"):
            passage_reference_from_retrieval_evidence(invalid_chunk, version)

        invalid_page_type = RetrievalEvidence(
            evidence_id="ev-invalid-page-type",
            text="Text",
            source_id="source-paper",
            filename="paper.pdf",
            page=1.5,
            document_id="doc-7",
            chunk_id="chunk-1",
        )
        with self.assertRaisesRegex(ValueError, "page must be an integer"):
            passage_reference_from_retrieval_evidence(invalid_page_type, version)

    def test_register_rag_document_version_creates_source_and_preserves_given_ids(self) -> None:
        state = ResearchState(user_request="Register local RAG document")
        version = register_rag_document_version(
            state,
            source_id="paper.pdf:page-4",
            document_id="paper-doc-42",
            filename="paper.pdf",
            version_id="version-caller-supplied",
            captured_at="2026-09-27T00:00:00+00:00",
            content_hash="caller-supplied-sha256",
            metadata={"collection": "local"},
        )

        self.assertEqual(len(state.sources), 1)
        self.assertEqual(state.sources[0].source_id, "paper.pdf:page-4")
        self.assertEqual(state.sources[0].title, "paper.pdf")
        self.assertEqual(state.sources[0].metadata["filename"], "paper.pdf")
        self.assertEqual(version.source_id, state.sources[0].source_id)
        self.assertEqual(version.document_id, "paper-doc-42")
        self.assertEqual(version.version_id, "version-caller-supplied")
        self.assertEqual(version.captured_at, "2026-09-27T00:00:00+00:00")
        self.assertEqual(version.metadata["content_hash"], "caller-supplied-sha256")
        self.assertEqual(version.metadata["collection"], "local")

    def _registered_rag_version(self) -> DocumentVersion:
        return register_rag_document_version(
            self.state,
            source_id=self.source.source_id,
            document_id="doc-7",
            filename="paper.pdf",
            version_id="version-rag-7",
        )

    @staticmethod
    def _retrieval_item(**overrides) -> RetrievalEvidence:
        fields = {
            "evidence_id": "rag-result-4b",
            "text": "  Exact retrieved text.\nKeep it unchanged.  ",
            "source_id": "source-paper",
            "filename": "paper.pdf",
            "page": 4,
            "section": "Evaluation",
            "score": 0.81,
            "retrieval_metadata": {"rrf_score": 0.04, "rank_trace": [2, 1]},
            "document_id": "doc-7",
            "chunk_id": "chunk-4b",
            "retrieval_method": "existing_advanced_rag",
            "retrieval_score": 0.81,
            "reranker_score": 0.77,
            "semantic_score": 0.7,
            "lexical_score": 0.3,
        }
        fields.update(overrides)
        return RetrievalEvidence(**fields)

    def test_adapter_creates_complete_chain_preserving_text_and_scores(self) -> None:
        version = self._registered_rag_version()
        retrieval = self._retrieval_item()

        evidence = adapt_retrieval_evidence_to_state(retrieval, version, self.state)

        reference = next(
            item for item in self.state.passage_references
            if item.passage_reference_id == evidence.passage_reference_id
        )
        self.assertEqual(evidence.text, retrieval.text)
        self.assertEqual(evidence.source_id, retrieval.source_id)
        self.assertEqual(evidence.document_version_id, version.version_id)
        self.assertEqual(reference.document_version_id, version.version_id)
        self.assertEqual(reference.locator_type, "page")
        self.assertEqual(reference.page, 4)
        self.assertEqual(reference.chunk_id, "chunk-4b")
        self.assertEqual(reference.section, "Evaluation")
        self.assertEqual(evidence.metadata["retrieval_metadata"], retrieval.retrieval_metadata)
        self.assertEqual(evidence.metadata["retrieval_evidence_id"], retrieval.evidence_id)
        self.assertEqual(evidence.metadata["retrieval_result_score"], retrieval.score)
        self.assertEqual(
            evidence.metadata["retrieval_scores"],
            {
                "retrieval_score": retrieval.retrieval_score,
                "reranker_score": retrieval.reranker_score,
                "semantic_score": retrieval.semantic_score,
                "lexical_score": retrieval.lexical_score,
            },
        )
        self.assertIsNone(evidence.relevance_score)
        self.assertEqual(self.state.validate_provenance(), [])

    def test_adapter_rejects_unregistered_or_incompatible_document_versions(self) -> None:
        retrieval = self._retrieval_item()
        unregistered = DocumentVersion(document_id="doc-7", source_id=self.source.source_id)
        with self.assertRaisesRegex(ValueError, "must be registered"):
            adapt_retrieval_evidence_to_state(retrieval, unregistered, self.state)

        version = self._registered_rag_version()
        with self.assertRaisesRegex(ValueError, "source_id does not match"):
            adapt_retrieval_evidence_to_state(
                self._retrieval_item(source_id="other-source"), version, self.state
            )
        with self.assertRaisesRegex(ValueError, "document_id does not match"):
            adapt_retrieval_evidence_to_state(
                self._retrieval_item(document_id="other-document"), version, self.state
            )

    def test_repeated_chunk_reuses_passage_but_creates_distinct_evidence(self) -> None:
        version = self._registered_rag_version()
        first = adapt_retrieval_evidence_to_state(self._retrieval_item(), version, self.state)
        second = adapt_retrieval_evidence_to_state(
            self._retrieval_item(evidence_id="same-chunk-different-query"),
            version,
            self.state,
        )

        self.assertEqual(first.passage_reference_id, second.passage_reference_id)
        self.assertNotEqual(first.evidence_id, second.evidence_id)
        self.assertEqual(len(self.state.passage_references), 1)
        self.assertEqual(len(self.state.evidence), 2)

    def test_adapter_preserves_missing_document_id_without_inventing_it(self) -> None:
        version = self._registered_rag_version()
        retrieval = self._retrieval_item(document_id="")

        evidence = adapt_retrieval_evidence_to_state(retrieval, version, self.state)

        self.assertEqual(evidence.document_version_id, version.version_id)
        self.assertEqual(version.document_id, "doc-7")

    def test_existing_retrieval_path_remains_usable_without_adapter(self) -> None:
        retrieved = self._retrieval_item()
        response = StaticRetrievalProvider([retrieved]).retrieve("Exact retrieved text")

        self.assertEqual(response.evidence[0], retrieved)
        self.assertEqual(response.evidence[0].text, retrieved.text)
        self.assertEqual(self.state.document_versions, [])

    def test_synthetic_existing_rag_provider_to_provenance_flow(self) -> None:
        class SyntheticRag:
            @staticmethod
            def retrieve_chunks_advanced(**kwargs):
                return [
                    {
                        "text": "Synthetic RAG passage",
                        "filename": "synthetic.pdf",
                        "page": 6,
                        "section": "Results",
                        "document_id": "synthetic-doc",
                        "chunk_id": "synthetic-chunk-6",
                        "evidence_score": 0.6,
                        "rerank_score": 0.4,
                    }
                ]

        provider = ExistingRAGProvider(
            chunks=[],
            embedding_model=None,
            reranker=None,
            faiss_index=None,
            bm25_index=None,
            retrieval_module=SyntheticRag,
        )
        retrieved = provider.retrieve("synthetic query").evidence[0]
        version = register_rag_document_version(
            self.state,
            source_id=retrieved.source_id,
            document_id=retrieved.document_id,
            filename=retrieved.filename,
        )

        evidence = adapt_retrieval_evidence_to_state(retrieved, version, self.state)

        reference = self.state.passage_references[0]
        self.assertEqual(evidence.text, "Synthetic RAG passage")
        self.assertEqual(evidence.source_id, retrieved.source_id)
        self.assertEqual(evidence.document_version_id, version.version_id)
        self.assertEqual(reference.page, 6)
        self.assertEqual(reference.chunk_id, "synthetic-chunk-6")
        self.assertEqual(reference.section, "Results")
        self.assertEqual(self.state.validate_provenance(), [])

    def test_state_add_methods_enforce_source_and_version_relationships(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown source_id"):
            self.state.add_document_version(
                DocumentVersion(document_id="orphan-doc", source_id="missing-source")
            )

        version = DocumentVersion(document_id="paper-1", source_id=self.source.source_id)
        with self.assertRaisesRegex(ValueError, "unknown document_version_id"):
            self.state.add_passage_reference(
                PassageReference(
                    document_version_id=version.version_id,
                    locator_type="page",
                    page=1,
                )
            )

    def test_evidence_links_source_version_and_passage_and_hashes_text(self) -> None:
        version = self.state.add_document_version(
            DocumentVersion(document_id="paper-1", source_id=self.source.source_id)
        )
        reference = self.state.add_passage_reference(
            PassageReference(
                document_version_id=version.version_id,
                locator_type="page",
                page=4,
                section="Results",
            )
        )
        evidence = self.state.add_evidence(
            Evidence(
                evidence_id="evidence-1",
                text="  The result was significant.\n",
                source_id=self.source.source_id,
                relevance_score=0.91,
                page=4,
                chunk_id="chunk-4",
                metadata={"retrieval_method": "hybrid"},
                document_version_id=version.version_id,
                passage_reference_id=reference.passage_reference_id,
            )
        )

        self.assertEqual(evidence.source_id, self.source.source_id)
        self.assertEqual(evidence.document_version_id, version.version_id)
        self.assertEqual(evidence.passage_reference_id, reference.passage_reference_id)
        self.assertEqual(evidence.page, 4)
        self.assertEqual(evidence.chunk_id, "chunk-4")
        self.assertEqual(evidence.metadata["retrieval_method"], "hybrid")
        self.assertEqual(len(evidence.text_hash or ""), 64)
        self.assertEqual(self.state.validate_provenance(), [])

    def test_evidence_rejects_dangling_or_inconsistent_provenance(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown document_version_id"):
            self.state.add_evidence(
                Evidence(
                    text="Evidence text",
                    source_id=self.source.source_id,
                    document_version_id="missing-version",
                    passage_reference_id="missing-passage",
                )
            )

        version = self.state.add_document_version(
            DocumentVersion(document_id="paper-1", source_id=self.source.source_id)
        )
        reference = self.state.add_passage_reference(
            PassageReference(document_version_id=version.version_id, locator_type="page", page=1)
        )
        with self.assertRaisesRegex(ValueError, "source_id does not match"):
            self.state.add_evidence(
                Evidence(
                    text="Evidence text",
                    source_id="different-source",
                    document_version_id=version.version_id,
                    passage_reference_id=reference.passage_reference_id,
                )
            )
        with self.assertRaisesRegex(ValueError, "both document_version_id"):
            self.state.add_evidence(
                Evidence(
                    text="Partial provenance",
                    source_id=self.source.source_id,
                    document_version_id=version.version_id,
                )
            )

    def test_provenance_hash_is_deterministic_over_normalized_text(self) -> None:
        common = {
            "source_id": self.source.source_id,
            "document_version_id": "version-test",
            "passage_reference_id": "passage-test",
        }
        first = Evidence(text="Cafe\u0301   is useful.", **common)
        second = Evidence(text="Café is useful.", **common)

        self.assertEqual(first.text_hash, second.text_hash)

    def test_complete_chain_round_trips_and_legacy_incomplete_evidence_loads(self) -> None:
        action = self.state.add_search(SearchAction(query="paper finding"))
        result = self.state.add_search_result(
            SearchResult(
                source_id=self.source.source_id,
                title=self.source.title,
                result_id="result-chain",
                search_id=action.search_id,
                rank=1,
            )
        )
        version = self.state.add_document_version(
            DocumentVersion(document_id="paper-1", source_id=result.source_id)
        )
        reference = self.state.add_passage_reference(
            PassageReference(
                document_version_id=version.version_id,
                locator_type="sentence",
                sentence_index=8,
                section="Abstract",
            )
        )
        evidence = self.state.add_evidence(
            Evidence(
                text="The paper reports the finding.",
                source_id=result.source_id,
                document_version_id=version.version_id,
                passage_reference_id=reference.passage_reference_id,
            )
        )

        restored = ResearchState.from_dict(self.state.to_dict())

        self.assertEqual(restored.searches[0].search_id, action.search_id)
        self.assertEqual(restored.search_results[0].search_id, action.search_id)
        self.assertEqual(restored.search_results[0].result_id, "result-chain")
        self.assertEqual(restored.search_results[0].source_id, self.source.source_id)
        self.assertEqual(restored.document_versions[0].version_id, version.version_id)
        self.assertEqual(
            restored.passage_references[0].passage_reference_id,
            reference.passage_reference_id,
        )
        self.assertEqual(restored.evidence[0].evidence_id, evidence.evidence_id)
        self.assertEqual(restored.validate_provenance(), [])

        legacy = ResearchState(user_request="Old state")
        legacy.add_evidence(Evidence(text="Old unlinked evidence", source_id="old-source"))
        old_payload = legacy.to_dict()
        old_payload.pop("document_versions")
        old_payload.pop("passage_references")
        loaded = ResearchState.from_dict(old_payload)
        self.assertEqual(loaded.document_versions, [])
        self.assertEqual(loaded.passage_references, [])
        self.assertEqual(loaded.evidence[0].document_version_id, None)
        self.assertEqual(loaded.validate_provenance(), [])

    def test_invalid_serialized_provenance_is_reported(self) -> None:
        payload = ResearchState(user_request="Malformed state").to_dict()
        payload["document_versions"] = [
            {
                "document_id": "doc-1",
                "source_id": "missing-source",
                "version_id": "version-bad",
                "captured_at": "2026-01-01T00:00:00+00:00",
                "metadata": {},
            }
        ]

        with self.assertRaisesRegex(ValueError, "Invalid serialized provenance"):
            ResearchState.from_dict(payload)


if __name__ == "__main__":
    unittest.main()
