from __future__ import annotations

import json
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.scifact.scifact_adapter import (  # noqa: E402
    corpus_to_rag_chunks,
    load_scifact_dataset,
    load_claims,
    load_corpus,
    runtime_claim_payload,
)
from evaluation.scifact.metrics import (  # noqa: E402
    build_run_manifest,
    retrieval_metrics_for_claim,
    verification_metrics_for_claim,
)
from evaluation.scifact.run_scifact_evaluation import _retrieve_runtime_claim  # noqa: E402
from evaluation.scifact.run_scifact_evaluation import _runtime_evidence  # noqa: E402


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class ScifactInfrastructureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.corpus_rows = [
            {"doc_id": 10, "title": "Study Alpha", "abstract": ["First source sentence, with details.", "Second sentence follows."], "structured": False},
            {"doc_id": 20, "title": "Study Beta", "abstract": ["A contradictory finding appears here."], "structured": True},
        ]
        write_jsonl(self.root / "corpus.jsonl", self.corpus_rows)
        write_jsonl(self.root / "claims_dev.jsonl", [
            {
                "id": 7,
                "claim": "The intervention improves the outcome.",
                "evidence": {
                    "10": [{"label": "SUPPORT", "sentences": [0]}],
                    "20": [{"label": "CONTRADICT", "sentences": [0]}],
                },
                "cited_doc_ids": [10, 20, 999],
            },
        ])
        write_jsonl(self.root / "claims_test.jsonl", [
            {"id": 99, "claim": "An unlabeled test claim."},
        ])

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_corpus_and_claim_parsing_preserve_ids_sentences_and_annotations(self) -> None:
        corpus = load_corpus(self.root / "corpus.jsonl")
        claims = load_claims(self.root / "claims_dev.jsonl", corpus)
        self.assertEqual(corpus[0].doc_id, 10)
        self.assertEqual(corpus[0].abstract[0], "First source sentence, with details.")
        self.assertEqual(claims[0].runtime.claim_id, 7)
        self.assertEqual(claims[0].runtime.text, "The intervention improves the outcome.")
        gold = claims[0].annotations.evidence
        self.assertEqual([(item.doc_id, item.rationales[0].label, item.rationales[0].sentence_indices) for item in gold],
                         [(10, "SUPPORT", (0,)), (20, "CONTRADICT", (0,))])
        self.assertEqual(claims[0].annotations.cited_doc_ids, (10, 20, 999))

    def test_test_split_missing_evidence_is_unavailable_not_empty_gold(self) -> None:
        data = load_scifact_dataset(self.root, "test")
        annotations = data.claims[0].annotations
        self.assertFalse(annotations.labels_available)
        self.assertIsNone(annotations.evidence)

    def test_missing_required_dataset_files_fail_with_explicit_paths(self) -> None:
        from evaluation.scifact.scifact_adapter import load_scifact_dataset

        empty_root = self.root / "missing"
        empty_root.mkdir()
        with self.assertRaisesRegex(FileNotFoundError, "corpus.jsonl.*claims_dev.jsonl"):
            load_scifact_dataset(empty_root, "dev")

    def test_explicit_empty_evidence_remains_distinct_from_unavailable(self) -> None:
        path = self.root / "empty.jsonl"
        write_jsonl(path, [{"id": 8, "claim": "Claim with an empty annotation.", "evidence": {}, "cited_doc_ids": []}])
        record = load_claims(path, load_corpus(self.root / "corpus.jsonl"))[0]
        self.assertTrue(record.annotations.labels_available)
        self.assertEqual(record.annotations.evidence, ())
        measured = retrieval_metrics_for_claim(record.annotations, [], [1])
        self.assertEqual(measured["retrieval_scoring_status"], "no_annotated_gold_evidence")
        self.assertIsNone(measured["recall_at_k"]["1"])

    def test_chunk_adapter_preserves_document_and_zero_based_sentence_mapping(self) -> None:
        class FakeRag:
            @staticmethod
            def create_chunks(pages, filename):
                return [
                    {"filename": filename, "page": page["page"], "chunk_id": index, "text": page["text"]}
                    for index, page in enumerate(pages)
                ]

        chunks = corpus_to_rag_chunks(load_corpus(self.root / "corpus.jsonl"), FakeRag)
        self.assertEqual(chunks[0]["scifact_doc_id"], 10)
        self.assertEqual(chunks[0]["scifact_sentence_indices"], [0])
        self.assertEqual(chunks[0]["scifact_sentence_texts"], [self.corpus_rows[0]["abstract"][0]])
        self.assertEqual(chunks[2]["scifact_doc_id"], 20)
        self.assertEqual(chunks[2]["scifact_sentence_indices"], [0])
        self.assertEqual(chunks[0]["document_id"], "10")
        self.assertIsInstance(chunks[0]["chunk_id"], int)
        self.assertEqual(chunks[2]["chunk_id"], 2)
        self.assertFalse(chunks[0]["scifact_structured"])
        self.assertIn("Study Alpha", chunks[0]["text"])
        self.assertFalse(any("gold" in key.lower() or "cited_doc" in key.lower() for chunk in chunks for key in chunk))

    def test_runtime_claim_and_retrieval_path_cannot_receive_cited_doc_ids(self) -> None:
        claim = load_scifact_dataset(self.root, "dev").claims[0]
        runtime = runtime_claim_payload(claim.runtime)
        self.assertEqual(set(runtime), {"claim_id", "claim"})

        class RecordingProvider:
            query = None
            limit = None

            def retrieve(self, query, limit):
                self.query, self.limit = query, limit
                return "retrieved"

        provider = RecordingProvider()
        self.assertEqual(_retrieve_runtime_claim(provider, runtime, 20), "retrieved")
        self.assertEqual(provider.query, claim.runtime.text)
        self.assertEqual(provider.limit, 20)
        self.assertNotIn("cited_doc_ids", runtime)

    def test_sentence_mapping_is_not_exposed_on_verifier_evidence_items(self) -> None:
        retrieved = SimpleNamespace(
            evidence_id="scifact-10-1",
            chunk_id="scifact-10-1",
            document_id="10",
            text="Title: Study Alpha. Abstract sentence: First source sentence, with details.",
            retrieval_score=0.8,
            score=0.8,
            retrieval_metadata={
                "chunk": {
                    "scifact_doc_id": 10,
                    "scifact_sentence_indices": [0],
                    "scifact_sentence_texts": ["First source sentence, with details."],
                },
            },
        )
        scoring_refs, verifier_items = _runtime_evidence([retrieved])
        self.assertEqual(scoring_refs[0]["scifact_sentence_indices"], [0])
        self.assertEqual(verifier_items[0].metadata, {})
        self.assertIsNone(verifier_items[0].page)
        self.assertNotIn("scifact_sentence_indices", verifier_items[0].__dict__)
        self.assertNotIn("cited_doc_ids", verifier_items[0].__dict__)

    def test_recall_at_k_deduplicates_chunks_and_handles_short_retrieval(self) -> None:
        annotations = load_scifact_dataset(self.root, "dev").claims[0].annotations
        retrieved = [
            {"scifact_doc_id": 10},
            {"scifact_doc_id": 10},
            {"scifact_doc_id": 77},
            {"scifact_doc_id": 20},
        ]
        result = retrieval_metrics_for_claim(annotations, retrieved, [1, 2, 5, 20])
        self.assertEqual(result["recall_at_k"], {"1": 0.5, "2": 0.5, "5": 1.0, "20": 1.0})
        self.assertEqual(result["first_gold_rank"], 1)
        self.assertEqual(result["reciprocal_rank"], 1.0)

    def test_missing_gold_evidence_is_not_scored_as_zero_recall(self) -> None:
        annotations = load_scifact_dataset(self.root, "dev").claims[0].annotations
        result = retrieval_metrics_for_claim(annotations, [{"scifact_doc_id": 77}], [1, 5])
        self.assertEqual(result["recall_at_k"], {"1": 0.0, "5": 0.0})
        self.assertIsNone(result["first_gold_rank"])
        self.assertEqual(result["reciprocal_rank"], 0.0)

    def test_stance_alignment_requires_and_checks_retrieved_gold_sentence(self) -> None:
        path = self.root / "single-gold.jsonl"
        write_jsonl(path, [{
            "id": 70,
            "claim": "The intervention improves the outcome.",
            "evidence": {"10": [{"label": "SUPPORT", "sentences": [0]}]},
        }])
        annotations = load_claims(path, load_corpus(self.root / "corpus.jsonl"))[0].annotations
        retrieved = [{
            "scifact_doc_id": 10,
            "scifact_sentence_indices": [0],
            "scifact_sentence_texts": ["First source sentence, with details."],
            "text": "Title: Study Alpha. Abstract sentence: First source sentence, with details.",
        }]
        report = {"verifications": [{
            "verdict": "SUPPORTED",
            "assessment_complete": True,
            "missing_assessments": [],
            "evidence_assessments": [{"evidence_index": "E1", "component_indices": ["C1"], "stance": "SUPPORTS"}],
            "coverage_matrix": [{"component_index": "C1", "status": "SUPPORTED"}],
        }], "trace": []}
        result = verification_metrics_for_claim(annotations, retrieved, report)
        self.assertEqual(result["evidence_stance_alignment_status"], "aligned")
        self.assertEqual(result["gold_rationale_sentence_hits_in_verification_context"], 1)

    def test_missing_assessment_is_not_aligned_and_is_counted(self) -> None:
        path = self.root / "single-gold.jsonl"
        write_jsonl(path, [{
            "id": 71,
            "claim": "The intervention improves the outcome.",
            "evidence": {"10": [{"label": "SUPPORT", "sentences": [0]}]},
        }])
        annotations = load_claims(path, load_corpus(self.root / "corpus.jsonl"))[0].annotations
        retrieved = [{
            "scifact_doc_id": 10,
            "scifact_sentence_indices": [0],
            "scifact_sentence_texts": ["First source sentence, with details."],
            "text": "Title: Study Alpha. Abstract sentence: First source sentence, with details.",
        }]
        report = {"verifications": [{
            "verdict": "INSUFFICIENT_EVIDENCE",
            "assessment_complete": False,
            "missing_assessments": ["E1-C1"],
            "evidence_assessments": [],
            "coverage_matrix": [{"component_index": "C1", "status": "INCOMPLETE"}],
        }], "trace": []}
        result = verification_metrics_for_claim(annotations, retrieved, report)
        self.assertEqual(result["evidence_stance_alignment_status"], "missing_assessment")
        self.assertEqual(result["missing_assessments"], ["E1-C1"])
        self.assertTrue(result["unable_to_establish_sufficient_evidence"])

    def test_conflicting_retrieved_gold_labels_are_explicitly_ambiguous(self) -> None:
        claim = load_scifact_dataset(self.root, "dev").claims[0]
        retrieved = [
            {"scifact_doc_id": 10, "scifact_sentence_indices": [0]},
            {"scifact_doc_id": 20, "scifact_sentence_indices": [0]},
        ]
        report = {
            "verifications": [{
                "verdict": "INSUFFICIENT_EVIDENCE",
                "assessment_complete": True,
                "missing_assessments": [],
                "evidence_assessments": [
                    {"evidence_index": "E1", "component_indices": ["C1"], "stance": "SUPPORTS"},
                    {"evidence_index": "E2", "component_indices": ["C1"], "stance": "CONTRADICTS"},
                ],
                "coverage_matrix": [{"component_index": "C1", "status": "CONFLICTING"}],
            }],
            "trace": [],
        }
        result = verification_metrics_for_claim(claim.annotations, retrieved, report)
        self.assertTrue(result["retrieved_gold_annotations_conflict"])
        self.assertTrue(result["verification_ambiguous"])
        self.assertEqual(result["gold_labels_on_retrieved_evidence"], ["CONTRADICT", "SUPPORT"])
        self.assertTrue(result["unable_to_establish_sufficient_evidence"])

    def test_manifest_records_hashes_config_and_prompt_sources(self) -> None:
        project = self.root / "project"
        project.mkdir()
        (project / "rag.py").write_text("# frozen fixture\n", encoding="utf-8")
        corpus_path = self.root / "corpus.jsonl"
        claims_path = self.root / "claims_dev.jsonl"
        manifest = build_run_manifest(
            dataset_root=self.root,
            split="dev",
            corpus_path=corpus_path,
            claims_path=claims_path,
            output_dir=self.root / "out",
            project_root=project,
            rag_module=SimpleNamespace(
                EMBEDDING_MODEL_NAME="embedding-fixture",
                RERANKER_MODEL_NAME="reranker-fixture",
                FINAL_TOP_K=8,
            ),
            verification_config={"model": "fixture-model", "temperature": 0},
            top_k=[1, 5],
            run_experiment_a=False,
            run_experiment_b=True,
        )
        self.assertEqual(manifest["dataset"]["split"], "dev")
        self.assertEqual(len(manifest["dataset"]["files"]["corpus"]["sha256"]), 64)
        expected_rag_hash = hashlib.sha256((project / "rag.py").read_bytes()).hexdigest().upper()
        self.assertEqual(manifest["frozen_components"]["rag_py_sha256"], expected_rag_hash)
        self.assertIsNone(manifest["project"]["git_commit"])
        self.assertFalse(manifest["evaluation"]["gold_annotation_data_passed_to_runtime"])
        self.assertFalse(manifest["evaluation"]["cited_doc_ids_used_for_retrieval"])
        self.assertIsNone(manifest["evaluation"]["three_class_gold_mapping"])


if __name__ == "__main__":
    unittest.main()
