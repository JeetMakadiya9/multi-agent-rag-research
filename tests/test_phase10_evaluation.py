"""Focused protocol and leakage tests for Phase 10."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.experiments.evaluator import SystemOutput, gold_claim_label, run_experiment
from evaluation.experiments.experiment_config import ExperimentConfig
from evaluation.scifact.scifact_adapter import ClaimAnnotations


class Phase10EvaluationTests(unittest.TestCase):
    def test_gold_label_mapping_requires_unanimous_rationale_stance(self):
        support = ClaimAnnotations(True, True, (), ())
        self.assertEqual(gold_claim_label(support), (None, "no_rationale_label; NEI mapping not asserted"))
        from evaluation.scifact.scifact_adapter import GoldEvidenceDocument, GoldRationale
        annotations = ClaimAnnotations(
            True, True,
            (GoldEvidenceDocument(10, (GoldRationale("SUPPORT", (0,)),)),),
            (),
        )
        self.assertEqual(gold_claim_label(annotations)[0], "SUPPORTED")
        conflict = ClaimAnnotations(
            True, True,
            (GoldEvidenceDocument(10, (GoldRationale("SUPPORT", (0,)),
                                      GoldRationale("CONTRADICT", (1,)))),),
            (),
        )
        self.assertIsNone(gold_claim_label(conflict)[0])

    def test_same_runtime_input_is_passed_without_gold_to_all_callbacks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "corpus.jsonl").write_text(json.dumps({
                "doc_id": 10, "title": "Study", "abstract": ["Evidence sentence."],
                "structured": False,
            }) + "\n", encoding="utf-8")
            (root / "claims_dev.jsonl").write_text(json.dumps({
                "id": 1, "claim": "The evidence sentence supports this.",
                "evidence": {"10": [{"label": "SUPPORT", "sentences": [0]}]},
                "cited_doc_ids": [10],
            }) + "\n", encoding="utf-8")
            seen = []

            def system(name):
                def call(runtime, provider, config):
                    self.assertEqual(set(runtime), {"claim_id", "claim"})
                    self.assertNotIn("cited_doc_ids", runtime)
                    self.assertNotIn("evidence", runtime)
                    seen.append((name, dict(runtime)))
                    return SystemOutput(predicted_label="SUPPORTED",
                                        retrieved_chunks=[{"scifact_doc_id": 10}],
                                        selected_evidence_doc_ids=[10], retrieval_calls=1)
                return call

            config = ExperimentConfig(str(root), example_limit=1)
            result = run_experiment(config, object(),
                                    {name: system(name) for name in ("A", "B", "C")},
                                    root / "run")
            self.assertEqual(len(seen), 3)
            self.assertEqual(seen[0][1], seen[1][1])
            self.assertEqual(seen[1][1], seen[2][1])
            self.assertEqual(result["records"], 1)
            self.assertEqual(result["summary"]["A"]["accuracy"], 1.0)
            self.assertEqual(result["summary"]["A"]["gold_document_recall_at_k"]["1"], 1.0)
            self.assertTrue((root / "run" / "run_manifest.json").is_file())
            self.assertTrue((root / "run" / "per_example.jsonl").is_file())
            self.assertTrue((root / "run" / "summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
