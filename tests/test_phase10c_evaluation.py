"""Focused protocol and persistence tests for the Phase 10C dev runner."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation.experiments.evaluator import SystemOutput
from evaluation.experiments import run_phase10_dev as phase10c
from evaluation.scifact.scifact_adapter import load_scifact_dataset

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "scifact" / "data"


class Phase10CEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = load_scifact_dataset(DATA, "dev")

    def test_dev_split_contains_300_claims(self):
        self.assertEqual(len(self.dataset.claims), 300)

    def test_dev_scorable_rationale_claim_count_is_188(self):
        scorable = [r for r in self.dataset.claims if phase10c.gold_claim_label(r.annotations)[0]]
        self.assertEqual(len(scorable), 188)

    def test_selection_is_deterministic(self):
        first = phase10c.select_stratified_dev_ids(self.dataset, 12)
        second = phase10c.select_stratified_dev_ids(self.dataset, 12)
        self.assertEqual(first, second)

    def test_selection_preserves_dev_stratum_proportions_and_unlabeled_cases(self):
        selected = phase10c.select_stratified_dev_ids(self.dataset, 12)
        chosen = set(selected)
        labels = [phase10c.gold_claim_label(row.annotations)[0]
                  for row in self.dataset.claims if row.runtime.claim_id in chosen]
        self.assertEqual(labels.count("SUPPORTED"), 5)
        self.assertEqual(labels.count("CONTRADICTED"), 3)
        self.assertEqual(labels.count(None), 4)

    def test_selection_preserves_dev_file_order_within_strata(self):
        selected = phase10c.select_stratified_dev_ids(self.dataset, 6)
        self.assertEqual(selected[0] in {r.runtime.claim_id for r in self.dataset.claims}, True)
        self.assertEqual(len(selected), len(set(selected)))

    def test_selection_rejects_count_larger_than_dev_split(self):
        with self.assertRaisesRegex(ValueError, "exceeds the dev split"):
            phase10c.select_stratified_dev_ids(self.dataset, 301)

    def test_runner_loads_dev_only(self):
        original = phase10c.load_scifact_dataset
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(phase10c, "load_scifact_dataset", wraps=original) as loader, \
             patch.object(phase10c, "_ollama_model_details", return_value={"tag": "fixture"}), \
             patch.object(phase10c, "SYSTEMS", self.callbacks()):
            phase10c.run_phase10_dev(dataset_root=DATA, output_root=folder, count=2,
                                     provider=object())
        self.assertEqual([call.args[1] for call in loader.call_args_list], ["dev"])

    def callbacks(self, *, fail_system=None, labels=None, inspect=None):
        labels = labels or {}
        result = {}
        for system in "ABC":
            def callback(runtime, provider, config, system=system):
                if inspect:
                    inspect(system, runtime, provider, config)
                if system == fail_system:
                    raise RuntimeError("fixture failure")
                label = labels.get(system)
                return SystemOutput(predicted_label=label,
                    retrieved_chunks=[{"rank": 1, "scifact_doc_id": 100,
                                       "evidence_id": "E1"}],
                    selected_evidence_doc_ids=[100] if system != "A" else [],
                    raw_output={"system": system}, llm_calls=1, retrieval_calls=1,
                    verification_calls=int(system != "A"), controller_cycles=int(system == "C"))
            result[system] = callback
        return result

    def _run_fake(self, folder, *, callbacks=None, count=2, provider=None):
        with patch.object(phase10c, "_ollama_model_details", return_value={"tag": "fixture"}), \
             patch.object(phase10c, "SYSTEMS", callbacks or self.callbacks()):
            return phase10c.run_phase10_dev(dataset_root=DATA, output_root=folder, count=count,
                                             provider=provider or object())

    def test_a_b_c_receive_identical_runtime_payload_and_shared_provider(self):
        seen = []
        provider = object()
        callbacks = self.callbacks(inspect=lambda system, runtime, received, config:
            seen.append((system, runtime.copy(), received, config)))
        with tempfile.TemporaryDirectory() as folder:
            self._run_fake(folder, callbacks=callbacks, provider=provider)
        for claim_id in (seen[0][1]["claim_id"], seen[3][1]["claim_id"]):
            claim_rows = [entry for entry in seen if entry[1]["claim_id"] == claim_id]
            self.assertEqual([entry[0] for entry in claim_rows], list("ABC"))
            self.assertEqual({frozenset(entry[1].items()) for entry in claim_rows},
                             {frozenset(claim_rows[0][1].items())})
            self.assertTrue(all(entry[2] is provider for entry in claim_rows))

    def test_runtime_callback_never_receives_gold_fields(self):
        seen = []
        with tempfile.TemporaryDirectory() as folder:
            self._run_fake(folder, callbacks=self.callbacks(inspect=lambda s, r, p, c: seen.append(r.copy())))
        self.assertTrue(seen)
        self.assertTrue(all(set(row) == {"claim_id", "claim"} for row in seen))

    def test_gold_join_happens_after_all_three_callbacks(self):
        observations = []
        def inspect(system, runtime, provider, config):
            path = Path(active_folder) / "phase10_dev_run" / "per_example.jsonl"
            rows = ([json.loads(line) for line in path.read_text().splitlines()]
                    if path.exists() else [])
            observations.append((system, runtime["claim_id"],
                                 any(row["example_id"] == runtime["claim_id"] for row in rows)))
        with tempfile.TemporaryDirectory() as folder:
            active_folder = folder
            self._run_fake(folder, callbacks=self.callbacks(inspect=inspect))
        self.assertEqual([row[0] for row in observations[:3]], list("ABC"))
        self.assertTrue(all(not joined for _, _, joined in observations))

    def test_gold_free_attempt_log_persists_each_execution(self):
        with tempfile.TemporaryDirectory() as folder:
            self._run_fake(folder)
            attempts = [json.loads(line) for line in
                (Path(folder) / "phase10_dev_run" / "runtime_attempts.jsonl").read_text().splitlines()]
        self.assertEqual(len(attempts), 6)
        self.assertTrue(all("gold_label" not in row and "gold_evidence_doc_ids" not in row for row in attempts))

    def test_scored_rows_join_gold_after_execution(self):
        with tempfile.TemporaryDirectory() as folder:
            self._run_fake(folder)
            rows = [json.loads(line) for line in
                (Path(folder) / "phase10_dev_run" / "per_example.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 6)
        self.assertTrue(all("gold_label" in row and row["split"] == "dev" for row in rows))

    def test_failure_remains_failed_and_is_counted(self):
        with tempfile.TemporaryDirectory() as folder:
            self._run_fake(folder, callbacks=self.callbacks(fail_system="B"))
            rows = [json.loads(line) for line in
                (Path(folder) / "phase10_dev_run" / "per_example.jsonl").read_text().splitlines()]
            attempts = [json.loads(line) for line in
                (Path(folder) / "phase10_dev_run" / "runtime_attempts.jsonl").read_text().splitlines()]
        self.assertEqual(sum(row["status"] == "FAILED" for row in rows), 2)
        self.assertEqual(sum(row["status"] == "FAILED" for row in attempts), 2)

    def test_invalid_prediction_is_not_scored_as_correct(self):
        rows = [{"gold_label": "SUPPORTED", "predicted_label": "BOGUS"}]
        metrics = phase10c._classification(rows)
        self.assertEqual(metrics["accuracy"], 0)
        self.assertEqual(metrics["denominator"], 1)
        self.assertEqual(metrics["invalid_or_failed_predictions"], 1)

    def test_failed_prediction_stays_in_primary_accuracy_denominator(self):
        rows = [{"gold_label": "CONTRADICTED", "predicted_label": None}]
        metrics = phase10c._classification(rows)
        self.assertEqual(metrics["accuracy"], 0)
        self.assertEqual(metrics["denominator"], 1)

    def test_unlabeled_example_is_excluded_from_classification_denominator(self):
        metrics = phase10c._classification([{"gold_label": None, "predicted_label": "SUPPORTED"}])
        self.assertIsNone(metrics["accuracy"])
        self.assertEqual(metrics["denominator"], 0)
        self.assertEqual(metrics["unscorable_gold_examples"], 1)

    def test_zero_f1_is_reported_as_zero_when_class_support_exists(self):
        metrics = phase10c._classification([
            {"gold_label": "SUPPORTED", "predicted_label": "CONTRADICTED"},
            {"gold_label": "CONTRADICTED", "predicted_label": "SUPPORTED"}])
        self.assertEqual(metrics["macro_f1_over_gold_supported_classes"], 0)

    def test_classification_confusion_matrix_preserves_invalid_bucket(self):
        metrics = phase10c._classification([
            {"gold_label": "SUPPORTED", "predicted_label": "INVALID"}])
        self.assertEqual(metrics["confusion_matrix"]["SUPPORTED"]["INVALID_OR_FAILED"], 1)

    def test_system_a_classification_is_not_applicable(self):
        with tempfile.TemporaryDirectory() as folder:
            summary = self._run_fake(folder)
        self.assertEqual(summary["systems"]["A"]["classification"], "NOT APPLICABLE")

    def test_system_a_selected_evidence_is_not_applicable(self):
        with tempfile.TemporaryDirectory() as folder:
            summary = self._run_fake(folder)
        self.assertEqual(summary["systems"]["A"]["selected_evidence_document_metrics"]["status"],
                         "NOT APPLICABLE")

    def test_system_b_and_c_selection_metrics_have_explicit_denominators(self):
        with tempfile.TemporaryDirectory() as folder:
            summary = self._run_fake(folder)
        for system in "BC":
            self.assertIn("claim_denominator", summary["systems"][system]["selected_evidence_document_metrics"])

    def test_retrieval_denominator_excludes_empty_annotations(self):
        with tempfile.TemporaryDirectory() as folder:
            summary = self._run_fake(folder, count=4)
        self.assertEqual(summary["systems"]["A"]["attempted"], 4)
        self.assertEqual(summary["systems"]["A"]["retrieval"]["scorable_claim_denominator"], 3)

    def test_aggregate_can_be_recomputed_from_stored_per_example_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            report = self._run_fake(folder)
            rows = [json.loads(line) for line in
                (Path(folder) / "phase10_dev_run" / "per_example.jsonl").read_text().splitlines()]
            attempts = [json.loads(line) for line in
                (Path(folder) / "phase10_dev_run" / "runtime_attempts.jsonl").read_text().splitlines()]
        self.assertEqual(phase10c.aggregate_stored_records(attempts, rows), report["systems"])

    def test_runner_refuses_to_overwrite_prior_phase10c_results(self):
        with tempfile.TemporaryDirectory() as folder:
            self._run_fake(folder)
            with self.assertRaises(FileExistsError):
                self._run_fake(folder)

    def test_resume_does_not_repeat_persisted_system_claim_pairs(self):
        calls = []
        callbacks = self.callbacks(inspect=lambda s, r, p, c: calls.append((r["claim_id"], s)))
        with tempfile.TemporaryDirectory() as folder:
            self._run_fake(folder, callbacks=callbacks)
            calls.clear()
            with patch.object(phase10c, "_ollama_model_details", return_value={"tag": "fixture"}), \
                 patch.object(phase10c, "SYSTEMS", callbacks):
                phase10c.run_phase10_dev(dataset_root=DATA, output_root=folder, count=2,
                                         provider=object(), resume=True)
        self.assertEqual(calls, [])

    def test_result_manifest_marks_dev_and_no_test_usage(self):
        with tempfile.TemporaryDirectory() as folder:
            self._run_fake(folder)
            manifest = json.loads((Path(folder) / "phase10_dev_run" / "run_manifest.json").read_text())
        self.assertEqual(manifest["dataset"]["split"], "dev")
        self.assertEqual(len(manifest["selection"]["claim_ids"]), 2)
        self.assertTrue(manifest["gold_leakage"]["gold_annotations_passed_to_execution"] is False)
        self.assertFalse(manifest["failure_handling"]["automatic_retries"])
        self.assertEqual(manifest["status"], "PARTIAL")
        self.assertEqual(manifest["execution_status"], "COMPLETED")
        self.assertEqual(manifest["dataset"]["gold_label_counts"]["SUPPORTED"], 124)

    def test_system_c_identity_is_documented_as_real_controller(self):
        self.assertIs(phase10c.SYSTEMS["C"], phase10c.run_system_c)
        with tempfile.TemporaryDirectory() as folder:
            self._run_fake(folder)
            manifest = json.loads((Path(folder) / "phase10_dev_run" / "run_manifest.json").read_text())
        self.assertIn("real 8C controller", manifest["systems"]["C"])

    def test_controller_and_verification_efficiency_fields_persist(self):
        with tempfile.TemporaryDirectory() as folder:
            self._run_fake(folder)
            rows = [json.loads(line) for line in
                (Path(folder) / "phase10_dev_run" / "per_example.jsonl").read_text().splitlines()]
        c_rows = [r for r in rows if r["system"] == "C"]
        self.assertTrue(all(r["controller_cycles"] == 1 and r["verification_calls"] == 1 for r in c_rows))

    def test_dataset_ids_selected_are_all_development_claims(self):
        ids = set(phase10c.select_stratified_dev_ids(self.dataset, 20))
        self.assertLessEqual(ids, {r.runtime.claim_id for r in self.dataset.claims})

    def test_paired_uncertainty_is_not_computed_for_small_n(self):
        rows = [{"example_id": 1, "system": "B", "gold_label": "SUPPORTED",
                 "predicted_label": "SUPPORTED"},
                {"example_id": 1, "system": "C", "gold_label": "SUPPORTED",
                 "predicted_label": "CONTRADICTED"}]
        self.assertEqual(phase10c.paired_accuracy_bootstrap(rows)["status"], "NOT COMPUTED")

    def test_paired_bootstrap_is_reproducible_and_uses_paired_denominator(self):
        rows = []
        for claim_id in range(12):
            label = "SUPPORTED" if claim_id % 2 else "CONTRADICTED"
            rows.extend([{"example_id": claim_id, "system": "B", "gold_label": label,
                          "predicted_label": label},
                         {"example_id": claim_id, "system": "C", "gold_label": label,
                          "predicted_label": "CONTRADICTED" if label == "SUPPORTED" else label}])
        first = phase10c.paired_accuracy_bootstrap(rows, seed=17, replicates=1000)
        second = phase10c.paired_accuracy_bootstrap(rows, seed=17, replicates=1000)
        self.assertEqual(first, second)
        self.assertEqual(first["paired_denominator"], 12)
        self.assertIn("percentile_95_ci", first)

    def test_resume_rejects_a_different_selection_or_config(self):
        with tempfile.TemporaryDirectory() as folder:
            self._run_fake(folder, count=2)
            with patch.object(phase10c, "_ollama_model_details", return_value={"tag": "fixture"}), \
                 patch.object(phase10c, "SYSTEMS", self.callbacks()):
                with self.assertRaisesRegex(ValueError, "does not match"):
                    phase10c.run_phase10_dev(dataset_root=DATA, output_root=folder,
                                             count=4, provider=object(), resume=True)


if __name__ == "__main__":
    unittest.main()
