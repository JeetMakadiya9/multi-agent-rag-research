"""Offline, deterministic checks for Phase 10D artifact analysis."""
import unittest
from pathlib import Path

from evaluation.experiments import phase10d_analysis as analysis


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "evaluation" / "experiments" / "results"
RUN = BASE / "phase10_dev_run"
SOURCES = {
    "combined": analysis.read_json(BASE / "phase10_dev.json"),
    "rows": analysis.read_jsonl(RUN / "per_example.jsonl"),
    "attempts": analysis.read_jsonl(RUN / "runtime_attempts.jsonl"),
    "manifest": analysis.read_json(RUN / "run_manifest.json"),
    "summary": analysis.read_json(RUN / "summary.json"),
}


class Phase10DAnalysisTests(unittest.TestCase):
    def test_required_artifacts_load_and_identify_dev_scope(self):
        self.assertEqual(SOURCES["combined"]["dataset"]["split"], "dev")
        self.assertEqual(len(SOURCES["combined"]["selection"]["claim_ids"]), 30)
        self.assertEqual(len(SOURCES["rows"]), 90)
        self.assertEqual(len(SOURCES["attempts"]), 90)

    def test_count_reconciliation_and_pair_integrity(self):
        rec = analysis.reconcile(SOURCES["rows"], SOURCES["attempts"], SOURCES["combined"],
                                 SOURCES["summary"], SOURCES["manifest"])
        self.assertEqual((rec["runtime_attempt_count"], rec["successful_attempts"], rec["failed_attempts"]), (90, 81, 9))
        self.assertEqual(rec["successful_A_B_C_triples"], 26)
        self.assertEqual((rec["labeled_claim_count"], rec["unlabeled_claim_count"]), (19, 11))
        self.assertFalse([d for d in rec["discrepancy_log"] if d["severity"] == "inconsistency"])

    def test_missing_frozen_source_is_recorded_not_substituted(self):
        hashes = analysis.source_hashes(ROOT)
        self.assertEqual(hashes["evaluation/scifact/scifact_metrics.py"]["status"], "MISSING")
        self.assertEqual(sum(x["status"] == "MATCH" for x in hashes.values()), 7)

    def test_failure_taxonomy(self):
        cases = [
            ({"error": "URLError: <urlopen error [WinError 10061] target actively refused>"},
             ("model_endpoint_failure", "connection_refused")),
            ({"error": "Phase 8C execution failed: Phase 5 execution failed (TaskExecutionError)."},
             ("controller_execution_failure", "phase5_task_execution_error")),
            ({"error": "JSON schema malformed"}, ("invalid_model_output", None)),
            ({"error": "timeout waiting for service"}, ("timeout", None)),
        ]
        for attempt, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(analysis.classify_failure(attempt), expected)

    def test_exact_failure_mapping_and_persistence(self):
        failures = analysis.build_failure_records(SOURCES["attempts"], SOURCES["rows"], SOURCES["manifest"])
        self.assertEqual([(r["claim_id"], r["system"]) for r in failures], [
            (230, "A"), (230, "B"), (230, "C"), (249, "A"), (249, "B"), (249, "C"),
            (268, "A"), (268, "B"), (859, "C")])
        self.assertEqual(sum(r["primary_category"] == "model_endpoint_failure" for r in failures), 6)
        self.assertEqual(sum(r["primary_category"] == "controller_execution_failure" for r in failures), 3)
        self.assertTrue(all("NO — unique" in r["retry_within_final_run"] for r in failures))
        self.assertTrue(all(r["persisted_in_runtime_attempts"] and r["persisted_in_per_example"] for r in failures))

    def test_failure_stages_are_not_conflated(self):
        failures = analysis.build_failure_records(SOURCES["attempts"], SOURCES["rows"], SOURCES["manifest"])
        endpoint = next(r for r in failures if r["claim_id"] == 230 and r["system"] == "A")
        self.assertTrue(endpoint["inference_started"].startswith("NO"))
        self.assertTrue(endpoint["retrieval_occurred"].startswith("YES"))
        self.assertTrue(endpoint["verification_occurred"].startswith("NO"))
        failed_c = next(r for r in failures if r["claim_id"] == 859 and r["system"] == "C")
        self.assertEqual(failed_c["inference_started"], "UNKNOWN / NOT RECORDED")
        self.assertTrue(failed_c["verification_occurred"].startswith("YES"))
        self.assertEqual(failed_c["controller_failure_trace"]["cycles"][0]["execution_status"], "FAILED")

    def test_prediction_state_separates_failure_invalid_missing_valid(self):
        self.assertEqual(analysis.prediction_state({"status": "FAILED", "predicted_label": None}, "B"), "FAILED")
        self.assertEqual(analysis.prediction_state({"status": "SUCCESS", "predicted_label": None}, "B"), "MISSING")
        self.assertEqual(analysis.prediction_state({"status": "SUCCESS", "predicted_label": "MAYBE"}, "B"), "INVALID")
        self.assertEqual(analysis.prediction_state({"status": "SUCCESS", "predicted_label": "INSUFFICIENT_EVIDENCE"}, "B"), "VALID")
        self.assertEqual(analysis.prediction_state({"status": "SUCCESS", "predicted_label": None}, "A"), "NOT_APPLICABLE")

    def test_classification_denominators_include_failures_exclude_unlabeled(self):
        b, c = analysis.classification_metrics(SOURCES["rows"], "B"), analysis.classification_metrics(SOURCES["rows"], "C")
        self.assertEqual((b["correct_numerator"], b["gold_labeled_denominator"], b["valid_prediction_denominator"]), (11, 19, 17))
        self.assertEqual((c["correct_numerator"], c["gold_labeled_denominator"], c["valid_prediction_denominator"]), (11, 19, 16))
        self.assertEqual(b["outcome_counts"]["failed_execution"], 2)
        self.assertEqual(c["outcome_counts"]["failed_execution"], 3)
        self.assertEqual((b["unlabeled_count"], c["unlabeled_count"]), (11, 11))

    def test_system_a_null_label_field_is_not_mislabeled_as_missing_prediction(self):
        rec = analysis.reconcile(SOURCES["rows"], SOURCES["attempts"], SOURCES["combined"],
                                 SOURCES["summary"], SOURCES["manifest"])
        a = rec["prediction_value_presence_by_system"]["A"]
        self.assertEqual(a["raw_null_prediction_fields"], 30)
        self.assertEqual(a["successful_attempts_with_missing_prediction"], 0)
        self.assertEqual(a["classification_not_applicable"], 30)

    def test_unlabeled_claims_have_no_correctness_label(self):
        result = analysis.build_unlabeled_analysis(SOURCES["rows"])
        self.assertEqual(result["unlabeled_claim_count"], 11)
        self.assertEqual(result["classification_correctness"], "N/A — no gold label; unlabeled is distinct from incorrect.")
        self.assertTrue(all(x["classification_correctness"] == "N/A — no gold label" for x in result["claims"]))

    def test_bc_agreement_separates_valid_agreement_and_failed_pairs(self):
        result = analysis.build_agreement(SOURCES["rows"])
        self.assertEqual(result["B_C_valid_prediction_agreement"], {
            "both_valid_same": 26, "both_valid_different": 0, "valid_pair_denominator": 26})
        self.assertEqual(result["B_C_prediction_agreement_counts"]["both_failed"], 2)
        self.assertEqual(result["B_C_prediction_agreement_counts"]["one_failed"], 2)
        self.assertFalse(result["all_completed_B_C_disagreements"])
        self.assertEqual({x["claim_id"] for x in result["B_C_differences_or_missing_outputs"]}, {230, 249, 268, 859})

    def test_retrieval_analysis_reconstructs_rank_one_document_overlap(self):
        result = analysis.build_retrieval_analysis(SOURCES["rows"])
        self.assertEqual(result["gold_evidence_claim_denominator"], 19)
        self.assertEqual([result["Recall@1_by_system"][s]["hits"] for s in "ABC"], [17, 17, 14])
        self.assertEqual([result["Recall@1_by_system"][s]["claim_denominator"] for s in "ABC"], [19, 19, 16])
        self.assertEqual(result["unavailable_ranked_doc_outputs_by_system"]["C"], [230, 249, 859])

    def test_missing_retrieval_is_not_counted_as_a_hit(self):
        self.assertIs(analysis.retrieval_hit_at_1({"gold_evidence_doc_ids": [12], "retrieved_chunks": []}), False)
        self.assertIsNone(analysis.retrieval_hit_at_1({"gold_evidence_doc_ids": [], "retrieved_chunks": []}))
        self.assertIs(analysis.retrieval_hit_at_1({"gold_evidence_doc_ids": [12]}), False)

    def test_efficiency_keeps_missing_failed_c_counters_visible(self):
        result = analysis.build_efficiency(SOURCES["rows"], SOURCES["attempts"])
        self.assertEqual(result["A"]["latency_seconds"]["success"]["count"], 27)
        self.assertEqual(result["A"]["latency_seconds"]["failed"]["count"], 3)
        for field in ("llm_calls", "retrieval_calls", "verification_calls", "controller_cycles"):
            self.assertEqual(result["C"]["call_counts"][field]["failed"]["missing_values"], 3)

    def test_persisted_citation_aggregate_reconciles_to_phase10c_summary(self):
        result = analysis.build_evidence_analysis(SOURCES["rows"])
        for system in ("B", "C"):
            metric = result["systems"][system]["phase10c_persisted_citation_metric"]
            self.assertEqual(metric["citation_row_denominator"], 19)
            self.assertAlmostEqual(metric["metrics"]["recall"]["mean"], 11 / 19)
            self.assertAlmostEqual(metric["metrics"]["precision"]["mean"], 1.0)

    def test_all_saved_metric_checks_reconcile(self):
        checks = analysis.compare_saved_metrics(SOURCES["rows"], SOURCES["attempts"],
                                                 SOURCES["combined"], SOURCES["summary"])
        self.assertTrue(checks)
        self.assertFalse([x for x in checks if x.get("match") is False])

    def test_controller_trace_counts_separate_failed_dispatches(self):
        result = analysis.build_system_c_analysis(SOURCES["rows"], SOURCES["attempts"])
        self.assertEqual(result["successful_claims_with_controller_traces"], 27)
        self.assertEqual(result["successful_trace_cycle_entries"], 54)
        self.assertEqual(result["successful_accepted_8A_validations"], 54)
        self.assertEqual(result["successful_completed_8B_executions"], 54)
        self.assertEqual(result["successful_claims_with_continuation_authorization"], 27)
        self.assertEqual(result["failed_dispatch_cycles_in_error_traces"], 3)

    def test_discrepancy_detection_reports_missing_record(self):
        rows = list(SOURCES["rows"])
        rows.pop()
        rec = analysis.reconcile(rows, SOURCES["attempts"], SOURCES["combined"], SOURCES["summary"], SOURCES["manifest"])
        self.assertTrue(any(x["item"] == "attempt_pairs_vs_scored_pairs" for x in rec["discrepancy_log"]))

    def test_claim_matrix_has_30_rows_and_a_is_not_applicable(self):
        matrix = analysis.build_claim_matrix(SOURCES["rows"])
        self.assertEqual(len(matrix), 30)
        self.assertTrue(all(row["A_validity"] == "NOT APPLICABLE" for row in matrix))
        self.assertTrue(all(row["A_prediction"].startswith("N/A") for row in matrix))

    def test_analysis_declares_no_external_calls_or_inference(self):
        report = analysis.analyze(root=ROOT, generated_at="2026-09-29T00:00:00+00:00")
        self.assertEqual(report["external_calls"], {"llm": False, "search_or_api": False, "inference_rerun": False})
        self.assertEqual(report["artifact_reconciliation"]["selected_claim_count"], 30)
        self.assertEqual(report["generated_at"], "2026-09-29T00:00:00+00:00")


if __name__ == "__main__":
    unittest.main()
