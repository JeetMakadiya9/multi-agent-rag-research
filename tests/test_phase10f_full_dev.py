from types import SimpleNamespace
import unittest

from evaluation.experiments.phase10e_infrastructure import (
    OutputState, canonical_fingerprint, validate_resume_configuration,
)
from evaluation.experiments.run_phase10f import (
    _aggregate, _classification, _failure_category, _validate_attempt,
    audit_c_traces, select_dev_claim_ids,
)


def claim(cid):
    return SimpleNamespace(runtime=SimpleNamespace(claim_id=cid))


def attempt(system, status, state):
    return {"system": system, "execution_status": status,
        "output_availability_status": {"retrieval": state, "evidence": state},
        "latency_seconds": 1.0 if status == "SUCCESS" else None,
        "llm_calls": 1 if status == "SUCCESS" else None,
        "retrieval_calls": 1 if status == "SUCCESS" else None,
        "verification_calls": 0, "controller_cycles": 0 if system == "C" else None}


def score(system, status, prediction, state, chunks, gold=(42,)):
    return {"system": system, "claim_id": 7, "status": status,
        "gold_label": "SUPPORTED", "predicted_label": prediction,
        "gold_evidence_doc_ids": list(gold), "retrieval_output_status": state,
        "retrieval": {"retrieval_scoring_status": "scored" if state in {
            OutputState.VALID_EMPTY.value, OutputState.VALID_NONEMPTY.value} else state,
            "gold_evidence_document_count": len(gold), "reciprocal_rank": None},
        "retrieved_chunks": chunks, "citation": None}


class Phase10FTests(unittest.TestCase):
    def test_launch_selection_is_exact_canonical_prefix(self):
        self.assertEqual(select_dev_claim_ids([claim(i) for i in (91, 11, 8, 6)], full_dev=False), [91, 11, 8])
        with self.assertRaisesRegex(ValueError, "Expected 3"):
            select_dev_claim_ids([claim(91), claim(11)], full_dev=False)

    def test_full_dev_selection_keeps_all_claims_in_file_order(self):
        self.assertEqual(select_dev_claim_ids([claim(i) for i in range(300)], full_dev=True), list(range(300)))

    def test_attempt_runtime_schema_excludes_gold_fields(self):
        row = {"run_id": "r", "attempt_id": "a", "system": "A", "claim_id": 7,
            "runtime_input": {"claim_id": 7, "claim": "text"}, "configuration_fingerprint": "f",
            "model_identity": {}, "retrieval_configuration_identity": {}, "execution_status": "SUCCESS",
            "output_availability_status": {}, "started_at_utc": "t", "completed_at_utc": "t", "output": {}}
        _validate_attempt(row)
        row["runtime_input"]["gold_label"] = "SUPPORTED"
        with self.assertRaisesRegex(ValueError, "unexpected fields"):
            _validate_attempt(row)

    def test_classification_excludes_failed_invalid_and_unlabeled(self):
        result = _classification([
            {"gold_label": "SUPPORTED", "status": "SUCCESS", "predicted_label": "SUPPORTED"},
            {"gold_label": "SUPPORTED", "status": "FAILED", "predicted_label": "SUPPORTED"},
            {"gold_label": "CONTRADICTED", "status": "SUCCESS", "predicted_label": "BAD"},
            {"gold_label": None, "status": "SUCCESS", "predicted_label": "SUPPORTED"}])
        self.assertEqual(result["accuracy"]["numerator"], 1)
        self.assertEqual(result["accuracy"]["denominator"], 1)
        self.assertEqual(result["accuracy"]["value"], 1.0)
        self.assertEqual(result["failed_execution_predictions_excluded"], 1)
        self.assertEqual(result["invalid_predictions_excluded"], 1)
        self.assertEqual(result["unlabeled_claims"], 1)

    def test_goldless_claim_is_not_counted_as_incorrect(self):
        result = _classification([{"gold_label": None, "status": "SUCCESS", "predicted_label": "SUPPORTED"}])
        self.assertEqual(result["accuracy"]["denominator"], 0)
        self.assertEqual(result["unlabeled_claims"], 1)
        self.assertIsNone(result["accuracy"]["value"])

    def test_retrieval_failure_is_excluded_not_scored_as_miss(self):
        attempts = [attempt("A", "SUCCESS", OutputState.VALID_NONEMPTY.value),
            attempt("B", "FAILED", OutputState.EXECUTION_FAILED.value),
            attempt("C", "SUCCESS", OutputState.VALID_EMPTY.value)]
        scores = [score("A", "SUCCESS", None, OutputState.VALID_NONEMPTY.value, [{"scifact_doc_id": 42}]),
            score("B", "FAILED", None, OutputState.EXECUTION_FAILED.value, []),
            score("C", "SUCCESS", None, OutputState.VALID_EMPTY.value, [])]
        result = _aggregate(attempts, scores, 1, resumed_attempt_count=0)
        self.assertEqual(result["retrieval"]["B"]["gold_evidence_claims_with_valid_output"], 0)
        self.assertEqual(result["retrieval"]["B"]["Recall@1"]["denominator"], 0)
        self.assertEqual(result["retrieval"]["B"]["unavailable_or_failed_retrievals_excluded"], 1)
        self.assertEqual(result["retrieval"]["C"]["Recall@1"]["numerator"], 0)
        self.assertEqual(result["retrieval"]["C"]["Recall@1"]["denominator"], 1)

    def test_valid_empty_retrieval_is_a_real_miss(self):
        result = _aggregate([attempt("A", "SUCCESS", OutputState.VALID_EMPTY.value)],
            [score("A", "SUCCESS", None, OutputState.VALID_EMPTY.value, [])], 1, resumed_attempt_count=0)
        self.assertEqual(result["retrieval"]["A"]["gold_evidence_claims_with_valid_output"], 1)
        self.assertEqual(result["retrieval"]["A"]["Recall@1"]["denominator"], 1)
        self.assertEqual(result["retrieval"]["A"]["Recall@1"]["numerator"], 0)

    def test_failure_categories_use_recorded_reason(self):
        self.assertEqual(_failure_category({"execution_status": "FAILED", "error_type": "ConnectionRefusedError"}), "local_model_endpoint_failure")
        self.assertEqual(_failure_category({"execution_status": "TIMEOUT", "error_type": "TimeoutError"}), "timeout")
        self.assertEqual(_failure_category({"execution_status": "FAILED", "error_stage": "phase8_controller"}), "controller_failure")
        self.assertEqual(_failure_category({"execution_status": "FAILED"}), "other_recorded_execution_failure")

    def test_identical_resume_configuration_allowed(self):
        config = {"dataset": {"sha256": "abc"}, "selection": {"ids": [1, 2, 3]}, "model": {"name": "qwen3:4b"}}
        self.assertEqual(validate_resume_configuration(config, dict(config)), canonical_fingerprint(config))

    def test_resume_configuration_rejects_changed_identity_fields(self):
        stored = {"dataset": {"sha256": "abc"}, "selection": {"ids": [1, 2, 3]}, "model": {"name": "qwen3:4b"}}
        for field, value in (("dataset", {"sha256": "changed"}), ("selection", {"ids": [1, 2, 4]}), ("model", {"name": "other"})):
            current = dict(stored)
            current[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                validate_resume_configuration(stored, current)

    def test_real_controller_trace_is_checked_from_persisted_fields(self):
        cycle = {"decision": {"proposal": {"action_type": "RETRIEVE_EVIDENCE"}},
            "validation": {"validation_status": "ACCEPTED"}, "approval_id": "approval-1",
            "execution": {"status": "COMPLETED", "dispatch": {"capability": "local_retrieval", "status": "COMPLETED"}}}
        row = {"system": "C", "status": "SUCCESS", "claim_id": 9, "selected_evidence_doc_ids": [],
            "output": {"raw_output": {"controller_traces": [{"trace": {"cycles": [cycle]}}],
                "research_state": {"task_executions": [{"execution_id": "exec-1", "status": "COMPLETED"}],
                    "research_loops": []}}}}
        audit = audit_c_traces([row])
        self.assertEqual(audit["successful_c_executions"], 1)
        self.assertEqual(audit["pass"], 1)

    def test_gold_fields_on_attempt_record_are_rejected(self):
        row = {"run_id": "r", "attempt_id": "a", "system": "A", "claim_id": 7,
            "runtime_input": {"claim_id": 7, "claim": "text"}, "configuration_fingerprint": "f",
            "model_identity": {}, "retrieval_configuration_identity": {}, "execution_status": "SUCCESS",
            "output_availability_status": {}, "started_at_utc": "t", "completed_at_utc": "t", "output": {},
            "gold_label": "SUPPORTED"}
        with self.assertRaisesRegex(ValueError, "gold fields"):
            _validate_attempt(row)

    def test_verification_action_requires_continuation_authorization_trace(self):
        cycle = {"decision": {"proposal": {"action_type": "VERIFY_CLAIM"}},
            "validation": {"valid": True}, "approval_id": "auth-1",
            "execution": {"status": "COMPLETED", "dispatch": {"capability": "VERIFY_CLAIM", "status": "COMPLETED"}}}
        row = {"system": "C", "status": "SUCCESS", "claim_id": 10,
            "selected_evidence_doc_ids": [42], "output": {"raw_output": {
                "controller_traces": [{"trace": {"cycles": [cycle]}}]}}}
        audit = audit_c_traces([row])
        self.assertEqual(audit["limited"], 1)
        self.assertTrue(audit["records"][0]["continuation_authorization_required"])
    def test_missing_controller_trace_is_limited_not_inferred(self):
        audit = audit_c_traces([{"system": "C", "status": "SUCCESS", "claim_id": 9,
            "output": {}, "selected_evidence_doc_ids": []}])
        self.assertEqual(audit["limited"], 1)
        self.assertEqual(audit["records"][0]["status"], "LIMITED")


if __name__ == "__main__":
    unittest.main()
