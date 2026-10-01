"""Deterministic infrastructure-only tests for Phase 10E remediation."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.experiments.phase10e_infrastructure import (
    JsonlRecoveryError, OutputState, append_jsonl, canonical_fingerprint,
    make_configuration_lock, model_health_preflight, output_state,
    recover_jsonl, retrieval_availability_summary, validate_attempt_record,
    validate_resume_configuration, resume_pending_pairs, validate_scored_record,
    exclusive_run_lock,
    interrupted_attempt_record, load_resume_manifest,
)


def rec(attempt_id="a1", claim=1, system="A", status="SUCCESS"):
    return {"attempt_id": attempt_id, "example_id": claim, "system": system,
            "status": status, "runtime_input": {"claim_id": claim, "claim": "safe"}}


class FakeResponse:
    def __init__(self, value):
        self.value = value
    def __enter__(self):
        return self
    def __exit__(self, *_):
        return False
    def read(self):
        return json.dumps(self.value).encode()


class Phase10ERemediationTests(unittest.TestCase):
    def test_output_execution_failed_is_not_empty(self):
        self.assertEqual(output_state(execution_succeeded=False, output_available=True,
            schema_valid=True, item_count=0), OutputState.EXECUTION_FAILED)

    def test_output_unavailable_is_distinct(self):
        self.assertEqual(output_state(execution_succeeded=True, output_available=False,
            schema_valid=None, item_count=None), OutputState.OUTPUT_UNAVAILABLE)

    def test_valid_empty_is_explicit(self):
        self.assertEqual(output_state(execution_succeeded=True, output_available=True,
            schema_valid=True, item_count=0), OutputState.VALID_EMPTY)

    def test_valid_nonempty_is_explicit(self):
        self.assertEqual(output_state(execution_succeeded=True, output_available=True,
            schema_valid=True, item_count=2), OutputState.VALID_NONEMPTY)

    def test_invalid_output_is_not_empty(self):
        self.assertEqual(output_state(execution_succeeded=True, output_available=True,
            schema_valid=False, item_count=0), OutputState.VALID_OUTPUT_INVALID)

    def test_missing_count_is_unavailable(self):
        self.assertEqual(output_state(execution_succeeded=True, output_available=True,
            schema_valid=True, item_count=None), OutputState.OUTPUT_UNAVAILABLE)

    def test_retrieval_failure_not_counted_as_miss(self):
        rows = [{"gold_evidence_doc_ids": [9], "retrieval_output_status": "EXECUTION_FAILED",
                 "retrieved_chunks": []}]
        result = retrieval_availability_summary(rows)
        self.assertEqual(result["valid_output_claims"], 0)
        self.assertEqual(result["gold_document_denominator_over_valid_outputs"], 0)
        self.assertEqual(result["output_status_counts"]["EXECUTION_FAILED"], 1)

    def test_retrieval_unavailable_not_counted_as_miss(self):
        result = retrieval_availability_summary([{"gold_evidence_doc_ids": [9],
            "retrieval_output_status": "OUTPUT_UNAVAILABLE", "retrieved_chunks": []}])
        self.assertEqual(result["gold_document_hits_at_k"]["1"], 0)
        self.assertIsNone(result["recall_at_k_over_valid_outputs"]["1"])
        self.assertEqual(result["unavailable_or_failed_claims"], 1)

    def test_valid_empty_retrieval_is_a_real_miss(self):
        result = retrieval_availability_summary([{"gold_evidence_doc_ids": [9],
            "retrieval_output_status": "VALID_EMPTY", "retrieved_chunks": []}])
        self.assertEqual(result["valid_empty_output_claims"], 1)
        self.assertEqual(result["gold_document_denominator_over_valid_outputs"], 1)
        self.assertEqual(result["recall_at_k_over_valid_outputs"]["1"], 0)

    def test_valid_retrieval_document_overlap(self):
        result = retrieval_availability_summary([{"gold_evidence_doc_ids": [9],
            "retrieval_output_status": "VALID_NONEMPTY",
            "retrieved_chunks": [{"scifact_doc_id": 9}]}])
        self.assertEqual(result["gold_document_hits_at_k"]["1"], 1)
        self.assertEqual(result["recall_at_k_over_valid_outputs"]["1"], 1)

    def test_empty_jsonl(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.jsonl"; p.touch()
            records, report = recover_jsonl(p)
            self.assertEqual(records, []); self.assertEqual(report.records, 0)

    def test_single_complete_record(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.jsonl"; append_jsonl(p, rec())
            self.assertEqual(recover_jsonl(p)[0], [rec()])

    def test_multiple_complete_records(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.jsonl"; append_jsonl(p, rec())
            append_jsonl(p, rec("a2", 2, "B"))
            self.assertEqual(len(recover_jsonl(p)[0]), 2)

    def test_valid_final_record_without_newline_is_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.jsonl"; p.write_text(json.dumps(rec()), encoding="utf-8")
            rows, report = recover_jsonl(p)
            self.assertEqual(rows, [rec()]); self.assertTrue(report.normalized_final_newline)
            self.assertTrue(p.read_bytes().endswith(b"\n"))

    def test_complete_records_and_truncated_tail_quarantined(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.jsonl"
            p.write_bytes((json.dumps(rec()) + "\n{" + '"attempt_id":"partial"').encode())
            rows, report = recover_jsonl(p)
            self.assertEqual(rows, [rec()]); self.assertTrue(report.recovered)
            self.assertTrue(Path(report.quarantined_path).exists())
            self.assertEqual(len(p.read_text(encoding="utf-8").splitlines()), 1)

    def test_truncated_string_tail_quarantined(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.jsonl"; p.write_bytes(b'{"message":"unfinished')
            rows, report = recover_jsonl(p)
            self.assertEqual(rows, []); self.assertTrue(report.recovered)

    def test_malformed_complete_line_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.jsonl"; p.write_bytes(b'{bad}\n')
            with self.assertRaises(JsonlRecoveryError): recover_jsonl(p)

    def test_malformed_line_before_final_tail_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.jsonl"; p.write_bytes(b'{bad}\n{"tail":')
            with self.assertRaises(JsonlRecoveryError): recover_jsonl(p)

    def test_malformed_unterminated_final_line_is_not_assumed_truncated(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.jsonl"; p.write_bytes(b'{bad}')
            with self.assertRaisesRegex(JsonlRecoveryError, "Malformed complete"):
                recover_jsonl(p)

    def test_duplicate_attempt_ids_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.jsonl"; append_jsonl(p, rec()); append_jsonl(p, rec())
            with self.assertRaisesRegex(JsonlRecoveryError, "Duplicate"):
                recover_jsonl(p, unique_key="attempt_id")

    def test_schema_invalid_record_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.jsonl"; append_jsonl(p, {"bad": 1})
            with self.assertRaises(JsonlRecoveryError):
                recover_jsonl(p, validator=validate_attempt_record)

    def test_scored_schema_invalid_record_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "scores.jsonl"; append_jsonl(p, {"example_id": 1})
            with self.assertRaisesRegex(JsonlRecoveryError, "Schema-invalid"):
                recover_jsonl(p, validator=validate_scored_record)

    def test_identical_resume_lock_allowed(self):
        configuration = {"dataset": {"hash": "a"}, "selection": [1, 2]}
        self.assertEqual(validate_resume_configuration(configuration, configuration),
                         canonical_fingerprint(configuration))

    def test_resume_dataset_hash_change_rejected(self): self._reject_change("dataset.hash", "a", "b")
    def test_resume_selected_claims_change_rejected(self): self._reject_change("selection", [1], [2])
    def test_resume_seed_change_rejected(self): self._reject_change("seed", 7, 8)
    def test_resume_model_digest_change_rejected(self): self._reject_change("model.digest", "d1", "d2")
    def test_resume_model_name_change_rejected(self): self._reject_change("model.name", "m1", "m2")
    def test_resume_prompt_change_rejected(self): self._reject_change("prompt.hash", "p1", "p2")
    def test_resume_retrieval_change_rejected(self): self._reject_change("retrieval.top_k", 20, 10)
    def test_resume_schema_change_rejected(self): self._reject_change("schema", "v1", "v2")
    def test_resume_system_definition_change_rejected(self): self._reject_change("systems.C", "real", "fake")

    def _reject_change(self, key, old, new):
        def build(value):
            if "." not in key: return {key: value}
            a, b = key.split("."); return {a: {b: value}}
        with self.assertRaisesRegex(ValueError, key.replace(".", r"\.")):
            validate_resume_configuration(build(old), build(new))

    def test_configuration_lock_fingerprint_stable(self):
        value = {"selection": [3, 1], "model": {"name": "x"}}
        self.assertEqual(make_configuration_lock(value), make_configuration_lock(value))

    def test_preflight_no_endpoint_records_failure_without_chat(self):
        calls = []
        def opener(req, timeout):
            calls.append(req.full_url)
            raise ConnectionRefusedError("refused")
        result = model_health_preflight(opener=opener)
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["failure_category"], "connection_refused")
        self.assertEqual(calls, ["http://127.0.0.1:11434/api/tags"])

    def test_preflight_model_missing_stops_before_chat(self):
        calls = []
        def opener(req, timeout):
            calls.append(req.full_url); return FakeResponse({"models": []})
        result = model_health_preflight(opener=opener)
        self.assertFalse(result["model_present"]); self.assertEqual(len(calls), 1)

    def test_preflight_success_captures_digest_and_json(self):
        def opener(req, timeout):
            if req.full_url.endswith("/api/tags"):
                return FakeResponse({"models": [{"name": "qwen3:4b", "digest": "sha256:abc"}]})
            if req.full_url.endswith("/api/show"):
                return FakeResponse({"details": {"family": "qwen"}, "model_info": {"x": 1}})
            return FakeResponse({"message": {"content": '{"status":"ok"}'}})
        result = model_health_preflight(opener=opener)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["model_digest"], "sha256:abc")
        self.assertTrue(result["json_capability"])

    def test_preflight_digest_unavailable_is_not_health_failure(self):
        def opener(req, timeout):
            if req.full_url.endswith("/api/tags"):
                return FakeResponse({"models": [{"name": "qwen3:4b"}]})
            if req.full_url.endswith("/api/show"):
                return FakeResponse({"details": {}, "model_info": {}})
            return FakeResponse({"message": {"content": '{"status":"ok"}'}})
        result = model_health_preflight(opener=opener)
        self.assertEqual(result["status"], "PASS")
        self.assertIsNone(result["model_digest"])
        self.assertEqual(result["digest_status"], "unavailable")

    def test_preflight_malformed_json_fails(self):
        def opener(req, timeout):
            if req.full_url.endswith("/api/tags"):
                return FakeResponse({"models": [{"name": "qwen3:4b"}]})
            if req.full_url.endswith("/api/show"):
                return FakeResponse({"details": {}, "model_info": {}})
            return FakeResponse({"message": {"content": "not-json"}})
        result = model_health_preflight(opener=opener)
        self.assertEqual(result["status"], "FAIL")
        self.assertFalse(result["json_capability"])

    def test_synthetic_resume_skips_persisted_and_failed_attempts(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "attempts.jsonl"
            append_jsonl(p, rec("ok-a", 1, "A", "SUCCESS"))
            append_jsonl(p, rec("fail-b", 1, "B", "FAILED"))
            append_jsonl(p, rec("ok-c", 1, "C", "SUCCESS"))
            rows, _ = recover_jsonl(p, validator=validate_attempt_record,
                                    unique_key="attempt_id")
            expected = [(1, "A"), (1, "B"), (1, "C"), (2, "A")]
            self.assertEqual(resume_pending_pairs(expected, rows), [(2, "A")])
            append_jsonl(p, rec("resume-2a", 2, "A", "SUCCESS"))
            rows, _ = recover_jsonl(p, validator=validate_attempt_record,
                                    unique_key="attempt_id")
            self.assertEqual(resume_pending_pairs(expected, rows), [])

    def test_synthetic_resume_rejects_duplicate_pair(self):
        with self.assertRaisesRegex(ValueError, "Duplicate claim/system"):
            resume_pending_pairs([(1, "A")], [rec("a", 1, "A"), rec("b", 1, "A")])

    def test_interrupted_attempt_is_persisted_and_not_retried(self):
        interrupted = rec("i1", 1, "C", "INTERRUPTED")
        validate_attempt_record(interrupted)
        self.assertEqual(resume_pending_pairs([(1, "C")], [interrupted]), [])

    def test_partial_c_keeps_valid_retrieval_but_evidence_unavailable(self):
        row = {"gold_evidence_doc_ids": [7], "retrieval_output_status": "VALID_NONEMPTY",
               "retrieved_chunks": [{"scifact_doc_id": 7}],
               "evidence_output_status": "OUTPUT_UNAVAILABLE"}
        availability = retrieval_availability_summary([row])
        self.assertEqual(availability["valid_output_claims"], 1)
        self.assertEqual(availability["gold_document_hits_at_k"]["1"], 1)
        self.assertEqual(row["evidence_output_status"], "OUTPUT_UNAVAILABLE")

    def test_timeout_preflight_is_recorded_as_failure(self):
        def opener(req, timeout):
            raise TimeoutError("deadline")
        result = model_health_preflight(opener=opener)
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["failure_category"], "timeout")

    def test_exclusive_run_lock_rejects_second_writer(self):
        with tempfile.TemporaryDirectory() as d:
            lock = Path(d) / "run.lock"
            with exclusive_run_lock(lock):
                with self.assertRaisesRegex(RuntimeError, "locked"):
                    with exclusive_run_lock(lock):
                        pass
            self.assertFalse(lock.exists())

    def test_stale_run_lock_is_recovered_after_dead_pid(self):
        with tempfile.TemporaryDirectory() as d:
            lock = Path(d) / "run.lock"
            lock.write_text(json.dumps({"pid": 2_000_000_000}), encoding="utf-8")
            with exclusive_run_lock(lock):
                self.assertTrue(lock.exists())
            self.assertFalse(lock.exists())

    def test_malformed_run_lock_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            lock = Path(d) / "run.lock"; lock.write_text("not-json", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "malformed"):
                with exclusive_run_lock(lock):
                    pass

    def test_inflight_attempt_becomes_unknown_completion_failure(self):
        row = interrupted_attempt_record({"attempt_id": "x", "example_id": 5,
            "system": "C", "runtime_input": {"claim_id": 5, "claim": "c"},
            "started_at_utc": "2026-10-01T00:00:00Z"})
        self.assertEqual(row["status"], "INTERRUPTED")
        self.assertEqual(row["retrieval_output_status"], "OUTPUT_UNAVAILABLE")
        self.assertEqual(row["interruption_provenance"]["retrieval_occurred"], "UNKNOWN")
        self.assertFalse(row["interruption_provenance"]["automatic_retry"])

    def test_missing_resume_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, "Resume artifact missing"):
                load_resume_manifest(Path(d) / "run_manifest.json")

    def test_corrupted_resume_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "run_manifest.json"; p.write_text("{broken", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Corrupted resume manifest"):
                load_resume_manifest(p)


if __name__ == "__main__":
    unittest.main()
