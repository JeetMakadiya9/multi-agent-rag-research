import copy
import hashlib
import json
from pathlib import Path
import unittest

from evaluation.experiments.phase10e_infrastructure import canonical_fingerprint
from evaluation.experiments.run_phase10f import (
    PHASE10F_PRE_REMEDIATION_RUNNER_SHA256,
    PHASE10F_RUNNER_SOURCE_PATH,
    _phase10f_resume_configuration,
    _phase10f_resume_plan,
    audit_c_traces,
)


RUN_ID = "run-test"
TASK_ID = "task-test"
RETRIEVAL_EXEC_ID = "exec-retrieval"
VERIFY_EXEC_ID = "exec-verify"
AUTH_ID = "continue-test"


def attempt(claim_id, system, status="SUCCESS", run_id=RUN_ID, attempt_id=None):
    return {"claim_id": claim_id, "system": system, "execution_status": status,
        "run_id": run_id, "attempt_id": attempt_id or f"{claim_id}-{system}-{status}"}


def trace_row(*, mutate=None, verify=True, selected=None, malformed=False):
    retrieval = {"decision": {"proposal": {"action_type": "RETRIEVE_EVIDENCE"}},
        "validation": {"validation_status": "ACCEPTED"}, "approval_id": "approval-r",
        "execution": {"status": "COMPLETED", "dispatch": {
            "capability": "local_retrieval", "status": "COMPLETED"}}}
    verification = {"decision": {"proposal": {"action_type": "VERIFY_CLAIM"}},
        "validation": {"validation_status": "ACCEPTED"}, "approval_id": "approval-v",
        "execution": {"status": "COMPLETED", "dispatch": {
            "capability": "evidence_verification", "status": "COMPLETED"}}}
    if mutate:
        mutate(retrieval, verification)
    cycles = [retrieval, verification] if verify else [retrieval]
    executions = [{"run_id": RUN_ID, "task_id": TASK_ID, "execution_id": RETRIEVAL_EXEC_ID,
        "status": "COMPLETED", "continuation_of_execution_id": None,
        "continuation_authorization_id": None}]
    loops = []
    if verify:
        executions.append({"run_id": RUN_ID, "task_id": TASK_ID, "execution_id": VERIFY_EXEC_ID,
            "status": "COMPLETED", "continuation_of_execution_id": RETRIEVAL_EXEC_ID,
            "continuation_authorization_id": AUTH_ID})
        loops.append({"continuation_authorizations": [{"run_id": RUN_ID, "task_id": TASK_ID,
            "authorization_id": AUTH_ID, "previous_execution_id": RETRIEVAL_EXEC_ID}],
            "transitions": [{"from_stage": "RESEARCH", "to_stage": "EVIDENCE_REVIEW",
                "authority": "researcher"}]})
    raw = {"controller_traces": [{"trace": {"cycles": cycles}}],
        "research_state": {"task_executions": executions, "research_loops": loops}}
    if malformed:
        raw["controller_traces"] = [{"trace": {"cycles": "not-a-list"}}]
    return {"system": "C", "status": "SUCCESS", "claim_id": 11,
        "selected_evidence_doc_ids": [42] if selected is None and verify else (selected or []),
        "output": {"raw_output": raw}}


class Phase10FResumeRemediationTests(unittest.TestCase):
    def test_completed_a_pair_is_recognized(self):
        plan = _phase10f_resume_plan([(1, "A")], [attempt(1, "A")], expected_run_id=RUN_ID)
        self.assertEqual(plan["completed_pairs"], [(1, "A")])
        self.assertEqual(plan["scheduled_pairs"], [])

    def test_completed_b_pair_is_recognized(self):
        plan = _phase10f_resume_plan([(1, "B")], [attempt(1, "B")], expected_run_id=RUN_ID)
        self.assertEqual(plan["completed_pairs"], [(1, "B")])

    def test_completed_c_pair_is_recognized(self):
        plan = _phase10f_resume_plan([(1, "C")], [attempt(1, "C")], expected_run_id=RUN_ID)
        self.assertEqual(plan["completed_pairs"], [(1, "C")])

    def test_multiple_claims_across_all_systems_are_recognized(self):
        expected = [(claim, system) for claim in (1, 3, 5) for system in ("A", "B", "C")]
        rows = [attempt(claim, system) for claim, system in expected]
        plan = _phase10f_resume_plan(expected, rows, expected_run_id=RUN_ID)
        self.assertEqual(plan["existing_attempt_count"], 9)
        self.assertEqual(plan["missing_pairs"], [])
        self.assertEqual(plan["scheduled_pairs"], [])

    def test_duplicate_claim_system_pair_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate Phase10F"):
            _phase10f_resume_plan([(1, "A")], [attempt(1, "A", attempt_id="a"),
                attempt(1, "A", attempt_id="b")], expected_run_id=RUN_ID)

    def test_missing_pair_is_scheduled_once(self):
        plan = _phase10f_resume_plan([(1, "A"), (1, "B")], [attempt(1, "A")], expected_run_id=RUN_ID)
        self.assertEqual(plan["missing_pairs"], [(1, "B")])
        self.assertEqual(plan["scheduled_pairs"], [(1, "B")])

    def test_failed_attempt_is_terminal_but_not_successful(self):
        plan = _phase10f_resume_plan([(1, "A")], [attempt(1, "A", "FAILED")], expected_run_id=RUN_ID)
        self.assertEqual(plan["completed_pairs"], [])
        self.assertEqual(plan["failed_terminal_pairs"], [(1, "A")])
        self.assertEqual(plan["scheduled_pairs"], [])

    def test_interruption_marker_is_preserved_and_not_retried(self):
        plan = _phase10f_resume_plan([(1, "C")], [attempt(1, "C", "INTERRUPTED")], expected_run_id=RUN_ID)
        self.assertEqual(plan["failed_terminal_pairs"], [(1, "C")])
        self.assertEqual(plan["completed_pairs"], [])
        self.assertEqual(plan["scheduled_pairs"], [])

    def test_historical_runner_migration_is_allowed_only_for_exact_remediation_source(self):
        old = {"run_id": RUN_ID, "dataset": {"hash": "same"}, "selection": [1, 3, 5],
            "source_sha256": {PHASE10F_RUNNER_SOURCE_PATH: PHASE10F_PRE_REMEDIATION_RUNNER_SHA256,
                "system.py": "fixed"}}
        current = copy.deepcopy(old)
        current["source_sha256"][PHASE10F_RUNNER_SOURCE_PATH] = hashlib.sha256(
            Path("evaluation/experiments/run_phase10f.py").read_bytes()).hexdigest().upper()
        stored_lock = {"configuration": old, "fingerprint_sha256": canonical_fingerprint(old)}
        current_lock = {"configuration": current, "fingerprint_sha256": canonical_fingerprint(current)}
        result = _phase10f_resume_configuration(stored_lock, current_lock)
        self.assertTrue(result["compatible"])
        self.assertTrue(result["runner_source_migration_only"])
        self.assertEqual(result["stored_fingerprint"], stored_lock["fingerprint_sha256"])

    def test_incompatible_configuration_is_rejected_with_field(self):
        old = {"run_id": RUN_ID, "dataset": {"hash": "a"}, "selection": [1],
            "source_sha256": {PHASE10F_RUNNER_SOURCE_PATH: PHASE10F_PRE_REMEDIATION_RUNNER_SHA256,
                "system.py": "fixed"}}
        new = copy.deepcopy(old)
        new["source_sha256"][PHASE10F_RUNNER_SOURCE_PATH] = hashlib.sha256(
            Path("evaluation/experiments/run_phase10f.py").read_bytes()).hexdigest().upper()
        new["dataset"]["hash"] = "changed"
        current = {"configuration": new, "fingerprint_sha256": canonical_fingerprint(new)}
        stored = {"configuration": old, "fingerprint_sha256": canonical_fingerprint(old)}
        with self.assertRaisesRegex(ValueError, "dataset.hash"):
            _phase10f_resume_configuration(stored, current)

    def test_incompatible_manifest_run_identity_is_rejected(self):
        expected = [(1, "A")]
        with self.assertRaisesRegex(ValueError, "run identity"):
            _phase10f_resume_plan(expected, [attempt(1, "A", run_id="other")], expected_run_id=RUN_ID)

    def test_missing_identifier_is_rejected(self):
        row = attempt(1, "A")
        del row["claim_id"]
        with self.assertRaisesRegex(ValueError, "missing identifiers"):
            _phase10f_resume_plan([(1, "A")], [row], expected_run_id=RUN_ID)

    def test_repeated_resume_plan_is_idempotent(self):
        expected = [(claim, system) for claim in (1, 3, 5) for system in ("A", "B", "C")]
        rows = [attempt(claim, system) for claim, system in expected]
        first = _phase10f_resume_plan(expected, rows, expected_run_id=RUN_ID)
        second = _phase10f_resume_plan(expected, rows, expected_run_id=RUN_ID)
        self.assertEqual(first["scheduled_pairs"], second["scheduled_pairs"])
        self.assertEqual(first["scheduled_pairs"], [])
        self.assertEqual(first["existing_attempt_count"], second["existing_attempt_count"])

    def test_unrecognized_stored_runner_hash_is_rejected(self):
        stored = {"run_id": RUN_ID, "source_sha256": {PHASE10F_RUNNER_SOURCE_PATH: "unknown"}}
        current = copy.deepcopy(stored)
        current["source_sha256"][PHASE10F_RUNNER_SOURCE_PATH] = hashlib.sha256(
            Path("evaluation/experiments/run_phase10f.py").read_bytes()).hexdigest().upper()
        stored_lock = {"configuration": stored, "fingerprint_sha256": canonical_fingerprint(stored)}
        current_lock = {"configuration": current, "fingerprint_sha256": canonical_fingerprint(current)}
        with self.assertRaisesRegex(ValueError, "unrecognized stored runner"):
            _phase10f_resume_configuration(stored_lock, current_lock)


class Phase10FTraceAuditRemediationTests(unittest.TestCase):
    def audit(self, row):
        return audit_c_traces([row])["records"][0]

    def test_complete_structured_c_trace_passes(self):
        record = self.audit(trace_row())
        self.assertEqual(record["overall"], "PASS")
        self.assertEqual(record["stages"]["phase_8a_validation"], "PASS")
        self.assertEqual(record["stages"]["phase_8b_execution"], "PASS")
        self.assertEqual(record["stages"]["phase_7b_dispatch"], "PASS")
        self.assertEqual(record["stages"]["phase_6_verification"], "PASS")
        self.assertEqual(record["stages"]["continuation_authorization"], "PASS")
        self.assertEqual(record["stages"]["researcher_transition"], "PASS")
        self.assertEqual(record["stages"]["completion"], "PASS")

    def test_missing_8a_validation_fails(self):
        record = self.audit(trace_row(mutate=lambda r, v: r.pop("validation")))
        self.assertEqual(record["stages"]["phase_8a_validation"], "FAIL")
        self.assertEqual(record["overall"], "FAIL")

    def test_missing_8b_execution_fails(self):
        record = self.audit(trace_row(mutate=lambda r, v: r.pop("execution")))
        self.assertEqual(record["stages"]["phase_8b_execution"], "FAIL")

    def test_missing_7b_dispatch_fails(self):
        record = self.audit(trace_row(mutate=lambda r, v: r["execution"].pop("dispatch")))
        self.assertEqual(record["stages"]["phase_7b_dispatch"], "FAIL")

    def test_missing_phase6_verification_when_required_fails(self):
        record = self.audit(trace_row(mutate=lambda r, v: v["execution"]["dispatch"].update(
            {"capability": "local_retrieval"})))
        self.assertEqual(record["stages"]["phase_6_verification"], "FAIL")

    def test_missing_continuation_authorization_fails(self):
        def mutate_state(row):
            raw=row["output"]["raw_output"]["research_state"]
            raw["research_loops"][0]["continuation_authorizations"]=[]
        row=trace_row(); mutate_state(row)
        record=self.audit(row)
        self.assertEqual(record["stages"]["continuation_authorization"], "FAIL")

    def test_mismatched_continuation_authorization_id_fails(self):
        def mutate_state(row):
            raw=row["output"]["raw_output"]["research_state"]
            raw["task_executions"][1]["continuation_authorization_id"]="wrong-id"
        row=trace_row(); mutate_state(row)
        self.assertEqual(self.audit(row)["stages"]["continuation_authorization"], "FAIL")

    def test_missing_researcher_authorized_transition_fails(self):
        def mutate_state(row):
            row["output"]["raw_output"]["research_state"]["research_loops"][0]["transitions"]=[]
        row=trace_row(); mutate_state(row)
        self.assertEqual(self.audit(row)["stages"]["researcher_transition"], "FAIL")

    def test_legitimately_optional_verification_and_continuation(self):
        record=self.audit(trace_row(verify=False, selected=[]))
        self.assertEqual(record["overall"], "PASS")
        self.assertEqual(record["stages"]["phase_6_verification"], "NOT_APPLICABLE")
        self.assertEqual(record["stages"]["continuation_authorization"], "NOT_APPLICABLE")

    def test_malformed_trace_fails_closed(self):
        record=self.audit(trace_row(malformed=True))
        self.assertEqual(record["overall"], "FAIL")
        self.assertEqual(record["stages"]["phase_8a_validation"], "FAIL")

    def test_all_historical_successful_c_traces_are_auditable(self):
        path=Path("evaluation/experiments/results/phase10f_launch_validation/per_example.jsonl")
        rows=[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        report=audit_c_traces(rows)
        self.assertEqual(report["successful_c_executions"],3)
        self.assertEqual(report["pass"],3)
        self.assertEqual(report["limited"],0)
        for item in report["records"]:
            self.assertEqual(item["stages"]["continuation_authorization"],"PASS")
            self.assertEqual(len(item["evidence"]["continuation_links"]),1)


if __name__ == "__main__":
    unittest.main()
