from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
for path in (str(ROOT), str(TESTS)):
    if path not in sys.path:
        sys.path.insert(0, path)

import test_research_autonomy as fixtures
from research_autonomy import (
    ResearchActionProposal, ResearchActionType, ResearchDecision,
    ResearchStopAssessment, ValidationStatus, build_autonomous_research_context,
)
from research_autonomous_controller import (
    AutonomousControllerPolicy, AutonomousControllerRequest,
    AutonomousControllerStopReason, run_autonomous_controller,
)
from research_autonomous_execution import (
    AutonomousExecutionApproval, AutonomousExecutionRequest,
    AutonomousExecutionStatus, execution_policy_digest, provider_bindings_for_action,
)
from research_capabilities import LocalRetrievalRequest
from research_retrieval import StaticRetrievalProvider


def _policy(f, **overrides):
    values = dict(maximum_cycles=2, maximum_executions=2,
                  maximum_capability_invocations=2, maximum_retrieval_actions=2,
                  maximum_verification_actions=2, maximum_total_research_actions=2)
    values.update(overrides)
    return AutonomousControllerPolicy(**values)


def _stop(context, *, reason="Enough evidence has been gathered."):
    proposal = ResearchActionProposal(
        ResearchActionType.STOP, reason, estimated_information_gain=0.0,
        priority="critical", stop_assessment=ResearchStopAssessment(
            "caller-directed stop", "The controller fixture requests a safe stop."))
    return fixtures._decision_for(context, proposal)


def _request(f, decision, validation, context, provider):
    return AutonomousExecutionRequest(
        decision, validation, f[2].run_id, f[7].loop_id, f[7].current_iteration_id,
        f[0].plan_id, f[0].revision, execution_policy_digest(f[8]),
        capability_request=LocalRetrievalRequest(decision.proposal.query),
        provider_refs=provider_bindings_for_action(
            ResearchActionType.RETRIEVE_EVIDENCE,
            retrieval_provider=provider))


def _controller(f, decision_provider, *, approval_provider=None, policy=None, provider=None,
                request_factory=None):
    execution_policy = f[8]
    selected_provider = provider or StaticRetrievalProvider([])
    return AutonomousControllerRequest(
        plan=f[0], run=f[2], review=f[1], loop=f[7], state=f[4],
        execution_policy=execution_policy, policy=policy or _policy(f),
        decision_provider=decision_provider,
        request_factory=request_factory or (lambda d, v, c: _request(f, d, v, c, selected_provider)),
        approval_provider=approval_provider or (lambda r: AutonomousExecutionApproval.for_request(r, reviewer="researcher")),
        retrieval_provider=selected_provider)


class ResearchAutonomousControllerTests(unittest.TestCase):
    def test_stop_decision_is_validated_and_stops_without_execution(self):
        f = fixtures._fixture()
        result = run_autonomous_controller(_controller(f, _stop))
        self.assertEqual(result.stop_reason, AutonomousControllerStopReason.DECISION_STOP)
        self.assertEqual(result.executions_completed, 0)
        self.assertEqual(result.trace.cycles[0].validation.validation_status, ValidationStatus.ACCEPTED)

    def test_exactly_one_decision_is_requested_for_a_stop_cycle(self):
        f = fixtures._fixture()
        seen = []
        result = run_autonomous_controller(_controller(f, lambda c: (seen.append(c.context_id), _stop(c))[1]))
        self.assertEqual(len(seen), 1)
        self.assertEqual(result.cycles_completed, 1)

    def test_stale_decision_context_prevents_execution(self):
        f = fixtures._fixture()
        old = build_autonomous_research_context(f[0], f[2], f[1], f[7], f[4], f[8])
        decision = _stop(old)
        f[4].unresolved_questions.append("state changed after the snapshot")
        result = run_autonomous_controller(_controller(f, lambda _: decision))
        self.assertEqual(result.stop_reason, AutonomousControllerStopReason.STALE_STATE)
        self.assertEqual(result.executions_completed, 0)

    def test_invalid_proposal_is_rejected_by_8a(self):
        f = fixtures._fixture()
        def choose(context):
            d = _stop(context)
            invalid = ResearchActionProposal(ResearchActionType.RETRIEVE_EVIDENCE, "foreign task",
                                             task_id="foreign", query="q")
            return fixtures._decision_for(context, invalid)
        result = run_autonomous_controller(_controller(f, choose))
        self.assertEqual(result.stop_reason, AutonomousControllerStopReason.VALIDATION_REJECTED)
        self.assertIsNone(result.trace.cycles[0].execution)

    def test_decision_generation_error_is_typed_and_not_retried(self):
        f = fixtures._fixture()
        calls = []
        def failed(_):
            calls.append(1)
            raise RuntimeError("provider unavailable")
        result = run_autonomous_controller(_controller(f, failed))
        self.assertEqual(result.stop_reason, AutonomousControllerStopReason.ERROR)
        self.assertEqual(calls, [1])
        self.assertIn("provider unavailable", result.failure_reason)

    def test_non_decision_return_is_rejected_as_provider_failure(self):
        f = fixtures._fixture()
        result = run_autonomous_controller(_controller(f, lambda _: None))
        self.assertEqual(result.stop_reason, AutonomousControllerStopReason.NO_ACTION)
        self.assertEqual(result.trace.cycles[0].status, "NO_ACTION")

    def test_missing_live_loop_is_returned_as_stale_state(self):
        f = fixtures._fixture()
        f[4].research_loops.clear()
        called = []
        result = run_autonomous_controller(_controller(f, lambda c: called.append(c)))
        self.assertEqual(result.stop_reason, AutonomousControllerStopReason.STALE_STATE)
        self.assertEqual(called, [])

    def test_no_researcher_approval_pauses_before_8b(self):
        f = fixtures._fixture()
        def choose(context):
            p = ResearchActionProposal(ResearchActionType.RETRIEVE_EVIDENCE,
                "Retrieve one bounded source", task_id=f[3].task_id,
                question_ids=(f[6].question_id,), query="bounded retrieval")
            return fixtures._decision_for(context, p)
        result = run_autonomous_controller(_controller(f, choose, approval_provider=lambda _: None))
        self.assertEqual(result.stop_reason, AutonomousControllerStopReason.WAIT_FOR_RESEARCHER)
        self.assertEqual(result.executions_completed, 0)
        self.assertEqual(result.trace.cycles[0].status, "WAITING_FOR_RESEARCHER")

    def test_approved_proposal_passes_through_8b_once(self):
        f = fixtures._fixture()
        def choose(context):
            p = ResearchActionProposal(ResearchActionType.RETRIEVE_EVIDENCE,
                "Retrieve one bounded source", task_id=f[3].task_id,
                question_ids=(f[6].question_id,), query="bounded retrieval")
            return fixtures._decision_for(context, p)
        result = run_autonomous_controller(_controller(f, choose))
        self.assertEqual(result.executions_completed, 1)
        self.assertEqual(result.trace.cycles[0].execution.status, AutonomousExecutionStatus.COMPLETED)
        self.assertIsNotNone(result.trace.cycles[0].execution.trace.controlled_execution_id)

    def test_successful_action_waits_without_existing_continuation_authorization(self):
        f = fixtures._fixture()
        def choose(context):
            p = ResearchActionProposal(ResearchActionType.RETRIEVE_EVIDENCE,
                "Retrieve one bounded source", task_id=f[3].task_id,
                question_ids=(f[6].question_id,), query="bounded retrieval")
            return fixtures._decision_for(context, p)
        result = run_autonomous_controller(_controller(f, choose))
        self.assertEqual(result.stop_reason, AutonomousControllerStopReason.WAIT_FOR_RESEARCHER)
        self.assertEqual(f[7].continuation_authorizations, ())

    def test_permitted_continuation_reobserves_state_before_next_decision(self):
        f = list(fixtures._fixture())
        f[8] = replace(f[8], allow_continuation=True)
        f = tuple(f)
        observed = []
        def choose(context):
            observed.append(context.context_id)
            if len(observed) == 1:
                proposal = ResearchActionProposal(ResearchActionType.RETRIEVE_EVIDENCE,
                    "Retrieve one bounded source", task_id=f[3].task_id,
                    question_ids=(f[6].question_id,), query="bounded retrieval")
                return fixtures._decision_for(context, proposal)
            return _stop(context, reason="The updated state is sufficient to stop.")
        # This isolates cycle orchestration. The production continuation predicate
        # reads the registered Phase 7C authorizations from ResearchState.
        with patch("research_autonomous_controller._has_existing_continuation", return_value=True):
            result = run_autonomous_controller(_controller(f, choose))
        self.assertEqual(result.cycles_completed, 2)
        self.assertEqual(result.executions_completed, 1)
        self.assertEqual(result.stop_reason, AutonomousControllerStopReason.DECISION_STOP)
        self.assertEqual(len(set(observed)), 2)

    def test_cycle_trace_links_decision_validation_request_and_execution(self):
        f = fixtures._fixture()
        def choose(context):
            return fixtures._decision_for(context, ResearchActionProposal(
                ResearchActionType.RETRIEVE_EVIDENCE, "Retrieve", task_id=f[3].task_id,
                question_ids=(f[6].question_id,), query="q"))
        cycle = run_autonomous_controller(_controller(f, choose)).trace.cycles[0]
        self.assertEqual(cycle.decision.trace.decision_id, cycle.validation.decision_id)
        self.assertEqual(cycle.request_id, cycle.execution.trace.request_id)
        self.assertEqual(cycle.decision.trace.decision_id, cycle.execution.trace.decision_id)

    def test_trace_preserves_provenance_ids(self):
        f = fixtures._fixture()
        def choose(context):
            p = ResearchActionProposal(ResearchActionType.RETRIEVE_EVIDENCE,
                "Retrieve with provenance", task_id=f[3].task_id,
                question_ids=(f[6].question_id,), query="q", provenance_ids=())
            return fixtures._decision_for(context, p)
        cycle = run_autonomous_controller(_controller(f, choose)).trace.cycles[0]
        self.assertEqual(cycle.execution.trace.provenance_ids, cycle.decision.proposal.provenance_ids)

    def test_execution_artifact_and_evidence_ids_are_retained(self):
        f = fixtures._fixture()
        def choose(context):
            p = ResearchActionProposal(ResearchActionType.RETRIEVE_EVIDENCE,
                "Retrieve", task_id=f[3].task_id, question_ids=(f[6].question_id,), query="q")
            return fixtures._decision_for(context, p)
        cycle = run_autonomous_controller(_controller(f, choose)).trace.cycles[0]
        self.assertEqual(cycle.execution.trace.artifact_ids,
                         tuple(item.artifact_id for item in f[4].research_artifacts))
        self.assertEqual(cycle.execution.trace.evidence_ids, ())
        self.assertIsNotNone(cycle.execution.trace.controlled_execution_id)

    def test_execution_failure_is_returned_without_retry(self):
        f = fixtures._fixture()
        calls = []
        class FailingProvider:
            def retrieve(self, query, limit=8):
                raise RuntimeError("retrieval failed")
        def choose(context):
            calls.append(context.context_id)
            return fixtures._decision_for(context, ResearchActionProposal(
                ResearchActionType.RETRIEVE_EVIDENCE, "Retrieve", task_id=f[3].task_id,
                question_ids=(f[6].question_id,), query="q"))
        result = run_autonomous_controller(_controller(f, choose, provider=FailingProvider()))
        self.assertEqual(len(calls), 1)
        self.assertEqual(result.executions_completed, 0)
        self.assertEqual(result.stop_reason, AutonomousControllerStopReason.EXECUTION_FAILED)
        self.assertEqual(result.trace.cycles[0].execution.status, AutonomousExecutionStatus.FAILED)

    def test_policy_rejects_zero_cycle_limit(self):
        with self.assertRaisesRegex(ValueError, "maximum_cycles"):
            AutonomousControllerPolicy(0, 0, 0)

    def test_policy_rejects_negative_execution_limit(self):
        with self.assertRaisesRegex(ValueError, "maximum_executions"):
            AutonomousControllerPolicy(1, -1, 1)

    def test_policy_rejects_negative_capability_limit(self):
        with self.assertRaisesRegex(ValueError, "maximum_capability_invocations"):
            AutonomousControllerPolicy(1, 1, -1)

    def test_policy_rejects_negative_retrieval_limit(self):
        with self.assertRaisesRegex(ValueError, "maximum_retrieval_actions"):
            AutonomousControllerPolicy(1, 1, 1, maximum_retrieval_actions=-1)

    def test_policy_rejects_negative_verification_limit(self):
        with self.assertRaisesRegex(ValueError, "maximum_verification_actions"):
            AutonomousControllerPolicy(1, 1, 1, maximum_verification_actions=-1)

    def test_policy_rejects_negative_total_action_limit(self):
        with self.assertRaisesRegex(ValueError, "maximum_total_research_actions"):
            AutonomousControllerPolicy(1, 1, 1, maximum_total_research_actions=-1)

    def test_policy_rejects_non_boolean_review_flag(self):
        with self.assertRaisesRegex(ValueError, "require_researcher_review"):
            AutonomousControllerPolicy(1, 1, 1, require_researcher_review="yes")

    def test_policy_rejects_cycles_above_registered_loop_limit(self):
        f = fixtures._fixture()
        with self.assertRaisesRegex(ValueError, "ResearchLoop limit"):
            _controller(f, _stop, policy=_policy(f, maximum_cycles=f[7].max_iterations + 1))

    def test_policy_rejects_executions_above_existing_execution_budget(self):
        f = fixtures._fixture()
        with self.assertRaisesRegex(ValueError, "task-execution budget"):
            _controller(f, _stop, policy=_policy(f, maximum_executions=f[8].maximum_task_executions + 1))

    def test_policy_rejects_capability_calls_above_existing_budget(self):
        f = fixtures._fixture()
        with self.assertRaisesRegex(ValueError, "capability.*budget"):
            _controller(f, _stop, policy=_policy(f, maximum_capability_invocations=f[8].maximum_capability_invocations + 1))

    def test_policy_requires_bounded_execution_policy(self):
        f = fixtures._fixture()
        with self.assertRaisesRegex(ValueError, "BoundedExecutionPolicy"):
            AutonomousControllerRequest(f[0], f[2], f[1], f[7], f[4], object(), _policy(f), _stop,
                                       lambda *args: None, lambda _: None)

    def test_policy_requires_execution_request_factory(self):
        f = fixtures._fixture()
        with self.assertRaisesRegex(ValueError, "callbacks"):
            AutonomousControllerRequest(f[0], f[2], f[1], f[7], f[4], f[8], _policy(f), _stop,
                                       None, lambda _: None)

    def test_policy_requires_approval_callback(self):
        f = fixtures._fixture()
        with self.assertRaisesRegex(ValueError, "callbacks"):
            AutonomousControllerRequest(f[0], f[2], f[1], f[7], f[4], f[8], _policy(f), _stop,
                                       lambda *args: None, None)

    def test_request_requires_controller_policy(self):
        f = fixtures._fixture()
        with self.assertRaisesRegex(ValueError, "AutonomousControllerPolicy"):
            AutonomousControllerRequest(f[0], f[2], f[1], f[7], f[4], f[8], object(), _stop,
                                       lambda *args: None, lambda _: None)

    def test_request_rejects_blank_controller_identity(self):
        f = fixtures._fixture()
        with self.assertRaisesRegex(ValueError, "controller_id"):
            AutonomousControllerRequest(f[0], f[2], f[1], f[7], f[4], f[8], _policy(f), _stop,
                                       lambda *args: None, lambda _: None, controller_id=" ")

    def test_request_factory_must_return_typed_8b_request(self):
        f = fixtures._fixture()
        def choose(context):
            return fixtures._decision_for(context, ResearchActionProposal(
                ResearchActionType.RETRIEVE_EVIDENCE, "Retrieve", task_id=f[3].task_id,
                question_ids=(f[6].question_id,), query="q"))
        result = run_autonomous_controller(_controller(f, choose, request_factory=lambda *args: object()))
        self.assertEqual(result.stop_reason, AutonomousControllerStopReason.ERROR)
        self.assertIn("AutonomousExecutionRequest", result.failure_reason)

    def test_request_factory_cannot_swap_decision_identity(self):
        f = fixtures._fixture()
        def choose(context):
            return fixtures._decision_for(context, ResearchActionProposal(
                ResearchActionType.RETRIEVE_EVIDENCE, "Retrieve", task_id=f[3].task_id,
                question_ids=(f[6].question_id,), query="q"))
        def bad_factory(d, v, c):
            request = _request(f, d, v, c)
            other = fixtures._decision_for(c, ResearchActionProposal(
                ResearchActionType.RETRIEVE_EVIDENCE, "Different rationale", task_id=f[3].task_id,
                question_ids=(f[6].question_id,), query="q"))
            return replace(request, decision=other)
        result = run_autonomous_controller(_controller(f, choose, request_factory=bad_factory))
        self.assertEqual(result.stop_reason, AutonomousControllerStopReason.ERROR)
        self.assertIsNotNone(result.failure_reason)

    def test_request_factory_cannot_swap_validation_identity(self):
        f = fixtures._fixture()
        def choose(context):
            return fixtures._decision_for(context, ResearchActionProposal(
                ResearchActionType.RETRIEVE_EVIDENCE, "Retrieve", task_id=f[3].task_id,
                question_ids=(f[6].question_id,), query="q"))
        def bad_factory(d, v, c):
            return replace(_request(f, d, v, c), prior_validation=replace(v, rejection_reasons=("bad",), validation_status=ValidationStatus.REJECTED))
        result = run_autonomous_controller(_controller(f, choose, request_factory=bad_factory))
        self.assertEqual(result.stop_reason, AutonomousControllerStopReason.ERROR)

    def test_approval_callback_cannot_synthesize_approval_when_missing(self):
        f = fixtures._fixture()
        def choose(context):
            return fixtures._decision_for(context, ResearchActionProposal(
                ResearchActionType.RETRIEVE_EVIDENCE, "Retrieve", task_id=f[3].task_id,
                question_ids=(f[6].question_id,), query="q"))
        result = run_autonomous_controller(_controller(f, choose, approval_provider=lambda _: None))
        self.assertEqual(result.stop_reason, AutonomousControllerStopReason.WAIT_FOR_RESEARCHER)
        self.assertFalse(any(c.execution is not None for c in result.trace.cycles))

    def test_existing_terminal_loop_is_not_decided_again(self):
        f = fixtures._fixture()
        from research_loop import StopReason, TransitionAuthority, stop_research_loop
        stop_research_loop(f[4], f[7].loop_id, StopReason.RESEARCHER_REQUESTED_STOP,
                           authority=TransitionAuthority.RESEARCHER, message="Terminal fixture.")
        calls = []
        result = run_autonomous_controller(_controller(f, lambda c: (calls.append(c), _stop(c))[1]))
        self.assertEqual(result.stop_reason, AutonomousControllerStopReason.DECISION_STOP)
        self.assertEqual(calls, [])

    def test_unresolved_blocker_stops_before_decision_provider(self):
        f = fixtures._fixture()
        from research_loop import UnresolvedWork, UnresolvedWorkKind, add_unresolved_work
        add_unresolved_work(f[4], f[7].loop_id, UnresolvedWork(
            UnresolvedWorkKind.MISSING_EVIDENCE, "Blocking evidence gap", blocks_progress=True))
        calls = []
        result = run_autonomous_controller(_controller(f, lambda c: (calls.append(c), _stop(c))[1]))
        self.assertEqual(result.stop_reason, AutonomousControllerStopReason.BLOCKED)
        self.assertEqual(calls, [])

    def test_result_serializes_with_trace_and_stop_reason(self):
        f = fixtures._fixture()
        result = run_autonomous_controller(_controller(f, _stop))
        payload = result.to_json()
        self.assertIn(result.stop_reason.value, payload)
        self.assertIn(result.trace.controller_id, payload)

    def test_policy_is_serializable(self):
        f = fixtures._fixture()
        self.assertEqual(_policy(f).to_dict()["maximum_cycles"], 2)

    def test_controller_request_serializes_scope_and_runtime_binding_requirement(self):
        f = fixtures._fixture()
        request = _controller(f, _stop)
        payload = request.to_dict()
        self.assertEqual(payload["loop_id"], f[7].loop_id)
        self.assertEqual(payload["policy"]["maximum_cycles"], 2)
        self.assertTrue(payload["runtime_bindings_required"])

    def test_cycle_trace_is_serializable(self):
        f = fixtures._fixture()
        result = run_autonomous_controller(_controller(f, _stop))
        cycle_dict = result.trace.cycles[0].to_dict()
        self.assertEqual(cycle_dict["cycle_number"], 1)
        self.assertEqual(cycle_dict["decision"]["trace"]["decision_id"],
                         result.trace.cycles[0].decision.trace.decision_id)

    def test_controller_preserves_epistemic_stop_assessment(self):
        f = fixtures._fixture()
        result = run_autonomous_controller(_controller(f, _stop))
        proposal = result.trace.cycles[0].decision.proposal
        self.assertIsNotNone(proposal.stop_assessment)
        self.assertEqual(proposal.stop_assessment.evidence_summary,
                         "The controller fixture requests a safe stop.")

    def test_controller_does_not_select_a_capability(self):
        f = fixtures._fixture()
        seen = []
        result = run_autonomous_controller(_controller(f, lambda c: (seen.append(c), _stop(c))[1]))
        self.assertEqual(result.trace.cycles[0].decision.proposal.required_capabilities, ())
        self.assertEqual(len(seen), 1)

    def test_cycle_count_is_finite_from_caller_policy(self):
        f = fixtures._fixture()
        result = run_autonomous_controller(_controller(f, _stop, policy=_policy(f, maximum_cycles=1)))
        self.assertLessEqual(result.cycles_completed, 1)

    def test_zero_execution_budget_stops_before_request_creation(self):
        f = fixtures._fixture()
        calls = []
        def choose(context):
            return fixtures._decision_for(context, ResearchActionProposal(
                ResearchActionType.RETRIEVE_EVIDENCE, "Retrieve", task_id=f[3].task_id,
                question_ids=(f[6].question_id,), query="q"))
        result = run_autonomous_controller(_controller(f, choose,
            policy=_policy(f, maximum_executions=0), request_factory=lambda *a: calls.append(a)))
        self.assertEqual(result.stop_reason, AutonomousControllerStopReason.MAXIMUM_EXECUTIONS)
        self.assertEqual(calls, [])

    def test_zero_capability_budget_blocks_capability_proposal(self):
        f = fixtures._fixture()
        def choose(context):
            return fixtures._decision_for(context, ResearchActionProposal(
                ResearchActionType.RETRIEVE_EVIDENCE, "Retrieve", task_id=f[3].task_id,
                question_ids=(f[6].question_id,), query="q"))
        result = run_autonomous_controller(_controller(f, choose,
            policy=_policy(f, maximum_capability_invocations=0)))
        self.assertEqual(result.stop_reason, AutonomousControllerStopReason.LOOP_LIMIT)
        self.assertIsNone(result.trace.cycles[0].execution)

    def test_zero_retrieval_budget_blocks_retrieval_proposal(self):
        f = fixtures._fixture()
        def choose(context):
            return fixtures._decision_for(context, ResearchActionProposal(
                ResearchActionType.RETRIEVE_EVIDENCE, "Retrieve", task_id=f[3].task_id,
                question_ids=(f[6].question_id,), query="q"))
        result = run_autonomous_controller(_controller(f, choose,
            policy=_policy(f, maximum_retrieval_actions=0)))
        self.assertEqual(result.stop_reason, AutonomousControllerStopReason.LOOP_LIMIT)

    def test_zero_total_action_budget_blocks_execution(self):
        f = fixtures._fixture()
        def choose(context):
            return fixtures._decision_for(context, ResearchActionProposal(
                ResearchActionType.RETRIEVE_EVIDENCE, "Retrieve", task_id=f[3].task_id,
                question_ids=(f[6].question_id,), query="q"))
        result = run_autonomous_controller(_controller(f, choose,
            policy=_policy(f, maximum_total_research_actions=0)))
        self.assertEqual(result.stop_reason, AutonomousControllerStopReason.MAXIMUM_EXECUTIONS)


if __name__ == "__main__":
    unittest.main()
