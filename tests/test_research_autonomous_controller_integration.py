"""Deterministic integration checks across Phase 7, 8A, 8B, and 8C."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
for path in (str(ROOT), str(TESTS)):
    if path not in sys.path:
        sys.path.insert(0, path)

import test_research_capabilities as capability_fixtures
from research_autonomy import ResearchActionType
from research_autonomous_controller import (
    AutonomousControllerPolicy,
    AutonomousControllerRequest,
    AutonomousControllerStopReason,
    run_autonomous_controller,
)
from research_autonomous_execution import (
    ApprovalStatus,
    AutonomousExecutionApproval,
    AutonomousExecutionRequest,
    AutonomousExecutionStatus,
    execution_policy_digest,
    provider_bindings_for_action,
)
from research_bounded_loop import BoundedExecutionPolicy
from research_capabilities import CapabilityType, LocalRetrievalRequest
from research_continuation import ResearchContinuationAction, apply_research_continuation
from research_loop import (
    LoopStage,
    ResearchLoopStatus,
    begin_loop_iteration,
    create_research_loop,
    start_research_loop,
    transition_research_loop,
)
from research_retrieval import (
    RetrievalEvidence,
    RetrievalResponse,
    register_rag_document_version,
)
from research_state import ResearchClaim


def _proposal_payload(task_id: str, question_id: str, query: str) -> str:
    return json.dumps({
        "action_type": "RETRIEVE_EVIDENCE",
        "rationale": "Retrieve local evidence for the approved research question.",
        "task_id": task_id,
        "question_ids": [question_id],
        "claim_ids": [],
        "evidence_ids": [],
        "artifact_ids": [],
        "query": query,
        "problem_statement": None,
        "candidate_id": None,
        "estimated_information_gain": 0.5,
        "priority": "medium",
        "unresolved_questions_addressed": [],
        "stop_after_action": False,
        "stop_assessment": None,
        "provenance_ids": [],
    })


class QueueDecisionProvider:
    """An LLMProvider-compatible test binding; 8A still parses and validates output."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []
        self.calls = 0

    def generate(self, messages, *, temperature=0.0, timeout=120):
        self.calls += 1
        self.prompts.append(messages[1]["content"])
        if not self.responses:
            raise AssertionError("8A requested more decisions than the test scenario supplied.")
        return self.responses.pop(0)


class SequencedRetrievalProvider:
    """Returns deterministic, distinct evidence rows for repeated local retrieval."""

    def __init__(self, *, fail=False):
        self.calls = []
        self.fail = fail

    def retrieve(self, query, limit=8):
        self.calls.append((query, limit))
        if self.fail:
            raise RuntimeError("synthetic local retrieval failure")
        number = len(self.calls)
        evidence = [RetrievalEvidence(
            evidence_id=f"integration-evidence-{number}",
            text=f"Synthetic local source {number} reports a measured method limitation.",
            source_id=f"integration-source-{number}",
            filename=f"local-study-{number}.txt", page=number,
            section="Results", score=0.9, document_id=f"integration-doc-{number}",
            chunk_id=f"integration-chunk-{number}", retrieval_method="synthetic_local",
        )]
        return RetrievalResponse(
            query=query,
            evidence=evidence,
            provider="sequenced_integration_fixture",
        )


def _scenario(*, maximum_iterations=3, responses=None, retrieval_provider=None,
              controller_policy=None, execution_policy_override=None, approval_callback=None):
    plan, review, run, task, state, _prior_iteration, question = capability_fixtures._plan("integration-domain")
    loop = create_research_loop(plan, run, review, state, max_iterations=maximum_iterations)
    start_research_loop(state, loop.loop_id)
    loop = transition_research_loop(state, loop.loop_id, LoopStage.RESEARCH,
                                    reason="Approved integration fixture enters research.")
    iteration = begin_loop_iteration(state, loop.loop_id, (task.task_id,))
    loop = state.research_loops[0]
    retrieval = retrieval_provider or SequencedRetrievalProvider()
    execution_policy = execution_policy_override or BoundedExecutionPolicy(
        maximum_iterations=maximum_iterations,
        maximum_capability_invocations=8,
        maximum_task_executions=8,
        allowed_capabilities=(CapabilityType.LOCAL_RETRIEVAL,),
        allowed_stages=(LoopStage.RESEARCH,),
        allowed_task_ids=(task.task_id,),
        allow_continuation=True,
    )
    llm = QueueDecisionProvider(responses or [
        _proposal_payload(task.task_id, question.question_id, "query one"),
        _proposal_payload(task.task_id, question.question_id, "query two"),
        _proposal_payload(task.task_id, question.question_id, "query three"),
    ])
    policy = controller_policy or AutonomousControllerPolicy(
        maximum_cycles=1, maximum_executions=1, maximum_capability_invocations=1,
        maximum_retrieval_actions=1, maximum_verification_actions=0,
        maximum_total_research_actions=1,
    )

    def request_factory(decision, validation, context):
        retrieval_number = len(retrieval.calls) + 1
        version = register_rag_document_version(
            state,
            source_id=f"integration-source-{retrieval_number}",
            document_id=f"integration-doc-{retrieval_number}",
            filename=f"local-study-{retrieval_number}.txt",
            version_id=f"integration-doc-{retrieval_number}-v1",
        )
        return AutonomousExecutionRequest(
            decision, validation, run.run_id, loop.loop_id, context.iteration_id,
            plan.plan_id, plan.revision, execution_policy_digest(execution_policy),
            capability_request=LocalRetrievalRequest(
                decision.proposal.query, limit=1,
                evidence_document_versions={1: version},
            ),
            provider_refs=provider_bindings_for_action(
                ResearchActionType.RETRIEVE_EVIDENCE, retrieval_provider=retrieval),
        )

    def approve(execution_request):
        if approval_callback is not None:
            return approval_callback(execution_request)
        return AutonomousExecutionApproval.for_request(execution_request,
                                                        reviewer="synthetic integration researcher")

    def controller_request(current_policy=policy):
        current_loop = next(item for item in state.research_loops if item.loop_id == loop.loop_id)
        return AutonomousControllerRequest(
            plan=plan, run=run, review=review, loop=current_loop, state=state,
            execution_policy=execution_policy, policy=current_policy,
            decision_provider=llm, request_factory=request_factory, approval_provider=approve,
            retrieval_provider=retrieval,
        )

    return {
        "plan": plan, "review": review, "run": run, "task": task, "state": state,
        "question": question, "loop_id": loop.loop_id, "iteration": iteration,
        "retrieval": retrieval, "execution_policy": execution_policy, "llm": llm,
        "policy": policy, "request_factory": request_factory, "approve": approve,
        "controller_request": controller_request,
    }


def _parse_context(prompt):
    marker = "Context (read-only):\n"
    return json.loads(prompt.split(marker, 1)[1])


def _run_cycle(scenario, policy=None):
    return run_autonomous_controller(scenario["controller_request"](policy or scenario["policy"]))


class ResearchAutonomousControllerIntegrationTests(unittest.TestCase):
    def test_two_real_cycles_with_explicit_phase7c_continuation(self):
        f = _scenario()
        first = _run_cycle(f)
        self.assertEqual(first.trace.cycles[0].execution.status, AutonomousExecutionStatus.COMPLETED)
        self.assertEqual(first.stop_reason, AutonomousControllerStopReason.MAXIMUM_CYCLES)
        first_execution_id = first.trace.cycles[0].execution.trace.controlled_execution_id
        continuation = apply_research_continuation(
            f["plan"], f["run"], f["review"], f["state"].research_loops[0], f["state"],
            action=ResearchContinuationAction.CONTINUE_SAME_TASK,
            previous_execution_id=first_execution_id,
        )
        second = _run_cycle(f)
        second_cycle = second.trace.cycles[0]
        self.assertEqual(second_cycle.execution.status, AutonomousExecutionStatus.COMPLETED)
        self.assertEqual(second.stop_reason, AutonomousControllerStopReason.MAXIMUM_CYCLES)
        self.assertNotEqual(second_cycle.execution.trace.controlled_execution_id, first_execution_id)
        self.assertEqual(second_cycle.execution.trace.iteration_id, continuation.iteration_id)
        self.assertEqual(second_cycle.execution.dispatch.execution_id,
                         second_cycle.execution.trace.controlled_execution_id)
        self.assertEqual(len(f["state"].task_executions), 2)

    def test_second_real_cycle_observes_cycle_one_evidence(self):
        f = _scenario()
        first = _run_cycle(f)
        first_evidence = tuple(item.evidence_id for item in f["state"].evidence)
        apply_research_continuation(
            f["plan"], f["run"], f["review"], f["state"].research_loops[0], f["state"],
            action=ResearchContinuationAction.CONTINUE_SAME_TASK,
            previous_execution_id=first.trace.cycles[0].execution.trace.controlled_execution_id,
        )
        second = _run_cycle(f)
        seen = _parse_context(f["llm"].prompts[1])
        self.assertEqual(tuple(item["evidence_id"] for item in seen["evidence"]), first_evidence)
        self.assertGreater(len(f["state"].evidence), len(first_evidence))
        self.assertNotEqual(first.trace.cycles[0].context_id, second.trace.cycles[0].context_id)

    def test_real_8a_provider_receives_cycle_one_and_two_contexts(self):
        f = _scenario()
        first = _run_cycle(f)
        apply_research_continuation(
            f["plan"], f["run"], f["review"], f["state"].research_loops[0], f["state"],
            action=ResearchContinuationAction.CONTINUE_SAME_TASK,
            previous_execution_id=first.trace.cycles[0].execution.trace.controlled_execution_id,
        )
        _run_cycle(f)
        self.assertEqual(f["llm"].calls, 2)
        self.assertEqual(len(f["llm"].prompts), 2)

    def test_8a_validation_is_accepted_before_each_real_8b_dispatch(self):
        f = _scenario()
        first = _run_cycle(f)
        apply_research_continuation(
            f["plan"], f["run"], f["review"], f["state"].research_loops[0], f["state"],
            action=ResearchContinuationAction.CONTINUE_SAME_TASK,
            previous_execution_id=first.trace.cycles[0].execution.trace.controlled_execution_id,
        )
        second = _run_cycle(f)
        for result in (first, second):
            cycle = result.trace.cycles[0]
            self.assertEqual(cycle.validation.validation_status.value, "ACCEPTED")
            self.assertEqual(cycle.validation.decision_id, cycle.decision.trace.decision_id)
            self.assertEqual(cycle.execution.trace.decision_id, cycle.decision.trace.decision_id)

    def test_8b_trace_contains_phase7_controlled_execution_id(self):
        f = _scenario()
        result = _run_cycle(f)
        trace = result.trace.cycles[0].execution.trace
        self.assertIsNotNone(trace.dispatch_id)
        self.assertIsNotNone(trace.controlled_execution_id)
        self.assertTrue(any(x.execution_id == trace.controlled_execution_id
                            for x in f["state"].task_executions))

    def test_caller_approval_is_bound_to_exact_first_request(self):
        f = _scenario()
        result = _run_cycle(f)
        approval_id = result.trace.cycles[0].approval_id
        execution_event = next(event for event in f["state"].events
                               if event.agent == "autonomous_execution"
                               and event.action == "proposal_execution_trace")
        trace_data = json.loads(execution_event.message)
        self.assertEqual(trace_data["approval_id"], approval_id)
        self.assertEqual(trace_data["decision_id"], result.trace.cycles[0].decision.trace.decision_id)

    def test_missing_researcher_approval_waits_before_8b(self):
        f = _scenario(approval_callback=lambda _request: None)
        result = _run_cycle(f)
        self.assertEqual(result.stop_reason, AutonomousControllerStopReason.WAIT_FOR_RESEARCHER)
        self.assertIsNone(result.trace.cycles[0].execution)
        self.assertEqual(f["state"].task_executions, [])
        self.assertEqual(f["retrieval"].calls, [])

    def test_missing_phase7c_continuation_authorization_waits(self):
        f = _scenario()
        first = _run_cycle(f)
        before = len(f["state"].task_executions)
        second = _run_cycle(f)
        cycle = second.trace.cycles[0]
        self.assertEqual(second.stop_reason, AutonomousControllerStopReason.WAIT_FOR_RESEARCHER)
        self.assertEqual(cycle.status, "WAITING_FOR_RESEARCHER")
        self.assertIn("Phase 7C", cycle.reason)
        self.assertIsNone(cycle.execution)
        self.assertEqual(len(f["state"].task_executions), before)
        self.assertEqual(f["retrieval"].calls, [("query one", 1)])

    def test_phase7c_authorization_is_attached_to_second_8b_request(self):
        f = _scenario()
        first = _run_cycle(f)
        continuation = apply_research_continuation(
            f["plan"], f["run"], f["review"], f["state"].research_loops[0], f["state"],
            action=ResearchContinuationAction.CONTINUE_SAME_TASK,
            previous_execution_id=first.trace.cycles[0].execution.trace.controlled_execution_id,
        )
        second = _run_cycle(f)
        self.assertEqual(f["state"].task_executions[-1].continuation_authorization_id,
                         continuation.continuation_authorization_id)

    def test_phase7c_authorization_is_not_created_by_controller(self):
        f = _scenario()
        result = _run_cycle(f)
        self.assertEqual(f["state"].research_loops[0].continuation_authorizations, ())
        self.assertEqual(result.stop_reason, AutonomousControllerStopReason.MAXIMUM_CYCLES)

    def test_deterministic_execution_failure_is_traced_without_retry(self):
        provider = SequencedRetrievalProvider(fail=True)
        f = _scenario(retrieval_provider=provider)
        result = _run_cycle(f)
        cycle = result.trace.cycles[0]
        self.assertEqual(result.stop_reason, AutonomousControllerStopReason.EXECUTION_FAILED)
        self.assertEqual(cycle.execution.status, AutonomousExecutionStatus.FAILED)
        self.assertIn("Phase 5 execution failed", cycle.execution.trace.reason)
        self.assertEqual(len(f["state"].task_executions), 1)
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(f["llm"].calls, 1)

    def test_failure_leaves_phase7_and_8b_trace_events(self):
        f = _scenario(retrieval_provider=SequencedRetrievalProvider(fail=True))
        result = _run_cycle(f)
        actions = [(event.agent, event.action) for event in f["state"].events]
        self.assertIn(("autonomous_execution", "proposal_execution_trace"), actions)
        self.assertIn(("ResearchLoop", "capability_result_recorded"), actions)
        self.assertEqual(result.trace.cycles[0].execution.trace.status, AutonomousExecutionStatus.FAILED)

    def test_maximum_cycle_policy_stops_after_one_real_execution(self):
        f = _scenario()
        result = _run_cycle(f)
        self.assertEqual(result.stop_reason, AutonomousControllerStopReason.MAXIMUM_CYCLES)
        self.assertEqual(result.cycles_completed, 1)
        self.assertEqual(result.executions_completed, 1)
        self.assertEqual(f["llm"].calls, 1)

    def test_zero_execution_policy_prevents_request_and_capability_execution(self):
        f = _scenario()
        no_exec = AutonomousControllerPolicy(1, 0, 1, 1, 0, 1)
        result = _run_cycle(f, no_exec)
        self.assertEqual(result.stop_reason, AutonomousControllerStopReason.MAXIMUM_EXECUTIONS)
        self.assertIsNone(result.trace.cycles[0].execution)
        self.assertEqual(f["retrieval"].calls, [])

    def test_registered_loop_iteration_bound_is_honored_by_8a(self):
        f = _scenario(maximum_iterations=1)
        result = _run_cycle(f)
        self.assertEqual(result.stop_reason, AutonomousControllerStopReason.DECISION_STOP)
        self.assertEqual(result.executions_completed, 0)
        self.assertEqual(result.trace.cycles[0].decision.proposal.action_type, ResearchActionType.STOP)
        self.assertEqual(f["retrieval"].calls, [])

    def test_cycle_trace_links_controller_decision_validation_and_execution(self):
        f = _scenario()
        result = _run_cycle(f)
        trace = result.trace
        cycle = trace.cycles[0]
        self.assertEqual(trace.run_id, f["run"].run_id)
        self.assertEqual(trace.loop_id, f["loop_id"])
        self.assertEqual(cycle.execution.trace.request_id, cycle.request_id)
        self.assertEqual(cycle.execution.trace.decision_id, cycle.decision.trace.decision_id)
        self.assertEqual(cycle.execution.trace.validation.status.value, "ACCEPTED")

    def test_execution_artifact_and_evidence_trace_resolve_in_state(self):
        f = _scenario()
        result = _run_cycle(f)
        execution_trace = result.trace.cycles[0].execution.trace
        self.assertTrue(execution_trace.artifact_ids)
        self.assertTrue(execution_trace.evidence_ids)
        self.assertTrue(set(execution_trace.artifact_ids).issubset(
            {item.artifact_id for item in f["state"].research_artifacts}))
        self.assertTrue(set(execution_trace.evidence_ids).issubset(
            {item.evidence_id for item in f["state"].evidence}))

    def test_retrieval_does_not_promote_evidence_to_a_verified_claim(self):
        f = _scenario()
        _run_cycle(f)
        self.assertEqual(len(f["state"].evidence), 1)
        self.assertEqual(f["state"].claims, [])
        self.assertEqual(f["state"].research_artifacts[0].artifact_type, "retrieval_result")

    def test_claim_epistemic_status_is_preserved_in_next_8a_context(self):
        f = _scenario()
        first = _run_cycle(f)
        evidence = f["state"].evidence[0]
        f["state"].add_claim(ResearchClaim(
            text="Synthetic claim not yet verified.", status="NOT_VALIDATED",
            evidence_ids=[evidence.evidence_id], source_ids=[evidence.source_id],
            verification_reason="Integration fixture status preservation."))
        apply_research_continuation(
            f["plan"], f["run"], f["review"], f["state"].research_loops[0], f["state"],
            action=ResearchContinuationAction.CONTINUE_SAME_TASK,
            previous_execution_id=first.trace.cycles[0].execution.trace.controlled_execution_id,
        )
        _run_cycle(f)
        context = _parse_context(f["llm"].prompts[1])
        self.assertEqual(context["claims"][0]["status"], "NOT_VALIDATED")
        self.assertEqual(f["state"].claims[0].status, "NOT_VALIDATED")

    def test_state_records_real_dispatch_request_result_and_execution_events(self):
        f = _scenario()
        _run_cycle(f)
        event_actions = {(event.agent, event.action) for event in f["state"].events}
        self.assertIn(("ResearchLoop", "capability_request_recorded"), event_actions)
        self.assertIn(("ResearchLoop", "capability_result_recorded"), event_actions)
        self.assertIn(("autonomous_execution", "proposal_execution_trace"), event_actions)
        self.assertTrue(f["state"].task_executions)

    def test_state_lineage_remains_valid_after_two_cycles_and_phase7c(self):
        f = _scenario()
        first = _run_cycle(f)
        apply_research_continuation(
            f["plan"], f["run"], f["review"], f["state"].research_loops[0], f["state"],
            action=ResearchContinuationAction.CONTINUE_SAME_TASK,
            previous_execution_id=first.trace.cycles[0].execution.trace.controlled_execution_id,
        )
        _run_cycle(f)
        self.assertEqual(f["state"].validate_lineage(), [])
        self.assertEqual(f["state"].research_loops[0].validate_references(f["state"]), [])

    def test_malformed_8a_json_fails_typed_without_execution(self):
        f = _scenario(responses=["{not-json"])
        result = _run_cycle(f)
        self.assertEqual(result.stop_reason, AutonomousControllerStopReason.ERROR)
        self.assertIn("ResearchDecisionGenerationError", result.failure_reason)
        self.assertEqual(f["retrieval"].calls, [])
        self.assertEqual(f["state"].task_executions, [])

    def test_invalid_8b_request_callback_fails_closed(self):
        f = _scenario()
        f["request_factory"] = lambda *_args: object()
        original_request = f["controller_request"]
        def request_with_bad_factory(current_policy=None):
            base = original_request(current_policy or f["policy"])
            return AutonomousControllerRequest(
                plan=base.plan, run=base.run, review=base.review, loop=base.loop, state=base.state,
                execution_policy=base.execution_policy, policy=base.policy,
                decision_provider=base.decision_provider, request_factory=f["request_factory"],
                approval_provider=base.approval_provider, retrieval_provider=base.retrieval_provider)
        f["controller_request"] = request_with_bad_factory
        result = _run_cycle(f)
        self.assertEqual(result.stop_reason, AutonomousControllerStopReason.ERROR)
        self.assertIn("AutonomousExecutionRequest", result.failure_reason)
        self.assertEqual(f["retrieval"].calls, [])

    def test_mismatched_researcher_approval_is_rejected_by_real_8b(self):
        def wrong_approval(request):
            approval = AutonomousExecutionApproval.for_request(request, reviewer="test researcher")
            return AutonomousExecutionApproval(
                approval_id=approval.approval_id, decision_id="another-decision",
                run_id=approval.run_id, loop_id=approval.loop_id,
                iteration_id=approval.iteration_id, plan_id=approval.plan_id,
                plan_revision=approval.plan_revision, action_type=approval.action_type,
                task_id=approval.task_id, scope_digest=approval.scope_digest,
                reviewer=approval.reviewer, status=ApprovalStatus.APPROVED,
                created_at=approval.created_at)
        f = _scenario(approval_callback=wrong_approval)
        result = _run_cycle(f)
        self.assertEqual(result.stop_reason, AutonomousControllerStopReason.EXECUTION_REJECTED)
        self.assertEqual(result.trace.cycles[0].execution.status, AutonomousExecutionStatus.INVALID)
        self.assertEqual(f["retrieval"].calls, [])

    def test_request_manifest_requires_rebinding_and_has_no_supported_deserializer(self):
        f = _scenario()
        request = f["controller_request"]()
        manifest = json.loads(request.to_json())
        self.assertTrue(manifest["runtime_bindings_required"])
        self.assertNotIn("request_factory", manifest)
        self.assertFalse(hasattr(AutonomousControllerRequest, "from_json"))

    def test_final_stop_reason_is_typed_and_serializable(self):
        f = _scenario()
        result = _run_cycle(f)
        serialized = json.loads(result.to_json())
        self.assertEqual(serialized["stop_reason"], result.stop_reason.value)
        self.assertEqual(result.stop_reason, AutonomousControllerStopReason.MAXIMUM_CYCLES)

    def test_phase7_loop_remains_running_after_capability_success_until_researcher_action(self):
        f = _scenario()
        _run_cycle(f)
        self.assertEqual(f["state"].research_loops[0].status, ResearchLoopStatus.RUNNING)
        self.assertIn(f["task"].task_id, f["state"].research_loops[0].unresolved_task_ids)
        self.assertNotIn(f["task"].task_id, f["state"].research_loops[0].completed_task_ids)


if __name__ == "__main__":
    unittest.main()
