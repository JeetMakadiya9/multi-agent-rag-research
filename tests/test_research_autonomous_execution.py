from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
for path in (str(ROOT), str(TESTS)):
    if path not in sys.path:
        sys.path.insert(0, path)

import test_research_autonomy as autonomy_fixtures
import test_research_capability_dispatch as dispatch_fixtures
import test_research_verification_capability as verification_fixtures
import test_research_improvement_capability as improvement_fixtures
import test_research_prior_work_capability as prior_work_fixtures
import test_research_experiment_planning_capability as experiment_fixtures
import test_research_synthesis_critique as synthesis_fixtures
from research_autonomy import (
    ResearchActionProposal, ResearchActionType, ResearchDecision, ResearcherOverride,
    ResearcherOverrideAction, ValidationStatus, validate_research_decision,
)
from research_autonomous_execution import (
    ApprovalStatus, AutonomousExecutionApproval, AutonomousExecutionRequest,
    AutonomousExecutionStatus, AutonomousExecutionTrace, execute_autonomous_proposal,
    execution_policy_digest, provider_bindings_for_action,
)
from research_capabilities import CapabilityType, LocalRetrievalRequest
from research_retrieval import StaticRetrievalProvider


def _capability_case(action):
    """Build a real Phase 6 fixture, active loop, 8A decision, and scoped 8B approval."""
    from research_autonomy import build_autonomous_research_context
    from research_bounded_loop import BoundedExecutionPolicy
    from research_loop import LoopStage
    from research_prior_work_capability import PriorWorkInvestigationRequest
    from research_state import PriorWorkSearchScope
    from research_verification_capability import EvidenceVerificationRequest

    providers = {}
    synthesis = None
    question_ids = ()
    claim_ids = evidence_ids = artifact_ids = ()
    candidate_id = problem_statement = query = None
    capability_request = None

    if action == "VERIFY_CLAIM":
        f = verification_fixtures._fixture()
        f = {"plan": f["plan"], "review": f["review"], "run": f["run"],
             "task": f["task"], "state": f["state"]}
        stage = LoopStage.VERIFICATION
        from research_state import ResearchClaim
        stored = f["state"].add_claim(ResearchClaim(
            text="System X uses BM25.", status="SUPPORTED", claim_id="claim-8b-focused",
            evidence_ids=[f["state"].evidence[0].evidence_id],
            source_ids=[f["state"].evidence[0].source_id],
            verification_reason="Synthetic stored claim for bridge testing."))
        claim_ids = (stored.claim_id,)
        evidence_ids = tuple(stored.evidence_ids)
        capability_request = EvidenceVerificationRequest(stored.text, evidence_ids)
        providers["llm_provider"] = verification_fixtures.QueueLLM(verification_fixtures._stance("SUPPORTS"))
    elif action == "GENERATE_IMPROVEMENT":
        f = improvement_fixtures._fixture()
        stage = LoopStage.IMPROVEMENT
        request = improvement_fixtures._request(f)
        capability_request = request
        problem_statement = request.problem_statement
        evidence_ids = request.evidence_ids
        claim_ids = request.verified_claim_ids
        response = improvement_fixtures._response(
            improvement_fixtures._candidate(evidence_id=f["evidence"].evidence_id,
                                            claim_id=f["claim"].claim_id))
        providers["llm_provider"] = improvement_fixtures.CountingLLM(response)
    elif action == "INVESTIGATE_PRIOR_WORK":
        f = prior_work_fixtures._fixture()
        f["task"] = f["tasks"][2]
        stage = LoopStage.PRIOR_WORK
        candidate_id = f["candidate"].candidate_id
        artifact_ids = (f["candidate_artifact"].artifact_id,)
        question_ids = (f["state_question"].question_id,)
        capability_request = PriorWorkInvestigationRequest(
            f["candidate_artifact"].artifact_id, candidate_id,
            PriorWorkSearchScope(maximum_queries=1, maximum_results=2,
                                 source_types=("paper",), provider_name="static"),
            queries=("documented limitation",), question_id=f["state_question"].question_id)

        class AssessingLLM:
            def generate(self, messages, *, temperature=0.0, timeout=120):
                import json
                payload = json.loads(messages[1]["content"])
                return prior_work_fixtures._assessments(
                    [item["search_result_id"] for item in payload["results"]])

        providers["llm_provider"] = AssessingLLM()
        providers["search_provider"] = f["provider"]
    elif action == "PLAN_EXPERIMENT":
        f = experiment_fixtures._fixture()
        stage = LoopStage.EXPERIMENT_PLANNING
        request = experiment_fixtures._request(f)
        capability_request = request
        candidate_id = request.candidate_id
        artifact_ids = (request.improvement_artifact_id,)
        providers["llm_provider"] = experiment_fixtures.QueueLLM(experiment_fixtures._plan_response())
    elif action == "CRITIQUE":
        f = synthesis_fixtures._analysis_fixture()
        stage = LoopStage.CRITIQUE
        synthesis = synthesis_fixtures._synthesize(f)
        loop = f["state"].research_loops[-1]
        from research_loop import transition_research_loop
        loop = transition_research_loop(f["state"], loop.loop_id, LoopStage.CRITIQUE,
                                        reason="Caller enters critique for this focused bridge test.")
        f["loop"] = loop
        f["loop_iteration"] = loop.loop_iterations[-1]
        # Synthesis is tied to the previous stage value only by stage at creation;
        # its loop and iteration identities remain the same for critique.
        f["loop"] = loop
    else:
        raise AssertionError(f"Unsupported test action {action}")

    if action != "CRITIQUE":
        f["loop"] = dispatch_fixtures._advance(f, stage)
    plan, run, review, state = f["plan"], f["run"], f["review"], f["state"]
    loop = f.get("loop") or state.research_loops[-1]
    iteration_id = f.get("loop_iteration_id") or f["loop_iteration"].iteration_id
    if not question_ids and plan.questions:
        question_ids = (plan.questions[0].question_id,)
    policy = BoundedExecutionPolicy(
        maximum_iterations=loop.max_iterations, maximum_capability_invocations=20, maximum_task_executions=20,
        allowed_capabilities=tuple(CapabilityType), allowed_stages=tuple(LoopStage),
        allowed_task_ids=(f["task"].task_id,), allow_continuation=True,
    )
    proposal = ResearchActionProposal(
        ResearchActionType(action), f"Explicit focused bridge proposal for {action}.",
        task_id=f["task"].task_id if action in {"VERIFY_CLAIM", "GENERATE_IMPROVEMENT",
                                                "INVESTIGATE_PRIOR_WORK", "PLAN_EXPERIMENT"} else None,
        question_ids=question_ids, claim_ids=claim_ids, evidence_ids=evidence_ids,
        artifact_ids=artifact_ids, problem_statement=problem_statement,
        candidate_id=candidate_id, query=query,
    )
    context = build_autonomous_research_context(plan, run, review, loop, state, policy)
    decision = autonomy_fixtures._decision_for(context, proposal)
    validation = validate_research_decision(decision, plan=plan, run=run, review=review,
                                            loop=loop, state=state, policy=policy)
    request = AutonomousExecutionRequest(
        decision, validation, run.run_id, loop.loop_id, iteration_id,
        plan.plan_id, plan.revision, execution_policy_digest(policy),
        capability_request=capability_request, synthesis=synthesis,
        provider_refs=provider_bindings_for_action(ResearchActionType(action), **providers),
    )
    approval = AutonomousExecutionApproval.for_request(request, reviewer="researcher")
    return plan, run, review, loop, state, policy, request, approval, providers


def _authorized_request(f, action="RETRIEVE_EVIDENCE", *, override=None, retrieval_provider=None):
    context = autonomy_fixtures._get_context(f)
    proposal = ResearchActionProposal(
        action_type=ResearchActionType(action),
        rationale="This explicit next action addresses the unresolved research question.",
        task_id=f[3].task_id if action == "RETRIEVE_EVIDENCE" else None,
        question_ids=(f[6].question_id,),
        query="synthetic retrieval query" if action == "RETRIEVE_EVIDENCE" else None,
    )
    decision = autonomy_fixtures._decision_for(context, proposal)
    validation = validate_research_decision(
        decision, plan=f[0], run=f[2], review=f[1], loop=f[7], state=f[4], policy=f[8],
        researcher_override=override,
    )
    request = AutonomousExecutionRequest(
        decision, validation, f[2].run_id, f[7].loop_id, f[7].current_iteration_id,
        f[0].plan_id, f[0].revision, execution_policy_digest(f[8]),
        capability_request=LocalRetrievalRequest("synthetic retrieval query") if action == "RETRIEVE_EVIDENCE" else None,
        researcher_override=override,
        provider_refs=provider_bindings_for_action(
            ResearchActionType(action),
            retrieval_provider=(retrieval_provider or StaticRetrievalProvider([])) if action == "RETRIEVE_EVIDENCE" else None,
        ),
    )
    approval = AutonomousExecutionApproval.for_request(request, reviewer="researcher")
    return request, approval


class ResearchAutonomousExecutionTests(unittest.TestCase):
    def test_accepted_proposal_dispatches_exactly_one_existing_capability(self):
        f = autonomy_fixtures._fixture()
        request, approval = _authorized_request(f)
        provider = StaticRetrievalProvider(autonomy_fixtures.capability_fixtures._items())
        result = execute_autonomous_proposal(f[0], f[2], f[1], f[7], f[4], f[8], request, approval,
                                             retrieval_provider=provider)
        self.assertEqual(result.status, AutonomousExecutionStatus.COMPLETED)
        self.assertEqual(result.dispatch.capability, CapabilityType.LOCAL_RETRIEVAL)
        self.assertEqual(len(f[4].task_executions), 1)
        self.assertEqual(result.trace.controlled_execution_id, result.dispatch.execution_id)
        self.assertEqual(result.trace.artifact_ids, result.dispatch.artifact_ids)

    def test_request_and_trace_round_trip(self):
        f = autonomy_fixtures._fixture()
        request, approval = _authorized_request(f)
        self.assertEqual(AutonomousExecutionRequest.from_json(request.to_json()), request)
        result = execute_autonomous_proposal(f[0], f[2], f[1], f[7], f[4], f[8], request, approval,
                                            retrieval_provider=StaticRetrievalProvider([]))
        self.assertEqual(AutonomousExecutionTrace.from_json(result.trace.to_json()), result.trace)
        from research_autonomous_execution import AutonomousExecutionResult
        self.assertEqual(AutonomousExecutionResult.from_json(result.to_json()), result)
        self.assertEqual(result.status, AutonomousExecutionStatus.COMPLETED)

    def test_wrong_scope_approval_fails_without_mutating_state(self):
        f = autonomy_fixtures._fixture()
        request, approval = _authorized_request(f)
        state_before = f[4].to_dict()
        stale = AutonomousExecutionApproval(
            approval.approval_id, approval.decision_id, approval.run_id, approval.loop_id,
            approval.iteration_id, approval.plan_id, approval.plan_revision + 1,
            approval.action_type, approval.task_id, approval.scope_digest, approval.reviewer,
            approval.status, approval.created_at,
        )
        result = execute_autonomous_proposal(f[0], f[2], f[1], f[7], f[4], f[8], request, stale,
                                             retrieval_provider=StaticRetrievalProvider([]))
        self.assertEqual(result.status, AutonomousExecutionStatus.INVALID)
        self.assertEqual(f[4].to_dict(), state_before)

    def test_stale_iteration_and_policy_are_rejected_without_mutation(self):
        f = autonomy_fixtures._fixture()
        request, approval = _authorized_request(f)
        before = f[4].to_dict()
        stale_iteration = AutonomousExecutionRequest(
            request.decision, request.prior_validation, request.run_id, request.loop_id,
            "iteration_stale", request.plan_id, request.plan_revision, request.policy_digest,
            request.capability_request,
        )
        stale_approval = AutonomousExecutionApproval.for_request(stale_iteration, reviewer="researcher")
        result = execute_autonomous_proposal(f[0], f[2], f[1], f[7], f[4], f[8], stale_iteration,
                                             stale_approval, retrieval_provider=StaticRetrievalProvider([]))
        self.assertEqual(result.status, AutonomousExecutionStatus.INVALID)
        changed_policy = type(f[8])(
            f[8].maximum_iterations, f[8].maximum_capability_invocations + 1,
            f[8].maximum_task_executions, f[8].allowed_capabilities, f[8].allowed_stages,
            f[8].allowed_task_ids, f[8].allow_continuation, f[8].stop_on_failure,
            f[8].stop_on_unresolved_blocker,
        )
        result = execute_autonomous_proposal(f[0], f[2], f[1], f[7], f[4], changed_policy, request, approval,
                                             retrieval_provider=StaticRetrievalProvider([]))
        self.assertEqual(result.status, AutonomousExecutionStatus.INVALID)
        self.assertEqual(f[4].to_dict(), before)

    def test_stale_plan_run_and_terminal_loop_reject_before_execution(self):
        f = autonomy_fixtures._fixture()
        request, approval = _authorized_request(f)
        before = f[4].to_dict()
        wrong_run = replace(f[2], run_id="wrong-run")
        result = execute_autonomous_proposal(f[0], wrong_run, f[1], f[7], f[4], f[8], request, approval,
                                             retrieval_provider=StaticRetrievalProvider([]))
        self.assertEqual(result.status, AutonomousExecutionStatus.INVALID)
        f[0].revision += 1
        result = execute_autonomous_proposal(f[0], f[2], f[1], f[7], f[4], f[8], request, approval,
                                             retrieval_provider=StaticRetrievalProvider([]))
        self.assertEqual(result.status, AutonomousExecutionStatus.INVALID)
        f[0].revision -= 1
        from research_loop import LoopStage, StopReason, TransitionAuthority, transition_research_loop
        transition_research_loop(f[4], f[7].loop_id, LoopStage.STOPPED, reason="test stop",
                                 authority=TransitionAuthority.RESEARCHER,
                                 stop_reason=StopReason.RESEARCHER_REQUESTED_STOP)
        terminal = f[4].research_loops[0]
        result = execute_autonomous_proposal(f[0], f[2], f[1], terminal, f[4], f[8], request, approval,
                                             retrieval_provider=StaticRetrievalProvider([]))
        self.assertEqual(result.status, AutonomousExecutionStatus.INVALID)
        self.assertEqual(f[4].task_executions, [])
        self.assertNotEqual(f[4].to_dict(), before)  # only this test's explicit terminal transition mutated it

    def test_explicit_researcher_acceptance_is_respected(self):
        f = autonomy_fixtures._fixture()
        request, _ = _authorized_request(f)
        accept = ResearcherOverride(request.decision.trace.decision_id, ResearcherOverrideAction.ACCEPT,
                                    "researcher", "Approved this proposal.")
        accepted = AutonomousExecutionRequest(
            request.decision, request.prior_validation, request.run_id, request.loop_id,
            request.iteration_id, request.plan_id, request.plan_revision, request.policy_digest,
            request.capability_request, researcher_override=accept, provider_refs=request.provider_refs,
        )
        approval = AutonomousExecutionApproval.for_request(accepted, reviewer="researcher")
        result = execute_autonomous_proposal(f[0], f[2], f[1], f[7], f[4], f[8], accepted, approval,
                                             retrieval_provider=StaticRetrievalProvider([]))
        self.assertEqual(result.status, AutonomousExecutionStatus.COMPLETED)
        self.assertEqual(len(f[4].task_executions), 1)

    def test_controlled_failure_is_recorded_and_not_retried(self):
        f = autonomy_fixtures._fixture()
        class BrokenRetrieval:
            name = "broken-synthetic"
            def retrieve(self, query, limit=8):
                raise RuntimeError("synthetic provider failure")
        provider = BrokenRetrieval()
        request, approval = _authorized_request(f, retrieval_provider=provider)
        result = execute_autonomous_proposal(f[0], f[2], f[1], f[7], f[4], f[8], request, approval,
                                             retrieval_provider=provider)
        self.assertEqual(result.status, AutonomousExecutionStatus.FAILED)
        self.assertIsNotNone(result.trace.controlled_execution_id)
        attempts = len(f[4].task_executions)
        duplicate = execute_autonomous_proposal(f[0], f[2], f[1], f[7], f[4], f[8], request, approval,
                                               retrieval_provider=provider)
        self.assertEqual(duplicate.status, AutonomousExecutionStatus.ALREADY_EXECUTED)
        self.assertEqual(len(f[4].task_executions), attempts)

    def test_provider_change_after_approval_is_rejected(self):
        f = autonomy_fixtures._fixture()
        request, approval = _authorized_request(f)
        class DifferentProvider:
            name = "different_retrieval"
            def retrieve(self, query, limit=8):
                return StaticRetrievalProvider([]).retrieve(query, limit)
        before = f[4].to_dict()
        result = execute_autonomous_proposal(f[0], f[2], f[1], f[7], f[4], f[8], request, approval,
                                             retrieval_provider=DifferentProvider())
        self.assertEqual(result.status, AutonomousExecutionStatus.INVALID)
        self.assertEqual(f[4].to_dict(), before)

    def test_researcher_approval_round_trip(self):
        request, approval = _authorized_request(autonomy_fixtures._fixture())
        self.assertEqual(AutonomousExecutionApproval.from_json(approval.to_json()), approval)

    def test_missing_approval_fails_closed(self):
        f = autonomy_fixtures._fixture()
        request, approval = _authorized_request(f)
        revoked = AutonomousExecutionApproval(
            approval.approval_id, approval.decision_id, approval.run_id, approval.loop_id,
            approval.iteration_id, approval.plan_id, approval.plan_revision, approval.action_type,
            approval.task_id, approval.scope_digest, approval.reviewer, ApprovalStatus.REVOKED,
            approval.created_at,
        )
        before = f[4].to_dict()
        result = execute_autonomous_proposal(f[0], f[2], f[1], f[7], f[4], f[8], request, revoked,
                                             retrieval_provider=StaticRetrievalProvider([]))
        self.assertEqual(result.status, AutonomousExecutionStatus.INVALID)
        self.assertEqual(f[4].to_dict(), before)

    def test_absent_approval_returns_structured_rejection(self):
        f = autonomy_fixtures._fixture()
        request, _ = _authorized_request(f)
        before = f[4].to_dict()
        result = execute_autonomous_proposal(f[0], f[2], f[1], f[7], f[4], f[8], request,
                                             retrieval_provider=StaticRetrievalProvider([]))
        self.assertEqual(result.status, AutonomousExecutionStatus.INVALID)
        self.assertIsNone(result.trace.approval_id)
        self.assertEqual(f[4].to_dict(), before)

    def test_rejected_or_superseded_authorization_does_not_execute(self):
        for auth_status, result_status in ((ApprovalStatus.REJECTED, AutonomousExecutionStatus.REJECTED),
                                           (ApprovalStatus.SUPERSEDED, AutonomousExecutionStatus.SUPERSEDED)):
            f = autonomy_fixtures._fixture()
            request, approved = _authorized_request(f)
            approval = AutonomousExecutionApproval(
                approved.approval_id, approved.decision_id, approved.run_id, approved.loop_id,
                approved.iteration_id, approved.plan_id, approved.plan_revision, approved.action_type,
                approved.task_id, approved.scope_digest, approved.reviewer, auth_status, approved.created_at,
            )
            result = execute_autonomous_proposal(f[0], f[2], f[1], f[7], f[4], f[8], request, approval,
                                                 retrieval_provider=StaticRetrievalProvider([]))
            self.assertEqual(result.status, result_status)
            self.assertEqual(f[4].task_executions, [])

    def test_duplicate_proposal_is_idempotently_reported(self):
        f = autonomy_fixtures._fixture()
        request, approval = _authorized_request(f)
        provider = StaticRetrievalProvider([])
        first = execute_autonomous_proposal(f[0], f[2], f[1], f[7], f[4], f[8], request, approval,
                                            retrieval_provider=provider)
        count = len(f[4].task_executions)
        second = execute_autonomous_proposal(f[0], f[2], f[1], f[7], f[4], f[8], request, approval,
                                             retrieval_provider=provider)
        self.assertEqual(second.status, AutonomousExecutionStatus.ALREADY_EXECUTED)
        self.assertEqual(len(f[4].task_executions), count)

    def test_completed_task_requires_phase7c_continuation_then_executes_new_record(self):
        from research_continuation import ResearchContinuationAction, apply_research_continuation
        from research_loop import select_loop_task

        f = autonomy_fixtures._fixture()
        first_request, first_approval = _authorized_request(f)
        first = execute_autonomous_proposal(f[0], f[2], f[1], f[7], f[4], f[8], first_request,
                                            first_approval, retrieval_provider=StaticRetrievalProvider([]))
        self.assertEqual(first.status, AutonomousExecutionStatus.COMPLETED)
        continuation = apply_research_continuation(
            f[0], f[2], f[1], f[4].research_loops[0], f[4],
            action=ResearchContinuationAction.CONTINUE_SAME_TASK,
            previous_execution_id=first.trace.controlled_execution_id,
        )
        loop = select_loop_task(f[4], continuation.loop_id, f[3].task_id)
        context = autonomy_fixtures.build_autonomous_research_context(f[0], f[2], f[1], loop, f[4], f[8])
        proposal = ResearchActionProposal(ResearchActionType.RETRIEVE_EVIDENCE,
            "Continue the same authorized task after explicit caller authorization.",
            task_id=f[3].task_id, question_ids=(f[6].question_id,), query="synthetic retrieval query")
        decision = autonomy_fixtures._decision_for(context, proposal)
        decision = ResearchDecision(decision.proposal, replace(decision.trace, decision_id="decision_continuation"))
        validation = validate_research_decision(decision, plan=f[0], run=f[2], review=f[1], loop=loop,
                                                state=f[4], policy=f[8])
        no_continue = AutonomousExecutionRequest(
            decision, validation, f[2].run_id, loop.loop_id, loop.current_iteration_id,
            f[0].plan_id, f[0].revision, execution_policy_digest(f[8]),
            capability_request=LocalRetrievalRequest("synthetic retrieval query"),
            provider_refs=provider_bindings_for_action(ResearchActionType.RETRIEVE_EVIDENCE,
                                                       retrieval_provider=StaticRetrievalProvider([])),
        )
        no_continue_approval = AutonomousExecutionApproval.for_request(no_continue, reviewer="researcher")
        before = len(f[4].task_executions)
        blocked = execute_autonomous_proposal(f[0], f[2], f[1], loop, f[4], f[8], no_continue,
                                              no_continue_approval,
                                              retrieval_provider=StaticRetrievalProvider([]))
        self.assertEqual(blocked.status, AutonomousExecutionStatus.CONTINUATION_REQUIRED)
        self.assertEqual(len(f[4].task_executions), before)
        continued = AutonomousExecutionRequest(
            decision, validation, f[2].run_id, loop.loop_id, loop.current_iteration_id,
            f[0].plan_id, f[0].revision, execution_policy_digest(f[8]),
            capability_request=LocalRetrievalRequest("synthetic retrieval query"),
            continuation_authorization_id=continuation.continuation_authorization_id,
            provider_refs=provider_bindings_for_action(ResearchActionType.RETRIEVE_EVIDENCE,
                                                       retrieval_provider=StaticRetrievalProvider([])),
        )
        approved = AutonomousExecutionApproval.for_request(continued, reviewer="researcher")
        second = execute_autonomous_proposal(f[0], f[2], f[1], loop, f[4], f[8], continued, approved,
                                             retrieval_provider=StaticRetrievalProvider([]))
        self.assertEqual(second.status, AutonomousExecutionStatus.COMPLETED)
        self.assertNotEqual(first.trace.controlled_execution_id, second.trace.controlled_execution_id)
        self.assertEqual(len(f[4].task_executions), before + 1)

    def test_modify_does_not_mutate_or_execute_original_proposal(self):
        f = autonomy_fixtures._fixture()
        original, approval = _authorized_request(f)
        modified_proposal = ResearchActionProposal(
            ResearchActionType.RETRIEVE_EVIDENCE, "Modified query", task_id=f[3].task_id,
            question_ids=(f[6].question_id,), query="different query",
        )
        override = ResearcherOverride(original.decision.trace.decision_id, ResearcherOverrideAction.MODIFY,
                                      "researcher", "Use a narrower query", modified_proposal)
        self.assertEqual(original.decision.proposal.query, "synthetic retrieval query")
        # A MODIFY override is checked when attached to the request and requires a new decision.
        changed = AutonomousExecutionRequest(
            original.decision, original.prior_validation, original.run_id, original.loop_id,
            original.iteration_id, original.plan_id, original.plan_revision, original.policy_digest,
            original.capability_request, researcher_override=override, provider_refs=original.provider_refs,
        )
        changed_approval = AutonomousExecutionApproval.for_request(changed, reviewer="researcher")
        rejected = execute_autonomous_proposal(f[0], f[2], f[1], f[7], f[4], f[8], changed, changed_approval,
                                               retrieval_provider=StaticRetrievalProvider([]))
        self.assertEqual(rejected.status, AutonomousExecutionStatus.MODIFICATION_REQUIRED)
        self.assertEqual(original.decision.proposal.query, "synthetic retrieval query")

    def test_no_direct_execution_for_researcher_reject(self):
        f = autonomy_fixtures._fixture()
        override = ResearcherOverride("will-replace", ResearcherOverrideAction.REJECT, "researcher", "No")
        request, approval = _authorized_request(f)
        override = ResearcherOverride(request.decision.trace.decision_id, ResearcherOverrideAction.REJECT,
                                      "researcher", "Do not execute")
        rejected_request = AutonomousExecutionRequest(
            request.decision, request.prior_validation, request.run_id, request.loop_id,
            request.iteration_id, request.plan_id, request.plan_revision, request.policy_digest,
            request.capability_request, None, override,
        )
        rejected_approval = AutonomousExecutionApproval.for_request(rejected_request, reviewer="researcher")
        result = execute_autonomous_proposal(f[0], f[2], f[1], f[7], f[4], f[8], rejected_request,
                                             rejected_approval, retrieval_provider=StaticRetrievalProvider([]))
        self.assertEqual(result.status, AutonomousExecutionStatus.REJECTED)
        self.assertEqual(f[4].task_executions, [])

    def test_synthesis_routes_to_phase7d_without_capability_execution(self):
        f = autonomy_fixtures._fixture()
        # Advance the actual controlled loop to the synthesis stage.
        from research_loop import LoopStage, TransitionAuthority, transition_research_loop
        loop = f[7]
        for stage in (LoopStage.EVIDENCE_REVIEW, LoopStage.VERIFICATION, LoopStage.SYNTHESIS):
            loop = transition_research_loop(f[4], loop.loop_id, stage,
                                            reason="test stage", authority=TransitionAuthority.SYSTEM)
        f = (*f[:7], loop, f[8])
        context = autonomy_fixtures._get_context(f)
        proposal = ResearchActionProposal(ResearchActionType.SYNTHESIZE, "Summarize stored evidence.")
        decision = autonomy_fixtures._decision_for(context, proposal)
        validation = validate_research_decision(decision, plan=f[0], run=f[2], review=f[1], loop=f[7], state=f[4], policy=f[8])
        request = AutonomousExecutionRequest(decision, validation, f[2].run_id, loop.loop_id,
                                             loop.current_iteration_id, f[0].plan_id, f[0].revision,
                                             execution_policy_digest(f[8]))
        approval = AutonomousExecutionApproval.for_request(request, reviewer="researcher")
        result = execute_autonomous_proposal(f[0], f[2], f[1], loop, f[4], f[8], request, approval)
        self.assertEqual(result.status, AutonomousExecutionStatus.COMPLETED)
        self.assertIsNotNone(result.synthesis)
        self.assertEqual(f[4].task_executions, [])

    def test_verify_claim_routes_to_existing_phase6_verifier(self):
        from research_capability_dispatch import DispatchStatus
        case = _capability_case("VERIFY_CLAIM")
        result = execute_autonomous_proposal(*case[:6], case[6], case[7], **case[8])
        self.assertEqual(result.status, AutonomousExecutionStatus.COMPLETED)
        self.assertEqual(result.dispatch.status, DispatchStatus.COMPLETED)
        self.assertEqual(result.dispatch.capability, CapabilityType.EVIDENCE_VERIFICATION)
        self.assertEqual(result.trace.artifact_ids, result.dispatch.artifact_ids)
        self.assertEqual(result.trace.evidence_ids, result.dispatch.evidence_ids)

    def test_improvement_routes_to_phase6_and_keeps_candidate_epistemic_status(self):
        case = _capability_case("GENERATE_IMPROVEMENT")
        result = execute_autonomous_proposal(*case[:6], case[6], case[7], **case[8])
        self.assertEqual(result.status, AutonomousExecutionStatus.COMPLETED)
        artifact = next(x for x in case[4].research_artifacts if x.artifact_id in result.trace.artifact_ids)
        self.assertEqual(artifact.artifact_type, "improvement_candidate")
        self.assertTrue(artifact.improvement_candidates)
        self.assertTrue(all(x.status == "CANDIDATE" for x in artifact.improvement_candidates))
        self.assertFalse(any("novel" in x.status.lower() or "validated" in x.status.lower()
                             for x in artifact.improvement_candidates))

    def test_prior_work_routes_to_existing_phase6_api_and_preserves_search_ids(self):
        case = _capability_case("INVESTIGATE_PRIOR_WORK")
        result = execute_autonomous_proposal(*case[:6], case[6], case[7], **case[8])
        self.assertEqual(result.status, AutonomousExecutionStatus.COMPLETED)
        self.assertEqual(result.dispatch.capability, CapabilityType.PRIOR_WORK_INVESTIGATION)
        self.assertTrue(result.trace.artifact_ids)
        self.assertTrue(result.dispatch.search_action_ids)
        self.assertEqual(result.trace.artifact_ids, result.dispatch.artifact_ids)

    def test_experiment_planning_routes_to_existing_api_and_remains_review_gated(self):
        case = _capability_case("PLAN_EXPERIMENT")
        result = execute_autonomous_proposal(*case[:6], case[6], case[7], **case[8])
        self.assertEqual(result.status, AutonomousExecutionStatus.COMPLETED)
        artifact = next(x for x in case[4].research_artifacts if x.artifact_id in result.trace.artifact_ids)
        self.assertEqual(artifact.artifact_type, "experiment_plan")
        self.assertEqual(artifact.experiment_plan.status, "REQUIRES_RESEARCHER_REVIEW")
        self.assertEqual(case[4].task_executions[-1].status.value, "COMPLETED")

    def test_critique_routes_to_phase7d_without_dispatching_a_phase5_task(self):
        case = _capability_case("CRITIQUE")
        execution_count = len(case[4].task_executions)
        result = execute_autonomous_proposal(*case[:6], case[6], case[7], **case[8])
        self.assertEqual(result.status, AutonomousExecutionStatus.COMPLETED)
        self.assertIsNotNone(result.critique)
        self.assertIsNone(result.dispatch)
        self.assertEqual(len(case[4].task_executions), execution_count)

    def test_approval_bound_to_another_decision_is_rejected_before_dispatch(self):
        from dataclasses import replace
        f = autonomy_fixtures._fixture()
        request, approval = _authorized_request(f)
        wrong = replace(approval, decision_id="different-decision")
        before = f[4].to_dict()
        result = execute_autonomous_proposal(f[0], f[2], f[1], f[7], f[4], f[8], request, wrong,
                                             retrieval_provider=StaticRetrievalProvider([]))
        self.assertEqual(result.status, AutonomousExecutionStatus.INVALID)
        self.assertEqual(f[4].to_dict(), before)
        self.assertEqual(f[4].task_executions, [])

    def test_approval_bound_to_another_run_is_rejected(self):
        case = _capability_case("VERIFY_CLAIM")
        wrong = replace(case[7], run_id="other-run")
        before = case[4].to_dict()
        execution_count = len(case[4].task_executions)
        result = execute_autonomous_proposal(*case[:6], case[6], wrong, **case[8])
        self.assertEqual(result.status, AutonomousExecutionStatus.INVALID)
        self.assertEqual(case[4].to_dict(), before)
        self.assertEqual(len(case[4].task_executions), execution_count)

    def test_approval_bound_to_another_loop_is_rejected(self):
        case = _capability_case("VERIFY_CLAIM")
        execution_count = len(case[4].task_executions)
        result = execute_autonomous_proposal(*case[:6], case[6], replace(case[7], loop_id="other-loop"), **case[8])
        self.assertEqual(result.status, AutonomousExecutionStatus.INVALID)
        self.assertEqual(len(case[4].task_executions), execution_count)

    def test_approval_bound_to_another_iteration_is_rejected(self):
        case = _capability_case("VERIFY_CLAIM")
        execution_count = len(case[4].task_executions)
        result = execute_autonomous_proposal(*case[:6], case[6], replace(case[7], iteration_id="other-iteration"), **case[8])
        self.assertEqual(result.status, AutonomousExecutionStatus.INVALID)
        self.assertEqual(len(case[4].task_executions), execution_count)

    def test_approval_bound_to_another_plan_is_rejected(self):
        case = _capability_case("VERIFY_CLAIM")
        execution_count = len(case[4].task_executions)
        result = execute_autonomous_proposal(*case[:6], case[6], replace(case[7], plan_id="other-plan"), **case[8])
        self.assertEqual(result.status, AutonomousExecutionStatus.INVALID)
        self.assertEqual(len(case[4].task_executions), execution_count)

    def test_unregistered_run_rejected_before_execution(self):
        case = _capability_case("VERIFY_CLAIM")
        execution_count = len(case[4].task_executions)
        case[4].research_runs.clear()
        result = execute_autonomous_proposal(*case[:6], case[6], case[7], **case[8])
        self.assertEqual(result.status, AutonomousExecutionStatus.INVALID)
        self.assertEqual(len(case[4].task_executions), execution_count)

    def test_terminal_run_rejected_before_execution(self):
        case = _capability_case("VERIFY_CLAIM")
        execution_count = len(case[4].task_executions)
        case[1].status = "completed"
        result = execute_autonomous_proposal(*case[:6], case[6], case[7], **case[8])
        self.assertEqual(result.status, AutonomousExecutionStatus.INVALID)
        self.assertEqual(len(case[4].task_executions), execution_count)

    def test_stage_change_after_proposal_rejects_stale_action(self):
        from research_loop import LoopStage, transition_research_loop
        f = autonomy_fixtures._fixture()
        request, approval = _authorized_request(f)
        transition_research_loop(f[4], f[7].loop_id, LoopStage.EVIDENCE_REVIEW,
                                 reason="Caller moved the stage after proposal validation.")
        before = len(f[4].task_executions)
        result = execute_autonomous_proposal(f[0], f[2], f[1], f[4].research_loops[0], f[4], f[8],
                                             request, approval, retrieval_provider=StaticRetrievalProvider([]))
        self.assertEqual(result.status, AutonomousExecutionStatus.INVALID)
        self.assertEqual(len(f[4].task_executions), before)

    def test_authorized_task_removed_after_proposal_is_rejected(self):
        f = autonomy_fixtures._fixture()
        request, approval = _authorized_request(f)
        f[2].authorized_task_ids.clear()
        before = len(f[4].task_executions)
        result = execute_autonomous_proposal(f[0], f[2], f[1], f[7], f[4], f[8], request, approval,
                                             retrieval_provider=StaticRetrievalProvider([]))
        self.assertEqual(result.status, AutonomousExecutionStatus.INVALID)
        self.assertEqual(len(f[4].task_executions), before)

    def test_capability_removed_from_policy_after_approval_is_rejected(self):
        f = autonomy_fixtures._fixture()
        request, approval = _authorized_request(f)
        from dataclasses import replace
        policy = replace(f[8], allowed_capabilities=())
        result = execute_autonomous_proposal(f[0], f[2], f[1], f[7], f[4], policy, request, approval,
                                             retrieval_provider=StaticRetrievalProvider([]))
        self.assertEqual(result.status, AutonomousExecutionStatus.INVALID)
        self.assertEqual(f[4].task_executions, [])

    def test_stored_evidence_removed_after_proposal_is_rejected(self):
        case = _capability_case("VERIFY_CLAIM")
        execution_count = len(case[4].task_executions)
        case[4].evidence.clear()
        result = execute_autonomous_proposal(*case[:6], case[6], case[7], **case[8])
        self.assertEqual(result.status, AutonomousExecutionStatus.INVALID)
        self.assertEqual(len(case[4].task_executions), execution_count)

    def test_identical_current_state_validation_is_deterministic(self):
        case = _capability_case("VERIFY_CLAIM")
        from research_autonomous_execution import _validate_action_request
        first = validate_research_decision(case[6].decision, plan=case[0], run=case[1], review=case[2],
                                           loop=case[3], state=case[4], policy=case[5])
        second = validate_research_decision(case[6].decision, plan=case[0], run=case[1], review=case[2],
                                            loop=case[3], state=case[4], policy=case[5])
        self.assertEqual(first.validation_status, ValidationStatus.ACCEPTED)
        self.assertEqual(second.validation_status, first.validation_status)
        self.assertEqual(second.rejection_reasons, first.rejection_reasons)
        self.assertEqual(first.context_id, second.context_id)

    def test_success_trace_is_persisted_in_existing_event_log(self):
        f = autonomy_fixtures._fixture()
        request, approval = _authorized_request(f)
        result = execute_autonomous_proposal(f[0], f[2], f[1], f[7], f[4], f[8], request, approval,
                                             retrieval_provider=StaticRetrievalProvider([]))
        events = [x for x in f[4].events if x.agent == "autonomous_execution"]
        self.assertEqual(result.status, AutonomousExecutionStatus.COMPLETED)
        self.assertEqual(len(events), 1)
        self.assertEqual(AutonomousExecutionTrace.from_json(events[0].message), result.trace)

    def test_one_invocation_records_one_execution_and_does_not_select_next_task(self):
        f = autonomy_fixtures._fixture()
        request, approval = _authorized_request(f)
        result = execute_autonomous_proposal(f[0], f[2], f[1], f[7], f[4], f[8], request, approval,
                                             retrieval_provider=StaticRetrievalProvider([]))
        self.assertEqual(result.status, AutonomousExecutionStatus.COMPLETED)
        self.assertEqual(len(f[4].task_executions), 1)
        self.assertEqual(len([x for x in f[4].events if x.agent == "autonomous_execution"]), 1)
        self.assertEqual(f[4].research_loops[0].current_task_id, f[3].task_id)

    def test_failed_execution_requires_explicit_retry_reference(self):
        class Broken:
            name = "test-broken"
            def retrieve(self, query, limit=8):
                raise RuntimeError("test provider failure")
        f = autonomy_fixtures._fixture()
        provider = Broken()
        request, approval = _authorized_request(f, retrieval_provider=provider)
        result = execute_autonomous_proposal(f[0], f[2], f[1], f[7], f[4], f[8], request, approval,
                                             retrieval_provider=provider)
        self.assertEqual(result.status, AutonomousExecutionStatus.FAILED)
        before = len(f[4].task_executions)
        retry_needed = execute_autonomous_proposal(f[0], f[2], f[1], f[7], f[4], f[8], request, approval,
                                                   retrieval_provider=provider)
        self.assertEqual(retry_needed.status, AutonomousExecutionStatus.ALREADY_EXECUTED)
        self.assertEqual(len(f[4].task_executions), before)

    def test_malformed_free_form_action_is_rejected_by_typed_proposal_parser(self):
        payload = autonomy_fixtures._payload(action="RUN_SHELL")
        with self.assertRaises(Exception):
            ResearchActionProposal.from_dict(__import__("json").loads(payload))

    def test_rejected_proposal_does_not_add_execution_or_execution_trace_event(self):
        f = autonomy_fixtures._fixture()
        request, approval = _authorized_request(f)
        f[2].authorized_task_ids.clear()
        before = f[4].to_dict()
        result = execute_autonomous_proposal(f[0], f[2], f[1], f[7], f[4], f[8], request, approval,
                                             retrieval_provider=StaticRetrievalProvider([]))
        self.assertEqual(result.status, AutonomousExecutionStatus.INVALID)
        self.assertEqual(f[4].to_dict(), before)

    def test_typed_request_cannot_pair_retrieval_proposal_with_verification_request(self):
        f = autonomy_fixtures._fixture()
        request, _ = _authorized_request(f)
        from research_verification_capability import EvidenceVerificationRequest
        with self.assertRaisesRegex(ValueError, "requires a typed"):
            replace(request, capability_request=EvidenceVerificationRequest("claim", ("evidence",)))

    def test_experiment_and_candidate_statuses_are_not_promoted_by_bridge(self):
        case = _capability_case("PLAN_EXPERIMENT")
        result = execute_autonomous_proposal(*case[:6], case[6], case[7], **case[8])
        artifact = next(x for x in case[4].research_artifacts if x.artifact_id in result.trace.artifact_ids)
        plan = artifact.experiment_plan
        self.assertEqual(plan.status, "REQUIRES_RESEARCHER_REVIEW")
        self.assertTrue(any("does not establish" in item.lower() or "not establish" in item.lower()
                            for item in plan.limitations))


if __name__ == "__main__":
    unittest.main()
