from __future__ import annotations

import json
import sys
import unittest
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import test_research_capabilities as capability_fixtures
import test_research_improvement_capability as improvement_fixtures
from llm import StaticProvider
from research_autonomy import (
    AutonomousResearchContext,
    ResearchActionProposal,
    ResearchActionType,
    ResearchDecision,
    ResearchDecisionGenerationError,
    ResearchStopAssessment,
    ResearcherOverride,
    ResearcherOverrideAction,
    ValidationStatus,
    apply_researcher_override,
    autonomous_decide,
    build_autonomous_research_context,
    validate_research_decision,
)
from research_bounded_loop import BoundedExecutionPolicy
from research_capabilities import CapabilityType
from research_capabilities import LocalRetrievalRequest, execute_local_retrieval
from research_loop import (
    LoopStage,
    ResearchLoopStatus,
    TransitionAuthority,
    begin_loop_iteration,
    create_research_loop,
    select_loop_task,
    start_research_loop,
    transition_research_loop,
)
from research_state import ResearchClaim
from research_retrieval import StaticRetrievalProvider, register_rag_document_version


class CountingProvider(StaticProvider):
    def __init__(self, response):
        super().__init__(response)
        self.calls = []

    def generate(self, messages, *, temperature=0.0, timeout=120):
        self.calls.append((messages, temperature, timeout))
        return super().generate(messages, temperature=temperature, timeout=timeout)


def _payload(action="RETRIEVE_EVIDENCE", **overrides):
    value = {
        "action_type": action,
        "rationale": "The unresolved question needs source-backed evidence.",
        "task_id": "task-retrieve" if action == "RETRIEVE_EVIDENCE" else None,
        "question_ids": ["plan-question"],
        "claim_ids": [],
        "evidence_ids": [],
        "artifact_ids": [],
        "query": "synthetic local query" if action == "RETRIEVE_EVIDENCE" else None,
        "problem_statement": None,
        "candidate_id": None,
        "estimated_information_gain": 0.7,
        "priority": "high",
        "unresolved_questions_addressed": [],
        "stop_after_action": False,
        "stop_assessment": None,
        "provenance_ids": [],
    }
    value.update(overrides)
    return json.dumps(value)


def _fixture(*, stages=None, capabilities=None, maximum_invocations=4, maximum_executions=4):
    plan, review, run, task, state, iteration, question = capability_fixtures._plan("domain-neutral")
    loop = create_research_loop(plan, run, review, state, max_iterations=3)
    start_research_loop(state, loop.loop_id)
    transition_research_loop(state, loop.loop_id, LoopStage.RESEARCH,
                             reason="Test context enters research.", authority=TransitionAuthority.SYSTEM)
    begin_loop_iteration(state, loop.loop_id, (task.task_id,))
    loop = select_loop_task(state, loop.loop_id, task.task_id)
    policy = BoundedExecutionPolicy(
        maximum_iterations=3,
        maximum_capability_invocations=maximum_invocations,
        maximum_task_executions=maximum_executions,
        allowed_capabilities=tuple(capabilities if capabilities is not None else CapabilityType),
        allowed_stages=tuple(stages if stages is not None else LoopStage),
        allowed_task_ids=(task.task_id,),
    )
    return plan, review, run, task, state, iteration, question, loop, policy


def _get_context(f):
    plan, review, run, _, state, _, _, loop, policy = f
    return build_autonomous_research_context(plan, run, review, loop, state, policy)


def _decide(f, response=None):
    context = _get_context(f)
    return context, autonomous_decide(context, CountingProvider(response or _payload()))


class ResearchAutonomyTests(unittest.TestCase):
    def validate(self, f, decision, override=None):
        plan, review, run, _, state, _, _, loop, policy = f
        return validate_research_decision(decision, plan=plan, run=run, review=review,
            loop=loop, state=state, policy=policy, researcher_override=override)

    def test_valid_typed_proposal_is_accepted_without_execution(self):
        f = _fixture()
        before = f[4].to_dict()
        context, decision = _decide(f)
        trace = self.validate(f, decision)
        self.assertEqual(trace.validation_status, ValidationStatus.ACCEPTED)
        self.assertEqual(decision.proposal.action_type, ResearchActionType.RETRIEVE_EVIDENCE)
        self.assertEqual(decision.proposal.required_capabilities, (CapabilityType.LOCAL_RETRIEVAL,))
        self.assertIn("explicitly_authorized_task", decision.proposal.required_authorizations)
        self.assertEqual(before, f[4].to_dict())
        self.assertEqual(f[4].task_executions, [])
        self.assertEqual(f[4].research_artifacts, [])

    def test_provider_receives_only_serialized_read_only_context_and_is_explicit(self):
        f = _fixture()
        context = _get_context(f)
        provider = CountingProvider(_payload())
        autonomous_decide(context, provider)
        self.assertEqual(len(provider.calls), 1)
        messages, temperature, _ = provider.calls[0]
        self.assertEqual(temperature, 0.0)
        self.assertIn(context.context_id, messages[1]["content"])
        self.assertNotIn("ResearchState(", messages[1]["content"])
        with self.assertRaisesRegex(ValueError, "LLMProvider"):
            autonomous_decide(context, object())

    def test_context_and_decision_json_round_trip(self):
        f = _fixture()
        context, decision = _decide(f)
        self.assertEqual(AutonomousResearchContext.from_json(context.to_json()), context)
        self.assertEqual(ResearchDecision.from_json(decision.to_json()), decision)

    def test_malformed_json_is_rejected(self):
        with self.assertRaises(ResearchDecisionGenerationError):
            _decide(_fixture(), "{not-json")

    def test_unknown_action_is_rejected(self):
        with self.assertRaises(ResearchDecisionGenerationError):
            _decide(_fixture(), _payload(action="CALL_TOOL"))

    def test_missing_action_field_is_rejected(self):
        payload = json.loads(_payload())
        del payload["action_type"]
        with self.assertRaises(ResearchDecisionGenerationError):
            _decide(_fixture(), json.dumps(payload))

    def test_invalid_task_id_is_rejected_by_validator(self):
        f = _fixture()
        _, decision = _decide(f)
        proposal = ResearchActionProposal.from_dict({**decision.proposal.to_dict(), "task_id": "foreign-task"})
        invalid = ResearchDecision(proposal, decision.trace)
        trace = self.validate(f, invalid)
        self.assertEqual(trace.validation_status, ValidationStatus.REJECTED)
        self.assertTrue(any("task" in reason for reason in trace.rejection_reasons))

    def test_unauthorized_task_is_rejected(self):
        f = _fixture()
        _, decision = _decide(f)
        bad = ResearchActionProposal.from_dict({**decision.proposal.to_dict(), "task_id": "other-task"})
        trace = self.validate(f, ResearchDecision(bad, decision.trace))
        self.assertEqual(trace.validation_status, ValidationStatus.REJECTED)

    def test_unauthorized_capability_is_rejected(self):
        f = _fixture(capabilities=())
        _, decision = _decide(f)
        trace = self.validate(f, decision)
        self.assertEqual(trace.validation_status, ValidationStatus.REJECTED)
        self.assertTrue(any("allowlist" in x for x in trace.rejection_reasons))

    def test_wrong_run_reference_is_rejected(self):
        f = _fixture()
        _, decision = _decide(f)
        trace = decision.trace
        wrong = type(trace)(trace.decision_id, trace.context_id, trace.loop_id, "run_other",
            trace.iteration_id, trace.action_type, trace.rationale, trace.provider, trace.model,
            trace.created_at, trace.provenance_ids)
        rejected = self.validate(f, ResearchDecision(decision.proposal, wrong))
        self.assertEqual(rejected.validation_status, ValidationStatus.REJECTED)

    def test_wrong_loop_reference_is_rejected(self):
        f = _fixture()
        _, decision = _decide(f)
        trace = decision.trace
        wrong = type(trace)(trace.decision_id, trace.context_id, "loop_other", trace.run_id,
            trace.iteration_id, trace.action_type, trace.rationale, trace.provider, trace.model,
            trace.created_at, trace.provenance_ids)
        self.assertEqual(self.validate(f, ResearchDecision(decision.proposal, wrong)).validation_status,
                         ValidationStatus.REJECTED)

    def test_wrong_plan_revision_context_fails_closed(self):
        f = _fixture()
        _, decision = _decide(f)
        f[0].revision += 1
        trace = self.validate(f, decision)
        self.assertEqual(trace.validation_status, ValidationStatus.REJECTED)

    def test_invalid_claim_reference_is_rejected(self):
        f = _fixture()
        payload = _payload("VERIFY_CLAIM", task_id="task-retrieve", query=None,
                           claim_ids=["missing-claim"], evidence_ids=["missing-evidence"])
        _, decision = _decide(f, payload)
        self.assertEqual(self.validate(f, decision).validation_status, ValidationStatus.REJECTED)

    def test_invalid_evidence_reference_is_rejected(self):
        f = _fixture()
        payload = _payload("GENERATE_IMPROVEMENT", task_id="task-retrieve", query=None,
                           problem_statement="A documented limitation", evidence_ids=["fake-evidence"])
        _, decision = _decide(f, payload)
        self.assertEqual(self.validate(f, decision).validation_status, ValidationStatus.REJECTED)

    def test_wrong_stage_proposal_is_rejected(self):
        f = _fixture()
        context = _get_context(f)
        synthesis = ResearchActionProposal(ResearchActionType.SYNTHESIZE, "Synthesize stored findings.")
        decision = _decision_for(context, synthesis)
        self.assertEqual(self.validate(f, decision).validation_status, ValidationStatus.REJECTED)

    def test_terminal_loop_rejects_proposal(self):
        f = _fixture()
        transition_research_loop(f[4], f[7].loop_id, LoopStage.STOPPED,
            reason="test terminal", authority=TransitionAuthority.RESEARCHER,
            stop_reason=__import__("research_loop").StopReason.RESEARCHER_REQUESTED_STOP)
        f = (*f[:7], f[4].research_loops[0], f[8])
        context = _get_context(f)
        decision = _decision_for(context, ResearchActionProposal(ResearchActionType.STOP, "Stop." ,
            stop_assessment=ResearchStopAssessment("researcher_requested_stop", "Researcher stopped.")))
        self.assertEqual(self.validate(f, decision).validation_status, ValidationStatus.REJECTED)

    def test_budget_exhaustion_overrides_llm_and_makes_no_provider_call(self):
        f = _fixture(maximum_invocations=0)
        context = _get_context(f)
        provider = CountingProvider(_payload())
        decision = autonomous_decide(context, provider)
        self.assertEqual(provider.calls, [])
        self.assertEqual(decision.proposal.action_type, ResearchActionType.STOP)
        self.assertEqual(decision.proposal.stop_assessment.stop_reason,
                         "maximum_capability_invocations_reached")
        self.assertEqual(self.validate(f, decision).validation_status, ValidationStatus.ACCEPTED)

    def test_request_researcher_review_is_a_typed_proposal_not_approval(self):
        f = _fixture()
        context = _get_context(f)
        proposal = ResearchActionProposal(ResearchActionType.REQUEST_RESEARCHER_REVIEW,
                                          "A human decision is needed.")
        decision = _decision_for(context, proposal)
        events_before = len(f[4].events)
        self.assertEqual(self.validate(f, decision).validation_status, ValidationStatus.ACCEPTED)
        self.assertEqual(f[1].status.value, "APPROVED")
        self.assertEqual(len(f[4].events), events_before)  # proposal itself records no new event

    def test_return_to_planning_requires_researcher_acceptance(self):
        f = _fixture()
        transition_research_loop(f[4], f[7].loop_id, LoopStage.RESEARCHER_REVIEW,
                                 reason="Human gate.")
        f = (*f[:7], f[4].research_loops[0], f[8])
        context = _get_context(f)
        proposal = ResearchActionProposal(ResearchActionType.RETURN_TO_PLANNING, "Researcher chose replanning.")
        decision = _decision_for(context, proposal)
        self.assertEqual(self.validate(f, decision).validation_status, ValidationStatus.REJECTED)
        override = ResearcherOverride(decision.trace.decision_id, ResearcherOverrideAction.ACCEPT,
                                      "researcher", "Explicitly return to planning.")
        result = apply_researcher_override(decision, override)
        self.assertTrue(result.executable_after_validation)
        self.assertEqual(self.validate(f, decision, override).validation_status, ValidationStatus.ACCEPTED)

    def test_valid_stop_is_accepted_and_does_not_change_loop(self):
        f = _fixture()
        context = _get_context(f)
        proposal = ResearchActionProposal(ResearchActionType.STOP, "No useful next action is identified.",
            estimated_information_gain=0.1,
            stop_assessment=ResearchStopAssessment("evidence_sufficient", "Available evidence was reviewed."))
        decision = _decision_for(context, proposal)
        before = f[4].to_dict()
        self.assertEqual(self.validate(f, decision).validation_status, ValidationStatus.ACCEPTED)
        self.assertEqual(f[4].to_dict(), before)

    def test_trace_serialization_preserves_provenance_and_validation(self):
        f = _fixture()
        _, decision = _decide(f)
        trace = self.validate(f, decision)
        restored = type(trace).from_json(trace.to_json())
        self.assertEqual(restored, trace)
        self.assertEqual(restored.validation_status, ValidationStatus.ACCEPTED)

    def test_stale_snapshot_is_rejected_after_state_changes(self):
        f = _fixture()
        _, decision = _decide(f)
        f[4].unresolved_questions.append("A new unresolved question.")
        trace = self.validate(f, decision)
        self.assertEqual(trace.validation_status, ValidationStatus.REJECTED)
        self.assertTrue(any("stale" in reason for reason in trace.rejection_reasons))

    def test_deterministic_validator_returns_same_result(self):
        f = _fixture()
        _, decision = _decide(f)
        self.assertEqual(self.validate(f, decision), self.validate(f, decision))

    def test_researcher_accept_reject_modify_and_stop_are_explicit(self):
        f = _fixture()
        _, decision = _decide(f)
        accept = ResearcherOverride(decision.trace.decision_id, "ACCEPT", "researcher", "Proceed to validation.")
        reject = ResearcherOverride(decision.trace.decision_id, "REJECT", "researcher", "Do not proceed.")
        modified_proposal = ResearchActionProposal(ResearchActionType.STOP, "Researcher changed course.",
            stop_assessment=ResearchStopAssessment("researcher_requested_stop", "Human request."))
        modify = ResearcherOverride(decision.trace.decision_id, "MODIFY", "researcher", "Change proposed action.", modified_proposal)
        stop = ResearcherOverride(decision.trace.decision_id, "STOP", "researcher", "Stop this research.")
        self.assertTrue(apply_researcher_override(decision, accept).executable_after_validation)
        self.assertFalse(apply_researcher_override(decision, reject).executable_after_validation)
        self.assertEqual(apply_researcher_override(decision, modify).proposal, modified_proposal)
        self.assertFalse(apply_researcher_override(decision, stop).executable_after_validation)

    def test_rejected_proposal_does_not_mutate_state(self):
        f = _fixture()
        before = deepcopy(f[4].to_dict())
        _, decision = _decide(f, _payload().replace('task-retrieve', 'not-authorized'))
        trace = self.validate(f, decision)
        self.assertEqual(trace.validation_status, ValidationStatus.REJECTED)
        self.assertEqual(before, f[4].to_dict())

    def test_context_preserves_epistemic_categories_for_candidate_artifacts(self):
        fixture = improvement_fixtures._fixture("climate science")
        candidate_data = improvement_fixtures._candidate(
            fixture["evidence"].evidence_id, fixture["claim"].claim_id)
        generated = improvement_fixtures._execute(
            fixture, improvement_fixtures.CountingLLM(improvement_fixtures._response(candidate_data)))
        loop = create_research_loop(fixture["plan"], fixture["run"], fixture["review"],
                                    fixture["state"], max_iterations=3)
        start_research_loop(fixture["state"], loop.loop_id)
        transition_research_loop(fixture["state"], loop.loop_id, LoopStage.RESEARCH,
                                 reason="Snapshot stored candidate context.", authority=TransitionAuthority.SYSTEM)
        begin_loop_iteration(fixture["state"], loop.loop_id, (fixture["task"].task_id,))
        loop = select_loop_task(fixture["state"], loop.loop_id, fixture["task"].task_id)
        policy = BoundedExecutionPolicy(3, 4, 4, tuple(CapabilityType), tuple(LoopStage),
                                        tuple(fixture["run"].authorized_task_ids))
        context = build_autonomous_research_context(fixture["plan"], fixture["run"], fixture["review"],
                                                     loop, fixture["state"], policy)
        artifact = next(x for x in context.artifacts if x.artifact_id == generated.artifact_id)
        self.assertEqual(artifact.candidate_ids, (generated.candidates[0].candidate_id,))
        self.assertIn("status=CANDIDATE", artifact.epistemic_notes[0])
        self.assertIn("novelty=NOT_ASSESSED", artifact.epistemic_notes[0])
        self.assertIn("validation=NOT_VALIDATED", artifact.epistemic_notes[0])
        self.assertNotIn("VERIFIED", artifact.epistemic_notes[0])

    def test_read_only_context_retains_stored_evidence_and_provenance(self):
        f = _fixture()
        provider = StaticRetrievalProvider(capability_fixtures._items())
        version = register_rag_document_version(f[4], source_id="source-one", document_id="doc-one",
                                                filename="paper-one.pdf", version_id="version-one")
        execute_local_retrieval(f[0], f[2], f[3], f[1],
            LocalRetrievalRequest("synthetic query", limit=1, question_id=f[6].question_id,
                                  evidence_document_versions={1: version}),
            provider, f[4], iteration_id=f[5].iteration_id)
        evidence = f[4].evidence[0]
        context = _get_context(f)
        snapshot = next(item for item in context.evidence if item.evidence_id == evidence.evidence_id)
        self.assertEqual(snapshot.text, evidence.text)
        self.assertEqual(snapshot.source_id, evidence.source_id)
        self.assertEqual(snapshot.document_version_id, evidence.document_version_id)
        self.assertEqual(snapshot.passage_reference_id, evidence.passage_reference_id)
        self.assertEqual(snapshot.search_result_id, evidence.search_result_id)
        self.assertTrue(set(snapshot.provenance_ids).issuperset({
            evidence.evidence_id, evidence.source_id, evidence.search_result_id,
            evidence.document_version_id, evidence.passage_reference_id,
        }))
        self.assertEqual(f[4].validate_lineage(), [])

    def test_context_is_domain_neutral(self):
        f = _fixture()
        context = _get_context(f)
        self.assertEqual(context.research_domain, "domain-neutral")
        self.assertTrue(context.objective)

    def test_no_tool_or_capability_calls_are_made_by_decide_or_validate(self):
        f = _fixture()
        before = (len(f[4].task_executions), len(f[4].research_artifacts), len(f[4].research_loops[0].capability_requests))
        _, decision = _decide(f)
        self.validate(f, decision)
        after = (len(f[4].task_executions), len(f[4].research_artifacts), len(f[4].research_loops[0].capability_requests))
        self.assertEqual(before, after)

    def test_provenance_references_must_exist_in_snapshot(self):
        f = _fixture()
        context = _get_context(f)
        proposal = ResearchActionProposal(ResearchActionType.RETRIEVE_EVIDENCE, "Need evidence.",
            task_id="task-retrieve", query="query", question_ids=("plan-question",), provenance_ids=("fake-source",))
        decision = _decision_for(context, proposal)
        self.assertEqual(self.validate(f, decision).validation_status, ValidationStatus.REJECTED)

    def test_required_plan_review_and_run_authorization_are_rechecked(self):
        f = _fixture()
        _, decision = _decide(f)
        f[2].approval_review_id = "another-review"
        self.assertEqual(self.validate(f, decision).validation_status, ValidationStatus.REJECTED)

    def test_claim_must_be_linked_to_stored_run_evidence(self):
        f = _fixture()
        context = _get_context(f)
        proposal = ResearchActionProposal(ResearchActionType.VERIFY_CLAIM, "Verify stored claim.",
            task_id="task-retrieve", claim_ids=("claim-fake",), evidence_ids=("evidence-fake",))
        decision = _decision_for(context, proposal)
        self.assertEqual(self.validate(f, decision).validation_status, ValidationStatus.REJECTED)

    def test_action_properties_are_not_llm_controlled(self):
        f = _fixture()
        _, decision = _decide(f)
        self.assertEqual(decision.proposal.required_authorizations[:2],
                         ("approved_exact_plan_revision", "registered_research_run"))
        with self.assertRaises(TypeError):
            decision.proposal.required_capabilities[0] = CapabilityType.EXPERIMENT_PLANNING


def _decision_for(context, proposal):
    from research_autonomy import ResearchDecisionTrace
    trace = ResearchDecisionTrace("decision_test", context.context_id, context.loop_id, context.run_id,
        context.iteration_id, proposal.action_type, proposal.rationale, "test", None,
        "2026-01-01T00:00:00+00:00", proposal.provenance_ids)
    return ResearchDecision(proposal, trace)


if __name__ == "__main__":
    unittest.main()
