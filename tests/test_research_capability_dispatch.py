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

import test_research_capabilities as local_tests
import test_research_verification_capability as verification_tests
import test_research_improvement_capability as improvement_tests
import test_research_prior_work_capability as prior_tests
import test_research_experiment_planning_capability as experiment_tests

from research_capabilities import CapabilityType, LocalRetrievalRequest
from research_capability_dispatch import (
    CapabilityDispatchError,
    DispatchStatus,
    ResearchCapabilityDispatchResult,
    dispatch_research_capability,
)
from research_loop import (
    LoopStage,
    ResearchLoopStatus,
    StopReason,
    TransitionAuthority,
    begin_loop_iteration,
    create_research_loop,
    start_research_loop,
    transition_research_loop,
)
from research_plan_review import PlanReview, ReviewStatus
from research_planning import ResearchObjective, ResearchPlan, ResearchPlanQuestion, ResearchRequest, ResearchTask
from research_retrieval import StaticRetrievalProvider
from research_run_authorization import authorize_research_run
from research_state import ResearchState
from research_task_authorization import authorize_research_tasks
from research_tools import SearchResult, StaticSearchProvider
from research_state import PriorWorkSearchScope
from llm import StaticProvider


def _advance(fixture, target: LoopStage):
    state = fixture["state"]
    loop = fixture.get("loop")
    if loop is None:
        loop = create_research_loop(
            fixture["plan"], fixture["run"], fixture["review"], state, max_iterations=10,
        )
        fixture["loop"] = loop
    path = {
        LoopStage.RESEARCH: (LoopStage.RESEARCH,),
        LoopStage.EVIDENCE_REVIEW: (LoopStage.RESEARCH, LoopStage.EVIDENCE_REVIEW),
        LoopStage.VERIFICATION: (LoopStage.RESEARCH, LoopStage.EVIDENCE_REVIEW, LoopStage.VERIFICATION),
        LoopStage.IMPROVEMENT: (LoopStage.RESEARCH, LoopStage.EVIDENCE_REVIEW, LoopStage.VERIFICATION,
                                LoopStage.SYNTHESIS, LoopStage.CRITIQUE, LoopStage.IMPROVEMENT),
        LoopStage.PRIOR_WORK: (LoopStage.RESEARCH, LoopStage.EVIDENCE_REVIEW, LoopStage.VERIFICATION,
                               LoopStage.SYNTHESIS, LoopStage.CRITIQUE, LoopStage.IMPROVEMENT,
                               LoopStage.PRIOR_WORK),
        LoopStage.EXPERIMENT_PLANNING: (LoopStage.RESEARCH, LoopStage.EVIDENCE_REVIEW,
                                        LoopStage.VERIFICATION, LoopStage.SYNTHESIS,
                                        LoopStage.CRITIQUE, LoopStage.IMPROVEMENT,
                                        LoopStage.PRIOR_WORK, LoopStage.EXPERIMENT_PLANNING),
    }[target]
    start_research_loop(state, loop.loop_id)
    for stage in path:
        loop = transition_research_loop(state, loop.loop_id, stage, reason=f"Test enters {stage.value}.")
    iteration = begin_loop_iteration(state, loop.loop_id, (fixture["task"].task_id,))
    fixture["loop"] = next(item for item in state.research_loops if item.loop_id == loop.loop_id)
    fixture["loop_iteration_id"] = iteration.iteration_id
    return fixture["loop"]


def _dispatch(fixture, stage, capability, request, *, retrieval_provider=None, llm_provider=None,
              search_provider=None, **kwargs):
    loop = _advance(fixture, stage)
    return dispatch_research_capability(
        fixture["plan"], fixture["run"], fixture["task"], fixture["review"], loop, fixture["state"],
        capability=capability, capability_request=request, iteration_id=fixture["loop_iteration_id"],
        retrieval_provider=retrieval_provider, llm_provider=llm_provider, search_provider=search_provider,
        **kwargs,
    )


class ResearchCapabilityDispatchTests(unittest.TestCase):
    def test_all_five_existing_capabilities_dispatch_once_through_public_apis(self):
        # Each Phase 6 fixture uses static providers and synthetic stored state.
        f = local_tests._plan()
        result = _dispatch(
            dict(plan=f[0], review=f[1], run=f[2], task=f[3], state=f[4], iteration=f[5]),
            LoopStage.RESEARCH, CapabilityType.LOCAL_RETRIEVAL,
            LocalRetrievalRequest("synthetic query", limit=1),
            retrieval_provider=StaticRetrievalProvider(local_tests._items()),
        )
        self.assertEqual(result.status, DispatchStatus.COMPLETED)
        self.assertEqual(result.capability, CapabilityType.LOCAL_RETRIEVAL)
        self.assertEqual(len(result.search_action_ids), 1)

        f = verification_tests._fixture()
        result = _dispatch(f, LoopStage.VERIFICATION, CapabilityType.EVIDENCE_VERIFICATION,
                           verification_tests.EvidenceVerificationRequest(
                               "System X uses BM25.", (f["evidence"].evidence_id,)),
                           llm_provider=verification_tests.QueueLLM(verification_tests._stance("SUPPORTS")))
        self.assertEqual(result.status, DispatchStatus.COMPLETED)

        f = improvement_tests._fixture()
        result = _dispatch(
            f, LoopStage.IMPROVEMENT, CapabilityType.IMPROVEMENT_GENERATION,
            improvement_tests._request(f),
            llm_provider=improvement_tests.CountingLLM(improvement_tests._response(
                improvement_tests._candidate(evidence_id=f["evidence"].evidence_id,
                                             claim_id=f["claim"].claim_id))),
        )
        self.assertEqual(result.status, DispatchStatus.COMPLETED)

        f = prior_tests._fixture()
        f["task"] = f["tasks"][2]
        class AssessingLLM:
            def generate(self, messages, *, temperature=0.0, timeout=120):
                payload = json.loads(messages[1]["content"])
                return prior_tests._assessments([item["search_result_id"] for item in payload["results"]])
        prior_request = prior_tests.PriorWorkInvestigationRequest(
            f["candidate_artifact"].artifact_id, f["candidate"].candidate_id,
            PriorWorkSearchScope(maximum_queries=1, maximum_results=2,
                                 source_types=("paper",), provider_name="static"),
            queries=("documented limitation",), question_id=f["state_question"].question_id,
        )
        result = _dispatch(f, LoopStage.PRIOR_WORK, CapabilityType.PRIOR_WORK_INVESTIGATION,
                           prior_request, search_provider=f["provider"], llm_provider=AssessingLLM())
        self.assertEqual(result.status, DispatchStatus.COMPLETED)
        self.assertTrue(result.search_action_ids)

        f = experiment_tests._fixture()
        result = _dispatch(f, LoopStage.EXPERIMENT_PLANNING, CapabilityType.EXPERIMENT_PLANNING,
                           experiment_tests._request(f),
                           llm_provider=experiment_tests.QueueLLM(experiment_tests._plan_response()))
        self.assertEqual(result.status, DispatchStatus.COMPLETED)
        self.assertEqual(len(f["state"].research_loops[0].capability_results), 1)

    def test_dispatch_requires_one_valid_explicit_capability_and_typed_request(self):
        f = local_tests._plan()
        fixture = dict(plan=f[0], review=f[1], run=f[2], task=f[3], state=f[4], iteration=f[5])
        loop = _advance(fixture, LoopStage.RESEARCH)
        common = dict(plan=f[0], run=f[2], task=f[3], review=f[1], loop=loop, state=f[4],
                      iteration_id=fixture["loop_iteration_id"], retrieval_provider=StaticRetrievalProvider([]))
        with self.assertRaises(TypeError):
            dispatch_research_capability(**common, capability_request=LocalRetrievalRequest("q"))
        for invalid in ([], "made_up"):
            with self.assertRaises(CapabilityDispatchError):
                dispatch_research_capability(**common, capability=invalid,
                                             capability_request=LocalRetrievalRequest("q"))
        with self.assertRaisesRegex(CapabilityDispatchError, "requires a typed"):
            dispatch_research_capability(**common, capability=CapabilityType.LOCAL_RETRIEVAL,
                                         capability_request=verification_tests.EvidenceVerificationRequest("claim", ("e",)))

    def test_wrong_stage_is_rejected_before_provider_or_phase5_execution(self):
        f = local_tests._plan()
        fixture = dict(plan=f[0], review=f[1], run=f[2], task=f[3], state=f[4], iteration=f[5])
        loop = _advance(fixture, LoopStage.RESEARCH)
        provider = local_tests.CountingProvider(local_tests._items())
        with self.assertRaisesRegex(CapabilityDispatchError, "incompatible with stage"):
            dispatch_research_capability(
                f[0], f[2], f[3], f[1], loop, f[4], capability=CapabilityType.EVIDENCE_VERIFICATION,
                capability_request=verification_tests.EvidenceVerificationRequest("claim", ("missing",)),
                iteration_id=fixture["loop_iteration_id"], llm_provider=verification_tests.QueueLLM("unused"),
            )
        self.assertEqual(provider.calls, [])
        self.assertEqual(f[4].task_executions, [])
        self.assertEqual(f[4].research_loops[0].capability_requests, ())

    def test_task_plan_revision_run_loop_iteration_and_gate_are_validated(self):
        f = local_tests._plan()
        fixture = dict(plan=f[0], review=f[1], run=f[2], task=f[3], state=f[4], iteration=f[5])
        loop = _advance(fixture, LoopStage.RESEARCH)
        args = dict(capability=CapabilityType.LOCAL_RETRIEVAL,
                    capability_request=LocalRetrievalRequest("query"),
                    iteration_id=fixture["loop_iteration_id"], retrieval_provider=StaticRetrievalProvider([]))
        wrong_run = authorize_research_run(f[0], f[1], ResearchState(user_request="other"))
        with self.assertRaises(CapabilityDispatchError):
            dispatch_research_capability(f[0], wrong_run, f[3], f[1], loop, f[4], **args)
        with self.assertRaises(CapabilityDispatchError):
            dispatch_research_capability(f[0], f[2], f[3], f[1], loop, f[4], **{**args, "iteration_id": "wrong"})
        gate = transition_research_loop(f[4], loop.loop_id, LoopStage.RESEARCHER_REVIEW,
                                       reason="Explicit researcher decision required.")
        with self.assertRaisesRegex(CapabilityDispatchError, "RUNNING loop"):
            dispatch_research_capability(f[0], f[2], f[3], f[1], gate, f[4], **args)

    def test_plan_revision_mismatch_and_unapproved_task_are_rejected_before_execution(self):
        f = local_tests._plan()
        fixture = dict(plan=f[0], review=f[1], run=f[2], task=f[3], state=f[4], iteration=f[5])
        loop = _advance(fixture, LoopStage.RESEARCH)
        args = dict(capability=CapabilityType.LOCAL_RETRIEVAL,
                    capability_request=LocalRetrievalRequest("query"),
                    iteration_id=fixture["loop_iteration_id"], retrieval_provider=StaticRetrievalProvider([]))
        f[0].revision += 1
        with self.assertRaises(CapabilityDispatchError):
            dispatch_research_capability(f[0], f[2], f[3], f[1], loop, f[4], **args)
        f[0].revision -= 1
        f[2].authorized_task_ids.clear()
        with self.assertRaises(CapabilityDispatchError):
            dispatch_research_capability(f[0], f[2], f[3], f[1], loop, f[4], **args)
        self.assertEqual(f[4].task_executions, [])

    def test_explicit_retry_uses_phase5_retry_reference_on_a_new_loop(self):
        f = local_tests._plan()
        fixture = dict(plan=f[0], review=f[1], run=f[2], task=f[3], state=f[4], iteration=f[5])
        loop = _advance(fixture, LoopStage.RESEARCH)

        class FailingProvider:
            def retrieve(self, query, limit=8):
                raise RuntimeError("synthetic provider failure")

        first = dispatch_research_capability(
            f[0], f[2], f[3], f[1], loop, f[4], capability=CapabilityType.LOCAL_RETRIEVAL,
            capability_request=LocalRetrievalRequest("query"), iteration_id=fixture["loop_iteration_id"],
            retrieval_provider=FailingProvider(),
        )
        self.assertEqual(first.status, DispatchStatus.FAILED)
        retry_loop = create_research_loop(f[0], f[2], f[1], f[4], max_iterations=2)
        fixture["loop"] = retry_loop
        retry_loop = _advance(fixture, LoopStage.RESEARCH)
        second = dispatch_research_capability(
            f[0], f[2], f[3], f[1], retry_loop, f[4], capability=CapabilityType.LOCAL_RETRIEVAL,
            capability_request=LocalRetrievalRequest("query"), iteration_id=fixture["loop_iteration_id"],
            retrieval_provider=StaticRetrievalProvider([]), retry_of_execution_id=first.execution_id,
        )
        self.assertEqual(second.status, DispatchStatus.COMPLETED)
        self.assertEqual(len(f[4].task_executions), 2)
        self.assertEqual(f[4].task_executions[1].retry_of_execution_id, first.execution_id)

    def test_successful_dispatch_and_loop_history_survive_state_round_trip(self):
        f = local_tests._plan()
        fixture = dict(plan=f[0], review=f[1], run=f[2], task=f[3], state=f[4], iteration=f[5])
        loop = _advance(fixture, LoopStage.RESEARCH)
        result = dispatch_research_capability(
            f[0], f[2], f[3], f[1], loop, f[4], capability=CapabilityType.LOCAL_RETRIEVAL,
            capability_request=LocalRetrievalRequest("query"), iteration_id=fixture["loop_iteration_id"],
            retrieval_provider=StaticRetrievalProvider(local_tests._items()),
        )
        restored = ResearchState.from_dict(json.loads(json.dumps(f[4].to_dict())))
        persisted = restored.research_loops[0]
        self.assertEqual(persisted.capability_results[0].execution_id, result.execution_id)
        self.assertEqual(persisted.capability_results[0].artifact_ids, result.artifact_ids)
        self.assertEqual(restored.validate_lineage(), [])

    def test_phase5_failure_is_recorded_and_not_retried_or_advanced(self):
        f = local_tests._plan()
        fixture = dict(plan=f[0], review=f[1], run=f[2], task=f[3], state=f[4], iteration=f[5])
        loop = _advance(fixture, LoopStage.RESEARCH)
        class FailingProvider:
            def retrieve(self, query, limit=8):
                raise RuntimeError("synthetic provider issue")
        result = dispatch_research_capability(
            f[0], f[2], f[3], f[1], loop, f[4], capability=CapabilityType.LOCAL_RETRIEVAL,
            capability_request=LocalRetrievalRequest("query"), iteration_id=fixture["loop_iteration_id"],
            retrieval_provider=FailingProvider(),
        )
        self.assertEqual(result.status, DispatchStatus.FAILED)
        self.assertEqual(len(f[4].task_executions), 1)
        self.assertEqual(f[4].task_executions[0].status.value, "FAILED")
        self.assertEqual(f[4].research_artifacts, [])
        self.assertEqual(f[4].research_loops[0].current_stage, LoopStage.FAILED)

    def test_dispatch_result_serialization_and_no_automatic_stage_advance(self):
        f = local_tests._plan()
        fixture = dict(plan=f[0], review=f[1], run=f[2], task=f[3], state=f[4], iteration=f[5])
        original_stage = _advance(fixture, LoopStage.RESEARCH).current_stage
        result = dispatch_research_capability(
            f[0], f[2], f[3], f[1], f[4].research_loops[0], f[4],
            capability=CapabilityType.LOCAL_RETRIEVAL, capability_request=LocalRetrievalRequest("query"),
            iteration_id=fixture["loop_iteration_id"], retrieval_provider=StaticRetrievalProvider([]),
        )
        self.assertEqual(ResearchCapabilityDispatchResult.from_json(result.to_json()), result)
        loop = f[4].research_loops[0]
        self.assertEqual(loop.current_stage, original_stage)
        self.assertEqual(len(loop.capability_requests), 1)
        self.assertEqual(len(loop.capability_results), 1)
        self.assertEqual(loop.capability_results[0].execution_id, result.execution_id)
        self.assertEqual(loop.status, ResearchLoopStatus.RUNNING)
        self.assertNotIn(CapabilityType.EVIDENCE_VERIFICATION.value,
                         [item.capability.value for item in loop.capability_requests])

    def test_domain_neutral_dispatch_uses_same_contract(self):
        for domain in ("NLP", "medicine", "agriculture", "physics", "cybersecurity"):
            with self.subTest(domain=domain):
                plan, review, run, task, state, *_ = local_tests._plan(domain)
                fixture = {"plan": plan, "review": review, "run": run, "task": task, "state": state}
                loop = _advance(fixture, LoopStage.RESEARCH)
                result = dispatch_research_capability(
                    plan, run, task, review, loop, state, capability=CapabilityType.LOCAL_RETRIEVAL,
                    capability_request=LocalRetrievalRequest("synthetic query"),
                    iteration_id=fixture["loop_iteration_id"],
                    retrieval_provider=StaticRetrievalProvider(local_tests._items()),
                )
                self.assertEqual(result.status, DispatchStatus.COMPLETED)


if __name__ == "__main__":
    unittest.main()
