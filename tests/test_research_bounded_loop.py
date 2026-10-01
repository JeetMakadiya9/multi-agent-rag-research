from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
for path in (str(ROOT), str(TESTS)):
    if path not in sys.path:
        sys.path.insert(0, path)

import test_research_capabilities as capability_fixtures
import test_research_synthesis_critique as synthesis_fixtures
from research_bounded_loop import (
    BeginIteration,
    BoundedExecutionPolicy,
    BoundedLoopOutcome,
    Critique,
    DispatchCapability,
    FinishIteration,
    ResearcherDecision,
    SelectTask,
    StartLoop,
    Synthesize,
    TransitionStage,
    run_bounded_research_loop,
)
from research_capabilities import CapabilityType, LocalRetrievalRequest
from research_capability_dispatch import DispatchStatus
from research_continuation import ResearchContinuationAction
from research_loop import (
    LoopIterationStatus,
    LoopStage,
    ResearchLoopStatus,
    StopReason,
    TransitionAuthority,
    create_research_loop,
)
from research_retrieval import StaticRetrievalProvider
from research_state import ResearchState


def _fixture(*, maximum_loop_iterations=5):
    plan, review, run, task, state, *_ = capability_fixtures._plan()
    loop = create_research_loop(plan, run, review, state, max_iterations=maximum_loop_iterations)
    policy = BoundedExecutionPolicy(
        maximum_iterations=maximum_loop_iterations,
        maximum_capability_invocations=5,
        maximum_task_executions=5,
        allowed_capabilities=(CapabilityType.LOCAL_RETRIEVAL,),
        allowed_stages=tuple(LoopStage),
        allowed_task_ids=(task.task_id,),
        allow_continuation=True,
    )
    return {"plan": plan, "review": review, "run": run, "task": task,
            "state": state, "loop": loop, "policy": policy}


def _start_research_commands(task_id):
    return (
        StartLoop(step_id="start"),
        TransitionStage(LoopStage.RESEARCH, "Caller enters research.", TransitionAuthority.SYSTEM,
                        step_id="research"),
        BeginIteration((task_id,), step_id="iteration"),
        SelectTask(task_id, step_id="select"),
    )


class FailingProvider:
    def __init__(self):
        self.calls = []

    def retrieve(self, query, limit=8):
        self.calls.append((query, limit))
        raise RuntimeError("controlled synthetic failure")


class ResearchBoundedLoopTests(unittest.TestCase):
    def test_valid_bounded_sequence_dispatches_only_explicit_capability(self):
        f = _fixture()
        provider = capability_fixtures.CountingProvider(capability_fixtures._items())
        commands = _start_research_commands(f["task"].task_id) + (
            DispatchCapability(f["task"].task_id, CapabilityType.LOCAL_RETRIEVAL,
                               LocalRetrievalRequest("synthetic query", limit=1), step_id="retrieve"),
        )
        result = run_bounded_research_loop(
            f["plan"], f["run"], f["review"], f["loop"], f["state"], f["policy"], commands,
            retrieval_provider=provider,
        )
        self.assertEqual(result.outcome, BoundedLoopOutcome.SEQUENCE_FINISHED)
        self.assertEqual(len(result.dispatches), 1)
        self.assertEqual(result.dispatches[0].status, DispatchStatus.COMPLETED)
        self.assertEqual(len(f["state"].task_executions), 1)
        self.assertEqual(len(f["state"].research_artifacts), 1)
        self.assertEqual(len(f["state"].research_loops[0].capability_requests), 1)
        self.assertEqual(f["state"].validate_lineage(), [])

    def test_policy_bounds_reject_zero_iterations_and_stop_before_next_iteration(self):
        f = _fixture()
        with self.assertRaisesRegex(ValueError, "maximum_iterations"):
            BoundedExecutionPolicy(0, 1, 1, (), (LoopStage.RESEARCH,), (f["task"].task_id,))
        commands = _start_research_commands(f["task"].task_id) + (
            FinishIteration(LoopIterationStatus.COMPLETED, "Caller finished iteration.", step_id="finish"),
            BeginIteration((f["task"].task_id,), step_id="second-iteration"),
        )
        policy = BoundedExecutionPolicy(1, 5, 5, (),
            (LoopStage.INTAKE, LoopStage.PLANNING, LoopStage.RESEARCH), (f["task"].task_id,))
        result = run_bounded_research_loop(f["plan"], f["run"], f["review"], f["loop"],
                                           f["state"], policy, commands)
        self.assertEqual(result.outcome, BoundedLoopOutcome.LIMIT_REACHED)
        self.assertEqual(len(f["state"].research_loops[0].loop_iterations), 1)
        self.assertIsNone(f["state"].research_loops[0].current_iteration_id)

    def test_capability_limit_and_capability_allowlist_fail_closed(self):
        f = _fixture()
        zero_calls = BoundedExecutionPolicy(2, 0, 2, (CapabilityType.LOCAL_RETRIEVAL,),
            f["policy"].allowed_stages, f["policy"].allowed_task_ids)
        provider = capability_fixtures.CountingProvider(capability_fixtures._items())
        result = run_bounded_research_loop(
            f["plan"], f["run"], f["review"], f["loop"], f["state"], zero_calls,
            _start_research_commands(f["task"].task_id) + (DispatchCapability(
                f["task"].task_id, CapabilityType.LOCAL_RETRIEVAL, LocalRetrievalRequest("q"),
                step_id="retrieve"),), retrieval_provider=provider,
        )
        self.assertEqual(result.outcome, BoundedLoopOutcome.LIMIT_REACHED)
        self.assertEqual(provider.calls, [])
        self.assertEqual(f["state"].task_executions, [])

        g = _fixture()
        disallowed = BoundedExecutionPolicy(2, 2, 2, (), g["policy"].allowed_stages,
                                             g["policy"].allowed_task_ids)
        with self.assertRaisesRegex(ValueError, "outside the caller policy"):
            run_bounded_research_loop(
                g["plan"], g["run"], g["review"], g["loop"], g["state"], disallowed,
                _start_research_commands(g["task"].task_id) + (DispatchCapability(
                    g["task"].task_id, CapabilityType.LOCAL_RETRIEVAL, LocalRetrievalRequest("q"),
                    step_id="retrieve"),), retrieval_provider=provider,
            )
        self.assertEqual(g["state"].task_executions, [])

    def test_policy_cannot_expand_loop_bound_and_wrong_context_rejected_pre_mutation(self):
        f = _fixture(maximum_loop_iterations=1)
        too_large = BoundedExecutionPolicy(2, 1, 1, (), f["policy"].allowed_stages,
                                           f["policy"].allowed_task_ids)
        with self.assertRaisesRegex(ValueError, "cannot expand"):
            run_bounded_research_loop(f["plan"], f["run"], f["review"], f["loop"],
                                      f["state"], too_large, ())
        other_run = type(f["run"])()
        with self.assertRaisesRegex(ValueError, "revision"):
            run_bounded_research_loop(f["plan"], other_run, f["review"], f["loop"],
                                      f["state"], f["policy"], ())
        self.assertEqual(f["state"].research_loops[0].status, ResearchLoopStatus.CREATED)

    def test_policy_rejects_unapproved_task_before_any_state_mutation(self):
        f = _fixture()
        commands = _start_research_commands("unauthorized-task")
        with self.assertRaisesRegex(ValueError, "outside the exact authorized"):
            run_bounded_research_loop(f["plan"], f["run"], f["review"], f["loop"],
                                      f["state"], f["policy"], commands)
        self.assertEqual(f["state"].research_loops[0].status, ResearchLoopStatus.CREATED)
        self.assertEqual(len(f["state"].iterations), len(capability_fixtures._plan()[5:6]))

    def test_plan_revision_mismatch_and_task_execution_bound_fail_before_dispatch(self):
        f = _fixture()
        f["plan"].revision += 1
        with self.assertRaisesRegex(ValueError, "revision"):
            run_bounded_research_loop(f["plan"], f["run"], f["review"], f["loop"],
                                      f["state"], f["policy"], ())
        f["plan"].revision -= 1

        provider = capability_fixtures.CountingProvider(capability_fixtures._items())
        no_executions = BoundedExecutionPolicy(2, 2, 0, (CapabilityType.LOCAL_RETRIEVAL,),
            f["policy"].allowed_stages, f["policy"].allowed_task_ids)
        commands = _start_research_commands(f["task"].task_id) + (DispatchCapability(
            f["task"].task_id, CapabilityType.LOCAL_RETRIEVAL, LocalRetrievalRequest("q"),
            step_id="retrieve"),)
        result = run_bounded_research_loop(f["plan"], f["run"], f["review"], f["loop"],
                                           f["state"], no_executions, commands,
                                           retrieval_provider=provider)
        self.assertEqual(result.outcome, BoundedLoopOutcome.LIMIT_REACHED)
        self.assertEqual(provider.calls, [])
        self.assertEqual(f["state"].task_executions, [])

    def test_dispatch_requires_explicit_task_selection(self):
        f = _fixture()
        provider = capability_fixtures.CountingProvider(capability_fixtures._items())
        commands = (
            StartLoop(step_id="start"),
            TransitionStage(LoopStage.RESEARCH, "Caller enters research.", TransitionAuthority.SYSTEM,
                            step_id="research"),
            BeginIteration((f["task"].task_id,), step_id="iteration"),
            DispatchCapability(f["task"].task_id, CapabilityType.LOCAL_RETRIEVAL,
                               LocalRetrievalRequest("q"), step_id="retrieve"),
        )
        with self.assertRaisesRegex(ValueError, "explicit SelectTask"):
            run_bounded_research_loop(f["plan"], f["run"], f["review"], f["loop"],
                                      f["state"], f["policy"], commands,
                                      retrieval_provider=provider)
        self.assertEqual(provider.calls, [])
        self.assertEqual(f["state"].task_executions, [])

    def test_researcher_gate_waits_and_does_not_auto_continue(self):
        f = _fixture()
        commands = (
            StartLoop(step_id="start"),
            TransitionStage(LoopStage.RESEARCH, "Enter research.", TransitionAuthority.SYSTEM,
                            step_id="research"),
            TransitionStage(LoopStage.RESEARCHER_REVIEW, "Researcher decision required.",
                            TransitionAuthority.SYSTEM, step_id="gate"),
        )
        result = run_bounded_research_loop(f["plan"], f["run"], f["review"], f["loop"],
                                           f["state"], f["policy"], commands)
        self.assertEqual(result.outcome, BoundedLoopOutcome.WAITING_FOR_RESEARCHER)
        self.assertEqual(result.loop_status, ResearchLoopStatus.WAITING_FOR_RESEARCHER)
        self.assertEqual(result.dispatches, ())

    def test_explicit_researcher_stop_is_terminal(self):
        f = _fixture()
        commands = _start_research_commands(f["task"].task_id)[:2] + (
            TransitionStage(LoopStage.STOPPED, "Researcher stops.", TransitionAuthority.RESEARCHER,
                            StopReason.RESEARCHER_REQUESTED_STOP, step_id="stop"),
        )
        result = run_bounded_research_loop(f["plan"], f["run"], f["review"], f["loop"],
                                           f["state"], f["policy"], commands)
        self.assertEqual(result.outcome, BoundedLoopOutcome.STOPPED)
        self.assertEqual(result.loop_status, ResearchLoopStatus.STOPPED)

    def test_capability_failure_is_preserved_and_not_retried_automatically(self):
        f = _fixture()
        provider = FailingProvider()
        commands = _start_research_commands(f["task"].task_id) + (DispatchCapability(
            f["task"].task_id, CapabilityType.LOCAL_RETRIEVAL, LocalRetrievalRequest("q"),
            step_id="retrieve"),)
        result = run_bounded_research_loop(f["plan"], f["run"], f["review"], f["loop"],
                                           f["state"], f["policy"], commands,
                                           retrieval_provider=provider)
        self.assertEqual(result.outcome, BoundedLoopOutcome.FAILED)
        self.assertEqual(provider.calls, [("q", 8)])
        self.assertEqual(len(f["state"].task_executions), 1)
        self.assertEqual(f["state"].task_executions[0].status.value, "FAILED")
        self.assertEqual(f["state"].research_artifacts, [])
        self.assertEqual(f["state"].research_loops[0].status, ResearchLoopStatus.FAILED)

    def test_continuation_requires_explicit_decision_and_is_not_dispatched_automatically(self):
        f = _fixture()
        provider = StaticRetrievalProvider(capability_fixtures._items())
        dispatch = DispatchCapability(f["task"].task_id, CapabilityType.LOCAL_RETRIEVAL,
                                      LocalRetrievalRequest("q", limit=1), step_id="exec-1")
        continuation = ResearcherDecision(ResearchContinuationAction.CONTINUE_SAME_TASK,
                                          previous_execution_id="will-be-replaced", step_id="continue")
        # The execution ID is not available when the immutable command sequence is built.
        # This interface therefore requires an explicit prior ID; use a deterministic
        # two-call boundary to prove that the decision itself never dispatches.
        first = run_bounded_research_loop(
            f["plan"], f["run"], f["review"], f["loop"], f["state"], f["policy"],
            _start_research_commands(f["task"].task_id) + (dispatch,), retrieval_provider=provider,
        )
        self.assertEqual(first.outcome, BoundedLoopOutcome.SEQUENCE_FINISHED)
        execution_id = f["state"].task_executions[-1].execution_id
        f["loop"] = f["state"].research_loops[0]
        decision = ResearcherDecision(ResearchContinuationAction.CONTINUE_SAME_TASK,
                                      previous_execution_id=execution_id, step_id="continue")
        second = run_bounded_research_loop(
            f["plan"], f["run"], f["review"], f["loop"], f["state"], f["policy"], (decision,),
        )
        self.assertEqual(second.outcome, BoundedLoopOutcome.SEQUENCE_FINISHED)
        self.assertEqual(len(f["state"].task_executions), 1)
        self.assertEqual(f["state"].research_loops[0].current_task_id, f["task"].task_id)
        self.assertEqual(len(f["state"].research_loops[0].continuation_authorizations), 1)

    def test_explicit_continuation_reference_dispatches_a_new_execution(self):
        f = _fixture()
        provider = StaticRetrievalProvider(capability_fixtures._items())
        first_commands = _start_research_commands(f["task"].task_id) + (DispatchCapability(
            f["task"].task_id, CapabilityType.LOCAL_RETRIEVAL, LocalRetrievalRequest("q", limit=1),
            step_id="first"),)
        run_bounded_research_loop(f["plan"], f["run"], f["review"], f["loop"], f["state"],
                                  f["policy"], first_commands, retrieval_provider=provider)
        prior_id = f["state"].task_executions[-1].execution_id
        decision = ResearcherDecision(ResearchContinuationAction.CONTINUE_SAME_TASK,
                                      previous_execution_id=prior_id, step_id="continue")
        next_dispatch = DispatchCapability(f["task"].task_id, CapabilityType.LOCAL_RETRIEVAL,
                                           LocalRetrievalRequest("q2", limit=1),
                                           continuation_step_id="continue", step_id="second")
        second = run_bounded_research_loop(
            f["plan"], f["run"], f["review"], f["state"].research_loops[0], f["state"],
            f["policy"], (decision, next_dispatch), retrieval_provider=provider,
        )
        self.assertEqual(second.outcome, BoundedLoopOutcome.SEQUENCE_FINISHED)
        self.assertEqual(len(f["state"].task_executions), 2)
        self.assertNotEqual(f["state"].task_executions[0].execution_id,
                            f["state"].task_executions[1].execution_id)
        self.assertEqual(f["state"].task_executions[1].continuation_of_execution_id, prior_id)
        self.assertIsNone(f["state"].task_executions[1].retry_of_execution_id)
        self.assertEqual(f["state"].task_executions[0].status.value, "COMPLETED")

    def test_7d_synthesis_and_critique_are_explicit_commands(self):
        f = synthesis_fixtures._analysis_fixture("agriculture")
        current_loop = f["state"].research_loops[-1]
        f["loop"] = current_loop
        policy = BoundedExecutionPolicy(
            maximum_iterations=current_loop.max_iterations,
            maximum_capability_invocations=3,
            maximum_task_executions=4,
            allowed_capabilities=(),
            allowed_stages=(LoopStage.SYNTHESIS, LoopStage.CRITIQUE),
            allowed_task_ids=tuple(f["run"].authorized_task_ids),
        )
        commands = (Synthesize(step_id="synthesize"),
                    TransitionStage(LoopStage.CRITIQUE, "Caller requests critique.",
                                   TransitionAuthority.SYSTEM, step_id="to-critique"),
                    Critique("synthesize", step_id="critique"))
        result = run_bounded_research_loop(
            f["plan"], f["run"], f["review"], current_loop, f["state"], policy, commands,
        )
        self.assertEqual(len(result.syntheses), 1)
        self.assertEqual(len(result.critiques), 1)
        self.assertEqual(result.syntheses[0].run_id, f["run"].run_id)
        self.assertEqual(result.critiques[0].synthesis_id, result.syntheses[0].synthesis_id)
        self.assertEqual(f["state"].validate_lineage(), [])

    def test_result_round_trip_and_old_state_payload_remain_compatible(self):
        f = _fixture()
        result = run_bounded_research_loop(f["plan"], f["run"], f["review"], f["loop"],
                                           f["state"], f["policy"], (StartLoop(step_id="start"),))
        self.assertEqual(type(result).from_json(result.to_json()), result)
        legacy = f["state"].to_dict()
        legacy.pop("research_loops", None)
        restored = ResearchState.from_dict(legacy)
        self.assertEqual(restored.research_loops, [])

    def test_wrong_stage_and_terminal_loop_rejected_without_execution(self):
        f = _fixture()
        provider = StaticRetrievalProvider(capability_fixtures._items())
        wrong = _start_research_commands(f["task"].task_id) + (
            DispatchCapability(f["task"].task_id, CapabilityType.LOCAL_RETRIEVAL,
                               LocalRetrievalRequest("q"), step_id="retrieve"),)
        # Starting in INTAKE then requesting a capability without moving to RESEARCH fails at dispatch.
        policy = BoundedExecutionPolicy(2, 2, 2, (CapabilityType.LOCAL_RETRIEVAL,),
            (LoopStage.INTAKE, LoopStage.PLANNING), (f["task"].task_id,))
        with self.assertRaisesRegex(ValueError, "outside the caller policy"):
            run_bounded_research_loop(f["plan"], f["run"], f["review"], f["loop"],
                                      f["state"], policy, wrong, retrieval_provider=provider)
        self.assertEqual(f["state"].task_executions, [])

        g = _fixture()
        stopped = TransitionStage(LoopStage.STOPPED, "Stop.", TransitionAuthority.RESEARCHER,
                                  StopReason.RESEARCHER_REQUESTED_STOP, step_id="stop")
        run_bounded_research_loop(g["plan"], g["run"], g["review"], g["loop"], g["state"],
                                  g["policy"], (StartLoop(), TransitionStage(
                                      LoopStage.RESEARCH, "Research.", TransitionAuthority.SYSTEM), stopped))
        terminal = g["state"].research_loops[0]
        with self.assertRaisesRegex(ValueError, "Terminal ResearchLoop"):
            run_bounded_research_loop(g["plan"], g["run"], g["review"], terminal, g["state"],
                                      g["policy"], ())


if __name__ == "__main__":
    unittest.main()
