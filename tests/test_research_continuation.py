from __future__ import annotations

import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
for path in (str(ROOT), str(TESTS)):
    if path not in sys.path:
        sys.path.insert(0, path)

import test_research_capabilities as local_tests
from research_capabilities import CapabilityType, LocalRetrievalRequest
from research_capability_dispatch import DispatchStatus, dispatch_research_capability
from research_continuation import (
    ResearchContinuationAction,
    apply_research_continuation,
)
from research_loop import (
    LoopStage, ResearchLoopStatus, StopReason, begin_loop_iteration,
    create_research_loop, start_research_loop, transition_research_loop,
)
from research_retrieval import StaticRetrievalProvider
from research_task_execution import TaskExecutionAuthorizationError, execute_authorized_task


def _fixture():
    plan, review, run, task, state, *_ = local_tests._plan()
    loop = create_research_loop(plan, run, review, state, max_iterations=5)
    start_research_loop(state, loop.loop_id)
    transition_research_loop(state, loop.loop_id, LoopStage.RESEARCH, reason="enter research")
    record = begin_loop_iteration(state, loop.loop_id, (task.task_id,))
    return plan, review, run, task, state, state.research_loops[0], record.iteration_id


def _dispatch(fixture, *, iteration_id=None, authorization_id=None):
    plan, review, run, task, state, loop, default_iteration = fixture
    return dispatch_research_capability(
        plan, run, task, review, state.research_loops[0], state,
        capability=CapabilityType.LOCAL_RETRIEVAL,
        capability_request=LocalRetrievalRequest("synthetic query"),
        iteration_id=iteration_id or default_iteration,
        retrieval_provider=StaticRetrievalProvider([]),
        continuation_authorization_id=authorization_id,
    )


class ResearchContinuationTests(unittest.TestCase):
    def test_successful_dispatch_completes_execution_but_keeps_task_active(self):
        fixture = _fixture()
        result = _dispatch(fixture)
        loop = fixture[4].research_loops[0]
        self.assertEqual(result.status, DispatchStatus.COMPLETED)
        self.assertEqual(fixture[4].task_executions[0].status.value, "COMPLETED")
        self.assertIn(fixture[3].task_id, loop.unresolved_task_ids)
        self.assertNotIn(fixture[3].task_id, loop.completed_task_ids)
        self.assertEqual(loop.current_task_id, fixture[3].task_id)
        self.assertEqual(fixture[3].status.value, "pending")

    def test_completed_execution_cannot_be_retried_or_repeated_silently(self):
        fixture = _fixture()
        first = _dispatch(fixture)
        plan, review, run, task, state, _, iteration_id = fixture
        handler = lambda _task, _ctx: __import__("research_task_execution").TaskExecutionResult(success=True)
        with self.assertRaisesRegex(TaskExecutionAuthorizationError, "completed.*cannot be.*retry|completed task execution"):
            execute_authorized_task(plan, run, task, review, handler, state,
                                    iteration_id=iteration_id, retry_of_execution_id=first.execution_id)
        with self.assertRaises(TaskExecutionAuthorizationError):
            execute_authorized_task(plan, run, task, review, handler, state, iteration_id=iteration_id)
        self.assertEqual(len(state.task_executions), 1)

    def test_same_task_continuation_creates_a_distinct_second_execution(self):
        fixture = _fixture()
        first = _dispatch(fixture)
        plan, review, run, task, state, loop, _ = fixture
        original_execution = state.task_executions[0].to_dict()
        decision = apply_research_continuation(
            plan, run, review, state.research_loops[0], state,
            action=ResearchContinuationAction.CONTINUE_SAME_TASK,
            previous_execution_id=first.execution_id,
        )
        self.assertFalse(decision.execution_occurred)
        self.assertEqual(decision.task_id, task.task_id)
        self.assertNotEqual(decision.iteration_id, first.iteration_id)
        self.assertTrue(decision.continuation_authorization_id)
        self.assertEqual(len(state.task_executions), 1)
        second = _dispatch(fixture, iteration_id=decision.iteration_id,
                           authorization_id=decision.continuation_authorization_id)
        old, new = state.task_executions
        self.assertNotEqual(old.execution_id, new.execution_id)
        self.assertEqual(old.status.value, "COMPLETED")
        self.assertEqual(old.to_dict(), original_execution)
        self.assertEqual(new.task_id, old.task_id)
        self.assertEqual(new.run_id, old.run_id)
        self.assertEqual((new.plan_id, new.plan_revision), (old.plan_id, old.plan_revision))
        self.assertIsNone(new.retry_of_execution_id)
        self.assertEqual(new.continuation_of_execution_id, old.execution_id)
        self.assertEqual(new.continuation_authorization_id, decision.continuation_authorization_id)
        self.assertEqual(second.status, DispatchStatus.COMPLETED)
        self.assertEqual(state.validate_lineage(), [])
        restored = type(state).from_dict(json.loads(json.dumps(state.to_dict())))
        self.assertEqual(restored.task_executions[1].continuation_authorization_id,
                         decision.continuation_authorization_id)
        self.assertEqual(restored.validate_lineage(), [])

    def test_continue_with_task_requires_and_selects_explicit_task_without_dispatch(self):
        fixture = _fixture()
        plan, review, run, first_task, state, loop, iteration_id = fixture
        second_task = replace(first_task, description="An explicitly selected follow-up task.",
                              task_id="task-explicit-follow-up")
        plan.tasks.append(second_task)
        run.authorized_task_ids.append(second_task.task_id)
        loop = state.research_loops[0]
        state.research_loops[0] = replace(loop, unresolved_task_ids=loop.unresolved_task_ids + (second_task.task_id,))
        fixture = (plan, review, run, first_task, state, state.research_loops[0], iteration_id)
        first = _dispatch(fixture)
        decision = apply_research_continuation(
            plan, run, review, state.research_loops[0], state,
            action=ResearchContinuationAction.CONTINUE_WITH_TASK,
            previous_execution_id=first.execution_id, task_id=second_task.task_id,
        )
        self.assertEqual(decision.task_id, second_task.task_id)
        self.assertIsNone(decision.continuation_authorization_id)
        self.assertFalse(decision.execution_occurred)
        self.assertEqual(len(state.task_executions), 1)
        self.assertEqual(state.research_loops[0].current_task_id, second_task.task_id)

    def test_return_to_planning_requires_existing_review_transition(self):
        fixture = _fixture()
        plan, review, run, task, state, loop, _ = fixture
        transition_research_loop(state, loop.loop_id, LoopStage.RESEARCHER_REVIEW,
                                 reason="researcher review requested")
        result = apply_research_continuation(
            plan, run, review, state.research_loops[0], state,
            action=ResearchContinuationAction.RETURN_TO_PLANNING,
        )
        self.assertEqual(result.stage, LoopStage.PLANNING)
        self.assertEqual(result.status, ResearchLoopStatus.RUNNING)

    def test_continuation_requires_explicit_source_and_authorization(self):
        fixture = _fixture()
        first = _dispatch(fixture)
        plan, review, run, task, state, _, iteration_id = fixture
        with self.assertRaisesRegex(ValueError, "explicit previous_execution_id"):
            apply_research_continuation(plan, run, review, state.research_loops[0], state,
                                        action=ResearchContinuationAction.CONTINUE_SAME_TASK)
        with self.assertRaisesRegex(Exception, "continuation authorization"):
            _dispatch(fixture)
        self.assertEqual(len(state.task_executions), 1)
        self.assertEqual(len(state.research_loops[0].capability_requests), 1)

    def test_continuation_source_must_be_completed_and_from_this_loop(self):
        fixture = _fixture()
        with self.assertRaisesRegex(ValueError, "completed execution"):
            apply_research_continuation(fixture[0], fixture[2], fixture[1], fixture[5], fixture[4],
                                        action=ResearchContinuationAction.CONTINUE_SAME_TASK,
                                        previous_execution_id="unknown")

    def test_continuation_round_trip_and_legacy_execution_payload(self):
        fixture = _fixture()
        first = _dispatch(fixture)
        plan, review, run, task, state, _, _ = fixture
        decision = apply_research_continuation(plan, run, review, state.research_loops[0], state,
                                               action=ResearchContinuationAction.CONTINUE_SAME_TASK,
                                               previous_execution_id=first.execution_id)
        restored = type(state).from_dict(json.loads(json.dumps(state.to_dict())))
        auth = restored.research_loops[0].continuation_authorizations[0]
        self.assertEqual(auth.authorization_id, decision.continuation_authorization_id)
        legacy = state.task_executions[0].to_dict()
        legacy.pop("continuation_of_execution_id")
        legacy.pop("continuation_authorization_id")
        from research_task_execution import ResearchTaskExecution
        self.assertIsNone(ResearchTaskExecution.from_dict(legacy).continuation_of_execution_id)

    def test_stop_and_wait_are_caller_actions_and_terminal_loops_stay_terminal(self):
        for action, expected in ((ResearchContinuationAction.STOP, ResearchLoopStatus.STOPPED),
                                 (ResearchContinuationAction.WAIT_FOR_RESEARCHER, ResearchLoopStatus.WAITING_FOR_RESEARCHER)):
            fixture = _fixture()
            plan, review, run, task, state, loop, _ = fixture
            result = apply_research_continuation(plan, run, review, loop, state, action=action)
            self.assertEqual(result.status, expected)
            if expected is ResearchLoopStatus.STOPPED:
                with self.assertRaisesRegex(ValueError, "Terminal"):
                    apply_research_continuation(plan, run, review, state.research_loops[0], state,
                                                action=ResearchContinuationAction.CONTINUE_SAME_TASK,
                                                previous_execution_id="any")

    def test_caller_choice_does_not_dispatch_or_call_a_provider(self):
        fixture = _fixture()
        first = _dispatch(fixture)
        state = fixture[4]
        before = len(state.task_executions)
        decision = apply_research_continuation(
            fixture[0], fixture[2], fixture[1], state.research_loops[0], state,
            action=ResearchContinuationAction.CONTINUE_SAME_TASK,
            previous_execution_id=first.execution_id,
        )
        self.assertFalse(decision.execution_occurred)
        self.assertEqual(len(state.task_executions), before)

    def test_invalid_continuation_leaves_state_unchanged(self):
        fixture = _fixture()
        first = _dispatch(fixture)
        plan, review, run, task, state, loop, _ = fixture
        before = state.to_dict()
        with self.assertRaisesRegex(ValueError, "completed execution"):
            apply_research_continuation(
                plan, run, review, state.research_loops[0], state,
                action=ResearchContinuationAction.CONTINUE_SAME_TASK,
                previous_execution_id="not-an-execution",
            )
        self.assertEqual(state.to_dict(), before)


if __name__ == "__main__":
    unittest.main()
