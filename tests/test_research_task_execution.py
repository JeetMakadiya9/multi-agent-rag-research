from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research_plan_review import PlanReview, ReviewStatus
from research_planning import ResearchObjective, ResearchPlan, ResearchPlanQuestion, ResearchRequest, ResearchTask, TaskStatus
from research_run_authorization import authorize_research_run
from research_state import ResearchRun, ResearchState
from research_task_authorization import authorize_research_tasks
from research_task_execution import (
    ResearchTaskExecution,
    TaskExecutionAuthorizationError,
    TaskExecutionContext,
    TaskExecutionError,
    TaskExecutionRecordingError,
    TaskExecutionResult,
    TaskExecutionStatus,
    execute_authorized_task,
)


def make_plan(*, plan_id: str = "plan-execution", revision: int = 1, domain: str = "general") -> ResearchPlan:
    question = ResearchPlanQuestion(question_id="plan-question-1", text="What is reported?")
    task = ResearchTask(
        description="Perform a deterministic planned operation.",
        question_id=question.question_id,
        task_id="task-execute-1",
        status=TaskStatus.READY,
    )
    return ResearchPlan(
        plan_id=plan_id,
        revision=revision,
        request=ResearchRequest("Research a topic.", "Describe what is reported.", domain=domain),
        objective=ResearchObjective("Investigate reported work."),
        questions=[question],
        tasks=[task],
    )


def make_review(plan: ResearchPlan, *, review_id: str = "review-execution", status: ReviewStatus = ReviewStatus.APPROVED) -> PlanReview:
    return PlanReview(
        plan_id=plan.plan_id,
        plan_revision=plan.revision,
        status=status,
        reviewer=None if status is ReviewStatus.PENDING else "researcher",
        requested_changes=("Narrow scope.",) if status is ReviewStatus.CHANGES_REQUESTED else (),
        review_id=review_id,
        reviewed_at="2026-09-27T10:00:00+00:00",
    )


def authorized_setup(*, domain: str = "general", with_iteration: bool = False):
    plan = make_plan(domain=domain)
    review = make_review(plan)
    state = ResearchState(user_request="Controlled execution")
    run = authorize_research_run(plan, review, state)
    authorize_research_tasks(plan, run, [plan.tasks[0].task_id], review, state)
    iteration = state.create_iteration(run.run_id) if with_iteration else None
    return plan, review, run, state, plan.tasks[0], iteration


class ResearchTaskExecutionTests(unittest.TestCase):
    def test_authorized_task_invokes_handler_and_records_result(self) -> None:
        plan, review, run, state, task, _ = authorized_setup()
        calls = []

        def handler(received_task, context):
            calls.append((received_task, context))
            return TaskExecutionResult(True, output="deterministic output", artifact_ids=("artifact-1",))

        execution = execute_authorized_task(plan, run, task, review, handler, state)
        self.assertEqual(len(calls), 1)
        self.assertEqual(execution.status, TaskExecutionStatus.COMPLETED)
        self.assertTrue(execution.result.success)
        self.assertEqual(execution.result.output, "deterministic output")
        self.assertEqual(execution.result.artifact_ids, ("artifact-1",))
        self.assertIn(execution, state.task_executions)

    def test_unauthorized_task_is_rejected_before_handler_invocation(self) -> None:
        plan, review, run, state, _, _ = authorized_setup()
        unselected = ResearchTask("Not selected.", plan.questions[0].question_id, task_id="task-not-authorized")
        plan.tasks.append(unselected)
        calls = []
        with self.assertRaisesRegex(TaskExecutionAuthorizationError, "not authorized"):
            execute_authorized_task(plan, run, unselected, review, lambda *_: calls.append(True), state)
        self.assertEqual(calls, [])
        self.assertEqual(state.task_executions, [])

    def test_task_not_present_in_plan_is_rejected_before_handler(self) -> None:
        plan, review, run, state, _, _ = authorized_setup()
        foreign = ResearchTask("Foreign.", plan.questions[0].question_id, task_id="foreign-task")
        calls = []
        with self.assertRaisesRegex(TaskExecutionAuthorizationError, "not authorized|not present"):
            execute_authorized_task(plan, run, foreign, review, lambda *_: calls.append(True), state)
        self.assertEqual(calls, [])

    def test_unlinked_run_is_rejected_before_handler(self) -> None:
        plan, review, _, state, task, _ = authorized_setup()
        calls = []
        with self.assertRaises(TaskExecutionAuthorizationError):
            execute_authorized_task(plan, ResearchRun(), task, review, lambda *_: calls.append(True), state)
        self.assertEqual(calls, [])

    def test_wrong_plan_and_wrong_revision_are_rejected(self) -> None:
        plan, review, run, state, task, _ = authorized_setup()
        other_plan = make_plan(plan_id="other-plan")
        calls = []
        with self.assertRaisesRegex(TaskExecutionAuthorizationError, "plan_id"):
            execute_authorized_task(other_plan, run, other_plan.tasks[0], make_review(other_plan), lambda *_: calls.append(True), state)
        new_revision = make_plan(plan_id=plan.plan_id, revision=2)
        with self.assertRaisesRegex(TaskExecutionAuthorizationError, "plan_revision"):
            execute_authorized_task(new_revision, run, new_revision.tasks[0], review, lambda *_: calls.append(True), state)
        self.assertEqual(calls, [])

    def test_wrong_or_nonapproved_review_is_rejected(self) -> None:
        plan, review, run, state, task, _ = authorized_setup()
        wrong_review = make_review(plan, review_id="different-review")
        calls = []
        with self.assertRaisesRegex(TaskExecutionAuthorizationError, "review"):
            execute_authorized_task(plan, run, task, wrong_review, lambda *_: calls.append(True), state)

        pending = make_review(plan, review_id="pending-review", status=ReviewStatus.PENDING)
        pending_run = ResearchRun(
            plan_id=plan.plan_id, plan_revision=plan.revision,
            approval_review_id=pending.review_id, authorized_task_ids=[task.task_id],
        )
        state.add_run(pending_run)
        with self.assertRaisesRegex(TaskExecutionAuthorizationError, "does not approve"):
            execute_authorized_task(plan, pending_run, task, pending, lambda *_: calls.append(True), state)
        self.assertEqual(calls, [])

    def test_run_without_authorized_task_ids_is_rejected(self) -> None:
        plan, review, _, state, task, _ = authorized_setup()
        unselected_run = authorize_research_run(plan, review, state)
        calls = []
        with self.assertRaisesRegex(TaskExecutionAuthorizationError, "no explicitly authorized"):
            execute_authorized_task(plan, unselected_run, task, review, lambda *_: calls.append(True), state)
        self.assertEqual(calls, [])

    def test_invalid_plan_is_rejected_before_handler(self) -> None:
        plan, review, run, state, task, _ = authorized_setup()
        plan.tasks[0].dependency_ids.append("missing")
        calls = []
        with self.assertRaisesRegex(TaskExecutionAuthorizationError, "Invalid ResearchPlan"):
            execute_authorized_task(plan, run, task, review, lambda *_: calls.append(True), state)
        self.assertEqual(calls, [])

    def test_invalid_mutated_run_is_rejected_before_handler(self) -> None:
        plan, review, run, state, task, _ = authorized_setup()
        run.status = "paused"  # type: ignore[assignment]
        calls = []
        with self.assertRaisesRegex(TaskExecutionAuthorizationError, "ResearchRun validation"):
            execute_authorized_task(plan, run, task, review, lambda *_: calls.append(True), state)
        self.assertEqual(calls, [])

    def test_terminal_research_run_cannot_start_new_task_execution(self) -> None:
        plan, review, run, state, task, _ = authorized_setup()
        run.status = "completed"
        calls = []
        with self.assertRaisesRegex(TaskExecutionAuthorizationError, "does not allow task execution"):
            execute_authorized_task(plan, run, task, review, lambda *_: calls.append(True), state)
        self.assertEqual(calls, [])

    def test_execution_record_has_run_plan_task_question_and_timing_lineage(self) -> None:
        plan, review, run, state, task, iteration = authorized_setup(with_iteration=True)
        execution = execute_authorized_task(
            plan, run, task, review,
            lambda _task, _context: TaskExecutionResult(True, output="done"),
            state, iteration_id=iteration.iteration_id,
        )
        self.assertEqual(execution.run_id, run.run_id)
        self.assertEqual(execution.plan_id, plan.plan_id)
        self.assertEqual(execution.plan_revision, plan.revision)
        self.assertEqual(execution.task_id, task.task_id)
        self.assertEqual(execution.plan_question_id, task.question_id)
        self.assertEqual(execution.iteration_id, iteration.iteration_id)
        self.assertIsNotNone(execution.started_at)
        self.assertIsNotNone(execution.ended_at)

    def test_iteration_lineage_is_optional_and_only_resolved_when_supplied(self) -> None:
        plan, review, run, state, task, _ = authorized_setup()
        execution = execute_authorized_task(plan, run, task, review, lambda *_: TaskExecutionResult(True), state)
        self.assertIsNone(execution.iteration_id)

    def test_invalid_iteration_lineage_is_rejected(self) -> None:
        plan, review, run, state, task, _ = authorized_setup()
        other = state.add_run()
        other_iteration = state.create_iteration(other.run_id)
        calls = []
        with self.assertRaisesRegex(TaskExecutionAuthorizationError, "different ResearchRun"):
            execute_authorized_task(plan, run, task, review, lambda *_: calls.append(True), state,
                                    iteration_id=other_iteration.iteration_id)
        self.assertEqual(calls, [])

    def test_task_handler_receives_only_task_copy_and_small_context(self) -> None:
        plan, review, run, state, task, _ = authorized_setup()
        observed = {}

        def handler(received_task, context):
            observed["task"] = received_task
            observed["context"] = context
            return TaskExecutionResult(True)

        execution = execute_authorized_task(plan, run, task, review, handler, state)
        self.assertIsNot(observed["task"], task)
        self.assertIsInstance(observed["context"], TaskExecutionContext)
        self.assertEqual(observed["context"].execution_id, execution.execution_id)
        self.assertEqual(observed["context"].run_id, run.run_id)
        self.assertFalse(hasattr(observed["context"], "state"))

    def test_invalid_lifecycle_transitions_are_rejected(self) -> None:
        record = ResearchTaskExecution("run", "plan", 1, "task", "question")
        with self.assertRaisesRegex(ValueError, "Cannot complete execution from CREATED"):
            record.complete(TaskExecutionResult(True))
        record.start()
        record.complete(TaskExecutionResult(True, output="ok"))
        with self.assertRaisesRegex(ValueError, "Cannot start execution from COMPLETED"):
            record.start()
        with self.assertRaisesRegex(ValueError, "Cannot fail execution from COMPLETED"):
            record.fail("late failure")

    def test_handler_exception_is_recorded_failed_and_error_text_is_sanitized(self) -> None:
        plan, review, run, state, task, _ = authorized_setup()

        def handler(*_):
            raise ValueError("secret token=do-not-store")

        with self.assertRaises(TaskExecutionError) as raised:
            execute_authorized_task(plan, run, task, review, handler, state)
        failed = raised.exception.execution
        self.assertEqual(failed.status, TaskExecutionStatus.FAILED)
        self.assertFalse(failed.result.success)
        self.assertIsNone(failed.result.output)
        self.assertEqual(failed.result.error, "Handler raised ValueError.")
        self.assertNotIn("secret token", failed.result.error)
        self.assertIn(failed, state.task_executions)

    def test_handler_failure_result_is_retained_and_raises(self) -> None:
        plan, review, run, state, task, _ = authorized_setup()
        with self.assertRaises(TaskExecutionError) as raised:
            execute_authorized_task(
                plan, run, task, review,
                lambda *_: TaskExecutionResult(False, error="deterministic handler failure"),
                state,
            )
        record = raised.exception.execution
        self.assertEqual(record.status, TaskExecutionStatus.FAILED)
        self.assertEqual(record.result.error, "deterministic handler failure")
        self.assertIsNone(record.result.output)

    def test_invalid_handler_result_is_recorded_failed(self) -> None:
        plan, review, run, state, task, _ = authorized_setup()
        with self.assertRaisesRegex(TaskExecutionError, "invalid result") as raised:
            execute_authorized_task(plan, run, task, review, lambda *_: {"success": True}, state)  # type: ignore[arg-type]
        self.assertEqual(raised.exception.execution.status, TaskExecutionStatus.FAILED)

    def test_retry_creates_new_record_and_preserves_failed_history(self) -> None:
        plan, review, run, state, task, _ = authorized_setup()
        with self.assertRaises(TaskExecutionError) as failed_error:
            execute_authorized_task(plan, run, task, review, lambda *_: (_ for _ in ()).throw(RuntimeError("x")), state)
        first = failed_error.exception.execution
        snapshot = first.to_dict()
        second = execute_authorized_task(
            plan, run, task, review, lambda *_: TaskExecutionResult(True, output="retry ok"),
            state, retry_of_execution_id=first.execution_id,
        )
        self.assertNotEqual(first.execution_id, second.execution_id)
        self.assertEqual(first.to_dict(), snapshot)
        self.assertEqual(second.retry_of_execution_id, first.execution_id)
        self.assertEqual(second.status, TaskExecutionStatus.COMPLETED)
        self.assertEqual(len(state.task_executions), 2)

    def test_retry_requires_explicit_latest_failure_and_same_authorization(self) -> None:
        plan, review, run, state, task, _ = authorized_setup()
        with self.assertRaises(TaskExecutionError) as failed_error:
            execute_authorized_task(plan, run, task, review, lambda *_: TaskExecutionResult(False, error="failed"), state)
        first = failed_error.exception.execution
        calls = []
        with self.assertRaisesRegex(TaskExecutionAuthorizationError, "explicitly reference"):
            execute_authorized_task(plan, run, task, review, lambda *_: calls.append(True), state)
        self.assertEqual(calls, [])
        other_review = make_review(plan, review_id="another-review")
        with self.assertRaises(TaskExecutionAuthorizationError):
            execute_authorized_task(plan, run, task, other_review, lambda *_: calls.append(True), state,
                                    retry_of_execution_id=first.execution_id)
        self.assertEqual(calls, [])

    def test_retry_reference_to_unknown_or_wrong_task_is_rejected(self) -> None:
        plan, review, run, state, task, _ = authorized_setup()
        calls = []
        with self.assertRaisesRegex(TaskExecutionAuthorizationError, "does not reference"):
            execute_authorized_task(plan, run, task, review, lambda *_: calls.append(True), state,
                                    retry_of_execution_id="missing-execution")
        self.assertEqual(calls, [])

    def test_duplicate_active_execution_is_rejected(self) -> None:
        plan, review, run, state, task, _ = authorized_setup()
        active = ResearchTaskExecution(run.run_id, plan.plan_id, plan.revision, task.task_id, task.question_id)
        state.add_task_execution(active)
        calls = []
        with self.assertRaisesRegex(TaskExecutionAuthorizationError, "active execution"):
            execute_authorized_task(plan, run, task, review, lambda *_: calls.append(True), state)
        self.assertEqual(calls, [])

    def test_completed_execution_cannot_be_silently_repeated(self) -> None:
        plan, review, run, state, task, _ = authorized_setup()
        execute_authorized_task(plan, run, task, review, lambda *_: TaskExecutionResult(True), state)
        calls = []
        with self.assertRaisesRegex(TaskExecutionAuthorizationError, "completed task execution"):
            execute_authorized_task(plan, run, task, review, lambda *_: calls.append(True), state)
        self.assertEqual(calls, [])
        self.assertEqual(len(state.task_executions), 1)

    def test_planning_status_is_not_changed_by_execution(self) -> None:
        plan, review, run, state, task, _ = authorized_setup()
        status_before = task.status
        execute_authorized_task(plan, run, task, review, lambda *_: TaskExecutionResult(True), state)
        self.assertEqual(task.status, status_before)

    def test_state_round_trip_preserves_execution_history_and_result(self) -> None:
        plan, review, run, state, task, _ = authorized_setup()
        execute_authorized_task(
            plan, run, task, review,
            lambda *_: TaskExecutionResult(True, output="recorded", evidence_ids=("evidence-ref",)), state,
        )
        restored = ResearchState.from_dict(json.loads(json.dumps(state.to_dict())))
        execution = restored.task_executions[0]
        self.assertEqual(execution.status, TaskExecutionStatus.COMPLETED)
        self.assertEqual(execution.result.output, "recorded")
        self.assertEqual(execution.result.evidence_ids, ("evidence-ref",))
        self.assertEqual(execution.plan_question_id, task.question_id)

    def test_legacy_state_without_execution_collection_loads(self) -> None:
        state = ResearchState(user_request="Legacy")
        payload = state.to_dict()
        payload.pop("task_executions")
        self.assertEqual(ResearchState.from_dict(payload).task_executions, [])

    def test_malformed_execution_payload_is_rejected(self) -> None:
        state = ResearchState(user_request="Malformed")
        payload = state.to_dict()
        payload["task_executions"] = [{"execution_id": "incomplete"}]
        with self.assertRaisesRegex(ValueError, "Malformed ResearchTaskExecution"):
            ResearchState.from_dict(payload)

    def test_malformed_lifecycle_shape_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "COMPLETED execution requires"):
            ResearchTaskExecution(
                "run", "plan", 1, "task", "question", status=TaskExecutionStatus.COMPLETED,
            )

    def test_execution_result_contract_rejects_success_with_error_and_failed_output(self) -> None:
        with self.assertRaisesRegex(ValueError, "successful TaskExecutionResult cannot contain an error"):
            TaskExecutionResult(True, error="error")
        with self.assertRaisesRegex(ValueError, "cannot claim successful output"):
            TaskExecutionResult(False, output="claimed success", error="failed")

    def test_state_rejects_execution_not_bound_to_authorized_run(self) -> None:
        plan, review, run, state, task, _ = authorized_setup()
        invalid = ResearchTaskExecution(run.run_id, plan.plan_id, plan.revision, "other-task", task.question_id)
        with self.assertRaisesRegex(ValueError, "not authorized"):
            state.add_task_execution(invalid)

    def test_state_recording_failure_before_handler_rolls_back(self) -> None:
        class FailingAddState(ResearchState):
            def add_task_execution(self, execution):
                super().add_task_execution(execution)
                raise RuntimeError("simulated add failure")

        plan, review, run, state, task, _ = authorized_setup()
        failing = FailingAddState.from_dict(state.to_dict())
        failing.research_runs[0] = run
        calls = []
        before_updated_at = failing.updated_at
        with self.assertRaises(TaskExecutionRecordingError):
            execute_authorized_task(plan, run, task, review, lambda *_: calls.append(True), failing)
        self.assertEqual(calls, [])
        self.assertEqual(failing.task_executions, [])
        self.assertEqual(failing.updated_at, before_updated_at)

    def test_completion_recording_failure_is_retained_as_failed_not_success(self) -> None:
        class FailOnCompletionTouchState(ResearchState):
            def touch(self) -> None:
                self.touch_calls = getattr(self, "touch_calls", 0) + 1
                if self.touch_calls == 3:
                    self.updated_at = "partial-update"
                    raise RuntimeError("completion timestamp failed")
                self.updated_at = f"touch-{self.touch_calls}"

        plan, review, run, state, task, _ = authorized_setup()
        failing = FailOnCompletionTouchState.from_dict(state.to_dict())
        failing.research_runs[0] = run
        calls = []
        with self.assertRaises(TaskExecutionRecordingError) as raised:
            execute_authorized_task(
                plan, run, task, review,
                lambda *_: calls.append(True) or TaskExecutionResult(True, output="must not appear as success"),
                failing,
            )
        self.assertEqual(calls, [True])
        failed = raised.exception.execution
        self.assertEqual(failed.status, TaskExecutionStatus.FAILED)
        self.assertFalse(failed.result.success)
        self.assertIsNone(failed.result.output)
        self.assertEqual(failing.task_executions, [failed])
        self.assertEqual(failing.updated_at, "touch-2")

    def test_handler_not_called_when_run_not_registered_in_state(self) -> None:
        plan, review, run, _, task, _ = authorized_setup()
        calls = []
        with self.assertRaisesRegex(TaskExecutionAuthorizationError, "registered"):
            execute_authorized_task(plan, run, task, review, lambda *_: calls.append(True), ResearchState("empty"))
        self.assertEqual(calls, [])

    def test_execution_is_domain_agnostic(self) -> None:
        for domain in ("NLP", "computer vision", "cybersecurity", "medicine", "agriculture", "physics"):
            with self.subTest(domain=domain):
                plan, review, run, state, task, _ = authorized_setup(domain=domain)
                record = execute_authorized_task(plan, run, task, review,
                                                 lambda *_: TaskExecutionResult(True, output="ok"), state)
                self.assertEqual(record.plan_id, plan.plan_id)

    def test_execution_contract_does_not_call_llm_or_search_providers(self) -> None:
        # The only invoked callback is this local deterministic handler. The
        # context exposes no provider, state, or model handle.
        plan, review, run, state, task, _ = authorized_setup()
        record = execute_authorized_task(plan, run, task, review,
                                         lambda _task, context: TaskExecutionResult(True, output=context.task_id), state)
        self.assertEqual(record.result.output, task.task_id)
        self.assertFalse(hasattr(record, "retrieval_response"))
        self.assertFalse(hasattr(record, "llm_response"))

    def test_execution_record_json_dict_round_trip(self) -> None:
        plan, review, run, state, task, _ = authorized_setup()
        record = execute_authorized_task(plan, run, task, review,
                                         lambda *_: TaskExecutionResult(True, output="json"), state)
        restored = ResearchTaskExecution.from_dict(json.loads(json.dumps(record.to_dict())))
        self.assertEqual(restored.to_dict(), record.to_dict())


if __name__ == "__main__":
    unittest.main()
