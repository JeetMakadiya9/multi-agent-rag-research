"""Controlled, deterministic execution contract for explicitly authorized tasks.

Handlers are caller-supplied synchronous functions. They receive only a copy
of the selected plan task and a small immutable context, never ResearchState.
This module performs no search, retrieval, model inference, or orchestration.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple
from uuid import uuid4

from research_plan_review import PlanReview, is_execution_approved
from research_planning import ResearchPlan, ResearchTask
from research_state import ResearchRun, ResearchState


def _id() -> str:
    return f"execution_{uuid4().hex[:10]}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _valid_timestamp(value: object, name: str, *, optional: bool = False) -> None:
    if value is None and optional:
        return
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty ISO 8601 timestamp.")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid ISO 8601 timestamp.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must include a timezone.")


def _validate_ids(values: object, field_name: str) -> Tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{field_name} must be a list or tuple of strings.")
    normalized = tuple(values)
    for index, value in enumerate(normalized):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name}[{index}] must be a non-empty string.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} must not contain duplicate IDs.")
    return normalized


class TaskExecutionStatus(str, Enum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class TaskExecutionResult:
    """Typed handler result; success is separate from factual correctness."""

    success: bool
    output: Optional[str] = None
    error: Optional[str] = None
    artifact_ids: Tuple[str, ...] = ()
    evidence_ids: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.success, bool):
            raise ValueError("TaskExecutionResult.success must be a boolean.")
        if self.output is not None and not isinstance(self.output, str):
            raise ValueError("TaskExecutionResult.output must be a string when supplied.")
        if self.error is not None and (not isinstance(self.error, str) or not self.error.strip()):
            raise ValueError("TaskExecutionResult.error must be a non-empty string when supplied.")
        object.__setattr__(self, "artifact_ids", _validate_ids(self.artifact_ids, "artifact_ids"))
        object.__setattr__(self, "evidence_ids", _validate_ids(self.evidence_ids, "evidence_ids"))
        if self.success and self.error is not None:
            raise ValueError("A successful TaskExecutionResult cannot contain an error.")
        if not self.success:
            if self.error is None:
                raise ValueError("A failed TaskExecutionResult requires an error.")
            if self.output is not None:
                raise ValueError("A failed TaskExecutionResult cannot claim successful output.")

    def to_dict(self) -> Dict[str, object]:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "artifact_ids": list(self.artifact_ids),
            "evidence_ids": list(self.evidence_ids),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "TaskExecutionResult":
        required = {"success", "output", "error", "artifact_ids", "evidence_ids"}
        if not isinstance(data, dict) or set(data) != required:
            raise ValueError("Malformed TaskExecutionResult payload.")
        return cls(**data)  # type: ignore[arg-type]


@dataclass(frozen=True)
class TaskExecutionContext:
    """Narrow context supplied to a task handler; contains no ResearchState."""

    execution_id: str
    run_id: str
    plan_id: str
    plan_revision: int
    task_id: str
    plan_question_id: str
    iteration_id: Optional[str] = None

    def __post_init__(self) -> None:
        for name in ("execution_id", "run_id", "plan_id", "task_id", "plan_question_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"TaskExecutionContext.{name} must be a non-empty string.")
        if not isinstance(self.plan_revision, int) or isinstance(self.plan_revision, bool) or self.plan_revision < 1:
            raise ValueError("TaskExecutionContext.plan_revision must be a positive integer.")
        if self.iteration_id is not None and (not isinstance(self.iteration_id, str) or not self.iteration_id.strip()):
            raise ValueError("TaskExecutionContext.iteration_id must be a non-empty string when supplied.")


@dataclass
class ResearchTaskExecution:
    """Append-only execution attempt linked to an authorized run and task."""

    run_id: str
    plan_id: str
    plan_revision: int
    task_id: str
    plan_question_id: str
    execution_id: str = field(default_factory=_id)
    status: TaskExecutionStatus = TaskExecutionStatus.CREATED
    created_at: str = field(default_factory=_now)
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    iteration_id: Optional[str] = None
    retry_of_execution_id: Optional[str] = None
    continuation_of_execution_id: Optional[str] = None
    continuation_authorization_id: Optional[str] = None
    result: Optional[TaskExecutionResult] = None

    def __post_init__(self) -> None:
        for name in ("run_id", "plan_id", "task_id", "plan_question_id", "execution_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"ResearchTaskExecution.{name} must be a non-empty string.")
        if not isinstance(self.plan_revision, int) or isinstance(self.plan_revision, bool) or self.plan_revision < 1:
            raise ValueError("ResearchTaskExecution.plan_revision must be a positive integer.")
        if not isinstance(self.status, TaskExecutionStatus):
            try:
                self.status = TaskExecutionStatus(self.status)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Unsupported task execution status: {self.status!r}.") from exc
        _valid_timestamp(self.created_at, "ResearchTaskExecution.created_at")
        _valid_timestamp(self.started_at, "ResearchTaskExecution.started_at", optional=True)
        _valid_timestamp(self.ended_at, "ResearchTaskExecution.ended_at", optional=True)
        for name in ("iteration_id", "retry_of_execution_id", "continuation_of_execution_id",
                     "continuation_authorization_id"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"ResearchTaskExecution.{name} must be a non-empty string when supplied.")
        if self.retry_of_execution_id and (self.continuation_of_execution_id or self.continuation_authorization_id):
            raise ValueError("Retry and continuation references are mutually exclusive.")
        if bool(self.continuation_of_execution_id) != bool(self.continuation_authorization_id):
            raise ValueError("A continuation requires both continuation references.")
        if self.result is not None and not isinstance(self.result, TaskExecutionResult):
            raise ValueError("ResearchTaskExecution.result must be a TaskExecutionResult when supplied.")
        self._validate_lifecycle_shape()

    def _validate_lifecycle_shape(self) -> None:
        if not isinstance(self.status, TaskExecutionStatus):
            raise ValueError("ResearchTaskExecution.status is invalid.")
        if self.status is TaskExecutionStatus.CREATED:
            if self.started_at is not None or self.ended_at is not None or self.result is not None:
                raise ValueError("CREATED execution cannot have start/end timestamps or a result.")
        elif self.status is TaskExecutionStatus.RUNNING:
            if self.started_at is None or self.ended_at is not None or self.result is not None:
                raise ValueError("RUNNING execution requires started_at and no end/result.")
        elif self.status is TaskExecutionStatus.COMPLETED:
            if self.started_at is None or self.ended_at is None or self.result is None or not self.result.success:
                raise ValueError("COMPLETED execution requires timestamps and a successful result.")
        elif self.status is TaskExecutionStatus.FAILED:
            if self.started_at is None or self.ended_at is None or self.result is None or self.result.success:
                raise ValueError("FAILED execution requires timestamps and a failed result.")

    def start(self) -> None:
        if self.status is not TaskExecutionStatus.CREATED:
            raise ValueError(f"Cannot start execution from {self.status.value}.")
        self.status = TaskExecutionStatus.RUNNING
        self.started_at = _now()

    def complete(self, result: TaskExecutionResult) -> None:
        if self.status is not TaskExecutionStatus.RUNNING:
            raise ValueError(f"Cannot complete execution from {self.status.value}.")
        if not isinstance(result, TaskExecutionResult) or not result.success:
            raise ValueError("Completing an execution requires a successful TaskExecutionResult.")
        self.status = TaskExecutionStatus.COMPLETED
        self.ended_at = _now()
        self.result = result

    def fail(self, error: str) -> None:
        if self.status is not TaskExecutionStatus.RUNNING:
            raise ValueError(f"Cannot fail execution from {self.status.value}.")
        failure = TaskExecutionResult(success=False, error=error)
        self.status = TaskExecutionStatus.FAILED
        self.ended_at = _now()
        self.result = failure

    def to_dict(self) -> Dict[str, object]:
        return {
            "run_id": self.run_id,
            "plan_id": self.plan_id,
            "plan_revision": self.plan_revision,
            "task_id": self.task_id,
            "plan_question_id": self.plan_question_id,
            "execution_id": self.execution_id,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "iteration_id": self.iteration_id,
            "retry_of_execution_id": self.retry_of_execution_id,
            "continuation_of_execution_id": self.continuation_of_execution_id,
            "continuation_authorization_id": self.continuation_authorization_id,
            "result": self.result.to_dict() if self.result is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "ResearchTaskExecution":
        required = {
            "run_id", "plan_id", "plan_revision", "task_id", "plan_question_id",
            "execution_id", "status", "created_at", "started_at", "ended_at",
            "iteration_id", "retry_of_execution_id", "continuation_of_execution_id",
            "continuation_authorization_id", "result",
        }
        legacy_required = required - {"continuation_of_execution_id", "continuation_authorization_id"}
        if not isinstance(data, dict) or frozenset(data) not in {frozenset(required), frozenset(legacy_required)}:
            raise ValueError("Malformed ResearchTaskExecution payload.")
        values = dict(data)
        values.setdefault("continuation_of_execution_id", None)
        values.setdefault("continuation_authorization_id", None)
        if values["result"] is not None:
            if not isinstance(values["result"], dict):
                raise ValueError("Malformed ResearchTaskExecution.result payload.")
            values["result"] = TaskExecutionResult.from_dict(values["result"])
        try:
            return cls(**values)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ValueError(f"Malformed ResearchTaskExecution payload: {exc}") from exc


class TaskExecutionError(RuntimeError):
    """A handler or task result failed; the attached record remains in state."""

    def __init__(self, message: str, execution: ResearchTaskExecution):
        self.execution = execution
        super().__init__(message)


class TaskExecutionRecordingError(RuntimeError):
    """ResearchState could not safely record an execution transition."""

    def __init__(self, message: str, execution: Optional[ResearchTaskExecution] = None):
        self.execution = execution
        super().__init__(message)


class TaskExecutionAuthorizationError(ValueError):
    """The supplied task is not authorized for this exact run and plan."""


TaskHandler = Callable[[ResearchTask, TaskExecutionContext], TaskExecutionResult]


def execute_authorized_task(
    plan: ResearchPlan,
    run: ResearchRun,
    task: ResearchTask,
    review: PlanReview,
    handler: TaskHandler,
    state: ResearchState,
    *,
    iteration_id: Optional[str] = None,
    retry_of_execution_id: Optional[str] = None,
    continuation_authorization_id: Optional[str] = None,
) -> ResearchTaskExecution:
    """Validate authorization, record an attempt, then invoke one handler.

    The supplied handler is the only work boundary. It is invoked only after
    plan/run/review/task authorization and initial state recording succeed.
    Retries require an explicit reference to the latest failed attempt for the
    same run/task. Concurrent execution is not supported.
    """
    _validate_authorization(plan, run, task, review, handler, state)
    canonical_task = next(candidate for candidate in plan.tasks if candidate.task_id == task.task_id)
    if canonical_task != task:
        raise TaskExecutionAuthorizationError("Supplied ResearchTask does not match the task in the approved plan.")

    if not any(existing is run for existing in state.research_runs):
        raise TaskExecutionAuthorizationError("ResearchRun must be registered in ResearchState before execution.")
    if iteration_id is not None:
        iteration = next((item for item in state.iterations if item.iteration_id == iteration_id), None)
        if iteration is None:
            raise TaskExecutionAuthorizationError(f"Unknown ResearchIteration {iteration_id!r}.")
        if iteration.run_id != run.run_id:
            raise TaskExecutionAuthorizationError("ResearchIteration belongs to a different ResearchRun.")

    prior = [item for item in state.task_executions if item.run_id == run.run_id and item.task_id == task.task_id]
    if any(item.status in {TaskExecutionStatus.CREATED, TaskExecutionStatus.RUNNING} for item in prior):
        raise TaskExecutionAuthorizationError("An active execution already exists for this run/task.")
    continuation_of_execution_id = None
    if prior and prior[-1].status is TaskExecutionStatus.COMPLETED:
        if retry_of_execution_id is not None:
            raise TaskExecutionAuthorizationError("A completed execution cannot be referenced as a retry.")
        continuation = _resolve_continuation_authorization(
            state, continuation_authorization_id, plan, run, task, iteration_id, prior[-1],
        )
        continuation_of_execution_id = prior[-1].execution_id
    elif continuation_authorization_id is not None:
        raise TaskExecutionAuthorizationError("Continuation authorization requires a preceding completed execution.")
    if prior and prior[-1].status is TaskExecutionStatus.FAILED:
        latest = prior[-1]
        if continuation_authorization_id is not None:
            raise TaskExecutionAuthorizationError("A failed execution uses retry, not continuation semantics.")
        if retry_of_execution_id != latest.execution_id:
            raise TaskExecutionAuthorizationError(
                "Retry must explicitly reference the latest failed execution for this run/task."
            )
    elif not prior and retry_of_execution_id is not None:
        raise TaskExecutionAuthorizationError("retry_of_execution_id does not reference a prior failed attempt.")

    execution = ResearchTaskExecution(
        run_id=run.run_id,
        plan_id=plan.plan_id,
        plan_revision=plan.revision,
        task_id=task.task_id,
        plan_question_id=task.question_id,
        iteration_id=iteration_id,
        retry_of_execution_id=retry_of_execution_id,
        continuation_of_execution_id=continuation_of_execution_id,
        continuation_authorization_id=continuation_authorization_id,
    )
    _register_and_start(state, execution)
    context = TaskExecutionContext(
        execution_id=execution.execution_id,
        run_id=run.run_id,
        plan_id=plan.plan_id,
        plan_revision=plan.revision,
        task_id=task.task_id,
        plan_question_id=task.question_id,
        iteration_id=iteration_id,
    )

    try:
        result = handler(deepcopy(canonical_task), context)
    except Exception as exc:
        execution.fail(f"Handler raised {type(exc).__name__}.")
        _touch_after_terminal_transition(state, execution)
        raise TaskExecutionError(
            f"Task handler failed with {type(exc).__name__}; execution {execution.execution_id} is FAILED.",
            execution,
        ) from exc

    if not isinstance(result, TaskExecutionResult):
        execution.fail("Handler returned an invalid TaskExecutionResult.")
        _touch_after_terminal_transition(state, execution)
        raise TaskExecutionError("Task handler returned an invalid result; execution is FAILED.", execution)
    if not result.success:
        execution.fail(result.error or "Handler reported failure.")
        _touch_after_terminal_transition(state, execution)
        raise TaskExecutionError("Task handler reported failure; execution is FAILED.", execution)

    before = (execution.status, execution.ended_at, execution.result)
    previous_updated_at = state.updated_at
    execution.complete(result)
    try:
        state.touch()
    except Exception as exc:
        # Do not leave an unrecorded success-looking result in the state.
        state.updated_at = previous_updated_at
        execution.status, execution.ended_at, execution.result = before
        execution.fail("Successful handler result could not be recorded in ResearchState.")
        raise TaskExecutionRecordingError(
            f"Handler returned successfully, but completion recording failed: {type(exc).__name__}.",
            execution,
        ) from exc
    return execution


def _validate_authorization(
    plan: ResearchPlan,
    run: ResearchRun,
    task: ResearchTask,
    review: PlanReview,
    handler: TaskHandler,
    state: ResearchState,
) -> None:
    if not isinstance(plan, ResearchPlan):
        raise TaskExecutionAuthorizationError("A ResearchPlan is required.")
    errors = plan.validate()
    if errors:
        raise TaskExecutionAuthorizationError("Invalid ResearchPlan: " + "; ".join(errors))
    if not isinstance(run, ResearchRun):
        raise TaskExecutionAuthorizationError("A ResearchRun is required.")
    if not isinstance(task, ResearchTask):
        raise TaskExecutionAuthorizationError("A ResearchTask from the plan is required.")
    if not isinstance(review, PlanReview):
        raise TaskExecutionAuthorizationError("The approving PlanReview is required.")
    if not isinstance(state, ResearchState):
        raise TaskExecutionAuthorizationError("A ResearchState is required to retain execution history.")
    if not callable(handler):
        raise TaskExecutionAuthorizationError("An explicit callable task handler is required.")
    try:
        ResearchRun(
            run_id=run.run_id,
            created_at=run.created_at,
            status=run.status,
            metadata=dict(run.metadata),
            plan_id=run.plan_id,
            plan_revision=run.plan_revision,
            approval_review_id=run.approval_review_id,
            authorized_task_ids=list(run.authorized_task_ids) if run.authorized_task_ids is not None else None,
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise TaskExecutionAuthorizationError(f"ResearchRun validation failed: {exc}") from exc
    if not run.plan_id or run.plan_id != plan.plan_id:
        raise TaskExecutionAuthorizationError("ResearchRun plan_id does not match the supplied ResearchPlan.")
    if run.status not in {"created", "running"}:
        raise TaskExecutionAuthorizationError(
            f"ResearchRun status {run.status!r} does not allow task execution."
        )
    if run.plan_revision != plan.revision:
        raise TaskExecutionAuthorizationError("ResearchRun plan_revision does not match the supplied ResearchPlan.")
    if not run.approval_review_id:
        raise TaskExecutionAuthorizationError("ResearchRun has no approval_review_id.")
    if run.authorized_task_ids is None:
        raise TaskExecutionAuthorizationError("ResearchRun has no explicitly authorized task IDs.")
    try:
        parsed = PlanReview.from_dict(review.to_dict())
        if parsed != review:
            raise ValueError("Serialized PlanReview does not match the supplied review.")
        review.validate_against(plan)
    except (AttributeError, TypeError, ValueError) as exc:
        raise TaskExecutionAuthorizationError(f"PlanReview validation failed: {exc}") from exc
    if review.review_id != run.approval_review_id:
        raise TaskExecutionAuthorizationError("PlanReview ID does not match ResearchRun approval_review_id.")
    if not is_execution_approved(plan, review):
        raise TaskExecutionAuthorizationError("The supplied PlanReview does not approve this plan revision.")
    if task.task_id not in run.authorized_task_ids:
        raise TaskExecutionAuthorizationError(f"Task {task.task_id!r} is not authorized for this ResearchRun.")
    if task.task_id not in {candidate.task_id for candidate in plan.tasks}:
        raise TaskExecutionAuthorizationError(f"Task {task.task_id!r} is not present in the supplied ResearchPlan.")


def _resolve_continuation_authorization(state, authorization_id, plan, run, task, iteration_id, previous):
    if not authorization_id:
        raise TaskExecutionAuthorizationError(
            "A completed task execution cannot be silently repeated; explicit continuation authorization is required."
        )
    if previous.status is not TaskExecutionStatus.COMPLETED:
        raise TaskExecutionAuthorizationError("Continuation must reference a completed execution.")
    matches = []
    for loop in state.research_loops:
        matches.extend(
            item for item in getattr(loop, "continuation_authorizations", ())
            if item.authorization_id == authorization_id
        )
    if len(matches) != 1:
        raise TaskExecutionAuthorizationError("Unknown or duplicate continuation authorization.")
    record = matches[0]
    expected = (run.run_id, plan.plan_id, plan.revision, task.task_id, previous.execution_id, iteration_id)
    actual = (record.run_id, record.plan_id, record.plan_revision, record.task_id,
              record.previous_execution_id, record.iteration_id)
    if actual != expected:
        raise TaskExecutionAuthorizationError(
            "Continuation authorization does not match the exact run, plan revision, task, source execution, and iteration."
        )
    return record


def _register_and_start(state: ResearchState, execution: ResearchTaskExecution) -> None:
    previous_records = list(state.task_executions)
    previous_updated_at = state.updated_at
    try:
        state.add_task_execution(execution)
        execution.start()
        state.touch()
    except Exception as exc:
        state.task_executions[:] = previous_records
        state.updated_at = previous_updated_at
        raise TaskExecutionRecordingError(
            f"Execution could not be safely registered; handler was not invoked: {type(exc).__name__}.",
            execution,
        ) from exc


def _touch_after_terminal_transition(state: ResearchState, execution: ResearchTaskExecution) -> None:
    previous_updated_at = state.updated_at
    try:
        state.touch()
    except Exception as exc:
        state.updated_at = previous_updated_at
        raise TaskExecutionRecordingError(
            f"Execution {execution.execution_id} is {execution.status.value}, but state timestamp recording failed: "
            f"{type(exc).__name__}.",
            execution,
        ) from exc


__all__ = [
    "TaskExecutionStatus",
    "TaskExecutionResult",
    "TaskExecutionContext",
    "ResearchTaskExecution",
    "TaskExecutionError",
    "TaskExecutionRecordingError",
    "TaskExecutionAuthorizationError",
    "TaskHandler",
    "execute_authorized_task",
]
