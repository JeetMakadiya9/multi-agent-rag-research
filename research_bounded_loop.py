"""Bounded orchestration of caller-supplied research-loop commands.

This module deliberately contains no task/capability selection policy. Callers
provide an immutable execution policy and an ordered command sequence. Phase 5,
Phase 6, Phase 7A-C, and Phase 7D remain the authorities for execution,
capability behavior, authorization, loop transitions, and synthesis/critique.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
from typing import Optional, Sequence, Union
from uuid import uuid4

from research_capabilities import CapabilityType
from research_capability_dispatch import (
    CapabilityRequest,
    DispatchStatus,
    ResearchCapabilityDispatchResult,
    dispatch_research_capability,
)
from research_continuation import (
    ResearchContinuationAction,
    apply_research_continuation,
)
from research_loop import (
    LoopIterationStatus,
    LoopStage,
    ResearchLoop,
    ResearchLoopStatus,
    StopReason,
    TransitionAuthority,
    begin_loop_iteration,
    complete_loop_task,
    finish_loop_iteration,
    select_loop_task,
    start_research_loop,
    transition_research_loop,
)
from research_plan_review import PlanReview, is_execution_approved
from research_planning import ResearchPlan, ResearchTask
from research_state import ResearchRun, ResearchState
from research_synthesis_critique import (
    ResearchCritique,
    ResearchSynthesis,
    critique_research_synthesis,
    synthesize_research_state,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


def _enum(enum_type, value, name):
    try:
        return value if isinstance(value, enum_type) else enum_type(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unsupported {name}: {value!r}.") from exc


def _unique_strings(values, name, *, allow_empty=True):
    if not isinstance(values, (tuple, list)):
        raise ValueError(f"{name} must be a sequence of strings.")
    values = tuple(values)
    if not allow_empty and not values:
        raise ValueError(f"{name} must not be empty.")
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"{name} must contain non-empty strings.")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must not contain duplicates.")
    return values


class BoundedLoopOutcome(str, Enum):
    SEQUENCE_FINISHED = "SEQUENCE_FINISHED"
    WAITING_FOR_RESEARCHER = "WAITING_FOR_RESEARCHER"
    STOPPED = "STOPPED"
    FAILED = "FAILED"
    LIMIT_REACHED = "LIMIT_REACHED"
    BLOCKED = "BLOCKED"


class BoundedEventType(str, Enum):
    LOOP_STARTED = "LOOP_STARTED"
    STAGE_TRANSITIONED = "STAGE_TRANSITIONED"
    ITERATION_STARTED = "ITERATION_STARTED"
    ITERATION_FINISHED = "ITERATION_FINISHED"
    TASK_SELECTED = "TASK_SELECTED"
    CAPABILITY_DISPATCHED = "CAPABILITY_DISPATCHED"
    SYNTHESIS_CREATED = "SYNTHESIS_CREATED"
    CRITIQUE_CREATED = "CRITIQUE_CREATED"
    RESEARCHER_DECISION_APPLIED = "RESEARCHER_DECISION_APPLIED"
    TASK_MARKED_COMPLETE = "TASK_MARKED_COMPLETE"


@dataclass(frozen=True)
class BoundedExecutionPolicy:
    """Explicit limits and allowlists for one bounded command sequence."""

    maximum_iterations: int
    maximum_capability_invocations: int
    maximum_task_executions: int
    allowed_capabilities: tuple[CapabilityType, ...]
    allowed_stages: tuple[LoopStage, ...]
    allowed_task_ids: tuple[str, ...]
    allow_continuation: bool = False
    stop_on_failure: bool = True
    stop_on_unresolved_blocker: bool = True

    def __post_init__(self):
        for name in ("maximum_iterations", "maximum_capability_invocations", "maximum_task_executions"):
            value = getattr(self, name)
            minimum = 1 if name == "maximum_iterations" else 0
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        if not isinstance(self.allow_continuation, bool):
            raise ValueError("allow_continuation must be boolean.")
        if not isinstance(self.stop_on_failure, bool) or not isinstance(self.stop_on_unresolved_blocker, bool):
            raise ValueError("Policy stop flags must be boolean.")
        capabilities = tuple(_enum(CapabilityType, item, "allowed capability") for item in self.allowed_capabilities)
        stages = tuple(_enum(LoopStage, item, "allowed stage") for item in self.allowed_stages)
        if len(capabilities) != len(set(capabilities)) or len(stages) != len(set(stages)):
            raise ValueError("Policy capability/stage allowlists must not contain duplicates.")
        if not stages:
            raise ValueError("allowed_stages must not be empty.")
        object.__setattr__(self, "allowed_capabilities", capabilities)
        object.__setattr__(self, "allowed_stages", stages)
        object.__setattr__(self, "allowed_task_ids", _unique_strings(self.allowed_task_ids, "allowed_task_ids"))


@dataclass(frozen=True)
class StartLoop:
    step_id: str = field(default_factory=lambda: _id("step"))

    def __post_init__(self):
        _validate_step_id(self.step_id)


@dataclass(frozen=True)
class TransitionStage:
    target: LoopStage
    reason: str
    authority: TransitionAuthority
    stop_reason: Optional[StopReason] = None
    step_id: str = field(default_factory=lambda: _id("step"))

    def __post_init__(self):
        _validate_step_id(self.step_id)
        object.__setattr__(self, "target", _enum(LoopStage, self.target, "target stage"))
        object.__setattr__(self, "authority", _enum(TransitionAuthority, self.authority, "transition authority"))
        if self.stop_reason is not None:
            object.__setattr__(self, "stop_reason", _enum(StopReason, self.stop_reason, "stop reason"))
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("TransitionStage.reason must be non-empty.")


@dataclass(frozen=True)
class BeginIteration:
    task_ids: tuple[str, ...]
    step_id: str = field(default_factory=lambda: _id("step"))

    def __post_init__(self):
        _validate_step_id(self.step_id)
        object.__setattr__(self, "task_ids", _unique_strings(self.task_ids, "task_ids", allow_empty=False))


@dataclass(frozen=True)
class SelectTask:
    task_id: str
    step_id: str = field(default_factory=lambda: _id("step"))

    def __post_init__(self):
        _validate_step_id(self.step_id)
        _validate_id(self.task_id, "task_id")


@dataclass(frozen=True)
class DispatchCapability:
    task_id: str
    capability: CapabilityType
    request: CapabilityRequest
    retry_of_execution_id: Optional[str] = None
    continuation_step_id: Optional[str] = None
    step_id: str = field(default_factory=lambda: _id("step"))

    def __post_init__(self):
        _validate_step_id(self.step_id)
        _validate_id(self.task_id, "task_id")
        object.__setattr__(self, "capability", _enum(CapabilityType, self.capability, "capability"))
        if self.retry_of_execution_id is not None:
            _validate_id(self.retry_of_execution_id, "retry_of_execution_id")
        if self.continuation_step_id is not None:
            _validate_step_id(self.continuation_step_id)
        if self.retry_of_execution_id is not None and self.continuation_step_id is not None:
            raise ValueError("A dispatch cannot be both a retry and a continuation.")


@dataclass(frozen=True)
class Synthesize:
    step_id: str = field(default_factory=lambda: _id("step"))
    artifact_ids: Optional[tuple[str, ...]] = None

    def __post_init__(self):
        _validate_step_id(self.step_id)
        if self.artifact_ids is not None:
            object.__setattr__(self, "artifact_ids", _unique_strings(self.artifact_ids, "artifact_ids"))


@dataclass(frozen=True)
class Critique:
    synthesis_step_id: str
    step_id: str = field(default_factory=lambda: _id("step"))

    def __post_init__(self):
        _validate_step_id(self.step_id)
        _validate_step_id(self.synthesis_step_id)


@dataclass(frozen=True)
class ResearcherDecision:
    action: ResearchContinuationAction
    previous_execution_id: Optional[str] = None
    task_id: Optional[str] = None
    step_id: str = field(default_factory=lambda: _id("step"))

    def __post_init__(self):
        _validate_step_id(self.step_id)
        object.__setattr__(self, "action", _enum(ResearchContinuationAction, self.action, "researcher decision"))
        if self.previous_execution_id is not None:
            _validate_id(self.previous_execution_id, "previous_execution_id")
        if self.task_id is not None:
            _validate_id(self.task_id, "task_id")


@dataclass(frozen=True)
class FinishIteration:
    status: LoopIterationStatus
    reason: str
    step_id: str = field(default_factory=lambda: _id("step"))

    def __post_init__(self):
        _validate_step_id(self.step_id)
        object.__setattr__(self, "status", _enum(LoopIterationStatus, self.status, "iteration status"))
        if self.status is LoopIterationStatus.RUNNING:
            raise ValueError("FinishIteration requires a terminal iteration status.")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("FinishIteration.reason must be non-empty.")


@dataclass(frozen=True)
class MarkTaskComplete:
    task_id: str
    execution_id: str
    step_id: str = field(default_factory=lambda: _id("step"))

    def __post_init__(self):
        _validate_step_id(self.step_id)
        _validate_id(self.task_id, "task_id")
        _validate_id(self.execution_id, "execution_id")


LoopCommand = Union[
    StartLoop, TransitionStage, BeginIteration, SelectTask, DispatchCapability,
    Synthesize, Critique, ResearcherDecision, FinishIteration, MarkTaskComplete,
]


@dataclass(frozen=True)
class BoundedLoopEvent:
    step_id: str
    event_type: BoundedEventType
    stage: LoopStage
    iteration_id: Optional[str] = None
    task_id: Optional[str] = None
    execution_id: Optional[str] = None
    artifact_ids: tuple[str, ...] = ()
    record_id: Optional[str] = None
    message: str = ""

    def __post_init__(self):
        _validate_step_id(self.step_id)
        object.__setattr__(self, "event_type", _enum(BoundedEventType, self.event_type, "event type"))
        object.__setattr__(self, "stage", _enum(LoopStage, self.stage, "event stage"))
        for name in ("iteration_id", "task_id", "execution_id", "record_id"):
            value = getattr(self, name)
            if value is not None:
                _validate_id(value, name)
        object.__setattr__(self, "artifact_ids", _unique_strings(self.artifact_ids, "artifact_ids"))
        if not isinstance(self.message, str):
            raise ValueError("BoundedLoopEvent.message must be a string.")

    def to_dict(self):
        return {"step_id": self.step_id, "event_type": self.event_type.value, "stage": self.stage.value,
                "iteration_id": self.iteration_id, "task_id": self.task_id,
                "execution_id": self.execution_id, "artifact_ids": list(self.artifact_ids),
                "record_id": self.record_id, "message": self.message}

    @classmethod
    def from_dict(cls, data):
        keys = {"step_id", "event_type", "stage", "iteration_id", "task_id", "execution_id",
                "artifact_ids", "record_id", "message"}
        if not isinstance(data, dict) or set(data) != keys:
            raise ValueError("Malformed BoundedLoopEvent payload.")
        return cls(**{**data, "artifact_ids": tuple(data["artifact_ids"])})


@dataclass(frozen=True)
class BoundedLoopResult:
    loop_id: str
    run_id: str
    outcome: BoundedLoopOutcome
    loop_status: ResearchLoopStatus
    current_stage: LoopStage
    iterations_used: int
    capability_invocations_used: int
    task_executions_used: int
    events: tuple[BoundedLoopEvent, ...]
    dispatches: tuple[ResearchCapabilityDispatchResult, ...] = ()
    syntheses: tuple[ResearchSynthesis, ...] = ()
    critiques: tuple[ResearchCritique, ...] = ()
    error: Optional[str] = None
    result_id: str = field(default_factory=lambda: _id("bounded"))
    created_at: str = field(default_factory=_now)

    def __post_init__(self):
        for name in ("loop_id", "run_id", "result_id"):
            _validate_id(getattr(self, name), name)
        object.__setattr__(self, "outcome", _enum(BoundedLoopOutcome, self.outcome, "outcome"))
        object.__setattr__(self, "loop_status", _enum(ResearchLoopStatus, self.loop_status, "loop status"))
        object.__setattr__(self, "current_stage", _enum(LoopStage, self.current_stage, "current stage"))
        for name in ("iterations_used", "capability_invocations_used", "task_executions_used"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        for name, typ in (("events", BoundedLoopEvent), ("dispatches", ResearchCapabilityDispatchResult),
                          ("syntheses", ResearchSynthesis), ("critiques", ResearchCritique)):
            values = tuple(getattr(self, name))
            if any(not isinstance(item, typ) for item in values):
                raise ValueError(f"{name} must contain {typ.__name__} values.")
            object.__setattr__(self, name, values)
        if self.error is not None and (not isinstance(self.error, str) or not self.error.strip()):
            raise ValueError("error must be non-empty when supplied.")
        _validate_timestamp(self.created_at)

    def to_dict(self):
        return {"loop_id": self.loop_id, "run_id": self.run_id, "outcome": self.outcome.value,
                "loop_status": self.loop_status.value, "current_stage": self.current_stage.value,
                "iterations_used": self.iterations_used,
                "capability_invocations_used": self.capability_invocations_used,
                "task_executions_used": self.task_executions_used,
                "events": [item.to_dict() for item in self.events],
                "dispatches": [item.to_dict() for item in self.dispatches],
                "syntheses": [item.to_dict() for item in self.syntheses],
                "critiques": [item.to_dict() for item in self.critiques],
                "error": self.error, "result_id": self.result_id, "created_at": self.created_at}

    def to_json(self):
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, data):
        keys = {"loop_id", "run_id", "outcome", "loop_status", "current_stage", "iterations_used",
                "capability_invocations_used", "task_executions_used", "events", "dispatches",
                "syntheses", "critiques", "error", "result_id", "created_at"}
        if not isinstance(data, dict) or set(data) != keys:
            raise ValueError("Malformed BoundedLoopResult payload.")
        try:
            return cls(**{**data,
                          "events": tuple(BoundedLoopEvent.from_dict(item) for item in data["events"]),
                          "dispatches": tuple(ResearchCapabilityDispatchResult.from_dict(item)
                                               for item in data["dispatches"]),
                          "syntheses": tuple(ResearchSynthesis.from_dict(item) for item in data["syntheses"]),
                          "critiques": tuple(ResearchCritique.from_dict(item) for item in data["critiques"])})
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Malformed BoundedLoopResult: {exc}") from exc

    @classmethod
    def from_json(cls, payload):
        try:
            return cls.from_dict(json.loads(payload))
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Malformed BoundedLoopResult JSON: {exc}") from exc


def run_bounded_research_loop(
    plan: ResearchPlan,
    run: ResearchRun,
    review: PlanReview,
    loop: ResearchLoop,
    state: ResearchState,
    policy: BoundedExecutionPolicy,
    commands: Sequence[LoopCommand],
    *,
    retrieval_provider=None,
    llm_provider=None,
    search_provider=None,
) -> BoundedLoopResult:
    """Apply exactly the supplied commands, stopping at any explicit bound/gate.

    The sequence is not transactional: completed state changes and history are
    preserved if a later command fails, matching the underlying recorded
    execution semantics. Invalid context and command lists are rejected before
    the first state mutation.
    """
    _validate_context(plan, run, review, loop, state, policy)
    if not isinstance(commands, (tuple, list)):
        raise ValueError("commands must be an explicit ordered sequence.")
    commands = tuple(commands)
    if not commands:
        raise ValueError("commands must contain at least one explicit workflow action.")
    if any(not isinstance(item, (StartLoop, TransitionStage, BeginIteration, SelectTask,
                                 DispatchCapability, Synthesize, Critique, ResearcherDecision,
                                 FinishIteration, MarkTaskComplete)) for item in commands):
        raise ValueError("commands must contain only supported typed loop commands.")
    step_ids = [item.step_id for item in commands]
    if len(step_ids) != len(set(step_ids)):
        raise ValueError("Command step_id values must be unique.")
    _preflight_commands(plan, run, policy, commands)

    events = []
    dispatches = []
    syntheses = {}
    critiques = []
    continuation_results = {}
    explicitly_selected_tasks = set()
    failed_dispatch = False
    for index, command in enumerate(commands):
        current = _registered_loop(state, loop.loop_id)
        if current.status in {ResearchLoopStatus.COMPLETED, ResearchLoopStatus.STOPPED, ResearchLoopStatus.FAILED}:
            return _result(current, run, state, policy, events, dispatches, syntheses, critiques,
                           BoundedLoopOutcome.STOPPED if current.status is ResearchLoopStatus.STOPPED
                           else BoundedLoopOutcome.FAILED if current.status is ResearchLoopStatus.FAILED
                           else BoundedLoopOutcome.SEQUENCE_FINISHED)
        if policy.stop_on_unresolved_blocker and any(item.blocks_progress for item in current.unresolved_work):
            return _result(current, run, state, policy, events, dispatches, syntheses, critiques,
                           BoundedLoopOutcome.BLOCKED, error="A blocking UnresolvedWork item prevents progress.")
        if current.status is ResearchLoopStatus.WAITING_FOR_RESEARCHER and not isinstance(command, ResearcherDecision):
            return _result(current, run, state, policy, events, dispatches, syntheses, critiques,
                           BoundedLoopOutcome.WAITING_FOR_RESEARCHER)

        if isinstance(command, StartLoop):
            if current.current_stage not in policy.allowed_stages:
                raise ValueError("Current stage is outside the caller policy.")
            updated = start_research_loop(state, current.loop_id)
            events.append(_event(command, BoundedEventType.LOOP_STARTED, updated))
        elif isinstance(command, TransitionStage):
            if command.target not in policy.allowed_stages:
                raise ValueError(f"Stage {command.target.value} is outside the caller policy.")
            updated = transition_research_loop(
                state, current.loop_id, command.target, reason=command.reason,
                authority=command.authority, stop_reason=command.stop_reason,
            )
            events.append(_event(command, BoundedEventType.STAGE_TRANSITIONED, updated,
                                 message=command.reason))
        elif isinstance(command, BeginIteration):
            _require_stage(current, policy)
            if len(current.loop_iterations) >= policy.maximum_iterations:
                return _result(current, run, state, policy, events, dispatches, syntheses, critiques,
                               BoundedLoopOutcome.LIMIT_REACHED,
                               error="Caller maximum_iterations bound reached.")
            record = begin_loop_iteration(state, current.loop_id, command.task_ids)
            if record is None:
                current = _registered_loop(state, current.loop_id)
                return _result(current, run, state, policy, events, dispatches, syntheses, critiques,
                               BoundedLoopOutcome.LIMIT_REACHED,
                               error="ResearchLoop maximum_iterations bound reached.")
            updated = _registered_loop(state, current.loop_id)
            events.append(_event(command, BoundedEventType.ITERATION_STARTED, updated,
                                 iteration_id=record.iteration_id,
                                 message=f"Started iteration {record.iteration_number}."))
        elif isinstance(command, SelectTask):
            _require_stage(current, policy)
            updated = select_loop_task(state, current.loop_id, command.task_id)
            explicitly_selected_tasks.add(command.task_id)
            events.append(_event(command, BoundedEventType.TASK_SELECTED, updated, task_id=command.task_id))
        elif isinstance(command, DispatchCapability):
            _require_stage(current, policy)
            if command.task_id not in explicitly_selected_tasks:
                raise ValueError("Dispatch requires an explicit SelectTask or continuation decision in this command sequence.")
            if command.capability not in policy.allowed_capabilities:
                raise ValueError(f"Capability {command.capability.value} is outside the caller policy.")
            if len(current.capability_requests) >= policy.maximum_capability_invocations:
                return _result(current, run, state, policy, events, dispatches, syntheses, critiques,
                               BoundedLoopOutcome.LIMIT_REACHED,
                               error="Caller maximum_capability_invocations bound reached.")
            if _run_execution_count(state, run.run_id) >= policy.maximum_task_executions:
                return _result(current, run, state, policy, events, dispatches, syntheses, critiques,
                               BoundedLoopOutcome.LIMIT_REACHED,
                               error="Caller maximum_task_executions bound reached.")
            continuation_auth_id = None
            if command.continuation_step_id is not None:
                continuation = continuation_results.get(command.continuation_step_id)
                if continuation is None or continuation.continuation_authorization_id is None:
                    raise ValueError("Dispatch continuation_step_id must reference a preceding CONTINUE_SAME_TASK decision.")
                continuation_auth_id = continuation.continuation_authorization_id
            task = next(item for item in plan.tasks if item.task_id == command.task_id)
            dispatch = dispatch_research_capability(
                plan, run, task, review, current, state, capability=command.capability,
                capability_request=command.request, iteration_id=current.current_iteration_id,
                retrieval_provider=retrieval_provider, llm_provider=llm_provider,
                search_provider=search_provider, retry_of_execution_id=command.retry_of_execution_id,
                continuation_authorization_id=continuation_auth_id,
            )
            dispatches.append(dispatch)
            updated = _registered_loop(state, current.loop_id)
            events.append(_event(command, BoundedEventType.CAPABILITY_DISPATCHED, updated,
                                 task_id=command.task_id, execution_id=dispatch.execution_id,
                                 artifact_ids=dispatch.artifact_ids,
                                 message=dispatch.status.value))
            if dispatch.status is DispatchStatus.FAILED:
                failed_dispatch = True
                if policy.stop_on_failure or updated.status is ResearchLoopStatus.FAILED:
                    return _result(updated, run, state, policy, events, dispatches, syntheses, critiques,
                                   BoundedLoopOutcome.FAILED, error=dispatch.error)
        elif isinstance(command, Synthesize):
            _require_stage(current, policy)
            if current.current_stage is not LoopStage.SYNTHESIS:
                raise ValueError("Synthesis command requires the SYNTHESIS stage.")
            synthesis = synthesize_research_state(
                plan, run, review, current, state,
                iteration_id=current.current_iteration_id,
                artifact_ids=command.artifact_ids,
            )
            syntheses[command.step_id] = synthesis
            events.append(_event(command, BoundedEventType.SYNTHESIS_CREATED, current,
                                 record_id=synthesis.synthesis_id))
        elif isinstance(command, Critique):
            _require_stage(current, policy)
            if current.current_stage is not LoopStage.CRITIQUE:
                raise ValueError("Critique command requires the CRITIQUE stage.")
            synthesis = syntheses.get(command.synthesis_step_id)
            if synthesis is None:
                raise ValueError("Critique must reference an earlier synthesis step in this command sequence.")
            critique = critique_research_synthesis(
                plan, run, review, current, state, synthesis,
                iteration_id=current.current_iteration_id,
            )
            critiques.append(critique)
            events.append(_event(command, BoundedEventType.CRITIQUE_CREATED, current,
                                 record_id=critique.critique_id))
        elif isinstance(command, ResearcherDecision):
            if command.action in {ResearchContinuationAction.CONTINUE_SAME_TASK,
                                  ResearchContinuationAction.CONTINUE_WITH_TASK}:
                if not policy.allow_continuation:
                    raise ValueError("Caller policy does not allow continuation.")
                if len(current.loop_iterations) >= policy.maximum_iterations:
                    return _result(current, run, state, policy, events, dispatches, syntheses, critiques,
                                   BoundedLoopOutcome.LIMIT_REACHED,
                                   error="Caller maximum_iterations bound prevents continuation.")
            updated_result = apply_research_continuation(
                plan, run, review, current, state, action=command.action,
                previous_execution_id=command.previous_execution_id, task_id=command.task_id,
            )
            continuation_results[command.step_id] = updated_result
            updated = _registered_loop(state, current.loop_id)
            if updated_result.task_id is not None and command.action in {
                ResearchContinuationAction.CONTINUE_SAME_TASK,
                ResearchContinuationAction.CONTINUE_WITH_TASK,
            }:
                explicitly_selected_tasks.add(updated_result.task_id)
            events.append(_event(command, BoundedEventType.RESEARCHER_DECISION_APPLIED, updated,
                                 iteration_id=updated_result.iteration_id,
                                 task_id=updated_result.task_id,
                                 record_id=updated_result.continuation_authorization_id,
                                 message=command.action.value))
            if updated.status is ResearchLoopStatus.STOPPED:
                return _result(updated, run, state, policy, events, dispatches, syntheses, critiques,
                               BoundedLoopOutcome.STOPPED)
            if updated.status is ResearchLoopStatus.WAITING_FOR_RESEARCHER:
                return _result(updated, run, state, policy, events, dispatches, syntheses, critiques,
                               BoundedLoopOutcome.WAITING_FOR_RESEARCHER)
        elif isinstance(command, FinishIteration):
            _require_stage(current, policy)
            record = finish_loop_iteration(state, current.loop_id, status=command.status, reason=command.reason)
            updated = _registered_loop(state, current.loop_id)
            events.append(_event(command, BoundedEventType.ITERATION_FINISHED, updated,
                                 iteration_id=record.iteration_id, message=command.reason))
            if updated.status is ResearchLoopStatus.STOPPED:
                return _result(updated, run, state, policy, events, dispatches, syntheses, critiques,
                               BoundedLoopOutcome.LIMIT_REACHED,
                               error="ResearchLoop maximum_iterations bound reached.")
            if updated.status is ResearchLoopStatus.FAILED:
                return _result(updated, run, state, policy, events, dispatches, syntheses, critiques,
                               BoundedLoopOutcome.FAILED, error=command.reason)
        elif isinstance(command, MarkTaskComplete):
            _require_stage(current, policy)
            updated = complete_loop_task(state, current.loop_id, command.task_id, command.execution_id)
            events.append(_event(command, BoundedEventType.TASK_MARKED_COMPLETE, updated,
                                 task_id=command.task_id, execution_id=command.execution_id))

        current = _registered_loop(state, loop.loop_id)
        if current.status is ResearchLoopStatus.WAITING_FOR_RESEARCHER:
            # A gate is crossed only by a caller-supplied decision command.
            next_command = commands[index + 1] if index + 1 < len(commands) else None
            if not isinstance(next_command, ResearcherDecision):
                return _result(current, run, state, policy, events, dispatches, syntheses, critiques,
                               BoundedLoopOutcome.WAITING_FOR_RESEARCHER)
        if current.status is ResearchLoopStatus.STOPPED:
            return _result(current, run, state, policy, events, dispatches, syntheses, critiques,
                           BoundedLoopOutcome.STOPPED)
        if current.status is ResearchLoopStatus.FAILED:
            return _result(current, run, state, policy, events, dispatches, syntheses, critiques,
                           BoundedLoopOutcome.FAILED)

    current = _registered_loop(state, loop.loop_id)
    if current.status is ResearchLoopStatus.WAITING_FOR_RESEARCHER:
        outcome = BoundedLoopOutcome.WAITING_FOR_RESEARCHER
    elif current.status is ResearchLoopStatus.STOPPED:
        outcome = BoundedLoopOutcome.STOPPED
    elif current.status is ResearchLoopStatus.FAILED or failed_dispatch:
        outcome = BoundedLoopOutcome.FAILED
    else:
        outcome = BoundedLoopOutcome.SEQUENCE_FINISHED
    return _result(current, run, state, policy, events, dispatches, syntheses, critiques, outcome)


def _validate_context(plan, run, review, loop, state, policy):
    if not isinstance(plan, ResearchPlan) or plan.validate():
        raise ValueError("A structurally valid ResearchPlan is required.")
    if not isinstance(run, ResearchRun) or not isinstance(review, PlanReview):
        raise ValueError("ResearchRun and PlanReview are required.")
    if not isinstance(loop, ResearchLoop) or not isinstance(state, ResearchState):
        raise ValueError("A typed ResearchLoop and ResearchState are required.")
    if not isinstance(policy, BoundedExecutionPolicy):
        raise ValueError("A typed BoundedExecutionPolicy is required.")
    review.validate_against(plan)
    if not is_execution_approved(plan, review):
        raise ValueError("The supplied PlanReview does not approve this exact plan revision.")
    if (run.plan_id != plan.plan_id or run.plan_revision != plan.revision
            or run.approval_review_id != review.review_id
            or loop.run_id != run.run_id or loop.plan_id != plan.plan_id
            or loop.plan_revision != plan.revision):
        raise ValueError("Run, review, loop, and exact approved plan revision do not match.")
    if run.status not in {"created", "running"}:
        raise ValueError("ResearchRun status does not allow bounded execution.")
    if not any(item is run for item in state.research_runs):
        raise ValueError("ResearchRun must be registered in ResearchState.")
    if not any(item == loop for item in state.research_loops):
        raise ValueError("ResearchLoop must be registered in ResearchState.")
    if loop.validate_references(state):
        raise ValueError("ResearchLoop references are invalid.")
    if state.validate_lineage():
        raise ValueError("ResearchState lineage is invalid.")
    authorized = set(run.authorized_task_ids or ())
    plan_task_ids = {item.task_id for item in plan.tasks}
    if not authorized.issubset(plan_task_ids):
        raise ValueError("ResearchRun has task IDs outside the exact approved plan.")
    if not set(policy.allowed_task_ids).issubset(authorized):
        raise ValueError("Policy allowed_task_ids must be explicitly authorized on the supplied ResearchRun.")
    if policy.maximum_iterations > loop.max_iterations:
        raise ValueError("Caller policy cannot expand ResearchLoop.max_iterations.")
    if loop.status in {ResearchLoopStatus.COMPLETED, ResearchLoopStatus.STOPPED, ResearchLoopStatus.FAILED}:
        raise ValueError("Terminal ResearchLoop cannot be executed.")


def _preflight_commands(plan, run, policy, commands):
    task_ids = {item.task_id for item in plan.tasks}
    authorized = set(run.authorized_task_ids or ())
    for command in commands:
        refs = ()
        if isinstance(command, BeginIteration):
            refs = command.task_ids
        elif isinstance(command, (SelectTask, DispatchCapability, MarkTaskComplete)):
            refs = (command.task_id,)
        elif isinstance(command, ResearcherDecision) and command.task_id:
            refs = (command.task_id,)
        for task_id in refs:
            if task_id not in task_ids or task_id not in authorized:
                raise ValueError(f"Command references task {task_id!r} outside the exact authorized plan/run.")
            if task_id not in set(policy.allowed_task_ids):
                raise ValueError(f"Task {task_id!r} is outside the caller policy.")


def _registered_loop(state, loop_id):
    matches = [item for item in state.research_loops if item.loop_id == loop_id]
    if len(matches) != 1:
        raise ValueError(f"Unknown or duplicate ResearchLoop ID {loop_id!r}.")
    return matches[0]


def _require_stage(loop, policy):
    if loop.current_stage not in policy.allowed_stages:
        raise ValueError(f"Stage {loop.current_stage.value} is outside the caller policy.")
    if loop.status is not ResearchLoopStatus.RUNNING:
        raise ValueError("This command requires a RUNNING research loop.")


def _run_execution_count(state, run_id):
    return sum(item.run_id == run_id for item in state.task_executions)


def _event(command, event_type, loop, *, iteration_id=None, task_id=None,
           execution_id=None, artifact_ids=(), record_id=None, message=""):
    return BoundedLoopEvent(command.step_id, event_type, loop.current_stage,
                            iteration_id=iteration_id or loop.current_iteration_id,
                            task_id=task_id, execution_id=execution_id,
                            artifact_ids=tuple(artifact_ids), record_id=record_id, message=message)


def _result(loop, run, state, policy, events, dispatches, syntheses, critiques,
            outcome, error=None):
    return BoundedLoopResult(
        loop_id=loop.loop_id, run_id=run.run_id, outcome=outcome,
        loop_status=loop.status, current_stage=loop.current_stage,
        iterations_used=len(loop.loop_iterations),
        capability_invocations_used=len(loop.capability_requests),
        task_executions_used=_run_execution_count(state, run.run_id),
        events=tuple(events), dispatches=tuple(dispatches),
        syntheses=tuple(syntheses.values()), critiques=tuple(critiques), error=error,
    )


def _validate_id(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")


def _validate_step_id(value):
    _validate_id(value, "step_id")


def _validate_timestamp(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("created_at must be a timezone-aware timestamp.")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("created_at must be a timezone-aware timestamp.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("created_at must include a timezone.")


__all__ = [
    "BoundedLoopOutcome", "BoundedEventType", "BoundedExecutionPolicy", "StartLoop",
    "TransitionStage", "BeginIteration", "SelectTask", "DispatchCapability",
    "Synthesize", "Critique", "ResearcherDecision", "FinishIteration",
    "MarkTaskComplete", "LoopCommand", "BoundedLoopEvent", "BoundedLoopResult",
    "run_bounded_research_loop",
]
