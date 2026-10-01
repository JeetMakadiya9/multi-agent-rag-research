"""Deterministic, researcher-controlled orchestration contract.

This module records workflow state and typed capability boundaries only. It
does not call a capability, create research tasks, search, or make scientific
decisions. Existing Phase 4/5 authorization and Phase 6 capability modules
remain authoritative for those operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
import json
from typing import TYPE_CHECKING, Optional, Tuple
from uuid import uuid4

from research_capabilities import CapabilityType
from research_plan_review import PlanReview, is_execution_approved
from research_planning import ResearchPlan
from research_state import ResearchIteration, ResearchRun
from research_task_execution import ResearchTaskExecution, TaskExecutionStatus

if TYPE_CHECKING:
    from research_state import ResearchState


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:10]}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _timestamp(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a timezone-aware ISO-8601 timestamp.")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a timezone-aware ISO-8601 timestamp.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must include a timezone.")


def _enum(enum_type: type[Enum], value: object, field_name: str):
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        options = ", ".join(item.value for item in enum_type)
        raise ValueError(f"Unsupported {field_name} {value!r}; expected one of: {options}.") from exc


def _ids(values: object, field_name: str, *, allow_empty: bool = True) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{field_name} must be a sequence of IDs.")
    normalized = tuple(values)
    if not allow_empty and not normalized:
        raise ValueError(f"{field_name} must not be empty.")
    if any(not isinstance(item, str) or not item.strip() for item in normalized):
        raise ValueError(f"{field_name} must contain non-empty strings.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} must not contain duplicates.")
    return normalized


class LoopStage(str, Enum):
    INTAKE = "INTAKE"
    PLANNING = "PLANNING"
    RESEARCH = "RESEARCH"
    EVIDENCE_REVIEW = "EVIDENCE_REVIEW"
    VERIFICATION = "VERIFICATION"
    SYNTHESIS = "SYNTHESIS"
    CRITIQUE = "CRITIQUE"
    IMPROVEMENT = "IMPROVEMENT"
    PRIOR_WORK = "PRIOR_WORK"
    EXPERIMENT_PLANNING = "EXPERIMENT_PLANNING"
    RESEARCHER_REVIEW = "RESEARCHER_REVIEW"
    COMPLETED = "COMPLETED"
    STOPPED = "STOPPED"
    FAILED = "FAILED"


class ResearchLoopStatus(str, Enum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    WAITING_FOR_RESEARCHER = "WAITING_FOR_RESEARCHER"
    COMPLETED = "COMPLETED"
    STOPPED = "STOPPED"
    FAILED = "FAILED"


class StopReason(str, Enum):
    RESEARCHER_REQUESTED_STOP = "researcher_requested_stop"
    MAXIMUM_ITERATIONS_REACHED = "maximum_iterations_reached"
    NO_AUTHORIZED_WORK = "no_authorized_work"
    UNRESOLVED_BLOCKER = "unresolved_blocker"
    COMPLETED = "completed"
    FAILED = "failed"
    WAITING_FOR_RESEARCHER = "waiting_for_researcher"


class TransitionAuthority(str, Enum):
    SYSTEM = "system"
    RESEARCHER = "researcher"


class LoopIterationStatus(str, Enum):
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    STOPPED = "STOPPED"


class UnresolvedWorkKind(str, Enum):
    MISSING_EVIDENCE = "missing_evidence"
    UNRESOLVED_CONTRADICTION = "unresolved_contradiction"
    MISSING_AUTHORIZED_TASK = "missing_authorized_task"
    MISSING_RESEARCHER_DECISION = "missing_researcher_decision"
    BASELINE_UNRESOLVED = "baseline_unresolved"
    DATASET_UNRESOLVED = "dataset_unresolved"
    PRIOR_WORK_INCOMPLETE = "prior_work_incomplete"
    EXPERIMENT_PLAN_REQUIRES_REVIEW = "experiment_plan_requires_review"
    CAPABILITY_FAILURE = "capability_failure"
    OTHER = "other"


class CapabilityInvocationStatus(str, Enum):
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


LEGAL_TRANSITIONS: dict[LoopStage, frozenset[LoopStage]] = {
    LoopStage.INTAKE: frozenset({LoopStage.PLANNING, LoopStage.STOPPED, LoopStage.FAILED}),
    LoopStage.PLANNING: frozenset({LoopStage.RESEARCH, LoopStage.RESEARCHER_REVIEW, LoopStage.STOPPED, LoopStage.FAILED}),
    LoopStage.RESEARCH: frozenset({LoopStage.EVIDENCE_REVIEW, LoopStage.RESEARCHER_REVIEW, LoopStage.STOPPED, LoopStage.FAILED}),
    LoopStage.EVIDENCE_REVIEW: frozenset({LoopStage.VERIFICATION, LoopStage.RESEARCHER_REVIEW, LoopStage.STOPPED, LoopStage.FAILED}),
    LoopStage.VERIFICATION: frozenset({LoopStage.SYNTHESIS, LoopStage.RESEARCH, LoopStage.RESEARCHER_REVIEW, LoopStage.STOPPED, LoopStage.FAILED}),
    LoopStage.SYNTHESIS: frozenset({LoopStage.CRITIQUE, LoopStage.STOPPED, LoopStage.FAILED}),
    LoopStage.CRITIQUE: frozenset({LoopStage.RESEARCH, LoopStage.IMPROVEMENT, LoopStage.RESEARCHER_REVIEW, LoopStage.STOPPED, LoopStage.FAILED}),
    LoopStage.IMPROVEMENT: frozenset({LoopStage.PRIOR_WORK, LoopStage.RESEARCHER_REVIEW, LoopStage.STOPPED, LoopStage.FAILED}),
    LoopStage.PRIOR_WORK: frozenset({LoopStage.EXPERIMENT_PLANNING, LoopStage.RESEARCH, LoopStage.RESEARCHER_REVIEW, LoopStage.STOPPED, LoopStage.FAILED}),
    LoopStage.EXPERIMENT_PLANNING: frozenset({LoopStage.RESEARCHER_REVIEW, LoopStage.STOPPED, LoopStage.FAILED}),
    LoopStage.RESEARCHER_REVIEW: frozenset({LoopStage.RESEARCH, LoopStage.PLANNING, LoopStage.COMPLETED, LoopStage.STOPPED, LoopStage.FAILED}),
    LoopStage.COMPLETED: frozenset(),
    LoopStage.STOPPED: frozenset(),
    LoopStage.FAILED: frozenset(),
}


_CAPABILITY_STAGE = {
    LoopStage.RESEARCH: frozenset({CapabilityType.LOCAL_RETRIEVAL}),
    LoopStage.EVIDENCE_REVIEW: frozenset({CapabilityType.EVIDENCE_VERIFICATION}),
    LoopStage.VERIFICATION: frozenset({CapabilityType.EVIDENCE_VERIFICATION}),
    LoopStage.IMPROVEMENT: frozenset({CapabilityType.IMPROVEMENT_GENERATION}),
    LoopStage.PRIOR_WORK: frozenset({CapabilityType.PRIOR_WORK_INVESTIGATION}),
    LoopStage.EXPERIMENT_PLANNING: frozenset({CapabilityType.EXPERIMENT_PLANNING}),
}


def capability_compatible_with_stage(stage: LoopStage, capability: CapabilityType) -> bool:
    """Return whether this explicitly chosen capability is allowed at a stage."""
    stage = _enum(LoopStage, stage, "stage")
    capability = _enum(CapabilityType, capability, "capability")
    return capability in _CAPABILITY_STAGE.get(stage, frozenset())


@dataclass(frozen=True)
class LoopTransition:
    from_stage: LoopStage
    to_stage: LoopStage
    authority: TransitionAuthority
    reason: str
    transitioned_at: str = field(default_factory=_now)

    def __post_init__(self) -> None:
        object.__setattr__(self, "from_stage", _enum(LoopStage, self.from_stage, "transition.from_stage"))
        object.__setattr__(self, "to_stage", _enum(LoopStage, self.to_stage, "transition.to_stage"))
        object.__setattr__(self, "authority", _enum(TransitionAuthority, self.authority, "transition.authority"))
        if self.to_stage not in LEGAL_TRANSITIONS[self.from_stage]:
            raise ValueError(f"Illegal research-loop transition: {self.from_stage.value} -> {self.to_stage.value}.")
        if (self.from_stage is LoopStage.RESEARCHER_REVIEW
                and self.to_stage in {LoopStage.RESEARCH, LoopStage.PLANNING, LoopStage.COMPLETED}
                and self.authority is not TransitionAuthority.RESEARCHER):
            raise ValueError("Researcher-review continuation/completion requires researcher authority.")
        if self.to_stage is LoopStage.COMPLETED and self.authority is not TransitionAuthority.RESEARCHER:
            raise ValueError("Completion transitions require researcher authority.")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("LoopTransition.reason must be non-empty.")
        _timestamp(self.transitioned_at, "LoopTransition.transitioned_at")

    def to_dict(self) -> dict[str, str]:
        return {"from_stage": self.from_stage.value, "to_stage": self.to_stage.value,
                "authority": self.authority.value, "reason": self.reason,
                "transitioned_at": self.transitioned_at}

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "LoopTransition":
        keys = {"from_stage", "to_stage", "authority", "reason", "transitioned_at"}
        if not isinstance(data, dict) or set(data) != keys:
            raise ValueError("Malformed LoopTransition payload.")
        return cls(**data)  # type: ignore[arg-type]


@dataclass(frozen=True)
class LoopIterationRecord:
    """Loop-specific status for an existing ResearchState iteration ID."""

    iteration_id: str
    run_id: str
    iteration_number: int
    stage: LoopStage
    task_ids: Tuple[str, ...]
    status: LoopIterationStatus = LoopIterationStatus.RUNNING
    started_at: str = field(default_factory=_now)
    ended_at: Optional[str] = None
    transition_reason: str = ""

    def __post_init__(self) -> None:
        for name in ("iteration_id", "run_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"LoopIterationRecord.{name} must be non-empty.")
        if not isinstance(self.iteration_number, int) or isinstance(self.iteration_number, bool) or self.iteration_number < 1:
            raise ValueError("LoopIterationRecord.iteration_number must be positive.")
        object.__setattr__(self, "stage", _enum(LoopStage, self.stage, "LoopIterationRecord.stage"))
        object.__setattr__(self, "status", _enum(LoopIterationStatus, self.status, "LoopIterationRecord.status"))
        object.__setattr__(self, "task_ids", _ids(self.task_ids, "LoopIterationRecord.task_ids", allow_empty=False))
        _timestamp(self.started_at, "LoopIterationRecord.started_at")
        if self.ended_at is not None:
            _timestamp(self.ended_at, "LoopIterationRecord.ended_at")
        if not isinstance(self.transition_reason, str):
            raise ValueError("LoopIterationRecord.transition_reason must be a string.")
        if not self.transition_reason.strip():
            raise ValueError("LoopIterationRecord.transition_reason must be non-empty.")
        if self.stage in {LoopStage.INTAKE, LoopStage.COMPLETED, LoopStage.STOPPED, LoopStage.FAILED}:
            raise ValueError("LoopIterationRecord.stage must be an active workflow stage.")
        if (self.status is LoopIterationStatus.RUNNING) != (self.ended_at is None):
            raise ValueError("A running loop iteration has no end time; terminal iterations require one.")
        if self.ended_at is not None and datetime.fromisoformat(self.ended_at) < datetime.fromisoformat(self.started_at):
            raise ValueError("LoopIterationRecord.ended_at cannot precede started_at.")

    def to_dict(self) -> dict[str, object]:
        return {"iteration_id": self.iteration_id, "run_id": self.run_id,
                "iteration_number": self.iteration_number, "stage": self.stage.value,
                "task_ids": list(self.task_ids), "status": self.status.value,
                "started_at": self.started_at, "ended_at": self.ended_at,
                "transition_reason": self.transition_reason}

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "LoopIterationRecord":
        keys = {"iteration_id", "run_id", "iteration_number", "stage", "task_ids", "status",
                "started_at", "ended_at", "transition_reason"}
        if not isinstance(data, dict) or set(data) != keys:
            raise ValueError("Malformed LoopIterationRecord payload.")
        return cls(**data)  # type: ignore[arg-type]


@dataclass(frozen=True)
class UnresolvedWork:
    kind: UnresolvedWorkKind
    description: str
    task_id: Optional[str] = None
    artifact_id: Optional[str] = None
    blocks_progress: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _enum(UnresolvedWorkKind, self.kind, "UnresolvedWork.kind"))
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("UnresolvedWork.description must be non-empty.")
        for name in ("task_id", "artifact_id"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"UnresolvedWork.{name} must be non-empty when supplied.")
        if not isinstance(self.blocks_progress, bool):
            raise ValueError("UnresolvedWork.blocks_progress must be boolean.")

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind.value, "description": self.description,
                "task_id": self.task_id, "artifact_id": self.artifact_id,
                "blocks_progress": self.blocks_progress}

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "UnresolvedWork":
        keys = {"kind", "description", "task_id", "artifact_id", "blocks_progress"}
        if not isinstance(data, dict) or set(data) != keys:
            raise ValueError("Malformed UnresolvedWork payload.")
        return cls(**data)  # type: ignore[arg-type]


@dataclass(frozen=True)
class CapabilityInvocationRequest:
    capability: CapabilityType
    loop_id: str
    run_id: str
    task_id: str
    iteration_id: str
    invocation_id: str = field(default_factory=lambda: _id("capreq"))

    def __post_init__(self) -> None:
        object.__setattr__(self, "capability", _enum(CapabilityType, self.capability, "capability"))
        for name in ("loop_id", "run_id", "task_id", "iteration_id", "invocation_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"CapabilityInvocationRequest.{name} must be non-empty.")

    def to_dict(self) -> dict[str, str]:
        return {"capability": self.capability.value, "loop_id": self.loop_id, "run_id": self.run_id,
                "task_id": self.task_id, "iteration_id": self.iteration_id,
                "invocation_id": self.invocation_id}

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "CapabilityInvocationRequest":
        keys = {"capability", "loop_id", "run_id", "task_id", "iteration_id", "invocation_id"}
        if not isinstance(data, dict) or set(data) != keys:
            raise ValueError("Malformed CapabilityInvocationRequest payload.")
        return cls(**data)  # type: ignore[arg-type]


@dataclass(frozen=True)
class CapabilityInvocationResult:
    capability: CapabilityType
    invocation_id: str
    execution_id: str
    status: CapabilityInvocationStatus
    artifact_ids: Tuple[str, ...] = ()
    evidence_ids: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "capability", _enum(CapabilityType, self.capability, "capability"))
        object.__setattr__(self, "status", _enum(CapabilityInvocationStatus, self.status, "CapabilityInvocationResult.status"))
        for name in ("invocation_id", "execution_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"CapabilityInvocationResult.{name} must be non-empty.")
        object.__setattr__(self, "artifact_ids", _ids(self.artifact_ids, "CapabilityInvocationResult.artifact_ids"))
        object.__setattr__(self, "evidence_ids", _ids(self.evidence_ids, "CapabilityInvocationResult.evidence_ids"))
        if self.status is CapabilityInvocationStatus.COMPLETED and not self.artifact_ids:
            raise ValueError("Completed capability results must reference at least one stored artifact.")
        if self.status is CapabilityInvocationStatus.FAILED and self.artifact_ids:
            raise ValueError("Failed capability results cannot claim successful artifacts.")
        if self.status is CapabilityInvocationStatus.FAILED and self.evidence_ids:
            raise ValueError("Failed capability results cannot claim successful Evidence outputs.")

    def to_dict(self) -> dict[str, object]:
        return {"capability": self.capability.value, "invocation_id": self.invocation_id,
                "execution_id": self.execution_id, "status": self.status.value,
                "artifact_ids": list(self.artifact_ids), "evidence_ids": list(self.evidence_ids)}

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "CapabilityInvocationResult":
        keys = {"capability", "invocation_id", "execution_id", "status", "artifact_ids", "evidence_ids"}
        if not isinstance(data, dict) or set(data) != keys:
            raise ValueError("Malformed CapabilityInvocationResult payload.")
        return cls(**data)  # type: ignore[arg-type]


@dataclass(frozen=True)
class TaskContinuationAuthorization:
    """Caller-issued permission for one new execution after a completion."""

    run_id: str
    plan_id: str
    plan_revision: int
    task_id: str
    previous_execution_id: str
    iteration_id: str
    authorization_id: str = field(default_factory=lambda: _id("continue"))
    authorized_at: str = field(default_factory=_now)

    def __post_init__(self) -> None:
        for name in ("run_id", "plan_id", "task_id", "previous_execution_id", "iteration_id", "authorization_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"TaskContinuationAuthorization.{name} must be non-empty.")
        if not isinstance(self.plan_revision, int) or isinstance(self.plan_revision, bool) or self.plan_revision < 1:
            raise ValueError("TaskContinuationAuthorization.plan_revision must be positive.")
        _timestamp(self.authorized_at, "TaskContinuationAuthorization.authorized_at")

    def to_dict(self) -> dict[str, object]:
        return {"run_id": self.run_id, "plan_id": self.plan_id, "plan_revision": self.plan_revision,
                "task_id": self.task_id, "previous_execution_id": self.previous_execution_id,
                "iteration_id": self.iteration_id, "authorization_id": self.authorization_id,
                "authorized_at": self.authorized_at}

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "TaskContinuationAuthorization":
        keys = {"run_id", "plan_id", "plan_revision", "task_id", "previous_execution_id",
                "iteration_id", "authorization_id", "authorized_at"}
        if not isinstance(data, dict) or set(data) != keys:
            raise ValueError("Malformed TaskContinuationAuthorization payload.")
        return cls(**data)  # type: ignore[arg-type]


@dataclass(frozen=True)
class ResearchLoop:
    run_id: str
    plan_id: str
    plan_revision: int
    max_iterations: int
    loop_id: str = field(default_factory=lambda: _id("loop"))
    current_stage: LoopStage = LoopStage.INTAKE
    status: ResearchLoopStatus = ResearchLoopStatus.CREATED
    current_task_id: Optional[str] = None
    current_iteration_id: Optional[str] = None
    iteration_number: int = 0
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    stop_reason: Optional[StopReason] = None
    unresolved_task_ids: Tuple[str, ...] = ()
    completed_task_ids: Tuple[str, ...] = ()
    loop_iterations: Tuple[LoopIterationRecord, ...] = ()
    transitions: Tuple[LoopTransition, ...] = ()
    unresolved_work: Tuple[UnresolvedWork, ...] = ()
    capability_requests: Tuple[CapabilityInvocationRequest, ...] = ()
    capability_results: Tuple[CapabilityInvocationResult, ...] = ()
    continuation_authorizations: Tuple[TaskContinuationAuthorization, ...] = ()

    def __post_init__(self) -> None:
        for name in ("loop_id", "run_id", "plan_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"ResearchLoop.{name} must be non-empty.")
        if not isinstance(self.plan_revision, int) or isinstance(self.plan_revision, bool) or self.plan_revision < 1:
            raise ValueError("ResearchLoop.plan_revision must be positive.")
        if not isinstance(self.max_iterations, int) or isinstance(self.max_iterations, bool) or self.max_iterations < 1:
            raise ValueError("ResearchLoop.max_iterations must be a positive integer supplied by the caller.")
        object.__setattr__(self, "current_stage", _enum(LoopStage, self.current_stage, "ResearchLoop.current_stage"))
        object.__setattr__(self, "status", _enum(ResearchLoopStatus, self.status, "ResearchLoop.status"))
        if self.stop_reason is not None:
            object.__setattr__(self, "stop_reason", _enum(StopReason, self.stop_reason, "ResearchLoop.stop_reason"))
        for name in ("current_task_id", "current_iteration_id"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"ResearchLoop.{name} must be non-empty when supplied.")
        if not isinstance(self.iteration_number, int) or isinstance(self.iteration_number, bool) or self.iteration_number < 0:
            raise ValueError("ResearchLoop.iteration_number must be a non-negative integer.")
        _timestamp(self.created_at, "ResearchLoop.created_at")
        _timestamp(self.updated_at, "ResearchLoop.updated_at")
        if datetime.fromisoformat(self.updated_at) < datetime.fromisoformat(self.created_at):
            raise ValueError("ResearchLoop.updated_at cannot precede created_at.")
        for name in ("unresolved_task_ids", "completed_task_ids"):
            object.__setattr__(self, name, _ids(getattr(self, name), f"ResearchLoop.{name}"))
        if set(self.unresolved_task_ids) & set(self.completed_task_ids):
            raise ValueError("A task cannot be both unresolved and completed.")
        specs = (("loop_iterations", LoopIterationRecord), ("transitions", LoopTransition),
                 ("unresolved_work", UnresolvedWork), ("capability_requests", CapabilityInvocationRequest),
                 ("capability_results", CapabilityInvocationResult),
                 ("continuation_authorizations", TaskContinuationAuthorization))
        for name, kind in specs:
            raw = getattr(self, name)
            if not isinstance(raw, (tuple, list)) or any(not isinstance(item, kind) for item in raw):
                raise ValueError(f"ResearchLoop.{name} must contain typed {kind.__name__} records.")
            object.__setattr__(self, name, tuple(raw))
        if len(self.loop_iterations) > self.max_iterations:
            raise ValueError("ResearchLoop exceeds max_iterations.")
        ids = [item.iteration_id for item in self.loop_iterations]
        numbers = [item.iteration_number for item in self.loop_iterations]
        if len(ids) != len(set(ids)) or len(numbers) != len(set(numbers)):
            raise ValueError("ResearchLoop iteration references/numbers must be unique.")
        if numbers != sorted(numbers):
            raise ValueError("ResearchLoop iterations must be stored in order.")
        if self.iteration_number != (numbers[-1] if numbers else 0):
            raise ValueError("ResearchLoop.iteration_number must match its latest recorded iteration.")
        running = [item for item in self.loop_iterations if item.status is LoopIterationStatus.RUNNING]
        if len(running) > 1:
            raise ValueError("ResearchLoop cannot have concurrent active iterations.")
        if self.current_iteration_id != (running[0].iteration_id if running else None):
            raise ValueError("ResearchLoop.current_iteration_id must match its active iteration record.")
        request_ids = [item.invocation_id for item in self.capability_requests]
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("ResearchLoop capability invocation IDs must be unique.")
        result_ids = [item.invocation_id for item in self.capability_results]
        if len(result_ids) != len(set(result_ids)) or not set(result_ids).issubset(set(request_ids)):
            raise ValueError("ResearchLoop capability results must uniquely resolve to invocation requests.")
        continuation_ids = [item.authorization_id for item in self.continuation_authorizations]
        if len(continuation_ids) != len(set(continuation_ids)):
            raise ValueError("ResearchLoop continuation authorization IDs must be unique.")
        self._validate_transition_history()
        self._validate_status_shape()

    def _validate_transition_history(self) -> None:
        stage = LoopStage.INTAKE
        for transition in self.transitions:
            if transition.from_stage is not stage:
                raise ValueError("ResearchLoop transition history is discontinuous.")
            if transition.to_stage not in LEGAL_TRANSITIONS[stage]:
                raise ValueError("ResearchLoop transition history contains an illegal transition.")
            stage = transition.to_stage
        if stage is not self.current_stage:
            raise ValueError("ResearchLoop current_stage does not match its transition history.")

    def _validate_status_shape(self) -> None:
        if self.status is ResearchLoopStatus.CREATED:
            valid = self.current_stage is LoopStage.INTAKE and self.stop_reason is None and self.current_iteration_id is None
        elif self.status is ResearchLoopStatus.RUNNING:
            valid = self.current_stage not in {LoopStage.INTAKE, LoopStage.RESEARCHER_REVIEW, LoopStage.COMPLETED, LoopStage.STOPPED, LoopStage.FAILED} and self.stop_reason is None
        elif self.status is ResearchLoopStatus.WAITING_FOR_RESEARCHER:
            valid = self.current_stage is LoopStage.RESEARCHER_REVIEW and self.stop_reason is StopReason.WAITING_FOR_RESEARCHER
        elif self.status is ResearchLoopStatus.COMPLETED:
            valid = self.current_stage is LoopStage.COMPLETED and self.stop_reason is StopReason.COMPLETED and self.current_iteration_id is None
        elif self.status is ResearchLoopStatus.STOPPED:
            valid = self.current_stage is LoopStage.STOPPED and self.stop_reason in {
                StopReason.RESEARCHER_REQUESTED_STOP, StopReason.MAXIMUM_ITERATIONS_REACHED,
                StopReason.NO_AUTHORIZED_WORK, StopReason.UNRESOLVED_BLOCKER,
            } and self.current_iteration_id is None
        else:
            valid = self.current_stage is LoopStage.FAILED and self.stop_reason is StopReason.FAILED and self.current_iteration_id is None
        if self.current_stage is LoopStage.STOPPED and self.stop_reason is StopReason.RESEARCHER_REQUESTED_STOP:
            valid = valid and bool(self.transitions) and self.transitions[-1].authority is TransitionAuthority.RESEARCHER
        if self.stop_reason is StopReason.MAXIMUM_ITERATIONS_REACHED:
            valid = valid and len(self.loop_iterations) >= self.max_iterations
        if not valid:
            raise ValueError("ResearchLoop status, current_stage, and stop_reason are inconsistent.")

    def validate_references(self, state: "ResearchState") -> list[str]:
        errors: list[str] = []
        runs = {item.run_id: item for item in state.research_runs}
        run = runs.get(self.run_id)
        if run is None:
            return [f"ResearchLoop {self.loop_id!r} references a missing ResearchRun."]
        if run.plan_id != self.plan_id or run.plan_revision != self.plan_revision or not run.approval_review_id:
            errors.append(f"ResearchLoop {self.loop_id!r} plan/revision/approval does not match its ResearchRun.")
        authorized = set(run.authorized_task_ids or ())
        referenced_tasks = set(self.unresolved_task_ids) | set(self.completed_task_ids)
        if self.current_task_id:
            referenced_tasks.add(self.current_task_id)
        referenced_tasks.update(item.task_id for item in self.unresolved_work if item.task_id is not None)
        if not referenced_tasks.issubset(authorized):
            errors.append(f"ResearchLoop {self.loop_id!r} references tasks not authorized for its run.")
        if self.stop_reason is StopReason.NO_AUTHORIZED_WORK and authorized:
            errors.append(f"ResearchLoop {self.loop_id!r} claims no authorized work despite authorized tasks.")
        if self.stop_reason is StopReason.UNRESOLVED_BLOCKER and not any(item.blocks_progress for item in self.unresolved_work):
            errors.append(f"ResearchLoop {self.loop_id!r} claims an unresolved blocker without a blocking work item.")
        artifacts = {item.artifact_id: item for item in state.research_artifacts}
        existing_iterations = {item.iteration_id: item for item in state.iterations}
        for record in self.loop_iterations:
            iteration = existing_iterations.get(record.iteration_id)
            if iteration is None or iteration.run_id != self.run_id or iteration.number != record.iteration_number:
                errors.append(f"ResearchLoop {self.loop_id!r} has an invalid ResearchIteration reference.")
            if not set(record.task_ids).issubset(authorized):
                errors.append(f"ResearchLoop {self.loop_id!r} iteration has unauthorized task IDs.")
        for work in self.unresolved_work:
            if work.artifact_id:
                artifact = artifacts.get(work.artifact_id)
                if artifact is None or artifact.run_id != self.run_id:
                    errors.append(f"ResearchLoop {self.loop_id!r} unresolved work has an invalid artifact reference.")
        if self.current_task_id and self.current_iteration_id:
            current = next((item for item in self.loop_iterations if item.iteration_id == self.current_iteration_id), None)
            if current is None or self.current_task_id not in current.task_ids:
                errors.append(f"ResearchLoop {self.loop_id!r} current task is not part of its current iteration.")
        iterations = {item.iteration_id for item in self.loop_iterations}
        for request in self.capability_requests:
            if request.loop_id != self.loop_id or request.run_id != self.run_id or request.task_id not in authorized or request.iteration_id not in iterations:
                errors.append(f"ResearchLoop {self.loop_id!r} has a mismatched capability request.")
        requests = {item.invocation_id: item for item in self.capability_requests}
        executions = {item.execution_id: item for item in state.task_executions}
        artifacts = {item.artifact_id: item for item in state.research_artifacts}
        evidence = {item.evidence_id: item for item in state.evidence}
        for result in self.capability_results:
            request = requests.get(result.invocation_id)
            execution = executions.get(result.execution_id)
            expected_status = TaskExecutionStatus.COMPLETED if result.status is CapabilityInvocationStatus.COMPLETED else TaskExecutionStatus.FAILED
            if request is None or request.capability is not result.capability or execution is None:
                errors.append(f"ResearchLoop {self.loop_id!r} has an unresolved capability result.")
                continue
            if (execution.run_id != self.run_id or execution.task_id != request.task_id
                    or execution.iteration_id != request.iteration_id or execution.status is not expected_status):
                errors.append(f"ResearchLoop {self.loop_id!r} capability result does not match its Phase 5 execution.")
            for artifact_id in result.artifact_ids:
                artifact = artifacts.get(artifact_id)
                if artifact is None or artifact.execution_id != execution.execution_id or artifact.run_id != self.run_id or artifact.task_id != request.task_id:
                    errors.append(f"ResearchLoop {self.loop_id!r} capability result references an invalid artifact.")
            for evidence_id in result.evidence_ids:
                item = evidence.get(evidence_id)
                if item is None:
                    errors.append(f"ResearchLoop {self.loop_id!r} capability result references missing Evidence.")
        for authorization in self.continuation_authorizations:
            source = executions.get(authorization.previous_execution_id)
            iteration = next((item for item in self.loop_iterations if item.iteration_id == authorization.iteration_id), None)
            if (authorization.run_id != self.run_id or authorization.plan_id != self.plan_id
                    or authorization.plan_revision != self.plan_revision
                    or authorization.task_id not in authorized or source is None
                    or source.status is not TaskExecutionStatus.COMPLETED
                    or source.run_id != self.run_id or source.plan_id != self.plan_id
                    or source.plan_revision != self.plan_revision or source.task_id != authorization.task_id
                    or iteration is None or authorization.task_id not in iteration.task_ids):
                errors.append(f"ResearchLoop {self.loop_id!r} has an invalid continuation authorization.")
        return errors

    def to_dict(self) -> dict[str, object]:
        return {
            "loop_id": self.loop_id, "run_id": self.run_id, "plan_id": self.plan_id,
            "plan_revision": self.plan_revision, "current_stage": self.current_stage.value,
            "status": self.status.value, "current_task_id": self.current_task_id,
            "current_iteration_id": self.current_iteration_id, "iteration_number": self.iteration_number,
            "created_at": self.created_at, "updated_at": self.updated_at,
            "stop_reason": self.stop_reason.value if self.stop_reason else None,
            "unresolved_task_ids": list(self.unresolved_task_ids), "completed_task_ids": list(self.completed_task_ids),
            "max_iterations": self.max_iterations,
            "loop_iterations": [item.to_dict() for item in self.loop_iterations],
            "transitions": [item.to_dict() for item in self.transitions],
            "unresolved_work": [item.to_dict() for item in self.unresolved_work],
            "capability_requests": [item.to_dict() for item in self.capability_requests],
            "capability_results": [item.to_dict() for item in self.capability_results],
            "continuation_authorizations": [item.to_dict() for item in self.continuation_authorizations],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "ResearchLoop":
        keys = {"loop_id", "run_id", "plan_id", "plan_revision", "current_stage", "status",
                "current_task_id", "current_iteration_id", "iteration_number", "created_at", "updated_at",
                "stop_reason", "unresolved_task_ids", "completed_task_ids", "max_iterations",
                "loop_iterations", "transitions", "unresolved_work", "capability_requests", "capability_results"}
        if not isinstance(data, dict) or set(data) not in (keys, keys | {"continuation_authorizations"}):
            raise ValueError("Malformed ResearchLoop payload.")
        values = dict(data)
        values.setdefault("continuation_authorizations", [])
        for name, parser in (("loop_iterations", LoopIterationRecord.from_dict), ("transitions", LoopTransition.from_dict),
                             ("unresolved_work", UnresolvedWork.from_dict),
                             ("capability_requests", CapabilityInvocationRequest.from_dict),
                             ("capability_results", CapabilityInvocationResult.from_dict)):
            raw = values[name]
            if not isinstance(raw, (list, tuple)):
                raise ValueError(f"Malformed ResearchLoop.{name}.")
            values[name] = tuple(parser(item) for item in raw)
        raw = values["continuation_authorizations"]
        if not isinstance(raw, (list, tuple)):
            raise ValueError("Malformed ResearchLoop.continuation_authorizations.")
        values["continuation_authorizations"] = tuple(TaskContinuationAuthorization.from_dict(item) for item in raw)
        for name in ("unresolved_task_ids", "completed_task_ids"):
            if not isinstance(values[name], (list, tuple)):
                raise ValueError(f"Malformed ResearchLoop.{name}.")
            values[name] = tuple(values[name])
        try:
            return cls(**values)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Malformed ResearchLoop payload: {exc}") from exc

    @classmethod
    def from_json(cls, payload: str) -> "ResearchLoop":
        try:
            data = json.loads(payload)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Malformed ResearchLoop JSON: {exc}") from exc
        return cls.from_dict(data)


def create_research_loop(plan: ResearchPlan, run: ResearchRun, review: PlanReview,
                         state: "ResearchState", *, max_iterations: int) -> ResearchLoop:
    """Register a loop bound to the exact already-approved plan/run."""
    from research_state import ResearchState
    if not isinstance(plan, ResearchPlan) or plan.validate():
        raise ValueError("A structurally valid ResearchPlan is required to create a loop.")
    if not isinstance(run, ResearchRun) or not isinstance(review, PlanReview) or not isinstance(state, ResearchState):
        raise ValueError("create_research_loop requires ResearchRun, PlanReview, and ResearchState.")
    try:
        validated_run = ResearchRun(
            run_id=run.run_id, created_at=run.created_at, status=run.status, metadata=dict(run.metadata),
            plan_id=run.plan_id, plan_revision=run.plan_revision,
            approval_review_id=run.approval_review_id,
            authorized_task_ids=list(run.authorized_task_ids) if run.authorized_task_ids is not None else None,
        )
        validated_review = PlanReview.from_dict(review.to_dict())
        if validated_run != run or validated_review != review:
            raise ValueError("Serialized run/review differs from the supplied record.")
        review.validate_against(plan)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"Research loop authorization validation failed: {exc}") from exc
    if run.status in {"completed", "failed"}:
        raise ValueError("A completed or failed ResearchRun cannot start a research loop.")
    if not is_execution_approved(plan, review):
        raise ValueError("Research loop requires explicit approval of this exact plan revision.")
    if run.plan_id != plan.plan_id or run.plan_revision != plan.revision or run.approval_review_id != review.review_id:
        raise ValueError("ResearchRun does not reference the exact approved plan revision and review.")
    if not any(item is run for item in state.research_runs):
        raise ValueError("ResearchRun must already be registered in ResearchState.")
    if run.authorized_task_ids is not None:
        plan_task_ids = {item.task_id for item in plan.tasks}
        if not set(run.authorized_task_ids).issubset(plan_task_ids):
            raise ValueError("ResearchRun contains task IDs outside the supplied plan.")
    loop = ResearchLoop(run_id=run.run_id, plan_id=plan.plan_id, plan_revision=plan.revision,
                        max_iterations=max_iterations,
                        unresolved_task_ids=tuple(run.authorized_task_ids or ()))
    old_events, old_updated = list(state.events), state.updated_at
    state.add_research_loop(loop)
    try:
        state.log("ResearchLoop", "loop_created", f"Created loop {loop.loop_id} for run {run.run_id}.")
    except Exception:
        state.research_loops.remove(loop)
        state.events[:] = old_events
        state.updated_at = old_updated
        raise
    return loop


def _get_loop(state: "ResearchState", loop_id: str) -> ResearchLoop:
    matches = [item for item in state.research_loops if item.loop_id == loop_id]
    if len(matches) != 1:
        raise ValueError(f"Unknown or duplicate ResearchLoop ID {loop_id!r}.")
    return matches[0]


def _replace_loop(state: "ResearchState", prior: ResearchLoop, updated: ResearchLoop,
                  action: str, message: str) -> ResearchLoop:
    index = state.research_loops.index(prior)
    old_events = list(state.events)
    old_updated_at = state.updated_at
    state.research_loops[index] = updated
    try:
        state.log("ResearchLoop", action, message)
        errors = updated.validate_references(state)
        if errors:
            raise ValueError("ResearchLoop update has invalid references: " + "; ".join(errors))
    except Exception:
        state.research_loops[index] = prior
        state.events[:] = old_events
        state.updated_at = old_updated_at
        raise
    return updated


def start_research_loop(state: "ResearchState", loop_id: str) -> ResearchLoop:
    loop = _get_loop(state, loop_id)
    if loop.status is not ResearchLoopStatus.CREATED:
        raise ValueError("Only a CREATED research loop can be started.")
    run = next(item for item in state.research_runs if item.run_id == loop.run_id)
    if not run.authorized_task_ids:
        return _transition(state, loop, LoopStage.STOPPED, authority=TransitionAuthority.SYSTEM,
                           reason="No authorized tasks are available.", stop_reason=StopReason.NO_AUTHORIZED_WORK)
    return _transition(state, loop, LoopStage.PLANNING, authority=TransitionAuthority.SYSTEM,
                       reason="Approved plan/run lineage is registered; loop entered planning stage.")


def transition_research_loop(state: "ResearchState", loop_id: str, target_stage: LoopStage,
                             *, reason: str, authority: TransitionAuthority = TransitionAuthority.SYSTEM,
                             stop_reason: Optional[StopReason] = None) -> ResearchLoop:
    loop = _get_loop(state, loop_id)
    try:
        return _transition(state, loop, target_stage, authority=authority, reason=reason, stop_reason=stop_reason)
    except ValueError as exc:
        state.log("ResearchLoop", "transition_rejected", f"Loop {loop_id}: {exc}")
        raise


def _transition(state: "ResearchState", loop: ResearchLoop, target_stage: LoopStage,
                *, authority: TransitionAuthority, reason: str,
                stop_reason: Optional[StopReason] = None) -> ResearchLoop:
    target_stage = _enum(LoopStage, target_stage, "target_stage")
    authority = _enum(TransitionAuthority, authority, "authority")
    if loop.status in {ResearchLoopStatus.COMPLETED, ResearchLoopStatus.STOPPED, ResearchLoopStatus.FAILED}:
        raise ValueError("Terminal research loops cannot transition.")
    if target_stage not in LEGAL_TRANSITIONS[loop.current_stage]:
        raise ValueError(f"Illegal research-loop transition: {loop.current_stage.value} -> {target_stage.value}.")
    if (loop.current_stage is LoopStage.RESEARCHER_REVIEW
            and target_stage in {LoopStage.RESEARCH, LoopStage.PLANNING, LoopStage.COMPLETED}
            and authority is not TransitionAuthority.RESEARCHER):
        raise ValueError("Leaving RESEARCHER_REVIEW to continue/complete requires researcher authority.")
    if (target_stage not in {LoopStage.RESEARCHER_REVIEW, LoopStage.STOPPED, LoopStage.FAILED}
            and any(item.blocks_progress for item in loop.unresolved_work)):
        raise ValueError("Blocking unresolved work must be resolved before continuing the workflow.")
    if target_stage is LoopStage.STOPPED:
        if stop_reason is None:
            raise ValueError("Stopping a loop requires a typed StopReason.")
        stop_reason = _enum(StopReason, stop_reason, "stop_reason")
        if stop_reason is StopReason.NO_AUTHORIZED_WORK:
            run = next((item for item in state.research_runs if item.run_id == loop.run_id), None)
            if run is None or run.authorized_task_ids:
                raise ValueError("no_authorized_work requires a registered run with no authorized task IDs.")
        if stop_reason is StopReason.UNRESOLVED_BLOCKER and not any(
            item.blocks_progress for item in loop.unresolved_work
        ):
            raise ValueError("unresolved_blocker requires at least one blocking unresolved-work item.")
        if stop_reason is StopReason.MAXIMUM_ITERATIONS_REACHED and len(loop.loop_iterations) < loop.max_iterations:
            raise ValueError("maximum_iterations_reached requires the configured iteration bound to be reached.")
        if stop_reason is StopReason.RESEARCHER_REQUESTED_STOP and authority is not TransitionAuthority.RESEARCHER:
            raise ValueError("Researcher-requested stop requires researcher authority.")
        if stop_reason in {StopReason.COMPLETED, StopReason.FAILED, StopReason.WAITING_FOR_RESEARCHER}:
            raise ValueError(f"{stop_reason.value} is not a STOPPED loop reason.")
        status = ResearchLoopStatus.STOPPED
    elif target_stage is LoopStage.FAILED:
        stop_reason, status = StopReason.FAILED, ResearchLoopStatus.FAILED
    elif target_stage is LoopStage.COMPLETED:
        if authority is not TransitionAuthority.RESEARCHER:
            raise ValueError("Completing a loop requires an explicit researcher decision.")
        stop_reason, status = StopReason.COMPLETED, ResearchLoopStatus.COMPLETED
    elif target_stage is LoopStage.RESEARCHER_REVIEW:
        stop_reason, status = StopReason.WAITING_FOR_RESEARCHER, ResearchLoopStatus.WAITING_FOR_RESEARCHER
    else:
        if stop_reason is not None:
            raise ValueError("Non-terminal stage transitions cannot have a stop reason.")
        stop_reason, status = None, ResearchLoopStatus.RUNNING
    if status in {ResearchLoopStatus.COMPLETED, ResearchLoopStatus.STOPPED, ResearchLoopStatus.FAILED} and loop.current_iteration_id:
        closing_status = {
            ResearchLoopStatus.COMPLETED: LoopIterationStatus.COMPLETED,
            ResearchLoopStatus.STOPPED: LoopIterationStatus.STOPPED,
            ResearchLoopStatus.FAILED: LoopIterationStatus.FAILED,
        }[status]
        loop = _close_active_iteration(state, loop, closing_status, reason)
    transition = LoopTransition(loop.current_stage, target_stage, authority, reason)
    updated = replace(loop, current_stage=target_stage, status=status, stop_reason=stop_reason,
                      transitions=loop.transitions + (transition,), updated_at=_now())
    return _replace_loop(state, loop, updated, "stage_transition",
                         f"{loop.current_stage.value} -> {target_stage.value}: {reason}")


def begin_loop_iteration(state: "ResearchState", loop_id: str, task_ids: Tuple[str, ...]) -> Optional[LoopIterationRecord]:
    loop = _get_loop(state, loop_id)
    if loop.status is not ResearchLoopStatus.RUNNING:
        raise ValueError("A loop iteration requires a RUNNING loop.")
    if loop.current_iteration_id is not None:
        raise ValueError("A loop iteration is already active.")
    tasks = _ids(task_ids, "task_ids", allow_empty=False)
    run = next(item for item in state.research_runs if item.run_id == loop.run_id)
    authorized = set(run.authorized_task_ids or ())
    if not set(tasks).issubset(authorized):
        raise ValueError("Loop iteration includes task IDs not authorized for this ResearchRun.")
    if not set(tasks).issubset(set(loop.unresolved_task_ids)):
        raise ValueError("Loop iteration may include only tasks still unresolved in this loop.")
    if any(item.blocks_progress for item in loop.unresolved_work):
        raise ValueError("Blocking unresolved work prevents a new loop iteration.")
    if len(loop.loop_iterations) >= loop.max_iterations:
        _transition(state, loop, LoopStage.STOPPED, authority=TransitionAuthority.SYSTEM,
                    reason="Maximum configured loop iterations reached.",
                    stop_reason=StopReason.MAXIMUM_ITERATIONS_REACHED)
        return None
    state_iterations = list(state.iterations)
    events = list(state.events)
    old_updated = state.updated_at
    try:
        research_iteration = state.create_iteration(loop.run_id, status="running")
        record = LoopIterationRecord(
            iteration_id=research_iteration.iteration_id, run_id=loop.run_id,
            iteration_number=research_iteration.number, stage=loop.current_stage, task_ids=tasks,
            transition_reason=f"Started at {loop.current_stage.value}.",
        )
        updated = replace(loop, current_iteration_id=record.iteration_id,
                          current_task_id=tasks[0], iteration_number=record.iteration_number,
                          loop_iterations=loop.loop_iterations + (record,), updated_at=_now())
        _replace_loop(state, loop, updated, "iteration_created",
                      f"Started loop iteration {record.iteration_number} ({record.iteration_id}).")
        return record
    except Exception:
        state.iterations[:] = state_iterations
        state.events[:] = events
        state.updated_at = old_updated
        raise


def finish_loop_iteration(state: "ResearchState", loop_id: str, *, status: LoopIterationStatus,
                          reason: str) -> LoopIterationRecord:
    loop = _get_loop(state, loop_id)
    status = _enum(LoopIterationStatus, status, "status")
    if status is LoopIterationStatus.RUNNING:
        raise ValueError("finish_loop_iteration requires a terminal iteration status.")
    if loop.current_iteration_id is None:
        raise ValueError("No loop iteration is active.")
    _close_active_iteration(state, loop, status, reason)
    updated_loop = _get_loop(state, loop_id)
    if status is LoopIterationStatus.FAILED:
        fail_research_loop(state, loop_id, f"Loop iteration failed: {reason}")
    elif len(updated_loop.loop_iterations) >= loop.max_iterations:
        _transition(state, updated_loop, LoopStage.STOPPED, authority=TransitionAuthority.SYSTEM,
                    reason="Maximum configured loop iterations reached.",
                    stop_reason=StopReason.MAXIMUM_ITERATIONS_REACHED)
    return next(item for item in updated_loop.loop_iterations if item.iteration_id == loop.current_iteration_id)


def select_loop_task(state: "ResearchState", loop_id: str, task_id: str) -> ResearchLoop:
    loop = _get_loop(state, loop_id)
    if loop.status is not ResearchLoopStatus.RUNNING or loop.current_iteration_id is None:
        raise ValueError("Selecting a task requires an active loop iteration.")
    active = next(item for item in loop.loop_iterations if item.iteration_id == loop.current_iteration_id)
    if task_id not in active.task_ids:
        raise ValueError("Selected task is not part of the active loop iteration.")
    if task_id in loop.completed_task_ids:
        raise ValueError("A completed task cannot become the current task again.")
    updated = replace(loop, current_task_id=task_id, updated_at=_now())
    return _replace_loop(state, loop, updated, "task_selected", f"Selected authorized task {task_id}.")


def _close_active_iteration(state: "ResearchState", loop: ResearchLoop,
                             status: LoopIterationStatus, reason: str) -> ResearchLoop:
    if loop.current_iteration_id is None:
        raise ValueError("No loop iteration is active.")
    record_index = next(i for i, item in enumerate(loop.loop_iterations) if item.iteration_id == loop.current_iteration_id)
    record = loop.loop_iterations[record_index]
    updated_record = replace(record, status=status, ended_at=_now(), transition_reason=reason)
    updated_records = loop.loop_iterations[:record_index] + (updated_record,) + loop.loop_iterations[record_index + 1:]
    state_iteration = next(item for item in state.iterations if item.iteration_id == record.iteration_id)
    old_status, old_updated = state_iteration.status, state.updated_at
    events = list(state.events)
    state_iteration.status = "failed" if status is LoopIterationStatus.FAILED else "completed"
    updated_loop = replace(loop, current_iteration_id=None, current_task_id=None,
                           loop_iterations=updated_records, updated_at=_now())
    try:
        _replace_loop(state, loop, updated_loop, "iteration_finished",
                      f"Finished loop iteration {record.iteration_number} as {status.value}: {reason}")
    except Exception:
        state_iteration.status = old_status
        state.updated_at = old_updated
        state.events[:] = events
        raise
    return updated_loop


def add_unresolved_work(state: "ResearchState", loop_id: str, item: UnresolvedWork) -> ResearchLoop:
    if not isinstance(item, UnresolvedWork):
        raise ValueError("A typed UnresolvedWork item is required.")
    loop = _get_loop(state, loop_id)
    if item.task_id and item.task_id not in set(loop.unresolved_task_ids) | set(loop.completed_task_ids):
        raise ValueError("UnresolvedWork task_id must refer to a task tracked by this loop.")
    if any(existing == item for existing in loop.unresolved_work):
        raise ValueError("Duplicate unresolved work item.")
    updated = replace(loop, unresolved_work=loop.unresolved_work + (item,), updated_at=_now())
    return _replace_loop(state, loop, updated, "unresolved_work_recorded", item.description)


def clear_unresolved_work(state: "ResearchState", loop_id: str, item: UnresolvedWork) -> ResearchLoop:
    loop = _get_loop(state, loop_id)
    if item not in loop.unresolved_work:
        raise ValueError("UnresolvedWork item is not recorded on this loop.")
    updated = replace(loop, unresolved_work=tuple(x for x in loop.unresolved_work if x != item), updated_at=_now())
    return _replace_loop(state, loop, updated, "unresolved_work_cleared", item.description)


def request_capability_invocation(state: "ResearchState", loop_id: str, capability: CapabilityType,
                                  task_id: str) -> CapabilityInvocationRequest:
    loop = _get_loop(state, loop_id)
    capability = _enum(CapabilityType, capability, "capability")
    if loop.status is not ResearchLoopStatus.RUNNING or loop.current_iteration_id is None:
        raise ValueError("Capability invocation requires a running loop iteration.")
    active_iteration = next(item for item in loop.loop_iterations if item.iteration_id == loop.current_iteration_id)
    if task_id not in active_iteration.task_ids:
        raise ValueError("Capability request task is not part of the active loop iteration.")
    if capability not in _CAPABILITY_STAGE.get(loop.current_stage, frozenset()):
        raise ValueError(f"Capability {capability.value!r} is not available at stage {loop.current_stage.value!r}.")
    existing = {item.invocation_id for item in loop.capability_requests}
    request = CapabilityInvocationRequest(capability, loop.loop_id, loop.run_id, task_id, loop.current_iteration_id)
    if request.invocation_id in existing:
        raise ValueError("Duplicate capability invocation ID.")
    updated = replace(loop, capability_requests=loop.capability_requests + (request,), updated_at=_now())
    _replace_loop(state, loop, updated, "capability_request_recorded",
                  f"Explicit {capability.value} request recorded for task {task_id}.")
    return request


def record_capability_result(state: "ResearchState", loop_id: str,
                             result: CapabilityInvocationResult) -> ResearchLoop:
    if not isinstance(result, CapabilityInvocationResult):
        raise ValueError("A typed CapabilityInvocationResult is required.")
    loop = _get_loop(state, loop_id)
    request = next((item for item in loop.capability_requests if item.invocation_id == result.invocation_id), None)
    if request is None or request.capability is not result.capability:
        raise ValueError("Capability result does not resolve to a matching recorded request.")
    if any(item.invocation_id == result.invocation_id for item in loop.capability_results):
        raise ValueError("Capability invocation already has a recorded result.")
    execution = next((item for item in state.task_executions if item.execution_id == result.execution_id), None)
    expected = TaskExecutionStatus.COMPLETED if result.status is CapabilityInvocationStatus.COMPLETED else TaskExecutionStatus.FAILED
    if execution is None or execution.run_id != loop.run_id or execution.task_id != request.task_id or execution.iteration_id != request.iteration_id or execution.status is not expected:
        raise ValueError("Capability result must reference the matching persisted Phase 5 execution.")
    artifacts = {item.artifact_id: item for item in state.research_artifacts}
    for artifact_id in result.artifact_ids:
        artifact = artifacts.get(artifact_id)
        if artifact is None or artifact.execution_id != execution.execution_id or artifact.run_id != loop.run_id or artifact.task_id != request.task_id:
            raise ValueError(f"Capability result references unknown or mismatched artifact {artifact_id!r}.")
    known_evidence = {item.evidence_id for item in state.evidence}
    if not set(result.evidence_ids).issubset(known_evidence):
        raise ValueError("Capability result references Evidence absent from ResearchState.")
    updated = replace(loop, capability_results=loop.capability_results + (result,), updated_at=_now())
    updated = _replace_loop(state, loop, updated, "capability_result_recorded",
                            f"Recorded {result.status.value} result for {request.capability.value}.")
    if result.status is CapabilityInvocationStatus.FAILED:
        return fail_research_loop(state, loop_id, f"Capability {request.capability.value} failed.")
    return updated


def complete_loop_task(state: "ResearchState", loop_id: str, task_id: str, execution_id: str) -> ResearchLoop:
    """Mark the research objective complete after an explicit caller decision.

    A COMPLETED Phase 5 execution alone does not call this function and does
    not move the task out of unresolved work.
    """
    loop = _get_loop(state, loop_id)
    if loop.status is not ResearchLoopStatus.RUNNING:
        raise ValueError("A task can only be marked completed while the research loop is RUNNING.")
    if task_id not in set(loop.unresolved_task_ids) | set(loop.completed_task_ids):
        raise ValueError("Task is not authorized/tracked by this loop.")
    execution = next((item for item in state.task_executions if item.execution_id == execution_id), None)
    if execution is None or execution.run_id != loop.run_id or execution.task_id != task_id or execution.status is not TaskExecutionStatus.COMPLETED:
        raise ValueError("Task completion requires a matching completed Phase 5 execution.")
    recorded_iteration = next((item for item in loop.loop_iterations if item.iteration_id == execution.iteration_id), None)
    if recorded_iteration is None or task_id not in recorded_iteration.task_ids:
        raise ValueError("Task completion requires an execution from a loop iteration that included this task.")
    if task_id in loop.completed_task_ids:
        raise ValueError("Task is already marked completed in this loop.")
    active = next((item for item in loop.loop_iterations if item.iteration_id == loop.current_iteration_id), None)
    candidates = active.task_ids if active is not None else ()
    remaining = tuple(x for x in candidates if x != task_id and x in loop.unresolved_task_ids)
    next_task = remaining[0] if loop.current_task_id == task_id and remaining else (
        None if loop.current_task_id == task_id else loop.current_task_id
    )
    updated = replace(loop, completed_task_ids=loop.completed_task_ids + (task_id,),
                      unresolved_task_ids=tuple(x for x in loop.unresolved_task_ids if x != task_id),
                      current_task_id=next_task,
                      updated_at=_now())
    return _replace_loop(state, loop, updated, "task_completed",
                         f"Caller marked research task {task_id} complete after execution {execution_id}.")


def stop_research_loop(state: "ResearchState", loop_id: str, reason: StopReason,
                       *, authority: TransitionAuthority = TransitionAuthority.SYSTEM,
                       message: str = "") -> ResearchLoop:
    loop = _get_loop(state, loop_id)
    reason = _enum(StopReason, reason, "stop_reason")
    if reason is StopReason.FAILED:
        return fail_research_loop(state, loop_id, message or "Loop failed.")
    if reason is StopReason.COMPLETED:
        return transition_research_loop(state, loop_id, LoopStage.COMPLETED,
                                        reason=message or "Researcher marked loop completed.",
                                        authority=authority)
    if reason is StopReason.WAITING_FOR_RESEARCHER:
        return transition_research_loop(state, loop_id, LoopStage.RESEARCHER_REVIEW,
                                        reason=message or "Researcher decision is required.",
                                        authority=TransitionAuthority.SYSTEM)
    return transition_research_loop(state, loop_id, LoopStage.STOPPED, authority=authority,
                                    reason=message or reason.value, stop_reason=reason)


def fail_research_loop(state: "ResearchState", loop_id: str, message: str) -> ResearchLoop:
    loop = _get_loop(state, loop_id)
    return _transition(state, loop, LoopStage.FAILED, authority=TransitionAuthority.SYSTEM,
                       reason=message or "Research loop failed.")


__all__ = [
    "LoopStage", "ResearchLoopStatus", "StopReason", "TransitionAuthority", "LoopIterationStatus",
    "UnresolvedWorkKind", "CapabilityInvocationStatus", "LEGAL_TRANSITIONS", "LoopTransition",
    "LoopIterationRecord", "UnresolvedWork", "CapabilityInvocationRequest", "CapabilityInvocationResult",
    "TaskContinuationAuthorization",
    "ResearchLoop", "capability_compatible_with_stage", "create_research_loop", "start_research_loop", "transition_research_loop",
    "begin_loop_iteration", "finish_loop_iteration", "add_unresolved_work", "clear_unresolved_work",
    "request_capability_invocation", "record_capability_result", "complete_loop_task",
    "select_loop_task", "stop_research_loop", "fail_research_loop",
]
