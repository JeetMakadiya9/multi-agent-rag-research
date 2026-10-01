"""Explicit one-capability dispatch from a controlled ResearchLoop.

The dispatcher validates loop/task/run lineage and then delegates to exactly
one existing Phase 6 public capability API. It does not select capabilities,
chain calls, or implement capability behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import json
from typing import Optional, Union
from uuid import uuid4

from research_capabilities import (
    CapabilityResult,
    CapabilityType,
    LocalRetrievalRequest,
    execute_local_retrieval,
)
from research_experiment_planning_capability import (
    ExperimentPlanningRequest,
    ExperimentPlanningResult,
    execute_experiment_planning,
)
from research_improvement_capability import (
    ImprovementGenerationRequest,
    ImprovementGenerationResult,
    execute_improvement_generation,
)
from research_loop import (
    CapabilityInvocationResult,
    CapabilityInvocationStatus,
    LoopStage,
    ResearchLoop,
    ResearchLoopStatus,
    capability_compatible_with_stage,
    complete_loop_task,
    record_capability_result,
    request_capability_invocation,
)
from research_plan_review import PlanReview, is_execution_approved
from research_planning import ResearchPlan, ResearchTask
from research_prior_work_capability import (
    PriorWorkInvestigationRequest,
    PriorWorkInvestigationResult,
    execute_prior_work_investigation,
)
from research_state import ResearchArtifact, ResearchRun, ResearchState
from research_task_execution import (
    ResearchTaskExecution,
    TaskExecutionError,
    TaskExecutionRecordingError,
    TaskExecutionStatus,
)
from research_verification_capability import (
    EvidenceVerificationRequest,
    VerificationCapabilityResult,
    execute_evidence_verification,
)


CapabilityRequest = Union[
    LocalRetrievalRequest,
    EvidenceVerificationRequest,
    ImprovementGenerationRequest,
    PriorWorkInvestigationRequest,
    ExperimentPlanningRequest,
]
CapabilityResponse = Union[
    CapabilityResult,
    VerificationCapabilityResult,
    ImprovementGenerationResult,
    PriorWorkInvestigationResult,
    ExperimentPlanningResult,
]


class DispatchStatus(str, Enum):
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class CapabilityDispatchError(ValueError):
    """Dispatch was rejected before a Phase 5 execution could be made."""


@dataclass(frozen=True)
class ResearchCapabilityDispatchResult:
    """Typed summary of one persisted capability invocation and execution."""

    dispatch_id: str
    loop_id: str
    run_id: str
    plan_id: str
    plan_revision: int
    task_id: str
    iteration_id: str
    capability: CapabilityType
    execution_id: str
    status: DispatchStatus
    artifact_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    search_action_ids: tuple[str, ...] = ()
    search_result_ids: tuple[str, ...] = ()
    created_at: str = ""
    error: Optional[str] = None

    def __post_init__(self) -> None:
        for name in ("dispatch_id", "loop_id", "run_id", "plan_id", "task_id", "iteration_id", "execution_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string.")
        if not isinstance(self.plan_revision, int) or isinstance(self.plan_revision, bool) or self.plan_revision < 1:
            raise ValueError("plan_revision must be a positive integer.")
        try:
            capability = self.capability if isinstance(self.capability, CapabilityType) else CapabilityType(self.capability)
        except (TypeError, ValueError) as exc:
            raise ValueError("Unknown capability in dispatch result.") from exc
        object.__setattr__(self, "capability", capability)
        try:
            status = self.status if isinstance(self.status, DispatchStatus) else DispatchStatus(self.status)
        except (TypeError, ValueError) as exc:
            raise ValueError("Unknown dispatch status.") from exc
        object.__setattr__(self, "status", status)
        for name in ("artifact_ids", "evidence_ids", "search_action_ids", "search_result_ids"):
            values = getattr(self, name)
            if not isinstance(values, (tuple, list)):
                raise ValueError(f"{name} must be a sequence of IDs.")
            values = tuple(values)
            if any(not isinstance(item, str) or not item.strip() for item in values) or len(values) != len(set(values)):
                raise ValueError(f"{name} must contain unique non-empty IDs.")
            object.__setattr__(self, name, values)
        if status is DispatchStatus.COMPLETED and (self.error is not None or not self.artifact_ids):
            raise ValueError("A completed dispatch requires persisted artifact IDs and cannot contain an error.")
        if status is DispatchStatus.FAILED and (not isinstance(self.error, str) or not self.error.strip()):
            raise ValueError("A failed dispatch requires an inspectable error summary.")
        if not isinstance(self.created_at, str) or not self.created_at.strip():
            raise ValueError("created_at must be a timezone-aware timestamp.")
        try:
            timestamp = datetime.fromisoformat(self.created_at)
        except ValueError as exc:
            raise ValueError("created_at must be a timezone-aware timestamp.") from exc
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("created_at must be a timezone-aware timestamp.")

    def to_dict(self) -> dict[str, object]:
        return {
            "dispatch_id": self.dispatch_id, "loop_id": self.loop_id, "run_id": self.run_id,
            "plan_id": self.plan_id, "plan_revision": self.plan_revision, "task_id": self.task_id,
            "iteration_id": self.iteration_id, "capability": self.capability.value,
            "execution_id": self.execution_id, "status": self.status.value,
            "artifact_ids": list(self.artifact_ids), "evidence_ids": list(self.evidence_ids),
            "search_action_ids": list(self.search_action_ids), "search_result_ids": list(self.search_result_ids),
            "created_at": self.created_at, "error": self.error,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "ResearchCapabilityDispatchResult":
        keys = {
            "dispatch_id", "loop_id", "run_id", "plan_id", "plan_revision", "task_id", "iteration_id",
            "capability", "execution_id", "status", "artifact_ids", "evidence_ids", "search_action_ids",
            "search_result_ids", "created_at", "error",
        }
        if not isinstance(data, dict) or set(data) != keys:
            raise ValueError("Malformed ResearchCapabilityDispatchResult payload.")
        values = dict(data)
        for name in ("artifact_ids", "evidence_ids", "search_action_ids", "search_result_ids"):
            if not isinstance(values[name], (list, tuple)):
                raise ValueError(f"Malformed dispatch result {name}.")
            values[name] = tuple(values[name])
        try:
            return cls(**values)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Malformed ResearchCapabilityDispatchResult: {exc}") from exc

    @classmethod
    def from_json(cls, payload: str) -> "ResearchCapabilityDispatchResult":
        try:
            data = json.loads(payload)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Malformed dispatch result JSON: {exc}") from exc
        return cls.from_dict(data)


_REQUEST_TYPES = {
    CapabilityType.LOCAL_RETRIEVAL: LocalRetrievalRequest,
    CapabilityType.EVIDENCE_VERIFICATION: EvidenceVerificationRequest,
    CapabilityType.IMPROVEMENT_GENERATION: ImprovementGenerationRequest,
    CapabilityType.PRIOR_WORK_INVESTIGATION: PriorWorkInvestigationRequest,
    CapabilityType.EXPERIMENT_PLANNING: ExperimentPlanningRequest,
}


def dispatch_research_capability(
    plan: ResearchPlan,
    run: ResearchRun,
    task: ResearchTask,
    review: PlanReview,
    loop: ResearchLoop,
    state: ResearchState,
    *,
    capability: CapabilityType,
    capability_request: CapabilityRequest,
    iteration_id: str,
    retrieval_provider=None,
    llm_provider=None,
    search_provider=None,
    retry_of_execution_id: Optional[str] = None,
    continuation_authorization_id: Optional[str] = None,
) -> ResearchCapabilityDispatchResult:
    """Dispatch one explicitly selected existing Phase 6 capability.

    The typed request object fixes the one-of request shape. All authorization
    and execution remain in the existing PlanReview/Phase 5/capability APIs.
    """
    try:
        capability = capability if isinstance(capability, CapabilityType) else CapabilityType(capability)
    except (TypeError, ValueError) as exc:
        raise CapabilityDispatchError("Exactly one supported CapabilityType must be explicitly selected.") from exc
    _validate_dispatch_inputs(plan, run, task, review, loop, state, capability,
                              capability_request, iteration_id, retrieval_provider,
                              llm_provider, search_provider, continuation_authorization_id)
    invocation = request_capability_invocation(state, loop.loop_id, capability, task.task_id)
    try:
        response = _call_one_capability(
            capability, plan, run, task, review, capability_request, state,
            iteration_id=iteration_id, retrieval_provider=retrieval_provider,
            llm_provider=llm_provider, search_provider=search_provider,
            retry_of_execution_id=retry_of_execution_id,
            continuation_authorization_id=continuation_authorization_id,
        )
    except (TaskExecutionError, TaskExecutionRecordingError) as exc:
        execution = exc.execution
        if not isinstance(execution, ResearchTaskExecution) or execution.status is not TaskExecutionStatus.FAILED:
            raise CapabilityDispatchError("Phase 5 capability failure did not expose a persisted FAILED execution.") from exc
        failed = CapabilityInvocationResult(
            capability, invocation.invocation_id, execution.execution_id, CapabilityInvocationStatus.FAILED,
        )
        record_capability_result(state, loop.loop_id, failed)
        return ResearchCapabilityDispatchResult(
            dispatch_id=invocation.invocation_id, loop_id=loop.loop_id, run_id=run.run_id,
            plan_id=plan.plan_id, plan_revision=plan.revision, task_id=task.task_id,
            iteration_id=iteration_id, capability=capability, execution_id=execution.execution_id,
            status=DispatchStatus.FAILED, created_at=_now(),
            error=f"Phase 5 execution failed ({type(exc).__name__}).",
        )

    execution_id, artifact_ids, evidence_ids, action_ids, result_ids = _extract_persisted_result(
        capability, response, state, run, task,
    )
    completed = CapabilityInvocationResult(
        capability, invocation.invocation_id, execution_id, CapabilityInvocationStatus.COMPLETED,
        artifact_ids=artifact_ids, evidence_ids=evidence_ids,
    )
    record_capability_result(state, loop.loop_id, completed)
    return ResearchCapabilityDispatchResult(
        dispatch_id=invocation.invocation_id, loop_id=loop.loop_id, run_id=run.run_id,
        plan_id=plan.plan_id, plan_revision=plan.revision, task_id=task.task_id,
        iteration_id=iteration_id, capability=capability, execution_id=execution_id,
        status=DispatchStatus.COMPLETED, artifact_ids=artifact_ids, evidence_ids=evidence_ids,
        search_action_ids=action_ids, search_result_ids=result_ids, created_at=_now(),
    )


def _validate_dispatch_inputs(plan, run, task, review, loop, state, capability, request,
                              iteration_id, retrieval_provider, llm_provider, search_provider,
                              continuation_authorization_id=None) -> None:
    if not isinstance(plan, ResearchPlan) or plan.validate():
        raise CapabilityDispatchError("A structurally valid ResearchPlan is required.")
    if not isinstance(run, ResearchRun) or not isinstance(task, ResearchTask) or not isinstance(review, PlanReview):
        raise CapabilityDispatchError("ResearchRun, ResearchTask, and PlanReview are required.")
    if not isinstance(state, ResearchState):
        raise CapabilityDispatchError("ResearchState is required.")
    if not isinstance(loop, ResearchLoop):
        raise CapabilityDispatchError("A registered ResearchLoop is required.")
    try:
        capability = capability if isinstance(capability, CapabilityType) else CapabilityType(capability)
    except (TypeError, ValueError) as exc:
        raise CapabilityDispatchError("Exactly one supported CapabilityType must be explicitly selected.") from exc
    if not isinstance(request, _REQUEST_TYPES[capability]):
        raise CapabilityDispatchError(f"{capability.value} requires a typed {_REQUEST_TYPES[capability].__name__}.")
    try:
        review.validate_against(plan)
        if PlanReview.from_dict(review.to_dict()) != review:
            raise ValueError("Serialized PlanReview differs from the supplied review.")
    except (AttributeError, TypeError, ValueError) as exc:
        raise CapabilityDispatchError(f"PlanReview does not match the supplied plan: {exc}") from exc
    if not is_execution_approved(plan, review):
        raise CapabilityDispatchError("The supplied PlanReview does not approve this exact plan revision.")
    if (run.plan_id != plan.plan_id or run.plan_revision != plan.revision
            or run.approval_review_id != review.review_id or task.task_id not in (run.authorized_task_ids or ())):
        raise CapabilityDispatchError("Run, approval, plan revision, or task authorization does not match.")
    if run.status not in {"created", "running"}:
        raise CapabilityDispatchError(f"ResearchRun status {run.status!r} does not allow capability dispatch.")
    if not any(item is run for item in state.research_runs):
        raise CapabilityDispatchError("ResearchRun must be registered in ResearchState.")
    canonical_task = next((item for item in plan.tasks if item.task_id == task.task_id), None)
    if canonical_task is None or canonical_task != task:
        raise CapabilityDispatchError("ResearchTask must exactly match a task in the approved plan.")
    registered = [item for item in state.research_loops if item.loop_id == loop.loop_id]
    if len(registered) != 1 or registered[0] != loop:
        raise CapabilityDispatchError("ResearchLoop must resolve uniquely to the current ResearchState record.")
    if loop.run_id != run.run_id or loop.plan_id != plan.plan_id or loop.plan_revision != plan.revision:
        raise CapabilityDispatchError("ResearchLoop is not linked to the supplied run and exact plan revision.")
    if loop.validate_references(state):
        raise CapabilityDispatchError("ResearchLoop has invalid ResearchState references.")
    if loop.status is not ResearchLoopStatus.RUNNING:
        raise CapabilityDispatchError("Dispatch requires a RUNNING loop; researcher gates and terminal loops cannot dispatch.")
    if loop.current_iteration_id != iteration_id or not iteration_id:
        raise CapabilityDispatchError("Dispatch iteration must be the loop's active ResearchIteration.")
    iteration = next((item for item in state.iterations if item.iteration_id == iteration_id), None)
    if iteration is None or iteration.run_id != run.run_id:
        raise CapabilityDispatchError("Dispatch iteration must exist and belong to the supplied ResearchRun.")
    if loop.current_task_id != task.task_id:
        raise CapabilityDispatchError("Only the currently selected loop task may be dispatched.")
    active = next((item for item in loop.loop_iterations if item.iteration_id == iteration_id), None)
    if active is None or task.task_id not in active.task_ids:
        raise CapabilityDispatchError("Task must belong to the active loop iteration.")
    prior = [item for item in state.task_executions if item.run_id == run.run_id and item.task_id == task.task_id]
    if prior and prior[-1].status is TaskExecutionStatus.COMPLETED:
        if not continuation_authorization_id:
            raise CapabilityDispatchError("Completed task execution requires explicit continuation authorization.")
        matches = [item for item in loop.continuation_authorizations
                   if item.authorization_id == continuation_authorization_id]
        if len(matches) != 1:
            raise CapabilityDispatchError("Unknown or duplicate continuation authorization.")
        authorization = matches[0]
        if (authorization.run_id != run.run_id or authorization.plan_id != plan.plan_id
                or authorization.plan_revision != plan.revision or authorization.task_id != task.task_id
                or authorization.previous_execution_id != prior[-1].execution_id
                or authorization.iteration_id != iteration_id):
            raise CapabilityDispatchError("Continuation authorization does not match this exact execution context.")
    elif continuation_authorization_id is not None:
        raise CapabilityDispatchError("Continuation authorization requires a prior completed execution.")
    if not capability_compatible_with_stage(loop.current_stage, capability):
        raise CapabilityDispatchError(f"{capability.value} is incompatible with stage {loop.current_stage.value}.")
    if capability is CapabilityType.LOCAL_RETRIEVAL and retrieval_provider is None:
        raise CapabilityDispatchError("LOCAL_RETRIEVAL requires an explicit retrieval provider.")
    if capability in {CapabilityType.EVIDENCE_VERIFICATION, CapabilityType.IMPROVEMENT_GENERATION,
                      CapabilityType.EXPERIMENT_PLANNING} and llm_provider is None:
        raise CapabilityDispatchError(f"{capability.value} requires an explicit LLM provider.")
    if capability is CapabilityType.PRIOR_WORK_INVESTIGATION and (search_provider is None or llm_provider is None):
        raise CapabilityDispatchError("PRIOR_WORK_INVESTIGATION requires explicit search and LLM providers.")


def _call_one_capability(capability, plan, run, task, review, request, state, *, iteration_id,
                         retrieval_provider, llm_provider, search_provider, retry_of_execution_id,
                         continuation_authorization_id=None):
    if capability is CapabilityType.LOCAL_RETRIEVAL:
        return execute_local_retrieval(plan, run, task, review, request, retrieval_provider, state,
                                        iteration_id=iteration_id, retry_of_execution_id=retry_of_execution_id,
                                        continuation_authorization_id=continuation_authorization_id)
    if capability is CapabilityType.EVIDENCE_VERIFICATION:
        return execute_evidence_verification(plan, run, task, review, request, state, llm_provider,
                                             iteration_id=iteration_id, retry_of_execution_id=retry_of_execution_id,
                                             continuation_authorization_id=continuation_authorization_id)
    if capability is CapabilityType.IMPROVEMENT_GENERATION:
        return execute_improvement_generation(plan, run, task, review, request, state, llm_provider,
                                              iteration_id=iteration_id, retry_of_execution_id=retry_of_execution_id,
                                              continuation_authorization_id=continuation_authorization_id)
    if capability is CapabilityType.PRIOR_WORK_INVESTIGATION:
        return execute_prior_work_investigation(plan, run, task, review, request, state, search_provider,
                                                llm_provider, iteration_id=iteration_id,
                                                retry_of_execution_id=retry_of_execution_id,
                                                continuation_authorization_id=continuation_authorization_id)
    if capability is CapabilityType.EXPERIMENT_PLANNING:
        return execute_experiment_planning(plan, run, task, review, request, state, llm_provider,
                                           iteration_id=iteration_id, retry_of_execution_id=retry_of_execution_id,
                                           continuation_authorization_id=continuation_authorization_id)
    raise CapabilityDispatchError(f"Unsupported capability {capability!r}.")


def _extract_persisted_result(capability, response, state, run, task):
    expected_types = {
        CapabilityType.LOCAL_RETRIEVAL: CapabilityResult,
        CapabilityType.EVIDENCE_VERIFICATION: VerificationCapabilityResult,
        CapabilityType.IMPROVEMENT_GENERATION: ImprovementGenerationResult,
        CapabilityType.PRIOR_WORK_INVESTIGATION: PriorWorkInvestigationResult,
        CapabilityType.EXPERIMENT_PLANNING: ExperimentPlanningResult,
    }
    if not isinstance(response, expected_types[capability]) or response.capability is not capability:
        raise CapabilityDispatchError("Capability returned a malformed or mismatched typed result.")
    execution_id = response.execution_id
    execution = next((item for item in state.task_executions if item.execution_id == execution_id), None)
    if (execution is None or execution.status is not TaskExecutionStatus.COMPLETED
            or execution.run_id != run.run_id or execution.task_id != task.task_id):
        raise CapabilityDispatchError("Capability result does not reference a persisted completed Phase 5 execution.")
    artifact_ids = (response.artifact_id,)
    artifacts = {item.artifact_id: item for item in state.research_artifacts}
    for artifact_id in artifact_ids:
        artifact = artifacts.get(artifact_id)
        if artifact is None or artifact.execution_id != execution_id or artifact.run_id != run.run_id or artifact.task_id != task.task_id:
            raise CapabilityDispatchError("Capability result references an absent or mismatched artifact.")
    evidence_ids = tuple(getattr(response, "evidence_ids", ()))
    known_evidence = {item.evidence_id for item in state.evidence}
    if not set(evidence_ids).issubset(known_evidence):
        raise CapabilityDispatchError("Capability result references Evidence absent from ResearchState.")
    action_ids = tuple(getattr(response, "search_action_ids", ()))
    one_action = getattr(response, "search_action_id", None)
    if one_action:
        action_ids = (one_action,)
    result_ids = tuple(getattr(response, "search_result_ids", ()))
    known_actions = {item.search_id for item in state.searches}
    known_results = {item.result_id for item in state.search_results}
    if not set(action_ids).issubset(known_actions) or not set(result_ids).issubset(known_results):
        raise CapabilityDispatchError("Capability result references missing SearchAction/SearchResult records.")
    return execution_id, artifact_ids, evidence_ids, action_ids, result_ids


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "CapabilityDispatchError", "CapabilityRequest", "DispatchStatus",
    "ResearchCapabilityDispatchResult", "dispatch_research_capability",
]
