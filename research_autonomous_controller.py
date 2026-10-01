"""Finite 8C orchestration over the existing 8A and 8B boundaries.

The controller owns only cycle ordering and stopping. Decision generation and
validation remain in :mod:`research_autonomy`; execution and its human approval
gate remain in :mod:`research_autonomous_execution`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
import json
from typing import Callable, Optional
from uuid import uuid4

from research_autonomy import (
    AutonomousResearchContext, ResearchActionType, ResearchDecision,
    ResearchDecisionTrace, ValidationStatus, autonomous_decide,
    build_autonomous_research_context, validate_research_decision,
)
from research_autonomous_execution import (
    AutonomousExecutionApproval, AutonomousExecutionRequest,
    AutonomousExecutionResult, AutonomousExecutionStatus,
    execute_autonomous_proposal,
)
from research_bounded_loop import BoundedExecutionPolicy
from research_loop import ResearchLoop, ResearchLoopStatus
from research_plan_review import PlanReview
from research_planning import ResearchPlan
from research_state import ResearchRun, ResearchState


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


class AutonomousControllerStopReason(str, Enum):
    COMPLETED = "COMPLETED"
    DECISION_STOP = "DECISION_STOP"
    WAIT_FOR_RESEARCHER = "WAIT_FOR_RESEARCHER"
    MAXIMUM_CYCLES = "MAXIMUM_CYCLES"
    MAXIMUM_EXECUTIONS = "MAXIMUM_EXECUTIONS"
    LOOP_LIMIT = "LOOP_LIMIT"
    VALIDATION_REJECTED = "VALIDATION_REJECTED"
    EXECUTION_REJECTED = "EXECUTION_REJECTED"
    EXECUTION_FAILED = "EXECUTION_FAILED"
    BLOCKED = "BLOCKED"
    STALE_STATE = "STALE_STATE"
    NO_ACTION = "NO_ACTION"
    ERROR = "ERROR"


@dataclass(frozen=True)
class AutonomousControllerPolicy:
    """Caller-owned finite bounds for a single controller invocation."""

    maximum_cycles: int
    maximum_executions: int
    maximum_capability_invocations: int
    maximum_retrieval_actions: int = 0
    maximum_verification_actions: int = 0
    maximum_total_research_actions: int = 0
    require_researcher_review: bool = True

    def __post_init__(self):
        for name in ("maximum_cycles", "maximum_executions", "maximum_capability_invocations",
                     "maximum_retrieval_actions", "maximum_verification_actions",
                     "maximum_total_research_actions"):
            value = getattr(self, name)
            minimum = 1 if name == "maximum_cycles" else 0
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        if not isinstance(self.require_researcher_review, bool):
            raise ValueError("require_researcher_review must be boolean.")

    def to_dict(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class AutonomousControllerRequest:
    plan: ResearchPlan
    run: ResearchRun
    review: PlanReview
    loop: ResearchLoop
    state: ResearchState
    execution_policy: BoundedExecutionPolicy
    policy: AutonomousControllerPolicy
    decision_provider: object
    request_factory: Callable[[ResearchDecision, ResearchDecisionTrace, AutonomousResearchContext], AutonomousExecutionRequest]
    approval_provider: Callable[[AutonomousExecutionRequest], Optional[AutonomousExecutionApproval]]
    retrieval_provider: object = None
    llm_provider: object = None
    search_provider: object = None
    controller_id: str = field(default_factory=lambda: _id("controller"))

    def __post_init__(self):
        if not isinstance(self.policy, AutonomousControllerPolicy):
            raise ValueError("An explicit AutonomousControllerPolicy is required.")
        if not isinstance(self.execution_policy, BoundedExecutionPolicy):
            raise ValueError("A BoundedExecutionPolicy is required.")
        if not isinstance(self.loop, ResearchLoop):
            raise ValueError("A typed ResearchLoop is required.")
        if self.policy.maximum_cycles > self.loop.max_iterations:
            raise ValueError("maximum_cycles cannot exceed the registered ResearchLoop limit.")
        if self.policy.maximum_executions > self.execution_policy.maximum_task_executions:
            raise ValueError("maximum_executions cannot exceed the registered task-execution budget.")
        if self.policy.maximum_capability_invocations > self.execution_policy.maximum_capability_invocations:
            raise ValueError("maximum_capability_invocations cannot exceed the execution policy budget.")
        if not callable(self.request_factory) or not callable(self.approval_provider):
            raise ValueError("Typed execution-request and researcher-approval callbacks are required.")
        if not isinstance(self.controller_id, str) or not self.controller_id.strip():
            raise ValueError("controller_id must be non-empty text.")

    def to_dict(self):
        """Serialize the request manifest; runtime providers/callbacks remain bindings."""
        execution = self.execution_policy
        return {
            "controller_id": self.controller_id, "run_id": self.run.run_id,
            "plan_id": self.plan.plan_id, "plan_revision": self.plan.revision,
            "review_id": self.review.review_id, "loop_id": self.loop.loop_id,
            "execution_policy": {
                "maximum_iterations": execution.maximum_iterations,
                "maximum_capability_invocations": execution.maximum_capability_invocations,
                "maximum_task_executions": execution.maximum_task_executions,
                "allowed_capabilities": [item.value for item in execution.allowed_capabilities],
                "allowed_stages": [item.value for item in execution.allowed_stages],
                "allowed_task_ids": list(execution.allowed_task_ids),
                "allow_continuation": execution.allow_continuation,
                "stop_on_failure": execution.stop_on_failure,
                "stop_on_unresolved_blocker": execution.stop_on_unresolved_blocker,
            },
            "policy": self.policy.to_dict(),
            "runtime_bindings_required": True,
        }

    def to_json(self):
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)


@dataclass(frozen=True)
class AutonomousControllerCycle:
    cycle_number: int
    context_id: str
    decision: Optional[ResearchDecision] = None
    validation: Optional[ResearchDecisionTrace] = None
    request_id: Optional[str] = None
    approval_id: Optional[str] = None
    execution: Optional[AutonomousExecutionResult] = None
    status: str = "STARTED"
    reason: Optional[str] = None

    def to_dict(self):
        return {
            "cycle_number": self.cycle_number, "context_id": self.context_id,
            "decision": self.decision.to_dict() if self.decision else None,
            "validation": self.validation.to_dict() if self.validation else None,
            "request_id": self.request_id, "approval_id": self.approval_id,
            "execution": self.execution.to_dict() if self.execution else None,
            "status": self.status, "reason": self.reason,
        }


@dataclass(frozen=True)
class AutonomousControllerTrace:
    controller_id: str
    run_id: str
    loop_id: str
    created_at: str
    cycles: tuple[AutonomousControllerCycle, ...]

    def to_dict(self):
        return {"controller_id": self.controller_id, "run_id": self.run_id,
                "loop_id": self.loop_id, "created_at": self.created_at,
                "cycles": [item.to_dict() for item in self.cycles]}

    def to_json(self):
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)


@dataclass(frozen=True)
class AutonomousControllerResult:
    stop_reason: AutonomousControllerStopReason
    cycles_completed: int
    executions_completed: int
    trace: AutonomousControllerTrace
    failure_reason: Optional[str] = None

    def __post_init__(self):
        object.__setattr__(self, "stop_reason", AutonomousControllerStopReason(self.stop_reason))
        if self.cycles_completed != len(self.trace.cycles):
            raise ValueError("cycles_completed must match trace length.")

    def to_dict(self):
        return {"stop_reason": self.stop_reason.value, "cycles_completed": self.cycles_completed,
                "executions_completed": self.executions_completed, "failure_reason": self.failure_reason,
                "trace": self.trace.to_dict()}

    def to_json(self):
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)


def run_autonomous_controller(request: AutonomousControllerRequest) -> AutonomousControllerResult:
    """Run at most the caller's finite cycle/execution bounds; never retries."""
    if not isinstance(request, AutonomousControllerRequest):
        raise ValueError("A typed AutonomousControllerRequest is required.")
    cycles: list[AutonomousControllerCycle] = []
    executions = 0
    execution_attempts = 0
    capability_invocations = retrieval_actions = verification_actions = 0
    reason = AutonomousControllerStopReason.COMPLETED
    failure = None
    for number in range(1, request.policy.maximum_cycles + 1):
        try:
            loop = _current_loop(request)
        except Exception as exc:
            reason = AutonomousControllerStopReason.STALE_STATE
            failure = f"{type(exc).__name__}: {exc}"
            cycles.append(AutonomousControllerCycle(number, "unavailable", status="STALE_STATE", reason=failure))
            break
        if loop.status in {ResearchLoopStatus.COMPLETED, ResearchLoopStatus.STOPPED}:
            reason = AutonomousControllerStopReason.DECISION_STOP
            break
        if loop.status is ResearchLoopStatus.FAILED:
            reason = AutonomousControllerStopReason.EXECUTION_FAILED
            break
        if loop.status is ResearchLoopStatus.WAITING_FOR_RESEARCHER:
            reason = AutonomousControllerStopReason.WAIT_FOR_RESEARCHER
            break
        if any(item.blocks_progress for item in loop.unresolved_work):
            reason = AutonomousControllerStopReason.BLOCKED
            break
        try:
            context = build_autonomous_research_context(
                request.plan, request.run, request.review, loop, request.state, request.execution_policy)
            decision = _get_decision(request, context)
            validation = validate_research_decision(
                decision, plan=request.plan, run=request.run, review=request.review,
                loop=loop, state=request.state, policy=request.execution_policy)
        except _NoAction:
            reason = AutonomousControllerStopReason.NO_ACTION
            cycles.append(AutonomousControllerCycle(number, context.context_id, status="NO_ACTION",
                                                    reason="8A returned no decision."))
            break
        except Exception as exc:
            reason = AutonomousControllerStopReason.ERROR
            failure = f"{type(exc).__name__}: {exc}"
            cycles.append(AutonomousControllerCycle(number, "unavailable", status="FAILED", reason=failure))
            break
        if validation.validation_status is not ValidationStatus.ACCEPTED:
            reason = AutonomousControllerStopReason.STALE_STATE if any("stale" in item for item in validation.rejection_reasons) else AutonomousControllerStopReason.VALIDATION_REJECTED
            cycles.append(AutonomousControllerCycle(number, context.context_id, decision, validation,
                                                    status="VALIDATION_REJECTED", reason="; ".join(validation.rejection_reasons)))
            break
        action = decision.proposal.action_type
        if action is ResearchActionType.STOP:
            cycles.append(AutonomousControllerCycle(number, context.context_id, decision, validation, status="STOPPED"))
            reason = AutonomousControllerStopReason.DECISION_STOP
            break
        if action in {ResearchActionType.RETRIEVE_EVIDENCE, ResearchActionType.VERIFY_CLAIM,
                      ResearchActionType.GENERATE_IMPROVEMENT, ResearchActionType.INVESTIGATE_PRIOR_WORK,
                      ResearchActionType.PLAN_EXPERIMENT} and capability_invocations >= request.policy.maximum_capability_invocations:
            cycles.append(AutonomousControllerCycle(number, context.context_id, decision, validation,
                status="CAPABILITY_LIMIT", reason="Controller capability-invocation limit reached."))
            reason = AutonomousControllerStopReason.LOOP_LIMIT
            break
        if execution_attempts >= request.policy.maximum_total_research_actions:
            cycles.append(AutonomousControllerCycle(number, context.context_id, decision, validation,
                status="ACTION_LIMIT", reason="Controller total research-action limit reached."))
            reason = AutonomousControllerStopReason.MAXIMUM_EXECUTIONS
            break
        if action is ResearchActionType.RETRIEVE_EVIDENCE and retrieval_actions >= request.policy.maximum_retrieval_actions:
            cycles.append(AutonomousControllerCycle(number, context.context_id, decision, validation,
                status="RETRIEVAL_LIMIT", reason="Controller retrieval-action limit reached."))
            reason = AutonomousControllerStopReason.LOOP_LIMIT
            break
        if action is ResearchActionType.VERIFY_CLAIM and verification_actions >= request.policy.maximum_verification_actions:
            cycles.append(AutonomousControllerCycle(number, context.context_id, decision, validation,
                status="VERIFICATION_LIMIT", reason="Controller verification-action limit reached."))
            reason = AutonomousControllerStopReason.LOOP_LIMIT
            break
        if decision.proposal.action_type in {ResearchActionType.REQUEST_RESEARCHER_REVIEW,
                                             ResearchActionType.RETURN_TO_PLANNING}:
            # These controlled non-capability actions still go through 8B below.
            pass
        if execution_attempts >= request.policy.maximum_executions:
            cycles.append(AutonomousControllerCycle(number, context.context_id, decision, validation,
                                                    status="EXECUTION_LIMIT"))
            reason = AutonomousControllerStopReason.MAXIMUM_EXECUTIONS
            break
        try:
            execution_request = request.request_factory(decision, validation, context)
            if not isinstance(execution_request, AutonomousExecutionRequest):
                raise ValueError("request_factory must return an AutonomousExecutionRequest.")
            if execution_request.decision != decision or execution_request.prior_validation != validation:
                raise ValueError("8B request must carry this exact 8A decision and validation.")
            if execution_request.retry_of_execution_id is not None:
                raise ValueError("8C never creates or submits automatic retry authorization.")
            task_id = decision.proposal.task_id
            if task_id and decision.proposal.required_capabilities:
                prior = _latest_task_execution(request.state, request.run.run_id, task_id)
                if prior is not None and prior.status.value == "FAILED":
                    cycles.append(AutonomousControllerCycle(number, context.context_id, decision, validation,
                        execution_request.request_id, status="WAITING_FOR_RESEARCHER",
                        reason="A failed execution requires explicit Phase 5 retry authorization."))
                    reason = AutonomousControllerStopReason.WAIT_FOR_RESEARCHER
                    break
                if prior is not None and prior.status.value == "COMPLETED":
                    authorization = _find_continuation_authorization(
                        loop, task_id, execution_request.iteration_id, prior.execution_id)
                    if authorization is None:
                        cycles.append(AutonomousControllerCycle(number, context.context_id, decision, validation,
                            execution_request.request_id, status="WAITING_FOR_RESEARCHER",
                            reason="Phase 7C continuation authorization is required for this task and iteration."))
                        reason = AutonomousControllerStopReason.WAIT_FOR_RESEARCHER
                        break
                    if execution_request.continuation_authorization_id is None:
                        execution_request = replace(
                            execution_request, continuation_authorization_id=authorization.authorization_id)
            approval = request.approval_provider(execution_request)
            if request.policy.require_researcher_review and not isinstance(approval, AutonomousExecutionApproval):
                cycles.append(AutonomousControllerCycle(number, context.context_id, decision, validation,
                    execution_request.request_id, status="WAITING_FOR_RESEARCHER",
                    reason="Proposal-scoped researcher approval was not supplied."))
                reason = AutonomousControllerStopReason.WAIT_FOR_RESEARCHER
                break
            result = execute_autonomous_proposal(
                request.plan, request.run, request.review, _current_loop(request), request.state,
                request.execution_policy, execution_request, approval,
                retrieval_provider=request.retrieval_provider, llm_provider=request.llm_provider,
                search_provider=request.search_provider)
            cycles.append(AutonomousControllerCycle(number, context.context_id, decision, validation,
                execution_request.request_id, approval.approval_id if isinstance(approval, AutonomousExecutionApproval) else None,
                result, result.status.value, result.trace.reason))
            execution_attempts += 1
            if result.dispatch is not None:
                capability_invocations += 1
                if result.dispatch.capability.value == "local_retrieval":
                    retrieval_actions += 1
                elif result.dispatch.capability.value == "evidence_verification":
                    verification_actions += 1
        except Exception as exc:
            reason = AutonomousControllerStopReason.ERROR
            failure = f"{type(exc).__name__}: {exc}"
            cycles.append(AutonomousControllerCycle(number, context.context_id, decision, validation,
                                                    status="FAILED", reason=failure))
            break
        result = cycles[-1].execution
        if result is None or result.status is not AutonomousExecutionStatus.COMPLETED:
            reason = (AutonomousControllerStopReason.EXECUTION_FAILED
                      if result and result.status is AutonomousExecutionStatus.FAILED
                      else AutonomousControllerStopReason.EXECUTION_REJECTED)
            failure = result.trace.reason if result else "8B returned no execution result."
            break
        executions += 1
        if result.trace.action_type is ResearchActionType.REQUEST_RESEARCHER_REVIEW:
            reason = AutonomousControllerStopReason.WAIT_FOR_RESEARCHER
            break
        if result.trace.action_type is ResearchActionType.STOP:
            reason = AutonomousControllerStopReason.DECISION_STOP
            break
        # A successful execution does not create continuation authorization.
        if number < request.policy.maximum_cycles and (
                not request.execution_policy.allow_continuation
                or not _has_existing_continuation(request.state, loop.loop_id)):
            reason = AutonomousControllerStopReason.WAIT_FOR_RESEARCHER
            break
    else:
        reason = AutonomousControllerStopReason.MAXIMUM_CYCLES
    if (len(cycles) >= request.policy.maximum_cycles and reason is AutonomousControllerStopReason.COMPLETED):
        reason = AutonomousControllerStopReason.MAXIMUM_CYCLES
    trace = AutonomousControllerTrace(request.controller_id, request.run.run_id, request.loop.loop_id,
                                      _now(), tuple(cycles))
    return AutonomousControllerResult(reason, len(cycles), executions, trace, failure)


def _current_loop(request):
    loops = [item for item in request.state.research_loops if item.loop_id == request.loop.loop_id]
    if len(loops) != 1:
        raise ValueError("ResearchLoop no longer resolves uniquely in current ResearchState.")
    return loops[0]


def _get_decision(request, context):
    provider = request.decision_provider
    decision = autonomous_decide(context, provider) if callable(getattr(provider, "generate", None)) else provider(context)
    if decision is None:
        raise _NoAction
    if not isinstance(decision, ResearchDecision):
        raise ValueError("8A decision provider must return one ResearchDecision.")
    return decision


class _NoAction(Exception):
    pass


def _has_existing_continuation(state, loop_id):
    loop = next((item for item in state.research_loops if item.loop_id == loop_id), None)
    return bool(loop and loop.continuation_authorizations)


def _latest_task_execution(state, run_id, task_id):
    matches = [item for item in state.task_executions
               if item.run_id == run_id and item.task_id == task_id]
    return matches[-1] if matches else None


def _find_continuation_authorization(loop, task_id, iteration_id, previous_execution_id):
    matches = [item for item in loop.continuation_authorizations
               if item.task_id == task_id and item.iteration_id == iteration_id
               and item.previous_execution_id == previous_execution_id]
    return matches[0] if len(matches) == 1 else None


__all__ = ["AutonomousControllerPolicy", "AutonomousControllerRequest", "AutonomousControllerCycle",
           "AutonomousControllerTrace", "AutonomousControllerResult", "AutonomousControllerStopReason",
           "run_autonomous_controller"]
