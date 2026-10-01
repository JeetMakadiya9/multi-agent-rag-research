"""Caller-controlled continuation decisions for an active research loop.

This module authorizes a next step only. It never dispatches a capability.
Continuation of a completed execution is separate from retry of a failed one.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from research_loop import (
    LoopIterationStatus, LoopStage, ResearchLoop, ResearchLoopStatus, StopReason,
    TaskContinuationAuthorization, TransitionAuthority, _get_loop,
    begin_loop_iteration, finish_loop_iteration, stop_research_loop,
    transition_research_loop,
)
from research_plan_review import PlanReview, is_execution_approved
from research_planning import ResearchPlan, ResearchTask
from research_state import ResearchRun, ResearchState
from research_task_execution import TaskExecutionStatus


class ResearchContinuationAction(str, Enum):
    STOP = "STOP"
    WAIT_FOR_RESEARCHER = "WAIT_FOR_RESEARCHER"
    CONTINUE_SAME_TASK = "CONTINUE_SAME_TASK"
    CONTINUE_WITH_TASK = "CONTINUE_WITH_TASK"
    RETURN_TO_PLANNING = "RETURN_TO_PLANNING"


@dataclass(frozen=True)
class ResearchContinuationResult:
    action: ResearchContinuationAction
    loop_id: str
    status: ResearchLoopStatus
    stage: LoopStage
    task_id: Optional[str] = None
    iteration_id: Optional[str] = None
    continuation_authorization_id: Optional[str] = None
    execution_occurred: bool = False


def apply_research_continuation(
    plan: ResearchPlan,
    run: ResearchRun,
    review: PlanReview,
    loop: ResearchLoop,
    state: ResearchState,
    *,
    action: ResearchContinuationAction,
    previous_execution_id: Optional[str] = None,
    task_id: Optional[str] = None,
) -> ResearchContinuationResult:
    """Apply one explicit caller decision; successful continuation is not execution."""
    try:
        action = action if isinstance(action, ResearchContinuationAction) else ResearchContinuationAction(action)
    except (TypeError, ValueError) as exc:
        raise ValueError("A supported explicit ResearchContinuationAction is required.") from exc
    _validate(plan, run, review, loop, state)

    if action is ResearchContinuationAction.STOP:
        if task_id is not None or previous_execution_id is not None:
            raise ValueError("STOP does not accept a task or source execution.")
        updated = stop_research_loop(
            state, loop.loop_id, StopReason.RESEARCHER_REQUESTED_STOP,
            authority=TransitionAuthority.RESEARCHER,
            message="Researcher explicitly stopped the research loop.",
        )
        return _result(action, updated)
    if action is ResearchContinuationAction.WAIT_FOR_RESEARCHER:
        if task_id is not None or previous_execution_id is not None:
            raise ValueError("WAIT_FOR_RESEARCHER does not accept a task or source execution.")
        updated = stop_research_loop(
            state, loop.loop_id, StopReason.WAITING_FOR_RESEARCHER,
            message="Caller explicitly paused for a researcher decision.",
        )
        return _result(action, updated)
    if action is ResearchContinuationAction.RETURN_TO_PLANNING:
        if task_id is not None or previous_execution_id is not None:
            raise ValueError("RETURN_TO_PLANNING does not accept a task or source execution.")
        updated = transition_research_loop(
            state, loop.loop_id, LoopStage.PLANNING,
            reason="Researcher explicitly returned the active workflow to planning.",
            authority=TransitionAuthority.RESEARCHER,
        )
        return _result(action, updated)

    source = _completed_source(loop, state, previous_execution_id)
    if action is ResearchContinuationAction.CONTINUE_SAME_TASK:
        if task_id is not None:
            raise ValueError("CONTINUE_SAME_TASK uses the task ID from its completed source execution.")
        selected_task_id = source.task_id
    else:
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("CONTINUE_WITH_TASK requires an explicit authorized task_id.")
        selected_task_id = task_id
        if selected_task_id == source.task_id:
            raise ValueError("CONTINUE_WITH_TASK must select a different task; use CONTINUE_SAME_TASK.")
        task = next((item for item in plan.tasks if item.task_id == selected_task_id), None)
        if task is None or selected_task_id not in (run.authorized_task_ids or ()):
            raise ValueError("CONTINUE_WITH_TASK task must belong to the exact approved plan and authorized run.")
        if selected_task_id not in loop.unresolved_task_ids:
            raise ValueError("CONTINUE_WITH_TASK task must remain unresolved in this loop.")
        existing_attempts = [item for item in state.task_executions
                             if item.run_id == run.run_id and item.task_id == selected_task_id]
        if existing_attempts:
            raise ValueError("CONTINUE_WITH_TASK cannot retry or continue a previously executed task implicitly.")

    if loop.status not in {ResearchLoopStatus.RUNNING, ResearchLoopStatus.WAITING_FOR_RESEARCHER}:
        raise ValueError("Continuation requires a nonterminal active loop.")
    if loop.current_task_id != source.task_id:
        raise ValueError("The completed source execution must belong to the loop's currently selected task.")
    if source.iteration_id != loop.current_iteration_id:
        raise ValueError("The completed source execution must belong to the active loop iteration.")
    if len(loop.loop_iterations) >= loop.max_iterations:
        # Let existing bounded-loop behavior stop the loop deterministically.
        if loop.current_iteration_id:
            finish_loop_iteration(state, loop.loop_id, status=LoopIterationStatus.COMPLETED,
                                  reason="Caller requested continuation at the iteration bound.")
        stopped = _get_loop(state, loop.loop_id)
        if stopped.status is ResearchLoopStatus.RUNNING:
            stop_research_loop(state, loop.loop_id, StopReason.MAXIMUM_ITERATIONS_REACHED,
                                message="Maximum configured loop iterations reached.")
        raise ValueError("Maximum loop iterations reached; continuation was not authorized.")

    before = state.to_dict()
    try:
        if loop.current_iteration_id is not None:
            finish_loop_iteration(state, loop.loop_id, status=LoopIterationStatus.COMPLETED,
                                  reason=f"Caller authorized {action.value.lower()} after completed execution.")
        current = _get_loop(state, loop.loop_id)
        if current.status is ResearchLoopStatus.STOPPED:
            raise ValueError("Maximum loop iterations reached; continuation was not authorized.")
        if current.status is ResearchLoopStatus.WAITING_FOR_RESEARCHER:
            transition_research_loop(state, loop.loop_id, LoopStage.RESEARCH,
                                     reason="Researcher explicitly authorized continuation.",
                                     authority=TransitionAuthority.RESEARCHER)
        record = begin_loop_iteration(state, loop.loop_id, (selected_task_id,))
        if record is None:
            raise ValueError("Maximum loop iterations reached; continuation was not authorized.")
        updated = _get_loop(state, loop.loop_id)
        auth_id = None
        if action is ResearchContinuationAction.CONTINUE_SAME_TASK:
            authorization = TaskContinuationAuthorization(
                run_id=run.run_id, plan_id=plan.plan_id, plan_revision=plan.revision,
                task_id=selected_task_id, previous_execution_id=source.execution_id,
                iteration_id=record.iteration_id,
            )
            updated = _replace_continuation_authorization(state, updated, authorization)
            auth_id = authorization.authorization_id
        return ResearchContinuationResult(
            action, updated.loop_id, updated.status, updated.current_stage,
            selected_task_id, record.iteration_id, auth_id, False,
        )
    except Exception:
        _restore_state(state, before)
        raise


def _validate(plan, run, review, loop, state) -> None:
    if not isinstance(plan, ResearchPlan) or plan.validate():
        raise ValueError("A structurally valid ResearchPlan is required.")
    if not isinstance(run, ResearchRun) or not isinstance(review, PlanReview):
        raise ValueError("ResearchRun and PlanReview are required.")
    if not isinstance(state, ResearchState) or not isinstance(loop, ResearchLoop):
        raise ValueError("A ResearchState and typed ResearchLoop are required.")
    review.validate_against(plan)
    if not is_execution_approved(plan, review):
        raise ValueError("The supplied review does not approve this exact plan revision.")
    if (run.plan_id != plan.plan_id or run.plan_revision != plan.revision
            or run.approval_review_id != review.review_id
            or loop.run_id != run.run_id or loop.plan_id != plan.plan_id
            or loop.plan_revision != plan.revision):
        raise ValueError("Run, loop, review, and exact approved plan revision do not match.")
    if not any(item is run for item in state.research_runs):
        raise ValueError("ResearchRun must be registered in ResearchState.")
    if not any(item == loop for item in state.research_loops):
        raise ValueError("ResearchLoop must be registered in ResearchState.")
    errors = loop.validate_references(state)
    if errors:
        raise ValueError("ResearchLoop references are invalid: " + "; ".join(errors))
    if loop.status in {ResearchLoopStatus.COMPLETED, ResearchLoopStatus.STOPPED, ResearchLoopStatus.FAILED}:
        raise ValueError("Terminal research loops cannot accept continuation actions.")


def _completed_source(loop, state, execution_id):
    if not isinstance(execution_id, str) or not execution_id.strip():
        raise ValueError("An explicit previous_execution_id is required for continuation.")
    matches = [item for item in state.task_executions if item.execution_id == execution_id]
    if len(matches) != 1 or matches[0].status is not TaskExecutionStatus.COMPLETED:
        raise ValueError("Continuation source must resolve to one completed execution.")
    execution = matches[0]
    results = [item for item in loop.capability_results
               if item.execution_id == execution_id and item.status.value == "COMPLETED"]
    if len(results) != 1:
        raise ValueError("Continuation source must be a successful capability execution recorded on this loop.")
    if execution.run_id != loop.run_id or execution.plan_id != loop.plan_id or execution.plan_revision != loop.plan_revision:
        raise ValueError("Continuation source belongs to another run or approved plan revision.")
    return execution


def _replace_continuation_authorization(state, loop, authorization):
    from dataclasses import replace
    from research_loop import _replace_loop
    if any(item.authorization_id == authorization.authorization_id for item in loop.continuation_authorizations):
        raise ValueError("Duplicate continuation authorization ID.")
    updated = replace(loop, continuation_authorizations=loop.continuation_authorizations + (authorization,))
    return _replace_loop(state, loop, updated, "continuation_authorized",
                         f"Explicit continuation authorized for task {authorization.task_id}.")


def _restore_state(state, payload):
    restored = ResearchState.from_dict(payload)
    state.__dict__.clear()
    state.__dict__.update(restored.__dict__)


def _result(action, loop):
    return ResearchContinuationResult(action, loop.loop_id, loop.status, loop.current_stage,
                                      loop.current_task_id, loop.current_iteration_id, None, False)


__all__ = ["ResearchContinuationAction", "ResearchContinuationResult", "apply_research_continuation"]
