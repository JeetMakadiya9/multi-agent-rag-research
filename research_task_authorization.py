"""Deterministic task-selection authorization for an approved ResearchRun.

This module records which planned tasks are authorized for future execution.
It does not execute tasks, expand dependencies, or make claims about task
quality or scientific validity.
"""

from __future__ import annotations

from typing import Optional, Sequence

from research_plan_review import PlanReview, is_execution_approved
from research_planning import ResearchPlan
from research_state import ResearchRun, ResearchState


class ResearchTaskAuthorizationError(ValueError):
    """The supplied run, approval, or task selection is not valid."""


class ResearchTaskAuthorizationRecordingError(RuntimeError):
    """Task authorization passed validation but could not be recorded in state."""


def authorize_research_tasks(
    plan: ResearchPlan,
    run: ResearchRun,
    task_ids: Sequence[str],
    review: PlanReview,
    state: Optional[ResearchState] = None,
) -> ResearchRun:
    """Record an explicit, ordered task selection on an approved run.

    The caller must supply the exact plan, run, task IDs, and approval review.
    The review is revalidated; no review history is searched and approval is
    never carried across plan revisions. If ``state`` is supplied, ``run``
    must be the exact object already registered in that state. All validation
    completes before the run or state is mutated.
    """
    if not isinstance(plan, ResearchPlan):
        raise ResearchTaskAuthorizationError("A ResearchPlan is required.")
    if not isinstance(run, ResearchRun):
        raise ResearchTaskAuthorizationError("A ResearchRun is required.")
    if not isinstance(review, PlanReview):
        raise ResearchTaskAuthorizationError("The approving PlanReview is required.")
    if state is not None and not isinstance(state, ResearchState):
        raise ResearchTaskAuthorizationError("state must be a ResearchState when supplied.")

    plan_errors = plan.validate()
    if plan_errors:
        raise ResearchTaskAuthorizationError(
            "Cannot authorize tasks from an invalid ResearchPlan: " + "; ".join(plan_errors)
        )

    # Validate the current mutable run fields using the model's own contract,
    # without replacing or mutating the caller's registered record.
    try:
        ResearchRun(
            run_id=run.run_id,
            created_at=run.created_at,
            status=run.status,
            metadata=dict(run.metadata),
            plan_id=run.plan_id,
            plan_revision=run.plan_revision,
            approval_review_id=run.approval_review_id,
            authorized_task_ids=(
                list(run.authorized_task_ids)
                if run.authorized_task_ids is not None
                else None
            ),
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise ResearchTaskAuthorizationError(f"ResearchRun validation failed: {exc}") from exc

    if run.plan_id is None or run.plan_revision is None or run.approval_review_id is None:
        raise ResearchTaskAuthorizationError(
            "ResearchRun has no complete Phase 4C plan approval lineage."
        )
    if run.plan_id != plan.plan_id:
        raise ResearchTaskAuthorizationError(
            f"ResearchRun plan_id {run.plan_id!r} does not match supplied plan {plan.plan_id!r}."
        )
    if run.plan_revision != plan.revision:
        raise ResearchTaskAuthorizationError(
            f"ResearchRun plan revision {run.plan_revision} does not match supplied plan revision {plan.revision}."
        )

    try:
        validated_review = PlanReview.from_dict(review.to_dict())
        if validated_review != review:
            raise ValueError("Serialized PlanReview does not match the supplied review record.")
        review.validate_against(plan)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ResearchTaskAuthorizationError(f"PlanReview validation failed: {exc}") from exc

    if review.plan_id != plan.plan_id:
        raise ResearchTaskAuthorizationError("PlanReview plan_id does not match the supplied plan.")
    if review.plan_revision != plan.revision:
        raise ResearchTaskAuthorizationError("PlanReview revision does not match the supplied plan revision.")
    if review.review_id != run.approval_review_id:
        raise ResearchTaskAuthorizationError(
            f"ResearchRun approval_review_id {run.approval_review_id!r} does not match supplied "
            f"review {review.review_id!r}."
        )
    if not is_execution_approved(plan, review):
        raise ResearchTaskAuthorizationError(
            f"Review {review.review_id!r} does not approve plan {plan.plan_id!r} revision {plan.revision}."
        )

    if run.authorized_task_ids is not None:
        raise ResearchTaskAuthorizationError(
            f"ResearchRun {run.run_id!r} already has an authorized task set."
        )
    if isinstance(task_ids, (str, bytes)) or not isinstance(task_ids, (list, tuple)):
        raise ResearchTaskAuthorizationError("task_ids must be a non-empty ordered sequence of task ID strings.")
    if not task_ids:
        raise ResearchTaskAuthorizationError("At least one task ID must be authorized.")
    for index, task_id in enumerate(task_ids):
        if not isinstance(task_id, str) or not task_id.strip():
            raise ResearchTaskAuthorizationError(
                f"task_ids[{index}] must be a non-empty string."
            )
    if len(set(task_ids)) != len(task_ids):
        raise ResearchTaskAuthorizationError("Duplicate task IDs are not allowed.")

    plan_task_ids = {task.task_id for task in plan.tasks}
    unknown = [task_id for task_id in task_ids if task_id not in plan_task_ids]
    if unknown:
        raise ResearchTaskAuthorizationError(
            "Task ID(s) are not present in the exact ResearchPlan revision: "
            + ", ".join(repr(task_id) for task_id in unknown)
        )

    if state is not None and not any(existing is run for existing in state.research_runs):
        raise ResearchTaskAuthorizationError(
            "ResearchRun must be registered in the supplied ResearchState before task authorization."
        )

    # The explicit selection is the entire authorized set, in caller order.
    # Dependencies and task statuses remain untouched.
    selected_ids = list(task_ids)
    prior_updated_at = state.updated_at if state is not None else None
    try:
        run.authorized_task_ids = selected_ids
        if state is not None:
            state.touch()
    except Exception as exc:
        run.authorized_task_ids = None
        if state is not None and state.updated_at != prior_updated_at:
            state.updated_at = prior_updated_at  # type: ignore[assignment]
        raise ResearchTaskAuthorizationRecordingError(
            "Task authorization was valid, but recording failed and was rolled back: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    return run


__all__ = [
    "ResearchTaskAuthorizationError",
    "ResearchTaskAuthorizationRecordingError",
    "authorize_research_tasks",
]
