"""Explicit authorization boundary from an approved plan to a ResearchRun.

The resulting run records which plan revision and human review authorized it.
Authorization is not scientific validation, and this module performs no
research execution.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from research_plan_review import PlanReview, is_execution_approved
from research_planning import ResearchPlan
from research_state import ResearchRun, ResearchState


class ResearchRunAuthorizationError(ValueError):
    """The supplied plan/review pair does not authorize a ResearchRun."""


class ResearchRunRegistrationError(RuntimeError):
    """Authorization passed, but the ResearchState could not record the run."""


def authorize_research_run(
    plan: ResearchPlan,
    review: PlanReview,
    state: Optional[ResearchState] = None,
    *,
    run_metadata: Optional[Dict[str, Any]] = None,
) -> ResearchRun:
    """Create a run only for the exact structurally valid approved revision.

    The caller supplies the review explicitly. No review history is searched
    and no approval is carried forward to another plan revision. When a state
    is supplied, registration uses ``ResearchState.add_run`` after every
    authorization check has passed.
    """
    if not isinstance(plan, ResearchPlan):
        raise ResearchRunAuthorizationError("A ResearchPlan is required to authorize a run.")
    if not isinstance(review, PlanReview):
        raise ResearchRunAuthorizationError("A PlanReview is required to authorize a run.")
    if state is not None and not isinstance(state, ResearchState):
        raise ResearchRunAuthorizationError("state must be a ResearchState when supplied.")
    if run_metadata is not None and not isinstance(run_metadata, dict):
        raise ResearchRunAuthorizationError("run_metadata must be a dictionary when supplied.")

    plan_errors = plan.validate()
    if plan_errors:
        raise ResearchRunAuthorizationError(
            "Cannot authorize a run from an invalid ResearchPlan: " + "; ".join(plan_errors)
        )

    try:
        # Re-parse the serialized form so even an improperly mutated frozen
        # review cannot bypass Phase 4B's constructor validation.
        validated_review = PlanReview.from_dict(review.to_dict())
        if validated_review != review:
            raise ValueError("Serialized PlanReview does not match the supplied review record.")
        review.validate_against(plan)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ResearchRunAuthorizationError(f"PlanReview validation failed: {exc}") from exc

    if not is_execution_approved(plan, review):
        raise ResearchRunAuthorizationError(
            f"Review {review.review_id!r} does not approve plan {plan.plan_id!r} "
            f"revision {plan.revision}."
        )

    # Construct only after all deterministic authorization checks have passed.
    run = ResearchRun(
        plan_id=plan.plan_id,
        plan_revision=plan.revision,
        approval_review_id=review.review_id,
        metadata=dict(run_metadata) if run_metadata is not None else {},
    )
    if state is None:
        return run

    prior_runs = list(state.research_runs)
    prior_updated_at = state.updated_at
    try:
        registered = state.add_run(run)
        if registered is not run:
            raise RuntimeError("ResearchState.add_run returned a different ResearchRun record.")
    except Exception as exc:
        # ResearchState.add_run is append-after-validation. Restore the prior
        # snapshot only if an overridden/failing implementation mutated it.
        if state.research_runs != prior_runs:
            state.research_runs[:] = prior_runs
        if state.updated_at != prior_updated_at:
            state.updated_at = prior_updated_at
        raise ResearchRunRegistrationError(
            "Plan approval was valid, but ResearchState registration failed; no run was returned: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    return run


__all__ = [
    "ResearchRunAuthorizationError",
    "ResearchRunRegistrationError",
    "authorize_research_run",
]
