"""Deterministic human review records for research-plan authorization.

Approval means that a researcher authorized future execution of one exact
plan revision. It does not establish scientific correctness, novelty, or truth.
This module records human decisions only; it does not evaluate plans or execute
them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
from typing import Any, Dict, Iterable, Optional, Tuple
from uuid import uuid4

from research_planning import ResearchPlan


def _new_review_id() -> str:
    return f"review_{uuid4().hex[:10]}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ReviewStatus(str, Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    CHANGES_REQUESTED = "CHANGES_REQUESTED"
    REJECTED = "REJECTED"


def _timestamp(value: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("PlanReview.reviewed_at must be a non-empty ISO 8601 timestamp.")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(
            "PlanReview.reviewed_at must be a valid ISO 8601 timestamp with a timezone."
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(
            "PlanReview.reviewed_at must include a timezone, for example +00:00."
        )
    return parsed


@dataclass(frozen=True)
class PlanReview:
    """One append-only human decision about one ResearchPlan revision."""

    plan_id: str
    plan_revision: int
    status: ReviewStatus = ReviewStatus.PENDING
    reviewer: Optional[str] = None
    reviewed_at: str = field(default_factory=_now)
    comments: str = ""
    requested_changes: Tuple[str, ...] = ()
    review_id: str = field(default_factory=_new_review_id)

    def __post_init__(self) -> None:
        if not isinstance(self.plan_id, str) or not self.plan_id.strip():
            raise ValueError("PlanReview.plan_id is required.")
        if not isinstance(self.review_id, str) or not self.review_id.strip():
            raise ValueError("PlanReview.review_id is required.")
        if (
            not isinstance(self.plan_revision, int)
            or isinstance(self.plan_revision, bool)
            or self.plan_revision < 1
        ):
            raise ValueError("PlanReview.plan_revision must be a positive integer.")
        if not isinstance(self.status, ReviewStatus):
            try:
                status = ReviewStatus(self.status)
            except (TypeError, ValueError) as exc:
                supported = ", ".join(item.value for item in ReviewStatus)
                raise ValueError(
                    f"Unsupported review status {self.status!r}; expected one of: {supported}."
                ) from exc
            object.__setattr__(self, "status", status)
        _timestamp(self.reviewed_at)
        if not isinstance(self.comments, str):
            raise ValueError("PlanReview.comments must be a string.")
        if self.reviewer is not None and (
            not isinstance(self.reviewer, str) or not self.reviewer.strip()
        ):
            raise ValueError("PlanReview.reviewer must be a non-empty string when supplied.")

        changes = self.requested_changes
        if not isinstance(changes, (list, tuple)):
            raise ValueError("PlanReview.requested_changes must be a sequence of strings.")
        normalized_changes = tuple(changes)
        for index, change in enumerate(normalized_changes):
            if not isinstance(change, str) or not change.strip():
                raise ValueError(
                    f"PlanReview.requested_changes[{index}] must be a non-empty string."
                )
        object.__setattr__(self, "requested_changes", normalized_changes)

        if self.status in {
            ReviewStatus.APPROVED,
            ReviewStatus.CHANGES_REQUESTED,
            ReviewStatus.REJECTED,
        } and self.reviewer is None:
            raise ValueError(f"{self.status.value} review requires a reviewer.")
        if self.status is ReviewStatus.CHANGES_REQUESTED and not normalized_changes:
            raise ValueError(
                "CHANGES_REQUESTED review requires at least one requested change."
            )

    def validate_against(self, plan: ResearchPlan) -> None:
        """Raise a clear error unless this review matches a valid plan revision."""
        if not isinstance(plan, ResearchPlan):
            raise ValueError("PlanReview must be validated against a ResearchPlan.")
        plan_errors = plan.validate()
        if plan_errors:
            raise ValueError(
                "Review cannot be validated against a structurally invalid ResearchPlan: "
                + "; ".join(plan_errors)
            )
        if self.plan_id != plan.plan_id:
            raise ValueError(
                f"Review refers to plan ID {self.plan_id!r}, but the supplied plan is "
                f"{plan.plan_id!r}."
            )
        if self.plan_revision != plan.revision:
            raise ValueError(
                f"Review revision {self.plan_revision} does not match plan revision {plan.revision}."
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize every review field into JSON-compatible values."""
        return {
            "review_id": self.review_id,
            "plan_id": self.plan_id,
            "plan_revision": self.plan_revision,
            "status": self.status.value,
            "reviewer": self.reviewer,
            "reviewed_at": self.reviewed_at,
            "comments": self.comments,
            "requested_changes": list(self.requested_changes),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlanReview":
        if not isinstance(data, dict):
            raise ValueError("PlanReview payload must be a JSON object.")
        required = {"review_id", "plan_id", "plan_revision", "status", "reviewed_at"}
        optional = {"reviewer", "comments", "requested_changes"}
        missing = sorted(required - data.keys())
        if missing:
            raise ValueError(
                "Malformed PlanReview: missing required field(s): " + ", ".join(missing) + "."
            )
        unknown = sorted(data.keys() - required - optional)
        if unknown:
            raise ValueError(
                "Malformed PlanReview: unknown field(s): " + ", ".join(unknown) + "."
            )
        try:
            return cls(**data)
        except TypeError as exc:
            raise ValueError(f"Malformed PlanReview payload: {exc}") from exc

    @classmethod
    def from_json(cls, payload: str) -> "PlanReview":
        try:
            data = json.loads(payload)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Malformed PlanReview JSON: {exc}") from exc
        return cls.from_dict(data)


def is_execution_approved(plan: ResearchPlan, review: PlanReview) -> bool:
    """Return whether this exact valid plan revision has explicit approval.

    This is an authorization check only. It performs no execution and makes no
    statement about the scientific quality or truth of the plan.
    """
    if not isinstance(plan, ResearchPlan) or not isinstance(review, PlanReview):
        return False
    try:
        review.validate_against(plan)
    except (AttributeError, TypeError, ValueError):
        return False
    return review.status is ReviewStatus.APPROVED


def latest_review(
    reviews: Iterable[PlanReview],
    *,
    plan_id: str,
    plan_revision: Optional[int] = None,
) -> Optional[PlanReview]:
    """Return the latest matching review by timestamp, then review ID.

    The review records remain unchanged; callers retain the full history.
    Review IDs provide a deterministic tie-break when timestamps are equal.
    """
    if not isinstance(plan_id, str) or not plan_id.strip():
        raise ValueError("latest_review requires a non-empty plan_id.")
    if plan_revision is not None and (
        not isinstance(plan_revision, int)
        or isinstance(plan_revision, bool)
        or plan_revision < 1
    ):
        raise ValueError("plan_revision must be a positive integer when supplied.")
    matches = []
    seen_review_ids: set[str] = set()
    for review in reviews:
        if not isinstance(review, PlanReview):
            raise ValueError("Review history may contain only PlanReview records.")
        if review.review_id in seen_review_ids:
            raise ValueError(f"Duplicate review_id {review.review_id!r} in review history.")
        seen_review_ids.add(review.review_id)
        if review.plan_id != plan_id:
            continue
        if plan_revision is not None and review.plan_revision != plan_revision:
            continue
        parsed = _timestamp(review.reviewed_at).astimezone(timezone.utc)
        matches.append((parsed, review.review_id, review))
    if not matches:
        return None
    return max(matches, key=lambda item: (item[0], item[1]))[2]


__all__ = [
    "ReviewStatus",
    "PlanReview",
    "is_execution_approved",
    "latest_review",
]
