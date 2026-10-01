from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research_planning import ResearchObjective, ResearchPlan, ResearchRequest
from research_plan_review import PlanReview, ReviewStatus
from research_run_authorization import (
    ResearchRunAuthorizationError,
    ResearchRunRegistrationError,
    authorize_research_run,
)
from research_state import ResearchRun, ResearchState


def make_plan(*, domain: str = "general", revision: int = 1) -> ResearchPlan:
    return ResearchPlan(
        request=ResearchRequest(
            original_text="Investigate a research question.",
            research_objective="Understand what previous research reports.",
            domain=domain,
        ),
        objective=ResearchObjective("Investigate whether approach X affects outcome Y."),
        revision=revision,
    )


def make_review(
    plan: ResearchPlan,
    status: ReviewStatus = ReviewStatus.APPROVED,
    *,
    plan_id: str | None = None,
    plan_revision: int | None = None,
    review_id: str = "review-approved",
) -> PlanReview:
    return PlanReview(
        plan_id=plan.plan_id if plan_id is None else plan_id,
        plan_revision=plan.revision if plan_revision is None else plan_revision,
        status=status,
        reviewer=None if status is ReviewStatus.PENDING else "researcher",
        review_id=review_id,
        reviewed_at="2026-09-27T10:00:00+00:00",
        requested_changes=("Revise the scope.",) if status is ReviewStatus.CHANGES_REQUESTED else (),
    )


class ResearchRunAuthorizationTests(unittest.TestCase):
    def test_matching_approved_plan_authorizes_new_run(self) -> None:
        plan = make_plan(revision=1)
        review = make_review(plan)
        run = authorize_research_run(plan, review)
        self.assertEqual(run.status, "created")
        self.assertEqual(run.plan_id, plan.plan_id)
        self.assertEqual(run.plan_revision, 1)
        self.assertEqual(run.approval_review_id, review.review_id)

    def test_plan_id_mismatch_is_rejected(self) -> None:
        plan = make_plan()
        review = make_review(plan, plan_id="another-plan")
        state = ResearchState(user_request="No mutation on failure")
        before_updated_at = state.updated_at
        with self.assertRaisesRegex(ResearchRunAuthorizationError, "Review refers to plan ID"):
            authorize_research_run(plan, review, state)
        self.assertEqual(state.research_runs, [])
        self.assertEqual(state.updated_at, before_updated_at)

    def test_plan_revision_mismatch_is_rejected(self) -> None:
        plan = make_plan(revision=2)
        review = make_review(plan, plan_revision=1)
        state = ResearchState(user_request="No mutation on failure")
        with self.assertRaisesRegex(ResearchRunAuthorizationError, "does not match plan revision"):
            authorize_research_run(plan, review, state)
        self.assertEqual(state.research_runs, [])

    def test_pending_changes_requested_and_rejected_reviews_cannot_authorize(self) -> None:
        plan = make_plan()
        reviews = (
            make_review(plan, ReviewStatus.PENDING),
            make_review(plan, ReviewStatus.CHANGES_REQUESTED, review_id="review-changes"),
            make_review(plan, ReviewStatus.REJECTED, review_id="review-rejected"),
        )
        state = ResearchState(user_request="No non-approved runs")
        for review in reviews:
            with self.subTest(status=review.status):
                with self.assertRaisesRegex(ResearchRunAuthorizationError, "does not approve"):
                    authorize_research_run(plan, review, state)
                self.assertEqual(state.research_runs, [])

    def test_explicitly_supplied_review_is_used_without_substitution(self) -> None:
        plan = make_plan()
        approved = make_review(plan, review_id="review-approved")
        pending = make_review(plan, ReviewStatus.PENDING, review_id="review-pending")
        # Having an approval elsewhere does not change the supplied pending decision.
        history = [approved, pending]
        state = ResearchState(user_request="Explicit review choice")
        with self.assertRaisesRegex(ResearchRunAuthorizationError, "does not approve"):
            authorize_research_run(plan, history[1], state)
        self.assertEqual(state.research_runs, [])

    def test_wrong_argument_types_and_malformed_plan_are_rejected(self) -> None:
        plan = make_plan()
        review = make_review(plan)
        with self.assertRaisesRegex(ResearchRunAuthorizationError, "ResearchPlan is required"):
            authorize_research_run(object(), review)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ResearchRunAuthorizationError, "PlanReview is required"):
            authorize_research_run(plan, object())  # type: ignore[arg-type]

        state = ResearchState(user_request="Invalid plan")
        object.__setattr__(plan, "revision", 0)
        with self.assertRaisesRegex(ResearchRunAuthorizationError, "invalid ResearchPlan"):
            authorize_research_run(plan, review, state)
        self.assertEqual(state.research_runs, [])

    def test_malformed_approved_review_is_rejected_before_run_creation(self) -> None:
        plan = make_plan()
        review = make_review(plan)
        # Defensive check for a corrupted object that bypassed frozen fields.
        object.__setattr__(review, "reviewer", None)
        state = ResearchState(user_request="Invalid review")
        with self.assertRaisesRegex(ResearchRunAuthorizationError, "APPROVED review requires a reviewer"):
            authorize_research_run(plan, review, state)
        self.assertEqual(state.research_runs, [])

    def test_malformed_requested_changes_are_rejected(self) -> None:
        plan = make_plan()
        review = make_review(plan)
        object.__setattr__(review, "requested_changes", (" ",))
        with self.assertRaisesRegex(ResearchRunAuthorizationError, r"requested_changes\[0\]"):
            authorize_research_run(plan, review)

    def test_successful_state_integration_records_exact_authorization(self) -> None:
        plan = make_plan(revision=3)
        review = make_review(plan, review_id="review-human-42")
        state = ResearchState(user_request="Authorized research")
        run = authorize_research_run(
            plan,
            review,
            state,
            run_metadata={"reason": "researcher approved", "batch": "pilot-1"},
        )
        self.assertEqual(state.research_runs, [run])
        self.assertIs(state.research_runs[0], run)
        self.assertEqual(run.plan_id, plan.plan_id)
        self.assertEqual(run.plan_revision, 3)
        self.assertEqual(run.approval_review_id, "review-human-42")
        self.assertEqual(run.metadata, {"reason": "researcher approved", "batch": "pilot-1"})

    def test_authorized_run_round_trip_preserves_plan_and_review_links(self) -> None:
        plan = make_plan(revision=4)
        review = make_review(plan, review_id="review-round-trip")
        state = ResearchState(user_request="Round trip")
        run = authorize_research_run(plan, review, state)
        restored = ResearchState.from_dict(state.to_dict())
        self.assertEqual(restored.research_runs, [run])
        restored_run = restored.research_runs[0]
        self.assertEqual(restored_run.plan_id, plan.plan_id)
        self.assertEqual(restored_run.plan_revision, 4)
        self.assertEqual(restored_run.approval_review_id, review.review_id)

    def test_legacy_research_run_positional_constructor_and_payload_remain_valid(self) -> None:
        legacy = ResearchRun("run-legacy", "2026-01-01T00:00:00+00:00", "created", {"old": True})
        self.assertEqual(legacy.metadata, {"old": True})
        self.assertIsNone(legacy.plan_id)
        self.assertIsNone(legacy.plan_revision)
        self.assertIsNone(legacy.approval_review_id)

        state = ResearchState(user_request="Legacy")
        state.add_run(legacy)
        payload = state.to_dict()
        for item in payload["research_runs"]:
            item.pop("plan_id")
            item.pop("plan_revision")
            item.pop("approval_review_id")
        restored = ResearchState.from_dict(payload)
        self.assertEqual(restored.research_runs[0], legacy)

    def test_research_run_requires_complete_typed_authorization_linkage(self) -> None:
        with self.assertRaisesRegex(ValueError, "provide plan_id, plan_revision"):
            ResearchRun(plan_id="plan-1")
        with self.assertRaisesRegex(ValueError, "positive integer"):
            ResearchRun(plan_id="plan-1", plan_revision=True, approval_review_id="review-1")
        with self.assertRaisesRegex(ValueError, "non-empty string"):
            ResearchRun(plan_id=" ", plan_revision=1, approval_review_id="review-1")

    def test_same_approved_review_can_authorize_two_distinct_runs(self) -> None:
        plan = make_plan()
        review = make_review(plan)
        first = authorize_research_run(plan, review)
        second = authorize_research_run(plan, review)
        self.assertNotEqual(first.run_id, second.run_id)
        self.assertEqual(first.approval_review_id, second.approval_review_id)

    def test_state_registration_failure_is_explicit_and_rolls_back_partial_append(self) -> None:
        class PartiallyFailingState(ResearchState):
            def add_run(self, run=None):
                super().add_run(run)
                raise RuntimeError("simulated persistence failure")

        plan = make_plan()
        review = make_review(plan)
        state = PartiallyFailingState(user_request="Registration failure")
        before_updated_at = state.updated_at
        with self.assertRaisesRegex(ResearchRunRegistrationError, "registration failed"):
            authorize_research_run(plan, review, state)
        self.assertEqual(state.research_runs, [])
        self.assertEqual(state.updated_at, before_updated_at)

    def test_state_registration_error_before_mutation_is_not_swallowed(self) -> None:
        class FailingState(ResearchState):
            def add_run(self, run=None):
                raise RuntimeError("registration unavailable")

        plan = make_plan()
        state = FailingState(user_request="Registration failure")
        with self.assertRaisesRegex(ResearchRunRegistrationError, "registration unavailable"):
            authorize_research_run(plan, make_review(plan), state)
        self.assertEqual(state.research_runs, [])

    def test_old_revision_approval_needs_new_review_for_new_revision(self) -> None:
        plan_v1 = make_plan(revision=1)
        approval_v1 = make_review(plan_v1, review_id="approval-v1")
        plan_v2 = ResearchPlan(
            request=plan_v1.request,
            objective=plan_v1.objective,
            plan_id=plan_v1.plan_id,
            revision=2,
        )
        with self.assertRaisesRegex(ResearchRunAuthorizationError, "does not match plan revision"):
            authorize_research_run(plan_v2, approval_v1)
        approval_v2 = make_review(plan_v2, review_id="approval-v2")
        run = authorize_research_run(plan_v2, approval_v2)
        self.assertEqual(run.plan_revision, 2)
        self.assertEqual(run.approval_review_id, "approval-v2")

    def test_authorization_is_domain_neutral(self) -> None:
        for domain in ("NLP", "computer vision", "cybersecurity", "agriculture", "medicine"):
            with self.subTest(domain=domain):
                plan = make_plan(domain=domain)
                run = authorize_research_run(plan, make_review(plan))
                self.assertEqual(run.plan_id, plan.plan_id)
                self.assertEqual(run.plan_revision, plan.revision)

    def test_authorization_records_no_scientific_quality_or_result(self) -> None:
        plan = make_plan()
        run = authorize_research_run(plan, make_review(plan))
        self.assertEqual(run.status, "created")
        self.assertFalse(hasattr(run, "scientific_validity"))
        self.assertFalse(hasattr(run, "novelty_score"))
        self.assertFalse(hasattr(run, "expected_result"))


if __name__ == "__main__":
    unittest.main()
