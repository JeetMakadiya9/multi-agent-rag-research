from __future__ import annotations

from dataclasses import replace
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research_planning import ResearchObjective, ResearchPlan, ResearchRequest
from research_plan_review import (
    PlanReview,
    ReviewStatus,
    is_execution_approved,
    latest_review,
)


def make_plan(*, domain: str = "general", revision: int = 1) -> ResearchPlan:
    return ResearchPlan(
        request=ResearchRequest(
            original_text="Investigate a research question.",
            research_objective="Understand what prior work reports.",
            domain=domain,
        ),
        objective=ResearchObjective(
            "Investigate whether method X improves outcome Y."
        ),
        revision=revision,
    )


def review_for(
    plan: ResearchPlan,
    status: ReviewStatus,
    **overrides,
) -> PlanReview:
    defaults = {
        "review_id": f"review-{status.value.lower()}-{plan.revision}",
        "plan_id": plan.plan_id,
        "plan_revision": plan.revision,
        "status": status,
        "reviewer": None if status is ReviewStatus.PENDING else "researcher",
        "reviewed_at": "2026-09-27T12:00:00+00:00",
        "comments": "",
        "requested_changes": (),
    }
    defaults.update(overrides)
    return PlanReview(**defaults)


class ResearchPlanReviewTests(unittest.TestCase):
    def test_pending_review_can_be_created_without_reviewer(self) -> None:
        plan = make_plan()
        review = PlanReview(
            plan_id=plan.plan_id,
            plan_revision=plan.revision,
            status=ReviewStatus.PENDING,
            reviewer=None,
        )
        self.assertEqual(review.status, ReviewStatus.PENDING)
        self.assertIsNone(review.reviewer)
        self.assertTrue(review.review_id.startswith("review_"))

    def test_each_review_status_is_representable(self) -> None:
        plan = make_plan()
        approved = review_for(plan, ReviewStatus.APPROVED)
        changes = review_for(
            plan, ReviewStatus.CHANGES_REQUESTED,
            requested_changes=["Narrow the research question."],
        )
        rejected = review_for(plan, ReviewStatus.REJECTED)
        self.assertEqual(approved.status, ReviewStatus.APPROVED)
        self.assertEqual(changes.requested_changes, ("Narrow the research question.",))
        self.assertEqual(rejected.status, ReviewStatus.REJECTED)

    def test_review_id_and_plan_id_are_required(self) -> None:
        plan = make_plan()
        with self.assertRaisesRegex(ValueError, "plan_id is required"):
            PlanReview(" ", plan.revision)
        with self.assertRaisesRegex(ValueError, "review_id is required"):
            PlanReview(plan.plan_id, plan.revision, review_id=" ")

    def test_plan_revision_must_be_positive_integer(self) -> None:
        plan = make_plan()
        with self.assertRaisesRegex(ValueError, "positive integer"):
            PlanReview(plan.plan_id, 0)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            PlanReview(plan.plan_id, True)  # type: ignore[arg-type]

    def test_approval_and_final_decisions_require_reviewer(self) -> None:
        plan = make_plan()
        for status in (
            ReviewStatus.APPROVED,
            ReviewStatus.CHANGES_REQUESTED,
            ReviewStatus.REJECTED,
        ):
            with self.subTest(status=status):
                with self.assertRaisesRegex(ValueError, "requires a reviewer"):
                    review_for(plan, status, reviewer=None)

    def test_pending_review_may_have_no_reviewer(self) -> None:
        pending = review_for(make_plan(), ReviewStatus.PENDING, reviewer=None)
        self.assertIsNone(pending.reviewer)

    def test_review_timestamp_must_be_present_and_timezone_aware(self) -> None:
        plan = make_plan()
        with self.assertRaisesRegex(ValueError, "non-empty ISO 8601"):
            review_for(plan, ReviewStatus.PENDING, reviewed_at="")
        with self.assertRaisesRegex(ValueError, "include a timezone"):
            review_for(plan, ReviewStatus.PENDING, reviewed_at="2026-09-27T12:00:00")
        with self.assertRaisesRegex(ValueError, "valid ISO 8601"):
            review_for(plan, ReviewStatus.PENDING, reviewed_at="yesterday")

    def test_changes_requested_requires_at_least_one_meaningful_change(self) -> None:
        plan = make_plan()
        with self.assertRaisesRegex(ValueError, "at least one requested change"):
            review_for(plan, ReviewStatus.CHANGES_REQUESTED)
        with self.assertRaisesRegex(ValueError, "must be a non-empty string"):
            review_for(plan, ReviewStatus.CHANGES_REQUESTED, requested_changes=["  "])

    def test_comments_and_requested_changes_are_preserved_as_human_feedback(self) -> None:
        plan = make_plan()
        review = review_for(
            plan,
            ReviewStatus.CHANGES_REQUESTED,
            comments="Please keep the scope manageable.",
            requested_changes=["Narrow question 2", "Add a requirement for study context"],
        )
        self.assertEqual(review.comments, "Please keep the scope manageable.")
        self.assertEqual(
            review.requested_changes,
            ("Narrow question 2", "Add a requirement for study context"),
        )

    def test_review_validates_against_matching_plan_and_revision(self) -> None:
        plan = make_plan()
        review_for(plan, ReviewStatus.PENDING).validate_against(plan)

    def test_review_rejects_wrong_plan_id(self) -> None:
        plan = make_plan()
        other = make_plan()
        review = review_for(plan, ReviewStatus.APPROVED)
        with self.assertRaisesRegex(ValueError, "Review refers to plan ID"):
            review.validate_against(other)

    def test_review_rejects_wrong_revision(self) -> None:
        original = make_plan(revision=1)
        updated = replace(original, revision=2)
        review = review_for(original, ReviewStatus.APPROVED)
        with self.assertRaisesRegex(ValueError, "Review revision 1 does not match plan revision 2"):
            review.validate_against(updated)

    def test_review_rejects_structurally_invalid_plan(self) -> None:
        plan = make_plan()
        review = review_for(plan, ReviewStatus.PENDING)
        object.__setattr__(plan, "revision", 0)
        with self.assertRaisesRegex(ValueError, "structurally invalid ResearchPlan"):
            review.validate_against(plan)

    def test_approved_exact_revision_is_execution_eligible(self) -> None:
        plan = make_plan()
        review = review_for(plan, ReviewStatus.APPROVED)
        self.assertTrue(is_execution_approved(plan, review))

    def test_non_approved_statuses_are_not_execution_eligible(self) -> None:
        plan = make_plan()
        reviews = [
            review_for(plan, ReviewStatus.PENDING, reviewer=None),
            review_for(
                plan, ReviewStatus.CHANGES_REQUESTED,
                requested_changes=["Revise the scope."],
            ),
            review_for(plan, ReviewStatus.REJECTED),
        ]
        for review in reviews:
            with self.subTest(status=review.status):
                self.assertFalse(is_execution_approved(plan, review))

    def test_mismatched_or_malformed_inputs_are_not_execution_eligible(self) -> None:
        plan = make_plan()
        approved = review_for(plan, ReviewStatus.APPROVED)
        self.assertFalse(is_execution_approved(make_plan(), approved))
        self.assertFalse(is_execution_approved(replace(plan, revision=2), approved))
        self.assertFalse(is_execution_approved(plan, object()))  # type: ignore[arg-type]

    def test_old_revision_approval_does_not_approve_new_revision(self) -> None:
        revision_one = make_plan(revision=1)
        revision_two = replace(revision_one, revision=2)
        approval = review_for(revision_one, ReviewStatus.APPROVED)
        self.assertTrue(is_execution_approved(revision_one, approval))
        self.assertFalse(is_execution_approved(revision_two, approval))

    def test_serialization_preserves_every_review_field(self) -> None:
        plan = make_plan()
        review = review_for(
            plan,
            ReviewStatus.CHANGES_REQUESTED,
            reviewer="researcher-7",
            comments="Please narrow the scope.",
            requested_changes=["Narrow question 2"],
        )
        data = review.to_dict()
        self.assertEqual(data["review_id"], review.review_id)
        self.assertEqual(data["plan_id"], plan.plan_id)
        self.assertEqual(data["plan_revision"], plan.revision)
        self.assertEqual(data["status"], "CHANGES_REQUESTED")
        self.assertEqual(data["reviewer"], "researcher-7")
        self.assertEqual(data["reviewed_at"], review.reviewed_at)
        self.assertEqual(data["comments"], "Please narrow the scope.")
        self.assertEqual(data["requested_changes"], ["Narrow question 2"])
        self.assertEqual(PlanReview.from_dict(data), review)

    def test_json_round_trip_preserves_review(self) -> None:
        plan = make_plan()
        review = review_for(plan, ReviewStatus.APPROVED, comments="Approved for future execution.")
        self.assertEqual(PlanReview.from_json(review.to_json()), review)

    def test_malformed_review_payloads_fail_clearly(self) -> None:
        plan = make_plan()
        payload = review_for(plan, ReviewStatus.PENDING).to_dict()
        missing = dict(payload)
        del missing["plan_revision"]
        with self.assertRaisesRegex(ValueError, "missing required field.*plan_revision"):
            PlanReview.from_dict(missing)
        unknown = dict(payload, scientific_score=0.9)
        with self.assertRaisesRegex(ValueError, "unknown field.*scientific_score"):
            PlanReview.from_dict(unknown)
        invalid_status = dict(payload, status="MAYBE")
        with self.assertRaisesRegex(ValueError, "Unsupported review status"):
            PlanReview.from_dict(invalid_status)
        with self.assertRaisesRegex(ValueError, "Malformed PlanReview JSON"):
            PlanReview.from_json("{")

    def test_multiple_reviews_are_preserved_and_latest_is_explicit(self) -> None:
        plan = make_plan()
        changes = review_for(
            plan,
            ReviewStatus.CHANGES_REQUESTED,
            review_id="review-1",
            reviewed_at="2026-09-27T10:00:00+00:00",
            requested_changes=["Narrow the scope."],
        )
        approved = review_for(
            plan,
            ReviewStatus.APPROVED,
            review_id="review-2",
            reviewed_at="2026-09-27T11:00:00+00:00",
        )
        history = [changes, approved]
        self.assertIs(latest_review(history, plan_id=plan.plan_id), approved)
        self.assertEqual(history, [changes, approved])
        self.assertIs(latest_review(history, plan_id=plan.plan_id, plan_revision=1), approved)
        self.assertIsNone(latest_review(history, plan_id=plan.plan_id, plan_revision=2))

    def test_latest_review_orders_timezone_offsets_and_ties_deterministically(self) -> None:
        plan = make_plan()
        earlier = review_for(
            plan,
            ReviewStatus.PENDING,
            reviewer=None,
            review_id="review-z",
            reviewed_at="2026-09-27T12:30:00+01:00",
        )
        later = review_for(
            plan,
            ReviewStatus.PENDING,
            reviewer=None,
            review_id="review-a",
            reviewed_at="2026-09-27T12:00:00+00:00",
        )
        tied = review_for(
            plan,
            ReviewStatus.PENDING,
            reviewer=None,
            review_id="review-b",
            reviewed_at="2026-09-27T12:00:00+00:00",
        )
        self.assertIs(latest_review([later, earlier], plan_id=plan.plan_id), later)
        self.assertIs(latest_review([later, tied], plan_id=plan.plan_id), tied)

    def test_history_rejects_invalid_filter_or_non_review_records(self) -> None:
        plan = make_plan()
        with self.assertRaisesRegex(ValueError, "non-empty plan_id"):
            latest_review([], plan_id=" ")
        with self.assertRaisesRegex(ValueError, "positive integer"):
            latest_review([], plan_id=plan.plan_id, plan_revision=0)
        with self.assertRaisesRegex(ValueError, "only PlanReview records"):
            latest_review([object()], plan_id=plan.plan_id)  # type: ignore[list-item]

    def test_review_history_rejects_duplicate_review_ids(self) -> None:
        plan = make_plan()
        review = review_for(plan, ReviewStatus.PENDING, review_id="same-review")
        duplicate = review_for(
            plan,
            ReviewStatus.PENDING,
            review_id="same-review",
            reviewed_at="2026-09-27T13:00:00+00:00",
        )
        with self.assertRaisesRegex(ValueError, "Duplicate review_id"):
            latest_review([review, duplicate], plan_id=plan.plan_id)

    def test_domain_neutral_approval_examples(self) -> None:
        for domain in ("AI/ML", "medicine", "agriculture", "physics"):
            with self.subTest(domain=domain):
                plan = make_plan(domain=domain)
                review = review_for(plan, ReviewStatus.APPROVED)
                self.assertTrue(is_execution_approved(plan, review))

    def test_approval_is_authorization_not_a_scientific_verdict(self) -> None:
        plan = make_plan()
        review = review_for(plan, ReviewStatus.APPROVED)
        self.assertIn("whether", plan.objective.text)
        self.assertTrue(is_execution_approved(plan, review))
        self.assertFalse(hasattr(review, "verdict"))
        self.assertFalse(hasattr(review, "scientific_correctness"))


if __name__ == "__main__":
    unittest.main()
