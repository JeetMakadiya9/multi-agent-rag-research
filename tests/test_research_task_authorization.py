from __future__ import annotations

import sys
import unittest
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research_plan_review import PlanReview, ReviewStatus
from research_planning import ResearchObjective, ResearchPlan, ResearchPlanQuestion, ResearchRequest, ResearchTask, TaskStatus
from research_run_authorization import authorize_research_run
from research_state import ResearchRun, ResearchState
from research_task_authorization import ResearchTaskAuthorizationError, authorize_research_tasks


def make_plan(*, plan_id: str = "plan-task-auth", revision: int = 1, domain: str = "general", task_prefix: str = "task") -> ResearchPlan:
    question = ResearchPlanQuestion(
        question_id=f"question-{plan_id}",
        text="What does the literature report?",
    )
    tasks = [
        ResearchTask("Search source material.", question.question_id, task_id=f"{task_prefix}-1", status=TaskStatus.PENDING),
        ResearchTask("Compare reported methods.", question.question_id, task_id=f"{task_prefix}-2", status=TaskStatus.READY),
        ResearchTask("Summarize limitations.", question.question_id, task_id=f"{task_prefix}-3", dependency_ids=[f"{task_prefix}-1", f"{task_prefix}-2"]),
    ]
    return ResearchPlan(
        plan_id=plan_id,
        revision=revision,
        request=ResearchRequest("Investigate a topic.", "Understand reported work.", domain=domain),
        objective=ResearchObjective("Describe what has been reported."),
        questions=[question],
        tasks=tasks,
    )


def make_review(plan: ResearchPlan, *, review_id: str = "review-task-auth", status: ReviewStatus = ReviewStatus.APPROVED) -> PlanReview:
    return PlanReview(
        plan_id=plan.plan_id,
        plan_revision=plan.revision,
        status=status,
        reviewer=None if status is ReviewStatus.PENDING else "researcher",
        requested_changes=("Narrow the scope.",) if status is ReviewStatus.CHANGES_REQUESTED else (),
        review_id=review_id,
        reviewed_at="2026-09-27T10:00:00+00:00",
    )


def make_run(plan: ResearchPlan, review: PlanReview, state: ResearchState | None = None) -> ResearchRun:
    return authorize_research_run(plan, review, state)


class ResearchTaskAuthorizationTests(unittest.TestCase):
    def test_single_and_multiple_task_authorization_preserves_exact_order(self) -> None:
        plan = make_plan()
        review = make_review(plan)
        single = make_run(plan, review)
        multi = make_run(plan, review)
        self.assertEqual(authorize_research_tasks(plan, single, ["task-2"], review).authorized_task_ids, ["task-2"])
        self.assertEqual(authorize_research_tasks(plan, multi, ["task-3", "task-1", "task-2"], review).authorized_task_ids,
                         ["task-3", "task-1", "task-2"])

    def test_run_plan_id_and_revision_must_match_exact_plan(self) -> None:
        plan = make_plan()
        review = make_review(plan)
        run = make_run(plan, review)
        different_plan = make_plan(plan_id="another-plan")
        with self.assertRaisesRegex(ResearchTaskAuthorizationError, "plan_id"):
            authorize_research_tasks(different_plan, run, ["task-1"], make_review(different_plan))

        changed_revision = make_plan(plan_id=plan.plan_id, revision=2)
        with self.assertRaisesRegex(ResearchTaskAuthorizationError, "revision"):
            authorize_research_tasks(changed_revision, run, ["task-1"], review)
        self.assertIsNone(run.authorized_task_ids)

    def test_legacy_run_without_approval_lineage_cannot_authorize_tasks(self) -> None:
        plan = make_plan()
        with self.assertRaisesRegex(ResearchTaskAuthorizationError, "no complete Phase 4C"):
            authorize_research_tasks(plan, ResearchRun(), ["task-1"], make_review(plan))

    def test_supplied_review_must_match_run_reference_and_be_approved(self) -> None:
        plan = make_plan()
        approved = make_review(plan, review_id="approved-review")
        run = make_run(plan, approved)
        wrong_review = make_review(plan, review_id="other-review")
        with self.assertRaisesRegex(ResearchTaskAuthorizationError, "approval_review_id"):
            authorize_research_tasks(plan, run, ["task-1"], wrong_review)

        for status in (ReviewStatus.PENDING, ReviewStatus.CHANGES_REQUESTED, ReviewStatus.REJECTED):
            with self.subTest(status=status):
                review = make_review(plan, review_id=f"{status.value}-review", status=status)
                mismatched_run = ResearchRun(
                    plan_id=plan.plan_id,
                    plan_revision=plan.revision,
                    approval_review_id=review.review_id,
                )
                with self.assertRaisesRegex(ResearchTaskAuthorizationError, "does not approve"):
                    authorize_research_tasks(plan, mismatched_run, ["task-1"], review)

    def test_review_plan_and_revision_are_revalidated(self) -> None:
        plan = make_plan()
        review = make_review(plan)
        run = make_run(plan, review)
        wrong_id = PlanReview(
            plan_id="other-plan", plan_revision=1, status=ReviewStatus.APPROVED,
            reviewer="researcher", review_id=review.review_id,
            reviewed_at=review.reviewed_at,
        )
        with self.assertRaisesRegex(ResearchTaskAuthorizationError, "PlanReview validation failed"):
            authorize_research_tasks(plan, run, ["task-1"], wrong_id)
        wrong_revision = PlanReview(
            plan_id=plan.plan_id, plan_revision=2, status=ReviewStatus.APPROVED,
            reviewer="researcher", review_id=review.review_id,
            reviewed_at=review.reviewed_at,
        )
        with self.assertRaisesRegex(ResearchTaskAuthorizationError, "PlanReview validation failed"):
            authorize_research_tasks(plan, run, ["task-1"], wrong_revision)

    def test_structurally_invalid_plan_is_rejected(self) -> None:
        plan = make_plan()
        review = make_review(plan)
        run = make_run(plan, review)
        plan.tasks[2].dependency_ids = ["missing-task"]
        with self.assertRaisesRegex(ResearchTaskAuthorizationError, "invalid ResearchPlan"):
            authorize_research_tasks(plan, run, ["task-1"], review)
        self.assertIsNone(run.authorized_task_ids)

    def test_unknown_task_ids_are_rejected_without_partial_authorization(self) -> None:
        plan = make_plan()
        review = make_review(plan)
        run = make_run(plan, review)
        for ids in (["task-unknown"], ["task-1", "task-unknown"]):
            with self.subTest(ids=ids):
                with self.assertRaisesRegex(ResearchTaskAuthorizationError, "not present"):
                    authorize_research_tasks(plan, run, ids, review)
                self.assertIsNone(run.authorized_task_ids)

    def test_duplicate_and_empty_task_ids_are_rejected(self) -> None:
        plan = make_plan()
        review = make_review(plan)
        for ids, message in ((["task-1", "task-1"], "Duplicate"), ([], "At least one")):
            run = make_run(plan, review)
            with self.subTest(ids=ids):
                with self.assertRaisesRegex(ResearchTaskAuthorizationError, message):
                    authorize_research_tasks(plan, run, ids, review)
                self.assertIsNone(run.authorized_task_ids)

    def test_invalid_task_id_shapes_are_rejected(self) -> None:
        plan = make_plan()
        review = make_review(plan)
        for ids in ("task-1", [" "], [None]):
            run = make_run(plan, review)
            with self.subTest(ids=ids):
                with self.assertRaises(ResearchTaskAuthorizationError):
                    authorize_research_tasks(plan, run, ids, review)  # type: ignore[arg-type]
                self.assertIsNone(run.authorized_task_ids)

    def test_old_revision_approval_cannot_authorize_new_revision_task(self) -> None:
        old_plan = make_plan(revision=1)
        old_review = make_review(old_plan, review_id="approval-v1")
        old_run = make_run(old_plan, old_review)
        new_plan = make_plan(plan_id=old_plan.plan_id, revision=2)
        new_plan.tasks.append(ResearchTask("New revision task.", new_plan.questions[0].question_id, task_id="task-4"))
        new_run = ResearchRun(plan_id=new_plan.plan_id, plan_revision=2, approval_review_id=old_review.review_id)
        with self.assertRaisesRegex(ResearchTaskAuthorizationError, "does not match supplied plan revision"):
            authorize_research_tasks(new_plan, old_run, ["task-4"], old_review)

        new_review = make_review(new_plan, review_id="approval-v2")
        authorized_new_run = make_run(new_plan, new_review)
        self.assertEqual(authorize_research_tasks(new_plan, authorized_new_run, ["task-4"], new_review).authorized_task_ids,
                         ["task-4"])

    def test_tasks_from_another_plan_are_not_members_of_this_plan(self) -> None:
        first = make_plan(plan_id="plan-one")
        second = make_plan(plan_id="plan-two", task_prefix="other-task")
        review = make_review(first)
        run = make_run(first, review)
        other_task_id = second.tasks[0].task_id
        with self.assertRaisesRegex(ResearchTaskAuthorizationError, "not present"):
            authorize_research_tasks(first, run, [other_task_id], review)

    def test_dependency_is_not_expanded_and_task_status_is_unchanged(self) -> None:
        plan = make_plan()
        review = make_review(plan)
        run = make_run(plan, review)
        statuses = [task.status for task in plan.tasks]
        authorize_research_tasks(plan, run, ["task-3"], review)
        self.assertEqual(run.authorized_task_ids, ["task-3"])
        self.assertEqual([task.status for task in plan.tasks], statuses)
        self.assertEqual(plan.tasks[2].dependency_ids, ["task-1", "task-2"])
        self.assertEqual(run.status, "created")

    def test_research_state_round_trip_preserves_authorized_ids_and_approval(self) -> None:
        plan = make_plan()
        review = make_review(plan)
        state = ResearchState(user_request="Task authorization")
        run = make_run(plan, review, state)
        authorize_research_tasks(plan, run, ["task-2", "task-1"], review, state)
        self.assertEqual(state.research_runs[0].authorized_task_ids, ["task-2", "task-1"])
        restored = ResearchState.from_dict(state.to_dict())
        restored_run = restored.research_runs[0]
        self.assertEqual(restored_run.authorized_task_ids, ["task-2", "task-1"])
        self.assertEqual(restored_run.plan_id, plan.plan_id)
        self.assertEqual(restored_run.plan_revision, plan.revision)
        self.assertEqual(restored_run.approval_review_id, review.review_id)

    def test_json_compatible_state_payload_round_trip_preserves_authorized_ids(self) -> None:
        plan = make_plan()
        review = make_review(plan)
        state = ResearchState(user_request="JSON round trip")
        run = make_run(plan, review, state)
        authorize_research_tasks(plan, run, ["task-3", "task-1"], review, state)
        json_payload = json.loads(json.dumps(state.to_dict()))
        restored = ResearchState.from_dict(json_payload)
        self.assertEqual(restored.research_runs[0].authorized_task_ids, ["task-3", "task-1"])

    def test_old_run_payload_loads_without_authorized_task_ids(self) -> None:
        state = ResearchState(user_request="Old state")
        state.add_run(ResearchRun(run_id="legacy-run"))
        payload = state.to_dict()
        payload["research_runs"][0].pop("authorized_task_ids")
        restored = ResearchState.from_dict(payload)
        self.assertIsNone(restored.research_runs[0].authorized_task_ids)

    def test_failed_authorization_does_not_mutate_state(self) -> None:
        plan = make_plan()
        review = make_review(plan)
        state = ResearchState(user_request="Failure is atomic")
        run = make_run(plan, review, state)
        before = state.to_dict()
        with self.assertRaises(ResearchTaskAuthorizationError):
            authorize_research_tasks(plan, run, ["missing"], review, state)
        self.assertEqual(state.to_dict(), before)
        self.assertIsNone(run.authorized_task_ids)

    def test_unregistered_run_is_rejected_when_state_is_supplied(self) -> None:
        plan = make_plan()
        review = make_review(plan)
        state = ResearchState(user_request="Must use state record")
        run = make_run(plan, review)
        with self.assertRaisesRegex(ResearchTaskAuthorizationError, "must be registered"):
            authorize_research_tasks(plan, run, ["task-1"], review, state)
        self.assertEqual(state.research_runs, [])

    def test_repeated_authorization_is_rejected_without_overwriting_first_set(self) -> None:
        plan = make_plan()
        review = make_review(plan)
        run = make_run(plan, review)
        authorize_research_tasks(plan, run, ["task-1"], review)
        with self.assertRaisesRegex(ResearchTaskAuthorizationError, "already has"):
            authorize_research_tasks(plan, run, ["task-2"], review)
        self.assertEqual(run.authorized_task_ids, ["task-1"])

    def test_distinct_authorized_runs_can_share_the_same_approval(self) -> None:
        plan = make_plan()
        review = make_review(plan)
        first, second = make_run(plan, review), make_run(plan, review)
        authorize_research_tasks(plan, first, ["task-1"], review)
        authorize_research_tasks(plan, second, ["task-2"], review)
        self.assertNotEqual(first.run_id, second.run_id)
        self.assertEqual(first.approval_review_id, second.approval_review_id)

    def test_domain_neutral_plans_use_the_same_authorization_contract(self) -> None:
        for domain in ("NLP", "computer vision", "cybersecurity", "agriculture", "medicine", "physics"):
            with self.subTest(domain=domain):
                plan = make_plan(domain=domain)
                review = make_review(plan)
                run = make_run(plan, review)
                self.assertEqual(authorize_research_tasks(plan, run, ["task-1"], review).authorized_task_ids,
                                 ["task-1"])

    def test_authorized_means_selected_not_executed_or_scientifically_validated(self) -> None:
        plan = make_plan()
        review = make_review(plan)
        run = make_run(plan, review)
        authorize_research_tasks(plan, run, ["task-1"], review)
        self.assertEqual(run.status, "created")
        self.assertEqual(plan.tasks[0].status, TaskStatus.PENDING)
        self.assertFalse(hasattr(run, "scientific_validity"))
        self.assertFalse(hasattr(run, "novelty_score"))
        self.assertFalse(hasattr(run, "execution_result"))

    def test_mutated_malformed_review_is_rejected(self) -> None:
        plan = make_plan()
        review = make_review(plan)
        run = make_run(plan, review)
        object.__setattr__(review, "reviewer", " ")
        with self.assertRaisesRegex(ResearchTaskAuthorizationError, "PlanReview validation failed"):
            authorize_research_tasks(plan, run, ["task-1"], review)

    def test_state_recording_failure_rolls_back_authorized_ids_and_timestamp(self) -> None:
        class FailingTouchState(ResearchState):
            def touch(self) -> None:
                self.updated_at = "changed-before-error"
                raise RuntimeError("touch failed")

        plan = make_plan()
        review = make_review(plan)
        state = ResearchState(user_request="Rollback")
        run = make_run(plan, review, state)
        before_updated_at = state.updated_at
        failing_state = FailingTouchState.from_dict(state.to_dict())
        # Reuse the exact run object because state membership is identity-based.
        failing_state.research_runs[0] = run
        with self.assertRaisesRegex(RuntimeError, "recording failed"):
            authorize_research_tasks(plan, run, ["task-1"], review, failing_state)
        self.assertIsNone(run.authorized_task_ids)
        self.assertEqual(failing_state.updated_at, before_updated_at)


if __name__ == "__main__":
    unittest.main()
