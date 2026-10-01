from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research_planning import (
    EvidenceRequirement,
    Priority,
    QuestionStatus,
    ResearchObjective,
    ResearchPlan,
    ResearchPlanQuestion,
    ResearchRequest,
    ResearchTask,
    TaskStatus,
)


def make_valid_plan() -> ResearchPlan:
    question = ResearchPlanQuestion(
        question_id="q-methods",
        text="Which approaches have been investigated?",
        priority=Priority.HIGH,
    )
    requirement = EvidenceRequirement(
        requirement_id="er-approaches",
        question_id=question.question_id,
        description="Evidence describing investigated approaches.",
        expected_evidence_category="approach description",
    )
    first = ResearchTask(
        task_id="task-find",
        description="Locate relevant sources.",
        question_id=question.question_id,
        evidence_requirement_ids=[requirement.requirement_id],
    )
    second = ResearchTask(
        task_id="task-compare",
        description="Compare the approaches described by the sources.",
        question_id=question.question_id,
        category="comparison",
        status=TaskStatus.READY,
        dependency_ids=[first.task_id],
    )
    return ResearchPlan(
        plan_id="plan-example",
        request=ResearchRequest(
            request_id="request-example",
            original_text="Study approaches to a research problem.",
            research_objective="Understand prior approaches and their reported properties.",
            domain="general research",
            constraints=["Use traceable sources."],
            created_at="2026-09-27T00:00:00+00:00",
        ),
        objective=ResearchObjective(
            objective_id="objective-example",
            text="Map previously investigated approaches.",
            scope_description="Describe reported approaches without asserting which is best.",
        ),
        questions=[question],
        evidence_requirements=[requirement],
        tasks=[first, second],
        created_at="2026-09-27T00:00:00+00:00",
    )


class ResearchPlanningContractTests(unittest.TestCase):
    def test_request_preserves_original_text_objective_domain_constraints_and_time(self) -> None:
        request = ResearchRequest(
            original_text="Investigate crop disease detection.",
            research_objective="Compare reported detection approaches.",
            domain="agriculture",
            constraints=["Focus on field conditions."],
            created_at="2026-01-01T00:00:00+00:00",
        )
        self.assertTrue(request.request_id.startswith("request_"))
        self.assertEqual(request.original_text, "Investigate crop disease detection.")
        self.assertEqual(request.research_objective, "Compare reported detection approaches.")
        self.assertEqual(request.domain, "agriculture")
        self.assertEqual(request.constraints, ["Focus on field conditions."])

    def test_request_domain_is_optional(self) -> None:
        request = ResearchRequest("Research a question.", "Establish what is known.")
        self.assertIsNone(request.domain)

    def test_objective_question_requirement_task_and_plan_are_typed(self) -> None:
        plan = make_valid_plan()
        self.assertIsInstance(plan.objective, ResearchObjective)
        self.assertIsInstance(plan.questions[0], ResearchPlanQuestion)
        self.assertIsInstance(plan.evidence_requirements[0], EvidenceRequirement)
        self.assertIsInstance(plan.tasks[0], ResearchTask)
        self.assertEqual(plan.revision, 1)

    def test_question_associations_resolve_from_plan_relations(self) -> None:
        plan = make_valid_plan()
        question_id = plan.questions[0].question_id
        self.assertEqual(
            plan.requirements_for_question(question_id), plan.evidence_requirements
        )
        self.assertEqual(plan.tasks_for_question(question_id), plan.tasks)
        with self.assertRaisesRegex(ValueError, "Unknown research question ID"):
            plan.tasks_for_question("unknown")

    def test_requirement_is_distinct_from_evidence_and_can_be_optional(self) -> None:
        plan = make_valid_plan()
        requirement = EvidenceRequirement(
            question_id=plan.questions[0].question_id,
            description="Optional context about study conditions.",
            required=False,
            expected_evidence_category="study context",
        )
        expanded = ResearchPlan(
            request=plan.request,
            objective=plan.objective,
            questions=plan.questions,
            evidence_requirements=[*plan.evidence_requirements, requirement],
            tasks=plan.tasks,
        )
        self.assertFalse(expanded.evidence_requirements[1].required)

    def test_task_dependencies_are_preserved_and_may_be_non_linear(self) -> None:
        plan = make_valid_plan()
        self.assertEqual(plan.tasks[1].dependency_ids, [plan.tasks[0].task_id])
        self.assertEqual(plan.tasks[0].status, TaskStatus.PENDING)
        self.assertEqual(plan.questions[0].status, QuestionStatus.PENDING)

    def test_self_dependency_is_rejected(self) -> None:
        plan = make_valid_plan()
        with self.assertRaisesRegex(ValueError, "cannot depend on itself"):
            ResearchPlan(
                request=plan.request,
                objective=plan.objective,
                questions=plan.questions,
                evidence_requirements=plan.evidence_requirements,
                tasks=[ResearchTask(
                    task_id="self-task",
                    description="Invalid self dependency.",
                    question_id=plan.questions[0].question_id,
                    dependency_ids=["self-task"],
                )],
            )

    def test_unknown_dependency_is_rejected(self) -> None:
        plan = make_valid_plan()
        with self.assertRaisesRegex(ValueError, "unknown dependency ID 'missing'"):
            ResearchPlan(
                request=plan.request,
                objective=plan.objective,
                questions=plan.questions,
                tasks=[ResearchTask(
                    description="Has an unknown predecessor.",
                    question_id=plan.questions[0].question_id,
                    dependency_ids=["missing"],
                )],
            )

    def test_circular_task_dependencies_are_rejected(self) -> None:
        plan = make_valid_plan()
        qid = plan.questions[0].question_id
        with self.assertRaisesRegex(ValueError, "Circular task dependency"):
            ResearchPlan(
                request=plan.request,
                objective=plan.objective,
                questions=plan.questions,
                tasks=[
                    ResearchTask(task_id="a", description="A", question_id=qid, dependency_ids=["b"]),
                    ResearchTask(task_id="b", description="B", question_id=qid, dependency_ids=["a"]),
                ],
            )

    def test_circular_parent_questions_are_rejected(self) -> None:
        plan = make_valid_plan()
        with self.assertRaisesRegex(ValueError, "Circular parent-question"):
            ResearchPlan(
                request=plan.request,
                objective=plan.objective,
                questions=[
                    ResearchPlanQuestion(question_id="a", text="A", parent_question_id="b"),
                    ResearchPlanQuestion(question_id="b", text="B", parent_question_id="a"),
                ],
            )

    def test_duplicate_entity_ids_are_rejected(self) -> None:
        plan = make_valid_plan()
        duplicate = ResearchTask(
            task_id=plan.tasks[0].task_id,
            description="Duplicate task ID.",
            question_id=plan.questions[0].question_id,
        )
        with self.assertRaisesRegex(ValueError, "Duplicate ResearchTask task_id"):
            ResearchPlan(
                request=plan.request,
                objective=plan.objective,
                questions=plan.questions,
                evidence_requirements=plan.evidence_requirements,
                tasks=[plan.tasks[0], duplicate],
            )

    def test_duplicate_relationship_entries_are_rejected(self) -> None:
        plan = make_valid_plan()
        req_id = plan.evidence_requirements[0].requirement_id
        with self.assertRaisesRegex(ValueError, "duplicate evidence requirement IDs"):
            ResearchPlan(
                request=plan.request,
                objective=plan.objective,
                questions=plan.questions,
                evidence_requirements=plan.evidence_requirements,
                tasks=[ResearchTask(
                    description="Duplicate link.", question_id=plan.questions[0].question_id,
                    evidence_requirement_ids=[req_id, req_id],
                )],
            )
        task = ResearchTask(description="A task.", question_id=plan.questions[0].question_id)
        task.dependency_ids = [plan.tasks[0].task_id, plan.tasks[0].task_id]
        with self.assertRaisesRegex(ValueError, "duplicate dependency IDs"):
            ResearchPlan(
                request=plan.request, objective=plan.objective,
                questions=plan.questions, tasks=[plan.tasks[0], task],
            )

    def test_invalid_priority_and_status_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported priority"):
            ResearchPlanQuestion("Question", priority="urgent")  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "Unsupported task status"):
            ResearchTask("Task", "q", status="verified")  # type: ignore[arg-type]

    def test_missing_question_requirement_and_task_references_are_rejected(self) -> None:
        plan = make_valid_plan()
        with self.assertRaisesRegex(ValueError, "unknown question_id 'missing'"):
            ResearchPlan(
                request=plan.request,
                objective=plan.objective,
                evidence_requirements=[EvidenceRequirement("missing", "Need evidence")],
            )
        with self.assertRaisesRegex(ValueError, "unknown evidence requirement ID 'missing'"):
            ResearchPlan(
                request=plan.request,
                objective=plan.objective,
                questions=plan.questions,
                tasks=[ResearchTask(
                    "Task", plan.questions[0].question_id,
                    evidence_requirement_ids=["missing"],
                )],
            )
        with self.assertRaisesRegex(ValueError, "unknown question_id 'missing'"):
            ResearchPlan(
                request=plan.request,
                objective=plan.objective,
                tasks=[ResearchTask("Task", "missing")],
            )

    def test_task_requirement_must_belong_to_its_question(self) -> None:
        q1 = ResearchPlanQuestion("First question", question_id="q1")
        q2 = ResearchPlanQuestion("Second question", question_id="q2")
        requirement = EvidenceRequirement("q2", "Second question evidence", requirement_id="er2")
        task = ResearchTask("First question task", "q1", evidence_requirement_ids=["er2"])
        with self.assertRaisesRegex(ValueError, "belong to different questions"):
            ResearchPlan(
                request=make_valid_plan().request,
                objective=make_valid_plan().objective,
                questions=[q1, q2],
                evidence_requirements=[requirement],
                tasks=[task],
            )

    def test_empty_required_text_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "original_text must be a non-empty"):
            ResearchRequest("  ", "Objective")
        with self.assertRaisesRegex(ValueError, "ResearchObjective.text"):
            ResearchObjective(" ")
        with self.assertRaisesRegex(ValueError, "EvidenceRequirement.description"):
            EvidenceRequirement("q", "")
        with self.assertRaisesRegex(ValueError, "ResearchTask.description"):
            ResearchTask("", "q")

    def test_json_serialization_is_valid_and_round_trips(self) -> None:
        plan = make_valid_plan()
        payload = plan.to_json()
        parsed = json.loads(payload)
        self.assertEqual(parsed["request"]["original_text"], plan.request.original_text)
        restored = ResearchPlan.from_json(payload)
        self.assertEqual(restored, plan)
        self.assertEqual(restored.to_dict(), plan.to_dict())

    def test_to_dict_and_from_dict_round_trip_preserves_relationships(self) -> None:
        plan = make_valid_plan()
        restored = ResearchPlan.from_dict(plan.to_dict())
        question_id = restored.questions[0].question_id
        requirement_id = restored.requirements_for_question(question_id)[0].requirement_id
        self.assertEqual(restored.tasks_for_question(question_id)[0].evidence_requirement_ids, [requirement_id])
        self.assertEqual(restored.tasks[1].dependency_ids, [restored.tasks[0].task_id])

    def test_malformed_serialized_plan_is_rejected_without_regenerating_ids(self) -> None:
        payload = make_valid_plan().to_dict()
        payload["questions"][0].pop("question_id")
        with self.assertRaisesRegex(ValueError, "missing required field.*question_id"):
            ResearchPlan.from_dict(payload)
        payload = make_valid_plan().to_dict()
        payload["unexpected"] = "must not be silently discarded"
        with self.assertRaisesRegex(ValueError, "unknown field.*unexpected"):
            ResearchPlan.from_dict(payload)
        with self.assertRaisesRegex(ValueError, "Malformed ResearchPlan JSON"):
            ResearchPlan.from_json("{")

    def test_domain_agnostic_requests_round_trip_across_fields(self) -> None:
        examples = [
            ("AI/ML", "Compare approaches to a general prediction problem."),
            ("medicine", "Summarize outcomes reported for an intervention."),
            ("agriculture", "Understand conditions associated with crop disease."),
            ("physics", "Review explanations for an observed material property."),
        ]
        for domain, text in examples:
            with self.subTest(domain=domain):
                request = ResearchRequest(text, "Describe what prior research reports.", domain=domain)
                plan = ResearchPlan(request, ResearchObjective("Describe reported findings."))
                self.assertEqual(ResearchPlan.from_json(plan.to_json()).request.domain, domain)

    def test_priority_is_a_small_typed_vocabulary(self) -> None:
        self.assertEqual(
            [item.value for item in Priority], ["low", "medium", "high", "critical"]
        )

    def test_task_completion_is_not_a_verification_verdict(self) -> None:
        task = ResearchTask("Review reported findings.", "q", status=TaskStatus.COMPLETED)
        self.assertEqual(task.status.value, "completed")
        self.assertFalse(hasattr(task, "verdict"))


if __name__ == "__main__":
    unittest.main()
