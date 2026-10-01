"""Typed, deterministic contracts for research intake and planning.

These records describe what a researcher wants investigated. They do not
execute searches, establish facts, or represent verified evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
from typing import Any, Dict, List, Optional
from uuid import uuid4


def _new_id(prefix: str) -> str:
    """Generate an entity identifier using the project's short UUID style."""
    return f"{prefix}_{uuid4().hex[:10]}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class QuestionStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class TaskStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    FAILED = "failed"
    CANCELLED = "cancelled"


def _enum_value(enum_type: type[Enum], value: Any, field_name: str) -> Enum:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        supported = ", ".join(item.value for item in enum_type)
        raise ValueError(
            f"Unsupported {field_name} {value!r}; expected one of: {supported}."
        ) from exc


def _require_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")


@dataclass
class ResearchRequest:
    """Original researcher-provided request, kept separate from the plan."""

    original_text: str
    research_objective: str
    request_id: str = field(default_factory=lambda: _new_id("request"))
    domain: Optional[str] = None
    constraints: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=_now)

    def __post_init__(self) -> None:
        _require_text(self.original_text, "ResearchRequest.original_text")
        _require_text(self.research_objective, "ResearchRequest.research_objective")
        _require_text(self.request_id, "ResearchRequest.request_id")
        _require_text(self.created_at, "ResearchRequest.created_at")
        if self.domain is not None:
            _require_text(self.domain, "ResearchRequest.domain")
        if not isinstance(self.constraints, list):
            raise ValueError("ResearchRequest.constraints must be a list of strings.")
        for index, constraint in enumerate(self.constraints):
            _require_text(constraint, f"ResearchRequest.constraints[{index}]")


@dataclass
class ResearchObjective:
    """Structured objective recorded as a planning artifact, not a finding."""

    text: str
    objective_id: str = field(default_factory=lambda: _new_id("objective"))
    scope_description: Optional[str] = None

    def __post_init__(self) -> None:
        _require_text(self.text, "ResearchObjective.text")
        _require_text(self.objective_id, "ResearchObjective.objective_id")
        if self.scope_description is not None:
            _require_text(self.scope_description, "ResearchObjective.scope_description")


@dataclass
class ResearchPlanQuestion:
    """A question in a plan, distinct from an execution-state question."""

    text: str
    priority: Priority = Priority.MEDIUM
    status: QuestionStatus = QuestionStatus.PENDING
    question_id: str = field(default_factory=lambda: _new_id("planq"))
    created_at: str = field(default_factory=_now)
    parent_question_id: Optional[str] = None

    def __post_init__(self) -> None:
        _require_text(self.text, "ResearchPlanQuestion.text")
        _require_text(self.question_id, "ResearchPlanQuestion.question_id")
        _require_text(self.created_at, "ResearchPlanQuestion.created_at")
        self.priority = _enum_value(Priority, self.priority, "priority")  # type: ignore[assignment]
        self.status = _enum_value(QuestionStatus, self.status, "question status")  # type: ignore[assignment]
        if self.parent_question_id is not None:
            _require_text(self.parent_question_id, "ResearchPlanQuestion.parent_question_id")


@dataclass
class EvidenceRequirement:
    """A description of evidence to seek; it is not evidence itself."""

    question_id: str
    description: str
    priority: Priority = Priority.MEDIUM
    required: bool = True
    expected_evidence_category: Optional[str] = None
    requirement_id: str = field(default_factory=lambda: _new_id("requirement"))

    def __post_init__(self) -> None:
        _require_text(self.question_id, "EvidenceRequirement.question_id")
        _require_text(self.description, "EvidenceRequirement.description")
        _require_text(self.requirement_id, "EvidenceRequirement.requirement_id")
        if not isinstance(self.required, bool):
            raise ValueError("EvidenceRequirement.required must be a boolean.")
        self.priority = _enum_value(Priority, self.priority, "priority")  # type: ignore[assignment]
        if self.expected_evidence_category is not None:
            _require_text(
                self.expected_evidence_category,
                "EvidenceRequirement.expected_evidence_category",
            )


@dataclass
class ResearchTask:
    """A future executable unit; completion does not imply a verified claim."""

    description: str
    question_id: str
    category: str = "research"
    priority: Priority = Priority.MEDIUM
    status: TaskStatus = TaskStatus.PENDING
    evidence_requirement_ids: List[str] = field(default_factory=list)
    dependency_ids: List[str] = field(default_factory=list)
    task_id: str = field(default_factory=lambda: _new_id("task"))
    created_at: str = field(default_factory=_now)

    def __post_init__(self) -> None:
        _require_text(self.description, "ResearchTask.description")
        _require_text(self.question_id, "ResearchTask.question_id")
        _require_text(self.category, "ResearchTask.category")
        _require_text(self.task_id, "ResearchTask.task_id")
        _require_text(self.created_at, "ResearchTask.created_at")
        self.priority = _enum_value(Priority, self.priority, "priority")  # type: ignore[assignment]
        self.status = _enum_value(TaskStatus, self.status, "task status")  # type: ignore[assignment]
        for name, values in (
            ("evidence_requirement_ids", self.evidence_requirement_ids),
            ("dependency_ids", self.dependency_ids),
        ):
            if not isinstance(values, list):
                raise ValueError(f"ResearchTask.{name} must be a list of IDs.")
            for index, value in enumerate(values):
                _require_text(value, f"ResearchTask.{name}[{index}]")


@dataclass
class ResearchPlan:
    """Standalone planning contract with typed, validated relationships."""

    request: ResearchRequest
    objective: ResearchObjective
    questions: List[ResearchPlanQuestion] = field(default_factory=list)
    evidence_requirements: List[EvidenceRequirement] = field(default_factory=list)
    tasks: List[ResearchTask] = field(default_factory=list)
    plan_id: str = field(default_factory=lambda: _new_id("plan"))
    created_at: str = field(default_factory=_now)
    revision: int = 1

    def __post_init__(self) -> None:
        errors = self.validate()
        if errors:
            raise ValueError("Invalid ResearchPlan: " + "; ".join(errors))

    def validate(self) -> List[str]:
        """Return deterministic structural errors without executing research."""
        errors: List[str] = []
        if not isinstance(self.request, ResearchRequest):
            errors.append("request must be a ResearchRequest.")
        if not isinstance(self.objective, ResearchObjective):
            errors.append("objective must be a ResearchObjective.")
        collection_specs = (
            ("questions", self.questions, ResearchPlanQuestion),
            ("evidence_requirements", self.evidence_requirements, EvidenceRequirement),
            ("tasks", self.tasks, ResearchTask),
        )
        for name, values, expected_type in collection_specs:
            if not isinstance(values, list):
                errors.append(f"{name} must be a list.")
            elif any(not isinstance(item, expected_type) for item in values):
                errors.append(f"{name} must contain only {expected_type.__name__} records.")
        if errors:
            return errors
        _append_duplicate_errors(errors, "ResearchPlanQuestion", self.questions, "question_id")
        _append_duplicate_errors(errors, "EvidenceRequirement", self.evidence_requirements, "requirement_id")
        _append_duplicate_errors(errors, "ResearchTask", self.tasks, "task_id")

        if not isinstance(self.plan_id, str) or not self.plan_id.strip():
            errors.append("plan_id must be a non-empty string.")
        if not isinstance(self.created_at, str) or not self.created_at.strip():
            errors.append("created_at must be a non-empty string.")
        if not isinstance(self.revision, int) or isinstance(self.revision, bool) or self.revision < 1:
            errors.append("revision must be a positive integer.")

        all_ids: List[tuple[str, str]] = []
        if isinstance(self.request, ResearchRequest):
            all_ids.append(("request_id", self.request.request_id))
        if isinstance(self.objective, ResearchObjective):
            all_ids.append(("objective_id", self.objective.objective_id))
        all_ids.extend(("question_id", item.question_id) for item in self.questions)
        all_ids.extend(("requirement_id", item.requirement_id) for item in self.evidence_requirements)
        all_ids.extend(("task_id", item.task_id) for item in self.tasks)
        all_ids.append(("plan_id", self.plan_id))
        seen_ids: Dict[str, str] = {}
        for field_name, value in all_ids:
            if value in seen_ids:
                errors.append(
                    f"Duplicate ID {value!r} across {seen_ids[value]} and {field_name}."
                )
            else:
                seen_ids[value] = field_name

        question_by_id = {item.question_id: item for item in self.questions}
        requirement_by_id = {item.requirement_id: item for item in self.evidence_requirements}
        task_by_id = {item.task_id: item for item in self.tasks}

        for question in self.questions:
            parent_id = question.parent_question_id
            if parent_id is not None and parent_id not in question_by_id:
                errors.append(
                    f"ResearchPlanQuestion {question.question_id!r} references unknown "
                    f"parent_question_id {parent_id!r}."
                )
            elif parent_id == question.question_id:
                errors.append(f"ResearchPlanQuestion {question.question_id!r} cannot parent itself.")
        errors.extend(_question_cycle_errors(question_by_id))

        for requirement in self.evidence_requirements:
            if requirement.question_id not in question_by_id:
                errors.append(
                    f"EvidenceRequirement {requirement.requirement_id!r} references unknown "
                    f"question_id {requirement.question_id!r}."
                )

        for task in self.tasks:
            if task.question_id not in question_by_id:
                errors.append(
                    f"ResearchTask {task.task_id!r} references unknown question_id {task.question_id!r}."
                )
            if len(set(task.evidence_requirement_ids)) != len(task.evidence_requirement_ids):
                errors.append(f"ResearchTask {task.task_id!r} has duplicate evidence requirement IDs.")
            for requirement_id in task.evidence_requirement_ids:
                requirement = requirement_by_id.get(requirement_id)
                if requirement is None:
                    errors.append(
                        f"ResearchTask {task.task_id!r} references unknown evidence requirement "
                        f"ID {requirement_id!r}."
                    )
                elif requirement.question_id != task.question_id:
                    errors.append(
                        f"ResearchTask {task.task_id!r} and EvidenceRequirement "
                        f"{requirement_id!r} belong to different questions."
                    )
            if len(set(task.dependency_ids)) != len(task.dependency_ids):
                errors.append(f"ResearchTask {task.task_id!r} has duplicate dependency IDs.")
            for dependency_id in task.dependency_ids:
                if dependency_id == task.task_id:
                    errors.append(f"ResearchTask {task.task_id!r} cannot depend on itself.")
                elif dependency_id not in task_by_id:
                    errors.append(
                        f"ResearchTask {task.task_id!r} references unknown dependency ID "
                        f"{dependency_id!r}."
                    )
        errors.extend(_task_cycle_errors(task_by_id))
        return errors

    def requirements_for_question(self, question_id: str) -> List[EvidenceRequirement]:
        """Return requirements associated by their question_id, in plan order."""
        if question_id not in {item.question_id for item in self.questions}:
            raise ValueError(f"Unknown research question ID: {question_id!r}.")
        return [item for item in self.evidence_requirements if item.question_id == question_id]

    def tasks_for_question(self, question_id: str) -> List[ResearchTask]:
        """Return tasks associated by their question_id, in plan order."""
        if question_id not in {item.question_id for item in self.questions}:
            raise ValueError(f"Unknown research question ID: {question_id!r}.")
        return [item for item in self.tasks if item.question_id == question_id]

    def to_dict(self) -> Dict[str, Any]:
        errors = self.validate()
        if errors:
            raise ValueError("Invalid ResearchPlan: " + "; ".join(errors))
        return {
            "plan_id": self.plan_id,
            "created_at": self.created_at,
            "revision": self.revision,
            "request": {
                "request_id": self.request.request_id,
                "original_text": self.request.original_text,
                "research_objective": self.request.research_objective,
                "domain": self.request.domain,
                "constraints": list(self.request.constraints),
                "created_at": self.request.created_at,
            },
            "objective": {
                "objective_id": self.objective.objective_id,
                "text": self.objective.text,
                "scope_description": self.objective.scope_description,
            },
            "questions": [
                {
                    "question_id": item.question_id,
                    "text": item.text,
                    "priority": item.priority.value,
                    "status": item.status.value,
                    "created_at": item.created_at,
                    "parent_question_id": item.parent_question_id,
                }
                for item in self.questions
            ],
            "evidence_requirements": [
                {
                    "requirement_id": item.requirement_id,
                    "question_id": item.question_id,
                    "description": item.description,
                    "priority": item.priority.value,
                    "required": item.required,
                    "expected_evidence_category": item.expected_evidence_category,
                }
                for item in self.evidence_requirements
            ],
            "tasks": [
                {
                    "task_id": item.task_id,
                    "description": item.description,
                    "question_id": item.question_id,
                    "category": item.category,
                    "priority": item.priority.value,
                    "status": item.status.value,
                    "evidence_requirement_ids": list(item.evidence_requirement_ids),
                    "dependency_ids": list(item.dependency_ids),
                    "created_at": item.created_at,
                }
                for item in self.tasks
            ],
        }

    def to_json(self) -> str:
        """Return stable JSON for logging, persistence, and tests."""
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResearchPlan":
        if not isinstance(data, dict):
            raise ValueError("ResearchPlan payload must be a JSON object.")
        _require_keys(data, {"plan_id", "created_at", "revision", "request", "objective", "questions", "evidence_requirements", "tasks"}, "ResearchPlan")
        _reject_unknown_keys(data, {"plan_id", "created_at", "revision", "request", "objective", "questions", "evidence_requirements", "tasks"}, "ResearchPlan")
        request_data = _mapping(data["request"], "request")
        _require_keys(request_data, {"request_id", "original_text", "research_objective", "created_at"}, "ResearchRequest")
        _reject_unknown_keys(request_data, {"request_id", "original_text", "research_objective", "domain", "constraints", "created_at"}, "ResearchRequest")
        objective_data = _mapping(data["objective"], "objective")
        _require_keys(objective_data, {"objective_id", "text"}, "ResearchObjective")
        _reject_unknown_keys(objective_data, {"objective_id", "text", "scope_description"}, "ResearchObjective")
        questions_data = _list_of_mappings(data["questions"], "questions")
        for item in questions_data:
            _require_keys(item, {"question_id", "text", "priority", "status", "created_at"}, "ResearchPlanQuestion")
            _reject_unknown_keys(item, {"question_id", "text", "priority", "status", "created_at", "parent_question_id"}, "ResearchPlanQuestion")
        requirements_data = _list_of_mappings(data["evidence_requirements"], "evidence_requirements")
        for item in requirements_data:
            _require_keys(item, {"requirement_id", "question_id", "description", "priority", "required"}, "EvidenceRequirement")
            _reject_unknown_keys(item, {"requirement_id", "question_id", "description", "priority", "required", "expected_evidence_category"}, "EvidenceRequirement")
        tasks_data = _list_of_mappings(data["tasks"], "tasks")
        for item in tasks_data:
            _require_keys(item, {"task_id", "description", "question_id", "category", "priority", "status", "created_at"}, "ResearchTask")
            _reject_unknown_keys(item, {"task_id", "description", "question_id", "category", "priority", "status", "evidence_requirement_ids", "dependency_ids", "created_at"}, "ResearchTask")
        try:
            request = ResearchRequest(**request_data)
            objective = ResearchObjective(**objective_data)
            questions = [ResearchPlanQuestion(**item) for item in questions_data]
            requirements = [EvidenceRequirement(**item) for item in requirements_data]
            tasks = [ResearchTask(**item) for item in tasks_data]
            return cls(
                request=request,
                objective=objective,
                questions=questions,
                evidence_requirements=requirements,
                tasks=tasks,
                plan_id=data["plan_id"],
                created_at=data["created_at"],
                revision=data["revision"],
            )
        except (TypeError, KeyError) as exc:
            raise ValueError(f"Malformed ResearchPlan payload: {exc}") from exc

    @classmethod
    def from_json(cls, payload: str) -> "ResearchPlan":
        try:
            data = json.loads(payload)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Malformed ResearchPlan JSON: {exc}") from exc
        return cls.from_dict(data)


def _append_duplicate_errors(errors: List[str], label: str, items: List[Any], field_name: str) -> None:
    values = [getattr(item, field_name) for item in items]
    if len(set(values)) != len(values):
        seen: set[str] = set()
        for value in values:
            if value in seen:
                errors.append(f"Duplicate {label} {field_name} {value!r}.")
                break
            seen.add(value)


def _question_cycle_errors(questions: Dict[str, ResearchPlanQuestion]) -> List[str]:
    errors: List[str] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(question_id: str) -> None:
        if question_id in visited:
            return
        if question_id in visiting:
            errors.append(f"Circular parent-question relationship includes {question_id!r}.")
            return
        visiting.add(question_id)
        parent_id = questions[question_id].parent_question_id
        if parent_id in questions:
            visit(parent_id)
        visiting.remove(question_id)
        visited.add(question_id)

    for question_id in questions:
        visit(question_id)
    return errors


def _task_cycle_errors(tasks: Dict[str, ResearchTask]) -> List[str]:
    errors: List[str] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(task_id: str) -> None:
        if task_id in visited:
            return
        if task_id in visiting:
            errors.append(f"Circular task dependency includes {task_id!r}.")
            return
        visiting.add(task_id)
        for dependency_id in tasks[task_id].dependency_ids:
            if dependency_id in tasks:
                visit(dependency_id)
        visiting.remove(task_id)
        visited.add(task_id)

    for task_id in tasks:
        visit(task_id)
    return errors


def _mapping(value: Any, name: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"ResearchPlan {name} must be a JSON object.")
    return value


def _list_of_mappings(value: Any, name: str) -> List[Dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"ResearchPlan {name} must be a list of JSON objects.")
    return value


def _require_keys(data: Dict[str, Any], keys: set[str], label: str) -> None:
    missing = sorted(keys - data.keys())
    if missing:
        raise ValueError(f"Malformed {label}: missing required field(s): {', '.join(missing)}.")


def _reject_unknown_keys(data: Dict[str, Any], keys: set[str], label: str) -> None:
    unknown = sorted(data.keys() - keys)
    if unknown:
        raise ValueError(f"Malformed {label}: unknown field(s): {', '.join(unknown)}.")


__all__ = [
    "Priority",
    "QuestionStatus",
    "TaskStatus",
    "ResearchRequest",
    "ResearchObjective",
    "ResearchPlanQuestion",
    "EvidenceRequirement",
    "ResearchTask",
    "ResearchPlan",
]
