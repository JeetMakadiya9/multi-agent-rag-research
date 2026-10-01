"""Typed autonomous research proposals with deterministic validation.

This is a decision layer only.  It gives an ``LLMProvider`` a serialized,
read-only snapshot and accepts one closed, structured proposal in return.  It
never receives mutable ``ResearchState`` in the generation call, never calls a
tool/capability, and never persists or executes a proposal.  The caller must
validate a proposal and separately submit any accepted action through the
existing Phase 5–7E boundaries.

Provider generation can be nondeterministic.  Context construction and
proposal validation are deterministic for an unchanged state/policy snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import math
from typing import Optional, Sequence
from uuid import uuid4

from llm import LLMProvider
from research_bounded_loop import BoundedExecutionPolicy
from research_capabilities import CapabilityType
from research_loop import (
    LEGAL_TRANSITIONS,
    LoopStage,
    ResearchLoop,
    ResearchLoopStatus,
    capability_compatible_with_stage,
)
from research_plan_review import PlanReview, is_execution_approved
from research_planning import ResearchPlan, ResearchTask
from research_state import ResearchArtifact, ResearchRun, ResearchState


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


def _text(value: object, name: str, *, optional: bool = False) -> None:
    if value is None and optional:
        return
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text.")


def _ids(values: object, name: str, *, optional: bool = True) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)):
        raise ValueError(f"{name} must be a sequence of IDs.")
    result = tuple(values)
    if not optional and not result:
        raise ValueError(f"{name} must not be empty.")
    if any(not isinstance(item, str) or not item.strip() for item in result):
        raise ValueError(f"{name} must contain non-empty IDs.")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicates.")
    return result


def _timestamp(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a timezone-aware ISO-8601 timestamp.")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a timezone-aware ISO-8601 timestamp.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must include a timezone.")


def _enum(enum_type, value, name):
    try:
        return value if isinstance(value, enum_type) else enum_type(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unsupported {name}: {value!r}.") from exc


class ResearchActionType(str, Enum):
    RETRIEVE_EVIDENCE = "RETRIEVE_EVIDENCE"
    VERIFY_CLAIM = "VERIFY_CLAIM"
    SYNTHESIZE = "SYNTHESIZE"
    CRITIQUE = "CRITIQUE"
    GENERATE_IMPROVEMENT = "GENERATE_IMPROVEMENT"
    INVESTIGATE_PRIOR_WORK = "INVESTIGATE_PRIOR_WORK"
    PLAN_EXPERIMENT = "PLAN_EXPERIMENT"
    REQUEST_RESEARCHER_REVIEW = "REQUEST_RESEARCHER_REVIEW"
    RETURN_TO_PLANNING = "RETURN_TO_PLANNING"
    STOP = "STOP"


class ResearcherOverrideAction(str, Enum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    MODIFY = "MODIFY"
    STOP = "STOP"


class ValidationStatus(str, Enum):
    PENDING = "PENDING"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"


_ACTION_CAPABILITY = {
    ResearchActionType.RETRIEVE_EVIDENCE: CapabilityType.LOCAL_RETRIEVAL,
    ResearchActionType.VERIFY_CLAIM: CapabilityType.EVIDENCE_VERIFICATION,
    ResearchActionType.GENERATE_IMPROVEMENT: CapabilityType.IMPROVEMENT_GENERATION,
    ResearchActionType.INVESTIGATE_PRIOR_WORK: CapabilityType.PRIOR_WORK_INVESTIGATION,
    ResearchActionType.PLAN_EXPERIMENT: CapabilityType.EXPERIMENT_PLANNING,
}


@dataclass(frozen=True)
class ContextTask:
    task_id: str
    description: str
    question_id: str
    category: str
    priority: str
    status: str

    def to_dict(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class ContextQuestion:
    question_id: str
    text: str
    priority: str
    status: str

    def to_dict(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class ContextEvidence:
    evidence_id: str
    text: str
    source_id: str
    document_version_id: Optional[str]
    passage_reference_id: Optional[str]
    search_result_id: Optional[str]
    text_hash: Optional[str]
    provenance_ids: tuple[str, ...]

    def to_dict(self):
        return {**{name: getattr(self, name) for name in self.__dataclass_fields__
                   if name != "provenance_ids"}, "provenance_ids": list(self.provenance_ids)}


@dataclass(frozen=True)
class ContextClaim:
    claim_id: str
    text: str
    status: str
    evidence_ids: tuple[str, ...]
    verification_reason: str

    def to_dict(self):
        return {**{name: getattr(self, name) for name in self.__dataclass_fields__
                   if name != "evidence_ids"}, "evidence_ids": list(self.evidence_ids)}


@dataclass(frozen=True)
class ContextArtifact:
    artifact_id: str
    artifact_type: str
    capability: str
    task_id: str
    evidence_ids: tuple[str, ...]
    claim: Optional[str] = None
    verdict: Optional[str] = None
    candidate_ids: tuple[str, ...] = ()
    epistemic_notes: tuple[str, ...] = ()
    summary: str = ""

    def __post_init__(self):
        for name in ("artifact_id", "artifact_type", "capability", "task_id"):
            _text(getattr(self, name), name)
        for name in ("claim", "verdict"):
            if getattr(self, name) is not None:
                _text(getattr(self, name), name)
        for name in ("evidence_ids", "candidate_ids", "epistemic_notes"):
            values = tuple(getattr(self, name))
            if any(not isinstance(x, str) or not x.strip() for x in values):
                raise ValueError(f"{name} must contain non-empty strings.")
            object.__setattr__(self, name, values)
        if not isinstance(self.summary, str):
            raise ValueError("summary must be text.")

    def to_dict(self):
        value = {name: getattr(self, name) for name in self.__dataclass_fields__
                 if name not in {"evidence_ids", "candidate_ids", "epistemic_notes"}}
        value["evidence_ids"] = list(self.evidence_ids)
        value["candidate_ids"] = list(self.candidate_ids)
        value["epistemic_notes"] = list(self.epistemic_notes)
        return value


@dataclass(frozen=True)
class BudgetSnapshot:
    maximum_iterations: int
    iterations_used: int
    maximum_capability_invocations: int
    capability_invocations_used: int
    maximum_task_executions: int
    task_executions_used: int
    hard_stop_reasons: tuple[str, ...] = ()

    def to_dict(self):
        return {**{name: getattr(self, name) for name in self.__dataclass_fields__
                   if name != "hard_stop_reasons"},
                "hard_stop_reasons": list(self.hard_stop_reasons)}


@dataclass(frozen=True)
class AutonomousResearchContext:
    """Immutable, purpose-built context; deliberately contains no state object."""

    context_id: str
    run_id: str
    plan_id: str
    plan_revision: int
    loop_id: str
    current_task_id: Optional[str]
    iteration_id: Optional[str]
    iteration_number: int
    current_stage: LoopStage
    loop_status: ResearchLoopStatus
    objective: str
    research_domain: Optional[str]
    questions: tuple[ContextQuestion, ...]
    tasks: tuple[ContextTask, ...]
    authorized_task_ids: tuple[str, ...]
    authorized_capabilities: tuple[CapabilityType, ...]
    evidence: tuple[ContextEvidence, ...]
    claims: tuple[ContextClaim, ...]
    artifacts: tuple[ContextArtifact, ...]
    unresolved_questions: tuple[str, ...]
    unresolved_contradictions: tuple[str, ...]
    unresolved_work: tuple[str, ...]
    completed_task_ids: tuple[str, ...]
    failed_task_ids: tuple[str, ...]
    provenance_ids: tuple[str, ...]
    budgets: BudgetSnapshot
    created_at: str = field(default_factory=_now)

    def __post_init__(self):
        for name in ("context_id", "run_id", "plan_id", "loop_id", "objective"):
            _text(getattr(self, name), f"AutonomousResearchContext.{name}")
        if self.research_domain is not None:
            _text(self.research_domain, "research_domain")
        if not isinstance(self.plan_revision, int) or isinstance(self.plan_revision, bool) or self.plan_revision < 1:
            raise ValueError("context plan_revision must be a positive integer.")
        if not isinstance(self.iteration_number, int) or isinstance(self.iteration_number, bool) or self.iteration_number < 0:
            raise ValueError("context iteration_number must be non-negative.")
        object.__setattr__(self, "current_stage", _enum(LoopStage, self.current_stage, "current_stage"))
        object.__setattr__(self, "loop_status", _enum(ResearchLoopStatus, self.loop_status, "loop_status"))
        if self.iteration_id is not None:
            _text(self.iteration_id, "iteration_id")
        if self.current_task_id is not None:
            _text(self.current_task_id, "current_task_id")
        for name, kind in (("questions", ContextQuestion), ("tasks", ContextTask),
                           ("evidence", ContextEvidence), ("claims", ContextClaim), ("artifacts", ContextArtifact)):
            items = tuple(getattr(self, name))
            if any(not isinstance(item, kind) for item in items):
                raise ValueError(f"{name} must contain {kind.__name__} snapshots.")
            object.__setattr__(self, name, items)
        for name in ("authorized_task_ids", "unresolved_questions", "unresolved_contradictions",
                     "unresolved_work", "completed_task_ids", "failed_task_ids", "provenance_ids"):
            object.__setattr__(self, name, _ids(getattr(self, name), name))
        capabilities = tuple(_enum(CapabilityType, item, "authorized capability") for item in self.authorized_capabilities)
        if len(capabilities) != len(set(capabilities)):
            raise ValueError("authorized_capabilities must not contain duplicates.")
        object.__setattr__(self, "authorized_capabilities", capabilities)
        if not isinstance(self.budgets, BudgetSnapshot):
            raise ValueError("budgets must be a BudgetSnapshot.")
        _timestamp(self.created_at, "created_at")

    def to_dict(self):
        return {
            "context_id": self.context_id, "run_id": self.run_id, "plan_id": self.plan_id,
            "plan_revision": self.plan_revision, "loop_id": self.loop_id,
            "current_task_id": self.current_task_id, "iteration_id": self.iteration_id,
            "iteration_number": self.iteration_number,
            "current_stage": self.current_stage.value, "loop_status": self.loop_status.value,
            "objective": self.objective, "research_domain": self.research_domain,
            "questions": [x.to_dict() for x in self.questions], "tasks": [x.to_dict() for x in self.tasks],
            "authorized_task_ids": list(self.authorized_task_ids),
            "authorized_capabilities": [x.value for x in self.authorized_capabilities],
            "evidence": [x.to_dict() for x in self.evidence], "claims": [x.to_dict() for x in self.claims],
            "artifacts": [x.to_dict() for x in self.artifacts],
            "unresolved_questions": list(self.unresolved_questions),
            "unresolved_contradictions": list(self.unresolved_contradictions),
            "unresolved_work": list(self.unresolved_work),
            "completed_task_ids": list(self.completed_task_ids), "failed_task_ids": list(self.failed_task_ids),
            "provenance_ids": list(self.provenance_ids),
            "budgets": self.budgets.to_dict(), "created_at": self.created_at,
        }

    def to_json(self):
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, data):
        required = {"context_id", "run_id", "plan_id", "plan_revision", "loop_id", "current_task_id", "iteration_id",
                    "iteration_number", "current_stage", "loop_status", "objective", "research_domain",
                    "questions", "tasks", "authorized_task_ids", "authorized_capabilities", "evidence",
                    "claims", "artifacts", "unresolved_questions", "unresolved_contradictions",
                    "unresolved_work", "completed_task_ids", "failed_task_ids", "provenance_ids",
                    "budgets", "created_at"}
        if not isinstance(data, dict) or set(data) != required:
            raise ValueError("Malformed AutonomousResearchContext payload.")
        try:
            return cls(**{**data,
                          "questions": tuple(ContextQuestion(**x) for x in data["questions"]),
                          "tasks": tuple(ContextTask(**x) for x in data["tasks"]),
                          "authorized_task_ids": tuple(data["authorized_task_ids"]),
                          "authorized_capabilities": tuple(data["authorized_capabilities"]),
                          "evidence": tuple(ContextEvidence(**{**x, "provenance_ids": tuple(x["provenance_ids"])}) for x in data["evidence"]),
                          "claims": tuple(ContextClaim(**{**x, "evidence_ids": tuple(x["evidence_ids"])}) for x in data["claims"]),
                          "artifacts": tuple(ContextArtifact(**{**x, "evidence_ids": tuple(x["evidence_ids"]),
                                                                  "candidate_ids": tuple(x["candidate_ids"]),
                                                                  "epistemic_notes": tuple(x["epistemic_notes"])}) for x in data["artifacts"]),
                          "unresolved_questions": tuple(data["unresolved_questions"]),
                          "unresolved_contradictions": tuple(data["unresolved_contradictions"]),
                          "unresolved_work": tuple(data["unresolved_work"]),
                          "completed_task_ids": tuple(data["completed_task_ids"]),
                          "failed_task_ids": tuple(data["failed_task_ids"]),
                          "provenance_ids": tuple(data["provenance_ids"]),
                          "budgets": BudgetSnapshot(**{**data["budgets"], "hard_stop_reasons": tuple(data["budgets"]["hard_stop_reasons"])})})
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Malformed AutonomousResearchContext: {exc}") from exc

    @classmethod
    def from_json(cls, payload):
        try:
            return cls.from_dict(json.loads(payload))
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Malformed AutonomousResearchContext JSON: {exc}") from exc


@dataclass(frozen=True)
class ResearchStopAssessment:
    stop_reason: str
    evidence_summary: str
    unresolved_questions: tuple[str, ...] = ()
    remaining_known_limitations: tuple[str, ...] = ()
    assessment: str = ""

    def __post_init__(self):
        for name in ("stop_reason", "evidence_summary"):
            _text(getattr(self, name), name)
        for name in ("unresolved_questions", "remaining_known_limitations"):
            values = tuple(getattr(self, name))
            if any(not isinstance(x, str) or not x.strip() for x in values):
                raise ValueError(f"{name} must contain non-empty text.")
            object.__setattr__(self, name, values)
        if not isinstance(self.assessment, str):
            raise ValueError("assessment must be text.")

    def to_dict(self):
        return {"stop_reason": self.stop_reason, "evidence_summary": self.evidence_summary,
                "unresolved_questions": list(self.unresolved_questions),
                "remaining_known_limitations": list(self.remaining_known_limitations),
                "assessment": self.assessment}

    @classmethod
    def from_dict(cls, data):
        keys = {"stop_reason", "evidence_summary", "unresolved_questions", "remaining_known_limitations", "assessment"}
        if not isinstance(data, dict) or set(data) != keys:
            raise ValueError("Malformed ResearchStopAssessment payload.")
        return cls(**{**data, "unresolved_questions": tuple(data["unresolved_questions"]),
                      "remaining_known_limitations": tuple(data["remaining_known_limitations"])})


@dataclass(frozen=True)
class ResearchActionProposal:
    """Closed action proposal. Parameters are typed fields, not a tool-call map."""

    action_type: ResearchActionType
    rationale: str
    task_id: Optional[str] = None
    question_ids: tuple[str, ...] = ()
    claim_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    artifact_ids: tuple[str, ...] = ()
    query: Optional[str] = None
    problem_statement: Optional[str] = None
    candidate_id: Optional[str] = None
    estimated_information_gain: Optional[float] = None
    priority: str = "medium"
    unresolved_questions_addressed: tuple[str, ...] = ()
    stop_after_action: bool = False
    stop_assessment: Optional[ResearchStopAssessment] = None
    provenance_ids: tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "action_type", _enum(ResearchActionType, self.action_type, "action_type"))
        _text(self.rationale, "rationale")
        if self.task_id is not None:
            _text(self.task_id, "task_id")
        for name in ("question_ids", "claim_ids", "evidence_ids", "artifact_ids",
                     "unresolved_questions_addressed", "provenance_ids"):
            object.__setattr__(self, name, _ids(getattr(self, name), name))
        for name in ("query", "problem_statement", "candidate_id"):
            if getattr(self, name) is not None:
                _text(getattr(self, name), name)
        if self.estimated_information_gain is not None:
            value = self.estimated_information_gain
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("estimated_information_gain must be a finite estimate from 0 to 1.")
        if self.priority not in {"low", "medium", "high", "critical"}:
            raise ValueError("priority must be low, medium, high, or critical.")
        if not isinstance(self.stop_after_action, bool):
            raise ValueError("stop_after_action must be boolean.")
        if self.action_type is ResearchActionType.STOP:
            if not isinstance(self.stop_assessment, ResearchStopAssessment):
                raise ValueError("STOP proposal requires a typed stop_assessment.")
        elif self.stop_assessment is not None:
            raise ValueError("Only STOP proposals may include stop_assessment.")
        if self.action_type in _ACTION_CAPABILITY and not self.task_id:
            raise ValueError(f"{self.action_type.value} requires a target task_id.")
        if self.action_type is ResearchActionType.RETRIEVE_EVIDENCE and not self.query:
            raise ValueError("RETRIEVE_EVIDENCE requires a query.")
        if self.action_type is ResearchActionType.VERIFY_CLAIM and not self.claim_ids:
            raise ValueError("VERIFY_CLAIM requires stored claim_ids.")
        if self.action_type is ResearchActionType.VERIFY_CLAIM and not self.evidence_ids:
            raise ValueError("VERIFY_CLAIM requires stored evidence_ids.")
        if self.action_type is ResearchActionType.GENERATE_IMPROVEMENT and (not self.problem_statement or not self.evidence_ids):
            raise ValueError("GENERATE_IMPROVEMENT requires a problem_statement and stored evidence_ids.")
        if self.action_type in {ResearchActionType.INVESTIGATE_PRIOR_WORK, ResearchActionType.PLAN_EXPERIMENT} and not self.candidate_id:
            raise ValueError(f"{self.action_type.value} requires a candidate_id.")

    @property
    def required_capabilities(self) -> tuple[CapabilityType, ...]:
        """Capabilities requested; this is a proposal, not a grant."""
        capability = _ACTION_CAPABILITY.get(self.action_type)
        return (capability,) if capability else ()

    @property
    def required_authorizations(self) -> tuple[str, ...]:
        required = ["approved_exact_plan_revision", "registered_research_run"]
        if self.action_type in _ACTION_CAPABILITY:
            required.append("explicitly_authorized_task")
        if self.evidence_ids or self.claim_ids or self.artifact_ids:
            required.append("stored_reference_resolution")
        if self.action_type is ResearchActionType.RETURN_TO_PLANNING:
            required.append("explicit_researcher_acceptance")
        return tuple(required)

    def to_dict(self):
        return {"action_type": self.action_type.value, "rationale": self.rationale, "task_id": self.task_id,
                "question_ids": list(self.question_ids), "claim_ids": list(self.claim_ids),
                "evidence_ids": list(self.evidence_ids), "artifact_ids": list(self.artifact_ids),
                "query": self.query, "problem_statement": self.problem_statement,
                "candidate_id": self.candidate_id, "estimated_information_gain": self.estimated_information_gain,
                "priority": self.priority, "unresolved_questions_addressed": list(self.unresolved_questions_addressed),
                "stop_after_action": self.stop_after_action,
                "stop_assessment": self.stop_assessment.to_dict() if self.stop_assessment else None,
                "provenance_ids": list(self.provenance_ids),
                "required_capabilities": [x.value for x in self.required_capabilities],
                "required_authorizations": list(self.required_authorizations)}

    @classmethod
    def from_dict(cls, data):
        keys = {"action_type", "rationale", "task_id", "question_ids", "claim_ids", "evidence_ids",
                "artifact_ids", "query", "problem_statement", "candidate_id", "estimated_information_gain",
                "priority", "unresolved_questions_addressed", "stop_after_action", "stop_assessment", "provenance_ids",
                "required_capabilities", "required_authorizations"}
        if not isinstance(data, dict) or set(data) != keys:
            raise ValueError("Malformed ResearchActionProposal payload.")
        try:
            proposal = cls(**{**{k: v for k, v in data.items()
                                if k not in {"required_capabilities", "required_authorizations"}},
                          "question_ids": tuple(data["question_ids"]),
                          "claim_ids": tuple(data["claim_ids"]), "evidence_ids": tuple(data["evidence_ids"]),
                          "artifact_ids": tuple(data["artifact_ids"]),
                          "unresolved_questions_addressed": tuple(data["unresolved_questions_addressed"]),
                          "stop_assessment": ResearchStopAssessment.from_dict(data["stop_assessment"]) if data["stop_assessment"] else None,
                          "provenance_ids": tuple(data["provenance_ids"])})
            if data["required_capabilities"] != [x.value for x in proposal.required_capabilities]:
                raise ValueError("Serialized required_capabilities do not match the typed action.")
            if data["required_authorizations"] != list(proposal.required_authorizations):
                raise ValueError("Serialized required_authorizations do not match the typed action.")
            return proposal
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Malformed ResearchActionProposal: {exc}") from exc


@dataclass(frozen=True)
class ResearchDecisionTrace:
    decision_id: str
    context_id: str
    loop_id: str
    run_id: str
    iteration_id: Optional[str]
    action_type: ResearchActionType
    rationale: str
    provider: str
    model: Optional[str]
    created_at: str
    provenance_ids: tuple[str, ...] = ()
    validation_status: ValidationStatus = ValidationStatus.PENDING
    rejection_reasons: tuple[str, ...] = ()

    def __post_init__(self):
        for name in ("decision_id", "context_id", "loop_id", "run_id", "provider", "rationale"):
            _text(getattr(self, name), name)
        if self.iteration_id is not None:
            _text(self.iteration_id, "iteration_id")
        if self.model is not None:
            _text(self.model, "model")
        object.__setattr__(self, "action_type", _enum(ResearchActionType, self.action_type, "action_type"))
        object.__setattr__(self, "validation_status", _enum(ValidationStatus, self.validation_status, "validation_status"))
        object.__setattr__(self, "provenance_ids", _ids(self.provenance_ids, "provenance_ids"))
        object.__setattr__(self, "rejection_reasons", _ids(self.rejection_reasons, "rejection_reasons"))
        _timestamp(self.created_at, "created_at")
        if self.validation_status is ValidationStatus.REJECTED and not self.rejection_reasons:
            raise ValueError("Rejected trace requires rejection_reasons.")
        if self.validation_status is not ValidationStatus.REJECTED and self.rejection_reasons:
            raise ValueError("Only rejected traces may contain rejection_reasons.")

    def to_dict(self):
        return {"decision_id": self.decision_id, "context_id": self.context_id, "loop_id": self.loop_id,
                "run_id": self.run_id, "iteration_id": self.iteration_id, "action_type": self.action_type.value,
                "rationale": self.rationale, "provider": self.provider, "model": self.model,
                "created_at": self.created_at, "provenance_ids": list(self.provenance_ids),
                "validation_status": self.validation_status.value,
                "rejection_reasons": list(self.rejection_reasons)}

    def to_json(self):
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, data):
        keys = {"decision_id", "context_id", "loop_id", "run_id", "iteration_id", "action_type", "rationale",
                "provider", "model", "created_at", "provenance_ids", "validation_status", "rejection_reasons"}
        if not isinstance(data, dict) or set(data) != keys:
            raise ValueError("Malformed ResearchDecisionTrace payload.")
        return cls(**{**data, "provenance_ids": tuple(data["provenance_ids"]),
                      "rejection_reasons": tuple(data["rejection_reasons"])})

    @classmethod
    def from_json(cls, payload):
        try:
            return cls.from_dict(json.loads(payload))
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Malformed ResearchDecisionTrace JSON: {exc}") from exc


@dataclass(frozen=True)
class ResearchDecision:
    proposal: ResearchActionProposal
    trace: ResearchDecisionTrace

    def __post_init__(self):
        if not isinstance(self.proposal, ResearchActionProposal) or not isinstance(self.trace, ResearchDecisionTrace):
            raise ValueError("ResearchDecision requires typed proposal and trace.")
        if self.proposal.action_type is not self.trace.action_type or self.proposal.rationale != self.trace.rationale:
            raise ValueError("Decision trace must describe its proposal.")

    def to_dict(self):
        return {"proposal": self.proposal.to_dict(), "trace": self.trace.to_dict()}

    def to_json(self):
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict) or set(data) != {"proposal", "trace"}:
            raise ValueError("Malformed ResearchDecision payload.")
        return cls(ResearchActionProposal.from_dict(data["proposal"]), ResearchDecisionTrace.from_dict(data["trace"]))

    @classmethod
    def from_json(cls, payload):
        try:
            return cls.from_dict(json.loads(payload))
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Malformed ResearchDecision JSON: {exc}") from exc


class ResearchDecisionGenerationError(ValueError):
    """Provider output failed the strict autonomous decision contract."""


@dataclass(frozen=True)
class ResearcherOverride:
    decision_id: str
    action: ResearcherOverrideAction
    researcher: str
    reason: str
    modified_proposal: Optional[ResearchActionProposal] = None
    created_at: str = field(default_factory=_now)

    def __post_init__(self):
        for name in ("decision_id", "researcher", "reason"):
            _text(getattr(self, name), name)
        object.__setattr__(self, "action", _enum(ResearcherOverrideAction, self.action, "override action"))
        if self.action is ResearcherOverrideAction.MODIFY:
            if not isinstance(self.modified_proposal, ResearchActionProposal):
                raise ValueError("MODIFY override requires a typed modified_proposal.")
        elif self.modified_proposal is not None:
            raise ValueError("Only MODIFY override may include modified_proposal.")
        _timestamp(self.created_at, "created_at")

    def to_dict(self):
        return {"decision_id": self.decision_id, "action": self.action.value, "researcher": self.researcher,
                "reason": self.reason, "modified_proposal": self.modified_proposal.to_dict() if self.modified_proposal else None,
                "created_at": self.created_at}


@dataclass(frozen=True)
class ResearcherOverrideResult:
    action: ResearcherOverrideAction
    proposal: Optional[ResearchActionProposal]
    executable_after_validation: bool
    note: str


def build_autonomous_research_context(
    plan: ResearchPlan, run: ResearchRun, review: PlanReview, loop: ResearchLoop,
    state: ResearchState, policy: BoundedExecutionPolicy,
) -> AutonomousResearchContext:
    """Build a bounded snapshot after deterministic lineage/authorization checks."""
    _validate_context_inputs(plan, run, review, loop, state, policy)
    authorized = tuple(x for x in (run.authorized_task_ids or ()) if x in set(policy.allowed_task_ids))
    tasks = tuple(ContextTask(x.task_id, x.description, x.question_id, x.category,
                              x.priority.value, x.status.value) for x in plan.tasks)
    questions = tuple(ContextQuestion(x.question_id, x.text, x.priority.value, x.status.value) for x in plan.questions)
    searches = {item.search_id: item for item in state.searches}
    results = {item.result_id: item for item in state.search_results}
    sources = {item.source_id for item in state.sources}
    versions = {item.version_id for item in state.document_versions}
    passages = {item.passage_reference_id for item in state.passage_references}
    state_evidence = {item.evidence_id: item for item in state.evidence}
    evidence_items = []
    provenance = set()
    for evidence_id, evidence in state_evidence.items():
        result = results.get(evidence.search_result_id) if evidence.search_result_id else None
        search = searches.get(result.search_id) if result else None
        if search is None or search.run_id != run.run_id:
            continue
        lineage_ids = [evidence_id, evidence.source_id, search.search_id, result.result_id]
        if evidence.document_version_id:
            lineage_ids.append(evidence.document_version_id)
        if evidence.passage_reference_id:
            lineage_ids.append(evidence.passage_reference_id)
        if evidence.source_id not in sources or (evidence.document_version_id and evidence.document_version_id not in versions) or (evidence.passage_reference_id and evidence.passage_reference_id not in passages):
            raise ValueError(f"Evidence {evidence_id!r} has unresolved provenance.")
        provenance.update(lineage_ids)
        evidence_items.append(ContextEvidence(evidence_id, evidence.text, evidence.source_id,
            evidence.document_version_id, evidence.passage_reference_id, evidence.search_result_id,
            evidence.text_hash, tuple(lineage_ids)))
    evidence_ids = {item.evidence_id for item in evidence_items}
    claims = tuple(ContextClaim(x.claim_id, x.text, x.status, tuple(i for i in x.evidence_ids if i in evidence_ids),
                                x.verification_reason)
                   for x in state.claims if any(i in evidence_ids for i in x.evidence_ids))
    artifacts = []
    for item in state.research_artifacts:
        if item.run_id != run.run_id:
            continue
        candidate_ids = tuple(x.candidate_id for x in item.improvement_candidates)
        epistemic_notes = tuple(
            f"{x.candidate_id}: status={x.status}; novelty={x.novelty_status}; validation={x.validation_status}; "
            f"proposed_change={x.proposed_change}"
            for x in item.improvement_candidates
        )
        summary = ""
        if item.artifact_type == "prior_work_investigation":
            summary = "; ".join(x.limitation for x in item.prior_work_findings if x.limitation)
        elif item.artifact_type == "verification_result":
            summary = "; ".join(x.explanation for x in item.verification_assessments if x.explanation)
        elif item.artifact_type == "improvement_candidate":
            summary = "; ".join(f"{x.title}: {x.proposed_change}" for x in item.improvement_candidates)
        elif item.artifact_type == "experiment_plan" and item.experiment_plan:
            summary = item.experiment_plan.objective
        artifacts.append(ContextArtifact(item.artifact_id, item.artifact_type, item.capability,
            item.task_id, item.evidence_ids, item.verification_claim, item.verification_verdict,
            candidate_ids, epistemic_notes, summary))
        provenance.add(item.artifact_id)
        provenance.update(item.evidence_ids)
    unresolved = tuple(x.description for x in loop.unresolved_work)
    hard_stops = []
    iterations_used = len(loop.loop_iterations)
    invocations_used = len(loop.capability_requests)
    executions_used = sum(x.run_id == run.run_id for x in state.task_executions)
    if iterations_used >= min(policy.maximum_iterations, loop.max_iterations):
        hard_stops.append("maximum_iterations_reached")
    if invocations_used >= policy.maximum_capability_invocations:
        hard_stops.append("maximum_capability_invocations_reached")
    if executions_used >= policy.maximum_task_executions:
        hard_stops.append("maximum_task_executions_reached")
    if loop.status in {ResearchLoopStatus.COMPLETED, ResearchLoopStatus.STOPPED, ResearchLoopStatus.FAILED}:
        hard_stops.append("terminal_loop")
    if loop.status is ResearchLoopStatus.WAITING_FOR_RESEARCHER:
        hard_stops.append("waiting_for_researcher")
    if not authorized:
        hard_stops.append("no_authorized_work")
    available_caps = tuple(x for x in policy.allowed_capabilities if capability_compatible_with_stage(loop.current_stage, x))
    budgets = BudgetSnapshot(min(policy.maximum_iterations, loop.max_iterations), iterations_used,
        policy.maximum_capability_invocations, invocations_used, policy.maximum_task_executions,
        executions_used, tuple(hard_stops))
    snapshot = {
        "run_id": run.run_id, "plan_id": plan.plan_id, "plan_revision": plan.revision,
        "approval_review_id": review.review_id, "run_status": run.status,
        "loop_id": loop.loop_id, "current_task_id": loop.current_task_id,
        "iteration_id": loop.current_iteration_id,
        "iteration_number": loop.iteration_number, "stage": loop.current_stage.value,
        "loop_status": loop.status.value, "domain": plan.request.domain,
        "objective": plan.objective.text, "tasks": [x.to_dict() for x in tasks],
        "questions": [x.to_dict() for x in questions], "authorized_task_ids": list(authorized),
        "authorized_capabilities": [x.value for x in available_caps],
        "evidence": [x.to_dict() for x in evidence_items], "claims": [x.to_dict() for x in claims],
        "artifacts": [x.to_dict() for x in artifacts], "budgets": budgets.to_dict(),
        "unresolved_questions": list(state.unresolved_questions), "unresolved_work": list(unresolved),
        "unresolved_contradictions": list(state.contradictions),
        "completed_task_ids": list(loop.completed_task_ids),
        "failed_task_ids": sorted({x.task_id for x in state.task_executions
                                    if x.run_id == run.run_id and x.status.value == "FAILED"}),
        "provenance_ids": sorted(provenance),
    }
    context_id = "context_" + hashlib.sha256(json.dumps(snapshot, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:20]
    return AutonomousResearchContext(context_id, run.run_id, plan.plan_id, plan.revision, loop.loop_id,
        loop.current_task_id, loop.current_iteration_id, loop.iteration_number, loop.current_stage, loop.status,
        plan.objective.text, plan.request.domain, questions, tasks, authorized, available_caps,
        tuple(evidence_items), claims, tuple(artifacts), tuple(state.unresolved_questions),
        tuple(state.contradictions), unresolved, tuple(loop.completed_task_ids),
        tuple(sorted({x.task_id for x in state.task_executions
                      if x.run_id == run.run_id and x.status.value == "FAILED"})),
        tuple(sorted(provenance)), budgets)


def autonomous_decide(context: AutonomousResearchContext, provider: LLMProvider) -> ResearchDecision:
    """Ask one provider for one typed proposal; never execute or persist it."""
    if not isinstance(context, AutonomousResearchContext):
        raise ValueError("A typed AutonomousResearchContext is required.")
    if not callable(getattr(provider, "generate", None)):
        raise ValueError("An explicit LLMProvider is required.")
    if "terminal_loop" in context.budgets.hard_stop_reasons:
        raise ResearchDecisionGenerationError("A terminal ResearchLoop cannot receive a new decision.")
    if "waiting_for_researcher" in context.budgets.hard_stop_reasons:
        raise ResearchDecisionGenerationError("A researcher decision is required before autonomy can continue.")
    decision_id = _id("decision")
    provider_name = type(provider).__name__
    model = getattr(provider, "model", None)
    if context.budgets.hard_stop_reasons and "terminal_loop" not in context.budgets.hard_stop_reasons:
        assessment = ResearchStopAssessment(
            stop_reason=context.budgets.hard_stop_reasons[0],
            evidence_summary="A deterministic loop bound or control gate prevents further work.",
            unresolved_questions=context.unresolved_questions,
            remaining_known_limitations=context.unresolved_work,
            assessment="System hard bounds override any model preference.",
        )
        proposal = ResearchActionProposal(ResearchActionType.STOP,
            "A deterministic safety or budget boundary prevents additional work.",
            estimated_information_gain=0.0, priority="critical", stop_assessment=assessment)
    else:
        prompt = _decision_prompt(context)
        try:
            raw = provider.generate([{"role": "system", "content": _SYSTEM_PROMPT},
                                     {"role": "user", "content": prompt}], temperature=0.0)
            payload = json.loads(raw)
            proposal = _proposal_from_provider(payload)
        except Exception as exc:
            raise ResearchDecisionGenerationError(f"Autonomous decision generation failed: {type(exc).__name__}: {exc}") from exc
    trace = ResearchDecisionTrace(decision_id, context.context_id, context.loop_id, context.run_id,
        context.iteration_id, proposal.action_type, proposal.rationale, provider_name, model, _now(),
        proposal.provenance_ids)
    return ResearchDecision(proposal, trace)


_SYSTEM_PROMPT = """You propose exactly one action for a controlled research system. Return only a JSON object matching the user schema. Never call tools, mutate state, claim authorization, or claim execution. Preserve epistemic distinctions: a candidate gap is not proven, a candidate improvement is not validated, a hypothesis is not fact, and insufficient evidence is not support. Choose only the closed action types in the schema. Existing stored IDs must be copied exactly from context; do not invent IDs. estimated_information_gain is a subjective estimate, not a measured metric. STOP is a recommendation, not a scientific conclusion."""


def _decision_prompt(context):
    schema = {
        "action_type": [x.value for x in ResearchActionType],
        "required_fields": ["action_type", "rationale", "task_id", "question_ids", "claim_ids", "evidence_ids",
            "artifact_ids", "query", "problem_statement", "candidate_id", "estimated_information_gain",
            "priority", "unresolved_questions_addressed", "stop_after_action", "stop_assessment", "provenance_ids"],
        "stop_assessment": {"stop_reason": "text", "evidence_summary": "text", "unresolved_questions": [],
                            "remaining_known_limitations": [], "assessment": "text"},
    }
    return "Return exactly one JSON object with these exact keys and types:\n" + json.dumps(schema, sort_keys=True) + \
        "\nContext (read-only):\n" + context.to_json()


def _proposal_from_provider(data):
    keys = {"action_type", "rationale", "task_id", "question_ids", "claim_ids", "evidence_ids", "artifact_ids",
            "query", "problem_statement", "candidate_id", "estimated_information_gain", "priority",
            "unresolved_questions_addressed", "stop_after_action", "stop_assessment", "provenance_ids"}
    if not isinstance(data, dict) or set(data) != keys:
        raise ValueError("Provider output must be a JSON object with exactly the proposal schema keys.")
    return ResearchActionProposal.from_dict({**data, "required_capabilities": [
        _ACTION_CAPABILITY[ResearchActionType(data["action_type"])].value
    ] if data["action_type"] in {x.value for x in _ACTION_CAPABILITY} else [],
        "required_authorizations": _expected_authorizations(data)})


def _expected_authorizations(data):
    action = ResearchActionType(data["action_type"])
    required = ["approved_exact_plan_revision", "registered_research_run"]
    if action in _ACTION_CAPABILITY:
        required.append("explicitly_authorized_task")
    if data["evidence_ids"] or data["claim_ids"] or data["artifact_ids"]:
        required.append("stored_reference_resolution")
    if action is ResearchActionType.RETURN_TO_PLANNING:
        required.append("explicit_researcher_acceptance")
    return required


def validate_research_decision(
    decision: ResearchDecision, *, plan: ResearchPlan, run: ResearchRun, review: PlanReview,
    loop: ResearchLoop, state: ResearchState, policy: BoundedExecutionPolicy,
    researcher_override: Optional[ResearcherOverride] = None,
) -> ResearchDecisionTrace:
    """Deterministically validate one proposal, returning an auditable trace.

    This function is read-only.  A rejected result contains all detected
    reasons; it never repairs a proposal or dispatches a capability.
    """
    if not isinstance(decision, ResearchDecision):
        raise ValueError("A typed ResearchDecision is required.")
    reasons = []
    try:
        _validate_context_inputs(plan, run, review, loop, state, policy)
        current = build_autonomous_research_context(plan, run, review, loop, state, policy)
        if decision.trace.context_id != current.context_id:
            reasons.append("decision context is stale or belongs to different state")
        if (decision.trace.loop_id != loop.loop_id or decision.trace.run_id != run.run_id
                or decision.trace.iteration_id != loop.current_iteration_id):
            reasons.append("decision loop/run/iteration references do not match current state")
        if loop.status in {ResearchLoopStatus.COMPLETED, ResearchLoopStatus.STOPPED, ResearchLoopStatus.FAILED}:
            reasons.append("terminal research loop rejects new decisions")
        proposal = decision.proposal
        if loop.status is ResearchLoopStatus.WAITING_FOR_RESEARCHER and proposal.action_type not in {
            ResearchActionType.REQUEST_RESEARCHER_REVIEW, ResearchActionType.STOP,
            ResearchActionType.RETURN_TO_PLANNING,
        }:
            reasons.append("researcher decision is required before further work")
        hard = set(current.budgets.hard_stop_reasons)
        nonterminal_hard = hard - {"terminal_loop", "waiting_for_researcher"}
        if nonterminal_hard and proposal.action_type is not ResearchActionType.STOP:
            reasons.append("deterministic hard budget requires STOP")
        if proposal.action_type is ResearchActionType.STOP and nonterminal_hard:
            if proposal.stop_assessment.stop_reason not in nonterminal_hard:
                reasons.append("STOP reason does not match the active deterministic bound")
        if proposal.task_id:
            task = next((x for x in plan.tasks if x.task_id == proposal.task_id), None)
            if task is None:
                reasons.append("selected task does not exist in the exact plan")
            if proposal.task_id not in (run.authorized_task_ids or ()):
                reasons.append("selected task is not authorized for this ResearchRun")
            if proposal.task_id not in policy.allowed_task_ids:
                reasons.append("selected task is outside the caller's execution policy")
        expected_capability = _ACTION_CAPABILITY.get(proposal.action_type)
        if expected_capability:
            if expected_capability not in policy.allowed_capabilities:
                reasons.append("required capability is outside the caller's allowlist")
            if expected_capability not in current.authorized_capabilities:
                reasons.append("required capability is unavailable at the current stage")
            if not capability_compatible_with_stage(loop.current_stage, expected_capability):
                reasons.append("current stage does not permit the proposed capability")
        elif proposal.action_type in {ResearchActionType.SYNTHESIZE, ResearchActionType.CRITIQUE}:
            permitted = LoopStage.SYNTHESIS if proposal.action_type is ResearchActionType.SYNTHESIZE else LoopStage.CRITIQUE
            if loop.current_stage is not permitted or loop.current_stage not in policy.allowed_stages:
                reasons.append("current stage/policy does not permit this analysis action")
        elif proposal.action_type is ResearchActionType.RETURN_TO_PLANNING:
            if loop.current_stage is not LoopStage.RESEARCHER_REVIEW:
                reasons.append("RETURN_TO_PLANNING requires the researcher-review stage")
            if not _override_matches(researcher_override, decision, ResearcherOverrideAction.ACCEPT):
                reasons.append("RETURN_TO_PLANNING requires explicit researcher ACCEPT")
        elif proposal.action_type is ResearchActionType.REQUEST_RESEARCHER_REVIEW:
            if loop.current_stage is LoopStage.RESEARCHER_REVIEW:
                reasons.append("loop is already waiting at the researcher-review gate")
            elif LoopStage.RESEARCHER_REVIEW not in LEGAL_TRANSITIONS[loop.current_stage]:
                reasons.append("current stage cannot enter the researcher-review gate")
        task_question_ids = {x.question_id for x in plan.questions}
        task_ids = {x.task_id for x in plan.tasks}
        question_ids = {x.question_id for x in state.research_questions if x.run_id == run.run_id}
        known_questions = task_question_ids | question_ids
        if not set(proposal.question_ids).issubset(known_questions):
            reasons.append("proposal references an unknown or foreign question")
        claim_ids = {x.claim_id for x in state.claims if set(x.evidence_ids) & {e.evidence_id for e in current.evidence}}
        if not set(proposal.claim_ids).issubset(claim_ids):
            reasons.append("proposal references an unknown claim or a claim outside this run's evidence")
        evidence_ids = {x.evidence_id for x in current.evidence}
        if not set(proposal.evidence_ids).issubset(evidence_ids):
            reasons.append("proposal references evidence outside the run's stored provenance")
        artifact_ids = {x.artifact_id for x in current.artifacts}
        if not set(proposal.artifact_ids).issubset(artifact_ids):
            reasons.append("proposal references an unknown or foreign artifact")
        if proposal.action_type in {ResearchActionType.INVESTIGATE_PRIOR_WORK, ResearchActionType.PLAN_EXPERIMENT}:
            candidates = {candidate for artifact in current.artifacts for candidate in artifact.candidate_ids}
            if proposal.candidate_id not in candidates:
                reasons.append("candidate_id does not resolve to a stored candidate in this run")
            candidate_artifacts = {artifact.artifact_id for artifact in current.artifacts
                                   if proposal.candidate_id in artifact.candidate_ids}
            if not candidate_artifacts.intersection(proposal.artifact_ids):
                reasons.append("candidate action must reference its stored candidate artifact")
        valid_provenance = set(current.provenance_ids)
        if not set(proposal.provenance_ids).issubset(valid_provenance):
            reasons.append("proposal contains unknown provenance references")
        if proposal.action_type is ResearchActionType.VERIFY_CLAIM:
            linked = {e for claim in current.claims if claim.claim_id in proposal.claim_ids for e in claim.evidence_ids}
            if not set(proposal.evidence_ids).issubset(linked):
                reasons.append("verification evidence must be linked to the selected stored claim")
        if proposal.action_type is ResearchActionType.GENERATE_IMPROVEMENT and not set(proposal.evidence_ids).issubset(evidence_ids):
            reasons.append("improvement evidence references must resolve in state")
        if researcher_override is not None:
            if researcher_override.decision_id != decision.trace.decision_id:
                reasons.append("researcher override references a different decision")
            elif researcher_override.action is ResearcherOverrideAction.REJECT:
                reasons.append("researcher explicitly rejected the proposal")
            elif researcher_override.action is ResearcherOverrideAction.STOP and proposal.action_type is not ResearchActionType.STOP:
                reasons.append("researcher STOP override requires a STOP proposal")
            elif researcher_override.action is ResearcherOverrideAction.MODIFY and researcher_override.modified_proposal != proposal:
                reasons.append("modified proposal must be reissued and validated as a new decision")
    except (AttributeError, TypeError, ValueError) as exc:
        reasons.append(f"deterministic context validation failed: {exc}")
    status = ValidationStatus.REJECTED if reasons else ValidationStatus.ACCEPTED
    return ResearchDecisionTrace(decision.trace.decision_id, decision.trace.context_id,
        decision.trace.loop_id, decision.trace.run_id, decision.trace.iteration_id,
        decision.proposal.action_type, decision.proposal.rationale, decision.trace.provider,
        decision.trace.model, decision.trace.created_at, decision.proposal.provenance_ids,
        status, tuple(dict.fromkeys(reasons)))


def apply_researcher_override(decision: ResearchDecision, override: ResearcherOverride) -> ResearcherOverrideResult:
    """Represent a human response without mutating state or executing work."""
    if not isinstance(decision, ResearchDecision) or not isinstance(override, ResearcherOverride):
        raise ValueError("Typed decision and researcher override are required.")
    if override.decision_id != decision.trace.decision_id:
        raise ValueError("ResearcherOverride refers to another decision.")
    if override.action is ResearcherOverrideAction.ACCEPT:
        return ResearcherOverrideResult(override.action, decision.proposal, True,
                                        "Accepted proposal still requires deterministic validation.")
    if override.action is ResearcherOverrideAction.MODIFY:
        return ResearcherOverrideResult(override.action, override.modified_proposal, True,
                                        "Modified proposal must be issued as a new decision and revalidated.")
    if override.action is ResearcherOverrideAction.STOP:
        return ResearcherOverrideResult(override.action, None, False,
                                        "Researcher requested stop; caller must apply it through controlled loop policy.")
    return ResearcherOverrideResult(override.action, None, False,
                                    "Proposal rejected by researcher; no execution is authorized.")


def _override_matches(override, decision, action):
    return isinstance(override, ResearcherOverride) and override.decision_id == decision.trace.decision_id and override.action is action


def _validate_context_inputs(plan, run, review, loop, state, policy):
    if not isinstance(plan, ResearchPlan) or plan.validate():
        raise ValueError("A structurally valid ResearchPlan is required.")
    if not isinstance(run, ResearchRun) or not isinstance(review, PlanReview):
        raise ValueError("A typed ResearchRun and PlanReview are required.")
    if not isinstance(loop, ResearchLoop) or not isinstance(state, ResearchState):
        raise ValueError("A typed ResearchLoop and ResearchState are required.")
    if not isinstance(policy, BoundedExecutionPolicy):
        raise ValueError("A typed BoundedExecutionPolicy is required.")
    review.validate_against(plan)
    if not is_execution_approved(plan, review):
        raise ValueError("PlanReview does not approve this exact plan revision.")
    if run.plan_id != plan.plan_id or run.plan_revision != plan.revision or run.approval_review_id != review.review_id:
        raise ValueError("Run approval lineage does not match the exact approved plan revision.")
    if loop.run_id != run.run_id or loop.plan_id != plan.plan_id or loop.plan_revision != plan.revision:
        raise ValueError("ResearchLoop does not match the approved plan and run.")
    if not any(item is run for item in state.research_runs):
        raise ValueError("ResearchRun must be registered in ResearchState.")
    loops = [item for item in state.research_loops if item.loop_id == loop.loop_id]
    if len(loops) != 1 or loops[0] != loop:
        raise ValueError("ResearchLoop must resolve uniquely in ResearchState.")
    if loop.validate_references(state):
        raise ValueError("ResearchLoop references are invalid.")
    if state.validate_lineage():
        raise ValueError("ResearchState lineage is invalid.")
    if run.status not in {"created", "running"}:
        raise ValueError("ResearchRun status does not allow an autonomous decision.")
    task_ids = {task.task_id for task in plan.tasks}
    authorized = set(run.authorized_task_ids or ())
    if not authorized or not authorized.issubset(task_ids):
        raise ValueError("ResearchRun has no valid explicitly authorized tasks.")
    if not set(policy.allowed_task_ids).issubset(authorized):
        raise ValueError("Caller policy contains tasks not authorized for the run.")
    if policy.maximum_iterations > loop.max_iterations:
        raise ValueError("Caller policy cannot expand the ResearchLoop iteration bound.")
    if any(stage not in policy.allowed_stages for stage in (loop.current_stage,)):
        raise ValueError("Current ResearchLoop stage is outside caller policy.")


__all__ = ["ResearchActionType", "ResearcherOverrideAction", "ValidationStatus", "ContextTask", "ContextQuestion",
            "ContextEvidence", "ContextClaim", "ContextArtifact", "BudgetSnapshot", "AutonomousResearchContext",
            "ResearchStopAssessment", "ResearchActionProposal", "ResearchDecisionTrace", "ResearchDecision",
            "ResearchDecisionGenerationError", "ResearcherOverride", "ResearcherOverrideResult",
            "build_autonomous_research_context", "autonomous_decide", "validate_research_decision",
            "apply_researcher_override"]
