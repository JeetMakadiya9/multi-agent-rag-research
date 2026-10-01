"""Structured research state for the Multi-Agent RAG system.

This module is intentionally independent of LangGraph and Streamlit.  It gives
all future agents a common, serializable state object while keeping the core
research data model deterministic and testable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
import hashlib
import unicodedata
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Literal, Optional
from uuid import uuid4

if TYPE_CHECKING:
    from research_tools import SearchResult
    from research_task_execution import ResearchTaskExecution
    from research_loop import ResearchLoop


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:10]}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _passage_text_hash(text: str) -> str:
    normalized = unicodedata.normalize("NFC", " ".join(text.split()))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


@dataclass
class ResearchQuestion:
    text: str
    priority: int = 1
    status: str = "open"
    question_id: str = field(default_factory=lambda: _id("rq"))
    created_at: str = field(default_factory=_now)
    # Optional, appended lineage fields preserve legacy positional calls.
    run_id: Optional[str] = None
    iteration_id: Optional[str] = None
    parent_question_id: Optional[str] = None


@dataclass
class ResearchRun:
    """Identity and lifecycle status for one research execution."""

    run_id: str = field(default_factory=lambda: _id("run"))
    created_at: str = field(default_factory=_now)
    status: Literal["created", "running", "completed", "failed"] = "created"
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Optional plan authorization linkage; appended to preserve old positional calls.
    plan_id: Optional[str] = None
    plan_revision: Optional[int] = None
    approval_review_id: Optional[str] = None
    # None means no task authorization has been recorded (including legacy
    # runs). A populated list is the explicit task selection for this run.
    authorized_task_ids: Optional[List[str]] = None

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("ResearchRun requires run_id.")
        if self.status not in {"created", "running", "completed", "failed"}:
            raise ValueError(f"Unsupported ResearchRun status: {self.status!r}.")
        authorization_fields = (self.plan_id, self.plan_revision, self.approval_review_id)
        if any(value is not None for value in authorization_fields) and not all(
            value is not None for value in authorization_fields
        ):
            raise ValueError(
                "ResearchRun plan authorization must provide plan_id, plan_revision, "
                "and approval_review_id together."
            )
        if self.plan_id is not None and (
            not isinstance(self.plan_id, str) or not self.plan_id.strip()
        ):
            raise ValueError("ResearchRun plan_id must be a non-empty string when supplied.")
        if self.approval_review_id is not None and (
            not isinstance(self.approval_review_id, str)
            or not self.approval_review_id.strip()
        ):
            raise ValueError(
                "ResearchRun approval_review_id must be a non-empty string when supplied."
            )
        if self.plan_revision is not None and (
            not isinstance(self.plan_revision, int)
            or isinstance(self.plan_revision, bool)
            or self.plan_revision < 1
        ):
            raise ValueError("ResearchRun plan_revision must be a positive integer when supplied.")
        if self.authorized_task_ids is not None:
            if not isinstance(self.authorized_task_ids, list) or not self.authorized_task_ids:
                raise ValueError("ResearchRun authorized_task_ids must be a non-empty list when supplied.")
            if any(not isinstance(task_id, str) or not task_id.strip() for task_id in self.authorized_task_ids):
                raise ValueError("ResearchRun authorized_task_ids must contain non-empty strings.")
            if len(set(self.authorized_task_ids)) != len(self.authorized_task_ids):
                raise ValueError("ResearchRun authorized_task_ids must not contain duplicates.")


@dataclass
class ResearchIteration:
    """One numbered research cycle within a ResearchRun."""

    run_id: str
    number: int
    iteration_id: str = field(default_factory=lambda: _id("iteration"))
    created_at: str = field(default_factory=_now)
    status: Literal["created", "running", "completed", "failed"] = "created"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("ResearchIteration requires run_id.")
        if not self.iteration_id:
            raise ValueError("ResearchIteration requires iteration_id.")
        if not isinstance(self.number, int) or isinstance(self.number, bool) or self.number < 1:
            raise ValueError("ResearchIteration number must be a positive integer.")
        if self.status not in {"created", "running", "completed", "failed"}:
            raise ValueError(f"Unsupported ResearchIteration status: {self.status!r}.")


@dataclass
class Source:
    title: str
    url: str = ""
    source_type: str = "unknown"
    published_at: Optional[str] = None
    author: Optional[str] = None
    source_id: str = field(default_factory=lambda: _id("src"))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentVersion:
    """A captured/versioned representation of a logical document."""

    document_id: str
    source_id: str
    version_id: str = field(default_factory=lambda: _id("version"))
    captured_at: str = field(default_factory=_now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.document_id:
            raise ValueError("DocumentVersion requires document_id.")
        if not self.source_id:
            raise ValueError("DocumentVersion requires source_id.")
        if not self.version_id:
            raise ValueError("DocumentVersion requires version_id.")


@dataclass
class PassageReference:
    """A typed locator inside one captured document version."""

    document_version_id: str
    locator_type: Literal["page", "chunk", "sentence", "span"]
    page: Optional[int] = None
    chunk_id: Optional[str] = None
    sentence_index: Optional[int] = None
    section: Optional[str] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    passage_reference_id: str = field(default_factory=lambda: _id("passage"))

    def __post_init__(self) -> None:
        if not self.document_version_id:
            raise ValueError("PassageReference requires document_version_id.")
        if not self.passage_reference_id:
            raise ValueError("PassageReference requires passage_reference_id.")
        if self.chunk_id is not None and (
            not isinstance(self.chunk_id, str) or not self.chunk_id.strip()
        ):
            raise ValueError("PassageReference chunk_id must be a non-empty string when supplied.")

        if self.locator_type == "page":
            if not isinstance(self.page, int) or isinstance(self.page, bool) or self.page < 1:
                raise ValueError("Page locator requires a 1-based page number.")
            self._validate_optional_span()
            if self.sentence_index is not None:
                raise ValueError("Page locator cannot include sentence_index.")
        elif self.locator_type == "chunk":
            if not isinstance(self.chunk_id, str) or not self.chunk_id.strip():
                raise ValueError("Chunk locator requires chunk_id.")
            if self.page is not None and (
                not isinstance(self.page, int) or isinstance(self.page, bool) or self.page < 1
            ):
                raise ValueError("Chunk locator page must be 1-based when supplied.")
            if self.sentence_index is not None or self.start_char is not None or self.end_char is not None:
                raise ValueError("Chunk locator cannot include sentence or character-span fields.")
        elif self.locator_type == "sentence":
            if (
                not isinstance(self.sentence_index, int)
                or isinstance(self.sentence_index, bool)
                or self.sentence_index < 0
            ):
                raise ValueError("Sentence locator requires a zero-based sentence_index.")
            if any(value is not None for value in (self.page, self.chunk_id, self.start_char, self.end_char)):
                raise ValueError("Sentence locator cannot include page, chunk, or character-span fields.")
        elif self.locator_type == "span":
            if (
                not isinstance(self.start_char, int)
                or isinstance(self.start_char, bool)
                or not isinstance(self.end_char, int)
                or isinstance(self.end_char, bool)
            ):
                raise ValueError("Span locator requires start_char and end_char.")
            if self.start_char < 0 or self.end_char <= self.start_char:
                raise ValueError("Span locator requires 0 <= start_char < end_char.")
            if self.page is not None and (
                not isinstance(self.page, int) or isinstance(self.page, bool) or self.page < 1
            ):
                raise ValueError("Span locator page must be 1-based when supplied.")
            if self.chunk_id is not None or self.sentence_index is not None:
                raise ValueError("Span locator cannot include chunk_id or sentence_index.")
        else:
            raise ValueError(f"Unsupported passage locator_type: {self.locator_type!r}")

    def _validate_optional_span(self) -> None:
        if (self.start_char is None) != (self.end_char is None):
            raise ValueError("Page locator character span must provide both start_char and end_char.")
        if self.start_char is not None and (
            not isinstance(self.start_char, int)
            or isinstance(self.start_char, bool)
            or not isinstance(self.end_char, int)
            or isinstance(self.end_char, bool)
            or self.start_char < 0
            or self.end_char <= self.start_char
        ):
            raise ValueError("Page locator requires 0 <= start_char < end_char.")


@dataclass
class Evidence:
    text: str
    source_id: str
    relevance_score: Optional[float] = None
    page: Optional[int] = None
    chunk_id: Optional[str] = None
    evidence_id: str = field(default_factory=lambda: _id("ev"))
    metadata: Dict[str, Any] = field(default_factory=dict)
    document_version_id: Optional[str] = None
    passage_reference_id: Optional[str] = None
    text_hash: Optional[str] = None
    # Optional bridge to a persisted search result; legacy and RAG evidence may omit it.
    search_result_id: Optional[str] = None

    def __post_init__(self) -> None:
        if (
            self.document_version_id is not None
            and self.passage_reference_id is not None
            and self.text_hash is None
        ):
            self.text_hash = _passage_text_hash(self.text)


@dataclass
class ResearchClaim:
    text: str
    status: str = "unverified"
    confidence: Optional[float] = None
    claim_id: str = field(default_factory=lambda: _id("cl"))
    evidence_ids: List[str] = field(default_factory=list)
    source_ids: List[str] = field(default_factory=list)
    verification_reason: str = ""


@dataclass
class SearchAction:
    query: str
    provider: str = "unknown"
    result_count: int = 0
    purpose: str = "research"
    search_id: str = field(default_factory=lambda: _id("search"))
    created_at: str = field(default_factory=_now)
    # Optional, appended lineage fields preserve legacy positional calls.
    run_id: Optional[str] = None
    iteration_id: Optional[str] = None
    question_id: Optional[str] = None


@dataclass
class AgentEvent:
    agent: str
    action: str
    message: str = ""
    event_id: str = field(default_factory=lambda: _id("evt"))
    created_at: str = field(default_factory=_now)


@dataclass(frozen=True)
class VerificationEvidenceAssessment:
    """Transient Experiment B stance mapped back to an existing Evidence ID."""

    evidence_id: str
    component_ids: tuple[str, ...]
    stance: Literal["SUPPORTS", "CONTRADICTS", "NEUTRAL"]

    def __post_init__(self) -> None:
        if not isinstance(self.evidence_id, str) or not self.evidence_id.strip():
            raise ValueError("VerificationEvidenceAssessment.evidence_id must be non-empty.")
        values = tuple(self.component_ids)
        if not values or any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError("VerificationEvidenceAssessment.component_ids must contain non-empty IDs.")
        if len(values) != len(set(values)):
            raise ValueError("VerificationEvidenceAssessment.component_ids must not contain duplicates.")
        object.__setattr__(self, "component_ids", values)
        if self.stance not in {"SUPPORTS", "CONTRADICTS", "NEUTRAL"}:
            raise ValueError("VerificationEvidenceAssessment.stance is invalid.")

    def to_dict(self) -> Dict[str, Any]:
        return {"evidence_id": self.evidence_id, "component_ids": list(self.component_ids), "stance": self.stance}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VerificationEvidenceAssessment":
        if not isinstance(data, dict) or set(data) != {"evidence_id", "component_ids", "stance"}:
            raise ValueError("Malformed VerificationEvidenceAssessment payload.")
        return cls(**data)


@dataclass(frozen=True)
class VerificationUnitAssessment:
    """Typed, non-persistent Experiment B assessment for one claim unit."""

    claim_id: str
    claim: str
    verdict: str
    confidence: float
    explanation: str
    evidence_assessments: tuple[VerificationEvidenceAssessment, ...] = ()
    validator_overrides: tuple[str, ...] = ()
    assessment_complete: bool = True
    missing_assessments: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("claim_id", "claim", "verdict"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"VerificationUnitAssessment.{name} must be non-empty.")
        if not isinstance(self.confidence, (int, float)) or isinstance(self.confidence, bool):
            raise ValueError("VerificationUnitAssessment.confidence must be numeric.")
        if not isinstance(self.explanation, str):
            raise ValueError("VerificationUnitAssessment.explanation must be a string.")
        assessments = tuple(self.evidence_assessments)
        if any(not isinstance(item, VerificationEvidenceAssessment) for item in assessments):
            raise ValueError("VerificationUnitAssessment.evidence_assessments must be typed assessments.")
        if not isinstance(self.assessment_complete, bool):
            raise ValueError("VerificationUnitAssessment.assessment_complete must be a boolean.")
        for name in ("validator_overrides", "missing_assessments"):
            values = tuple(getattr(self, name))
            if any(not isinstance(value, str) for value in values):
                raise ValueError(f"VerificationUnitAssessment.{name} must contain strings.")
            object.__setattr__(self, name, values)
        object.__setattr__(self, "evidence_assessments", assessments)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "claim": self.claim,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "evidence_assessments": [item.to_dict() for item in self.evidence_assessments],
            "validator_overrides": list(self.validator_overrides),
            "assessment_complete": self.assessment_complete,
            "missing_assessments": list(self.missing_assessments),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VerificationUnitAssessment":
        required = {
            "claim_id", "claim", "verdict", "confidence", "explanation", "evidence_assessments",
            "validator_overrides", "assessment_complete", "missing_assessments",
        }
        if not isinstance(data, dict) or set(data) != required:
            raise ValueError("Malformed VerificationUnitAssessment payload.")
        values = dict(data)
        if not isinstance(values["evidence_assessments"], (list, tuple)):
            raise ValueError("Malformed VerificationUnitAssessment evidence_assessments.")
        values["evidence_assessments"] = tuple(
            VerificationEvidenceAssessment.from_dict(item) for item in values["evidence_assessments"]
        )
        values["validator_overrides"] = tuple(values["validator_overrides"])
        values["missing_assessments"] = tuple(values["missing_assessments"])
        return cls(**values)


@dataclass(frozen=True)
class ResearchImprovementCandidate:
    """System-generated proposal, explicitly not a verified or novel finding."""

    title: str
    problem_statement: str
    motivation: str
    proposed_change: str
    rationale: str
    expected_benefit: str
    assumptions: tuple[str, ...]
    risks: tuple[str, ...]
    validation_needed: tuple[str, ...]
    supporting_evidence_ids: tuple[str, ...]
    supporting_claim_ids: tuple[str, ...] = ()
    supporting_verification_artifact_ids: tuple[str, ...] = ()
    candidate_id: str = field(default_factory=lambda: _id("candidate"))
    status: Literal["CANDIDATE"] = "CANDIDATE"
    problem_origin: Literal["researcher_provided"] = "researcher_provided"
    generation_origin: Literal["system_generated"] = "system_generated"
    novelty_status: Literal["NOT_ASSESSED"] = "NOT_ASSESSED"
    validation_status: Literal["NOT_VALIDATED"] = "NOT_VALIDATED"
    generated_at: str = field(default_factory=_now)

    def __post_init__(self) -> None:
        for name in (
            "candidate_id", "title", "problem_statement", "motivation", "proposed_change",
            "rationale", "expected_benefit",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"ResearchImprovementCandidate.{name} must be non-empty.")
        if self.status != "CANDIDATE" or self.problem_origin != "researcher_provided":
            raise ValueError("ResearchImprovementCandidate epistemic origin/status is invalid.")
        if self.generation_origin != "system_generated":
            raise ValueError("ResearchImprovementCandidate.generation_origin must be system_generated.")
        if self.novelty_status != "NOT_ASSESSED" or self.validation_status != "NOT_VALIDATED":
            raise ValueError("Generated candidates cannot be marked novel or validated.")
        for name in (
            "assumptions", "risks", "validation_needed", "supporting_evidence_ids",
            "supporting_claim_ids", "supporting_verification_artifact_ids",
        ):
            values = tuple(getattr(self, name))
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise ValueError(f"ResearchImprovementCandidate.{name} must contain non-empty strings.")
            if len(values) != len(set(values)):
                raise ValueError(f"ResearchImprovementCandidate.{name} must not contain duplicates.")
            object.__setattr__(self, name, values)
        if not self.assumptions or not self.risks or not self.validation_needed or not self.supporting_evidence_ids:
            raise ValueError("Candidate requires assumptions, risks, validation needs, and supporting Evidence IDs.")
        if not isinstance(self.generated_at, str) or not self.generated_at.strip():
            raise ValueError("ResearchImprovementCandidate.generated_at must be non-empty.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "problem_statement": self.problem_statement,
            "motivation": self.motivation,
            "proposed_change": self.proposed_change,
            "rationale": self.rationale,
            "expected_benefit": self.expected_benefit,
            "assumptions": list(self.assumptions),
            "risks": list(self.risks),
            "validation_needed": list(self.validation_needed),
            "supporting_evidence_ids": list(self.supporting_evidence_ids),
            "supporting_claim_ids": list(self.supporting_claim_ids),
            "supporting_verification_artifact_ids": list(self.supporting_verification_artifact_ids),
            "candidate_id": self.candidate_id,
            "status": self.status,
            "problem_origin": self.problem_origin,
            "generation_origin": self.generation_origin,
            "novelty_status": self.novelty_status,
            "validation_status": self.validation_status,
            "generated_at": self.generated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResearchImprovementCandidate":
        required = {
            "title", "problem_statement", "motivation", "proposed_change", "rationale", "expected_benefit",
            "assumptions", "risks", "validation_needed", "supporting_evidence_ids", "supporting_claim_ids",
            "supporting_verification_artifact_ids", "candidate_id", "status", "problem_origin", "generation_origin",
            "novelty_status", "validation_status", "generated_at",
        }
        if not isinstance(data, dict) or set(data) != required:
            raise ValueError("Malformed ResearchImprovementCandidate payload.")
        values = dict(data)
        for name in (
            "assumptions", "risks", "validation_needed", "supporting_evidence_ids",
            "supporting_claim_ids", "supporting_verification_artifact_ids",
        ):
            if not isinstance(values[name], (list, tuple)):
                raise ValueError(f"Malformed ResearchImprovementCandidate.{name}.")
            values[name] = tuple(values[name])
        return cls(**values)


@dataclass(frozen=True)
class ExperimentDesignElement:
    """A proposed control/ablation and why it may isolate an effect."""

    description: str
    rationale: str

    def __post_init__(self) -> None:
        for name in ("description", "rationale"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"ExperimentDesignElement.{name} must be non-empty text.")

    def to_dict(self) -> Dict[str, str]:
        return {"description": self.description, "rationale": self.rationale}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentDesignElement":
        if not isinstance(data, dict) or set(data) != {"description", "rationale"}:
            raise ValueError("Malformed ExperimentDesignElement payload.")
        return cls(**data)


@dataclass(frozen=True)
class ExperimentMetric:
    """A proposed measurement, not a measured result or guaranteed-suitable metric."""

    name: str
    measures: str
    rationale: str
    suitability: Literal["REQUIRES_RESEARCHER_REVIEW"] = "REQUIRES_RESEARCHER_REVIEW"

    def __post_init__(self) -> None:
        for name in ("name", "measures", "rationale"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"ExperimentMetric.{name} must be non-empty text.")
        if self.suitability != "REQUIRES_RESEARCHER_REVIEW":
            raise ValueError("ExperimentMetric suitability must remain subject to researcher review.")

    def to_dict(self) -> Dict[str, str]:
        return {"name": self.name, "measures": self.measures, "rationale": self.rationale,
                "suitability": self.suitability}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentMetric":
        if not isinstance(data, dict) or set(data) != {"name", "measures", "rationale", "suitability"}:
            raise ValueError("Malformed ExperimentMetric payload.")
        return cls(**data)


@dataclass(frozen=True)
class ResearchExperimentPlan:
    """Candidate validation design. It is a proposal, never an experiment result."""

    improvement_artifact_id: str
    candidate_id: str
    objective: str
    hypothesis: str
    proposed_method: str
    baseline: str
    dataset_status: Literal["RESEARCHER_SPECIFIED", "DATASET_REQUIRES_RESEARCHER_SELECTION"]
    selected_dataset: Optional[str] = None
    domain: Optional[str] = None
    desired_validation_type: Optional[str] = None
    researcher_constraints: tuple[str, ...] = ()
    researcher_requirements: tuple[str, ...] = ()
    data_requirements: tuple[str, ...] = ()
    experimental_setup: tuple[str, ...] = ()
    controls: tuple[ExperimentDesignElement, ...] = ()
    ablations: tuple[ExperimentDesignElement, ...] = ()
    metrics: tuple[ExperimentMetric, ...] = ()
    expected_observations: tuple[str, ...] = ()
    interpretation_criteria: tuple[str, ...] = ()
    confounders: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    reproducibility_requirements: tuple[str, ...] = ()
    resource_requirements: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    unresolved_requirements: tuple[str, ...] = ()
    prior_work_artifact_ids: tuple[str, ...] = ()
    verification_artifact_ids: tuple[str, ...] = ()
    supporting_evidence_ids: tuple[str, ...] = ()
    supporting_search_result_ids: tuple[str, ...] = ()
    supporting_claim_ids: tuple[str, ...] = ()
    status: Literal["REQUIRES_RESEARCHER_REVIEW"] = "REQUIRES_RESEARCHER_REVIEW"
    experiment_plan_id: str = field(default_factory=lambda: _id("expplan"))
    created_at: str = field(default_factory=_now)

    _TEXT_FIELDS: ClassVar[tuple[str, ...]] = (
        "data_requirements", "experimental_setup", "expected_observations", "interpretation_criteria",
        "confounders", "limitations", "reproducibility_requirements", "resource_requirements",
        "assumptions", "unresolved_requirements", "researcher_constraints", "researcher_requirements",
    )
    _ID_FIELDS: ClassVar[tuple[str, ...]] = (
        "prior_work_artifact_ids", "verification_artifact_ids", "supporting_evidence_ids",
        "supporting_search_result_ids", "supporting_claim_ids",
    )

    def __post_init__(self) -> None:
        for name in ("experiment_plan_id", "improvement_artifact_id", "candidate_id", "created_at",
                     "objective", "hypothesis", "proposed_method", "baseline"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"ResearchExperimentPlan.{name} must be non-empty.")
        if self.status != "REQUIRES_RESEARCHER_REVIEW":
            raise ValueError("A generated experiment plan must require researcher review.")
        if self.dataset_status not in {"RESEARCHER_SPECIFIED", "DATASET_REQUIRES_RESEARCHER_SELECTION"}:
            raise ValueError("ResearchExperimentPlan.dataset_status is invalid.")
        if self.dataset_status == "RESEARCHER_SPECIFIED" and (not isinstance(self.selected_dataset, str) or not self.selected_dataset.strip()):
            raise ValueError("RESEARCHER_SPECIFIED dataset status requires selected_dataset text.")
        if self.dataset_status == "DATASET_REQUIRES_RESEARCHER_SELECTION" and self.selected_dataset is not None:
            raise ValueError("An unselected dataset must not contain a fabricated dataset name.")
        for name in ("domain", "desired_validation_type"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"ResearchExperimentPlan.{name} must be non-empty text when supplied.")
        for name in self._TEXT_FIELDS + self._ID_FIELDS:
            raw = getattr(self, name)
            if not isinstance(raw, (tuple, list)):
                raise ValueError(f"ResearchExperimentPlan.{name} must be a sequence.")
            values = tuple(raw)
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise ValueError(f"ResearchExperimentPlan.{name} must contain non-empty strings.")
            if name in self._ID_FIELDS and len(values) != len(set(values)):
                raise ValueError(f"ResearchExperimentPlan.{name} must not contain duplicate IDs.")
            object.__setattr__(self, name, values)
        for name, cls in (("controls", ExperimentDesignElement), ("ablations", ExperimentDesignElement),
                          ("metrics", ExperimentMetric)):
            raw = getattr(self, name)
            if not isinstance(raw, (tuple, list)) or any(not isinstance(item, cls) for item in raw):
                raise ValueError(f"ResearchExperimentPlan.{name} must contain typed {cls.__name__} records.")
            object.__setattr__(self, name, tuple(raw))
        if self.dataset_status == "DATASET_REQUIRES_RESEARCHER_SELECTION" and not any(
            "DATASET_REQUIRES_RESEARCHER_SELECTION" in item for item in self.unresolved_requirements
        ):
            raise ValueError("Unselected dataset must be listed as an unresolved requirement.")
        if self.baseline == "BASELINE_REQUIRES_RESEARCHER_SPECIFICATION" and not any(
            "BASELINE_REQUIRES_RESEARCHER_SPECIFICATION" in item for item in self.unresolved_requirements
        ):
            raise ValueError("Unspecified baseline must be listed as an unresolved requirement.")

    def to_dict(self) -> Dict[str, Any]:
        result = {name: list(getattr(self, name)) for name in self._TEXT_FIELDS + self._ID_FIELDS}
        result.update({
            "experiment_plan_id": self.experiment_plan_id,
            "improvement_artifact_id": self.improvement_artifact_id,
            "candidate_id": self.candidate_id,
            "objective": self.objective,
            "hypothesis": self.hypothesis,
            "proposed_method": self.proposed_method,
            "baseline": self.baseline,
            "dataset_status": self.dataset_status,
            "selected_dataset": self.selected_dataset,
            "domain": self.domain,
            "desired_validation_type": self.desired_validation_type,
            "controls": [item.to_dict() for item in self.controls],
            "ablations": [item.to_dict() for item in self.ablations],
            "metrics": [item.to_dict() for item in self.metrics],
            "status": self.status,
            "created_at": self.created_at,
        })
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResearchExperimentPlan":
        keys = {
            "experiment_plan_id", "improvement_artifact_id", "candidate_id", "objective", "hypothesis",
            "proposed_method", "baseline", "dataset_status", "selected_dataset", "domain", "desired_validation_type", "data_requirements",
            "experimental_setup", "controls", "ablations", "metrics", "expected_observations",
            "interpretation_criteria", "confounders", "limitations", "reproducibility_requirements",
            "resource_requirements", "assumptions", "unresolved_requirements", "prior_work_artifact_ids",
            "verification_artifact_ids", "supporting_evidence_ids", "supporting_search_result_ids",
            "supporting_claim_ids", "status", "created_at", "researcher_constraints", "researcher_requirements",
        }
        if not isinstance(data, dict) or set(data) != keys:
            raise ValueError("Malformed ResearchExperimentPlan payload.")
        values = dict(data)
        for name in cls._TEXT_FIELDS + cls._ID_FIELDS:
            if not isinstance(values[name], (list, tuple)):
                raise ValueError(f"Malformed ResearchExperimentPlan.{name}.")
            values[name] = tuple(values[name])
        for name, parser in (("controls", ExperimentDesignElement.from_dict),
                             ("ablations", ExperimentDesignElement.from_dict),
                             ("metrics", ExperimentMetric.from_dict)):
            if not isinstance(values[name], (list, tuple)):
                raise ValueError(f"Malformed ResearchExperimentPlan.{name}.")
            values[name] = tuple(parser(item) for item in values[name])
        return cls(**values)


@dataclass(frozen=True)
class PriorWorkSearchScope:
    """Bounded, explicit search limits; not a claim of exhaustive coverage."""

    maximum_queries: int = 3
    maximum_results: int = 15
    source_types: tuple[str, ...] = ()
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    provider_name: Optional[str] = None

    def __post_init__(self) -> None:
        for name, value, maximum in (
            ("maximum_queries", self.maximum_queries, 5),
            ("maximum_results", self.maximum_results, 50),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= maximum:
                raise ValueError(f"PriorWorkSearchScope.{name} must be between 1 and {maximum}.")
        if not isinstance(self.source_types, (list, tuple)):
            raise ValueError("PriorWorkSearchScope.source_types must be a sequence of strings.")
        source_types = tuple(self.source_types)
        if any(not isinstance(value, str) or not value.strip() for value in source_types):
            raise ValueError("PriorWorkSearchScope.source_types must contain non-empty strings.")
        source_types = tuple(value.strip() for value in source_types)
        if len(source_types) != len(set(source_types)):
            raise ValueError("PriorWorkSearchScope.source_types must not contain duplicates.")
        object.__setattr__(self, "source_types", source_types)
        for name in ("date_from", "date_to"):
            value = getattr(self, name)
            if value is not None:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"PriorWorkSearchScope.{name} must be a YYYY-MM-DD date.")
                try:
                    parsed = date.fromisoformat(value)
                except ValueError as exc:
                    raise ValueError(f"PriorWorkSearchScope.{name} must be a YYYY-MM-DD date.") from exc
                if parsed.isoformat() != value:
                    raise ValueError(f"PriorWorkSearchScope.{name} must be a YYYY-MM-DD date.")
        if self.date_from and self.date_to and date.fromisoformat(self.date_from) > date.fromisoformat(self.date_to):
            raise ValueError("PriorWorkSearchScope.date_from cannot be after date_to.")
        if self.provider_name is not None and (not isinstance(self.provider_name, str) or not self.provider_name.strip()):
            raise ValueError("PriorWorkSearchScope.provider_name must be non-empty when supplied.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "maximum_queries": self.maximum_queries,
            "maximum_results": self.maximum_results,
            "source_types": list(self.source_types),
            "date_from": self.date_from,
            "date_to": self.date_to,
            "provider_name": self.provider_name,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PriorWorkSearchScope":
        required = {"maximum_queries", "maximum_results", "source_types", "date_from", "date_to", "provider_name"}
        if not isinstance(data, dict) or set(data) != required or not isinstance(data["source_types"], (list, tuple)):
            raise ValueError("Malformed PriorWorkSearchScope payload.")
        return cls(**{**data, "source_types": tuple(data["source_types"])})


@dataclass(frozen=True)
class PriorWorkQueryRecord:
    """One executed query, tied to its actual SearchAction and SearchResults."""

    query: str
    search_action_id: str
    origin: Literal["researcher_provided", "system_generated"]
    search_result_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("query", "search_action_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"PriorWorkQueryRecord.{name} must be non-empty.")
        if self.origin not in {"researcher_provided", "system_generated"}:
            raise ValueError("PriorWorkQueryRecord.origin is invalid.")
        ids = tuple(self.search_result_ids)
        if any(not isinstance(value, str) or not value.strip() for value in ids) or len(ids) != len(set(ids)):
            raise ValueError("PriorWorkQueryRecord.search_result_ids must contain unique non-empty IDs.")
        object.__setattr__(self, "search_result_ids", ids)

    def to_dict(self) -> Dict[str, Any]:
        return {"query": self.query, "search_action_id": self.search_action_id, "origin": self.origin,
                "search_result_ids": list(self.search_result_ids)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PriorWorkQueryRecord":
        required = {"query", "search_action_id", "origin", "search_result_ids"}
        if not isinstance(data, dict) or set(data) != required or not isinstance(data["search_result_ids"], (list, tuple)):
            raise ValueError("Malformed PriorWorkQueryRecord payload.")
        return cls(**{**data, "search_result_ids": tuple(data["search_result_ids"])})


@dataclass(frozen=True)
class PriorWorkResultAssessment:
    """System relevance assessment for a discovered result, not a novelty verdict."""

    search_result_id: str
    relevance: Literal["RELEVANT", "POTENTIALLY_RELEVANT", "NOT_RELEVANT"]
    reason: str

    def __post_init__(self) -> None:
        for name in ("search_result_id", "reason"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"PriorWorkResultAssessment.{name} must be non-empty.")
        if self.relevance not in {"RELEVANT", "POTENTIALLY_RELEVANT", "NOT_RELEVANT"}:
            raise ValueError("PriorWorkResultAssessment.relevance is invalid.")

    def to_dict(self) -> Dict[str, Any]:
        return {"search_result_id": self.search_result_id, "relevance": self.relevance, "reason": self.reason}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PriorWorkResultAssessment":
        if not isinstance(data, dict) or set(data) != {"search_result_id", "relevance", "reason"}:
            raise ValueError("Malformed PriorWorkResultAssessment payload.")
        return cls(**data)


@dataclass(frozen=True)
class PriorWorkFinding:
    """A bounded comparison of a potentially relevant persisted SearchResult."""

    search_result_id: str
    source_id: str
    relevance: Literal["RELEVANT", "POTENTIALLY_RELEVANT"]
    relationship: Literal["DIRECT_OVERLAP", "PARTIAL_OVERLAP", "RELATED_DISTINCT", "INSUFFICIENT_INFORMATION"]
    problem_overlap: Optional[str]
    method_overlap: Optional[str]
    dataset_overlap: Optional[str]
    evaluation_overlap: Optional[str]
    differences: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    finding_id: str = field(default_factory=lambda: _id("priorwork"))

    def __post_init__(self) -> None:
        for name in ("finding_id", "search_result_id", "source_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"PriorWorkFinding.{name} must be non-empty.")
        if self.relevance not in {"RELEVANT", "POTENTIALLY_RELEVANT"}:
            raise ValueError("PriorWorkFinding cannot represent NOT_RELEVANT results.")
        if self.relationship not in {"DIRECT_OVERLAP", "PARTIAL_OVERLAP", "RELATED_DISTINCT", "INSUFFICIENT_INFORMATION"}:
            raise ValueError("PriorWorkFinding.relationship is invalid.")
        for name in ("problem_overlap", "method_overlap", "dataset_overlap", "evaluation_overlap"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"PriorWorkFinding.{name} must be text or None.")
        for name in ("differences", "limitations"):
            raw_values = getattr(self, name)
            if not isinstance(raw_values, (tuple, list)):
                raise ValueError(f"PriorWorkFinding.{name} must be a sequence of strings.")
            values = tuple(raw_values)
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise ValueError(f"PriorWorkFinding.{name} must contain non-empty strings.")
            object.__setattr__(self, name, values)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "search_result_id": self.search_result_id, "source_id": self.source_id,
            "relevance": self.relevance, "relationship": self.relationship,
            "problem_overlap": self.problem_overlap, "method_overlap": self.method_overlap,
            "dataset_overlap": self.dataset_overlap, "evaluation_overlap": self.evaluation_overlap,
            "differences": list(self.differences), "limitations": list(self.limitations),
            "finding_id": self.finding_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PriorWorkFinding":
        required = {
            "search_result_id", "source_id", "relevance", "relationship", "problem_overlap",
            "method_overlap", "dataset_overlap", "evaluation_overlap", "differences", "limitations", "finding_id",
        }
        if not isinstance(data, dict) or set(data) != required:
            raise ValueError("Malformed PriorWorkFinding payload.")
        values = dict(data)
        for name in ("differences", "limitations"):
            if not isinstance(values[name], (list, tuple)):
                raise ValueError(f"Malformed PriorWorkFinding.{name}.")
            values[name] = tuple(values[name])
        return cls(**values)


@dataclass(frozen=True)
class ResearchArtifact:
    """Typed reference record for an output produced by a task capability.

    Retrieval artifacts record retrieved material. Verification artifacts
    record assessments of existing Evidence. Improvement artifacts store only
    generated, unvalidated candidates and their supporting input references.
    """

    execution_id: str
    run_id: str
    task_id: str
    search_action_id: Optional[str]
    search_result_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    artifact_type: Literal["retrieval_result", "verification_result", "improvement_candidate", "prior_work_investigation", "experiment_plan"] = "retrieval_result"
    capability: Literal["local_retrieval", "evidence_verification", "improvement_generation", "prior_work_investigation", "experiment_planning"] = "local_retrieval"
    artifact_id: str = field(default_factory=lambda: _id("artifact"))
    created_at: str = field(default_factory=_now)
    verification_claim: Optional[str] = None
    verification_verdict: Optional[str] = None
    verification_assessments: tuple[VerificationUnitAssessment, ...] = ()
    improvement_problem_statement: Optional[str] = None
    improvement_problem_origin: Optional[Literal["researcher_provided"]] = None
    improvement_candidates: tuple[ResearchImprovementCandidate, ...] = ()
    improvement_claim_ids: tuple[str, ...] = ()
    improvement_verification_artifact_ids: tuple[str, ...] = ()
    prior_work_candidate_artifact_id: Optional[str] = None
    prior_work_candidate_id: Optional[str] = None
    prior_work_scope: Optional[PriorWorkSearchScope] = None
    prior_work_provider: Optional[str] = None
    prior_work_queries: tuple[PriorWorkQueryRecord, ...] = ()
    prior_work_assessments: tuple[PriorWorkResultAssessment, ...] = ()
    prior_work_findings: tuple[PriorWorkFinding, ...] = ()
    prior_work_status: Optional[Literal["COMPLETED", "ZERO_RESULTS"]] = None
    prior_work_uncertainties: tuple[str, ...] = ()
    experiment_plan: Optional[ResearchExperimentPlan] = None

    def __post_init__(self) -> None:
        for name in ("execution_id", "run_id", "task_id", "artifact_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"ResearchArtifact.{name} must be a non-empty string.")
        if self.artifact_type not in {"retrieval_result", "verification_result", "improvement_candidate", "prior_work_investigation", "experiment_plan"}:
            raise ValueError(f"Unsupported ResearchArtifact artifact_type: {self.artifact_type!r}.")
        expected_capability = {
            "retrieval_result": "local_retrieval",
            "verification_result": "evidence_verification",
            "improvement_candidate": "improvement_generation",
            "prior_work_investigation": "prior_work_investigation",
            "experiment_plan": "experiment_planning",
        }[self.artifact_type]
        if self.capability != expected_capability:
            raise ValueError(f"Unsupported ResearchArtifact capability: {self.capability!r}.")
        for name in ("search_result_ids", "evidence_ids"):
            values = getattr(self, name)
            if not isinstance(values, (list, tuple)):
                raise ValueError(f"ResearchArtifact.{name} must be a sequence of IDs.")
            normalized = tuple(values)
            if any(not isinstance(value, str) or not value.strip() for value in normalized):
                raise ValueError(f"ResearchArtifact.{name} must contain non-empty strings.")
            if len(set(normalized)) != len(normalized):
                raise ValueError(f"ResearchArtifact.{name} must not contain duplicates.")
            object.__setattr__(self, name, normalized)
        if self.artifact_type == "retrieval_result":
            if not isinstance(self.search_action_id, str) or not self.search_action_id.strip():
                raise ValueError("Retrieval ResearchArtifact requires a search_action_id.")
            if self.verification_claim is not None or self.verification_verdict is not None or self.verification_assessments:
                raise ValueError("Retrieval ResearchArtifact cannot contain verification fields.")
            if self.improvement_problem_statement is not None or self.improvement_candidates:
                raise ValueError("Retrieval ResearchArtifact cannot contain improvement fields.")
        else:
            if self.artifact_type not in {"prior_work_investigation"} and (self.search_action_id is not None or self.search_result_ids):
                raise ValueError("Non-retrieval ResearchArtifact cannot contain retrieval action/result references.")
            if self.artifact_type == "verification_result":
                if not isinstance(self.verification_claim, str) or not self.verification_claim.strip():
                    raise ValueError("Verification ResearchArtifact requires a verification_claim.")
                if self.verification_verdict not in {"SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"}:
                    raise ValueError("Verification ResearchArtifact has an invalid verification_verdict.")
                if not self.evidence_ids:
                    raise ValueError("Verification ResearchArtifact requires existing evidence_ids.")
                assessments = tuple(self.verification_assessments)
                if len(assessments) != 1 or any(not isinstance(item, VerificationUnitAssessment) for item in assessments):
                    raise ValueError("Verification ResearchArtifact requires exactly one typed claim assessment.")
                if any(item.verdict not in {"SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"} for item in assessments):
                    raise ValueError("Verification ResearchArtifact contains an invalid assessment verdict.")
                if assessments[0].claim != self.verification_claim or assessments[0].verdict != self.verification_verdict:
                    raise ValueError("Verification ResearchArtifact claim/verdict must match its typed assessment.")
                if self.improvement_problem_statement is not None or self.improvement_candidates:
                    raise ValueError("Verification ResearchArtifact cannot contain improvement fields.")
                object.__setattr__(self, "verification_assessments", assessments)
            elif self.artifact_type == "improvement_candidate":
                if self.verification_claim is not None or self.verification_verdict is not None or self.verification_assessments:
                    raise ValueError("Improvement ResearchArtifact cannot contain verification fields.")
                if not isinstance(self.improvement_problem_statement, str) or not self.improvement_problem_statement.strip():
                    raise ValueError("Improvement ResearchArtifact requires an improvement_problem_statement.")
                if self.improvement_problem_origin != "researcher_provided":
                    raise ValueError("Improvement problem origin must be researcher_provided.")
                if not self.evidence_ids:
                    raise ValueError("Improvement ResearchArtifact requires supporting Evidence IDs.")
                candidates = tuple(self.improvement_candidates)
                if any(not isinstance(item, ResearchImprovementCandidate) for item in candidates):
                    raise ValueError("Improvement ResearchArtifact candidates must be typed.")
                if any(item.problem_statement != self.improvement_problem_statement for item in candidates):
                    raise ValueError("Candidate problem statements must match their ResearchArtifact.")
                if any(not set(item.supporting_evidence_ids).issubset(self.evidence_ids) for item in candidates):
                    raise ValueError("Candidate references Evidence outside its ResearchArtifact.")
                object.__setattr__(self, "improvement_candidates", candidates)
            elif self.artifact_type == "prior_work_investigation":
                if self.search_action_id is not None or self.evidence_ids:
                    raise ValueError("Prior-work artifact cannot claim one search action or acquired Evidence.")
                if (
                    self.verification_claim is not None or self.verification_verdict is not None
                    or self.verification_assessments or self.improvement_problem_statement is not None
                    or self.improvement_problem_origin is not None or self.improvement_candidates
                    or self.improvement_claim_ids or self.improvement_verification_artifact_ids
                ):
                    raise ValueError("Prior-work ResearchArtifact cannot contain verification or improvement payloads.")
                for name, expected_type in (
                    ("prior_work_scope", PriorWorkSearchScope),
                    ("prior_work_candidate_artifact_id", str),
                    ("prior_work_candidate_id", str),
                    ("prior_work_provider", str),
                    ("prior_work_status", str),
                ):
                    value = getattr(self, name)
                    if not isinstance(value, expected_type) or (isinstance(value, str) and not value.strip()):
                        raise ValueError(f"Prior-work ResearchArtifact requires {name}.")
                if self.prior_work_status not in {"COMPLETED", "ZERO_RESULTS"}:
                    raise ValueError("Prior-work ResearchArtifact has an invalid status.")
                if self.prior_work_status == "ZERO_RESULTS" and self.search_result_ids:
                    raise ValueError("ZERO_RESULTS prior-work artifact cannot contain SearchResults.")
                if self.prior_work_status == "COMPLETED" and not self.search_result_ids:
                    raise ValueError("Prior-work artifact with no SearchResults must use ZERO_RESULTS status.")
                if not isinstance(self.prior_work_queries, (tuple, list)):
                    raise ValueError("prior_work_queries must be a sequence.")
                query_records = tuple(self.prior_work_queries)
                if not query_records or any(not isinstance(item, PriorWorkQueryRecord) for item in query_records):
                    raise ValueError("Prior-work artifact requires typed query records.")
                if len(query_records) > self.prior_work_scope.maximum_queries:
                    raise ValueError("Prior-work artifact exceeds its query budget.")
                if len(self.search_result_ids) > self.prior_work_scope.maximum_results:
                    raise ValueError("Prior-work artifact exceeds its total result budget.")
                query_ids = [item.search_action_id for item in query_records]
                if len(query_ids) != len(set(query_ids)):
                    raise ValueError("Prior-work artifact query records contain duplicate SearchAction IDs.")
                flattened_results = tuple(result_id for item in query_records for result_id in item.search_result_ids)
                if flattened_results != self.search_result_ids:
                    raise ValueError("Prior-work query records do not match artifact SearchResult IDs and order.")
                object.__setattr__(self, "prior_work_queries", query_records)
                for name, cls in (
                    ("prior_work_assessments", PriorWorkResultAssessment),
                    ("prior_work_findings", PriorWorkFinding),
                ):
                    values = tuple(getattr(self, name))
                    if any(not isinstance(item, cls) for item in values):
                        raise ValueError(f"{name} must contain typed records.")
                    object.__setattr__(self, name, values)
                if tuple(item.search_result_id for item in self.prior_work_assessments) != self.search_result_ids:
                    raise ValueError("Prior-work assessments must cover every SearchResult in retrieval order.")
                if any(item.search_result_id not in self.search_result_ids for item in self.prior_work_findings):
                    raise ValueError("Prior-work finding references a SearchResult outside the artifact.")
                relevance_by_result = {item.search_result_id: item.relevance for item in self.prior_work_assessments}
                expected_findings = {
                    item.search_result_id for item in self.prior_work_assessments
                    if item.relevance != "NOT_RELEVANT"
                }
                actual_findings = {item.search_result_id for item in self.prior_work_findings}
                if expected_findings != actual_findings or any(
                    item.relevance != relevance_by_result[item.search_result_id]
                    for item in self.prior_work_findings
                ):
                    raise ValueError("Prior-work findings must match all and only relevant result assessments.")
                if any(
                    assessment.relevance == "NOT_RELEVANT"
                    and any(f.search_result_id == assessment.search_result_id for f in self.prior_work_findings)
                    for assessment in self.prior_work_assessments
                ):
                    raise ValueError("NOT_RELEVANT SearchResults cannot become prior-work findings.")
                finding_ids = [item.finding_id for item in self.prior_work_findings]
                if len(finding_ids) != len(set(finding_ids)):
                    raise ValueError("Prior-work artifact contains duplicate finding IDs.")
                if not isinstance(self.prior_work_uncertainties, (tuple, list)):
                    raise ValueError("prior_work_uncertainties must be a sequence of strings.")
                uncertainties = tuple(self.prior_work_uncertainties)
                if any(not isinstance(value, str) or not value.strip() for value in uncertainties):
                    raise ValueError("prior_work_uncertainties must contain non-empty text.")
                object.__setattr__(self, "prior_work_uncertainties", uncertainties)
            else:
                if self.search_action_id is not None or self.search_result_ids:
                    raise ValueError("Experiment-plan artifact cannot contain direct retrieval references.")
                if self.verification_claim is not None or self.verification_verdict is not None or self.verification_assessments:
                    raise ValueError("Experiment-plan artifact cannot contain verification result fields.")
                if self.improvement_problem_statement is not None or self.improvement_problem_origin is not None or self.improvement_candidates:
                    raise ValueError("Experiment-plan artifact cannot contain improvement-generation fields.")
                if self.improvement_claim_ids or self.improvement_verification_artifact_ids:
                    raise ValueError("Experiment-plan artifact cannot contain improvement-generation references.")
                if any((
                    self.prior_work_candidate_artifact_id, self.prior_work_candidate_id, self.prior_work_scope,
                    self.prior_work_provider, self.prior_work_queries, self.prior_work_assessments,
                    self.prior_work_findings, self.prior_work_status, self.prior_work_uncertainties,
                )):
                    raise ValueError("Experiment-plan artifact cannot contain prior-work payload fields.")
                if not isinstance(self.experiment_plan, ResearchExperimentPlan):
                    raise ValueError("Experiment-plan artifact requires a typed ResearchExperimentPlan.")
                if tuple(self.evidence_ids) != self.experiment_plan.supporting_evidence_ids:
                    raise ValueError("Experiment-plan artifact evidence_ids must match the plan support references.")
            for name in ("improvement_claim_ids", "improvement_verification_artifact_ids"):
                values = tuple(getattr(self, name))
                if any(not isinstance(value, str) or not value.strip() for value in values):
                    raise ValueError(f"ResearchArtifact.{name} must contain non-empty IDs.")
                if len(values) != len(set(values)):
                    raise ValueError(f"ResearchArtifact.{name} must not contain duplicates.")
                object.__setattr__(self, name, values)
            if self.artifact_type != "improvement_candidate" and (self.improvement_claim_ids or self.improvement_verification_artifact_ids):
                raise ValueError("Non-improvement ResearchArtifact cannot contain improvement references.")
            if self.artifact_type == "improvement_candidate" and any(
                not set(candidate.supporting_claim_ids).issubset(self.improvement_claim_ids)
                or not set(candidate.supporting_verification_artifact_ids).issubset(self.improvement_verification_artifact_ids)
                for candidate in self.improvement_candidates
            ):
                raise ValueError("Improvement candidate references were not supplied to its artifact.")
        if self.artifact_type != "prior_work_investigation" and any((
            self.prior_work_candidate_artifact_id, self.prior_work_candidate_id, self.prior_work_scope,
            self.prior_work_provider, self.prior_work_queries, self.prior_work_assessments,
            self.prior_work_findings, self.prior_work_status, self.prior_work_uncertainties,
        )):
            raise ValueError("Non-prior-work ResearchArtifact cannot contain prior-work fields.")
        if self.artifact_type != "experiment_plan" and self.experiment_plan is not None:
            raise ValueError("Non-experiment-plan ResearchArtifact cannot contain an experiment_plan.")
        if not isinstance(self.created_at, str) or not self.created_at.strip():
            raise ValueError("ResearchArtifact.created_at must be a non-empty timestamp.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "search_action_id": self.search_action_id,
            "search_result_ids": list(self.search_result_ids),
            "evidence_ids": list(self.evidence_ids),
            "artifact_type": self.artifact_type,
            "capability": self.capability,
            "artifact_id": self.artifact_id,
            "created_at": self.created_at,
            "verification_claim": self.verification_claim,
            "verification_verdict": self.verification_verdict,
            "verification_assessments": [item.to_dict() for item in self.verification_assessments],
            "improvement_problem_statement": self.improvement_problem_statement,
            "improvement_problem_origin": self.improvement_problem_origin,
            "improvement_candidates": [item.to_dict() for item in self.improvement_candidates],
            "improvement_claim_ids": list(self.improvement_claim_ids),
            "improvement_verification_artifact_ids": list(self.improvement_verification_artifact_ids),
            "prior_work_candidate_artifact_id": self.prior_work_candidate_artifact_id,
            "prior_work_candidate_id": self.prior_work_candidate_id,
            "prior_work_scope": self.prior_work_scope.to_dict() if self.prior_work_scope is not None else None,
            "prior_work_provider": self.prior_work_provider,
            "prior_work_queries": [item.to_dict() for item in self.prior_work_queries],
            "prior_work_assessments": [item.to_dict() for item in self.prior_work_assessments],
            "prior_work_findings": [item.to_dict() for item in self.prior_work_findings],
            "prior_work_status": self.prior_work_status,
            "prior_work_uncertainties": list(self.prior_work_uncertainties),
            "experiment_plan": self.experiment_plan.to_dict() if self.experiment_plan is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResearchArtifact":
        required = {
            "execution_id", "run_id", "task_id", "search_action_id",
            "search_result_ids", "evidence_ids", "artifact_type", "capability",
            "artifact_id", "created_at",
        }
        verification_fields = {"verification_claim", "verification_verdict", "verification_assessments"}
        improvement_fields = {
            "improvement_problem_statement", "improvement_problem_origin", "improvement_candidates",
            "improvement_claim_ids", "improvement_verification_artifact_ids",
        }
        prior_work_fields = {
            "prior_work_candidate_artifact_id", "prior_work_candidate_id", "prior_work_scope",
            "prior_work_provider", "prior_work_queries", "prior_work_assessments",
            "prior_work_findings", "prior_work_status", "prior_work_uncertainties",
        }
        experiment_fields = {"experiment_plan"}
        groups = (verification_fields, improvement_fields, prior_work_fields, experiment_fields)
        allowed_shapes = tuple(
            required | set().union(*(groups[index] for index in range(len(groups)) if mask & (1 << index)))
            for mask in range(1 << len(groups))
        )
        if not isinstance(data, dict) or set(data) not in allowed_shapes:
            raise ValueError("Malformed ResearchArtifact payload.")
        try:
            values = dict(data)
            values.setdefault("verification_claim", None)
            values.setdefault("verification_verdict", None)
            raw_assessments = values.setdefault("verification_assessments", [])
            if not isinstance(raw_assessments, (list, tuple)):
                raise ValueError("Malformed ResearchArtifact verification_assessments.")
            values["verification_assessments"] = tuple(
                VerificationUnitAssessment.from_dict(item) for item in raw_assessments
            )
            values.setdefault("improvement_problem_statement", None)
            values.setdefault("improvement_problem_origin", None)
            raw_candidates = values.setdefault("improvement_candidates", [])
            if not isinstance(raw_candidates, (list, tuple)):
                raise ValueError("Malformed ResearchArtifact improvement_candidates.")
            values["improvement_candidates"] = tuple(
                ResearchImprovementCandidate.from_dict(item) for item in raw_candidates
            )
            for name in ("improvement_claim_ids", "improvement_verification_artifact_ids"):
                raw = values.setdefault(name, [])
                if not isinstance(raw, (list, tuple)):
                    raise ValueError(f"Malformed ResearchArtifact.{name}.")
                values[name] = tuple(raw)
            values.setdefault("prior_work_candidate_artifact_id", None)
            values.setdefault("prior_work_candidate_id", None)
            raw_scope = values.setdefault("prior_work_scope", None)
            if raw_scope is not None:
                values["prior_work_scope"] = PriorWorkSearchScope.from_dict(raw_scope)
            values.setdefault("prior_work_provider", None)
            for name, parser in (
                ("prior_work_queries", PriorWorkQueryRecord.from_dict),
                ("prior_work_assessments", PriorWorkResultAssessment.from_dict),
                ("prior_work_findings", PriorWorkFinding.from_dict),
            ):
                raw = values.setdefault(name, [])
                if not isinstance(raw, (list, tuple)):
                    raise ValueError(f"Malformed ResearchArtifact.{name}.")
                values[name] = tuple(parser(item) for item in raw)
            values.setdefault("prior_work_status", None)
            raw_uncertainties = values.setdefault("prior_work_uncertainties", [])
            if not isinstance(raw_uncertainties, (list, tuple)):
                raise ValueError("Malformed ResearchArtifact.prior_work_uncertainties.")
            values["prior_work_uncertainties"] = tuple(raw_uncertainties)
            raw_experiment_plan = values.setdefault("experiment_plan", None)
            if raw_experiment_plan is not None:
                values["experiment_plan"] = ResearchExperimentPlan.from_dict(raw_experiment_plan)
            return cls(**values)
        except TypeError as exc:
            raise ValueError(f"Malformed ResearchArtifact payload: {exc}") from exc


@dataclass
class ResearchState:
    """Single source of truth shared by the research workflow."""

    user_request: str
    research_objective: str = ""
    status: str = "created"
    iteration: int = 0

    research_questions: List[ResearchQuestion] = field(default_factory=list)
    sources: List[Source] = field(default_factory=list)
    evidence: List[Evidence] = field(default_factory=list)
    claims: List[ResearchClaim] = field(default_factory=list)
    searches: List[SearchAction] = field(default_factory=list)
    events: List[AgentEvent] = field(default_factory=list)
    unresolved_questions: List[str] = field(default_factory=list)
    contradictions: List[str] = field(default_factory=list)

    final_report: str = ""
    errors: List[str] = field(default_factory=list)

    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    # Appended to preserve the positional order of existing state fields.
    search_results: List["SearchResult"] = field(default_factory=list)
    document_versions: List[DocumentVersion] = field(default_factory=list)
    passage_references: List[PassageReference] = field(default_factory=list)
    # Lineage records are independent of the legacy scalar `iteration` above.
    research_runs: List[ResearchRun] = field(default_factory=list)
    iterations: List[ResearchIteration] = field(default_factory=list)
    # Appended to preserve legacy positional construction and old payloads.
    task_executions: List["ResearchTaskExecution"] = field(default_factory=list)
    # Phase 6 typed capability outputs; absent in older state payloads.
    research_artifacts: List[ResearchArtifact] = field(default_factory=list)
    # Phase 7A records are appended so positional construction and old state payloads remain valid.
    research_loops: List["ResearchLoop"] = field(default_factory=list)

    def touch(self) -> None:
        self.updated_at = _now()

    def add_run(self, run: Optional[ResearchRun] = None) -> ResearchRun:
        run = run or ResearchRun()
        if run.run_id in {item.run_id for item in self.research_runs}:
            raise ValueError(f"Duplicate research run_id {run.run_id!r}.")
        self.research_runs.append(run)
        self.touch()
        return run

    def add_iteration(self, iteration: ResearchIteration) -> ResearchIteration:
        if iteration.run_id not in {item.run_id for item in self.research_runs}:
            raise ValueError(f"ResearchIteration references unknown run_id {iteration.run_id!r}.")
        if iteration.iteration_id in {item.iteration_id for item in self.iterations}:
            raise ValueError(f"Duplicate iteration_id {iteration.iteration_id!r}.")
        if any(item.run_id == iteration.run_id and item.number == iteration.number for item in self.iterations):
            raise ValueError(
                f"Duplicate iteration number {iteration.number} for run {iteration.run_id!r}."
            )
        self.iterations.append(iteration)
        self.touch()
        return iteration

    def create_iteration(
        self,
        run_id: str,
        *,
        status: Literal["created", "running", "completed", "failed"] = "created",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ResearchIteration:
        if run_id not in {item.run_id for item in self.research_runs}:
            raise ValueError(f"ResearchIteration references unknown run_id {run_id!r}.")
        number = max((item.number for item in self.iterations if item.run_id == run_id), default=0) + 1
        return self.add_iteration(
            ResearchIteration(run_id=run_id, number=number, status=status, metadata=dict(metadata or {}))
        )

    def add_task_execution(self, execution: "ResearchTaskExecution") -> "ResearchTaskExecution":
        """Register a CREATED execution for a task authorized on a known run."""
        from research_task_execution import ResearchTaskExecution, TaskExecutionStatus

        if not isinstance(execution, ResearchTaskExecution):
            raise TypeError("execution must be a ResearchTaskExecution.")
        if execution.status is not TaskExecutionStatus.CREATED:
            raise ValueError("A new ResearchTaskExecution must be CREATED when registered.")
        if execution.execution_id in {item.execution_id for item in self.task_executions}:
            raise ValueError(f"Duplicate task execution_id {execution.execution_id!r}.")
        run = next((item for item in self.research_runs if item.run_id == execution.run_id), None)
        if run is None:
            raise ValueError(f"TaskExecution references unknown run_id {execution.run_id!r}.")
        if (
            run.plan_id != execution.plan_id
            or run.plan_revision != execution.plan_revision
            or run.approval_review_id is None
        ):
            raise ValueError("TaskExecution plan/revision does not match its ResearchRun authorization.")
        if run.authorized_task_ids is None or execution.task_id not in run.authorized_task_ids:
            raise ValueError(f"TaskExecution task_id {execution.task_id!r} is not authorized for its ResearchRun.")
        if execution.iteration_id is not None:
            iteration = next((item for item in self.iterations if item.iteration_id == execution.iteration_id), None)
            if iteration is None or iteration.run_id != run.run_id:
                raise ValueError("TaskExecution iteration_id is missing or belongs to another ResearchRun.")

        prior = [
            item for item in self.task_executions
            if item.run_id == execution.run_id and item.task_id == execution.task_id
        ]
        if any(item.status in {TaskExecutionStatus.CREATED, TaskExecutionStatus.RUNNING} for item in prior):
            raise ValueError("An active TaskExecution already exists for this run/task.")
        if prior and prior[-1].status is TaskExecutionStatus.COMPLETED:
            if execution.retry_of_execution_id is not None:
                raise ValueError("A completed TaskExecution cannot be referenced as a retry.")
            if execution.continuation_of_execution_id != prior[-1].execution_id:
                raise ValueError("Continuation must explicitly reference the latest completed TaskExecution.")
            matches = [
                authorization
                for loop in self.research_loops
                for authorization in getattr(loop, "continuation_authorizations", ())
                if authorization.authorization_id == execution.continuation_authorization_id
            ]
            if len(matches) != 1:
                raise ValueError("TaskExecution requires one registered continuation authorization.")
            authorization = matches[0]
            if (authorization.run_id != execution.run_id or authorization.plan_id != execution.plan_id
                    or authorization.plan_revision != execution.plan_revision
                    or authorization.task_id != execution.task_id
                    or authorization.previous_execution_id != prior[-1].execution_id
                    or authorization.iteration_id != execution.iteration_id):
                raise ValueError("TaskExecution does not match its continuation authorization.")
        elif execution.continuation_of_execution_id is not None or execution.continuation_authorization_id is not None:
            raise ValueError("Continuation references require a preceding completed TaskExecution.")
        if prior and prior[-1].status is TaskExecutionStatus.FAILED:
            if prior[-1].status is not TaskExecutionStatus.FAILED:
                raise ValueError("Only a failed TaskExecution can be retried.")
            if execution.retry_of_execution_id != prior[-1].execution_id:
                raise ValueError("Retry must reference the latest failed TaskExecution.")
            if execution.continuation_of_execution_id is not None:
                raise ValueError("A failed TaskExecution can only use retry semantics, not continuation.")
        elif not prior and execution.retry_of_execution_id is not None:
            raise ValueError("retry_of_execution_id does not reference a prior execution.")

        self.task_executions.append(execution)
        self.touch()
        return execution

    def add_research_loop(self, loop: "ResearchLoop") -> "ResearchLoop":
        """Register a typed loop after validating its existing run/iteration links."""
        from research_loop import ResearchLoop

        if not isinstance(loop, ResearchLoop):
            raise TypeError("loop must be a ResearchLoop.")
        if loop.loop_id in {item.loop_id for item in self.research_loops}:
            raise ValueError(f"Duplicate research_loop loop_id {loop.loop_id!r}.")
        errors = loop.validate_references(self)
        if errors:
            raise ValueError("Invalid ResearchLoop references: " + "; ".join(errors))
        self.research_loops.append(loop)
        self.touch()
        return loop

    def add_research_artifact(self, artifact: ResearchArtifact) -> ResearchArtifact:
        """Register a typed capability artifact while its execution is RUNNING."""
        from research_task_execution import TaskExecutionStatus

        if not isinstance(artifact, ResearchArtifact):
            raise TypeError("artifact must be a ResearchArtifact.")
        if artifact.artifact_id in {item.artifact_id for item in self.research_artifacts}:
            raise ValueError(f"Duplicate ResearchArtifact artifact_id {artifact.artifact_id!r}.")
        execution = next(
            (item for item in self.task_executions if item.execution_id == artifact.execution_id),
            None,
        )
        if execution is None or execution.status is not TaskExecutionStatus.RUNNING:
            raise ValueError("ResearchArtifact must reference a currently RUNNING TaskExecution.")
        if execution.run_id != artifact.run_id or execution.task_id != artifact.task_id:
            raise ValueError("ResearchArtifact run/task lineage does not match its TaskExecution.")
        if any(item.execution_id == artifact.execution_id for item in self.research_artifacts):
            raise ValueError("A TaskExecution may create only one capability artifact.")
        known_evidence = {item.evidence_id: item for item in self.evidence}
        if artifact.artifact_type == "retrieval_result":
            search = next(
                (item for item in self.searches if item.search_id == artifact.search_action_id),
                None,
            )
            if search is None or search.run_id != artifact.run_id:
                raise ValueError("Retrieval ResearchArtifact SearchAction is missing or belongs to another run.")
            results = [item for item in self.search_results if item.search_id == search.search_id]
            if tuple(item.result_id for item in results) != artifact.search_result_ids:
                raise ValueError("ResearchArtifact SearchResult IDs do not match its SearchAction results.")
            result_ids = set(artifact.search_result_ids)
            for evidence_id in artifact.evidence_ids:
                evidence = known_evidence.get(evidence_id)
                if evidence is None or evidence.search_result_id not in result_ids:
                    raise ValueError("Retrieval ResearchArtifact Evidence must resolve to one of its SearchResults.")
        elif artifact.artifact_type == "verification_result":
            errors = self.validate_provenance()
            if errors:
                raise ValueError("Verification artifact requires valid state provenance: " + "; ".join(errors))
            versions = {item.version_id: item for item in self.document_versions}
            passages = {item.passage_reference_id: item for item in self.passage_references}
            results = {item.result_id: item for item in self.search_results if item.result_id is not None}
            searches = {item.search_id: item for item in self.searches}
            for evidence_id in artifact.evidence_ids:
                evidence = known_evidence.get(evidence_id)
                if evidence is None:
                    raise ValueError(f"Verification ResearchArtifact references unknown Evidence {evidence_id!r}.")
                version = versions.get(evidence.document_version_id or "")
                passage = passages.get(evidence.passage_reference_id or "")
                result = results.get(evidence.search_result_id or "")
                if version is None or passage is None or result is None:
                    raise ValueError(
                        f"Verification Evidence {evidence_id!r} must resolve Source, DocumentVersion, "
                        "PassageReference, and SearchResult provenance."
                    )
                if version.source_id != evidence.source_id or passage.document_version_id != version.version_id:
                    raise ValueError(f"Verification Evidence {evidence_id!r} has inconsistent source provenance.")
                if result.source_id != evidence.source_id:
                    raise ValueError(f"Verification Evidence {evidence_id!r} source mismatches its SearchResult.")
                search = searches.get(result.search_id or "")
                if search is None or (search.run_id is not None and search.run_id != artifact.run_id):
                    raise ValueError(
                        f"Verification Evidence {evidence_id!r} SearchAction is missing or belongs to another run."
                    )
            assessed_evidence_ids = {
                assessment.evidence_id
                for unit in artifact.verification_assessments
                for assessment in unit.evidence_assessments
            }
            if not assessed_evidence_ids.issubset(set(artifact.evidence_ids)):
                raise ValueError("Verification assessments reference Evidence outside the artifact evidence_ids.")
        elif artifact.artifact_type == "improvement_candidate":
            errors = self.validate_provenance()
            if errors:
                raise ValueError("Improvement artifact requires valid state provenance: " + "; ".join(errors))
            versions = {item.version_id: item for item in self.document_versions}
            passages = {item.passage_reference_id: item for item in self.passage_references}
            results = {item.result_id: item for item in self.search_results if item.result_id is not None}
            searches = {item.search_id: item for item in self.searches}
            for evidence_id in artifact.evidence_ids:
                evidence = known_evidence.get(evidence_id)
                if evidence is None:
                    raise ValueError(f"Improvement ResearchArtifact references unknown Evidence {evidence_id!r}.")
                version = versions.get(evidence.document_version_id or "")
                passage = passages.get(evidence.passage_reference_id or "")
                result = results.get(evidence.search_result_id or "")
                if version is None or passage is None or result is None:
                    raise ValueError(f"Improvement Evidence {evidence_id!r} has incomplete provenance.")
                if version.source_id != evidence.source_id or passage.document_version_id != version.version_id:
                    raise ValueError(f"Improvement Evidence {evidence_id!r} has inconsistent source provenance.")
                if result.source_id != evidence.source_id:
                    raise ValueError(f"Improvement Evidence {evidence_id!r} source mismatches its SearchResult.")
                search = searches.get(result.search_id or "")
                if search is None or search.run_id != artifact.run_id:
                    raise ValueError(f"Improvement Evidence {evidence_id!r} has invalid run/search lineage.")

            claims = {item.claim_id: item for item in self.claims}
            for claim_id in artifact.improvement_claim_ids:
                claim = claims.get(claim_id)
                if claim is None or claim.status != "SUPPORTED":
                    raise ValueError(f"Improvement artifact claim {claim_id!r} is missing or not SUPPORTED.")
                if not set(claim.evidence_ids).issubset(set(artifact.evidence_ids)):
                    raise ValueError(f"Improvement artifact claim {claim_id!r} references unselected Evidence.")
            artifacts = {item.artifact_id: item for item in self.research_artifacts}
            for verification_id in artifact.improvement_verification_artifact_ids:
                verification = artifacts.get(verification_id)
                verification_execution = next(
                    (item for item in self.task_executions if item.execution_id == (verification.execution_id if verification else None)),
                    None,
                )
                if (
                    verification is None
                    or verification.artifact_type != "verification_result"
                    or verification.verification_verdict != "SUPPORTED"
                    or verification.run_id != artifact.run_id
                    or verification_execution is None
                    or verification_execution.status.value != "COMPLETED"
                    or verification_execution.result is None
                    or verification.artifact_id not in verification_execution.result.artifact_ids
                ):
                    raise ValueError(
                        f"Improvement artifact verification {verification_id!r} is missing, unsupported, or belongs to another run."
                    )
                if not set(verification.evidence_ids).issubset(set(artifact.evidence_ids)):
                    raise ValueError(f"Improvement artifact verification {verification_id!r} references unselected Evidence.")
            candidate_ids = [item.candidate_id for item in artifact.improvement_candidates]
            if len(candidate_ids) != len(set(candidate_ids)):
                raise ValueError("Improvement artifact contains duplicate candidate IDs.")
            existing_candidate_ids = {
                candidate.candidate_id
                for previous in self.research_artifacts
                for candidate in previous.improvement_candidates
            }
            if existing_candidate_ids.intersection(candidate_ids):
                raise ValueError("Improvement artifact contains a candidate ID already stored in ResearchState.")
        elif artifact.artifact_type == "prior_work_investigation":
            candidate_artifact = next(
                (item for item in self.research_artifacts if item.artifact_id == artifact.prior_work_candidate_artifact_id),
                None,
            )
            candidate = next(
                (
                    candidate
                    for candidate in (candidate_artifact.improvement_candidates if candidate_artifact else ())
                    if candidate.candidate_id == artifact.prior_work_candidate_id
                ),
                None,
            )
            if (
                candidate_artifact is None
                or candidate_artifact.artifact_type != "improvement_candidate"
                or candidate_artifact.run_id != artifact.run_id
                or candidate is None
                or candidate.status != "CANDIDATE"
                or candidate.novelty_status != "NOT_ASSESSED"
                or candidate.validation_status != "NOT_VALIDATED"
            ):
                raise ValueError("Prior-work artifact must reference a stored candidate from the same ResearchRun.")
            if artifact.evidence_ids:
                raise ValueError("Prior-work search results cannot be represented as acquired Evidence without capture provenance.")
            if not artifact.prior_work_queries:
                raise ValueError("Prior-work artifact must record its executed query set.")
            if len(artifact.search_result_ids) > artifact.prior_work_scope.maximum_results:
                raise ValueError("Prior-work artifact exceeds its total result budget.")
            searches = {item.search_id: item for item in self.searches}
            results_by_id = {item.result_id: item for item in self.search_results if item.result_id is not None}
            assessments = {item.search_result_id: item for item in artifact.prior_work_assessments}
            query_result_ids = []
            for query_record in artifact.prior_work_queries:
                search = searches.get(query_record.search_action_id)
                if (
                    search is None
                    or search.run_id != artifact.run_id
                    or search.query != query_record.query
                    or search.provider != artifact.prior_work_provider
                ):
                    raise ValueError("Prior-work query does not resolve to a SearchAction in the same run/provider.")
                action_results = [item for item in self.search_results if item.search_id == search.search_id]
                action_result_ids = tuple(item.result_id for item in action_results)
                if action_result_ids != query_record.search_result_ids or search.result_count != len(action_results):
                    raise ValueError("Prior-work query SearchResult IDs do not match its SearchAction.")
                if any(item.rank != rank for rank, item in enumerate(action_results, start=1)):
                    raise ValueError("Prior-work SearchResult ranks are not in persisted retrieval order.")
                query_result_ids.extend(query_record.search_result_ids)
            if tuple(query_result_ids) != artifact.search_result_ids:
                raise ValueError("Prior-work artifact SearchResult lineage does not match its executed queries.")
            if set(assessments) != set(artifact.search_result_ids):
                raise ValueError("Prior-work relevance assessments must cover every SearchResult exactly once.")
            for finding in artifact.prior_work_findings:
                result = results_by_id.get(finding.search_result_id)
                assessment = assessments.get(finding.search_result_id)
                if (
                    result is None
                    or assessment is None
                    or assessment.relevance == "NOT_RELEVANT"
                    or finding.source_id != result.source_id
                ):
                    raise ValueError("Prior-work finding must resolve to a relevant SearchResult and its Source.")
            if (artifact.prior_work_status == "ZERO_RESULTS") != (not artifact.search_result_ids):
                raise ValueError("Prior-work status does not match the recorded SearchResult count.")
        else:
            experiment = artifact.experiment_plan
            candidate_artifact = next(
                (item for item in self.research_artifacts if item.artifact_id == experiment.improvement_artifact_id),
                None,
            )
            candidate = next(
                (item for item in (candidate_artifact.improvement_candidates if candidate_artifact else ())
                 if item.candidate_id == experiment.candidate_id),
                None,
            )
            if (
                candidate_artifact is None or candidate_artifact.artifact_type != "improvement_candidate"
                or candidate_artifact.run_id != artifact.run_id or candidate is None
                or candidate.status != "CANDIDATE" or candidate.novelty_status != "NOT_ASSESSED"
                or candidate.validation_status != "NOT_VALIDATED"
            ):
                raise ValueError("Experiment plan must reference a stored candidate from the same ResearchRun.")
            if not set(candidate.supporting_evidence_ids).issubset(set(artifact.evidence_ids)):
                raise ValueError("Experiment plan must preserve the candidate's supporting Evidence IDs.")
            prior_artifacts = {item.artifact_id: item for item in self.research_artifacts}
            prior_result_ids = set()
            for prior_id in experiment.prior_work_artifact_ids:
                prior = prior_artifacts.get(prior_id)
                if prior is None or prior.artifact_type != "prior_work_investigation" or prior.run_id != artifact.run_id:
                    raise ValueError(f"Experiment plan prior-work artifact {prior_id!r} is missing or belongs to another run.")
                prior_result_ids.update(prior.search_result_ids)
            if not set(experiment.supporting_search_result_ids).issubset(prior_result_ids):
                raise ValueError("Experiment plan references SearchResults outside its prior-work artifacts.")
            for verification_id in experiment.verification_artifact_ids:
                verification = prior_artifacts.get(verification_id)
                if (
                    verification is None or verification.artifact_type != "verification_result"
                    or verification.run_id != artifact.run_id
                    or not set(verification.evidence_ids).issubset(set(artifact.evidence_ids))
                ):
                    raise ValueError(f"Experiment plan verification artifact {verification_id!r} is missing or inconsistent.")
            known_claims = {item.claim_id: item for item in self.claims}
            for claim_id in experiment.supporting_claim_ids:
                claim = known_claims.get(claim_id)
                if claim is None or claim.status != "SUPPORTED" or not set(claim.evidence_ids).issubset(set(artifact.evidence_ids)):
                    raise ValueError(f"Experiment plan claim {claim_id!r} is missing, unsupported, or has unresolved Evidence.")
            versions = {item.version_id: item for item in self.document_versions}
            passages = {item.passage_reference_id: item for item in self.passage_references}
            results = {item.result_id: item for item in self.search_results if item.result_id is not None}
            searches = {item.search_id: item for item in self.searches}
            for evidence_id in artifact.evidence_ids:
                evidence = known_evidence.get(evidence_id)
                version = versions.get(evidence.document_version_id or "") if evidence else None
                passage = passages.get(evidence.passage_reference_id or "") if evidence else None
                result = results.get(evidence.search_result_id or "") if evidence else None
                search = searches.get(result.search_id or "") if result else None
                if (
                    evidence is None or version is None or passage is None or result is None or search is None
                    or version.source_id != evidence.source_id or passage.document_version_id != version.version_id
                    or result.source_id != evidence.source_id or search.run_id != artifact.run_id
                ):
                    raise ValueError(f"Experiment plan Evidence {evidence_id!r} has invalid provenance/run lineage.")
        self.research_artifacts.append(artifact)
        self.touch()
        return artifact

    def add_question(
        self,
        text: str,
        priority: int = 1,
        *,
        run_id: Optional[str] = None,
        iteration_id: Optional[str] = None,
        parent_question_id: Optional[str] = None,
    ) -> ResearchQuestion:
        question = ResearchQuestion(
            text=text,
            priority=priority,
            run_id=run_id,
            iteration_id=iteration_id,
            parent_question_id=parent_question_id,
        )
        return self.add_research_question(question)

    def add_research_question(self, question: ResearchQuestion) -> ResearchQuestion:
        if question.question_id in {item.question_id for item in self.research_questions}:
            raise ValueError(f"Duplicate question_id {question.question_id!r}.")
        iterations = {item.iteration_id: item for item in self.iterations}
        runs = {item.run_id for item in self.research_runs}
        if question.iteration_id is not None:
            iteration = iterations.get(question.iteration_id)
            if iteration is None:
                raise ValueError(f"ResearchQuestion references unknown iteration_id {question.iteration_id!r}.")
            if question.run_id is not None and question.run_id != iteration.run_id:
                raise ValueError("ResearchQuestion run_id does not match its iteration's run_id.")
            question.run_id = iteration.run_id
        if question.run_id is not None and question.run_id not in runs:
            raise ValueError(f"ResearchQuestion references unknown run_id {question.run_id!r}.")
        if question.parent_question_id is not None:
            parent = next(
                (item for item in self.research_questions if item.question_id == question.parent_question_id),
                None,
            )
            if parent is None:
                raise ValueError(
                    f"ResearchQuestion references unknown parent_question_id {question.parent_question_id!r}."
                )
            if question.run_id is not None and parent.run_id != question.run_id:
                raise ValueError("Parent question must belong to the same research run.")
        self.research_questions.append(question)
        self.touch()
        return question

    def add_source(self, source: Source) -> Source:
        self.sources.append(source)
        self.touch()
        return source

    def add_document_version(self, version: DocumentVersion) -> DocumentVersion:
        if version.source_id not in {source.source_id for source in self.sources}:
            raise ValueError(f"DocumentVersion references unknown source_id {version.source_id!r}.")
        if version.version_id in {item.version_id for item in self.document_versions}:
            raise ValueError(f"Duplicate document version_id {version.version_id!r}.")
        self.document_versions.append(version)
        self.touch()
        return version

    def add_passage_reference(self, reference: PassageReference) -> PassageReference:
        if reference.document_version_id not in {item.version_id for item in self.document_versions}:
            raise ValueError(
                f"PassageReference references unknown document_version_id "
                f"{reference.document_version_id!r}."
            )
        if reference.passage_reference_id in {
            item.passage_reference_id for item in self.passage_references
        }:
            raise ValueError(
                f"Duplicate passage_reference_id {reference.passage_reference_id!r}."
            )
        self.passage_references.append(reference)
        self.touch()
        return reference

    def add_evidence(self, evidence: Evidence) -> Evidence:
        has_version = evidence.document_version_id is not None
        has_passage = evidence.passage_reference_id is not None
        if has_version != has_passage:
            raise ValueError(
                "Evidence provenance must provide both document_version_id and passage_reference_id."
            )
        if has_version:
            versions = {item.version_id: item for item in self.document_versions}
            passages = {
                item.passage_reference_id: item for item in self.passage_references
            }
            version = versions.get(evidence.document_version_id or "")
            reference = passages.get(evidence.passage_reference_id or "")
            if version is None:
                raise ValueError(
                    f"Evidence references unknown document_version_id "
                    f"{evidence.document_version_id!r}."
                )
            if reference is None:
                raise ValueError(
                    f"Evidence references unknown passage_reference_id "
                    f"{evidence.passage_reference_id!r}."
                )
            if reference.document_version_id != version.version_id:
                raise ValueError("Evidence passage and document version do not match.")
            if evidence.source_id != version.source_id:
                raise ValueError("Evidence source_id does not match its DocumentVersion source_id.")
            expected_hash = _passage_text_hash(evidence.text)
            if evidence.text_hash != expected_hash:
                raise ValueError("Evidence text_hash does not match its normalized evidence text.")
        if evidence.search_result_id is not None:
            result = next(
                (item for item in self.search_results if item.result_id == evidence.search_result_id),
                None,
            )
            if result is None:
                raise ValueError(f"Evidence references unknown search_result_id {evidence.search_result_id!r}.")
            if evidence.source_id != result.source_id:
                raise ValueError("Evidence source_id does not match its SearchResult source_id.")
        self.evidence.append(evidence)
        self.touch()
        return evidence

    def add_claim(self, claim: ResearchClaim) -> ResearchClaim:
        self.claims.append(claim)
        self.touch()
        return claim

    def add_search(self, search: SearchAction) -> SearchAction:
        self._apply_search_lineage(search)
        if search.search_id in {item.search_id for item in self.searches}:
            raise ValueError(f"Duplicate search_id {search.search_id!r}.")
        self.searches.append(search)
        self.touch()
        return search

    def add_search_result(self, result: "SearchResult") -> "SearchResult":
        # Linked persisted results need durable instance identity. Preserve
        # legacy unlinked provider results with no ID.
        if result.search_id is not None and result.result_id is None:
            result.result_id = _id("result")
        if result.result_id is not None and result.result_id in {
            item.result_id for item in self.search_results if item.result_id is not None
        }:
            raise ValueError(f"Duplicate result_id {result.result_id!r}.")
        if result.search_id is not None and result.search_id not in {
            item.search_id for item in self.searches
        }:
            raise ValueError(f"SearchResult references unknown search_id {result.search_id!r}.")
        if result.rank is not None and (
            not isinstance(result.rank, int) or isinstance(result.rank, bool) or result.rank < 1
        ):
            raise ValueError("SearchResult rank must be a positive integer when supplied.")
        self.search_results.append(result)
        self.touch()
        return result

    def _apply_search_lineage(self, search: SearchAction) -> None:
        runs = {item.run_id for item in self.research_runs}
        iterations = {item.iteration_id: item for item in self.iterations}
        questions = {item.question_id: item for item in self.research_questions}
        if search.iteration_id is not None:
            iteration = iterations.get(search.iteration_id)
            if iteration is None:
                raise ValueError(f"SearchAction references unknown iteration_id {search.iteration_id!r}.")
            if search.run_id is not None and search.run_id != iteration.run_id:
                raise ValueError("SearchAction run_id does not match its iteration's run_id.")
            search.run_id = iteration.run_id
        question = None
        if search.question_id is not None:
            question = questions.get(search.question_id)
            if question is None:
                raise ValueError(f"SearchAction references unknown question_id {search.question_id!r}.")
            if search.run_id is None and question.run_id is not None:
                search.run_id = question.run_id
        if search.run_id is not None and search.run_id not in runs:
            raise ValueError(f"SearchAction references unknown run_id {search.run_id!r}.")
        if question is not None and question.run_id is not None and search.run_id != question.run_id:
            raise ValueError("SearchAction and ResearchQuestion belong to different research runs.")

    def validate_provenance(self) -> List[str]:
        """Return integrity problems without rejecting legacy unlinked evidence."""
        errors: List[str] = []
        source_ids = {source.source_id for source in self.sources}
        versions = {item.version_id: item for item in self.document_versions}
        passages = {
            item.passage_reference_id: item for item in self.passage_references
        }

        if len(versions) != len(self.document_versions):
            errors.append("Duplicate DocumentVersion version_id.")
        if len(passages) != len(self.passage_references):
            errors.append("Duplicate PassageReference passage_reference_id.")

        for version in self.document_versions:
            if version.source_id not in source_ids:
                errors.append(
                    f"DocumentVersion {version.version_id!r} references missing source "
                    f"{version.source_id!r}."
                )
        for reference in self.passage_references:
            if reference.document_version_id not in versions:
                errors.append(
                    f"PassageReference {reference.passage_reference_id!r} references missing "
                    f"DocumentVersion {reference.document_version_id!r}."
                )
        for evidence in self.evidence:
            has_version = evidence.document_version_id is not None
            has_passage = evidence.passage_reference_id is not None
            if not has_version and not has_passage:
                continue  # Legacy evidence remains valid without typed provenance.
            if has_version != has_passage:
                errors.append(f"Evidence {evidence.evidence_id!r} has incomplete provenance links.")
                continue
            version = versions.get(evidence.document_version_id or "")
            reference = passages.get(evidence.passage_reference_id or "")
            if version is None:
                errors.append(f"Evidence {evidence.evidence_id!r} references a missing DocumentVersion.")
            if reference is None:
                errors.append(f"Evidence {evidence.evidence_id!r} references a missing PassageReference.")
            if version is not None and evidence.source_id != version.source_id:
                errors.append(f"Evidence {evidence.evidence_id!r} source_id mismatches its DocumentVersion.")
            if version is not None and reference is not None and reference.document_version_id != version.version_id:
                errors.append(f"Evidence {evidence.evidence_id!r} passage belongs to another DocumentVersion.")
            if evidence.text_hash != _passage_text_hash(evidence.text):
                errors.append(f"Evidence {evidence.evidence_id!r} text_hash does not match its text.")
        return errors

    def log(self, agent: str, action: str, message: str = "") -> AgentEvent:
        event = AgentEvent(agent=agent, action=action, message=message)
        self.events.append(event)
        self.touch()
        return event

    def request_more_research(self, question: str) -> None:
        if question not in self.unresolved_questions:
            self.unresolved_questions.append(question)
        self.touch()

    def resolve_research_question(self, question: str) -> None:
        self.unresolved_questions = [q for q in self.unresolved_questions if q != question]
        self.touch()

    def add_contradiction(self, description: str) -> None:
        if description not in self.contradictions:
            self.contradictions.append(description)
        self.touch()

    def summary(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "iteration": self.iteration,
            "research_questions": len(self.research_questions),
            "sources": len(self.sources),
            "evidence": len(self.evidence),
            "claims": len(self.claims),
            "searches": len(self.searches),
            "research_runs": len(self.research_runs),
            "iterations": len(self.iterations),
            "research_loops": len(self.research_loops),
            "unresolved_questions": len(self.unresolved_questions),
            "contradictions": len(self.contradictions),
            "errors": len(self.errors),
        }

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResearchState":
        state = cls(
            user_request=data.get("user_request", ""),
            research_objective=data.get("research_objective", ""),
            status=data.get("status", "created"),
            iteration=int(data.get("iteration", 0)),
            final_report=data.get("final_report", ""),
            created_at=data.get("created_at", _now()),
            updated_at=data.get("updated_at", _now()),
        )
        state.research_questions = [ResearchQuestion(**x) for x in data.get("research_questions", [])]
        state.sources = [Source(**x) for x in data.get("sources", [])]
        state.claims = [ResearchClaim(**x) for x in data.get("claims", [])]
        state.searches = [SearchAction(**x) for x in data.get("searches", [])]
        if data.get("search_results"):
            # Import lazily to keep the state module independent at import time.
            from research_tools import SearchResult

            state.search_results = [SearchResult(**x) for x in data["search_results"]]
        state.document_versions = [
            DocumentVersion(**x) for x in data.get("document_versions", [])
        ]
        state.passage_references = [
            PassageReference(**x) for x in data.get("passage_references", [])
        ]
        state.research_runs = [ResearchRun(**x) for x in data.get("research_runs", [])]
        state.iterations = [ResearchIteration(**x) for x in data.get("iterations", [])]
        state.research_artifacts = [
            ResearchArtifact.from_dict(x) for x in data.get("research_artifacts", [])
        ]
        raw_loops = data.get("research_loops", [])
        if not isinstance(raw_loops, (list, tuple)):
            raise ValueError("Malformed ResearchState research_loops collection.")
        if raw_loops:
            from research_loop import ResearchLoop

            state.research_loops = [ResearchLoop.from_dict(x) for x in raw_loops]
        if data.get("task_executions"):
            from research_task_execution import ResearchTaskExecution

            state.task_executions = [
                ResearchTaskExecution.from_dict(x) for x in data["task_executions"]
            ]
        state.evidence = [Evidence(**x) for x in data.get("evidence", [])]
        state.events = [AgentEvent(**x) for x in data.get("events", [])]
        state.unresolved_questions = list(data.get("unresolved_questions", []))
        state.contradictions = list(data.get("contradictions", []))
        state.errors = list(data.get("errors", []))
        provenance_errors = state.validate_provenance()
        if provenance_errors:
            raise ValueError("Invalid serialized provenance: " + "; ".join(provenance_errors))
        lineage_errors = state.validate_lineage()
        if lineage_errors:
            raise ValueError("Invalid serialized lineage: " + "; ".join(lineage_errors))
        return state

    def validate_lineage(self) -> List[str]:
        """Return lineage integrity issues while allowing legacy unlinked records."""
        errors: List[str] = []
        runs = {item.run_id: item for item in self.research_runs}
        iterations = {item.iteration_id: item for item in self.iterations}
        questions = {item.question_id: item for item in self.research_questions}
        searches = {item.search_id: item for item in self.searches}
        results = {item.result_id: item for item in self.search_results if item.result_id is not None}
        if len(runs) != len(self.research_runs):
            errors.append("Duplicate ResearchRun run_id.")
        if len(iterations) != len(self.iterations):
            errors.append("Duplicate ResearchIteration iteration_id.")
        if len(questions) != len(self.research_questions):
            errors.append("Duplicate ResearchQuestion question_id.")
        if len(searches) != len(self.searches):
            errors.append("Duplicate SearchAction search_id.")
        if len(results) != sum(item.result_id is not None for item in self.search_results):
            errors.append("Duplicate SearchResult result_id.")
        iteration_numbers = [(item.run_id, item.number) for item in self.iterations]
        if len(set(iteration_numbers)) != len(iteration_numbers):
            errors.append("Duplicate ResearchIteration number within a run.")
        for iteration in self.iterations:
            if iteration.run_id not in runs:
                errors.append(f"ResearchIteration {iteration.iteration_id!r} references missing ResearchRun.")
            if not isinstance(iteration.number, int) or isinstance(iteration.number, bool) or iteration.number < 1:
                errors.append(f"ResearchIteration {iteration.iteration_id!r} has invalid number.")
        for question in self.research_questions:
            if question.run_id is not None and question.run_id not in runs:
                errors.append(f"ResearchQuestion {question.question_id!r} references missing ResearchRun.")
            if question.iteration_id is not None:
                iteration = iterations.get(question.iteration_id)
                if iteration is None:
                    errors.append(f"ResearchQuestion {question.question_id!r} references missing iteration.")
                elif question.run_id != iteration.run_id:
                    errors.append(f"ResearchQuestion {question.question_id!r} run/iteration mismatch.")
            if question.parent_question_id is not None:
                parent = questions.get(question.parent_question_id)
                if parent is None:
                    errors.append(f"ResearchQuestion {question.question_id!r} references missing parent question.")
                elif question.run_id is not None and parent.run_id != question.run_id:
                    errors.append(f"ResearchQuestion {question.question_id!r} parent belongs to another run.")
        for search in self.searches:
            if search.run_id is not None and search.run_id not in runs:
                errors.append(f"SearchAction {search.search_id!r} references missing ResearchRun.")
            if search.iteration_id is not None:
                iteration = iterations.get(search.iteration_id)
                if iteration is None:
                    errors.append(f"SearchAction {search.search_id!r} references missing iteration.")
                elif search.run_id != iteration.run_id:
                    errors.append(f"SearchAction {search.search_id!r} run/iteration mismatch.")
            if search.question_id is not None:
                question = questions.get(search.question_id)
                if question is None:
                    errors.append(f"SearchAction {search.search_id!r} references missing question.")
                elif question.run_id is not None and search.run_id != question.run_id:
                    errors.append(f"SearchAction {search.search_id!r} question belongs to another run.")
        for result in self.search_results:
            if result.search_id is not None and result.search_id not in searches:
                errors.append(f"SearchResult {result.result_id!r} references missing SearchAction.")
        for evidence in self.evidence:
            if evidence.search_result_id is not None:
                result = results.get(evidence.search_result_id)
                if result is None:
                    errors.append(f"Evidence {evidence.evidence_id!r} references missing SearchResult.")
                elif evidence.source_id != result.source_id:
                    errors.append(f"Evidence {evidence.evidence_id!r} source mismatches its SearchResult.")
        execution_by_id: Dict[str, ResearchTaskExecution] = {}
        prior_executions: List[ResearchTaskExecution] = []
        for execution in self.task_executions:
            if execution.execution_id in execution_by_id:
                errors.append(f"Duplicate ResearchTaskExecution execution_id {execution.execution_id!r}.")
            else:
                execution_by_id[execution.execution_id] = execution
            run = runs.get(execution.run_id)
            if run is None:
                errors.append(f"ResearchTaskExecution {execution.execution_id!r} references missing ResearchRun.")
            elif (
                run.plan_id != execution.plan_id
                or run.plan_revision != execution.plan_revision
                or run.approval_review_id is None
            ):
                errors.append(f"ResearchTaskExecution {execution.execution_id!r} plan/run authorization mismatch.")
            elif run.authorized_task_ids is None or execution.task_id not in run.authorized_task_ids:
                errors.append(f"ResearchTaskExecution {execution.execution_id!r} task is not authorized on its run.")
            if execution.iteration_id is not None:
                iteration = iterations.get(execution.iteration_id)
                if iteration is None or iteration.run_id != execution.run_id:
                    errors.append(f"ResearchTaskExecution {execution.execution_id!r} has invalid iteration lineage.")
            try:
                execution._validate_lifecycle_shape()
            except (AttributeError, TypeError, ValueError) as exc:
                errors.append(f"ResearchTaskExecution {execution.execution_id!r} has invalid lifecycle: {exc}")
            if execution.retry_of_execution_id is not None:
                prior_target = next(
                    (item for item in prior_executions if item.execution_id == execution.retry_of_execution_id),
                    None,
                )
                if (
                    prior_target is None
                    or prior_target.status.value != "FAILED"
                    or prior_target.run_id != execution.run_id
                    or prior_target.task_id != execution.task_id
                ):
                    errors.append(f"ResearchTaskExecution {execution.execution_id!r} has invalid retry lineage.")
            if execution.continuation_of_execution_id is not None:
                prior_target = next(
                    (item for item in prior_executions if item.execution_id == execution.continuation_of_execution_id),
                    None,
                )
                authorizations = [
                    authorization
                    for loop in self.research_loops
                    for authorization in getattr(loop, "continuation_authorizations", ())
                    if authorization.authorization_id == execution.continuation_authorization_id
                ]
                prior_for_task = [item for item in prior_executions
                                  if item.run_id == execution.run_id and item.task_id == execution.task_id]
                if (prior_target is None or prior_target.status.value != "COMPLETED"
                        or not prior_for_task or prior_for_task[-1].execution_id != prior_target.execution_id
                        or prior_target.run_id != execution.run_id or prior_target.task_id != execution.task_id
                        or prior_target.plan_id != execution.plan_id
                        or prior_target.plan_revision != execution.plan_revision
                        or len(authorizations) != 1
                        or authorizations[0].previous_execution_id != prior_target.execution_id
                        or authorizations[0].run_id != execution.run_id
                        or authorizations[0].plan_id != execution.plan_id
                        or authorizations[0].plan_revision != execution.plan_revision
                        or authorizations[0].task_id != execution.task_id
                        or authorizations[0].iteration_id != execution.iteration_id):
                    errors.append(f"ResearchTaskExecution {execution.execution_id!r} has invalid continuation lineage.")
            prior_executions.append(execution)
        artifact_ids: set[str] = set()
        artifacts_by_execution: Dict[str, ResearchArtifact] = {}
        for artifact in self.research_artifacts:
            if artifact.artifact_id in artifact_ids:
                errors.append(f"Duplicate ResearchArtifact artifact_id {artifact.artifact_id!r}.")
            artifact_ids.add(artifact.artifact_id)
            if artifact.execution_id in artifacts_by_execution:
                errors.append(f"Duplicate ResearchArtifact for execution {artifact.execution_id!r}.")
            artifacts_by_execution[artifact.execution_id] = artifact
            execution = execution_by_id.get(artifact.execution_id)
            if execution is None:
                errors.append(f"ResearchArtifact {artifact.artifact_id!r} references missing execution.")
                continue
            if execution.run_id != artifact.run_id or execution.task_id != artifact.task_id:
                errors.append(f"ResearchArtifact {artifact.artifact_id!r} execution lineage mismatch.")
            evidence_by_id = {item.evidence_id: item for item in self.evidence}
            if artifact.artifact_type == "retrieval_result":
                search = searches.get(artifact.search_action_id or "")
                if search is None or search.run_id != artifact.run_id:
                    errors.append(f"ResearchArtifact {artifact.artifact_id!r} references invalid SearchAction.")
                    continue
                action_results = [item for item in self.search_results if item.search_id == search.search_id]
                if tuple(item.result_id for item in action_results) != artifact.search_result_ids:
                    errors.append(f"ResearchArtifact {artifact.artifact_id!r} SearchResult references mismatch.")
                if any(
                    evidence_id not in evidence_by_id
                    or evidence_by_id[evidence_id].search_result_id not in set(artifact.search_result_ids)
                    for evidence_id in artifact.evidence_ids
                ):
                    errors.append(f"ResearchArtifact {artifact.artifact_id!r} has invalid Evidence references.")
            elif artifact.artifact_type == "verification_result":
                if not artifact.verification_claim or not artifact.verification_verdict:
                    errors.append(f"Verification ResearchArtifact {artifact.artifact_id!r} has incomplete result data.")
                for evidence_id in artifact.evidence_ids:
                    evidence = evidence_by_id.get(evidence_id)
                    if evidence is None:
                        errors.append(f"Verification ResearchArtifact {artifact.artifact_id!r} references missing Evidence.")
                    elif (
                        evidence.document_version_id not in {item.version_id for item in self.document_versions}
                        or evidence.passage_reference_id not in {item.passage_reference_id for item in self.passage_references}
                        or evidence.search_result_id not in results
                    ):
                        errors.append(f"Verification ResearchArtifact {artifact.artifact_id!r} has incomplete Evidence provenance.")
                    else:
                        result = results[evidence.search_result_id]
                        search = searches.get(result.search_id or "")
                        if search is None or (search.run_id is not None and search.run_id != artifact.run_id):
                            errors.append(f"Verification ResearchArtifact {artifact.artifact_id!r} has invalid run/search lineage.")
                assessed_ids = {
                    assessment.evidence_id
                    for unit in artifact.verification_assessments
                    for assessment in unit.evidence_assessments
                }
                if not assessed_ids.issubset(set(artifact.evidence_ids)):
                    errors.append(f"Verification ResearchArtifact {artifact.artifact_id!r} has dangling assessments.")
            elif artifact.artifact_type == "improvement_candidate":
                if (
                    not artifact.improvement_problem_statement
                    or artifact.improvement_problem_origin != "researcher_provided"
                    or not artifact.evidence_ids
                ):
                    errors.append(f"Improvement ResearchArtifact {artifact.artifact_id!r} has incomplete input lineage.")
                for evidence_id in artifact.evidence_ids:
                    evidence = evidence_by_id.get(evidence_id)
                    if evidence is None:
                        errors.append(f"Improvement ResearchArtifact {artifact.artifact_id!r} references missing Evidence.")
                        continue
                    if (
                        evidence.document_version_id not in {item.version_id for item in self.document_versions}
                        or evidence.passage_reference_id not in {item.passage_reference_id for item in self.passage_references}
                        or evidence.search_result_id not in results
                    ):
                        errors.append(f"Improvement ResearchArtifact {artifact.artifact_id!r} has incomplete Evidence provenance.")
                        continue
                    result = results[evidence.search_result_id]
                    if result.source_id != evidence.source_id:
                        errors.append(f"Improvement ResearchArtifact {artifact.artifact_id!r} Evidence source mismatch.")
                    search = searches.get(result.search_id or "")
                    if search is None or search.run_id != artifact.run_id:
                        errors.append(f"Improvement ResearchArtifact {artifact.artifact_id!r} has invalid run/search lineage.")
                claims = {item.claim_id: item for item in self.claims}
                for claim_id in artifact.improvement_claim_ids:
                    claim = claims.get(claim_id)
                    if claim is None or claim.status != "SUPPORTED":
                        errors.append(f"Improvement ResearchArtifact {artifact.artifact_id!r} references a missing or unsupported claim.")
                    elif not set(claim.evidence_ids).issubset(set(artifact.evidence_ids)):
                        errors.append(f"Improvement ResearchArtifact {artifact.artifact_id!r} claim references unselected Evidence.")
                for verification_id in artifact.improvement_verification_artifact_ids:
                    verification = next(
                        (item for item in self.research_artifacts if item.artifact_id == verification_id),
                        None,
                    )
                    verification_execution = (
                        execution_by_id.get(verification.execution_id) if verification is not None else None
                    )
                    if (
                        verification is None
                        or verification.artifact_type != "verification_result"
                        or verification.verification_verdict != "SUPPORTED"
                        or verification.run_id != artifact.run_id
                        or not set(verification.evidence_ids).issubset(set(artifact.evidence_ids))
                        or verification_execution is None
                        or verification_execution.status.value != "COMPLETED"
                        or verification_execution.result is None
                        or verification.artifact_id not in verification_execution.result.artifact_ids
                    ):
                        errors.append(f"Improvement ResearchArtifact {artifact.artifact_id!r} references invalid verification input.")
                candidate_ids = [item.candidate_id for item in artifact.improvement_candidates]
                if len(candidate_ids) != len(set(candidate_ids)):
                    errors.append(f"Improvement ResearchArtifact {artifact.artifact_id!r} contains duplicate candidate IDs.")
                for candidate in artifact.improvement_candidates:
                    if (
                        candidate.problem_statement != artifact.improvement_problem_statement
                        or not set(candidate.supporting_evidence_ids).issubset(set(artifact.evidence_ids))
                        or not set(candidate.supporting_claim_ids).issubset(set(artifact.improvement_claim_ids))
                        or not set(candidate.supporting_verification_artifact_ids).issubset(set(artifact.improvement_verification_artifact_ids))
                    ):
                        errors.append(f"Improvement ResearchArtifact {artifact.artifact_id!r} candidate references mismatch.")
            elif artifact.artifact_type == "prior_work_investigation":
                candidate_artifact = next(
                    (item for item in self.research_artifacts if item.artifact_id == artifact.prior_work_candidate_artifact_id),
                    None,
                )
                candidate = next(
                    (item for item in (candidate_artifact.improvement_candidates if candidate_artifact else ())
                     if item.candidate_id == artifact.prior_work_candidate_id),
                    None,
                )
                if (
                    candidate_artifact is None
                    or candidate_artifact.artifact_type != "improvement_candidate"
                    or candidate_artifact.run_id != artifact.run_id
                    or candidate is None
                    or candidate.status != "CANDIDATE"
                    or candidate.novelty_status != "NOT_ASSESSED"
                    or candidate.validation_status != "NOT_VALIDATED"
                ):
                    errors.append(f"Prior-work ResearchArtifact {artifact.artifact_id!r} references an invalid candidate.")
                if artifact.evidence_ids:
                    errors.append(f"Prior-work ResearchArtifact {artifact.artifact_id!r} incorrectly stores Evidence.")
                if not artifact.prior_work_queries or len(artifact.prior_work_queries) > artifact.prior_work_scope.maximum_queries:
                    errors.append(f"Prior-work ResearchArtifact {artifact.artifact_id!r} has invalid query coverage.")
                if len(artifact.search_result_ids) > artifact.prior_work_scope.maximum_results:
                    errors.append(f"Prior-work ResearchArtifact {artifact.artifact_id!r} exceeds its result budget.")
                query_result_ids = []
                assessments = {item.search_result_id: item for item in artifact.prior_work_assessments}
                for query_record in artifact.prior_work_queries:
                    search = searches.get(query_record.search_action_id)
                    if (
                        search is None or search.run_id != artifact.run_id
                        or search.query != query_record.query or search.provider != artifact.prior_work_provider
                    ):
                        errors.append(f"Prior-work ResearchArtifact {artifact.artifact_id!r} has invalid SearchAction lineage.")
                        continue
                    action_results = [item for item in self.search_results if item.search_id == search.search_id]
                    ids = tuple(item.result_id for item in action_results)
                    if (ids != query_record.search_result_ids
                            or search.result_count != len(action_results)
                            or any(item.rank != index for index, item in enumerate(action_results, 1))):
                        errors.append(f"Prior-work ResearchArtifact {artifact.artifact_id!r} has invalid SearchResult order.")
                    query_result_ids.extend(query_record.search_result_ids)
                if tuple(query_result_ids) != artifact.search_result_ids:
                    errors.append(f"Prior-work ResearchArtifact {artifact.artifact_id!r} SearchResult references mismatch.")
                if set(assessments) != set(artifact.search_result_ids):
                    errors.append(f"Prior-work ResearchArtifact {artifact.artifact_id!r} assessments do not cover SearchResults.")
                for finding in artifact.prior_work_findings:
                    result = results.get(finding.search_result_id)
                    assessment = assessments.get(finding.search_result_id)
                    if (
                        result is None or assessment is None or assessment.relevance == "NOT_RELEVANT"
                        or assessment.relevance != finding.relevance or result.source_id != finding.source_id
                    ):
                        errors.append(f"Prior-work ResearchArtifact {artifact.artifact_id!r} has invalid finding lineage.")
                if (artifact.prior_work_status == "ZERO_RESULTS") != (not artifact.search_result_ids):
                    errors.append(f"Prior-work ResearchArtifact {artifact.artifact_id!r} has inconsistent result status.")
            else:
                experiment = artifact.experiment_plan
                improvement = next(
                    (item for item in self.research_artifacts if item.artifact_id == experiment.improvement_artifact_id),
                    None,
                )
                candidate = next(
                    (item for item in (improvement.improvement_candidates if improvement else ())
                     if item.candidate_id == experiment.candidate_id),
                    None,
                )
                if improvement is None or improvement.artifact_type != "improvement_candidate" or improvement.run_id != artifact.run_id or candidate is None:
                    errors.append(f"Experiment plan {artifact.artifact_id!r} has invalid improvement-candidate lineage.")
                elif not set(candidate.supporting_evidence_ids).issubset(set(artifact.evidence_ids)):
                    errors.append(f"Experiment plan {artifact.artifact_id!r} omits candidate supporting Evidence.")
                prior_ids = set()
                for prior_id in experiment.prior_work_artifact_ids:
                    prior = next((item for item in self.research_artifacts if item.artifact_id == prior_id), None)
                    if prior is None or prior.artifact_type != "prior_work_investigation" or prior.run_id != artifact.run_id:
                        errors.append(f"Experiment plan {artifact.artifact_id!r} has invalid prior-work reference.")
                    else:
                        prior_ids.update(prior.search_result_ids)
                if not set(experiment.supporting_search_result_ids).issubset(prior_ids):
                    errors.append(f"Experiment plan {artifact.artifact_id!r} has dangling prior-work SearchResult references.")
                for verification_id in experiment.verification_artifact_ids:
                    verification = next((item for item in self.research_artifacts if item.artifact_id == verification_id), None)
                    if (
                        verification is None or verification.artifact_type != "verification_result"
                        or verification.run_id != artifact.run_id
                        or not set(verification.evidence_ids).issubset(set(artifact.evidence_ids))
                    ):
                        errors.append(f"Experiment plan {artifact.artifact_id!r} has invalid verification artifact reference.")
                claims = {item.claim_id: item for item in self.claims}
                for claim_id in experiment.supporting_claim_ids:
                    claim = claims.get(claim_id)
                    if claim is None or claim.status != "SUPPORTED" or not set(claim.evidence_ids).issubset(set(artifact.evidence_ids)):
                        errors.append(f"Experiment plan {artifact.artifact_id!r} has invalid supporting claim reference.")
                for evidence_id in artifact.evidence_ids:
                    evidence = evidence_by_id.get(evidence_id)
                    if evidence is None:
                        errors.append(f"Experiment plan {artifact.artifact_id!r} references missing Evidence.")
                        continue
                    result = results.get(evidence.search_result_id or "")
                    search = searches.get(result.search_id or "") if result else None
                    if (
                        evidence.document_version_id not in {item.version_id for item in self.document_versions}
                        or evidence.passage_reference_id not in {item.passage_reference_id for item in self.passage_references}
                        or result is None or search is None or search.run_id != artifact.run_id
                        or result.source_id != evidence.source_id
                    ):
                        errors.append(f"Experiment plan {artifact.artifact_id!r} Evidence has invalid provenance/run lineage.")
            if execution.status.value != "COMPLETED" or execution.result is None:
                errors.append(f"ResearchArtifact {artifact.artifact_id!r} has no completed execution.")
            elif (
                artifact.artifact_id not in execution.result.artifact_ids
                or any(evidence_id not in execution.result.evidence_ids for evidence_id in artifact.evidence_ids)
            ):
                errors.append(f"ResearchArtifact {artifact.artifact_id!r} is not linked from its execution result.")
        loop_ids = [getattr(loop, "loop_id", None) for loop in self.research_loops]
        if len(loop_ids) != len(set(loop_ids)):
            errors.append("Duplicate ResearchLoop loop_id.")
        for loop in self.research_loops:
            try:
                errors.extend(loop.validate_references(self))
            except (AttributeError, TypeError, ValueError) as exc:
                errors.append(f"ResearchLoop {getattr(loop, 'loop_id', '<unknown>')!r} is malformed: {exc}")
        return errors


__all__ = [
    "ResearchQuestion",
    "ResearchRun",
    "ResearchIteration",
    "Source",
    "DocumentVersion",
    "PassageReference",
    "Evidence",
    "ResearchClaim",
    "SearchAction",
    "AgentEvent",
    "ResearchImprovementCandidate",
    "PriorWorkSearchScope",
    "PriorWorkQueryRecord",
    "PriorWorkResultAssessment",
    "PriorWorkFinding",
    "ExperimentDesignElement",
    "ExperimentMetric",
    "ResearchExperimentPlan",
    "ResearchArtifact",
    "ResearchState",
]
