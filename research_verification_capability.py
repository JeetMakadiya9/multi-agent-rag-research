"""Explicit, authorized verification of already stored research Evidence.

This adapter delegates stance assessment and deterministic verdicts to the
existing Experiment B implementation. It never retrieves or creates Evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from claim_verification import (
    CONTRADICTED,
    INSUFFICIENT_EVIDENCE,
    SUPPORTED,
    ClaimVerification,
    EvidenceItem,
    VerificationReport,
)
from experiment_b_verification import (
    ExperimentBConfig,
    VALID_VERDICTS,
    run_experiment_b_verification,
)
from llm import LLMProvider
from research_capabilities import CapabilityType
from research_plan_review import PlanReview
from research_planning import ResearchPlan, ResearchTask
from research_state import (
    Evidence,
    ResearchArtifact,
    ResearchRun,
    ResearchState,
    VerificationEvidenceAssessment,
    VerificationUnitAssessment,
)
from research_task_execution import (
    ResearchTaskExecution,
    TaskExecutionContext,
    TaskExecutionError,
    TaskExecutionRecordingError,
    TaskExecutionResult,
    execute_authorized_task,
)
from verification_units import analyze_verification_units


_VALID_VERDICTS = {SUPPORTED, CONTRADICTED, INSUFFICIENT_EVIDENCE}


class VerificationCapabilityError(ValueError):
    """A verification request or its referenced stored Evidence is invalid."""


@dataclass(frozen=True)
class EvidenceVerificationRequest:
    """A claim and IDs of existing Evidence records; raw text is not accepted."""

    claim: str
    evidence_ids: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.claim, str) or not self.claim.strip():
            raise VerificationCapabilityError("EVIDENCE_VERIFICATION claim must be non-empty.")
        if not isinstance(self.evidence_ids, (tuple, list)):
            raise VerificationCapabilityError("EVIDENCE_VERIFICATION evidence_ids must be a sequence of IDs.")
        if not self.evidence_ids:
            raise VerificationCapabilityError("EVIDENCE_VERIFICATION requires at least one evidence_id.")
        ids = tuple(self.evidence_ids)
        if any(not isinstance(item, str) or not item.strip() for item in ids):
            raise VerificationCapabilityError("EVIDENCE_VERIFICATION evidence_ids must contain non-empty IDs.")
        if len(ids) != len(set(ids)):
            raise VerificationCapabilityError("EVIDENCE_VERIFICATION evidence_ids must not contain duplicates.")
        normalized_claim = self.claim.strip()
        units = analyze_verification_units(normalized_claim)
        if len(units) != 1:
            raise VerificationCapabilityError("EVIDENCE_VERIFICATION accepts one claim unit per request.")
        object.__setattr__(self, "claim", normalized_claim)
        object.__setattr__(self, "evidence_ids", ids)


@dataclass(frozen=True)
class VerificationCapabilityResult:
    """Completed Experiment B assessment and its persisted artifact reference."""

    capability: CapabilityType
    execution_id: str
    run_id: str
    task_id: str
    artifact_id: str
    claim: str
    verdict: str
    evidence_ids: Tuple[str, ...]
    assessments: Tuple[VerificationUnitAssessment, ...]
    explanation: str
    success: bool = True

    def __post_init__(self) -> None:
        if self.capability is not CapabilityType.EVIDENCE_VERIFICATION:
            raise ValueError("VerificationCapabilityResult must identify EVIDENCE_VERIFICATION.")
        for name in ("execution_id", "run_id", "task_id", "artifact_id", "claim"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"VerificationCapabilityResult.{name} must be non-empty.")
        if self.verdict not in _VALID_VERDICTS:
            raise ValueError("VerificationCapabilityResult has an invalid Experiment B verdict.")
        evidence_ids = tuple(self.evidence_ids)
        if not evidence_ids or len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("VerificationCapabilityResult evidence_ids must be non-empty and unique.")
        object.__setattr__(self, "evidence_ids", evidence_ids)
        assessments = tuple(self.assessments)
        if not assessments or any(not isinstance(item, VerificationUnitAssessment) for item in assessments):
            raise ValueError("VerificationCapabilityResult assessments must be typed and non-empty.")
        object.__setattr__(self, "assessments", assessments)
        if not isinstance(self.explanation, str):
            raise ValueError("VerificationCapabilityResult.explanation must be a string.")
        if self.success is not True:
            raise ValueError("A returned VerificationCapabilityResult represents a successful execution.")

    def to_dict(self) -> dict[str, object]:
        return {
            "capability": self.capability.value,
            "execution_id": self.execution_id,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "artifact_id": self.artifact_id,
            "claim": self.claim,
            "verdict": self.verdict,
            "evidence_ids": list(self.evidence_ids),
            "assessments": [item.to_dict() for item in self.assessments],
            "explanation": self.explanation,
            "success": self.success,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "VerificationCapabilityResult":
        required = {
            "capability", "execution_id", "run_id", "task_id", "artifact_id", "claim", "verdict",
            "evidence_ids", "assessments", "explanation", "success",
        }
        if not isinstance(data, dict) or set(data) != required:
            raise ValueError("Malformed VerificationCapabilityResult payload.")
        values = dict(data)
        try:
            values["capability"] = CapabilityType(values["capability"])
            values["evidence_ids"] = tuple(values["evidence_ids"])  # type: ignore[arg-type]
            values["assessments"] = tuple(
                VerificationUnitAssessment.from_dict(item) for item in values["assessments"]  # type: ignore[union-attr]
            )
            return cls(**values)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Malformed VerificationCapabilityResult payload: {exc}") from exc


class _VerificationStateAccess:
    """Read-only provenance view plus the one allowed artifact write."""

    __slots__ = ("__state",)

    def __init__(self, state: ResearchState):
        self.__state = state

    @property
    def evidence(self):
        return tuple(self.__state.evidence)

    @property
    def sources(self):
        return tuple(self.__state.sources)

    @property
    def document_versions(self):
        return tuple(self.__state.document_versions)

    @property
    def passage_references(self):
        return tuple(self.__state.passage_references)

    @property
    def search_results(self):
        return tuple(self.__state.search_results)

    @property
    def searches(self):
        return tuple(self.__state.searches)

    def validate_provenance(self):
        return self.__state.validate_provenance()

    def add_research_artifact(self, artifact: ResearchArtifact) -> ResearchArtifact:
        return self.__state.add_research_artifact(artifact)


class _ArtifactSnapshot:
    def __init__(self, state: ResearchState):
        self.state = state
        self.length = len(state.research_artifacts)

    def rollback(self) -> None:
        del self.state.research_artifacts[self.length:]


def execute_evidence_verification(
    plan: ResearchPlan,
    run: ResearchRun,
    task: ResearchTask,
    review: PlanReview,
    request: EvidenceVerificationRequest,
    state: ResearchState,
    llm_provider: LLMProvider,
    *,
    iteration_id: Optional[str] = None,
    retry_of_execution_id: Optional[str] = None,
    continuation_authorization_id: Optional[str] = None,
) -> VerificationCapabilityResult:
    """Verify existing Evidence through Phase 5 and frozen Experiment B.

    ``llm_provider`` is mandatory, so this API never silently chooses an
    Ollama/network provider. Experiment B receives no retrieval provider and
    cannot fetch additional evidence.
    """
    _validate_entry(request, state, llm_provider)
    snapshot = _ArtifactSnapshot(state)
    access = _VerificationStateAccess(state)
    produced: dict[str, object] = {}

    def _verify(_task: ResearchTask, context: TaskExecutionContext) -> TaskExecutionResult:
        evidence = _resolve_evidence(access, request.evidence_ids, context.run_id)
        normalized = tuple(_to_experiment_b_item(item, access) for item in evidence)
        config = ExperimentBConfig(
            max_evidence_items=len(normalized),
            max_judge_retries=0,
            unit_specific_retrieval=False,
            max_targeted_retrieval_retries=0,
            answer_revision_enabled=False,
        )
        report = run_experiment_b_verification(
            request.claim,
            request.claim,
            normalized,
            retrieval_provider=None,
            llm_provider=llm_provider,
            config=config,
        )
        assessment = _normalize_report(report, request, evidence)
        artifact = ResearchArtifact(
            execution_id=context.execution_id,
            run_id=context.run_id,
            task_id=context.task_id,
            search_action_id=None,
            evidence_ids=request.evidence_ids,
            artifact_type="verification_result",
            capability=CapabilityType.EVIDENCE_VERIFICATION.value,
            verification_claim=request.claim,
            verification_verdict=assessment[0].verdict,
            verification_assessments=assessment,
        )
        access.add_research_artifact(artifact)
        produced["artifact"] = artifact
        produced["assessments"] = assessment
        produced["explanation"] = assessment[0].explanation
        output = f"Experiment B completed with deterministic verdict {assessment[0].verdict}."
        return TaskExecutionResult(
            success=True,
            output=output,
            artifact_ids=(artifact.artifact_id,),
            evidence_ids=request.evidence_ids,
        )

    try:
        execution = execute_authorized_task(
            plan,
            run,
            task,
            review,
            _verify,
            state,
            iteration_id=iteration_id,
            retry_of_execution_id=retry_of_execution_id,
            continuation_authorization_id=continuation_authorization_id,
        )
    except (TaskExecutionError, TaskExecutionRecordingError):
        snapshot.rollback()
        raise

    artifact = produced.get("artifact")
    assessments = produced.get("assessments")
    if not isinstance(artifact, ResearchArtifact) or not isinstance(assessments, tuple):
        snapshot.rollback()
        raise RuntimeError("EVIDENCE_VERIFICATION completed without a persisted verification artifact.")
    return VerificationCapabilityResult(
        capability=CapabilityType.EVIDENCE_VERIFICATION,
        execution_id=execution.execution_id,
        run_id=execution.run_id,
        task_id=execution.task_id,
        artifact_id=artifact.artifact_id,
        claim=request.claim,
        verdict=artifact.verification_verdict or "",
        evidence_ids=artifact.evidence_ids,
        assessments=assessments,
        explanation=str(produced["explanation"]),
    )


def _validate_entry(
    request: EvidenceVerificationRequest,
    state: ResearchState,
    llm_provider: LLMProvider,
) -> None:
    if not isinstance(request, EvidenceVerificationRequest):
        raise VerificationCapabilityError("An EvidenceVerificationRequest is required.")
    try:
        EvidenceVerificationRequest(request.claim, request.evidence_ids)
    except (AttributeError, TypeError, ValueError) as exc:
        raise VerificationCapabilityError(f"Invalid EVIDENCE_VERIFICATION request: {exc}") from exc
    if not isinstance(state, ResearchState):
        raise VerificationCapabilityError("A ResearchState is required to resolve Evidence and persist the artifact.")
    if not callable(getattr(llm_provider, "generate", None)):
        raise VerificationCapabilityError("An explicit LLMProvider with generate() is required by Experiment B.")


def _resolve_evidence(
    state: _VerificationStateAccess,
    evidence_ids: Sequence[str],
    run_id: str,
) -> tuple[Evidence, ...]:
    known = {item.evidence_id: item for item in state.evidence}
    missing = [item for item in evidence_ids if item not in known]
    if missing:
        raise VerificationCapabilityError("Unknown ResearchState Evidence ID(s): " + ", ".join(missing))
    provenance_errors = state.validate_provenance()
    if provenance_errors:
        raise VerificationCapabilityError("ResearchState provenance is invalid: " + "; ".join(provenance_errors))
    versions = {item.version_id: item for item in state.document_versions}
    passages = {item.passage_reference_id: item for item in state.passage_references}
    sources = {item.source_id for item in state.sources}
    results = {item.result_id: item for item in state.search_results if item.result_id is not None}
    searches = {item.search_id: item for item in state.searches}
    selected = tuple(known[item] for item in evidence_ids)
    for evidence in selected:
        if not isinstance(evidence.text, str) or not evidence.text.strip():
            raise VerificationCapabilityError(f"Evidence {evidence.evidence_id!r} has no usable stored text.")
        version = versions.get(evidence.document_version_id or "")
        passage = passages.get(evidence.passage_reference_id or "")
        result = results.get(evidence.search_result_id or "")
        if evidence.source_id not in sources or version is None or passage is None or result is None:
            raise VerificationCapabilityError(
                f"Evidence {evidence.evidence_id!r} lacks complete Source/SearchResult/DocumentVersion/PassageReference provenance."
            )
        if version.source_id != evidence.source_id or passage.document_version_id != version.version_id:
            raise VerificationCapabilityError(f"Evidence {evidence.evidence_id!r} has mismatched document provenance.")
        if result.source_id != evidence.source_id:
            raise VerificationCapabilityError(f"Evidence {evidence.evidence_id!r} source mismatches its SearchResult.")
        search = searches.get(result.search_id or "")
        if search is None or (search.run_id is not None and search.run_id != run_id):
            raise VerificationCapabilityError(
                f"Evidence {evidence.evidence_id!r} SearchAction is missing or belongs to another ResearchRun."
            )
    return selected


def _to_experiment_b_item(evidence: Evidence, state: _VerificationStateAccess) -> EvidenceItem:
    version = next(item for item in state.document_versions if item.version_id == evidence.document_version_id)
    return EvidenceItem(
        evidence_id=evidence.evidence_id,
        text=evidence.text,
        source=evidence.source_id,
        page=evidence.page,
        score=float(evidence.relevance_score or 0.0),
        metadata=dict(evidence.metadata),
        source_id=evidence.source_id,
        document_id=version.document_id,
        chunk_id=evidence.chunk_id or "",
    )


def _normalize_report(
    report: VerificationReport,
    request: EvidenceVerificationRequest,
    evidence: Sequence[Evidence],
) -> Tuple[VerificationUnitAssessment, ...]:
    if not isinstance(report, VerificationReport):
        raise VerificationCapabilityError("Experiment B returned a malformed VerificationReport.")
    if report.errors:
        raise VerificationCapabilityError("Experiment B assessment failed: " + "; ".join(report.errors))
    if len(report.verifications) != 1 or not isinstance(report.verifications[0], ClaimVerification):
        raise VerificationCapabilityError("Experiment B must return exactly one typed claim assessment.")
    if report.original_answer != request.claim:
        raise VerificationCapabilityError("Experiment B report does not match the requested claim.")
    verification = report.verifications[0]
    if verification.claim != request.claim:
        raise VerificationCapabilityError("Experiment B assessment does not match the requested claim.")
    if verification.verdict not in VALID_VERDICTS or verification.verdict not in _VALID_VERDICTS:
        raise VerificationCapabilityError("Experiment B returned an invalid deterministic verdict.")
    if not verification.assessment_complete or verification.missing_assessments:
        raise VerificationCapabilityError("Experiment B returned an incomplete evidence assessment.")
    if not isinstance(verification.evidence_assessments, list):
        raise VerificationCapabilityError("Experiment B returned malformed evidence assessment rows.")
    units = analyze_verification_units(request.claim)
    evidence_by_index = {f"E{index}": item for index, item in enumerate(evidence, start=1)}
    stance_records = []
    seen_pairs: set[tuple[str, str]] = set()
    for row in verification.evidence_assessments:
        if not isinstance(row, dict):
            raise VerificationCapabilityError("Experiment B returned a malformed evidence assessment row.")
        evidence_index = row.get("evidence_index")
        component_ids = row.get("component_indices")
        stance = row.get("stance")
        if evidence_index not in evidence_by_index or not isinstance(component_ids, list) or not component_ids:
            raise VerificationCapabilityError("Experiment B assessment references an unknown evidence/component.")
        if stance not in {"SUPPORTS", "CONTRADICTS", "NEUTRAL"}:
            raise VerificationCapabilityError("Experiment B returned an invalid evidence stance.")
        known_components = {item.component_id for item in units[0].components}
        if any(component_id not in known_components for component_id in component_ids):
            raise VerificationCapabilityError("Experiment B assessment references an unknown verification component.")
        evidence_id = evidence_by_index[evidence_index].evidence_id
        for component_id in component_ids:
            pair = (evidence_id, component_id)
            if pair in seen_pairs:
                raise VerificationCapabilityError("Experiment B returned duplicate evidence/component assessments.")
            seen_pairs.add(pair)
        stance_records.append(VerificationEvidenceAssessment(
            evidence_id=evidence_id,
            component_ids=tuple(component_ids),
            stance=stance,
        ))
    expected_pairs = {
        (item.evidence_id, component.component_id)
        for item in evidence
        for component in units[0].components
    }
    if seen_pairs != expected_pairs:
        raise VerificationCapabilityError("Experiment B assessment did not cover every supplied evidence/component pair.")
    assessment = VerificationUnitAssessment(
        claim_id=verification.claim_id,
        claim=verification.claim,
        verdict=verification.verdict,
        confidence=verification.confidence,
        explanation=verification.explanation,
        evidence_assessments=tuple(stance_records),
        validator_overrides=tuple(verification.validator_overrides),
        assessment_complete=verification.assessment_complete,
        missing_assessments=tuple(verification.missing_assessments),
    )
    return (assessment,)


__all__ = [
    "EvidenceVerificationRequest",
    "VerificationCapabilityResult",
    "VerificationCapabilityError",
    "execute_evidence_verification",
]
