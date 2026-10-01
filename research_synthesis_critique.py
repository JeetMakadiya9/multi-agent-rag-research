"""Read-only, caller-invoked synthesis and critique over stored run artifacts.

No retrieval, provider calls, capability dispatch, task mutation, or research
conclusion is produced from unstored inputs. Results are typed snapshots whose
references resolve to existing ResearchState records.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
import re
from typing import Optional, Sequence
from uuid import uuid4

from research_loop import LoopStage, ResearchLoop, ResearchLoopStatus
from research_plan_review import PlanReview, is_execution_approved
from research_planning import ResearchPlan
from research_state import ResearchArtifact, ResearchRun, ResearchState
from research_task_execution import TaskExecutionStatus


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _enum(enum_type, value, name):
    try:
        return value if isinstance(value, enum_type) else enum_type(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unsupported {name}: {value!r}.") from exc


def _strings(value, name, *, unique=True):
    if not isinstance(value, (tuple, list)):
        raise ValueError(f"{name} must be a sequence of strings.")
    values = tuple(value)
    if any(not isinstance(item, str) or not item.strip() for item in values):
        raise ValueError(f"{name} must contain non-empty strings.")
    if unique and len(values) != len(set(values)):
        raise ValueError(f"{name} must not contain duplicate values.")
    return values


def _timestamp(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a timezone-aware ISO-8601 timestamp.")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a timezone-aware ISO-8601 timestamp.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must include a timezone.")


class FindingClassification(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    UNVERIFIED = "UNVERIFIED"
    DOCUMENTED_LIMITATION = "DOCUMENTED_LIMITATION"
    UNRESOLVED_CONTRADICTION = "UNRESOLVED_CONTRADICTION"


class FindingOrigin(str, Enum):
    VERIFICATION_ARTIFACT = "VERIFICATION_ARTIFACT"
    RESEARCH_CLAIM = "RESEARCH_CLAIM"
    RESEARCHER_PROVIDED_CONTEXT = "RESEARCHER_PROVIDED_CONTEXT"
    STORED_PRIOR_WORK = "STORED_PRIOR_WORK"
    STORED_CONTRADICTION = "STORED_CONTRADICTION"


class QuestionCoverageStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    PARTIALLY_ADDRESSED = "PARTIALLY_ADDRESSED"
    UNRESOLVED_CONTRADICTION = "UNRESOLVED_CONTRADICTION"
    UNRESOLVED = "UNRESOLVED"


class CritiqueIssueType(str, Enum):
    UNVERIFIED_CLAIM = "UNVERIFIED_CLAIM"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    CONTRADICTORY_EVIDENCE = "CONTRADICTORY_EVIDENCE"
    MISSING_COVERAGE = "MISSING_COVERAGE"
    PROVENANCE_PROBLEM = "PROVENANCE_PROBLEM"
    MISSING_EXPERIMENTAL_VALIDATION = "MISSING_EXPERIMENTAL_VALIDATION"
    DUPLICATED_FINDING = "DUPLICATED_FINDING"


@dataclass(frozen=True)
class EvidenceProvenanceReference:
    evidence_id: str
    source_id: str
    document_version_id: Optional[str]
    passage_reference_id: Optional[str]
    search_result_id: Optional[str]
    search_action_id: Optional[str]
    text_hash: Optional[str]

    def __post_init__(self):
        for name in ("evidence_id", "source_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"EvidenceProvenanceReference.{name} must be non-empty.")
        for name in ("document_version_id", "passage_reference_id", "search_result_id", "search_action_id", "text_hash"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"EvidenceProvenanceReference.{name} must be non-empty when supplied.")

    def to_dict(self):
        return {name: getattr(self, name) for name in (
            "evidence_id", "source_id", "document_version_id", "passage_reference_id",
            "search_result_id", "search_action_id", "text_hash",
        )}

    @classmethod
    def from_dict(cls, data):
        keys = {"evidence_id", "source_id", "document_version_id", "passage_reference_id",
                "search_result_id", "search_action_id", "text_hash"}
        if not isinstance(data, dict) or set(data) != keys:
            raise ValueError("Malformed EvidenceProvenanceReference payload.")
        return cls(**data)


@dataclass(frozen=True)
class SynthesisFinding:
    statement: str
    classification: FindingClassification
    evidence_ids: tuple[str, ...] = ()
    claim_ids: tuple[str, ...] = ()
    verification_artifact_ids: tuple[str, ...] = ()
    task_ids: tuple[str, ...] = ()
    question_id: Optional[str] = None
    source_artifact_ids: tuple[str, ...] = ()
    origin: FindingOrigin = FindingOrigin.VERIFICATION_ARTIFACT
    finding_id: str = field(default_factory=lambda: _id("finding"))

    def __post_init__(self):
        if not isinstance(self.statement, str) or not self.statement.strip():
            raise ValueError("SynthesisFinding.statement must be non-empty.")
        object.__setattr__(self, "classification", _enum(FindingClassification, self.classification, "finding classification"))
        object.__setattr__(self, "origin", _enum(FindingOrigin, self.origin, "finding origin"))
        for name in ("evidence_ids", "claim_ids", "verification_artifact_ids", "task_ids", "source_artifact_ids"):
            object.__setattr__(self, name, _strings(getattr(self, name), f"SynthesisFinding.{name}"))
        if not isinstance(self.finding_id, str) or not self.finding_id.strip():
            raise ValueError("SynthesisFinding.finding_id must be non-empty.")
        if self.question_id is not None and (not isinstance(self.question_id, str) or not self.question_id.strip()):
            raise ValueError("SynthesisFinding.question_id must be non-empty when supplied.")
        if not any((self.evidence_ids, self.claim_ids, self.verification_artifact_ids,
                    self.task_ids, self.question_id, self.source_artifact_ids)):
            raise ValueError("Every synthesis finding must reference existing research context.")

    def to_dict(self):
        return {"statement": self.statement, "classification": self.classification.value,
                "evidence_ids": list(self.evidence_ids), "claim_ids": list(self.claim_ids),
                "verification_artifact_ids": list(self.verification_artifact_ids),
                "task_ids": list(self.task_ids), "question_id": self.question_id,
                "source_artifact_ids": list(self.source_artifact_ids), "origin": self.origin.value,
                "finding_id": self.finding_id}

    @classmethod
    def from_dict(cls, data):
        keys = {"statement", "classification", "evidence_ids", "claim_ids",
                "verification_artifact_ids", "task_ids", "question_id", "source_artifact_ids", "origin", "finding_id"}
        if not isinstance(data, dict) or set(data) != keys:
            raise ValueError("Malformed SynthesisFinding payload.")
        return cls(**{**data, "evidence_ids": tuple(data["evidence_ids"]),
                      "claim_ids": tuple(data["claim_ids"]),
                      "verification_artifact_ids": tuple(data["verification_artifact_ids"]),
                      "task_ids": tuple(data["task_ids"]),
                      "source_artifact_ids": tuple(data["source_artifact_ids"])})


@dataclass(frozen=True)
class QuestionCoverage:
    question_id: str
    question: str
    status: QuestionCoverageStatus
    task_ids: tuple[str, ...] = ()
    verification_artifact_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self):
        for name in ("question_id", "question"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"QuestionCoverage.{name} must be non-empty.")
        object.__setattr__(self, "status", _enum(QuestionCoverageStatus, self.status, "question coverage status"))
        for name in ("task_ids", "verification_artifact_ids", "evidence_ids"):
            object.__setattr__(self, name, _strings(getattr(self, name), f"QuestionCoverage.{name}"))

    def to_dict(self):
        return {"question_id": self.question_id, "question": self.question, "status": self.status.value,
                "task_ids": list(self.task_ids), "verification_artifact_ids": list(self.verification_artifact_ids),
                "evidence_ids": list(self.evidence_ids)}

    @classmethod
    def from_dict(cls, data):
        keys = {"question_id", "question", "status", "task_ids", "verification_artifact_ids", "evidence_ids"}
        if not isinstance(data, dict) or set(data) != keys:
            raise ValueError("Malformed QuestionCoverage payload.")
        return cls(**{**data, "task_ids": tuple(data["task_ids"]),
                      "verification_artifact_ids": tuple(data["verification_artifact_ids"]),
                      "evidence_ids": tuple(data["evidence_ids"])})


@dataclass(frozen=True)
class ResearchSynthesis:
    run_id: str
    plan_id: str
    plan_revision: int
    loop_id: str
    iteration_id: str
    objective: str
    source_artifact_ids: tuple[str, ...]
    evidence_references: tuple[EvidenceProvenanceReference, ...]
    findings: tuple[SynthesisFinding, ...]
    question_coverage: tuple[QuestionCoverage, ...]
    claim_ids: tuple[str, ...] = ()
    candidate_improvement_ids: tuple[str, ...] = ()
    prior_work_finding_ids: tuple[str, ...] = ()
    experiment_plan_artifact_ids: tuple[str, ...] = ()
    synthesis_id: str = field(default_factory=lambda: _id("synthesis"))
    generated_at: str = field(default_factory=_now)

    def __post_init__(self):
        for name in ("run_id", "plan_id", "loop_id", "iteration_id", "objective", "synthesis_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"ResearchSynthesis.{name} must be non-empty.")
        if not isinstance(self.plan_revision, int) or isinstance(self.plan_revision, bool) or self.plan_revision < 1:
            raise ValueError("ResearchSynthesis.plan_revision must be positive.")
        _timestamp(self.generated_at, "ResearchSynthesis.generated_at")
        for name in ("source_artifact_ids", "claim_ids", "candidate_improvement_ids",
                     "prior_work_finding_ids", "experiment_plan_artifact_ids"):
            object.__setattr__(self, name, _strings(getattr(self, name), f"ResearchSynthesis.{name}"))
        for name, typ in (("evidence_references", EvidenceProvenanceReference),
                          ("findings", SynthesisFinding), ("question_coverage", QuestionCoverage)):
            values = tuple(getattr(self, name))
            if any(not isinstance(value, typ) for value in values):
                raise ValueError(f"ResearchSynthesis.{name} must contain {typ.__name__} values.")
            object.__setattr__(self, name, values)
        if len({item.evidence_id for item in self.evidence_references}) != len(self.evidence_references):
            raise ValueError("ResearchSynthesis evidence references must be unique.")

    def to_dict(self):
        return {"run_id": self.run_id, "plan_id": self.plan_id, "plan_revision": self.plan_revision,
                "loop_id": self.loop_id, "iteration_id": self.iteration_id, "objective": self.objective,
                "source_artifact_ids": list(self.source_artifact_ids),
                "evidence_references": [item.to_dict() for item in self.evidence_references],
                "findings": [item.to_dict() for item in self.findings],
                "question_coverage": [item.to_dict() for item in self.question_coverage],
                "claim_ids": list(self.claim_ids), "candidate_improvement_ids": list(self.candidate_improvement_ids),
                "prior_work_finding_ids": list(self.prior_work_finding_ids),
                "experiment_plan_artifact_ids": list(self.experiment_plan_artifact_ids),
                "synthesis_id": self.synthesis_id, "generated_at": self.generated_at}

    def to_json(self):
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, data):
        keys = {"run_id", "plan_id", "plan_revision", "loop_id", "iteration_id", "objective",
                "source_artifact_ids", "evidence_references", "findings", "question_coverage", "claim_ids",
                "candidate_improvement_ids", "prior_work_finding_ids", "experiment_plan_artifact_ids",
                "synthesis_id", "generated_at"}
        if not isinstance(data, dict) or set(data) != keys:
            raise ValueError("Malformed ResearchSynthesis payload.")
        try:
            return cls(**{**data,
                "source_artifact_ids": tuple(data["source_artifact_ids"]),
                "evidence_references": tuple(EvidenceProvenanceReference.from_dict(x) for x in data["evidence_references"]),
                "findings": tuple(SynthesisFinding.from_dict(x) for x in data["findings"]),
                "question_coverage": tuple(QuestionCoverage.from_dict(x) for x in data["question_coverage"]),
                "claim_ids": tuple(data["claim_ids"]),
                "candidate_improvement_ids": tuple(data["candidate_improvement_ids"]),
                "prior_work_finding_ids": tuple(data["prior_work_finding_ids"]),
                "experiment_plan_artifact_ids": tuple(data["experiment_plan_artifact_ids"])})
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Malformed ResearchSynthesis: {exc}") from exc

    @classmethod
    def from_json(cls, payload):
        try:
            return cls.from_dict(json.loads(payload))
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Malformed ResearchSynthesis JSON: {exc}") from exc


@dataclass(frozen=True)
class CritiqueIssue:
    issue_type: CritiqueIssueType
    description: str
    evidence_ids: tuple[str, ...] = ()
    claim_ids: tuple[str, ...] = ()
    artifact_ids: tuple[str, ...] = ()
    question_id: Optional[str] = None
    issue_id: str = field(default_factory=lambda: _id("issue"))

    def __post_init__(self):
        object.__setattr__(self, "issue_type", _enum(CritiqueIssueType, self.issue_type, "critique issue type"))
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("CritiqueIssue.description must be non-empty.")
        for name in ("evidence_ids", "claim_ids", "artifact_ids"):
            object.__setattr__(self, name, _strings(getattr(self, name), f"CritiqueIssue.{name}"))
        if not isinstance(self.issue_id, str) or not self.issue_id.strip():
            raise ValueError("CritiqueIssue.issue_id must be non-empty.")
        if self.question_id is not None and (not isinstance(self.question_id, str) or not self.question_id.strip()):
            raise ValueError("CritiqueIssue.question_id must be non-empty when supplied.")

    def to_dict(self):
        return {"issue_type": self.issue_type.value, "description": self.description,
                "evidence_ids": list(self.evidence_ids), "claim_ids": list(self.claim_ids),
                "artifact_ids": list(self.artifact_ids), "question_id": self.question_id,
                "issue_id": self.issue_id}

    @classmethod
    def from_dict(cls, data):
        keys = {"issue_type", "description", "evidence_ids", "claim_ids", "artifact_ids", "question_id", "issue_id"}
        if not isinstance(data, dict) or set(data) != keys:
            raise ValueError("Malformed CritiqueIssue payload.")
        return cls(**{**data, "evidence_ids": tuple(data["evidence_ids"]),
                      "claim_ids": tuple(data["claim_ids"]), "artifact_ids": tuple(data["artifact_ids"])})


@dataclass(frozen=True)
class ResearchCritique:
    synthesis_id: str
    run_id: str
    plan_id: str
    plan_revision: int
    loop_id: str
    iteration_id: str
    source_artifact_ids: tuple[str, ...]
    issues: tuple[CritiqueIssue, ...]
    candidate_improvement_ids: tuple[str, ...] = ()
    novelty_status: str = "NOT_ASSESSED"
    critique_id: str = field(default_factory=lambda: _id("critique"))
    created_at: str = field(default_factory=_now)

    def __post_init__(self):
        for name in ("synthesis_id", "run_id", "plan_id", "loop_id", "iteration_id", "critique_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"ResearchCritique.{name} must be non-empty.")
        if not isinstance(self.plan_revision, int) or isinstance(self.plan_revision, bool) or self.plan_revision < 1:
            raise ValueError("ResearchCritique.plan_revision must be positive.")
        if self.novelty_status != "NOT_ASSESSED":
            raise ValueError("ResearchCritique cannot declare a novelty assessment.")
        _timestamp(self.created_at, "ResearchCritique.created_at")
        object.__setattr__(self, "source_artifact_ids", _strings(self.source_artifact_ids, "ResearchCritique.source_artifact_ids"))
        object.__setattr__(self, "candidate_improvement_ids", _strings(self.candidate_improvement_ids, "ResearchCritique.candidate_improvement_ids"))
        values = tuple(self.issues)
        if any(not isinstance(item, CritiqueIssue) for item in values):
            raise ValueError("ResearchCritique.issues must contain CritiqueIssue values.")
        object.__setattr__(self, "issues", values)

    def to_dict(self):
        return {"synthesis_id": self.synthesis_id, "run_id": self.run_id, "plan_id": self.plan_id,
                "plan_revision": self.plan_revision, "loop_id": self.loop_id, "iteration_id": self.iteration_id,
                "source_artifact_ids": list(self.source_artifact_ids), "issues": [item.to_dict() for item in self.issues],
                "candidate_improvement_ids": list(self.candidate_improvement_ids),
                "novelty_status": self.novelty_status, "critique_id": self.critique_id, "created_at": self.created_at}

    def to_json(self):
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, data):
        keys = {"synthesis_id", "run_id", "plan_id", "plan_revision", "loop_id", "iteration_id",
                "source_artifact_ids", "issues", "candidate_improvement_ids", "novelty_status", "critique_id", "created_at"}
        if not isinstance(data, dict) or set(data) != keys:
            raise ValueError("Malformed ResearchCritique payload.")
        try:
            return cls(**{**data, "source_artifact_ids": tuple(data["source_artifact_ids"]),
                          "issues": tuple(CritiqueIssue.from_dict(x) for x in data["issues"]),
                          "candidate_improvement_ids": tuple(data["candidate_improvement_ids"])})
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Malformed ResearchCritique: {exc}") from exc

    @classmethod
    def from_json(cls, payload):
        try:
            return cls.from_dict(json.loads(payload))
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Malformed ResearchCritique JSON: {exc}") from exc


def synthesize_research_state(plan: ResearchPlan, run: ResearchRun, review: PlanReview,
                              loop: ResearchLoop, state: ResearchState, *, iteration_id: str,
                              artifact_ids: Optional[Sequence[str]] = None) -> ResearchSynthesis:
    """Return a deterministic synthesis of completed artifacts in one approved run."""
    _validate_context(plan, run, review, loop, state, iteration_id, LoopStage.SYNTHESIS)
    artifacts = _resolve_artifacts(plan, run, state, artifact_ids)
    executions = {item.execution_id: item for item in state.task_executions}
    task_by_id = {item.task_id: item for item in plan.tasks}
    evidence_ids = tuple(dict.fromkeys(eid for item in artifacts for eid in item.evidence_ids))
    evidence_by_id = {item.evidence_id: item for item in state.evidence}
    evidence_refs = tuple(_evidence_reference(state, evidence_by_id[eid]) for eid in evidence_ids)
    claims = _linked_claims(state, evidence_ids)
    findings = []
    verification_artifacts = [item for item in artifacts if item.artifact_type == "verification_result"]
    for artifact in verification_artifacts:
        execution = executions[artifact.execution_id]
        task = task_by_id[execution.task_id]
        status = FindingClassification(artifact.verification_verdict)
        linked_claim_ids = tuple(claim.claim_id for claim in claims if _same_text(claim.text, artifact.verification_claim))
        findings.append(SynthesisFinding(
            statement=artifact.verification_claim or "", classification=status,
            evidence_ids=artifact.evidence_ids, claim_ids=linked_claim_ids,
            verification_artifact_ids=(artifact.artifact_id,), task_ids=(task.task_id,),
            question_id=task.question_id,
        ))
    verified_claim_texts = {_same_text_key(item.verification_claim) for item in verification_artifacts}
    for claim in claims:
        if _same_text_key(claim.text) not in verified_claim_texts:
            findings.append(SynthesisFinding(
                statement=claim.text, classification=_claim_classification(claim.status),
            evidence_ids=tuple(claim.evidence_ids), claim_ids=(claim.claim_id,),
                origin=FindingOrigin.RESEARCH_CLAIM,
            ))
    findings.extend(_stored_context_findings(artifacts, executions, task_by_id))
    findings.extend(_all_contradiction_findings(findings, verification_artifacts, executions, task_by_id))
    coverage = _question_coverage(plan, artifacts, executions)
    candidate_ids = tuple(candidate.candidate_id for artifact in artifacts
                          for candidate in artifact.improvement_candidates)
    prior_finding_ids = tuple(finding.finding_id for artifact in artifacts
                              for finding in artifact.prior_work_findings)
    experiment_ids = tuple(item.artifact_id for item in artifacts if item.artifact_type == "experiment_plan")
    return ResearchSynthesis(
        run_id=run.run_id, plan_id=plan.plan_id, plan_revision=plan.revision,
        loop_id=loop.loop_id, iteration_id=iteration_id, objective=plan.objective.text,
        source_artifact_ids=tuple(item.artifact_id for item in artifacts),
        evidence_references=evidence_refs, findings=tuple(findings), question_coverage=coverage,
        claim_ids=tuple(item.claim_id for item in claims), candidate_improvement_ids=candidate_ids,
        prior_work_finding_ids=prior_finding_ids, experiment_plan_artifact_ids=experiment_ids,
    )


def critique_research_synthesis(plan: ResearchPlan, run: ResearchRun, review: PlanReview,
                                loop: ResearchLoop, state: ResearchState, synthesis: ResearchSynthesis,
                                *, iteration_id: str) -> ResearchCritique:
    """Critique one synthesis against its same-run stored inputs; no follow-up is run."""
    _validate_context(plan, run, review, loop, state, iteration_id, LoopStage.CRITIQUE)
    if not isinstance(synthesis, ResearchSynthesis):
        raise ValueError("A typed ResearchSynthesis is required.")
    if (synthesis.run_id != run.run_id or synthesis.plan_id != plan.plan_id
            or synthesis.plan_revision != plan.revision or synthesis.loop_id != loop.loop_id
            or synthesis.iteration_id != iteration_id):
        raise ValueError("ResearchSynthesis belongs to another run, plan revision, loop, or iteration.")
    artifacts = _resolve_artifacts(plan, run, state, synthesis.source_artifact_ids)
    if tuple(item.artifact_id for item in artifacts) != synthesis.source_artifact_ids:
        raise ValueError("ResearchSynthesis source artifact ordering is stale or mismatched.")
    current = _synthesis_reference_snapshot(plan, run, loop, iteration_id, artifacts, state)
    _validate_synthesis_references(synthesis, current, plan, artifacts, state)
    issues = []
    verification_artifacts = [item for item in artifacts if item.artifact_type == "verification_result"]
    claims = _linked_claims(state, tuple(ref.evidence_id for ref in synthesis.evidence_references))
    verified_claim_texts = {_same_text_key(item.verification_claim) for item in verification_artifacts}
    for claim in claims:
        if _same_text_key(claim.text) not in verified_claim_texts:
            issues.append(CritiqueIssue(
                CritiqueIssueType.UNVERIFIED_CLAIM,
                "Stored claim has no matching verification artifact in this synthesis; this does not establish that it is false.",
                tuple(claim.evidence_ids), (claim.claim_id,), (),
            ))
    for artifact in verification_artifacts:
        if artifact.verification_verdict == "INSUFFICIENT_EVIDENCE":
            issues.append(CritiqueIssue(
                CritiqueIssueType.INSUFFICIENT_EVIDENCE,
                "Stored verification result reports insufficient evidence for this claim.",
                artifact.evidence_ids, (), (artifact.artifact_id,),
            ))
    for finding in synthesis.findings:
        if finding.classification is FindingClassification.UNRESOLVED_CONTRADICTION:
            issues.append(CritiqueIssue(
                CritiqueIssueType.CONTRADICTORY_EVIDENCE,
                "Stored verification artifacts disagree on this claim; no result was selected as the winner.",
                finding.evidence_ids, finding.claim_ids, finding.verification_artifact_ids,
                finding.question_id,
            ))
    for coverage in synthesis.question_coverage:
        if coverage.status in {QuestionCoverageStatus.UNRESOLVED, QuestionCoverageStatus.PARTIALLY_ADDRESSED,
                               QuestionCoverageStatus.INSUFFICIENT_EVIDENCE,
                               QuestionCoverageStatus.UNRESOLVED_CONTRADICTION}:
            issues.append(CritiqueIssue(
                CritiqueIssueType.MISSING_COVERAGE,
                f"Planned question coverage is {coverage.status.value}; task completion is not treated as evidence coverage.",
                coverage.evidence_ids, (), coverage.verification_artifact_ids, coverage.question_id,
            ))
    incomplete_refs = [item.evidence_id for item in synthesis.evidence_references
                       if not all((item.source_id, item.document_version_id, item.passage_reference_id,
                                   item.search_result_id, item.search_action_id, item.text_hash))]
    if incomplete_refs:
        issues.append(CritiqueIssue(
            CritiqueIssueType.PROVENANCE_PROBLEM,
            "Some selected Evidence lacks one or more Source/SearchResult/DocumentVersion/PassageReference links.",
            tuple(incomplete_refs), (), synthesis.source_artifact_ids,
        ))
    candidate_ids = synthesis.candidate_improvement_ids
    experiment_candidate_ids = {
        artifact.experiment_plan.candidate_id for artifact in artifacts
        if artifact.artifact_type == "experiment_plan" and artifact.experiment_plan is not None
    }
    for artifact in artifacts:
        if artifact.artifact_type != "improvement_candidate":
            continue
        for candidate in artifact.improvement_candidates:
            if candidate.candidate_id not in experiment_candidate_ids:
                issues.append(CritiqueIssue(
                    CritiqueIssueType.MISSING_EXPERIMENTAL_VALIDATION,
                    "Stored candidate remains NOT_VALIDATED and has no linked experiment plan; no experiment was created.",
                    candidate.supporting_evidence_ids, candidate.supporting_claim_ids,
                    (artifact.artifact_id,),
                ))
    return ResearchCritique(
        synthesis_id=synthesis.synthesis_id, run_id=run.run_id, plan_id=plan.plan_id,
        plan_revision=plan.revision, loop_id=loop.loop_id, iteration_id=iteration_id,
        source_artifact_ids=synthesis.source_artifact_ids, issues=tuple(issues),
        candidate_improvement_ids=candidate_ids,
    )


def _validate_context(plan, run, review, loop, state, iteration_id, required_stage):
    if not isinstance(plan, ResearchPlan) or plan.validate():
        raise ValueError("A structurally valid ResearchPlan is required.")
    if not isinstance(run, ResearchRun) or not isinstance(review, PlanReview):
        raise ValueError("ResearchRun and PlanReview are required.")
    if not isinstance(loop, ResearchLoop) or not isinstance(state, ResearchState):
        raise ValueError("A typed ResearchLoop and ResearchState are required.")
    review.validate_against(plan)
    if not is_execution_approved(plan, review):
        raise ValueError("PlanReview does not approve this exact plan revision.")
    if (run.plan_id != plan.plan_id or run.plan_revision != plan.revision
            or run.approval_review_id != review.review_id or run.status not in {"created", "running"}
            or loop.run_id != run.run_id or loop.plan_id != plan.plan_id
            or loop.plan_revision != plan.revision):
        raise ValueError("Run/loop/review do not match the exact approved plan revision.")
    if not any(item is run for item in state.research_runs):
        raise ValueError("ResearchRun must be registered in ResearchState.")
    if not any(item == loop for item in state.research_loops):
        raise ValueError("ResearchLoop must be registered in ResearchState.")
    if loop.status is not ResearchLoopStatus.RUNNING or loop.current_stage is not required_stage:
        raise ValueError(f"Controlled analysis requires a RUNNING loop at {required_stage.value} stage.")
    if not isinstance(iteration_id, str) or not iteration_id or loop.current_iteration_id != iteration_id:
        raise ValueError("Analysis iteration must be the loop's active ResearchIteration.")
    iteration = next((item for item in state.iterations if item.iteration_id == iteration_id), None)
    loop_iteration = next((item for item in loop.loop_iterations if item.iteration_id == iteration_id), None)
    if (iteration is None or iteration.run_id != run.run_id or iteration.status != "running"
            or loop_iteration is None or loop_iteration.status.value != "RUNNING"):
        raise ValueError("Analysis requires an active iteration belonging to the supplied ResearchRun.")
    tasks = {item.task_id for item in plan.tasks}
    authorized = set(run.authorized_task_ids or ())
    if not authorized or not authorized.issubset(tasks):
        raise ValueError("ResearchRun task authorization does not resolve to tasks in the approved plan.")
    if not set(loop_iteration.task_ids).issubset(authorized):
        raise ValueError("Active loop iteration includes tasks not authorized for this run.")
    errors = loop.validate_references(state)
    if errors:
        raise ValueError("ResearchLoop references are invalid: " + "; ".join(errors))
    lineage_errors = state.validate_lineage()
    if lineage_errors:
        raise ValueError("ResearchState lineage is invalid: " + "; ".join(lineage_errors))


def _resolve_artifacts(plan, run, state, artifact_ids):
    by_id = {item.artifact_id: item for item in state.research_artifacts}
    if artifact_ids is None:
        candidates = [item for item in state.research_artifacts if item.run_id == run.run_id]
    else:
        ids = _strings(artifact_ids, "artifact_ids")
        candidates = []
        for artifact_id in ids:
            artifact = by_id.get(artifact_id)
            if artifact is None:
                raise ValueError(f"Unknown ResearchArtifact ID: {artifact_id}")
            if artifact.run_id != run.run_id:
                raise ValueError(f"ResearchArtifact {artifact_id!r} belongs to another ResearchRun.")
            candidates.append(artifact)
    task_ids = {item.task_id for item in plan.tasks}
    authorized = set(run.authorized_task_ids or ())
    executions = {item.execution_id: item for item in state.task_executions}
    result = []
    for artifact in candidates:
        execution = executions.get(artifact.execution_id)
        if (artifact.run_id != run.run_id or artifact.task_id not in task_ids
                or artifact.task_id not in authorized or execution is None
                or execution.status is not TaskExecutionStatus.COMPLETED
                or artifact.artifact_id not in (execution.result.artifact_ids if execution.result else ())):
            raise ValueError(f"ResearchArtifact {artifact.artifact_id!r} lacks completed authorized same-run execution lineage.")
        result.append(artifact)
    return tuple(result)


def _evidence_reference(state, evidence):
    result = next((item for item in state.search_results if item.result_id == evidence.search_result_id), None)
    search = next((item for item in state.searches if result is not None and item.search_id == result.search_id), None)
    return EvidenceProvenanceReference(
        evidence_id=evidence.evidence_id, source_id=evidence.source_id,
        document_version_id=evidence.document_version_id,
        passage_reference_id=evidence.passage_reference_id,
        search_result_id=evidence.search_result_id,
        search_action_id=search.search_id if search is not None else None,
        text_hash=evidence.text_hash,
    )


def _linked_claims(state, evidence_ids):
    selected = set(evidence_ids)
    if not selected:
        return ()
    return tuple(item for item in state.claims if item.evidence_ids and set(item.evidence_ids).issubset(selected))


def _same_text(left, right):
    return _same_text_key(left) == _same_text_key(right)


def _same_text_key(value):
    return " ".join(value.casefold().split())


def _claim_classification(status):
    if not isinstance(status, str):
        raise ValueError("Stored ResearchClaim status must be a string.")
    # ResearchClaim.status is not a substitute for a same-run, persisted
    # deterministic verification artifact. Unmatched claims remain unverified.
    return FindingClassification.UNVERIFIED


def _stored_context_findings(artifacts, executions, tasks):
    """Expose stored context without presenting it as verified fact."""
    result = []
    for artifact in artifacts:
        execution = executions[artifact.execution_id]
        task = tasks[execution.task_id]
        if artifact.artifact_type == "improvement_candidate":
            result.append(SynthesisFinding(
                statement=artifact.improvement_problem_statement or "",
                classification=FindingClassification.UNVERIFIED,
                evidence_ids=artifact.evidence_ids, task_ids=(task.task_id,),
                question_id=task.question_id, source_artifact_ids=(artifact.artifact_id,),
                origin=FindingOrigin.RESEARCHER_PROVIDED_CONTEXT,
            ))
        elif artifact.artifact_type == "prior_work_investigation":
            for prior_finding in artifact.prior_work_findings:
                for limitation in prior_finding.limitations:
                    result.append(SynthesisFinding(
                        statement=limitation,
                        classification=FindingClassification.DOCUMENTED_LIMITATION,
                        task_ids=(task.task_id,), question_id=task.question_id,
                        source_artifact_ids=(artifact.artifact_id,),
                        origin=FindingOrigin.STORED_PRIOR_WORK,
                    ))
    return result


def _question_coverage(plan, artifacts, executions):
    tasks = {item.task_id: item for item in plan.tasks}
    verification_by_task = {}
    for artifact in artifacts:
        if artifact.artifact_type == "verification_result":
            execution = executions[artifact.execution_id]
            verification_by_task.setdefault(execution.task_id, []).append(artifact)
    result = []
    for question in plan.questions:
        task_ids = tuple(item.task_id for item in plan.tasks if item.question_id == question.question_id)
        all_verifications = [artifact for task_id in task_ids for artifact in verification_by_task.get(task_id, ())]
        evidence_ids = tuple(dict.fromkeys(eid for item in all_verifications for eid in item.evidence_ids))
        artifact_ids = tuple(item.artifact_id for item in all_verifications)
        assessed_tasks = {executions[item.execution_id].task_id for item in all_verifications}
        if not all_verifications:
            status = QuestionCoverageStatus.UNRESOLVED
        elif len(assessed_tasks) < len(task_ids):
            status = QuestionCoverageStatus.PARTIALLY_ADDRESSED
        else:
            verdicts = {item.verification_verdict for item in all_verifications}
            if any(_has_conflicting_verdicts(group) for group in _group_claims(all_verifications).values()):
                status = QuestionCoverageStatus.UNRESOLVED_CONTRADICTION
            elif "INSUFFICIENT_EVIDENCE" in verdicts:
                status = QuestionCoverageStatus.INSUFFICIENT_EVIDENCE
            elif verdicts == {"SUPPORTED"}:
                status = QuestionCoverageStatus.SUPPORTED
            elif verdicts == {"CONTRADICTED"}:
                status = QuestionCoverageStatus.CONTRADICTED
            else:
                status = QuestionCoverageStatus.PARTIALLY_ADDRESSED
        result.append(QuestionCoverage(question.question_id, question.text, status,
                                       task_ids, artifact_ids, evidence_ids))
    return tuple(result)


def _claim_key(artifact):
    return re.sub(r"\s+", " ", (artifact.verification_claim or "").casefold()).strip()


def _group_claims(artifacts):
    groups = {}
    for item in artifacts:
        groups.setdefault(_claim_key(item), []).append(item)
    return groups


def _has_conflicting_verdicts(items):
    values = {item.verification_verdict for item in items}
    return "SUPPORTED" in values and "CONTRADICTED" in values


def _contradiction_findings(findings):
    by_claim = {}
    for finding in findings:
        by_claim.setdefault(" ".join(finding.statement.casefold().split()), []).append(finding)
    result = []
    for group in by_claim.values():
        statuses = {item.classification for item in group}
        if FindingClassification.SUPPORTED in statuses and FindingClassification.CONTRADICTED in statuses:
            result.append(SynthesisFinding(
                statement=group[0].statement,
                classification=FindingClassification.UNRESOLVED_CONTRADICTION,
                evidence_ids=tuple(dict.fromkeys(eid for item in group for eid in item.evidence_ids)),
                claim_ids=tuple(dict.fromkeys(cid for item in group for cid in item.claim_ids)),
                verification_artifact_ids=tuple(dict.fromkeys(aid for item in group for aid in item.verification_artifact_ids)),
                task_ids=tuple(dict.fromkeys(tid for item in group for tid in item.task_ids)),
                question_id=group[0].question_id,
                origin=FindingOrigin.STORED_CONTRADICTION,
            ))
    return result


def _all_contradiction_findings(findings, artifacts, executions, tasks):
    """Preserve both verifier-level and per-evidence stance conflicts."""
    result = _contradiction_findings(findings)
    existing = {artifact_id for item in result for artifact_id in item.verification_artifact_ids}
    for artifact in artifacts:
        stances = {assessment.stance for unit in artifact.verification_assessments
                   for assessment in unit.evidence_assessments}
        if "SUPPORTS" not in stances or "CONTRADICTS" not in stances:
            continue
        execution = executions[artifact.execution_id]
        task = tasks[execution.task_id]
        refs = (artifact.artifact_id,)
        if artifact.artifact_id in existing:
            continue
        result.append(SynthesisFinding(
            statement=artifact.verification_claim or "",
            classification=FindingClassification.UNRESOLVED_CONTRADICTION,
            evidence_ids=artifact.evidence_ids,
            verification_artifact_ids=refs,
            task_ids=(task.task_id,), question_id=task.question_id,
            origin=FindingOrigin.STORED_CONTRADICTION,
        ))
        existing.add(artifact.artifact_id)
    return result


def _synthesis_reference_snapshot(plan, run, loop, iteration_id, artifacts, state):
    """Small derived signature used to reject stale or cross-run synthesis inputs."""
    return {
        "source_artifact_ids": tuple(item.artifact_id for item in artifacts),
        "evidence_ids": tuple(dict.fromkeys(eid for item in artifacts for eid in item.evidence_ids)),
        "run_id": run.run_id, "plan_id": plan.plan_id, "plan_revision": plan.revision,
        "loop_id": loop.loop_id, "iteration_id": iteration_id, "objective": plan.objective.text,
    }


def _validate_synthesis_references(synthesis, signature, plan, artifacts, state):
    if (synthesis.source_artifact_ids != signature["source_artifact_ids"]
            or synthesis.objective != signature["objective"]
            or tuple(item.evidence_id for item in synthesis.evidence_references) != signature["evidence_ids"]):
        raise ValueError("ResearchSynthesis contains stale or unauthorized artifact/Evidence references.")
    evidence_by_id = {item.evidence_id: item for item in state.evidence}
    expected_refs = tuple(_evidence_reference(state, evidence_by_id[eid])
                          for eid in signature["evidence_ids"])
    if synthesis.evidence_references != expected_refs:
        raise ValueError("ResearchSynthesis contains stale or fabricated Evidence provenance references.")

    executions = {item.execution_id: item for item in state.task_executions}
    tasks = {item.task_id: item for item in plan.tasks}
    claims = _linked_claims(state, signature["evidence_ids"])
    expected_findings = []
    verification_artifacts = [item for item in artifacts if item.artifact_type == "verification_result"]
    for artifact in verification_artifacts:
        execution = executions[artifact.execution_id]
        task = tasks[execution.task_id]
        claim_ids = tuple(claim.claim_id for claim in claims
                          if _same_text(claim.text, artifact.verification_claim))
        expected_findings.append(SynthesisFinding(
            statement=artifact.verification_claim or "",
            classification=FindingClassification(artifact.verification_verdict),
            evidence_ids=artifact.evidence_ids, claim_ids=claim_ids,
            verification_artifact_ids=(artifact.artifact_id,), task_ids=(task.task_id,),
            question_id=task.question_id,
        ))
    verified_claim_texts = {_same_text_key(item.verification_claim) for item in verification_artifacts}
    for claim in claims:
        if _same_text_key(claim.text) not in verified_claim_texts:
            expected_findings.append(SynthesisFinding(
                statement=claim.text, classification=_claim_classification(claim.status),
                evidence_ids=tuple(claim.evidence_ids), claim_ids=(claim.claim_id,),
                origin=FindingOrigin.RESEARCH_CLAIM,
            ))
    expected_findings.extend(_stored_context_findings(artifacts, executions, tasks))
    expected_findings.extend(_all_contradiction_findings(expected_findings, verification_artifacts, executions, tasks))
    actual_findings = tuple(_finding_content(item) for item in synthesis.findings)
    if actual_findings != tuple(_finding_content(item) for item in expected_findings):
        raise ValueError("ResearchSynthesis findings do not match stored verification artifacts.")

    expected_coverage = _question_coverage(plan, artifacts, executions)
    if synthesis.question_coverage != expected_coverage:
        raise ValueError("ResearchSynthesis question coverage does not match stored artifacts.")
    expected_claim_ids = tuple(item.claim_id for item in claims)
    expected_candidate_ids = tuple(candidate.candidate_id for artifact in artifacts
                                   for candidate in artifact.improvement_candidates)
    expected_prior_ids = tuple(finding.finding_id for artifact in artifacts
                               for finding in artifact.prior_work_findings)
    expected_experiment_ids = tuple(item.artifact_id for item in artifacts
                                    if item.artifact_type == "experiment_plan")
    if (synthesis.claim_ids != expected_claim_ids
            or synthesis.candidate_improvement_ids != expected_candidate_ids
            or synthesis.prior_work_finding_ids != expected_prior_ids
            or synthesis.experiment_plan_artifact_ids != expected_experiment_ids):
        raise ValueError("ResearchSynthesis derived references do not match stored run artifacts.")


def _finding_content(finding):
    return (finding.statement, finding.classification, finding.evidence_ids,
            finding.claim_ids, finding.verification_artifact_ids, finding.task_ids,
            finding.question_id, finding.source_artifact_ids, finding.origin)


__all__ = [
    "FindingClassification", "FindingOrigin", "QuestionCoverageStatus", "CritiqueIssueType",
    "EvidenceProvenanceReference", "SynthesisFinding", "QuestionCoverage",
    "ResearchSynthesis", "CritiqueIssue", "ResearchCritique",
    "synthesize_research_state", "critique_research_synthesis",
]
