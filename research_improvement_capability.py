"""Explicitly generate unvalidated improvement candidates from stored inputs.

This capability runs only through the Phase 5 authorization boundary. It never
retrieves evidence, verifies claims, searches prior work, or plans experiments.
Generated candidates are proposals, not facts, proof of a gap, or novelty claims.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Optional, Sequence, Tuple

from llm import LLMProvider
from research_capabilities import CapabilityType
from research_plan_review import PlanReview
from research_planning import ResearchPlan, ResearchTask
from research_state import (
    Evidence,
    ResearchArtifact,
    ResearchImprovementCandidate,
    ResearchRun,
    ResearchState,
)
from research_task_execution import (
    TaskExecutionContext,
    TaskExecutionError,
    TaskExecutionRecordingError,
    TaskExecutionResult,
    execute_authorized_task,
)


class ImprovementCapabilityError(ValueError):
    """Malformed generation input or invalid stored provenance/verification."""


@dataclass(frozen=True)
class ImprovementGenerationRequest:
    """Researcher-provided problem plus IDs of existing verified research input."""

    problem_statement: str
    evidence_ids: Tuple[str, ...]
    verified_claim_ids: Tuple[str, ...] = ()
    verification_artifact_ids: Tuple[str, ...] = ()
    domain: Optional[str] = None
    constraints: Tuple[str, ...] = ()
    number_of_candidates: int = 3

    def __post_init__(self) -> None:
        if not isinstance(self.problem_statement, str) or not self.problem_statement.strip():
            raise ImprovementCapabilityError("problem_statement must be non-empty researcher-provided text.")
        object.__setattr__(self, "problem_statement", self.problem_statement.strip())
        for name in ("evidence_ids", "verified_claim_ids", "verification_artifact_ids", "constraints"):
            raw = getattr(self, name)
            if not isinstance(raw, (tuple, list)):
                raise ImprovementCapabilityError(f"{name} must be a sequence.")
            values = tuple(raw)
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise ImprovementCapabilityError(f"{name} must contain non-empty strings.")
            values = tuple(value.strip() for value in values)
            if len(values) != len(set(values)):
                raise ImprovementCapabilityError(f"{name} must not contain duplicates.")
            object.__setattr__(self, name, values)
        if not self.evidence_ids:
            raise ImprovementCapabilityError("At least one stored evidence_id is required.")
        if not self.verified_claim_ids and not self.verification_artifact_ids:
            raise ImprovementCapabilityError(
                "At least one existing supported claim or supported verification artifact is required."
            )
        if self.domain is not None and (not isinstance(self.domain, str) or not self.domain.strip()):
            raise ImprovementCapabilityError("domain must be non-empty when supplied.")
        if not isinstance(self.number_of_candidates, int) or isinstance(self.number_of_candidates, bool) or not 1 <= self.number_of_candidates <= 10:
            raise ImprovementCapabilityError("number_of_candidates must be an integer from 1 through 10.")


@dataclass(frozen=True)
class ImprovementGenerationResult:
    capability: CapabilityType
    execution_id: str
    run_id: str
    task_id: str
    artifact_id: str
    problem_statement: str
    candidates: Tuple[ResearchImprovementCandidate, ...]
    evidence_ids: Tuple[str, ...]
    success: bool = True

    def __post_init__(self) -> None:
        if self.capability is not CapabilityType.IMPROVEMENT_GENERATION:
            raise ValueError("Result capability must be IMPROVEMENT_GENERATION.")
        for name in ("execution_id", "run_id", "task_id", "artifact_id", "problem_statement"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty.")
        candidates = tuple(self.candidates)
        if any(not isinstance(item, ResearchImprovementCandidate) for item in candidates):
            raise ValueError("candidates must contain typed ResearchImprovementCandidate records.")
        if len({item.candidate_id for item in candidates}) != len(candidates):
            raise ValueError("candidate IDs must be unique.")
        object.__setattr__(self, "candidates", candidates)
        evidence_ids = tuple(self.evidence_ids)
        if not evidence_ids or len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("evidence_ids must be non-empty and unique.")
        object.__setattr__(self, "evidence_ids", evidence_ids)
        if self.success is not True:
            raise ValueError("Returned result represents a completed execution.")

    def to_dict(self) -> dict[str, object]:
        return {
            "capability": self.capability.value,
            "execution_id": self.execution_id,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "artifact_id": self.artifact_id,
            "problem_statement": self.problem_statement,
            "candidates": [item.to_dict() for item in self.candidates],
            "evidence_ids": list(self.evidence_ids),
            "success": self.success,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "ImprovementGenerationResult":
        required = {
            "capability", "execution_id", "run_id", "task_id", "artifact_id",
            "problem_statement", "candidates", "evidence_ids", "success",
        }
        if not isinstance(data, dict) or set(data) != required:
            raise ValueError("Malformed ImprovementGenerationResult payload.")
        values = dict(data)
        try:
            values["capability"] = CapabilityType(values["capability"])
            values["candidates"] = tuple(ResearchImprovementCandidate.from_dict(item) for item in values["candidates"])
            values["evidence_ids"] = tuple(values["evidence_ids"])
            return cls(**values)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Malformed ImprovementGenerationResult payload: {exc}") from exc


class _ImprovementStateAccess:
    """Narrow read view and one artifact-write operation; no general state access."""

    __slots__ = ("__state",)

    def __init__(self, state: ResearchState):
        self.__state = state

    @property
    def evidence(self):
        return tuple(self.__state.evidence)

    @property
    def claims(self):
        return tuple(self.__state.claims)

    @property
    def research_artifacts(self):
        return tuple(self.__state.research_artifacts)

    @property
    def task_executions(self):
        return tuple(self.__state.task_executions)

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


_OUTPUT_KEYS = {
    "title", "motivation", "proposed_change", "rationale", "expected_benefit",
    "assumptions", "risks", "validation_needed", "supporting_evidence_ids",
    "supporting_claim_ids", "supporting_verification_artifact_ids",
}
_UNSUPPORTED_CERTAINTY = re.compile(
    r"\b(?:novel contribution|novel method|novel approach|novel idea|this is novel|"
    r"first (?:ever|method|approach|to )|no previous work|no prior work|"
    r"no one has|never been done|(?:this )?improvement is proven|is proven|proven to|"
    r"has been validated|will improve|guaranteed(?: to improve| improvement)?)\b",
    flags=re.IGNORECASE,
)


def execute_improvement_generation(
    plan: ResearchPlan,
    run: ResearchRun,
    task: ResearchTask,
    review: PlanReview,
    request: ImprovementGenerationRequest,
    state: ResearchState,
    llm_provider: LLMProvider,
    *,
    iteration_id: Optional[str] = None,
    retry_of_execution_id: Optional[str] = None,
    continuation_authorization_id: Optional[str] = None,
) -> ImprovementGenerationResult:
    """Generate candidate proposals from explicit, already-stored research inputs."""
    _validate_entry(request, state, llm_provider)
    snapshot = _ArtifactSnapshot(state)
    access = _ImprovementStateAccess(state)
    produced: dict[str, object] = {}

    def _generate(_task: ResearchTask, context: TaskExecutionContext) -> TaskExecutionResult:
        evidence = _resolve_evidence(access, request.evidence_ids, context.run_id)
        claims = _resolve_claims(access, request.verified_claim_ids, request.evidence_ids)
        verification_artifacts = _resolve_verification_artifacts(
            access, request.verification_artifact_ids, request.evidence_ids, context.run_id
        )
        prompt = _build_prompt(request, evidence, claims, verification_artifacts)
        raw = llm_provider.generate(
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            timeout=120,
        )
        candidates = _parse_candidates(raw, request, context)
        artifact = ResearchArtifact(
            execution_id=context.execution_id,
            run_id=context.run_id,
            task_id=context.task_id,
            search_action_id=None,
            evidence_ids=request.evidence_ids,
            artifact_type="improvement_candidate",
            capability=CapabilityType.IMPROVEMENT_GENERATION.value,
            improvement_problem_statement=request.problem_statement,
            improvement_problem_origin="researcher_provided",
            improvement_candidates=candidates,
            improvement_claim_ids=request.verified_claim_ids,
            improvement_verification_artifact_ids=request.verification_artifact_ids,
        )
        access.add_research_artifact(artifact)
        produced["artifact"] = artifact
        output = (
            f"Generated {len(candidates)} candidate improvement(s); novelty and effectiveness remain unassessed."
            if candidates else "No candidate improvements were generated; no proposal was fabricated."
        )
        return TaskExecutionResult(
            success=True,
            output=output,
            artifact_ids=(artifact.artifact_id,),
            evidence_ids=request.evidence_ids,
        )

    try:
        execution = execute_authorized_task(
            plan, run, task, review, _generate, state,
            iteration_id=iteration_id,
            retry_of_execution_id=retry_of_execution_id,
            continuation_authorization_id=continuation_authorization_id,
        )
    except (TaskExecutionError, TaskExecutionRecordingError):
        snapshot.rollback()
        raise
    artifact = produced.get("artifact")
    if not isinstance(artifact, ResearchArtifact):
        snapshot.rollback()
        raise RuntimeError("IMPROVEMENT_GENERATION completed without a persisted artifact.")
    return ImprovementGenerationResult(
        capability=CapabilityType.IMPROVEMENT_GENERATION,
        execution_id=execution.execution_id,
        run_id=execution.run_id,
        task_id=execution.task_id,
        artifact_id=artifact.artifact_id,
        problem_statement=request.problem_statement,
        candidates=artifact.improvement_candidates,
        evidence_ids=artifact.evidence_ids,
    )


_SYSTEM_PROMPT = """You generate candidate research improvements from given inputs.
The problem statement is researcher-provided context, not automatically a verified fact.
Evidence and supported findings are background/motivation, not proof that a proposal works.
Return only the requested JSON object. Proposals are hypotheses and must be marked by cautious wording.
Do not claim novelty, firstness, absence of prior work, proven effectiveness, or guaranteed outcomes.
Do not invent citations, datasets, measurements, or experimental results. Do not create a full experiment protocol;
state only what would need validation. Distinguish assumptions and risks. Every candidate must cite only supplied IDs."""


def _validate_entry(request: ImprovementGenerationRequest, state: ResearchState, provider: LLMProvider) -> None:
    if not isinstance(request, ImprovementGenerationRequest):
        raise ImprovementCapabilityError("An ImprovementGenerationRequest is required.")
    if not isinstance(state, ResearchState):
        raise ImprovementCapabilityError("ResearchState is required.")
    if provider is None or not callable(getattr(provider, "generate", None)):
        raise ImprovementCapabilityError("An explicit LLMProvider with generate() is required.")


def _resolve_evidence(access: _ImprovementStateAccess, evidence_ids: Sequence[str], run_id: str) -> tuple[Evidence, ...]:
    errors = access.validate_provenance()
    if errors:
        raise ImprovementCapabilityError("Stored Evidence provenance is invalid: " + "; ".join(errors))
    evidence_by_id = {item.evidence_id: item for item in access.evidence}
    versions = {item.version_id: item for item in access.document_versions}
    passages = {item.passage_reference_id: item for item in access.passage_references}
    results = {item.result_id: item for item in access.search_results if item.result_id is not None}
    searches = {item.search_id: item for item in access.searches}
    resolved = []
    for evidence_id in evidence_ids:
        evidence = evidence_by_id.get(evidence_id)
        if evidence is None:
            raise ImprovementCapabilityError(f"Unknown Evidence ID {evidence_id!r}.")
        if not evidence.text.strip():
            raise ImprovementCapabilityError(f"Evidence {evidence_id!r} has no text.")
        version = versions.get(evidence.document_version_id or "")
        passage = passages.get(evidence.passage_reference_id or "")
        result = results.get(evidence.search_result_id or "")
        source_ids = {item.source_id for item in access.sources}
        if (
            version is None or passage is None or result is None
            or evidence.source_id not in source_ids
            or version.source_id != evidence.source_id
            or passage.document_version_id != version.version_id
            or result.source_id != evidence.source_id
        ):
            raise ImprovementCapabilityError(f"Evidence {evidence_id!r} does not resolve complete source/document/passage/search provenance.")
        search = searches.get(result.search_id or "")
        if search is None or search.run_id != run_id:
            raise ImprovementCapabilityError(f"Evidence {evidence_id!r} does not belong to the authorized run lineage.")
        resolved.append(evidence)
    return tuple(resolved)


def _resolve_claims(access, claim_ids: Sequence[str], evidence_ids: Sequence[str]):
    claims = {item.claim_id: item for item in access.claims}
    selected = []
    for claim_id in claim_ids:
        claim = claims.get(claim_id)
        if claim is None or claim.status != "SUPPORTED":
            raise ImprovementCapabilityError(f"Claim {claim_id!r} is missing or not marked SUPPORTED.")
        if not claim.evidence_ids or not set(claim.evidence_ids).issubset(set(evidence_ids)):
            raise ImprovementCapabilityError(f"Supported claim {claim_id!r} is not grounded in the requested Evidence IDs.")
        selected.append(claim)
    return tuple(selected)


def _resolve_verification_artifacts(access, artifact_ids, evidence_ids, run_id):
    artifacts = {item.artifact_id: item for item in access.research_artifacts}
    executions = {item.execution_id: item for item in access.task_executions}
    selected = []
    for artifact_id in artifact_ids:
        artifact = artifacts.get(artifact_id)
        if (
            artifact is None
            or artifact.artifact_type != "verification_result"
            or artifact.verification_verdict != "SUPPORTED"
            or artifact.run_id != run_id
            or not set(artifact.evidence_ids).issubset(set(evidence_ids))
            or artifact.execution_id not in executions
            or executions[artifact.execution_id].status.value != "COMPLETED"
        ):
            raise ImprovementCapabilityError(f"Verification artifact {artifact_id!r} is missing, unsupported, or outside this run/evidence set.")
        selected.append(artifact)
    return tuple(selected)


def _build_prompt(request, evidence, claims, verification_artifacts) -> str:
    evidence_payload = [
        {
            "evidence_id": item.evidence_id,
            "text": item.text,
            "source_id": item.source_id,
            "document_version_id": item.document_version_id,
            "passage_reference_id": item.passage_reference_id,
        }
        for item in evidence
    ]
    claim_payload = [{"claim_id": item.claim_id, "text": item.text, "status": item.status} for item in claims]
    verification_payload = [
        {"artifact_id": item.artifact_id, "claim": item.verification_claim, "verdict": item.verification_verdict}
        for item in verification_artifacts
    ]
    return json.dumps({
        "task": "propose candidate improvements only",
        "problem_statement": {"origin": "researcher_provided", "text": request.problem_statement},
        "domain": request.domain,
        "constraints": list(request.constraints),
        "verified_claims": claim_payload,
        "supported_verification_artifacts": verification_payload,
        "stored_evidence": evidence_payload,
        "maximum_candidates": request.number_of_candidates,
        "required_candidate_fields": sorted(_OUTPUT_KEYS),
        "field_rules": {
            "assumptions": "non-empty list of explicit assumptions",
            "risks": "non-empty list of possible failure modes",
            "validation_needed": "non-empty list; do not write a full experimental protocol",
            "supporting_evidence_ids": "one or more exact supplied Evidence IDs",
            "supporting_claim_ids": "zero or more exact supplied supported claim IDs",
            "supporting_verification_artifact_ids": "zero or more exact supplied supported verification artifact IDs",
        },
    }, ensure_ascii=False)


def _parse_candidates(raw, request, context):
    if not isinstance(raw, str) or not raw.strip():
        raise ImprovementCapabilityError("LLM returned an empty improvement response.")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ImprovementCapabilityError("LLM returned malformed improvement JSON.") from exc
    if not isinstance(data, dict) or set(data) != {"candidates"} or not isinstance(data["candidates"], list):
        raise ImprovementCapabilityError("Improvement JSON must contain only a candidates list.")
    if len(data["candidates"]) > request.number_of_candidates:
        raise ImprovementCapabilityError("LLM returned more candidates than requested.")
    output = []
    for index, item in enumerate(data["candidates"]):
        if not isinstance(item, dict) or set(item) != _OUTPUT_KEYS:
            raise ImprovementCapabilityError(f"Candidate {index} does not match the required structured schema.")
        for key, value in item.items():
            serialized = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            if _UNSUPPORTED_CERTAINTY.search(serialized):
                raise ImprovementCapabilityError(f"Candidate {index} contains unsupported certainty/novelty language.")
        evidence_ids = _string_list(item, "supporting_evidence_ids", required=True)
        claim_ids = _string_list(item, "supporting_claim_ids", required=False)
        verification_ids = _string_list(item, "supporting_verification_artifact_ids", required=False)
        if not set(evidence_ids).issubset(set(request.evidence_ids)):
            raise ImprovementCapabilityError(f"Candidate {index} references Evidence not supplied in the request.")
        if not set(claim_ids).issubset(set(request.verified_claim_ids)):
            raise ImprovementCapabilityError(f"Candidate {index} references a claim not supplied in the request.")
        if not set(verification_ids).issubset(set(request.verification_artifact_ids)):
            raise ImprovementCapabilityError(f"Candidate {index} references a verification artifact not supplied in the request.")
        output.append(ResearchImprovementCandidate(
            title=_string(item, "title"),
            problem_statement=request.problem_statement,
            motivation=_string(item, "motivation"),
            proposed_change=_string(item, "proposed_change"),
            rationale=_string(item, "rationale"),
            expected_benefit=_string(item, "expected_benefit"),
            assumptions=_string_list(item, "assumptions", required=True),
            risks=_string_list(item, "risks", required=True),
            validation_needed=_string_list(item, "validation_needed", required=True),
            supporting_evidence_ids=evidence_ids,
            supporting_claim_ids=claim_ids,
            supporting_verification_artifact_ids=verification_ids,
        ))
    return tuple(output)


def _string(item, key):
    value = item.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ImprovementCapabilityError(f"Candidate field {key!r} must be non-empty text.")
    return value.strip()


def _string_list(item, key, *, required):
    value = item.get(key)
    if not isinstance(value, list) or (required and not value):
        raise ImprovementCapabilityError(f"Candidate field {key!r} must be {'a non-empty' if required else 'a'} list of strings.")
    if any(not isinstance(entry, str) or not entry.strip() for entry in value):
        raise ImprovementCapabilityError(f"Candidate field {key!r} must contain non-empty strings.")
    normalized = tuple(entry.strip() for entry in value)
    if len(normalized) != len(set(normalized)):
        raise ImprovementCapabilityError(f"Candidate field {key!r} must not contain duplicate values.")
    return normalized


__all__ = [
    "ImprovementCapabilityError",
    "ImprovementGenerationRequest",
    "ImprovementGenerationResult",
    "execute_improvement_generation",
]
