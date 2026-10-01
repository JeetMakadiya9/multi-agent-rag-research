"""Authorized, non-executing generation of candidate experiment plans.

This module turns stored candidate/research context into a structured proposal
for researcher review. It never searches, retrieves, verifies, trains, measures,
or executes an experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Optional, Tuple

from llm import LLMProvider
from research_capabilities import CapabilityType
from research_plan_review import PlanReview
from research_planning import ResearchPlan, ResearchTask
from research_state import (
    Evidence,
    ExperimentDesignElement,
    ExperimentMetric,
    ResearchArtifact,
    ResearchExperimentPlan,
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


class ExperimentPlanningError(ValueError):
    """Malformed planning input, stored reference, or generated plan."""


@dataclass(frozen=True)
class ExperimentPlanningRequest:
    improvement_artifact_id: str
    candidate_id: str
    prior_work_artifact_ids: Tuple[str, ...] = ()
    verification_artifact_ids: Tuple[str, ...] = ()
    evidence_ids: Tuple[str, ...] = ()
    domain: Optional[str] = None
    researcher_constraints: Tuple[str, ...] = ()
    researcher_requirements: Tuple[str, ...] = ()
    desired_validation_type: Optional[str] = None
    baseline: Optional[str] = None
    selected_dataset: Optional[str] = None

    def __post_init__(self) -> None:
        for name in ("improvement_artifact_id", "candidate_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ExperimentPlanningError(f"{name} must be a non-empty stored identifier.")
        for name in ("prior_work_artifact_ids", "verification_artifact_ids", "evidence_ids",
                     "researcher_constraints", "researcher_requirements"):
            values = getattr(self, name)
            if not isinstance(values, (tuple, list)):
                raise ExperimentPlanningError(f"{name} must be a sequence.")
            values = tuple(values)
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise ExperimentPlanningError(f"{name} must contain non-empty strings.")
            if name in {"prior_work_artifact_ids", "verification_artifact_ids", "evidence_ids"} and len(values) != len(set(values)):
                raise ExperimentPlanningError(f"{name} must not contain duplicate IDs.")
            object.__setattr__(self, name, values)
        for name in ("domain", "desired_validation_type", "baseline", "selected_dataset"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ExperimentPlanningError(f"{name} must be non-empty text when supplied.")


@dataclass(frozen=True)
class ExperimentPlanningResult:
    capability: CapabilityType
    execution_id: str
    run_id: str
    task_id: str
    artifact_id: str
    experiment_plan: ResearchExperimentPlan

    def __post_init__(self) -> None:
        if self.capability is not CapabilityType.EXPERIMENT_PLANNING:
            raise ValueError("ExperimentPlanningResult capability must be EXPERIMENT_PLANNING.")
        for name in ("execution_id", "run_id", "task_id", "artifact_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"ExperimentPlanningResult.{name} must be non-empty.")
        if not isinstance(self.experiment_plan, ResearchExperimentPlan):
            raise ValueError("ExperimentPlanningResult requires a typed ResearchExperimentPlan.")

    def to_dict(self) -> dict[str, object]:
        return {"capability": self.capability.value, "execution_id": self.execution_id,
                "run_id": self.run_id, "task_id": self.task_id, "artifact_id": self.artifact_id,
                "experiment_plan": self.experiment_plan.to_dict()}

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "ExperimentPlanningResult":
        required = {"capability", "execution_id", "run_id", "task_id", "artifact_id", "experiment_plan"}
        if not isinstance(data, dict) or set(data) != required or not isinstance(data["experiment_plan"], dict):
            raise ValueError("Malformed ExperimentPlanningResult payload.")
        values = dict(data)
        values["capability"] = CapabilityType(values["capability"])
        values["experiment_plan"] = ResearchExperimentPlan.from_dict(values["experiment_plan"])
        return cls(**values)  # type: ignore[arg-type]


class _ArtifactSnapshot:
    def __init__(self, state: ResearchState):
        self.state = state
        self.artifacts = list(state.research_artifacts)
        self.updated_at = state.updated_at

    def rollback(self) -> None:
        self.state.research_artifacts[:] = self.artifacts
        self.state.updated_at = self.updated_at


def execute_experiment_planning(
    plan: ResearchPlan,
    run: ResearchRun,
    task: ResearchTask,
    review: PlanReview,
    request: ExperimentPlanningRequest,
    state: ResearchState,
    llm_provider: LLMProvider,
    *,
    iteration_id: Optional[str] = None,
    retry_of_execution_id: Optional[str] = None,
    continuation_authorization_id: Optional[str] = None,
) -> ExperimentPlanningResult:
    """Generate one structured experiment proposal inside Phase 5 authorization."""
    _validate_entry(request, state, llm_provider, run)
    snapshot = _ArtifactSnapshot(state)
    produced: dict[str, object] = {}

    def handler(_task: ResearchTask, context: TaskExecutionContext) -> TaskExecutionResult:
        lineage_errors = state.validate_lineage() + state.validate_provenance()
        if lineage_errors:
            raise ExperimentPlanningError("ResearchState lineage/provenance is invalid: " + "; ".join(lineage_errors))
        candidate_artifact = _artifact(state, request.improvement_artifact_id)
        if (candidate_artifact is None or candidate_artifact.artifact_type != "improvement_candidate"
                or candidate_artifact.run_id != context.run_id):
            raise ExperimentPlanningError("improvement_artifact_id must resolve to a Phase 6C artifact in this ResearchRun.")
        candidate = next((item for item in candidate_artifact.improvement_candidates
                          if item.candidate_id == request.candidate_id), None)
        if candidate is None or candidate.status != "CANDIDATE" or candidate.novelty_status != "NOT_ASSESSED" or candidate.validation_status != "NOT_VALIDATED":
            raise ExperimentPlanningError("candidate_id must resolve to an unvalidated Phase 6C candidate in the artifact.")
        _require_completed_artifact_execution(state, candidate_artifact)

        prior_artifacts = []
        for artifact_id in request.prior_work_artifact_ids:
            artifact = _artifact(state, artifact_id)
            if (artifact is None or artifact.artifact_type != "prior_work_investigation"
                    or artifact.run_id != context.run_id
                    or artifact.prior_work_candidate_id != candidate.candidate_id
                    or artifact.prior_work_candidate_artifact_id != candidate_artifact.artifact_id):
                raise ExperimentPlanningError(f"Prior-work artifact {artifact_id!r} is missing, mismatched, or belongs to another run/candidate.")
            _require_completed_artifact_execution(state, artifact)
            prior_artifacts.append(artifact)

        verification_ids = tuple(dict.fromkeys(
            tuple(candidate.supporting_verification_artifact_ids) + request.verification_artifact_ids
        ))
        verifications = []
        for artifact_id in verification_ids:
            artifact = _artifact(state, artifact_id)
            if artifact is None or artifact.artifact_type != "verification_result" or artifact.run_id != context.run_id:
                raise ExperimentPlanningError(f"Verification artifact {artifact_id!r} is missing or belongs to another run.")
            _require_completed_artifact_execution(state, artifact)
            verifications.append(artifact)

        evidence_ids = tuple(dict.fromkeys(
            tuple(candidate.supporting_evidence_ids) + request.evidence_ids
            + tuple(eid for artifact in verifications for eid in artifact.evidence_ids)
        ))
        evidence = _resolve_evidence(state, evidence_ids, context.run_id)
        claims = _resolve_supported_claims(state, candidate.supporting_claim_ids, evidence_ids)
        prior_context = _prior_work_context(state, prior_artifacts)
        verification_context = [
            {"artifact_id": item.artifact_id, "claim": item.verification_claim,
             "verdict": item.verification_verdict, "evidence_ids": list(item.evidence_ids)}
            for item in verifications
        ]
        payload = _build_context(request, candidate_artifact, candidate, evidence, claims,
                                 prior_context, verification_context)
        generated = _parse_plan(
            llm_provider.generate(_prompt(payload), temperature=0.0),
            request=request,
            candidate_artifact_id=candidate_artifact.artifact_id,
            candidate_id=candidate.candidate_id,
            prior_work_artifact_ids=tuple(item.artifact_id for item in prior_artifacts),
            verification_artifact_ids=verification_ids,
            evidence_ids=tuple(item.evidence_id for item in evidence),
            search_result_ids=tuple(dict.fromkeys(
                result_id for item in prior_artifacts for result_id in item.search_result_ids
                if any(f.search_result_id == result_id for f in item.prior_work_findings)
            )),
            claim_ids=tuple(item.claim_id for item in claims),
        )
        artifact = ResearchArtifact(
            execution_id=context.execution_id, run_id=context.run_id, task_id=context.task_id,
            search_action_id=None, evidence_ids=generated.supporting_evidence_ids,
            artifact_type="experiment_plan", capability="experiment_planning",
            experiment_plan=generated,
        )
        state.add_research_artifact(artifact)
        produced["artifact"] = artifact
        produced["plan"] = generated
        return TaskExecutionResult(
            success=True,
            output="Created a candidate experiment plan requiring researcher review; no experiment was executed.",
            artifact_ids=(artifact.artifact_id,),
            evidence_ids=generated.supporting_evidence_ids,
        )

    try:
        execution = execute_authorized_task(
            plan, run, task, review, handler, state, iteration_id=iteration_id,
            retry_of_execution_id=retry_of_execution_id,
            continuation_authorization_id=continuation_authorization_id,
        )
    except (TaskExecutionError, TaskExecutionRecordingError):
        snapshot.rollback()
        raise
    except Exception:
        snapshot.rollback()
        raise
    artifact = produced.get("artifact")
    experiment_plan = produced.get("plan")
    if not isinstance(artifact, ResearchArtifact) or not isinstance(experiment_plan, ResearchExperimentPlan) or execution.result is None:
        snapshot.rollback()
        raise RuntimeError("Experiment planning completed without a persisted typed plan artifact.")
    return ExperimentPlanningResult(
        capability=CapabilityType.EXPERIMENT_PLANNING, execution_id=execution.execution_id,
        run_id=run.run_id, task_id=task.task_id, artifact_id=artifact.artifact_id,
        experiment_plan=experiment_plan,
    )


def _validate_entry(request, state, provider, run) -> None:
    if not isinstance(request, ExperimentPlanningRequest):
        raise ExperimentPlanningError("An ExperimentPlanningRequest is required.")
    if not isinstance(state, ResearchState):
        raise ExperimentPlanningError("A ResearchState is required.")
    if not isinstance(run, ResearchRun):
        raise ExperimentPlanningError("A ResearchRun is required.")
    if not callable(getattr(provider, "generate", None)):
        raise ExperimentPlanningError("An explicit LLMProvider with generate() is required.")


def _artifact(state: ResearchState, artifact_id: str):
    return next((item for item in state.research_artifacts if item.artifact_id == artifact_id), None)


def _require_completed_artifact_execution(state: ResearchState, artifact: ResearchArtifact) -> None:
    execution = next((item for item in state.task_executions if item.execution_id == artifact.execution_id), None)
    if (execution is None or execution.status.value != "COMPLETED" or execution.result is None
            or artifact.artifact_id not in execution.result.artifact_ids):
        raise ExperimentPlanningError(f"Supporting artifact {artifact.artifact_id!r} lacks a completed execution reference.")


def _resolve_evidence(state: ResearchState, evidence_ids: Tuple[str, ...], run_id: str) -> tuple[Evidence, ...]:
    evidence_by_id = {item.evidence_id: item for item in state.evidence}
    results = {item.result_id: item for item in state.search_results if item.result_id is not None}
    searches = {item.search_id: item for item in state.searches}
    versions = {item.version_id: item for item in state.document_versions}
    passages = {item.passage_reference_id: item for item in state.passage_references}
    resolved = []
    for evidence_id in evidence_ids:
        evidence = evidence_by_id.get(evidence_id)
        if evidence is None:
            raise ExperimentPlanningError(f"Evidence ID {evidence_id!r} does not resolve in ResearchState.")
        result = results.get(evidence.search_result_id or "")
        search = searches.get(result.search_id or "") if result else None
        version = versions.get(evidence.document_version_id or "")
        passage = passages.get(evidence.passage_reference_id or "")
        if (result is None or search is None or search.run_id != run_id or version is None or passage is None
                or version.source_id != evidence.source_id or passage.document_version_id != version.version_id
                or result.source_id != evidence.source_id):
            raise ExperimentPlanningError(f"Evidence {evidence_id!r} has incomplete or out-of-run provenance.")
        resolved.append(evidence)
    return tuple(resolved)


def _resolve_supported_claims(state: ResearchState, claim_ids, evidence_ids):
    claims = {item.claim_id: item for item in state.claims}
    output = []
    for claim_id in claim_ids:
        claim = claims.get(claim_id)
        if claim is None or claim.status != "SUPPORTED" or not set(claim.evidence_ids).issubset(set(evidence_ids)):
            raise ExperimentPlanningError(f"Candidate supporting claim {claim_id!r} is missing, unsupported, or ungrounded.")
        output.append(claim)
    return tuple(output)


def _prior_work_context(state: ResearchState, artifacts):
    results = {item.result_id: item for item in state.search_results if item.result_id is not None}
    context = []
    for artifact in artifacts:
        findings = {item.search_result_id: item for item in artifact.prior_work_findings}
        for result_id, finding in findings.items():
            result = results.get(result_id)
            if result is None:
                raise ExperimentPlanningError(f"Prior-work finding SearchResult {result_id!r} is missing.")
            context.append({
                "artifact_id": artifact.artifact_id,
                "search_result_id": result_id,
                "source_id": result.source_id,
                "title": result.title,
                "published_at": result.published_at,
                "relevance": finding.relevance,
                "relationship": finding.relationship,
                "problem_overlap": finding.problem_overlap,
                "method_overlap": finding.method_overlap,
                "dataset_overlap": finding.dataset_overlap,
                "evaluation_overlap": finding.evaluation_overlap,
                "differences": list(finding.differences),
                "limitations": list(finding.limitations),
            })
    return context


def _build_context(request, candidate_artifact, candidate, evidence, claims, prior_context, verification_context):
    return {
        "candidate": {
            "artifact_id": candidate_artifact.artifact_id,
            "candidate_id": candidate.candidate_id,
            "title": candidate.title,
            "problem_statement": candidate.problem_statement,
            "motivation": candidate.motivation,
            "proposed_change": candidate.proposed_change,
            "rationale": candidate.rationale,
            "expected_benefit": candidate.expected_benefit,
            "assumptions": list(candidate.assumptions),
            "risks": list(candidate.risks),
            "validation_needed": list(candidate.validation_needed),
            "status": candidate.status,
            "novelty_status": candidate.novelty_status,
            "validation_status": candidate.validation_status,
        },
        "domain": request.domain,
        "researcher_constraints": list(request.researcher_constraints),
        "researcher_requirements": list(request.researcher_requirements),
        "desired_validation_type": request.desired_validation_type,
        "researcher_baseline": request.baseline,
        "researcher_selected_dataset": request.selected_dataset,
        "supported_claims": [{"claim_id": c.claim_id, "text": c.text, "evidence_ids": list(c.evidence_ids)} for c in claims],
        "evidence": [{"evidence_id": e.evidence_id, "text": e.text, "source_id": e.source_id,
                      "document_version_id": e.document_version_id,
                      "passage_reference_id": e.passage_reference_id,
                      "search_result_id": e.search_result_id, "text_hash": e.text_hash} for e in evidence],
        "verification_artifacts": verification_context,
        "prior_work_findings": prior_context,
    }


_SCHEMA = {
    "objective": "string",
    "hypothesis": "testable, explicitly hypothetical string",
    "proposed_method": "string",
    "baseline": "exact researcher baseline or BASELINE_REQUIRES_RESEARCHER_SPECIFICATION",
    "dataset_status": "RESEARCHER_SPECIFIED|DATASET_REQUIRES_RESEARCHER_SELECTION",
    "selected_dataset": "exact researcher dataset or null",
    "data_requirements": ["string"],
    "experimental_setup": ["string"],
    "controls": [{"description": "string", "rationale": "string"}],
    "ablations": [{"description": "string", "rationale": "string"}],
    "metrics": [{"name": "string", "measures": "string", "rationale": "string",
                 "suitability": "REQUIRES_RESEARCHER_REVIEW"}],
    "expected_observations": ["qualitative expected observation, not a result"],
    "interpretation_criteria": ["conditional interpretation rule, not a conclusion"],
    "confounders": ["string"],
    "limitations": ["string"],
    "reproducibility_requirements": ["string or TO_BE_SPECIFIED"],
    "resource_requirements": ["string or UNKNOWN/TO_BE_ESTIMATED"],
    "assumptions": ["PLANNING_ASSUMPTION: string"],
    "unresolved_requirements": ["string"],
}


def _prompt(context: dict) -> list[dict[str, str]]:
    system = (
        "You are a domain-agnostic experiment-planning assistant. Produce only a candidate plan for researcher review; "
        "do not execute an experiment, search, retrieve, verify, or create evidence. Return a single JSON object with "
        "exactly these keys and value shapes: " + json.dumps(_SCHEMA) + "\n"
        "Epistemic rules: a hypothesis is not a fact; a plan is not a result; expected outcomes are not observations; "
        "metrics are proposed and require researcher review; feasibility and scientific validity are not established; "
        "do not claim novelty, a proven gap, success, validation, or that a candidate works. Do not invent baselines, "
        "datasets, thresholds, measured values, or performance predictions. If no baseline was supplied, use "
        "BASELINE_REQUIRES_RESEARCHER_SPECIFICATION and include that exact unresolved requirement. If no dataset was "
        "selected by the researcher, set selected_dataset to null, dataset_status to "
        "DATASET_REQUIRES_RESEARCHER_SELECTION, and include that exact unresolved requirement. If no metric is "
        "appropriate from supplied context, return an empty metrics list and include METRIC_REQUIRES_RESEARCHER_SELECTION. "
        "Use TO_BE_SPECIFIED or UNKNOWN for missing reproducibility/resources. Preserve all supplied references and "
        "constraints; do not invent IDs. Prior-work observations are context, not requirements or proof of novelty. "
        "Use qualitative expected observations only; no numeric predictions. Mark assumptions explicitly."
    )
    user = "Use only this stored candidate and explicitly supplied state context to propose a plan:\n" + json.dumps(context, ensure_ascii=False)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _parse_plan(raw: str, *, request, candidate_artifact_id, candidate_id, prior_work_artifact_ids,
                verification_artifact_ids, evidence_ids, search_result_ids, claim_ids) -> ResearchExperimentPlan:
    expected = set(_SCHEMA)
    if not isinstance(raw, str):
        raise ExperimentPlanningError("LLM response must be JSON text.")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ExperimentPlanningError("LLM response is malformed JSON.") from exc
    if not isinstance(value, dict) or set(value) != expected:
        raise ExperimentPlanningError("LLM response does not match the exact experiment-plan schema.")
    # Researcher-owned constraints/requirements are copied from the request;
    # they are not model-generated fields in the response schema.
    generated_text_fields = set(ResearchExperimentPlan._TEXT_FIELDS) - {
        "researcher_constraints", "researcher_requirements",
    }
    arrays = generated_text_fields | {"controls", "ablations", "metrics"}
    for name in arrays:
        if not isinstance(value[name], list):
            raise ExperimentPlanningError(f"Generated {name} must be a JSON list.")
    for name in generated_text_fields:
        if any(not isinstance(item, str) or not item.strip() for item in value[name]):
            raise ExperimentPlanningError(f"Generated {name} must contain non-empty strings.")
    if request.baseline is None:
        if value["baseline"] != "BASELINE_REQUIRES_RESEARCHER_SPECIFICATION":
            raise ExperimentPlanningError("The model cannot invent a baseline that the researcher did not specify.")
    elif value["baseline"] != request.baseline:
        raise ExperimentPlanningError("Generated baseline does not preserve the researcher-specified baseline.")
    if request.selected_dataset is None:
        if value["selected_dataset"] is not None or value["dataset_status"] != "DATASET_REQUIRES_RESEARCHER_SELECTION":
            raise ExperimentPlanningError("The model cannot select or invent a dataset.")
    elif value["selected_dataset"] != request.selected_dataset or value["dataset_status"] != "RESEARCHER_SPECIFIED":
        raise ExperimentPlanningError("Generated dataset does not preserve the researcher-selected dataset.")
    if not isinstance(value["objective"], str) or not value["objective"].strip():
        raise ExperimentPlanningError("Generated objective must be non-empty.")
    if not isinstance(value["hypothesis"], str) or not re.search(r"\b(whether|hypothes|test|investigat)\b", value["hypothesis"], re.I):
        raise ExperimentPlanningError("Generated hypothesis must remain explicitly testable/hypothetical.")
    if not isinstance(value["proposed_method"], str) or not value["proposed_method"].strip():
        raise ExperimentPlanningError("Generated proposed_method must be non-empty.")
    unresolved = value["unresolved_requirements"]
    if request.baseline is None and not any("BASELINE_REQUIRES_RESEARCHER_SPECIFICATION" in x for x in unresolved):
        raise ExperimentPlanningError("Unspecified baseline must remain an unresolved requirement.")
    if request.selected_dataset is None and not any("DATASET_REQUIRES_RESEARCHER_SELECTION" in x for x in unresolved):
        raise ExperimentPlanningError("Unselected dataset must remain an unresolved requirement.")
    if not value["metrics"] and not any("METRIC_REQUIRES_RESEARCHER_SELECTION" in x for x in unresolved):
        raise ExperimentPlanningError("An empty metric proposal must remain unresolved.")
    _validate_plan_language(value)
    controls = tuple(ExperimentDesignElement.from_dict(item) for item in value["controls"])
    ablations = tuple(ExperimentDesignElement.from_dict(item) for item in value["ablations"])
    metrics = tuple(ExperimentMetric.from_dict(item) for item in value["metrics"])
    return ResearchExperimentPlan(
        improvement_artifact_id=candidate_artifact_id, candidate_id=candidate_id,
        objective=value["objective"], hypothesis=value["hypothesis"], proposed_method=value["proposed_method"],
        baseline=value["baseline"], dataset_status=value["dataset_status"], selected_dataset=value["selected_dataset"],
        domain=request.domain, desired_validation_type=request.desired_validation_type,
        researcher_constraints=request.researcher_constraints, researcher_requirements=request.researcher_requirements,
        data_requirements=tuple(value["data_requirements"]), experimental_setup=tuple(value["experimental_setup"]),
        controls=controls, ablations=ablations, metrics=metrics,
        expected_observations=tuple(value["expected_observations"]),
        interpretation_criteria=tuple(value["interpretation_criteria"]), confounders=tuple(value["confounders"]),
        limitations=tuple(value["limitations"]), reproducibility_requirements=tuple(value["reproducibility_requirements"]),
        resource_requirements=tuple(value["resource_requirements"]), assumptions=tuple(value["assumptions"]),
        unresolved_requirements=tuple(value["unresolved_requirements"]),
        prior_work_artifact_ids=prior_work_artifact_ids, verification_artifact_ids=verification_artifact_ids,
        supporting_evidence_ids=evidence_ids, supporting_search_result_ids=search_result_ids,
        supporting_claim_ids=claim_ids,
    )


def _validate_plan_language(value: dict) -> None:
    all_text = [value["objective"], value["hypothesis"], value["proposed_method"], value["baseline"]]
    for field_name in set(ResearchExperimentPlan._TEXT_FIELDS) - {"researcher_constraints", "researcher_requirements"}:
        all_text.extend(value[field_name])
    for field_name in ("controls", "ablations", "metrics"):
        for row in value[field_name]:
            if not isinstance(row, dict):
                raise ExperimentPlanningError(f"Generated {field_name} entries must be objects.")
            all_text.extend(str(v) for v in row.values())
    forbidden = re.compile(r"\b(this is novel|novel contribution|proven research gap|no previous work|first[- ]ever|candidate works|hypothesis is proven|scientifically valid|experiment succeeded|results confirm)\b", re.I)
    if any(forbidden.search(text) for text in all_text):
        raise ExperimentPlanningError("Generated plan contains an unsupported novelty, validity, or result claim.")
    numeric_performance = re.compile(r"\b(accuracy|precision|recall|f1|auroc|auc|auprc|rmse|mae|latency)\b[^\n]{0,32}\b\d+(?:\.\d+)?\s*%?", re.I)
    for text in value["expected_observations"]:
        if not isinstance(text, str) or not text.strip():
            raise ExperimentPlanningError("expected_observations must contain non-empty qualitative text.")
        if "%" in text or numeric_performance.search(text):
            raise ExperimentPlanningError("Expected observations cannot contain fabricated numeric performance predictions.")
    for text in value["interpretation_criteria"]:
        if not isinstance(text, str) or not text.strip():
            raise ExperimentPlanningError("interpretation_criteria must contain non-empty conditional rules.")
        if not re.search(r"\b(if|would|under those conditions)\b", text, re.I):
            raise ExperimentPlanningError("Interpretation criteria must remain conditional, not actual conclusions.")


__all__ = [
    "ExperimentPlanningError", "ExperimentPlanningRequest", "ExperimentPlanningResult",
    "execute_experiment_planning",
]
