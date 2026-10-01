"""Explicit, authorized research capabilities.

Phase 6 exposes explicit, authorized research capabilities. LOCAL_RETRIEVAL
records retrieved material without verification; EVIDENCE_VERIFICATION is
implemented in ``research_verification_capability`` and must be requested
separately. IMPROVEMENT_GENERATION is implemented in
``research_improvement_capability`` and produces only unvalidated candidate
proposals. No autonomous task selection or web search is performed here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Optional, Tuple

from research_plan_review import PlanReview
from research_planning import ResearchPlan, ResearchTask
from research_retrieval import RetrievalProvider
from research_state import (
    DocumentVersion,
    Evidence,
    PassageReference,
    ResearchArtifact,
    ResearchRun,
    ResearchState,
    SearchAction,
    Source,
)
from research_task_execution import (
    ResearchTaskExecution,
    TaskExecutionContext,
    TaskExecutionError,
    TaskExecutionRecordingError,
    TaskExecutionResult,
    execute_authorized_task,
)
from research_workflow import retrieve_for_research


class CapabilityType(str, Enum):
    LOCAL_RETRIEVAL = "local_retrieval"
    EVIDENCE_VERIFICATION = "evidence_verification"
    IMPROVEMENT_GENERATION = "improvement_generation"
    PRIOR_WORK_INVESTIGATION = "prior_work_investigation"
    EXPERIMENT_PLANNING = "experiment_planning"


class CapabilityStatus(str, Enum):
    SUCCESS = "SUCCESS"
    NO_RESULTS = "NO_RESULTS"


class CapabilityRequestError(ValueError):
    """A capability request is malformed or refers to unavailable state."""


@dataclass(frozen=True)
class LocalRetrievalRequest:
    """Typed request for the existing local retrieval provider boundary."""

    query: str
    limit: int = 8
    purpose: str = "retrieval"
    question_id: Optional[str] = None
    evidence_document_versions: Mapping[int, DocumentVersion] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_request_values(self)
        object.__setattr__(
            self,
            "evidence_document_versions",
            MappingProxyType(dict(self.evidence_document_versions)),
        )


@dataclass(frozen=True)
class CapabilityResult:
    """References persisted output from one authorized capability execution."""

    capability: CapabilityType
    status: CapabilityStatus
    execution_id: str
    artifact_id: str
    search_action_id: str
    search_result_ids: Tuple[str, ...]
    evidence_ids: Tuple[str, ...]
    result_count: int

    def __post_init__(self) -> None:
        if self.capability is not CapabilityType.LOCAL_RETRIEVAL:
            raise ValueError("Unsupported research capability result type.")
        if not isinstance(self.status, CapabilityStatus):
            object.__setattr__(self, "status", CapabilityStatus(self.status))
        for name in ("execution_id", "artifact_id", "search_action_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"CapabilityResult.{name} must be a non-empty string.")
        for name in ("search_result_ids", "evidence_ids"):
            values = tuple(getattr(self, name))
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise ValueError(f"CapabilityResult.{name} must contain non-empty IDs.")
            if len(values) != len(set(values)):
                raise ValueError(f"CapabilityResult.{name} must not contain duplicate IDs.")
            object.__setattr__(self, name, values)
        if not isinstance(self.result_count, int) or isinstance(self.result_count, bool) or self.result_count < 0:
            raise ValueError("CapabilityResult.result_count must be a non-negative integer.")
        if self.result_count != len(self.search_result_ids):
            raise ValueError("CapabilityResult.result_count must match search_result_ids.")
        expected = CapabilityStatus.SUCCESS if self.result_count else CapabilityStatus.NO_RESULTS
        if self.status is not expected:
            raise ValueError("CapabilityResult status must match its result_count.")


class _RetrievalStateAccess:
    """Narrow adapter for the existing recorder; exposes no general state API."""

    __slots__ = ("__state", "created_search_ids")

    def __init__(self, state: ResearchState):
        self.__state = state
        self.created_search_ids: list[str] = []

    @property
    def research_runs(self):
        return tuple(self.__state.research_runs)

    @property
    def iterations(self):
        return tuple(self.__state.iterations)

    @property
    def research_questions(self):
        return tuple(self.__state.research_questions)

    @property
    def sources(self):
        return tuple(self.__state.sources)

    @property
    def searches(self):
        return tuple(self.__state.searches)

    @property
    def search_results(self):
        return tuple(self.__state.search_results)

    @property
    def document_versions(self):
        return tuple(self.__state.document_versions)

    @property
    def passage_references(self):
        return tuple(self.__state.passage_references)

    @property
    def evidence(self):
        return tuple(self.__state.evidence)

    def add_source(self, value: Source) -> Source:
        return self.__state.add_source(value)

    def add_search(self, value: SearchAction) -> SearchAction:
        created = self.__state.add_search(value)
        self.created_search_ids.append(created.search_id)
        return created

    def add_search_result(self, value):
        return self.__state.add_search_result(value)

    def add_passage_reference(self, value: PassageReference) -> PassageReference:
        return self.__state.add_passage_reference(value)

    def add_evidence(self, value: Evidence) -> Evidence:
        return self.__state.add_evidence(value)

    def log(self, agent: str, action: str, message: str = ""):
        return self.__state.log(agent, action, message)


class _RetrievalMutationSnapshot:
    """Rollback newly appended retrieval/artifact records after a failed run."""

    _COLLECTIONS = (
        "sources", "searches", "search_results", "passage_references",
        "evidence", "events", "research_artifacts",
    )

    def __init__(self, state: ResearchState):
        self.state = state
        self.lengths = {name: len(getattr(state, name)) for name in self._COLLECTIONS}

    def rollback(self) -> None:
        # Current Phase 5 explicitly does not support concurrent state mutation.
        for name, length in self.lengths.items():
            del getattr(self.state, name)[length:]


def execute_local_retrieval(
    plan: ResearchPlan,
    run: ResearchRun,
    task: ResearchTask,
    review: PlanReview,
    request: LocalRetrievalRequest,
    provider: RetrievalProvider,
    state: ResearchState,
    *,
    iteration_id: Optional[str] = None,
    retry_of_execution_id: Optional[str] = None,
    continuation_authorization_id: Optional[str] = None,
) -> CapabilityResult:
    """Execute LOCAL_RETRIEVAL only inside Phase 5's authorized task boundary.

    The function does not expose its ResearchState to an arbitrary handler.
    It records through a retrieval-only state façade and rolls those records
    back if retrieval, artifact recording, or task-result recording fails.
    """
    _validate_request(request, provider, state, run, iteration_id)
    snapshot = _RetrievalMutationSnapshot(state)
    access = _RetrievalStateAccess(state)
    produced: dict[str, object] = {}

    def _run_capability(_task: ResearchTask, context: TaskExecutionContext) -> TaskExecutionResult:
        recorded = retrieve_for_research(
            provider,
            access,  # type: ignore[arg-type]
            query=request.query,
            limit=request.limit,
            run_id=context.run_id,
            iteration_id=context.iteration_id,
            question_id=request.question_id,
            evidence_document_versions=request.evidence_document_versions,
            purpose=request.purpose,
        )
        if len(access.created_search_ids) != 1:
            raise RuntimeError("LOCAL_RETRIEVAL did not create exactly one SearchAction.")
        action_id = access.created_search_ids[0]
        result_ids = tuple(
            result.result_id for result in recorded.search_response.results
        )
        if any(result_id is None for result_id in result_ids):
            raise RuntimeError("LOCAL_RETRIEVAL returned a SearchResult without persisted identity.")
        evidence_ids = tuple(
            item.evidence_id for item in state.evidence[snapshot.lengths["evidence"]:]
        )
        artifact = ResearchArtifact(
            execution_id=context.execution_id,
            run_id=context.run_id,
            task_id=context.task_id,
            search_action_id=action_id,
            search_result_ids=result_ids,  # type: ignore[arg-type]
            evidence_ids=evidence_ids,
        )
        state.add_research_artifact(artifact)
        produced["artifact"] = artifact
        produced["result_count"] = len(result_ids)
        output = f"Retrieved {len(result_ids)} result(s); material is not automatically verified."
        return TaskExecutionResult(
            success=True,
            output=output,
            artifact_ids=(artifact.artifact_id,),
            evidence_ids=evidence_ids,
        )

    try:
        execution = execute_authorized_task(
            plan,
            run,
            task,
            review,
            _run_capability,
            state,
            iteration_id=iteration_id,
            retry_of_execution_id=retry_of_execution_id,
            continuation_authorization_id=continuation_authorization_id,
        )
    except (TaskExecutionError, TaskExecutionRecordingError):
        snapshot.rollback()
        raise

    artifact = produced.get("artifact")
    if not isinstance(artifact, ResearchArtifact) or execution.result is None:
        # This indicates an internal contract defect after an otherwise
        # successful execution; do not return a false capability success.
        snapshot.rollback()
        raise RuntimeError("LOCAL_RETRIEVAL completed without a persisted ResearchArtifact.")
    result_count = int(produced["result_count"])
    return CapabilityResult(
        capability=CapabilityType.LOCAL_RETRIEVAL,
        status=CapabilityStatus.SUCCESS if result_count else CapabilityStatus.NO_RESULTS,
        execution_id=execution.execution_id,
        artifact_id=artifact.artifact_id,
        search_action_id=artifact.search_action_id,
        search_result_ids=artifact.search_result_ids,
        evidence_ids=artifact.evidence_ids,
        result_count=result_count,
    )


def _validate_request_values(request: LocalRetrievalRequest) -> None:
    if not isinstance(request.query, str) or not request.query.strip():
        raise CapabilityRequestError("LOCAL_RETRIEVAL query must be a non-empty string.")
    if not isinstance(request.limit, int) or isinstance(request.limit, bool) or request.limit < 1:
        raise CapabilityRequestError("LOCAL_RETRIEVAL limit must be a positive integer.")
    if not isinstance(request.purpose, str) or not request.purpose.strip():
        raise CapabilityRequestError("LOCAL_RETRIEVAL purpose must be a non-empty string.")
    if request.question_id is not None and (
        not isinstance(request.question_id, str) or not request.question_id.strip()
    ):
        raise CapabilityRequestError("LOCAL_RETRIEVAL question_id must be a non-empty string when supplied.")
    if not isinstance(request.evidence_document_versions, Mapping):
        raise CapabilityRequestError("evidence_document_versions must be a rank-to-DocumentVersion mapping.")
    for rank, version in request.evidence_document_versions.items():
        if not isinstance(rank, int) or isinstance(rank, bool) or rank < 1:
            raise CapabilityRequestError("Evidence selection ranks must be positive 1-based integers.")
        if not isinstance(version, DocumentVersion):
            raise CapabilityRequestError("Evidence selections must reference DocumentVersion records.")


def _validate_request(
    request: LocalRetrievalRequest,
    provider: RetrievalProvider,
    state: ResearchState,
    run: ResearchRun,
    iteration_id: Optional[str],
) -> None:
    if not isinstance(request, LocalRetrievalRequest):
        raise CapabilityRequestError("A LocalRetrievalRequest is required.")
    try:
        _validate_request_values(request)
    except (AttributeError, TypeError, ValueError) as exc:
        raise CapabilityRequestError(f"Invalid LOCAL_RETRIEVAL request: {exc}") from exc
    if not isinstance(state, ResearchState):
        raise CapabilityRequestError("A ResearchState is required for capability artifact recording.")
    if not callable(getattr(provider, "retrieve", None)):
        raise CapabilityRequestError("LOCAL_RETRIEVAL requires a RetrievalProvider with retrieve().")
    if not isinstance(run, ResearchRun):
        raise CapabilityRequestError("A ResearchRun is required.")
    if iteration_id is not None:
        iteration = next((item for item in state.iterations if item.iteration_id == iteration_id), None)
        if iteration is None or iteration.run_id != run.run_id:
            raise CapabilityRequestError("iteration_id must resolve to an iteration of this ResearchRun.")
    if request.question_id is not None:
        question = next(
            (item for item in state.research_questions if item.question_id == request.question_id),
            None,
        )
        if question is None or question.run_id != run.run_id:
            raise CapabilityRequestError("question_id must resolve to a ResearchQuestion in this run.")
        if iteration_id is not None and question.iteration_id is not None and question.iteration_id != iteration_id:
            raise CapabilityRequestError("question_id belongs to another ResearchIteration.")
    registered_versions = {item.version_id: item for item in state.document_versions}
    for rank, version in request.evidence_document_versions.items():
        registered = registered_versions.get(version.version_id)
        if registered is None or registered != version:
            raise CapabilityRequestError(
                f"Evidence selection at rank {rank} must use an exactly registered DocumentVersion."
            )


__all__ = [
    "CapabilityType",
    "CapabilityStatus",
    "CapabilityRequestError",
    "LocalRetrievalRequest",
    "CapabilityResult",
    "execute_local_retrieval",
]
