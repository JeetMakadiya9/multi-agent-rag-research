"""Explicit, bounded prior-work investigation over the existing SearchProvider API.

Search results are recorded as SearchResults, not promoted to Evidence. The
capability reports bounded similarities and uncertainties; it makes no novelty
or research-gap determination.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import re
from typing import Optional, Sequence, Tuple

from llm import LLMProvider
from research_capabilities import CapabilityType
from research_plan_review import PlanReview
from research_planning import ResearchPlan, ResearchTask
from research_state import (
    PriorWorkFinding,
    PriorWorkQueryRecord,
    PriorWorkResultAssessment,
    PriorWorkSearchScope,
    ResearchArtifact,
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
from research_tools import SearchManager, SearchRequest, SearchResponse, SearchResult, SearchProvider


class PriorWorkCapabilityError(ValueError):
    """Invalid request, stored candidate, provider response, or analysis."""


@dataclass(frozen=True)
class PriorWorkInvestigationRequest:
    candidate_artifact_id: str
    candidate_id: str
    scope: PriorWorkSearchScope
    queries: Tuple[str, ...] = ()
    question_id: Optional[str] = None

    def __post_init__(self) -> None:
        for name in ("candidate_artifact_id", "candidate_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise PriorWorkCapabilityError(f"{name} must be a non-empty stored identifier.")
        if not isinstance(self.scope, PriorWorkSearchScope):
            raise PriorWorkCapabilityError("scope must be a typed PriorWorkSearchScope.")
        if not isinstance(self.queries, (tuple, list)):
            raise PriorWorkCapabilityError("queries must be a sequence of strings.")
        queries = tuple(self.queries)
        if any(not isinstance(query, str) or not query.strip() for query in queries):
            raise PriorWorkCapabilityError("queries must contain non-empty strings.")
        if len(queries) != len(set(queries)):
            raise PriorWorkCapabilityError("queries must not contain duplicates.")
        if len(queries) > self.scope.maximum_queries:
            raise PriorWorkCapabilityError("queries exceeds the configured query budget.")
        object.__setattr__(self, "queries", queries)
        if self.question_id is not None and (not isinstance(self.question_id, str) or not self.question_id.strip()):
            raise PriorWorkCapabilityError("question_id must be non-empty when supplied.")


@dataclass(frozen=True)
class PriorWorkInvestigationResult:
    capability: CapabilityType
    execution_id: str
    run_id: str
    task_id: str
    artifact_id: str
    candidate_id: str
    search_action_ids: Tuple[str, ...]
    search_result_ids: Tuple[str, ...]
    findings: Tuple[PriorWorkFinding, ...]
    status: str


class _BoundedProvider:
    """Enforce configured date/source constraints around any SearchProvider."""
    def __init__(self, provider: SearchProvider, scope: PriorWorkSearchScope, remaining: int):
        self.provider, self.scope, self.remaining = provider, scope, remaining

    @property
    def name(self) -> str:
        return self.provider.name

    def search(self, request: SearchRequest) -> SearchResponse:
        # Some SearchProvider implementations (including the existing static
        # provider) do not apply source/date filters. Bounded overfetch lets us
        # apply those constraints before consuming this query's result budget.
        provider_request = replace(request, limit=min(500, max(request.limit, self.remaining * 10)))
        response = self.provider.search(provider_request)
        from datetime import date
        filtered = []
        for result in response.results:
            if self.scope.source_types and result.source_type not in self.scope.source_types:
                continue
            if self.scope.date_from or self.scope.date_to:
                if not result.published_at:
                    continue
                try:
                    published = date.fromisoformat(result.published_at[:10])
                except ValueError:
                    continue
                if self.scope.date_from and published < date.fromisoformat(self.scope.date_from):
                    continue
                if self.scope.date_to and published > date.fromisoformat(self.scope.date_to):
                    continue
            filtered.append(result)
        return SearchResponse(
            query=response.query,
            results=filtered[: self.remaining],
            provider=response.provider,
            searched_at=response.searched_at,
            metadata=dict(response.metadata),
        )


class _MutationSnapshot:
    _FIELDS = ("sources", "searches", "search_results", "events", "research_artifacts")

    def __init__(self, state: ResearchState):
        self.state = state
        self.lengths = {name: len(getattr(state, name)) for name in self._FIELDS}
        self.updated_at = state.updated_at

    def rollback(self) -> None:
        for name, length in self.lengths.items():
            del getattr(self.state, name)[length:]
        self.state.updated_at = self.updated_at


def execute_prior_work_investigation(
    plan: ResearchPlan,
    run: ResearchRun,
    task: ResearchTask,
    review: PlanReview,
    request: PriorWorkInvestigationRequest,
    state: ResearchState,
    search_provider: SearchProvider,
    llm_provider: LLMProvider,
    *,
    iteration_id: Optional[str] = None,
    retry_of_execution_id: Optional[str] = None,
    continuation_authorization_id: Optional[str] = None,
) -> PriorWorkInvestigationResult:
    """Run explicit searches and LLM-assisted comparisons only after Phase 5 authorization."""
    _validate_entry(request, state, search_provider, llm_provider, run)
    snapshot = _MutationSnapshot(state)
    produced: dict[str, object] = {}

    def handler(_task: ResearchTask, context: TaskExecutionContext) -> TaskExecutionResult:
        if request.scope.provider_name and request.scope.provider_name != search_provider.name:
            raise PriorWorkCapabilityError("Search provider does not match the explicitly requested provider_name.")
        lineage_errors = state.validate_lineage() + state.validate_provenance()
        if lineage_errors:
            raise PriorWorkCapabilityError("ResearchState provenance/lineage is invalid: " + "; ".join(lineage_errors))
        candidate_artifact = next((a for a in state.research_artifacts if a.artifact_id == request.candidate_artifact_id), None)
        candidate = next((c for c in (candidate_artifact.improvement_candidates if candidate_artifact else ())
                          if c.candidate_id == request.candidate_id), None)
        if candidate_artifact is None or candidate_artifact.artifact_type != "improvement_candidate" or candidate_artifact.run_id != context.run_id or candidate is None:
            raise PriorWorkCapabilityError("Candidate ID must resolve to a stored improvement candidate in this ResearchRun.")
        if candidate.status != "CANDIDATE" or candidate.novelty_status != "NOT_ASSESSED" or candidate.validation_status != "NOT_VALIDATED":
            raise PriorWorkCapabilityError("Prior-work investigation requires an unassessed, unvalidated candidate.")
        candidate_execution = next((e for e in state.task_executions if e.execution_id == candidate_artifact.execution_id), None)
        if (
            candidate_execution is None or candidate_execution.status.value != "COMPLETED"
            or candidate_execution.result is None
            or candidate_artifact.artifact_id not in candidate_execution.result.artifact_ids
            or not set(candidate.supporting_evidence_ids).issubset(set(candidate_artifact.evidence_ids))
        ):
            raise PriorWorkCapabilityError("Candidate must resolve through a completed stored improvement artifact.")
        evidence_ids = set(candidate.supporting_evidence_ids)
        if not evidence_ids or not evidence_ids.issubset({e.evidence_id for e in state.evidence}):
            raise PriorWorkCapabilityError("Candidate supporting Evidence IDs do not resolve in ResearchState.")
        if request.question_id is not None:
            question = next((q for q in state.research_questions if q.question_id == request.question_id), None)
            if question is None or question.run_id != context.run_id:
                raise PriorWorkCapabilityError("question_id must resolve to a ResearchQuestion in this run.")

        queries = request.queries or _generate_queries(candidate, request.scope, llm_provider)
        if not queries or len(queries) > request.scope.maximum_queries:
            raise PriorWorkCapabilityError("Generated query set is empty or exceeds its configured budget.")
        origin = "researcher_provided" if request.queries else "system_generated"
        query_records = []
        all_results = []
        remaining = request.scope.maximum_results
        for query in queries:
            bounded = _BoundedProvider(search_provider, request.scope, remaining)
            response = SearchManager(bounded).search(
                SearchRequest(query=query, limit=max(1, remaining), source_types=list(request.scope.source_types) or None,
                              date_from=request.scope.date_from, date_to=request.scope.date_to),
                state=state,
                run_id=context.run_id,
                iteration_id=iteration_id,
                question_id=request.question_id,
            )
            if response.provider != search_provider.name:
                raise PriorWorkCapabilityError("SearchProvider returned a response with a mismatched provider name.")
            ids = tuple(item.result_id for item in response.results)
            if any(not result_id for result_id in ids):
                raise PriorWorkCapabilityError("SearchManager did not persist stable SearchResult IDs.")
            search_action = next((a for a in reversed(state.searches) if a.run_id == context.run_id and a.query == query and a.provider == response.provider), None)
            if search_action is None:
                raise PriorWorkCapabilityError("SearchManager did not persist a SearchAction for the query.")
            query_records.append(PriorWorkQueryRecord(query, search_action.search_id, origin, ids))
            all_results.extend(response.results)
            remaining -= len(response.results)

        assessments, findings, uncertainties = _assess_results(candidate, all_results, llm_provider)
        result_ids = tuple(item.result_id for item in all_results)
        artifact = ResearchArtifact(
            execution_id=context.execution_id, run_id=context.run_id, task_id=context.task_id,
            search_action_id=None, search_result_ids=result_ids, evidence_ids=(),
            artifact_type="prior_work_investigation", capability="prior_work_investigation",
            prior_work_candidate_artifact_id=request.candidate_artifact_id,
            prior_work_candidate_id=request.candidate_id, prior_work_scope=request.scope,
            prior_work_provider=search_provider.name, prior_work_queries=tuple(query_records),
            prior_work_assessments=assessments, prior_work_findings=findings,
            prior_work_status="COMPLETED" if result_ids else "ZERO_RESULTS",
            prior_work_uncertainties=uncertainties,
        )
        state.add_research_artifact(artifact)
        produced["artifact"] = artifact
        return TaskExecutionResult(success=True, output=f"Recorded bounded prior-work search results ({len(result_ids)} result(s)); novelty was not assessed.", artifact_ids=(artifact.artifact_id,))

    try:
        execution = execute_authorized_task(plan, run, task, review, handler, state,
                                            iteration_id=iteration_id,
                                            retry_of_execution_id=retry_of_execution_id,
                                            continuation_authorization_id=continuation_authorization_id)
    except (TaskExecutionError, TaskExecutionRecordingError):
        snapshot.rollback()
        raise
    except Exception:
        # execute_authorized_task stores handler failures as TaskExecutionError; this path
        # is for recording/capability exceptions surfaced before that wrapper can retain it.
        snapshot.rollback()
        raise
    artifact = produced.get("artifact")
    if not isinstance(artifact, ResearchArtifact) or execution.result is None:
        snapshot.rollback()
        raise RuntimeError("Prior-work execution completed without a persisted artifact.")
    return PriorWorkInvestigationResult(
        capability=CapabilityType.PRIOR_WORK_INVESTIGATION, execution_id=execution.execution_id,
        run_id=run.run_id, task_id=task.task_id, artifact_id=artifact.artifact_id,
        candidate_id=request.candidate_id,
        search_action_ids=tuple(item.search_action_id for item in artifact.prior_work_queries),
        search_result_ids=artifact.search_result_ids, findings=artifact.prior_work_findings,
        status=artifact.prior_work_status or "COMPLETED",
    )


def _validate_entry(request, state, search_provider, llm_provider, run) -> None:
    if not isinstance(request, PriorWorkInvestigationRequest):
        raise PriorWorkCapabilityError("A PriorWorkInvestigationRequest is required.")
    if not isinstance(state, ResearchState):
        raise PriorWorkCapabilityError("A ResearchState is required.")
    if not isinstance(run, ResearchRun):
        raise PriorWorkCapabilityError("A ResearchRun is required.")
    if not callable(getattr(search_provider, "search", None)) or not isinstance(getattr(search_provider, "name", None), str):
        raise PriorWorkCapabilityError("An explicit SearchProvider with name and search() is required.")
    if not callable(getattr(llm_provider, "generate", None)):
        raise PriorWorkCapabilityError("An explicit LLMProvider with generate() is required.")


def _generate_queries(candidate, scope: PriorWorkSearchScope, provider: LLMProvider) -> Tuple[str, ...]:
    prompt = ({"role": "system", "content": "Create bounded search queries for prior-work investigation. Return JSON only: {\"queries\":[string,...]}. Do not state that novelty is established, that no prior work exists, or that a research gap is proven."},
              {"role": "user", "content": json.dumps({"problem": candidate.problem_statement, "candidate": candidate.proposed_change, "maximum_queries": scope.maximum_queries}, ensure_ascii=False)})
    data = _parse_json(provider.generate(prompt, temperature=0.0), {"queries"})
    queries = data["queries"]
    if not isinstance(queries, list) or not queries or len(queries) > scope.maximum_queries:
        raise PriorWorkCapabilityError("LLM query response must contain a non-empty bounded queries list.")
    values = tuple(queries)
    if any(not isinstance(value, str) or not value.strip() for value in values) or len(values) != len(set(values)):
        raise PriorWorkCapabilityError("LLM queries must be unique non-empty strings.")
    return values


def _assess_results(candidate, results: Sequence[SearchResult], provider: LLMProvider):
    if not results:
        return (), (), ("No SearchResults were returned within the configured search scope.",)
    payload = [{"search_result_id": r.result_id, "title": r.title, "snippet": r.snippet,
                "content": r.content, "source_type": r.source_type, "published_at": r.published_at}
               for r in results]
    schema = {"assessments": [{"search_result_id": "exact ID", "relevance": "RELEVANT|POTENTIALLY_RELEVANT|NOT_RELEVANT", "reason": "text", "relationship": "DIRECT_OVERLAP|PARTIAL_OVERLAP|RELATED_DISTINCT|INSUFFICIENT_INFORMATION|null", "problem_overlap": "text|null", "method_overlap": "text|null", "dataset_overlap": "text|null", "evaluation_overlap": "text|null", "differences": ["text"], "limitations": ["text"]}], "uncertainties": ["text"]}
    prompt = ({"role": "system", "content": "Compare each returned SearchResult with the candidate problem only. Return JSON matching this schema exactly: " + json.dumps(schema) + " Search results are not verified evidence. Do not claim novelty, first-ever status, absence of prior work, or a proven gap. Do not infer facts absent from result text. Every result must appear once in original order. NOT_RELEVANT must have relationship null and does not become a finding."},
              {"role": "user", "content": json.dumps({"problem": candidate.problem_statement, "proposed_change": candidate.proposed_change, "results": payload}, ensure_ascii=False)})
    data = _parse_json(provider.generate(prompt, temperature=0.0), {"assessments", "uncertainties"})
    rows, uncertainties = data["assessments"], data["uncertainties"]
    if not isinstance(rows, list) or not isinstance(uncertainties, list):
        raise PriorWorkCapabilityError("LLM comparison response must contain assessments and uncertainties lists.")
    if len(rows) != len(results):
        raise PriorWorkCapabilityError("LLM comparison response must assess every SearchResult exactly once.")
    if any(not isinstance(item, str) or not item.strip() for item in uncertainties):
        raise PriorWorkCapabilityError("LLM uncertainties must be non-empty strings.")
    forbidden = re.compile(r"\b(this is novel|first[- ]ever|no (?:prior|previous) work|never been done|proven research gap|novel contribution)\b", re.I)
    if any(forbidden.search(item) for item in uncertainties):
        raise PriorWorkCapabilityError("Comparison output contains an unsupported novelty or prior-work conclusion.")
    assessments, findings = [], []
    for result, row in zip(results, rows):
        required = {"search_result_id", "relevance", "reason", "relationship", "problem_overlap", "method_overlap", "dataset_overlap", "evaluation_overlap", "differences", "limitations"}
        if not isinstance(row, dict) or set(row) != required or row["search_result_id"] != result.result_id:
            raise PriorWorkCapabilityError("Malformed or mismatched SearchResult comparison row.")
        texts = [row.get(key) for key in ("reason", "problem_overlap", "method_overlap", "dataset_overlap", "evaluation_overlap") if isinstance(row.get(key), str)] + list(row.get("differences", [])) + list(row.get("limitations", []))
        if any(forbidden.search(text) for text in texts):
            raise PriorWorkCapabilityError("Comparison output contains an unsupported novelty or prior-work conclusion.")
        assessment = PriorWorkResultAssessment(row["search_result_id"], row["relevance"], row["reason"])
        assessments.append(assessment)
        if assessment.relevance == "NOT_RELEVANT":
            if row["relationship"] is not None:
                raise PriorWorkCapabilityError("NOT_RELEVANT output must not include a relationship classification.")
            continue
        if row["relationship"] not in {"DIRECT_OVERLAP", "PARTIAL_OVERLAP", "RELATED_DISTINCT", "INSUFFICIENT_INFORMATION"}:
            raise PriorWorkCapabilityError("Relevant result requires a supported relationship classification.")
        for key in ("differences", "limitations"):
            if not isinstance(row[key], list) or any(not isinstance(v, str) or not v.strip() for v in row[key]):
                raise PriorWorkCapabilityError(f"Comparison {key} must be a list of non-empty strings.")
        for key in ("problem_overlap", "method_overlap", "dataset_overlap", "evaluation_overlap"):
            if row[key] is not None and (not isinstance(row[key], str) or not row[key].strip()):
                raise PriorWorkCapabilityError(f"Comparison {key} must be non-empty text or null.")
        findings.append(PriorWorkFinding(
            search_result_id=result.result_id, source_id=result.source_id, relevance=assessment.relevance,
            relationship=row["relationship"], problem_overlap=row["problem_overlap"], method_overlap=row["method_overlap"],
            dataset_overlap=row["dataset_overlap"], evaluation_overlap=row["evaluation_overlap"],
            differences=tuple(row["differences"]), limitations=tuple(row["limitations"]),
        ))
    if any(not isinstance(item, str) or not item.strip() for item in uncertainties):
        raise PriorWorkCapabilityError("uncertainties must contain non-empty strings.")
    return tuple(assessments), tuple(findings), tuple(uncertainties)


def _parse_json(raw: str, expected_keys: set[str]) -> dict:
    if not isinstance(raw, str):
        raise PriorWorkCapabilityError("LLM response must be JSON text.")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PriorWorkCapabilityError("LLM response is malformed JSON.") from exc
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise PriorWorkCapabilityError("LLM response does not match the required JSON object schema.")
    return value


__all__ = ["PriorWorkCapabilityError", "PriorWorkInvestigationRequest", "PriorWorkInvestigationResult", "execute_prior_work_investigation"]
