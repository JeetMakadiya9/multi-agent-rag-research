from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_capabilities import CapabilityType, LocalRetrievalRequest, execute_local_retrieval
from research_improvement_capability import ImprovementGenerationRequest, execute_improvement_generation
from research_plan_review import PlanReview, ReviewStatus
from research_planning import ResearchObjective, ResearchPlan, ResearchPlanQuestion, ResearchRequest, ResearchTask
from research_prior_work_capability import (
    PriorWorkCapabilityError, PriorWorkInvestigationRequest, execute_prior_work_investigation,
)
from research_retrieval import RetrievalEvidence, StaticRetrievalProvider, register_rag_document_version
from research_run_authorization import authorize_research_run
from research_state import PriorWorkSearchScope, ResearchClaim, ResearchState
from research_task_authorization import authorize_research_tasks
from research_task_execution import TaskExecutionAuthorizationError, TaskExecutionError, TaskExecutionStatus
from research_tools import SearchResult, StaticSearchProvider


class QueueLLM:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def generate(self, messages, *, temperature=0.0, timeout=120):
        self.calls.append(messages)
        if not self.responses:
            raise AssertionError("Unexpected LLM call")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _candidate_response(evidence_id, claim_id):
    return json.dumps({"candidates": [{
        "title": "Candidate approach", "motivation": "The limitation motivates investigation.",
        "proposed_change": "Investigate an adaptive method.", "rationale": "It may address the limitation.",
        "expected_benefit": "Could reduce the limitation's effect.",
        "assumptions": ["The limitation applies to the target setting."],
        "risks": ["The method may add complexity."],
        "validation_needed": ["Effectiveness requires validation."],
        "supporting_evidence_ids": [evidence_id], "supporting_claim_ids": [claim_id],
        "supporting_verification_artifact_ids": [],
    }]})


def _assessments(result_ids, *, relevance="RELEVANT"):
    return json.dumps({"assessments": [{
        "search_result_id": rid, "relevance": relevance, "reason": "The result concerns the documented problem.",
        "relationship": "PARTIAL_OVERLAP" if relevance != "NOT_RELEVANT" else None,
        "problem_overlap": "Similar problem", "method_overlap": None, "dataset_overlap": None,
        "evaluation_overlap": None, "differences": ["Different setting"], "limitations": ["Only result metadata was available."],
    } for rid in result_ids], "uncertainties": ["Search scope is bounded and cannot establish novelty."]})


def _fixture(domain="agriculture"):
    q = ResearchPlanQuestion("What limitation is documented?", question_id=f"q-{domain}")
    tasks = [
        ResearchTask("Retrieve stored material", q.question_id, task_id=f"retrieve-{domain}"),
        ResearchTask("Generate a candidate", q.question_id, task_id=f"improve-{domain}"),
        ResearchTask("Investigate prior work", q.question_id, task_id=f"prior-{domain}"),
    ]
    plan = ResearchPlan(request=ResearchRequest("Study a documented problem", "Explore candidate responses", domain=domain),
                        objective=ResearchObjective("Review relevant prior work"), questions=[q], tasks=tasks,
                        plan_id=f"plan-{domain}")
    review = PlanReview(plan.plan_id, plan.revision, ReviewStatus.APPROVED, reviewer="researcher")
    state = ResearchState(user_request="synthetic prior-work test")
    run = authorize_research_run(plan, review, state)
    authorize_research_tasks(plan, run, [task.task_id for task in tasks], review, state)
    iteration = state.create_iteration(run.run_id)
    state_question = state.add_question(q.text, run_id=run.run_id, iteration_id=iteration.iteration_id)
    version = register_rag_document_version(state, source_id=f"src-{domain}", document_id=f"doc-{domain}",
                                            filename="local.pdf", version_id=f"ver-{domain}")
    evidence = RetrievalEvidence(evidence_id=f"chunk-{domain}", text="A documented limitation affects performance.",
                                 source_id=f"src-{domain}", filename="local.pdf", page=2,
                                 document_id=f"doc-{domain}", chunk_id=f"chunk-{domain}")
    execute_local_retrieval(plan, run, tasks[0], review,
                            LocalRetrievalRequest("documented limitation", limit=1, evidence_document_versions={1: version}),
                            StaticRetrievalProvider([evidence]), state, iteration_id=iteration.iteration_id)
    stored_evidence = state.evidence[0]
    claim = state.add_claim(ResearchClaim(text="A documented limitation affects performance.", status="SUPPORTED",
                                          claim_id=f"claim-{domain}", evidence_ids=[stored_evidence.evidence_id],
                                          source_ids=[stored_evidence.source_id], verification_reason="Synthetic fixture input."))
    candidate_llm = QueueLLM(_candidate_response(stored_evidence.evidence_id, claim.claim_id))
    candidate_result = execute_improvement_generation(
        plan, run, tasks[1], review,
        ImprovementGenerationRequest("Known limitation: performance is affected.", (stored_evidence.evidence_id,), verified_claim_ids=(claim.claim_id,)),
        state, candidate_llm, iteration_id=iteration.iteration_id,
    )
    candidate_artifact = next(a for a in state.research_artifacts if a.artifact_id == candidate_result.artifact_id)
    candidate = candidate_artifact.improvement_candidates[0]
    provider = StaticSearchProvider([
        SearchResult(f"paper-{domain}-a", "Related analysis", snippet="The limitation is studied in a different context.",
                     content="Related method and problem", source_type="paper", published_at="2024-02-01"),
        SearchResult(f"paper-{domain}-b", "Another study", snippet="A different evaluation of the same issue.",
                     content="Related issue, different evaluation", source_type="paper", published_at="2023-05-01"),
    ])
    return dict(plan=plan, run=run, review=review, state=state, tasks=tasks, iteration=iteration,
                state_question=state_question,
                candidate=candidate, candidate_artifact=candidate_artifact, provider=provider)


class ResearchPriorWorkCapabilityTests(unittest.TestCase):
    def _request(self, f, **kwargs):
        values = dict(candidate_artifact_id=f["candidate_artifact"].artifact_id,
                      candidate_id=f["candidate"].candidate_id,
                      scope=PriorWorkSearchScope(maximum_queries=2, maximum_results=4,
                                                 source_types=("paper",), provider_name="static"),
                      queries=("documented limitation method", "limitation evaluation"),
                      question_id=f["state_question"].question_id)
        values.update(kwargs)
        return PriorWorkInvestigationRequest(**values)

    def _execute(self, f, llm, request=None, **kwargs):
        return execute_prior_work_investigation(f["plan"], f["run"], f["tasks"][2], f["review"],
                                                request or self._request(f), f["state"], f["provider"], llm,
                                                iteration_id=f["iteration"].iteration_id, **kwargs)

    def test_explicit_prior_work_records_search_lineage_and_typed_findings(self):
        f = _fixture()
        # SearchManager can persist repeated sources, and each result is linked to its own query.
        llm = QueueLLM(_assessments(["placeholder"]))
        # Resolve generated persisted IDs only after provider calls; a tiny provider-aware response is used below.
        class AssessingLLM:
            def __init__(self): self.calls = []
            def generate(self, messages, *, temperature=0.0, timeout=120):
                self.calls.append(messages)
                user = json.loads(messages[1]["content"])
                return _assessments([item["search_result_id"] for item in user["results"]])
        llm = AssessingLLM()
        before_evidence = len(f["state"].evidence)
        result = self._execute(f, llm)
        self.assertEqual(result.capability, CapabilityType.PRIOR_WORK_INVESTIGATION)
        self.assertEqual(len(result.search_action_ids), 2)
        self.assertGreater(len(result.search_result_ids), 0)
        self.assertEqual(len(result.search_result_ids), len(result.findings))
        artifact = next(a for a in f["state"].research_artifacts if a.artifact_id == result.artifact_id)
        self.assertEqual(artifact.prior_work_candidate_id, f["candidate"].candidate_id)
        self.assertEqual(artifact.evidence_ids, ())
        self.assertEqual(len(f["state"].evidence), before_evidence)
        self.assertEqual(f["state"].task_executions[-1].status, TaskExecutionStatus.COMPLETED)
        self.assertEqual(f["state"].validate_lineage(), [])
        self.assertEqual(len(llm.calls), 1)
        restored = ResearchState.from_dict(f["state"].to_dict())
        self.assertEqual(restored.validate_lineage(), [])
        restored_artifact = next(a for a in restored.research_artifacts if a.artifact_id == result.artifact_id)
        self.assertEqual(restored_artifact.prior_work_findings, artifact.prior_work_findings)

    def test_query_generation_is_explicit_bounded_and_tagged_system_generated(self):
        f = _fixture("physics")
        class AssessingLLM:
            calls = 0
            def generate(self, messages, *, temperature=0.0, timeout=120):
                self.calls += 1
                if self.calls == 1: return json.dumps({"queries": ["limitation performance method"]})
                ids = [x["search_result_id"] for x in json.loads(messages[1]["content"])["results"]]
                return _assessments(ids)
        llm = AssessingLLM()
        result = self._execute(f, llm, self._request(f, queries=()))
        artifact = next(a for a in f["state"].research_artifacts if a.artifact_id == result.artifact_id)
        self.assertEqual(len(artifact.prior_work_queries), 1)
        self.assertEqual(artifact.prior_work_queries[0].origin, "system_generated")
        self.assertEqual(llm.calls, 2)

    def test_zero_results_is_successful_and_serializes_without_evidence_creation(self):
        f = _fixture("climate")
        f["provider"] = StaticSearchProvider([])
        class NoResultsLLM:
            calls = 0
            def generate(self, messages, *, temperature=0.0, timeout=120):
                self.calls += 1
                return _assessments([])
        llm = NoResultsLLM()
        before = len(f["state"].evidence)
        result = self._execute(f, llm)
        self.assertEqual(result.status, "ZERO_RESULTS")
        self.assertEqual(result.findings, ())
        self.assertEqual(llm.calls, 0)
        restored = ResearchState.from_dict(f["state"].to_dict())
        self.assertEqual(restored.validate_lineage(), [])
        self.assertEqual(len(restored.evidence), before)

    def test_source_type_filter_and_result_budget_are_recorded_without_promoting_search_hits(self):
        f = _fixture("bounded")
        f["provider"] = StaticSearchProvider([
            SearchResult("nonpaper", "Web result", snippet="limitation method", source_type="webpage"),
            SearchResult("paper", "Paper result", snippet="limitation method", source_type="paper"),
        ])
        class RelevantOnly:
            def generate(self, messages, *, temperature=0.0, timeout=120):
                ids = [item["search_result_id"] for item in json.loads(messages[1]["content"])["results"]]
                rows = json.loads(_assessments(ids))
                rows["assessments"][0]["relevance"] = "NOT_RELEVANT"
                rows["assessments"][0]["relationship"] = None
                return json.dumps(rows)
        result = self._execute(f, RelevantOnly(), self._request(f, scope=PriorWorkSearchScope(
            maximum_queries=2, maximum_results=1, source_types=("paper",), provider_name="static")))
        artifact = next(a for a in f["state"].research_artifacts if a.artifact_id == result.artifact_id)
        self.assertEqual(len(artifact.search_result_ids), 1)
        self.assertEqual(len(artifact.prior_work_queries), 2)
        self.assertEqual(artifact.prior_work_queries[1].search_result_ids, ())
        stored = next(item for item in f["state"].search_results if item.result_id == artifact.search_result_ids[0])
        self.assertEqual(stored.source_id, "paper")
        self.assertEqual(artifact.prior_work_findings, ())
        self.assertEqual(len(f["state"].evidence), 1)
        self.assertEqual(artifact.prior_work_scope.maximum_results, 1)
        self.assertEqual(f["state"].validate_lineage(), [])

    def test_unapproved_task_never_calls_search_or_llm(self):
        f = _fixture("unauthorized")
        calls = []
        class ExplodingProvider:
            name = "static"
            def search(self, request): calls.append("search"); raise AssertionError("unauthorized search")
        with self.assertRaises(TaskExecutionAuthorizationError):
            execute_prior_work_investigation(f["plan"], f["run"], ResearchTask("No", f["plan"].questions[0].question_id),
                f["review"], self._request(f), f["state"], ExplodingProvider(), QueueLLM("{}"))
        self.assertEqual(calls, [])

    def test_invalid_candidate_and_scope_fail_without_search(self):
        f = _fixture("invalid")
        with self.assertRaises((PriorWorkCapabilityError, TaskExecutionError)):
            self._execute(f, QueueLLM("{}"), self._request(f, candidate_id="missing"))
        self.assertFalse(any(a.artifact_type == "prior_work_investigation" for a in f["state"].research_artifacts))
        with self.assertRaises(ValueError):
            PriorWorkSearchScope(maximum_queries=6)
        with self.assertRaises(PriorWorkCapabilityError):
            PriorWorkInvestigationRequest(f["candidate_artifact"].artifact_id, f["candidate"].candidate_id,
                                          PriorWorkSearchScope(), queries=("same", "same"))

    def test_malformed_comparison_fails_and_rolls_back_search_records_but_keeps_failure(self):
        f = _fixture("malformed")
        previous = (len(f["state"].searches), len(f["state"].search_results), len(f["state"].research_artifacts))
        with self.assertRaises(TaskExecutionError):
            self._execute(f, QueueLLM('{"assessments": []}'))
        self.assertEqual((len(f["state"].searches), len(f["state"].search_results), len(f["state"].research_artifacts)), previous)
        self.assertEqual(f["state"].task_executions[-1].status, TaskExecutionStatus.FAILED)

    def test_artifact_write_failure_rolls_back_search_and_source_records(self):
        f = _fixture("artifact-write")
        before = {name: len(getattr(f["state"], name)) for name in ("sources", "searches", "search_results", "research_artifacts")}
        class Good:
            def generate(self, messages, *, temperature=0.0, timeout=120):
                ids = [x["search_result_id"] for x in json.loads(messages[1]["content"])["results"]]
                return _assessments(ids)
        with patch.object(f["state"], "add_research_artifact", side_effect=RuntimeError("state write rejected")):
            with self.assertRaises(TaskExecutionError):
                self._execute(f, Good())
        self.assertEqual({name: len(getattr(f["state"], name)) for name in before}, before)
        self.assertEqual(f["state"].task_executions[-1].status, TaskExecutionStatus.FAILED)

    def test_unsupported_novelty_claim_is_rejected(self):
        f = _fixture("novelty")
        class BadLLM:
            def generate(self, messages, *, temperature=0.0, timeout=120):
                ids = [x["search_result_id"] for x in json.loads(messages[1]["content"])["results"]]
                rows = json.loads(_assessments(ids))
                rows["assessments"][0]["reason"] = "This is novel and the first-ever method."
                return json.dumps(rows)
        with self.assertRaises(TaskExecutionError):
            self._execute(f, BadLLM())
        self.assertFalse(any(a.artifact_type == "prior_work_investigation" for a in f["state"].research_artifacts))

    def test_retry_preserves_failed_attempt_and_candidate_unchanged(self):
        f = _fixture("retry")
        candidate_snapshot = f["candidate"].to_dict()
        with self.assertRaises(TaskExecutionError):
            self._execute(f, QueueLLM("invalid json"))
        failed = f["state"].task_executions[-1]
        class Good:
            def generate(self, messages, *, temperature=0.0, timeout=120):
                ids = [x["search_result_id"] for x in json.loads(messages[1]["content"])["results"]]
                return _assessments(ids)
        result = self._execute(f, Good(), retry_of_execution_id=failed.execution_id)
        self.assertEqual(failed.status, TaskExecutionStatus.FAILED)
        self.assertEqual(f["state"].task_executions[-1].retry_of_execution_id, failed.execution_id)
        self.assertEqual(f["candidate"].to_dict(), candidate_snapshot)
        self.assertEqual(f["state"].validate_lineage(), [])


if __name__ == "__main__":
    unittest.main()
