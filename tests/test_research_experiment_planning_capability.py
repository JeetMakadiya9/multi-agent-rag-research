from __future__ import annotations

import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_capabilities import CapabilityType, LocalRetrievalRequest, execute_local_retrieval
from research_experiment_planning_capability import (
    ExperimentPlanningError,
    ExperimentPlanningRequest,
    ExperimentPlanningResult,
    execute_experiment_planning,
)
from research_improvement_capability import ImprovementGenerationRequest, execute_improvement_generation
from research_prior_work_capability import PriorWorkInvestigationRequest, execute_prior_work_investigation
from research_plan_review import PlanReview, ReviewStatus
from research_planning import ResearchObjective, ResearchPlan, ResearchPlanQuestion, ResearchRequest, ResearchTask
from research_retrieval import RetrievalEvidence, StaticRetrievalProvider, register_rag_document_version
from research_run_authorization import authorize_research_run
from research_state import PriorWorkSearchScope, ResearchArtifact, ResearchClaim, ResearchState
from research_task_authorization import authorize_research_tasks
from research_task_execution import TaskExecutionAuthorizationError, TaskExecutionError, TaskExecutionStatus
from research_tools import SearchResult, StaticSearchProvider


class QueueLLM:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def generate(self, messages, *, temperature=0.0, timeout=120):
        self.calls.append((messages, temperature, timeout))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _candidate_response(evidence_id, claim_id):
    return json.dumps({"candidates": [{
        "title": "Candidate approach", "motivation": "The documented limitation motivates investigation.",
        "proposed_change": "Investigate an adaptive approach.", "rationale": "It may address the limitation.",
        "expected_benefit": "Could reduce sensitivity to the identified condition.",
        "assumptions": ["The limitation applies to the target setting."],
        "risks": ["The method may add complexity."],
        "validation_needed": ["Effectiveness requires empirical validation."],
        "supporting_evidence_ids": [evidence_id], "supporting_claim_ids": [claim_id],
        "supporting_verification_artifact_ids": [],
    }]})


def _plan_response(**overrides):
    value = {
        "objective": "Assess whether the proposed approach addresses the documented limitation.",
        "hypothesis": "Investigate whether the proposed approach reduces sensitivity under the specified condition.",
        "proposed_method": "Compare the candidate method with a researcher-selected baseline.",
        "baseline": "BASELINE_REQUIRES_RESEARCHER_SPECIFICATION",
        "dataset_status": "DATASET_REQUIRES_RESEARCHER_SELECTION",
        "selected_dataset": None,
        "data_requirements": ["Data suitable for the target condition must be identified."],
        "experimental_setup": ["The setup requires researcher specification before execution."],
        "controls": [], "ablations": [],
        "metrics": [{"name": "Suitable outcome measure", "measures": "The outcome relevant to the objective.",
                     "rationale": "A measure should match the research objective.",
                     "suitability": "REQUIRES_RESEARCHER_REVIEW"}],
        "expected_observations": ["The comparison may show differences that require interpretation."],
        "interpretation_criteria": ["If the prespecified measure changes under matched conditions, that would warrant further analysis."],
        "confounders": ["Differences in data composition could affect interpretation."],
        "limitations": ["This plan does not establish feasibility or effectiveness."],
        "reproducibility_requirements": ["Protocol details remain TO_BE_SPECIFIED."],
        "resource_requirements": ["Resource needs are UNKNOWN/TO_BE_ESTIMATED."],
        "assumptions": ["PLANNING_ASSUMPTION: the candidate is applicable to the target setting."],
        "unresolved_requirements": ["BASELINE_REQUIRES_RESEARCHER_SPECIFICATION",
                                    "DATASET_REQUIRES_RESEARCHER_SELECTION"],
    }
    value.update(overrides)
    return json.dumps(value)


def _fixture(domain="agriculture"):
    question = ResearchPlanQuestion("What limitation is documented?", question_id=f"q-{domain}")
    retrieval_task = ResearchTask("Retrieve stored source material", question.question_id, task_id=f"retrieve-{domain}")
    improve_task = ResearchTask("Generate a candidate improvement", question.question_id, task_id=f"improve-{domain}")
    prior_task = ResearchTask("Investigate supplied prior work", question.question_id, task_id=f"prior-{domain}")
    planning_task = ResearchTask("Prepare a validation plan", question.question_id, task_id=f"experiment-{domain}")
    plan = ResearchPlan(
        request=ResearchRequest("Investigate a documented limitation", "Explore candidate responses", domain=domain),
        objective=ResearchObjective("Assess candidate approaches using stored research inputs."),
        questions=[question], tasks=[retrieval_task, improve_task, prior_task, planning_task], plan_id=f"plan-{domain}",
    )
    review = PlanReview(plan.plan_id, plan.revision, ReviewStatus.APPROVED, reviewer="researcher")
    state = ResearchState(user_request="Synthetic experiment-planning test")
    run = authorize_research_run(plan, review, state)
    authorize_research_tasks(plan, run, [t.task_id for t in (retrieval_task, improve_task, prior_task, planning_task)], review, state)
    iteration = state.create_iteration(run.run_id)
    version = register_rag_document_version(state, source_id=f"src-{domain}", document_id=f"doc-{domain}",
                                            filename="synthetic.pdf", version_id=f"ver-{domain}")
    result = RetrievalEvidence(evidence_id=f"chunk-{domain}", text="A documented limitation affects performance.",
                               source_id=f"src-{domain}", filename="synthetic.pdf", page=2,
                               document_id=f"doc-{domain}", chunk_id=f"chunk-{domain}")
    execute_local_retrieval(plan, run, retrieval_task, review,
                            LocalRetrievalRequest("documented limitation", limit=1,
                                                  evidence_document_versions={1: version}),
                            StaticRetrievalProvider([result]), state, iteration_id=iteration.iteration_id)
    evidence = state.evidence[0]
    claim = state.add_claim(ResearchClaim(text="The limitation affects performance.", status="SUPPORTED",
                                          claim_id=f"claim-{domain}", evidence_ids=[evidence.evidence_id],
                                          source_ids=[evidence.source_id], verification_reason="Synthetic test input."))
    generation = execute_improvement_generation(
        plan, run, improve_task, review,
        ImprovementGenerationRequest("Researcher limitation: performance is affected.", (evidence.evidence_id,),
                                     verified_claim_ids=(claim.claim_id,)),
        state, QueueLLM(_candidate_response(evidence.evidence_id, claim.claim_id)),
        iteration_id=iteration.iteration_id,
    )
    artifact = next(a for a in state.research_artifacts if a.artifact_id == generation.artifact_id)
    return {"plan": plan, "review": review, "run": run, "task": planning_task, "prior_task": prior_task, "state": state,
            "iteration": iteration, "evidence": evidence, "candidate_artifact": artifact,
            "candidate": artifact.improvement_candidates[0]}


def _request(f, **overrides):
    values = {"improvement_artifact_id": f["candidate_artifact"].artifact_id,
              "candidate_id": f["candidate"].candidate_id, "domain": f["plan"].request.domain,
              "researcher_constraints": ("Do not assume data availability.",),
              "researcher_requirements": ("Require a controlled comparison.",)}
    values.update(overrides)
    return ExperimentPlanningRequest(**values)


class ResearchExperimentPlanningCapabilityTests(unittest.TestCase):
    def _execute(self, f, llm, request=None, **kwargs):
        return execute_experiment_planning(f["plan"], f["run"], f["task"], f["review"],
                                           request or _request(f), f["state"], llm,
                                           iteration_id=f["iteration"].iteration_id, **kwargs)

    def test_authorized_request_stores_typed_review_required_plan_and_round_trips(self):
        f = _fixture()
        llm = QueueLLM(_plan_response())
        before_evidence = list(f["state"].evidence)
        candidate_before = f["candidate"]
        task_status_before = f["task"].status
        result = self._execute(f, llm)
        self.assertEqual(result.capability, CapabilityType.EXPERIMENT_PLANNING)
        self.assertEqual(llm.calls[0][1], 0.0)
        self.assertEqual(result.experiment_plan.status, "REQUIRES_RESEARCHER_REVIEW")
        self.assertEqual(result.experiment_plan.candidate_id, f["candidate"].candidate_id)
        self.assertEqual(result.experiment_plan.supporting_evidence_ids, (f["evidence"].evidence_id,))
        self.assertEqual(result.experiment_plan.researcher_constraints, ("Do not assume data availability.",))
        self.assertEqual(result.experiment_plan.researcher_requirements, ("Require a controlled comparison.",))
        self.assertFalse(hasattr(result.experiment_plan, "novelty_score"))
        self.assertEqual(f["candidate"], candidate_before)
        self.assertEqual(f["task"].status, task_status_before)
        self.assertEqual(result.experiment_plan.baseline, "BASELINE_REQUIRES_RESEARCHER_SPECIFICATION")
        self.assertIsNone(result.experiment_plan.selected_dataset)
        artifact = next(a for a in f["state"].research_artifacts if a.artifact_id == result.artifact_id)
        self.assertIs(artifact.experiment_plan, result.experiment_plan)
        self.assertEqual(artifact.artifact_type, "experiment_plan")
        self.assertEqual(artifact.capability, "experiment_planning")
        self.assertEqual(f["state"].evidence, before_evidence)
        self.assertEqual(f["state"].task_executions[-1].status, TaskExecutionStatus.COMPLETED)
        restored = ResearchState.from_dict(f["state"].to_dict())
        self.assertEqual(restored.validate_lineage(), [])
        loaded = next(a for a in restored.research_artifacts if a.artifact_id == result.artifact_id)
        self.assertEqual(loaded.experiment_plan, result.experiment_plan)
        self.assertEqual(ExperimentPlanningResult.from_dict(result.to_dict()), result)

    def test_unauthorized_execution_does_not_call_llm_or_create_plan(self):
        f = _fixture("physics")
        llm = QueueLLM(_plan_response())
        before = len(f["state"].research_artifacts)
        with self.assertRaises(TaskExecutionAuthorizationError):
            execute_experiment_planning(f["plan"], f["run"], ResearchTask("Other", f["plan"].questions[0].question_id),
                                        f["review"], _request(f), f["state"], llm)
        self.assertEqual(llm.calls, [])
        self.assertEqual(len(f["state"].research_artifacts), before)

    def test_authorization_rejects_wrong_revision_review_and_unregistered_run_before_llm(self):
        f = _fixture("chemistry")
        llm = QueueLLM(_plan_response())
        with self.assertRaises(TaskExecutionAuthorizationError):
            execute_experiment_planning(f["plan"], replace(f["run"], plan_revision=f["run"].plan_revision + 1),
                                        f["task"], f["review"], _request(f), f["state"], llm)
        other_review = PlanReview(f["plan"].plan_id, f["plan"].revision, ReviewStatus.APPROVED,
                                  reviewer="researcher", review_id="different-review")
        with self.assertRaises(TaskExecutionAuthorizationError):
            execute_experiment_planning(f["plan"], f["run"], f["task"], other_review, _request(f), f["state"], llm)
        with self.assertRaises(TaskExecutionAuthorizationError):
            execute_experiment_planning(f["plan"], replace(f["run"], run_id="unregistered-run"), f["task"],
                                        f["review"], _request(f), f["state"], llm)
        self.assertEqual(llm.calls, [])

    def test_valid_prior_work_reference_is_preserved_and_wrong_run_is_rejected(self):
        f = _fixture("energy")
        class AssessmentLLM:
            calls = 0
            def generate(self, messages, *, temperature=0.0, timeout=120):
                self.calls += 1
                results = json.loads(messages[1]["content"])["results"]
                return json.dumps({"assessments": [{
                    "search_result_id": item["search_result_id"], "relevance": "RELEVANT",
                    "reason": "The result concerns the candidate problem.", "relationship": "PARTIAL_OVERLAP",
                    "problem_overlap": "Related limitation", "method_overlap": None,
                    "dataset_overlap": None, "evaluation_overlap": None,
                    "differences": ["Different conditions"], "limitations": ["Limited source metadata."],
                } for item in results], "uncertainties": ["Scope is bounded."]})
        prior_llm = AssessmentLLM()
        prior = execute_prior_work_investigation(
            f["plan"], f["run"], f["prior_task"], f["review"],
            PriorWorkInvestigationRequest(
                f["candidate_artifact"].artifact_id, f["candidate"].candidate_id,
                PriorWorkSearchScope(maximum_queries=1, maximum_results=2, provider_name="static"),
                queries=("candidate related method",),
            ), f["state"], StaticSearchProvider([SearchResult("prior-paper", "Related study", snippet="Related limitation.")]),
            prior_llm, iteration_id=f["iteration"].iteration_id,
        )
        llm = QueueLLM(_plan_response())
        result = self._execute(f, llm, _request(f, prior_work_artifact_ids=(prior.artifact_id,)))
        self.assertEqual(result.experiment_plan.prior_work_artifact_ids, (prior.artifact_id,))
        self.assertTrue(result.experiment_plan.supporting_search_result_ids)
        self.assertEqual(prior_llm.calls, 1)
        self.assertEqual(len(llm.calls), 1)

        other = _fixture("biology")
        with self.assertRaises(TaskExecutionError):
            self._execute(other, QueueLLM(_plan_response()),
                          _request(other, prior_work_artifact_ids=(prior.artifact_id,)))
        self.assertEqual(other["state"].task_executions[-1].status, TaskExecutionStatus.FAILED)

    def test_artifact_insertion_failure_rolls_back_artifact_but_records_failed_execution(self):
        f = _fixture("mathematics")
        llm = QueueLLM(_plan_response())
        before = list(f["state"].research_artifacts)
        original = f["state"].add_research_artifact
        def fail_insert(artifact):
            original(artifact)
            raise RuntimeError("simulated state persistence failure")
        f["state"].add_research_artifact = fail_insert
        with self.assertRaises(TaskExecutionError):
            self._execute(f, llm)
        f["state"].add_research_artifact = original
        self.assertEqual(f["state"].research_artifacts, before)
        self.assertEqual(f["state"].task_executions[-1].status, TaskExecutionStatus.FAILED)
        self.assertIn("RuntimeError", f["state"].task_executions[-1].result.error or "")

    def test_request_rejects_duplicate_or_blank_references_and_unstored_candidate(self):
        f = _fixture("climate")
        with self.assertRaises(ExperimentPlanningError):
            _request(f, evidence_ids=(f["evidence"].evidence_id, f["evidence"].evidence_id))
        with self.assertRaises(ExperimentPlanningError):
            _request(f, prior_work_artifact_ids=("",))
        llm = QueueLLM(_plan_response())
        with self.assertRaises(TaskExecutionError):
            self._execute(f, llm, _request(f, candidate_id="invented-candidate"))
        self.assertEqual(llm.calls, [])

    def test_researcher_baseline_and_dataset_are_preserved_but_not_inferred(self):
        f = _fixture("medicine")
        request = _request(f, baseline="Researcher baseline B", selected_dataset="Researcher dataset D")
        response = _plan_response(baseline="Researcher baseline B", dataset_status="RESEARCHER_SPECIFIED",
                                  selected_dataset="Researcher dataset D")
        response_data = json.loads(response)
        response_data["unresolved_requirements"] = []
        result = self._execute(f, QueueLLM(json.dumps(response_data)), request)
        self.assertEqual(result.experiment_plan.baseline, "Researcher baseline B")
        self.assertEqual(result.experiment_plan.selected_dataset, "Researcher dataset D")
        self.assertEqual(result.experiment_plan.dataset_status, "RESEARCHER_SPECIFIED")

    def test_unsupported_novelty_and_fabricated_numeric_outcome_are_rejected_and_rolled_back(self):
        f = _fixture("cybersecurity")
        initial_artifact_ids = [a.artifact_id for a in f["state"].research_artifacts]
        with self.assertRaises(TaskExecutionError):
            self._execute(f, QueueLLM(_plan_response(objective="This is novel and proven.")))
        self.assertEqual([a.artifact_id for a in f["state"].research_artifacts], initial_artifact_ids)
        self.assertEqual(f["state"].task_executions[-1].status, TaskExecutionStatus.FAILED)
        with self.assertRaises(TaskExecutionError):
            self._execute(f, QueueLLM(_plan_response(expected_observations=["Accuracy may increase by 20%."])),
                          retry_of_execution_id=f["state"].task_executions[-1].execution_id)
        self.assertEqual([a.artifact_id for a in f["state"].research_artifacts], initial_artifact_ids)
        self.assertEqual(f["state"].task_executions[-1].status, TaskExecutionStatus.FAILED)

    def test_missing_evidence_and_bad_json_fail_without_retrieval_or_artifact(self):
        f = _fixture("biology")
        before = len(f["state"].research_artifacts)
        llm = QueueLLM("not-json")
        with self.assertRaises(TaskExecutionError):
            self._execute(f, llm, _request(f, evidence_ids=("unknown-evidence",)))
        self.assertEqual(llm.calls, [])
        failed_execution_id = f["state"].task_executions[-1].execution_id
        with self.assertRaises(TaskExecutionError):
            self._execute(f, llm, retry_of_execution_id=failed_execution_id)
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(len(f["state"].research_artifacts), before)
        self.assertEqual(f["state"].task_executions[-1].status, TaskExecutionStatus.FAILED)

    def test_research_state_old_payload_without_experiment_plan_still_loads(self):
        f = _fixture("vision")
        payload = f["state"].to_dict()
        for artifact in payload["research_artifacts"]:
            artifact.pop("experiment_plan", None)
        restored = ResearchState.from_dict(payload)
        self.assertEqual(restored.validate_lineage(), [])
        self.assertFalse(any(item.artifact_type == "experiment_plan" for item in restored.research_artifacts))


if __name__ == "__main__":
    unittest.main()
