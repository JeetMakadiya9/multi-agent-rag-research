from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research_capabilities import CapabilityType
from research_improvement_capability import (
    ImprovementCapabilityError,
    ImprovementGenerationRequest,
    ImprovementGenerationResult,
    execute_improvement_generation,
)
from research_plan_review import PlanReview, ReviewStatus
from research_planning import ResearchObjective, ResearchPlan, ResearchPlanQuestion, ResearchRequest, ResearchTask
from research_retrieval import RetrievalEvidence, StaticRetrievalProvider, register_rag_document_version
from research_run_authorization import authorize_research_run
from research_state import ResearchClaim, ResearchState
from research_task_authorization import authorize_research_tasks
from research_task_execution import (
    TaskExecutionAuthorizationError,
    TaskExecutionError,
    TaskExecutionStatus,
)


class CountingLLM:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def generate(self, messages, *, temperature=0.0, timeout=120):
        self.calls.append((messages, temperature, timeout))
        if self.error:
            raise self.error
        return self.response


def _candidate(evidence_id="ev-med", claim_id="claim-supported", verification_id=None, *, title="Candidate A"):
    return {
        "title": title,
        "motivation": "The documented limitation motivates investigation.",
        "proposed_change": "Investigate an adaptive approach to address the limitation.",
        "rationale": "The proposal may address the stated mechanism.",
        "expected_benefit": "Could reduce sensitivity to the identified condition.",
        "assumptions": ["The limitation is relevant to the target setting."],
        "risks": ["The added method may increase computational cost."],
        "validation_needed": ["Effectiveness requires empirical validation."],
        "supporting_evidence_ids": [evidence_id],
        "supporting_claim_ids": [claim_id] if claim_id else [],
        "supporting_verification_artifact_ids": [verification_id] if verification_id else [],
    }


def _response(*candidates):
    return json.dumps({"candidates": list(candidates)})


def _fixture(domain="medicine"):
    question = ResearchPlanQuestion("What limitation has been documented?", question_id=f"question-{domain}")
    retrieval_task = ResearchTask("Retrieve existing source material.", question.question_id, task_id=f"retrieve-{domain}")
    verification_task = ResearchTask("Verify a researcher-specified claim.", question.question_id, task_id=f"verify-{domain}")
    improve_task = ResearchTask("Generate candidate improvements.", question.question_id, task_id=f"improve-{domain}")
    unauthorized_task = ResearchTask("Not authorized.", question.question_id, task_id=f"unauthorized-{domain}")
    plan = ResearchPlan(
        request=ResearchRequest("Investigate a documented limitation.", "Explore possible approaches.", domain=domain),
        objective=ResearchObjective("Generate candidate improvements from existing verified input."),
        questions=[question],
        tasks=[retrieval_task, verification_task, improve_task, unauthorized_task],
        plan_id=f"plan-{domain}",
    )
    review = PlanReview(
        plan_id=plan.plan_id,
        plan_revision=plan.revision,
        status=ReviewStatus.APPROVED,
        reviewer="researcher",
    )
    state = ResearchState(user_request="Synthetic 6C fixture")
    run = authorize_research_run(plan, review, state)
    authorize_research_tasks(plan, run, [retrieval_task.task_id, verification_task.task_id, improve_task.task_id], review, state)
    iteration = state.create_iteration(run.run_id)
    version = register_rag_document_version(
        state, source_id=f"source-{domain}", document_id=f"doc-{domain}",
        filename="synthetic-source.pdf", version_id=f"version-{domain}",
    )
    retrieval = RetrievalEvidence(
        evidence_id=f"chunk-{domain}",
        text="The prior method requires a large labeled dataset.",
        source_id=f"source-{domain}", filename="synthetic-source.pdf", page=4,
        section="Limitations", document_id=f"doc-{domain}", chunk_id=f"chunk-{domain}",
    )
    local = StaticRetrievalProvider([retrieval])
    from research_capabilities import LocalRetrievalRequest, execute_local_retrieval
    execute_local_retrieval(
        plan, run, retrieval_task, review,
        LocalRetrievalRequest("method requires labeled dataset", limit=1, evidence_document_versions={1: version}),
        local, state, iteration_id=iteration.iteration_id,
    )
    evidence = state.evidence[0]
    claim = state.add_claim(ResearchClaim(
        text="The method requires labeled data.", status="SUPPORTED",
        claim_id=f"claim-{domain}", evidence_ids=[evidence.evidence_id],
        source_ids=[evidence.source_id], verification_reason="Synthetic supported result.",
    ))
    return {
        "plan": plan, "review": review, "run": run, "task": improve_task,
        "verification_task": verification_task,
        "unauthorized_task": unauthorized_task, "state": state,
        "iteration": iteration, "evidence": evidence, "claim": claim,
        "retrieval_task": retrieval_task,
    }


def _request(fixture, **overrides):
    values = {
        "problem_statement": "Researcher-provided limitation: method requires a large labeled dataset.",
        "evidence_ids": (fixture["evidence"].evidence_id,),
        "verified_claim_ids": (fixture["claim"].claim_id,),
        "domain": fixture["plan"].request.domain,
    }
    values.update(overrides)
    return ImprovementGenerationRequest(**values)


def _execute(fixture, llm, request=None, **kwargs):
    return execute_improvement_generation(
        fixture["plan"], fixture["run"], fixture["task"], fixture["review"],
        request or _request(fixture), fixture["state"], llm,
        iteration_id=fixture["iteration"].iteration_id, **kwargs,
    )


class ResearchImprovementCapabilityTests(unittest.TestCase):
    def test_authorized_generation_persists_candidate_artifact_not_evidence(self):
        fixture = _fixture()
        llm = CountingLLM(_response(_candidate(evidence_id=fixture["evidence"].evidence_id, claim_id=fixture["claim"].claim_id)))
        with patch("research_capabilities.execute_local_retrieval", side_effect=AssertionError("must not retrieve")) as retrieve, \
             patch("research_verification_capability.run_experiment_b_verification", side_effect=AssertionError("must not verify")) as verify:
            result = _execute(fixture, llm)
        retrieve.assert_not_called()
        verify.assert_not_called()
        self.assertEqual(CapabilityType.IMPROVEMENT_GENERATION.value, "improvement_generation")
        self.assertEqual(result.capability, CapabilityType.IMPROVEMENT_GENERATION)
        self.assertEqual(result.candidates[0].status, "CANDIDATE")
        self.assertEqual(result.candidates[0].novelty_status, "NOT_ASSESSED")
        self.assertEqual(result.candidates[0].validation_status, "NOT_VALIDATED")
        self.assertEqual(result.candidates[0].problem_origin, "researcher_provided")
        self.assertEqual(result.candidates[0].generation_origin, "system_generated")
        self.assertEqual(result.candidates[0].supporting_evidence_ids, (fixture["evidence"].evidence_id,))
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(len(fixture["state"].evidence), 1)
        artifact = fixture["state"].research_artifacts[-1]
        self.assertEqual(artifact.artifact_type, "improvement_candidate")
        self.assertEqual(artifact.capability, "improvement_generation")
        self.assertEqual(artifact.execution_id, result.execution_id)
        self.assertEqual(artifact.evidence_ids, (fixture["evidence"].evidence_id,))
        self.assertEqual(fixture["state"].task_executions[-1].status, TaskExecutionStatus.COMPLETED)
        self.assertEqual(fixture["task"].status.value, "pending")
        self.assertEqual(fixture["state"].validate_lineage(), [])

    def test_domain_neutral_generation_and_multiple_candidate_ids(self):
        for domain in ("NLP", "computer vision", "cybersecurity", "agriculture"):
            with self.subTest(domain=domain):
                fixture = _fixture(domain)
                evidence_id, claim_id = fixture["evidence"].evidence_id, fixture["claim"].claim_id
                llm = CountingLLM(_response(
                    _candidate(evidence_id, claim_id, title="Candidate one"),
                    _candidate(evidence_id, claim_id, title="Candidate two"),
                ))
                result = _execute(fixture, llm, _request(fixture, number_of_candidates=2))
                self.assertEqual(len(result.candidates), 2)
                self.assertNotEqual(result.candidates[0].candidate_id, result.candidates[1].candidate_id)
                self.assertEqual(result.candidates[0].supporting_evidence_ids, result.candidates[1].supporting_evidence_ids)

    def test_no_candidate_is_a_valid_explicit_completed_output(self):
        fixture = _fixture("physics")
        result = _execute(fixture, CountingLLM(_response()))
        self.assertEqual(result.candidates, ())
        self.assertIn("No candidate improvements", fixture["state"].task_executions[-1].result.output)
        self.assertEqual(fixture["state"].validate_lineage(), [])

    def test_authorization_failure_does_not_call_provider_or_add_artifact(self):
        fixture = _fixture("unauthorized")
        llm = CountingLLM(_response())
        before = len(fixture["state"].research_artifacts)
        with self.assertRaises(TaskExecutionAuthorizationError):
            execute_improvement_generation(
                fixture["plan"], fixture["run"], fixture["unauthorized_task"], fixture["review"],
                _request(fixture), fixture["state"], llm,
            )
        self.assertEqual(llm.calls, [])
        self.assertEqual(len(fixture["state"].research_artifacts), before)

    def test_bad_evidence_ids_and_request_values_rejected_before_provider(self):
        fixture = _fixture("bad-request")
        for values in (
            {"evidence_ids": ("missing",)},
            {"evidence_ids": (fixture["evidence"].evidence_id, fixture["evidence"].evidence_id)},
        ):
            with self.subTest(values=values):
                llm = CountingLLM(_response())
                with self.assertRaises((ImprovementCapabilityError, TaskExecutionError)):
                    _execute(fixture, llm, _request(fixture, **values))
                self.assertEqual(llm.calls, [])
                # First invalid handler attempt is recorded FAILED; use a fresh fixture for any further attempt.
                fixture = _fixture("bad-request-next")
        with self.assertRaises(ImprovementCapabilityError):
            ImprovementGenerationRequest(" ", ("e1",), verified_claim_ids=("c1",))
        with self.assertRaises(ImprovementCapabilityError):
            ImprovementGenerationRequest("problem", ("e1",), verified_claim_ids=("c1",), number_of_candidates=0)
        with self.assertRaises(ImprovementCapabilityError):
            ImprovementGenerationRequest("problem", ("e1",), number_of_candidates=1)

    def test_unverified_claim_and_unresolvable_raw_text_are_rejected(self):
        fixture = _fixture("unverified")
        fixture["claim"].status = "unverified"
        llm = CountingLLM(_response())
        with self.assertRaises(TaskExecutionError):
            _execute(fixture, llm)
        self.assertEqual(llm.calls, [])
        self.assertFalse(any(a.artifact_type == "improvement_candidate" for a in fixture["state"].research_artifacts))
        self.assertEqual(fixture["state"].task_executions[-1].status, TaskExecutionStatus.FAILED)

    def test_schema_and_unsupported_epistemic_claims_fail_without_artifact(self):
        phrases = ("This is novel.", "This is the first method.", "No previous work exists.", "This improvement is proven.")
        for phrase in phrases:
            with self.subTest(phrase=phrase):
                fixture = _fixture(f"unsupported-{phrases.index(phrase)}")
                value = _candidate(fixture["evidence"].evidence_id, fixture["claim"].claim_id)
                value["rationale"] = phrase
                llm = CountingLLM(_response(value))
                with self.assertRaises(TaskExecutionError):
                    _execute(fixture, llm)
                self.assertFalse(any(a.artifact_type == "improvement_candidate" for a in fixture["state"].research_artifacts))
                self.assertEqual(fixture["state"].task_executions[-1].status, TaskExecutionStatus.FAILED)
        fixture = _fixture("malformed")
        with self.assertRaises(TaskExecutionError):
            _execute(fixture, CountingLLM('{"candidates": ['))
        self.assertFalse(any(a.artifact_type == "improvement_candidate" for a in fixture["state"].research_artifacts))

    def test_llm_failure_and_artifact_write_failure_are_recorded_and_rolled_back(self):
        fixture = _fixture("provider-failure")
        with self.assertRaises(TaskExecutionError):
            _execute(fixture, CountingLLM(error=RuntimeError("synthetic failure")))
        self.assertEqual(fixture["state"].task_executions[-1].status, TaskExecutionStatus.FAILED)
        self.assertFalse(any(a.artifact_type == "improvement_candidate" for a in fixture["state"].research_artifacts))

        fixture = _fixture("artifact-failure")
        response = _response(_candidate(fixture["evidence"].evidence_id, fixture["claim"].claim_id))
        with patch.object(fixture["state"], "add_research_artifact", side_effect=RuntimeError("write failure")):
            with self.assertRaises(TaskExecutionError):
                _execute(fixture, CountingLLM(response))
        self.assertFalse(any(a.artifact_type == "improvement_candidate" for a in fixture["state"].research_artifacts))
        self.assertEqual(fixture["state"].task_executions[-1].status, TaskExecutionStatus.FAILED)

    def test_retry_creates_new_execution_and_old_failure_remains(self):
        fixture = _fixture("retry")
        with self.assertRaises(TaskExecutionError):
            _execute(fixture, CountingLLM("not json"))
        failed = fixture["state"].task_executions[-1]
        self.assertEqual(failed.status, TaskExecutionStatus.FAILED)
        result = _execute(
            fixture,
            CountingLLM(_response(_candidate(fixture["evidence"].evidence_id, fixture["claim"].claim_id))),
            retry_of_execution_id=failed.execution_id,
        )
        self.assertEqual(len(fixture["state"].task_executions), 3)  # retrieval execution plus two generation attempts
        self.assertEqual(failed.status, TaskExecutionStatus.FAILED)
        self.assertEqual(fixture["state"].task_executions[-1].retry_of_execution_id, failed.execution_id)
        self.assertEqual(result.execution_id, fixture["state"].task_executions[-1].execution_id)

    def test_candidate_and_result_serialization_and_legacy_state(self):
        fixture = _fixture("roundtrip")
        result = _execute(
            fixture, CountingLLM(_response(_candidate(fixture["evidence"].evidence_id, fixture["claim"].claim_id)))
        )
        self.assertEqual(ImprovementGenerationResult.from_dict(result.to_dict()), result)
        restored = ResearchState.from_dict(fixture["state"].to_dict())
        self.assertEqual(restored.validate_lineage(), [])
        candidate = restored.research_artifacts[-1].improvement_candidates[0]
        self.assertEqual(candidate, result.candidates[0])
        self.assertEqual(candidate.supporting_evidence_ids, (fixture["evidence"].evidence_id,))
        legacy = ResearchState.from_dict({"user_request": "legacy"})
        self.assertEqual(legacy.research_artifacts, [])
        self.assertEqual(legacy.evidence, [])

    def test_supported_verification_artifact_can_be_a_grounding_input(self):
        fixture = _fixture("verification-input")
        from research_verification_capability import EvidenceVerificationRequest, execute_evidence_verification

        class VerificationStub:
            def generate(self, messages, *, temperature=0.0, timeout=120):
                return json.dumps({
                    "assessments": [{"evidence_index": "E1", "component_indices": ["C1"], "stance": "SUPPORTS"}],
                    "reason": "The stored passage supports the claim.",
                })

        verification = execute_evidence_verification(
            fixture["plan"], fixture["run"], fixture["verification_task"], fixture["review"],
            EvidenceVerificationRequest("The prior method requires labeled data.", (fixture["evidence"].evidence_id,)),
            fixture["state"], VerificationStub(), iteration_id=fixture["iteration"].iteration_id,
        )
        self.assertEqual(verification.verdict, "SUPPORTED")
        candidate = _candidate(fixture["evidence"].evidence_id, claim_id=None, verification_id=verification.artifact_id)
        llm = CountingLLM(_response(candidate))
        result = _execute(
            fixture, llm,
            _request(fixture, verified_claim_ids=(), verification_artifact_ids=(verification.artifact_id,)),
        )
        self.assertEqual(result.candidates[0].supporting_verification_artifact_ids, (verification.artifact_id,))


if __name__ == "__main__":
    unittest.main()
