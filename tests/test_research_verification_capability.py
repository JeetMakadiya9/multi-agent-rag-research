from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from claim_verification import CONTRADICTED, INSUFFICIENT_EVIDENCE, SUPPORTED
from research_capabilities import CapabilityType, LocalRetrievalRequest, execute_local_retrieval
from research_plan_review import PlanReview, ReviewStatus
from research_planning import ResearchObjective, ResearchPlan, ResearchPlanQuestion, ResearchRequest, ResearchTask
from research_retrieval import RetrievalEvidence, StaticRetrievalProvider, register_rag_document_version
from research_run_authorization import authorize_research_run
from research_state import ResearchArtifact, ResearchState, VerificationUnitAssessment
from research_task_authorization import authorize_research_tasks
from research_task_execution import TaskExecutionAuthorizationError, TaskExecutionError, TaskExecutionStatus
from research_verification_capability import (
    EvidenceVerificationRequest,
    VerificationCapabilityError,
    VerificationCapabilityResult,
    execute_evidence_verification,
)


def _stance(stance: str, evidence_index: str = "E1", component: str = "C1") -> str:
    return json.dumps({
        "assessments": [{
            "evidence_index": evidence_index,
            "component_indices": [component],
            "stance": stance,
        }],
        "reason": "Deterministic test assessment.",
    })


class QueueLLM:
    def __init__(self, *responses: str):
        self.responses = list(responses)
        self.calls = []

    def generate(self, messages, *, temperature=0.0, timeout=120):
        self.calls.append((messages, temperature, timeout))
        if not self.responses:
            raise AssertionError("Unexpected Experiment B provider call")
        return self.responses.pop(0)


class CountingRetrievalProvider(StaticRetrievalProvider):
    def __init__(self, evidence):
        super().__init__(evidence)
        self.calls = []

    def retrieve(self, query, limit=8):
        self.calls.append((query, limit))
        return super().retrieve(query, limit)


def _fixture(domain: str = "general", *, evidence_text: str = "System X uses BM25."):
    question = ResearchPlanQuestion("What does System X use?", question_id=f"plan-question-{domain}")
    retrieval_task = ResearchTask("Retrieve source material.", question.question_id, task_id=f"retrieve-{domain}")
    verify_task = ResearchTask("Verify a researcher-specified claim.", question.question_id, task_id=f"verify-{domain}")
    plan = ResearchPlan(
        request=ResearchRequest("Research a domain-neutral topic.", "Inspect sourced claims.", domain=domain),
        objective=ResearchObjective("Assess a stated claim against stored evidence."),
        questions=[question],
        tasks=[retrieval_task, verify_task],
        plan_id=f"plan-{domain}",
    )
    review = PlanReview(
        plan_id=plan.plan_id,
        plan_revision=plan.revision,
        status=ReviewStatus.APPROVED,
        reviewer="researcher",
    )
    state = ResearchState(user_request="Synthetic Phase 6B test")
    run = authorize_research_run(plan, review, state)
    authorize_research_tasks(plan, run, [retrieval_task.task_id, verify_task.task_id], review, state)
    iteration = state.create_iteration(run.run_id)
    version = register_rag_document_version(
        state,
        source_id=f"source-{domain}",
        document_id=f"document-{domain}",
        filename="synthetic-document.pdf",
        version_id=f"version-{domain}",
    )
    retrieved = RetrievalEvidence(
        evidence_id=f"chunk-{domain}",
        text=evidence_text,
        source_id=f"source-{domain}",
        filename="synthetic-document.pdf",
        page=2,
        section="Results",
        document_id=f"document-{domain}",
        chunk_id=f"chunk-{domain}",
        retrieval_metadata={"test": "synthetic"},
    )
    provider = CountingRetrievalProvider([retrieved])
    retrieval_result = execute_local_retrieval(
        plan,
        run,
        retrieval_task,
        review,
        LocalRetrievalRequest("System X uses BM25", limit=1, evidence_document_versions={1: version}),
        provider,
        state,
        iteration_id=iteration.iteration_id,
    )
    evidence = state.evidence[0]
    return {
        "plan": plan,
        "review": review,
        "run": run,
        "retrieval_task": retrieval_task,
        "task": verify_task,
        "state": state,
        "iteration": iteration,
        "evidence": evidence,
        "provider": provider,
        "retrieval_result": retrieval_result,
    }


def _execute(fixture, provider, evidence_ids=None, **kwargs):
    return execute_evidence_verification(
        fixture["plan"],
        fixture["run"],
        fixture["task"],
        fixture["review"],
        EvidenceVerificationRequest(
            "System X uses BM25.",
            tuple(evidence_ids or (fixture["evidence"].evidence_id,)),
        ),
        fixture["state"],
        provider,
        iteration_id=fixture["iteration"].iteration_id,
        **kwargs,
    )


class ResearchVerificationCapabilityTests(unittest.TestCase):
    def test_explicit_capability_and_successful_authorized_verification(self):
        fixture = _fixture("NLP")
        task_status_before = fixture["task"].status
        llm = QueueLLM(_stance("SUPPORTS"))
        result = _execute(fixture, llm)
        self.assertEqual(CapabilityType.EVIDENCE_VERIFICATION.value, "evidence_verification")
        self.assertEqual(result.capability, CapabilityType.EVIDENCE_VERIFICATION)
        self.assertEqual(result.verdict, SUPPORTED)
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(fixture["task"].status, task_status_before)
        self.assertIn(fixture["evidence"].text, llm.calls[0][0][1]["content"])
        self.assertTrue(result.success)
        self.assertEqual(fixture["state"].task_executions[-1].status, TaskExecutionStatus.COMPLETED)

    def test_contradicted_and_insufficient_are_successful_verification_outcomes(self):
        for stance, verdict in (("CONTRADICTS", CONTRADICTED), ("NEUTRAL", INSUFFICIENT_EVIDENCE)):
            with self.subTest(verdict=verdict):
                fixture = _fixture(f"domain-{verdict}")
                result = _execute(fixture, QueueLLM(_stance(stance)))
                self.assertEqual(result.verdict, verdict)
                self.assertEqual(fixture["state"].task_executions[-1].status, TaskExecutionStatus.COMPLETED)

    def test_partial_component_coverage_remains_insufficient(self):
        fixture = _fixture("vision", evidence_text="System X uses BM25.")
        llm = QueueLLM(
            _stance("SUPPORTS", component="C1"),
            _stance("NEUTRAL", component="C2"),
        )
        result = execute_evidence_verification(
            fixture["plan"], fixture["run"], fixture["task"], fixture["review"],
            EvidenceVerificationRequest("System X uses BM25 and FAISS.", (fixture["evidence"].evidence_id,)),
            fixture["state"], llm,
        )
        self.assertEqual(result.verdict, INSUFFICIENT_EVIDENCE)

    def test_conflicting_stances_remain_insufficient(self):
        fixture = _fixture("cybersecurity")
        second = fixture["state"].evidence[0]
        # Reuse the existing registered source/version/passage lineage while
        # keeping Evidence identity distinct for this synthetic conflict case.
        from research_state import Evidence
        from research_retrieval import adapt_retrieval_evidence_to_state
        item = RetrievalEvidence(
            evidence_id="chunk-conflict", text="System X does not use BM25.",
            source_id=second.source_id, filename="synthetic-document.pdf", page=3,
            document_id=f"document-cybersecurity", chunk_id="chunk-conflict",
        )
        # Add a distinct stored SearchResult and adapt the chunk through the
        # existing provenance adapter, without invoking retrieval.
        from research_tools import SearchResult
        from research_state import SearchAction
        search = fixture["state"].add_search(SearchAction(
            query="conflicting synthetic retrieval",
            provider="test",
            result_count=1,
            run_id=fixture["run"].run_id,
        ))
        search_result = fixture["state"].add_search_result(SearchResult(
            source_id=second.source_id, title="", content=item.text, source_type="unknown",
            search_id=search.search_id, rank=1,
        ))
        version = fixture["state"].document_versions[0]
        adapt_retrieval_evidence_to_state(item, version, fixture["state"], search_result_id=search_result.result_id)
        llm = QueueLLM(_stance("SUPPORTS", "E1"), _stance("CONTRADICTS", "E2"))
        result = _execute(fixture, llm, [second.evidence_id, fixture["state"].evidence[-1].evidence_id])
        self.assertEqual(result.verdict, INSUFFICIENT_EVIDENCE)
        self.assertEqual(len(llm.calls), 2)

    def test_invalid_authorization_never_invokes_experiment_b(self):
        fixture = _fixture("medicine")
        unauthorized = ResearchTask("Not authorized.", fixture["task"].question_id, task_id="unauthorized")
        with patch("research_verification_capability.run_experiment_b_verification") as verifier:
            with self.assertRaises(TaskExecutionAuthorizationError):
                execute_evidence_verification(
                    fixture["plan"], fixture["run"], unauthorized, fixture["review"],
                    EvidenceVerificationRequest("System X uses BM25.", (fixture["evidence"].evidence_id,)),
                    fixture["state"], QueueLLM(),
                )
        verifier.assert_not_called()

    def test_invalid_revision_review_and_unregistered_run_never_invoke_experiment_b(self):
        for variant in ("revision", "review", "unregistered"):
            with self.subTest(variant=variant):
                fixture = _fixture(f"authorization-{variant}")
                if variant == "revision":
                    fixture["run"].plan_revision += 1
                elif variant == "review":
                    from research_plan_review import PlanReview, ReviewStatus
                    replacement = PlanReview(
                        plan_id=fixture["plan"].plan_id,
                        plan_revision=fixture["plan"].revision,
                        status=ReviewStatus.APPROVED,
                        reviewer="another reviewer",
                    )
                    fixture["review"] = replacement
                else:
                    fixture["state"].research_runs.clear()
                with patch("research_verification_capability.run_experiment_b_verification") as verifier:
                    with self.assertRaises(TaskExecutionAuthorizationError):
                        _execute(fixture, QueueLLM(_stance("SUPPORTS")))
                verifier.assert_not_called()

    def test_unknown_evidence_fails_execution_without_artifact(self):
        fixture = _fixture("agriculture")
        prior_artifacts = list(fixture["state"].research_artifacts)
        with patch("research_verification_capability.run_experiment_b_verification") as verifier:
            with self.assertRaises(TaskExecutionError) as raised:
                execute_evidence_verification(
                    fixture["plan"], fixture["run"], fixture["task"], fixture["review"],
                    EvidenceVerificationRequest("System X uses BM25.", ("made-up-evidence",)),
                    fixture["state"], QueueLLM(),
                )
        verifier.assert_not_called()
        self.assertEqual(raised.exception.execution.status, TaskExecutionStatus.FAILED)
        self.assertEqual(fixture["state"].research_artifacts, prior_artifacts)

    def test_request_rejects_empty_claim_evidence_empty_ids_and_duplicates(self):
        for claim, ids in ((" ", ("ev",)), ("A claim.", ()), ("A claim.", (" ",)), ("A claim.", ("ev", "ev"))):
            with self.subTest(claim=claim, ids=ids), self.assertRaises(VerificationCapabilityError):
                EvidenceVerificationRequest(claim, ids)
        with self.assertRaises(TypeError):
            EvidenceVerificationRequest("Claim.", ("ev",), raw_evidence_text="made up")

    def test_invalid_provenance_is_rejected_before_experiment_b(self):
        fixture = _fixture("physics")
        prior_artifacts = list(fixture["state"].research_artifacts)
        fixture["state"].evidence[0].passage_reference_id = "missing-passage"
        with patch("research_verification_capability.run_experiment_b_verification") as verifier:
            with self.assertRaises(TaskExecutionError):
                _execute(fixture, QueueLLM())
        verifier.assert_not_called()
        self.assertEqual(fixture["state"].research_artifacts, prior_artifacts)
        self.assertEqual(fixture["state"].task_executions[-1].status, TaskExecutionStatus.FAILED)

    def test_malformed_experiment_b_response_fails_and_can_be_explicitly_retried(self):
        fixture = _fixture("biology")
        prior_artifacts = list(fixture["state"].research_artifacts)
        with self.assertRaises(TaskExecutionError) as raised:
            _execute(fixture, QueueLLM("not-json"))
        failed = raised.exception.execution
        self.assertEqual(failed.status, TaskExecutionStatus.FAILED)
        self.assertEqual(fixture["state"].research_artifacts, prior_artifacts)
        result = _execute(fixture, QueueLLM(_stance("SUPPORTS")), retry_of_execution_id=failed.execution_id)
        self.assertNotEqual(result.execution_id, failed.execution_id)
        failed_record = next(item for item in fixture["state"].task_executions if item.execution_id == failed.execution_id)
        self.assertEqual(failed_record.status, TaskExecutionStatus.FAILED)
        self.assertEqual(fixture["state"].task_executions[-1].retry_of_execution_id, failed.execution_id)

    def test_artifact_preserves_existing_evidence_and_provenance_without_creating_evidence(self):
        fixture = _fixture("climate")
        evidence_before = list(fixture["state"].evidence)
        result = _execute(fixture, QueueLLM(_stance("SUPPORTS")))
        artifact = fixture["state"].research_artifacts[-1]
        self.assertEqual(artifact.artifact_type, "verification_result")
        self.assertEqual(artifact.capability, "evidence_verification")
        self.assertIsNone(artifact.search_action_id)
        self.assertEqual(artifact.evidence_ids, (evidence_before[0].evidence_id,))
        self.assertEqual(fixture["state"].evidence, evidence_before)
        self.assertEqual(artifact.verification_assessments[0].evidence_assessments[0].evidence_id,
                         evidence_before[0].evidence_id)
        self.assertEqual(artifact.verification_verdict, result.verdict)
        self.assertTrue(artifact.execution_id in {item.execution_id for item in fixture["state"].task_executions})
        self.assertEqual(fixture["state"].validate_lineage(), [])

    def test_artifact_and_capability_result_round_trip(self):
        fixture = _fixture("chemistry")
        result = _execute(fixture, QueueLLM(_stance("SUPPORTS")))
        self.assertEqual(VerificationCapabilityResult.from_dict(json.loads(json.dumps(result.to_dict()))), result)
        restored = ResearchState.from_dict(json.loads(json.dumps(fixture["state"].to_dict())))
        artifact = restored.research_artifacts[-1]
        self.assertEqual(artifact.verification_claim, result.claim)
        self.assertEqual(artifact.verification_verdict, SUPPORTED)
        self.assertEqual(artifact.evidence_ids, result.evidence_ids)
        self.assertEqual(restored.validate_lineage(), [])

    def test_frozen_experiment_b_receives_no_retrieval_provider(self):
        from experiment_b_verification import run_experiment_b_verification
        fixture = _fixture("energy")
        with patch(
            "research_verification_capability.run_experiment_b_verification",
            wraps=run_experiment_b_verification,
        ) as verifier:
            _execute(fixture, QueueLLM(_stance("SUPPORTS")))
        self.assertEqual(verifier.call_count, 1)
        self.assertIsNone(verifier.call_args.kwargs["retrieval_provider"])
        self.assertFalse(verifier.call_args.kwargs["config"].answer_revision_enabled)

    def test_artifact_recording_failure_marks_execution_failed_and_rolls_back(self):
        fixture = _fixture("automotive")
        prior_artifacts = list(fixture["state"].research_artifacts)
        with patch.object(fixture["state"], "add_research_artifact", side_effect=RuntimeError("write failed")):
            with self.assertRaises(TaskExecutionError) as raised:
                _execute(fixture, QueueLLM(_stance("SUPPORTS")))
        self.assertEqual(raised.exception.execution.status, TaskExecutionStatus.FAILED)
        self.assertEqual(fixture["state"].research_artifacts, prior_artifacts)

    def test_malformed_artifact_or_dangling_evidence_is_rejected(self):
        fixture = _fixture("engineering")
        _execute(fixture, QueueLLM(_stance("SUPPORTS")))
        payload = json.loads(json.dumps(fixture["state"].to_dict()))
        payload["research_artifacts"][-1]["evidence_ids"] = ["not-real"]
        with self.assertRaisesRegex(ValueError, "Invalid serialized lineage"):
            ResearchState.from_dict(payload)
        malformed = ResearchArtifact(
            execution_id="x", run_id="r", task_id="t", search_action_id=None,
            evidence_ids=("ev",), artifact_type="verification_result", capability="evidence_verification",
            verification_claim="Claim.", verification_verdict=SUPPORTED,
            verification_assessments=(VerificationUnitAssessment("u", "Claim.", SUPPORTED, 1.0, "ok"),),
        )
        with self.assertRaisesRegex(ValueError, "exactly one typed claim assessment"):
            ResearchArtifact.from_dict({**malformed.to_dict(), "verification_assessments": []})

    def test_verification_is_explicit_and_local_retrieval_does_not_auto_verify(self):
        fixture = _fixture("economics")
        self.assertEqual(len(fixture["state"].research_artifacts), 1)
        self.assertEqual(fixture["state"].research_artifacts[0].artifact_type, "retrieval_result")
        self.assertEqual(len(fixture["provider"].calls), 1)
        self.assertEqual(len(fixture["state"].task_executions), 1)

    def test_domain_neutral_requests_use_the_same_contract(self):
        for domain in ("NLP", "computer vision", "cybersecurity", "medicine", "agriculture", "physics"):
            with self.subTest(domain=domain):
                fixture = _fixture(domain)
                result = _execute(fixture, QueueLLM(_stance("NEUTRAL")))
                self.assertEqual(result.verdict, INSUFFICIENT_EVIDENCE)
                self.assertEqual(result.capability, CapabilityType.EVIDENCE_VERIFICATION)

    def test_legacy_state_without_artifacts_still_loads(self):
        state = ResearchState(user_request="Old payload")
        payload = state.to_dict()
        payload.pop("research_artifacts")
        restored = ResearchState.from_dict(payload)
        self.assertEqual(restored.research_artifacts, [])

    def test_legacy_retrieval_artifact_payload_without_verification_fields_loads(self):
        fixture = _fixture("legacy")
        payload = json.loads(json.dumps(fixture["state"].to_dict()))
        artifact = payload["research_artifacts"][0]
        artifact.pop("verification_claim")
        artifact.pop("verification_verdict")
        artifact.pop("verification_assessments")
        restored = ResearchState.from_dict(payload)
        self.assertEqual(restored.research_artifacts[0].artifact_type, "retrieval_result")
        self.assertEqual(restored.validate_lineage(), [])


if __name__ == "__main__":
    unittest.main()
