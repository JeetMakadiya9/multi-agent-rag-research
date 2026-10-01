from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research_capabilities import (
    CapabilityRequestError,
    CapabilityStatus,
    CapabilityType,
    LocalRetrievalRequest,
    execute_local_retrieval,
)
from research_plan_review import PlanReview, ReviewStatus
from research_planning import (
    ResearchObjective,
    ResearchPlan,
    ResearchPlanQuestion,
    ResearchRequest,
    ResearchTask,
)
from research_retrieval import (
    ExistingRAGProvider,
    RetrievalEvidence,
    StaticRetrievalProvider,
    register_rag_document_version,
)
from research_run_authorization import authorize_research_run
from research_state import ResearchArtifact, ResearchState
from research_task_authorization import authorize_research_tasks
from research_task_execution import TaskExecutionError, TaskExecutionStatus
from research_task_execution import TaskExecutionAuthorizationError


def _plan(domain: str = "general"):
    question = ResearchPlanQuestion("What is reported?", question_id="plan-question")
    task = ResearchTask("Retrieve relevant local documents.", question.question_id, task_id="task-retrieve")
    plan = ResearchPlan(
        request=ResearchRequest("Research a topic.", "Understand reported work.", domain=domain),
        objective=ResearchObjective("Investigate existing material."),
        questions=[question],
        tasks=[task],
        plan_id=f"plan-{domain}",
    )
    review = PlanReview(
        plan_id=plan.plan_id,
        plan_revision=plan.revision,
        status=ReviewStatus.APPROVED,
        reviewer="researcher",
    )
    state = ResearchState(user_request="Phase 6 synthetic capability test")
    run = authorize_research_run(plan, review, state)
    authorize_research_tasks(plan, run, [task.task_id], review, state)
    iteration = state.create_iteration(run.run_id)
    question_state = state.add_question(
        "What local sources describe this topic?",
        run_id=run.run_id,
        iteration_id=iteration.iteration_id,
    )
    return plan, review, run, task, state, iteration, question_state


def _items():
    return [
        RetrievalEvidence(
            evidence_id="chunk-1",
            text="A synthetic retrieved passage about method limitations.",
            source_id="source-one",
            filename="paper-one.pdf",
            page=2,
            section="Limitations",
            document_id="doc-one",
            chunk_id="chunk-1",
            retrieval_metadata={"fixture": "synthetic"},
        ),
        RetrievalEvidence(
            evidence_id="chunk-2",
            text="Another synthetic passage about evaluation methods.",
            source_id="source-two",
            filename="paper-two.pdf",
            page=1,
            section="Evaluation",
            document_id="doc-two",
            chunk_id="chunk-2",
        ),
    ]


class CountingProvider(StaticRetrievalProvider):
    def __init__(self, evidence):
        super().__init__(evidence)
        self.calls = []

    def retrieve(self, query: str, limit: int = 8):
        self.calls.append((query, limit))
        return super().retrieve(query, limit)


class ResearchCapabilitiesTests(unittest.TestCase):
    def _execute(self, *, versions=None, provider=None, request=None, domain="general", **kwargs):
        plan, review, run, task, state, iteration, question = _plan(domain)
        if versions is None:
            versions = {}
        provider = provider or CountingProvider(_items())
        request = request or LocalRetrievalRequest(
            "synthetic passage method limitations",
            limit=2,
            question_id=question.question_id,
            evidence_document_versions=versions,
        )
        result = execute_local_retrieval(
            plan,
            run,
            task,
            review,
            request,
            provider,
            state,
            iteration_id=iteration.iteration_id,
            **kwargs,
        )
        return result, (plan, review, run, task, state, iteration, question), provider

    def test_success_records_authorized_execution_and_typed_retrieval_artifact(self):
        result, (_, _, run, task, state, iteration, _), provider = self._execute()
        self.assertEqual(provider.calls, [("synthetic passage method limitations", 2)])
        self.assertEqual(result.capability, CapabilityType.LOCAL_RETRIEVAL)
        self.assertEqual(result.status, CapabilityStatus.SUCCESS)
        self.assertEqual(len(state.task_executions), 1)
        execution = state.task_executions[0]
        self.assertEqual(execution.status, TaskExecutionStatus.COMPLETED)
        artifact = state.research_artifacts[0]
        self.assertEqual(artifact.artifact_type, "retrieval_result")
        self.assertEqual(artifact.capability, "local_retrieval")
        self.assertEqual((artifact.execution_id, artifact.run_id, artifact.task_id),
                         (execution.execution_id, run.run_id, task.task_id))
        self.assertEqual(result.artifact_id, artifact.artifact_id)
        self.assertEqual(len(artifact.search_result_ids), 2)
        action = state.searches[0]
        self.assertEqual(action.query, "synthetic passage method limitations")
        self.assertEqual(action.provider, "static_retrieval")
        self.assertEqual(action.purpose, "retrieval")
        self.assertEqual(action.result_count, 2)
        self.assertEqual(action.question_id, state.research_questions[0].question_id)
        self.assertEqual([item.rank for item in state.search_results], [1, 2])
        self.assertEqual(state.search_results[0].metadata["retrieval_metadata"]["fixture"], "synthetic")
        self.assertEqual(state.searches[0].iteration_id, iteration.iteration_id)
        self.assertEqual(execution.result.artifact_ids, (artifact.artifact_id,))
        self.assertEqual(state.validate_lineage(), [])

    def test_local_capability_runs_existing_rag_retrieval_code_on_synthetic_chunks(self):
        import rag

        plan, review, run, task, state, iteration, _ = _plan()
        chunks = [{
            "chunk_id": "synthetic-chunk",
            "document_id": "synthetic-document",
            "filename": "synthetic.pdf",
            "page": 1,
            "section": "Results",
            "text": "Synthetic local retrieval passage about reported results.",
        }]
        reranker = type(
            "DeterministicReranker",
            (),
            {"predict": lambda self, pairs, show_progress_bar=False: [1.0 for _ in pairs]},
        )()
        provider = ExistingRAGProvider(
            chunks=chunks,
            embedding_model=None,
            reranker=reranker,
            faiss_index=None,
            bm25_index=None,
            retrieval_module=rag,
        )
        version = register_rag_document_version(
            state,
            source_id="synthetic.pdf:page-1",
            document_id="synthetic-document",
            filename="synthetic.pdf",
            version_id="synthetic-document-v1",
        )
        result = execute_local_retrieval(
            plan,
            run,
            task,
            review,
            LocalRetrievalRequest(
                "synthetic local retrieval passage",
                limit=1,
                evidence_document_versions={1: version},
            ),
            provider,
            state,
            iteration_id=iteration.iteration_id,
        )
        self.assertEqual(result.result_count, 1)
        self.assertEqual(len(state.searches), 1)
        self.assertEqual(len(state.search_results), 1)
        self.assertEqual(len(state.evidence), 1)
        self.assertEqual(state.search_results[0].content, state.evidence[0].text)
        self.assertEqual(state.evidence[0].document_version_id, version.version_id)
        self.assertEqual(state.validate_lineage(), [])
        self.assertEqual(state.validate_provenance(), [])

    def test_selected_result_becomes_provenance_evidence_linked_to_artifact(self):
        plan, review, run, task, state, iteration, question = _plan()
        version = register_rag_document_version(
            state,
            source_id="source-one",
            document_id="doc-one",
            filename="paper-one.pdf",
            version_id="doc-one-v1",
        )
        request = LocalRetrievalRequest(
            "synthetic passage method limitations",
            limit=2,
            question_id=question.question_id,
            evidence_document_versions={1: version},
        )
        result = execute_local_retrieval(
            plan, run, task, review, request, CountingProvider(_items()), state,
            iteration_id=iteration.iteration_id,
        )
        artifact = state.research_artifacts[0]
        evidence = state.evidence[0]
        passage = next(item for item in state.passage_references
                       if item.passage_reference_id == evidence.passage_reference_id)
        search_result = next(item for item in state.search_results
                             if item.result_id == evidence.search_result_id)
        self.assertEqual(evidence.text, _items()[0].text)
        self.assertEqual(evidence.source_id, "source-one")
        self.assertEqual(evidence.document_version_id, version.version_id)
        self.assertEqual(passage.page, 2)
        self.assertEqual(passage.chunk_id, "chunk-1")
        self.assertEqual(artifact.evidence_ids, (evidence.evidence_id,))
        self.assertEqual(result.evidence_ids, (evidence.evidence_id,))
        self.assertEqual(search_result.source_id, evidence.source_id)
        self.assertFalse(hasattr(evidence, "verified"))
        self.assertEqual(state.validate_provenance(), [])

    def test_unmapped_results_remain_search_results_without_fabricated_document_provenance(self):
        result, (_, _, _, _, state, _, _), _ = self._execute()
        self.assertEqual(len(state.search_results), 2)
        self.assertEqual(state.evidence, [])
        self.assertEqual(state.document_versions, [])
        self.assertEqual(state.passage_references, [])
        self.assertEqual(result.evidence_ids, ())

    def test_no_results_is_distinct_from_failure_and_still_records_action(self):
        provider = CountingProvider(_items())
        result, (_, _, _, _, state, _, _), _ = self._execute(
            provider=provider,
            request=LocalRetrievalRequest("qzxneverword", limit=2),
        )
        self.assertEqual(result.status, CapabilityStatus.NO_RESULTS)
        self.assertEqual(result.result_count, 0)
        self.assertEqual(len(state.searches), 1)
        self.assertEqual(state.searches[0].result_count, 0)
        self.assertEqual(len(state.research_artifacts), 1)

    def test_unauthorized_task_never_invokes_provider(self):
        plan, review, run, task, state, iteration, _ = _plan()
        unselected = ResearchTask("Not authorized.", task.question_id, task_id="task-unselected")
        plan.tasks.append(unselected)
        provider = CountingProvider(_items())
        with self.assertRaises(TaskExecutionAuthorizationError):
            execute_local_retrieval(
                plan, run, unselected, review,
                LocalRetrievalRequest("method limitations"), provider, state,
                iteration_id=iteration.iteration_id,
            )
        self.assertEqual(provider.calls, [])
        self.assertEqual(state.task_executions, [])
        self.assertEqual(state.searches, [])
        self.assertEqual(state.research_artifacts, [])

    def test_unlinked_run_and_wrong_review_never_invoke_provider(self):
        from research_state import ResearchRun

        plan, review, run, task, state, iteration, _ = _plan()
        provider = CountingProvider(_items())
        with self.assertRaises(TaskExecutionAuthorizationError):
            execute_local_retrieval(
                plan, ResearchRun(), task, review,
                LocalRetrievalRequest("method limitations"), provider, state,
            )
        self.assertEqual(provider.calls, [])

        wrong_review = PlanReview(
            plan_id=plan.plan_id,
            plan_revision=plan.revision,
            status=ReviewStatus.APPROVED,
            reviewer="researcher",
            review_id="another-review",
        )
        with self.assertRaises(TaskExecutionAuthorizationError):
            execute_local_retrieval(
                plan, run, task, wrong_review,
                LocalRetrievalRequest("method limitations"), provider, state,
                iteration_id=iteration.iteration_id,
            )
        self.assertEqual(provider.calls, [])
        self.assertEqual(state.task_executions, [])

    def test_invalid_request_fails_before_provider_or_task_execution(self):
        plan, review, run, task, state, iteration, _ = _plan()
        provider = CountingProvider(_items())
        with self.assertRaises(CapabilityRequestError):
            execute_local_retrieval(
                plan, run, task, review,
                LocalRetrievalRequest("valid query"), provider, state,
                iteration_id="missing-iteration",
            )
        self.assertEqual(provider.calls, [])
        self.assertEqual(state.task_executions, [])

    def test_malformed_request_is_revalidated_before_capability_invocation(self):
        plan, review, run, task, state, iteration, _ = _plan()
        request = LocalRetrievalRequest("valid query")
        object.__setattr__(request, "query", "   ")
        provider = CountingProvider(_items())
        with self.assertRaises(CapabilityRequestError):
            execute_local_retrieval(
                plan, run, task, review, request, provider, state,
                iteration_id=iteration.iteration_id,
            )
        self.assertEqual(provider.calls, [])

    def test_unregistered_document_version_is_rejected_before_provider_call(self):
        plan, review, run, task, state, iteration, _ = _plan()
        from research_state import DocumentVersion
        version = DocumentVersion("doc-one", "source-one", version_id="unregistered")
        provider = CountingProvider(_items())
        with self.assertRaisesRegex(CapabilityRequestError, "exactly registered"):
            execute_local_retrieval(
                plan, run, task, review,
                LocalRetrievalRequest("method limitations", evidence_document_versions={1: version}),
                provider, state, iteration_id=iteration.iteration_id,
            )
        self.assertEqual(provider.calls, [])

    def test_task_handler_does_not_receive_research_state(self):
        import research_capabilities

        original = research_capabilities.retrieve_for_research
        observed = {}

        def inspect_access(provider, access, **kwargs):
            observed["has_retrieval_methods"] = all(
                hasattr(access, name) for name in ("add_search", "add_search_result", "add_evidence")
            )
            observed["has_unrelated_state"] = any(
                hasattr(access, name) for name in ("claims", "task_executions", "research_artifacts", "final_report")
            )
            return original(provider, access, **kwargs)

        with patch("research_capabilities.retrieve_for_research", side_effect=inspect_access):
            result, (_, _, _, _, state, _, _), _ = self._execute()
        self.assertTrue(observed["has_retrieval_methods"])
        self.assertFalse(observed["has_unrelated_state"])
        self.assertEqual(result.execution_id, state.task_executions[0].execution_id)

    def test_provider_failure_records_failed_execution_and_rolls_back_no_artifacts(self):
        class BrokenProvider:
            def __init__(self):
                self.calls = 0
            def retrieve(self, query, limit=8):
                self.calls += 1
                raise RuntimeError("synthetic provider failure")

        provider = BrokenProvider()
        plan, review, run, task, state, iteration, _ = _plan()
        with self.assertRaises(TaskExecutionError) as raised:
            execute_local_retrieval(
                plan, run, task, review, LocalRetrievalRequest("method limitations"),
                provider, state, iteration_id=iteration.iteration_id,
            )
        self.assertEqual(provider.calls, 1)
        self.assertEqual(raised.exception.execution.status, TaskExecutionStatus.FAILED)
        self.assertEqual(state.searches, [])
        self.assertEqual(state.research_artifacts, [])

    def test_artifact_persistence_failure_rolls_back_retrieval_records(self):
        plan, review, run, task, state, iteration, _ = _plan()
        provider = CountingProvider(_items())
        with patch.object(state, "add_research_artifact", side_effect=RuntimeError("storage failure")):
            with self.assertRaises(TaskExecutionError) as raised:
                execute_local_retrieval(
                    plan, run, task, review, LocalRetrievalRequest("method limitations"),
                    provider, state, iteration_id=iteration.iteration_id,
                )
        self.assertEqual(raised.exception.execution.status, TaskExecutionStatus.FAILED)
        self.assertEqual(state.searches, [])
        self.assertEqual(state.search_results, [])
        self.assertEqual(state.sources, [])
        self.assertEqual(state.evidence, [])
        self.assertEqual(state.research_artifacts, [])
        self.assertEqual(state.validate_lineage(), [])

    def test_retry_uses_phase5_history_and_creates_new_capability_artifact(self):
        class FirstFailsThenSucceeds:
            def __init__(self):
                self.calls = 0
            def retrieve(self, query, limit=8):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("first attempt")
                return StaticRetrievalProvider(_items()).retrieve(query, limit)

        provider = FirstFailsThenSucceeds()
        plan, review, run, task, state, iteration, _ = _plan()
        with self.assertRaises(TaskExecutionError) as raised:
            execute_local_retrieval(
                plan, run, task, review, LocalRetrievalRequest("method limitations"), provider,
                state, iteration_id=iteration.iteration_id,
            )
        failed = raised.exception.execution
        successful = execute_local_retrieval(
            plan, run, task, review, LocalRetrievalRequest("method limitations"), provider,
            state, iteration_id=iteration.iteration_id,
            retry_of_execution_id=failed.execution_id,
        )
        self.assertEqual(provider.calls, 2)
        self.assertEqual(failed.status, TaskExecutionStatus.FAILED)
        self.assertEqual(state.task_executions[-1].retry_of_execution_id, failed.execution_id)
        self.assertEqual(len(state.research_artifacts), 1)
        self.assertEqual(state.research_artifacts[0].execution_id, successful.execution_id)

    def test_state_json_round_trip_resolves_artifact_execution_and_retrieval_records(self):
        plan, review, run, task, state, iteration, question = _plan()
        version = register_rag_document_version(
            state,
            source_id="source-one",
            document_id="doc-one",
            filename="paper-one.pdf",
            version_id="doc-one-roundtrip-v1",
        )
        provider = CountingProvider(_items())
        result = execute_local_retrieval(
            plan,
            run,
            task,
            review,
            LocalRetrievalRequest(
                "synthetic passage method limitations",
                limit=2,
                question_id=question.question_id,
                evidence_document_versions={1: version},
            ),
            provider,
            state,
            iteration_id=iteration.iteration_id,
        )
        restored = ResearchState.from_dict(json.loads(json.dumps(state.to_dict())))
        artifact = restored.research_artifacts[0]
        execution = next(item for item in restored.task_executions if item.execution_id == artifact.execution_id)
        search = next(item for item in restored.searches if item.search_id == artifact.search_action_id)
        self.assertEqual(execution.result.artifact_ids, (artifact.artifact_id,))
        self.assertEqual([item.result_id for item in restored.search_results if item.search_id == search.search_id],
                         list(artifact.search_result_ids))
        self.assertEqual(artifact.artifact_id, result.artifact_id)
        self.assertEqual(len(restored.evidence), 1)
        self.assertTrue(restored.evidence[0].passage_reference_id)
        self.assertEqual(restored.evidence[0].document_version_id, version.version_id)
        self.assertEqual(restored.validate_lineage(), [])

    def test_duplicate_artifacts_and_invalid_artifact_evidence_links_are_rejected(self):
        _, (_, _, _, _, state, _, _), _ = self._execute()
        artifact = state.research_artifacts[0]
        with self.assertRaisesRegex(ValueError, "Duplicate ResearchArtifact artifact_id"):
            state.add_research_artifact(artifact)

        payload = state.to_dict()
        payload["research_artifacts"][0]["evidence_ids"] = ["missing-evidence"]
        with self.assertRaisesRegex(ValueError, "Invalid serialized lineage"):
            ResearchState.from_dict(payload)

    def test_legacy_state_without_artifact_collection_still_loads(self):
        state = ResearchState(user_request="Old state")
        payload = state.to_dict()
        payload.pop("research_artifacts")
        self.assertEqual(ResearchState.from_dict(payload).research_artifacts, [])

    def test_malformed_artifact_record_is_rejected(self):
        payload = ResearchState(user_request="Malformed").to_dict()
        payload["research_artifacts"] = [{"artifact_id": "partial"}]
        with self.assertRaisesRegex(ValueError, "Malformed ResearchArtifact"):
            ResearchState.from_dict(payload)

    def test_capability_result_rejects_unknown_or_inconsistent_artifact_references(self):
        with self.assertRaisesRegex(ValueError, "status must match"):
            from research_capabilities import CapabilityResult
            CapabilityResult(
                capability=CapabilityType.LOCAL_RETRIEVAL,
                status=CapabilityStatus.NO_RESULTS,
                execution_id="exec", artifact_id="art", search_action_id="search",
                search_result_ids=("result",), evidence_ids=(), result_count=1,
            )

    def test_capability_behavior_is_domain_neutral(self):
        for domain in ("NLP", "medicine", "agriculture", "physics"):
            with self.subTest(domain=domain):
                result, (_, _, _, _, state, _, _), _ = self._execute(domain=domain)
                self.assertEqual(result.capability, CapabilityType.LOCAL_RETRIEVAL)
                self.assertEqual(len(state.research_artifacts), 1)

    def test_retrieval_artifact_does_not_include_scientific_quality_or_verdict_fields(self):
        self._execute()
        artifact = ResearchArtifact(
            execution_id="execution", run_id="run", task_id="task", search_action_id="search"
        )
        fields = set(artifact.to_dict())
        self.assertFalse({"confidence", "verified", "novelty", "verdict", "quality_score"} & fields)


if __name__ == "__main__":
    unittest.main()
