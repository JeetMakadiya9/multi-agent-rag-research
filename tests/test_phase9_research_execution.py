"""End-to-end Phase 9 research execution over the existing bounded stack.

The corpus, 8A provider, and Experiment B judge are deterministic fixtures. Retrieval
still runs through ExistingRAGProvider and the project's frozen rag.py implementation;
verification still runs through Phase 6, Phase 7B, Phase 8B, and Experiment B.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
for path in (str(ROOT), str(TESTS)):
    if path not in sys.path:
        sys.path.insert(0, path)

from claim_verification import CONTRADICTED, INSUFFICIENT_EVIDENCE, SUPPORTED
from research_autonomy import ResearchActionType
from research_autonomous_controller import (
    AutonomousControllerPolicy,
    AutonomousControllerRequest,
    AutonomousControllerStopReason,
    run_autonomous_controller,
)
from research_autonomous_execution import (
    AutonomousExecutionApproval,
    AutonomousExecutionRequest,
    AutonomousExecutionStatus,
    execution_policy_digest,
    provider_bindings_for_action,
)
from research_bounded_loop import BoundedExecutionPolicy
from research_capabilities import CapabilityType, LocalRetrievalRequest
from research_continuation import ResearchContinuationAction, apply_research_continuation
from research_loop import (
    LoopStage,
    ResearchLoopStatus,
    TransitionAuthority,
    begin_loop_iteration,
    create_research_loop,
    start_research_loop,
    transition_research_loop,
)
from research_plan_review import PlanReview, ReviewStatus
from research_planning import (
    EvidenceRequirement,
    ResearchObjective,
    ResearchPlan,
    ResearchPlanQuestion,
    ResearchRequest,
    ResearchTask,
)
from research_retrieval import ExistingRAGProvider, RetrievalEvidence, register_rag_document_version
from research_run_authorization import authorize_research_run
from research_state import Evidence, ResearchClaim, ResearchQuestion, ResearchState
from research_task_authorization import authorize_research_tasks
from research_verification_capability import EvidenceVerificationRequest
from verification_units import analyze_verification_units


RESEARCH_QUESTION = (
    "Assess what the available AquaSort test records establish about dye removal, "
    "energy use, and the limits of the reported test."
)
CLAIMS = (
    "AquaSort removed 20% of blue dye in the reported single-sample run.",
    "AquaSort removed 90% of blue dye in the reported single-sample run.",
    "AquaSort reduced energy use by 30%.",
    "The AquaSort test used one sample.",
)
EXPECTED_VERDICTS = (SUPPORTED, CONTRADICTED, INSUFFICIENT_EVIDENCE, SUPPORTED)
EXPECTED_STANCES = ("SUPPORTS", "CONTRADICTS", "NEUTRAL", "SUPPORTS")


def _corpus():
    return [
        {
            "chunk_id": "aquasort-results-1", "document_id": "aquasort-report-1",
            "filename": "aquasort-results.pdf", "page": 2, "section": "Results",
            "text": "Testing records show the AquaSort cartridge removed 20% of blue dye from one sample during one controlled run.",
        },
        {
            "chunk_id": "aquasort-methods-1", "document_id": "aquasort-report-1",
            "filename": "aquasort-results.pdf", "page": 3, "section": "Methods",
            "text": "The report describes AquaSort operation in blue dye samples but does not measure energy use.",
        },
        {
            "chunk_id": "aquasort-limitations-1", "document_id": "aquasort-report-1",
            "filename": "aquasort-results.pdf", "page": 5, "section": "Limitations",
            "text": "The AquaSort test used only one sample and did not report replicates.",
        },
        {
            "chunk_id": "unrelated-astronomy-1", "document_id": "unrelated-astronomy",
            "filename": "astronomy-notes.pdf", "page": 1, "section": "Overview",
            "text": "The Moon is made of rock and orbits Earth once each month.",
        },
    ]


def _retrieval_provider(*, fail=False):
    import rag

    class FixedReranker:
        def predict(self, pairs, show_progress_bar=False):
            return [1.0 for _ in pairs]

    class Provider(ExistingRAGProvider):
        def __init__(self):
            super().__init__(chunks=_corpus(), embedding_model=None, reranker=FixedReranker(),
                             faiss_index=None, bm25_index=None, retrieval_module=rag)
            self.calls = []

        def retrieve(self, query, limit=8):
            self.calls.append((query, limit))
            if fail:
                raise RuntimeError("synthetic local retrieval failure")
            return super().retrieve(query, limit)

    return Provider()


class ScriptedResearcher:
    """Deterministic 8A binding; actual proposal parsing and validation remain in 8A."""

    def __init__(self, *, task_override=None):
        self.prompts = []
        self.decisions = []
        self.synthesis_issued = False
        self.task_override = task_override

    def generate(self, messages, *, temperature=0.0, timeout=120):
        prompt = messages[1]["content"]
        self.prompts.append(prompt)
        context = json.loads(prompt.split("Context (read-only):\n", 1)[1])
        stage = context["current_stage"]
        if stage == LoopStage.RESEARCH.value:
            data = _proposal("RETRIEVE_EVIDENCE", task_id=self.task_override or context["current_task_id"],
                             question_ids=[context["questions"][0]["question_id"]],
                             query="AquaSort blue dye sample energy replicates")
        elif stage == LoopStage.EVIDENCE_REVIEW.value:
            verification_count = sum(item["artifact_type"] == "verification_result"
                                     for item in context["artifacts"])
            claim = context["claims"][verification_count]
            data = _proposal("VERIFY_CLAIM", task_id=context["current_task_id"],
                             question_ids=[context["questions"][0]["question_id"]],
                             claim_ids=[claim["claim_id"]], evidence_ids=claim["evidence_ids"])
        elif stage == LoopStage.SYNTHESIS.value and not self.synthesis_issued:
            self.synthesis_issued = True
            data = _proposal("SYNTHESIZE", artifact_ids=[item["artifact_id"]
                             for item in context["artifacts"]])
        else:
            data = _proposal("STOP", stop_assessment={
                "stop_reason": "phase9_fixture_complete",
                "evidence_summary": "All bounded fixture claims received typed Experiment B verdicts.",
                "unresolved_questions": [], "remaining_known_limitations": [],
                "assessment": "The deterministic research workflow reached its explicit stopping action.",
            })
        self.decisions.append(data["action_type"])
        return json.dumps(data)


class ScriptedExperimentBJudge:
    """Deterministic pair assessor; Experiment B still aggregates and validates verdicts."""

    def __init__(self):
        self.responses = []
        self.prompts = []

    def generate(self, messages, *, temperature=0.0, timeout=120):
        self.prompts.append(messages[1]["content"])
        index = len(self.responses)
        claim = CLAIMS[index]
        component_ids = [item.component_id for item in analyze_verification_units(claim)[0].components]
        self.responses.append(claim)
        return json.dumps({
            "assessments": [{
                "evidence_index": "E1", "component_indices": component_ids,
                "stance": EXPECTED_STANCES[index],
            }],
            "reason": "Deterministic Phase 9 fixture assessment.",
        })


def _proposal(action_type, *, task_id=None, question_ids=(), claim_ids=(), evidence_ids=(),
              artifact_ids=(), query=None, stop_assessment=None):
    return {
        "action_type": action_type,
        "rationale": "Perform the next explicitly scoped Phase 9 research action.",
        "task_id": task_id,
        "question_ids": list(question_ids),
        "claim_ids": list(claim_ids),
        "evidence_ids": list(evidence_ids),
        "artifact_ids": list(artifact_ids),
        "query": query,
        "problem_statement": None,
        "candidate_id": None,
        "estimated_information_gain": 0.5,
        "priority": "medium",
        "unresolved_questions_addressed": [],
        "stop_after_action": False,
        "stop_assessment": stop_assessment,
        "provenance_ids": [],
    }


def _new_fixture(*, retrieval=None, approval_provider=None, authorized=True):
    question = ResearchPlanQuestion(
        "What do the AquaSort records establish about dye removal, energy use, and test limitations?",
        question_id="phase9-main-question",
    )
    requirement = EvidenceRequirement(question.question_id,
                                      "Retrieve passage-level results and limitations from local records.",
                                      requirement_id="phase9-evidence-requirement")
    task = ResearchTask(
        "Retrieve and assess locally stored AquaSort test evidence.", question.question_id,
        evidence_requirement_ids=[requirement.requirement_id], task_id="phase9-research-task",
    )
    other_task = ResearchTask(
        "Review the separately authorized comparison question.", question.question_id,
        task_id="phase9-other-authorized-task",
    )
    plan = ResearchPlan(
        request=ResearchRequest(RESEARCH_QUESTION,
                                "Assess scoped technical claims against locally available passages.",
                                domain="deterministic-test-corpus"),
        objective=ResearchObjective("Determine what the retrieved test record supports and leaves unresolved."),
        questions=[question], evidence_requirements=[requirement], tasks=[task, other_task], plan_id="phase9-plan",
    )
    review = PlanReview(plan.plan_id, plan.revision, ReviewStatus.APPROVED, reviewer="fixture researcher")
    state = ResearchState(user_request=RESEARCH_QUESTION)
    run = authorize_research_run(plan, review, state)
    authorized_tasks = [task.task_id] if authorized else [other_task.task_id]
    authorize_research_tasks(plan, run, authorized_tasks, review, state)
    iteration = state.create_iteration(run.run_id)
    research_question = state.add_research_question(ResearchQuestion(
        text=question.text, question_id=question.question_id,
        run_id=run.run_id,
    ))
    retrieval = retrieval or _retrieval_provider()
    judge = ScriptedExperimentBJudge()
    versions = []
    for index, chunk in enumerate(_corpus()[:3], start=1):
        versions.append(register_rag_document_version(
            state, source_id=f"{chunk['filename']}:page-{chunk['page']}",
            document_id=chunk["document_id"], filename=chunk["filename"],
            version_id=f"phase9-document-version-{index}",
        ))
    loop = create_research_loop(plan, run, review, state, max_iterations=7)
    start_research_loop(state, loop.loop_id)
    loop = transition_research_loop(state, loop.loop_id, LoopStage.RESEARCH,
                                    reason="Researcher approved the bounded local evidence review.")
    active_task = task if authorized else other_task
    begin_loop_iteration(state, loop.loop_id, (active_task.task_id,))
    loop = state.research_loops[0]
    llm = ScriptedResearcher(task_override=task.task_id if not authorized else None)
    execution_policy = BoundedExecutionPolicy(
        maximum_iterations=7, maximum_capability_invocations=6,
        maximum_task_executions=6,
        allowed_capabilities=(CapabilityType.LOCAL_RETRIEVAL, CapabilityType.EVIDENCE_VERIFICATION),
        allowed_stages=(LoopStage.RESEARCH, LoopStage.EVIDENCE_REVIEW,
                        LoopStage.VERIFICATION, LoopStage.SYNTHESIS),
        allowed_task_ids=(active_task.task_id,), allow_continuation=True,
    )
    controller_policy = AutonomousControllerPolicy(
        maximum_cycles=1, maximum_executions=1, maximum_capability_invocations=1,
        maximum_retrieval_actions=1, maximum_verification_actions=1,
        maximum_total_research_actions=1, require_researcher_review=True,
    )

    def request_factory(decision, validation, context):
        proposal = decision.proposal
        capability_request = None
        if proposal.action_type is ResearchActionType.RETRIEVE_EVIDENCE:
            capability_request = LocalRetrievalRequest(
                proposal.query, limit=8,
                question_id=proposal.question_ids[0] if proposal.question_ids else None,
                evidence_document_versions={i: version for i, version in enumerate(versions, start=1)},
            )
        elif proposal.action_type is ResearchActionType.VERIFY_CLAIM:
            claim = next(item for item in state.claims if item.claim_id == proposal.claim_ids[0])
            capability_request = EvidenceVerificationRequest(claim.text, tuple(proposal.evidence_ids))
        refs = provider_bindings_for_action(
            proposal.action_type,
            retrieval_provider=retrieval,
            llm_provider=judge,
        ) if proposal.action_type in {
            ResearchActionType.RETRIEVE_EVIDENCE,
            ResearchActionType.VERIFY_CLAIM,
        } else ()
        return AutonomousExecutionRequest(
            decision, validation, run.run_id, loop.loop_id, context.iteration_id,
            plan.plan_id, plan.revision, execution_policy_digest(execution_policy),
            capability_request=capability_request, provider_refs=refs,
        )

    approve = approval_provider or (lambda request: AutonomousExecutionApproval.for_request(
        request, reviewer="fixture researcher"))

    def controller_request():
        current = next(item for item in state.research_loops if item.loop_id == loop.loop_id)
        return AutonomousControllerRequest(
            plan=plan, run=run, review=review, loop=current, state=state,
            execution_policy=execution_policy, policy=controller_policy,
            decision_provider=llm, request_factory=request_factory,
            approval_provider=approve, retrieval_provider=retrieval, llm_provider=judge,
        )

    return {
        "plan": plan, "review": review, "run": run, "task": task, "state": state,
        "question": question, "research_question": research_question, "loop_id": loop.loop_id,
        "retrieval": retrieval, "judge": judge, "llm": llm, "versions": versions,
        "execution_policy": execution_policy, "controller_policy": controller_policy,
        "controller_request": controller_request, "request_factory": request_factory,
        "approval_provider": approve,
    }


def _one_controller_call(fixture):
    return run_autonomous_controller(fixture["controller_request"]())


def _continue_same_task(fixture, result):
    execution_id = result.trace.cycles[-1].execution.trace.controlled_execution_id
    return apply_research_continuation(
        fixture["plan"], fixture["run"], fixture["review"],
        fixture["state"].research_loops[0], fixture["state"],
        action=ResearchContinuationAction.CONTINUE_SAME_TASK,
        previous_execution_id=execution_id,
    )


def _complete_workflow():
    fixture = _new_fixture()
    controller_results = []
    retrieval_result = _one_controller_call(fixture)
    controller_results.append(retrieval_result)
    assert retrieval_result.stop_reason is AutonomousControllerStopReason.MAXIMUM_CYCLES
    assert retrieval_result.trace.cycles[0].execution.status is AutonomousExecutionStatus.COMPLETED
    assert len(fixture["state"].evidence) == 3

    evidence = {item.chunk_id: item for item in fixture["state"].evidence}
    linked_rows = (
        (CLAIMS[0], evidence["aquasort-results-1"]),
        (CLAIMS[1], evidence["aquasort-results-1"]),
        (CLAIMS[2], evidence["aquasort-methods-1"]),
        (CLAIMS[3], evidence["aquasort-limitations-1"]),
    )
    for claim_text, source_evidence in linked_rows:
        fixture["state"].add_claim(ResearchClaim(
            text=claim_text, status="NOT_VALIDATED",
            evidence_ids=[source_evidence.evidence_id], source_ids=[source_evidence.source_id],
            verification_reason="Researcher-supplied claim awaiting Phase 6 verification.",
        ))

    _continue_same_task(fixture, retrieval_result)
    transition_research_loop(
        fixture["state"], fixture["loop_id"], LoopStage.EVIDENCE_REVIEW,
        reason="Researcher authorized verification of the retrieved passages.",
        authority=TransitionAuthority.RESEARCHER,
    )
    verification_results = []
    for index, expected in enumerate(EXPECTED_VERDICTS):
        result = _one_controller_call(fixture)
        controller_results.append(result)
        verification_results.append(result)
        cycle = result.trace.cycles[0]
        assert cycle.validation.validation_status.value == "ACCEPTED"
        assert cycle.execution.status is AutonomousExecutionStatus.COMPLETED
        assert cycle.execution.dispatch.status.value == "COMPLETED"
        assert fixture["state"].research_artifacts[-1].verification_verdict == expected
        verified_artifact = next(
            item for item in reversed(fixture["state"].research_artifacts)
            if item.artifact_type == "verification_result"
            and item.verification_claim == CLAIMS[index]
        )
        verified_claim = next(item for item in fixture["state"].claims
                              if item.text == CLAIMS[index])
        # Phase 6 persists the authoritative typed verdict as a ResearchArtifact;
        # the caller reflects that verdict in the mutable ResearchClaim summary.
        verified_claim.status = verified_artifact.verification_verdict
        verified_claim.verification_reason = verified_artifact.verification_assessments[0].explanation
        if index < len(EXPECTED_VERDICTS) - 1:
            _continue_same_task(fixture, result)

    transition_research_loop(
        fixture["state"], fixture["loop_id"], LoopStage.VERIFICATION,
        reason="All selected claims have typed Phase 6 verification artifacts.",
        authority=TransitionAuthority.RESEARCHER,
    )
    transition_research_loop(
        fixture["state"], fixture["loop_id"], LoopStage.SYNTHESIS,
        reason="Researcher authorized synthesis over stored verification artifacts.",
        authority=TransitionAuthority.RESEARCHER,
    )
    synthesis_result = _one_controller_call(fixture)
    controller_results.append(synthesis_result)
    assert synthesis_result.trace.cycles[0].execution.synthesis is not None
    stop_result = _one_controller_call(fixture)
    controller_results.append(stop_result)
    assert stop_result.stop_reason is AutonomousControllerStopReason.DECISION_STOP
    researcher_stop = apply_research_continuation(
        fixture["plan"], fixture["run"], fixture["review"],
        fixture["state"].research_loops[0], fixture["state"],
        action=ResearchContinuationAction.STOP,
    )

    synthesis = synthesis_result.trace.cycles[0].execution.synthesis
    final = _structured_result(fixture, synthesis, stop_result, controller_results, researcher_stop)
    return fixture | {
        "controller_results": tuple(controller_results),
        "retrieval_result": retrieval_result,
        "verification_results": tuple(verification_results),
        "synthesis_result": synthesis_result,
        "stop_result": stop_result,
        "researcher_stop": researcher_stop,
        "synthesis": synthesis,
        "final": final,
    }


def _structured_result(fixture, synthesis, stop_result, controller_results, researcher_stop):
    findings = [item.to_dict() for item in synthesis.findings]
    provenance = [item.to_dict() for item in synthesis.evidence_references]
    contradictions = [item for item in findings if item["classification"] == CONTRADICTED]
    limitations = [item for item in findings if item["classification"] == SUPPORTED
                   and "one sample" in item["statement"].lower()]
    return {
        "research_question": RESEARCH_QUESTION,
        "research_status": "COMPLETED",
        "claims_investigated": findings,
        "evidence_used": [item.evidence_id for item in synthesis.evidence_references],
        "evidence_provenance": provenance,
        "verification_status": {item["statement"]: item["classification"] for item in findings},
        "contradictions": contradictions,
        "documented_limitations": limitations,
        "unresolved_questions": list(fixture["state"].unresolved_questions),
        "final_synthesis": synthesis.to_dict(),
        "trace_references": {
            "run_id": fixture["run"].run_id,
            "loop_id": fixture["loop_id"],
            "controller_ids": [item.trace.controller_id for item in controller_results],
            "execution_ids": [cycle.execution.trace.controlled_execution_id
                              for result in controller_results
                              for cycle in result.trace.cycles if cycle.execution],
            "artifact_ids": list(synthesis.source_artifact_ids),
        },
        "stop_reason": researcher_stop.status.value + ":" + fixture["state"].research_loops[0].stop_reason.value,
    }


class Phase9ResearchExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workflow = _complete_workflow()

    def test_user_research_request_is_registered_in_research_state(self):
        self.assertEqual(self.workflow["state"].user_request, RESEARCH_QUESTION)

    def test_research_run_is_registered_and_bound_to_approved_plan(self):
        f = self.workflow
        self.assertIn(f["run"], f["state"].research_runs)
        self.assertEqual(f["run"].plan_id, f["plan"].plan_id)
        self.assertEqual(f["run"].approval_review_id, f["review"].review_id)

    def test_all_decisions_went_through_actual_8a_provider(self):
        self.assertEqual(self.workflow["llm"].decisions,
                         ["RETRIEVE_EVIDENCE", "VERIFY_CLAIM", "VERIFY_CLAIM",
                          "VERIFY_CLAIM", "VERIFY_CLAIM", "SYNTHESIZE", "STOP"])

    def test_all_8a_validation_traces_are_accepted(self):
        for result in self.workflow["controller_results"]:
            self.assertEqual(result.trace.cycles[0].validation.validation_status.value, "ACCEPTED")

    def test_each_decision_is_bound_to_its_exact_8b_execution(self):
        for result in self.workflow["controller_results"]:
            cycle = result.trace.cycles[0]
            if cycle.decision.proposal.action_type is ResearchActionType.STOP:
                self.assertIsNone(cycle.execution)
                continue
            self.assertEqual(cycle.execution.trace.decision_id, cycle.decision.trace.decision_id)
            self.assertEqual(cycle.execution.trace.request_id, cycle.request_id)
            self.assertEqual(cycle.execution.trace.approval_id, cycle.approval_id)

    def test_retrieval_used_existing_advanced_rag_provider(self):
        self.assertIsInstance(self.workflow["retrieval"], ExistingRAGProvider)
        self.assertEqual(self.workflow["state"].searches[0].provider, "existing_advanced_rag")

    def test_retrieval_executed_once_inside_controlled_dispatch(self):
        self.assertEqual(len(self.workflow["retrieval"].calls), 1)
        dispatch = self.workflow["retrieval_result"].trace.cycles[0].execution.dispatch
        self.assertEqual(dispatch.capability.value, "local_retrieval")

    def test_rag_returned_multiple_real_retrieval_evidence_objects(self):
        self.assertGreaterEqual(len(self.workflow["state"].evidence), 3)
        self.assertTrue(all(isinstance(item, Evidence) for item in self.workflow["state"].evidence))
        self.assertEqual(len(self.workflow["retrieval_result"].trace.cycles[0].execution.trace.evidence_ids), 3)

    def test_relevant_results_include_blue_dye_test_passage(self):
        texts = [item.text for item in self.workflow["state"].evidence]
        self.assertTrue(any("removed 20% of blue dye" in text for text in texts))

    def test_relevant_results_include_the_documented_test_limitation(self):
        self.assertTrue(any("one sample" in item.text and "replicates" in item.text
                            for item in self.workflow["state"].evidence))

    def test_irrelevant_astronomy_passage_is_not_returned_or_used(self):
        self.assertFalse(any("Moon" in item.text for item in self.workflow["state"].evidence))
        self.assertFalse(any("unrelated-astronomy-1" in item.chunk_id
                             for item in self.workflow["state"].evidence))

    def test_evidence_has_source_document_version_and_passage_provenance(self):
        state = self.workflow["state"]
        self.assertEqual(state.validate_provenance(), [])
        for item in state.evidence:
            self.assertTrue(item.source_id)
            self.assertTrue(item.document_version_id)
            self.assertTrue(item.passage_reference_id)
            self.assertTrue(item.search_result_id)
            self.assertTrue(item.text_hash)

    def test_evidence_search_lineage_resolves_to_current_research_run(self):
        f = self.workflow
        results = {item.result_id: item for item in f["state"].search_results}
        searches = {item.search_id: item for item in f["state"].searches}
        for item in f["state"].evidence:
            result = results[item.search_result_id]
            search = searches[result.search_id]
            self.assertEqual(search.run_id, f["run"].run_id)
            self.assertEqual(search.iteration_id, f["retrieval_result"].trace.cycles[0].execution.trace.iteration_id)

    def test_evidence_task_and_capability_execution_lineage_resolves(self):
        f = self.workflow
        executions = {item.execution_id: item for item in f["state"].task_executions}
        for artifact in f["state"].research_artifacts:
            self.assertIn(artifact.execution_id, executions)
            self.assertEqual(artifact.run_id, f["run"].run_id)
            self.assertEqual(artifact.task_id, f["task"].task_id)

    def test_retrieval_produced_persisted_artifact_and_provenance_links(self):
        dispatch = self.workflow["retrieval_result"].trace.cycles[0].execution.dispatch
        self.assertTrue(dispatch.artifact_ids)
        self.assertEqual(set(dispatch.evidence_ids), {item.evidence_id for item in self.workflow["state"].evidence})

    def test_following_8a_context_contains_fresh_retrieved_evidence(self):
        prompt = self.workflow["llm"].prompts[1]
        context = json.loads(prompt.split("Context (read-only):\n", 1)[1])
        self.assertEqual(len(context["evidence"]), 3)
        self.assertEqual(context["current_stage"], LoopStage.EVIDENCE_REVIEW.value)

    def test_8a_context_ids_change_after_retrieval_and_explicit_continuation(self):
        prompts = self.workflow["llm"].prompts
        contexts = [json.loads(item.split("Context (read-only):\n", 1)[1]) for item in prompts]
        self.assertNotEqual(contexts[0]["iteration_id"], contexts[1]["iteration_id"])
        self.assertNotEqual(contexts[0]["evidence"], contexts[1]["evidence"])

    def test_verification_capability_runs_through_phase7b_dispatch(self):
        f = self.workflow
        dispatches = [cycle.execution.dispatch for result in f["verification_results"]
                      for cycle in result.trace.cycles]
        self.assertEqual(len(dispatches), 4)
        self.assertTrue(all(item.capability is CapabilityType.EVIDENCE_VERIFICATION for item in dispatches))

    def test_experiment_b_judge_was_called_for_each_claim_assessment(self):
        self.assertEqual(len(self.workflow["judge"].prompts), 4)
        self.assertEqual(self.workflow["judge"].responses, list(CLAIMS))

    def test_supported_claim_uses_typed_supported_verification_artifact(self):
        artifact = next(item for item in self.workflow["state"].research_artifacts
                        if item.verification_claim == CLAIMS[0])
        self.assertEqual(artifact.verification_verdict, SUPPORTED)
        self.assertEqual(artifact.verification_assessments[0].verdict, SUPPORTED)

    def test_contradicted_claim_is_preserved_as_contradicted(self):
        artifact = next(item for item in self.workflow["state"].research_artifacts
                        if item.verification_claim == CLAIMS[1])
        self.assertEqual(artifact.verification_verdict, CONTRADICTED)

    def test_relevant_but_non_entailing_evidence_is_not_counted_as_support(self):
        artifact = next(item for item in self.workflow["state"].research_artifacts
                        if item.verification_claim == CLAIMS[2])
        self.assertEqual(artifact.verification_assessments[0].evidence_assessments[0].stance, "NEUTRAL")
        self.assertEqual(artifact.verification_verdict, INSUFFICIENT_EVIDENCE)

    def test_energy_reduction_claim_remains_insufficient_not_supported(self):
        finding = next(item for item in self.workflow["synthesis"].findings if item.statement == CLAIMS[2])
        self.assertEqual(finding.classification.value, INSUFFICIENT_EVIDENCE)

    def test_typed_verdicts_update_research_claim_epistemic_status(self):
        claims = {item.text: item for item in self.workflow["state"].claims}
        self.assertEqual([claims[text].status for text in CLAIMS], list(EXPECTED_VERDICTS))
        self.assertTrue(all(claims[text].verification_reason for text in CLAIMS))

    def test_next_8a_context_preserves_prior_claim_status(self):
        contexts = [json.loads(prompt.split("Context (read-only):\n", 1)[1])
                    for prompt in self.workflow["llm"].prompts]
        claim_rows = {item["text"]: item for item in contexts[2]["claims"]}
        self.assertEqual(claim_rows[CLAIMS[0]]["status"], SUPPORTED)
        self.assertEqual(claim_rows[CLAIMS[1]]["status"], "NOT_VALIDATED")

    def test_supported_documented_limitation_is_preserved(self):
        artifact = next(item for item in self.workflow["state"].research_artifacts
                        if item.verification_claim == CLAIMS[3])
        self.assertEqual(artifact.verification_verdict, SUPPORTED)
        limitation = next(item for item in self.workflow["final"]["documented_limitations"])
        self.assertIn("one sample", limitation["statement"])

    def test_verification_artifacts_reference_existing_evidence(self):
        evidence_ids = {item.evidence_id for item in self.workflow["state"].evidence}
        for artifact in self.workflow["state"].research_artifacts:
            if artifact.artifact_type == "verification_result":
                self.assertTrue(set(artifact.evidence_ids).issubset(evidence_ids))

    def test_each_phase7c_continuation_is_explicit_and_bound_to_previous_execution(self):
        f = self.workflow
        authorizations = [loop_auth for loop_auth in f["state"].research_loops[0].continuation_authorizations]
        self.assertEqual(len(authorizations), 4)
        completed = {item.execution_id for item in f["state"].task_executions}
        self.assertTrue(all(item.previous_execution_id in completed for item in authorizations))

    def test_no_execution_was_retried_automatically(self):
        executions = self.workflow["state"].task_executions
        self.assertEqual(len(executions), 5)
        self.assertTrue(all(item.retry_of_execution_id is None for item in executions))

    def test_all_phase6_actions_were_bounded_by_registered_policy(self):
        f = self.workflow
        self.assertEqual(len(f["state"].task_executions), 5)
        self.assertLessEqual(len(f["state"].research_loops[0].loop_iterations), 7)
        self.assertLessEqual(len(f["state"].research_loops[0].capability_requests), 6)

    def test_synthesis_was_generated_by_the_existing_8b_action(self):
        cycle = self.workflow["synthesis_result"].trace.cycles[0]
        self.assertEqual(cycle.decision.proposal.action_type, ResearchActionType.SYNTHESIZE)
        self.assertIsNotNone(cycle.execution.synthesis)
        self.assertEqual(cycle.execution.status, AutonomousExecutionStatus.COMPLETED)

    def test_final_synthesis_classifies_all_investigated_claims(self):
        statuses = {item.statement: item.classification.value for item in self.workflow["synthesis"].findings}
        self.assertEqual([statuses[claim] for claim in CLAIMS], list(EXPECTED_VERDICTS))

    def test_final_synthesis_carries_passage_level_evidence_references(self):
        refs = self.workflow["synthesis"].evidence_references
        self.assertGreaterEqual(len(refs), 3)
        for item in refs:
            self.assertTrue(item.source_id)
            self.assertTrue(item.document_version_id)
            self.assertTrue(item.passage_reference_id)
            self.assertTrue(item.search_result_id)
            self.assertTrue(item.search_action_id)
            self.assertTrue(item.text_hash)

    def test_final_result_contains_trace_ids_back_to_the_research_run(self):
        final = self.workflow["final"]
        self.assertEqual(final["trace_references"]["run_id"], self.workflow["run"].run_id)
        self.assertEqual(final["trace_references"]["loop_id"], self.workflow["loop_id"])
        self.assertEqual(len(final["trace_references"]["execution_ids"]), 6)

    def test_final_result_is_structured_and_json_serializable(self):
        final = self.workflow["final"]
        for key in ("research_question", "research_status", "claims_investigated", "evidence_used",
                    "evidence_provenance", "verification_status", "contradictions",
                    "documented_limitations", "unresolved_questions", "final_synthesis",
                    "trace_references", "stop_reason"):
            self.assertIn(key, final)
        json.loads(json.dumps(final))

    def test_final_stop_reason_is_typed_and_loop_is_stopped(self):
        self.assertEqual(self.workflow["stop_result"].stop_reason,
                         AutonomousControllerStopReason.DECISION_STOP)
        self.assertEqual(self.workflow["state"].research_loops[0].status, ResearchLoopStatus.STOPPED)

    def test_research_state_lineage_and_provenance_validate_after_completion(self):
        state = self.workflow["state"]
        self.assertEqual(state.validate_lineage(), [])
        self.assertEqual(state.validate_provenance(), [])


class Phase9NegativePathTests(unittest.TestCase):
    def test_missing_researcher_approval_waits_before_retrieval_dispatch(self):
        fixture = _new_fixture(approval_provider=lambda _request: None)
        result = _one_controller_call(fixture)
        self.assertEqual(result.stop_reason, AutonomousControllerStopReason.WAIT_FOR_RESEARCHER)
        self.assertEqual(fixture["retrieval"].calls, [])
        self.assertEqual(fixture["state"].task_executions, [])

    def test_missing_phase7c_continuation_waits_without_second_execution(self):
        fixture = _new_fixture()
        first = _one_controller_call(fixture)
        result = _one_controller_call(fixture)
        self.assertEqual(result.stop_reason, AutonomousControllerStopReason.WAIT_FOR_RESEARCHER)
        self.assertIn("Phase 7C", result.trace.cycles[0].reason)
        self.assertEqual(len(fixture["state"].task_executions), 1)
        self.assertEqual(len(fixture["retrieval"].calls), 1)
        self.assertTrue(first.trace.cycles[0].execution.trace.controlled_execution_id)

    def test_unapproved_task_is_rejected_before_phase6_capability_execution(self):
        fixture = _new_fixture(authorized=False)
        result = _one_controller_call(fixture)
        self.assertIn(result.stop_reason, {
            AutonomousControllerStopReason.VALIDATION_REJECTED,
            AutonomousControllerStopReason.DECISION_STOP,
        })
        self.assertEqual(fixture["retrieval"].calls, [])
        self.assertEqual(fixture["state"].task_executions, [])

    def test_failed_phase6_retrieval_is_traced_and_never_retried(self):
        provider = _retrieval_provider(fail=True)
        fixture = _new_fixture(retrieval=provider)
        result = _one_controller_call(fixture)
        self.assertEqual(result.stop_reason, AutonomousControllerStopReason.EXECUTION_FAILED)
        self.assertEqual(result.trace.cycles[0].execution.status, AutonomousExecutionStatus.FAILED)
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(len(fixture["state"].task_executions), 1)

    def test_failed_capability_keeps_phase7b_and_8b_failure_trace(self):
        fixture = _new_fixture(retrieval=_retrieval_provider(fail=True))
        result = _one_controller_call(fixture)
        actions = {(event.agent, event.action) for event in fixture["state"].events}
        self.assertIn(("ResearchLoop", "capability_result_recorded"), actions)
        self.assertIn(("autonomous_execution", "proposal_execution_trace"), actions)
        self.assertEqual(result.trace.cycles[0].execution.trace.status, AutonomousExecutionStatus.FAILED)

    def test_no_evidence_request_is_rejected_instead_of_fabricating_insufficient_verdict(self):
        from research_verification_capability import EvidenceVerificationRequest, VerificationCapabilityError
        with self.assertRaises(VerificationCapabilityError):
            EvidenceVerificationRequest("AquaSort reduced energy use by 30%.", ())


if __name__ == "__main__":
    unittest.main()
