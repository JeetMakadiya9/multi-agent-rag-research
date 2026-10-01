"""Executable System C binding to the existing bounded Phase 8C workflow."""
from __future__ import annotations

import json
from typing import Any

from evaluation.experiments.evaluator import SystemExecutionError, SystemOutput
from evaluation.experiments.phase10e_infrastructure import OutputState


def _proposal(action: str, *, task_id: str, question_id: str,
              claim_id: str | None = None, evidence_ids: tuple[str, ...] = (),
              query: str | None = None) -> dict[str, Any]:
    return {
        "action_type": action,
        "rationale": "Execute the next approved, bounded SciFact evidence action.",
        "task_id": task_id,
        "question_ids": [question_id],
        "claim_ids": [claim_id] if claim_id else [],
        "evidence_ids": list(evidence_ids),
        "artifact_ids": [],
        "query": query,
        "problem_statement": None,
        "candidate_id": None,
        "estimated_information_gain": 0.5,
        "priority": "medium",
        "unresolved_questions_addressed": [],
        "stop_after_action": False,
        "stop_assessment": None,
        "provenance_ids": [],
    }


class _Deterministic8AProvider:
    """Smoke-only decision binding; production 8A parsing and validation run."""
    def __init__(self, claim: str, evidence_limit: int = 8):
        self.claim = claim
        self.evidence_limit = evidence_limit
        self.calls = 0
        self.actions: list[str] = []

    def generate(self, messages, *, temperature=0.0, timeout=120):
        self.calls += 1
        prompt = messages[1]["content"]
        context = json.loads(prompt.split("Context (read-only):\n", 1)[1])
        task_id = context["current_task_id"]
        question_id = context["questions"][0]["question_id"]
        stage = context["current_stage"]
        if stage == "RESEARCH":
            action = _proposal("RETRIEVE_EVIDENCE", task_id=task_id,
                               question_id=question_id, query=self.claim)
        elif stage == "EVIDENCE_REVIEW":
            claim = context["claims"][0]
            action = _proposal("VERIFY_CLAIM", task_id=task_id,
                               question_id=question_id, claim_id=claim["claim_id"],
                               evidence_ids=tuple(claim["evidence_ids"][:self.evidence_limit]))
        else:
            raise RuntimeError(f"Unexpected Phase 8A stage in smoke binding: {stage}")
        self.actions.append(action["action_type"])
        return json.dumps(action)


class _CountingRetrieval:
    def __init__(self, provider):
        self.provider = provider
        self.responses = []
        self.calls = 0

    def retrieve(self, query, limit=8):
        response = self.provider.retrieve(query, limit=limit)
        self.calls += 1
        self.responses.append(response)
        return response


class _CountingJudge:
    def __init__(self, provider):
        self.provider = provider
        self.calls = 0

    def generate(self, messages, *, temperature=0.0, timeout=120):
        self.calls += 1
        return self.provider.generate(messages, temperature=temperature, timeout=timeout)


def run_system_c(runtime: dict[str, Any], provider: Any, config: Any, *,
                 llm_provider: Any = None, decision_provider: Any = None) -> SystemOutput:
    """Run retrieval and verification through the real Phase 8C controller.

    A preliminary retrieval captures exact rank-to-document provenance versions
    required by the existing Phase 6 capability. The approved controller then
    performs its own retrieval and verification invocations. Both retrievals
    are counted in the resulting cost.
    """
    from claim_verification import INSUFFICIENT_EVIDENCE
    from .local_model import OllamaCPUProvider
    from research_autonomous_controller import (
        AutonomousControllerPolicy, AutonomousControllerRequest, run_autonomous_controller,
    )
    from research_autonomous_execution import (
        AutonomousExecutionApproval, AutonomousExecutionRequest, execution_policy_digest,
        provider_bindings_for_action,
    )
    from research_bounded_loop import BoundedExecutionPolicy
    from research_capabilities import CapabilityType, LocalRetrievalRequest
    from research_continuation import ResearchContinuationAction, apply_research_continuation
    from research_loop import (
        LoopStage, TransitionAuthority, begin_loop_iteration, create_research_loop,
        start_research_loop, transition_research_loop,
    )
    from research_plan_review import PlanReview, ReviewStatus
    from research_planning import (
        EvidenceRequirement, ResearchObjective, ResearchPlan, ResearchPlanQuestion,
        ResearchRequest, ResearchTask,
    )
    from research_retrieval import register_rag_document_version
    from research_run_authorization import authorize_research_run
    from research_state import ResearchClaim, ResearchQuestion, ResearchState
    from research_task_authorization import authorize_research_tasks
    from research_verification_capability import EvidenceVerificationRequest
    from evaluation.scifact.run_scifact_evaluation import _runtime_evidence

    claim_id = runtime["claim_id"]
    claim_text = runtime["claim"]
    limit = config.top_k
    counted_retrieval = _CountingRetrieval(provider)
    # Determine the actual retrieved rank/source bindings before constructing
    # the typed Phase 6 request. This call is exposed as extra System C cost.
    preflight = counted_retrieval.retrieve(claim_text, limit=limit)
    preflight_refs, _ = _runtime_evidence(preflight.evidence)
    retrieval_status = (OutputState.VALID_NONEMPTY.value if preflight_refs
                        else OutputState.VALID_EMPTY.value)
    state = ResearchState(user_request=claim_text)
    versions = {}
    for rank, item in enumerate(preflight.evidence, start=1):
        version = register_rag_document_version(
            state,
            source_id=item.source_id,
            document_id=item.document_id or item.source_id,
            filename=item.filename or item.source_id,
            version_id=f"phase10c-{claim_id}-{rank}",
        )
        versions[rank] = version

    question = ResearchPlanQuestion(claim_text, question_id=f"phase10-question-{claim_id}")
    requirement = EvidenceRequirement(
        question.question_id, "Retrieve and verify evidence relevant to this SciFact claim.",
        requirement_id=f"phase10-req-{claim_id}",
    )
    task = ResearchTask(
        "Retrieve evidence and verify the supplied claim against the local corpus.",
        question.question_id, evidence_requirement_ids=[requirement.requirement_id],
        task_id=f"phase10-task-{claim_id}",
    )
    plan = ResearchPlan(
        request=ResearchRequest(claim_text, "Verify this claim using the local SciFact corpus.", domain="SciFact"),
        objective=ResearchObjective("Retrieve and verify one claim with bounded research actions."),
        questions=[question], evidence_requirements=[requirement], tasks=[task],
        plan_id=f"phase10-plan-{claim_id}",
    )
    review = PlanReview(plan.plan_id, plan.revision, ReviewStatus.APPROVED,
                        reviewer="Phase 10 smoke researcher")
    run = authorize_research_run(plan, review, state)
    authorize_research_tasks(plan, run, [task.task_id], review, state)
    state.create_iteration(run.run_id)
    state.add_research_question(ResearchQuestion(
        text=claim_text, question_id=question.question_id, run_id=run.run_id,
    ))
    loop = create_research_loop(plan, run, review, state, max_iterations=4)
    start_research_loop(state, loop.loop_id)
    transition_research_loop(state, loop.loop_id, LoopStage.RESEARCH,
                             reason="Researcher approved bounded evidence retrieval.")
    begin_loop_iteration(state, loop.loop_id, (task.task_id,))

    execution_policy = BoundedExecutionPolicy(
        maximum_iterations=4, maximum_capability_invocations=2, maximum_task_executions=2,
        allowed_capabilities=(CapabilityType.LOCAL_RETRIEVAL, CapabilityType.EVIDENCE_VERIFICATION),
        allowed_stages=(LoopStage.RESEARCH, LoopStage.EVIDENCE_REVIEW, LoopStage.VERIFICATION),
        allowed_task_ids=(task.task_id,), allow_continuation=True,
    )
    controller_policy = AutonomousControllerPolicy(
        maximum_cycles=1, maximum_executions=1, maximum_capability_invocations=1,
        maximum_retrieval_actions=1, maximum_verification_actions=1,
        maximum_total_research_actions=1, require_researcher_review=True,
    )
    decision_provider = decision_provider or _Deterministic8AProvider(
        claim_text, evidence_limit=config.verification_evidence_limit)
    # Phase 6's no-retry capability expects machine-readable pair assessments.
    # Constrained JSON keeps the actual local judge inside that existing
    # contract without changing Experiment B or the frozen verifier.
    judge = _CountingJudge(llm_provider or OllamaCPUProvider(
        model=config.model, response_format="json"))
    controller_results = []

    def request_factory(decision, validation, context):
        proposal = decision.proposal
        if proposal.action_type.value == "RETRIEVE_EVIDENCE":
            capability_request = LocalRetrievalRequest(
                proposal.query or claim_text, limit=limit,
                question_id=question.question_id, evidence_document_versions=versions,
            )
        elif proposal.action_type.value == "VERIFY_CLAIM":
            target = next(item for item in state.claims if item.claim_id == proposal.claim_ids[0])
            capability_request = EvidenceVerificationRequest(target.text, tuple(proposal.evidence_ids))
        else:
            raise RuntimeError(f"Unexpected action requested by Phase 8A: {proposal.action_type.value}")
        bindings = provider_bindings_for_action(
            proposal.action_type, retrieval_provider=counted_retrieval, llm_provider=judge,
        )
        return AutonomousExecutionRequest(
            decision, validation, run.run_id, loop.loop_id, context.iteration_id,
            plan.plan_id, plan.revision, execution_policy_digest(execution_policy),
            capability_request=capability_request, provider_refs=bindings,
        )

    def approval(request):
        # Explicitly approve only the scoped, already validated local action.
        return AutonomousExecutionApproval.for_request(request, reviewer="Phase 10 smoke policy")

    def invoke_controller():
        current = next(item for item in state.research_loops if item.loop_id == loop.loop_id)
        result = run_autonomous_controller(AutonomousControllerRequest(
            plan=plan, run=run, review=review, loop=current, state=state,
            execution_policy=execution_policy, policy=controller_policy,
            decision_provider=decision_provider, request_factory=request_factory,
            approval_provider=approval, retrieval_provider=counted_retrieval, llm_provider=judge,
        ))
        controller_results.append(result)
        cycle = result.trace.cycles[-1] if result.trace.cycles else None
        if cycle is None or cycle.execution is None or cycle.execution.status.value != "COMPLETED":
            raise RuntimeError(f"Phase 8C execution failed: {result.to_dict()}")
        return result

    try:
        retrieval_result = invoke_controller()
    except Exception as exc:
        partial = SystemOutput(
            retrieved_chunks=preflight_refs, retrieval_calls=counted_retrieval.calls,
            verification_calls=0, llm_calls=judge.calls,
            controller_cycles=sum(item.cycles_completed for item in controller_results),
            retrieval_output_status=retrieval_status,
            evidence_output_status=OutputState.OUTPUT_UNAVAILABLE.value,
            raw_output={"controller_traces": [item.to_dict() for item in controller_results],
                        "research_state": state.to_dict()},
            error=f"{type(exc).__name__}: {exc}", error_stage="phase8c_controller",
        )
        raise SystemExecutionError(f"Phase 8C execution failed: {type(exc).__name__}: {exc}", partial) from exc
    retrieval_cycle = retrieval_result.trace.cycles[-1]
    evidence_ids = list(retrieval_cycle.execution.dispatch.evidence_ids)
    if not evidence_ids:
        refs, _ = _runtime_evidence(counted_retrieval.responses[-1].evidence)
        return SystemOutput(
            predicted_label=INSUFFICIENT_EVIDENCE, retrieved_chunks=refs,
            selected_evidence_doc_ids=[], retrieval_calls=counted_retrieval.calls,
            verification_calls=0, llm_calls=0, controller_cycles=len(controller_results),
            retrieval_output_status=retrieval_status,
            evidence_output_status=OutputState.VALID_EMPTY.value,
            raw_output={"controller_traces": [item.to_dict() for item in controller_results],
                        "research_state": state.to_dict(), "note": "No evidence selected for verification."},
        )

    state.add_claim(ResearchClaim(
        text=claim_text, status="NOT_VALIDATED", evidence_ids=evidence_ids,
        source_ids=list(dict.fromkeys(item.source_id for item in state.evidence
                                      if item.evidence_id in evidence_ids)),
        verification_reason="System C claim registered for Phase 6 verification.",
    ))
    execution_id = retrieval_cycle.execution.trace.controlled_execution_id
    apply_research_continuation(
        plan, run, review, state.research_loops[0], state,
        action=ResearchContinuationAction.CONTINUE_SAME_TASK,
        previous_execution_id=execution_id,
    )
    transition_research_loop(
        state, loop.loop_id, LoopStage.EVIDENCE_REVIEW,
        reason="Researcher explicitly authorized claim verification after retrieval.",
        authority=TransitionAuthority.RESEARCHER,
    )
    try:
        verification_result = invoke_controller()
    except Exception as exc:
        partial = SystemOutput(
            predicted_label=None, retrieved_chunks=preflight_refs,
            selected_evidence_doc_ids=[], assessments=[],
            retrieval_calls=counted_retrieval.calls, verification_calls=1,
            llm_calls=judge.calls,
            controller_cycles=sum(item.cycles_completed for item in controller_results),
            retrieval_output_status=retrieval_status,
            evidence_output_status=OutputState.OUTPUT_UNAVAILABLE.value,
            raw_output={"controller_traces": [item.to_dict() for item in controller_results],
                        "research_state": state.to_dict()},
            error=f"{type(exc).__name__}: {exc}", error_stage="phase8c_verification",
        )
        raise SystemExecutionError(f"Phase 8C execution failed: {type(exc).__name__}: {exc}", partial) from exc

    verification_artifacts = [item for item in state.research_artifacts
                              if item.artifact_type == "verification_result"
                              and item.verification_claim == claim_text]
    verdict = verification_artifacts[-1].verification_verdict if verification_artifacts else None
    assessments = ([item.to_dict() for item in verification_artifacts[-1].verification_assessments]
                   if verification_artifacts else [])
    evidence_by_id = {item.evidence_id: item for item in state.evidence}
    selected_ids = {
        assessment.get("evidence_id")
        for unit in assessments
        for assessment in unit.get("evidence_assessments", [])
        if assessment.get("stance") in {"SUPPORTS", "CONTRADICTS"}
    }
    selected_docs = []
    for evidence_id in selected_ids:
        evidence = evidence_by_id.get(evidence_id)
        if evidence and evidence.document_version_id:
            version = next((item for item in state.document_versions
                            if item.version_id == evidence.document_version_id), None)
            if version:
                try:
                    selected_docs.append(int(version.document_id))
                except (TypeError, ValueError):
                    pass
    refs, _ = _runtime_evidence(counted_retrieval.responses[-1].evidence)
    return SystemOutput(
        predicted_label=verdict,
        retrieved_chunks=refs,
        selected_evidence_doc_ids=list(dict.fromkeys(selected_docs)),
        assessments=assessments,
        retrieval_calls=counted_retrieval.calls,
        verification_calls=1,
        llm_calls=judge.calls,
        controller_cycles=sum(item.cycles_completed for item in controller_results),
        raw_output={
            "controller_traces": [item.to_dict() for item in controller_results],
            "controller_decisions": decision_provider.actions,
            "research_state": state.to_dict(),
            "final_stop_reason": verification_result.stop_reason.value,
            "phase8c_actions": decision_provider.actions,
        },
        retrieval_output_status=retrieval_status,
        evidence_output_status=(OutputState.VALID_NONEMPTY.value if selected_docs
                                else OutputState.VALID_EMPTY.value),
    )


def make_system_c(*, llm_provider: Any = None, decision_provider_factory: Any = None):
    """Create a C binding; injectable providers support deterministic tests.

    Production smoke uses the default deterministic 8A action binding and real
    local Ollama verifier. Injected providers are reserved for unit/integration
    tests; the actual runner does not pass fixtures here.
    """
    def bound(runtime: dict[str, Any], provider: Any, config: Any) -> SystemOutput:
        decision_provider = (decision_provider_factory(runtime)
                             if decision_provider_factory else None)
        return run_system_c(runtime, provider, config,
                            llm_provider=llm_provider,
                            decision_provider=decision_provider)
    return bound
