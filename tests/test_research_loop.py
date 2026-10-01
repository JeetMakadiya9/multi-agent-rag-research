from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_capabilities import CapabilityType
from research_loop import (
    CapabilityInvocationRequest,
    CapabilityInvocationResult,
    CapabilityInvocationStatus,
    LEGAL_TRANSITIONS,
    LoopIterationStatus,
    LoopStage,
    ResearchLoop,
    ResearchLoopStatus,
    StopReason,
    TransitionAuthority,
    UnresolvedWork,
    UnresolvedWorkKind,
    add_unresolved_work,
    begin_loop_iteration,
    clear_unresolved_work,
    complete_loop_task,
    create_research_loop,
    fail_research_loop,
    finish_loop_iteration,
    record_capability_result,
    request_capability_invocation,
    select_loop_task,
    start_research_loop,
    stop_research_loop,
    transition_research_loop,
)
from research_plan_review import PlanReview, ReviewStatus
from research_planning import ResearchObjective, ResearchPlan, ResearchPlanQuestion, ResearchRequest, ResearchTask
from research_run_authorization import authorize_research_run
from research_state import ResearchState
from research_task_authorization import authorize_research_tasks
from research_task_execution import TaskExecutionError, TaskExecutionResult, execute_authorized_task


def _fixture(*, authorized=True, max_iterations=3, domain="general"):
    question = ResearchPlanQuestion("What does the available evidence establish?", question_id=f"q-{domain}")
    task_a = ResearchTask("Retrieve existing material.", question.question_id, task_id=f"t1-{domain}")
    task_b = ResearchTask("Verify an explicit claim.", question.question_id, task_id=f"t2-{domain}")
    plan = ResearchPlan(
        request=ResearchRequest("Study a research problem.", "Record supported findings.", domain=domain),
        objective=ResearchObjective("Collect and assess relevant information."),
        questions=[question], tasks=[task_a, task_b], plan_id=f"plan-{domain}",
    )
    review = PlanReview(plan.plan_id, plan.revision, ReviewStatus.APPROVED, reviewer="researcher")
    state = ResearchState(user_request="Synthetic loop contract fixture")
    run = authorize_research_run(plan, review, state)
    if authorized:
        authorize_research_tasks(plan, run, [task_a.task_id, task_b.task_id], review, state)
    loop = create_research_loop(plan, run, review, state, max_iterations=max_iterations)
    return {"plan": plan, "review": review, "run": run, "state": state,
            "tasks": [task_a, task_b], "loop": loop}


class ResearchLoopTests(unittest.TestCase):
    def test_creation_is_typed_approved_and_separate_from_loop_status(self):
        f = _fixture()
        loop = f["loop"]
        self.assertEqual(loop.current_stage, LoopStage.INTAKE)
        self.assertEqual(loop.status, ResearchLoopStatus.CREATED)
        self.assertEqual(loop.plan_id, f["plan"].plan_id)
        self.assertEqual(loop.plan_revision, f["plan"].revision)
        self.assertEqual(loop.unresolved_task_ids, tuple(f["run"].authorized_task_ids))
        self.assertIs(f["state"].research_loops[0], loop)
        self.assertEqual(f["state"].validate_lineage(), [])

    def test_invalid_plan_run_review_and_revision_are_rejected(self):
        f = _fixture()
        state = ResearchState(user_request="Invalid loop")
        with self.assertRaises(ValueError):
            create_research_loop(None, f["run"], f["review"], state, max_iterations=2)
        with self.assertRaises(ValueError):
            create_research_loop(f["plan"], f["run"], f["review"], state, max_iterations=2)
        f["plan"].revision += 1
        with self.assertRaises(ValueError):
            create_research_loop(f["plan"], f["run"], f["review"], f["state"], max_iterations=2)

    def test_start_and_legal_transition_and_audit_event(self):
        f = _fixture()
        loop = start_research_loop(f["state"], f["loop"].loop_id)
        self.assertEqual(loop.current_stage, LoopStage.PLANNING)
        self.assertEqual(loop.status, ResearchLoopStatus.RUNNING)
        self.assertEqual(loop.transitions[-1].from_stage, LoopStage.INTAKE)
        self.assertEqual(f["state"].events[-1].action, "stage_transition")

    def test_illegal_transition_is_rejected_without_changing_loop(self):
        f = _fixture()
        loop = start_research_loop(f["state"], f["loop"].loop_id)
        prior = loop
        with self.assertRaisesRegex(ValueError, "Illegal research-loop transition"):
            transition_research_loop(f["state"], loop.loop_id, LoopStage.SYNTHESIS,
                                     reason="cannot skip the contract")
        self.assertIs(f["state"].research_loops[0], prior)
        self.assertEqual(f["state"].events[-1].action, "transition_rejected")

    def test_iteration_numbering_reuses_state_iteration_ids_and_is_bounded(self):
        f = _fixture(max_iterations=2)
        start_research_loop(f["state"], f["loop"].loop_id)
        transition_research_loop(f["state"], f["loop"].loop_id, LoopStage.RESEARCH,
                                 reason="Begin authorized retrieval work.")
        first = begin_loop_iteration(f["state"], f["loop"].loop_id, (f["tasks"][0].task_id,))
        self.assertEqual(first.iteration_number, 1)
        self.assertEqual(first.status, LoopIterationStatus.RUNNING)
        self.assertEqual(first.iteration_id, f["state"].iterations[-1].iteration_id)
        with self.assertRaisesRegex(ValueError, "already active"):
            begin_loop_iteration(f["state"], f["loop"].loop_id, (f["tasks"][1].task_id,))
        finish_loop_iteration(f["state"], f["loop"].loop_id,
                              status=LoopIterationStatus.COMPLETED, reason="Synthetic iteration done.")
        second = begin_loop_iteration(f["state"], f["loop"].loop_id, (f["tasks"][1].task_id,))
        self.assertEqual(second.iteration_number, 2)
        finish_loop_iteration(f["state"], f["loop"].loop_id,
                              status=LoopIterationStatus.COMPLETED, reason="Bound reached.")
        loop = f["state"].research_loops[0]
        self.assertEqual(loop.status, ResearchLoopStatus.STOPPED)
        self.assertEqual(loop.stop_reason, StopReason.MAXIMUM_ITERATIONS_REACHED)
        self.assertEqual(len(loop.loop_iterations), loop.max_iterations)

    def test_unauthorized_task_cannot_enter_iteration_or_capability_request(self):
        f = _fixture()
        start_research_loop(f["state"], f["loop"].loop_id)
        transition_research_loop(f["state"], f["loop"].loop_id, LoopStage.RESEARCH, reason="Research stage.")
        with self.assertRaisesRegex(ValueError, "not authorized"):
            begin_loop_iteration(f["state"], f["loop"].loop_id, ("unapproved-task",))

    def test_no_authorized_work_stops_instead_of_progressing(self):
        f = _fixture(authorized=False, domain="empty")
        loop = start_research_loop(f["state"], f["loop"].loop_id)
        self.assertEqual(loop.current_stage, LoopStage.STOPPED)
        self.assertEqual(loop.status, ResearchLoopStatus.STOPPED)
        self.assertEqual(loop.stop_reason, StopReason.NO_AUTHORIZED_WORK)

    def test_researcher_gate_waits_and_requires_explicit_researcher_transition(self):
        f = _fixture()
        start_research_loop(f["state"], f["loop"].loop_id)
        loop = transition_research_loop(f["state"], f["loop"].loop_id, LoopStage.RESEARCHER_REVIEW,
                                        reason="A researcher decision is needed.")
        self.assertEqual(loop.status, ResearchLoopStatus.WAITING_FOR_RESEARCHER)
        self.assertEqual(loop.stop_reason, StopReason.WAITING_FOR_RESEARCHER)
        with self.assertRaisesRegex(ValueError, "researcher authority"):
            transition_research_loop(f["state"], loop.loop_id, LoopStage.RESEARCH,
                                     reason="No implicit decision.")
        loop = transition_research_loop(f["state"], loop.loop_id, LoopStage.RESEARCH,
                                        reason="Researcher explicitly authorized continuation.",
                                        authority=TransitionAuthority.RESEARCHER)
        self.assertEqual(loop.status, ResearchLoopStatus.RUNNING)

    def test_researcher_can_complete_or_stop_after_review(self):
        f = _fixture(domain="complete")
        start_research_loop(f["state"], f["loop"].loop_id)
        transition_research_loop(f["state"], f["loop"].loop_id, LoopStage.RESEARCHER_REVIEW,
                                 reason="Review gate.")
        loop = transition_research_loop(f["state"], f["loop"].loop_id, LoopStage.COMPLETED,
                                        reason="Researcher closed the workflow.",
                                        authority=TransitionAuthority.RESEARCHER)
        self.assertEqual(loop.status, ResearchLoopStatus.COMPLETED)
        self.assertEqual(loop.stop_reason, StopReason.COMPLETED)

        other = _fixture(domain="stop")
        start_research_loop(other["state"], other["loop"].loop_id)
        transition_research_loop(other["state"], other["loop"].loop_id, LoopStage.RESEARCHER_REVIEW,
                                 reason="Review gate.")
        loop = stop_research_loop(other["state"], other["loop"].loop_id,
                                  StopReason.RESEARCHER_REQUESTED_STOP,
                                  authority=TransitionAuthority.RESEARCHER, message="Stop requested.")
        self.assertEqual(loop.status, ResearchLoopStatus.STOPPED)
        self.assertEqual(loop.stop_reason, StopReason.RESEARCHER_REQUESTED_STOP)

    def test_stop_reason_is_typed_and_failure_is_explicit(self):
        f = _fixture(domain="failure")
        loop = fail_research_loop(f["state"], f["loop"].loop_id, "Synthetic orchestration error.")
        self.assertEqual(loop.current_stage, LoopStage.FAILED)
        self.assertEqual(loop.status, ResearchLoopStatus.FAILED)
        self.assertEqual(loop.stop_reason, StopReason.FAILED)
        with self.assertRaises(ValueError):
            stop_research_loop(_fixture()["state"], _fixture()["loop"].loop_id, "not-a-reason")

    def test_blocking_unresolved_work_prevents_progress_until_cleared(self):
        f = _fixture()
        start_research_loop(f["state"], f["loop"].loop_id)
        blocker = UnresolvedWork(UnresolvedWorkKind.MISSING_EVIDENCE, "Required evidence is missing.",
                                 task_id=f["tasks"][0].task_id)
        add_unresolved_work(f["state"], f["loop"].loop_id, blocker)
        with self.assertRaisesRegex(ValueError, "Blocking unresolved"):
            transition_research_loop(f["state"], f["loop"].loop_id, LoopStage.RESEARCH,
                                     reason="Cannot progress.")
        clear_unresolved_work(f["state"], f["loop"].loop_id, blocker)
        self.assertEqual(f["state"].research_loops[0].unresolved_work, ())

    def test_blockers_prevent_forward_transition_at_any_stage_but_allow_review_gate(self):
        f = _fixture(domain="stage-blocker")
        start_research_loop(f["state"], f["loop"].loop_id)
        blocker = UnresolvedWork(UnresolvedWorkKind.MISSING_EVIDENCE, "Evidence is missing.")
        add_unresolved_work(f["state"], f["loop"].loop_id, blocker)
        with self.assertRaisesRegex(ValueError, "Blocking unresolved work"):
            transition_research_loop(f["state"], f["loop"].loop_id, LoopStage.RESEARCH,
                                     reason="Cannot advance with a blocker.")
        loop = transition_research_loop(f["state"], f["loop"].loop_id, LoopStage.RESEARCHER_REVIEW,
                                        reason="Escalate the blocker for researcher review.")
        self.assertEqual(loop.status, ResearchLoopStatus.WAITING_FOR_RESEARCHER)

    def test_iteration_cannot_repeat_a_task_already_marked_complete(self):
        f = _fixture(domain="completed-task")
        start_research_loop(f["state"], f["loop"].loop_id)
        transition_research_loop(f["state"], f["loop"].loop_id, LoopStage.RESEARCH, reason="Research.")
        task_id = f["tasks"][0].task_id
        begin_loop_iteration(f["state"], f["loop"].loop_id, (task_id,))
        iteration_id = f["state"].research_loops[0].current_iteration_id
        execution = execute_authorized_task(
            f["plan"], f["run"], f["tasks"][0], f["review"],
            lambda _task, _context: TaskExecutionResult(success=True, output="done"),
            f["state"], iteration_id=iteration_id,
        )
        complete_loop_task(f["state"], f["loop"].loop_id, task_id, execution.execution_id)
        finish_loop_iteration(f["state"], f["loop"].loop_id,
                              status=LoopIterationStatus.COMPLETED, reason="Task recorded.")
        with self.assertRaisesRegex(ValueError, "only tasks still unresolved"):
            begin_loop_iteration(f["state"], f["loop"].loop_id, (task_id,))

    def test_unresolved_blocker_stop_reason_requires_a_real_blocker(self):
        f = _fixture(domain="bogus-blocker-stop")
        with self.assertRaisesRegex(ValueError, "requires at least one blocking"):
            stop_research_loop(f["state"], f["loop"].loop_id, StopReason.UNRESOLVED_BLOCKER)

    def test_capability_boundary_is_typed_explicit_and_does_not_auto_invoke(self):
        f = _fixture()
        start_research_loop(f["state"], f["loop"].loop_id)
        transition_research_loop(f["state"], f["loop"].loop_id, LoopStage.RESEARCH, reason="Research stage.")
        begin_loop_iteration(f["state"], f["loop"].loop_id, (f["tasks"][0].task_id,))
        before_executions = len(f["state"].task_executions)
        before_artifacts = len(f["state"].research_artifacts)
        request = request_capability_invocation(f["state"], f["loop"].loop_id,
                                                CapabilityType.LOCAL_RETRIEVAL, f["tasks"][0].task_id)
        self.assertIsInstance(request, CapabilityInvocationRequest)
        self.assertEqual(request.run_id, f["run"].run_id)
        self.assertEqual(len(f["state"].task_executions), before_executions)
        self.assertEqual(len(f["state"].research_artifacts), before_artifacts)
        self.assertEqual(f["state"].research_loops[0].capability_requests, (request,))
        with self.assertRaisesRegex(ValueError, "not available"):
            request_capability_invocation(f["state"], f["loop"].loop_id,
                                          CapabilityType.EXPERIMENT_PLANNING, f["tasks"][0].task_id)
        result = CapabilityInvocationResult(CapabilityType.LOCAL_RETRIEVAL, request.invocation_id,
                                           "execution-stub", CapabilityInvocationStatus.COMPLETED,
                                           artifact_ids=("artifact-stub",))
        self.assertEqual(result.status, CapabilityInvocationStatus.COMPLETED)

    def test_multiple_authorized_task_ids_and_task_completion_require_phase5_record(self):
        f = _fixture()
        start_research_loop(f["state"], f["loop"].loop_id)
        transition_research_loop(f["state"], f["loop"].loop_id, LoopStage.RESEARCH, reason="Research stage.")
        with self.assertRaisesRegex(ValueError, "completed Phase 5 execution"):
            complete_loop_task(f["state"], f["loop"].loop_id, f["tasks"][0].task_id, "missing-execution")
        record = begin_loop_iteration(f["state"], f["loop"].loop_id,
                                      (f["tasks"][0].task_id, f["tasks"][1].task_id))
        self.assertEqual(record.task_ids, (f["tasks"][0].task_id, f["tasks"][1].task_id))
        select_loop_task(f["state"], f["loop"].loop_id, f["tasks"][1].task_id)
        execution = execute_authorized_task(
            f["plan"], f["run"], f["tasks"][1], f["review"],
            lambda _task, _context: TaskExecutionResult(success=True, output="synthetic handler result"),
            f["state"], iteration_id=record.iteration_id,
        )
        loop = complete_loop_task(f["state"], f["loop"].loop_id, f["tasks"][1].task_id, execution.execution_id)
        self.assertIn(f["tasks"][1].task_id, loop.completed_task_ids)
        self.assertNotIn(f["tasks"][1].task_id, loop.unresolved_task_ids)
        self.assertEqual(loop.current_task_id, f["tasks"][0].task_id)

    def test_capability_failure_is_recorded_and_fails_loop_without_claiming_success(self):
        f = _fixture(domain="failed-capability")
        start_research_loop(f["state"], f["loop"].loop_id)
        transition_research_loop(f["state"], f["loop"].loop_id, LoopStage.RESEARCH, reason="Research stage.")
        iteration = begin_loop_iteration(f["state"], f["loop"].loop_id, (f["tasks"][0].task_id,))
        request = request_capability_invocation(f["state"], f["loop"].loop_id,
                                                CapabilityType.LOCAL_RETRIEVAL, f["tasks"][0].task_id)
        with self.assertRaises(TaskExecutionError) as caught:
            execute_authorized_task(
                f["plan"], f["run"], f["tasks"][0], f["review"],
                lambda _task, _context: (_ for _ in ()).throw(RuntimeError("synthetic provider failure")),
                f["state"], iteration_id=iteration.iteration_id,
            )
        result = CapabilityInvocationResult(CapabilityType.LOCAL_RETRIEVAL, request.invocation_id,
                                            caught.exception.execution.execution_id,
                                            CapabilityInvocationStatus.FAILED)
        loop = record_capability_result(f["state"], f["loop"].loop_id, result)
        self.assertEqual(loop.capability_results, (result,))
        self.assertEqual(loop.status, ResearchLoopStatus.FAILED)
        self.assertEqual(loop.stop_reason, StopReason.FAILED)
        self.assertIsNone(loop.current_iteration_id)
        self.assertEqual(f["state"].iterations[-1].status, "failed")

    def test_duplicate_loop_and_malformed_reference_are_rejected(self):
        f = _fixture()
        with self.assertRaisesRegex(ValueError, "Duplicate research_loop"):
            f["state"].add_research_loop(f["loop"])
        malformed = ResearchLoop(run_id=f["run"].run_id, plan_id="wrong-plan", plan_revision=1,
                                 max_iterations=1, loop_id="loop-wrong")
        with self.assertRaisesRegex(ValueError, "Invalid ResearchLoop references"):
            f["state"].add_research_loop(malformed)

    def test_loop_and_state_serialization_round_trip_and_old_state_compatibility(self):
        f = _fixture()
        start_research_loop(f["state"], f["loop"].loop_id)
        loop_json = f["state"].research_loops[0].to_json()
        self.assertEqual(ResearchLoop.from_json(loop_json), f["state"].research_loops[0])
        payload = json.loads(json.dumps(f["state"].to_dict()))
        restored = ResearchState.from_dict(payload)
        self.assertEqual(restored.research_loops, f["state"].research_loops)
        self.assertEqual(restored.validate_lineage(), [])
        legacy = dict(payload)
        legacy.pop("research_loops")
        self.assertEqual(ResearchState.from_dict(legacy).research_loops, [])

    def test_malformed_loop_serialization_is_rejected(self):
        f = _fixture()
        data = f["loop"].to_dict()
        data["current_stage"] = "UNKNOWN"
        with self.assertRaises(ValueError):
            ResearchLoop.from_dict(data)
        with self.assertRaises(ValueError):
            ResearchLoop.from_json("not-json")

    def test_transition_graph_has_no_terminal_escape_or_automatic_stage_edges(self):
        for terminal in (LoopStage.COMPLETED, LoopStage.STOPPED, LoopStage.FAILED):
            self.assertEqual(LEGAL_TRANSITIONS[terminal], frozenset())
        self.assertEqual(LEGAL_TRANSITIONS[LoopStage.EXPERIMENT_PLANNING],
                         frozenset({LoopStage.RESEARCHER_REVIEW, LoopStage.STOPPED, LoopStage.FAILED}))
        self.assertNotIn(LoopStage.VERIFICATION, LEGAL_TRANSITIONS[LoopStage.RESEARCH])

    def test_domain_neutral_examples_use_the_same_contract(self):
        for domain in ("AI/ML", "medicine", "agriculture", "physics", "cybersecurity"):
            with self.subTest(domain=domain):
                f = _fixture(domain=domain)
                loop = start_research_loop(f["state"], f["loop"].loop_id)
                self.assertEqual(loop.run_id, f["run"].run_id)
                self.assertEqual(f["state"].validate_lineage(), [])

    def test_research_loop_does_not_insert_provenance_or_change_research_task_status(self):
        f = _fixture()
        before = (list(f["state"].evidence), [task.status for task in f["plan"].tasks])
        start_research_loop(f["state"], f["loop"].loop_id)
        transition_research_loop(f["state"], f["loop"].loop_id, LoopStage.RESEARCH, reason="Research stage.")
        begin_loop_iteration(f["state"], f["loop"].loop_id, (f["tasks"][0].task_id,))
        after = (list(f["state"].evidence), [task.status for task in f["plan"].tasks])
        self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
