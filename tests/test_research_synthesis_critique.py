from __future__ import annotations

import sys
import unittest
from pathlib import Path
from dataclasses import replace

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
for path in (str(ROOT), str(TESTS)):
    if path not in sys.path:
        sys.path.insert(0, path)

from test_research_verification_capability import QueueLLM, _execute, _fixture, _stance
from research_loop import (
    LoopStage,
    TransitionAuthority,
    begin_loop_iteration,
    create_research_loop,
    start_research_loop,
    transition_research_loop,
)
from research_synthesis_critique import (
    CritiqueIssueType,
    FindingClassification,
    QuestionCoverageStatus,
    ResearchCritique,
    ResearchSynthesis,
    critique_research_synthesis,
    synthesize_research_state,
)


def _analysis_fixture(domain="domain-neutral"):
    f = _fixture(domain)
    llm = QueueLLM(_stance("SUPPORTS"))
    _execute(f, llm)
    f["llm"] = llm
    return _with_active_analysis_loop(f)


def _with_active_analysis_loop(f):
    loop = create_research_loop(f["plan"], f["run"], f["review"], f["state"], max_iterations=2)
    start_research_loop(f["state"], loop.loop_id)
    transition_research_loop(f["state"], loop.loop_id, LoopStage.RESEARCH,
                             reason="Caller enters the research stage.")
    record = begin_loop_iteration(
        f["state"], loop.loop_id,
        (f["retrieval_task"].task_id, f["task"].task_id),
    )
    for stage in (LoopStage.EVIDENCE_REVIEW, LoopStage.VERIFICATION, LoopStage.SYNTHESIS):
        transition_research_loop(f["state"], loop.loop_id, stage,
                                 reason=f"Caller enters {stage.value}.")
    f.update({"loop": f["state"].research_loops[-1], "loop_iteration": record})
    return f


def _synthesize(f, artifact_ids=None):
    return synthesize_research_state(
        f["plan"], f["run"], f["review"], f["loop"], f["state"],
        iteration_id=f["loop_iteration"].iteration_id, artifact_ids=artifact_ids,
    )


def _critique(f, synthesis):
    transition_research_loop(f["state"], f["loop"].loop_id, LoopStage.CRITIQUE,
                             reason="Caller requests a critique.")
    f["loop"] = f["state"].research_loops[-1]
    return critique_research_synthesis(
        f["plan"], f["run"], f["review"], f["loop"], f["state"], synthesis,
        iteration_id=f["loop_iteration"].iteration_id,
    )


class ResearchSynthesisCritiqueTests(unittest.TestCase):
    def test_synthesis_preserves_stored_verdict_and_full_evidence_provenance(self):
        f = _analysis_fixture("medicine")
        before = f["state"].to_dict()
        llm_calls = len(f["llm"].calls)
        result = _synthesize(f)
        self.assertEqual(len(f["llm"].calls), llm_calls)
        self.assertEqual(len(result.findings), 1)
        finding = result.findings[0]
        artifact = f["state"].research_artifacts[-1]
        self.assertEqual(finding.classification.value, artifact.verification_verdict)
        self.assertEqual(finding.statement, artifact.verification_claim)
        self.assertEqual(finding.evidence_ids, artifact.evidence_ids)
        self.assertEqual(finding.verification_artifact_ids, (artifact.artifact_id,))
        ref = result.evidence_references[0]
        self.assertEqual(ref.evidence_id, f["evidence"].evidence_id)
        self.assertEqual(ref.source_id, f["evidence"].source_id)
        self.assertEqual(ref.search_result_id, f["evidence"].search_result_id)
        self.assertIsNotNone(ref.document_version_id)
        self.assertIsNotNone(ref.passage_reference_id)
        self.assertIsNotNone(ref.search_action_id)
        self.assertTrue(ref.text_hash)
        self.assertEqual(before, f["state"].to_dict())
        self.assertEqual(result.run_id, f["run"].run_id)
        self.assertEqual(result.plan_revision, f["plan"].revision)

    def test_question_coverage_does_not_treat_task_completion_as_evidence(self):
        f = _analysis_fixture("agriculture")
        result = _synthesize(f)
        self.assertEqual(len(result.question_coverage), 1)
        self.assertEqual(result.question_coverage[0].status, QuestionCoverageStatus.PARTIALLY_ADDRESSED)

    def test_insufficient_verdict_is_preserved_and_critique_reports_it(self):
        f = _fixture("physics")
        _execute(f, QueueLLM(_stance("NEUTRAL")))
        loop = create_research_loop(f["plan"], f["run"], f["review"], f["state"], max_iterations=2)
        start_research_loop(f["state"], loop.loop_id)
        transition_research_loop(f["state"], loop.loop_id, LoopStage.RESEARCH, reason="start")
        iteration = begin_loop_iteration(f["state"], loop.loop_id,
                                         (f["retrieval_task"].task_id, f["task"].task_id))
        for stage in (LoopStage.EVIDENCE_REVIEW, LoopStage.VERIFICATION, LoopStage.SYNTHESIS):
            transition_research_loop(f["state"], loop.loop_id, stage, reason="caller transition")
        f.update(loop=f["state"].research_loops[-1], loop_iteration=iteration)
        synthesis = _synthesize(f)
        self.assertEqual(synthesis.findings[0].classification, FindingClassification.INSUFFICIENT_EVIDENCE)
        critique = _critique(f, synthesis)
        self.assertIn(CritiqueIssueType.INSUFFICIENT_EVIDENCE,
                      {issue.issue_type for issue in critique.issues})
        self.assertNotEqual(synthesis.findings[0].classification, FindingClassification.SUPPORTED)

    def test_conflicting_stored_evidence_is_preserved_as_unresolved(self):
        from research_retrieval import RetrievalEvidence, adapt_retrieval_evidence_to_state
        from research_state import SearchAction
        from research_tools import SearchResult

        f = _fixture("economics")
        existing = f["evidence"]
        added = RetrievalEvidence(
            evidence_id="chunk-economics-conflict", text="System X does not use BM25.",
            source_id=existing.source_id, filename="synthetic-document.pdf", page=3,
            document_id="document-economics", chunk_id="chunk-economics-conflict",
        )
        search = f["state"].add_search(SearchAction(
            query="synthetic contradiction", provider="test", result_count=1,
            run_id=f["run"].run_id,
        ))
        result = f["state"].add_search_result(SearchResult(
            source_id=existing.source_id, title="", content=added.text, source_type="unknown",
            search_id=search.search_id, rank=1,
        ))
        adapt_retrieval_evidence_to_state(
            added, f["state"].document_versions[0], f["state"], search_result_id=result.result_id,
        )
        llm = QueueLLM(_stance("SUPPORTS", "E1"), _stance("CONTRADICTS", "E2"))
        _execute(f, llm, (existing.evidence_id, f["state"].evidence[-1].evidence_id))
        f["llm"] = llm
        _with_active_analysis_loop(f)
        synthesis = _synthesize(f)
        self.assertEqual(len(synthesis.evidence_references), 2)
        conflicts = [item for item in synthesis.findings
                     if item.classification is FindingClassification.UNRESOLVED_CONTRADICTION]
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(len(conflicts[0].evidence_ids), 2)
        critique = _critique(f, synthesis)
        self.assertIn(CritiqueIssueType.CONTRADICTORY_EVIDENCE,
                      {issue.issue_type for issue in critique.issues})

    def test_unverified_claim_is_reported_without_becoming_false(self):
        from research_state import ResearchClaim

        f = _analysis_fixture("psychology")
        claim = f["state"].add_claim(ResearchClaim(
            text="The intervention may affect outcome Y.", status="SUPPORTED",
            evidence_ids=[f["evidence"].evidence_id], source_ids=[f["evidence"].source_id],
        ))
        synthesis = _synthesize(f)
        item = next(item for item in synthesis.findings if claim.claim_id in item.claim_ids)
        self.assertEqual(item.classification, FindingClassification.UNVERIFIED)
        critique = _critique(f, synthesis)
        self.assertIn(CritiqueIssueType.UNVERIFIED_CLAIM, {issue.issue_type for issue in critique.issues})
        self.assertIn("does not establish", next(issue.description for issue in critique.issues
                                                  if issue.issue_type is CritiqueIssueType.UNVERIFIED_CLAIM).lower())

    def test_serialization_round_trip_for_synthesis_and_critique(self):
        f = _analysis_fixture("climate")
        synthesis = _synthesize(f)
        self.assertEqual(ResearchSynthesis.from_json(synthesis.to_json()), synthesis)
        critique = _critique(f, synthesis)
        self.assertEqual(ResearchCritique.from_json(critique.to_json()), critique)
        malformed = synthesis.to_dict()
        malformed["surprise"] = True
        with self.assertRaises(ValueError):
            ResearchSynthesis.from_dict(malformed)

    def test_only_explicit_same_run_artifacts_are_synthesized(self):
        f = _analysis_fixture("cybersecurity")
        artifact = next(item for item in f["state"].research_artifacts
                        if item.artifact_type == "verification_result")
        selected = _synthesize(f, [artifact.artifact_id])
        self.assertEqual(selected.source_artifact_ids, (artifact.artifact_id,))
        with self.assertRaisesRegex(ValueError, "Unknown ResearchArtifact"):
            _synthesize(f, ["made-up-artifact"])

    def test_wrong_loop_stage_and_cross_revision_inputs_are_rejected_without_mutation(self):
        f = _analysis_fixture("biology")
        transition_research_loop(f["state"], f["loop"].loop_id, LoopStage.CRITIQUE,
                                 reason="Test wrong analysis stage.")
        f["loop"] = f["state"].research_loops[-1]
        before = f["state"].to_dict()
        with self.assertRaisesRegex(ValueError, "requires a RUNNING loop at SYNTHESIS"):
            _synthesize(f)
        original_revision = f["plan"].revision
        f["plan"].revision += 1
        with self.assertRaises(ValueError):
            _synthesize(f)
        f["plan"].revision = original_revision
        self.assertEqual(before, f["state"].to_dict())

    def test_critique_rejects_fabricated_finding_verdict(self):
        f = _analysis_fixture("computer-vision")
        synthesis = _synthesize(f)
        fabricated = replace(synthesis, findings=(replace(
            synthesis.findings[0], classification=FindingClassification.CONTRADICTED),))
        with self.assertRaisesRegex(ValueError, "findings do not match stored"):
            _critique(f, fabricated)

    def test_no_claim_of_novelty_or_automatic_follow_up(self):
        f = _analysis_fixture("chemistry")
        synthesis = _synthesize(f)
        critique = _critique(f, synthesis)
        self.assertEqual(critique.novelty_status, "NOT_ASSESSED")
        self.assertEqual(len(f["state"].research_artifacts), 2)
        self.assertEqual(len(f["state"].evidence), 1)
        self.assertEqual(len(f["llm"].calls), 1)
        self.assertEqual(f["state"].research_loops[-1].current_stage, LoopStage.CRITIQUE)


if __name__ == "__main__":
    unittest.main()
