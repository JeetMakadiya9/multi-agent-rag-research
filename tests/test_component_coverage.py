from __future__ import annotations

import json
import unittest

from claim_verification import CONTRADICTED, INSUFFICIENT_EVIDENCE, SUPPORTED, EvidenceItem
from experiment_b_verification import ExperimentBConfig, run_experiment_b_verification, validate_judgement
from llm import StaticProvider
from verification_units import analyze_verification_units


def assessment_payload(rows):
    return json.dumps({"assessments": rows, "reason": "Evidence-level component assessment."})


def pair_payload(evidence_index, component_index, stance, reason="Evidence-level component assessment."):
    return json.dumps({"assessments": [{"evidence_index": evidence_index,
                                         "component_indices": [component_index], "stance": stance}],
                       "reason": reason})


class DeterministicComponentCoverageTests(unittest.TestCase):
    def setUp(self):
        self.unit = analyze_verification_units("System uses FAISS and BM25.")[0]
        self.assertEqual([x.component_id for x in self.unit.components], ["C1", "C2"])
        self.e1 = EvidenceItem("long-faiss-id", "System uses FAISS for vector retrieval.")
        self.e2 = EvidenceItem("long-bm25-id", "System uses BM25 for lexical retrieval.")

    def verdict(self, rows, evidence=None):
        return validate_judgement(self.unit, {"assessments": rows}, evidence or [self.e1, self.e2]).verdict

    def test_support_one_of_two_components_is_insufficient(self):
        result = validate_judgement(self.unit, {"assessments": [
            {"evidence_index": "E1", "component_indices": ["C1"], "stance": "SUPPORTS"}
        ]}, [self.e1])
        self.assertEqual(result.verdict, INSUFFICIENT_EVIDENCE)
        self.assertFalse(result.assessment_complete)
        self.assertEqual(result.missing_assessments, ["E1-C2"])
        self.assertEqual(len(result.supported_components), 1)
        self.assertEqual(len(result.unsupported_components), 1)

    def test_two_evidence_items_cover_both_components(self):
        self.assertEqual(self.verdict([
            {"evidence_index": "E1", "component_indices": ["C1"], "stance": "SUPPORTS"},
            {"evidence_index": "E1", "component_indices": ["C2"], "stance": "NEUTRAL"},
            {"evidence_index": "E2", "component_indices": ["C1"], "stance": "NEUTRAL"},
            {"evidence_index": "E2", "component_indices": ["C2"], "stance": "SUPPORTS"},
        ]), SUPPORTED)

    def test_one_component_contradicted_means_contradicted(self):
        self.assertEqual(self.verdict([
            {"evidence_index": "E1", "component_indices": ["C1"], "stance": "SUPPORTS"},
            {"evidence_index": "E1", "component_indices": ["C2"], "stance": "NEUTRAL"},
            {"evidence_index": "E2", "component_indices": ["C1"], "stance": "NEUTRAL"},
            {"evidence_index": "E2", "component_indices": ["C2"], "stance": "CONTRADICTS"},
        ]), CONTRADICTED)

    def test_same_component_support_and_contradiction_is_conflict(self):
        result = validate_judgement(self.unit, {"assessments": [
            {"evidence_index": "E1", "component_indices": ["C1"], "stance": "SUPPORTS"},
            {"evidence_index": "E2", "component_indices": ["C1"], "stance": "CONTRADICTS"},
            {"evidence_index": "E1", "component_indices": ["C2"], "stance": "NEUTRAL"},
            {"evidence_index": "E2", "component_indices": ["C2"], "stance": "NEUTRAL"},
        ]}, [self.e1, self.e2])
        self.assertEqual(result.verdict, INSUFFICIENT_EVIDENCE)
        self.assertTrue(result.assessment_complete)
        self.assertEqual(len(result.conflicting_components), 1)

    def test_neutral_only_is_insufficient(self):
        self.assertEqual(self.verdict([
            {"evidence_index": "E1", "component_indices": ["C1", "C2"], "stance": "NEUTRAL"}
        ], [self.e1]), INSUFFICIENT_EVIDENCE)

    def test_explicit_neutral_is_complete_and_distinct_from_a_missing_pair(self):
        unit = analyze_verification_units("System X uses BM25.")[0]
        result = validate_judgement(unit, {"assessments": [
            {"evidence_index": "E1", "component_indices": ["C1"], "stance": "SUPPORTS"},
            {"evidence_index": "E2", "component_indices": ["C1"], "stance": "NEUTRAL"},
        ]}, [self.e1, self.e2])
        self.assertTrue(result.assessment_complete)
        self.assertEqual(result.missing_assessments, [])
        self.assertEqual(result.verdict, SUPPORTED)

    def test_invalid_evidence_reference_never_supports(self):
        result = validate_judgement(self.unit, {"assessments": [
            {"evidence_index": "E404", "component_indices": ["C1", "C2"], "stance": "SUPPORTS"}
        ]}, [self.e1, self.e2])
        self.assertEqual(result.verdict, INSUFFICIENT_EVIDENCE)
        self.assertTrue(result.validation_errors)

    def test_invalid_component_reference_never_supports(self):
        result = validate_judgement(self.unit, {"assessments": [
            {"evidence_index": "E1", "component_indices": ["C404"], "stance": "SUPPORTS"}
        ]}, [self.e1])
        self.assertEqual(result.verdict, INSUFFICIENT_EVIDENCE)

    def test_three_components_require_all_three(self):
        unit = analyze_verification_units("System uses FAISS, BM25, and a cross-encoder.")[0]
        evidence = [EvidenceItem("a", "System uses FAISS."), EvidenceItem("b", "System uses BM25.")]
        rows = [{"evidence_index": f"E{i}", "component_indices": [f"C{i}"], "stance": "SUPPORTS"}
                for i in range(1, 3)]
        result = validate_judgement(unit, {"assessments": rows}, evidence)
        self.assertEqual(result.verdict, INSUFFICIENT_EVIDENCE)
        # All components have assessments; lexical anchoring also confirms all list items.
        evidence.append(EvidenceItem("c", "System uses a cross-encoder."))
        rows = [{"evidence_index": f"E{e}", "component_indices": [f"C{c}"],
                 "stance": "SUPPORTS" if e == c else "NEUTRAL"}
                for e in range(1, 4) for c in range(1, 4)]
        self.assertEqual(validate_judgement(unit, {"assessments": rows}, evidence).verdict, SUPPORTED)

    def test_case_5_omitted_e2_assessment_is_incomplete_not_neutral(self):
        unit = analyze_verification_units("System X uses BM25 for lexical retrieval.")[0]
        evidence = [
            EvidenceItem("source-a", "System X uses BM25 for lexical retrieval."),
            EvidenceItem("source-b", "System X does not use BM25; lexical retrieval is performed with TF-IDF."),
        ]
        result = validate_judgement(unit, {"assessments": [
            {"evidence_index": "E1", "component_indices": ["C1"], "stance": "SUPPORTS"},
        ]}, evidence)
        self.assertFalse(result.assessment_complete)
        self.assertEqual(result.missing_assessments, ["E2-C1"])
        self.assertEqual(result.coverage_matrix[0]["status"], "INCOMPLETE")
        self.assertEqual(result.verdict, INSUFFICIENT_EVIDENCE)

    def test_legacy_single_component_keeps_single_evidence_compatibility(self):
        unit = analyze_verification_units("System X uses BM25 for lexical retrieval.")[0]
        one_evidence = validate_judgement(unit, {
            "verdict": SUPPORTED, "supporting_evidence_ids": ["source-a"],
        }, [EvidenceItem("source-a", "System X uses BM25 for lexical retrieval.")])
        self.assertEqual(one_evidence.verdict, SUPPORTED)
        self.assertTrue(one_evidence.assessment_complete)

        two_evidence = validate_judgement(unit, {
            "verdict": SUPPORTED, "supporting_evidence_ids": ["source-a"],
        }, [
            EvidenceItem("source-a", "System X uses BM25 for lexical retrieval."),
            EvidenceItem("source-b", "System X does not use BM25; lexical retrieval is performed with TF-IDF."),
        ])
        self.assertEqual(two_evidence.verdict, INSUFFICIENT_EVIDENCE)
        self.assertFalse(two_evidence.assessment_complete)
        self.assertEqual(two_evidence.missing_assessments, ["E2-C1"])

    def test_stance_assessments_drive_engine_and_final_answer_revision(self):
        provider = StaticProvider([
            pair_payload("E1", "C1", "SUPPORTS"),
            pair_payload("E1", "C2", "NEUTRAL"),
        ])
        report = run_experiment_b_verification("Which methods?", "System uses FAISS and BM25.",
            [self.e1], llm_provider=provider,
            config=ExperimentBConfig(max_targeted_retrieval_retries=0, answer_revision_enabled=True))
        self.assertEqual(report.verifications[0].verdict, INSUFFICIENT_EVIDENCE)
        self.assertIn("Evidence supports part of the statement: System uses FAISS", report.revised_answer)
        self.assertIn("does not establish", report.revised_answer)
        self.assertNotEqual(report.verifications[0].llm_verdict, SUPPORTED)

    def test_conflicting_answer_revision_reports_unresolved_sources(self):
        provider = StaticProvider([
            pair_payload("E1", "C1", "SUPPORTS"),
            pair_payload("E2", "C1", "CONTRADICTS"),
        ])
        report = run_experiment_b_verification("Does System use BM25?", "System uses BM25.", [
            EvidenceItem("source-a", "System uses BM25."),
            EvidenceItem("source-b", "System does not use BM25."),
        ], llm_provider=provider,
            config=ExperimentBConfig(max_targeted_retrieval_retries=0, answer_revision_enabled=True))
        self.assertEqual(report.verifications[0].verdict, INSUFFICIENT_EVIDENCE)
        self.assertIn("sources conflict", report.revised_answer)
        self.assertIn("unresolved", report.revised_answer)
        self.assertNotEqual(report.revised_answer, "System uses BM25.")

    def test_pairwise_engine_detects_case5_conflict_deterministically(self):
        unit_text = "System X uses BM25 for lexical retrieval."
        provider = StaticProvider([
            pair_payload("E1", "C1", "SUPPORTS"),
            pair_payload("E2", "C1", "CONTRADICTS"),
        ])
        report = run_experiment_b_verification("Does System X use BM25?", unit_text, [
            EvidenceItem("source-a", unit_text),
            EvidenceItem("source-b", "System X does not use BM25; lexical retrieval is performed with TF-IDF."),
        ], llm_provider=provider,
            config=ExperimentBConfig(max_targeted_retrieval_retries=0, answer_revision_enabled=False))
        result = report.verifications[0]
        self.assertEqual(report.metrics["llm_calls"], 2)
        self.assertEqual(result.evidence_assessments, [
            {"evidence_index": "E1", "component_indices": ["C1"], "stance": "SUPPORTS"},
            {"evidence_index": "E2", "component_indices": ["C1"], "stance": "CONTRADICTS"},
        ])
        self.assertTrue(result.assessment_complete)
        self.assertEqual(result.missing_assessments, [])
        self.assertEqual(result.coverage_matrix[0]["status"], "CONFLICTING")
        self.assertEqual(result.conflicting_components, [unit_text])
        self.assertEqual(result.verdict, INSUFFICIENT_EVIDENCE)

    def test_invalid_pair_schema_retries_only_that_pair_and_preserves_valid_pair(self):
        provider = StaticProvider([
            pair_payload("E1", "C1", "SUPPORTS"),
            json.dumps({"assessments": [{"evidence_index": "E2", "component_indices": ["C1"]}],
                        "reason": "Missing stance."}),
            json.dumps({"task": "classification", "evidence": "E2", "component": "C1"}),
        ])
        report = run_experiment_b_verification("Does System X use BM25?",
            "System X uses BM25 for lexical retrieval.", [
                EvidenceItem("source-a", "System X uses BM25 for lexical retrieval."),
                EvidenceItem("source-b", "System X does not use BM25; lexical retrieval is performed with TF-IDF."),
            ], llm_provider=provider,
            config=ExperimentBConfig(max_judge_retries=1, max_targeted_retrieval_retries=0,
                                     answer_revision_enabled=False))
        result = report.verifications[0]
        self.assertEqual(report.metrics["llm_calls"], 3)
        self.assertEqual(result.evidence_assessments, [
            {"evidence_index": "E1", "component_indices": ["C1"], "stance": "SUPPORTS"},
        ])
        self.assertEqual(result.missing_assessments, ["E2-C1"])
        self.assertFalse(result.assessment_complete)
        self.assertEqual(result.verdict, INSUFFICIENT_EVIDENCE)
        self.assertTrue(any("Invalid assessment for E2-C1" in error for error in result.validation_errors))


if __name__ == "__main__":
    unittest.main()
