from __future__ import annotations

import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from claim_verification import (
    CONTRADICTED, INSUFFICIENT_EVIDENCE, SUPPORTED, EvidenceItem,
    evidence_from_retrieval_results,
)
from experiment_b_verification import (
    ExperimentBConfig, build_unit_retrieval_query, revise_answer,
    run_experiment_b_verification, validate_judgement,
)
from llm import OllamaProvider, StaticProvider
from research_pipeline import run_experiment_a, run_experiment_b
from research_retrieval import ExistingRAGProvider, RetrievalEvidence
from research_state import ResearchState
from verification_units import analyze_verification_units


def judge(verdict: str, *, support=(), refute=(), supported=(), unsupported=(),
          assessments=(), reason="Evidence-based result.") -> str:
    return json.dumps({"verdict": verdict, "supporting_evidence_ids": list(support),
        "contradicting_evidence_ids": list(refute), "supported_components": list(supported),
        "unsupported_components": list(unsupported), "reason": reason,
        "evidence_assessments": list(assessments)})


def pair_assessment(evidence_index: str, component_index: str, stance: str) -> str:
    return json.dumps({"assessments": [{"evidence_index": evidence_index,
        "component_indices": [component_index], "stance": stance}], "reason": "Pair assessed."})


class QueueRetriever:
    name = "queue-retriever"
    def __init__(self, responses): self.responses = list(responses); self.queries = []
    def retrieve(self, query, limit=8):
        self.queries.append(query)
        items = self.responses.pop(0) if self.responses else []
        return SimpleNamespace(query=query, evidence=items, provider=self.name)


class VerificationUnitTests(unittest.TestCase):
    def test_unit_preserves_original_text(self):
        text = "Company A acquired Company B in 2024."
        unit = analyze_verification_units(text)[0]
        self.assertEqual(unit.original_text, text)

    def test_date_relationship_remains_one_unit(self):
        unit = analyze_verification_units("Company A acquired Company B in 2024.")[0]
        self.assertEqual(unit.temporal_constraints, ["2024"])
        self.assertIn("acquired", unit.original_text)

    def test_condition_is_not_split_from_proposition(self):
        units = analyze_verification_units("If GPU acceleration is enabled, the system processes images faster.")
        self.assertEqual(len(units), 1)
        self.assertTrue(units[0].conditions)
        self.assertTrue(units[0].dependencies)

    def test_conditional_multi_outcome_sentence_is_conservatively_grouped(self):
        unit = analyze_verification_units(
            "When GPU mode is active, the service returns more images and consumes less CPU."
        )[0]
        self.assertEqual(len(unit.objects), 1)
        self.assertIn("consumes less CPU", unit.objects[0])

    def test_quantity_and_condition_are_attached(self):
        unit = analyze_verification_units("The model achieves 95% accuracy when inputs are normalized.")[0]
        self.assertTrue(any("95%" in x for x in unit.quantitative_constraints))
        self.assertTrue(unit.conditions)

    def test_negation_is_preserved(self):
        unit = analyze_verification_units("The system does not use PostgreSQL.")[0]
        self.assertIn("does not", unit.negation.lower())

    def test_causal_relation_is_preserved(self):
        unit = analyze_verification_units("Latency fell because the cache was warmed.")[0]
        self.assertIn("because", unit.causal_relation.lower())
        self.assertTrue(unit.dependencies)

    def test_comparison_direction_is_preserved(self):
        unit = analyze_verification_units("Model A is faster than Model B.")[0]
        self.assertIn("faster than Model B", unit.comparison_relation)

    def test_coordinated_objects_stay_in_one_unit(self):
        unit = analyze_verification_units("The pipeline uses FAISS, BM25, and a cross-encoder.")[0]
        self.assertEqual(len(unit.objects), 3)
        self.assertEqual(unit.subject.lower(), "pipeline")

    def test_technical_components_are_not_split_into_unrelated_units(self):
        units = analyze_verification_units("The parser performs OCR, layout parsing, and table extraction.")
        self.assertEqual(len(units), 1)
        self.assertEqual(len(units[0].objects), 3)


class DeterministicValidatorTests(unittest.TestCase):
    def test_simple_supported_claim(self):
        unit = analyze_verification_units("The report uses evidence.")[0]
        ev = [EvidenceItem("E1", "The report uses evidence.")]
        result = validate_judgement(unit, json.loads(judge(SUPPORTED, support=["E1"])), ev)
        self.assertEqual(result.verdict, SUPPORTED)

    def test_direct_contradiction_requires_refuting_id(self):
        unit = analyze_verification_units("The sensor is active.")[0]
        ev = [EvidenceItem("E1", "The sensor is inactive.")]
        data = json.loads(judge(CONTRADICTED, refute=["E1"]))
        self.assertEqual(validate_judgement(unit, data, ev).verdict, CONTRADICTED)
        data["contradicting_evidence_ids"] = []
        self.assertEqual(validate_judgement(unit, data, ev).verdict, INSUFFICIENT_EVIDENCE)

    def test_insufficient_evidence(self):
        unit = analyze_verification_units("The sensor is active.")[0]
        result = validate_judgement(unit, json.loads(judge(INSUFFICIENT_EVIDENCE)),
                                    [EvidenceItem("E1", "The manual lists sensor colors.")])
        self.assertEqual(result.verdict, INSUFFICIENT_EVIDENCE)

    def test_partial_compound_support_is_downgraded(self):
        unit = analyze_verification_units("The system uses FAISS and BM25.")[0]
        ev = [EvidenceItem("E1", "The system uses FAISS.")]
        data = json.loads(judge(SUPPORTED, support=["E1"], supported=["FAISS"], unsupported=["BM25"]))
        self.assertEqual(validate_judgement(unit, data, ev).verdict, INSUFFICIENT_EVIDENCE)

    def test_missing_date_downgrades_supported(self):
        unit = analyze_verification_units("Project S launched in 2024.")[0]
        ev = [EvidenceItem("E1", "Project S launched.")]
        result = validate_judgement(unit, json.loads(judge(SUPPORTED, support=["E1"])), ev)
        self.assertEqual(result.verdict, INSUFFICIENT_EVIDENCE)

    def test_missing_quantity_downgrades_supported(self):
        unit = analyze_verification_units("The model achieved 95% accuracy.")[0]
        ev = [EvidenceItem("E1", "The model achieved high accuracy.")]
        result = validate_judgement(unit, json.loads(judge(SUPPORTED, support=["E1"])), ev)
        self.assertEqual(result.verdict, INSUFFICIENT_EVIDENCE)

    def test_conflicting_evidence_without_provenance_resolution_is_insufficient(self):
        unit = analyze_verification_units("The service is available.")[0]
        ev = [EvidenceItem("E1", "The service is available."), EvidenceItem("E2", "The service is unavailable.")]
        data = json.loads(judge(SUPPORTED, support=["E1"], refute=["E2"]))
        result = validate_judgement(unit, data, ev)
        self.assertEqual(result.verdict, INSUFFICIENT_EVIDENCE)
        self.assertTrue(result.overrides)

    def test_same_component_conflict_remains_insufficient_even_with_temporal_metadata(self):
        unit = analyze_verification_units("The contract was active in 2024.")[0]
        ev = [EvidenceItem("E1", "The contract was active in 2024.", metadata={"event_date":"2024-06"}),
              EvidenceItem("E2", "The contract was inactive in 2023.", metadata={"event_date":"2023-08"})]
        result = validate_judgement(unit, json.loads(judge(SUPPORTED, support=["E1"], refute=["E2"])), ev)
        self.assertEqual(result.verdict, INSUFFICIENT_EVIDENCE)
        self.assertTrue(result.conflicting_components)

    def test_invalid_evidence_id_is_rejected(self):
        unit = analyze_verification_units("The report is final.")[0]
        result = validate_judgement(unit, json.loads(judge(SUPPORTED, support=["E404"])),
                                    [EvidenceItem("E1", "The report is final.")])
        self.assertEqual(result.verdict, INSUFFICIENT_EVIDENCE)
        self.assertTrue(any("Unknown evidence ID" in x for x in result.overrides))

    def test_inconsistent_supported_verdict_is_overridden(self):
        unit = analyze_verification_units("The system uses FAISS and BM25.")[0]
        ev = [EvidenceItem("E1", "The system uses FAISS.")]
        result = validate_judgement(unit, json.loads(judge(SUPPORTED, support=["E1"], unsupported=["BM25"])), ev)
        self.assertEqual(result.verdict, INSUFFICIENT_EVIDENCE)

    def test_missing_coordinated_object_is_detected_even_if_model_omits_it(self):
        unit = analyze_verification_units("The pipeline uses FAISS, BM25, and a cross-encoder.")[0]
        ev = [EvidenceItem("E1", "The pipeline uses FAISS.")]
        data = json.loads(judge(SUPPORTED, support=["E1"], supported=["FAISS"]))
        result = validate_judgement(unit, data, ev)
        self.assertEqual(result.verdict, INSUFFICIENT_EVIDENCE)
        self.assertTrue(any("coordinated object" in x for x in result.overrides))

    def test_evidence_assessment_ids_are_validated_and_attributed(self):
        unit = analyze_verification_units("The report is final.")[0]
        data = json.loads(judge(SUPPORTED, assessments=[{"evidence_id": "E1", "interpretation": "SUPPORTS"}]))
        result = validate_judgement(unit, data, [EvidenceItem("E1", "The report is final.")])
        self.assertEqual(result.verdict, SUPPORTED)


class ExperimentBEngineTests(unittest.TestCase):
    def test_adapter_conversion_preserves_provenance_and_score_components(self):
        item = RetrievalEvidence(evidence_id="EV7", text="Chunk content", source_id="src7",
            filename="paper.pdf", page=4, score=0.73, retrieval_metadata={"rank": 2},
            document_id="doc7", chunk_id="chunk7", retrieval_score=0.73,
            reranker_score=0.61, semantic_score=0.7, lexical_score=0.5)
        converted = evidence_from_retrieval_results([item])[0]
        self.assertEqual(converted.text, "Chunk content")
        self.assertEqual(converted.source_id, "src7")
        self.assertEqual(converted.document_id, "doc7")
        self.assertEqual(converted.chunk_id, "chunk7")
        self.assertEqual(converted.score, 0.73)
        self.assertEqual(converted.retrieval_score, 0.73)
        self.assertEqual(converted.reranker_score, 0.61)
        self.assertEqual(converted.semantic_score, 0.7)
        self.assertEqual(converted.lexical_score, 0.5)

    def test_native_rag_adapter_maps_evidence_score_and_reranker(self):
        native = SimpleNamespace(index=0, chunk={"text":"Chunk", "filename":"x.pdf", "page":2, "chunk_id":"c2"},
            evidence_score=0.64, rerank_score=0.81, semantic_score=0.7, lexical_score=0.5,
            to_dict=lambda: {"evidence_score":0.64,"rerank_score":0.81})
        adapted = ExistingRAGProvider._convert_result(native, fallback_index=1)
        self.assertEqual(adapted.score, 0.64)
        self.assertEqual(adapted.retrieval_score, 0.64)
        self.assertEqual(adapted.reranker_score, 0.81)
        self.assertEqual(adapted.chunk_id, "c2")

    def test_pipeline_integration_records_units_state_and_verified_answer(self):
        answer = "A factual statement."
        state = ResearchState(user_request="Check the statement")
        result = run_experiment_b("Why?", answer, [],
            retrieval_provider=QueueRetriever([[EvidenceItem("E1", answer, source="paper.pdf")]]),
            llm_provider=StaticProvider([judge(SUPPORTED, support=["E1"])]),
            verification_config=ExperimentBConfig(max_targeted_retrieval_retries=0, answer_revision_enabled=True),
            state=state)
        self.assertIsNone(result.error)
        self.assertEqual(result.final_answer, "A factual statement. [E1]")
        self.assertEqual(state.claims[0].status, SUPPORTED)
        self.assertTrue(any(event.action == "verification_unit" for event in state.events))

    def test_experiment_a_answer_remains_unchanged(self):
        result = run_experiment_a("Question?", "Baseline answer.", [])
        self.assertEqual(result.final_answer, "Baseline answer.")
        self.assertFalse(result.verification_enabled)

    def test_missing_evidence_returns_insufficient_without_llm_call(self):
        provider = StaticProvider([])
        report = run_experiment_b_verification("Q", "A factual statement.", [], llm_provider=provider)
        self.assertEqual(report.verifications[0].verdict, INSUFFICIENT_EVIDENCE)
        self.assertEqual(report.metrics["llm_calls"], 0)

    def test_malformed_output_retries_once(self):
        provider = StaticProvider(["not JSON", judge(SUPPORTED, support=["E1"])])
        report = run_experiment_b_verification("Q", "A factual statement.",
            [EvidenceItem("E1", "A factual statement.")], llm_provider=provider,
            config=ExperimentBConfig(max_judge_retries=1))
        self.assertEqual(report.verifications[0].verdict, SUPPORTED)
        self.assertEqual(report.metrics["llm_calls"], 2)

    def test_claim_specific_retrieval_includes_question_and_unit(self):
        retriever = QueueRetriever([[EvidenceItem("E1", "A factual statement.")]])
        provider = StaticProvider([judge(SUPPORTED, support=["E1"])])
        report = run_experiment_b_verification("Why is this true?", "A factual statement.", [],
            retrieval_provider=retriever, llm_provider=provider,
            config=ExperimentBConfig(max_targeted_retrieval_retries=0))
        self.assertEqual(report.verifications[0].verdict, SUPPORTED)
        self.assertEqual(len(retriever.queries), 1)
        self.assertIn("Why is this true?", retriever.queries[0])
        self.assertIn("A factual statement", retriever.queries[0])

    def test_ablation_can_disable_unit_specific_retrieval(self):
        retriever = QueueRetriever([[EvidenceItem("E2", "unused targeted text")]])
        provider = StaticProvider([judge(SUPPORTED, support=["E1"])])
        report = run_experiment_b_verification("Q", "A factual statement.",
            [EvidenceItem("E1", "A factual statement.")], retrieval_provider=retriever,
            llm_provider=provider, config=ExperimentBConfig(unit_specific_retrieval=False,
                max_targeted_retrieval_retries=1))
        self.assertEqual(retriever.queries, [])
        self.assertEqual(report.verifications[0].verdict, SUPPORTED)

    def test_pipeline_default_keeps_original_and_returns_separate_revision(self):
        answer = "A factual statement."
        result = run_experiment_b("Q", answer, [EvidenceItem("E1", answer)],
            llm_provider=StaticProvider([judge(SUPPORTED, support=["E1"])]))
        self.assertEqual(result.final_answer, answer)
        self.assertIn("[E1]", result.report.revised_answer)

    def test_insufficient_verdict_triggers_one_focused_retrieval_retry(self):
        retriever = QueueRetriever([
            [EvidenceItem("E1", "Company A acquired Company B in 2024.")],
            [EvidenceItem("E2", "Company A acquired Company B in 2024 for $2 billion.")],
        ])
        provider = StaticProvider([
            pair_assessment("E1", "C1", "SUPPORTS"),
            pair_assessment("E1", "C1", "SUPPORTS"),
            pair_assessment("E2", "C1", "SUPPORTS"),
        ])
        answer = "Company A acquired Company B in 2024 for $2 billion."
        report = run_experiment_b_verification("What happened?", answer, [],
            retrieval_provider=retriever, llm_provider=provider,
            config=ExperimentBConfig(max_targeted_retrieval_retries=1))
        self.assertEqual(len(retriever.queries), 2)
        self.assertNotEqual(retriever.queries[0], retriever.queries[1])
        self.assertEqual(report.verifications[0].verdict, SUPPORTED)
        self.assertEqual(report.metrics["llm_calls"], 3)
        self.assertEqual(report.metrics["retrieval_calls"], 2)

    def test_trace_records_queries_evidence_verdict_and_counts(self):
        retriever = QueueRetriever([[EvidenceItem("E1", "A factual statement.", source="doc")]])
        provider = StaticProvider([judge(SUPPORTED, support=["E1"])])
        report = run_experiment_b_verification("Q", "A factual statement.", [],
            retrieval_provider=retriever, llm_provider=provider,
            config=ExperimentBConfig(max_targeted_retrieval_retries=0))
        self.assertEqual(report.trace[1]["final_verdict"], SUPPORTED)
        self.assertTrue(report.trace[1]["retrieval_queries"])
        self.assertEqual(report.metrics["llm_calls"], 1)

    def test_answer_revision_cites_supported_evidence(self):
        verification = run_experiment_b_verification("Q", "A factual statement.",
            [EvidenceItem("E1", "A factual statement.")],
            llm_provider=StaticProvider([judge(SUPPORTED, support=["E1"])]),
            config=ExperimentBConfig(answer_revision_enabled=True))
        self.assertIn("A factual statement.", verification.revised_answer)
        self.assertIn("[E1]", verification.revised_answer)

    def test_answer_revision_qualifies_insufficient_assertion(self):
        report = run_experiment_b_verification("Q", "Company Z bought Firm K for $5 million.",
            [EvidenceItem("E1", "Company Z bought Firm K.")],
            llm_provider=StaticProvider([judge(INSUFFICIENT_EVIDENCE, support=["E1"], unsupported=["$5 million"])]),
            config=ExperimentBConfig(answer_revision_enabled=True))
        self.assertIn("does not establish", report.revised_answer)

    def test_contradicted_assertion_is_replaced_by_cited_evidence(self):
        ev = EvidenceItem("E2", "The sensor remained inactive.")
        report = run_experiment_b_verification("Q", "The sensor was active.", [ev],
            llm_provider=StaticProvider([judge(CONTRADICTED, refute=["E2"])]),
            config=ExperimentBConfig(answer_revision_enabled=True))
        self.assertNotIn("The sensor was active.", report.revised_answer)
        self.assertIn("The sensor remained inactive.", report.revised_answer)
        self.assertIn("[E2]", report.revised_answer)

    def test_explicit_ollama_thinking_and_json_settings_are_sent(self):
        event = b'{"message":{"role":"assistant","content":"{}"},"done":true}\n'
        with patch("llm.urllib.request.urlopen", return_value=io.BytesIO(event)) as open_url:
            OllamaProvider(think=False, format="json").generate([{"role": "user", "content": "test"}])
        payload = json.loads(open_url.call_args.args[0].data.decode("utf-8"))
        self.assertIs(payload["think"], False)
        self.assertEqual(payload["format"], "json")

    def test_ablation_config_exposes_validator_retry_and_revision_switches(self):
        config = ExperimentBConfig(validator_enabled=False, max_targeted_retrieval_retries=0,
                                   answer_revision_enabled=False, think=None, response_format=None)
        self.assertFalse(config.validator_enabled)
        self.assertEqual(config.max_targeted_retrieval_retries, 0)
        self.assertFalse(config.answer_revision_enabled)


if __name__ == "__main__":
    unittest.main()
