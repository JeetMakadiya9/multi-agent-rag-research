from __future__ import annotations

import io
import json
import unittest
from unittest.mock import patch

import rag
from claim_verification import (
    CONTRADICTED,
    INSUFFICIENT_EVIDENCE,
    SUPPORTED,
    Claim,
    EvidenceItem,
    build_verification_report,
    evidence_from_retrieval_results,
    extract_claims,
    verify_claim,
)
from llm import OllamaProvider, StaticProvider
from research_retrieval import ExistingRAGProvider, RetrievalEvidence
from research_pipeline import run_experiment_b
from research_state import ResearchState


class EvidenceConversionTests(unittest.TestCase):
    def test_raw_retrieval_result_becomes_evidence_and_survives_report(self) -> None:
        raw_results = [
            {
                "chunk": {
                    "text": "RAG combines retrieval and language generation.",
                    "filename": "synthetic.pdf",
                    "page": 7,
                    "chunk_id": "chunk-7",
                    "section": "Overview",
                    "parent_id": "parent-1",
                },
                "evidence_score": 0.91,
            }
        ]

        evidence = evidence_from_retrieval_results(raw_results)
        self.assertEqual(len(evidence), 1)
        item = evidence[0]
        self.assertEqual(item.text, "RAG combines retrieval and language generation.")
        self.assertEqual(item.source, "synthetic.pdf")
        self.assertEqual(item.page, 7)
        self.assertEqual(item.score, 0.91)
        self.assertEqual(item.metadata["chunk_id"], "chunk-7")

        provider = StaticProvider(
            [
                '{"claims":[{"claim_id":"C1","text":"RAG combines retrieval and language generation."}]}',
                '{"verdict":"SUPPORTED","confidence":0.99,"evidence_ids":["chunk-7"],"explanation":"The passage states this."}',
            ]
        )
        report = build_verification_report(
            "RAG combines retrieval and language generation.",
            evidence,
            llm_provider=provider,
        )
        self.assertEqual(len(report.verifications), 1)
        self.assertEqual(report.verifications[0].verdict, SUPPORTED)
        self.assertEqual(report.verifications[0].supporting_evidence[0].text, item.text)

    def test_adapter_evidence_fields_are_preserved(self) -> None:
        adapter_item = RetrievalEvidence(
            evidence_id="ev-9",
            text="Evidence body",
            source_id="source-9",
            filename="paper.pdf",
            page=9,
            section="Methods",
            score=0.77,
            retrieval_metadata={"rank": 2},
        )
        converted = evidence_from_retrieval_results([adapter_item])

        self.assertEqual(len(converted), 1)
        self.assertEqual(converted[0].evidence_id, "ev-9")
        self.assertEqual(converted[0].text, "Evidence body")
        self.assertEqual(converted[0].source, "source-9")
        self.assertEqual(converted[0].page, 9)
        self.assertEqual(converted[0].score, 0.77)
        self.assertEqual(converted[0].metadata["rank"], 2)

    def test_native_rag_evidence_score_survives_adapter_conversion(self) -> None:
        result = rag.RetrievalResult(
            index=0,
            chunk={"text": "Known text", "filename": "known.pdf", "page": 3},
            evidence_score=0.634,
        )
        converted = ExistingRAGProvider._convert_result(result, fallback_index=1)
        self.assertEqual(converted.score, 0.634)


class DeterministicClaimVerificationTests(unittest.TestCase):
    def test_ollama_provider_collects_streamed_content(self) -> None:
        events = (
            b'{"message":{"role":"assistant","content":"stream "},"done":false}\n'
            b'{"message":{"role":"assistant","content":"response"},"done":true}\n'
        )
        with patch("llm.urllib.request.urlopen", return_value=io.BytesIO(events)) as open_url:
            response = OllamaProvider().generate([{"role": "user", "content": "hello"}])

        self.assertEqual(response, "stream response")
        request = open_url.call_args.args[0]
        self.assertTrue(json.loads(request.data.decode("utf-8"))["stream"])

    def test_claim_extraction_uses_static_provider(self) -> None:
        provider = StaticProvider(
            '{"claims":[{"claim_id":"C1","text":"The sky is blue."}]}'
        )
        claims = extract_claims("The sky is blue.", llm_provider=provider)
        self.assertEqual(claims, [Claim("C1", "The sky is blue.")])

    def test_supported_claim(self) -> None:
        result = verify_claim(
            Claim("C1", "The sky is blue."),
            [EvidenceItem("E1", "The sky is blue.")],
            llm_provider=StaticProvider(
                '{"verdict":"SUPPORTED","confidence":0.9,"evidence_ids":["E1"],"explanation":"Directly stated."}'
            ),
        )
        self.assertEqual(result.verdict, SUPPORTED)
        self.assertEqual(len(result.supporting_evidence), 1)

    def test_contradicted_claim(self) -> None:
        result = verify_claim(
            Claim("C1", "The sky is green."),
            [EvidenceItem("E1", "The sky is blue.")],
            llm_provider=StaticProvider(
                '{"verdict":"CONTRADICTED","confidence":0.8,"evidence_ids":["E1"],"explanation":"Evidence says blue."}'
            ),
        )
        self.assertEqual(result.verdict, CONTRADICTED)

    def test_insufficient_evidence_verdict(self) -> None:
        result = verify_claim(
            Claim("C1", "The sky is green."),
            [EvidenceItem("E1", "The passage discusses clouds.")],
            llm_provider=StaticProvider(
                '{"verdict":"INSUFFICIENT_EVIDENCE","confidence":0.95,"evidence_ids":[],"explanation":"No color is established."}'
            ),
        )
        self.assertEqual(result.verdict, INSUFFICIENT_EVIDENCE)

    def test_empty_evidence_needs_no_provider_call(self) -> None:
        result = verify_claim(Claim("C1", "A claim."), [], llm_provider=StaticProvider([]))
        self.assertEqual(result.verdict, INSUFFICIENT_EVIDENCE)
        self.assertIn("No evidence", result.explanation)

    def test_full_report_uses_sequenced_static_responses(self) -> None:
        provider = StaticProvider(
            [
                '{"claims":[{"claim_id":"C1","text":"RAG uses retrieved evidence."}]}',
                '{"verdict":"SUPPORTED","confidence":0.88,"evidence_ids":["E1"],"explanation":"The passage says so."}',
            ]
        )
        report = build_verification_report(
            "RAG uses retrieved evidence.",
            [EvidenceItem("E1", "RAG uses retrieved evidence.", source="paper.pdf", page=2)],
            llm_provider=provider,
        )
        self.assertEqual(report.supported_count, 1)
        self.assertEqual(report.support_rate, 1.0)
        self.assertEqual(report.verifications[0].supporting_evidence[0].page, 2)


class ExperimentBPipelineTests(unittest.TestCase):
    def test_pipeline_converts_raw_evidence_once_and_reports_metrics(self) -> None:
        answer = "RAG combines retrieval and language generation."
        raw_results = [
            {
                "chunk": {
                    "text": answer,
                    "filename": "synthetic.pdf",
                    "page": 4,
                    "chunk_id": "chunk-4",
                },
                "evidence_score": 0.84,
            }
        ]
        provider = StaticProvider(
            [
                '{"claims":[{"claim_id":"C1","text":"RAG combines retrieval and language generation."}]}',
                '{"assessments":[{"evidence_index":"E1","component_indices":["C1"],"stance":"SUPPORTS"}],"reason":"The evidence states the component."}',
                '{"assessments":[{"evidence_index":"E1","component_indices":["C2"],"stance":"SUPPORTS"}],"reason":"The evidence states the component."}',
            ]
        )

        result = run_experiment_b("What does RAG combine?", answer, raw_results, llm_provider=provider)

        self.assertIsNone(result.error)
        self.assertEqual(result.original_answer, answer)
        self.assertEqual(result.final_answer, answer)
        self.assertEqual(result.report.verifications[0].verdict, SUPPORTED)
        self.assertEqual(result.report.verifications[0].supporting_evidence[0].text, answer)
        self.assertEqual(result.verification_metrics["supported_claims"], 1)
        self.assertEqual(result.to_dict()["original_answer"], answer)

    def test_filter_option_returns_supported_claims_without_signature_error(self) -> None:
        answer = "Claim A. Claim B."
        provider = StaticProvider(
            [
                '{"claims":[{"claim_id":"C1","text":"Claim A."},{"claim_id":"C2","text":"Claim B."}]}',
                '{"verdict":"SUPPORTED","confidence":0.9,"evidence_ids":["E1"],"explanation":"Supported."}',
                '{"verdict":"CONTRADICTED","confidence":0.8,"evidence_ids":["E1"],"explanation":"Contradicted."}',
            ]
        )
        raw_results = [{"chunk": {"text": "Claim A is supported; Claim B is false."}}]

        result = run_experiment_b(
            "Check the claims.",
            answer,
            raw_results,
            llm_provider=provider,
            remove_unverified_claims=True,
        )

        self.assertIsNone(result.error)
        self.assertEqual(result.final_answer, "- Claim A.")
        self.assertEqual(result.report.contradicted_count, 1)

    def test_state_records_claim_verdicts_contradictions_and_unresolved_claims(self) -> None:
        answer = "Fact one. Fact two. Fact three."
        provider = StaticProvider(
            [
                '{"claims":[{"claim_id":"C1","text":"Fact one."},{"claim_id":"C2","text":"Fact two."},{"claim_id":"C3","text":"Fact three."}]}',
                '{"verdict":"SUPPORTED","confidence":0.9,"evidence_ids":["E1"],"explanation":"Supported."}',
                '{"verdict":"CONTRADICTED","confidence":0.8,"evidence_ids":["E1"],"explanation":"Conflicting evidence."}',
                '{"verdict":"INSUFFICIENT_EVIDENCE","confidence":0.7,"evidence_ids":[],"explanation":"Not established."}',
            ]
        )
        state = ResearchState(user_request="Verify three facts")
        raw_results = [{"chunk": {"text": "Fact one and a conflict with fact two."}}]

        result = run_experiment_b(
            "Verify these facts.",
            answer,
            raw_results,
            llm_provider=provider,
            state=state,
        )

        self.assertIsNone(result.error)
        self.assertEqual([claim.status for claim in state.claims], [SUPPORTED, CONTRADICTED, INSUFFICIENT_EVIDENCE])
        self.assertEqual(len(state.contradictions), 1)
        self.assertEqual(state.unresolved_questions, ["Fact three."])
        self.assertEqual(len(state.evidence), 1)



if __name__ == "__main__":
    unittest.main()
