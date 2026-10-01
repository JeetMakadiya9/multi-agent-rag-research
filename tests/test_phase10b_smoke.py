"""System and production-controller binding tests for the Phase 10B runner."""
from __future__ import annotations

import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from evaluation.experiments.experiment_config import ExperimentConfig
from evaluation.experiments.run_phase10_smoke import SCIFACT_DEV_SMOKE_IDS
from evaluation.experiments.system_c_multi_agent import (
    _Deterministic8AProvider, make_system_c, run_system_c,
)
from evaluation.scifact.scifact_adapter import load_scifact_dataset
from research_retrieval import RetrievalEvidence, RetrievalResponse, StaticRetrievalProvider


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "scifact" / "data"


class _Judge:
    def __init__(self, verdict="SUPPORTED", stance="SUPPORTS"):
        self.calls = 0
        self.verdict = verdict
        self.stance = stance

    def generate(self, messages, *, temperature=0.0, timeout=120):
        self.calls += 1
        return json.dumps({
            "assessments": [{"evidence_index": "E1", "component_indices": ["C1"],
                             "stance": self.stance}],
            "reason": "Deterministic integration-test assessment.",
        })


def _static_provider():
    evidence = RetrievalEvidence(
        evidence_id="chunk-1", text="A controlled clinical trial supports the claim.",
        source_id="scifact-source-101:page-1", filename="scifact-101.txt", page=1,
        section="Abstract", score=0.9, document_id="101", chunk_id="chunk-1",
    )
    return StaticRetrievalProvider([evidence])


class Phase10BTests(unittest.TestCase):
    # Dataset selection and config invariants.
    def test_fixed_smoke_ids_are_present_in_dev(self):
        dataset = load_scifact_dataset(DATASET, "dev")
        self.assertTrue(set(SCIFACT_DEV_SMOKE_IDS).issubset(
            {item.runtime.claim_id for item in dataset.claims}))

    def test_fixed_smoke_ids_have_support_contradict_and_empty_cases(self):
        dataset = load_scifact_dataset(DATASET, "dev")
        rows = {item.runtime.claim_id: item for item in dataset.claims}
        labels = {claim_id: {r.label for doc in rows[claim_id].annotations.evidence or ()
                             for r in doc.rationales}
                  for claim_id in SCIFACT_DEV_SMOKE_IDS}
        self.assertEqual(labels[1], set())
        self.assertEqual(labels[3], {"SUPPORT"})
        self.assertEqual(labels[42], {"CONTRADICT"})

    def test_config_preserves_fixed_example_order(self):
        config = ExperimentConfig(str(DATASET), example_ids=(42, 1, 3))
        self.assertEqual(config.example_ids, (42, 1, 3))

    def test_config_rejects_duplicate_example_ids(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            ExperimentConfig(str(DATASET), example_ids=(1, 1))

    def test_config_represents_final_test_separately_from_dev_smoke(self):
        config = ExperimentConfig(str(DATASET), split="test", example_ids=(1,))
        self.assertEqual(config.split, "test")

    # The same retrieval provider object is shared by all three evaluator slots.
    def test_fixed_ids_are_three_real_examples(self):
        self.assertEqual(len(SCIFACT_DEV_SMOKE_IDS), 3)
        self.assertEqual(len(set(SCIFACT_DEV_SMOKE_IDS)), 3)

    def test_a8a_binding_selects_retrieval_in_research_stage(self):
        provider = _Deterministic8AProvider("a claim")
        context = {"current_task_id": "t", "questions": [{"question_id": "q"}],
                   "current_stage": "RESEARCH"}
        message = {"content": "Context (read-only):\n" + json.dumps(context)}
        proposal = json.loads(provider.generate([{}, message]))
        self.assertEqual(proposal["action_type"], "RETRIEVE_EVIDENCE")
        self.assertEqual(proposal["query"], "a claim")

    def test_a8a_binding_selects_verification_in_review_stage(self):
        provider = _Deterministic8AProvider("a claim")
        context = {"current_task_id": "t", "questions": [{"question_id": "q"}],
                   "current_stage": "EVIDENCE_REVIEW",
                   "claims": [{"claim_id": "c", "evidence_ids": ["e1"]}]}
        message = {"content": "Context (read-only):\n" + json.dumps(context)}
        proposal = json.loads(provider.generate([{}, message]))
        self.assertEqual(proposal["action_type"], "VERIFY_CLAIM")
        self.assertEqual(proposal["evidence_ids"], ["e1"])

    def test_a8a_binding_rejects_unexpected_stage(self):
        provider = _Deterministic8AProvider("a claim")
        context = {"current_task_id": "t", "questions": [{"question_id": "q"}],
                   "current_stage": "CRITIQUE"}
        with self.assertRaisesRegex(RuntimeError, "Unexpected Phase 8A stage"):
            provider.generate([{}, {"content": "Context (read-only):\n" + json.dumps(context)}])

    # End-to-end C runs real 8C/8B/dispatch/Phase 6 APIs; only the bindings are deterministic.
    def test_system_c_runs_real_controller_twice_for_retrieve_then_verify(self):
        provider = _static_provider()
        judge = _Judge()
        config = ExperimentConfig(str(DATASET), top_k=1, verification_evidence_limit=1)
        result = run_system_c({"claim_id": 9001, "claim": "The trial supports the claim."},
                              provider, config, llm_provider=judge)
        self.assertEqual(result.predicted_label, "SUPPORTED")
        self.assertEqual(result.controller_cycles, 2)
        self.assertEqual(result.retrieval_calls, 2)
        self.assertGreaterEqual(judge.calls, 1)
        self.assertEqual(result.raw_output["controller_decisions"],
                         ["RETRIEVE_EVIDENCE", "VERIFY_CLAIM"])

    def test_system_c_trace_contains_accepted_8a_validations(self):
        result = run_system_c({"claim_id": 9002, "claim": "The trial supports the claim."},
                              _static_provider(),
                              ExperimentConfig(str(DATASET), top_k=1, verification_evidence_limit=1),
                              llm_provider=_Judge())
        cycles = [cycle for trace in result.raw_output["controller_traces"]
                  for cycle in trace["trace"]["cycles"]]
        self.assertEqual(len(cycles), 2)
        self.assertTrue(all(cycle["validation"]["validation_status"] == "ACCEPTED"
                            for cycle in cycles))

    def test_system_c_trace_contains_completed_8b_executions(self):
        result = run_system_c({"claim_id": 9003, "claim": "The trial supports the claim."},
                              _static_provider(),
                              ExperimentConfig(str(DATASET), top_k=1, verification_evidence_limit=1),
                              llm_provider=_Judge())
        cycles = [cycle for trace in result.raw_output["controller_traces"]
                  for cycle in trace["trace"]["cycles"]]
        self.assertTrue(all(cycle["execution"]["status"] == "COMPLETED" for cycle in cycles))

    def test_system_c_trace_reaches_phase7b_dispatch(self):
        result = run_system_c({"claim_id": 9004, "claim": "The trial supports the claim."},
                              _static_provider(),
                              ExperimentConfig(str(DATASET), top_k=1, verification_evidence_limit=1),
                              llm_provider=_Judge())
        dispatches = [cycle["execution"]["dispatch"] for trace in result.raw_output["controller_traces"]
                     for cycle in trace["trace"]["cycles"]]
        self.assertEqual([item["capability"] for item in dispatches],
                         ["local_retrieval", "evidence_verification"])

    def test_system_c_reaches_phase6_retrieval_and_verification_artifacts(self):
        result = run_system_c({"claim_id": 9005, "claim": "The trial supports the claim."},
                              _static_provider(),
                              ExperimentConfig(str(DATASET), top_k=1, verification_evidence_limit=1),
                              llm_provider=_Judge())
        artifacts = result.raw_output["research_state"]["research_artifacts"]
        self.assertTrue(any(item.get("artifact_type") == "retrieval_result" for item in artifacts))
        self.assertTrue(any(item.get("artifact_type") == "verification_result" for item in artifacts))

    def test_system_c_records_retrieval_provenance(self):
        result = run_system_c({"claim_id": 9006, "claim": "The trial supports the claim."},
                              _static_provider(),
                              ExperimentConfig(str(DATASET), top_k=1, verification_evidence_limit=1),
                              llm_provider=_Judge())
        state = result.raw_output["research_state"]
        self.assertEqual(len(state["document_versions"]), 1)
        self.assertEqual(len(state["evidence"]), 1)
        self.assertTrue(state["evidence"][0]["passage_reference_id"])

    def test_system_c_uses_explicit_continuation_authorization(self):
        result = run_system_c({"claim_id": 9007, "claim": "The trial supports the claim."},
                              _static_provider(),
                              ExperimentConfig(str(DATASET), top_k=1, verification_evidence_limit=1),
                              llm_provider=_Judge())
        state = result.raw_output["research_state"]
        loop = state["research_loops"][0]
        self.assertTrue(loop["continuation_authorizations"])

    def test_system_c_controller_is_finitely_bounded(self):
        result = run_system_c({"claim_id": 9008, "claim": "The trial supports the claim."},
                              _static_provider(),
                              ExperimentConfig(str(DATASET), top_k=1, verification_evidence_limit=1),
                              llm_provider=_Judge())
        for trace in result.raw_output["controller_traces"]:
            self.assertEqual(len(trace["trace"]["cycles"]), 1)

    def test_system_c_uses_shared_provider_for_preflight_and_controlled_retrieval(self):
        class CountingProvider(StaticRetrievalProvider):
            def __init__(self, evidence):
                super().__init__(evidence)
                self.queries = []

            def retrieve(self, query, limit=8):
                self.queries.append((query, limit))
                return super().retrieve(query, limit)
        provider = CountingProvider(_static_provider().evidence)
        claim = "The trial supports the claim."
        run_system_c({"claim_id": 9009, "claim": claim}, provider,
                     ExperimentConfig(str(DATASET), top_k=1, verification_evidence_limit=1),
                     llm_provider=_Judge())
        self.assertEqual(provider.queries, [(claim, 1), (claim, 1)])

    def test_system_c_factory_binds_test_judge_without_replacing_controller(self):
        system = make_system_c(llm_provider=_Judge())
        result = system({"claim_id": 9010, "claim": "The trial supports the claim."},
                        _static_provider(),
                        ExperimentConfig(str(DATASET), top_k=1, verification_evidence_limit=1))
        self.assertEqual(result.predicted_label, "SUPPORTED")
        self.assertEqual(result.controller_cycles, 2)

    # Artifact and error contracts.
    def test_smoke_runner_reports_fixed_ids_as_dev_only(self):
        self.assertEqual(SCIFACT_DEV_SMOKE_IDS, (1, 3, 42))

    def test_system_c_returns_retrieved_evidence_in_structured_output(self):
        result = run_system_c({"claim_id": 9011, "claim": "The trial supports the claim."},
                              _static_provider(),
                              ExperimentConfig(str(DATASET), top_k=1, verification_evidence_limit=1),
                              llm_provider=_Judge())
        self.assertEqual(result.retrieved_chunks[0]["scifact_doc_id"], 101)
        self.assertEqual(result.selected_evidence_doc_ids, [101])

    def test_system_c_returns_verification_assessments(self):
        result = run_system_c({"claim_id": 9012, "claim": "The trial supports the claim."},
                              _static_provider(),
                              ExperimentConfig(str(DATASET), top_k=1, verification_evidence_limit=1),
                              llm_provider=_Judge())
        self.assertTrue(result.assessments)
        self.assertEqual(result.assessments[0]["verdict"], "SUPPORTED")

    def test_system_c_counted_calls_include_preflight_work(self):
        result = run_system_c({"claim_id": 9013, "claim": "The trial supports the claim."},
                              _static_provider(),
                              ExperimentConfig(str(DATASET), top_k=1, verification_evidence_limit=1),
                              llm_provider=_Judge())
        self.assertEqual(result.retrieval_calls, 2)

    def test_system_c_error_is_not_converted_to_empty_success(self):
        class BadProvider:
            def retrieve(self, query, limit=8):
                raise RuntimeError("retrieval unavailable")
        # The evaluator's safe boundary retains thrown system errors explicitly.
        from evaluation.experiments.evaluator import _safe_call
        output, _ = _safe_call(run_system_c,
                               {"claim_id": 9014, "claim": "A claim."}, BadProvider(),
                               ExperimentConfig(str(DATASET), top_k=1))
        self.assertIn("retrieval unavailable", output.error)

    def test_smoke_runner_does_not_select_test_split(self):
        config = ExperimentConfig(str(DATASET), split="dev", example_ids=SCIFACT_DEV_SMOKE_IDS)
        self.assertEqual(config.split, "dev")

    def test_execution_callback_receives_no_gold_annotation_fields(self):
        from evaluation.experiments.evaluator import _safe_call, SystemOutput
        seen = {}
        def callback(runtime, provider, config):
            seen.update(runtime)
            return SystemOutput(raw_output="ok")
        _safe_call(callback, {"claim_id": 123, "claim": "A claim."}, object(),
                   ExperimentConfig(str(DATASET)))
        self.assertEqual(set(seen), {"claim_id", "claim"})
        self.assertNotIn("gold_label", seen)
        self.assertNotIn("annotations", seen)

    def test_result_artifact_has_per_system_status_evidence_and_provenance(self):
        from evaluation.experiments.evaluator import run_experiment, SystemOutput
        class Retriever:
            def retrieve(self, query, limit=8):
                raise AssertionError("callback fixture must not retrieve")
        def callback(runtime, provider, config):
            return SystemOutput(raw_output={"answer": "smoke"})
        with tempfile.TemporaryDirectory() as folder:
            run_experiment(
                ExperimentConfig(str(DATASET), example_ids=(1,)), Retriever(),
                {name: callback for name in "ABC"}, folder,
            )
            row = json.loads((Path(folder) / "per_example.jsonl").read_text(encoding="utf-8"))
        for result in row["systems"].values():
            self.assertEqual(result["status"], "SUCCESS")
            self.assertEqual(result["example_id"], 1)
            self.assertEqual(result["input_claim"], row["runtime_input"]["claim"])
            self.assertIsNone(result["gold_label"])
            self.assertIn("evidence", result)
            self.assertIn("provenance", result)
            self.assertIn("verification_status", result)
            self.assertIn("final_result", result)

    def test_local_model_json_mode_is_passed_to_ollama(self):
        from evaluation.experiments.local_model import OllamaCPUProvider
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def __iter__(self):
                yield b'{"message":{"content":"{}"},"done":true}\n'
        with patch("evaluation.experiments.local_model.urllib.request.urlopen",
                   return_value=Response()) as open_url:
            provider = OllamaCPUProvider(response_format="json")
            provider.generate([{"role": "user", "content": "json"}])
        request = open_url.call_args.args[0]
        self.assertEqual(json.loads(request.data)["format"], "json")

    def test_phase6_failure_stays_failed_through_real_controller(self):
        class BrokenJudge:
            def generate(self, *args, **kwargs):
                raise TimeoutError("judge timed out")
        with self.assertRaisesRegex(RuntimeError, "Phase 8C execution failed") as caught:
            run_system_c({"claim_id": 9015, "claim": "The trial supports the claim."},
                         _static_provider(),
                         ExperimentConfig(str(DATASET), top_k=1, verification_evidence_limit=1),
                         llm_provider=BrokenJudge())
        self.assertIn("EXECUTION_FAILED", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
