"""Candidate verifier variants (docqa/agentic_candidates.py). Model replies are SCRIPTED: these tests check the
control flow and the guards, not model quality. The default verifier must stay arm_c_studio."""
import json

from docqa import agentic_candidates as ac
from docqa.agentic import StudioArmCVerifier
from docqa.verify import INSUFFICIENT, SUPPORTED, VERIFIERS
from test_adversarial_campaign import RawVerifier
from test_agentic_verifier import CLAIM, NEU, SUP, Store, ev
from webrag.model_service import STUDIO_VERIFIER

PASSAGE = "Feeding rapamycin to adult Drosophila produces the life span extension seen in some TOR mutants."


def reply(*items):
    return json.dumps({"assessments": [{"id": pid, "stance": s, "quote": q} for pid, s, q in items]})


def test_a_near_copy_is_repaired_to_the_exact_span():
    assert ac.repair("feeding rapamycin to adult Drosoph. produces the life span extension", PASSAGE) == \
        "Feeding rapamycin to adult Drosophila produces the life span extension"


def test_exact_quotes_are_left_alone():
    assert ac.repair("feeding rapamycin to adult Drosophila produces", PASSAGE) is None


def test_a_quote_that_adds_a_negation_is_not_repaired():
    text = "Microglia solely function in disease states in the central nervous system (CNS)."
    assert ac.repair("Microglia are not solely functioning in disease states in the central nervous system (CNS).",
                     text) is None


def test_a_quote_with_different_numbers_is_not_repaired():
    assert ac.repair("lowered systolic blood pressure by 19 mmHg", SUP) is None


def test_a_dissimilar_quote_is_not_repaired():
    assert ac.repair("an entirely different sentence that is not in the passage at all", PASSAGE) is None


def test_the_candidate_keeps_a_repaired_stance_and_records_the_repair():
    abbreviated = "In a randomised trial of 400 adults, drug X lowered systolic blood press. by 9 mmHg."
    model = RawVerifier(reply(("P1", "SUPPORTS", abbreviated), ("P2", "NEUTRAL", "")))
    verdict = ac.StudioArmCQuoteRepairVerifier(model).verify(CLAIM, ev(SUP, NEU), search=Store().rank)
    assert verdict.label == SUPPORTED
    repairs = verdict.agentic_trace["quote_repairs"]
    assert repairs[0]["quote"] == abbreviated and repairs[0]["repaired_to"] in SUP


def test_the_default_still_demotes_the_same_reply():
    abbreviated = "In a randomised trial of 400 adults, drug X lowered systolic blood press. by 9 mmHg."
    model = RawVerifier(reply(("P1", "SUPPORTS", abbreviated), ("P2", "NEUTRAL", "")),
                        planner=[{"action": "ADJUDICATE"}], adjudicator=[{"verdict": "INSUFFICIENT", "passage": "P1",
                                                                          "quote": ""}])
    verdict = StudioArmCVerifier(model).verify(CLAIM, ev(SUP, NEU), search=Store().rank)
    assert verdict.label == INSUFFICIENT and "quote_repairs" not in verdict.agentic_trace


class Recorder:
    """A model client that records the system prompts it receives and answers every role with a neutral reply."""

    def __init__(self):
        self.systems = []

    def chat(self, messages, **kw):
        from docqa.llm import LLMResponse
        self.systems.append(messages[0]["content"])
        user = messages[1]["content"]
        if "Passages:" in user:
            return LLMResponse(reply(("P1", "NEUTRAL", "")), "stop", 10, 100, 0.01)
        return LLMResponse(json.dumps({"action": "ADJUDICATE", "verdict": "INSUFFICIENT", "passage": "P1",
                                       "quote": ""}), "stop", 10, 100, 0.01)


def test_prompt_p2_replaces_only_the_verifier_prompt():
    from evaluation.experiments.agentic_verify import prompts
    client = Recorder()
    ac.StudioArmCPromptP2Verifier(client).verify(CLAIM, ev(NEU), search=Store().rank)
    assert client.systems[0] == ac.VERIFIER_PROMPT_P2
    assert prompts.VERIFIER_SYSTEM["v1"] not in client.systems                  # the study prompt is not used
    assert all(s == ac.VERIFIER_PROMPT_P2 or s in prompts.PLANNER_SYSTEM.values()
               or s in prompts.ADJUDICATOR_SYSTEM.values() for s in client.systems)


def test_the_thinking_variant_falls_back_to_the_given_client_when_it_is_not_local_ollama():
    client = Recorder()
    model = ac.ThinkingStudioModel(client)
    assert model.thinking is None
    verdict = ac.StudioArmCThinkingVerifier(client).verify(CLAIM, ev(NEU), search=Store().rank)
    assert verdict.label == INSUFFICIENT


def test_candidates_are_opt_in_and_the_default_is_unchanged():
    assert STUDIO_VERIFIER == "arm_c_studio" and VERIFIERS["arm_c_studio"] is StudioArmCVerifier
    assert VERIFIERS["arm_c_studio_qr"] is ac.StudioArmCQuoteRepairVerifier


# ------------------------------------------------------------------------------------------------ round 3 candidate
def test_the_corrected_judge_differs_from_the_default_only_in_batch_size_and_source_cap():
    import dataclasses
    from docqa.agentic import frozen_settings
    from test_agentic_verifier import FakeModel
    candidate = ac.StudioArmCAllPassagesVerifier(FakeModel({}))
    frozen = frozen_settings()
    assert candidate.settings.verifier_batch == 1 and frozen.verifier_batch == 8
    assert dataclasses.replace(candidate.settings, verifier_batch=frozen.verifier_batch) == frozen
    assert candidate.verify_per_source == 8 and not hasattr(StudioArmCVerifier(FakeModel({})), "verify_per_source")
    assert (candidate.accept_unsided, candidate.require_complete_assessment, candidate.check_cited_stance) == \
        (False, True, True)                                                             # Studio rules 1-3 kept


def test_the_corrected_judge_asks_the_verifier_about_every_passage_separately():
    from test_agentic_verifier import CON, FakeModel
    model = FakeModel({SUP: "SUPPORTS", NEU: "NEUTRAL", CON: "NEUTRAL"})
    verdict = ac.StudioArmCAllPassagesVerifier(model).verify(CLAIM, ev(SUP, NEU, CON), search=Store().rank)
    verifier_calls = [r for r in model.requests if r["role"] == "verifier"]
    assert len(verifier_calls) == 3 and all(r["user"].count("[P") == 1 for r in verifier_calls)
    assert verdict.label == SUPPORTED and verdict.agentic_trace["unlisted_passages"] == []
    default = FakeModel({SUP: "SUPPORTS", NEU: "NEUTRAL", CON: "NEUTRAL"})
    StudioArmCVerifier(default).verify(CLAIM, ev(SUP, NEU, CON), search=Store().rank)
    assert len([r for r in default.requests if r["role"] == "verifier"]) == 1           # the default batches


class CapStore:
    """Records the ranking options the orchestrator asks for (passages from one source)."""

    def __init__(self, sources, models=None):
        self.calls = []
        self.sources = {s.id: s for s in sources}

    def rank(self, query, k=6, per_source=2):
        from webrag.evidence import Passage
        self.calls.append({"k": k, "per_source": per_source})
        return [Passage(f"S1:p1:{i}", "S1", 1, text, 1.0) for i, text in enumerate((SUP, NEU, NEU), 1)][:per_source]


def run_verify(verifier_name):
    from test_agentic_verifier import FakeModel
    from webrag.orchestrator import Orchestrator, Run, Settings
    from webrag.researcher import FULL_TEXT, Source
    model = FakeModel({SUP: "SUPPORTS"})
    orch = Orchestrator(Settings(save=False, verifier=verifier_name), client=model, store_factory=CapStore)
    source = Source("S1", "t", "https://example.org/a", "evaluation-packet", "web", "c1:1", FULL_TEXT, SUP, origin="S1")
    store = CapStore([source])
    run = Run(CLAIM, "verify", lambda event: None)
    orch._verify_mode(run, store, CLAIM)
    return store.calls, run.data


def test_verify_mode_lifts_the_source_cap_only_for_the_corrected_judge():
    import docqa.agentic_candidates  # noqa: F401
    default_calls, default_data = run_verify("arm_c_studio")
    candidate_calls, candidate_data = run_verify("arm_c_studio_all")
    assert default_calls[0] == {"k": 8, "per_source": 2}                                 # unchanged default
    assert candidate_calls[0] == {"k": 8, "per_source": 8}
    assert len(default_data["evidence_pool"]) == 2 and len(candidate_data["evidence_pool"]) == 3
