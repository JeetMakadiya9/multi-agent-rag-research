"""Offline tests for the opt-in rank tie-break verifier (docqa/ranked.py) and its handling in the WebRAG application
(orchestrator, deep research). A scripted fake model replaces Ollama; nothing is sent anywhere."""
import json

import pytest

from docqa.adjudicate import ADJUDICATED
from docqa.index import Evidence
from docqa.llm import LLMError, LLMResponse
from docqa.ranked import DECIDED_CONFLICTS, RANK_TIEBREAK, RankedQuoteStanceVerifier
from docqa.verify import CONTRADICTED, INSUFFICIENT, SUPPORTED, VERIFIERS, QuoteStanceVerifier, Verdict

CLAIM = "The bridge opened in 1932."
SUP = Evidence("E1", "doc-a", 1, "The bridge opened to traffic in 1932 after six years of construction.")
CON = Evidence("E2", "doc-b", 1, "The bridge opened in 1934, two years later than first planned.")
NEU = Evidence("E3", "doc-c", 1, "The city is known for its many museums and parks.")


def stance(s, quote, same=True):
    return json.dumps({"quote": quote, "same_subject_and_property": same, "stance": s})


class FakeModel:
    """Answers per-passage stance requests from ``stances`` (keyed by passage text); counts every request."""

    def __init__(self, stances):
        self.stances, self.requests = dict(stances), 0

    def chat(self, messages, *, response_format=None, temperature=0.0, num_predict=None, timeout=None):
        self.requests += 1
        user = messages[1]["content"]
        for passage, reply in self.stances.items():
            if passage in user:
                if isinstance(reply, Exception):
                    raise reply
                return LLMResponse(reply, "stop", 10, 100, 0.2)
        raise AssertionError("unexpected request")


def conflicting():
    return FakeModel({SUP.text: stance("SUPPORTS", "The bridge opened to traffic in 1932"),
                      CON.text: stance("CONTRADICTS", "The bridge opened in 1934"),
                      NEU.text: stance("NEUTRAL", "")})


# ------------------------------------------------------------------------------------------- registry and defaults
def test_registered_as_an_opt_in_mode_and_the_default_is_unchanged():
    from webrag.orchestrator import Settings
    assert VERIFIERS["quote_stance_ranked"] is RankedQuoteStanceVerifier
    assert VERIFIERS["quote_stance"] is QuoteStanceVerifier
    assert Settings().verifier == "quote_stance"
    assert DECIDED_CONFLICTS == {ADJUDICATED: "adjudicated", RANK_TIEBREAK: "decided by rank"}


# ------------------------------------------------------------------------------------------- the decision rule
@pytest.mark.parametrize("order, label, first", [((SUP, CON, NEU), SUPPORTED, "E1"),
                                                 ((CON, SUP, NEU), CONTRADICTED, "E2"),
                                                 ((NEU, CON, SUP), CONTRADICTED, "E2")])
def test_a_conflict_is_decided_by_the_first_sided_passage_in_evidence_order(order, label, first):
    model = conflicting()
    verdict = RankedQuoteStanceVerifier(model).verify(CLAIM, list(order))
    assert (verdict.label, verdict.sub_state) == (label, RANK_TIEBREAK)
    assert f"{first} (the first passage with a stance)" in verdict.note and "no extra model call" in verdict.note
    assert verdict.stances == {"E1": "SUPPORTS", "E2": "CONTRADICTS", "E3": "NEUTRAL"}   # both sides stay visible
    assert model.requests == 3 and verdict.calls == 3                                    # one call per passage only


def test_everything_except_a_conflict_matches_quote_stance():
    cases = [[SUP, NEU], [CON, NEU], [NEU], []]
    for evidence in cases:
        ranked = RankedQuoteStanceVerifier(conflicting()).verify(CLAIM, evidence)
        plain = QuoteStanceVerifier(conflicting()).verify(CLAIM, evidence)
        assert (ranked.label, ranked.sub_state, ranked.stances) == (plain.label, plain.sub_state, plain.stances)
    assert RankedQuoteStanceVerifier(conflicting()).verify(CLAIM, []).sub_state == "NO_EVIDENCE"


def test_an_ungrounded_stance_cannot_create_a_conflict():
    model = FakeModel({SUP.text: stance("SUPPORTS", "The bridge opened to traffic in 1932"),
                       CON.text: stance("CONTRADICTS", "a sentence that is not in the passage")})
    verdict = RankedQuoteStanceVerifier(model).verify(CLAIM, [CON, SUP])
    assert (verdict.label, verdict.sub_state) == (SUPPORTED, "")        # the unquoted stance became NEUTRAL first


def test_unassessable_evidence_is_never_decided():
    model = FakeModel({SUP.text: stance("SUPPORTS", "The bridge opened to traffic in 1932"),
                       CON.text: "not json"})
    verdict = RankedQuoteStanceVerifier(model).verify(CLAIM, [SUP, CON])
    assert (verdict.label, verdict.sub_state, verdict.parse_ok) == (INSUFFICIENT, "ASSESSMENT_FAILED", False)


def test_a_service_failure_propagates_as_with_quote_stance():
    model = FakeModel({SUP.text: LLMError("timeout", "no response within 180s")})
    with pytest.raises(LLMError):
        RankedQuoteStanceVerifier(model).verify(CLAIM, [SUP])


def test_quote_attestation_wraps_the_ranked_verifier_without_changing_its_verdict():
    from docqa.attestation import supports_attestation, with_attestation
    assert supports_attestation(VERIFIERS["quote_stance_ranked"])
    attested = with_attestation(RankedQuoteStanceVerifier(conflicting()))
    assert isinstance(attested, RankedQuoteStanceVerifier)
    verdict = attested.verify(CLAIM, [SUP, CON])
    assert (verdict.label, verdict.sub_state) == (SUPPORTED, RANK_TIEBREAK)


# ------------------------------------------------------------------------------------------- application handling
def test_deep_research_labels_cautions_and_reviews_rank_decided_statements():
    from webrag import deep
    passages = [{"stance": "SUPPORTS", "source_id": "S1", "key": "S1:p1", "text": CLAIM},
                {"stance": "CONTRADICTS", "source_id": "S2", "key": "S2:p1", "text": "It opened in 1934."}]
    claim = {"label": SUPPORTED, "sub_state": RANK_TIEBREAK, "text": CLAIM, "passages": passages, "parse_ok": True}
    assert deep.strength(claim, {}) == "rank decided"
    kinds = {c["kind"] for c in deep.cautions(claim)}
    assert "rank_decided_conflict" in kinds and "adjudicated_conflict" not in kinds    # kept out of the answer
    assert deep.strength(dict(claim, label=CONTRADICTED), {}) == "rank decided"
    assert "rank decided" in deep.EVIDENCE_POLICY["labels"]
    assert "rank_decided_conflict" in deep.EVIDENCE_POLICY["cautions"]
    adjudicated = dict(claim, sub_state=ADJUDICATED)
    assert deep.strength(adjudicated, {}) == "adjudicated"
    assert "rank_decided_conflict" not in {c["kind"] for c in deep.cautions(adjudicated)}
    plain = dict(claim, sub_state="")
    assert deep.strength(plain, {}) != "rank decided" and not {c["kind"] for c in deep.cautions(plain)} & {
        "rank_decided_conflict", "adjudicated_conflict"}


def test_verify_mode_records_the_disagreement_for_a_rank_decided_verdict():
    from webrag.orchestrator import Orchestrator, Run

    class Fixed:
        name = "fixed"

        def verify(self, claim, evidence):
            return Verdict(SUPPORTED, RANK_TIEBREAK, stances={e.id: s for e, s in zip(evidence, ("SUPPORTS", "CONTRADICTS"))},
                           note="conflicting evidence decided by rank: E1 (the first passage with a stance) SUPPORTS")

    class Passage:
        def __init__(self, sid, text):
            self.source_id, self.text, self.key = sid, text, f"{sid}:p1"

        def as_evidence(self, eid):
            return Evidence(eid, self.source_id, 1, self.text)

    class Store:
        sources = {"S1": type("Src", (), {"origin": "a.example"})(), "S2": type("Src", (), {"origin": "b.example"})()}

        def rank(self, claim, k):
            return [Passage("S1", SUP.text), Passage("S2", CON.text)]

    orch = Orchestrator.__new__(Orchestrator)
    orch.verifier, orch.settings = Fixed(), type("S", (), {"question_passages": 4})()
    orch._passage_record = lambda p, store: {"source_id": p.source_id, "key": p.key, "text": p.text}
    run = Run(CLAIM, "verify", lambda event: None)
    orch._verify_mode(run, Store(), CLAIM)
    claim = run.data["claims"][0]
    assert (claim["label"], claim["sub_state"]) == (SUPPORTED, RANK_TIEBREAK)
    text = run.data["disagreements"][0]["text"]
    assert "rank tie-break" in text and "adjudication" not in text and "(SUPPORTED)" in text
    assert set(run.data["disagreements"][0]["sources"]) == {"S1", "S2"}
