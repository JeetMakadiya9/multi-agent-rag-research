"""Offline tests for the opt-in conflict-adjudicated verifier (docqa/adjudicate.py) and its handling in the WebRAG
application (orchestrator, deep research, studio vocabulary). A scripted fake model replaces Ollama; nothing is
sent anywhere."""
import json

import pytest

from docqa import adjudicate
from docqa.adjudicate import (ADJUDICATED, ADJUDICATOR_PROMPT, QUOTE_PROMPT_V3, AdjudicatedQuoteStanceV3Verifier,
                              AdjudicatedQuoteStanceVerifier)
from docqa.index import Evidence
from docqa.llm import LLMError, LLMResponse
from docqa.verify import (CONTRADICTED, INSUFFICIENT, QUOTE_PROMPT_V2, SUPPORTED, VERIFIERS, QuoteStanceVerifier)

CLAIM = "The bridge opened in 1932."
SUP = Evidence("E1", "doc-a", 1, "The bridge opened to traffic in 1932 after six years of construction.")
CON = Evidence("E2", "doc-b", 1, "The bridge opened in 1934, two years later than first planned.")
NEU = Evidence("E3", "doc-c", 1, "The city is known for its many museums and parks.")


def stance(s, quote, same=True):
    return json.dumps({"quote": quote, "same_subject_and_property": same, "stance": s})


class FakeModel:
    """Answers per-passage stance requests from ``stances`` (keyed by passage text) and adjudicator requests from
    the ``adjudications`` queue (each item a reply string, an LLMResponse-like tuple or an exception)."""

    def __init__(self, stances, adjudications=()):
        self.stances, self.adjudications = dict(stances), list(adjudications)
        self.requests = []

    def chat(self, messages, *, response_format=None, temperature=0.0, num_predict=None, timeout=None):
        system, user = messages[0]["content"], messages[1]["content"]
        self.requests.append({"system": system, "user": user, "schema": response_format})
        if system == ADJUDICATOR_PROMPT:
            item = self.adjudications.pop(0)
            if isinstance(item, Exception):
                raise item
            text, reason = item if isinstance(item, tuple) else (item, "stop")
            return LLMResponse(text, reason, 10, 100, 0.5)
        for passage, reply in self.stances.items():
            if passage in user:
                return LLMResponse(reply, "stop", 10, 100, 0.2)
        raise AssertionError("unexpected request")

    def adjudicator_requests(self):
        return [r for r in self.requests if r["system"] == ADJUDICATOR_PROMPT]


def conflicting_model(adjudications):
    return FakeModel({SUP.text: stance("SUPPORTS", "The bridge opened to traffic in 1932"),
                      CON.text: stance("CONTRADICTS", "The bridge opened in 1934"),
                      NEU.text: stance("NEUTRAL", "")}, adjudications)


def decision(d, passage="P1", quote="The bridge opened to traffic in 1932 after six years of construction."):
    return json.dumps({"decision": d, "passage": passage, "quote": quote})


SUPPORTED_BOTH_ORDERS = [decision("SUPPORTED", "P1"), decision("SUPPORTED", "P2")]       # P2 = SUP when reversed


# ------------------------------------------------------------------------------------------- registry and defaults
def test_new_modes_are_registered_and_the_previous_modes_are_unchanged():
    assert VERIFIERS["quote_stance"] is QuoteStanceVerifier
    assert VERIFIERS["quote_stance_adjudicated"] is AdjudicatedQuoteStanceVerifier
    assert VERIFIERS["quote_stance_v3_adjudicated"] is AdjudicatedQuoteStanceV3Verifier
    assert {"frozen_b", "quote_stance_v1", "property_stance", "checked_stance"} <= set(VERIFIERS)


def test_the_application_default_stays_quote_stance():
    from webrag.orchestrator import Settings
    assert Settings().verifier == "quote_stance"


def test_stance_prompts():
    assert AdjudicatedQuoteStanceVerifier(FakeModel({})).prompt == QUOTE_PROMPT_V2      # v2 judgement unchanged
    v3 = AdjudicatedQuoteStanceV3Verifier(FakeModel({}))
    assert v3.prompt == QUOTE_PROMPT_V3 and "no effect or no association" in v3.prompt
    with pytest.raises(ValueError):
        AdjudicatedQuoteStanceVerifier(FakeModel({}), stance_version=7)


# ------------------------------------------------------------------------------------------- trigger conditions
def test_unanimous_evidence_is_never_adjudicated():
    model = FakeModel({SUP.text: stance("SUPPORTS", "The bridge opened to traffic in 1932"), NEU.text: stance("NEUTRAL", "")})
    verdict = AdjudicatedQuoteStanceVerifier(model).verify(CLAIM, [SUP, NEU])
    assert verdict.label == SUPPORTED and verdict.sub_state == "" and model.adjudicator_requests() == []


def test_missing_or_neutral_evidence_is_not_adjudicated_and_stays_insufficient():
    model = FakeModel({NEU.text: stance("NEUTRAL", "")})
    verifier = AdjudicatedQuoteStanceVerifier(model)
    neutral = verifier.verify(CLAIM, [NEU])
    assert (neutral.label, neutral.sub_state) == (INSUFFICIENT, "NO_DECISIVE_EVIDENCE")
    empty = verifier.verify(CLAIM, [])
    assert (empty.label, empty.sub_state) == (INSUFFICIENT, "NO_EVIDENCE")
    assert model.adjudicator_requests() == []


def test_unassessable_passages_are_not_adjudicated():
    model = FakeModel({SUP.text: stance("SUPPORTS", "The bridge opened to traffic in 1932"), CON.text: "not json"})
    verdict = AdjudicatedQuoteStanceVerifier(model).verify(CLAIM, [SUP, CON])
    assert (verdict.label, verdict.sub_state) == (INSUFFICIENT, "ASSESSMENT_FAILED")
    assert model.adjudicator_requests() == []


def test_an_ungrounded_stance_cannot_create_a_conflict():
    model = FakeModel({SUP.text: stance("SUPPORTS", "The bridge opened to traffic in 1932"),
                       CON.text: stance("CONTRADICTS", "a sentence that is not in the passage at all")})
    verdict = AdjudicatedQuoteStanceVerifier(model).verify(CLAIM, [SUP, CON])
    assert verdict.label == SUPPORTED and model.adjudicator_requests() == []


# ------------------------------------------------------------------------------------------- adjudication outcomes
def test_conflict_is_adjudicated_once_and_the_disagreement_stays_visible():
    model = conflicting_model(SUPPORTED_BOTH_ORDERS)
    verdict = AdjudicatedQuoteStanceVerifier(model).verify(CLAIM, [SUP, CON, NEU])
    assert (verdict.label, verdict.sub_state) == (SUPPORTED, ADJUDICATED)
    assert verdict.stances == {"E1": "SUPPORTS", "E2": "CONTRADICTS", "E3": "NEUTRAL"}    # both sides kept
    assert len(model.adjudicator_requests()) == 2 and verdict.calls == 5            # 3 passages + 2 orders
    assert "reverse order" in verdict.note
    assert "E1" in verdict.note and verdict.parse_ok


def test_contradicted_decision_must_cite_a_contradicting_passage():
    con_quote = "The bridge opened in 1934, two years later"
    model = conflicting_model([decision("CONTRADICTED", "P2", con_quote), decision("CONTRADICTED", "P1", con_quote)])
    verdict = AdjudicatedQuoteStanceVerifier(model).verify(CLAIM, [SUP, CON, NEU])
    assert (verdict.label, verdict.sub_state) == (CONTRADICTED, ADJUDICATED)


@pytest.mark.parametrize("reply,reason", [
    (decision("UNRESOLVED", "NONE", ""), "unresolved"),
    (decision("SUPPORTED", "P2", "The bridge opened in 1934, two years later"), "judged CONTRADICTS"),
    (decision("SUPPORTED", "P9"), "not among the passages shown"),
    (decision("SUPPORTED", "NONE"), "not among the passages shown"),
    (decision("SUPPORTED", "P1", "The bridge was built of steel and painted red"), "quote is not in"),
])
def test_unusable_adjudications_leave_the_conflict_undecided(reply, reason):
    model = conflicting_model([reply])
    verdict = AdjudicatedQuoteStanceVerifier(model).verify(CLAIM, [SUP, CON, NEU])
    assert (verdict.label, verdict.sub_state) == (INSUFFICIENT, "CONFLICTING")
    assert reason in verdict.note


def test_malformed_output_is_retried_once_then_abstains():
    model = conflicting_model(["not json", '{"decision": "MAYBE"}'])
    verdict = AdjudicatedQuoteStanceVerifier(model).verify(CLAIM, [SUP, CON, NEU])
    assert (verdict.label, verdict.sub_state) == (INSUFFICIENT, "CONFLICTING")
    assert len(model.adjudicator_requests()) == 2 and "unusable after one retry" in verdict.note


def test_truncated_output_is_never_accepted():
    cut = (decision("SUPPORTED"), "length")
    model = conflicting_model([cut, cut])
    verdict = AdjudicatedQuoteStanceVerifier(model).verify(CLAIM, [SUP, CON, NEU])
    assert (verdict.label, verdict.sub_state) == (INSUFFICIENT, "CONFLICTING") and verdict.truncated


def test_a_retry_after_malformed_output_can_succeed():
    model = conflicting_model(["{", decision("SUPPORTED"), decision("SUPPORTED", "P2")])
    verdict = AdjudicatedQuoteStanceVerifier(model).verify(CLAIM, [SUP, CON, NEU])
    assert (verdict.label, verdict.sub_state) == (SUPPORTED, ADJUDICATED)


def test_a_service_failure_during_adjudication_abstains_without_raising():
    model = conflicting_model([LLMError("timeout", "no response within 180s")])
    verdict = AdjudicatedQuoteStanceVerifier(model).verify(CLAIM, [SUP, CON, NEU])
    assert (verdict.label, verdict.sub_state) == (INSUFFICIENT, "CONFLICTING")
    assert "timeout" in verdict.note


def test_a_service_failure_during_passage_judgement_propagates_as_before():
    class Down(FakeModel):
        def chat(self, *args, **kwargs):
            raise LLMError("unreachable", "connection refused")
    with pytest.raises(LLMError):
        AdjudicatedQuoteStanceVerifier(Down({})).verify(CLAIM, [SUP])
    with pytest.raises(LLMError):
        QuoteStanceVerifier(Down({})).verify(CLAIM, [SUP])


# ------------------------------------------------------------------------------------------- prompt and passage labels
def test_adjudicator_prompt_has_no_example_label_and_the_schema_limits_labels_to_the_shown_passages():
    assert not any(f"P{i}" in ADJUDICATOR_PROMPT for i in range(1, 10))
    model = conflicting_model(SUPPORTED_BOTH_ORDERS)
    AdjudicatedQuoteStanceVerifier(model).verify(CLAIM, [SUP, CON, NEU])
    request = model.adjudicator_requests()[0]
    assert request["schema"]["properties"]["passage"]["enum"] == ["P1", "P2", "NONE"]
    assert NEU.text not in request["user"]                       # only passages that took a side are shown
    assert "[P1] (judged supporting)" in request["user"] and "[P2] (judged contradicting)" in request["user"]


def test_at_most_six_passages_are_shown_with_both_sides_present():
    stances = {f"E{i}": "SUPPORTS" for i in range(1, 7)} | {f"E{i}": "CONTRADICTS" for i in range(7, 10)}
    evidence = [Evidence(f"E{i}", "d", 1, f"text {i}") for i in range(1, 10)]
    shown = adjudicate._shown(evidence, stances)
    assert len(shown) == 6 and [label for label, _, _ in shown] == [f"P{i}" for i in range(1, 7)]
    assert sum(s == "SUPPORTS" for _, _, s in shown) == 3 and sum(s == "CONTRADICTS" for _, _, s in shown) == 3


# ------------------------------------------------------------------------------------------- application handling
def test_deep_research_labels_and_cautions_adjudicated_statements():
    from webrag import deep
    passages = [{"stance": "SUPPORTS", "source_id": "S1", "key": "S1:p1", "text": CLAIM},
                {"stance": "CONTRADICTS", "source_id": "S2", "key": "S2:p1", "text": "It opened in 1934."}]
    claim = {"label": SUPPORTED, "sub_state": ADJUDICATED, "text": CLAIM, "passages": passages, "parse_ok": True}
    assert deep.strength(claim, {}) == "adjudicated"
    assert "adjudicated_conflict" in {c["kind"] for c in deep.cautions(claim)}       # kept out of the answer
    assert deep.strength(dict(claim, label=CONTRADICTED), {}) == "adjudicated"
    assert "adjudicated" in deep.EVIDENCE_POLICY["labels"]
    assert "adjudicated_conflict" in deep.EVIDENCE_POLICY["cautions"]
    plain = dict(claim, sub_state="")
    assert deep.strength(plain, {}) != "adjudicated"
    assert "adjudicated_conflict" not in {c["kind"] for c in deep.cautions(plain)}


def test_verify_mode_records_the_disagreement_for_an_adjudicated_verdict():
    from docqa.verify import Verdict
    from webrag.orchestrator import Orchestrator, Run

    class Fixed:
        name = "fixed"

        def verify(self, claim, evidence):
            return Verdict(SUPPORTED, ADJUDICATED, stances={e.id: s for e, s in zip(evidence, ("SUPPORTS", "CONTRADICTS"))},
                           note="adjudicated: E1 judged decisive over conflicting evidence")

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
    assert (claim["label"], claim["sub_state"]) == (SUPPORTED, ADJUDICATED)
    assert claim["verifier_note"].startswith("adjudicated: E1")
    assert run.data["disagreements"] and "adjudication" in run.data["disagreements"][0]["text"]
    assert set(run.data["disagreements"][0]["sources"]) == {"S1", "S2"}


# ------------------------------------------------------------------------------------------- order check
def test_the_reverse_order_call_shows_the_passages_reversed_and_relabelled():
    model = conflicting_model(SUPPORTED_BOTH_ORDERS)
    AdjudicatedQuoteStanceVerifier(model).verify(CLAIM, [SUP, CON, NEU])
    first, second = model.adjudicator_requests()
    assert first["user"].index(SUP.text) < first["user"].index(CON.text)
    assert second["user"].index(CON.text) < second["user"].index(SUP.text)
    assert "[P1] (judged contradicting)" in second["user"]


@pytest.mark.parametrize("second", [decision("UNRESOLVED", "NONE", ""),
                                    decision("CONTRADICTED", "P1", "The bridge opened in 1934, two years later"),
                                    "not json"])
def test_a_decision_not_reproduced_in_reverse_order_leaves_the_conflict_undecided(second):
    replies = [decision("SUPPORTED", "P1"), second] + (["not json"] if second == "not json" else [])
    model = conflicting_model(replies)
    verdict = AdjudicatedQuoteStanceVerifier(model).verify(CLAIM, [SUP, CON, NEU])
    assert (verdict.label, verdict.sub_state) == (INSUFFICIENT, "CONFLICTING")
    assert "not reproduced" in verdict.note


def test_the_no_check_variant_reproduces_the_first_version():
    model = conflicting_model([decision("SUPPORTED")])
    verdict = VERIFIERS["quote_stance_adjudicated_nocheck"](model).verify(CLAIM, [SUP, CON, NEU])
    assert (verdict.label, verdict.sub_state) == (SUPPORTED, ADJUDICATED)
    assert len(model.adjudicator_requests()) == 1
