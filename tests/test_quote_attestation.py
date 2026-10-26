"""Record-only quote attestation (docqa/attestation.py) on its own: the record, the verifier mixin and its data flow.
A scripted fake model client answers per passage; no model, no network. docqa/verify.py is used unchanged.
Application integration (adjudicated verifier, Settings, orchestrator, report): tests/test_quote_attestation_integration.py."""
import dataclasses
import json
from types import SimpleNamespace

import pytest

from docqa.attestation import (ATTESTATION_RECORD_VERSION, attestation_record, attestations, supports_attestation,
                               with_attestation)
from docqa.index import Evidence
from docqa.verify import VERIFIERS, FrozenBVerifier, QuoteStanceVerifier, Verdict, quote_in_passage


# ------------------------------------------------------------------------------------------- the attestation record
def record(quote, passage):
    return attestation_record(quote, passage)


def test_exact_quote_is_located_with_original_offsets():
    passage = "The bridge opened in 1932. It is 503 metres long."
    r = record("It is 503 metres long.", passage)
    assert (r["status"], r["tier"], r["reason"]) == ("LOCATED", "EXACT", "UNIQUE_EXACT_MATCH")
    [span] = r["spans"]
    assert passage[span["start"]:span["end"]] == span["text"] == "It is 503 metres long."


def test_whitespace_variant_is_located_in_the_canonical_tier_and_maps_back():
    passage = "The tower was completed\n   in 1889 for the fair."
    r = record("The tower was completed in 1889", passage)
    assert (r["status"], r["tier"]) == ("LOCATED", "CANONICAL")
    span = r["spans"][0]
    assert passage[span["start"]:span["end"]] == span["text"]


@pytest.mark.parametrize("quote, passage, reason", [
    ("The tower is tall and 300 metres was built in 1889.",          # reordered words
     "The tower is 300 metres tall and was built in 1889.", "NO_CONTIGUOUS_MATCH"),
    ("The tower was built in 1889.",                                  # non-contiguous words
     "The tower is 300 metres tall and was built in 1889.", "NO_CONTIGUOUS_MATCH"),
    ("the drug was effective in adults",                             # a dropped negation
     "In the trial the drug was not effective in adults.", "NO_CONTIGUOUS_MATCH"),
    ("effective dose was low", "The ineffective dose was low.", "BOUNDARY_VIOLATION"),   # inside a longer word
])
def test_quotes_the_legacy_check_accepts_are_unattested(quote, passage, reason):
    assert quote_in_passage(quote, passage) is True                   # the frozen check's known tolerance
    r = record(quote, passage)
    assert r["status"] == "UNATTESTED" and r["reason"] == reason and r["spans"] == []


def test_repeated_sentence_is_ambiguous_and_every_span_is_recorded():
    passage = "Sales rose. Costs fell. Sales rose."
    r = record("Sales rose.", passage)
    assert r["status"] == "AMBIGUOUS" and len(r["spans"]) == 2
    assert all(passage[s["start"]:s["end"]] == s["text"] for s in r["spans"])


def test_negation_cue_is_recorded_without_changing_the_status():
    r = record("effective in adults.", "The drug was not effective in adults.")
    assert r["status"] == "LOCATED" and r["spans"][0]["negation_cue"] == "not"


@pytest.mark.parametrize("quote", ["", "   ", "\n"])
def test_empty_quote_is_empty(quote):
    assert record(quote, "Some passage.")["status"] == "EMPTY"


def test_record_is_plain_json():
    r = record("Sales rose.", "Sales rose.")
    assert json.loads(json.dumps(r)) == r and r["attestation_version"] and r["canonicalization"]


# ------------------------------------------------------------------------------------------- scripted verifier
class ScriptedClient:
    """Answers each passage with the next scripted reply for that passage (matched by a marker in its text)."""

    def __init__(self, replies):
        self.replies = {k: list(v) for k, v in replies.items()}
        self.calls = []

    def chat(self, messages, **kwargs):
        user = messages[-1]["content"]
        marker = next(k for k in self.replies if k in user)
        self.calls.append(marker)
        reply = self.replies[marker].pop(0)
        text = reply if isinstance(reply, str) else json.dumps(reply)
        return SimpleNamespace(text=text, seconds=0.01, truncated=False)


def stance(quote, s, same=True):
    return {"quote": quote, "same_subject_and_property": same, "stance": s}


P1 = "Alpha passage. The bridge opened in 1932."
P2 = "Beta passage. The bridge was not opened in 1932 but later."
EVIDENCE = [Evidence("E1", "S1", 1, P1), Evidence("E2", "S2", 1, P2)]
CLAIM = "The bridge opened in 1932."


def replies():
    return {"Alpha": [stance("The bridge opened in 1932.", "SUPPORTS")],
            # a reordered "quote" the frozen check accepts (all words present): recorded as UNATTESTED
            "Beta": [stance("The bridge was opened in 1932 not", "SUPPORTS")]}


def verdict_fields(v):
    return {f.name: getattr(v, f.name) for f in dataclasses.fields(v) if f.name != "seconds"}


def test_attestation_never_changes_the_verdict():
    plain = QuoteStanceVerifier(ScriptedClient(replies())).verify(CLAIM, EVIDENCE)
    attested_verifier = with_attestation(QuoteStanceVerifier(ScriptedClient(replies())))
    attested = attested_verifier.verify(CLAIM, EVIDENCE)
    assert verdict_fields(attested) == verdict_fields(plain)
    assert attested.label == "SUPPORTED"                              # the frozen check accepted both quotes
    assert attestations(plain) == {}


def test_records_are_tied_to_the_right_passage_and_quote():
    verdict = with_attestation(QuoteStanceVerifier(ScriptedClient(replies()))).verify(CLAIM, EVIDENCE)
    found = attestations(verdict)
    assert set(found) == {"E1", "E2"}
    [e1], [e2] = found["E1"], found["E2"]
    assert e1["quote"] == "The bridge opened in 1932." and e1["attestation"]["status"] == "LOCATED"
    assert P1[e1["attestation"]["spans"][0]["start"]:e1["attestation"]["spans"][0]["end"]] == e1["quote"]
    assert e2["quote"] == "The bridge was opened in 1932 not"
    assert e2["legacy_quote_check"] is True and e2["attestation"]["status"] == "UNATTESTED"
    assert e1["stance"] == verdict.stances["E1"] and e2["stance"] == verdict.stances["E2"]
    assert e1["record_version"] == ATTESTATION_RECORD_VERSION and e1["checked_text"] == CLAIM


def test_the_quote_after_a_retry_is_the_one_attested():
    client = ScriptedClient({"Alpha": ["not json", stance("The bridge opened in 1932.", "SUPPORTS")],
                             "Beta": [stance("", "NEUTRAL")]})
    verdict = with_attestation(QuoteStanceVerifier(client)).verify(CLAIM, EVIDENCE)
    assert client.calls == ["Alpha", "Alpha", "Beta"] and verdict.calls == 3
    assert attestations(verdict)["E1"][0]["attestation"]["status"] == "LOCATED"
    assert attestations(verdict)["E2"][0]["attestation"]["status"] == "EMPTY"


def test_a_passage_without_usable_output_is_recorded_as_not_assessed():
    client = ScriptedClient({"Alpha": ["bad", "still bad"], "Beta": [stance("", "NEUTRAL")]})
    verdict = with_attestation(QuoteStanceVerifier(client)).verify(CLAIM, EVIDENCE)
    assert verdict.sub_state == "ASSESSMENT_FAILED"                    # unchanged frozen behaviour
    assert attestations(verdict)["E1"] == [{"record_version": ATTESTATION_RECORD_VERSION, "checked_text": CLAIM,
                                            "assessed": False}]


def test_a_shared_verifier_keeps_no_request_state():
    client = ScriptedClient({"Alpha": [stance("The bridge opened in 1932.", "SUPPORTS")] * 2,
                             "Beta": [stance("", "NEUTRAL")] * 2})
    verifier = with_attestation(QuoteStanceVerifier(client))
    first = verifier.verify(CLAIM, EVIDENCE)
    second = verifier.verify(CLAIM, EVIDENCE[:1])
    assert set(attestations(first)) == {"E1", "E2"} and set(attestations(second)) == {"E1"}
    assert len(attestations(first)["E1"]) == 1                        # the second call did not append to the first
    assert "quote_attestations" not in vars(verifier)


def test_a_passage_judged_twice_keeps_both_entries_in_order():
    verifier = with_attestation(QuoteStanceVerifier(ScriptedClient(
        {"Alpha": [stance("Alpha passage.", "NEUTRAL"), stance("The bridge opened in 1932.", "SUPPORTS")]})))
    verdict, notes = Verdict("INSUFFICIENT"), []
    verifier.judge_passage("fact one", EVIDENCE[0], verdict, notes)
    verifier.judge_passage("fact two", EVIDENCE[0], verdict, notes)
    entries = attestations(verdict)["E1"]
    assert [e["checked_text"] for e in entries] == ["fact one", "fact two"]




# ------------------------------------------------------------------------------------------- configuration
def test_only_quote_based_verifiers_qualify():
    with pytest.raises(ValueError):
        with_attestation(FrozenBVerifier(None))
    assert supports_attestation(VERIFIERS["quote_stance"]) and not supports_attestation(VERIFIERS["frozen_b"])
