"""verifier_v2 (docqa/verifier_v2.py): relevance gate, one structured call per passage, calibrated aggregation,
optional NLI filter and cached judgments. The model and the NLI filter are scripted; nothing is sent anywhere."""
import json

import pytest

from docqa.index import Evidence
from docqa.llm import LLMError, LLMResponse
from docqa.nli import NLIScore, nli_from_env
from docqa.verify import CONTRADICTED, INSUFFICIENT, SUPPORTED, VERIFIERS
from docqa import verifier_v2 as v2
from webrag.model_service import STUDIO_VERIFIER

CLAIM = "Drug X lowers blood pressure in adults."
SUP = "In a randomised trial of 400 adults, drug X lowered systolic blood pressure by 9 mmHg."
SUP2 = "A second trial in 250 adults confirmed that drug X lowered blood pressure after eight weeks."
CON = "In adults, drug X had no effect on blood pressure compared with placebo in a large trial."
NEU = "Drug X is metabolised in the liver and has a half-life of twelve hours in most patients."
KIDS = "In children, drug X raised blood pressure slightly during a short pilot study of 30 patients."


def reply(stance="NEUTRAL", strength="strong", tests=True, gate=(True, True, True), quote=""):
    return json.dumps({"same_population": gate[0], "same_intervention": gate[1], "same_outcome": gate[2],
                       "tests_claim": tests, "stance": stance, "strength": strength, "quote": quote})


def quote_of(text, words=8):
    return " ".join(text.split()[:words])


class Model:
    """Replies chosen by the passage in the prompt (a dict passage -> reply, or a list consumed in order)."""

    def __init__(self, replies, model="qwen3:4b"):
        self.replies, self.model, self.requests = replies, model, []

    def chat(self, messages, *, response_format=None, temperature=0.0, num_predict=None, timeout=None):
        user = messages[1]["content"]
        self.requests.append({"system": messages[0]["content"], "user": user, "format": response_format,
                              "temperature": temperature})
        passage = user.split("Passage:\n", 1)[1].split(v2.RETRY_SUFFIX, 1)[0]
        item = self.replies[passage] if isinstance(self.replies, dict) else self.replies.pop(0)
        if isinstance(item, list):
            item = item.pop(0)
        if isinstance(item, Exception):
            raise item
        return LLMResponse(item, "stop", 20, 200, 0.1)


class FakeNLI:
    model_name = "fake-nli"

    def __init__(self, scores=None, fail=None):
        self.scores, self.fail, self.calls = scores or {}, fail, 0

    def score(self, pairs):
        self.calls += 1
        if self.fail:
            raise RuntimeError(self.fail)
        return [self.scores.get(text, NLIScore(0.2, 0.3, 0.5)) for text, _ in pairs]


def ev(*texts):
    return [Evidence(f"E{i}", f"doc{i}", 1, t) for i, t in enumerate(texts, 1)]


def verifier(model, nli=None, cache=None):
    return v2.VerifierV2(model, nli=nli, cache=cache or v2.VerdictCache())


# ------------------------------------------------------------------------------------- one structured call per passage
def test_each_passage_gets_its_own_call_with_the_json_schema_and_no_passage_ids():
    model = Model({SUP: reply("SUPPORTS", quote=quote_of(SUP)), NEU: reply()})
    verifier(model).verify(CLAIM, ev(SUP, NEU))
    assert len(model.requests) == 2
    for request in model.requests:
        assert request["format"] == v2.SCHEMA and request["temperature"] == 0.0
        assert request["user"].count("Passage:") == 1 and "[P" not in request["user"]


def test_the_schema_puts_the_relevance_gate_before_the_stance():
    order = v2.SCHEMA["required"]
    assert order.index("tests_claim") < order.index("stance") and set(v2.GATE_FIELDS) <= set(order[:3])
    assert v2.SCHEMA["properties"]["stance"]["enum"] == ["SUPPORTS", "CONTRADICTS", "NEUTRAL"]


# ------------------------------------------------------------------------------------------- calibrated aggregation
def test_strong_grounded_support_decides_supported_with_confidence():
    model = Model({SUP: reply("SUPPORTS", quote=quote_of(SUP)), SUP2: reply("SUPPORTS", quote=quote_of(SUP2)),
                   NEU: reply()})
    verdict = verifier(model).verify(CLAIM, ev(SUP, SUP2, NEU))
    assert verdict.label == SUPPORTED and verdict.stances == {"E1": "SUPPORTS", "E2": "SUPPORTS", "E3": "NEUTRAL"}
    assert verdict.confidence == {"level": "high", "agreeing": 2, "opposing": 0, "tested": 3, "judged": 3}
    assert verdict.abstain_reason == ""


def test_a_single_strong_contradiction_decides_with_medium_confidence():
    verdict = verifier(Model({CON: reply("CONTRADICTS", quote=quote_of(CON)), NEU: reply()})).verify(CLAIM, ev(CON, NEU))
    assert verdict.label == CONTRADICTED and verdict.confidence["level"] == "medium"


def test_a_passage_that_does_not_test_the_claim_cannot_contradict_it():
    model = Model({KIDS: reply("CONTRADICTS", tests=False, gate=(False, True, True), quote=quote_of(KIDS)),
                   NEU: reply(tests=False, gate=(True, True, False))})
    verdict = verifier(model).verify(CLAIM, ev(KIDS, NEU))
    assert verdict.label == INSUFFICIENT and verdict.sub_state == "NO_DECISIVE_EVIDENCE"
    assert verdict.stances["E1"] == "NEUTRAL"
    assert verdict.passage_reasons["E1"] == "does not test the claim (different population)"
    assert verdict.abstain_reason.startswith("No passage tested this statement")
    assert verdict.confidence["level"] == "abstained"


def test_a_gate_mismatch_overrides_tests_claim_true():
    model = Model({KIDS: reply("CONTRADICTS", tests=True, gate=(False, True, True), quote=quote_of(KIDS))})
    verdict = verifier(model).verify(CLAIM, ev(KIDS))
    assert verdict.label == INSUFFICIENT and verdict.stances["E1"] == "NEUTRAL"


def test_weak_evidence_is_shown_but_does_not_decide():
    verdict = verifier(Model({CON: reply("CONTRADICTS", "weak", quote=quote_of(CON))})).verify(CLAIM, ev(CON))
    assert verdict.label == INSUFFICIENT and verdict.stances["E1"] == "CONTRADICTS"     # still visible
    assert "weak or indirect" in verdict.abstain_reason


def test_an_ungrounded_quote_demotes_the_stance():
    verdict = verifier(Model({SUP: reply("SUPPORTS", quote="drug X cured every patient in the study")})).verify(
        CLAIM, ev(SUP))
    assert verdict.label == INSUFFICIENT and verdict.stances["E1"] == "NEUTRAL"
    assert verdict.passage_reasons["E1"] == "its quote was not found in the passage"


def test_strong_evidence_both_ways_is_conflicting_and_explained():
    model = Model({SUP: reply("SUPPORTS", quote=quote_of(SUP)), CON: reply("CONTRADICTS", quote=quote_of(CON))})
    verdict = verifier(model).verify(CLAIM, ev(SUP, CON))
    assert verdict.label == INSUFFICIENT and verdict.sub_state == "CONFLICTING"
    assert "both ways" in verdict.abstain_reason


def test_no_evidence_is_reported_with_its_reason():
    verdict = verifier(Model({})).verify(CLAIM, [])
    assert verdict.sub_state == "NO_EVIDENCE" and verdict.abstain_reason == "No evidence was retrieved for this statement."


# ------------------------------------------------------------------------------------------- invalid replies
def test_an_invalid_reply_is_retried_and_a_valid_retry_is_used():
    model = Model({SUP: ["not json", reply("SUPPORTS", quote=quote_of(SUP))]})
    verdict = verifier(model).verify(CLAIM, ev(SUP))
    assert verdict.label == SUPPORTED and verdict.calls == 2
    assert model.requests[1]["user"].endswith(v2.RETRY_SUFFIX)


@pytest.mark.parametrize("bad", ["not json", json.dumps({"stance": "SUPPORTS"}),
                                 reply("MAYBE"), reply(strength="certain"),
                                 json.dumps({"same_population": "yes", "same_intervention": True, "same_outcome": True,
                                             "tests_claim": True, "stance": "NEUTRAL", "strength": "strong",
                                             "quote": ""})])
def test_two_invalid_replies_fail_the_claim_closed_and_are_never_neutral(bad):
    verdict = verifier(Model({SUP: [bad, bad], NEU: reply()})).verify(CLAIM, ev(SUP, NEU))
    assert verdict.sub_state == "ASSESSMENT_FAILED" and not verdict.parse_ok
    assert "E1" not in verdict.stances and verdict.stances == {"E2": "NEUTRAL"}
    assert verdict.abstain_reason == "The verifier's output was unusable, so nothing was decided."


def test_check_reply_accepts_only_complete_valid_objects():
    assert v2.check_reply(reply("supports", quote="x")) ["stance"] == "SUPPORTS"
    assert v2.check_reply("Here it is: " + reply()) is not None                 # wrapped in prose
    for bad in ("", "[]", json.dumps({"stance": "NEUTRAL"}), reply(stance="")):
        assert v2.check_reply(bad) is None


def test_a_service_failure_reaches_the_studio():
    with pytest.raises(LLMError):
        verifier(Model({SUP: LLMError("timeout", "no response")})).verify(CLAIM, ev(SUP))


# ------------------------------------------------------------------------------------------- NLI filter
def test_a_confidently_neutral_passage_is_skipped_without_a_model_call():
    model = Model({SUP: reply("SUPPORTS", quote=quote_of(SUP))})
    nli = FakeNLI({NEU: NLIScore(0.02, 0.95, 0.03), SUP: NLIScore(0.9, 0.05, 0.05)})
    verdict = verifier(model, nli=nli).verify(CLAIM, ev(SUP, NEU))
    assert len(model.requests) == 1 and verdict.label == SUPPORTED
    assert verdict.verifier_trace["cache"]["skipped_by_nli"] == 1 and verdict.verifier_trace["nli"].startswith("on")


def test_a_contradiction_the_nli_model_does_not_see_does_not_decide():
    nli = FakeNLI({CON: NLIScore(0.3, 0.6, 0.1)})
    verdict = verifier(Model({CON: reply("CONTRADICTS", quote=quote_of(CON))}), nli=nli).verify(CLAIM, ev(CON))
    assert verdict.label == INSUFFICIENT and "not confirmed by the NLI check" in verdict.abstain_reason


def test_a_contradiction_the_nli_model_confirms_decides():
    nli = FakeNLI({CON: NLIScore(0.05, 0.15, 0.8)})
    verdict = verifier(Model({CON: reply("CONTRADICTS", quote=quote_of(CON))}), nli=nli).verify(CLAIM, ev(CON))
    assert verdict.label == CONTRADICTED


def test_an_unavailable_nli_model_is_reported_and_the_verifier_still_runs():
    verdict = verifier(Model({SUP: reply("SUPPORTS", quote=quote_of(SUP))}), nli=FakeNLI(fail="not in cache")).verify(
        CLAIM, ev(SUP))
    assert verdict.label == SUPPORTED and verdict.verifier_trace["nli"] == "unavailable: not in cache"


def test_the_nli_setting_can_switch_the_filter_off():
    assert nli_from_env({"WEBRAG_NLI_MODEL": "off"}) is None
    configured = nli_from_env({})
    assert configured.model_name == "cross-encoder/nli-deberta-v3-xsmall" and configured.allow_download is False


# ------------------------------------------------------------------------------------------- cache
def test_a_repeated_claim_is_answered_from_the_cache():
    cache = v2.VerdictCache()
    first = Model({SUP: reply("SUPPORTS", quote=quote_of(SUP))})
    verifier(first, cache=cache).verify(CLAIM, ev(SUP))
    again = Model({})
    verdict = verifier(again, cache=cache).verify(CLAIM, ev(SUP))
    assert again.requests == [] and verdict.label == SUPPORTED and verdict.calls == 0
    assert verdict.verifier_trace["cache"]["hits"] == 1


def test_the_cache_key_includes_the_model_and_skips_invalid_judgments():
    cache = v2.VerdictCache()
    verifier(Model({SUP: reply("SUPPORTS", quote=quote_of(SUP))}), cache=cache).verify(CLAIM, ev(SUP))
    other = Model({SUP: reply("SUPPORTS", quote=quote_of(SUP))}, model="larger-model")
    verifier(other, cache=cache).verify(CLAIM, ev(SUP))
    assert len(other.requests) == 1                                              # another model: judged again
    failing = v2.VerdictCache()
    verifier(Model({NEU: ["bad", "bad"]}), cache=failing).verify(CLAIM, ev(NEU))
    assert failing.get(failing.key("qwen3:4b", CLAIM, NEU, False)) is None


def test_the_cache_is_bounded():
    cache = v2.VerdictCache(max_entries=2)
    for i in range(5):
        cache.put(str(i), {"eid": "E1"})
    assert cache.get("0") is None and cache.get("4") == {"eid": "E1"}


# ------------------------------------------------------------------------------------------- registry
def test_verifier_v2_is_registered_and_opt_in():
    assert VERIFIERS["verifier_v2"] is v2.VerifierV2 and STUDIO_VERIFIER == "arm_c_studio"
    assert v2.VerifierV2.uses_search is False
