"""Passage-id matching between the verifier prompt and its reply (``arm_c_studio_all_idfix``,
docqa/agentic_candidates.py).

The defect: the study's judge labels passages with run-wide ids ([P2], [P3] ...) while the verifier prompt's JSON example
answers ``"id": "P1"``. A model that copies the example (as qwen3:4b did in 517 of 668 single-passage replies in round
3) is not matched, and the frozen judge reads the passage as NEUTRAL without saying so. The earlier scripted model
echoed the requested id, so the tests never saw it. The models below answer by POSITION within each call (P1, P2 ...,
whatever ids the prompt showed), like the real model. Replies are scripted; nothing is sent anywhere.

Levels: the reply check alone; the wrapper (``IdCheckedStudioModel``) alone; the wrapper inside the frozen
``arms.judge``; the whole candidate verifier (``arms.run_agentic`` and the Studio's rules)."""
import json
import re

import pytest

from docqa import agentic_candidates as ac
from docqa.agentic import StudioArmCVerifier, StudioModel, frozen_settings
from docqa.llm import LLMResponse
from docqa.verify import CONTRADICTED, INSUFFICIENT, SUPPORTED, VERIFIERS
from evaluation.experiments.agentic_verify import arms, prompts
from evaluation.experiments.agentic_verify.llm import LLMError as StudyLLMError
from evaluation.experiments.agentic_verify.llm import Usage
from test_agentic_verifier import CLAIM, CON, FOUND, NEU, NEU2, SUP, FakeModel, Store, ev
from webrag.model_service import STUDIO_VERIFIER

VERIFIER = prompts.VERIFIER_SYSTEM["v1"]
ADJUDICATE = [{"action": "ADJUDICATE"}]
NO_DECISION = [{"verdict": "INSUFFICIENT", "passage": "P1", "quote": ""}]
EXTRA = "Systolic pressure was recorded twice daily by trained nurses during the whole study period."


def shown(user):
    """(id, text) of each passage in a verifier prompt, as the model sees it."""
    body = user.split("Passages:\n", 1)[1].split(ac.VERIFIER_RETRY_SUFFIX, 1)[0]
    return re.findall(r"\[(P\d+)\] (.*?)(?=\n\n\[P\d+\] |\Z)", body, re.S)


def quote_of(text):
    return " ".join(text.split()[:6])


class PositionalModel(FakeModel):
    """Like ``FakeModel``, but the verifier numbers its reply entries P1, P2 ... by position within each call, as the
    prompt's example does, instead of echoing the ids it was shown. ``reverse`` lists the entries last to first."""

    def __init__(self, stances, reverse=False, **kw):
        super().__init__(stances, **kw)
        self.reverse = reverse

    def chat(self, messages, **kw):
        system, user = messages[0]["content"], messages[1]["content"]
        if system != VERIFIER:
            return super().chat(messages, **kw)
        self.requests.append({"role": "verifier", "user": user})
        entries = [{"id": f"P{i}", "stance": self.stances.get(text.strip(), "NEUTRAL"), "quote": quote_of(text)}
                   for i, (_, text) in enumerate(shown(user), 1)]
        return LLMResponse(json.dumps({"assessments": entries[::-1] if self.reverse else entries}), "stop", 20, 300, 0.2)


class ScriptedVerifier(FakeModel):
    """Verifier replies are taken from ``replies`` in order (a raw string, or a callable of the passages shown)."""

    def __init__(self, replies, **kw):
        super().__init__({}, **kw)
        self.replies = list(replies)

    def chat(self, messages, **kw):
        system, user = messages[0]["content"], messages[1]["content"]
        if system != VERIFIER:
            return super().chat(messages, **kw)
        self.requests.append({"role": "verifier", "user": user})
        item = self.replies.pop(0)
        return LLMResponse(item(shown(user)) if callable(item) else item, "stop", 20, 300, 0.2)


def reply(*entries):
    return json.dumps({"assessments": [dict(zip(("id", "stance", "quote"), e)) for e in entries]})


def verifier_requests(model):
    return [r for r in model.requests if r["role"] == "verifier"]


def candidate(model):
    return ac.StudioArmCAllPassagesIdCheckedVerifier(model)


# Replies the check must reject for a call that showed one passage (P1); the code each one is recorded under.
INVALID_FOR_ONE = {
    "not json": "no_list",
    json.dumps({"verdict": "SUPPORTS"}): "no_list",
    json.dumps({"assessments": "P1 SUPPORTS"}): "no_list",
    json.dumps({"assessments": ["P1"]}): "malformed_entry",
    json.dumps({"assessments": [{"id": 1, "stance": "NEUTRAL"}]}): "malformed_entry",
    json.dumps({"assessments": [{"stance": "NEUTRAL"}]}): "malformed_entry",
    reply(("P2", "NEUTRAL", "")): "unexpected_id",                                 # the wrong id
    reply(("p1", "NEUTRAL", "")): "unexpected_id",                                 # ids are exact
    reply(("P1", "NEUTRAL", ""), ("P1", "NEUTRAL", "")): "duplicate_id",
    reply(("P1", "SUPPORTS", "a"), ("P1", "CONTRADICTS", "b")): "duplicate_id",
    reply(("P1", "NEUTRAL", ""), ("P9", "NEUTRAL", "")): "unexpected_id",
    reply(("P1", "MAYBE", "")): "invalid_stance",
    reply(("P1", "", "")): "invalid_stance",
    json.dumps({"assessments": [{"id": "P1", "quote": ""}]}): "invalid_stance",   # no stance: never NEUTRAL
    json.dumps({"assessments": [{"id": "P1", "stance": None}]}): "invalid_stance",
    json.dumps({"assessments": [{"id": "P1", "stance": ["NEUTRAL"]}]}): "invalid_stance",
    json.dumps({"assessments": []}): "missing_entry",
}


# ------------------------------------------------------------------------- the defect, reproduced on the old candidate
def test_the_round3_candidate_loses_a_non_p1_verdict_when_the_model_answers_by_position():
    model = PositionalModel({SUP: "SUPPORTS"}, planner=ADJUDICATE, adjudicator=NO_DECISION)
    verdict = ac.StudioArmCAllPassagesVerifier(model).verify(CLAIM, ev(NEU, SUP), search=Store().rank)
    assert [i for i, _ in shown(verifier_requests(model)[1]["user"])] == ["P2"]     # shown as P2, answered as P1
    assert verdict.label == INSUFFICIENT                                            # the SUPPORTS verdict is lost
    assert verdict.agentic_trace["unlisted_passages"] == ["P2"]


# ----------------------------------------------------------------------------------------------- the reply check
def test_valid_replies_with_each_stance_are_accepted_whatever_their_order():
    entries, reason = ac.check_reply(reply(("P2", "neutral", ""), ("[P1]", "SUPPORTS", "a"), ("P3", "Contradicts", "b")),
                                     ["P1", "P2", "P3"])
    assert reason == "" and [e["id"] for e in entries] == ["[P1]", "P2", "P3"]


@pytest.mark.parametrize("text,code", list(INVALID_FOR_ONE.items()))
def test_invalid_replies_are_rejected_with_their_code(text, code):
    entries, reason = ac.check_reply(text, ["P1"])
    assert entries is None and reason.split(":", 1)[0] == code


def test_a_reply_that_leaves_out_one_of_several_passages_is_rejected():
    entries, reason = ac.check_reply(reply(("P1", "NEUTRAL", "")), ["P1", "P2"])
    assert entries is None and reason == "missing_entry: no entry for P2"


# ------------------------------------------------------------------------------------------ prompt parsing
def test_the_prompt_is_parsed_back_exactly_with_nonconsecutive_ids_and_the_retry_suffix():
    user = prompts.verifier_user(CLAIM, [("P3", SUP), ("P7", "a passage that quotes [P3] in its text")])
    claim, passages, suffix = ac.split_verifier_user(user + ac.VERIFIER_RETRY_SUFFIX)
    assert claim == CLAIM and [p for p, _ in passages] == ["P3", "P7"] and suffix == ac.VERIFIER_RETRY_SUFFIX


@pytest.mark.parametrize("user", ["not a verifier prompt", "Claim: x\n\nno passages here",
                                  prompts.verifier_user(CLAIM, []), "Claim: x\n\nPassages:\nno label here"])
def test_prompts_without_parsable_passages_are_not_parsed(user):
    assert ac.split_verifier_user(user) is None


# ---------------------------------------------------------------------------------------------- the wrapper alone
def call_wrapper(client, passages, max_passages=None, suffix=""):
    model = ac.IdCheckedStudioModel(client, max_passages=max_passages)
    text = model.complete("verifier", VERIFIER, prompts.verifier_user(CLAIM, passages) + suffix, 100, Usage())
    return model, text


def test_the_wrapper_shows_nonconsecutive_ids_as_p1_p2_and_maps_the_reply_back():
    client = PositionalModel({SUP: "SUPPORTS", CON: "CONTRADICTS"}, reverse=True)
    model, text = call_wrapper(client, [("P3", SUP), ("P7", CON)], suffix=ac.VERIFIER_RETRY_SUFFIX)
    seen = verifier_requests(client)[0]["user"]
    assert shown(seen) == [("P1", SUP), ("P2", CON)] and seen.endswith(ac.VERIFIER_RETRY_SUFFIX)
    assert seen == prompts.verifier_user(CLAIM, [("P1", SUP), ("P2", CON)]) + ac.VERIFIER_RETRY_SUFFIX
    mapped = {e["id"]: e["stance"] for e in json.loads(text)["assessments"]}
    assert mapped == {"P3": "SUPPORTS", "P7": "CONTRADICTS"}
    assert model.reply_checks == [{"shown": ["P1", "P2"], "study_ids": ["P3", "P7"], "id_check": "ok"}]
    assert model.reply_ids[-1]["returned_ids"] == ["P3", "P7"] and model.assessed == {"P3": True, "P7": True}


def test_the_wrapper_rejects_unread_a_prompt_with_more_passages_than_one_call_holds():
    client = PositionalModel({})
    model, text = call_wrapper(client, [("P2", NEU + "\n\n[P5] " + NEU2)], max_passages=1)   # splits into two
    assert text == ac.INVALID_REPLY and model.reply_checks[0]["code"] == "ambiguous_prompt"


def test_the_wrapper_rejects_unread_a_prompt_it_cannot_parse():
    model = ac.IdCheckedStudioModel(ScriptedVerifier([reply(("P1", "SUPPORTS", "x"))]))
    text = model.complete("verifier", VERIFIER, "not a verifier prompt", 100, Usage())
    assert text == ac.INVALID_REPLY and model.reply_checks[0]["code"] == "unparsable_prompt"


def test_the_wrapper_leaves_planner_and_adjudicator_calls_alone():
    client = FakeModel({}, planner=ADJUDICATE)
    model = ac.IdCheckedStudioModel(client)
    text = model.complete("planner", prompts.PLANNER_SYSTEM["v0"], "anything", 100, Usage())
    assert json.loads(text) == ADJUDICATE[0] and model.reply_checks == []


# ------------------------------------------------------------------------------- inside the frozen judge (arms.judge)
def passages_numbered(*items):
    return [arms.Passage(pid, None, f"c{pid}", text, "claim", rank) for rank, (pid, text) in enumerate(items, 1)]


def test_the_frozen_judge_gets_each_original_passage_its_own_verdict_in_one_batch():
    settings = frozen_settings()                                                       # batch 8: one call
    client = PositionalModel({SUP: "SUPPORTS", CON: "CONTRADICTS"})
    stances, invalid, _ = arms.judge(CLAIM, passages_numbered(("P4", NEU), ("P5", SUP), ("P6", CON)),
                                     ac.IdCheckedStudioModel(client), Usage(), settings)
    assert [(s.pid, s.stance, s.raw_stance) for s in stances] == \
        [("P4", "NEUTRAL", "NEUTRAL"), ("P5", "SUPPORTS", "SUPPORTS"), ("P6", "CONTRADICTS", "CONTRADICTS")]
    assert invalid == 0


def test_the_frozen_judge_without_the_wrapper_reads_the_same_replies_as_neutral():
    client = PositionalModel({SUP: "SUPPORTS", CON: "CONTRADICTS"})
    stances, _, _ = arms.judge(CLAIM, passages_numbered(("P4", NEU), ("P5", SUP), ("P6", CON)),
                               StudioModel(client), Usage(), frozen_settings())
    assert [s.stance for s in stances] == ["NEUTRAL", "NEUTRAL", "NEUTRAL"]           # the defect, for contrast


def test_an_invalid_first_reply_and_a_valid_retry_give_the_retry_verdict():
    client = ScriptedVerifier([reply(("P2", "SUPPORTS", quote_of(SUP))), reply(("P1", "SUPPORTS", quote_of(SUP)))])
    model = ac.IdCheckedStudioModel(client)
    stances, invalid, _ = arms.judge(CLAIM, passages_numbered(("P2", SUP)), model, Usage(), frozen_settings())
    assert [(s.pid, s.stance) for s in stances] == [("P2", "SUPPORTS")] and invalid == 1
    assert [c["id_check"] for c in model.reply_checks] == ["rejected", "ok"]
    assert verifier_requests(client)[1]["user"].endswith(ac.VERIFIER_RETRY_SUFFIX)    # the judge's own retry


@pytest.mark.parametrize("text,code", list(INVALID_FOR_ONE.items()))
def test_two_invalid_replies_make_the_frozen_judge_raise_instead_of_reading_neutral(text, code):
    model = ac.IdCheckedStudioModel(ScriptedVerifier([text, text]))
    with pytest.raises(StudyLLMError, match="VERIFIER_INVALID"):
        arms.judge(CLAIM, passages_numbered(("P2", SUP)), model, Usage(), frozen_settings())
    assert [c["code"] for c in model.reply_checks] == [code, code]


def test_an_empty_passage_list_makes_no_call():
    client = ScriptedVerifier([])
    assert arms.judge(CLAIM, [], ac.IdCheckedStudioModel(client), Usage(), frozen_settings()) == ([], 0, 0)
    assert verifier_requests(client) == []


# --------------------------------------------------------------------------------- the whole candidate verifier
def test_a_single_passage_labelled_p5_keeps_its_verdict():
    model = PositionalModel({SUP: "SUPPORTS"})
    verdict = candidate(model).verify(CLAIM, ev(NEU, NEU2, EXTRA, NEU + " Again.", SUP), search=Store().rank)
    assert all([i for i, _ in shown(r["user"])] == ["P1"] for r in verifier_requests(model))
    assert verdict.label == SUPPORTED and verdict.stances["E5"] == "SUPPORTS"
    trace = verdict.agentic_trace
    assert [r["returned_ids"] for r in trace["reply_identifiers"]] == [["P1"], ["P2"], ["P3"], ["P4"], ["P5"]]
    assert trace["unlisted_passages"] == [] and trace["reply_reconstruction"]["consistent"] is True
    assert {d["reply_entry"] for d in verdict.passage_diagnostics.values()} == {"valid"}


def test_contradicts_and_neutral_on_later_passages_are_kept():
    verdict = candidate(PositionalModel({CON: "CONTRADICTS"})).verify(CLAIM, ev(NEU, NEU2, CON), search=Store().rank)
    assert verdict.stances == {"E1": "NEUTRAL", "E2": "NEUTRAL", "E3": "CONTRADICTS"} and verdict.label == CONTRADICTED


def test_genuine_neutral_judgements_stay_neutral_and_are_not_failures():
    model = PositionalModel({}, planner=ADJUDICATE, adjudicator=NO_DECISION)
    verdict = candidate(model).verify(CLAIM, ev(NEU, NEU2), search=Store().rank)
    assert verdict.label == INSUFFICIENT and verdict.sub_state == "NO_DECISIVE_EVIDENCE" and verdict.parse_ok
    assert verdict.stances == {"E1": "NEUTRAL", "E2": "NEUTRAL"}
    assert verdict.agentic_trace["unlisted_passages"] == []


def test_passages_found_by_a_planner_search_are_matched_too():
    model = PositionalModel({FOUND: "SUPPORTS"},
                            planner=[{"action": "SEARCH", "query": "drug X blood pressure"}, {"action": "ADJUDICATE"}])
    verdict = candidate(model).verify(CLAIM, ev(NEU, NEU2), search=Store().rank)
    assert verdict.label == SUPPORTED and verdict.stances["S1"] == "SUPPORTS"
    checks = [r["id_check"] for r in verdict.agentic_trace["reply_identifiers"]]
    assert [c["study_ids"] for c in checks] == [["P1"], ["P2"], ["P3"]] and {c["id_check"] for c in checks} == {"ok"}


@pytest.mark.parametrize("text,code", list(INVALID_FOR_ONE.items()))
def test_no_invalid_reply_is_ever_read_as_neutral_by_the_candidate(text, code):
    model = ScriptedVerifier([reply(("P1", "NEUTRAL", "")), text, text])            # P1 valid, P2 twice invalid
    verdict = candidate(model).verify(CLAIM, ev(NEU, SUP), search=Store().rank)
    assert verdict.sub_state == "ASSESSMENT_FAILED" and not verdict.parse_ok and verdict.label == INSUFFICIENT
    assert "VERIFIER_INVALID" in verdict.error
    checks = [r["id_check"] for r in verdict.agentic_trace["reply_identifiers"]]
    assert [c["id_check"] for c in checks] == ["ok", "rejected", "rejected"] and checks[1]["code"] == code
    assert "E2" not in verdict.stances                                                 # no verdict for P2 at all


def test_a_wrong_id_then_a_valid_retry_decides_the_claim():
    model = ScriptedVerifier([reply(("P7", "SUPPORTS", "x")), reply(("P1", "SUPPORTS", quote_of(SUP)))])
    verdict = candidate(model).verify(CLAIM, ev(SUP), search=Store().rank)
    assert verdict.label == SUPPORTED
    assert [r["id_check"]["id_check"] for r in verdict.agentic_trace["reply_identifiers"]] == ["rejected", "ok"]


def test_a_passage_whose_text_contains_a_label_fails_closed():
    model = PositionalModel({SUP: "SUPPORTS"})
    verdict = candidate(model).verify(CLAIM, ev(SUP + "\n\n[P2] " + NEU), search=Store().rank)
    assert verdict.sub_state == "ASSESSMENT_FAILED"
    assert {r["id_check"]["code"] for r in verdict.agentic_trace["reply_identifiers"]} == {"ambiguous_prompt"}


# ------------------------------------------------------------------------------------------------- nothing else moved
def test_the_fixed_candidate_differs_from_the_round3_candidate_only_in_its_model_wrapper():
    old, new = ac.StudioArmCAllPassagesVerifier, ac.StudioArmCAllPassagesIdCheckedVerifier
    assert issubclass(new, old) and old.model_class is StudioModel and new.model_class is ac.IdCheckedStudioModel
    a, b = old(FakeModel({})), new(FakeModel({}))
    assert a.settings == b.settings and a.verify_per_source == b.verify_per_source
    assert (a.accept_unsided, a.require_complete_assessment, a.check_cited_stance) == \
        (b.accept_unsided, b.require_complete_assessment, b.check_cited_stance)


def test_the_default_verifier_is_unchanged():
    assert STUDIO_VERIFIER == "arm_c_studio" and VERIFIERS["arm_c_studio"] is StudioArmCVerifier
    assert StudioArmCVerifier.model_class is StudioModel
    assert VERIFIERS["arm_c_studio_all_idfix"] is ac.StudioArmCAllPassagesIdCheckedVerifier
