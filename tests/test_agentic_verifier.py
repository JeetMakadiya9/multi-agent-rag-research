"""Offline tests for agentic verification (docqa/agentic.py), the Research Studio's default verifier: it runs the
agentic verification study's frozen arm-C code (evaluation/experiments/agentic_verify) on the Studio's evidence. A
scripted fake model answers each role by its exact system prompt; nothing is sent anywhere."""
import hashlib
import json
import re

import pytest

from docqa.adjudicate import ADJUDICATED
from docqa.agentic import (ArmC0Verifier, StudioArmCVerifier, ArmCVerifier, FROZEN_CONFIG,
                           frozen_settings)
from docqa.index import Evidence
from docqa.llm import LLMError, LLMResponse
from docqa.ranked import ADJUDICATED_UNSIDED, RANK_TIEBREAK
from docqa.verify import CONTRADICTED, INSUFFICIENT, SUPPORTED, VERIFIERS
from evaluation.experiments.agentic_verify import arms, prompts
from webrag.evidence import Passage

CLAIM = "Drug X lowers blood pressure in adults."
SUP = "In a randomised trial of 400 adults, drug X lowered systolic blood pressure by 9 mmHg."
CON = "In adults, drug X had no effect on blood pressure compared with placebo in a large trial."
NEU = "Drug X is metabolised in the liver and has a half-life of twelve hours in most patients."
NEU2 = "Blood pressure is measured with a cuff placed on the upper arm of the seated patient."
FOUND = "A meta-analysis found that drug X reduced blood pressure in adults across eleven trials."

VERIFIER, PLANNER, ADJUDICATOR = (prompts.VERIFIER_SYSTEM["v1"], prompts.PLANNER_SYSTEM["v0"],
                                  prompts.ADJUDICATOR_SYSTEM["v2"])


class FakeModel:
    """Verifier replies come from ``stances`` (passage text -> stance, or "INVALID" for an unusable reply); planner and
    adjudicator replies are taken from queues (a dict, a raw string, a callable of the user message, or an exception)."""

    def __init__(self, stances, planner=(), adjudicator=()):
        self.stances, self.planner, self.adjudicator = dict(stances), list(planner), list(adjudicator)
        self.requests = []

    def chat(self, messages, *, response_format=None, temperature=0.0, num_predict=None, timeout=None):
        system, user = messages[0]["content"], messages[1]["content"]
        role = {VERIFIER: "verifier", PLANNER: "planner", ADJUDICATOR: "adjudicator"}[system]
        self.requests.append({"role": role, "user": user, "num_predict": num_predict, "format": response_format,
                              "temperature": temperature})
        if role == "verifier":
            body = user.split("Passages:\n", 1)[1].split("\n\nReturn valid JSON", 1)[0]     # the retry's suffix
            entries = re.findall(r"\[(P\d+)\] (.*?)(?=\n\n\[P\d+\] |\Z)", body, re.S)
            if any(self.stances.get(text.strip()) == "INVALID" for _, text in entries):
                return LLMResponse("not json", "stop", 5, 50, 0.1)
            if any(isinstance(self.stances.get(text.strip()), Exception) for _, text in entries):
                raise next(self.stances[t.strip()] for _, t in entries if isinstance(self.stances.get(t.strip()), Exception))
            reply = {"assessments": [{"id": pid, "stance": self.stances.get(text.strip(), "NEUTRAL"),
                                      "quote": " ".join(text.split()[:6])} for pid, text in entries]}
            return LLMResponse(json.dumps(reply), "stop", 20, 300, 0.2)
        item = (self.planner if role == "planner" else self.adjudicator).pop(0)
        if isinstance(item, Exception):
            raise item
        if callable(item):
            item = item(user)
        return LLMResponse(item if isinstance(item, str) else json.dumps(item), "stop", 10, 100, 0.1)

    def roles(self):
        return [r["role"] for r in self.requests]


def pid_of(user, text):
    return re.search(r"\[(P\d+)\] \(marked \w+\) " + re.escape(text[:30]), user).group(1)


def decide(verdict, text, quote=None):
    return lambda user: {"verdict": verdict, "passage": pid_of(user, text), "quote": quote or text[:40]}


def ev(*texts):
    return [Evidence(f"E{i}", f"doc{i}", 1, t) for i, t in enumerate(texts, 1)]


class Store:
    """Ranks ``FOUND`` (and the given passages) for any planner query; records the queries."""

    def __init__(self, extra=(FOUND,)):
        self.extra, self.queries = list(extra), []
        self.sources = {f"W{i}": type("Src", (), {"origin": f"w{i}.example"})() for i in range(1, 9)}

    def rank(self, query, k=6):
        self.queries.append((query, k))
        return [Passage(f"W{i}:p1:x", f"W{i}", 1, t, 1.0) for i, t in enumerate(self.extra, 1)][:k]


# ------------------------------------------------------------------------------------------- default and exact code
def test_the_studio_default_is_the_arm_c_studio_variant_and_alternatives_stay_available():
    from webrag.model_service import STUDIO_VERIFIER, studio_options
    from webrag.orchestrator import Settings
    assert STUDIO_VERIFIER == "arm_c_studio" and studio_options(env={})["verifier"] == "arm_c_studio"
    assert VERIFIERS["arm_c_studio"] is StudioArmCVerifier and VERIFIERS["arm_c"] is ArmCVerifier
    assert VERIFIERS["arm_c0"] is ArmC0Verifier
    assert studio_options("quote_stance", env={})["verifier"] == "quote_stance"     # the previous verifier
    assert Settings().verifier == "quote_stance"         # the command-line tool and evaluation harness are unchanged


def test_the_workflow_is_the_frozen_study_code_with_its_frozen_settings():
    config = json.loads(FROZEN_CONFIG.read_text(encoding="utf-8"))
    folder = FROZEN_CONFIG.parent
    for name in ("arms.py", "llm.py", "prompts.py"):
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == config["code_sha256"][name], name
    s = frozen_settings()
    assert (s.verifier, s.planner, s.adjudicator, s.trigger, s.fallback, s.search_when) == \
        ("v1", "v0", "v2", "unresolved", "rank", "empty")
    assert (s.k, s.search_k, s.max_searches, s.max_turns, s.adjudicate_max, s.skip_exhausted_planner) == \
        (8, 4, 2, 3, 6, True)
    assert (s.verifier_tokens, s.planner_tokens, s.adjudicator_tokens) == (900, 300, 200)


# ------------------------------------------------------------------------------------------- the three paths
def test_unanimous_evidence_takes_one_verifier_call_and_nothing_else():
    model = FakeModel({SUP: "SUPPORTS"})
    verdict = StudioArmCVerifier(model).verify(CLAIM, ev(SUP, NEU), search=Store().rank)
    assert (verdict.label, verdict.sub_state) == (SUPPORTED, "")
    assert model.roles() == ["verifier"]
    assert model.requests[0]["format"] == "json" and model.requests[0]["num_predict"] == 900
    assert verdict.stances == {"E1": "SUPPORTS", "E2": "NEUTRAL"}
    assert verdict.agentic_trace["actions"] == [{"turn": 0, "action": "RESOLVED"}]


def test_conflicting_evidence_goes_straight_to_the_adjudicator_and_both_sides_stay():
    model = FakeModel({SUP: "SUPPORTS", CON: "CONTRADICTS"}, adjudicator=[decide("CONTRADICTED", CON)])
    store = Store()
    verdict = StudioArmCVerifier(model).verify(CLAIM, ev(SUP, CON, NEU), search=store.rank)
    assert (verdict.label, verdict.sub_state) == (CONTRADICTED, ADJUDICATED)
    assert model.roles() == ["verifier", "adjudicator"] and store.queries == []        # no planner, no search
    assert verdict.stances == {"E1": "SUPPORTS", "E2": "CONTRADICTS", "E3": "NEUTRAL"}
    shown = re.findall(r"\[(P\d+)\] \(marked (\w+)\)", model.requests[1]["user"])
    assert [s for _, s in shown][:2] == ["SUPPORTS", "CONTRADICTS"]                    # sided passages first
    assert "adjudicator judged CONTRADICTED" in verdict.note


def test_without_a_sided_passage_the_planner_searches_the_run_evidence_and_new_passages_are_judged():
    model = FakeModel({FOUND: "SUPPORTS"}, planner=[{"action": "SEARCH", "query": "drug X blood pressure meta-analysis"},
                                                    {"action": "ADJUDICATE"}])
    store = Store()
    verdict = StudioArmCVerifier(model).verify(CLAIM, ev(NEU, NEU2), search=store.rank)
    assert model.roles() == ["verifier", "planner", "verifier", "planner"]             # resolved: no adjudicator
    assert store.queries == [("drug X blood pressure meta-analysis", 8)]              # search_k 4, twice for dedup
    assert (verdict.label, verdict.sub_state) == (SUPPORTED, "")
    assert verdict.stances == {"E1": "NEUTRAL", "E2": "NEUTRAL", "S1": "SUPPORTS"}
    [found] = verdict.searched_passages
    assert found["passage"].text == FOUND and found["query"] == "drug X blood pressure meta-analysis"
    assert "1 planner search(es)" in verdict.note


def test_search_is_bounded_and_unresolved_evidence_then_goes_to_the_adjudicator():
    many = [f"Unrelated passage number {i} about hospital parking and cafeteria opening hours." for i in range(12)]
    model = FakeModel({}, planner=[{"action": "SEARCH", "query": f"q{i}"} for i in range(5)],
                      adjudicator=[{"verdict": "INSUFFICIENT", "passage": "P1", "quote": ""}])
    store = Store(extra=many)
    verdict = StudioArmCVerifier(model).verify(CLAIM, ev(NEU, NEU2), search=store.rank)
    assert len(store.queries) == 2 and model.roles().count("planner") == 2           # at most 2 searches
    assert model.roles()[-1] == "adjudicator"
    assert len(verdict.searched_passages) == 8                                         # 4 new passages per search
    assert (verdict.label, verdict.sub_state) == (INSUFFICIENT, "NO_DECISIVE_EVIDENCE")


# ------------------------------------------------------------------------------------------- unsided decisions
def unsided_model():
    return FakeModel({}, planner=[{"action": "ADJUDICATE"}], adjudicator=[decide("SUPPORTED", NEU)])


def test_the_default_declines_a_decision_without_any_sided_passage_and_records_it():
    verdict = StudioArmCVerifier(unsided_model()).verify(CLAIM, ev(NEU, NEU2), search=Store().rank)
    assert (verdict.label, verdict.sub_state) == (INSUFFICIENT, "NO_DECISIVE_EVIDENCE")
    assert "proposed SUPPORTED, which is not accepted" in verdict.note
    assert verdict.agentic_trace["adjudication"]["verdict"] == "SUPPORTED"            # the proposal is kept


def test_the_exact_mode_accepts_it_as_the_study_did_and_the_studio_cautions_it():
    from webrag import deep
    verdict = ArmCVerifier(unsided_model()).verify(CLAIM, ev(NEU, NEU2), search=Store().rank)
    assert (verdict.label, verdict.sub_state) == (SUPPORTED, ADJUDICATED_UNSIDED)
    claim = {"label": SUPPORTED, "sub_state": ADJUDICATED_UNSIDED, "text": CLAIM, "parse_ok": True,
             "passages": [{"stance": "NEUTRAL", "source_id": "S1", "key": "S1:p1", "text": NEU}]}
    assert deep.strength(claim, {}) == "adjudicated unsided"
    assert "unsided_adjudication" in {c["kind"] for c in deep.cautions(claim)}       # kept out of the answer


# ------------------------------------------------------------------------------------------- fall-backs and failures
def test_an_ungrounded_adjudication_falls_back_to_the_rank_rule():
    model = FakeModel({SUP: "SUPPORTS", CON: "CONTRADICTS"},
                      adjudicator=[decide("CONTRADICTED", CON, quote="words that are not in any passage at all")])
    verdict = StudioArmCVerifier(model).verify(CLAIM, ev(SUP, CON), search=Store().rank)
    assert (verdict.label, verdict.sub_state) == (SUPPORTED, RANK_TIEBREAK)           # E1 is ranked first
    assert "study fall-back" in verdict.note


def test_an_insufficient_adjudication_keeps_the_conflict_undecided():
    model = FakeModel({SUP: "SUPPORTS", CON: "CONTRADICTS"},
                      adjudicator=[{"verdict": "INSUFFICIENT", "passage": "P1", "quote": ""}])
    verdict = StudioArmCVerifier(model).verify(CLAIM, ev(SUP, CON), search=Store().rank)
    assert (verdict.label, verdict.sub_state) == (INSUFFICIENT, "CONFLICTING")


def test_an_unusable_verifier_reply_is_an_assessment_failure_not_a_verdict():
    model = FakeModel({SUP: "INVALID"})
    verdict = StudioArmCVerifier(model).verify(CLAIM, ev(SUP), search=Store().rank)
    assert (verdict.label, verdict.sub_state, verdict.parse_ok) == (INSUFFICIENT, "ASSESSMENT_FAILED", False)
    assert model.roles() == ["verifier", "verifier"]                                   # one retry


@pytest.mark.parametrize("where", ["verifier", "adjudicator"])
def test_a_service_failure_is_raised_to_the_studio_and_never_replaced(where):
    error = LLMError("timeout", "no response within 180s")
    if where == "verifier":
        model = FakeModel({SUP: error})
    else:
        model = FakeModel({SUP: "SUPPORTS", CON: "CONTRADICTS"}, adjudicator=[error])
    with pytest.raises(LLMError, match="timeout"):
        StudioArmCVerifier(model).verify(CLAIM, ev(SUP, CON), search=Store().rank)


def test_no_evidence_is_never_searched_or_decided():
    model = FakeModel({})
    verdict = StudioArmCVerifier(model).verify(CLAIM, [], search=Store().rank)
    assert (verdict.label, verdict.sub_state) == (INSUFFICIENT, "NO_EVIDENCE") and model.requests == []


def test_arm_c0_adjudicates_without_planning_or_search():
    model = FakeModel({}, adjudicator=[{"verdict": "INSUFFICIENT", "passage": "P1", "quote": ""}])
    store = Store()
    verdict = ArmC0Verifier(model).verify(CLAIM, ev(NEU, NEU2), search=store.rank)
    assert model.roles() == ["verifier", "adjudicator"] and store.queries == []
    assert verdict.agentic_trace["study_arm"] == "C0"


def test_quote_attestation_records_where_each_arm_c_quote_occurs_without_changing_the_verdict():
    from docqa.attestation import supports_attestation
    assert supports_attestation(VERIFIERS["arm_c_studio"]) and supports_attestation(VERIFIERS["arm_c"])
    plain = StudioArmCVerifier(FakeModel({SUP: "SUPPORTS", CON: "CONTRADICTS"}, adjudicator=[decide("SUPPORTED", SUP)]))
    attesting = StudioArmCVerifier(FakeModel({SUP: "SUPPORTS", CON: "CONTRADICTS"}, adjudicator=[decide("SUPPORTED", SUP)]))
    attesting.attest = True
    a, b = plain.verify(CLAIM, ev(SUP, CON, NEU)), attesting.verify(CLAIM, ev(SUP, CON, NEU))
    assert (a.label, a.sub_state, a.stances) == (b.label, b.sub_state, b.stances)       # record only
    records = b.quote_attestations
    assert set(records) == {"E1", "E2", "E3"} and "quote_attestations" not in a.__dict__
    assert records["E1"][0]["quote"] == " ".join(SUP.split()[:6]) and records["E1"][0]["legacy_quote_check"] is True
    assert records["E1"][0]["attestation"]["status"] == "LOCATED"


def test_a_failed_planner_search_fails_closed():
    def broken(query, k):
        raise OSError("index unavailable")
    model = FakeModel({}, planner=[{"action": "SEARCH", "query": "drug X"}])
    verdict = StudioArmCVerifier(model).verify(CLAIM, ev(NEU, NEU2), search=broken)
    assert (verdict.label, verdict.sub_state, verdict.parse_ok) == (INSUFFICIENT, "ASSESSMENT_FAILED", False)
    assert "planner search failed: OSError" in verdict.note and "fails closed" in verdict.note


def test_a_search_without_an_evidence_store_fails_closed():
    model = FakeModel({}, planner=[{"action": "SEARCH", "query": "drug X"}])
    verdict = StudioArmCVerifier(model).verify(CLAIM, ev(NEU, NEU2), search=None)
    assert (verdict.label, verdict.sub_state) == (INSUFFICIENT, "ASSESSMENT_FAILED")
    assert "no evidence store was available" in verdict.error


def test_the_historical_and_the_studio_variant_differ_only_on_unsided_decisions():
    for make in (lambda: FakeModel({SUP: "SUPPORTS", CON: "CONTRADICTS"}, adjudicator=[decide("CONTRADICTED", CON)]),
                 lambda: FakeModel({SUP: "SUPPORTS"})):
        a = ArmCVerifier(make()).verify(CLAIM, ev(SUP, CON), search=Store().rank)
        b = StudioArmCVerifier(make()).verify(CLAIM, ev(SUP, CON), search=Store().rank)
        assert (a.label, a.sub_state, a.stances) == (b.label, b.sub_state, b.stances)
    exact = ArmCVerifier(unsided_model()).verify(CLAIM, ev(NEU, NEU2), search=Store().rank)
    studio = StudioArmCVerifier(unsided_model()).verify(CLAIM, ev(NEU, NEU2), search=Store().rank)
    assert (exact.label, studio.label) == (SUPPORTED, INSUFFICIENT)
    assert exact.agentic_trace["variant"] == "historical method"
    assert studio.agentic_trace["variant"].startswith("Studio-adapted variant")


# ------------------------------------------------------------------------------------------- the Studio pipeline
def test_the_studio_orchestrator_runs_the_agentic_workflow_and_records_what_ran():
    from webrag.orchestrator import Orchestrator, Run, Settings
    model = FakeModel({FOUND: "SUPPORTS"}, planner=[{"action": "SEARCH", "query": "drug X adults trials"},
                                                    {"action": "ADJUDICATE"}])
    orch = Orchestrator(Settings(verifier="arm_c_studio"), client=model, providers=[])
    assert isinstance(orch.verifier, StudioArmCVerifier)
    store = Store()
    first = [Passage("W7:p1:a", "W7", 1, NEU, 2.0), Passage("W8:p1:b", "W8", 1, NEU2, 1.5)]
    store.rank = (lambda original: lambda query, k=6: first if query == CLAIM else original(query, k))(store.rank)
    orch._passage_record = lambda p, s: {"source_id": p.source_id, "key": p.key, "text": p.text}
    run = Run(CLAIM, "verify", lambda event: None)
    orch._verify_mode(run, store, CLAIM)
    claim = run.data["claims"][0]
    assert (claim["label"], claim["sub_state"], claim["verifier"]) == (SUPPORTED, "", "arm_c_studio")
    assert [p["stance"] for p in claim["passages"]] == ["NEUTRAL", "NEUTRAL", "SUPPORTS"]
    assert claim["passages"][2]["found_by_search"] == "drug X adults trials"
    assert claim["supporting_sources"] == ["W1"] and claim["independent_support"] == 1
    trace = claim["verifier_trace"]
    assert trace["workflow"] == "arm_c_studio" and trace["study_arm"] == "C"
    assert [a["action"] for a in trace["actions"]] == ["SEARCH", "ADJUDICATE"]
    assert trace["usage"]["by_role"] == {"planner": 2, "verifier": 2}                   # no adjudicator was needed


def test_a_service_failure_in_the_studio_is_recorded_as_service_error():
    from webrag.orchestrator import Orchestrator, Settings
    model = FakeModel({SUP: LLMError("unreachable", "connection refused")})
    orch = Orchestrator(Settings(verifier="arm_c_studio"), client=model, providers=[])
    orch._passage_record = lambda p, s: {"source_id": p.source_id, "key": p.key, "text": p.text}
    result = orch._verify(CLAIM, "input claim", [Passage("W1:p1:a", "W1", 1, SUP, 1.0)], Store())
    assert (result.label, result.sub_state, result.parse_ok) == (INSUFFICIENT, "SERVICE_ERROR", False)
    assert "unreachable" in result.error
