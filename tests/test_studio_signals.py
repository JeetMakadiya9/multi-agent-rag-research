"""Studio signals around a verdict: why a statement stays INSUFFICIENT, how much the passages agree, and the
per-process read cache (webrag/orchestrator.py, webrag/researcher.py). Offline; models and the web are scripted."""
import json

import pytest

from docqa.verify import CONTRADICTED, INSUFFICIENT, SUPPORTED, Verdict
from webrag import researcher as researcher_module
from webrag.orchestrator import ABSTAIN_REASONS, confidence_from_stances
from webrag.researcher import ReadCache, WebResearcher
from test_agentic_candidates import CapStore
from test_agentic_verifier import CLAIM, FakeModel, NEU, SUP


# ------------------------------------------------------------------------------------------------ confidence
@pytest.mark.parametrize("label,stances,level", [
    (SUPPORTED, {"a": "SUPPORTS", "b": "SUPPORTS", "c": "NEUTRAL"}, "high"),
    (SUPPORTED, {"a": "SUPPORTS", "b": "NEUTRAL"}, "medium"),
    (CONTRADICTED, {"a": "CONTRADICTS", "b": "SUPPORTS"}, "low"),
    (INSUFFICIENT, {"a": "NEUTRAL"}, "abstained"),
])
def test_confidence_counts_agreement_across_passages(label, stances, level):
    out = confidence_from_stances(label, stances)
    assert out["level"] == level and out["judged"] == len(stances)


# ------------------------------------------------------------------------------------------------ through the pipeline
def verify_with(verifier_name, model, monkeypatch=None):
    from webrag.orchestrator import Orchestrator, Run, Settings
    from webrag.researcher import FULL_TEXT, Source
    orch = Orchestrator(Settings(save=False, verifier=verifier_name), client=model, store_factory=CapStore)
    source = Source("S1", "t", "https://example.org/a", "evaluation-packet", "web", "c1:1", FULL_TEXT, SUP, origin="S1")
    store = CapStore([source])
    run = Run(CLAIM, "verify", lambda event: None)
    orch._verify_mode(run, store, CLAIM)
    return run.data["claims"][0]


def test_the_default_verifier_gets_a_reason_and_a_confidence():
    claim = verify_with("arm_c_studio", FakeModel({SUP: "SUPPORTS"}))
    assert claim["label"] == SUPPORTED and claim["abstain_reason"] == ""
    assert claim["confidence"]["level"] in ("medium", "high") and claim["confidence"]["agreeing"] >= 1


def test_verifier_v2_reasons_reach_each_passage_and_the_claim(monkeypatch):
    monkeypatch.setenv("WEBRAG_NLI_MODEL", "off")
    from test_verifier_v2 import Model, reply
    model = Model({SUP: reply("NEUTRAL"), NEU: reply("NEUTRAL")})
    claim = verify_with("verifier_v2", model)
    assert claim["label"] == INSUFFICIENT and claim["sub_state"] == "NO_DECISIVE_EVIDENCE"
    assert claim["abstain_reason"] and claim["confidence"]["level"] == "abstained"
    assert all(p.get("verifier_reason") for p in claim["passages"])
    assert claim["verifier_trace"]["workflow"] == "verifier_v2"


def test_every_insufficient_sub_state_has_a_plain_reason():
    for sub_state in ("NO_EVIDENCE", "NO_DECISIVE_EVIDENCE", "CONFLICTING", "ASSESSMENT_FAILED", "SERVICE_ERROR"):
        assert ABSTAIN_REASONS[sub_state].endswith(".")


# ------------------------------------------------------------------------------------------------ read cache
class Page:
    def __init__(self, text):
        self.content, self.text, self.url, self.content_type = text.encode(), text, "https://example.org/a", "text/plain"


class Session:
    def __init__(self):
        self.gets = 0

    def get(self, url):
        self.gets += 1
        return Page(f"page body {self.gets}")


def test_a_page_read_once_is_served_from_the_cache_until_it_expires():
    cache = ReadCache(ttl=60)
    session = Session()
    first = WebResearcher([], session, read_cache=cache)._read_document("https://example.org/a")
    second = WebResearcher([], session, read_cache=cache)._read_document("https://example.org/a")
    assert session.gets == 1 and first.text == second.text == "page body 1" and cache.hits == 1
    second.text = "changed by a caller"
    assert WebResearcher([], session, read_cache=cache)._read_document("https://example.org/a").text == "page body 1"
    expired = ReadCache(ttl=0)
    for _ in range(2):
        WebResearcher([], session, read_cache=expired)._read_document("https://example.org/b")
    assert session.gets == 3                                     # an expired entry is fetched again


def test_failed_reads_are_not_cached_and_the_cache_can_be_switched_off(monkeypatch):
    class Failing:
        gets = 0

        def get(self, url):
            Failing.gets += 1
            raise researcher_module.FetchError("http_error", "500")
    cache = ReadCache()
    for _ in range(2):
        with pytest.raises(researcher_module.FetchError):
            WebResearcher([], Failing(), read_cache=cache)._read_document("https://example.org/x")
    assert Failing.gets == 2
    monkeypatch.setenv("WEBRAG_READ_CACHE", "0")
    assert WebResearcher([], Session()).read_cache is None


def test_the_reading_settings_are_part_of_the_key():
    cache, session = ReadCache(), Session()
    WebResearcher([], session, read_cache=cache, classify=False)._read_document("https://example.org/a")
    WebResearcher([], session, read_cache=cache, classify=True)._read_document("https://example.org/a")
    assert session.gets == 2


def test_the_read_cache_is_bounded():
    cache = ReadCache(max_entries=2)
    for i in range(4):
        cache.put((str(i),), researcher_module.Read(str(i), "u", "text/plain"))
    assert cache.get(("0",)) is None and cache.get(("3",)).text == "3"
