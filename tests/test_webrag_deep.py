"""Offline tests for the deep-research engine: providers and their configuration, fetching and extraction, transcripts,
the source-aware planner, the bounded research loop (budgets, coverage, follow-up, counter-evidence, stopping), the
evidence policy, citation checks, the report and the server routes. No network, no model: every response below is a
SYNTHETIC fixture and the model, verifier and evidence index are deterministic fakes."""
from __future__ import annotations

import io
import json
import re
import threading
import urllib.error
import urllib.request
from dataclasses import replace
from http.server import ThreadingHTTPServer

import pytest

from docqa.llm import LLMError, LLMResponse
from docqa.pdf_utils import write_simple_pdf
from docqa.verify import CONTRADICTED, INSUFFICIENT, SUPPORTED, Verdict
from webrag import deep as deep_module
from webrag import extract
from webrag.agents import SourceAwarePlanner, Synthesizer, summarize, write_queries
from webrag.config import ConfigError, ConfigStore, public_view, validate
from webrag.deep import (BUDGETS, check_citations, completeness, independence, is_evidence, run_status, site_of,
                         source_quality, strength)
from webrag.evidence import Passage, cap_per_origin
from webrag.feeds import parse_feed
from webrag.net import Fetched, FetchError
from webrag.orchestrator import Orchestrator, Settings
from webrag.report import render_markdown
from webrag.researcher import (ABSTRACT_ONLY, FAILED, FULL_TEXT, METADATA_ONLY, SNIPPET_ONLY, Route, Source,
                               WebResearcher)
from webrag.search import (PROVIDERS, BraveSearch, CrossrefSearch, EuropePmcSearch, FeedSearch, OpenAlexSearch,
                           SearchError, SearchHit, SearxngSearch, WikipediaSearch, crossref_hit, default_providers,
                           enabled_names, excluded_host, is_evidence_provider, provider_catalog)
from webrag.server import JobQueue, check_provider, make_handler, studio_orchestrator

# ================================================================================================= fakes

class FakeSession:
    """Routes URLs to canned responses; anything unrouted is a 404. Never touches the network."""

    def __init__(self, routes=()):
        self.routes, self.requests_made, self.calls, self.allowed_origins = list(routes), 0, [], set()

    def get(self, url, *, params=None, headers=None, check_robots=True, max_bytes=None):
        self.requests_made += 1
        self.calls.append({"url": url, "params": dict(params or {}), "headers": dict(headers or {})})
        for match, result in self.routes:
            if match(url, params or {}):
                if isinstance(result, Exception):
                    raise result
                return result
        raise FetchError("http_404", url)


def page(body, ctype="text/html", url="https://site.example/x", charset=""):
    data = body if isinstance(body, bytes) else (body if isinstance(body, str) else json.dumps(body)).encode("utf-8")
    return Fetched(url, 200, ctype, data, charset)


class FakeProvider:
    """Hits chosen by a function of the query; records every call and the keyword arguments it received. A row is
    (title, url, snippet, abstract, published, publisher, authors), trailing fields optional."""

    def __init__(self, name, kind, hits_for, texts=None, error=None, supports_since=False):
        self.name, self.kind, self.hits_for, self.texts, self.error = name, kind, hits_for, texts or {}, error
        self.calls = []
        self.supports_since = supports_since

    def search(self, query, limit=3, **kwargs):
        self.calls.append((query, kwargs))
        if self.error:
            raise SearchError(self.error)
        rows = self.hits_for(query)
        return [SearchHit(self.name, query, i, r[0], r[1], r[2] if len(r) > 2 else "", kind=self.kind,
                          abstract=r[3] if len(r) > 3 else "", published=r[4] if len(r) > 4 else None,
                          publisher=r[5] if len(r) > 5 else "", authors=list(r[6]) if len(r) > 6 else [])
                for i, r in enumerate(rows[:limit], 1)]

    def fetch_text(self, hit):
        if hit.url not in self.texts:
            raise SearchError("article text unavailable")
        return {"text": self.texts[hit.url], "revised": "2026-09-01T00:00:00Z"}


class Store:
    """Sentence passages ranked by word overlap; supports incremental indexing like EvidenceStore."""

    def __init__(self, sources, models=None):
        self.sources, self.passages = {}, []
        self.add_sources(sources)

    def add_sources(self, sources):
        for s in sources:
            if s.id in self.sources:
                continue
            self.sources[s.id] = s
            for page_no, text in enumerate(s.pages or [s.text], 1):
                for i, sentence in enumerate(text.split(".")):
                    if sentence.strip():
                        self.passages.append(Passage(f"{s.id}:p{page_no}:{i}", s.id, page_no, sentence.strip() + ".", 0.0))

    @property
    def size(self):
        return len(self.passages)

    def rank(self, query, k=6, per_source=2):
        words = set(re.findall(r"[a-z0-9]+", query.lower())) - {"the", "a", "of", "in", "what", "how", "did", "do"}
        scored = [(len(words & set(re.findall(r"[a-z0-9]+", p.text.lower()))), p) for p in self.passages]
        scored = sorted([x for x in scored if x[0] > 0], key=lambda x: -x[0])
        kept, per = [], {}
        for score, p in scored:
            if per.get(p.source_id, 0) >= per_source:
                continue
            per[p.source_id] = per.get(p.source_id, 0) + 1
            kept.append(Passage(p.key, p.source_id, p.page, p.text, float(score)))
            if len(kept) == k:
                break
        return kept


class EvidenceVerifier:
    """SUPPORTS a passage that contains the claim; CONTRADICTS one that contains 'contrary' and shares its key words;
    the verdict follows the documented rules (support only, contradiction only, both = CONFLICTING)."""

    def __init__(self):
        self.calls = []

    def verify(self, claim, evidence):
        self.calls.append((claim, [e.text for e in evidence]))
        core = claim.rstrip(".").lower()
        words = set(re.findall(r"[a-z]{4,}", core))
        stances = {}
        for e in evidence:
            text = e.text.lower()
            if "contrary" in text and len(words & set(re.findall(r"[a-z]{4,}", text))) >= 2:
                stances[e.id] = "CONTRADICTS"
            elif core in text:
                stances[e.id] = "SUPPORTS"
            else:
                stances[e.id] = "NEUTRAL"
        s, c = "SUPPORTS" in stances.values(), "CONTRADICTS" in stances.values()
        if s and c:
            return Verdict(INSUFFICIENT, "CONFLICTING", stances=stances)
        if s:
            return Verdict(SUPPORTED, stances=stances)
        if c:
            return Verdict(CONTRADICTED, stances=stances)
        return Verdict(INSUFFICIENT, "NO_DECISIVE_EVIDENCE", stances=stances)


def prompt_kind(messages):
    system = messages[0]["content"]
    if system.startswith("You plan research."):
        return "plan"
    if system.startswith("Some subquestions"):
        return "follow-up"
    if system.startswith("These statements"):
        return "counter"
    if system.startswith("You write the opening"):
        return "summary"
    return "synthesis"


class DeepClient:
    model, base_url = "qwen3:4b", "http://127.0.0.1:11434"

    def __init__(self, replies, fail=()):
        self.replies, self.fail, self.seen = replies, set(fail), []
        self.history = []

    def describe(self):
        return {"model": "qwen3:4b", "ollama_version": "0.0-test", "model_digest": "d"}

    def chat(self, messages, *, response_format=None, temperature=0.0, num_predict=None, timeout=None):
        kind = prompt_kind(messages)
        self.seen.append((kind, messages))
        if kind in self.fail:
            raise LLMError("timeout", "slow")
        reply = self.replies.get(kind, {})
        reply = reply(messages) if callable(reply) else reply
        return LLMResponse(reply if isinstance(reply, str) else json.dumps(reply), "stop", 1, 1, 0.01)


# A small world of SYNTHETIC sources.
GROWTH = "Solar capacity grew 20 percent in 2024"
WIKI_TEXT = f"{GROWTH}. Panels convert sunlight into electricity. Rooftop installations are common in cities."
WEB_HTML = (f"<html><head><title>Solar report</title><meta name='author' content='A. Writer'></head><body><main><p>{GROWTH} "
            "according to the national grid operator, which publishes the figures every year for every region of the "
            "country and explains its counting method in detail.</p>" + "<p>Installers reported long waiting lists for new "
            "rooftop systems across the year, which the operator expects to continue.</p>" * 3 + "</main></body></html>")
STORAGE = "Storage batteries cost 100 dollars per kilowatt hour"
STORAGE_HTML = f"<html><body><article><p>{STORAGE} in the latest survey of grid projects and suppliers.</p>" + \
               "<p>Prices were collected from published tenders and supplier lists during the whole year.</p>" * 4 + "</article></body></html>"
COUNTER_HTML = (f"<html><body><article><p>Contrary to claims, storage batteries did not cost 100 dollars per kilowatt hour; "
                "audited tenders show higher prices.</p>" + "<p>The audit covered every tender published by the regulator.</p>" * 4
                + "</article></body></html>")
STATS_HTML = (f"<html><body><article><p>{GROWTH} and the energy agency counts every grid-connected installation in its "
              "annual statistics, published each spring for all regions.</p>" + "<p>Figures are revised when late "
              "registrations arrive from installers across the country.</p>" * 3 + "</article></body></html>")
WIKI_URL = "https://en.wikipedia.org/wiki/Solar_power"


def world(counter=True):
    # A Wikipedia provider is offered to every run below: the source policy must keep it out of deep research.
    wiki = FakeProvider("wikipedia", "encyclopedia",
                        lambda q: [("Solar power", WIKI_URL)] if "solar" in q.lower() else [], texts={WIKI_URL: WIKI_TEXT})
    scholarly = FakeProvider("openalex", "scholarly", lambda q: [
        ("Solar survey", "https://doi.org/10.1/solar", "", f"{GROWTH} in the global survey of installed capacity.", None, "",
         ["Ana Researcher"]),
        ("Record without abstract", "https://doi.org/10.1/empty", "", "")] if "solar" in q.lower() else [])

    def web(q):
        # Rows carry the publisher name a web search such as Brave records; a Wikipedia page is among the results.
        q = q.lower()
        rows = []
        if "solar" in q:
            rows += [("Grid report", "https://grid.example/report", "Grid report snippet.", "", None, "Grid Operator"),
                     ("Energy statistics", "https://stats.energy.example/solar", "Statistics snippet.", "", None, "Energy Agency"),
                     ("Solar power - Wikipedia", WIKI_URL, "Encyclopedia snippet about solar."),
                     ("Members only", "https://paywalled.example/a", "Teaser about solar panels."),
                     ("Down site", "https://down.example/b", "")]
        if "storage" in q or "batter" in q or "cost" in q:
            rows += [("Battery survey", "https://batteries.example/survey", "Battery snippet.", "", None, "Battery Council")]
        if counter and ("criticism" in q or "contrary" in q or "against" in q):
            rows += [("Battery audit", "https://audit.example/batteries", "Audit snippet.", "", None, "Audit Office")]
        return rows

    searxng = FakeProvider("searxng", "web", web)
    session = FakeSession([
        (lambda u, p: "wikipedia.org" in u, page(f"<html><body><main><p>{WIKI_TEXT}</p></main></body></html>", url=WIKI_URL)),
        (lambda u, p: u == "https://stats.energy.example/solar", page(STATS_HTML, url="https://stats.energy.example/solar")),
        (lambda u, p: u == "https://grid.example/report", page(WEB_HTML, url="https://grid.example/report")),
        (lambda u, p: u == "https://paywalled.example/a", page("<html><title>Subscribe</title><body><p>Subscribe to continue "
                                                                 "reading this article.</p></body></html>", url="https://paywalled.example/a")),
        (lambda u, p: u == "https://down.example/b", FetchError("timeout", "https://down.example/b")),
        (lambda u, p: u == "https://batteries.example/survey", page(STORAGE_HTML, url="https://batteries.example/survey")),
        (lambda u, p: u == "https://audit.example/batteries", page(COUNTER_HTML, url="https://audit.example/batteries")),
    ])
    return [wiki, scholarly, searxng], session


PLAN = {"central_question": "How fast is solar growing and what does storage cost?", "deliverable": "answer",
        "subquestions": [{"text": "How fast did solar capacity grow?", "terms": ["solar capacity"], "categories": ["encyclopedia", "scholarly", "web"]},
                         {"text": "What do storage batteries cost?", "terms": ["battery storage"], "categories": ["web"]}],
        "queries": [{"subquestion": 1, "category": "encyclopedia", "query": "solar capacity growth"},
                    {"subquestion": 1, "category": "scholarly", "query": "solar capacity growth survey"},
                    {"subquestion": 1, "category": "web", "query": "solar capacity growth 2024"},
                    {"subquestion": 1, "category": "news", "query": "not an available category"}],
        "counter_queries": [], "time_constraint": "2024", "entities": ["solar"]}
SYNTH = {"findings": [{"text": f"{GROWTH}.", "evidence": ["E1"], "subquestion": 1},
                      {"text": f"{STORAGE}.", "evidence": ["E2"], "subquestion": 2}],
         "disagreements": [], "gaps": ["Regional detail is missing."], "insufficient": False}
FOLLOW = {"queries": [{"subquestion": 1, "category": "web", "query": "storage battery cost per kilowatt hour"}]}
COUNTER = {"queries": [{"subquestion": 1, "category": "web", "query": "storage battery cost criticism"}]}
SUMMARY = {"answered": True, "direct_answer": {"text": f"{GROWTH}.", "findings": ["F1"]},
           "summary": [{"text": "Capacity rose by 20 percent.", "findings": ["F1"]},
                       {"text": "Capacity rose by 35 percent.", "findings": ["F1"]},
                       {"text": "Prices are falling.", "findings": []}]}


def deep_run(text="How fast is solar growing and what does storage cost?", mode="research", replies=None, depth="standard",
             providers=None, session=None, **options):
    providers_, session_ = world()
    client = DeepClient(replies or {"plan": PLAN, "synthesis": SYNTH, "follow-up": FOLLOW, "counter": COUNTER, "summary": SUMMARY})
    verifier = EvidenceVerifier()
    orch = Orchestrator(Settings(save=False, per_query=6), client=client, session=session or session_,
                        providers=providers if providers is not None else providers_, verifier=verifier, store_factory=Store)
    data = orch.run(text, mode, options={"depth": depth, **options})
    return data, client, verifier, orch


# ================================================================================================= providers
def test_registry_catalog_and_configuration_states(monkeypatch):
    for name in ("WEBRAG_PROVIDERS", "SEARXNG_URL", "BRAVE_API_KEY", "OPENALEX_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    rows = {r["id"]: r for r in provider_catalog(validate({}))}
    assert set(rows) == set(PROVIDERS) and all(r["implemented"] for r in rows.values())
    assert rows["openalex"]["active"] and rows["arxiv"]["active"]
    wiki = rows["wikipedia"]                                                       # source policy
    assert not wiki["enabled"] and not wiki["active"] and wiki["evidence"] is False and wiki["controlled_by"] == "policy"
    assert "not accepted as an evidence source" in wiki["policy"]
    assert rows["searxng"]["enabled"] and not rows["searxng"]["configured"] and not rows["searxng"]["active"]
    assert rows["searxng"]["missing"] == ["SEARXNG_URL (or a SearXNG address in Integrations)"]
    assert not rows["crossref"]["enabled"] and not rows["europepmc"]["enabled"]           # opt-in
    assert rows["feeds"]["enabled"] and not rows["feeds"]["configured"]
    monkeypatch.setenv("BRAVE_API_KEY", "synthetic-secret-123")
    config = validate({"providers": {"crossref": {"enabled": True}, "wikipedia": {"enabled": False}},
                       "searxng": {"url": "http://127.0.0.1:8888/"}, "feeds": [{"url": "https://news.example/rss", "category": "news"}]})
    rows = {r["id"]: r for r in provider_catalog(config)}
    assert rows["crossref"]["active"] and not rows["wikipedia"]["active"] and rows["searxng"]["active"] and rows["feeds"]["active"]
    assert rows["brave"]["active"] and rows["brave"]["secrets"] == {"BRAVE_API_KEY": True}
    assert "synthetic-secret-123" not in json.dumps(provider_catalog(config))
    monkeypatch.setenv("WEBRAG_PROVIDERS", "wikipedia,openalex")
    rows = {r["id"]: r for r in provider_catalog(config)}
    assert not rows["wikipedia"]["active"] and rows["openalex"]["active"]           # named in the environment: still excluded
    assert not rows["crossref"]["active"] and rows["crossref"]["controlled_by"] == "WEBRAG_PROVIDERS"


def test_default_providers_follow_the_configuration(monkeypatch):
    for name in ("WEBRAG_PROVIDERS", "SEARXNG_URL", "BRAVE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    session = FakeSession()
    # Without a studio configuration (command line, evaluation) the original default set is unchanged.
    assert [p.name for p in default_providers(session)] == ["wikipedia", "openalex", "arxiv"]
    config = validate({"providers": {"arxiv": {"enabled": False}, "europepmc": {"enabled": True}},
                       "searxng": {"url": "http://127.0.0.1:8888"}, "contact_email": "me@example.org",
                       "feeds": [{"url": "https://gov.example/feed", "name": "Gov", "category": "official"}]})
    names = [p.name for p in default_providers(session, None, config)]
    assert names == ["openalex", "europepmc", "searxng", "feeds"]                       # no Wikipedia in the studio
    assert "http://127.0.0.1:8888" in session.allowed_origins
    assert enabled_names({}) == ["openalex", "arxiv", "searxng", "brave", "feeds"]
    assert [p.name for p in default_providers(session, ["wikipedia", "openalex"], validate({}))] == ["openalex"]
    monkeypatch.setenv("WEBRAG_PROVIDERS", "wikipedia,arxiv")
    assert [p.name for p in default_providers(session, None, validate({}))] == ["arxiv"]
    assert [p.name for p in default_providers(session)] == ["wikipedia", "arxiv"]       # the command line: unchanged
    monkeypatch.delenv("WEBRAG_PROVIDERS")
    with pytest.raises(ValueError, match="no feed is configured"):
        default_providers(session, ["feeds"], validate({}))


def test_configuration_validation_refuses_secrets_and_unknown_settings(tmp_path):
    for bad, message in [({"api_key": "x"}, "unknown settings"), ({"providers": {"bing": {"enabled": True}}}, "unknown provider"),
                         ({"providers": {"wikipedia": {"enabled": "yes"}}}, "only"),
                         ({"searxng": {"url": "http://127.0.0.1:8888/?token=abc"}}, "must not carry"),
                         ({"searxng": {"url": "ftp://x.example"}}, "http"),
                         ({"feeds": [{"url": "https://user:pw@x.example/feed"}]}, "without a user name"),
                         ({"feeds": [{"url": "https://x.example/feed", "category": "tv"}]}, "category"),
                         ({"contact_email": "not-an-email"}, "email"),
                         ({"providers": {"wikipedia": {"enabled": True}}}, "not accepted as an evidence source")]:
        with pytest.raises(ConfigError, match=message):
            validate(bad)
    assert validate({"providers": {"wikipedia": {"enabled": False}}})["providers"] == {"wikipedia": {"enabled": False}}
    old = tmp_path / "old.json"                       # saved before the policy: the switch is dropped, the rest is kept
    old.write_text(json.dumps({"providers": {"wikipedia": {"enabled": True}, "crossref": {"enabled": True}}}), encoding="utf-8")
    assert ConfigStore(old).load()["providers"] == {"crossref": {"enabled": True}}
    with pytest.raises(ConfigError, match="not accepted"):
        ConfigStore(old).update({"providers": {"wikipedia": {"enabled": True}}})
    store = ConfigStore(tmp_path / "config.json")
    assert store.load()["providers"] == {} and store.error == ""
    saved = store.update({"providers": {"crossref": {"enabled": True}}, "feeds": [{"url": "https://a.example/feed"},
                                                                                 {"url": "https://a.example/feed"}]})
    assert saved["feeds"] == [{"url": "https://a.example/feed", "name": "", "category": "news"}]       # de-duplicated
    store.update({"providers": {"arxiv": {"enabled": False}}})
    assert store.load()["providers"] == {"crossref": {"enabled": True}, "arxiv": {"enabled": False}}  # merged by id
    (tmp_path / "config.json").write_text("{broken", encoding="utf-8")
    assert store.load()["providers"] == {} and "could not be used" in store.error


def test_public_view_reports_environment_without_values(monkeypatch):
    monkeypatch.setenv("BRAVE_API_KEY", "synthetic-secret-456")
    view = public_view(validate({}))
    assert view["environment"]["BRAVE_API_KEY"] is True and "synthetic-secret-456" not in json.dumps(view)


def test_crossref_parsing_licences_and_parameters():
    rows = {"message": {"items": [
        {"DOI": "10.1/open", "title": ["Open <i>study</i>"], "author": [{"given": "Ada", "family": "Lovelace"}],
         "published": {"date-parts": [[2021, 5, 3]]}, "container-title": ["J. Synthetic"], "abstract": "<jats:p>We find X.</jats:p>",
         "license": [{"URL": "https://creativecommons.org/licenses/by/4.0/"}],
         "link": [{"URL": "https://pub.example/open.pdf", "content-type": "application/pdf"}], "type": "journal-article"},
        {"DOI": "10.1/closed", "title": ["Closed"], "license": [{"URL": "https://publisher.example/tdm"}],
         "link": [{"URL": "https://pub.example/closed.pdf", "content-type": "application/pdf"}], "issued": {"date-parts": [[2020]]}},
        {"title": ["No DOI"]}]}}
    session = FakeSession([(lambda u, p: "crossref" in u, page(rows, "application/json"))])
    hits = CrossrefSearch(session, mailto="me@example.org").search("x", 4, since="2020-01-01")
    assert [h.title for h in hits] == ["Open study", "Closed"]
    assert hits[0].authors == ["Ada Lovelace"] and hits[0].published == "2021-05-03" and hits[0].abstract == "We find X."
    assert hits[0].pdf_url == "https://pub.example/open.pdf" and hits[1].pdf_url is None      # only openly licensed links
    assert hits[1].published == "2020" and hits[0].identifiers == {"doi": "10.1/open"}
    params = session.calls[0]["params"]
    assert params["mailto"] == "me@example.org" and params["filter"] == "from-pub-date:2020-01-01"
    assert crossref_hit({"DOI": "", "title": ["x"]}) is None
    broken = FakeSession([(lambda u, p: True, FetchError("rate_limited", "x"))])
    with pytest.raises(SearchError):
        CrossrefSearch(broken).search("x")


def test_native_recency_and_category_parameters():
    from datetime import date, timedelta
    recent = (date.today() - timedelta(days=100)).isoformat()
    openalex = FakeSession([(lambda u, p: True, page({"results": []}, "application/json"))])
    OpenAlexSearch(openalex).search("q", since="2020-01-01")
    assert openalex.calls[0]["params"]["filter"] == "from_publication_date:2020-01-01"
    searx = FakeSession([(lambda u, p: True, page({"results": []}, "application/json"))])
    SearxngSearch(searx, "http://127.0.0.1:8888").search("q", since=recent, categories="news")
    assert searx.calls[0]["params"]["time_range"] == "year" and searx.calls[0]["params"]["categories"] == "news"
    SearxngSearch(searx, "http://127.0.0.1:8888").search("q", since="2015-01-01")
    assert "time_range" not in searx.calls[1]["params"]                     # older limits are filtered afterwards
    brave = FakeSession([(lambda u, p: True, page({"web": {"results": []}}, "application/json"))])
    BraveSearch(brave, "k").search("q", since=recent)
    assert brave.calls[0]["params"]["freshness"] == "py" and brave.calls[0]["headers"]["X-Subscription-Token"] == "k"


RSS = b"""<?xml version="1.0"?>
<rss version="2.0" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:podcast="https://podcastindex.org/namespace/1.0"
     xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">
<channel><title>Energy Talk</title>
 <item><title>Solar storage episode</title><link>https://pod.example/ep1</link><guid>ep1</guid>
  <description>&lt;p&gt;Batteries and solar storage costs.&lt;/p&gt;</description><pubDate>Tue, 03 Sep 2024 10:00:00 GMT</pubDate>
  <dc:creator>Host Name</dc:creator><enclosure url="https://pod.example/ep1.mp3" type="audio/mpeg"/>
  <itunes:duration>00:42:10</itunes:duration>
  <podcast:transcript url="https://pod.example/ep1.json" type="application/json"/>
  <podcast:transcript url="https://pod.example/ep1.vtt" type="text/vtt"/></item>
 <item><title>Old wind episode</title><link>https://pod.example/ep0</link><pubDate>Mon, 01 Jan 2018 10:00:00 GMT</pubDate></item>
 <item><title>Unsafe</title><link>javascript:alert(1)</link></item>
</channel></rss>"""
ATOM = b"""<feed xmlns="http://www.w3.org/2005/Atom"><title>Ministry releases</title>
<entry><title>Solar subsidy report published</title><link rel="alternate" href="https://gov.example/reports/solar"/>
<id>tag:gov,2024:1</id><published>2024-06-01T00:00:00Z</published><summary>The solar subsidy report.</summary>
<author><name>Ministry of Energy</name></author></entry></feed>"""


def test_feed_parsing_rss_atom_and_podcast_transcripts():
    items = parse_feed(RSS, "https://pod.example/feed")
    assert [i.title for i in items] == ["Solar storage episode", "Old wind episode"]          # unsafe link dropped
    ep = items[0]
    assert (ep.published, ep.authors, ep.transcript_url, ep.transcript_type) == ("2024-09-03", ["Host Name"],
                                                                                  "https://pod.example/ep1.vtt", "text/vtt")
    assert ep.media_url == "https://pod.example/ep1.mp3" and ep.duration == "00:42:10" and ep.summary == "Batteries and solar storage costs."
    atom = parse_feed(ATOM)[0]
    assert (atom.url, atom.published, atom.authors) == ("https://gov.example/reports/solar", "2024-06-01", ["Ministry of Energy"])
    with pytest.raises(ValueError):
        parse_feed(b"<rss><channel><item>")
    with pytest.raises(ValueError):
        parse_feed(b"<html></html>")


def test_feed_search_matches_keywords_dates_categories_and_reports_failures():
    session = FakeSession([(lambda u, p: u == "https://pod.example/feed", page(RSS, "application/rss+xml", "https://pod.example/feed")),
                           (lambda u, p: u == "https://gov.example/feed", page(ATOM, "application/atom+xml", "https://gov.example/feed")),
                           (lambda u, p: u == "https://broken.example/feed", FetchError("http_500", "https://broken.example/feed"))])
    feeds = FeedSearch(session, [{"url": "https://pod.example/feed", "name": "Energy Talk", "category": "podcast"},
                                 {"url": "https://gov.example/feed", "name": "Ministry", "category": "official"},
                                 {"url": "https://broken.example/feed", "name": "Broken", "category": "news"}])
    hits = feeds.search("solar subsidy storage")
    assert [h.title for h in hits] == ["Solar storage episode", "Solar subsidy report published"]
    assert hits[0].category == "podcast" and hits[0].extra["transcript_url"].endswith(".vtt") and hits[0].publisher == "Energy Talk"
    assert [h.title for h in feeds.search("solar", since="2024-07-01")] == ["Solar storage episode"]
    assert [h.title for h in feeds.search("solar", categories="official")] == ["Solar subsidy report published"]
    with pytest.raises(SearchError, match="Broken"):
        feeds.search("solar", categories="news")                              # every feed of the category failed
    assert feeds.search("unrelated topic words") == []                        # an empty result is not an error
    assert len([c for c in session.calls if c["url"] == "https://pod.example/feed"]) == 1      # each feed read once


# ================================================================================================= extraction
def test_text_decoding_honours_declared_and_detected_encodings():
    latin = "Café über naïve".encode("latin-1")
    assert extract.decode_text(latin, "iso-8859-1") == "Café über naïve"
    assert page(latin, charset="iso-8859-1").text == "Café über naïve"
    meta = b"<html><head><meta charset='windows-1252'></head><body>\x93quoted\x94</body></html>"
    assert "“quoted”" in extract.decode_text(meta, sniff_html=True)
    assert extract.decode_text(b"\xef\xbb\xbfBOM text") == "BOM text"
    assert extract.decode_text("plain ünïcode".encode("utf-8")) == "plain ünïcode"
    assert extract.decode_text(b"bad \x81\x8d bytes", "no-such-codec")         # never raises


def test_html_main_content_metadata_and_gates():
    body = "<p>" + "The main article explains the method in detail and gives the results. " * 8 + "</p>"
    html = (f"<html lang='en-GB'><head><title>T</title><link rel='canonical' href='https://site.example/a'>"
            f"<meta name='citation_author' content='Ada Lovelace'><meta name='author' content='Ada Lovelace'></head>"
            f"<body><div>Sidebar text that is long enough to be kept by the line filter, about other topics.</div>"
            f"<main>{body}</main></body></html>")
    out = extract.html_to_text(html)
    assert out.used_main and "Sidebar" not in out.text and out.canonical_url == "https://site.example/a"
    assert out.authors == ["Ada Lovelace"] and out.language == "en-GB" and out.gate == ""
    assert extract.html_to_text("<html><title>Just a moment...</title><body><p>Checking your browser before accessing.</p></body></html>").gate == "bot check"
    assert extract.html_to_text("<html><body><p>Please sign in to continue reading this article.</p></body></html>").gate == "login required"
    assert extract.html_to_text("<html><body><p>404 Not Found: this page doesn't exist.</p></body></html>").gate == "error page"
    long_article = "<p>" + "A long article that mentions you can sign in to continue reading elsewhere. " * 30 + "</p>"
    assert extract.html_to_text(long_article).gate == ""                        # long content is never a gate


def test_pdf_diagnostics_scanned_malformed_and_page_limits(tmp_path):
    from pypdf import PdfWriter
    writer = PdfWriter()
    writer.add_blank_page(200, 200)
    writer.add_blank_page(200, 200)
    blank = io.BytesIO()
    writer.write(blank)
    scanned = extract.pdf_to_text(blank.getvalue())
    assert scanned.image_only and scanned.empty_pages == 2 and scanned.page_count == 2 and scanned.text == ""
    with pytest.raises(extract.ExtractionError) as caught:
        extract.pdf_to_text(b"%PDF-1.4 this is not really a pdf")
    assert caught.value.code == "pdf_unreadable"
    write_simple_pdf(tmp_path / "long.pdf", [[f"Page {i} text about solar capacity growth."] for i in range(1, 6)])
    long = extract.pdf_to_text((tmp_path / "long.pdf").read_bytes(), max_pages=3)
    assert long.page_count == 5 and len(long.pages) == 3 and long.truncated and long.pages[2].startswith("Page 3")


VTT = """WEBVTT

00:00:01.000 --> 00:00:04.000
<v Alice>Solar capacity grew quickly.

00:00:30.000 --> 00:00:35.500
<v Bob>Storage is still expensive.

00:01:10.000 --> 00:01:15.000
<v Bob>Prices fell last year.
"""


def test_transcripts_keep_timestamps_and_speakers():
    segments = extract.parse_transcript(VTT, "text/vtt")
    assert [(s.start, s.speaker) for s in segments] == [(1.0, "Alice"), (30.0, "Bob"), (70.0, "Bob")]
    pages, labels = extract.transcript_pages(segments, window=60)
    assert pages == ["Alice: Solar capacity grew quickly. Bob: Storage is still expensive.", "Bob: Prices fell last year."]
    assert labels == ["0:01–0:35", "1:10–1:15"]
    srt = "1\n00:00:02,000 --> 00:00:05,000\nHello there.\n\n2\n00:00:06,000 --> 00:00:08,000\nSecond line.\n"
    assert [s.text for s in extract.parse_transcript(srt, "application/x-subrip")] == ["Hello there.", "Second line."]
    js = json.dumps({"segments": [{"startTime": 3, "endTime": 4, "body": "Hi.", "speaker": "Host"}]})
    assert extract.parse_transcript(js, "application/json")[0].speaker == "Host"
    with pytest.raises(extract.ExtractionError):
        extract.parse_transcript("not a transcript", "text/plain")


# ================================================================================================= researcher (deep mode)
def test_deep_researcher_classifies_every_retrieval_outcome():
    providers, session = world()
    researcher = WebResearcher(providers, session, classify=True, per_query=6)
    corpus = researcher.gather(["solar"])
    by_title = {s.title: s for s in corpus.sources}
    # source policy: the Wikipedia provider is not queried, and the Wikipedia page a web search returned is not fetched
    assert providers[0].name == "wikipedia" and providers[0].calls == []
    assert "Solar power" not in by_title and "Solar power - Wikipedia" not in by_title and corpus.excluded_by_policy == 1
    assert not [c for c in session.calls if "wikipedia.org" in c["url"]] and "wikipedia" not in corpus.provider_status
    assert by_title["Solar survey"].access == ABSTRACT_ONLY and by_title["Solar survey"].identifiers == {"doi": "10.1/solar"}
    empty = by_title["Record without abstract"]
    assert empty.access == METADATA_ONLY and empty.retrieval == "metadata" and not empty.usable
    grid = by_title["Grid report"]
    assert grid.access == FULL_TEXT and grid.authors == ["A. Writer"] and grid.fingerprint and grid.diagnostics["main_content"]
    gated = by_title["Members only"]
    assert gated.access == SNIPPET_ONLY and gated.diagnostics["gate"] == "paywall" and gated.retrieval == "snippet"
    down = by_title["Down site"]
    assert down.access == FAILED and down.retrieval == "failed" and "timeout" in down.retrieval_error
    assert by_title["Grid report"].discovered_by == ["searxng"]
    legacy = {s.title: s for s in WebResearcher(providers, session, per_query=6).gather(["solar"]).sources}
    assert legacy["Record without abstract"].access == FAILED                  # the original mode is unchanged
    assert legacy["Members only"].access == FULL_TEXT or legacy["Members only"].access == SNIPPET_ONLY
    assert legacy["Solar power"].access == FULL_TEXT                           # including its provider selection


def test_transcript_sources_are_cited_by_time_and_never_claim_listening():
    hit = SearchHit("feeds", "q", 1, "Solar storage episode", "https://pod.example/ep1", "Batteries.", kind="web",
                    category="podcast", extra={"transcript_url": "https://pod.example/ep1.vtt", "transcript_type": "text/vtt",
                                               "media_url": "https://pod.example/ep1.mp3"})
    session = FakeSession([(lambda u, p: u.endswith(".vtt"), page(VTT, "text/vtt", "https://pod.example/ep1.vtt"))])
    source = WebResearcher([], session, classify=True).fetch("S1", hit, ["q"])
    assert source.access == FULL_TEXT and source.category == "podcast" and source.page_labels == ["0:01–0:35", "1:10–1:15"]
    assert "not listened to" in source.note and source.media_url.endswith(".mp3")
    store = Store([source])
    record = Orchestrator._passage_record(store.rank("storage expensive")[0], store)
    assert record["location"] == "0:01–0:35"


def test_open_access_resolution_budgets_and_unsupported_types(tmp_path):
    write_simple_pdf(tmp_path / "oa.pdf", [["Open access full text about solar capacity. " * 60]])
    pdf = (tmp_path / "oa.pdf").read_bytes()
    session = FakeSession([
        (lambda u, p: "api.openalex.org/works/doi:" in u, page({"best_oa_location": {"pdf_url": "https://repo.example/oa.pdf",
                                                                                     "license": "cc-by"},
                                                                "open_access": {"oa_status": "green"}}, "application/json")),
        (lambda u, p: u == "https://repo.example/oa.pdf", page(pdf, "application/pdf", "https://repo.example/oa.pdf")),
        (lambda u, p: u == "https://site.example/data.zip", page(b"PK..", "application/zip", "https://site.example/data.zip"))])
    from webrag.researcher import OpenAccessResolver
    resolver = OpenAccessResolver(session)
    researcher = WebResearcher([], session, classify=True, resolver=resolver, max_full_text=1)
    hit = SearchHit("crossref", "q", 1, "Paper", "https://doi.org/10.1/x", kind="scholarly", doi="https://doi.org/10.1/x",
                    abstract="An abstract.")
    source = researcher.fetch("S1", hit, ["q"])
    assert source.access == FULL_TEXT and source.read_url == "https://repo.example/oa.pdf" and source.license == "cc-by"
    second = researcher.fetch("S2", replace(hit, url="https://doi.org/10.1/y", doi="https://doi.org/10.1/y"), ["q"])
    assert second.access == ABSTRACT_ONLY and second.diagnostics["full_text_skipped"] == "full-text budget used up"
    assert researcher.full_text_attempts == 1
    zipped = researcher.fetch("S3", SearchHit("searxng", "q", 1, "Zip", "https://site.example/data.zip", kind="web"), ["q"])
    assert zipped.access == FAILED and "unsupported_type" in zipped.retrieval_error
    tight = WebResearcher([], FakeSession([(lambda u, p: True, page(WEB_HTML, url="https://grid.example/report"))]),
                          classify=True, max_total_chars=50)
    skipped = tight.fetch("S4", SearchHit("searxng", "q", 1, "Grid", "https://grid.example/report", kind="web"), ["q"])
    assert skipped.access == METADATA_ONLY and skipped.retrieval == "skipped" and "text budget" in skipped.note


def test_routes_send_queries_only_to_their_providers_and_dates_are_filtered():
    a = FakeProvider("searxng", "web", lambda q: [("Old", "https://a.example/old", "x", "", "2010-01-01"),
                                                  ("New", "https://a.example/new", "y", "", "2025-01-01")])
    b = FakeProvider("openalex", "scholarly", lambda q: [])
    researcher = WebResearcher([a, b], FakeSession(), classify=True, since="2020-01-01", fetch_full_text=False)
    corpus = researcher.gather(["q1"], routes=[Route("q1", "searxng", "web")])
    assert [q for q, _ in a.calls] == ["q1"] and b.calls == []
    assert [s.title for s in corpus.sources] == ["New"] and corpus.filtered_by_date == 1
    assert corpus.provider_status["openalex"]["queries"] == 0 and corpus.provider_status["searxng"]["queries"] == 1


# ================================================================================================= agents
def test_source_aware_planner_routes_queries_and_falls_back():
    client = DeepClient({"plan": PLAN})
    plan = SourceAwarePlanner(client, max_subquestions=4, max_queries=6).plan("How fast is solar growing?", "research",
                                                                               ["encyclopedia", "scholarly", "web"])
    assert [s["text"] for s in plan.subquestions] == ["How fast did solar capacity grow?", "What do storage batteries cost?"]
    assert all(q.category in {"encyclopedia", "scholarly", "web"} for q in plan.queries)                 # 'news' dropped
    assert plan.queries[0].query == "How fast is solar growing?"                                      # input always searched
    assert plan.context["time_constraint"] == "2024" and not plan.fallback
    system = client.seen[0][1][0]["content"]
    assert "'web': the general web" in system and "'news'" not in system                              # only active types
    bad = SourceAwarePlanner(DeepClient({"plan": "{broken"})).plan("Solar growth in Spain", "research", ["scholarly", "web"])
    assert bad.fallback and "unusable" in bad.note and [q.category for q in bad.queries] == ["scholarly", "web"]
    assert bad.queries[1].query == "Solar growth in Spain"
    none = SourceAwarePlanner(DeepClient({})).plan("x", "research", [])
    assert none.fallback and none.queries == []


def test_query_writer_and_keyword_fallback():
    queries, fell_back = write_queries(DeepClient({"follow-up": FOLLOW}), "follow-up", ["What do batteries cost?"], ["web"],
                                       ["solar"], 3)
    assert [q.query for q in queries] == ["storage battery cost per kilowatt hour"] and not fell_back
    queries, fell_back = write_queries(DeepClient({}, fail={"counter"}), "counter", ["Batteries cost 100 dollars."],
                                       ["web", "scholarly"], [], 2)
    assert fell_back and queries[0].query == "Batteries cost 100 dollars criticism" and queries[0].purpose == "counter"


def test_summary_keeps_only_cited_sentences_without_new_numbers():
    result = summarize(DeepClient({"summary": SUMMARY}), "Q?", [f"{GROWTH}."])
    assert result["available"] and result["direct_answer"]["findings"] == ["F1"]
    assert [p["text"] for p in result["summary"]] == ["Capacity rose by 20 percent."] and result["dropped"] == 2
    assert summarize(DeepClient({}), "Q?", [])["available"] is False
    assert summarize(DeepClient({}, fail={"summary"}), "Q?", ["x"])["available"] is False


def test_synthesizer_v3_tags_subquestions():
    passages = [Passage("S1:p1:0", "S1", 1, f"{GROWTH}.", 1.0)]
    sources = {"S1": Source("S1", "T", "https://a.example", "wikipedia", "encyclopedia", "url:a", FULL_TEXT, "x")}
    reply = {"findings": [{"text": "A.", "evidence": ["E1"], "subquestion": 2}, {"text": "B.", "evidence": ["E1"], "subquestion": 9}],
             "disagreements": [], "gaps": [], "insufficient": False}
    out = Synthesizer(DeepClient({"synthesis": reply}), version=3).synthesize("Q", ["s1", "s2"], passages, sources)
    assert [f.subquestion for f in out.findings] == [2, 0]                      # out-of-range numbers fall back to 0


# ================================================================================================= the research loop
def test_standard_depth_run_end_to_end():
    data, client, verifier, orch = deep_run()
    assert data["depth"] == "standard" and data["status"] in {"COMPLETE", "PARTIAL"} and data["outcome"] == "ANSWERED"
    # source policy: the Wikipedia provider passed in is never queried, and the Wikipedia page that the web search
    # returned is never fetched, recorded, indexed, cited or listed
    wiki = next(p for p in orch.providers if p.name == "wikipedia")
    assert wiki.calls == [] and "wikipedia" not in data["options"]["providers"]
    assert not any("wikipedia.org" in c["url"] for c in orch.session.calls)
    assert not any("wikipedia" in json.dumps(s).lower() for s in data["sources"])
    assert not any("wikipedia" in p["url"] for p in data["evidence_pool"])
    assert sum(r["excluded_by_policy"] for r in data["rounds"]) >= 1
    assert any("not accepted as an evidence source" in x for x in data["limitations"])
    bibliography = render_markdown(data).split("## Sources")[1].split("## Pipeline stages")[0]
    assert "wikipedia" not in bibliography.lower()
    rounds = data["rounds"]
    assert rounds[0]["searches"] <= BUDGETS["standard"].searches_per_round
    assert any(q["purpose"] == "follow-up" for r in rounds for q in r["queries"])                     # a follow-up round
    assert rounds[-1].get("purpose") == "counter-evidence"                                          # and a counter round
    claims = {c["text"]: c for c in data["claims"]}
    growth, storage = claims[f"{GROWTH}."], claims[f"{STORAGE}."]
    assert growth["label"] == SUPPORTED and growth["strength"] == "corroborated" and growth["citations"]["ok"]
    # corroborated because two supporting pages have different sites and different recorded publishers
    assert growth["independence"]["status"] == "established" and growth["independence"]["independent"] == 2
    assert storage["label"] == INSUFFICIENT and storage["sub_state"] == "CONFLICTING" and storage["strength"] == "contested"
    change = data["counter_evidence"]["changes"][0]
    assert change["before"]["label"] == SUPPORTED and change["after"]["sub_state"] == "CONFLICTING" and change["changed"]
    assert any(d["verified"] and "storage" in d["text"].lower() for d in data["disagreements"])    # contradiction kept
    ledger = {e["index"]: e for e in data["ledger"]}
    assert ledger[1]["status"] == "answered" and ledger[2]["status"] == "contested"
    assert ledger[0]["status"] == "answered with disagreement"
    summary = data["summary"]
    assert summary["available"] and summary["direct_answer"]["claims"] == [data["claims"].index(growth)]
    assert summary["dropped"] == 2
    used, limits = data["budget"]["used"], data["budget"]["limits"]
    assert all(used[k] <= limits[k] for k in ("rounds", "searches", "sources", "full_text", "chars"))
    assert used["rounds"] == 2 and len(rounds) == 3                  # the counter-evidence search is not a search round
    access = {s["title"]: s["access"] for s in data["sources"]}
    assert access["Members only"] == SNIPPET_ONLY and access["Record without abstract"] == METADATA_ONLY
    indexed = {p["source_id"] for p in data["evidence_pool"]}
    snippet_ids = {s["id"] for s in data["sources"] if s["access"] in {SNIPPET_ONLY, METADATA_ONLY, FAILED}}
    assert not indexed & snippet_ids                                         # snippets and metadata are never evidence
    assert data["stopping"]["reason"] == "round_limit" and data["stopping"]["completeness"] in {"budget_limited", "access_limited"}
    assert any("snippets (used for discovery, not as evidence)" in x for x in data["limitations"])
    assert data["plan"]["queries"] and isinstance(data["plan"]["queries"][0], str)                    # old record shape kept
    assert data["settings"]["depth"] == "standard" and "deep-research-v5" in data["settings"]["prompt_version"]
    assert [k for k, _ in client.seen].count("summary") == 1


def test_quick_depth_is_one_round_without_counter_evidence():
    data, _, _, _ = deep_run(depth="quick")
    assert len(data["rounds"]) == 1 and data["stopping"]["reason"] == "round_limit"
    assert "not part of the quick depth" in data["counter_evidence"]["skipped"]


def test_stops_when_coverage_is_sufficient_even_with_rounds_left():
    plan = dict(PLAN, subquestions=[PLAN["subquestions"][0]])
    data, client, _, _ = deep_run(depth="deep", replies={"plan": plan, "synthesis": dict(SYNTH, findings=SYNTH["findings"][:1]),
                                                         "summary": SUMMARY})
    assert len([r for r in data["rounds"] if r.get("purpose") != "counter-evidence"]) == 1
    assert data["stopping"]["reason"] == "retrieval_coverage_met" and data["stopping"]["completeness"] == "complete"
    assert "follow-up" not in [k for k, _ in client.seen]


def test_little_new_evidence_stops_the_loop():
    nothing = {"queries": [{"subquestion": 1, "category": "web", "query": "unrelated topic words"}]}
    data, _, _, _ = deep_run(depth="deep", replies={"plan": PLAN, "synthesis": SYNTH, "follow-up": nothing, "summary": SUMMARY,
                                                    "counter": COUNTER})
    assert data["stopping"]["reason"] == "little_new_evidence" and data["stopping"]["completeness"] in {"evidence_limited", "access_limited"}
    assert data["rounds"][1]["new_sources"] == 0


def test_no_new_query_stops_the_loop():
    """A follow-up query for a source type the open question cannot use is refused; the keyword fallback is used
    once, and when it would only repeat itself the loop stops instead of searching the same thing again."""
    stale = {"queries": [{"subquestion": 1, "category": "encyclopedia", "query": "solar capacity growth again"}]}
    data, _, _, _ = deep_run(depth="deep", replies={"plan": PLAN, "synthesis": SYNTH, "follow-up": stale, "summary": SUMMARY,
                                                    "counter": COUNTER})
    assert data["stopping"]["reason"] == "no_new_queries"
    follow = [q for r in data["rounds"] for q in r["queries"] if q["purpose"] == "follow-up"]
    assert [q["query"] for q in follow] == ["storage batteries cost"] and follow[0]["category"] == "web"


def test_budgets_are_enforced(monkeypatch):
    monkeypatch.setitem(deep_module.BUDGETS, "standard", replace(BUDGETS["standard"], searches=3, searches_per_round=3, sources=2))
    data, _, _, _ = deep_run()
    used = data["budget"]["used"]
    assert used["searches"] <= 3 and used["sources"] <= 2 and len(data["sources"]) <= 2
    assert data["stopping"]["reason"] in {"search_budget", "source_budget", "retrieval_coverage_met", "round_limit"}


def test_time_budget_stops_after_the_first_round(monkeypatch):
    monkeypatch.setitem(deep_module.BUDGETS, "standard", replace(BUDGETS["standard"], seconds=0))
    data, _, _, _ = deep_run()
    assert data["stopping"]["reason"] == "time_budget" and len(data["rounds"]) == 1
    assert data["counter_evidence"]["changes"] == [] and data["counter_evidence"]["skipped"]


def test_failing_providers_are_reported_not_hidden():
    broken = [FakeProvider("openalex", "scholarly", lambda q: [], error="timeout: slow"),
              FakeProvider("searxng", "web", lambda q: [], error="http_403: refused")]
    data, _, _, _ = deep_run(providers=broken)
    assert data["status"] == "FAILED" and "No search provider could be reached" in data["error"]   # an execution failure
    assert data["stopping"]["reason"] == "providers_failing" and data["stopping"]["completeness"] == "provider_limited"
    assert any("'searxng' failed" in x for x in data["limitations"]) and any("'openalex' failed" in x for x in data["limitations"])


def test_only_snippets_and_metadata_give_an_honest_empty_result():
    """Searches that ran and found only snippets and metadata: a completed run that could not support an answer."""
    only = [FakeProvider("searxng", "web", lambda q: [("Members only", "https://paywalled.example/a", "Teaser about solar.")]),
            FakeProvider("openalex", "scholarly", lambda q: [("Record", "https://doi.org/10.1/empty", "", "")])]
    _, session = world()
    data, _, _, _ = deep_run(providers=only, session=session)
    assert data["status"] == "COMPLETE" and data["outcome"] == "INSUFFICIENT_EVIDENCE" and data["claims"] == []
    assert "No readable source" in data["error"] and "2 source(s) were found" in data["error"]
    assert data["source_quality"]["read"] == 0 and data["summary"]["available"] is False and "direct_answer" not in data["summary"]
    assert all(e["status"] != "answered" for e in data["ledger"])


def test_deep_verification_of_a_claim():
    data, _, _, _ = deep_run(text=f"{GROWTH}.", mode="verify")
    assert data["outcome"] == SUPPORTED and data["claims"][0]["strength"] == "corroborated"
    assert data["summary"] == {"available": True, "verdict": SUPPORTED, "strength": "corroborated"}
    assert data["counter_evidence"]["skipped"] == "every supported finding already rests on publishers established as independent"


def test_model_failure_during_deep_research_fails_clearly():
    data, _, _, _ = deep_run(replies={"plan": PLAN, "synthesis": SYNTH}, depth="quick")
    assert data["status"] in {"COMPLETE", "PARTIAL"}                           # summary missing is not a failure
    providers, session = world()
    client = DeepClient({"plan": PLAN}, fail={"synthesis"})
    orch = Orchestrator(Settings(save=False), client=client, session=session, providers=providers,
                        verifier=EvidenceVerifier(), store_factory=Store)
    failed = orch.run("Q?", options={"depth": "quick"})
    assert failed["status"] == "FAILED" and "Local model error" in failed["error"]


def test_legacy_single_pass_is_untouched_without_a_depth():
    providers, session = world()
    client = DeepClient({"synthesis": SYNTH})
    orch = Orchestrator(Settings(save=False), client=client, session=session, providers=providers,
                        verifier=EvidenceVerifier(), store_factory=Store)
    client.replies["plan"] = PLAN
    data = orch.run("How fast is solar growing?")
    assert "depth" not in data and "ledger" not in data and data["stages"][1]["name"] == "plan"
    assert [s["name"] for s in data["stages"]][:3] == ["model check", "plan", "search and fetch"]
    with pytest.raises(ValueError):
        Settings(depth="bottomless")


# ================================================================================================= policy units
def test_strength_labels_and_citation_checks():
    sources = {"S1": {"access": FULL_TEXT}, "S2": {"access": ABSTRACT_ONLY}, "S3": {"access": SNIPPET_ONLY}}
    established = {"status": "established", "independent": 2, "groups": 2, "sources": 2}
    supported = {"label": SUPPORTED, "parse_ok": True, "independent_support": 2, "contradicting_sources": [],
                 "independence": established,
                 "passages": [{"source_id": "S1", "stance": "SUPPORTS"}, {"source_id": "S2", "stance": "SUPPORTS"}]}
    assert strength(supported, sources) == "corroborated"
    unknown = {"status": "unknown", "independent": 1, "groups": 2, "sources": 2}
    assert strength(dict(supported, independence=unknown), sources) == "independence unknown"
    single = {"status": "single", "independent": 1, "groups": 1, "sources": 2}
    assert strength(dict(supported, independence=single), sources) == "single source"
    # two distinct texts (independent_support 2) without an independence assessment are never corroboration
    assert strength({k: v for k, v in supported.items() if k != "independence"}, sources) == "single source"
    assert strength(dict(supported, passages=[{"source_id": "S2", "stance": "SUPPORTS"}]), sources) == "abstract only"
    assert strength({"label": INSUFFICIENT, "sub_state": "CONFLICTING"}, sources) == "contested"
    assert strength({"label": CONTRADICTED}, sources) == "contradicted"
    assert strength({"label": INSUFFICIENT, "sub_state": "SERVICE_ERROR"}, sources) == "unverified"
    claim = {"passages": [{"key": "S1:p1:a", "source_id": "S1", "page": 1, "text": "Exact words here."},
                          {"key": "S9:p1:a", "source_id": "S9", "page": 1, "text": "x"},
                          {"key": "S1:p7:a", "source_id": "S1", "page": 7, "text": "Exact words here."},
                          {"key": "S3:p1:a", "source_id": "S3", "page": 1, "text": "y"},
                          {"key": "S1:p1:b", "source_id": "S1", "page": 1, "text": "Paraphrased, not in the text."}]}
    check = check_citations(claim, sources, {"S1": "Some Exact   words here. More."}, {"S1": 2})
    assert not check["ok"] and len(check["problems"]) == 3 and check["unlocated"] == ["S1:p1:b"]
    assert any("unknown source S9" in p for p in check["problems"]) and any("location 7" in p for p in check["problems"])
    # a passage from a Wikipedia page fails the check even if such a source were ever recorded as readable
    wiki = {"W1": {"access": FULL_TEXT, "url": "https://example.org/r", "read_url": "https://en.m.wikipedia.org/wiki/X"}}
    flagged = check_citations({"passages": [{"key": "W1:p1:a", "source_id": "W1", "page": 1, "text": "x"}]}, wiki, {}, {})
    assert not flagged["ok"] and "excluded by the source policy" in flagged["problems"][0]


def test_completeness_labels():
    ledger = [{"status": "answered"}, {"status": "unanswered"}]
    readable = [{"access": FULL_TEXT}, {"access": FULL_TEXT}, {"access": FAILED}]
    unreadable = [{"access": FULL_TEXT}, {"access": FAILED}, {"access": METADATA_ONLY}]
    assert completeness({"reason": "coverage_sufficient"}, readable, [{"status": "answered"}]) == "complete"
    assert completeness({"reason": "round_limit"}, readable, ledger) == "budget_limited"
    assert completeness({"reason": "round_limit"}, unreadable, ledger) == "access_limited"
    assert completeness({"reason": "little_new_evidence"}, readable, ledger) == "evidence_limited"


def test_report_renders_deep_sections():
    data, _, _, _ = deep_run()
    report = render_markdown(data)
    for heading in ("## Direct answer", "## Summary", "## Coverage by question", "## Evidence quality",
                    "## Method and stopping reason", "## Sources"):
        assert heading in report
    assert "(finding 1)" in report and "corroborated" in report and "Round 2" in report
    assert "35 percent" not in report                                          # the dropped summary sentence


# ================================================================================================= server
class Instant:
    def __init__(self):
        self.options = []

    def run(self, text, mode, progress=None, options=None):
        self.options.append(options)
        return {"input": text, "mode": mode, "status": "COMPLETE", "outcome": "ANSWERED", "stages": []}


def call(url, data=None, headers=None, method=None):
    request = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


@pytest.fixture()
def server(tmp_path, monkeypatch):
    for name in ("WEBRAG_PROVIDERS", "SEARXNG_URL", "BRAVE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    runner = Instant()
    jobs = JobQueue(lambda: runner)
    store = ConfigStore(tmp_path / "config.json")
    tested = []

    def fake_test(provider_id, config):
        tested.append(provider_id)
        return {"id": provider_id, "ok": True, "results": 2}

    from webrag.studio import EvaluationStore, RunStore
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(jobs, 0))
    port = httpd.server_address[1]
    httpd.RequestHandlerClass = make_handler(jobs, port, runs=RunStore(tmp_path / "runs"), evaluations=EvaluationStore(tmp_path),
                                             config_store=store, provider_test=fake_test)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{port}", jobs, runner, store, tested
    httpd.shutdown()
    httpd.server_close()


def post(base, path, body):
    return call(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"}, "POST")


def wait(jobs, job_id):
    for _ in range(200):
        if jobs.get(job_id)["state"] == "done":
            return jobs.get(job_id)
        threading.Event().wait(0.02)
    raise AssertionError("job did not finish")


def test_run_options_are_validated_against_active_providers(server):
    base, jobs, runner, _, _ = server
    assert post(base, "/api/run", {"input": "Q?", "depth": "bottomless"})[0] == 400
    status, body = post(base, "/api/run", {"input": "Q?", "depth": "deep", "providers": ["crossref"]})
    assert status == 400 and "not active" in body["error"]
    assert post(base, "/api/run", {"input": "Q?", "depth": "deep", "categories": ["tv"]})[0] == 400
    assert post(base, "/api/run", {"input": "Q?", "depth": "deep", "recency": "decade"})[0] == 400
    assert post(base, "/api/run", {"input": "Q?", "depth": "deep", "providers": []})[0] == 400
    status, body = post(base, "/api/run", {"input": "Q?", "depth": "quick", "providers": ["wikipedia"]})
    assert status == 400 and "not active" in body["error"]                    # source policy: never a run provider
    status, body = post(base, "/api/run", {"input": "Q?", "mode": "verify", "depth": "quick", "providers": ["openalex"],
                                           "categories": ["scholarly"], "recency": "year"})
    assert status == 202
    job = wait(jobs, body["id"])
    options = runner.options[-1]
    assert options["depth"] == "quick" and options["providers"] == ["openalex"] and options["recency"] == "year"
    status, body = post(base, "/api/run", {"input": "Q?", "depth": "quick"})
    wait(jobs, body["id"])
    assert "wikipedia" not in runner.options[-1]["providers"]                   # nor among the defaults
    assert "config" in options and "config" not in json.dumps(job)            # the configuration stays on the server
    status, body = post(base, "/api/run", {"input": "Plain question?"})
    wait(jobs, body["id"])
    assert runner.options[-1] is None                                         # no depth: the original pipeline


def test_provider_catalog_and_configuration_routes(server):
    base, _, _, store, tested = server
    status, payload = call(base + "/api/providers")
    assert status == 200 and set(payload["depths"]) == {"quick", "standard", "deep"}
    rows = {r["id"]: r for r in payload["providers"]}
    assert not rows["wikipedia"]["active"] and rows["wikipedia"]["evidence"] is False and rows["openalex"]["active"]
    assert not rows["crossref"]["active"] and payload["config"]["environment"]["BRAVE_API_KEY"] is False
    status, refused = post(base, "/api/providers/config", {"providers": {"wikipedia": {"enabled": True}}})
    assert status == 400 and "not accepted as an evidence source" in refused["error"]
    status, payload = post(base, "/api/providers/config", {"providers": {"crossref": {"enabled": True}},
                                                           "feeds": [{"url": "https://news.example/rss", "category": "news"}]})
    rows = {r["id"]: r for r in payload["providers"]}
    assert status == 200 and rows["crossref"]["active"] and rows["feeds"]["active"]
    assert store.load()["providers"]["crossref"] == {"enabled": True}
    assert post(base, "/api/providers/config", {"brave_api_key": "x"})[0] == 400                 # never stores secrets
    assert post(base, "/api/run", {"input": "Q?", "depth": "quick", "providers": ["crossref", "feeds"]})[0] == 202
    assert post(base, "/api/providers/openalex/test", {})[1] == {"id": "openalex", "ok": True, "results": 2} and tested == ["openalex"]
    refused = post(base, "/api/providers/wikipedia/test", {})[1]
    assert refused["ok"] is False and refused["excluded"] and tested == ["openalex"]      # nothing sent for Wikipedia
    assert call(base + "/api/providers/config", method="GET")[0] == 404


def test_check_provider_uses_one_search_and_reports_configuration():
    session = FakeSession([(lambda u, p: "api.openalex.org" in u,
                            page({"results": [{"id": "W1", "display_name": "A", "doi": "https://doi.org/10.1/a"}]}, "application/json"))])
    result = check_provider("openalex", validate({}), session=session)
    assert result["ok"] and result["results"] == 1 and len(session.calls) == 1
    untouched = FakeSession()
    refused = check_provider("wikipedia", validate({}), session=untouched)
    assert refused["ok"] is False and refused["excluded"] and untouched.calls == []      # source policy: nothing is sent
    assert check_provider("searxng", validate({}), session=FakeSession())["configured"] is False
    assert check_provider("bing", {}, session=FakeSession())["error"] == "unknown provider"
    failing = FakeSession([(lambda u, p: True, FetchError("http_500", "https://x.example/?api_key=SECRET"))])
    result = check_provider("openalex", validate({}), session=failing)
    assert not result["ok"] and "SECRET" not in result["error"]


# ================================================================================================= source policy: Wikipedia
def test_wikipedia_hosts_and_provider_are_recognised():
    for url in ("https://en.wikipedia.org/wiki/X", "https://en.m.wikipedia.org/wiki/X", "http://WIKIPEDIA.ORG/x"):
        assert excluded_host(url) == "wikipedia.org"
    for url in ("https://notwikipedia.org/x", "https://wikipedia.org.evil.example/x", "https://example.org/wikipedia.org", ""):
        assert excluded_host(url) == ""
    assert not is_evidence_provider("wikipedia") and is_evidence_provider("openalex") and is_evidence_provider("a-fake")
    page_source = Source("S1", "T", "https://short.example/a", "searxng", "web", "url:a", FULL_TEXT, "Readable text.",
                         read_url="https://en.wikipedia.org/wiki/A")
    assert not is_evidence(page_source)                                        # the index gate checks the URL actually read
    assert is_evidence(replace(page_source, read_url="https://grid.example/report"))


def test_a_redirect_into_wikipedia_and_feed_links_to_it_never_become_sources():
    providers, session = world()
    session.routes.insert(0, (lambda u, p: u == "https://short.example/solar",
                              page("<html><body><main><p>" + (WIKI_TEXT + " ") * 6 + "</p></main></body></html>", url=WIKI_URL)))
    feeds = FakeProvider("feeds", "web", lambda q: [("Solar on Wikipedia", "https://en.m.wikipedia.org/wiki/Solar", "Feed item.")])
    web = FakeProvider("searxng", "web", lambda q: [("Short link", "https://short.example/solar", "Short snippet.", "", None, "Shortener"),
                                                    ("Grid report", "https://grid.example/report", "s", "", None, "Grid Operator")])
    researcher = WebResearcher([web, feeds], session, classify=True)
    corpus = researcher.gather(["solar"])
    assert [s.title for s in corpus.sources] == ["Grid report"] and corpus.excluded_by_policy == 2
    assert not [c for c in session.calls if "m.wikipedia.org" in c["url"]]     # the feed's Wikipedia link was not fetched
    data, _, _, _ = deep_run(providers=[web, feeds], session=session)
    assert all("wikipedia" not in (s["read_url"] or s["url"]) for s in data["sources"])
    assert all("wikipedia" not in p["url"] for c in data["claims"] for p in c["passages"])


def test_a_run_offered_only_wikipedia_searches_nothing():
    providers, session = world()
    wiki = providers[0]
    data, client, _, _ = deep_run(providers=[wiki], session=session)
    assert data["status"] == "FAILED" and "not accepted as an evidence source" in data["error"]
    assert wiki.calls == [] and session.calls == [] and data["options"]["providers"] == []
    assert "plan" not in [k for k, _ in client.seen]                         # nothing planned, nothing searched


def test_wikipedia_text_cannot_support_a_finding_or_satisfy_coverage():
    """With Wikipedia's text available to the fakes, the statement it alone supports is never supported."""
    wiki_only = "Rooftop installations are common in cities."               # only in WIKI_TEXT
    providers, session = world()
    synth = {"findings": [{"text": wiki_only, "evidence": ["E1"], "subquestion": 1}], "disagreements": [], "gaps": [],
             "insufficient": False}
    data, _, verifier, _ = deep_run(text=wiki_only, mode="verify", replies={"plan": PLAN, "synthesis": synth, "summary": SUMMARY})
    assert data["outcome"] != SUPPORTED and data["claims"][0]["supporting_sources"] == []
    assert all(wiki_only not in " ".join(texts) for _, texts in verifier.calls)  # its text never reached the verifier


def test_studio_orchestrator_never_carries_wikipedia(monkeypatch):
    monkeypatch.setenv("WEBRAG_PROVIDERS", "wikipedia,openalex")
    for name in ("SEARXNG_URL", "BRAVE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    assert [p.name for p in studio_orchestrator("qwen3:4b", "quote_stance").providers] == ["openalex"]
    assert [p.name for p in Orchestrator(Settings(save=False)).providers] == ["wikipedia", "openalex"]   # command line unchanged


# ================================================================================================= defect 1: response shapes
def json_session(body):
    return FakeSession([(lambda u, p: True, page(body, "application/json"))])


@pytest.mark.parametrize("search, body, expected", [
    (lambda s: BraveSearch(s, "k").search("q"), {"web": None}, "'web' as null"),
    (lambda s: BraveSearch(s, "k").search("q"), {"web": {"results": "none"}}, "'results' as a string"),
    (lambda s: OpenAlexSearch(s).search("q"), [{"results": []}], "an array at the top level"),
    (lambda s: OpenAlexSearch(s).search("q"), {"results": [{"id": "W1", "authorships": [1]}]}, "authorships that is not an object"),
    (lambda s: OpenAlexSearch(s).search("q"), {"results": [{"abstract_inverted_index": {"word": 3}}]}, "abstract index"),
    (lambda s: SearxngSearch(s, "http://127.0.0.1:8888").search("q"), ["not", "an", "object"], "top level"),
    (lambda s: WikipediaSearch(s).search("q"), {"query": {"search": [1, 2]}}, "search results that are not an object"),
    (lambda s: EuropePmcSearch(s).search("q"), {"resultList": {"result": {"id": 1}}}, "'result' as an object"),
    (lambda s: CrossrefSearch(s).search("q"), {"message": {"items": [{"DOI": "10.1/x", "title": "A string"}]}}, "'title' as a string"),
    (lambda s: CrossrefSearch(s).search("q"), {"message": {"items": [{"DOI": "10.1/x", "title": ["T"],
                                                                      "published": {"date-parts": [["2020"]]}}]}}, "date"),
])
def test_valid_json_of_the_wrong_shape_is_a_provider_error(search, body, expected):
    with pytest.raises(SearchError) as caught:
        search(json_session(body))
    assert str(caught.value).startswith("malformed_response:") and expected in str(caught.value)


def test_documented_nullable_and_absent_fields_are_still_accepted():
    hits = OpenAlexSearch(json_session({"results": [{"id": "W1", "display_name": "A", "doi": None, "primary_location": None,
                                                     "best_oa_location": None, "abstract_inverted_index": None,
                                                     "authorships": None, "cited_by_count": 3}]})).search("q")
    assert hits[0].url == "W1" and hits[0].abstract == "" and hits[0].extra["cited_by"] == "3"
    assert BraveSearch(json_session({"query": {}}), "k").search("q") == []          # no 'web' key: no web results
    assert EuropePmcSearch(json_session({"resultList": None})).search("q") == []
    from webrag.researcher import OpenAccessResolver
    assert OpenAccessResolver(json_session([1, 2])).resolve("10.1/x")["error"].startswith("malformed_response")
    assert OpenAccessResolver(json_session({"best_oa_location": None})).resolve("10.1/x") == \
        {"pdf_url": "", "landing_url": "", "license": "", "oa_status": ""}


def test_a_malformed_provider_does_not_discard_other_results():
    malformed = OpenAlexSearch(json_session([{"unexpected": "top-level list"}]))
    good = FakeProvider("searxng", "web", lambda q: [("Grid report", "https://grid.example/report", "s", "", None, "Grid Operator")])
    _, session = world()
    corpus = WebResearcher([malformed, good], session, classify=True).gather(["solar"])
    assert [s.title for s in corpus.sources] == ["Grid report"]                # the valid provider's results survive
    assert corpus.search_errors[0][0] == "openalex" and corpus.search_errors[0][2].startswith("malformed_response")
    status = corpus.provider_status
    assert status["openalex"]["succeeded"] == 0 and status["openalex"]["failures"] == {"malformed_response": 1}
    assert status["searxng"]["succeeded"] == 1 and status["searxng"]["failed"] == 0


def test_a_deep_run_survives_a_malformed_provider_and_reports_it():
    providers, session = world()
    brave = BraveSearch(json_session({"web": None}), "k")
    data, _, _, _ = deep_run(providers=[providers[1], providers[2], brave], session=session)
    assert data["status"] == "PARTIAL" and data["outcome"] == "ANSWERED"      # not FAILED, and not reported as clean
    assert any("'brave' failed" in x and "malformed_response" in x for x in data["limitations"])
    first = data["rounds"][0]["provider_status"]["brave"]
    assert first["succeeded"] == 0 and first["failed"] == first["queries"] > 0


class BrokenProvider:
    """A programmer error inside a provider: not a malformed response, so it must not be swallowed."""
    name, kind = "searxng", "web"

    def search(self, query, limit=3, **kwargs):
        raise RuntimeError("bug in provider code")


def test_programmer_errors_are_not_swallowed():
    with pytest.raises(RuntimeError, match="bug in provider code"):
        WebResearcher([BrokenProvider()], FakeSession(), classify=True).gather(["q"])
    data, _, _, _ = deep_run(providers=[BrokenProvider()])
    assert data["status"] == "FAILED" and data["error"] == "Unexpected error: RuntimeError: bug in provider code"


# ================================================================================================= defect 2: independence
def src(sid, url, publisher="", kind="web", authors=(), identifiers=None, origin=None, access=FULL_TEXT):
    """A source record as the researcher writes it: without a recorded name, the publisher field holds the domain."""
    host = url.split("/")[2]
    domain = host[4:] if host.startswith("www.") else host
    return {"id": sid, "url": url, "read_url": url, "domain": domain, "publisher": publisher or domain, "kind": kind,
            "authors": list(authors), "identifiers": identifiers or {}, "origin": origin or sid, "access": access}


def test_two_urls_on_the_same_site_are_one_publisher():
    same = independence([src("A", "https://news.example.com/a", "Example News"), src("B", "https://news.example.com/b", "Other Name")])
    assert same["status"] == "single" and same["groups"] == 1
    assert independence([src("A", "https://site.example/a"), src("B", "https://site.example/b")])["status"] == "single"


def test_one_publisher_across_subdomains_and_domains_is_one_publisher():
    assert site_of("news.example.com") == site_of("www.example.com") == "example.com"
    assert site_of("news.bbc.co.uk") == "bbc.co.uk" and site_of("127.0.0.1") == "127.0.0.1"
    across = independence([src("A", "https://news.example.com/a", "Example News"), src("B", "https://blog.example.com/b", "Example Blog")])
    assert across["status"] == "single"                                        # same site, different section names
    named = independence([src("A", "https://example-media.net/a", "Example Media"), src("B", "https://examplemedia.org/b", "Example  media")])
    assert named["status"] == "single"                                         # same recorded publisher, two domains


def test_distinct_publishers_with_metadata_are_independent():
    pages = independence([src("A", "https://alpha.example/a", "Alpha News"), src("B", "https://beta.example/b", "Beta Times")])
    assert pages["status"] == "established" and pages["independent"] == 2
    assert site_of("news.bbc.co.uk") != site_of("www.itv.co.uk")
    works = [src("P1", "https://doi.org/10.1/a", "J. One", "scholarly", ["Smith, Jane", "Lee, Kim"], {"doi": "10.1/a"}),
             src("P2", "https://doi.org/10.2/b", "J. Two", "scholarly", ["Ana Ortiz"], {"doi": "10.2/b"})]
    # deep-research-v2: different teams alone are not enough; both reference lists must be checkable
    assert independence(works)["status"] == "unknown"
    checked = [dict(w, text=f"Body of {w['id']}.\nReferences\n[1] An unrelated work, 2019.") for w in works]
    assert independence(checked)["status"] == "established"                    # different teams, no cross-citation
    claim = {"label": SUPPORTED, "parse_ok": True, "contradicting_sources": [], "independence": pages,
             "passages": [{"source_id": "A", "stance": "SUPPORTS"}, {"source_id": "B", "stance": "SUPPORTS"}]}
    assert strength(claim, {"A": {"access": FULL_TEXT}, "B": {"access": FULL_TEXT}}) == "corroborated"


def test_missing_or_ambiguous_publisher_metadata_leaves_independence_unknown():
    unnamed = independence([src("A", "https://alpha.example/a"), src("B", "https://beta.example/b")])
    assert unnamed["status"] == "unknown" and unnamed["groups"] == 2 and unnamed["independent"] == 1
    half = independence([src("A", "https://alpha.example/a", "Alpha News"), src("B", "https://beta.example/b")])
    assert half["status"] == "unknown"
    no_authors = independence([src("P1", "https://doi.org/10.1/a", "", "scholarly"), src("P2", "https://doi.org/10.2/b", "", "preprint")])
    assert no_authors["status"] == "unknown"
    same_team = independence([src("P1", "https://doi.org/10.1/a", "", "scholarly", ["Smith, Jane"]),
                              src("P2", "https://arxiv.org/abs/1", "", "preprint", ["J. Smith", "Bo Li"])])
    assert same_team["status"] == "single"                                     # a shared author: one team
    mixed = independence([src("P1", "https://doi.org/10.1/a", "", "scholarly", ["Ana Ortiz"]), src("A", "https://alpha.example/a", "Alpha News")])
    assert mixed["status"] == "unknown"                                        # a page about a study, and the study
    claim = {"label": SUPPORTED, "parse_ok": True, "contradicting_sources": [], "independence": unnamed,
             "passages": [{"source_id": "A", "stance": "SUPPORTS"}, {"source_id": "B", "stance": "SUPPORTS"}]}
    assert strength(claim, {"A": {"access": FULL_TEXT}, "B": {"access": FULL_TEXT}}) == "independence unknown"
    three = independence([src("A", "https://alpha.example/a", "Alpha News"), src("B", "https://beta.example/b", "Beta Times"),
                          src("C", "https://gamma.example/c")])
    assert three["status"] == "established" and three["independent"] == 2     # C is unknown, A and B are established


def test_near_identical_text_from_nominally_different_publishers_is_one_origin():
    copies = independence([src("A", "https://alpha.example/a", "Alpha News"), src("B", "https://beta.example/b", "Beta Times", origin="A")])
    assert copies["status"] == "single"
    work_twice = independence([src("P1", "https://doi.org/10.1/a", "", "scholarly", ["Ana Ortiz"], {"doi": "10.1/A"}),
                               src("P2", "https://repo.example/a.pdf", "", "scholarly", ["Bo Li"], {"doi": "10.1/a"})])
    assert work_twice["status"] == "single"                                    # one work found twice


def test_deduplication_still_limits_redundant_passages_but_does_not_decide_independence():
    copy_html = WEB_HTML.replace("<main>", "<main><p>Republished.</p>")
    session = FakeSession([(lambda u, p: u == "https://grid.example/report", page(WEB_HTML, url="https://grid.example/report")),
                           (lambda u, p: u == "https://copy.example/report", page(copy_html, url="https://copy.example/report")),
                           (lambda u, p: u == "https://grid.example/other", page(STATS_HTML, url="https://grid.example/other"))])
    web = FakeProvider("searxng", "web", lambda q: [("Grid report", "https://grid.example/report", "s", "", None, "Grid Operator"),
                                                    ("Copy", "https://copy.example/report", "s", "", None, "Copy Desk"),
                                                    ("Other page", "https://grid.example/other", "s", "", None, "Grid Operator")])
    sources = WebResearcher([web], session, classify=True).gather(["solar"]).sources
    grid, copy, other = sources
    assert copy.origin == grid.id and "not counted as independent" in copy.note and other.origin == other.id
    store = Store(sources)
    capped = cap_per_origin(store.passages, store.sources, k=20, per_source=1)
    assert len([p for p in capped if p.source_id in {grid.id, copy.id}]) == 1  # copies still give one passage
    records = [Orchestrator._source_record(s) for s in sources]
    quality = source_quality(records, [])
    assert quality["distinct_texts"] == 2 and quality["publisher_groups"] == 1  # two texts, one publisher
    assert independence([grid, other])["status"] == "single"                    # different texts, same site


def test_unknown_independence_is_never_corroboration_or_sufficient_coverage():
    providers, session = world()
    unnamed = FakeProvider("searxng", "web", lambda q: [r[:3] for r in providers[2].hits_for(q)])  # no recorded publishers
    data, _, _, _ = deep_run(depth="deep", providers=[providers[1], unnamed], session=session,
                             replies={"plan": dict(PLAN, subquestions=[PLAN["subquestions"][0]]),
                                      "synthesis": dict(SYNTH, findings=SYNTH["findings"][:1]), "summary": SUMMARY,
                                      "follow-up": FOLLOW, "counter": COUNTER})
    growth = next(c for c in data["claims"] if c["text"] == f"{GROWTH}.")
    assert growth["label"] == SUPPORTED and growth["independence"]["status"] == "unknown"
    assert growth["strength"] == "independence unknown" and growth["independent_support"] >= 2   # distinct texts, kept as recorded
    assert data["stopping"]["reason"] != "retrieval_coverage_met"                 # never counted as strong coverage
    assert {e["index"]: e for e in data["ledger"]}[1]["status"] == "partially answered"
    assert data["counter_evidence"]["targets"] == [0]                           # and challenged with counter-evidence


# ================================================================================================= defect 3: run status
def test_all_providers_returning_nothing_is_a_completed_run_without_an_answer():
    empty = [FakeProvider("searxng", "web", lambda q: []), FakeProvider("openalex", "scholarly", lambda q: [])]
    data, _, _, _ = deep_run(providers=empty)
    assert data["status"] == "COMPLETE" and data["outcome"] == "INSUFFICIENT_EVIDENCE" and data["claims"] == []
    assert data["error"] == "The searches returned no results for this input."
    assert data["summary"]["available"] is False and all(e["status"] == "not reached" for e in data["ledger"])
    verify, _, _, _ = deep_run(providers=empty, mode="verify", text=f"{GROWTH}.")
    assert verify["status"] == "COMPLETE" and verify["outcome"] == INSUFFICIENT and verify["summary"]["available"] is False


def test_sources_found_but_none_extracted_is_a_completed_run_without_an_answer():
    session = FakeSession([(lambda u, p: u.endswith(".pdf"), page(b"%PDF-1.4 broken", "application/pdf")),
                           (lambda u, p: u.endswith(".zip"), page(b"PK..", "application/zip")),
                           (lambda u, p: u.endswith("/tiny"), page("<html><body><p>Too short.</p></body></html>"))])
    web = FakeProvider("searxng", "web", lambda q: [("Broken PDF", "https://x.example/a.pdf"), ("Archive", "https://x.example/data.zip"),
                                                    ("Tiny page", "https://x.example/tiny")])
    data, _, _, _ = deep_run(providers=[web], session=session)
    assert data["status"] == "COMPLETE" and data["outcome"] == "INSUFFICIENT_EVIDENCE"
    assert "3 source(s) were found, but none could be read" in data["error"]
    assert {s["access"] for s in data["sources"]} == {FAILED} and data["summary"]["available"] is False
    failing = FakeProvider("openalex", "scholarly", lambda q: [], error="timeout: slow")
    partial, _, _, _ = deep_run(providers=[web, failing], session=session)
    assert partial["status"] == "PARTIAL" and partial["outcome"] == "INSUFFICIENT_EVIDENCE"   # some searches failed


def test_execution_failures_stay_failed():
    assert run_status({"stages": [], "search_errors": []}, failed=True, nothing_readable=True) == "FAILED"
    assert run_status({"status": "FAILED", "stages": []}, failed=False) == "FAILED"
    assert run_status({"stages": [{"status": "failed"}], "search_errors": []}, failed=False, nothing_readable=True) == "COMPLETE"
    assert run_status({"stages": [{"status": "partial"}]}, failed=False) == "PARTIAL"

    def broken_store(sources, models):
        raise RuntimeError("index unavailable")
    providers, session = world()
    orch = Orchestrator(Settings(save=False, per_query=6), client=DeepClient({"plan": PLAN}), session=session,
                        providers=providers, verifier=EvidenceVerifier(), store_factory=broken_store)
    data = orch.run("How fast is solar growing?", options={"depth": "quick"})
    assert data["status"] == "FAILED" and data["error"] == "Unexpected error: RuntimeError: index unavailable"


# ================================================================================================= evidence quality A:
# passage boundaries. All page texts below are SYNTHETIC; the first reproduces the page-split pattern of a saved run
# (a word hyphenated across a PDF page break, footers after it, a running header before the continuation).
from webrag.deep import cautions, fragment_caution, passage_boundary   # noqa: E402

HEADER = "Front. Sci. Rev., 2024, 0(0): 1-17"
SPLIT_PAGES = [
    f"{HEADER}\nREVIEW ARTICLE\nFraud detection has a long history. Studies have demonstrated that deep learning mod-\n"
    "Received month dd, yyyy; accepted month dd, yyyy\nE-mail: author@example.org\narXiv:0000.00000v1 [cs.LG] 1 Jan 2024",
    f"2 {HEADER}\nels outperform traditional methods in terms of accuracy and scalability. Graph neural networks model\n"
    "relations between accounts. Several models were compared on public data.",
    f"{HEADER} 3\nThe review ends here.",
]
SPLIT_PASSAGE = (f"2 {HEADER} els outperform traditional methods in terms of accuracy and scalability. Graph neural "
                 "networks model relations between accounts.")


def test_boundary_page_split_pattern_is_a_confirmed_mid_word_fragment():
    b = passage_boundary(SPLIT_PAGES, 2, SPLIT_PASSAGE)
    assert b["boundary"] == "mid_word" and "'models' occurs in the source" in b["evidence"]
    assert b["lead"].startswith("els outperform traditional methods") and b["lead"].endswith("scalability.")


def test_boundary_complete_sentence_and_document_start():
    page = "Fraud is costly. Graph neural networks model relations between accounts. They need labelled data."
    assert passage_boundary([page], 1, "Graph neural networks model relations between accounts.")["boundary"] == "complete"
    assert passage_boundary([page], 1, "Fraud is costly.")["boundary"] == "complete"            # starts the document
    # a sentence-ending last line on the previous page, a capitalised start after the running header
    assert passage_boundary(SPLIT_PAGES, 3, "The review ends here.")["boundary"] == "complete"


def test_boundary_contiguous_mid_word_and_mid_sentence_fragments():
    page = "Studies show that deep learning models outperform rule-based systems on most benchmarks."
    cut_word = passage_boundary([page], 1, "els outperform rule-based systems on most benchmarks.")
    assert cut_word["boundary"] == "mid_word" and "'models'" in cut_word["evidence"]
    cut_sentence = passage_boundary([page], 1, "outperform rule-based systems on most benchmarks.")
    assert cut_sentence["boundary"] == "mid_sentence"


def test_boundary_ambiguous_context_is_uncertain_not_truncated():
    heading = "3 Results\nGraph models improved recall on two datasets."
    assert passage_boundary([heading], 1, "Graph models improved recall on two datasets.")["boundary"] == "uncertain"
    after_end = "The study closed in 2020. e.g. graph models were used."
    assert passage_boundary([after_end], 1, "e.g. graph models were used.")["boundary"] == "uncertain"
    no_hyphen = ["Page one ends without a sentence end", "continues on the next page with more text."]
    assert passage_boundary(no_hyphen, 2, "continues on the next page with more text.")["boundary"] == "uncertain"
    unattested = ["A line that ends with qux-", "zzy words follow here."]                 # 'quxzzy' occurs nowhere
    assert passage_boundary(unattested, 2, "zzy words follow here.")["boundary"] == "uncertain"


def test_boundary_missing_page_context_is_uncertain():
    assert passage_boundary([], 1, "Anything.")["boundary"] == "uncertain"
    assert passage_boundary(["Some page text."], 3, "Some page text.")["boundary"] == "uncertain"
    missing = passage_boundary(["Some page text."], 1, "A passage that is not on the page.")
    assert missing["boundary"] == "uncertain" and "not found" in missing["evidence"]


def _support(text, page=2, key="S1:p2:a", pages=SPLIT_PAGES):
    record = {"key": key, "source_id": "S1", "page": page, "text": text, "stance": "SUPPORTS"}
    return record | passage_boundary(pages, page, text)


def test_fragment_caution_only_when_the_finding_rests_on_the_cut_off_lead():
    passage = _support(SPLIT_PASSAGE)
    wrong_subject = "Graph neural networks outperform traditional methods in accuracy and scalability."
    caution = fragment_caution(wrong_subject, [passage])
    assert caution["kind"] == "fragment_subject" and "begins mid-word" in caution["detail"]
    later = "Graph neural networks model relations between accounts."              # a later, complete sentence
    assert fragment_caution(later, [passage]) is None
    complete = _support("Graph neural networks model relations between accounts.", key="S1:p2:b")
    assert complete["boundary"] not in {"mid_word", "mid_sentence"}
    assert fragment_caution(wrong_subject, [complete]) is None                    # no fragment, no caution
    assert fragment_caution(wrong_subject, [dict(passage, stance="NEUTRAL")]) is None   # supporting passages only
    record = {"label": SUPPORTED, "text": wrong_subject, "passages": [passage]}
    assert [c["kind"] for c in cautions(record)] == ["fragment_subject"]
    assert cautions(dict(record, label=INSUFFICIENT)) == []                       # only verified statements


class CutStore(Store):
    """Like Store, but cuts the sentence holding 'deep learning models' inside the word 'models', as a chunker can."""

    def add_sources(self, sources):
        super().add_sources(sources)
        out = []
        for p in self.passages:
            at = p.text.find("learning mod")
            if at >= 0:
                cut = at + len("learning mod")
                out += [replace(p, key=p.key + "a", text=p.text[:cut]), replace(p, key=p.key + "b", text=p.text[cut:])]
            else:
                out.append(p)
        self.passages = out


class OverlapVerifier:
    """SUPPORTS a passage sharing three or more content words with the claim (a lenient fake verifier)."""

    def verify(self, claim, evidence):
        words = set(re.findall(r"[a-z]{4,}", claim.lower()))
        stances = {e.id: "SUPPORTS" if len(words & set(re.findall(r"[a-z]{4,}", e.text.lower()))) >= 3 else "NEUTRAL"
                   for e in evidence}
        supported = "SUPPORTS" in stances.values()
        return Verdict(SUPPORTED if supported else INSUFFICIENT, "" if supported else "NO_DECISIVE_EVIDENCE", stances=stances)


FRAUD_HTML = ("<html><body><article><p>Fraud detection is an old field. Studies have demonstrated that deep learning "
              "models outperform traditional methods in accuracy and scalability. Graph neural networks represent "
              "transactions as connected accounts across banks.</p>"
              + "<p>The review covers payment networks, card fraud and money laundering across many institutions.</p>" * 4
              + "</article></body></html>")
FRAUD_PLAN = {"subquestions": [], "queries": [{"subquestion": 0, "category": "web", "query": "graph neural networks fraud"}]}


def _fraud_run(replies, verifier=None):
    web = FakeProvider("searxng", "web", lambda q: [("Fraud review", "https://review.example/gnn", "snippet", "", None,
                                                     "Review Press")])
    session = FakeSession([(lambda u, p: u == "https://review.example/gnn", page(FRAUD_HTML, url="https://review.example/gnn"))])
    client = DeepClient(replies)
    orch = Orchestrator(Settings(save=False, per_query=6), client=client, session=session, providers=[web],
                        verifier=verifier or OverlapVerifier(), store_factory=CutStore)
    return orch.run("Do graph neural networks outperform traditional methods using connected accounts?",
                    options={"depth": "quick"}), client


def _label_of(messages, needle):
    return next(b.split("]")[0] for b in messages[1]["content"].split("\n[") if needle in b and b.startswith("E"))


def _cite_all(text):
    def summary(messages):
        labels = re.findall(r"\[(F\d+)\]", messages[1]["content"])
        return {"answered": True, "direct_answer": {"text": text, "findings": labels}, "summary": []}
    return summary


def test_fragment_finding_is_cautioned_listed_and_left_out_of_the_answer():
    def synthesis(messages):
        cut, whole = _label_of(messages, "els outperform"), _label_of(messages, "connected accounts")
        return {"findings": [{"text": "Graph neural networks outperform traditional methods in accuracy and scalability.",
                              "evidence": [cut], "subquestion": 0},
                             {"text": "Graph neural networks represent transactions as connected accounts across banks.",
                              "evidence": [whole], "subquestion": 0}],
                "disagreements": [], "gaps": [], "insufficient": False}

    data, client = _fraud_run({"plan": FRAUD_PLAN, "synthesis": synthesis,
                               "summary": _cite_all("Graph neural networks represent transactions as connected accounts across banks.")})
    assert data["status"] in {"COMPLETE", "PARTIAL"}, data["error"]
    assert data["sources"][0]["access"] == FULL_TEXT
    prompt = next(m for k, m in client.seen if k == "synthesis")
    assert "begins mid-word: the words before it are cut off" in prompt[1]["content"]
    assert "never name a subject for its first words" in prompt[0]["content"]
    claims = {c["text"]: c for c in data["claims"]}
    cut = claims["Graph neural networks outperform traditional methods in accuracy and scalability."]
    whole = claims["Graph neural networks represent transactions as connected accounts across banks."]
    assert cut["label"] == SUPPORTED and [w["kind"] for w in cut["cautions"]] == ["fragment_subject"]   # verdict kept
    assert any(p["boundary"] == "mid_word" for p in cut["passages"])
    assert whole["label"] == SUPPORTED and whole["cautions"] == []
    summary_prompt = next(m for k, m in client.seen if k == "summary")[1]["content"].split("Verified findings:")[1]
    assert "outperform" not in summary_prompt and "connected accounts" in summary_prompt
    assert data["summary"]["direct_answer"]["claims"] == [data["claims"].index(whole)]
    assert any("carry an evidence-quality caution" in x for x in data["limitations"])
    report = render_markdown(data)
    assert "Caution (fragment subject)" in report and "outperform traditional methods" in report.split("## Verified findings")[1]


# ================================================================================================= evidence quality C:
# retrieval coverage, answer coverage, completeness, the stopping trigger and the execution status are kept apart.
from webrag.deep import stopping_record   # noqa: E402

RETRIEVAL_STOP = {"reason": "retrieval_coverage_met", "detail": "Retrieval check met.", "also_reached": ["round_limit"],
                  "retrieval_coverage": {"met": True, "states": {"0": "strong", "1": "strong", "2": "strong", "3": "strong"}}}
READABLE = [{"access": FULL_TEXT}, {"access": FULL_TEXT}]


def test_retrieval_coverage_with_open_questions_is_partial_and_names_them():
    ledger = [{"index": 0, "status": "answered"}, {"index": 1, "status": "answered"},
              {"index": 2, "status": "partially answered"}, {"index": 3, "status": "unanswered"}]
    stop = stopping_record(RETRIEVAL_STOP, [], READABLE, ledger)
    assert stop["reason"] == "retrieval_coverage_met" and stop["completeness"] == "partial"
    assert "question 2 is partially answered; question 3 is unanswered" in stop["detail"]
    assert stop["answer_coverage"] == {"questions": 4, "answered": 2, "open": [{"index": 2, "status": "partially answered"},
                                                                              {"index": 3, "status": "unanswered"}]}
    assert stop["also_reached"] == ["round_limit"] and stop["retrieval_coverage"]["met"] is True
    assert RETRIEVAL_STOP["detail"] == "Retrieval check met."                       # the input is not modified


def test_retrieval_coverage_with_every_question_answered_is_complete():
    ledger = [{"index": 0, "status": "answered"}, {"index": 1, "status": "answered with disagreement"}]
    stop = stopping_record(RETRIEVAL_STOP, [], READABLE, ledger)
    assert stop["completeness"] == "complete" and stop["detail"] == "Retrieval check met."
    assert stop["answer_coverage"]["open"] == []
    budget = stopping_record({"reason": "round_limit", "detail": "One round."}, [{"index": 0, "state": "weak"}], READABLE,
                             [{"index": 0, "status": "partially answered"}])
    assert budget["completeness"] == "budget_limited" and budget["also_reached"] == []
    assert budget["retrieval_coverage"] == {"met": False, "states": {"0": "weak"}}
    assert stopping_record(None, [], READABLE, [])["reason"] == "round_limit"        # the default stop is unchanged


def test_quick_run_records_round_limit_with_retrieval_coverage_and_the_unanswered_question():
    plan = dict(PLAN, subquestions=[PLAN["subquestions"][0]])
    synth = dict(SYNTH, findings=[dict(SYNTH["findings"][0], subquestion=0)])     # answers the main question only
    data, _, _, _ = deep_run(depth="quick", replies={"plan": plan, "synthesis": synth, "summary": SUMMARY})
    stop = data["stopping"]
    assert stop["reason"] == "retrieval_coverage_met" and stop["also_reached"] == ["round_limit"]
    assert stop["retrieval_coverage"]["met"] is True
    assert stop["completeness"] == "partial" and "question 1 is unanswered" in stop["detail"]
    assert stop["answer_coverage"]["open"] == [{"index": 1, "status": "unanswered"}]
    assert {e["index"]: e["status"] for e in data["ledger"]} == {0: "answered", 1: "unanswered"}
    # the execution status is decided by the stages alone, exactly as before
    assert data["status"] == run_status(dict(data, status="RUNNING"), failed=False) and data["status"] != "FAILED"
    report = render_markdown(data)
    assert "Limits also reached when searching stopped: round limit." in report
    assert "Retrieval coverage (passages found, not answers): met. Answer coverage (verified): 1 of 2" in report
    assert "question 1 (unanswered)" in report


def test_execution_status_does_not_depend_on_completeness():
    for value in ("partial", "complete", "budget_limited"):
        assert run_status({"stages": [{"status": "ok"}], "stopping": {"completeness": value}}, failed=False) == "COMPLETE"


def test_a_v1_stopping_record_still_renders():
    data, _, _, _ = deep_run(depth="quick")
    data["stopping"] = {"reason": "coverage_sufficient", "detail": "Every question has passages.", "completeness": "partial"}
    report = render_markdown(data)
    assert "- Stopped: Every question has passages. Result: partial." in report and "Retrieval coverage" not in report


# ================================================================================================= evidence quality B:
# source dependence. Texts and identifiers are SYNTHETIC; the cross-reference case reproduces the pattern of a saved
# run (two works with different authors, where one's reference list holds the other's DOI).
REFS = "\nReferences\n[1] Someone, A. (2019). An unrelated earlier study of payment networks. J. Example, 1, 1-9."


def work(sid, authors, doi="", title="", text="", access=FULL_TEXT, **ids):
    record = src(sid, f"https://doi.org/{doi or sid}", "", "scholarly", authors, {"doi": doi, **ids} if doi else ids,
                 access=access)
    return dict(record, title=title, text=text)


def pair(result):
    return [(p["relation"], p["reason"]) for p in result["pairs"]]


def test_duplicate_versions_and_shared_authors_are_one_work_or_team():
    title = "Graph neural networks for financial fraud detection a review"
    preprint = work("S6", ["Ana Ortiz"], title=title, text="Body." + REFS, arxiv="2401.00001v2")
    journal = work("S2", ["Kim Lee"], doi="10.9/review", title=title.title(), text="Body." + REFS)
    duplicate = independence([preprint, journal])
    assert duplicate["status"] == "single" and pair(duplicate)[0][0] == "same" and "two versions" in pair(duplicate)[0][1]
    team = independence([work("A", ["Smith, Jane"], doi="10.1/a", text=REFS), work("B", ["J. Smith"], doi="10.1/b", text=REFS)])
    assert team["status"] == "single" and "shared author" in pair(team)[0][1]


def test_cross_references_by_doi_title_or_identifier_make_works_dependent():
    s2 = work("S2", ["Dawei Cheng", "Yi Zou"], doi="10.1007/s00000-024-00000-y",
              title="Graph neural networks for financial fraud detection: a review", text="Review body." + REFS)
    by_doi = work("S3", ["Rina Putri"], doi="10.5/jurnal", text="Article body.\nDaftar Pustaka\nCheng, D., Zou, Y. (2025). "
                  "Frontiers of Computer Science. https://doi.org/10.1007/s00000-024-00000-y\n")
    result = independence([s2, by_doi])
    assert result["status"] == "unknown" and result["independent"] == 1 and result["groups"] == 2
    assert pair(result)[0][0] == "dependent" and "S3 depends on S2" in pair(result)[0][1] and "DOI" in pair(result)[0][1]
    by_title = work("S4", ["Rina Putri"], doi="10.5/other", text="Body.\nReferences\nCheng D. Graph neural networks for "
                    "financial fraud\ndetection: a review. 2025.")
    assert pair(independence([s2, by_title]))[0] == ("dependent", "S4 depends on S2: its text names the title of S2")
    preprint = work("P", ["Ana Ortiz"], title="", text="Body." + REFS, arxiv="2411.05815v2")
    by_arxiv = work("Q", ["Kim Lee"], doi="10.6/q", text="Body.\nReferences\nOrtiz A. arXiv:2411.05815 (2024).")
    assert pair(independence([preprint, by_arxiv]))[0][0] == "dependent"
    record = {"label": SUPPORTED, "parse_ok": True, "contradicting_sources": [], "independence": result,
              "passages": [{"source_id": "S2", "stance": "SUPPORTS"}, {"source_id": "S3", "stance": "SUPPORTS"}]}
    assert strength(record, {"S2": {"access": FULL_TEXT}, "S3": {"access": FULL_TEXT}}) == "independence unknown"


def test_works_without_checkable_reference_lists_stay_unknown():
    a = work("A", ["Ana Ortiz"], doi="10.1/a", text="Body." + REFS)
    abstract = work("B", ["Kim Lee"], doi="10.1/b", text="An abstract only.", access=ABSTRACT_ONLY)
    result = independence([a, abstract])
    assert result["status"] == "unknown" and "no full-text reference list" in pair(result)[0][1] and "B" in pair(result)[0][1]
    no_heading = work("C", ["Kim Lee"], doi="10.1/c", text="Full text without any reference heading at all.")
    assert independence([a, no_heading])["status"] == "unknown"
    no_authors = work("D", [], doi="10.1/d", text="Body." + REFS)
    assert "author list is missing" in pair(independence([a, no_authors]))[0][1]


def test_independent_works_meeting_every_criterion_are_established():
    a = work("A", ["Ana Ortiz"], doi="10.1/a", title="Temporal motifs in transaction graphs for fraud", text="Body A." + REFS)
    b = work("B", ["Kim Lee"], doi="10.2/b", title="Isolation forests for card fraud at scale today", text="Body B." + REFS)
    result = independence([a, b])
    assert result["status"] == "established" and result["independent"] == 2
    assert pair(result)[0] == ("distinct", "different authors; both reference lists were checked and neither cites the other")
    record = {"label": SUPPORTED, "parse_ok": True, "contradicting_sources": [], "independence": result,
              "passages": [{"source_id": "A", "stance": "SUPPORTS"}, {"source_id": "B", "stance": "SUPPORTS"}]}
    assert strength(record, {"A": {"access": FULL_TEXT}, "B": {"access": FULL_TEXT}}) == "corroborated"


def test_pages_that_cite_each_other_are_dependent_and_unrelated_pages_unchanged():
    a = dict(src("A", "https://alpha.example/report", "Alpha News"), text="Our own reporting on the grid figures.")
    b = dict(src("B", "https://beta.example/story", "Beta Times"), text="As https://alpha.example/report said, figures rose.")
    linked = independence([a, b])
    assert linked["status"] == "unknown" and pair(linked)[0] == ("dependent", "B depends on A: its text links to A")
    plain = independence([dict(a, text="Grid figures."), dict(b, text="Other figures.")])
    assert plain["status"] == "established"                                   # no citation: the page rule is unchanged


def test_a_dependent_finding_is_reported_and_not_corroborated():
    claim = {"independence": independence([dict(src("A", "https://alpha.example/report", "Alpha News"), text="x"),
                                           dict(src("B", "https://beta.example/s", "Beta Times"),
                                                text="See alpha.example/report.")])}
    from webrag.report import _support_basis
    assert _support_basis(claim).endswith("independence not established; one cites another")


# ================================================================================================= evidence quality D:
# hedge and cause fidelity. Lexical warnings, not proof of meaning; every sentence is SYNTHETIC.
from webrag.deep import fidelity   # noqa: E402


def supports(*texts):
    return [{"key": f"S{i}:p1:x", "source_id": f"S{i}", "page": 1, "text": t, "stance": "SUPPORTS"}
            for i, t in enumerate(texts, 1)]


def kinds(found):
    return [c["kind"] for c in found[0]]


def test_hedge_loss_is_cautioned_and_a_kept_hedge_is_not():
    hedged = supports("Graph learning may improve early identification of fraud rings in payment data.")
    found = fidelity("Graph learning improves early identification of fraud rings.", hedged)
    assert kinds(found) == ["hedge_dropped"] and "'may'" in found[0][0]["detail"] and found[1] == "assessed"
    assert kinds(fidelity("Graph learning may improve early identification of fraud rings.", hedged)) == []
    assert kinds(fidelity("Graph learning could help early identification of fraud rings.", hedged)) == []


def test_mixed_strength_supporting_passages_give_no_hedge_caution():
    mixed = supports("Graph learning may improve early identification of fraud rings.",
                     "Graph learning improved early identification of fraud rings in two banks.")
    found = fidelity("Graph learning improves early identification of fraud rings.", mixed)
    assert kinds(found) == [] and "mixed hedging" in found[1]


def test_association_is_not_causation():
    found = fidelity("Shared devices cause higher fraud rates among new accounts.",
                     supports("Shared devices are associated with higher fraud rates among new accounts."))
    assert "causal_added" in kinds(found) and "association" in next(c for c in found[0] if c["kind"] == "causal_added")["detail"]
    stated = supports("Shared devices lead to higher fraud rates among new accounts, the audit found.")
    assert kinds(fidelity("Shared devices lead to higher fraud rates among new accounts.", stated)) == []


def test_negation_is_not_assessed():
    found = fidelity("Graph models do not improve recall on small fraud datasets.",
                     supports("Graph models may improve recall on small fraud datasets."))
    assert kinds(found) == [] and found[1].startswith("not assessed: negation")
    found = fidelity("Graph models improve recall on small fraud datasets.",
                     supports("Graph models did not clearly improve recall on small fraud datasets."))
    assert kinds(found) == [] and found[1].startswith("not assessed: negation")


def test_a_causal_word_in_an_unrelated_sentence_does_not_support_a_causal_finding():
    passage = supports("Losses fell in 2023 because banks hired more staff. Graph learning improved early "
                       "identification of fraud rings in the pilot.")
    found = fidelity("Graph learning improved early identification of fraud rings, which led to lower losses.", passage)
    assert kinds(found) == ["causal_added"]
    assert fidelity("Graph learning improved fraud detection.", supports("Unrelated text about weather."))[1].startswith(
        "not assessed: no supporting sentence")


def test_cautions_combine_and_only_cover_verified_statements():
    record = {"label": SUPPORTED, "text": "Shared devices cause higher fraud rates among new accounts.",
              "passages": supports("Shared devices might be associated with higher fraud rates among new accounts.")}
    assert sorted(c["kind"] for c in cautions(record)) == ["causal_added", "hedge_dropped"]
    assert cautions(dict(record, label=CONTRADICTED)) == []


def test_summary_sentences_that_add_a_cause_or_drop_a_shared_hedge_are_dropped():
    findings = ["Graph learning may improve early identification of fraud rings.",
                "Graph learning might reduce manual review time."]
    reply = {"answered": True,
             "direct_answer": {"text": "Graph learning may improve early identification of fraud rings.", "findings": ["F1"]},
             "summary": [{"text": "Graph learning improves early identification of fraud rings.", "findings": ["F1"]},
                         {"text": "Graph learning may reduce review time because rings are found early.", "findings": ["F1", "F2"]},
                         {"text": "Graph learning might reduce manual review time.", "findings": ["F2"]},
                         {"text": "Graph learning does not always reduce manual review time.", "findings": ["F2"]}]}
    summary = summarize(DeepClient({"summary": reply}), "Does graph learning help?", findings)
    assert summary["direct_answer"]["text"].startswith("Graph learning may improve")
    assert [p["text"] for p in summary["summary"]] == ["Graph learning might reduce manual review time.",
                                                       "Graph learning does not always reduce manual review time."]
    assert summary["dropped"] == 2


def test_v3_prompt_rules_and_unchanged_v1_v2_prompts():
    from webrag.agents import SUMMARY_PROMPT, SYNTHESIS_PROMPT, SYNTHESIS_PROMPT_V2, SYNTHESIS_PROMPT_V3
    assert "Keep the passage's certainty" in SYNTHESIS_PROMPT_V3 and "an association is not a cause" in SYNTHESIS_PROMPT_V3
    assert "begins mid-word" in SYNTHESIS_PROMPT_V3
    for prompt in (SYNTHESIS_PROMPT, SYNTHESIS_PROMPT_V2):
        assert "certainty" not in prompt and "mid-word" not in prompt
    assert "keep their hedges" in SUMMARY_PROMPT


# ================================================================================================= deep-research-v3:
# query allocation, routing, the query ledger, candidate limits, ranked selection, full text by priority, evidence
# gaps, provider coverage, claim IDs, the number caution, the evidence review and several claims per verification.
from webrag.agents import PlannedQuery   # noqa: E402
from webrag.deep import (DeepResearch, allocate_queries, evidence_gap, number_caution, review,   # noqa: E402
                         split_claims)


def pq(query, sub=0, purpose="answer", category="scholarly"):
    return PlannedQuery(query, category, sub, purpose)


def test_allocation_keeps_the_question_each_subquestion_and_counter_evidence():
    question = "Do graph models detect fraud better?"
    plan = [pq(question), pq("a1", 1), pq("a2", 1), pq("a3", 1), pq("b1", 2), pq("c1", 0, "counter"), pq("c2", 0, "counter")]
    run, omitted = allocate_queries(plan, 4, question)
    assert [q.query for q in run] == [question, "a1", "b1", "c1"]          # not the first four
    assert [q.query for q in omitted] == ["a2", "a3", "c2"]
    assert allocate_queries(plan[:3], 4, question) == (plan[:3], [])        # room for all: order unchanged
    small, _ = allocate_queries(plan, 2, question)
    assert [q.query for q in small] == [question, "a1"]                     # too small to reserve a place
    deep_budget, _ = allocate_queries(plan + [pq(f"x{i}", 3) for i in range(6)], 8, question)
    assert sum(q.purpose == "counter" for q in deep_budget) == 2            # a quarter of eight places


def test_routes_give_every_query_a_search_before_any_query_gets_two():
    d = DeepResearch.__new__(DeepResearch)
    d.names = ["openalex", "crossref", "europepmc", "arxiv"]
    routes, cut = d.routes_for([pq("q1"), pq("q2"), pq("q3")], 5)
    assert [(r.query, r.provider) for r in routes] == [("q1", "openalex"), ("q2", "openalex"), ("q3", "openalex"),
                                                       ("q1", "crossref"), ("q2", "crossref")]
    assert len(cut) == 7 and {r.query for r in routes} == {"q1", "q2", "q3"}


V3_PLAN = dict(PLAN, counter_queries=[{"subquestion": 1, "query": "solar capacity growth overstated"}],
               queries=PLAN["queries"] + [{"subquestion": 1, "category": "web", "query": "solar statistics agency"},
                                          {"subquestion": 2, "category": "web", "query": "battery tender prices"}])


def test_quick_run_ledger_shows_executed_skipped_and_counter_queries(monkeypatch):
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    monkeypatch.delenv("WEBRAG_PROVIDERS", raising=False)
    data, _, _, _ = deep_run(depth="quick", replies={"plan": V3_PLAN, "synthesis": SYNTH, "summary": SUMMARY})
    ledger = data["queries"]
    assert [q["id"] for q in ledger] == [f"Q{i}" for i in range(1, len(ledger) + 1)]
    counter = [q for q in ledger if q["purpose"] == "counter"]
    assert counter and counter[0]["status"] == "executed" and counter[0]["searches"]          # ran, not cut away
    skipped = [q for q in ledger if q["status"] == "skipped"]
    assert skipped and all("queries per round" in q["reason"] for q in skipped)
    assert data["rounds"][0]["omitted_queries"] == [{"query": q["query"], "purpose": q["purpose"], "subquestion": q["subquestion"]}
                                                     for q in skipped]
    executed = [q for q in ledger if q["status"] == "executed"]
    assert all(s["requested"] == 10 for q in executed for s in q["searches"])               # provider candidate limits
    assert data["rounds"][0]["candidate_limits"] == {"openalex": 10, "searxng": 10}
    assert data["rounds"][0]["candidates"] >= data["rounds"][0]["new_sources"]
    states = {p["id"]: p["state"] for p in data["provider_coverage"]}
    assert states["openalex"] == "searched" and states["searxng"] == "searched"
    assert states["wikipedia"] == "excluded" and states["crossref"] == "disabled" and states["brave"] == "not_configured"
    assert states["arxiv"] == "not_selected"                                                  # available, not passed in
    assert [c["id"] for c in data["claims"]] == ["C1", "C2"]
    kinds = [i["kind"] for i in data["review"]]
    assert "query_not_run" in kinds and all(i["refs"] for i in data["review"] if i["kind"] == "query_not_run")
    assert data["deep_version"] == "deep-research-v5"


def test_ledger_distinguishes_failed_no_provider_and_zero_results():
    failing = FakeProvider("openalex", "scholarly", lambda q: [], error="http_503: down")
    empty = FakeProvider("searxng", "web", lambda q: [])
    plan = {"subquestions": [], "queries": [{"subquestion": 0, "category": "scholarly", "query": "solar papers"},
                                            {"subquestion": 0, "category": "web", "query": "solar pages"}]}
    data, _, _, _ = deep_run(depth="quick", providers=[failing, empty], replies={"plan": plan})
    by_query = {q["query"]: q for q in data["queries"]}
    assert by_query["solar papers"]["status"] == "failed" and by_query["solar papers"]["searches"][0]["error"].startswith("http_503")
    assert by_query["solar pages"]["status"] == "executed" and by_query["solar pages"]["searches"][0]["returned"] == 0
    d = DeepResearch.__new__(DeepResearch)
    d.data, d.depth, d.budget, d.used = {}, "quick", BUDGETS["quick"], {"searches": 0}
    d._log_queries(1, "search", [pq("official only", category="official")], [], [], [], [])
    assert d.data["queries"][0]["status"] == "no_provider" and "'official'" in d.data["queries"][0]["reason"]


def test_candidate_limits_apply_to_deep_research_only():
    seen = []

    class Counting(FakeProvider):
        def search(self, query, limit=3, **kwargs):
            seen.append(limit)
            return super().search(query, limit, **kwargs)
    provider = Counting("openalex", "scholarly", lambda q: [])
    WebResearcher([provider], FakeSession()).gather(["q"])                                   # the single-pass pipeline
    WebResearcher([provider], FakeSession(), classify=True, per_provider={"openalex": 10}).gather(["q"])
    assert seen == [3, 10]


def ahit(title, url, rank=1, query="graph fraud detection", abstract="", pdf=None, venue=""):
    return SearchHit("openalex", query, rank, title, url, kind="scholarly", abstract=abstract, pdf_url=pdf,
                     doi=url if "doi.org" in url else None, extra={"venue": venue} if venue else {})


def test_selection_ranks_by_relevance_and_covers_every_query():
    r = WebResearcher([], FakeSession(), classify=True, max_sources=2)
    hits = [ahit("Unrelated protein folding", "https://doi.org/10.1/a", 1, abstract="Proteins fold."),
            ahit("Graph models for fraud detection", "https://doi.org/10.1/b", 2, abstract="Graph fraud detection works."),
            ahit("Counter study", "https://doi.org/10.1/c", 3, query="graph fraud criticism", abstract="Graph fraud criticism.")]
    chosen = [h.url for h, _ in r.select(hits)]
    assert chosen == ["https://doi.org/10.1/b", "https://doi.org/10.1/c"]                  # best of each query
    key = next(k for k in r._selection if k.endswith("10.1/b"))
    assert r.candidates == 3 and "best candidate" in r._selection[key]["reason"]
    plain = WebResearcher([], FakeSession(), max_sources=2)                                 # single-pass: rank order kept
    assert [h.url for h, _ in plain.select(hits)] == ["https://doi.org/10.1/a", "https://doi.org/10.1/b"]


def test_selection_prefers_venue_diversity_among_equals():
    r = WebResearcher([], FakeSession(), classify=True, max_sources=2)
    hits = [ahit("Graph fraud detection one", "https://doi.org/10.1/a", 1, abstract="Graph fraud detection.", venue="J A"),
            ahit("Graph fraud detection two", "https://doi.org/10.1/b", 1, abstract="Graph fraud detection.", venue="J A"),
            ahit("Graph fraud detection three", "https://doi.org/10.1/c", 1, abstract="Graph fraud detection.", venue="J B")]
    assert [h.url for h, _ in r.select(hits)] == ["https://doi.org/10.1/a", "https://doi.org/10.1/c"]


def test_full_text_goes_to_the_highest_ranked_sources(tmp_path):
    write_simple_pdf(tmp_path / "oa.pdf", [["Graph fraud detection full text. " * 80]])
    pdf = (tmp_path / "oa.pdf").read_bytes()
    session = FakeSession([(lambda u, p: u.endswith(".pdf"), page(pdf, "application/pdf", "https://repo.example/x.pdf"))])
    hits = [ahit("Weak match", "https://doi.org/10.1/a", 1, abstract="Something else.", pdf="https://repo.example/a.pdf"),
            ahit("Graph fraud detection", "https://doi.org/10.1/b", 2, abstract="Graph fraud detection.", pdf="https://repo.example/b.pdf"),
            ahit("Graph fraud detection again", "https://doi.org/10.1/c", 3, abstract="Graph fraud detection too.", pdf="https://repo.example/c.pdf")]
    r = WebResearcher([], session, classify=True, max_full_text=1)
    r.search = lambda queries, routes=None: (hits, [])
    corpus = r.gather(["graph fraud detection"])
    access = {s.url: s.access for s in corpus.sources}
    assert access == {"https://doi.org/10.1/b": FULL_TEXT, "https://doi.org/10.1/c": ABSTRACT_ONLY,
                      "https://doi.org/10.1/a": ABSTRACT_ONLY}
    assert r.full_text_attempts == 1
    skipped = [s.diagnostics.get("full_text_skipped", "") for s in corpus.sources if s.access == ABSTRACT_ONLY]
    assert all("all 1 attempt(s) went to higher-priority sources (1 read in full, 0 failed)" in x for x in skipped)
    assert r.full_text_outcome == {"attempts": 1, "succeeded": 1, "failed": 0}


def test_evidence_gaps_reach_the_follow_up_query_writer():
    assert "abstracts" in evidence_gap({"state": "weak", "full_text": False})
    assert "no relevant readable passage" in evidence_gap({"state": "none"})
    assert "one publisher" in evidence_gap({"state": "weak", "full_text": True, "independence": "single"})
    data, client, _, _ = deep_run()
    follow = next(m for k, m in client.seen if k == "follow-up")[1]["content"]
    assert "(evidence gap:" in follow and data["rounds"][0]["gaps"]
    queries, fell_back = write_queries(DeepClient({}, fail={"follow-up"}), "follow-up",
                                       ["What do storage batteries cost? (evidence gap: only abstracts were found)"],
                                       ["web"], [], 2)
    assert fell_back and "gap" not in queries[0].query and "abstracts" not in queries[0].query


def test_number_caution_only_for_figures_absent_from_support():
    support = [{"key": "S1:p1:a", "stance": "SUPPORTS", "text": "Capacity grew 20 percent to 1,200 gigawatts in 2024."}]
    assert number_caution("Capacity grew 20 percent to 1200 gigawatts in 2024.", support) is None
    caution = number_caution("Capacity grew 25 percent in 2024.", support)
    assert caution["kind"] == "number_unsupported" and "25" in caution["detail"]
    assert number_caution("Capacity grew 25 percent.", [dict(support[0], stance="NEUTRAL")]) is None
    record = {"label": SUPPORTED, "text": "Capacity grew 25 percent in 2024.", "passages": support}
    assert "number_unsupported" in [c["kind"] for c in cautions(record)]


def test_review_names_claims_refs_and_actions():
    dependent = {"status": "unknown", "pairs": [{"a": "S2", "b": "S3", "relation": "dependent",
                                                  "reason": "S3 depends on S2: its text contains the DOI of S2 (10.1/x)"}]}
    data = {"claims": [
        {"id": "C1", "label": SUPPORTED, "strength": "independence unknown", "independence": dependent, "citations": {"ok": True},
         "cautions": [{"kind": "hedge_dropped", "detail": "Hedged source."}], "passages": [{"key": "S2:p1:a", "stance": "SUPPORTS"}]},
        {"id": "C2", "label": SUPPORTED, "strength": "abstract only", "citations": {"ok": True}, "cautions": [],
         "passages": [{"key": "S5:p1:a", "stance": "SUPPORTS"}]},
        {"id": "C3", "label": INSUFFICIENT, "sub_state": "CONFLICTING", "passages": [{"key": "S1:p1:a", "stance": "CONTRADICTS"}]},
        {"id": "C4", "label": INSUFFICIENT, "sub_state": "SERVICE_ERROR", "parse_ok": False, "passages": []}],
        "queries": [{"id": "Q5", "query": "fraud graph criticism", "purpose": "counter", "status": "skipped", "reason": "budget"}],
        "counter_evidence": {"skipped": "counter-evidence search is not part of the quick depth"},
        "ledger": [{"index": 3, "status": "unanswered", "reason": "No verified statement."}]}
    items = review(data)
    by = {(i["claim"], i["kind"]): i for i in items}
    assert by[("C1", "caution:hedge_dropped")]["action"] == "excluded" and by[("C1", "caution:hedge_dropped")]["refs"] == ["S2:p1:a"]
    assert "contains the DOI of S2" in by[("C1", "corroboration")]["detail"]
    assert by[("C2", "abstract_only")]["action"] == "needs_evidence"
    assert by[("C3", "contested")]["action"] == "unresolved"
    assert by[("C4", "verification_failed")]["action"] == "not_reported"
    assert by[(None, "counter_query_not_run")]["refs"] == ["Q5"] and by[(None, "counter_query_not_run")]["action"] == "unresolved"
    assert (None, "counter_search_skipped") in by and by[(None, "open_question")]["detail"].startswith("Question 3 is unanswered")
    assert review({}) == []


def test_several_claims_are_verified_one_by_one():
    assert split_claims("- First claim.\n2. Second claim.\n\n* First claim.") == ["First claim.", "Second claim."]
    assert split_claims("One claim only.") == ["One claim only."]
    data, _, _, _ = deep_run(text=f"{GROWTH}.\n{STORAGE}.", mode="verify", depth="quick")
    assert data["outcome"] == "CLAIMS_ASSESSED" and [c["id"] for c in data["claims"]] == ["C1", "C2"]
    assert [c["text"] for c in data["claims"]] == [f"{GROWTH}.", f"{STORAGE}."]
    assert data["claims"][0]["label"] == SUPPORTED
    assert data["summary"]["claims"] == 2 and sum(data["summary"]["verdicts"].values()) == 2
    assert data["claims_mode"] == {"claims": 2, "limit": 8, "dropped": 0}
    if any(c["label"] not in {SUPPORTED, CONTRADICTED} for c in data["claims"]):
        assert data["ledger"][0]["status"] != "answered" and data["stopping"]["completeness"] != "complete"
    many, _, _, _ = deep_run(text=f"{GROWTH}.\nRooftop panels are banned in every city.", mode="verify", depth="quick")
    assert [c["label"] for c in many["claims"]][1] == INSUFFICIENT
    assert many["ledger"][0]["status"] == "partially answered" and "1 of 2 claims" in many["ledger"][0]["reason"]
    assert many["stopping"]["completeness"] != "complete"
    single, _, _, _ = deep_run(text=f"{GROWTH}.", mode="verify", depth="quick")
    assert single["outcome"] == single["claims"][0]["label"] and "claims_mode" not in single


# ================================================================================================= deep-research-v4:
# Gate 1 repairs. The sentences below follow the patterns of the 2026-10-21 live run (statement C1-C6), shortened;
# the sources are SYNTHETIC records and no model or provider is called.
from webrag.agents import _statement_ok, attributions, qualifier_words, scope_words   # noqa: E402
from webrag.deep import (counted_sources, infer_subquestions, secondary_reports, support_check)   # noqa: E402
from webrag.researcher import RELEVANCE_FLOOR, query_relevance   # noqa: E402

C1 = ("Graph neural networks (GNNs) may improve early fraud identification, reduce financial losses, and support more "
      "effective risk management in digital financial ecosystems.")
C1_SUPPORT = (". From an operational perspective, incorporating graph representation learning into fraud detection systems "
              "may improve early fraud identification, reduce financial losses, and support more effective risk management "
              "within increasingly interconnected digital financial ecosystems.")
C1_TOPICAL = (". Graph neural networks revolutionized fraud detection by enabling the direct modeling of relational "
              "structures in financial data. Kipf and Welling's work on graph convolutional networks established the "
              "theoretical foundation.")
C3 = "Graph neural networks revolutionized fraud detection by enabling direct modeling of relational structures in financial data."
C3_DEFINITION = (". Graph Neural Networks (GNNs) are a family of deep learning models specifically designed to learn from "
                 "graph-structured data by integrating node attributes with the relational information embedded in graph "
                 "topology.")


def sp(source, text, key=None, stance="SUPPORTS"):
    return {"key": key or f"{source}:p1:x", "source_id": source, "page": 1, "text": text, "stance": stance}


def test_corroboration_needs_a_passage_that_covers_the_finding_c1_pattern():
    rows = {r["source"]: r for r in support_check(C1, [sp("S2", C1_SUPPORT), sp("S6", C1_TOPICAL)])}
    assert rows["S2"]["counted"] and rows["S2"]["coverage"] >= 0.6
    assert not rows["S6"]["counted"] and rows["S6"]["coverage"] < 0.6          # same topic, not the claim
    two = [src("S2", "https://doi.org/10.1/a", "", "scholarly", ["Rina Putri"]), src("S6", "https://doi.org/10.2/b", "", "scholarly", ["Li Wei"])]
    two = [dict(s, text="Body.\nReferences\n[1] Unrelated.") for s in two]
    everyone = independence(two)                                                 # what v3 counted
    assert everyone["status"] == "established"
    counted = independence([s for s in two if s["id"] in counted_sources(C1, [sp("S2", C1_SUPPORT), sp("S6", C1_TOPICAL)])])
    record = {"label": SUPPORTED, "parse_ok": True, "contradicting_sources": [], "independence": counted,
              "support_check": list(rows.values()), "passages": [sp("S2", C1_SUPPORT), sp("S6", C1_TOPICAL)]}
    assert strength(record, {"S2": {"access": FULL_TEXT}, "S6": {"access": FULL_TEXT}}) == "single source"


def test_a_definition_does_not_corroborate_a_specific_claim_c3_pattern():
    rows = {r["source"]: r for r in support_check(C3, [sp("S6", C1_TOPICAL), sp("S2", C3_DEFINITION)])}
    assert rows["S6"]["counted"] and rows["S6"]["coverage"] == 1.0
    assert not rows["S2"]["counted"]
    assert counted_sources(C3, [sp("S6", C1_TOPICAL), sp("S2", C3_DEFINITION)]) == ["S6"]


def test_two_passages_that_each_state_the_claim_still_corroborate():
    finding = "Solar capacity grew 20 percent in 2024."
    rows = support_check(finding, [sp("A", "Solar capacity grew 20 percent in 2024 according to the operator."),
                                   sp("B", "The agency reports that solar capacity grew 20 percent in 2024.")])
    assert [r["counted"] for r in rows] == [True, True]
    assert support_check(finding, [sp("A", "Solar capacity grew 20 percent in 2024.", stance="NEUTRAL")]) == []


def test_review_reports_sources_not_counted_without_claiming_entailment():
    data = {"claims": [{"id": "C1", "label": SUPPORTED, "strength": "single source", "citations": {"ok": True}, "cautions": [],
                        "passages": [sp("S2", C1_SUPPORT), sp("S6", C1_TOPICAL, key="S6:p2:a")],
                        "support_check": support_check(C1, [sp("S2", C1_SUPPORT), sp("S6", C1_TOPICAL, key="S6:p2:a")])}]}
    item = next(i for i in review(data) if i["kind"] == "support_not_counted")
    assert item["claim"] == "C1" and item["refs"] == ["S6:p2:a"] and item["action"] == "needs_evidence"
    assert "S6 was judged to support" in item["detail"] and "neither proves nor disproves support" in item["detail"]


C2_SOURCE = ("Xu et al. (2024) demonstrated that many fraud graphs exhibit heterophilic structures, where fraudulent nodes "
             "are predominantly connected to legitimate nodes.")
C4_SOURCE = ("Experiments on two classic financial fraud datasets demonstrate that our proposed method achieves superior "
             "discriminative capability for fraudulent nodes compared to traditional graph algorithms.")
C6_SOURCE = ("Works on graph neural networks that can capture relationships among entities in a graph constructed from the "
             "credit card transaction data have also been examined.")


def caution_kinds(finding, *texts):
    return [c["kind"] for c in fidelity(finding, [sp(f"S{i}", t) for i, t in enumerate(texts, 1)])[0]]


def test_lost_quantifier_scope_modality_and_attribution_are_cautioned():
    # C2: "many" dropped (the live run's synthesizer did this); attribution kept
    assert caution_kinds("Xu et al. (2024) demonstrated that fraud graphs exhibit heterophilic structures where fraudulent "
                         "nodes are predominantly connected to legitimate nodes.", C2_SOURCE) == ["qualifier_dropped"]
    assert caution_kinds("Xu et al. (2024) demonstrated that many fraud graphs exhibit heterophilic structures where "
                         "fraudulent nodes are predominantly connected to legitimate nodes.", C2_SOURCE) == []
    assert caution_kinds("Many fraud graphs exhibit heterophilic structures where fraudulent nodes are predominantly "
                         "connected to legitimate nodes.", C2_SOURCE) == ["attribution_dropped"]
    # C4: the dataset scope
    assert caution_kinds("The proposed method achieves superior discriminative capability for fraudulent nodes compared "
                         "to traditional graph algorithms.", C4_SOURCE) == ["qualifier_dropped"]
    assert caution_kinds("On two classic financial fraud datasets, the proposed method achieves superior discriminative "
                         "capability for fraudulent nodes compared to traditional graph algorithms.", C4_SOURCE) == []
    # C6: "can" is modality, not a hedge, and still may not be dropped
    assert caution_kinds("Graph neural networks capture relationships among entities in a graph constructed from credit "
                         "card transaction data.", C6_SOURCE) == ["qualifier_dropped"]
    assert caution_kinds("Graph neural networks can capture relationships among entities in a graph constructed from "
                         "credit card transaction data.", C6_SOURCE) == []


def test_qualifier_rules_false_positives_and_false_negatives():
    unrelated = "Many papers were reviewed. Graph models capture relationships among accounts in transaction networks."
    assert caution_kinds("Graph models capture relationships among accounts in transaction networks.", unrelated) == []
    assert caution_kinds("Fraud graphs exhibit heterophilic structures with fraudulent nodes linked to legitimate nodes.",
                         "Many fraud graphs exhibit heterophilic structures with fraudulent nodes linked to legitimate nodes.",
                         "Fraud graphs exhibit heterophilic structures with fraudulent nodes linked to legitimate nodes.") == []
    assert qualifier_words("Rules cannot capture coordinated fraud.") == set()           # "cannot" is not "can"
    assert caution_kinds("Graph models cannot capture fraud rings across banks.",
                         "Graph models can capture fraud rings across banks.") == []       # negation: not judged
    # known limits of a lexical rule, kept visible: a faithful paraphrase is flagged ...
    assert caution_kinds("Numerous fraud graphs exhibit heterophilic structures with fraudulent nodes linked to legitimate nodes.",
                         "Many fraud graphs exhibit heterophilic structures with fraudulent nodes linked to legitimate nodes.") == ["qualifier_dropped"]
    # ... and a lost qualifier written without a listed word is missed
    assert caution_kinds("Fraud graphs exhibit heterophilic structures with fraudulent nodes linked to legitimate nodes.",
                         "A large share of fraud graphs exhibit heterophilic structures with fraudulent nodes linked to "
                         "legitimate nodes.") == []
    assert scope_words("Results on two classic financial fraud datasets.") == {"dataset"}
    assert attributions("Kipf and Welling (2017) showed this; Xu et al. reported that.") == {"kipf", "xu"}


def test_summary_sentences_must_keep_each_cited_findings_qualifiers_c2_c4_c6():
    c4 = ("Experiments on two classic financial fraud datasets demonstrate that FinGuard-GNN achieves superior "
          "discriminative capability for fraudulent nodes compared to traditional graph algorithms and machine learning methods.")
    c6 = "Graph neural networks can capture relationships among entities in a graph constructed from credit card transaction data."
    c2 = ("Xu et al. (2024) demonstrated that many fraud graphs exhibit heterophilic structures, reducing the "
          "effectiveness of conventional homophily-based GNN models.")
    live_answer = ("Graph neural networks (GNNs) may improve early fraud identification, reduce financial losses, and support "
                   "more effective risk management in digital financial ecosystems. They also capture relationships among "
                   "entities in a graph constructed from credit card transaction data.")
    assert not _statement_ok(live_answer, [C1, c6])                                       # "can" lost
    assert _statement_ok(live_answer.replace("They also capture", "They can also capture"), [C1, c6])
    assert not _statement_ok("Fraud graphs exhibit heterophilic structures, reducing effectiveness of conventional "
                             "homophily-based GNN models", [c2])                          # "many" and "Xu et al." lost
    assert _statement_ok("Xu et al. reported that many fraud graphs exhibit heterophilic structures.", [c2])
    assert not _statement_ok("FinGuard-GNN achieves superior discriminative capability for fraudulent nodes compared to "
                             "traditional methods", [c4])                                 # the datasets scope lost
    assert _statement_ok("On two classic financial fraud datasets, FinGuard-GNN outperformed traditional methods.", [c4])
    assert _statement_ok("Graph learning does not always reduce review time.", ["Graph learning might reduce review time."])


def test_live_run_summary_is_filtered_down_to_faithful_points():
    c2 = ("Xu et al. (2024) demonstrated that fraud graphs exhibit heterophilic structures where fraudulent nodes are "
          "predominantly connected to legitimate nodes rather than other fraudulent entities, reducing the effectiveness of "
          "conventional homophily-based GNN models.")
    c5 = ("Traditional machine learning approaches analyze transactions in isolation and fail to capture relational "
          "patterns indicative of coordinated fraud rings.")
    c4 = ("Experiments on two classic financial fraud datasets demonstrate that FinGuard-GNN achieves superior "
          "discriminative capability for fraudulent nodes compared to traditional graph algorithms and machine learning methods.")
    c6 = "Graph neural networks can capture relationships among entities in a graph constructed from credit card transaction data."
    reply = {"answered": True,
             "direct_answer": {"text": f"{C1} They also capture relationships among entities in a graph constructed from "
                                       f"credit card transaction data.", "findings": ["F1", "F5"]},
             "summary": [{"text": "GNNs may improve early fraud identification and reduce financial losses", "findings": ["F1"]},
                         {"text": "GNNs capture relationships among entities in credit card transaction graphs", "findings": ["F5"]},
                         {"text": "Fraud graphs exhibit heterophilic structures, reducing effectiveness of conventional "
                                  "homophily-based GNN models", "findings": ["F2"]},
                         {"text": "Traditional machine learning approaches fail to capture relational patterns indicative of "
                                  "coordinated fraud rings", "findings": ["F3"]},
                         {"text": "FinGuard-GNN achieves superior discriminative capability for fraudulent nodes compared to "
                                  "traditional methods", "findings": ["F4"]}]}
    out = summarize(DeepClient({"summary": reply}), "Do GNNs help fraud detection?", [C1, c2, c5, c4, c6])
    assert out["direct_answer"] is None and out["dropped"] == 4
    assert [p["text"] for p in out["summary"]] == ["GNNs may improve early fraud identification and reduce financial losses",
                                                   "Traditional machine learning approaches fail to capture relational patterns "
                                                   "indicative of coordinated fraud rings"]


def test_secondary_reports_name_the_attributed_work():
    claim = {"text": "Xu et al. (2024) demonstrated that many fraud graphs exhibit heterophilic structures.",
             "passages": [sp("S2", C2_SOURCE, key="S2:p9:a")]}
    reports = secondary_reports(claim, {"S2": {"authors": ["Rina Putri"]}})
    assert reports == [{"source": "S2", "passage": "S2:p9:a", "attributed_to": ["xu"]}]
    assert secondary_reports(claim, {"S2": {"authors": ["Xu, Lei", "Ana Ortiz"]}}) == []      # the authors' own result
    item = next(i for i in review({"claims": [dict(claim, id="C2", label=SUPPORTED, citations={"ok": True}, cautions=[],
                                                   secondary_reports=reports)]}) if i["kind"] == "secondary_report")
    assert "S2 reports this as the result of Xu" in item["detail"] and item["refs"] == ["S2:p9:a"]


LIVE_SUBS = [{"text": "What evidence supports the use of graph neural networks for financial transaction fraud detection, and "
                      "what are their main limitations compared with conventional machine-learning methods?"},
             {"text": "What specific evidence demonstrates the effectiveness of graph neural networks in detecting financial "
                      "transaction fraud?"},
             {"text": "What are the primary limitations of graph neural networks when applied to financial transaction fraud detection?"},
             {"text": "How do graph neural networks compare to conventional machine learning methods in terms of performance "
                      "metrics for financial fraud detection?"}]
LIVE_QUESTION = LIVE_SUBS[0]["text"]
LIVE_PLAN = [pq(LIVE_QUESTION[:160]), pq("graph neural networks financial transaction fraud detection effectiveness metrics"),
             pq("graph neural networks limitations financial fraud detection computational complexity"),
             pq("graph neural networks vs traditional machine learning financial fraud detection performance metrics"),
             pq("graph neural networks financial fraud detection overfitting risks in high-dimensional data", purpose="counter"),
             pq("conventional machine learning methods outperform graph neural networks in real-time financial fraud detection "
                "scenarios", purpose="counter")]


def test_subquestion_links_are_inferred_when_every_tag_is_the_same():
    linked, inferred = infer_subquestions(LIVE_PLAN, LIVE_SUBS, LIVE_QUESTION)
    assert [q.subquestion for q in linked] == [0, 1, 2, 3, 0, 3]
    assert inferred == {LIVE_PLAN[1].query, LIVE_PLAN[2].query, LIVE_PLAN[3].query, LIVE_PLAN[5].query}
    varied = [replace(LIVE_PLAN[1], subquestion=2), replace(LIVE_PLAN[2], subquestion=1)]
    assert infer_subquestions(varied, LIVE_SUBS)[0] == varied                     # informative planner tags are kept
    blank = [replace(q, category="") for q in LIVE_PLAN]                           # categories missing or identical
    assert [q.subquestion for q in infer_subquestions(blank, LIVE_SUBS, LIVE_QUESTION)[0]] == [0, 1, 2, 3, 0, 3]


def test_allocation_gives_the_comparison_subquestion_a_place_and_reports_the_rest():
    linked, _ = infer_subquestions(LIVE_PLAN, LIVE_SUBS, LIVE_QUESTION)
    run, omitted = allocate_queries(linked, 4, LIVE_QUESTION)
    assert [q.query[:40] for q in run] == [LIVE_QUESTION[:40], LIVE_PLAN[3].query[:40], LIVE_PLAN[2].query[:40],
                                           LIVE_PLAN[4].query[:40]]
    assert {q.subquestion for q in run if q.purpose == "answer"} == {0, 2, 3}       # the comparison is searched
    assert [q.subquestion for q in omitted] == [1, 3]                             # and the gap is not silent
    data = {"queries": [{"id": "Q1", "subquestion": 0, "status": "executed"}, {"id": "Q2", "subquestion": 3, "status": "executed"},
                        {"id": "Q3", "subquestion": 2, "status": "executed"}, {"id": "Q5", "subquestion": 1, "status": "skipped",
                                                                               "purpose": "answer", "query": "q", "reason": "budget"}],
            "ledger": [{"index": 0, "status": "answered"}, {"index": 1, "status": "answered"},
                       {"index": 2, "status": "answered"}, {"index": 3, "status": "answered"}]}
    gap = next(i for i in review(data) if i["kind"] == "subquestion_not_searched")
    assert gap["detail"].startswith("Question 1 had no executed query of its own") and gap["refs"] == ["Q5"]


def test_a_quick_run_with_uninformative_tags_records_inferred_links():
    plan = dict(PLAN, subquestions=PLAN["subquestions"] + [{"text": "Which agency publishes solar statistics?", "terms": []}],
                queries=[{"subquestion": 0, "category": "scholarly", "query": "solar capacity growth survey"},
                         {"subquestion": 0, "category": "web", "query": "storage batteries cost per kilowatt hour"},
                         {"subquestion": 0, "category": "web", "query": "solar statistics agency publishes"},
                         {"subquestion": 0, "category": "web", "query": "solar capacity grew 2024"}])
    data, _, _, _ = deep_run(depth="quick", replies={"plan": plan, "synthesis": SYNTH, "summary": SUMMARY})
    ledger = {q["query"]: q for q in data["queries"]}
    assert ledger["storage batteries cost per kilowatt hour"]["subquestion"] == 2
    assert ledger["storage batteries cost per kilowatt hour"]["tagged_by"] == "inferred"
    assert all("allocated" in q for q in data["queries"])
    assert len({q["subquestion"] for q in data["queries"] if q["status"] == "executed"}) >= 3


def test_relevance_floor_rejects_an_off_topic_result_with_an_abstract():
    fraud = "graph neural networks financial transaction fraud detection"
    off = SearchHit("arxiv", fraud, 1, "Contrastive Graph Neural Network Explanation", "https://arxiv.org/abs/2010.13663",
                    kind="preprint", abstract="Graph neural networks achieve strong results on graph tasks, but explaining "
                    "their predictions is hard. We propose a contrastive method to explain graph neural network decisions "
                    "on molecule and citation benchmarks, and study how explanations generalise across architectures.")
    on = SearchHit("openalex", fraud, 2, "Graph Neural Networks for Fraud Detection", "https://doi.org/10.1/f", kind="scholarly",
                   doi="https://doi.org/10.1/f", abstract="We model financial transaction networks with graph neural "
                   "networks to improve fraud detection across banks, comparing them with gradient boosting on two "
                   "transaction datasets and reporting precision and recall for each fraud type.")
    assert query_relevance(off, fraud) < RELEVANCE_FLOOR <= query_relevance(on, fraud)
    assert query_relevance(replace(off, abstract="Too short."), fraud) is None           # not judged without an abstract
    r = WebResearcher([], FakeSession(), classify=True, max_sources=5)
    assert [h.url for h, _ in r.select([off, on])] == ["https://doi.org/10.1/f"]
    assert r.below_floor == 1 and r.query_picks == {fraud: next(iter(k for k in r._selection if k.endswith("10.1/f")))}
    alone = WebResearcher([], FakeSession(), classify=True, max_sources=5)
    assert alone.select([off]) == [] and alone.query_picks == {fraud: None}       # no relevant result: recorded as such
    cli = WebResearcher([], FakeSession(), max_sources=5)                         # the single-pass pipeline: unchanged
    assert [h.url for h, _ in cli.select([off, on])] == ["https://arxiv.org/abs/2010.13663", "https://doi.org/10.1/f"]


def test_full_text_attempts_successes_and_failures_are_reported_separately(tmp_path):
    write_simple_pdf(tmp_path / "oa.pdf", [["Graph fraud detection full text. " * 80]])
    pdf = (tmp_path / "oa.pdf").read_bytes()
    session = FakeSession([(lambda u, p: u.endswith("/b.pdf"), page(pdf, "application/pdf", "https://repo.example/b.pdf")),
                           (lambda u, p: u.endswith("/c.pdf"), FetchError("http_403", "https://repo.example/c.pdf"))])
    hits = [ahit("Graph fraud detection", "https://doi.org/10.1/b", 1, abstract="Graph fraud detection.", pdf="https://repo.example/b.pdf"),
            ahit("Graph fraud detection again", "https://doi.org/10.1/c", 2, abstract="Graph fraud detection too.", pdf="https://repo.example/c.pdf"),
            ahit("Graph fraud detection third", "https://doi.org/10.1/d", 3, abstract="Graph fraud detection three.", pdf="https://repo.example/d.pdf")]
    r = WebResearcher([], session, classify=True, max_full_text=2)
    r.search = lambda queries, routes=None: (hits, [])
    corpus = r.gather(["graph fraud detection"])
    by_url = {s.url: s for s in corpus.sources}
    assert r.full_text_outcome == {"attempts": 2, "succeeded": 1, "failed": 1}
    assert by_url["https://doi.org/10.1/c"].diagnostics["full_text_error"] == "http_403"
    assert by_url["https://doi.org/10.1/d"].diagnostics["full_text_skipped"] == (
        "full-text budget: all 2 attempt(s) went to higher-priority sources (1 read in full, 1 failed)")


def test_passages_that_start_with_the_previous_full_stop_are_classified_after_it():
    page = ("Graph learning is studied widely. From an operational perspective, it may improve early identification. "
            "Several models were compared on public data")
    assert passage_boundary([page], 1, ". From an operational perspective, it may improve early identification.")["boundary"] == "complete"
    lower = "The first study closed. it was then repeated with new data."
    assert passage_boundary([lower], 1, ". it was then repeated with new data.")["boundary"] == "uncertain"
    cut = "Studies show that deep learning models outperform rule-based systems."
    assert passage_boundary([cut], 1, "els outperform rule-based systems.")["boundary"] == "mid_word"   # unchanged


def test_report_names_sources_not_counted_as_corroboration():
    from webrag.report import _support_basis
    claim = {"independence": {"status": "single", "groups": 1, "sources": 1},
             "support_check": support_check(C1, [sp("S2", C1_SUPPORT), sp("S6", C1_TOPICAL)])}
    assert _support_basis(claim) == "one publisher; S6 not counted (passage does not cover the statement)"
    assert _support_basis({"independence": {"status": "none"}, "support_check": []}) == "no supporting passage covers the statement"


# ================================================================================================= Gate 1 v4 repairs:
# support_unmatched, scope with punctuation, sentence-safe answer length. Sentences follow the 2026-10-21 v4 live run
# (statements C1 and C3), shortened; SYNTHETIC records, no model or provider is called.
from webrag.agents import MAX_STATEMENT_CHARS, fit_sentences   # noqa: E402
from webrag.deep import support_caution   # noqa: E402

V4_C1 = ("Graph Neural Networks (GNNs) demonstrate effectiveness in detecting financial transaction fraud by capturing "
         "relational structures among entities and outperforming conventional methods on the Elliptic Bitcoin Transaction "
         "Dataset using Graph Attention Networks (GAT) compared to Graph Convolutional Networks (GCN), GraphSAGE, and "
         "Graph Isomorphism Networks (GIN).")
V4_C3 = ("Graph Neural Networks (GNNs) outperform conventional machine learning methods in terms of accuracy, precision, "
         "recall, F1-score, ROC-AUC, and PR-AUC metrics for financial fraud detection, as demonstrated by their ability to "
         "capture interconnected fraud patterns and assign adaptive importance to relevant neighboring nodes.")
V4_S2 = (". Model evaluation was performed using Accuracy, Precision, Recall, F1-score, ROC-AUC, and PR-AUC metrics. The "
         "results show that attention-based graph learning provides superior performance in detecting fraudulent "
         "transactions by assigning adaptive importance to relevant neighboring nodes during information propagation.")
V4_S4 = (". The proposed model was evaluated on a large-scale, class-imbalanced transaction dataset comprising over 2.1 "
         "million transactions and benchmarked against Logistic Regression, Random Forest and XGBoost baselines.")


def test_support_unmatched_withholds_the_v4_c1_and_c3_patterns():
    c1 = {"label": SUPPORTED, "text": V4_C1, "passages": [sp("S2", V4_S2, key="S2:p1:a")]}
    c3 = {"label": SUPPORTED, "text": V4_C3, "passages": [sp("S2", V4_S2, key="S2:p1:a"), sp("S4", V4_S4, key="S4:p1:b")]}
    for record in (c1, c3):
        caution = support_caution(record)
        assert caution["kind"] == "support_unmatched" and "does not visibly cover the whole statement" in caution["detail"]
        assert "support_unmatched" in [c["kind"] for c in cautions(record)]
    assert "(best: S2," in support_caution(c3)["detail"]
    covered = {"label": SUPPORTED, "text": "Attention-based graph learning provides superior performance in detecting "
               "fraudulent transactions.", "passages": [sp("S2", V4_S2)]}
    assert support_caution(covered) is None                                        # a covering passage: no caution
    assert support_caution({"label": SUPPORTED, "text": V4_C1, "passages": []})["detail"].endswith(
        "(no supporting passage was recorded), so the retrieved text does not visibly cover the whole statement.")
    assert cautions(dict(c1, label=INSUFFICIENT)) == []                            # never applied to other verdicts
    assert cautions(dict(c1, label=CONTRADICTED)) == []


def test_unmatched_finding_stays_listed_and_is_withheld_everywhere_else():
    broad = "Graph neural networks outperform conventional machine learning in accuracy, precision and recall across every bank."
    whole = "Graph neural networks represent transactions as connected accounts across banks."

    def synthesis(messages):
        label = _label_of(messages, "connected accounts")
        return {"findings": [{"text": broad, "evidence": [label], "subquestion": 0},
                             {"text": whole, "evidence": [label], "subquestion": 0}],
                "disagreements": [], "gaps": [], "insufficient": False}

    data, client = _fraud_run({"plan": FRAUD_PLAN, "synthesis": synthesis, "summary": _cite_all(whole)})
    claims = {c["text"]: c for c in data["claims"]}
    assert claims[broad]["label"] == SUPPORTED                                     # the verdict is not changed
    assert "support_unmatched" in [w["kind"] for w in claims[broad]["cautions"]]
    assert claims[whole]["cautions"] == []
    summary_prompt = next(m for k, m in client.seen if k == "summary")[1]["content"].split("Verified findings:")[1]
    assert broad not in summary_prompt and whole in summary_prompt                  # not offered to the summary
    assert data["summary"]["direct_answer"]["claims"] == [data["claims"].index(claims[whole])]
    item = next(i for i in data["review"] if i["kind"] == "caution:support_unmatched")
    assert item["claim"] == claims[broad]["id"] and item["action"] == "excluded" and item["refs"]
    report = render_markdown(data)
    assert broad in report.split("## Verified findings")[1] and "Caution (support unmatched)" in report


def test_scope_phrases_with_punctuation_are_recognised():
    assert scope_words("evaluated on a large-scale, class-imbalanced transaction dataset") == {"dataset"}
    assert scope_words("tested in two regional; anonymised banks") == {"bank"}
    assert scope_words("The method was evaluated carefully.") == set()
    sentence = "The proposed model outperformed baselines when evaluated on a large-scale, class-imbalanced transaction dataset."
    assert caution_kinds("The proposed model outperformed baselines.", sentence) == ["qualifier_dropped"]
    assert caution_kinds("The proposed model outperformed baselines on one transaction dataset.", sentence) == []


def test_known_limit_a_changed_comparison_target_is_not_detected():
    # One graph model beating other graph models, written as graph models beating conventional methods: same words,
    # different claim. Word matching cannot see this; the limitation is documented (EVIDENCE_POLICY cautions _limits).
    sentence = ("Attention-based graph learning provided superior performance in detecting fraudulent transactions "
                "compared with other graph learning models.")
    finding = "Attention-based graph learning provided superior performance in detecting fraudulent transactions compared with conventional models."
    assert caution_kinds(finding, sentence) == []
    assert support_caution({"label": SUPPORTED, "text": finding, "passages": [sp("S2", sentence)]}) is None
    assert "comparison" in deep_module.EVIDENCE_POLICY["cautions"]["_limits"]


def test_answer_text_is_never_cut_inside_a_sentence():
    short = "Graph learning may improve early fraud identification."
    assert fit_sentences(short) == (short, False)
    sentences = [f"Sentence {i} reports that graph models may help detect fraud rings in payment networks." for i in range(12)]
    text, shortened = fit_sentences(" ".join(sentences))
    assert shortened and len(text) <= MAX_STATEMENT_CHARS and text.endswith(" […]")
    assert text[:-4].endswith("networks.") and text[:-4] == " ".join(sentences[:len(text[:-4].split(". ")) ])
    one = "Graph models " + "and more " * 120 + "detect fraud."
    cut, shortened = fit_sentences(one)
    assert shortened and len(cut) <= MAX_STATEMENT_CHARS and cut.endswith("…") and not cut.endswith(" …")
    assert one.startswith(cut[:-1])


def test_summarize_shortens_by_whole_sentences_and_checks_what_is_shown():
    findings = ["Graph learning may improve early fraud identification.",
                "Graph neural networks can capture relationships among accounts."]
    places = ["retail", "banking", "insurance", "lending", "payments", "brokerage", "remittance", "wallets", "cards",
              "exchanges", "leasing", "factoring"]
    filler = " ".join(f"Graph learning may improve early fraud identification in {p}." for p in places)
    long_ok = {"text": filler, "findings": ["F1"]}
    lost = {"text": filler + " Graph neural networks can capture relationships among accounts.", "findings": ["F1", "F2"]}
    out = summarize(DeepClient({"summary": {"answered": True, "direct_answer": long_ok, "summary": [lost]}}),
                    "Does graph learning help?", findings)
    answer = out["direct_answer"]
    assert answer["shortened"] and answer["text"].endswith(" […]") and len(answer["text"]) <= MAX_STATEMENT_CHARS
    assert out["summary"] == [] and out["dropped"] == 1        # its "can" sentence fell outside the limit: not shown
    plain = summarize(DeepClient({"summary": {"answered": True, "direct_answer": {"text": findings[0], "findings": ["F1"]},
                                              "summary": []}}), "q", findings)
    assert plain["direct_answer"] == {"text": findings[0], "findings": ["F1"]}       # under the limit: unchanged


def test_a_policy_refusal_is_not_reported_as_a_provider_failure():
    """Gate 1 v4 live run: arXiv's address was refused by the network guard, and the record called arXiv "failed"."""
    providers, session = world()
    refused = FakeProvider("openalex", "scholarly", lambda q: [],
                           error="blocked_host: export.arxiv.org resolves to a private or reserved address")
    broken = FakeProvider("searxng", "web", lambda q: [], error="http_503: down")
    data, _, _, _ = deep_run(depth="quick", providers=[refused, broken], session=session)
    states = {p["id"]: p for p in data["provider_coverage"]}
    assert states["openalex"]["state"] == "blocked_by_policy" and "not a provider failure" in states["openalex"]["detail"]
    assert states["searxng"]["state"] == "failed"                                  # a real failure stays a failure
    assert any("'openalex' was not contacted: the network-safety policy refused its address" in x for x in data["limitations"])
    assert any("Search provider 'searxng' failed" in x for x in data["limitations"])
    assert data["status"] != "COMPLETE"                                            # nothing was searched: not complete
