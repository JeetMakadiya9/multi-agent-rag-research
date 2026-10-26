"""Offline tests for search-provider configuration, metadata, de-duplication and HTTP safety (no network)."""
import json

import pytest

from webrag import dedupe
from webrag.net import FetchError, Fetched, PoliteSession, redact
from webrag.researcher import FULL_TEXT, Source, WebResearcher, provider_status
from webrag.search import (EuropePmcSearch, OpenAlexSearch, SearchError, SearchHit, SearxngSearch,
                           default_providers)


class FakeResponse:
    def __init__(self, status=200, body=b"", headers=None, url="https://site.org/page"):
        self.status_code, self.body, self.headers, self.url = status, body, headers or {}, url

    def iter_content(self, size):
        yield self.body

    def close(self):
        pass


def session_with(responses, resolver=lambda host: ["93.184.216.34"], **kwargs):
    session = PoliteSession(min_interval=0, resolver=resolver, respect_robots=False, **kwargs)
    queue, session.calls = list(responses), []

    def fake_get(url, **kw):
        session.calls.append({"url": url, **kw})
        return queue.pop(0)

    session.session.get = fake_get
    return session


class RecordingSession:
    def __init__(self, body):
        self.body, self.calls, self.requests_made, self.allowed_origins = body, [], 0, set()

    def get(self, url, *, params=None, headers=None, check_robots=True, max_bytes=None):
        self.calls.append({"url": url, "params": params or {}, "headers": headers or {}})
        if isinstance(self.body, Exception):
            raise self.body
        return Fetched(url, 200, "application/json", json.dumps(self.body).encode())


# ------------------------------------------------------------------------------------------------ HTTP safety

def test_redirect_to_a_private_address_is_refused():
    """Before the fix, requests followed redirects itself, so a public page could bounce the fetch to 127.0.0.1."""
    session = session_with([FakeResponse(302, headers={"Location": "http://127.0.0.1/admin"},
                                         url="https://public.org/a")])
    with pytest.raises(FetchError) as info:
        session.get("https://public.org/a")
    assert info.value.code == "blocked_host" and len(session.calls) == 1
    assert session.calls[0]["allow_redirects"] is False


def test_redirects_are_followed_hop_by_hop():
    session = session_with([FakeResponse(301, headers={"Location": "/b"}, url="https://public.org/a"),
                            FakeResponse(200, body=b"ok", headers={"Content-Type": "text/html"},
                                         url="https://public.org/b")])
    page = session.get("https://public.org/a")
    assert page.content == b"ok" and [c["url"] for c in session.calls] == ["https://public.org/a",
                                                                          "https://public.org/b"]
    loop = session_with([FakeResponse(302, headers={"Location": "/a"}, url="https://public.org/a")] * 7)
    with pytest.raises(FetchError, match="too many redirects"):
        loop.get("https://public.org/a")


def test_public_name_resolving_to_a_private_address_is_refused():
    session = session_with([], resolver=lambda host: ["10.0.0.5"])
    with pytest.raises(FetchError, match="resolves to a private"):
        session.get("https://intranet.example.org/")


def test_only_the_configured_local_origin_is_allowed():
    session = session_with([FakeResponse(200, body=b"{}", headers={"Content-Type": "application/json"},
                                         url="http://127.0.0.1:8888/search")],
                           allowed_origins=["http://127.0.0.1:8888"])
    assert session.get("http://127.0.0.1:8888/search").status == 200
    with pytest.raises(FetchError, match="blocked_host"):
        session.get("http://127.0.0.1:9999/other")


def test_secrets_are_redacted_from_urls_and_errors():
    assert redact("https://a.org/x?api_key=SECRET&q=1") == "https://a.org/x?api_key=REDACTED&q=1"
    session = session_with([FakeResponse(500, url="https://a.org/x?api_key=SECRET")])
    with pytest.raises(FetchError) as info:
        session.get("https://a.org/x?api_key=SECRET")
    assert "SECRET" not in str(info.value)


def test_openalex_key_travels_in_a_header_not_the_url(monkeypatch):
    monkeypatch.setenv("OPENALEX_API_KEY", "SECRET-KEY")
    session = RecordingSession({"results": []})
    OpenAlexSearch(session).search("q")
    call = session.calls[0]
    assert call["headers"] == {"Authorization": "Bearer SECRET-KEY"}
    assert "SECRET-KEY" not in json.dumps(call["params"]) and "SECRET-KEY" not in call["url"]


# ------------------------------------------------------------------------------------------------ providers

def test_provider_selection(monkeypatch):
    for name in ("WEBRAG_PROVIDERS", "SEARXNG_URL", "BRAVE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    session = PoliteSession(resolver=lambda h: ["93.184.216.34"])
    assert [p.name for p in default_providers(session)] == ["wikipedia", "openalex", "arxiv"]
    monkeypatch.setenv("WEBRAG_PROVIDERS", "wikipedia,europepmc")
    assert [p.name for p in default_providers(session)] == ["wikipedia", "europepmc"]
    with pytest.raises(ValueError, match="Unknown search provider"):
        default_providers(session, ["bing"])
    with pytest.raises(ValueError, match="SEARXNG_URL is not set"):
        default_providers(session, ["searxng"])
    monkeypatch.setenv("SEARXNG_URL", "http://127.0.0.1:8888/")
    providers = default_providers(session, ["searxng"])
    assert providers[0].name == "searxng" and "http://127.0.0.1:8888" in session.allowed_origins


def test_searxng_parsing_engines_and_json_refusal(monkeypatch):
    body = {"results": [{"title": "<b>Doc</b>", "url": "https://gov.example/report", "content": "A report.",
                         "engine": "duckduckgo"},
                        {"title": "bad", "url": "javascript:alert(1)", "content": "x"}]}
    session = RecordingSession(body)
    hits = SearxngSearch(session, "http://127.0.0.1:8888", engines="duckduckgo,wikipedia").search("q")
    assert [(h.title, h.url, h.extra["engine"]) for h in hits] == [("Doc", "https://gov.example/report", "duckduckgo")]
    assert session.calls[0]["params"]["engines"] == "duckduckgo,wikipedia"
    refused = RecordingSession(FetchError("http_403", "x"))
    with pytest.raises(SearchError, match="search.formats"):
        SearxngSearch(refused, "http://127.0.0.1:8888").search("q")


def test_europepmc_full_text_api_for_open_access_articles():
    """Regression (live smoke test): the open-access PDF link is robots-disallowed, so the full-text API is used."""
    xml = (b"<article><front><p>Front matter</p></front><body><sec><p>Statins reduced <b>major</b> events.</p>"
           b"<p>Second paragraph.</p></sec></body></article>")

    class XmlSession(RecordingSession):
        def get(self, url, **kw):
            self.calls.append({"url": url, **kw})
            return Fetched(url, 200, "application/xml", xml)

    session = XmlSession({})
    provider = EuropePmcSearch(session)
    hit = SearchHit("europepmc", "q", 1, "T", "https://europepmc.org/article/MED/1", kind="scholarly",
                    extra={"pmcid": "PMC9", "open_access": "Y"})
    assert provider.fetch_text(hit)["text"] == "Statins reduced major events.\nSecond paragraph."
    assert session.calls[0]["url"].endswith("/PMC9/fullTextXML")
    with pytest.raises(SearchError, match="no open-access"):
        provider.fetch_text(SearchHit("europepmc", "q", 1, "T", "u", extra={"pmcid": "", "open_access": "N"}))


def test_europepmc_parsing():
    body = {"resultList": {"result": [{
        "source": "MED", "id": "123", "pmid": "123", "pmcid": "PMC9", "doi": "10.1/abc", "title": "A <i>trial</i>",
        "abstractText": "<h4>Results</h4>The drug reduced pain.", "firstPublicationDate": "2021-05-01",
        "journalInfo": {"journal": {"title": "The Lancet"}},
        "fullTextUrlList": {"fullTextUrl": [{"documentStyle": "pdf", "availability": "Open access",
                                             "url": "https://europepmc.org/articles/PMC9?pdf=render"}]}}]}}
    hit = EuropePmcSearch(RecordingSession(body)).search("q")[0]
    assert (hit.title, hit.url, hit.doi, hit.abstract) == ("A trial", "https://europepmc.org/article/MED/123",
                                                          "https://doi.org/10.1/abc", "ResultsThe drug reduced pain.")
    assert hit.pdf_url.endswith("pdf=render") and hit.extra["venue"] == "The Lancet"
    assert dedupe.canonical_key(hit.url, hit.doi) == "doi:10.1/abc"       # merges with the same DOI elsewhere


# ------------------------------------------------------------------------------------------------ metadata and identity

class FakeProvider:
    def __init__(self, name, hits, kind="scholarly", error=None):
        self.name, self.hits, self.kind, self.error = name, hits, kind, error

    def search(self, query, limit=3):
        if self.error:
            raise SearchError(self.error)
        return [SearchHit(self.name, query, i, t, u, "", kind=self.kind, abstract=a, extra={"venue": "J. Test"})
                for i, (t, u, a) in enumerate(self.hits[:limit], 1)]


def test_sources_record_domain_publisher_and_provider_status():
    good = FakeProvider("openalex", [("Paper", "https://doi.org/10.1/x", "An abstract about fraud graphs.")])
    down = FakeProvider("arxiv", [], error="rate_limited: export.arxiv.org asked to retry after 99 s")
    corpus = WebResearcher([good, down], session=RecordingSession({}), fetch_full_text=False).gather(["q1", "q2"])
    source = corpus.sources[0]
    assert (source.domain, source.publisher) == ("doi.org", "J. Test")
    assert corpus.provider_status["arxiv"] == {"queries": 2, "failed": 2, "succeeded": 0, "hits": 0,
                                               "failures": {"rate_limited": 2}}
    assert corpus.provider_status["openalex"]["hits"] == 2


def test_followup_gather_skips_held_works_and_links_copies():
    text = " ".join(f"word{i}" for i in range(80))
    held = Source("S1", "Old", "https://doi.org/10.1/old", "openalex", "scholarly", "doi:10.1/old", FULL_TEXT, text,
                  origin="S1")
    provider = FakeProvider("openalex", [("Old again", "https://doi.org/10.1/old", "duplicate work"),
                                         ("Copy", "https://doi.org/10.1/copy", "Reprinted: " + text),
                                         ("New", "https://doi.org/10.1/new", "Completely different findings here.")])
    corpus = WebResearcher([provider], session=RecordingSession({}), fetch_full_text=False).gather(["q"], existing=[held])
    assert [(s.id, s.title) for s in corpus.sources] == [("S2", "Copy"), ("S3", "New")]
    assert corpus.sources[0].origin == "S1" and corpus.sources[1].origin == "S3"


def test_independent_sources_that_agree_are_not_merged():
    a = "A randomised trial of 400 adults found the vaccine reduced hospital admissions by about half compared " \
        "with placebo over two winter seasons in three countries, with mild side effects only."
    b = "Surveillance data from regional hospitals showed that vaccinated patients were admitted far less often; " \
        "the observational analysis estimated a fifty percent reduction after adjusting for age and comorbidity."
    assert dedupe.near_duplicate_groups([a, b]) == [0, 1]


def test_provider_status_without_errors():
    status = provider_status([FakeProvider("wikipedia", [])], [], ["q"], [])
    assert status == {"wikipedia": {"queries": 1, "failed": 0, "succeeded": 1, "hits": 0, "failures": {}}}
