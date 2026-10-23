"""Hosted front end helpers (webrag/hosted.py): capability checks, source search and run-record checks. Every record and
search result below is SYNTHETIC TEST DATA; no model, network or Streamlit is used."""
from __future__ import annotations

import json

import pytest

from webrag import hosted
from webrag.report import render_markdown
from webrag.search import SearchError, SearchHit

SAMPLE_RUN = {          # synthetic: shaped like Orchestrator.run output, values invented for the test
    "input": "When did the bridge open?", "mode": "research", "status": "COMPLETE", "outcome": "ANSWERED",
    "started": "2026-10-18T10:00:00Z", "stages": [], "error": "", "disagreements": [], "gaps": [], "limitations": [],
    "sources": [{"id": "S1", "title": "Bridge (synthetic)", "url": "https://example.org/bridge", "provider": "openalex",
                 "kind": "scholarly", "access": "ABSTRACT_ONLY", "published": "2020-05-01T00:00:00",
                 "retrieved_at": "2026-10-18T10:01:00Z"}],
    "claims": [{"id": "F1", "text": "The bridge opened in 1932.", "label": "SUPPORTED", "sub_state": "",
                "cited_sources": ["S1"], "supporting_sources": ["S1"], "contradicting_sources": [],
                "independent_support": 1, "independent_contradiction": 0, "cautions": [],
                "passages": [{"source_id": "S1", "url": "https://example.org/bridge", "stance": "SUPPORTS",
                              "text": "The bridge opened in 1932."}]},
               {"id": "F2", "text": "It is 900 metres long.", "label": "INSUFFICIENT", "sub_state": "",
                "cited_sources": ["S1"], "supporting_sources": [], "contradicting_sources": [],
                "independent_support": 0, "independent_contradiction": 0, "passages": []}],
}


class FakeProvider:
    def __init__(self, name, hits=(), error=None, supports_since=False):
        self.name, self.hits, self.error, self.supports_since = name, list(hits), error, supports_since
        self.calls = []

    def search(self, query, limit=3, **kwargs):
        self.calls.append((query, limit, kwargs))
        if self.error:
            raise self.error
        return self.hits[:limit]


def hit(provider, title, url="https://example.org/a"):
    return SearchHit(provider=provider, query="q", rank=1, title=title, url=url, kind="scholarly",
                     abstract="An abstract (synthetic).")


def test_search_providers_hide_secret_values(monkeypatch):
    monkeypatch.setenv("BRAVE_API_KEY", "secret-value-for-test")
    monkeypatch.delenv("SEARXNG_URL", raising=False)
    rows = {p["id"]: p for p in hosted.search_providers()}
    assert rows["brave"]["configured"] is True and rows["searxng"]["configured"] is False
    assert "secret-value-for-test" not in json.dumps(rows)
    assert "wikipedia" not in rows and "feeds" not in rows


def test_search_sources_reports_each_provider(monkeypatch):
    providers = {"openalex": FakeProvider("openalex", [hit("openalex", "Paper A")], supports_since=True),
                 "arxiv": FakeProvider("arxiv", error=SearchError("rate_limited"))}
    monkeypatch.setattr(hosted, "default_providers", lambda session, names, config=None: [providers[names[0]]])
    result = hosted.search_sources("  graph   networks ", ["openalex", "arxiv"], limit=2, since="2020-01-01",
                                   session=object())
    assert result["query"] == "graph networks"
    assert [h["title"] for h in result["hits"]] == ["Paper A"]
    assert result["errors"] == {"arxiv": "SearchError: rate_limited"}
    assert result["since_ignored"] == ["arxiv"]
    assert providers["openalex"].calls == [("graph networks", 2, {"since": "2020-01-01"})]
    assert result["hits"][0]["excluded_host"] == ""


def test_search_sources_filters_dates_for_providers_without_a_date_filter(monkeypatch):
    old, new, undated = hit("arxiv", "Old"), hit("arxiv", "New"), hit("arxiv", "Undated")
    old.published, new.published = "2019-08-31", "2026-06-18"
    provider = FakeProvider("arxiv", [old, new, undated])
    monkeypatch.setattr(hosted, "default_providers", lambda session, names, config=None: [provider])
    result = hosted.search_sources("x", ["arxiv"], limit=5, since="2022-01-01", session=object())
    assert [h["title"] for h in result["hits"]] == ["New", "Undated"]
    assert result["since_ignored"] == ["arxiv"] and provider.calls == [("x", 5, {})]


def test_search_sources_marks_excluded_hosts(monkeypatch):
    provider = FakeProvider("brave", [hit("brave", "Article", "https://en.wikipedia.org/wiki/X")])
    monkeypatch.setattr(hosted, "default_providers", lambda session, names, config=None: [provider])
    result = hosted.search_sources("x", ["brave"], session=object())
    assert result["hits"][0]["excluded_host"] == "wikipedia.org"


@pytest.mark.parametrize("query, providers, limit, message", [
    ("", ["openalex"], 3, "enter a search query"),
    ("x" * 301, ["openalex"], 3, "longer than"),
    ("x", [], 3, "at least one"),
    ("x", ["wikipedia"], 3, "not searchable"),
    ("x", ["feeds"], 3, "not searchable"),
    ("x", ["openalex"], 0, "between 1 and"),
    ("x", ["openalex"], 11, "between 1 and"),
])
def test_search_sources_rejects_bad_requests(query, providers, limit, message):
    with pytest.raises(ValueError, match=message):
        hosted.search_sources(query, providers, limit, session=object())


def test_load_run_accepts_a_run_and_renders_it():
    data = hosted.load_run(json.dumps(SAMPLE_RUN).encode("utf-8-sig"))
    assert hosted.label_counts(data) == {"SUPPORTED": 1, "CONTRADICTED": 0, "INSUFFICIENT": 1}
    assert [r["verdict"] for r in hosted.claim_rows(data)] == ["SUPPORTED", "INSUFFICIENT"]
    assert hosted.source_rows(data)[0] == {"id": "S1", "title": "Bridge (synthetic)", "provider": "openalex",
                                           "access": "ABSTRACT_ONLY", "published": "2020-05-01",
                                           "url": "https://example.org/bridge"}
    assert "The bridge opened in 1932." in render_markdown(data)


@pytest.mark.parametrize("raw, message", [
    (b"\xff\xfe", "not a JSON file"),
    (b"not json", "not a JSON file"),
    (b"[]", "JSON object"),
    (json.dumps({"mode": "chat", "input": "x", "status": "COMPLETE"}).encode(), "'mode'"),
    (json.dumps({"mode": "research", "status": "COMPLETE"}).encode(), "'input'"),
    (json.dumps({**SAMPLE_RUN, "sources": [{"title": "no id"}]}).encode(), "'id'"),
    (json.dumps({**SAMPLE_RUN, "claims": [{"text": "no label"}]}).encode(), "'label'"),
])
def test_load_run_rejects_other_files(raw, message):
    with pytest.raises(ValueError, match=message):
        hosted.load_run(raw)


def test_load_run_refuses_large_files(monkeypatch):
    monkeypatch.setattr(hosted, "MAX_RUN_BYTES", 10)
    with pytest.raises(ValueError, match="larger than"):
        hosted.load_run(json.dumps(SAMPLE_RUN).encode())
