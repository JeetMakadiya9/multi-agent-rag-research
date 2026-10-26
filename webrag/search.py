"""Search providers. Each turns a query into real search hits from a public API or a configured feed; none invents
results.

Default set: Wikipedia (MediaWiki API), OpenAlex (scholarly works) and arXiv (preprints), plus a general web search
when configured: SEARXNG_URL (a self-hosted SearXNG with JSON output enabled; optional SEARXNG_ENGINES and
SEARXNG_CATEGORIES) or BRAVE_API_KEY (Brave Search API). Opt-in: Europe PMC (biomedical and life-science literature,
keyless), Crossref (scholarly metadata by DOI, keyless) and feeds (RSS/Atom feeds the user lists: news, official
publications, podcasts with published transcripts). WEBRAG_PROVIDERS (comma-separated names) replaces the default set.
OPENALEX_API_KEY, when set, is sent to OpenAlex in an Authorization header (never in the URL); keys are free:
https://help.openalex.org/api/authentication/

``PROVIDERS`` describes every implemented provider (category, what it returns, what it needs); ``provider_catalog``
reports, for a configuration, which are implemented, configured, enabled and active, without exposing secret values.

Source policy: Wikipedia is not an accepted evidence source for the studio. With a studio configuration (``config``)
the Wikipedia provider is never built, and ``excluded_host`` lets the deep researcher leave out Wikipedia pages that
another provider returns. Without a configuration (the original command-line pipeline and the evaluation) provider
selection is unchanged.

Responses are checked against the shape each service documents before nested fields are read; a response of another
shape is a ``SearchError`` ("malformed_response: ...") for that provider and query, like any other failed search.
"""
from __future__ import annotations

import json
import math
import os
import re
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .extract import openalex_abstract, strip_tags
from .net import FetchError, PoliteSession

STOPWORDS = set("""a an and are as at be by did do does for from has have how in into is it its of on or that the
their them they this to was were what when where whether which who why will with than then there these those versus vs
compared compare investigate find relevant studies evidence claim verify about between over under more less""".split())


@dataclass
class SearchHit:
    provider: str
    query: str
    rank: int
    title: str
    url: str
    snippet: str = ""
    kind: str = "web"                   # web | encyclopedia | scholarly | preprint
    published: Optional[str] = None
    doi: Optional[str] = None
    abstract: str = ""
    pdf_url: Optional[str] = None
    extra: Dict[str, str] = field(default_factory=dict)
    authors: List[str] = field(default_factory=list)
    publisher: str = ""
    identifiers: Dict[str, str] = field(default_factory=dict)     # doi, arxiv, pmid, pmcid, isbn, guid ...
    category: str = ""                  # source category when it differs from ``kind`` (news, official, podcast)


class SearchError(Exception):
    pass


# ================================================================================================= provider registry
@dataclass(frozen=True)
class ProviderInfo:
    id: str
    name: str
    category: str                       # main source category
    categories: Tuple[str, ...]         # categories its results can belong to
    returns: str                        # what a search returns
    content: str                        # what the studio can then read from a result
    requires: Tuple[str, ...] = ()      # configuration needed before it can run
    secrets: Tuple[str, ...] = ()       # environment variables holding secrets (shown only as set / not set)
    default_on: bool = True             # part of the default set once configured
    query_style: str = "keywords"
    max_per_query: int = 3              # informational: the single-pass pipeline asks every provider for Settings.per_query
    # Deep research: candidates requested per search (one request, no paging), well inside the service's own page
    # limit. A larger pool costs no extra request; the source budget still caps what is read. 0 = unknown, so the
    # run's per-query setting is used.
    candidates: int = 0
    recency: str = ""                   # how a time range is applied: "native" (server-side filter) or "" (results
    limits: str = ""                    # are filtered afterwards by their recorded date)
    evidence: bool = True               # False: never used as an evidence source in the studio (source policy)


PROVIDERS: Dict[str, ProviderInfo] = {p.id: p for p in [
    ProviderInfo("wikipedia", "Wikipedia", "encyclopedia", ("encyclopedia",),
                 "article titles, URLs and search snippets", "the full article text through the MediaWiki API",
                 default_on=False, evidence=False,
                 limits="Not accepted as an evidence source in the studio; kept only for the original command-line "
                        "pipeline."),
    ProviderInfo("openalex", "OpenAlex", "scholarly", ("scholarly",),
                 "titles, DOIs, dates, venues, authors, abstracts and open-access PDF links",
                 "abstracts, and full text when an open-access PDF is linked", secrets=("OPENALEX_API_KEY",),
                 max_per_query=4, candidates=10, recency="native",
                 limits="Without a key, requests share a daily budget per network address (HTTP 429 when used up)."),
    ProviderInfo("arxiv", "arXiv", "preprint", ("preprint",),
                 "titles, abstracts, authors, dates and PDF links", "abstracts, and full text from the PDF",
                 query_style="boolean-keywords", candidates=8, limits="Queries are up to 5 keywords joined with AND; 3 s between requests."),
    ProviderInfo("crossref", "Crossref", "scholarly", ("scholarly",),
                 "DOI metadata: titles, authors, publishers, dates, licences and abstracts when deposited",
                 "abstracts when deposited; full text only from openly licensed links", default_on=False,
                 max_per_query=4, candidates=10, recency="native",
                 limits="Metadata service: many records have no abstract. A contact email is sent with requests when set."),
    ProviderInfo("europepmc", "Europe PMC", "scholarly", ("scholarly", "preprint"),
                 "biomedical articles with abstracts, journals, PMIDs and PMCIDs",
                 "abstracts, and open-access full text through its full-text API", default_on=False, candidates=8,
                 limits="Biomedical and life-science literature only."),
    ProviderInfo("searxng", "SearXNG", "web", ("web", "news", "official"),
                 "web results: titles, URLs, snippets, dates when the engine gives them",
                 "the page or PDF at each result, read under robots.txt", requires=("SEARXNG_URL",),
                 query_style="natural", max_per_query=5, candidates=10, recency="native",
                 limits="A self-hosted instance with JSON output enabled. Its upstream engines decide coverage."),
    ProviderInfo("brave", "Brave Search API", "web", ("web", "news", "official"),
                 "web results: titles, URLs, snippets and page age", "the page or PDF at each result, read under robots.txt",
                 requires=("BRAVE_API_KEY",), secrets=("BRAVE_API_KEY",), query_style="natural", max_per_query=5, candidates=10,
                 recency="native", limits="Paid API key; usage counts against its plan."),
    ProviderInfo("feeds", "RSS and Atom feeds", "news", ("news", "official", "podcast", "web"),
                 "entries of the feeds you list: titles, links, dates, authors, summaries, transcript links",
                 "the linked article or document, or a podcast transcript the feed publishes", requires=("feeds",),
                 default_on=True, query_style="keywords", max_per_query=5, candidates=8,
                 limits="Only the feeds you list are read; entries are matched to the query by their keywords."),
]}

# Which providers can serve each source category, in preference order.
CATEGORY_PROVIDERS: Dict[str, Tuple[str, ...]] = {
    "encyclopedia": ("wikipedia",),
    "scholarly": ("openalex", "crossref", "europepmc", "arxiv"),
    "web": ("searxng", "brave"),
    "news": ("feeds", "searxng", "brave"),
    "official": ("feeds", "searxng", "brave"),
    "podcast": ("feeds",),
}
SOURCE_CATEGORIES = tuple(CATEGORY_PROVIDERS)


def keywords(query: str, limit: int = 6) -> List[str]:
    words = [w for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]*", query) if w.lower() not in STOPWORDS and len(w) > 1]
    return list(dict.fromkeys(words))[:limit]


# ================================================================================================= source policy
# Hosts whose pages are never evidence in the studio, whichever provider finds them (a web search, a feed) and
# whether they are reached directly or by a redirect.
EVIDENCE_EXCLUDED_HOSTS = ("wikipedia.org",)
EXCLUSION_NOTE = "Wikipedia is not accepted as an evidence source"


def is_evidence_provider(name: str) -> bool:
    """False for providers the source policy keeps out of the studio (Wikipedia); unknown names are not judged here."""
    info = PROVIDERS.get(name)
    return info is None or info.evidence


def excluded_host(url: Optional[str]) -> str:
    """The excluded host ``url`` belongs to (itself or a subdomain, e.g. en.m.wikipedia.org), or ""."""
    try:
        host = (urllib.parse.urlsplit(str(url or "")).hostname or "").lower().rstrip(".")
    except ValueError:
        return ""
    return next((h for h in EVIDENCE_EXCLUDED_HOSTS if host == h or host.endswith("." + h)), "")


# ================================================================================================= response shapes
def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "a boolean"
    return {dict: "an object", list: "an array", str: "a string"}.get(type(value), "a number" if isinstance(value, (int, float))
                                                                     else type(value).__name__)


def _malformed(host: str, detail: str) -> SearchError:
    return SearchError(f"malformed_response: {host or 'the service'} {detail}")


def _member(data: Mapping[str, Any], key: str, kind: type, host: str, default: Any = None, nullable: bool = False) -> Any:
    """``data[key]`` when it has the JSON type the service documents; ``default`` when the key is absent (or null, for
    fields documented as nullable). Any other value means the response does not have the expected shape."""
    if key not in data or (nullable and data[key] is None):
        return default
    value = data[key]
    if isinstance(value, kind) and not isinstance(value, bool):
        return value
    raise _malformed(host, f"sent {key!r} as {_json_type(value)}, expected {_json_type(kind())}")


def _rows(value: List[Any], host: str, what: str) -> List[Dict[str, Any]]:
    """A list of result objects; a list holding anything else is a malformed response, not a short result."""
    bad = sum(not isinstance(row, dict) for row in value)
    if bad:
        raise _malformed(host, f"sent {bad} {what} that {'is' if bad == 1 else 'are'} not an object")
    return value


def _text(value: Any, host: str, what: str) -> str:
    """A text field: a string as it is, a number as text, null as empty; an object or array is malformed."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    raise _malformed(host, f"sent {what!r} as {_json_type(value)}, expected text")


def _json(session: PoliteSession, url: str, params: dict, headers: Optional[dict] = None) -> dict:
    host = urllib.parse.urlsplit(url).hostname or ""
    try:
        response = session.get(url, params=params, headers=headers, check_robots=False, max_bytes=5_000_000)
        data = json.loads(response.content)
    except FetchError as exc:
        raise SearchError(str(exc)) from None
    except ValueError:
        raise SearchError(f"malformed JSON from {host}") from None
    if not isinstance(data, dict):
        raise _malformed(host, f"sent {_json_type(data)} at the top level, expected an object")
    return data


class WikipediaSearch:
    name = "wikipedia"
    api = "https://en.wikipedia.org/w/api.php"

    def __init__(self, session: PoliteSession):
        self.session = session

    def search(self, query: str, limit: int = 3) -> List[SearchHit]:
        data = _json(self.session, self.api, {"action": "query", "list": "search", "srsearch": query,
                                               "srlimit": limit, "format": "json", "utf8": 1})
        host = "en.wikipedia.org"
        hits = []
        for rank, row in enumerate(_rows(_member(_member(data, "query", dict, host, {}), "search", list, host, []),
                                         host, "search results"), 1):
            title = _text(row.get("title"), host, "title")
            pageid = _text(row.get("pageid"), host, "pageid")
            url = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))
            hits.append(SearchHit(self.name, query, rank, title, url, strip_tags(_text(row.get("snippet"), host, "snippet")),
                                  kind="encyclopedia", extra={"pageid": pageid},
                                  publisher="Wikipedia", identifiers={"wikipedia_pageid": pageid}))
        return hits

    def fetch_text(self, hit: SearchHit) -> Dict[str, str]:
        """Plain-text article via the API (avoids scraping HTML); returns text and last-revision timestamp."""
        data = _json(self.session, self.api, {"action": "query", "prop": "extracts|revisions", "explaintext": 1,
                                               "rvprop": "timestamp", "pageids": hit.extra.get("pageid", ""),
                                               "format": "json", "redirects": 1})
        host = "en.wikipedia.org"
        pages = _rows(list(_member(_member(data, "query", dict, host, {}), "pages", dict, host, {}).values()), host, "pages")
        if not pages or "extract" not in pages[0]:
            raise SearchError("article text unavailable")
        revisions = _rows(_member(pages[0], "revisions", list, host, []), host, "revisions") or [{}]
        return {"text": _text(pages[0]["extract"], host, "extract"),
                "revised": _text(revisions[0].get("timestamp"), host, "timestamp")}


class OpenAlexSearch:
    name = "openalex"
    api = "https://api.openalex.org/works"

    def __init__(self, session: PoliteSession):
        self.session = session

    supports_since = True

    def search(self, query: str, limit: int = 4, since: Optional[str] = None) -> List[SearchHit]:
        params = {"search": query, "per-page": limit,
                  "select": "id,doi,display_name,publication_date,abstract_inverted_index,best_oa_location,"
                            "primary_location,type,cited_by_count,authorships"}
        if since:
            params["filter"] = f"from_publication_date:{since}"
        key = os.environ.get("OPENALEX_API_KEY", "").strip()
        headers = {"Authorization": f"Bearer {key}"} if key else None      # a header keeps the key out of URLs
        try:
            data = _json(self.session, self.api, params, headers)
        except SearchError as exc:
            if "rate_limited" in str(exc) and not os.environ.get("OPENALEX_API_KEY"):
                raise SearchError(f"{exc}. OpenAlex's free daily budget for requests without an API key is used up; "
                                  f"set OPENALEX_API_KEY (keys are free) or retry after midnight UTC") from None
            raise
        host = "api.openalex.org"
        hits = []
        for rank, row in enumerate(_rows(_member(data, "results", list, host, []), host, "results"), 1):
            primary = _member(row, "primary_location", dict, host, {}, nullable=True)       # documented as nullable
            best = _member(row, "best_oa_location", dict, host, {}, nullable=True)
            venue = _member(primary, "source", dict, host, {}, nullable=True)
            doi, ident = _text(row.get("doi"), host, "doi"), _text(row.get("id"), host, "id")
            url = doi or _text(primary.get("landing_page_url"), host, "landing_page_url") or ident
            source = _text(venue.get("display_name"), host, "display_name")
            authors = [_text(_member(a, "author", dict, host, {}, nullable=True).get("display_name"), host, "author")
                       for a in _rows(_member(row, "authorships", list, host, [], nullable=True), host, "authorships")]
            hits.append(SearchHit(self.name, query, rank, _text(row.get("display_name"), host, "display_name"), url,
                                  kind="scholarly", published=_text(row.get("publication_date"), host, "publication_date") or None,
                                  doi=doi or None, abstract=openalex_abstract(_inverted_index(row, host)),
                                  pdf_url=_text(best.get("pdf_url"), host, "pdf_url") or None,
                                  extra={"venue": source, "type": _text(row.get("type"), host, "type"),
                                         "cited_by": _text(row.get("cited_by_count"), host, "cited_by_count"),
                                         "oa_landing": _text(best.get("landing_page_url"), host, "landing_page_url"),
                                         "license": _text(best.get("license"), host, "license")},
                                  authors=[a for a in authors if a][:12], publisher=source,
                                  identifiers={k: v for k, v in {"doi": doi, "openalex": ident}.items() if v}))
        return hits


def _inverted_index(row: Mapping[str, Any], host: str) -> Optional[Dict[str, List[int]]]:
    """OpenAlex's abstract as {word: [positions]}, checked before it is rebuilt."""
    index = _member(row, "abstract_inverted_index", dict, host, None, nullable=True)
    if index is not None and not all(isinstance(places, list) and all(isinstance(p, int) and not isinstance(p, bool) for p in places)
                                     for places in index.values()):
        raise _malformed(host, "sent an abstract index whose positions are not lists of numbers")
    return index


class ArxivSearch:
    name = "arxiv"
    api = "https://export.arxiv.org/api/query"
    ns = {"a": "http://www.w3.org/2005/Atom"}

    def __init__(self, session: PoliteSession):
        self.session = session

    def search(self, query: str, limit: int = 3) -> List[SearchHit]:
        terms = keywords(query, 5)
        if not terms:
            return []
        search_query = " AND ".join(f"all:{t}" for t in terms)
        try:
            response = self.session.get(self.api, params={"search_query": search_query, "max_results": limit},
                                        check_robots=False, max_bytes=3_000_000)
            root = ET.fromstring(response.content)
        except FetchError as exc:
            raise SearchError(str(exc)) from None
        except ET.ParseError:
            raise SearchError("malformed arXiv response") from None
        hits = []
        for rank, entry in enumerate(root.findall("a:entry", self.ns), 1):
            abs_url = (entry.findtext("a:id", "", self.ns) or "").strip()
            pdf = next((link.get("href") for link in entry.findall("a:link", self.ns)
                        if link.get("title") == "pdf"), None)
            title = re.sub(r"\s+", " ", entry.findtext("a:title", "", self.ns)).strip()
            abstract = re.sub(r"\s+", " ", entry.findtext("a:summary", "", self.ns)).strip()
            authors = [re.sub(r"\s+", " ", a.findtext("a:name", "", self.ns)).strip()
                       for a in entry.findall("a:author", self.ns)]
            ident = re.sub(r"^https?://arxiv\.org/abs/", "", abs_url)
            hits.append(SearchHit(self.name, query, rank, title, abs_url, abstract[:300], kind="preprint",
                                  published=(entry.findtext("a:published", "", self.ns) or "")[:10] or None,
                                  abstract=abstract, pdf_url=pdf, authors=[a for a in authors if a][:12],
                                  publisher="arXiv", identifiers={"arxiv": ident} if ident else {}))
        return hits


class BraveSearch:
    name = "brave"
    api = "https://api.search.brave.com/res/v1/web/search"

    supports_since = True

    def __init__(self, session: PoliteSession, key: str):
        self.session, self.key = session, key

    def search(self, query: str, limit: int = 5, since: Optional[str] = None) -> List[SearchHit]:
        params = {"q": query, "count": limit}
        if since and _within_days(since, 366):
            params["freshness"] = "py"                      # past year; longer ranges are filtered afterwards
        data = _json(self.session, self.api, params,
                     headers={"X-Subscription-Token": self.key, "Accept": "application/json"})
        host = "api.search.brave.com"
        web = _member(data, "web", dict, host, {})          # absent: no web results; null or another type: malformed
        return [SearchHit(self.name, query, rank, strip_tags(_text(row.get("title"), host, "title")),
                          _text(row.get("url"), host, "url"), strip_tags(_text(row.get("description"), host, "description")),
                          published=_text(row.get("page_age"), host, "page_age") or None,
                          publisher=_text(_member(row, "profile", dict, host, {}, nullable=True).get("name"), host, "profile name"))
                for rank, row in enumerate(_rows(_member(web, "results", list, host, []), host, "results"), 1)
                if row.get("url")]


class SearxngSearch:
    """A SearXNG instance with ``search.formats: [html, json]`` in its settings.yml. Self-hosting on this machine is
    the intended setup; its configured origin is the only private address the HTTP client will contact."""

    name = "searxng"
    supports_categories = True

    def __init__(self, session: PoliteSession, base_url: str, engines: str = "", categories: str = ""):
        self.session, self.base_url = session, base_url.rstrip("/")
        self.engines, self.categories = engines.strip(), categories.strip()
        session.allowed_origins.add(_origin_of(self.base_url))

    supports_since = True

    def search(self, query: str, limit: int = 5, since: Optional[str] = None, categories: str = "") -> List[SearchHit]:
        params = {"q": query, "format": "json"}
        if self.engines:
            params["engines"] = self.engines
        if categories or self.categories:
            params["categories"] = categories or self.categories
        if since and _within_days(since, 366):
            params["time_range"] = "year"                   # SearXNG's widest range; older limits are applied afterwards
        try:
            data = _json(self.session, self.base_url + "/search", params)
        except SearchError as exc:
            if "http_403" in str(exc):
                raise SearchError(f"{exc}. SearXNG refused JSON output: add 'json' to search.formats in its "
                                  f"settings.yml") from None
            raise
        host = urllib.parse.urlsplit(self.base_url).hostname or ""
        rows = _rows(_member(data, "results", list, host, []), host, "results")[:limit]
        return [SearchHit(self.name, query, rank, strip_tags(_text(row.get("title"), host, "title")), _text(row.get("url"), host, "url"),
                          strip_tags(_text(row.get("content"), host, "content")),
                          published=_text(row.get("publishedDate"), host, "publishedDate") or None,
                          extra={"engine": _text(row.get("engine"), host, "engine")})
                for rank, row in enumerate(rows, 1)
                if _text(row.get("url"), host, "url").startswith(("http://", "https://"))]


class EuropePmcSearch:
    """Europe PMC REST API (keyless): biomedical and life-science articles and preprints, with abstracts and links to
    open-access full text."""

    name = "europepmc"
    api = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

    def __init__(self, session: PoliteSession):
        self.session = session

    def search(self, query: str, limit: int = 3) -> List[SearchHit]:
        data = _json(self.session, self.api, {"query": query, "format": "json", "pageSize": limit,
                                               "resultType": "core"})
        host = "www.ebi.ac.uk"
        results = _member(_member(data, "resultList", dict, host, {}, nullable=True), "result", list, host, [])
        hits = []
        for rank, row in enumerate(_rows(results, host, "results")[:limit], 1):
            source, ident = _text(row.get("source"), host, "source"), _text(row.get("id"), host, "id")
            if not (source and ident):
                continue
            links = _rows(_member(_member(row, "fullTextUrlList", dict, host, {}, nullable=True), "fullTextUrl", list, host, [],
                                  nullable=True), host, "full-text links")
            pdf = next((_text(l.get("url"), host, "url") for l in links if _text(l.get("documentStyle"), host, "documentStyle") == "pdf"
                        and _text(l.get("availability"), host, "availability").lower() in {"open access", "free"}), None) or None
            journal = _text(_member(_member(row, "journalInfo", dict, host, {}, nullable=True), "journal", dict, host, {},
                                    nullable=True).get("title"), host, "journal title")
            doi, pmid, pmcid = (_text(row.get(k), host, k) for k in ("doi", "pmid", "pmcid"))
            identifiers = {k: v for k, v in {"doi": doi, "pmid": pmid, "pmcid": pmcid}.items() if v}
            hits.append(SearchHit(self.name, query, rank, strip_tags(_text(row.get("title"), host, "title")),
                                  f"https://europepmc.org/article/{source}/{ident}",
                                  kind="preprint" if source == "PPR" else "scholarly",
                                  published=(_text(row.get("firstPublicationDate"), host, "firstPublicationDate")
                                             or _text(row.get("pubYear"), host, "pubYear") or None),
                                  doi=f"https://doi.org/{doi}" if doi else None,
                                  abstract=strip_tags(_text(row.get("abstractText"), host, "abstractText")), pdf_url=pdf,
                                  extra={"venue": journal, "pmid": pmid, "pmcid": pmcid,
                                         "open_access": _text(row.get("isOpenAccess"), host, "isOpenAccess").upper()},
                                  authors=[a.strip() for a in _text(row.get("authorString"), host, "authorString").rstrip(".").split(",")
                                           if a.strip()][:12], publisher=journal, identifiers=identifiers))
        return hits

    def fetch_text(self, hit: SearchHit) -> Dict[str, str]:
        """Open-access full text through Europe PMC's documented full-text API (JATS XML), for articles with a PMCID.
        The article's PDF link is disallowed by europepmc.org's robots.txt, which page fetching respects."""
        pmcid = hit.extra.get("pmcid", "")
        if not pmcid or hit.extra.get("open_access") != "Y":
            raise SearchError("no open-access full text")
        try:
            response = self.session.get(f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML",
                                        check_robots=False, max_bytes=8_000_000)
            root = ET.fromstring(response.content)
        except FetchError as exc:
            raise SearchError(str(exc)) from None
        except ET.ParseError:
            raise SearchError("malformed full-text XML") from None
        body = root.find(".//body")
        paragraphs = ["".join(p.itertext()).strip() for p in (body.iter("p") if body is not None else [])]
        text = "\n".join(re.sub(r"\s+", " ", p) for p in paragraphs if p)
        if not text:
            raise SearchError("full-text XML had no body text")
        return {"text": text}


class CrossrefSearch:
    """Crossref REST API (keyless): metadata deposited by publishers for DOIs. Abstracts exist only when the
    publisher deposited them; a full-text link is used only when it carries an open (Creative Commons) licence."""

    name = "crossref"
    api = "https://api.crossref.org/works"
    supports_since = True

    def __init__(self, session: PoliteSession, mailto: str = ""):
        self.session, self.mailto = session, mailto.strip()

    def search(self, query: str, limit: int = 4, since: Optional[str] = None) -> List[SearchHit]:
        params = {"query": query, "rows": limit,
                  "select": "DOI,title,author,published,issued,container-title,publisher,abstract,link,license,type,URL"}
        if since:
            params["filter"] = f"from-pub-date:{since}"
        if self.mailto:
            params["mailto"] = self.mailto                   # Crossref's documented "polite" identification
        data = _json(self.session, self.api, params)
        host = "api.crossref.org"
        items = _member(_member(data, "message", dict, host, {}), "items", list, host, [])
        return [hit for rank, row in enumerate(_rows(items, host, "items"), 1)
                if (hit := crossref_hit(row, query, rank)) is not None]


def crossref_hit(row: Mapping[str, Any], query: str = "", rank: int = 1) -> Optional[SearchHit]:
    host = "api.crossref.org"
    doi = _text(row.get("DOI"), host, "DOI").strip()
    titles = _member(row, "title", list, host, [], nullable=True)
    if not doi or not titles:
        return None
    authors = [" ".join(p for p in (_text(a.get("given"), host, "given"), _text(a.get("family"), host, "family")) if p)
               or _text(a.get("name"), host, "name")
               for a in _rows(_member(row, "author", list, host, [], nullable=True), host, "authors")]
    dated = (_member(row, "published", dict, host, None, nullable=True)
             or _member(row, "issued", dict, host, {}, nullable=True))
    parts = _member(dated, "date-parts", list, host, [], nullable=True)
    date_parts = parts[0] if parts else []
    if not isinstance(date_parts, list) or not all(p is None or (isinstance(p, int) and not isinstance(p, bool)) for p in date_parts):
        raise _malformed(host, "sent a date that is not a list of numbers")
    published = "-".join(f"{p:02d}" if i else str(p) for i, p in enumerate(date_parts) if p) or None
    licences = [_text(l.get("URL"), host, "licence URL")
                for l in _rows(_member(row, "license", list, host, [], nullable=True), host, "licences")]
    open_licence = any("creativecommons.org" in l for l in licences)
    links = _rows(_member(row, "link", list, host, [], nullable=True), host, "links")
    pdf = next((_text(l.get("URL"), host, "link URL") for l in links
                if _text(l.get("content-type"), host, "content-type") == "application/pdf" and l.get("URL")), None) if open_licence else None
    venues = _member(row, "container-title", list, host, [], nullable=True)
    venue = _text(venues[0], host, "container-title") if venues else ""
    return SearchHit("crossref", query, rank, strip_tags(_text(titles[0], host, "title")), f"https://doi.org/{doi}",
                     kind="scholarly", published=published, doi=f"https://doi.org/{doi}",
                     abstract=strip_tags(_text(row.get("abstract"), host, "abstract")),
                     pdf_url=pdf, extra={"venue": venue, "type": _text(row.get("type"), host, "type"),
                                         "license": licences[0] if licences else ""},
                     authors=[a for a in authors if a][:12], publisher=venue or _text(row.get("publisher"), host, "publisher"),
                     identifiers={"doi": doi})


class FeedSearch:
    """RSS and Atom feeds the user lists (news sites, official publication feeds, podcasts). A search reads each
    feed once per run and returns the entries whose title and summary contain the query's keywords."""

    name = "feeds"
    supports_since = True
    supports_categories = True

    def __init__(self, session: PoliteSession, feeds: Sequence[Mapping[str, str]]):
        self.session = session
        self.feeds = [dict(f) for f in feeds if f.get("url")]
        self._cache: Dict[str, Any] = {}

    def _entries(self, feed: Mapping[str, str]):
        from .feeds import parse_feed
        url = feed["url"]
        if url not in self._cache:
            try:
                page = self.session.get(url, max_bytes=5_000_000)
                self._cache[url] = parse_feed(page.content, page.url)
            except (FetchError, ValueError) as exc:
                self._cache[url] = SearchError(f"{feed.get('name') or url}: {exc}")
        return self._cache[url]

    def search(self, query: str, limit: int = 5, since: Optional[str] = None, categories: str = "") -> List[SearchHit]:
        terms = [t.lower() for t in keywords(query, 8)]
        if not terms:
            return []
        needed = max(1, math.ceil(len(terms) / 2))
        wanted = {c.strip() for c in categories.split(",") if c.strip()}
        scored, failures, considered = [], [], 0
        for feed in self.feeds:
            category = feed.get("category") or "news"
            if wanted and category not in wanted:
                continue
            considered += 1
            entries = self._entries(feed)
            if isinstance(entries, SearchError):
                failures.append(str(entries))
                continue
            for entry in entries:
                text = f"{entry.title} {entry.summary}".lower()
                score = sum(1 for t in terms if t in text)
                if score < needed or (since and entry.published and entry.published[:10] < since):
                    continue
                scored.append((score, entry.published or "", feed, category, entry))
        if considered and len(failures) == considered:          # every feed failed: a failure, not an empty result
            raise SearchError("; ".join(failures))
        scored.sort(key=lambda row: row[1], reverse=True)          # newest first ...
        scored.sort(key=lambda row: -row[0])                       # ... within the best keyword match
        hits = []
        for rank, (score, _, feed, category, entry) in enumerate(scored[:limit], 1):
            extra = {"feed": feed.get("name") or feed["url"], "feed_url": feed["url"]}
            if entry.transcript_url:
                extra.update(transcript_url=entry.transcript_url, transcript_type=entry.transcript_type)
            if entry.media_url:
                extra.update(media_url=entry.media_url, media_type=entry.media_type, duration=entry.duration)
            hits.append(SearchHit(self.name, query, rank, entry.title, entry.url, entry.summary[:500],
                                  kind="web", published=entry.published, extra=extra, authors=entry.authors,
                                  publisher=feed.get("name") or entry.feed_title, category=category,
                                  identifiers={"guid": entry.guid} if entry.guid else {}))
        return hits


def _within_days(since: str, days: int) -> bool:
    from datetime import date, timedelta
    try:
        return date.fromisoformat(since[:10]) >= date.today() - timedelta(days=days)
    except ValueError:
        return False


def _origin_of(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    return f"{parsed.scheme}://{parsed.netloc}".lower()


DEFAULT_PROVIDERS = ("wikipedia", "openalex", "arxiv", "searxng", "brave")
AVAILABLE_PROVIDERS = DEFAULT_PROVIDERS + ("europepmc", "crossref", "feeds")


def _configured(name: str, config: Optional[Mapping[str, Any]] = None) -> Tuple[bool, List[str]]:
    """Whether a provider has what it needs, and what is missing (names only, never values)."""
    config = config or {}
    if name == "searxng":
        ok = bool(os.environ.get("SEARXNG_URL", "").strip() or (config.get("searxng") or {}).get("url"))
        return ok, [] if ok else ["SEARXNG_URL (or a SearXNG address in Integrations)"]
    if name == "brave":
        ok = bool(os.environ.get("BRAVE_API_KEY", "").strip())
        return ok, [] if ok else ["BRAVE_API_KEY environment variable"]
    if name == "feeds":
        ok = bool(config.get("feeds"))
        return ok, [] if ok else ["at least one feed address in Integrations"]
    return True, []


def _build(name: str, session: PoliteSession, config: Optional[Mapping[str, Any]] = None):
    config = config or {}
    if name == "wikipedia":
        return WikipediaSearch(session)
    if name == "openalex":
        return OpenAlexSearch(session)
    if name == "arxiv":
        return ArxivSearch(session)
    if name == "europepmc":
        return EuropePmcSearch(session)
    if name == "crossref":
        return CrossrefSearch(session, str(config.get("contact_email") or ""))
    if name == "feeds":
        return FeedSearch(session, config.get("feeds") or [])
    if name == "searxng":
        settings = config.get("searxng") or {}
        url = os.environ.get("SEARXNG_URL", "").strip() or str(settings.get("url") or "")
        return SearxngSearch(session, url, os.environ.get("SEARXNG_ENGINES", "") or str(settings.get("engines") or ""),
                             os.environ.get("SEARXNG_CATEGORIES", "") or str(settings.get("categories") or ""))
    if name == "brave":
        return BraveSearch(session, os.environ["BRAVE_API_KEY"].strip())
    raise ValueError(name)


def enabled_names(config: Optional[Mapping[str, Any]] = None) -> List[str]:
    """Providers switched on in the configuration (default-on providers unless switched off; opt-in ones only when
    switched on), in registry order. Providers the source policy excludes are never switched on."""
    switches = (config or {}).get("providers") or {}
    return [pid for pid, info in PROVIDERS.items()
            if info.evidence and switches.get(pid, {}).get("enabled", info.default_on) is True]


def default_providers(session: PoliteSession, names: Optional[List[str]] = None,
                      config: Optional[Mapping[str, Any]] = None) -> list:
    """Providers by name: ``names``, else WEBRAG_PROVIDERS, else the configuration's enabled set (when a
    configuration is given), else the default set. Unconfigured default providers (SearXNG, Brave, feeds) are left out
    of the default set; naming an unknown or unconfigured provider is an error rather than a silent omission.

    With a configuration (the studio), providers the source policy excludes (Wikipedia) are left out even when they
    are named; without one (the original command-line pipeline and the evaluation) the selection is unchanged."""
    if not names and os.environ.get("WEBRAG_PROVIDERS", "").strip():
        names = [n.strip() for n in os.environ["WEBRAG_PROVIDERS"].split(",") if n.strip()]
    explicit = bool(names)
    if names:
        names = list(names)
    elif config is not None:
        names = enabled_names(config)
    else:
        names = list(DEFAULT_PROVIDERS)
    unknown = [n for n in names if n not in AVAILABLE_PROVIDERS]
    if unknown:
        raise ValueError(f"Unknown search provider(s) {unknown}; available: {list(AVAILABLE_PROVIDERS)}")
    if config is not None:
        names = [n for n in names if is_evidence_provider(n)]
    providers = []
    for name in names:
        ok, missing = _configured(name, config)
        if not ok:
            if explicit:
                need = {"searxng": "SEARXNG_URL is not set", "brave": "BRAVE_API_KEY is not set",
                        "feeds": "no feed is configured"}[name]
                raise ValueError(f"Provider '{name}' was requested but {need}.")
            continue
        provider = _build(name, session, config)
        if name == "searxng":
            session.allowed_origins.add(_origin_of(provider.base_url))
        providers.append(provider)
    return providers


def provider_catalog(config: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Any]]:
    """Every implemented provider with its configuration state. ``active`` means a run would use it now; secrets are
    reported only as set or not set."""
    override = [n.strip() for n in os.environ.get("WEBRAG_PROVIDERS", "").split(",") if n.strip()]
    enabled = set(enabled_names(config))
    rows = []
    for pid, info in PROVIDERS.items():
        configured, missing = _configured(pid, config)
        switched_on = info.evidence and (pid in override if override else pid in enabled)
        rows.append({
            "id": pid, "name": info.name, "category": info.category, "categories": list(info.categories),
            "returns": info.returns, "content": info.content, "query_style": info.query_style,
            "max_per_query": info.max_per_query, "candidates_per_query": info.candidates or None, "recency": info.recency or "filtered after search",
            "limits": info.limits, "default_on": info.default_on, "implemented": True,
            "configured": configured, "missing": missing, "enabled": switched_on,
            "active": switched_on and configured,
            "evidence": info.evidence,
            "policy": "" if info.evidence else f"{EXCLUSION_NOTE}, so it is never searched and its pages are never "
                                               f"used as evidence, even if WEBRAG_PROVIDERS names it.",
            "controlled_by": "policy" if not info.evidence else "WEBRAG_PROVIDERS" if override else "settings",
            "secrets": {name: bool(os.environ.get(name, "").strip()) for name in info.secrets},
        })
    return rows
