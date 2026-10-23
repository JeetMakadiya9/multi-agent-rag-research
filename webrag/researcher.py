"""Agent B - Web Researcher: runs real searches, fetches and extracts accessible content, and records exactly how much
of each source was read (full text, abstract only, snippet only, metadata only, or failed).

Two modes share this code:
* the original single-pass mode (``classify=False``, the default), unchanged for the command line, the evaluation
  harness and existing runs;
* the deep-research mode (``classify=True``), which also: tells metadata-only records apart from failed retrievals,
  follows a legal open-access copy for a DOI (``OpenAccessResolver``), reads podcast transcripts published in feeds
  (with timestamps), treats login, paywall, bot-check and error pages as not retrieved, reports scanned PDFs, keeps a
  total text budget, records retrieval diagnostics, identifiers and authors on every source, and applies the source
  policy: results from an excluded provider or host (Wikipedia) are dropped before fetching, and a page that redirects
  to an excluded host is dropped after it; neither becomes a source, so neither can be indexed, cited or listed.
"""
from __future__ import annotations

import hashlib
import re
import threading
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .dedupe import arxiv_id, canonical_key, near_duplicate_groups, normalize_doi
from .extract import (ExtractionError, html_to_text, parse_transcript, pdf_to_text, transcript_pages)
from .net import FetchError, PoliteSession
from .search import SearchError, SearchHit, excluded_host, is_evidence_provider, keywords

FULL_TEXT, ABSTRACT_ONLY, SNIPPET_ONLY, FAILED = "FULL_TEXT", "ABSTRACT_ONLY", "SNIPPET_ONLY", "FAILED"
METADATA_ONLY = "METADATA_ONLY"            # deep mode: a record with no content to read (or none obtained by budget)
ACCESS_LEVELS = (FULL_TEXT, ABSTRACT_ONLY, SNIPPET_ONLY, METADATA_ONLY, FAILED)
MIN_FULL_TEXT_CHARS = 1500
TRANSCRIPT_TYPES = {"text/vtt", "application/x-subrip", "application/srt", "text/srt"}


@dataclass
class Source:
    id: str
    title: str
    url: str
    provider: str
    kind: str
    key: str
    access: str = FAILED
    text: str = ""
    pages: Optional[List[str]] = None
    published: Optional[str] = None
    retrieved_at: str = ""
    queries: List[str] = field(default_factory=list)
    origin: str = ""                       # id of the source this one copies; its own id when independent
    note: str = ""
    read_url: Optional[str] = None         # the URL whose content was actually read
    extra: Dict[str, str] = field(default_factory=dict)
    domain: str = ""                       # host of the URL that was read (or found)
    publisher: str = ""                    # journal/venue, or the site (Wikipedia, arXiv, a web domain)
    # Recorded in deep-research runs (empty otherwise):
    category: str = ""                     # encyclopedia | scholarly | preprint | web | news | official | podcast
    authors: List[str] = field(default_factory=list)
    identifiers: Dict[str, str] = field(default_factory=dict)
    canonical_url: str = ""
    content_type: str = ""
    retrieval: str = ""                    # retrieved | abstract | snippet | metadata | failed | skipped
    retrieval_error: str = ""
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    fingerprint: str = ""                  # sha256 prefix of the text that was indexed
    license: str = ""
    discovered_by: List[str] = field(default_factory=list)
    page_labels: Optional[List[str]] = None    # transcript time ranges, one per "page"
    media_url: str = ""
    round: int = 1

    @property
    def usable(self) -> bool:
        return self.access not in {FAILED, METADATA_ONLY} and bool(self.text.strip())


@dataclass
class Corpus:
    sources: List[Source]
    hits: List[SearchHit]
    search_errors: List[Tuple[str, str, str]]          # (provider, query, error)
    requests_made: int = 0
    provider_status: Dict[str, Dict[str, object]] = field(default_factory=dict)
    filtered_by_date: int = 0                           # hits dropped because their recorded date is too old
    queries_sent: int = 0
    excluded_by_policy: int = 0                         # deep mode: results and redirects to excluded hosts dropped


@dataclass
class Route:
    """One query for one provider (deep mode): the planner decides which providers suit which queries."""
    query: str
    provider: str
    category: str = ""
    purpose: str = "answer"                # answer | counter | follow-up
    subquestion: int = -1


PUBLISHER_BY_PROVIDER = {"wikipedia": "Wikipedia", "arxiv": "arXiv"}


def _domain(url: Optional[str]) -> str:
    host = (urllib.parse.urlsplit(url or "").hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def provider_status(providers: Sequence, hits: Sequence[SearchHit], queries: Sequence[str],
                    errors: Sequence[Tuple[str, str, str]], sent: Optional[Dict[str, int]] = None) -> Dict[str, Dict[str, object]]:
    """Per provider: queries sent, failures by error code, and hits returned. ``sent`` overrides the number of queries
    each provider received (routed searches send different queries to different providers)."""
    status: Dict[str, Dict[str, object]] = {}
    for provider in providers:
        failures: Dict[str, int] = {}
        for name, _, error in errors:
            if name == provider.name:
                code = error.split(":", 1)[0].strip() or "error"
                failures[code] = failures.get(code, 0) + 1
        failed = sum(failures.values())
        count = sent.get(provider.name, 0) if sent is not None else len(queries)
        status[provider.name] = {"queries": count, "failed": failed, "succeeded": count - failed,
                                 "hits": sum(h.provider == provider.name for h in hits), "failures": failures}
    return status


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def candidate_score(hit: SearchHit, queries: Sequence[str]) -> float:
    """Deep mode: how promising a search result is before it is read. Relevance is the share of the queries' keywords
    found in its title and abstract (or snippet); readable content adds a little (an abstract, an open-access PDF or
    PMCID, a DOI that may resolve to an open copy, a web page that is read in full); lower provider ranks cost a
    little. A ranking aid for choosing what to read, never evidence and never part of a verdict."""
    terms = {t.lower() for q in queries for t in keywords(q, 8)}
    words = set(re.findall(r"[a-z0-9][a-z0-9\-]*", f"{hit.title} {hit.abstract or hit.snippet}".lower()))
    relevance = len(terms & words) / len(terms) if terms else 0.0
    readable = 0.2 if hit.abstract else 0.0
    if hit.kind not in {"scholarly", "preprint"}:
        readable += 0.2                                 # a web page or document is fetched and read in full
    elif hit.pdf_url or hit.extra.get("pmcid"):
        readable += 0.2
    elif hit.doi:
        readable += 0.1
    return round(relevance + readable - 0.03 * (max(hit.rank, 1) - 1), 4)


# Deep mode: a result whose abstract shares less than this share of a query's keywords is not read for that query.
# Calibrated on one live run (an off-topic work scored 0.62, every on-topic one 0.75 or more); a heuristic, not a
# measure of scientific relevance. Results without an abstract (web snippets, bare records) are not judged by it.
RELEVANCE_FLOOR = 0.7
MIN_ABSTRACT_CHARS = 200


def _stem(word: str) -> str:
    return word[:-1] if len(word) > 4 and word.endswith("s") and not word.endswith("ss") else word


def query_relevance(hit: SearchHit, query: str) -> Optional[float]:
    """Share of the query's keywords found in the result's title and abstract (plural endings ignored), or None when
    the result has no abstract to judge it by."""
    if len(hit.abstract or "") < MIN_ABSTRACT_CHARS:
        return None
    terms = {_stem(t.lower()) for t in keywords(query, 8)}
    words = {_stem(w) for w in re.findall(r"[a-z0-9][a-z0-9\-]*", f"{hit.title} {hit.abstract}".lower())}
    return round(len(terms & words) / len(terms), 2) if terms else None


def _venue(hit: SearchHit) -> str:
    """What a selected result counts as for diversity: its venue or publisher, else its site."""
    return (hit.extra.get("venue") or hit.publisher or _domain(hit.url)).strip().lower()


def _identifiers(hit: SearchHit) -> Dict[str, str]:
    ids = {k: v for k, v in hit.identifiers.items() if v}
    doi = normalize_doi(hit.doi or ids.get("doi")) or (normalize_doi(hit.url) if "doi.org/" in (hit.url or "") else None)
    if doi:
        ids["doi"] = doi
    arxiv = arxiv_id(hit.url) or arxiv_id(hit.doi or "")
    if arxiv:
        ids["arxiv"] = arxiv
    for key in ("pmid", "pmcid"):
        if hit.extra.get(key):
            ids[key] = hit.extra[key]
    return ids


class OpenAccessResolver:
    """A legal open-access copy for a DOI, from OpenAlex's record of the work (its open-access locations are compiled
    from publisher and repository data). Only locations OpenAlex marks as open access are returned; nothing behind a
    paywall is attempted."""

    api = "https://api.openalex.org/works/doi:"

    def __init__(self, session: PoliteSession):
        self.session = session
        self.lookups = 0

    def resolve(self, doi: str) -> Dict[str, str]:
        import json
        import os
        self.lookups += 1
        key = os.environ.get("OPENALEX_API_KEY", "").strip()
        try:
            page = self.session.get(self.api + urllib.parse.quote(doi, safe="/"), check_robots=False,
                                    params={"select": "best_oa_location,open_access"},
                                    headers={"Authorization": f"Bearer {key}"} if key else None, max_bytes=2_000_000)
            data = json.loads(page.content)
        except (FetchError, ValueError) as exc:
            return {"error": str(exc)}
        best = data.get("best_oa_location") if isinstance(data, dict) else None
        access = data.get("open_access") if isinstance(data, dict) else None
        if not isinstance(data, dict) or not isinstance(best or {}, dict) or not isinstance(access or {}, dict):
            return {"error": "malformed_response: api.openalex.org sent an unexpected shape for the work"}
        text = lambda record, key: record.get(key) if isinstance(record.get(key), str) else ""
        return {"pdf_url": text(best or {}, "pdf_url"), "landing_url": text(best or {}, "landing_page_url"),
                "license": text(best or {}, "license"), "oa_status": text(access or {}, "oa_status")}


@dataclass
class Read:
    text: str
    final_url: str
    content_type: str
    pages: Optional[List[str]] = None
    page_labels: Optional[List[str]] = None
    title: str = ""
    published: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    canonical_url: str = ""
    pdf_url: Optional[str] = None
    gate: str = ""
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class WebResearcher:
    def __init__(self, providers: Sequence, session: PoliteSession, per_query: int = 3, max_sources: int = 10,
                 fetch_full_text: bool = True, max_full_text: int = 4, max_chars: int = 80_000, workers: int = 4,
                 classify: bool = False, resolver: Optional[OpenAccessResolver] = None, max_pdf_pages: int = 30,
                 max_total_chars: Optional[int] = None, since: Optional[str] = None, round_number: int = 1,
                 per_provider: Optional[Dict[str, int]] = None):
        # deep mode applies the source policy to providers too: an excluded one (Wikipedia) is never queried
        self.providers = [p for p in providers if not classify or is_evidence_provider(p.name)]
        self.session = session
        self.per_query, self.max_sources = per_query, max_sources
        self.per_provider = dict(per_provider or {})     # deep mode: candidates requested per search, by provider
        self.route_results: List[Dict[str, Any]] = []    # one record per search: requested, returned, error
        self.candidates = 0                              # distinct works among the results, before the source cap
        self._selection: Dict[str, Dict[str, Any]] = {}  # deep mode: why each selected result was chosen
        self._planned: List[Tuple[Any, ...]] = []        # deep mode: full-text downloads waiting for a priority slot
        self.query_picks: Dict[str, Optional[str]] = {}  # deep mode: the work read first for each query (None: none relevant)
        self.below_floor = 0                             # deep mode: results left out by the relevance floor
        self.full_text_outcome = {"attempts": 0, "succeeded": 0, "failed": 0}
        self._defer = False                              # set by gather: queue full text instead of fetching at once
        self.fetch_full_text, self.max_full_text = fetch_full_text, max_full_text
        self.max_chars, self.workers = max_chars, workers
        self.classify, self.resolver, self.max_pdf_pages = classify, resolver, max_pdf_pages
        self.max_total_chars, self.since, self.round_number = max_total_chars, since, round_number
        self._full_text_lock = threading.Lock()
        self._full_text_used = 0
        self.full_text_attempts = 0
        self.chars_used = 0

    # ------------------------------------------------------------------ search
    def _call(self, provider, query: str, category: str = "") -> List[SearchHit]:
        kwargs: Dict[str, Any] = {}
        if self.since and getattr(provider, "supports_since", False):
            kwargs["since"] = self.since
        if category and getattr(provider, "supports_categories", False):
            kwargs["categories"] = {"news": "news", "podcast": "podcast", "official": "official"}.get(category, "")
            if provider.name == "searxng" and category in {"official", "podcast", "web"}:
                kwargs["categories"] = ""                     # SearXNG has no such category; use its default
        hits = provider.search(query, self.per_provider.get(provider.name, self.per_query), **kwargs)
        for hit in hits:
            if category and not hit.category and hit.kind not in {"scholarly", "preprint", "encyclopedia"}:
                hit.category = category
        return hits

    def search(self, queries: Sequence[str], routes: Optional[Sequence[Route]] = None
               ) -> Tuple[List[SearchHit], List[Tuple[str, str, str]]]:
        errors: List[Tuple[str, str, str]] = []
        by_name = {p.name: p for p in self.providers}

        def run(provider) -> List[SearchHit]:
            found: List[SearchHit] = []
            work = [(q, "") for q in queries] if routes is None else \
                [(r.query, r.category) for r in routes if r.provider == provider.name]
            for query, category in work:
                record = {"provider": provider.name, "query": query, "category": category,
                          "requested": self.per_provider.get(provider.name, self.per_query), "returned": 0, "error": ""}
                try:
                    hits = self._call(provider, query, category)
                    record["returned"] = len(hits)
                    found.extend(hits)
                except SearchError as exc:
                    errors.append((provider.name, query, str(exc)))
                    record["error"] = str(exc)
                self.route_results.append(record)
            return found

        active = self.providers if routes is None else [by_name[n] for n in dict.fromkeys(r.provider for r in routes) if n in by_name]
        if not active:
            return [], errors
        with ThreadPoolExecutor(max_workers=max(1, len(active))) as pool:
            results = list(pool.map(run, active))
        return [hit for group in results for hit in group], errors

    def select(self, hits: Sequence[SearchHit]) -> List[Tuple[SearchHit, List[str]]]:
        """De-duplicate by canonical key, up to ``max_sources``: every query's first-ranked hits come before any
        second-ranked hit, and within one rank the provider order decides (Wikipedia, OpenAlex, arXiv, then web)."""
        merged: Dict[str, Tuple[SearchHit, List[str]]] = {}
        order: List[str] = []
        self._discovered: Dict[str, List[str]] = {}
        for hit in sorted(hits, key=lambda h: (h.rank, self._provider_index(h.provider))):
            if not hit.url:
                continue
            key = canonical_key(hit.url, hit.doi)
            providers = self._discovered.setdefault(key, [])
            if hit.provider not in providers:
                providers.append(hit.provider)
            if key in merged:
                first, queries = merged[key]
                if hit.query not in queries:
                    queries.append(hit.query)
                first.abstract = first.abstract or hit.abstract
                first.pdf_url = first.pdf_url or hit.pdf_url
                if self.classify:                            # keep the richest metadata across providers
                    first.authors = first.authors or hit.authors
                    first.published = first.published or hit.published
                    first.doi = first.doi or hit.doi
                    first.identifiers = {**hit.identifiers, **first.identifiers}
                    for k, v in hit.extra.items():
                        first.extra.setdefault(k, v)
                continue
            merged[key] = (hit, [hit.query])
            order.append(key)
        self.candidates = len(order)
        if self.classify:
            order = self._rank(order, merged)
        return [merged[k] for k in order[:self.max_sources]]

    def _rank(self, order: List[str], merged: Dict[str, Tuple[SearchHit, List[str]]]) -> List[str]:
        """Deep mode: the order in which distinct works are read. First the best-scoring candidate of each query (so
        every query that ran, a counter-evidence query included, is represented), then the rest by score, each
        further work from an already chosen venue or site scoring a little less. Deterministic: ties keep the
        provider's rank order. A result with an abstract that is below ``RELEVANCE_FLOOR`` for every query that found
        it is not read at all: ranking first in a weak pool is not enough. A query whose results all fall below the
        floor gets no first pick, and that is recorded (``query_picks``)."""
        scores = {k: candidate_score(merged[k][0], merged[k][1]) for k in order}
        relevance = {k: [r for r in (query_relevance(merged[k][0], q) for q in merged[k][1]) if r is not None] for k in order}
        rejected = {k for k in order if relevance[k] and max(relevance[k]) < RELEVANCE_FLOOR}
        for k in rejected:
            self._selection[k] = {"score": scores[k], "relevance": max(relevance[k]),
                                  "reason": f"not read: below the relevance floor ({RELEVANCE_FLOOR}) for every query that found it"}
        self.below_floor = len(rejected)
        order = [k for k in order if k not in rejected]
        position = {k: i for i, k in enumerate(order)}
        queries = list(dict.fromkeys(q for k in list(order) + sorted(rejected) for q in merged[k][1]))
        picked: List[str] = []
        for query in queries:
            best = sorted((k for k in order if query in merged[k][1] and k not in picked),
                          key=lambda k: (-scores[k], position[k]))
            self.query_picks[query] = best[0] if best else None
            if best:
                picked.append(best[0])
                self._selection[best[0]] = {"score": scores[best[0]], "reason": f"best candidate for the query “{query}”"}
        rest = [k for k in order if k not in picked]
        used: Dict[str, int] = {}
        for k in picked:
            used[_venue(merged[k][0])] = used.get(_venue(merged[k][0]), 0) + 1
        while rest:
            best = min(rest, key=lambda k: (-(scores[k] - 0.15 * used.get(_venue(merged[k][0]), 0)), position[k]))
            rest.remove(best)
            picked.append(best)
            used[_venue(merged[best][0])] = used.get(_venue(merged[best][0]), 0) + 1
            self._selection[best] = {"score": scores[best], "reason": "ranked by relevance, readable content and venue diversity"}
        for i, k in enumerate(picked):
            self._selection[k]["rank"] = i + 1
            if i >= self.max_sources:
                self._selection[k]["reason"] = "not read: the run's source limit was reached"
        return picked

    def _provider_index(self, name: str) -> int:
        return next((i for i, p in enumerate(self.providers) if p.name == name), len(self.providers))

    # ------------------------------------------------------------------ fetch
    def _claim_full_text_slot(self) -> bool:
        with self._full_text_lock:
            if self._full_text_used >= self.max_full_text:
                return False
            self._full_text_used += 1
            self.full_text_attempts += 1
            return True

    def _claim_chars(self, n: int) -> bool:
        with self._full_text_lock:
            if self.max_total_chars is not None and self.chars_used + n > self.max_total_chars:
                return False
            self.chars_used += n
            return True

    def _read_document(self, url: str) -> Read:
        """Fetch a page, PDF, plain-text file or transcript and extract its text."""
        page = self.session.get(url)
        ctype = page.content_type
        if ctype == "application/pdf" or page.content[:5] == b"%PDF-":
            extracted = pdf_to_text(page.content, max_pages=self.max_pdf_pages)
            diagnostics = {"pdf_pages": extracted.page_count, "pages_read": len(extracted.pages or []),
                           "empty_pages": extracted.empty_pages, "image_only": extracted.image_only,
                           "truncated_pages": extracted.truncated}
            return Read(extracted.text, page.url, "application/pdf", extracted.pages, None, extracted.title,
                        extracted.published, diagnostics=diagnostics,
                        gate="scanned PDF without a text layer (no OCR)" if extracted.image_only else "")
        if self.classify and (ctype in TRANSCRIPT_TYPES or url.lower().split("?")[0].endswith((".vtt", ".srt"))):
            pages, labels = transcript_pages(parse_transcript(page.text, ctype or url))
            return Read("\n".join(pages), page.url, ctype or "text/vtt", pages, labels,
                        diagnostics={"transcript_windows": len(pages)})
        if ctype in {"text/html", "application/xhtml+xml", ""}:
            extracted = html_to_text(page.text)
            return Read(extracted.text, page.url, ctype or "text/html", None, None, extracted.title, extracted.published,
                        extracted.authors, extracted.canonical_url, extracted.pdf_url, extracted.gate,
                        {"main_content": extracted.used_main, "language": extracted.language})
        if ctype.startswith("text/"):
            return Read(page.text, page.url, ctype)
        raise FetchError("unsupported_type", ctype)

    def _read(self, url: str) -> Tuple[str, Optional[List[str]], str, Optional[str], str]:
        """Fetch a page or PDF and extract text: (text, pdf pages, title, published, final url)."""
        read = self._read_document(url)
        return read.text, read.pages, read.title, read.published, read.final_url

    def fetch(self, source_id: str, hit: SearchHit, queries: List[str]) -> Source:
        source = Source(source_id, hit.title or hit.url, hit.url, hit.provider, hit.kind, canonical_key(hit.url, hit.doi),
                        published=hit.published, retrieved_at=_now(), queries=queries, origin=source_id,
                        extra=dict(hit.extra))
        if self.classify:
            source.category = hit.category or hit.kind
            source.authors, source.identifiers = list(hit.authors), _identifiers(hit)
            source.license = hit.extra.get("license", "")
            source.media_url = hit.extra.get("media_url", "")
            source.round = self.round_number
            source.discovered_by = list(getattr(self, "_discovered", {}).get(source.key, [hit.provider]))
            if source.key in self._selection:
                source.diagnostics["selection"] = dict(self._selection[source.key])
        provider = next((p for p in self.providers if p.name == hit.provider), None)
        attempted = False
        try:
            if hit.provider == "wikipedia" and provider is not None:
                attempted = True
                article = provider.fetch_text(hit)
                source.text, source.access, source.read_url = article["text"], FULL_TEXT, hit.url
                source.published = source.published or (article.get("revised") or None)
                source.note = "Wikipedia article text via the MediaWiki API; date is the latest revision."
            elif hit.kind in {"scholarly", "preprint"}:
                attempted = self._plan_scholarly(source, hit) if self.classify and self._defer else self._fetch_scholarly(source, hit)
            elif self.classify and hit.extra.get("transcript_url"):
                attempted = True
                self._fetch_transcript(source, hit)
            else:
                attempted = True
                self._fetch_page(source, hit.url)
        except (FetchError, SearchError) as exc:
            source.note = f"Content not retrieved ({exc})."
            source.retrieval_error = str(exc)
        except ExtractionError as exc:
            source.note = f"Content could not be extracted ({exc})."
            source.retrieval_error = str(exc)
        except Exception as exc:                          # extraction bugs must not stop the other sources
            source.note = f"Content extraction failed ({type(exc).__name__})."
            source.retrieval_error = type(exc).__name__
        if not source.usable and hit.snippet:
            source.text, source.access = hit.snippet, SNIPPET_ONLY
            source.note = (source.note + " Only the search snippet is available.").strip()
        if source.usable:
            source.text = source.text[:self.max_chars]
        if self.classify:
            self._classify(source, hit, attempted)
        source.domain = _domain(source.read_url or source.url)
        source.publisher = PUBLISHER_BY_PROVIDER.get(hit.provider) or hit.extra.get("venue") or hit.publisher or source.domain
        return source

    def _fetch_page(self, source: Source, url: str) -> None:
        read = self._read_document(url)
        source.content_type = read.content_type
        source.diagnostics.update(read.diagnostics)
        if self.classify:
            source.authors = source.authors or read.authors
            source.canonical_url = read.canonical_url
            if read.gate:
                source.diagnostics["gate"] = read.gate
                source.note = f"Not read: the page looks like a {read.gate}."
                source.retrieval_error = read.gate
                return
        if len(read.text) >= 300:
            if self.classify and not self._claim_chars(len(read.text[:self.max_chars])):
                source.note = "Not read: the run's text budget was used up."
                source.retrieval = "skipped"
                return
            source.text, source.pages, source.access, source.read_url = read.text, read.pages, FULL_TEXT, read.final_url
            source.page_labels = read.page_labels
            source.title = source.title or read.title
            source.published = source.published or read.published
        else:
            source.note = "Page had too little extractable text."
            source.retrieval_error = "too little text"

    def _fetch_transcript(self, source: Source, hit: SearchHit) -> None:
        """A transcript the publisher links from the feed; the audio or video itself is not processed."""
        read = self._read_document(hit.extra["transcript_url"])
        if not self._claim_chars(len(read.text[:self.max_chars])):
            source.note, source.retrieval = "Not read: the run's text budget was used up.", "skipped"
            return
        source.text, source.pages, source.page_labels = read.text, read.pages, read.page_labels
        source.access, source.read_url, source.content_type = FULL_TEXT, read.final_url, read.content_type
        source.category = source.category or "podcast"
        source.diagnostics.update(read.diagnostics)
        source.note = ("Transcript published with the episode (passages are cited by time); the audio itself was not "
                       "listened to or transcribed.")

    def _fetch_scholarly(self, source: Source, hit: SearchHit) -> bool:
        """Returns whether a retrieval was attempted (the abstract from the search API counts as one)."""
        attempted = False
        if hit.abstract:
            attempted = True
            source.text, source.access, source.read_url = hit.abstract, ABSTRACT_ONLY, hit.url
            source.note = "Abstract only (from the search API)."
        provider = next((p for p in self.providers if p.name == hit.provider), None)
        api_full_text = provider is not None and hasattr(provider, "fetch_text") and hit.extra.get("pmcid")
        if api_full_text and self.fetch_full_text and self._claim_full_text_slot():
            attempted = True
            try:
                text = provider.fetch_text(hit)["text"]
            except SearchError as exc:
                source.note += f" Open-access full text not retrieved ({exc})."
                source.diagnostics["full_text_error"] = str(exc)
            else:
                if len(text) >= MIN_FULL_TEXT_CHARS and (not self.classify or self._claim_chars(len(text[:self.max_chars]))):
                    source.text, source.access, source.read_url = text, FULL_TEXT, hit.url
                    source.note = "Open-access full text (provider full-text API)."
                    return attempted
        pdf_url, landing = hit.pdf_url, ""
        if self.classify and not pdf_url and self.resolver is not None and self.fetch_full_text:
            doi = normalize_doi(hit.doi) or (normalize_doi(hit.url) if "doi.org/" in (hit.url or "") else None)
            if doi:
                found = self.resolver.resolve(doi)
                source.diagnostics["open_access_lookup"] = {k: v for k, v in found.items() if k != "error"} or {"error": found.get("error", "")}
                pdf_url, landing = found.get("pdf_url") or None, found.get("landing_url") or ""
                source.license = source.license or found.get("license", "")
        if not self.fetch_full_text or not (pdf_url or landing):
            return attempted
        if not self._claim_full_text_slot():
            if self.classify:
                source.diagnostics["full_text_skipped"] = "full-text budget used up"
            return attempted
        attempted = True
        try:
            read = self._read_document(pdf_url or landing)
            if not pdf_url and read.pdf_url:                    # an open-access landing page that links its PDF
                read = self._read_document(urllib.parse.urljoin(read.final_url, read.pdf_url))
        except FetchError as exc:
            source.note += f" Open-access full text not retrieved ({exc.code})."
            source.diagnostics["full_text_error"] = exc.code
            return attempted
        except ExtractionError as exc:
            source.note += f" Open-access full text could not be extracted ({exc.code})."
            source.diagnostics["full_text_error"] = exc.code
            return attempted
        source.diagnostics.update(read.diagnostics)
        if read.gate:
            source.note += f" The full-text address showed a {read.gate}."
            source.diagnostics["full_text_error"] = read.gate
            return attempted
        if len(read.text) >= MIN_FULL_TEXT_CHARS:
            if self.classify and not self._claim_chars(len(read.text[:self.max_chars])):
                source.diagnostics["full_text_skipped"] = "text budget used up"
                return attempted
            source.text, source.pages, source.access, source.read_url = read.text, read.pages, FULL_TEXT, read.final_url
            source.content_type = read.content_type
            source.note = "Open-access full text."
        return attempted

    def _plan_scholarly(self, source: Source, hit: SearchHit) -> bool:
        """Deep mode: take the abstract, find where open-access full text could come from (the provider's full-text
        API, the result's PDF link, or an open-access copy of its DOI), and queue the download. Downloads happen after
        every selected source is known, in priority order (``_download_by_priority``), so the full-text budget goes to
        the most promising sources rather than to whichever thread asks first."""
        attempted = False
        if hit.abstract:
            attempted = True
            source.text, source.access, source.read_url = hit.abstract, ABSTRACT_ONLY, hit.url
            source.note = "Abstract only (from the search API)."
        if not self.fetch_full_text:
            return attempted
        provider = next((p for p in self.providers if p.name == hit.provider), None)
        api = provider if provider is not None and hasattr(provider, "fetch_text") and hit.extra.get("pmcid") else None
        pdf_url, landing = hit.pdf_url, ""
        if not pdf_url and self.resolver is not None:
            doi = normalize_doi(hit.doi) or (normalize_doi(hit.url) if "doi.org/" in (hit.url or "") else None)
            if doi:
                found = self.resolver.resolve(doi)
                source.diagnostics["open_access_lookup"] = {k: v for k, v in found.items() if k != "error"} or {"error": found.get("error", "")}
                pdf_url, landing = found.get("pdf_url") or None, found.get("landing_url") or ""
                source.license = source.license or found.get("license", "")
        if api is not None or pdf_url or landing:
            with self._full_text_lock:
                self._planned.append((source, hit, api, pdf_url, landing))
        return attempted

    def _download_by_priority(self) -> None:
        """Deep mode: grant the remaining full-text attempts to the queued sources with the best selection score
        (ties: the order they were selected), download those, and record why the others were not read in full."""
        def priority(item):
            source = item[0]
            return (-(self._selection.get(source.key, {}).get("score", 0.0)), int(source.id[1:]) if source.id[1:].isdigit() else 0)
        queued = sorted(self._planned, key=priority)
        self._planned = []
        granted = queued[:max(0, self.max_full_text - self._full_text_used)]
        with self._full_text_lock:
            self._full_text_used += len(granted)
            self.full_text_attempts += len(granted)
        if granted:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                list(pool.map(lambda item: self._download_full_text(*item), granted))
        # Attempts are not reads: say how many of the granted attempts succeeded and how many failed.
        succeeded = sum(item[0].access == FULL_TEXT for item in granted)
        self.full_text_outcome = {"attempts": len(granted), "succeeded": succeeded, "failed": len(granted) - succeeded}
        for source, *_ in queued[len(granted):]:
            source.diagnostics["full_text_skipped"] = (
                f"full-text budget: all {len(granted)} attempt(s) went to higher-priority sources "
                f"({succeeded} read in full, {len(granted) - succeeded} failed)" if granted else "full-text budget used up")

    def _download_full_text(self, source: Source, hit: SearchHit, api, pdf_url: Optional[str], landing: str) -> None:
        """One granted full-text attempt: the provider's full-text API first, then the open-access PDF or page."""
        try:
            if api is not None:
                try:
                    text = api.fetch_text(hit)["text"]
                except SearchError as exc:
                    source.note += f" Open-access full text not retrieved ({exc})."
                    source.diagnostics["full_text_error"] = str(exc)
                else:
                    if len(text) >= MIN_FULL_TEXT_CHARS and self._claim_chars(len(text[:self.max_chars])):
                        self._upgrade(source, text, None, hit.url, "", "Open-access full text (provider full-text API).")
                        return
            if not (pdf_url or landing):
                return
            read = self._read_document(pdf_url or landing)
            if not pdf_url and read.pdf_url:                    # an open-access landing page that links its PDF
                read = self._read_document(urllib.parse.urljoin(read.final_url, read.pdf_url))
        except FetchError as exc:
            source.note += f" Open-access full text not retrieved ({exc.code})."
            source.diagnostics["full_text_error"] = exc.code
            return
        except ExtractionError as exc:
            source.note += f" Open-access full text could not be extracted ({exc.code})."
            source.diagnostics["full_text_error"] = exc.code
            return
        except Exception as exc:                          # extraction bugs must not stop the other sources
            source.diagnostics["full_text_error"] = type(exc).__name__
            return
        source.diagnostics.update(read.diagnostics)
        if read.gate:
            source.note += f" The full-text address showed a {read.gate}."
            source.diagnostics["full_text_error"] = read.gate
        elif len(read.text) < MIN_FULL_TEXT_CHARS:
            source.diagnostics["full_text_error"] = "too little text"
        elif not self._claim_chars(len(read.text[:self.max_chars])):
            source.diagnostics["full_text_skipped"] = "text budget used up"
        else:
            self._upgrade(source, read.text, read.pages, read.final_url, read.content_type, "Open-access full text.")

    def _upgrade(self, source: Source, text: str, pages: Optional[List[str]], url: str, ctype: str, note: str) -> None:
        source.text, source.pages, source.access, source.read_url = text[:self.max_chars], pages, FULL_TEXT, url
        source.content_type, source.note, source.retrieval, source.retrieval_error = ctype or source.content_type, note, "retrieved", ""
        source.fingerprint = hashlib.sha256(source.text.encode("utf-8")).hexdigest()[:16]
        source.diagnostics["chars"] = len(source.text)

    def _classify(self, source: Source, hit: SearchHit, attempted: bool) -> None:
        """Deep mode: say exactly what was obtained. A record with no content is METADATA_ONLY when nothing could be
        attempted, FAILED when an attempt failed; snippets stay labelled as snippets."""
        if source.access == FULL_TEXT:
            source.retrieval = "retrieved"
        elif source.access == ABSTRACT_ONLY:
            source.retrieval = "abstract"
        elif source.access == SNIPPET_ONLY:
            source.retrieval = "snippet"
        elif source.retrieval != "skipped":
            if attempted and source.retrieval_error:
                source.retrieval = "failed"
            else:
                source.access, source.retrieval = METADATA_ONLY, "metadata"
                if not source.note:
                    source.note = "Only the record's metadata is available (no abstract or open-access copy was found)."
        else:
            source.access = METADATA_ONLY
        if source.usable:
            source.fingerprint = hashlib.sha256(source.text.encode("utf-8")).hexdigest()[:16]
            source.diagnostics["chars"] = len(source.text)

    # ------------------------------------------------------------------ all together
    def gather(self, queries: Sequence[str], log: Callable[[str], None] = lambda message: None,
               existing: Sequence[Source] = (), routes: Optional[Sequence[Route]] = None) -> Corpus:
        """Search, select, fetch and de-duplicate. ``existing`` (used by follow-up research) are sources already
        held: their works are not fetched again, new IDs continue after theirs, and a new source that copies one of
        them is linked to it as the same origin. ``routes`` (deep mode) send each query only to the providers chosen
        for it; without routes every query goes to every provider, as before."""
        requests_before = self.session.requests_made
        hits, errors = self.search(queries, routes)
        filtered = excluded = 0
        if self.classify:                                     # source policy: never fetched, never a source
            kept = [h for h in hits if is_evidence_provider(h.provider) and not excluded_host(h.url)]
            excluded, hits = len(hits) - len(kept), kept
        if self.since:
            kept = [h for h in hits if not (h.published and str(h.published)[:10] < self.since)]
            filtered, hits = len(hits) - len(kept), kept
        log(f"{len(hits)} search hits from {len(self.providers)} provider(s); {len(errors)} search error(s)")
        held = {s.key for s in existing}
        if self.classify:                                     # works already held do not take a place among the new
            hits = [h for h in hits if canonical_key(h.url, h.doi) not in held]
        selected = [(hit, q) for hit, q in self.select(hits) if canonical_key(hit.url, hit.doi) not in held]
        start = len(existing)
        self._defer = self.classify
        try:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                sources = list(pool.map(lambda pair: self.fetch(f"S{start + pair[0] + 1}", *pair[1]),
                                        enumerate(selected)))
        finally:
            self._defer = False
        if self._planned:
            self._download_by_priority()
        if self.classify:                                     # a page that redirected to an excluded host
            landed = [s for s in sources if excluded_host(s.read_url)]
            excluded += len(landed)
            sources = [s for s in sources if s not in landed]
        pool_sources = list(existing) + sources
        origins = near_duplicate_groups([s.text if s.usable else "" for s in pool_sources])
        for source, origin in zip(pool_sources[start:], origins[start:]):
            if pool_sources[origin].id != source.id:
                source.origin = pool_sources[origin].origin or pool_sources[origin].id
                source.note = (source.note + f" Largely duplicates {source.origin}; not counted as independent.").strip()
        sent = None
        if routes is not None:
            sent = {}
            for route in routes:
                sent[route.provider] = sent.get(route.provider, 0) + 1
        return Corpus(sources, hits, errors, self.session.requests_made - requests_before,
                      provider_status(self.providers, hits, queries, errors, sent), filtered,
                      len(routes) if routes is not None else len(queries) * len(self.providers), excluded)
