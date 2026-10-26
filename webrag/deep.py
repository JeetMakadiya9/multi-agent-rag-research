"""Bounded deep research: rounds of search and retrieval, coverage tracked per subquestion from the evidence actually
collected, follow-up and counter-evidence searches, an explicit stopping rule, an evidence-quality policy, citation
checks and a summary written only from verified findings.

It runs inside ``Orchestrator.run`` when a research depth is chosen (``quick``, ``standard`` or ``deep``) and reuses
the orchestrator's model client, verifier, researcher, evidence store and record formats. Verdicts come from the
configured verifier exactly as before: nothing here upgrades, overrides or recalculates one. The quality labels and
coverage statuses are descriptions of the evidence behind each verdict, defined in ``EVIDENCE_POLICY``.

Budgets (``BUDGETS``) bound every resource: rounds, planned queries per round, searches (one query sent to one
provider), unique sources, full-text attempts, characters of text processed and elapsed time. Search stops at the
first of: sufficient coverage, a budget limit, little new evidence, no new query, or failing providers; the stopping
reason and whether the result is complete, budget-limited, evidence-limited, access-limited or provider-limited are
recorded and reported.
"""
from __future__ import annotations

import copy
import itertools
import re
import time
import urllib.parse
from dataclasses import asdict, dataclass, field, replace
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

from docqa.llm import LLMError
from docqa.adjudicate import ADJUDICATED
from docqa.ranked import ADJUDICATED_UNSIDED, DECIDED_CONFLICTS, RANK_TIEBREAK
from docqa.verify import CONTRADICTED, INSUFFICIENT, SUPPORTED

from .agents import (ASSOCIATION, CAUSAL, HEDGE, NEGATION, PlannedQuery, SourceAwarePlanner, SourcePlan, Synthesizer,
                     attributions, qualifier_words, scope_words, summarize, write_queries)
from .evidence import Passage, cap_per_origin, merge_passages
from .researcher import (ABSTRACT_ONLY, FAILED, FULL_TEXT, METADATA_ONLY, SNIPPET_ONLY, OpenAccessResolver, Route,
                         Source, WebResearcher)
from .search import (CATEGORY_PROVIDERS, EXCLUSION_NOTE, PROVIDERS, excluded_host, is_evidence_provider, keywords,
                     provider_catalog)

# v2: passage boundaries and evidence-quality cautions (cautioned findings stay out of the answer), citation-based
# source dependence with stricter independence for scholarly works, and the stop record split into retrieval
# coverage, answer coverage and simultaneous limits (``retrieval_coverage_met`` replaces ``coverage_sufficient``).
# v3: query allocation that keeps counter-evidence and every subquestion (no first-N cut), round-robin routing, a
# query ledger with execution status, provider candidate limits, ranked source selection, full text by priority,
# gap-annotated follow-up, provider coverage, claim IDs, the number caution, the evidence review and several claims
# per verification run.
# v4 (Gate 1 repairs): corroboration counts only sources whose passage covers the finding; qualifier, scope and
# attribution checks (findings and summary); subquestion links inferred when the planner's tags carry no information,
# allocation by coverage; a relevance floor for reading; full-text attempts reported as attempts, successes and
# failures; passages starting with the previous sentence's full stop classified from after it.
# v5 (after the v4 live smoke test): a SUPPORTED statement no supporting passage covers gets support_unmatched and
# stays out of the direct answer and summary; scope phrases may contain punctuation; summary text is shortened only
# between sentences.
DEEP_VERSION = "deep-research-v5"


@dataclass(frozen=True)
class Budget:
    rounds: int
    queries_per_round: int          # planned query strings per round
    searches: int                   # total searches: one query sent to one provider
    searches_per_round: int
    sources: int                    # unique sources selected over the whole run
    full_text: int                  # full-text retrieval attempts (PDFs, open-access copies, full-text APIs)
    chars: int                      # characters of extracted text processed
    seconds: int                    # elapsed time before searching stops (verification still finishes)
    subquestions: int
    findings: int
    counter_evidence: int           # supported findings to challenge with a counter-evidence search (0 = off)


# Conservative for a 4B local model on ordinary hardware (verification takes about half a minute per statement).
BUDGETS: Dict[str, Budget] = {
    "quick": Budget(rounds=1, queries_per_round=4, searches=10, searches_per_round=10, sources=10, full_text=4,
                    chars=400_000, seconds=480, subquestions=3, findings=6, counter_evidence=0),
    "standard": Budget(rounds=2, queries_per_round=6, searches=24, searches_per_round=14, sources=16, full_text=8,
                       chars=700_000, seconds=900, subquestions=4, findings=8, counter_evidence=2),
    "deep": Budget(rounds=3, queries_per_round=8, searches=44, searches_per_round=18, sources=24, full_text=12,
                   chars=1_200_000, seconds=1800, subquestions=5, findings=8, counter_evidence=3),
}
DEPTHS = tuple(BUDGETS)
RECENCY = {"any": None, "year": 365, "5y": 5 * 365}

EVIDENCE_POLICY = {
    "FULL_TEXT": "Substantive text of the document or page. Can support findings; being full text does not make it "
                 "accurate or authoritative.",
    "ABSTRACT_ONLY": "A scholarly abstract without the paper. Can support what the abstract itself states; findings "
                     "resting only on abstracts are labelled 'abstract only' because details may differ in the paper.",
    "SNIPPET_ONLY": "A search-result fragment. Used to discover sources only; never indexed as evidence in deep research.",
    "METADATA_ONLY": "Title, authors, dates or identifiers only. Never evidence for a factual claim.",
    "FAILED": "Retrieval was attempted and failed; the reason is recorded. Not evidence.",
    "labels": {
        "corroborated": "Verified SUPPORTED with supporting passages from at least two publishers that their metadata "
                        "establishes as independent, at least one of them full text, and no passage judged to "
                        "contradict it. Since deep-research-v4 a source counts only when one sentence of its passage "
                        "contains at least 60% of the finding's content words (a lexical screen that withholds "
                        "corroboration from passages that only define a concept or touch the topic; it does not prove "
                        "entailment).",
        "independence unknown": "Verified SUPPORTED by more than one source, but they are not established as "
                                "independent (for example different sites without a recorded publisher, one source "
                                "citing another, or scholarly works whose reference lists could not be checked); not "
                                "called corroborated.",
        "single source": "Verified SUPPORTED, but all supporting passages come from one publisher, one work or copies "
                         "of one text.",
        "abstract only": "Verified SUPPORTED, but every supporting passage is from an abstract.",
        "contested": "The verifier found passages for and against (INSUFFICIENT, CONFLICTING); both sides are kept.",
        "adjudicated": "The verifier found passages for and against, and the adjudication step judged one side "
                       "decisive (sub-state ADJUDICATED); both sides are kept and the statement is left out of the "
                       "direct answer and summary.",
        "rank decided": "The verifier found passages for and against, and the rank tie-break took the first "
                        "passage with a stance as decisive (sub-state RANK_TIEBREAK; no extra model call, and a "
                        "first-ranked passage is not a more reliable one); both sides are kept and the statement is "
                        "left out of the direct answer and summary.",
        "adjudicated unsided": "No passage was judged for or against the statement, yet the adjudication step of "
                               "the historical arm_c or arm_c0 decided it (sub-state ADJUDICATED_UNSIDED); the "
                               "statement is left out of the direct answer and summary.",
        "contradicted": "The verifier judged the statement CONTRADICTED by the retrieved passages.",
        "not established": "The verifier found no decisive passage (INSUFFICIENT).",
        "unverified": "The verifier could not assess the statement (failed or incomplete output).",
    },
    "independence": "Judged from metadata and from the sources' own retrieved text, never assumed. Near-copies (8-word "
                    "shingle containment of 60% or more), one work found twice (a shared identifier) and two versions "
                    "of one work (the same full title) count once; pages on the same site, or with the same recorded "
                    "publisher name, count as one publisher; scholarly works sharing an author count as one team. A "
                    "source whose text contains another's DOI, arXiv ID, PMID, PMCID, URL or full title (five words or "
                    "more) depends on it and is not independent of it. Pages are independent only when they have "
                    "different sites and different recorded publisher names and neither cites the other. Scholarly "
                    "works are independent only when their author lists share no author, both were read in full text "
                    "with a detectable reference list, and neither cites the other; different authors alone are not "
                    "enough. Otherwise independence is unknown. This is a heuristic over the retrieved text: it misses "
                    "citations in text that was not retrieved, indirect citation chains and references it cannot "
                    "match, and it is not a citation-graph search.",
    "boundaries": "Where each passage behind a finding starts, read from the source's own page text: mid-word or "
                  "mid-sentence only when the text before it confirms it (or, at a page break, a hyphenated word whose "
                  "joined form occurs in the source); complete when a sentence starts there; otherwise uncertain. "
                  "Repeated running headers and footers are skipped. Chunk boundaries themselves are not changed.",
    "queries": "Every planned query is listed with its status: executed (at least one search ran, even with no "
               "results), failed (every search for it failed), skipped (a query or search budget left no room; the "
               "reason is recorded) or no_provider (no active provider serves its source type). When a round has "
               "more queries than places, the user's wording, a share for counter-evidence and each subquestion in "
               "turn come first. A counter-evidence query that did not run is never reported as sought.",
    "selection": "Search results are read in order of a ranking aid (keyword relevance to the queries, readable "
                 "content, provider rank, venue diversity), with the best result of each query first. Full-text "
                 "attempts go to the highest-ranked sources that have an open-access copy. The ranking never "
                 "affects a verdict.",
    "review": "A deterministic list of what limits each statement's support (cautions, abstract-only support, "
              "corroboration not established, contested or failed checks) and of planned work not done. It "
              "restates the recorded checks; it is not an independent verification.",
    "cautions": {
        "fragment_subject": "The finding's best-matching supporting sentence is the cut-off start of a passage "
                            "confirmed to begin mid-word or mid-sentence, so its subject is not in the retrieved text.",
        "hedge_dropped": "Every best-matching supporting sentence is hedged (may, might, could, suggests, appears, "
                         "is associated with ...), but the finding states it without a hedge.",
        "causal_added": "The finding states cause and effect (causes, leads to, because, due to ...) that no "
                        "best-matching supporting sentence states, for example where the source reports an association.",
        "number_unsupported": "The finding states a number that appears in none of its supporting passages (units, "
                              "rounding and recomputed values are not judged).",
        "support_unmatched": "Verified SUPPORTED, but no passage judged supporting contains at least 60% of the "
                             "statement's content words, so the retrieved text does not visibly cover the whole "
                             "statement. Kept out of the direct answer and summary; not a judgement that it is false.",
        "qualifier_dropped": "Every best-matching supporting sentence carries a qualifier (many, some, most, can ...) or "
                             "a scope (on two datasets, in patients ...) that the finding leaves out.",
        "attribution_dropped": "The supporting sentence reports another work's result (\"Xu et al. found\") and the "
                               "finding states it without saying who reported it.",
        "adjudicated_conflict": "Verified SUPPORTED only after the adjudication step chose between passages "
                                "judged for and against it; the opposing passages remain listed. Kept out of the "
                                "direct answer and summary.",
        "rank_decided_conflict": "Verified SUPPORTED only because the rank tie-break took the first-ranked "
                                 "passage with a stance as decisive over passages judged against it; the opposing "
                                 "passages remain listed. Kept out of the direct answer and summary.",
        "unsided_adjudication": "Verified SUPPORTED only by the adjudication step of the historical arm_c or arm_c0, although no "
                                "passage was judged to support it. Kept out of the direct answer and summary.",
        "_rule": "A caution is a warning from fixed text checks, not a verdict and not proof that the finding is "
                 "false or true. Cautioned findings stay listed with their warning and evidence, and are left out of "
                 "the direct answer and summary.",
        "_limits": "Word matching cannot tell whether a statement keeps its source's comparison (one graph model "
                   "against other graph models is not graph models against conventional methods) or generalises one "
                   "model or dataset to a whole class of methods; such changes can pass every check.",
    },
    "excluded": "Wikipedia is not accepted as an evidence source: its provider is never searched and pages on "
                "wikipedia.org are dropped before they are read, so they are never indexed, cited or listed.",
}
INDEXED_ACCESS = {FULL_TEXT, ABSTRACT_ONLY}


def is_evidence(source: Source) -> bool:
    """The gate to the evidence index: readable text (full text or an abstract) from a host the source policy accepts."""
    return source.usable and source.access in INDEXED_ACCESS and not excluded_host(source.read_url or source.url)


# ================================================================================================= independence
# Text distinctness (``Source.origin``: near-copies share one origin, used to avoid redundant passages) is not
# independence. Two sources count as independent only when their recorded metadata says so; otherwise their
# independence is "unknown" and they are never called corroborating.
SCHOLARLY_KINDS = {"scholarly", "preprint"}
WORK_IDS = ("doi", "arxiv", "pmid", "pmcid")
SECOND_LEVEL = {"co", "com", "org", "net", "ac", "gov", "edu", "or", "ne", "go", "gob", "mil", "nic", "sch"}


def site_of(host: str) -> str:
    """The site a host belongs to: its last two labels, or three under a two-letter country code with a common second
    level (``news.bbc.co.uk`` -> ``bbc.co.uk``). Without a public-suffix list this can merge different sites on a shared
    hosting domain (``a.github.io``, ``b.github.io``); that only ever counts fewer independent publishers, never more."""
    host = (host or "").lower().strip(".")
    if not host or ":" in host or host.replace(".", "").isdigit():
        return host
    labels = host.split(".")
    if len(labels) >= 3 and len(labels[-1]) == 2 and labels[-2] in SECOND_LEVEL:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def _field(source: Any, key: str) -> Any:
    return source.get(key) if isinstance(source, dict) else getattr(source, key, None)


def _person(name: str) -> str:
    """An author as surname and first initial: "Smith, Jane", "Jane Q. Smith" and "J. Smith" all give "smith j"."""
    name = re.sub(r"[^\w\s,.'-]", "", str(name or "")).strip()
    if "," in name:
        surname, _, given = name.partition(",")
    else:
        parts = name.replace(".", " ").split()
        surname, given = (parts[-1], " ".join(parts[:-1])) if parts else ("", "")
    surname = re.sub(r"[^\w'-]", "", surname).lower()
    return f"{surname} {re.sub(r'[^a-z]', '', given.lower())[:1]}".strip() if surname else ""


# A reference-list heading on a line of its own (English and a few other languages seen in scholarly PDFs).
REFERENCE_HEADING = re.compile(r"(?im)^[ \t]*(?:\d+\.?|[ivx]+\.)?[ \t]*(?:references|reference list|bibliography|"
                               r"works cited|literature cited|cited literature|daftar pustaka|referensi|literatur|"
                               r"literaturverzeichnis|r[ée]f[ée]rences|referencias|bibliograf[íi]a)[ \t]*:?[ \t]*$")
MIN_TITLE_WORDS = 5                                     # shorter titles are too generic to match in another text


def _words(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _flat(text: str) -> str:
    """Lower case, words broken by line-end hyphens joined, whitespace collapsed."""
    return re.sub(r"\s+", " ", re.sub(r"-[ \t]*\n\s*", "", text)).lower()


def identity(source: Any) -> Dict[str, Any]:
    """What a source's recorded metadata, and its own retrieved text, say about who published it and what it cites.
    A publisher name counts only when a provider or feed recorded one; the domain the page was read from is not a
    publisher name."""
    url = _field(source, "read_url") or _field(source, "url") or ""
    parts = urllib.parse.urlsplit(str(url))
    host = (parts.hostname or "").lower()
    domain = (_field(source, "domain") or (host[4:] if host.startswith("www.") else host)).lower()
    publisher = re.sub(r"\s+", " ", str(_field(source, "publisher") or "")).strip()
    named = publisher if publisher and publisher.lower() not in {domain, host} else ""
    ids = _field(source, "identifiers") or {}
    text = str(_field(source, "text") or "")
    title = _words(str(_field(source, "title") or ""))
    link = f"{host[4:] if host.startswith('www.') else host}{parts.path.rstrip('/')}" if len(parts.path.strip("/")) > 1 else ""
    return {"id": _field(source, "id"), "origin": _field(source, "origin") or _field(source, "id"),
            "work": (_field(source, "kind") or "") in SCHOLARLY_KINDS, "site": site_of(domain),
            "name": named.lower(), "authors": {p for p in (_person(a) for a in _field(source, "authors") or []) if p},
            "ids": {f"{k}:{str(v).lower()}" for k, v in ids.items() if k in WORK_IDS and v},
            "title": title if len(title.split()) >= MIN_TITLE_WORDS else "",
            "link": "" if "doi.org" in host else link.lower(),
            "full": _field(source, "access") == FULL_TEXT, "references": bool(REFERENCE_HEADING.search(text)),
            "flat": _flat(text), "text_words": f" {_words(_flat(text))} " if text else ""}


def _mentions(a: Dict[str, Any], b: Dict[str, Any]) -> str:
    """How ``a``'s retrieved text refers to ``b``: its DOI, arXiv ID, PMID or PMCID, its URL, or its full title
    (at least ``MIN_TITLE_WORDS`` words); "" when none of these occurs. Absence proves nothing."""
    text = a["flat"]
    if not text:
        return ""
    for entry in sorted(b["ids"]):
        kind, value = entry.split(":", 1)
        value = re.sub(r"^https?://(dx\.)?doi\.org/", "", value)
        if kind == "doi" and value in text:
            return f"its text contains the DOI of {b['id']} ({value})"
        if kind == "arxiv" and re.search(rf"\b{re.escape(re.sub(r'v\d+$', '', value))}(v\d+)?\b", text):
            return f"its text contains the arXiv ID of {b['id']} ({value})"
        if kind == "pmid" and re.search(rf"pmid:?\s*{re.escape(value)}\b", text):
            return f"its text contains the PMID of {b['id']} ({value})"
        if kind == "pmcid" and re.search(rf"\b{re.escape(value)}\b", text):
            return f"its text contains the PMCID of {b['id']} ({value})"
    if b["link"] and b["link"] in text:
        return f"its text links to {b['id']}"
    if b["title"] and f" {b['title']} " in a["text_words"]:
        return f"its text names the title of {b['id']}"
    return ""


def relation_reason(a: Dict[str, Any], b: Dict[str, Any]) -> Tuple[str, str]:
    """Whether two identities are the "same" publisher (or work, or text), "dependent" (one's text cites or links to
    the other), "distinct" publishers, or "unknown"; and why."""
    if a["origin"] == b["origin"]:
        return "same", "copies of one text"
    if a["ids"] & b["ids"]:
        return "same", "one work found twice (a shared identifier)"
    if a["work"] and b["work"]:                         # scholarly works: the research team, not the journal
        if a["title"] and a["title"] == b["title"]:
            return "same", "two versions of one work (the same full title)"
        if a["authors"] & b["authors"]:
            return "same", "the same research team (a shared author)"
    elif not a["work"] and not b["work"]:
        if a["site"] and a["site"] == b["site"]:
            return "same", "the same site"
        if a["name"] and a["name"] == b["name"]:
            return "same", "the same recorded publisher name"
    for x, y in ((a, b), (b, a)):
        cited = _mentions(x, y)
        if cited:
            return "dependent", f"{x['id']} depends on {y['id']}: {cited}"
    if a["work"] and b["work"]:
        if not a["authors"] or not b["authors"]:
            return "unknown", "an author list is missing"
        unchecked = [x["id"] for x in (a, b) if not (x["full"] and x["references"])]
        if unchecked:
            return "unknown", (f"different authors, but no full-text reference list to check for a citation in "
                               f"{', '.join(map(str, unchecked))}")
        return "distinct", "different authors; both reference lists were checked and neither cites the other"
    if a["work"] or b["work"]:
        return "unknown", "a page and a scholarly work: their relation is not in the metadata"
    if a["name"] and b["name"]:
        return "distinct", "different sites and recorded publisher names; neither text cites the other"
    return "unknown", "different sites, but no recorded publisher name for one of them"


def relation(a: Dict[str, Any], b: Dict[str, Any]) -> str:
    return relation_reason(a, b)[0]


def independence(sources: Sequence[Any], largest_set: bool = True) -> Dict[str, Any]:
    """How independent ``sources`` are: they are grouped into publishers (any "same" relation joins two groups), and
    ``independent`` is the largest number of groups that are pairwise "distinct". ``status`` is "established" (two or
    more), "unknown" (several groups, none provably apart), "single" (one group) or "none" (no source)."""
    seen, unique = set(), []
    for s in sources:
        if _field(s, "id") not in seen:
            seen.add(_field(s, "id"))
            unique.append(identity(s))
    parent = list(range(len(unique)))
    pairs = {(i, j): relation_reason(unique[i], unique[j]) for i in range(len(unique)) for j in range(i)}

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for (i, j), (kind, _) in pairs.items():
        if kind == "same":
            parent[root(i)] = root(j)
    groups: Dict[int, List[int]] = {}
    for i in range(len(unique)):
        groups.setdefault(root(i), []).append(i)
    members = list(groups.values())
    best = 1 if members else 0
    if largest_set and len(members) > 1:
        apart = lambda g, h: all(pairs[max(i, j), min(i, j)][0] == "distinct" for i in g for j in h)
        candidates = members[:10]                       # a statement has a handful of supporting sources
        for size in range(len(candidates), 1, -1):
            if any(all(apart(candidates[x], candidates[y]) for x, y in itertools.combinations(combo, 2))
                   for combo in itertools.combinations(range(len(candidates)), size)):
                best = size
                break
    status = "none" if not members else "single" if len(members) == 1 else "established" if best >= 2 else "unknown"
    return {"status": status, "independent": best, "groups": len(members), "sources": len(unique),
            "members": [[unique[i]["id"] for i in g] for g in members],
            "pairs": [{"a": unique[j]["id"], "b": unique[i]["id"], "relation": kind, "reason": reason}
                      for (i, j), (kind, reason) in sorted(pairs.items(), key=lambda x: (x[0][1], x[0][0]))]}


# ================================================================================================= passage boundaries
# Chunk boundaries come from rag.py (unchanged) and can fall inside a word or a sentence, most often at a PDF page
# break. This only describes where a passage starts, from the source's own page text; it never moves a boundary and
# never changes a verdict. Whenever that text cannot establish the boundary, it is "uncertain", not a fragment.
FRAGMENTS = {"mid_word", "mid_sentence"}
_EDGE_LINES = 3                                         # lines at the top and bottom of a page checked for repetition
_TERMINAL = re.compile(r"[.!?:;][\"'”’)\]]*$")
_SENTENCE_END = re.compile(r"[.!?][\"'”’)\]]*(?=\s|$)")


def _furniture_key(line: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\d+", "", line)).strip().lower()


def _repeated_lines(pages: Sequence[str]) -> set:
    """Lines near the top or bottom of two or more pages, compared without digits: running headers and footers."""
    counts: Dict[str, int] = {}
    for text in pages:
        lines = [line for line in text.splitlines() if line.strip()]
        for key in {_furniture_key(line) for line in lines[:_EDGE_LINES] + lines[-_EDGE_LINES:]}:
            counts[key] = counts.get(key, 0) + 1
    return {key for key, n in counts.items() if n >= 2 and len(key) >= 4}


def _is_furniture(line: str, repeated: set) -> bool:
    """A blank line, a bare page number or a line repeated at the edge of several pages."""
    line = line.strip()
    return not line or re.fullmatch(r"[\W\d_]*", line) is not None or _furniture_key(line) in repeated


def _lead(body: str) -> str:
    """The passage text up to its first sentence end: for a fragment, the words whose subject is cut off."""
    end = _SENTENCE_END.search(body)
    return re.sub(r"\s+", " ", body[:end.end()] if end else body).strip()


def passage_boundary(pages: Sequence[str], page: int, text: str) -> Dict[str, str]:
    """Where a passage starts in its source: "mid_word" or "mid_sentence" (confirmed by the source text before it),
    "complete" (a sentence starts there) or "uncertain". ``lead`` is the text up to the passage's first sentence end."""
    from .studio import locate_passage
    if not pages or not 1 <= page <= len(pages) or not text.strip():
        return {"boundary": "uncertain", "evidence": "no page text is recorded for this passage", "lead": ""}
    page_text = pages[page - 1]
    span = locate_passage(page_text, text)
    if span is None:
        return {"boundary": "uncertain", "evidence": "the passage was not found in its page text", "lead": ""}
    repeated = _repeated_lines(pages)
    start, end = span
    if start == 0 or page_text[start - 1] == "\n":       # skip a running header the passage begins with
        while True:
            line_end = page_text.find("\n", start)
            if line_end == -1 or line_end >= end or not _is_furniture(page_text[start:line_end], repeated):
                break
            start = line_end + 1
    while start < end and page_text[start].isspace():
        start += 1
    # Chunks often begin with the full stop that ended the previous sentence (". From an operational ..."): the
    # passage then starts at a sentence boundary, after that mark (seen in 18 of 30 judged passages of a live run).
    leading = re.match(r"[.!?;:]+[\"'”’)\]]*\s+", page_text[start:end])
    if leading:
        start += leading.end()
    body = page_text[start:end]
    result = lambda kind, evidence: {"boundary": kind, "evidence": evidence, "lead": _lead(body)}
    if not body.strip():
        return {"boundary": "uncertain", "evidence": "the passage holds only page furniture", "lead": ""}
    first, word = body[0], re.match(r"[A-Za-z]+", body)
    lower = first.isalpha() and first.islower()
    raw = page_text[:start]
    contiguous = bool(raw) and not raw[-1].isspace()     # no space or line break between the text and the passage
    lines = raw.split("\n")
    if lines.pop().strip():                             # text on the passage's own line: running text, kept
        before = raw
    else:                                               # the passage starts a line: drop furniture lines above it
        while lines and _is_furniture(lines[-1], repeated):
            lines.pop()
        before = "\n".join(lines)
    if before.strip():
        tail = before.rstrip()
        if contiguous and raw[-1].isalnum() and first.isalnum():
            joined = re.search(r"\w+$", raw).group() + (word.group() if word else first)
            return result("mid_word", f"the passage starts inside the word '{joined}'")
        if lower and not contiguous and re.search(r"[A-Za-z]-$", tail):
            return result("mid_word", "the line before the passage ends with a hyphenated word part")
        if _TERMINAL.search(tail):
            if lower:
                return result("uncertain", "the text before it ends a sentence, but the passage starts in lower case")
            return result("complete", "the text before it ends a sentence")
        if lower:
            return result("mid_sentence", "the text before it has no sentence end and the passage starts in lower case")
        return result("uncertain", "the text before it has no sentence end (it may be a heading)")
    if page == 1:
        return result("complete", "the passage starts the document")
    previous = [line for line in pages[page - 2].splitlines() if line.strip()]
    while previous and _is_furniture(previous[-1], repeated):
        previous.pop()
    if lower and word:
        # A word broken over the page break: a line near the end of the previous page ends in "stem-", and "stem" +
        # the passage's first word is a word that occurs elsewhere in the source. Lines after it (footnotes, a
        # received date, an identifier stamp) are taken to be page furniture.
        whole = re.sub(r"-[ \t]*\n\s*", "", "\n\x00\n".join(pages)).lower()   # never joined across a page break
        for line in reversed(previous[-6:]):
            stem = re.search(r"([A-Za-z]+)-\s*$", line)
            if stem:
                joined = (stem.group(1) + word.group()).lower()
                if re.search(rf"\b{re.escape(joined)}\b", whole):
                    return result("mid_word", f"page {page - 1} ends with '{stem.group(1)}-' and '{joined}' occurs in "
                                              f"the source, so the passage starts inside that word")
                break
        return result("uncertain", "the passage starts a page in lower case, but the previous page does not show "
                                   "where its sentence began")
    if previous and _TERMINAL.search(previous[-1].rstrip()):
        return result("complete", f"the last line of page {page - 1} ends a sentence")
    return result("uncertain", f"the end of page {page - 1} does not show whether a sentence continues")


_CONTENT = re.compile(r"[a-z][a-z0-9\-]{3,}")
_COMMON = {"that", "this", "with", "from", "have", "were", "which", "their", "than", "these", "those", "such", "also",
           "into", "more", "been", "they", "them", "when", "where", "while", "other", "some", "most", "many", "between"}


def _content_words(text: str) -> set:
    return set(_CONTENT.findall(text.lower())) - _COMMON


_ABBREVIATION = re.compile(r"(?:\bet al|\be\.g|\bi\.e|\bvs|\bFig|\bEq|\bNo|\bcf)\.$", re.I)


def _sentences(text: str) -> List[str]:
    """Sentences of a passage; a full stop after "et al.", "e.g.", "Fig." and the like does not end one (otherwise
    "Xu et al. (2024) demonstrated ..." loses its attribution)."""
    out: List[str] = []
    for piece in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text)):
        if out and _ABBREVIATION.search(out[-1]):
            out[-1] = f"{out[-1]} {piece}"
        else:
            out.append(piece)
    return [s.strip() for s in out if s.strip()]


def best_sentences(finding: str, passages: Sequence[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], str, bool]]:
    """For each supporting passage, the sentence that shares the most content words with the finding (at least two):
    (passage, sentence, whether that sentence is the cut-off lead of a confirmed fragment). Ties keep the lead."""
    words = _content_words(finding)
    out = []
    for p in passages:
        if p.get("stance") != "SUPPORTS":
            continue
        lead = p.get("lead", "") if p.get("boundary") in FRAGMENTS else ""
        units = [(lead, True)] if lead else []
        rest = p.get("text", "")
        if lead:
            position = re.sub(r"\s+", " ", rest).find(lead)
            rest = re.sub(r"\s+", " ", rest)[position + len(lead):] if position >= 0 else rest
        units += [(s, False) for s in _sentences(rest)]
        scored = [(len(words & _content_words(s)), s, is_lead) for s, is_lead in units]
        top = max((score for score, _, _ in scored), default=0)
        if top >= 2:
            best = [(s, is_lead) for score, s, is_lead in scored if score == top]
            sentence, is_lead = next(((s, l) for s, l in best if l), best[0])
            out.append((p, sentence, is_lead))
    return out


def fragment_caution(finding: str, passages: Sequence[Dict[str, Any]]) -> Optional[Dict[str, str]]:
    """A caution when the finding's best-matching evidence is the cut-off lead of a passage confirmed to start inside
    a word or sentence: its subject is not in the retrieved text. A caution is not a verdict."""
    matches = best_sentences(finding, passages)
    if not matches:
        return None
    top = max(len(_content_words(finding) & _content_words(s)) for _, s, _ in matches)
    leads = [(p, s) for p, s, is_lead in matches if is_lead and len(_content_words(finding) & _content_words(s)) == top]
    if not leads:
        return None
    p, sentence = leads[0]
    return {"kind": "fragment_subject",
            "detail": f"The best-matching supporting text ({p.get('key')}) begins {p['boundary'].replace('_', '-')} "
                      f"(\"{sentence[:120]}\"), so the subject it describes is not in the retrieved passage."}


def fidelity(finding: str, passages: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, str]], str]:
    """Hedge and cause checks of a finding against the best-matching sentence of each supporting passage. Returns the
    cautions and what was assessed. Negation on either side is not judged (no caution); a hedge is "dropped" only when
    every matched sentence is hedged (mixed strength gives no caution); a causal word must appear in a matched
    sentence, not elsewhere in the passage. Lexical cues only: not proof of meaning either way."""
    matches = [s for _, s, _ in best_sentences(finding, passages)]
    if not matches:
        return [], "not assessed: no supporting sentence shares enough words with the finding"
    if NEGATION.search(finding) or any(NEGATION.search(s) for s in matches):
        return [], "not assessed: negation in the finding or its supporting text"
    found = []
    hedged = [HEDGE.search(s) for s in matches]
    if all(hedged) and not HEDGE.search(finding):
        found.append({"kind": "hedge_dropped",
                      "detail": f"The supporting text is hedged ('{hedged[0].group()}': \"{matches[0][:120]}\"), but "
                                f"the finding states it without a hedge."})
    cause = CAUSAL.search(finding)
    if cause and not any(CAUSAL.search(s) for s in matches):
        association = any(ASSOCIATION.search(s) for s in matches)
        found.append({"kind": "causal_added",
                      "detail": f"The finding says '{cause.group()}', but its best-matching supporting text states no "
                                f"cause" + (" (it describes an association)" if association else "") + "."})
    # Qualifiers and scope present in every matched sentence but missing from the finding ("many fraud graphs" ->
    # "fraud graphs"); hedge words are left to hedge_dropped above. Attribution present in a matched sentence ("Xu et
    # al. demonstrated ...") but missing from the finding turns a reported result into the finding's own claim.
    shared = set.intersection(*[qualifier_words(s) - _HEDGE_WORDS for s in matches]) - qualifier_words(finding)
    scopes = set.intersection(*[scope_words(s) for s in matches]) - scope_words(finding)
    if shared or scopes:
        lost = sorted(shared) + [f"scope: {s}" for s in sorted(scopes)]
        found.append({"kind": "qualifier_dropped",
                      "detail": f"The supporting text says {', '.join(repr(x) for x in lost)} (\"{matches[0][:120]}\"), "
                                f"which the finding leaves out, so it may claim more than the source."})
    # Named works only ("Xu et al.", "Kipf and Welling (2017)"): "according to the operator" on a page is too common
    # in sources that report their own figures to treat as a secondary report.
    named = lambda text: attributions(text) - {"according to"}
    credited = set.union(*[named(s) for s in matches]) - named(finding)
    if credited and not named(finding):
        found.append({"kind": "attribution_dropped",
                      "detail": f"The supporting text reports this as another work's result ({', '.join(sorted(credited))}), "
                                f"but the finding states it without saying who reported it."})
    status = "assessed" + ("; mixed hedging among the supporting sentences" if any(hedged) and not all(hedged) else "")
    return found, status


_HEDGE_WORDS = {"may", "might", "could"}            # judged by hedge_dropped, not counted again as qualifiers
SUPPORT_COVERAGE = 0.6


def _stem(word: str) -> str:
    return word[:-1] if len(word) > 4 and word.endswith("s") and not word.endswith("ss") else word


def _terms(text: str) -> set:
    return {_stem(w) for w in _content_words(text)}


def support_check(finding: str, passages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """For each source with a passage the verifier judged SUPPORTS: the largest share of the finding's content words
    found in one sentence of those passages, and whether that reaches ``SUPPORT_COVERAGE``. Only sources that reach it
    count toward corroboration; a source that only defines a concept or discusses the topic does not. This is a
    lexical screen that can withhold corroboration, never grant it: high overlap is not proof of entailment, and the
    verifier's verdict is unchanged."""
    words = _terms(finding)
    best: Dict[str, Dict[str, Any]] = {}
    for p in passages:
        if p.get("stance") != "SUPPORTS" or not words:
            continue
        units = _sentences(p.get("text", "")) or [p.get("text", "")]
        coverage = max(len(words & _terms(s)) / len(words) for s in units)
        source = p.get("source_id") or str(p.get("key", "")).split(":", 1)[0]     # keys are "<source>:p<page>:<hash>"
        row = best.get(source)
        if row is None or coverage > row["coverage"]:
            best[source] = {"source": source, "passage": p.get("key", ""), "coverage": round(coverage, 2)}
    rows = sorted(best.values(), key=lambda r: (-r["coverage"], r["source"]))
    for row in rows:
        row["counted"] = row["coverage"] >= SUPPORT_COVERAGE
    return rows


def counted_sources(finding: str, passages: Sequence[Dict[str, Any]]) -> List[str]:
    return [r["source"] for r in support_check(finding, passages) if r["counted"]]


def secondary_reports(claim: Dict[str, Any], sources: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Supporting sentences that attribute the result to someone who is not among the source's authors ("Xu et al.
    (2024) demonstrated ..." in a paper by others): the source reports that work's result and the original was not
    retrieved. Judged from author surnames only; a missing author list leaves every attribution counted as secondary."""
    out = []
    for p, sentence, _ in best_sentences(claim.get("text", ""), claim.get("passages", [])):
        authors = {_person(a).split()[0] for a in sources.get(p["source_id"], {}).get("authors") or [] if _person(a)}
        named = sorted(n for n in attributions(sentence) - {"according to"} if n not in authors)
        if named:
            out.append({"source": p["source_id"], "passage": p.get("key", ""), "attributed_to": named})
    return out


def cautions(claim: Dict[str, Any]) -> List[Dict[str, str]]:
    """Evidence-quality cautions on a verified statement (``EVIDENCE_POLICY['cautions']``): warnings from fixed text
    checks, not verdicts and not proof of entailment. Only statements verified SUPPORTED are checked."""
    if claim.get("label") != SUPPORTED:
        return []
    passages = claim.get("passages", [])
    found = [support_caution(claim)] + [fragment_caution(claim["text"], passages)] + \
        fidelity(claim["text"], passages)[0] + [number_caution(claim["text"], passages)]
    if claim.get("sub_state") == ADJUDICATED:
        found.append({"kind": "adjudicated_conflict",
                      "detail": "Passages were judged both for and against this statement; the adjudication step "
                                "judged the supporting side decisive."})
    elif claim.get("sub_state") == ADJUDICATED_UNSIDED:
        found.append({"kind": "unsided_adjudication",
                      "detail": "No passage was judged to support this statement; the adjudication step judged it "
                                "supported from passages the verifier had judged neutral."})
    elif claim.get("sub_state") == RANK_TIEBREAK:
        found.append({"kind": "rank_decided_conflict",
                      "detail": "Passages were judged both for and against this statement; the rank tie-break "
                                "took the first-ranked passage with a stance, which supports it, as decisive."})
    return [c for c in found if c]


def support_caution(claim: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """``support_unmatched``: the verifier judged the statement SUPPORTED, but no passage it judged supporting
    contains ``SUPPORT_COVERAGE`` of the statement's content words (``support_check``), so nothing in the retrieved
    text visibly covers the whole statement. A conservative filter that keeps the statement out of the direct answer
    and summary; it is not a finding that the statement is false, and high overlap elsewhere is not proof of support.
    (A live run put "GNNs outperform conventional ML in accuracy ... PR-AUC" in the answer on passages covering 13-29%.)"""
    rows = claim["support_check"] if "support_check" in claim else support_check(claim["text"], claim.get("passages", []))
    if any(r.get("counted") for r in rows):
        return None
    best = max((r["coverage"] for r in rows), default=0.0)
    where = f" (best: {rows[0]['source']}, {round(100 * best)}%)" if rows else " (no supporting passage was recorded)"
    return {"kind": "support_unmatched",
            "detail": f"No passage judged to support this statement contains at least {round(100 * SUPPORT_COVERAGE)}% "
                      f"of its content words{where}, so the retrieved text does not visibly cover the whole statement."}


_DIGITS = re.compile(r"\d+(?:[.,]\d+)*")


def _numbers(text: str) -> set:
    """Numbers as written, with thousands separators removed ("1,200" and "1200" match; "0.5" stays "0.5")."""
    return {re.sub(r",(?=\d{3}\b)", "", n) for n in _DIGITS.findall(text)}


def number_caution(finding: str, passages: Sequence[Dict[str, Any]]) -> Optional[Dict[str, str]]:
    """A caution when the finding states a number that appears in none of its supporting passages: a figure the
    verifier may have accepted without the text showing it. Units, rounding and recomputation are not judged."""
    support = " ".join(p.get("text", "") for p in passages if p.get("stance") == "SUPPORTS")
    if not support:
        return None
    missing = sorted(_numbers(finding) - _numbers(support))
    if not missing:
        return None
    return {"kind": "number_unsupported",
            "detail": f"The finding states {', '.join(missing[:4])}, which does not appear in its supporting passages."}


def _pages_of(deep: "DeepResearch") -> Dict[str, List[str]]:
    return {s.id: list(s.pages) if s.pages else [s.text] for s in deep.sources if s.usable}


def evidence_gap(state: Dict[str, Any]) -> str:
    """Why a question's retrieved evidence is not yet strong (``DeepResearch.assess``), in words a query writer can
    act on."""
    if state.get("state") == "none":
        return "no relevant readable passage was found; search other wording or another source type"
    if not state.get("full_text"):
        return "only abstracts were found; look for full text, primary reports or official documents"
    if state.get("independence") == "single":
        return "all relevant passages come from one publisher or work; look for independent work"
    return "the sources' independence is not established; look for work by other authors or publishers"


def infer_subquestions(queries: Sequence[PlannedQuery], subquestions: Sequence[Dict[str, Any]],
                       question: str = "") -> Tuple[List[PlannedQuery], set]:
    """Query-to-subquestion links when the planner's own tags carry no information: there are two or more
    subquestions but every query other than the user's wording has the same tag (a live run tagged all six with the
    main question). Each such query is then linked to the subquestion whose own words (found in no other subquestion)
    it shares most, then by all shared words, ties going to the earlier subquestion; a query sharing no such word keeps
    its tag. Planner tags that do vary are kept as they are. A lexical link, not a judgement of what the query will
    find. Returns the queries and the query texts whose link was inferred."""
    subs = [(i, _terms(s.get("text", "") + " " + " ".join(s.get("terms") or []))) for i, s in enumerate(subquestions) if i > 0]
    others = [q for q in queries if q.query.lower() != question[:160].lower()]
    if len(subs) < 2 or len({q.subquestion for q in others}) != 1:
        return list(queries), set()
    unique = {i: w - set().union(*[x for j, x in subs if j != i]) for i, w in subs}     # words of this subquestion only
    out, inferred = [], set()
    for q in queries:
        words = _terms(q.query)
        if q in others:
            score, earlier = max(((len(words & unique[i]), len(words & w)), -i) for i, w in subs)
            index = -earlier
            if score[0] >= 1 and index != q.subquestion:
                out.append(replace(q, subquestion=index))
                inferred.add(q.query)
                continue
        out.append(q)
    return out, inferred


def allocate_queries(queries: Sequence[PlannedQuery], limit: int, question: str = "") -> Tuple[List[PlannedQuery], List[PlannedQuery]]:
    """Which planned queries a round runs when there are more than ``limit``, instead of the first N: the user's own
    wording first; then a reserved share for counter-evidence (a quarter of the places, at least one, when the round
    has three or more); then one answer query per subquestion, the one adding the most words not yet searched first
    (so when not every subquestion fits, the one most covered by queries already chosen waits); then the remaining
    answer queries, each subquestion in turn; then whatever still fits. Returns (run, omitted), in a stable order."""
    queries = list(queries)
    if len(queries) <= limit:
        return queries, []
    indexes = list(range(len(queries)))
    user = [i for i in indexes if question and queries[i].query.lower() == question[:160].lower()][:1]
    counter = [i for i in indexes if queries[i].purpose == "counter" and i not in user]
    answer = [i for i in indexes if i not in user and i not in counter]
    by_sub: Dict[int, List[int]] = {}
    for i in answer:
        by_sub.setdefault(queries[i].subquestion, []).append(i)
    covered = set().union(*[_terms(queries[i].query) for i in user])
    firsts, pending = [], [group[0] for group in by_sub.values()]
    while pending:
        pick = max(pending, key=lambda i: (len(_terms(queries[i].query) - covered), -i))
        pending.remove(pick)
        firsts.append(pick)
        covered |= _terms(queries[pick].query)
    rest = [i for layer in itertools.zip_longest(*[group[1:] for group in by_sub.values()]) for i in layer if i is not None]
    turns = firsts + rest
    reserve = min(len(counter), max(1, limit // 4)) if counter and limit >= 3 else 0
    chosen = list(user)[:limit]
    for i in turns:
        if len(chosen) >= limit - reserve:
            break
        chosen.append(i)
    for i in counter + turns:
        if len(chosen) >= limit:
            break
        if i not in chosen:
            chosen.append(i)
    return [queries[i] for i in chosen], [queries[i] for i in indexes if i not in chosen]


def _passage(record: Dict[str, Any]) -> Passage:
    return Passage(record["key"], record["source_id"], record["page"], record["text"], record.get("score", 0.0))


class DeepResearch:
    """One deep-research run. ``orch`` is the Orchestrator; ``run`` its Run (data, stage and note helpers)."""

    def __init__(self, orch, run, options: Dict[str, Any], providers: Sequence):
        self.orch, self.run, self.data = orch, run, run.data
        self.depth = options.get("depth") if options.get("depth") in BUDGETS else "standard"
        self.budget = BUDGETS[self.depth]
        days = RECENCY.get(options.get("recency") or "any")
        self.since = (date.today() - timedelta(days=days)).isoformat() if days else None
        self.preferred = [c for c in options.get("categories") or [] if c in CATEGORY_PROVIDERS]
        self.config = options.get("config")              # the studio's provider settings (None: defaults)
        # Source policy: an excluded provider (Wikipedia) is never searched, whoever passed it in.
        self.excluded_providers = [p.name for p in providers if not is_evidence_provider(p.name)]
        self.providers = [p for p in providers if is_evidence_provider(p.name)]
        self.names = [p.name for p in self.providers]
        self.categories = [c for c, ids in CATEGORY_PROVIDERS.items() if any(i in self.names for i in ids)]
        self.started = time.monotonic()
        self.used = {"rounds": 0, "queries": 0, "searches": 0, "sources": 0, "full_text": 0, "chars": 0}
        self.sources: List[Source] = []
        self.store = None
        self.rounds: List[Dict[str, Any]] = []
        self.previous_queries: List[str] = []
        self.seen_passages: set = set()
        self._inferred: set = set()                     # query texts whose subquestion link was inferred
        self.resolver = OpenAccessResolver(orch.session) if "openalex" in self.names else None

    # ---------------------------------------------------------------------------------------------- bookkeeping
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def _publish_budget(self) -> None:
        self.data["budget"] = {"depth": self.depth, "limits": asdict(self.budget), "used": dict(self.used),
                               "elapsed": round(self.elapsed(), 1)}

    def remaining(self, key: str) -> int:
        limit = {"rounds": self.budget.rounds, "searches": self.budget.searches, "sources": self.budget.sources,
                 "full_text": self.budget.full_text, "chars": self.budget.chars}[key]
        return max(0, limit - self.used[key])

    # ---------------------------------------------------------------------------------------------- routing
    def routes_for(self, queries: Sequence[PlannedQuery], limit: int) -> Tuple[List[Route], List[Route]]:
        """Each planned query goes to the active providers of its source type, up to ``limit`` searches: every query
        gets its first provider before any query gets a second, so a later query is never left without a search
        while an earlier one is sent everywhere. Returns the routes to run and the ones cut by the limit."""
        options = [[p for p in CATEGORY_PROVIDERS.get(q.category, ()) if p in self.names] for q in queries]
        routes: List[Route] = []
        cut: List[Route] = []
        for level in range(max((len(o) for o in options), default=0)):
            for q, providers in zip(queries, options):
                if level < len(providers):
                    route = Route(q.query, providers[level], q.category, q.purpose, q.subquestion)
                    (routes if len(routes) < limit else cut).append(route)
        return routes, cut

    def candidate_limits(self) -> Dict[str, int]:
        """Candidates requested per search, by provider (``ProviderInfo.candidates``); providers without a declared
        limit use the run's per-query setting."""
        return {name: PROVIDERS[name].candidates for name in self.names if name in PROVIDERS and PROVIDERS[name].candidates}

    # ---------------------------------------------------------------------------------------------- one round
    def search_round(self, number: int, queries: Sequence[PlannedQuery], purpose: str = "") -> Dict[str, Any]:
        """One round of searching and reading. A counter-evidence round (``purpose``) uses the search, source and text
        budgets but not a search round: those are counted by the counter-evidence allowance."""
        started = time.perf_counter()
        queries, omitted = allocate_queries(queries, self.budget.queries_per_round, self.data.get("input", ""))
        limit = min(self.budget.searches_per_round, self.remaining("searches"))
        routes, cut = self.routes_for(queries, limit)
        self.run.note("search", f"round {number}: {len(routes)} searches across {len({r.provider for r in routes})} provider(s)")
        researcher = WebResearcher(self.providers, self.orch.session, per_query=self.orch.settings.per_query,
                                   max_sources=self.remaining("sources"), max_full_text=self.remaining("full_text"),
                                   classify=True, resolver=self.resolver, max_total_chars=self.remaining("chars"),
                                   since=self.since, round_number=number, per_provider=self.candidate_limits())
        corpus = researcher.gather([r.query for r in routes], lambda m: self.run.note("search", f"round {number}: {m}"),
                                   existing=self.sources, routes=routes)
        self._log_queries(number, purpose or "search", queries, omitted, routes, cut, researcher.route_results,
                          researcher.query_picks)
        if not purpose:
            self.used["rounds"] = number
        self.used["queries"] += len(queries)
        self.used["searches"] += len(routes)
        self.used["sources"] += len(corpus.sources)
        self.used["full_text"] += researcher.full_text_attempts
        self.used["chars"] += researcher.chars_used
        self.previous_queries += [q.query for q in queries]
        self.sources += corpus.sources
        evidence = [s for s in corpus.sources if is_evidence(s)]
        self._index(evidence)
        self.data["sources"] = [self.orch._source_record(s) for s in self.sources]
        self.data["search_errors"] += [{"provider": p, "query": q, "error": e, "round": number} for p, q, e in corpus.search_errors]
        record = {
            "number": number, "queries": [{"query": q.query, "category": q.category, "purpose": q.purpose,
                                           "subquestion": q.subquestion,
                                           "providers": [r.provider for r in routes if r.query == q.query and r.category == q.category]}
                                          for q in queries],
            "omitted_queries": [{"query": q.query, "purpose": q.purpose, "subquestion": q.subquestion} for q in omitted],
            "searches_cut": len(cut), "candidates": researcher.candidates,
            "below_relevance_floor": researcher.below_floor, "full_text": researcher.full_text_outcome,
            "candidate_limits": researcher.per_provider,
            "searches": len(routes), "hits": len(corpus.hits), "new_sources": len(corpus.sources),
            "indexed_sources": len(evidence), "filtered_by_date": corpus.filtered_by_date,
            "excluded_by_policy": corpus.excluded_by_policy,
            "errors": len(corpus.search_errors), "provider_status": corpus.provider_status,
            "requests": corpus.requests_made, "seconds": round(time.perf_counter() - started, 2),
            "access": {level: sum(s.access == level for s in corpus.sources)
                       for level in (FULL_TEXT, ABSTRACT_ONLY, SNIPPET_ONLY, METADATA_ONLY, FAILED)},
        }
        if purpose:
            record["purpose"] = purpose
        self.rounds.append(record)
        self.data["rounds"] = self.rounds
        self.data["requests"] = self.data.get("requests", 0) + corpus.requests_made
        self._publish_budget()
        status = "ok" if evidence and not corpus.search_errors else ("partial" if evidence or corpus.sources else "failed")
        self.run.stage(f"round {number}: search and read", status, time.perf_counter() - started,
                       f"{len(routes)} searches, {len(corpus.hits)} hits, {len(corpus.sources)} new sources, "
                       f"{len(evidence)} with readable text" + (f", {corpus.filtered_by_date} outside the time range"
                                                                if corpus.filtered_by_date else "")
                       + (f", {corpus.excluded_by_policy} Wikipedia result(s) left out" if corpus.excluded_by_policy else ""))
        return record

    def _log_queries(self, number: int, kind: str, queries: Sequence[PlannedQuery], omitted: Sequence[PlannedQuery],
                     routes: Sequence[Route], cut: Sequence[Route], results: Sequence[Dict[str, Any]],
                     picks: Optional[Dict[str, Optional[str]]] = None) -> None:
        """The query ledger (``data['queries']``): every query planned for a round and what happened to it.
        ``executed``: at least one search for it ran (zero results is still executed); ``failed``: every search for
        it failed; ``skipped``: not run because of a budget, with the reason; ``no_provider``: no active provider
        serves its source type. A skipped or failed counter-evidence query is never reported as sought.
        ``allocated``: the query was given a place in the round; ``tagged_by``: whether its subquestion link came from
        the planner or was inferred (``infer_subquestions``); ``relevant_candidate``: for an executed query, whether
        any result cleared the relevance floor (False: none did, so nothing was read for this query in particular)."""
        ledger = self.data.setdefault("queries", [])
        ran: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for r in results:
            ran.setdefault((r["query"], r["category"]), []).append(r)
        limit = min(self.budget.searches_per_round, self.remaining("searches"))   # logged before the round is counted
        rows = []
        for q in queries:
            searches = sorted(ran.get((q.query, q.category), []), key=lambda r: r["provider"])
            not_sent = sorted(r.provider for r in cut if r.query == q.query and r.category == q.category)
            if searches:
                ok = [s for s in searches if not s["error"]]
                status, reason = ("executed", "") if ok else ("failed", "every search for it failed")
                if not_sent:
                    reason = (reason + "; " if reason else "") + f"not sent to {', '.join(not_sent)}: the round's search limit ({limit}) was reached"
            elif not_sent:
                status, reason = "skipped", f"the round's search limit ({limit}) was reached before it could run"
            else:
                status, reason = "no_provider", f"no active provider serves the '{q.category}' source type"
            rows.append((q, True, status, reason, searches, not_sent))
        for q in omitted:
            rows.append((q, False, "skipped", f"the {self.depth} depth runs {self.budget.queries_per_round} queries per "
                                              f"round; the user's question, a share for counter-evidence and one query per "
                                              f"subquestion (the least covered first) were given places first", [], []))
        picks = picks if picks is not None else {}
        for q, allocated, status, reason, searches, not_sent in rows:
            entry = {"id": f"Q{len(ledger) + 1}", "round": number, "round_kind": kind, "query": q.query,
                     "category": q.category, "purpose": q.purpose, "subquestion": q.subquestion,
                     "tagged_by": "inferred" if q.query in getattr(self, "_inferred", set()) else "planner", "allocated": allocated,
                     "status": status, "reason": reason, "not_sent_to": not_sent,
                     "searches": [{"provider": s["provider"], "requested": s["requested"], "returned": s["returned"],
                                   "error": s["error"]} for s in searches]}
            if status == "executed" and q.query in picks:
                entry["relevant_candidate"] = picks[q.query] is not None
                if picks[q.query] is None and any(s["returned"] for s in searches):
                    entry["reason"] = "no result cleared the relevance floor, so none was read for this query in particular"
            ledger.append(entry)

    def _index(self, sources: Sequence[Source]) -> None:
        if not sources:
            return
        if self.store is None:
            self.store = self.orch.store_factory(list(sources), self.orch.models)
        elif hasattr(self.store, "add_sources"):
            self.store.add_sources(list(sources))
        else:                                               # a store without incremental indexing: rebuild
            self.store = self.orch.store_factory([s for s in self.sources if is_evidence(s)], self.orch.models)

    # ---------------------------------------------------------------------------------------------- coverage
    def assess(self, subquestions: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
        """Evidence state per subquestion from the passages the index returns for it. A passage counts only if it
        shares a keyword with the subquestion (a guard against ranking noise); this guides searching only and never
        affects a verdict."""
        states, new = [], 0
        for index, sub in enumerate(subquestions):
            terms = {t.lower() for t in keywords(sub["text"] + " " + " ".join(sub.get("terms") or []), 12)}
            passages = self.store.rank(sub["text"], k=4, per_source=2) if self.store is not None else []
            relevant = [p for p in passages if terms & set(re.findall(r"[a-z0-9\-]+", p.text.lower()))]
            publishers = independence([self.store.sources[p.source_id] for p in relevant])
            full = any(self.store.sources[p.source_id].access == FULL_TEXT for p in relevant)
            fresh = [p.key for p in relevant if p.key not in self.seen_passages]
            new += len(fresh)
            self.seen_passages.update(p.key for p in relevant)
            # Strong needs publishers established as independent; distinct texts of unknown independence are weak.
            state = "strong" if publishers["status"] == "established" and full else ("weak" if relevant else "none")
            states.append({"index": index, "state": state, "passages": [p.key for p in relevant],
                           "publishers": publishers["independent"], "groups": publishers["groups"],
                           "independence": publishers["status"], "full_text": full})
        return states, new

    # ---------------------------------------------------------------------------------------------- the loop
    def gather(self, plan: SourcePlan, subquestions: List[Dict[str, Any]]) -> Dict[str, Any]:
        queries, self._inferred = infer_subquestions(list(plan.queries) + [q for q in plan.counter_queries], subquestions,
                                                     self.data.get("input", ""))
        number, states, stop = 1, [], None
        while True:
            record = self.search_round(number, queries)
            assess_started = time.perf_counter()
            states, new = self.assess(subquestions)
            record["coverage"] = [{"subquestion": s["index"], "state": s["state"], "publishers": s["publishers"],
                                   "independence": s["independence"]} for s in states]
            record["new_relevant_passages"] = new
            weak = [s for s in states if s["state"] != "strong"]
            self.run.stage(f"round {number}: coverage", "ok", time.perf_counter() - assess_started,
                           f"{len(states) - len(weak)} of {len(states)} questions with strong evidence; {new} new relevant passages")
            stop = self.stopping(number, states, new, record)
            if stop:
                break
            # Each open question goes to the query writer with the gap that keeps it open, so a follow-up query can
            # target it (full text or primary sources for abstract-only evidence, independent work for one publisher).
            gaps = {s["index"]: evidence_gap(s) for s in weak}
            record["gaps"] = [{"subquestion": i, "gap": g} for i, g in gaps.items()]
            items = [f"{subquestions[s['index']]['text']} (evidence gap: {gaps[s['index']]})" for s in weak]
            categories = self._categories_for(weak, subquestions)
            started = time.perf_counter()
            self.run.note("follow-up", f"writing follow-up queries for {len(items)} open question(s)")
            planned, fell_back = write_queries(self.orch.client, "follow-up", items, categories, self.previous_queries,
                                               self.budget.queries_per_round)
            # write_queries numbers items from 1; map back to the subquestion indexes
            queries = [replace(q, subquestion=weak[q.subquestion - 1]["index"] if 1 <= q.subquestion <= len(weak) else 0)
                       for q in planned]
            self.run.stage(f"round {number + 1}: follow-up queries", "partial" if fell_back else "ok",
                           time.perf_counter() - started, f"{len(queries)} queries" + (" (keyword fallback)" if fell_back else ""))
            if not queries:
                stop = {"reason": "no_new_queries", "detail": "No new query could be formed for the open questions."}
                break
            number += 1
        return {"states": states, "stop": stop}

    def _categories_for(self, weak: Sequence[Dict[str, Any]], subquestions: Sequence[Dict[str, Any]]) -> List[str]:
        """Prefer source types not yet used for the weak questions (and full-text-capable ones when only abstracts
        were found), then the rest."""
        used = {s.category for s in self.sources if s.usable}
        wanted = []
        for s in weak:
            for c in subquestions[s["index"]].get("categories") or self.categories:
                if c in self.categories and c not in wanted:
                    wanted.append(c)
        fresh = [c for c in wanted if c not in used] + [c for c in wanted if c in used]
        return fresh or list(self.categories)

    def stopping(self, number: int, states: Sequence[Dict[str, Any]], new: int, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """The first stopping condition that holds (in the order below) is the ``reason``; every other one that held
        at the same time is listed in ``also_reached``. ``retrieval_coverage`` describes the retrieved passages only:
        whether the questions are answered is judged later from verified statements (the ledger)."""
        held = []
        if all(s["state"] == "strong" for s in states):
            held.append({"reason": "retrieval_coverage_met",
                         "detail": "Retrieval check met: every question had keyword-matched passages from at least two "
                                   "publishers established as independent by their metadata, including full text. This "
                                   "describes the passages retrieved, not whether the questions were answered."})
        if self.elapsed() >= self.budget.seconds:
            held.append({"reason": "time_budget", "detail": f"The {self.depth} time budget ({self.budget.seconds} s) was used up."})
        if number >= self.budget.rounds:
            held.append({"reason": "round_limit", "detail": f"The {self.depth} depth allows {self.budget.rounds} search round(s)."})
        if self.remaining("searches") == 0:
            held.append({"reason": "search_budget", "detail": f"All {self.budget.searches} searches of the {self.depth} budget were used."})
        if self.remaining("sources") == 0:
            held.append({"reason": "source_budget", "detail": f"The limit of {self.budget.sources} sources was reached."})
        if record["searches"] and record["errors"] >= record["searches"]:
            held.append({"reason": "providers_failing", "detail": "Every search in this round failed."})
        if number > 1 and new == 0:
            held.append({"reason": "little_new_evidence", "detail": "The last round found no new relevant passage."})
        if not held:
            return None
        return dict(held[0], also_reached=[h["reason"] for h in held[1:]],
                    retrieval_coverage={"met": all(s["state"] == "strong" for s in states),
                                        "states": {str(s["index"]): s["state"] for s in states}})

    # ---------------------------------------------------------------------------------------------- verification
    def challenge(self, claims: List[Any], subquestions: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        """Search for counter-evidence on supported findings whose supporting publishers are not established as
        independent, then re-verify them over their old and new passages. The original verdict is kept in the record;
        contradictions are never removed."""
        record: Dict[str, Any] = {"targets": [], "queries": [], "changes": [], "skipped": ""}
        held = self.store.sources if self.store is not None else {}
        # corroboration counts only sources whose passage covers the finding (``support_check``), as in the labels
        targets = [i for i, c in enumerate(claims) if c.label == SUPPORTED and independence(
            [held[s] for s in counted_sources(c.text, c.passages) if s in held])["status"] != "established"][:self.budget.counter_evidence]
        if not self.budget.counter_evidence:
            record["skipped"] = f"counter-evidence search is not part of the {self.depth} depth"
            return record
        if not targets:
            record["skipped"] = "every supported finding already rests on publishers established as independent"
            return record
        if self.elapsed() >= self.budget.seconds or self.remaining("searches") == 0 or self.remaining("sources") == 0:
            record["skipped"] = "the search budget was used up before counter-evidence could be sought"
            return record
        record["targets"] = targets
        started = time.perf_counter()
        planned, fell_back = write_queries(self.orch.client, "counter", [claims[i].text for i in targets],
                                           self.categories, self.previous_queries, len(targets) + 1)
        if not planned:
            record["skipped"] = "no counter-evidence query could be formed"
            return record
        record["queries"] = [{"query": q.query, "category": q.category} for q in planned]
        record["fallback"] = fell_back
        before_sources = len(self.sources)
        self.search_round(len(self.rounds) + 1, planned, purpose="counter-evidence")
        new_ids = {s.id for s in self.sources[before_sources:] if is_evidence(s)}
        for index in targets:
            before = claims[index]
            old = [_passage(p) for p in before.passages]
            ranked = self.store.rank(before.text, k=self.orch.settings.claim_passages) if self.store is not None else []
            passages = merge_passages(ranked, old, limit=self.orch.settings.claim_passages + 2)
            after = self.orch._verify(before.text, "finding", passages, self.store, before.cited_sources)
            after.subquestion = getattr(before, "subquestion", 0)
            used_new = sorted({p.source_id for p in passages if p.source_id in new_ids})
            changed = after.label != before.label or after.sub_state != before.sub_state
            record["changes"].append({
                "claim_index": index, "text": before.text, "changed": changed, "new_sources_judged": used_new,
                "before": {"label": before.label, "sub_state": before.sub_state, "supporting_sources": before.supporting_sources,
                           "contradicting_sources": before.contradicting_sources},
                "after": {"label": after.label, "sub_state": after.sub_state, "supporting_sources": after.supporting_sources,
                          "contradicting_sources": after.contradicting_sources},
                "reason": (f"{before.label} -> {after.label}{f' ({after.sub_state})' if after.sub_state else ''} after "
                           f"re-verification with counter-evidence sources {used_new}") if changed and used_new else
                          ("unchanged" + (f"; counter-evidence sources {used_new} were judged" if used_new else
                                          "; no counter-evidence source reached the top passages"))})
            claims[index] = after
        self.run.stage("counter-evidence", "partial" if fell_back else "ok", time.perf_counter() - started,
                       f"{len(planned)} queries for {len(targets)} finding(s); "
                       f"{sum(c['changed'] for c in record['changes'])} verdict(s) changed")
        return record


# ================================================================================================= evidence policy
def strength(claim: Dict[str, Any], sources: Dict[str, Dict[str, Any]]) -> str:
    """The documented evidence-quality label of a verified statement (``EVIDENCE_POLICY['labels']``)."""
    if not claim.get("parse_ok", True) or claim.get("sub_state") in {"SERVICE_ERROR", "ASSESSMENT_FAILED"}:
        return "unverified"
    if claim.get("sub_state") == ADJUDICATED:
        return "adjudicated"
    if claim.get("sub_state") == RANK_TIEBREAK:
        return "rank decided"
    if claim.get("sub_state") == ADJUDICATED_UNSIDED:
        return "adjudicated unsided"
    label = claim.get("label")
    if label == CONTRADICTED:
        return "contradicted"
    if label == INSUFFICIENT:
        return "contested" if claim.get("sub_state") == "CONFLICTING" else "not established"
    supporting = [p for p in claim.get("passages", []) if p.get("stance") == "SUPPORTS"]
    if "support_check" in claim:                        # deep-research-v4: only passages that cover the finding count
        counted = {r["source"] for r in claim["support_check"] if r.get("counted")}
        supporting = [p for p in supporting if p["source_id"] in counted] or supporting
    access = {sources.get(p["source_id"], {}).get("access") for p in supporting}
    if supporting and access <= {ABSTRACT_ONLY}:
        return "abstract only"
    # Independence comes from the supporting sources' metadata (``independence``), not from how different their texts
    # are; a record without that assessment is never called corroborated.
    published = claim.get("independence") or {}
    if published.get("status") == "established" and FULL_TEXT in access and not claim.get("contradicting_sources"):
        return "corroborated"
    if published.get("groups", 0) >= 2:
        return "independence unknown"
    return "single source"


def check_citations(claim: Dict[str, Any], sources: Dict[str, Dict[str, Any]], texts: Dict[str, str],
                    pages: Dict[str, int]) -> Dict[str, Any]:
    """Every passage behind a verdict must belong to a recorded, readable source, sit on a page or time window that
    exists, and be text that occurs in that source's extract (whitespace and line-break hyphenation normalised)."""
    from .studio import locate_passage
    problems, unlocated = [], []
    for p in claim.get("passages", []):
        source = sources.get(p.get("source_id"))
        if source is None:
            problems.append(f"{p.get('key')}: unknown source {p.get('source_id')}")
            continue
        if source.get("access") not in INDEXED_ACCESS:
            problems.append(f"{p.get('key')}: source {p['source_id']} was not read ({source.get('access')})")
        if excluded_host(source.get("read_url") or source.get("url")):
            problems.append(f"{p.get('key')}: source {p['source_id']} is excluded by the source policy ({EXCLUSION_NOTE})")
        page_count = pages.get(p["source_id"], 1)
        if not 1 <= int(p.get("page") or 1) <= max(1, page_count):
            problems.append(f"{p.get('key')}: location {p.get('page')} does not exist in {p['source_id']}")
        text = texts.get(p["source_id"], "")
        if text and locate_passage(text, p.get("text", "")) is None:
            unlocated.append(p.get("key"))            # usually the indexer's own normalisation; reported, not fatal
    return {"checked": len(claim.get("passages", [])), "problems": problems, "unlocated": unlocated, "ok": not problems}


def ledger_entry(index: int, sub: Dict[str, Any], state: Dict[str, Any], claims: Sequence[Dict[str, Any]],
                 sources: Sequence[Dict[str, Any]], rounds: Sequence[Dict[str, Any]], categories: Sequence[str]) -> Dict[str, Any]:
    # The main question (index 0) is answered by every finding; a subquestion by the findings tagged with it.
    mine = list(claims) if index == 0 else [c for c in claims if c.get("subquestion", 0) == index]
    labels = [c["strength"] for c in mine]
    queries = [q["query"] for r in rounds for q in r["queries"] if q.get("subquestion") == index]
    discovered = [s["id"] for s in sources if set(s.get("queries") or []) & set(queries)]
    retrieved = [s["id"] for s in sources if s["id"] in discovered and s.get("access") in INDEXED_ACCESS]
    supported = [c["text"] for c in mine if c["label"] == SUPPORTED]
    against = [c["text"] for c in mine if c["strength"] in {"contested", "contradicted", "adjudicated", "rank decided"}]
    if "corroborated" in labels:
        status = "answered with disagreement" if against else "answered"
        reason = ("Corroborated by publishers established as independent" + ("; some statements are contested" if against else "")
                  + ".")
    elif supported:
        status = "partially answered"
        if "independence unknown" in labels:
            reason = ("Supported by more than one source, but their metadata does not establish that they are "
                      "independent, so this is not counted as corroboration.")
        elif set(labels) & {"single source", "abstract only"}:
            reason = "Supported only by " + ("abstracts" if set(labels) & {"abstract only"} and "single source" not in labels
                                             else "a single publisher") + "; independent corroboration was not found."
        else:
            reason = ("Supported only by statements whose sources disagreed and were "
                      + ("decided by rank" if "rank decided" in labels and "adjudicated" not in labels
                         else "adjudicated") + "; independent corroboration was not found.")
    elif against:
        status, reason = "contested", "The evidence found disagrees or contradicts the drafted statements."
    elif state.get("state") in {"strong", "weak"}:
        status, reason = "unanswered", "Relevant passages were found, but no statement drawn from them could be verified."
    else:
        status, reason = "not reached", "No relevant readable source was found for this question."
    used = {s.get("category") for s in sources if s["id"] in retrieved}
    missing = [c for c in (sub.get("categories") or categories) if c not in used and c in categories]
    next_action = "" if status == "answered" else (
        f"Search {', '.join(missing[:2])} sources" if missing else "Look for independent full-text sources" if supported
        else "Reformulate the question or add a source type that covers it")
    return {"index": index, "text": sub["text"], "categories": sub.get("categories") or [], "queries": queries,
            "discovered": discovered, "retrieved": retrieved, "passages": state.get("passages", []),
            "independent_publishers": state.get("publishers", 0), "independence": state.get("independence", "none"),
            "evidence_state": state.get("state", "none"),
            "supporting": supported, "contradicting": against,
            "uncertain": [c["text"] for c in mine if c["strength"] in {"not established", "unverified"}],
            "status": status, "reason": reason, "next_action": next_action}


# "coverage_sufficient" is the deep-research-v1 name of "retrieval_coverage_met", kept so v1 records still map.
COMPLETENESS = {"retrieval_coverage_met": "complete", "coverage_sufficient": "complete", "time_budget": "budget_limited",
                "round_limit": "budget_limited", "search_budget": "budget_limited", "source_budget": "budget_limited",
                "little_new_evidence": "evidence_limited", "no_new_queries": "evidence_limited",
                "providers_failing": "provider_limited"}
ANSWERED = {"answered", "answered with disagreement"}


def completeness(stop: Dict[str, str], sources: Sequence[Dict[str, Any]], ledger: Sequence[Dict[str, Any]]) -> str:
    kind = COMPLETENESS.get(stop.get("reason", ""), "partial")
    open_questions = [e for e in ledger if e["status"] not in ANSWERED]
    if kind != "complete" and open_questions and sources:
        unreadable = sum(s.get("access") in {FAILED, METADATA_ONLY, SNIPPET_ONLY} for s in sources)
        if unreadable * 2 >= len(sources):
            return "access_limited"
    if kind == "complete" and open_questions:
        return "partial"
    return kind


def stopping_record(stop: Optional[Dict[str, Any]], states: Sequence[Dict[str, Any]], sources: Sequence[Dict[str, Any]],
                    ledger: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Why searching stopped, kept apart from what was answered. ``reason`` triggered the stop and ``also_reached``
    lists the other conditions that held with it; ``retrieval_coverage`` describes the retrieved passages;
    ``answer_coverage`` counts the ledger's verified answers; ``completeness`` combines the stop with the open
    questions. A stop on retrieval coverage names every question still open after verification. The execution status
    (``run_status``) is separate and unchanged."""
    stop = dict(stop or {"reason": "round_limit", "detail": "The planned rounds finished."})
    stop.setdefault("also_reached", [])
    stop.setdefault("retrieval_coverage", {"met": bool(states) and all(s.get("state") == "strong" for s in states),
                                           "states": {str(s["index"]): s.get("state", "none") for s in states}})
    still_open = [e for e in ledger if e["status"] not in ANSWERED]
    stop["answer_coverage"] = {"questions": len(ledger), "answered": len(ledger) - len(still_open),
                               "open": [{"index": e["index"], "status": e["status"]} for e in still_open]}
    if stop["reason"] == "retrieval_coverage_met" and still_open:
        name = lambda e: "the main question" if e["index"] == 0 else "question %d" % e["index"]
        named = "; ".join(f"{name(e)} is {e['status']}" for e in still_open)
        stop["detail"] += f" After verification, {len(still_open)} question(s) remain open: {named}."
    stop["completeness"] = completeness(stop, sources, ledger)
    return stop


def run_deep(orch, run, text: str, options: Dict[str, Any], providers: Sequence) -> None:
    """Fill ``run.data`` with a deep-research result. Raises only for model errors, which the orchestrator reports."""
    data, mode = run.data, run.data["mode"]
    deep = DeepResearch(orch, run, options, providers)
    data.update(depth=deep.depth, deep_version=DEEP_VERSION, evidence_policy=EVIDENCE_POLICY, rounds=[], queries=[],
                options={"depth": deep.depth, "providers": deep.names, "categories": deep.preferred,
                         "recency": options.get("recency") or "any", "since": deep.since})
    deep._publish_budget()
    if not deep.providers:                              # after the source policy: nothing accepted is left to search
        data["status"], data["error"] = "FAILED", "No search provider is active; enable one in Integrations."
        if deep.excluded_providers:
            data["error"] += f" ({EXCLUSION_NOTE}, so it is not searched.)"
        data["stopping"] = {"reason": "providers_failing", "detail": data["error"], "completeness": "provider_limited"}
        return

    # 1. plan
    started = time.perf_counter()
    run.note("plan", "planning subquestions, source types and queries")
    plan = SourceAwarePlanner(orch.client, deep.budget.subquestions, deep.budget.queries_per_round).plan(
        text, mode, deep.categories, deep.preferred)
    subquestions = [{"text": text, "terms": [], "categories": deep.categories}]
    if mode == "research":
        subquestions += plan.subquestions if not plan.fallback else []
    data["plan"] = {"queries": [q.query for q in plan.queries], "subquestions": [s["text"] for s in plan.subquestions],
                    "fallback": plan.fallback}
    data["research_plan"] = {**asdict(plan), "searched_categories": deep.categories}
    if plan.note:
        data["limitations"].append(plan.note)
    run.stage("plan", "partial" if plan.fallback else "ok", time.perf_counter() - started,
              f"{len(subquestions) - 1} subquestions, {len(plan.queries)} queries, {len(plan.counter_queries)} counter-evidence queries "
              f"over {', '.join(deep.categories)}")

    # 2. search rounds
    outcome = deep.gather(plan, subquestions)
    stop, states = outcome["stop"], outcome["states"]
    if deep.store is None:
        # Nothing readable was indexed. That is an execution failure only when no search could be carried out at all;
        # searches that ran and found nothing readable are a completed run with an insufficient-evidence outcome.
        unreachable = all(r["errors"] >= r["searches"] > 0 for r in deep.rounds)
        data["outcome"] = "INSUFFICIENT_EVIDENCE" if mode == "research" else INSUFFICIENT
        found = len(deep.sources)
        data["error"] = ("No search provider could be reached." if unreachable
                         else "The searches returned no results for this input." if not found
                         else f"No readable source (full text or abstract) was found for this input: {found} source(s) "
                              f"were found, but none could be read as full text or an abstract.")
        _finish(deep, data, [], subquestions, states, stop, plan, failed=unreachable, nothing_readable=True)
        return

    # 3. evidence pool, drafting and verification
    if mode == "verify":
        _verify_claim(deep, data, text, subquestions, states, stop, plan)
        return
    started = time.perf_counter()
    groups = [deep.store.rank(text, k=orch.settings.question_passages)] + \
             [deep.store.rank(s["text"], k=3) for s in subquestions[1:]]
    pool = cap_per_origin(merge_passages(*groups, limit=100), deep.store.sources,
                          k=min(16, orch.settings.question_passages + 2 * len(subquestions)), per_source=2)
    data["evidence_pool"] = [orch._passage_record(p, deep.store) for p in pool]
    run.stage("evidence pool", "ok" if pool else "failed", time.perf_counter() - started,
              f"{len(pool)} passages from {len({p.source_id for p in pool})} sources")
    if not pool:
        data["outcome"] = "INSUFFICIENT_EVIDENCE"
        _finish(deep, data, [], subquestions, states, stop, plan)
        return
    started = time.perf_counter()
    run.note("synthesize", "drafting cited findings per question")
    # Passages confirmed to start inside a word or sentence are marked for the model (see passage_boundary).
    pages_of = _pages_of(deep)
    notes = {}
    for p in pool:
        kind = passage_boundary(pages_of.get(p.source_id, []), p.page, p.text)["boundary"]
        if kind in FRAGMENTS:
            notes[p.key] = f"begins {kind.replace('_', '-')}: the words before it are cut off"
    synthesis = Synthesizer(orch.client, version=3, max_findings=deep.budget.findings).synthesize(
        text, [s["text"] for s in subquestions[1:]], pool, deep.store.sources, notes=notes)
    if not synthesis.parse_ok:
        run.stage("synthesize", "failed", time.perf_counter() - started, synthesis.error)
        data["error"] = "The local model's synthesis could not be parsed; no findings are reported."
        data["outcome"] = "INSUFFICIENT_EVIDENCE"
        _finish(deep, data, [], subquestions, states, stop, plan, failed=True)      # a required step produced nothing usable
        return
    run.stage("synthesize", "partial" if synthesis.truncated else "ok", time.perf_counter() - started,
              f"{len(synthesis.findings)} findings, {len(synthesis.disagreements)} disagreements")
    unknown = sorted({i for s in synthesis.findings + synthesis.disagreements for i in s.unknown_ids})
    if unknown:
        data["limitations"].append(f"The model cited passage IDs that were not supplied ({', '.join(unknown)}); those citations were discarded.")
    started = time.perf_counter()
    claims = []
    for number, finding in enumerate(synthesis.findings, 1):
        run.note("verify", f"checking finding {number} of {len(synthesis.findings)}")
        extra = deep.store.rank(finding.text, k=orch.settings.claim_passages)
        passages = merge_passages(finding.passages, extra, limit=orch.settings.claim_passages)
        result = orch._verify(finding.text, "finding", passages, deep.store, cited=[p.source_id for p in finding.passages])
        result.subquestion = finding.subquestion if finding.subquestion >= 0 else 0
        claims.append(result)
    failures = sum(not c.parse_ok for c in claims)
    run.stage("verify", "partial" if failures else "ok", time.perf_counter() - started,
              f"{sum(c.label == SUPPORTED for c in claims)} supported, {sum(c.label == CONTRADICTED for c in claims)} "
              f"contradicted, {sum(c.label == INSUFFICIENT for c in claims)} not established"
              + (f", {failures} checks failed" if failures else ""))
    data["counter_evidence"] = deep.challenge(claims, subquestions)
    data["disagreements"] = [{"text": d.text, "sources": list(dict.fromkeys(p.source_id for p in d.passages)), "verified": False}
                             for d in synthesis.disagreements if len(d.passages) >= 2]
    data["gaps"] = synthesis.gaps
    if synthesis.insufficient:
        data["limitations"].append("The synthesizer reported that the evidence does not fully answer the question.")
    _finish(deep, data, claims, subquestions, deep.assess(subquestions)[0], stop, plan, question=text)


MAX_CLAIMS = 8


def split_claims(text: str) -> List[str]:
    """The claims of a verification request: one per non-empty line (list markers removed), at most ``MAX_CLAIMS``.
    A single line is a single claim, exactly as before."""
    lines = [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", line).strip() for line in text.splitlines()]
    claims = list(dict.fromkeys(line for line in lines if line))
    return claims[:MAX_CLAIMS] or [text.strip()]


def _verify_claim(deep: DeepResearch, data: Dict[str, Any], claim: str, subquestions, states, stop, plan) -> None:
    """Claims mode: each claim is judged by the configured verifier against the passages retrieved for it; the
    verdict rules are the verifier's own. Several claims (one per line) share the run's searches and sources."""
    orch, run = deep.orch, deep.run
    started = time.perf_counter()
    texts = split_claims(claim)
    claims, pool = [], []
    for number, text in enumerate(texts, 1):
        passages = deep.store.rank(text, k=orch.settings.question_passages)
        pool = merge_passages(pool, passages, limit=200)
        run.note("verify", f"judging each passage against claim {number} of {len(texts)}")
        result = orch._verify(text, "input claim", passages, deep.store)
        result.subquestion = 0
        claims.append(result)
    data["evidence_pool"] = [orch._passage_record(p, deep.store) for p in pool]
    failures = sum(not c.parse_ok for c in claims)
    run.stage("verify", "partial" if failures else "ok", time.perf_counter() - started,
              "; ".join(f"{c.label}" + (f" ({c.sub_state})" if c.sub_state else "") for c in claims))
    data["counter_evidence"] = deep.challenge(claims, subquestions)
    if len(claims) > 1:
        data["claims_mode"] = {"claims": len(claims), "limit": MAX_CLAIMS, "dropped": max(0, len(
            [l for l in claim.splitlines() if l.strip()]) - MAX_CLAIMS)}
    # A run-level outcome only for a single claim; several claims each keep their own verdict.
    data["outcome"] = claims[0].label if len(claims) == 1 else "CLAIMS_ASSESSED"
    for c in claims:
        if c.sub_state == "CONFLICTING":
            data["disagreements"].append({"text": "Retrieved sources disagree about this claim." if len(claims) == 1
                                          else f"Retrieved sources disagree about: {c.text}",
                                          "sources": c.supporting_sources + c.contradicting_sources, "verified": True})
        elif c.sub_state in DECIDED_CONFLICTS:
            how = DECIDED_CONFLICTS[c.sub_state]
            data["disagreements"].append({"text": (f"Retrieved sources disagreed about this claim; {how} as "
                                                   f"{c.label}." if len(claims) == 1 else
                                                   f"Retrieved sources disagreed about: {c.text} ({how} as "
                                                   f"{c.label})"),
                                          "sources": c.supporting_sources + c.contradicting_sources, "verified": True})
    _finish(deep, data, claims, subquestions, deep.assess(subquestions)[0], stop, plan)


def _finish(deep: DeepResearch, data: Dict[str, Any], claims: List[Any], subquestions, states, stop, plan,
            question: str = "", failed: bool = False, nothing_readable: bool = False) -> None:
    """Record the claims, ledger, quality and summary. ``failed``: a required step could not be carried out (no
    search could be made, or the model's synthesis was unusable); only then is the run FAILED. ``nothing_readable``:
    the searches ran but no source could be read (see ``run_status``)."""
    orch, run = deep.orch, deep.run
    started = time.perf_counter()
    records = [asdict(c) | {"subquestion": getattr(c, "subquestion", 0)} for c in claims]
    by_id = {s["id"]: s for s in data["sources"]}
    texts = {s.id: s.text for s in deep.sources if s.usable}
    pages = {s.id: len(s.pages) if s.pages else 1 for s in deep.sources}
    pages_of = _pages_of(deep)
    for number, record in enumerate(records, 1):
        record["id"] = f"C{number}"                     # stable within the run; the review and report refer to it
        # independent_support (distinct texts, as in the original pipeline) is kept as recorded; the evidence label
        # uses the supporting publishers' independence instead
        # deep-research-v4: a supporting source counts toward corroboration only when one of its passages covers the
        # finding's content (``support_check``); the verifier's supporting_sources are kept as recorded
        record["support_check"] = support_check(record["text"], record["passages"])
        record["corroborating_sources"] = [r["source"] for r in record["support_check"] if r["counted"]]
        record["independence"] = independence([by_id[s] for s in record["corroborating_sources"] if s in by_id])
        record["strength"] = strength(record, by_id)
        record["secondary_reports"] = secondary_reports(record, by_id) if record["label"] == SUPPORTED else []
        record["citations"] = check_citations(record, by_id, texts, pages)
        for p in record["passages"]:
            p.update(passage_boundary(pages_of.get(p["source_id"], []), int(p.get("page") or 1), p.get("text", "")))
        record["cautions"] = cautions(record)
        if record["label"] == SUPPORTED:
            record["fidelity_check"] = fidelity(record["text"], record["passages"])[1]
    data["claims"] = records
    if data["mode"] == "research" and records:
        data["outcome"] = "ANSWERED" if any(c["label"] == SUPPORTED for c in records) else "INSUFFICIENT_EVIDENCE"
    for c in records:
        if c["label"] == INSUFFICIENT and c.get("sub_state") == "CONFLICTING" and data["mode"] == "research":
            data["disagreements"].append({"text": f"Sources disagree on: {c['text']}",
                                          "sources": c["supporting_sources"] + c["contradicting_sources"], "verified": True})
        elif c.get("sub_state") in DECIDED_CONFLICTS and data["mode"] == "research":
            data["disagreements"].append({"text": f"Sources disagreed on: {c['text']} "
                                                  f"({DECIDED_CONFLICTS[c['sub_state']]} as {c['label']})",
                                          "sources": c["supporting_sources"] + c["contradicting_sources"], "verified": True})
    broken = [c for c in records if not c["citations"]["ok"]]
    if broken:
        data["limitations"].append(f"{len(broken)} statement(s) cite passages that failed the citation check; they are "
                                   f"flagged and left out of the summary.")
    state_by_index = {s["index"]: s for s in states}
    data["ledger"] = [ledger_entry(i, sub, state_by_index.get(i, {}), records, data["sources"], deep.rounds, deep.categories)
                      for i, sub in enumerate(subquestions)]
    if data["mode"] == "verify" and len(records) > 1 and data["ledger"]:
        # Several claims: the request is answered only when every claim has a decisive verdict, not when one does.
        decisive = [c for c in records if c["label"] in {SUPPORTED, CONTRADICTED}]
        if len(decisive) < len(records):
            entry = data["ledger"][0]
            entry["status"] = ("partially answered" if decisive else
                               "contested" if any(c.get("sub_state") == "CONFLICTING" for c in records) else "unanswered")
            entry["reason"] = (f"{len(decisive)} of {len(records)} claims have a decisive verdict (supported or "
                               f"contradicted); the others are not established by the retrieved passages.")
    data["stopping"] = stopping_record(stop, states, data["sources"], data["ledger"])
    data["source_quality"] = source_quality(data["sources"], records)
    data["provider_coverage"] = provider_coverage(deep)
    _limitations(deep, data)
    data["review"] = review(data)
    cautioned = [c for c in records if c["label"] == SUPPORTED and c["cautions"]]
    if cautioned:
        data["limitations"].append(f"{len(cautioned)} verified statement(s) carry an evidence-quality caution; they are "
                                   f"listed with the caution and left out of the direct answer and summary.")
    if data["mode"] == "research":
        verified = [c for c in records if c["label"] == SUPPORTED and c["citations"]["ok"] and not c["cautions"]]
        run.note("summary", "writing the direct answer from verified findings")
        try:
            summary = summarize(orch.client, question or data["input"], [c["text"] for c in verified])
        except LLMError as exc:                         # the findings stand without a summary
            summary = {"available": False, "reason": f"The summary could not be written ({exc})."}
        index_of = {f"F{i}": records.index(c) for i, c in enumerate(verified, 1)}
        for part in [summary.get("direct_answer")] + list(summary.get("summary") or []):
            if part:
                part["claims"] = [index_of[f] for f in part.pop("findings") if f in index_of]
        data["summary"] = summary
        run.stage("summary", "ok" if summary.get("available") else "partial", time.perf_counter() - started,
                  "direct answer and summary from verified findings" if summary.get("available") else summary.get("reason", ""))
    elif len(records) > 1:
        counts: Dict[str, int] = {}
        for c in records:
            counts[c["label"]] = counts.get(c["label"], 0) + 1
        data["summary"] = {"available": True, "verdicts": counts, "claims": len(records)}
    else:
        label = records[0]["label"] if records else INSUFFICIENT
        data["summary"] = {"available": bool(records), "verdict": label, "strength": records[0]["strength"] if records else ""}
    deep._publish_budget()
    data["status"] = run_status(data, failed, nothing_readable)


# Failure codes that mean the request was refused here by the network-safety policy (webrag.net), not attempted.
POLICY_CODES = frozenset({"blocked_host", "proxy_policy"})
PROVIDER_STATES = {
    "searched": "searched in this run", "failed": "every search sent to it failed",
    "blocked_by_policy": "not contacted: the network-safety policy refused its address (a local decision, not a "
                         "provider failure)", "not_used": "active, but no query "
    "in this run was for its source types", "not_selected": "available, but not chosen for this run",
    "not_configured": "switched on, but its settings are missing", "disabled": "switched off",
    "excluded": "not accepted as an evidence source",
}


def provider_coverage(deep: "DeepResearch") -> List[Dict[str, Any]]:
    """Every implemented provider with what happened to it in this run, from the recorded searches: a provider is
    "searched" only when searches were sent to it, and an unconfigured or disabled one is never shown as searched."""
    sent: Dict[str, Dict[str, int]] = {}
    policy: Dict[str, set] = {}                         # failure codes per provider (FetchError / SearchError codes)
    for r in deep.rounds:
        for name, status in (r.get("provider_status") or {}).items():
            row = sent.setdefault(name, {"queries": 0, "failed": 0, "hits": 0})
            for key in row:
                row[key] += int(status.get(key) or 0)
            codes = policy.setdefault(name, set())
            codes.update(status.get("failures") or {})
    rows = []
    for item in provider_catalog(deep.config):
        pid, counts = item["id"], sent.get(item["id"], {"queries": 0, "failed": 0, "hits": 0})
        if counts["queries"]:
            state = "failed" if counts["failed"] >= counts["queries"] else "searched"
            if state == "failed" and policy.get(pid) and policy[pid] <= POLICY_CODES:
                state = "blocked_by_policy"
        elif pid in deep.names:
            state = "not_used"
        elif not item["evidence"]:
            state = "excluded"
        elif not item["enabled"]:
            state = "disabled"
        elif not item["configured"]:
            state = "not_configured"
        else:
            state = "not_selected"
        rows.append({"id": pid, "name": item["name"], "categories": item["categories"], "state": state,
                     "detail": PROVIDER_STATES[state], "missing": item["missing"] if state == "not_configured" else [],
                     **counts})
    known = {r["id"] for r in rows}
    for name, counts in sent.items():                   # a provider outside the registry (custom or test)
        if name not in known and counts["queries"]:
            state = "failed" if counts["failed"] >= counts["queries"] else "searched"
            rows.append({"id": name, "name": name, "categories": [], "state": state, "detail": PROVIDER_STATES[state],
                         "missing": [], **counts})
    return rows


REVIEW_ACTIONS = {"excluded": "left out of the direct answer and summary", "needs_evidence": "needs further evidence",
                  "unresolved": "left unresolved", "not_reported": "not reported as a finding", "note": "noted"}


def review(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """A deterministic evidence review of the finished run (no model is asked): for each statement, what limits its
    support, and for the run, what was planned but not done. Each item names the statement (``claim``), the passages
    or queries it rests on (``refs``) and what was done about it (``action``). It does not change a verdict, and it is
    not an independent check of the evidence: it restates what the recorded checks found."""
    items: List[Dict[str, Any]] = []

    def add(claim, kind, detail, action, refs=()):
        items.append({"claim": claim, "kind": kind, "detail": detail, "action": action, "refs": list(refs)})

    for c in data.get("claims") or []:
        cid, supporting = c.get("id", ""), [p["key"] for p in c.get("passages", []) if p.get("stance") == "SUPPORTS"]
        if not c.get("parse_ok", True) or c.get("sub_state") in {"SERVICE_ERROR", "ASSESSMENT_FAILED"}:
            add(cid, "verification_failed", "The verifier could not assess this statement; it is not counted as verified.",
                "not_reported")
            continue
        if c.get("sub_state") == ADJUDICATED and c.get("label") == CONTRADICTED:
            # (an adjudicated SUPPORTED statement is reviewed through its adjudicated_conflict caution below)
            add(cid, "adjudicated", "Passages were judged both for and against this statement; the adjudication step "
                                    "judged it contradicted. Both sides are kept.", "not_reported",
                [p["key"] for p in c.get("passages", []) if p.get("stance") in {"SUPPORTS", "CONTRADICTS"}])
        if c.get("sub_state") == RANK_TIEBREAK and c.get("label") == CONTRADICTED:
            # (a rank-decided SUPPORTED statement is reviewed through its rank_decided_conflict caution below)
            add(cid, "rank_decided", "Passages were judged both for and against this statement; the rank "
                                     "tie-break took the first-ranked passage with a stance, which contradicts it, as "
                                     "decisive. Both sides are kept.", "not_reported",
                [p["key"] for p in c.get("passages", []) if p.get("stance") in {"SUPPORTS", "CONTRADICTS"}])
        if c.get("sub_state") == ADJUDICATED_UNSIDED and c.get("label") == CONTRADICTED:
            add(cid, "adjudicated_unsided", "No passage was judged against this statement; the adjudication step "
                                            "judged it contradicted from passages the verifier had judged neutral.",
                "not_reported", [p["key"] for p in c.get("passages", [])])
        if c.get("label") == CONTRADICTED:
            add(cid, "contradicted", "The retrieved passages were judged to contradict this statement.", "not_reported",
                [p["key"] for p in c.get("passages", []) if p.get("stance") == "CONTRADICTS"])
        elif c.get("label") == INSUFFICIENT:
            if c.get("sub_state") == "CONFLICTING":
                add(cid, "contested", "Passages were judged both for and against this statement.", "unresolved",
                    [p["key"] for p in c.get("passages", []) if p.get("stance") in {"SUPPORTS", "CONTRADICTS"}])
            else:
                add(cid, "not_established", "No retrieved passage was decisive for this statement.", "not_reported")
            continue
        if c.get("label") != SUPPORTED:
            continue
        for w in c.get("cautions") or []:
            add(cid, f"caution:{w.get('kind', '')}", w.get("detail", ""), "excluded", supporting)
        if not (c.get("citations") or {}).get("ok", True):
            add(cid, "citation_problem", "; ".join(c["citations"].get("problems", [])[:3]), "excluded", supporting)
        for row in c.get("support_check") or []:
            if not row.get("counted"):
                add(cid, "support_not_counted",
                    f"{row['source']} was judged to support this statement, but its best passage contains only "
                    f"{round(100 * row['coverage'])}% of the statement's content words, so it is not counted as "
                    f"corroboration (a lexical screen: it neither proves nor disproves support).", "needs_evidence",
                    [row["passage"]])
        for report in c.get("secondary_reports") or []:
            add(cid, "secondary_report",
                f"{report['source']} reports this as the result of {', '.join(n.title() for n in report['attributed_to'])}; "
                f"that original work was not retrieved, so the statement rests on a secondary report.", "needs_evidence",
                [report["passage"]])
        label = c.get("strength", "")
        if label == "abstract only":
            add(cid, "abstract_only", "Supported only by abstracts; methods, numbers and subgroups may differ in the "
                                      "full papers.", "needs_evidence", supporting)
        elif label in {"single source", "independence unknown"}:
            pairs = [p for p in (c.get("independence") or {}).get("pairs") or [] if p.get("relation") == "dependent"]
            detail = ("Its supporting sources depend on one another: " + pairs[0]["reason"] + "." if pairs else
                      "All support comes from one publisher or work." if label == "single source" else
                      "Its supporting sources are not established as independent.")
            add(cid, "corroboration", detail, "needs_evidence", supporting)
    for q in data.get("queries") or []:
        if q.get("status") in {"skipped", "failed", "no_provider"}:
            kind = "counter_query_not_run" if q.get("purpose") == "counter" else "query_not_run"
            what = "A counter-evidence query" if q.get("purpose") == "counter" else "A planned query"
            add(None, kind, f"{what} was {q['status'].replace('_', ' ')}: “{q['query']}” ({q.get('reason') or 'no reason recorded'}).",
                "unresolved" if q.get("purpose") == "counter" else "note", [q["id"]])
    searched = {q.get("subquestion") for q in data.get("queries") or [] if q.get("status") == "executed"}
    for e in data.get("ledger") or []:
        if e.get("index") and data.get("queries") and e["index"] not in searched:
            add(None, "subquestion_not_searched",
                f"Question {e['index']} had no executed query of its own in this run; broader queries may still have "
                f"found sources for it, but it was not searched specifically.", "unresolved",
                [q["id"] for q in data["queries"] if q.get("subquestion") == e["index"]])
    counter = data.get("counter_evidence") or {}
    if counter.get("skipped") and any(c.get("label") == SUPPORTED for c in data.get("claims") or []):
        add(None, "counter_search_skipped", f"No counter-evidence search after verification: {counter['skipped']}.", "note")
    for e in data.get("ledger") or []:
        if e.get("status") not in ANSWERED:
            name = "The main question" if e.get("index") == 0 else f"Question {e.get('index')}"
            add(None, "open_question", f"{name} is {e.get('status')}: {e.get('reason', '')}", "unresolved")
    return items


def run_status(data: Dict[str, Any], failed: bool, nothing_readable: bool = False) -> str:
    """FAILED only for an execution failure. A run that carried out its searches but found nothing readable is
    COMPLETE (PARTIAL when some of its searches failed), as in the single-pass pipeline: its outcome says the evidence
    was insufficient and ``error`` says why. Otherwise PARTIAL when a step reported problems."""
    if failed or data.get("status") == "FAILED":
        return "FAILED"
    if nothing_readable:
        return "PARTIAL" if data.get("search_errors") else "COMPLETE"
    return "PARTIAL" if any(s["status"] in {"partial", "failed"} for s in data["stages"]) else "COMPLETE"


def source_quality(sources: Sequence[Dict[str, Any]], claims: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    usable = [s for s in sources if s.get("access") in INDEXED_ACCESS]
    supporting = [p for c in claims for p in c.get("passages", []) if p.get("stance") == "SUPPORTS"]
    by_id = {s["id"]: s for s in sources}
    return {
        "discovered": len(sources), "read": len(usable),
        "by_access": {level: sum(s.get("access") == level for s in sources)
                      for level in (FULL_TEXT, ABSTRACT_ONLY, SNIPPET_ONLY, METADATA_ONLY, FAILED)},
        "by_category": {c: sum((s.get("category") or s.get("kind")) == c for s in usable)
                        for c in sorted({(s.get("category") or s.get("kind") or "other") for s in usable})},
        # distinct texts (near-copies merged) are not independent publishers: both are reported, under their own names
        "distinct_texts": len({s.get("origin") or s["id"] for s in usable}),
        "publisher_groups": independence(usable, largest_set=False)["groups"],
        "domains": len({s.get("domain") for s in usable if s.get("domain")}),
        "publishers": len({s.get("publisher") for s in usable if s.get("publisher")}),
        "supporting_passages_by_access": {level: sum(by_id.get(p["source_id"], {}).get("access") == level for p in supporting)
                                          for level in (FULL_TEXT, ABSTRACT_ONLY)},
        "providers_used": sorted({p for s in sources for p in (s.get("discovered_by") or [s.get("provider")]) if p}),
    }


def _limitations(deep: DeepResearch, data: Dict[str, Any]) -> None:
    quality = data["source_quality"]
    access = quality["by_access"]
    data["limitations"].append(
        f"Of {quality['discovered']} sources found, {access[FULL_TEXT]} were read in full, {access[ABSTRACT_ONLY]} only as "
        f"abstracts, {access[SNIPPET_ONLY]} only as search snippets (used for discovery, not as evidence), "
        f"{access[METADATA_ONLY]} had only metadata and {access[FAILED]} could not be retrieved.")
    failing: Dict[str, Dict[str, int]] = {}
    for r in deep.rounds:
        for name, status in (r.get("provider_status") or {}).items():
            for code, n in (status.get("failures") or {}).items():
                failing.setdefault(name, {})[code] = failing.get(name, {}).get(code, 0) + n
    for name, codes in failing.items():
        example = next((e["error"] for e in data["search_errors"] if e["provider"] == name), "")
        count = sum(codes.values())
        if set(codes) <= POLICY_CODES:                  # refused here by the network-safety policy, never contacted
            data["limitations"].append(f"Search provider '{name}' was not contacted: the network-safety policy refused "
                                       f"its address {count} time(s) ({example}). This is a local policy decision on "
                                       f"this network, not a failure of the provider.")
        else:
            data["limitations"].append(f"Search provider '{name}' failed {count} time(s) ({example})")
    refused = sum(s.get("diagnostics", {}).get("full_text_error") in POLICY_CODES for s in data["sources"])
    if refused:
        data["limitations"].append(f"{refused} open-access full-text download(s) were refused by the network-safety "
                                   f"policy (their hosts' addresses were not allowed), so those sources stayed abstracts.")
    if not set(deep.names) & {"searxng", "brave"}:
        data["limitations"].append("No general web search provider was used, so news, official and other web sources "
                                   "were reachable only through configured feeds, if any.")
    if deep.since:
        data["limitations"].append(f"Results dated before {deep.since} were left out; results without a recorded date were kept.")
    if deep.excluded_providers:
        data["limitations"].append(f"{EXCLUSION_NOTE}, so {', '.join(deep.excluded_providers)} was not searched.")
    dropped = sum(r.get("excluded_by_policy", 0) for r in deep.rounds)
    if dropped:
        data["limitations"].append(f"{dropped} search result(s) on Wikipedia were left out unread ({EXCLUSION_NOTE}).")
    stop = data["stopping"]
    if stop["completeness"] != "complete":
        data["limitations"].append(f"Research stopped: {stop['detail']} The result is {stop['completeness'].replace('_', '-')}, "
                                   f"not an exhaustive search.")
