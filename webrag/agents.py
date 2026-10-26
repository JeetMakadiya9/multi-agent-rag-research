"""LLM-backed agents with strict output contracts.

Agent A (Planner) turns the user's input into subquestions and search queries; it is told not to state facts, and
its output is used only as search input. Agent E (Synthesizer) writes findings that must cite the supplied passage
IDs; citations to unknown IDs are dropped and reported. Both validate their JSON deterministically and fall back
safely when the model output is unusable.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from docqa.llm import LLMError, OllamaChat

from .evidence import Passage
from .numeric import unsupported_numbers
from .researcher import Source

# Recorded in every run. v1 = first release; planner-v2 = planner told not to name search engines (provider names
# stripped from queries). The synthesizer version is recorded separately (Settings.synthesizer_version).
PROMPT_VERSION = "webrag-agents-planner-v2"

PLAN_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {"subquestions": {"type": "array", "items": {"type": "string"}},
                   "search_queries": {"type": "array", "items": {"type": "string"}}},
    "required": ["subquestions", "search_queries"],
}

PLAN_PROMPT = (
    "You plan web research. Do NOT answer the question and do NOT state any facts.\n"
    "1. List at most 3 subquestions that must be answered to address the user's input.\n"
    "2. Write at most 4 short keyword search queries (3 to 8 words each) for encyclopedia and scholarly search "
    "engines. Use the key entities and technical terms only; do not name websites or search engines. Include one "
    "query aimed at limitations, criticism or contrary evidence.\n"
    "Return JSON only."
)

SYNTHESIS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "findings": {"type": "array", "items": {"type": "object", "properties": {
            "text": {"type": "string"}, "evidence": {"type": "array", "items": {"type": "string"}}},
            "required": ["text", "evidence"]}},
        "disagreements": {"type": "array", "items": {"type": "object", "properties": {
            "text": {"type": "string"}, "evidence": {"type": "array", "items": {"type": "string"}}},
            "required": ["text", "evidence"]}},
        "gaps": {"type": "array", "items": {"type": "string"}},
        "insufficient": {"type": "boolean"},
    },
    "required": ["findings", "disagreements", "gaps", "insufficient"],
}

SYNTHESIS_PROMPT = (
    "You are a careful research analyst. Use ONLY the numbered evidence passages; never use outside knowledge.\n"
    "Rules:\n"
    "1. 'findings': at most 6 atomic factual statements that answer the question. Each must be supported by the "
    "passages whose IDs you list (for example [\"E2\"]). Keep each finding to one sentence and include the context "
    "that matters (study, dataset, population, year, conditions) when the passage gives it.\n"
    "2. 'disagreements': places where passages report different or opposing results, citing the passages on each side.\n"
    "3. 'gaps': what the question needs that the passages do not establish.\n"
    "4. Set 'insufficient' to true if the passages cannot answer the question.\n"
    "5. Never invent titles, authors, dates, numbers, quotations or URLs. Cite passage IDs only.\n"
    "Return JSON only."
)

# v2: in research runs, findings carried years taken from the source label ("a 2023 study"), which the verifier
# cannot find in the passage text, and joined two facts in one sentence, which the verifier can only half-check.
# Passage labels no longer show dates, details must come from the passage text, and each finding states one fact.
SYNTHESIS_PROMPT_V2 = (
    "You are a careful research analyst. Use ONLY the text of the numbered evidence passages; never use outside "
    "knowledge. The label in brackets before each passage (source type and title) is only for orientation: never "
    "take a date, author, venue or number from it.\n"
    "Rules:\n"
    "1. 'findings': at most 6 statements that answer the question. Each finding states exactly ONE fact (one "
    "subject, one property, one value) in one sentence; never join two facts with 'and' - write two findings. "
    "Every detail in a finding (names, numbers, dates, conditions) must appear in the text of the passages whose IDs "
    "you list (for example [\"E2\"]).\n"
    "2. 'disagreements': places where passages report different or opposing results, citing the passages on each side.\n"
    "3. 'gaps': what the question needs that the passages do not establish.\n"
    "4. Set 'insufficient' to true if the passages cannot answer the question.\n"
    "5. Never invent titles, authors, dates, numbers, quotations or URLs. Cite passage IDs only.\n"
    "Return JSON only."
)


@dataclass
class Plan:
    subquestions: List[str]
    queries: List[str]
    fallback: bool = False
    note: str = ""
    directions: List[str] = field(default_factory=list)      # perspective planner: one direction per query
    style: str = "default"


# Opt-in experiment (inspired by STORM's perspective-guided question asking): the same limits as the default planner,
# but each query is written for a stated research direction, and only the directions that matter are used.
PERSPECTIVE_VERSION = "planner-perspective-v1"
PERSPECTIVE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {"subquestions": {"type": "array", "items": {"type": "string"}},
                   "queries": {"type": "array", "items": {
                       "type": "object",
                       "properties": {"direction": {"type": "string",
                                                    "enum": ["supporting", "challenging", "context"]},
                                      "query": {"type": "string"}},
                       "required": ["direction", "query"]}}},
    "required": ["subquestions", "queries"],
}

PERSPECTIVE_PROMPT = (
    "You plan web research. Do NOT answer the question and do NOT state any facts.\n"
    "Consider three research directions and use only those that matter for this input (a simple factual question "
    "may need only one):\n"
    "- supporting: evidence that establishes the answer or supports the central claim;\n"
    "- challenging: evidence that challenges, contradicts or qualifies it;\n"
    "- context: methodology, limitations, conditions or alternative explanations.\n"
    "Write at most 3 subquestions and at most 3 short keyword search queries (3 to 8 words each), each labelled with "
    "its direction. Use the key entities and technical terms only; do not name websites or search engines.\n"
    "Return JSON only."
)


@dataclass
class CitedStatement:
    text: str
    passages: List[Passage]
    unknown_ids: List[str] = field(default_factory=list)
    subquestion: int = -1              # synthesizer v3: the subquestion it answers (0 = the main question)


@dataclass
class Synthesis:
    findings: List[CitedStatement] = field(default_factory=list)
    disagreements: List[CitedStatement] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)
    insufficient: bool = False
    parse_ok: bool = True
    truncated: bool = False
    error: str = ""
    raw: List[str] = field(default_factory=list)


PROVIDER_WORDS = re.compile(r"\s*\b(?:on\s+|from\s+|in\s+)?(?:wikipedia|openalex|arxiv|google scholar)\b\s*", re.I)


def strip_provider_names(query: str) -> str:
    """The model tends to append engine names ("... OpenAlex"), which only adds noise to the search."""
    return re.sub(r"\s+", " ", PROVIDER_WORDS.sub(" ", query)).strip(" :,-")


def _clean_strings(values: Any, limit: int, max_len: int = 200) -> List[str]:
    if not isinstance(values, list):
        raise ValueError("expected a list")
    out = []
    for value in values:
        if not isinstance(value, str):
            raise ValueError("expected strings")
        value = re.sub(r"\s+", " ", value).strip().strip('"')
        if value and value.lower() not in {v.lower() for v in out}:
            out.append(value[:max_len])
    return out[:limit]


class Planner:
    """Agent A. ``style``: "default", or "perspective" (opt-in experiment; same query limit)."""

    def __init__(self, client: OllamaChat, max_queries: int = 4, style: str = "default"):
        if style not in {"default", "perspective"}:
            raise ValueError(f"unknown planner style {style!r}")
        self.client, self.max_queries, self.style = client, max_queries, style

    def plan(self, text: str, mode: str) -> Plan:
        task = "Claim to verify" if mode == "verify" else "Research question"
        perspective = self.style == "perspective"
        prompt, schema = (PERSPECTIVE_PROMPT, PERSPECTIVE_SCHEMA) if perspective else (PLAN_PROMPT, PLAN_SCHEMA)
        messages = [{"role": "system", "content": prompt}, {"role": "user", "content": f"{task}: {text}"}]
        directions: List[str] = []
        try:
            response = self.client.chat(messages, response_format=schema, num_predict=300)
            data = json.loads(response.text)
            subquestions = _clean_strings(data.get("subquestions"), 3)
            if perspective:
                rows = data.get("queries")
                if not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows):
                    raise ValueError("queries must be objects")
                pairs = [(str(r.get("direction", "")), strip_provider_names(str(r.get("query", "")))) for r in rows]
                raw_queries = [q for _, q in pairs]
                direction_of = {q.lower(): d for d, q in pairs}
            else:
                raw_queries = data.get("search_queries")
                if isinstance(raw_queries, list):
                    raw_queries = [strip_provider_names(q) if isinstance(q, str) else q for q in raw_queries]
            queries = _clean_strings(raw_queries, self.max_queries, 120)
            if not queries:
                raise ValueError("no queries")
            if perspective:
                directions = [direction_of.get(q.lower(), "") for q in queries]
        except (LLMError, ValueError, json.JSONDecodeError, AttributeError) as exc:
            return Plan([], [text[:200]], fallback=True, style=self.style,
                        note=f"Planner output unusable ({type(exc).__name__}); searched with the input text itself.")
        if text[:200] not in queries:            # the user's own wording is always searched as well
            queries = [text[:200]] + queries[:self.max_queries - 1]
            directions = (["question"] + directions[:self.max_queries - 1]) if perspective else []
        return Plan(subquestions, queries, directions=directions, style=self.style)


class Synthesizer:
    """Agent E (drafting step): cited findings from the passages. The orchestrator later lists as findings only the
    statements the verifier marks SUPPORTED, so unverified prose never reaches the answer.

    ``version`` 3 (deep research only) is version 2 plus the number of the subquestion each finding answers, so
    coverage can be tracked per subquestion; versions 1 and 2 are unchanged."""

    def __init__(self, client: OllamaChat, num_predict: int = 900, version: int = 2, max_findings: int = 6):
        self.client, self.num_predict, self.version, self.max_findings = client, num_predict, version, max_findings

    def synthesize(self, question: str, subquestions: Sequence[str], passages: Sequence[Passage],
                   sources: Dict[str, Source], notes: Optional[Dict[str, str]] = None) -> Synthesis:
        """``notes`` (version 3 only): a short remark per passage key, added to its label (deep research marks
        passages confirmed to begin inside a word or sentence)."""
        labelled = {f"E{i}": p for i, p in enumerate(passages, 1)}
        blocks = []
        for label, p in labelled.items():
            s = sources[p.source_id]
            date = f", {s.published[:10]}" if s.published and self.version == 1 else ""
            note = f"; {notes[p.key]}" if self.version == 3 and notes and p.key in notes else ""
            blocks.append(f"[{label}] ({s.kind}: {s.title}{date}; {s.access.replace('_', ' ').lower()}{note})\n{p.text}")
        if self.version == 3:
            sub = "".join(f"\n{i}. {q}" for i, q in enumerate(subquestions, 1))
            user = f"Question: {question}" + (f"\nSubquestions (0 = the main question):{sub}" if sub else "")
        else:
            sub = "".join(f"\n- {q}" for q in subquestions)
            user = f"Question: {question}" + (f"\nSubquestions:{sub}" if sub else "")
        user += "\n\nEvidence passages:\n" + "\n\n".join(blocks)
        prompt = {1: SYNTHESIS_PROMPT, 2: SYNTHESIS_PROMPT_V2, 3: SYNTHESIS_PROMPT_V3}[self.version]
        schema = SYNTHESIS_SCHEMA_V3 if self.version == 3 else SYNTHESIS_SCHEMA
        messages = [{"role": "system", "content": prompt}, {"role": "user", "content": user}]
        result = Synthesis(parse_ok=False, error="no attempt")
        for _ in range(2):
            response = self.client.chat(messages, response_format=schema, num_predict=self.num_predict)
            result.raw.append(response.text)
            result.truncated = result.truncated or response.truncated
            try:
                parsed = self._parse(response.text, labelled, len(subquestions) if self.version == 3 else None,
                                     self.max_findings)
            except (ValueError, json.JSONDecodeError) as exc:
                result.error = f"unparsable synthesis ({type(exc).__name__}: {exc})"
                continue
            parsed.raw, parsed.truncated = result.raw, result.truncated
            return parsed
        return result

    @staticmethod
    def _parse(text: str, labelled: Dict[str, Passage], subquestions: Optional[int] = None,
               max_findings: int = 6) -> Synthesis:
        data = json.loads(text)
        if not isinstance(data, dict) or not isinstance(data.get("insufficient"), bool):
            raise ValueError("schema mismatch")

        def statements(rows: Any, limit: int) -> List[CitedStatement]:
            if not isinstance(rows, list):
                raise ValueError("expected a list of statements")
            out = []
            for row in rows[:limit]:
                if not isinstance(row, dict) or not isinstance(row.get("text"), str) or not row["text"].strip():
                    raise ValueError("statement without text")
                ids = [str(x).strip().strip("[]") for x in (row.get("evidence") or []) if str(x).strip()]
                ids = list(dict.fromkeys(ids))
                statement = CitedStatement(row["text"].strip(), [labelled[i] for i in ids if i in labelled],
                                           [i for i in ids if i not in labelled])
                if subquestions is not None:
                    number = row.get("subquestion")
                    statement.subquestion = number if isinstance(number, int) and 0 <= number <= subquestions else 0
                out.append(statement)
            return out

        return Synthesis(findings=statements(data.get("findings"), max_findings),
                         disagreements=statements(data.get("disagreements"), 4),
                         gaps=_clean_strings(data.get("gaps") or [], 6, 300), insufficient=data["insufficient"])


# ================================================================================================= deep research
SYNTHESIS_SCHEMA_V3: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "findings": {"type": "array", "items": {"type": "object", "properties": {
            "text": {"type": "string"}, "evidence": {"type": "array", "items": {"type": "string"}},
            "subquestion": {"type": "integer"}}, "required": ["text", "evidence", "subquestion"]}},
        "disagreements": SYNTHESIS_SCHEMA["properties"]["disagreements"],
        "gaps": {"type": "array", "items": {"type": "string"}},
        "insufficient": {"type": "boolean"},
    },
    "required": ["findings", "disagreements", "gaps", "insufficient"],
}

SYNTHESIS_PROMPT_V3 = SYNTHESIS_PROMPT_V2.replace(
    "1. 'findings': at most 6 statements",
    "1. 'findings': at most 8 statements, covering the subquestions in turn,").replace(
    "you list (for example [\"E2\"]).",
    "you list (for example [\"E2\"]). Give 'subquestion': the number of the subquestion the finding answers "
    "(0 for the main question).").replace(
    "Return JSON only.",
    "6. A passage whose label says it 'begins mid-word' or 'begins mid-sentence' starts inside a sentence: never name a "
    "subject for its first words that its text does not show; leave that part out.\n"
    "7. Keep the passage's certainty: if it says 'may', 'might', 'could', 'suggests', 'appears' or 'is associated with', "
    "the finding says so too. Never state cause and effect ('causes', 'leads to', 'because', 'due to') unless the passage "
    "states it; an association is not a cause.\n"
    "8. Keep the passage's qualifiers and scope: quantities ('many', 'some', 'most'), 'can', and the conditions, datasets "
    "or populations a result was obtained on. When a passage reports another study's result ('Xu et al. found ...'), "
    "say who reported it. If a short finding cannot keep these, narrow the finding instead of dropping them.\n"
    "Return JSON only.")

CATEGORY_HINTS = {
    "encyclopedia": "encyclopedia articles (keywords, 2 to 6 words)",
    "scholarly": "scholarly papers and preprints (technical keywords, 3 to 8 words)",
    "web": "the general web: documentation, specialist and publisher sites (a short natural-language query)",
    "news": "news reporting (a short query naming the event, people or organisations)",
    "official": "official sources: government, regulators, international bodies (a query naming the issuing body or "
                "document type)",
    "podcast": "podcast and interview transcripts (a short topical query)",
}

SOURCE_PLAN_PROMPT = (
    "You plan research. Do NOT answer the question and do NOT state any facts.\n"
    "Return JSON with:\n"
    "- 'central_question': the question restated in one sentence; 'deliverable': what the user wants (answer, "
    "comparison, explanation, list ...).\n"
    "- 'subquestions': at most {max_sub} subquestions that must be answered, each with 'text', 'terms' (key terms, "
    "synonyms and alternative names), and 'categories' (which source types can answer it).\n"
    "- 'queries': search queries, each with 'subquestion' (0 = the main question), 'category' and 'query'. Write "
    "each query for its source type:{hints}\nWrite at most {max_queries} queries in total and do not repeat a query "
    "for every source type unless it suits them.\n"
    "- 'counter_queries': at most 2 queries looking for evidence that contradicts, criticises or qualifies the likely "
    "answer, each with 'subquestion' and 'query'.\n"
    "- 'time_constraint', 'geography', 'entities', 'primary_sources' (kinds of authoritative records that would settle "
    "it), 'competing_explanations' and 'uncertainties': short strings or lists; empty when not relevant.\n"
    "Do not name websites or search engines inside queries. Return JSON only."
)

SOURCE_PLAN_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "central_question": {"type": "string"}, "deliverable": {"type": "string"},
        "subquestions": {"type": "array", "items": {"type": "object", "properties": {
            "text": {"type": "string"}, "terms": {"type": "array", "items": {"type": "string"}},
            "categories": {"type": "array", "items": {"type": "string"}}}, "required": ["text"]}},
        "queries": {"type": "array", "items": {"type": "object", "properties": {
            "subquestion": {"type": "integer"}, "category": {"type": "string"}, "query": {"type": "string"}},
            "required": ["category", "query"]}},
        "counter_queries": {"type": "array", "items": {"type": "object", "properties": {
            "subquestion": {"type": "integer"}, "query": {"type": "string"}}, "required": ["query"]}},
        "time_constraint": {"type": "string"}, "geography": {"type": "string"},
        "entities": {"type": "array", "items": {"type": "string"}},
        "primary_sources": {"type": "array", "items": {"type": "string"}},
        "competing_explanations": {"type": "array", "items": {"type": "string"}},
        "uncertainties": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["subquestions", "queries"],
}

SOURCE_PLANNER_VERSION = "planner-source-aware-v1"


@dataclass
class PlannedQuery:
    query: str
    category: str
    subquestion: int = 0               # 0 = the main question; n = subquestion n
    purpose: str = "answer"            # answer | counter | follow-up


@dataclass
class SourcePlan:
    central_question: str
    deliverable: str
    subquestions: List[Dict[str, Any]]                 # {"text", "terms", "categories"}
    queries: List[PlannedQuery]
    counter_queries: List[PlannedQuery]
    context: Dict[str, Any] = field(default_factory=dict)  # time, geography, entities, primary sources, ...
    fallback: bool = False
    note: str = ""
    version: str = SOURCE_PLANNER_VERSION


def _clean_query(value: Any) -> str:
    return re.sub(r"\s+", " ", strip_provider_names(str(value or ""))).strip().strip('"')[:160]


def _string_list(value: Any, limit: int = 6, max_len: int = 160) -> List[str]:
    if isinstance(value, str):
        value = [value] if value.strip() else []
    if not isinstance(value, list):
        return []
    return [re.sub(r"\s+", " ", str(v)).strip()[:max_len] for v in value if str(v).strip()][:limit]


def fallback_queries(text: str, categories: Sequence[str], limit: int) -> List[PlannedQuery]:
    """Deterministic plan when the model's output is unusable: the input itself for the main source types, then its
    keywords for the rest. Never invents terms."""
    from .search import keywords
    words = " ".join(keywords(text, 6)) or text[:120]
    out: List[PlannedQuery] = []
    for i, category in enumerate(categories):
        query = text[:160] if category in {"web", "news", "official", "podcast"} or i == 0 else words
        out.append(PlannedQuery(query, category, 0))
    return out[:limit]


class SourceAwarePlanner:
    """Agent A for deep research: subquestions with terms and source types, queries written per source type, and
    queries for counter-evidence. Output is validated deterministically; when it is unusable, a keyword plan is used
    and the run says so. ``categories`` are the source types that have an active provider; nothing else is planned."""

    def __init__(self, client: OllamaChat, max_subquestions: int = 4, max_queries: int = 6):
        self.client, self.max_subquestions, self.max_queries = client, max_subquestions, max_queries

    def plan(self, text: str, mode: str, categories: Sequence[str], preferred: Sequence[str] = ()) -> SourcePlan:
        available = [c for c in categories]
        order = [c for c in preferred if c in available] + [c for c in available if c not in preferred]
        if not order:
            return SourcePlan(text, "", [{"text": text, "terms": [], "categories": []}], [], [], fallback=True,
                              note="No search provider is active, so nothing could be planned.")
        hints = "".join(f"\n  - '{c}': {CATEGORY_HINTS[c]}" for c in order)
        prompt = SOURCE_PLAN_PROMPT.format(max_sub=self.max_subquestions, hints=hints, max_queries=self.max_queries)
        task = "Claim to verify" if mode == "verify" else "Research question"
        preference = f"\nPreferred source types: {', '.join(preferred)}" if preferred else ""
        messages = [{"role": "system", "content": prompt}, {"role": "user", "content": f"{task}: {text}{preference}"}]
        try:
            response = self.client.chat(messages, response_format=SOURCE_PLAN_SCHEMA, num_predict=700)
            data = json.loads(response.text)
            if not isinstance(data, dict):
                raise ValueError("not an object")
            subquestions = []
            for row in data.get("subquestions") or []:
                if isinstance(row, dict) and str(row.get("text", "")).strip():
                    subquestions.append({"text": re.sub(r"\s+", " ", str(row["text"])).strip()[:240],
                                         "terms": _string_list(row.get("terms"), 8, 60),
                                         "categories": [c for c in _string_list(row.get("categories"), 6, 20) if c in available]})
                if len(subquestions) == self.max_subquestions:
                    break
            seen, queries = set(), []
            for row in data.get("queries") or []:
                if not isinstance(row, dict):
                    continue
                category, query = str(row.get("category", "")).strip().lower(), _clean_query(row.get("query"))
                number = row.get("subquestion") if isinstance(row.get("subquestion"), int) else 0
                if category not in available or not query or (query.lower(), category) in seen:
                    continue
                seen.add((query.lower(), category))
                queries.append(PlannedQuery(query, category, number if 0 <= number <= len(subquestions) else 0))
            if not queries:
                raise ValueError("no usable queries")
            counter = []
            for row in (data.get("counter_queries") or [])[:2]:
                if isinstance(row, dict) and _clean_query(row.get("query")):
                    number = row.get("subquestion") if isinstance(row.get("subquestion"), int) else 0
                    counter.append(PlannedQuery(_clean_query(row["query"]), order[0], number if 0 <= number <= len(subquestions) else 0, "counter"))
            context = {"time_constraint": str(data.get("time_constraint") or "")[:160],
                       "geography": str(data.get("geography") or "")[:160],
                       "entities": _string_list(data.get("entities"), 8, 80),
                       "primary_sources": _string_list(data.get("primary_sources"), 6),
                       "competing_explanations": _string_list(data.get("competing_explanations"), 4),
                       "uncertainties": _string_list(data.get("uncertainties"), 4)}
        except (LLMError, ValueError, json.JSONDecodeError, AttributeError, TypeError) as exc:
            return SourcePlan(text, "", [{"text": text, "terms": [], "categories": order}],
                              fallback_queries(text, order, self.max_queries), [], fallback=True,
                              note=f"Planner output unusable ({type(exc).__name__}); searched with the input text and its keywords.")
        # The user's own wording is always searched, for the first suitable source type.
        if not any(q.query.lower() == text[:160].lower() for q in queries):
            queries.insert(0, PlannedQuery(text[:160], order[0], 0))
        for sub in subquestions:
            sub["categories"] = sub["categories"] or order
        return SourcePlan(str(data.get("central_question") or text)[:300], str(data.get("deliverable") or "")[:120],
                          subquestions or [{"text": text, "terms": [], "categories": order}],
                          queries[:self.max_queries], counter, context)


QUERY_WRITER_PROMPT = {
    "follow-up": ("Some subquestions of a research question are not yet answered by the sources found. Write search "
                  "queries that could find sources answering them. Each query has 'subquestion' (its number), "
                  "'category' (one of: {categories}) and 'query'. Write each query for its source type. Do not repeat "
                  "earlier queries. At most {n} queries. Return JSON only."),
    "counter": ("These statements from a research draft were supported by few independent sources. Write search queries "
                "that could find evidence contradicting, criticising or qualifying them. Each query has 'subquestion' "
                "(the statement's number), 'category' (one of: {categories}) and 'query'. Do not repeat earlier "
                "queries. At most {n} queries. Return JSON only."),
}
QUERY_WRITER_SCHEMA: Dict[str, Any] = {"type": "object", "properties": {"queries": SOURCE_PLAN_SCHEMA["properties"]["queries"]},
                                       "required": ["queries"]}


def write_queries(client: OllamaChat, purpose: str, items: Sequence[str], categories: Sequence[str],
                  previous: Sequence[str], max_queries: int) -> Tuple[List[PlannedQuery], bool]:
    """Targeted queries for unanswered subquestions ("follow-up") or for counter-evidence ("counter"). Returns the
    queries and whether the deterministic keyword fallback was used."""
    from .search import keywords
    if not items or not categories or max_queries <= 0:
        return [], False
    seen = {re.sub(r"\s+", " ", q).strip().lower() for q in previous}
    numbered = "\n".join(f"{i}. {text}" for i, text in enumerate(items, 1))
    messages = [{"role": "system", "content": QUERY_WRITER_PROMPT[purpose].format(categories=", ".join(categories), n=max_queries)},
                {"role": "user", "content": f"{numbered}\n\nEarlier queries:\n" + "\n".join(f"- {q}" for q in previous)}]
    out: List[PlannedQuery] = []
    try:
        data = json.loads(client.chat(messages, response_format=QUERY_WRITER_SCHEMA, num_predict=300).text)
        for row in data.get("queries") or []:
            if not isinstance(row, dict):
                continue
            query, category = _clean_query(row.get("query")), str(row.get("category", "")).strip().lower()
            number = row.get("subquestion") if isinstance(row.get("subquestion"), int) else 1
            if not query or category not in categories or query.lower() in seen:
                continue
            seen.add(query.lower())
            out.append(PlannedQuery(query, category, max(0, min(number, len(items))), purpose))
            if len(out) == max_queries:
                break
        if out:
            return out, False
    except (LLMError, ValueError, json.JSONDecodeError, AttributeError, TypeError):
        pass
    suffix = " criticism" if purpose == "counter" else ""
    for i, text in enumerate(items, 1):
        text = re.sub(r"\s*\(evidence gap:.*\)\s*$", "", text)       # the gap note guides the model only
        query = (" ".join(keywords(text, 6)) or text[:120]) + suffix
        if query.lower() not in seen:
            seen.add(query.lower())
            out.append(PlannedQuery(query, categories[(i - 1) % len(categories)], i, purpose))
        if len(out) == max_queries:
            break
    return out, True


SUMMARY_PROMPT = (
    "You write the opening of a research report from VERIFIED findings only (numbered F1, F2, ...).\n"
    "Return JSON with 'answered' (true if the findings answer the question), 'direct_answer' ({'text', 'findings'}: "
    "one or two sentences) and 'summary' (at most 5 points, each {'text', 'findings'}).\n"
    "Every sentence restates only what the findings it cites say, and cites them (for example [\"F1\", \"F3\"]). "
    "Never add a fact, number, name or date that is not in a cited finding. Do not claim certainty or consensus the "
    "findings do not state: keep their hedges ('may', 'could', 'suggests') and never add cause and effect ('causes', "
    "'leads to', 'because') that they do not state. Keep each finding's qualifiers ('many', 'some', 'can'), its "
    "conditions and datasets, and its attribution ('Xu et al. reported ...'); a sentence that drops them is discarded. "
    "Return JSON only."
)
SUMMARY_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "answered": {"type": "boolean"},
        "direct_answer": {"type": "object", "properties": {"text": {"type": "string"},
                                                           "findings": {"type": "array", "items": {"type": "string"}}}},
        "summary": {"type": "array", "items": {"type": "object", "properties": {
            "text": {"type": "string"}, "findings": {"type": "array", "items": {"type": "string"}}}}},
    },
    "required": ["answered", "direct_answer", "summary"],
}
# Lexical cues for the fidelity checks (deep research). They are warnings, not proof of meaning: they miss paraphrased
# hedges and causes, and "can" (ability) is deliberately not a hedge.
HEDGE = re.compile(r"\b(?:may|might|could|possibly|possible|potentially|potential|perhaps|likely|unlikely|suggests?|"
                   r"suggested|suggesting|appears?|appeared|seems?|seemed|indicates?|indicated|indicating|associated|"
                   r"associations?|correlat\w*|preliminary)\b", re.I)
CAUSAL = re.compile(r"\b(?:caus(?:e|es|ed|ing|al|ally|ation)|leads? to|led to|leading to|results? in|resulted in|"
                    r"resulting in|because|due to|owing to|attributable to|drives?|drove|driven by|driving|therefore|"
                    r"thereby|consequently)\b", re.I)
ASSOCIATION = re.compile(r"\b(?:associat\w*|correlat\w*|linked (?:to|with)|relationship between)\b", re.I)
NEGATION = re.compile(r"\b(?:no|not|never|without|neither|nor|none|cannot|fail(?:s|ed|ing)?)\b|n't\b", re.I)
# Qualifiers that limit how far a statement reaches: quantities and modality ("can" here, unlike HEDGE, because
# "can capture" is weaker than "captures"), the scope a result was obtained on, and who reported it.
QUALIFIER = re.compile(r"\b(?:many|some|most|several|few|often|sometimes|frequently|typically|usually|generally|"
                       r"partly|partially|largely|can|may|might|could)\b", re.I)
SCOPE = re.compile(r"\b(?:on|in|across|using|from|among|for)\s+(?:[\w\-]+[,;:]?\s+){0,4}?"
                   r"(datasets?|data\s+sets?|benchmarks?|samples?|cohorts?|participants|patients|trials?|experiments?|"
                   r"simulations?|countries|banks)\b", re.I)
ATTRIBUTION = re.compile(r"\b([A-Z][A-Za-z\-']+)(?:\s+(?:and|&)\s+[A-Z][A-Za-z\-']+)?\s+et\s+al\b|"
                         r"\b([A-Z][A-Za-z\-']+)\s+(?:and|&)\s+[A-Z][A-Za-z\-']+\s+\(\d{4}\)|\baccording to\b")


def qualifier_words(text: str) -> set:
    return {m.lower() for m in QUALIFIER.findall(text)}


def scope_words(text: str) -> set:
    """The scope nouns of phrases such as "on two classic financial fraud datasets" ("dataset")."""
    return {re.sub(r"\s+", " ", m.lower()).rstrip("s") for m in SCOPE.findall(text)}


def attributions(text: str) -> set:
    """Who a statement is attributed to: the surnames of "Xu et al." or "Kipf and Welling (2017)", or "according to"."""
    found = set()
    for m in ATTRIBUTION.finditer(text):
        found.add((m.group(1) or m.group(2) or "according to").lower())
    return found


def _statement_ok(text: str, cited: Sequence[str]) -> bool:
    """A summary sentence may only contain numbers that its cited findings state as the same whole token
    (``webrag.numeric``: "5" is not supported by "25"); it may not add cause and
    effect that none of them states, nor drop a hedge that all of them carry; and for every finding it cites it must
    keep that finding's qualifiers ("many", "can" ...), scope ("datasets" ...) and attribution ("Xu et al."). Modality
    is not judged in a negated sentence. These are lexical checks: a paraphrase ("numerous" for "many") is not
    recognised, so a faithful paraphrase can be dropped and an unfaithful one can pass."""
    if unsupported_numbers(text, cited):                # whole numeric tokens (webrag/numeric.py), never substrings
        return False
    if CAUSAL.search(text) and not any(CAUSAL.search(c) for c in cited):
        return False
    negated = bool(NEGATION.search(text))
    if cited and all(HEDGE.search(c) for c in cited) and not HEDGE.search(text) and not negated:
        return False
    for finding in cited:
        if not negated and not qualifier_words(finding) <= qualifier_words(text):
            return False
        if not scope_words(finding) <= scope_words(text) or not attributions(finding) <= attributions(text):
            return False
    return True


MAX_STATEMENT_CHARS = 600


def fit_sentences(text: str, limit: int = MAX_STATEMENT_CHARS) -> Tuple[str, bool]:
    """``text`` within ``limit`` characters, never cut inside a sentence: whole sentences are kept and " […]" marks
    that later ones were left out; a first sentence longer than the limit is cut at a word and ends with "…".
    Returns the text and whether it was shortened."""
    if len(text) <= limit:
        return text, False
    marker, kept = " […]", ""
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        candidate = f"{kept} {sentence}".strip()
        if len(candidate) + len(marker) > limit:
            break
        kept = candidate
    if kept:
        return kept + marker, True
    return text[:limit - 1].rsplit(" ", 1)[0].rstrip(",;: ") + "…", True


def summarize(client: OllamaChat, question: str, findings: Sequence[str]) -> Dict[str, Any]:
    """Direct answer and executive summary restating verified findings. Sentences that cite no verified finding, or
    that carry a number absent from the findings they cite, are dropped and counted."""
    if not findings:
        return {"available": False, "reason": "No finding was verified, so no answer or summary is given."}
    labelled = {f"F{i}": text for i, text in enumerate(findings, 1)}
    user = f"Question: {question}\n\nVerified findings:\n" + "\n".join(f"[{k}] {v}" for k, v in labelled.items())
    try:
        data = json.loads(client.chat([{"role": "system", "content": SUMMARY_PROMPT}, {"role": "user", "content": user}],
                                      response_format=SUMMARY_SCHEMA, num_predict=500).text)
        if not isinstance(data, dict):
            raise ValueError("not an object")
    except (LLMError, ValueError, json.JSONDecodeError) as exc:
        return {"available": False, "reason": f"The summary could not be written ({type(exc).__name__}); the verified findings below stand on their own."}
    dropped = 0

    def clean(row: Any) -> Optional[Dict[str, Any]]:
        nonlocal dropped
        if not isinstance(row, dict) or not str(row.get("text", "")).strip():
            return None
        ids = [str(i).strip().strip("[]") for i in row.get("findings") or [] if str(i).strip().strip("[]") in labelled]
        text = re.sub(r"\s*\[(?:\"?F\d+\"?(?:,\s*)?)+\]", "", str(row["text"])).strip()
        # shortened first, then checked: what is shown must itself keep the cited findings' qualifiers
        text, shortened = fit_sentences(text)
        if not ids or not _statement_ok(text, [labelled[i] for i in ids]):
            dropped += 1
            return None
        part = {"text": text, "findings": list(dict.fromkeys(ids))}
        if shortened:
            part["shortened"] = True
        return part

    answer = clean(data.get("direct_answer"))
    points = [p for p in (clean(r) for r in (data.get("summary") or [])[:5]) if p]
    return {"available": bool(answer or points), "answered": bool(data.get("answered")) and bool(answer),
            "direct_answer": answer, "summary": points, "dropped": dropped,
            "reason": "" if (answer or points) else "The model's summary did not cite verified findings, so it was not used."}
