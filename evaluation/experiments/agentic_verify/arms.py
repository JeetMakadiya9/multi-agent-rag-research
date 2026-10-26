"""The compared arms. All share the retriever, the model, the verifier prompt and the aggregation rule.

* ``single_pass``  (B1 / B1-K): retrieve top-k for the claim, judge every passage once, aggregate.
* ``agentic``      (C): the B1 step, then an LLM planner chooses further SEARCH actions (bounded), new passages are
  judged by the same verifier, and an LLM adjudicator makes the final decision from the passages that took a side.
* ``adjudicated``  (C-noplan, ablation): the B1 step followed by the adjudicator, without planning or search.
"""
from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import prompts
from .llm import ChatClient, LLMError, Usage, parse_json

SUPPORTED, CONTRADICTED, INSUFFICIENT, FAILED = "SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE", "FAILED"
STANCES = ("SUPPORTS", "CONTRADICTS", "NEUTRAL")
_ALNUM = re.compile(r"[^0-9a-z]+")


@dataclass
class Passage:
    pid: str
    doc_id: Optional[int]
    chunk_id: str
    text: str
    source: str            # "claim" or "search<n>"
    rank: int


@dataclass
class Stance:
    pid: str
    stance: str
    grounded: bool
    raw_stance: str


@dataclass
class Settings:
    verifier: str = "v1"
    planner: str = "v0"
    adjudicator: str = "v0"
    k: int = 8                 # passages judged in the first (claim) retrieval
    search_k: int = 4          # new passages kept per planner search
    max_searches: int = 2
    max_turns: int = 3
    adjudicate_max: int = 6    # passages shown to the adjudicator
    verifier_batch: int = 8
    verifier_tokens: int = 900
    planner_tokens: int = 120
    adjudicator_tokens: int = 200
    trigger: str = "always"      # "unresolved": plan/search/adjudicate only when the first judgement is not unanimous
    fallback: str = "aggregate"  # "rank": an unusable adjudication falls back to the B1R rank rule, not the aggregate
    skip_exhausted_planner: bool = False   # do not ask the planner again once no search is left
    search_when: str = "unresolved"        # "empty": conflicting first evidence goes straight to adjudication;
                                           # the planner/search loop runs only when no passage took a side


@dataclass
class Result:
    prediction: str
    status: str                                   # "ok" or a failure code
    usage: Dict[str, Any]
    retrieval_calls: int
    latency_seconds: float
    passages: List[Dict[str, Any]] = field(default_factory=list)
    stances: List[Dict[str, Any]] = field(default_factory=list)
    actions: List[Dict[str, Any]] = field(default_factory=list)
    adjudication: Optional[Dict[str, Any]] = None
    aggregate_verdict: Optional[str] = None
    invalid_outputs: int = 0
    ungrounded_quotes: int = 0


def _norm(text: str) -> str:
    return _ALNUM.sub("", (text or "").lower())


def grounded(quote: Any, passage_text: str) -> bool:
    q = _norm(quote if isinstance(quote, str) else "")
    return len(q) >= 12 and q in _norm(passage_text)


class Retriever:
    """Wraps the frozen Phase 10 provider; returns Passage objects with SciFact document IDs."""

    def __init__(self, provider: Any):
        self.provider = provider
        self.calls = 0

    def search(self, query: str, limit: int, source: str) -> List[Passage]:
        self.calls += 1
        response = self.provider.retrieve(query, limit=limit)
        out = []
        for rank, item in enumerate(getattr(response, "evidence", []) or [], start=1):
            metadata = getattr(item, "retrieval_metadata", {}) or {}
            chunk = metadata.get("chunk", {}) if isinstance(metadata, dict) else {}
            doc_id = chunk.get("scifact_doc_id", getattr(item, "document_id", None))
            try:
                doc_id = int(doc_id)
            except (TypeError, ValueError):
                doc_id = None
            chunk_id = str(getattr(item, "chunk_id", "") or chunk.get("chunk_id", "") or f"{source}-{rank}")
            out.append(Passage("", doc_id, chunk_id, str(getattr(item, "text", "") or ""), source, rank))
        return out


def judge(claim: str, passages: Sequence[Passage], client: ChatClient, usage: Usage, s: Settings
          ) -> Tuple[List[Stance], int, int]:
    """Verifier role. Returns stances (one per passage), invalid-output count and ungrounded-quote count."""
    stances: List[Stance] = []
    invalid = ungrounded = 0
    for start in range(0, len(passages), s.verifier_batch):
        batch = passages[start:start + s.verifier_batch]
        user = prompts.verifier_user(claim, [(p.pid, p.text) for p in batch])
        parsed = None
        for attempt in range(2):
            suffix = "" if attempt == 0 else "\n\nReturn valid JSON with one assessment per passage."
            text = client.complete("verifier", prompts.VERIFIER_SYSTEM[s.verifier], user + suffix,
                                   s.verifier_tokens, usage)
            parsed = parse_json(text)
            if isinstance(parsed, dict) and isinstance(parsed.get("assessments"), list):
                break
            invalid += 1
            parsed = None
        by_id = {}
        if parsed is not None:
            for entry in parsed["assessments"]:
                if isinstance(entry, dict) and isinstance(entry.get("id"), str):
                    by_id.setdefault(entry["id"].strip().strip("[]"), entry)
        for p in batch:
            entry = by_id.get(p.pid, {})
            raw = str(entry.get("stance", "NEUTRAL")).upper().strip()
            raw = raw if raw in STANCES else "NEUTRAL"
            ok = raw == "NEUTRAL" or grounded(entry.get("quote"), p.text)
            if not ok:
                ungrounded += 1
            stances.append(Stance(p.pid, raw if ok else "NEUTRAL", ok, raw))
        if parsed is None:
            raise LLMError("VERIFIER_INVALID")
    return stances, invalid, ungrounded


def aggregate(stances: Sequence[Stance]) -> str:
    sup = any(st.stance == "SUPPORTS" for st in stances)
    con = any(st.stance == "CONTRADICTS" for st in stances)
    if sup and not con:
        return SUPPORTED
    if con and not sup:
        return CONTRADICTED
    return INSUFFICIENT


def rank_rule(passages: Sequence[Passage], stances: Sequence[Stance]) -> str:
    """The B1R rule: a unanimous set keeps its verdict; a mixed set is decided by the best-ranked sided passage
    (first-retrieval passages before search results)."""
    verdict = aggregate(stances)
    if verdict != INSUFFICIENT:
        return verdict
    order = {p.pid: (0 if p.source == "claim" else 1, p.rank) for p in passages}
    sided = sorted((order.get(st.pid, (9, 99)), st.stance) for st in stances if st.stance != "NEUTRAL")
    if not sided:
        return INSUFFICIENT
    return SUPPORTED if sided[0][1] == "SUPPORTS" else CONTRADICTED


def _fallback(passages, stances, s: "Settings") -> str:
    return rank_rule(passages, stances) if s.fallback == "rank" else aggregate(stances)


def _number(passages: List[Passage]) -> None:
    for i, p in enumerate(passages, start=1):
        p.pid = f"P{i}"


def adjudicate(claim: str, passages: Sequence[Passage], stances: Sequence[Stance], client: ChatClient,
               usage: Usage, s: Settings) -> Tuple[Optional[str], Dict[str, Any], int]:
    """Adjudicator role. Shows the passages that took a side (first), topped up with the best-ranked others."""
    stance_of = {st.pid: st.stance for st in stances}
    sided = [p for p in passages if stance_of.get(p.pid) != "NEUTRAL"]
    others = [p for p in passages if stance_of.get(p.pid) == "NEUTRAL"]
    shown = (sided + others)[:s.adjudicate_max]
    user = prompts.adjudicator_user(claim, [(p.pid, stance_of.get(p.pid, "NEUTRAL"), p.text) for p in shown])
    invalid = 0
    for attempt in range(2):
        suffix = "" if attempt == 0 else "\n\nReturn valid JSON with a verdict."
        parsed = parse_json(client.complete("adjudicator", prompts.ADJUDICATOR_SYSTEM[s.adjudicator], user + suffix,
                                            s.adjudicator_tokens, usage))
        verdict = str(parsed.get("verdict", "")).upper().strip() if isinstance(parsed, dict) else ""
        if verdict in ("SUPPORTED", "CONTRADICTED", "INSUFFICIENT"):
            break
        invalid += 1
    else:
        return None, {"valid": False, "shown": [p.pid for p in shown]}, invalid
    verdict = INSUFFICIENT if verdict == "INSUFFICIENT" else verdict
    cited = str(parsed.get("passage", "")).strip().strip("[]")
    text = next((p.text for p in shown if p.pid == cited), "")
    ok = verdict == INSUFFICIENT or grounded(parsed.get("quote"), text)
    return (verdict if ok else None), {"valid": True, "verdict": verdict, "passage": cited, "grounded": ok,
                                       "shown": [p.pid for p in shown]}, invalid


def _result(prediction, status, usage, retriever, start, passages, stances, **extra) -> Result:
    return Result(prediction=prediction, status=status, usage=usage.as_dict(), retrieval_calls=retriever.calls,
                  latency_seconds=round(time.perf_counter() - start, 3),
                  passages=[{k: v for k, v in asdict(p).items() if k != "text"} for p in passages],
                  stances=[asdict(st) for st in stances], **extra)


def run_single_pass(claim: str, retriever: Retriever, client: ChatClient, s: Settings, k: Optional[int] = None,
                    adjudicated: bool = False) -> Result:
    start, usage = time.perf_counter(), Usage()
    retriever.calls = 0
    passages: List[Passage] = []
    stances: List[Stance] = []
    try:
        passages = retriever.search(claim, k or s.k, "claim")
        _number(passages)
        stances, invalid, ungrounded = judge(claim, passages, client, usage, s)
        verdict = aggregate(stances)
        adjudication = None
        if adjudicated and not (s.trigger == "unresolved" and verdict != INSUFFICIENT):
            decided, adjudication, more_invalid = adjudicate(claim, passages, stances, client, usage, s)
            invalid += more_invalid
            final = decided or _fallback(passages, stances, s)
        else:
            final = verdict
        return _result(final, "ok", usage, retriever, start, passages, stances, adjudication=adjudication,
                       aggregate_verdict=verdict, invalid_outputs=invalid, ungrounded_quotes=ungrounded)
    except LLMError as exc:
        return _result(FAILED, str(exc) or "LLM_ERROR", usage, retriever, start, passages, stances)


def run_agentic(claim: str, retriever: Retriever, client: ChatClient, s: Settings) -> Result:
    start, usage = time.perf_counter(), Usage()
    retriever.calls = 0
    passages: List[Passage] = []
    stances: List[Stance] = []
    actions: List[Dict[str, Any]] = []
    invalid = ungrounded = 0
    try:
        passages = retriever.search(claim, s.k, "claim")
        _number(passages)
        stances, invalid, ungrounded = judge(claim, passages, client, usage, s)
        first = aggregate(stances)
        if s.trigger == "unresolved" and first != INSUFFICIENT:
            return _result(first, "ok", usage, retriever, start, passages, stances, actions=[{"turn": 0, "action":
                           "RESOLVED"}], aggregate_verdict=first, invalid_outputs=invalid, ungrounded_quotes=ungrounded)
        seen = {p.chunk_id for p in passages}
        queries: List[str] = []
        conflicting = any(st.stance != "NEUTRAL" for st in stances)
        turns = 0 if (s.search_when == "empty" and conflicting) else s.max_turns
        if turns == 0:
            actions.append({"turn": 0, "action": "ADJUDICATE_CONFLICT"})
        for turn in range(turns):
            searches_left = s.max_searches - len(queries)
            if s.skip_exhausted_planner and searches_left <= 0:
                actions.append({"turn": turn, "action": "ADJUDICATE_NO_SEARCH_LEFT"})
                break
            stance_of = {st.pid: st for st in stances}
            findings = [f"{p.pid} {stance_of[p.pid].stance}: {p.text[:300]}" for p in passages
                        if stance_of[p.pid].stance != "NEUTRAL"]
            action, query = None, None
            for attempt in range(2):
                suffix = "" if attempt == 0 else "\n\nReturn valid JSON with one action."
                parsed = parse_json(client.complete(
                    "planner", prompts.PLANNER_SYSTEM[s.planner],
                    prompts.planner_user(claim, findings, searches_left, queries) + suffix, s.planner_tokens, usage))
                if isinstance(parsed, dict) and parsed.get("action") in ("SEARCH", "ADJUDICATE"):
                    action = parsed["action"]
                    query = parsed.get("query") if isinstance(parsed.get("query"), str) else None
                    if action == "ADJUDICATE" or (query and query.strip()):
                        break
                invalid += 1
                action = None
            if action is None:
                actions.append({"turn": turn, "action": "INVALID"})
                break
            if action == "ADJUDICATE" or searches_left <= 0:
                actions.append({"turn": turn, "action": action if action == "ADJUDICATE" else "SEARCH_REFUSED"})
                break
            query = query.strip()[:300]
            queries.append(query)
            found = [p for p in retriever.search(query, s.search_k * 2, f"search{len(queries)}")
                     if p.chunk_id not in seen][:s.search_k]
            for p in found:
                seen.add(p.chunk_id)
            offset = len(passages)
            for i, p in enumerate(found, start=offset + 1):
                p.pid = f"P{i}"
            new_stances, more_invalid, more_ungrounded = judge(claim, found, client, usage, s) if found else ([], 0, 0)
            invalid += more_invalid
            ungrounded += more_ungrounded
            passages.extend(found)
            stances.extend(new_stances)
            actions.append({"turn": turn, "action": "SEARCH", "query": query, "new_passages": len(found)})
        verdict = aggregate(stances)
        if s.trigger == "unresolved" and verdict != INSUFFICIENT:
            decided, adjudication = verdict, None             # the searches resolved it: unanimous evidence stands
        else:
            decided, adjudication, more_invalid = adjudicate(claim, passages, stances, client, usage, s)
            invalid += more_invalid
        return _result(decided or _fallback(passages, stances, s), "ok", usage, retriever, start, passages, stances,
                       actions=actions,
                       adjudication=adjudication, aggregate_verdict=verdict, invalid_outputs=invalid,
                       ungrounded_quotes=ungrounded)
    except LLMError as exc:
        return _result(FAILED, str(exc) or "LLM_ERROR", usage, retriever, start, passages, stances, actions=actions)
