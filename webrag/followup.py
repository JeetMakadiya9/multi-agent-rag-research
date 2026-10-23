"""Bounded, verifier-driven follow-up research (inspired by Open Deep Research), opt-in.

When a finding is INSUFFICIENT because no passage was decisive (or only part of it was supported), the gap is a
specific, searchable fact. One extra round may then search for it:
* at most ``max_findings`` findings trigger it; CONFLICTING, failed assessments and service errors never do;
* at most 2 queries, written for those findings, never repeating the run's earlier queries;
* at most ``max_new_sources`` new sources, fetched by the same researcher (robots.txt, SSRF and rate-limit rules),
  de-duplicated against the run's sources (same work skipped, near-copies linked to their origin);
* only the triggering findings are re-ranked (over old plus new passages) and re-verified; every other finding is
  untouched. Each change keeps the original verdict and says what changed and why.
New sources are appended with ``followup_round: 1``; the original sources and passages stay as they were.

Inline: ``Settings(followup_rounds=1)``. On a saved run (writes a new derived run, never modifies the original):
    python -m webrag.followup --run <folder under outputs/webrag/runs>
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import time
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from docqa.llm import LLMError
from docqa.verify import INSUFFICIENT, SUPPORTED

from .agents import strip_provider_names
from .evidence import EvidenceStore, Passage, merge_passages
from .researcher import Source, WebResearcher
from .search import keywords

FOLLOWUP_VERSION = "followup-v1"
ELIGIBLE_SUB_STATES = {"NO_DECISIVE_EVIDENCE", "PARTLY_SUPPORTED", "NO_EVIDENCE", ""}
QUERY_SCHEMA = {"type": "object", "properties": {"queries": {"type": "array", "items": {"type": "string"}}},
                "required": ["queries"]}
QUERY_PROMPT = (
    "Some statements from a research draft could not be checked against the sources found so far. Write at most 2 "
    "short keyword search queries (3 to 8 words each) that would find an authoritative source stating or refuting "
    "them. Name the key entities and the specific fact. Include a query that could find disconfirming evidence when "
    "useful. Do not name websites or search engines and do not repeat the earlier queries. Return JSON only."
)


def eligible(claim: Dict[str, Any]) -> bool:
    return (claim.get("role", "finding") == "finding" and claim.get("label") == INSUFFICIENT
            and claim.get("parse_ok", True) and claim.get("sub_state", "") in ELIGIBLE_SUB_STATES)


def _norm(query: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", query.lower())).strip()


def plan_queries(client, statements: Sequence[str], previous: Sequence[str], max_queries: int = 2) -> List[str]:
    """Model-written queries for the gaps; falls back to the first statement's keywords. Never repeats a query."""
    seen = {_norm(q) for q in previous}
    candidates: List[str] = []
    user = "Statements:\n" + "\n".join(f"- {s}" for s in statements) + "\n\nEarlier queries:\n" + \
        "\n".join(f"- {q}" for q in previous)
    try:
        response = client.chat([{"role": "system", "content": QUERY_PROMPT}, {"role": "user", "content": user}],
                               response_format=QUERY_SCHEMA, num_predict=150)
        raw = json.loads(response.text).get("queries", [])
        candidates = [strip_provider_names(str(q)).strip()[:120] for q in raw if isinstance(q, str)]
    except (LLMError, ValueError, AttributeError, json.JSONDecodeError):
        candidates = []
    if not candidates and statements:
        candidates = [" ".join(keywords(statements[0], 8))]
    queries = []
    for q in candidates:
        if q and _norm(q) not in seen and len(q.split()) >= 2:
            seen.add(_norm(q))
            queries.append(q)
    return queries[:max_queries]


def _source_from_record(record: Dict[str, Any]) -> Source:
    allowed = {f.name for f in fields(Source)}
    return Source(**{k: v for k, v in record.items() if k in allowed})


def run_followup(data: Dict[str, Any], *, client, session, providers: Sequence, verify_fn: Callable[..., Any],
                 settings, models=None, store: Optional[EvidenceStore] = None, max_queries: int = 2,
                 max_new_sources: int = 6, max_findings: int = 2,
                 source_record: Callable[[Source], Dict[str, Any]] = asdict) -> Dict[str, Any]:
    """Return a new run dict with one follow-up round applied (``data`` itself is not modified)."""
    data = copy.deepcopy(data)
    started = time.perf_counter()
    triggered = [i for i, c in enumerate(data.get("claims", [])) if eligible(c)][:max_findings]
    record: Dict[str, Any] = {"version": FOLLOWUP_VERSION, "round": 1, "triggered_by": triggered, "queries": [],
                              "new_sources": [], "changes": [], "requests": 0, "skipped_reason": "",
                              "limits": {"max_queries": max_queries, "max_new_sources": max_new_sources,
                                         "max_findings": max_findings}}
    data["followup"] = record
    if not triggered:
        record["skipped_reason"] = "no finding was INSUFFICIENT for lack of decisive evidence"
        record["seconds"] = round(time.perf_counter() - started, 2)
        return data
    statements = [data["claims"][i]["text"] for i in triggered]
    previous = list((data.get("plan") or {}).get("queries") or []) + [data.get("input", "")]
    queries = plan_queries(client, statements, previous, max_queries)
    record["queries"] = queries
    if not queries:
        record["skipped_reason"] = "no new query could be formed"
        record["seconds"] = round(time.perf_counter() - started, 2)
        return data
    existing = [_source_from_record(s) for s in data.get("sources", [])]
    researcher = WebResearcher(providers, session, per_query=settings.per_query, max_sources=max_new_sources,
                               max_full_text=min(2, settings.max_full_text))
    corpus = researcher.gather(queries, existing=existing)
    record["requests"] = corpus.requests_made
    record["provider_status"] = corpus.provider_status
    record["new_sources"] = [s.id for s in corpus.sources]
    for s in corpus.sources:
        data.setdefault("sources", []).append(source_record(s) | {"followup_round": 1})
    usable_new = [s for s in corpus.sources if s.usable]
    if store is None:
        sources = [s for s in existing if s.usable] + usable_new
        guard = getattr(settings, "injection_guard", "off")
        if guard == "off":
            store = EvidenceStore(sources, models)
        else:
            from .guarded import GuardedEvidenceStore
            store = GuardedEvidenceStore(sources, models, guard=guard)
    else:
        store.add_sources(usable_new)
    new_ids = {s.id for s in usable_new}
    for index in triggered:
        before = data["claims"][index]
        old = [Passage(p["key"], p["source_id"], p["page"], p["text"], p.get("score", 0.0))
               for p in before.get("passages", [])]
        ranked = store.rank(before["text"], k=settings.claim_passages)
        passages = merge_passages(ranked, old, limit=settings.claim_passages)
        after = asdict(verify_fn(before["text"], "finding", passages, store, before.get("cited_sources", [])))
        used_new = sorted({p.source_id for p in passages if p.source_id in new_ids})
        changed = after["label"] != before["label"]
        if changed:
            reason = (f"{before['label']} -> {after['label']} after re-verification with passages from follow-up "
                      f"sources {used_new}" if used_new else
                      f"{before['label']} -> {after['label']} after re-ranking (no follow-up source among the passages)")
        else:
            reason = ("unchanged: no follow-up source reached the top passages" if not used_new else
                      f"unchanged although follow-up sources {used_new} were judged")
        record["changes"].append({"claim_index": index, "text": before["text"], "changed": changed, "reason": reason,
                                  "followup_sources_used": used_new,
                                  "before": {k: before.get(k) for k in ("label", "sub_state", "supporting_sources",
                                                                       "contradicting_sources")},
                                  "after": {k: after.get(k) for k in ("label", "sub_state", "supporting_sources",
                                                                     "contradicting_sources")},
                                  "original_claim": before})
        data["claims"][index] = after
    if data.get("mode") == "research":
        data["outcome"] = "ANSWERED" if any(c.get("label") == SUPPORTED for c in data["claims"]) \
            else "INSUFFICIENT_EVIDENCE"
    changed = [c for c in record["changes"] if c["changed"]]
    data.setdefault("limitations", []).append(
        f"Follow-up research (1 round, {len(queries)} queries, {len(usable_new)} new readable sources) re-checked "
        f"{len(triggered)} finding(s); {len(changed)} changed verdict.")
    record["seconds"] = round(time.perf_counter() - started, 2)
    return data


def main(argv=None) -> int:
    from .orchestrator import RUNS_DIR, Orchestrator, Settings

    parser = argparse.ArgumentParser(prog="python -m webrag.followup")
    parser.add_argument("--run", required=True, help="folder name under outputs/webrag/runs")
    args = parser.parse_args(argv)
    source = RUNS_DIR / args.run / "run.json"
    data = json.loads(source.read_text(encoding="utf-8"))
    if data.get("followup"):
        raise SystemExit("This run already contains a follow-up round; follow-up is limited to one round.")
    saved = data.get("settings") or {}
    settings = Settings(model=saved.get("model", "qwen3:4b"), verifier=saved.get("verifier", "quote_stance"),
                        providers=saved.get("providers"), save=True)
    orchestrator = Orchestrator(settings)
    result = run_followup(data, client=orchestrator.client, session=orchestrator.session,
                          providers=orchestrator.providers, verify_fn=orchestrator._verify, settings=settings,
                          source_record=orchestrator._source_record)
    result["derived_from"] = args.run
    result["settings"] = dict(saved) | {"followup_rounds": 1, "followup_version": FOLLOWUP_VERSION}
    folder = orchestrator._save(result)
    print(json.dumps({k: result["followup"][k] for k in ("triggered_by", "queries", "new_sources", "requests",
                                                          "seconds", "skipped_reason")}, indent=1))
    for change in result["followup"]["changes"]:
        print(f"- [{change['claim_index']}] {change['reason']}: {change['text'][:90]}")
    print(f"Saved: {folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
