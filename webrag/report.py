"""Plain-text/Markdown rendering of a run (the web interface renders the same JSON)."""
from __future__ import annotations

from typing import Any, Dict, List

LABEL_TEXT = {"SUPPORTED": "supported", "CONTRADICTED": "contradicted by the evidence",
              "INSUFFICIENT": "not established by the evidence"}


def _links(ids: List[str], sources: Dict[str, Dict[str, Any]]) -> str:
    return ", ".join(f"[{i}]({sources[i]['url']})" for i in ids if i in sources) or "no source"


def _excerpt(text: str, limit: int = 280) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit - 3] + "..."


def _support_basis(claim: Dict[str, Any], count_key: str = "independent_support") -> str:
    """Deep research: how independent the supporting publishers are (from metadata). Otherwise the original count."""
    independence = claim.get("independence")
    if not isinstance(independence, dict):
        return f"{claim.get(count_key, 0)} independent source(s)"
    status, groups = independence.get("status"), independence.get("groups", 0)
    # deep-research-v4: sources the verifier judged supporting whose passage does not cover the finding
    uncounted = [r["source"] for r in claim.get("support_check") or [] if not r.get("counted")]
    note = f"; {', '.join(uncounted)} not counted (passage does not cover the statement)" if uncounted else ""
    if status == "established":
        return f"{independence.get('independent', 0)} publishers established as independent{note}"
    if status == "unknown":
        cites = any(p.get("relation") == "dependent" for p in independence.get("pairs") or [])
        return (f"{independence.get('sources', 0)} sources from {groups} publishers, independence not established"
                + ("; one cites another" if cites else "") + note)
    return ("one publisher" if status == "single" else "no supporting passage covers the statement") + note


def _answer_sections(data: Dict[str, Any]) -> List[str]:
    """Deep research: the direct answer and summary, each sentence pointing to the verified findings it restates."""
    summary = data.get("summary") or {}
    if data.get("mode") != "research" or not summary.get("available"):
        return ["", "## Direct answer", summary.get("reason") or "No answer was established."] if data.get("depth") and data.get("mode") == "research" else []
    findings = [c for c in data.get("claims", []) if c.get("label") == "SUPPORTED"]
    numbers = {id(c): i for i, c in enumerate(findings, 1)}

    def refs(part: Dict[str, Any]) -> str:
        claims = data.get("claims", [])
        nums = [numbers.get(id(claims[i])) for i in part.get("claims", []) if 0 <= i < len(claims)]
        nums = [str(n) for n in nums if n]
        return f" (finding{'s' if len(nums) > 1 else ''} {', '.join(nums)})" if nums else ""

    lines = ["", "## Direct answer"]
    answer = summary.get("direct_answer")
    lines.append((answer["text"] + refs(answer)) if answer else "The verified findings do not add up to a direct answer.")
    if summary.get("summary"):
        lines += ["", "## Summary"] + [f"- {p['text']}{refs(p)}" for p in summary["summary"]]
    lines.append("")
    lines.append("_Written by the local model from verified findings only; each sentence names the findings it restates._")
    return lines


def _deep_sections(data: Dict[str, Any]) -> List[str]:
    if not data.get("depth"):
        return []
    lines = []
    ledger = data.get("ledger") or []
    if ledger:
        lines += ["", "## Coverage by question", "", "| Question | Status | Why |", "|---|---|---|"]
        for entry in ledger:
            text = entry["text"].replace("|", "/")
            lines.append(f"| {text} | {entry['status']} | {entry['reason'].replace('|', '/')} |")
    quality = data.get("source_quality") or {}
    if quality:
        access = quality.get("by_access", {})
        diversity = (f"distinct texts (near-copies merged): {quality['distinct_texts']}; publisher groups: "
                     f"{quality.get('publisher_groups', 0)} (independence is judged per finding)"
                     if "distinct_texts" in quality else f"independent origins among them: {quality.get('independent_origins', 0)}")
        lines += ["", "## Evidence quality",
                  f"- Sources found: {quality.get('discovered', 0)}; read (full text or abstract): {quality.get('read', 0)}; "
                  f"{diversity}.",
                  f"- Full text {access.get('FULL_TEXT', 0)}, abstract only {access.get('ABSTRACT_ONLY', 0)}, snippet only "
                  f"{access.get('SNIPPET_ONLY', 0)} (discovery only), metadata only {access.get('METADATA_ONLY', 0)}, "
                  f"failed {access.get('FAILED', 0)}.",
                  "- Source types read: " + (", ".join(f"{k} {v}" for k, v in (quality.get("by_category") or {}).items()) or "none") + "."]
    stop = data.get("stopping") or {}
    rounds = data.get("rounds") or []
    budget = (data.get("budget") or {})
    lines += ["", "## Method and stopping reason",
              f"- Depth: {data['depth']}; searched with {', '.join((data.get('options') or {}).get('providers') or []) or 'no provider'}."]
    for r in rounds:
        queries = "; ".join(f"{q['query']} [{q['category']}{'' if q.get('purpose') == 'answer' else ', ' + q['purpose']}]"
                            for q in r.get("queries", []))
        lines.append(f"- Round {r['number']}{' (counter-evidence)' if r.get('purpose') == 'counter-evidence' else ''}: "
                     f"{r['searches']} searches, {r['new_sources']} new sources, {r['indexed_sources']} read. Queries: {queries or 'none'}.")
    if budget.get("used"):
        used, limits = budget["used"], budget.get("limits", {})
        lines.append(f"- Budget used: {used.get('searches', 0)}/{limits.get('searches', '?')} searches, "
                     f"{used.get('sources', 0)}/{limits.get('sources', '?')} sources, "
                     f"{used.get('full_text', 0)}/{limits.get('full_text', '?')} full-text attempts.")
    if stop:
        lines.append(f"- Stopped: {stop.get('detail', '')} Result: {stop.get('completeness', '').replace('_', '-')}.")
        if stop.get("also_reached"):
            lines.append("- Limits also reached when searching stopped: "
                         + ", ".join(r.replace("_", " ") for r in stop["also_reached"]) + ".")
        retrieval, answers = stop.get("retrieval_coverage"), stop.get("answer_coverage")
        if isinstance(retrieval, dict) and isinstance(answers, dict):
            lines.append(f"- Retrieval coverage (passages found, not answers): {'met' if retrieval.get('met') else 'not met'}. "
                         f"Answer coverage (verified): {answers.get('answered', 0)} of {answers.get('questions', 0)} "
                         f"question(s) answered" + (", open: " + ", ".join(
                             f"{'main question' if o['index'] == 0 else 'question %d' % o['index']} ({o['status']})"
                             for o in answers.get("open") or []) if answers.get("open") else "") + ".")
    lines.append("- Verdicts come from the verifier; evidence labels follow the documented evidence policy.")
    lines += _execution_lines(data)
    items = data.get("review") or []
    if items:
        lines += ["", "## Evidence review",
                  "_A deterministic restatement of the recorded checks, not an independent verification._"]
        for item in items:
            who = f"[{item['claim']}] " if item.get("claim") else ""
            refs = f" (refs: {', '.join(item['refs'][:4])})" if item.get("refs") else ""
            lines.append(f"- {who}{item.get('detail', '')} — {item.get('action', '').replace('_', ' ')}{refs}")
    return lines


def _execution_lines(data: Dict[str, Any]) -> List[str]:
    """deep-research-v3: what was searched, as recorded: the query ledger and each provider's state."""
    queries, providers = data.get("queries") or [], data.get("provider_coverage") or []
    if not queries and not providers:
        return []
    lines = ["", "## Search execution"]
    if queries:
        counts: Dict[str, int] = {}
        for q in queries:
            counts[q.get("status", "")] = counts.get(q.get("status", ""), 0) + 1
        lines.append("- Queries: " + ", ".join(f"{n} {s.replace('_', ' ')}" for s, n in counts.items()) + ".")
        counter = [q for q in queries if q.get("purpose") == "counter"]
        if counter:
            ran = sum(q.get("status") == "executed" for q in counter)
            lines.append(f"- Counter-evidence queries: {ran} of {len(counter)} executed.")
        for q in queries:
            if q.get("status") != "executed":
                lines.append(f"- {q['id']} {q.get('status', '').replace('_', ' ')}: “{q['query']}” — {q.get('reason', '')}")
    if providers:
        groups: Dict[str, List[str]] = {}
        for p in providers:
            groups.setdefault(p.get("state", ""), []).append(p.get("name", p.get("id", "")))
        lines.append("- Providers: " + "; ".join(f"{s.replace('_', ' ')}: {', '.join(names)}" for s, names in groups.items()) + ".")
    return lines


def _attestation_section(data: Dict[str, Any]) -> List[str]:
    """Opt-in record-only quote attestation (docqa/attestation.py): where each verifier quote was found in its
    passage. Location only: it never changed a stance or verdict, and a located quote may still not support a claim."""
    rows = []
    for c in data.get("claims") or []:
        for p in c.get("passages") or []:
            for entry in p.get("quote_attestation") or []:
                found = entry.get("attestation") or {}
                status = (found.get("status") or "error") if entry.get("assessed") else "not assessed"
                tier = f" ({found['tier'].lower()})" if found.get("tier") else ""
                cue = next((s["negation_cue"] for s in found.get("spans") or [] if s.get("negation_cue")), None)
                legacy = {True: "accepted", False: "rejected"}.get(entry.get("legacy_quote_check"), "n/a")
                rows.append(f"| {_excerpt(c['text'], 60).replace('|', '/')} | {p.get('key', '')} | {entry.get('stance', '')} | "
                            f"{status.lower().replace('_', ' ')}{tier} | {legacy} | {cue or ''} |")
    if not rows:
        return []
    return ["", "## Quote attestation (record only)",
            "Where each verifier quote occurs in the passage it judged. This never changed a verdict; a located quote "
            "can still fail to support the claim.", "",
            "| Statement | Passage | Stance | Quote location | Verifier's own quote check | Negation just before |",
            "|---|---|---|---|---|---|"] + rows


def render_markdown(data: Dict[str, Any]) -> str:
    sources = {s["id"]: s for s in data.get("sources", [])}
    lines = [f"# {'Claim verification' if data['mode'] == 'verify' else 'Research report'}", "",
             f"**Input:** {data['input']}", f"**Status:** {data['status']}    **Outcome:** {data.get('outcome')}"]
    if data.get("error"):
        lines.append(f"**Error:** {data['error']}")
    lines += _answer_sections(data)
    claims = data.get("claims", [])
    if data["mode"] == "verify" and len(claims) > 1:              # several claims, each with its own verdict
        lines += ["", "## Claims", "", "| Claim | Verdict | Evidence label | Cautions |", "|---|---|---|---|"]
        for c in claims:
            lines.append(f"| {c.get('id', '')} {c['text'].replace('|', '/')} | {c['label']}"
                         + (f" ({c['sub_state']})" if c.get("sub_state") else "") + f" | {c.get('strength', '')} | "
                         + (", ".join(w["kind"].replace("_", " ") for w in c.get("cautions") or []) or "none") + " |")
        for c in claims:
            lines += ["", f"### {c.get('id', '')} {c['text']}",
                      f"Verdict: {c['label']}. Supporting sources: {_links(c['supporting_sources'], sources)} "
                      f"({_support_basis(c)}); contradicting sources: {_links(c['contradicting_sources'], sources)}."]
            lines += [f"- {p['stance'].lower().replace('_', ' ')}: [{p['source_id']}]({p['url']}) {_excerpt(p['text'], 200)}"
                      for p in c["passages"] if p.get("stance") in {"SUPPORTS", "CONTRADICTS"}]
    elif data["mode"] == "verify" and claims:
        c = claims[0]
        support = (_support_basis(c) if isinstance(c.get("independence"), dict)
                   else f"{c['independent_support']} independent")
        lines += ["", f"## Verdict: {c['label']}" + (f" ({c['sub_state']})" if c["sub_state"] else ""),
                  f"Supporting sources: {_links(c['supporting_sources'], sources)} "
                  f"({support}); contradicting sources: "
                  f"{_links(c['contradicting_sources'], sources)} ({c['independent_contradiction']} independent)."]
        for stance in ("SUPPORTS", "CONTRADICTS", "NEUTRAL", "NOT_ASSESSED"):
            rows = [p for p in c["passages"] if p["stance"] == stance]
            if rows:
                lines += ["", f"### Passages judged {stance}"]
                lines += [f"- [{p['source_id']}]({p['url']}) {_excerpt(p['text'])}" for p in rows]
    elif claims:
        supported = [c for c in claims if c["label"] == "SUPPORTED"]
        others = [c for c in claims if c["label"] != "SUPPORTED"]
        lines += ["", "## Verified findings"]
        if not supported:
            lines.append("No finding could be verified against the retrieved evidence.")
        for i, c in enumerate(supported, 1):
            label = f"; {c['strength']}" if c.get("strength") else ""
            lines.append(f"{i}. {'(' + c['id'] + ') ' if c.get('id') else ''}{c['text']} — "
                         f"{_links(c['supporting_sources'], sources)} ({_support_basis(c)}{label})")
            # deep research: warnings from fixed text checks; the finding is kept out of the direct answer and summary
            lines += [f"   - Caution ({w['kind'].replace('_', ' ')}): {w['detail']} Not used in the direct answer or summary."
                      for w in c.get("cautions") or []]
        if others:
            lines += ["", "## Statements not verified"]
            for c in others:
                basis = c["contradicting_sources"] if c["label"] == "CONTRADICTED" else c["cited_sources"]
                lines.append(f"- {c['text']} — {LABEL_TEXT[c['label']]}"
                             + (f" ({c['sub_state'].replace('_', ' ').lower()})" if c["sub_state"] else "")
                             + f"; sources: {_links(basis, sources)}")
    if data.get("disagreements"):
        lines += ["", "## Disagreements between sources"]
        for d in data["disagreements"]:
            tag = "detected by the verifier" if d["verified"] else "identified by the model, not independently checked"
            lines.append(f"- {d['text']} ({tag}; {_links(d['sources'], sources)})")
    if data.get("gaps"):
        lines += ["", "## Gaps"] + [f"- {g}" for g in data["gaps"]]
    if data.get("limitations"):
        lines += ["", "## Limitations"] + [f"- {x}" for x in data["limitations"]]
    lines += _deep_sections(data)
    lines += _attestation_section(data)
    if sources:
        lines += ["", "## Sources"]
        for s in sources.values():
            date = f", {s['published'][:10]}" if s.get("published") else ""
            where = s.get("publisher") or s.get("domain") or ""
            ids = "".join(f"; {k.upper()} {v}" for k, v in (s.get("identifiers") or {}).items() if k in {"doi", "arxiv", "pmid", "pmcid"})
            authors = f"{', '.join(s['authors'][:3])}{' et al.' if len(s['authors']) > 3 else ''}. " if s.get("authors") else ""
            lines.append(f"- **{s['id']}** {authors}[{s['title']}]({s['url']}) — {where + ', ' if where else ''}"
                         f"{s['provider']}, {s.get('category') or s['kind']}{date}{ids}; "
                         f"{s['access'].replace('_', ' ').lower()}; retrieved {s['retrieved_at']}"
                         + (f". {s['note']}" if s.get("note") else ""))
    if data.get("stages"):
        lines += ["", "## Pipeline stages"]
        lines += [f"- {s['name']}: {s['status']} ({s['seconds']} s) {s['detail']}" for s in data["stages"]]
    if data.get("saved_to"):
        lines += ["", f"Run record: {data['saved_to']}"]
    return "\n".join(lines)
