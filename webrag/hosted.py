"""The hosted app's quick source search and saved-report viewer (``streamlit_cloud/tools_page.py``): source search
without a model and checks on an uploaded run record. Research itself runs in the Research Studio
(``webrag/hosted_studio.py``). Nothing here imports Streamlit or a model."""
from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from .net import PoliteSession
from .search import default_providers, excluded_host, provider_catalog

# Evidence providers that can search from a query alone. RSS/Atom feeds need a feed list, which only the Studio keeps.
SEARCHABLE = ("openalex", "arxiv", "crossref", "europepmc", "searxng", "brave")
MAX_QUERY_CHARS = 300
MAX_RESULTS = 10
MAX_RUN_BYTES = 20_000_000
LABELS = ("SUPPORTED", "CONTRADICTED", "INSUFFICIENT")


def search_providers() -> List[Dict[str, Any]]:
    """The searchable evidence providers with their configuration state (secrets only as set / not set)."""
    return [{"id": row["id"], "name": row["name"], "configured": row["configured"], "missing": row["missing"],
             "returns": row["returns"], "limits": row["limits"]}
            for row in provider_catalog({}) if row["id"] in SEARCHABLE]


# ================================================================================================ source search
def search_sources(query: str, providers: List[str], limit: int = 5, since: Optional[str] = None,
                   session: Optional[PoliteSession] = None) -> Dict[str, Any]:
    """Search results from each provider, as the providers return them. No model reads or judges them; a provider
    that fails is reported with its error and the others still run. ``since`` (YYYY-MM-DD) is sent to providers that
    filter by date; results of the others (``since_ignored``) are filtered by their recorded date, and results
    without a date are kept."""
    query = " ".join(str(query or "").split())
    if not query:
        raise ValueError("enter a search query")
    if len(query) > MAX_QUERY_CHARS:
        raise ValueError(f"the query is longer than {MAX_QUERY_CHARS} characters")
    if not providers:
        raise ValueError("choose at least one source to search")
    unknown = [p for p in providers if p not in SEARCHABLE]
    if unknown:
        raise ValueError(f"not searchable here: {', '.join(unknown)}")
    if not isinstance(limit, int) or not 1 <= limit <= MAX_RESULTS:
        raise ValueError(f"results per source must be between 1 and {MAX_RESULTS}")
    session = session or PoliteSession()
    hits: List[Dict[str, Any]] = []
    errors: Dict[str, str] = {}
    since_ignored: List[str] = []
    for name in providers:
        try:
            [provider] = default_providers(session, [name], config={})
            kwargs = {}
            if since:
                if getattr(provider, "supports_since", False):
                    kwargs["since"] = since
                else:
                    since_ignored.append(name)
            for hit in provider.search(query, limit=limit, **kwargs):
                if name in since_ignored and hit.published and hit.published[:10] < since:
                    continue                    # no server-side date filter: filtered by the recorded date instead
                row = asdict(hit)
                row["excluded_host"] = excluded_host(hit.url)
                hits.append(row)
        except Exception as exc:        # one provider's failure (network, quota, bad response) never hides the rest
            errors[name] = f"{type(exc).__name__}: {exc}"
    return {"query": query, "providers": list(providers), "limit": limit, "since": since, "hits": hits,
            "errors": errors, "since_ignored": since_ignored, "requests_made": getattr(session, "requests_made", None)}


# ================================================================================================ saved run records
def load_run(raw: bytes) -> Dict[str, Any]:
    """A run record exported from the Studio or the command line, checked for the fields the viewer reads."""
    if len(raw) > MAX_RUN_BYTES:
        raise ValueError(f"the file is larger than {MAX_RUN_BYTES // 1_000_000} MB")
    try:
        data = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"not a JSON file: {exc}") from None
    problems = run_problems(data)
    if problems:
        raise ValueError("not a run record: " + "; ".join(problems))
    return data


def run_problems(data: Any) -> List[str]:
    if not isinstance(data, dict):
        return ["the file does not hold a JSON object"]
    problems = []
    if data.get("mode") not in ("research", "verify"):
        problems.append("'mode' must be 'research' or 'verify'")
    for key in ("input", "status"):
        if not isinstance(data.get(key), str):
            problems.append(f"'{key}' is missing")
    for key in ("claims", "sources"):
        if not isinstance(data.get(key, []), list):
            problems.append(f"'{key}' must be a list")
    if not problems:
        if any(not isinstance(s, dict) or "id" not in s for s in data.get("sources", [])):
            problems.append("every source needs an 'id'")
        if any(not isinstance(c, dict) or "text" not in c or "label" not in c for c in data.get("claims", [])):
            problems.append("every claim needs 'text' and 'label'")
    return problems


def label_counts(data: Dict[str, Any]) -> Dict[str, int]:
    counts = {label: 0 for label in LABELS}
    for claim in data.get("claims", []):
        counts[claim["label"]] = counts.get(claim["label"], 0) + 1
    return counts


def claim_rows(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [{"id": c.get("id", ""), "statement": c["text"], "verdict": c["label"], "detail": c.get("sub_state") or "",
             "evidence label": c.get("strength") or "",
             "supporting sources": ", ".join(c.get("supporting_sources") or []),
             "contradicting sources": ", ".join(c.get("contradicting_sources") or []),
             "cautions": ", ".join(str(w.get("kind", "")).replace("_", " ") for w in c.get("cautions") or [])}
            for c in data.get("claims", [])]


def source_rows(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [{"id": s["id"], "title": s.get("title") or "", "provider": s.get("provider") or "",
             "access": s.get("access") or "", "published": str(s.get("published") or "")[:10],
             "url": s.get("url") or ""}
            for s in data.get("sources", [])]
