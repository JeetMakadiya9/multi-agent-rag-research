"""Turn a saved run record (outputs/webrag/runs/<name>/run.json) into a publishable showcase example.

    python showcase/make_examples.py <run.json> [<run.json> ...]

Kept: the question or claim, recorded stages, the model-written plan, statements and summary, the verifier's verdicts
and per-passage stances (as source ids), program-written cautions and limitations, coverage counts, and bibliographic
metadata of each source (title, link, authors, provider, date, access).

Removed: all third-party text (full source texts, pages, passage text, the evidence pool), notes the verifier may have
quoted from passages, raw search-error messages, saving paths, and service addresses of the recording computer. The
result is written to showcase/data/examples/<name>.json.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

OUT = Path(__file__).resolve().parent / "data" / "examples"
SOURCE_FIELDS = ("id", "title", "url", "provider", "kind", "category", "access", "published", "publisher", "domain",
                 "authors", "identifiers", "round")
CLAIM_FIELDS = ("id", "text", "label", "sub_state", "strength", "subquestion", "supporting_sources",
                "contradicting_sources", "cited_sources", "independent_support", "independent_contradiction")
_LOOPBACK = re.compile(r"\s*\(?https?://(?:127\.0\.0\.1|localhost)(?::\d+)?/?\)?")


def _clean(text: Any) -> str:
    return _LOOPBACK.sub("", str(text or "")).strip()


def example(data: Dict[str, Any], name: str) -> Dict[str, Any]:
    model = data.get("model") if isinstance(data.get("model"), dict) else {}
    settings = data.get("settings") if isinstance(data.get("settings"), dict) else {}
    claims: List[Dict[str, Any]] = []
    for claim in data.get("claims") or []:
        if not isinstance(claim, dict):
            continue
        row = {key: claim.get(key) for key in CLAIM_FIELDS}
        row["cautions"] = [{"kind": w.get("kind"), "detail": _clean(w.get("detail"))}
                           for w in claim.get("cautions") or [] if isinstance(w, dict)]
        row["passages"] = [{"source_id": p.get("source_id"), "stance": p.get("stance"), "page": p.get("page")}
                           for p in claim.get("passages") or [] if isinstance(p, dict)]
        claims.append(row)
    summary = data.get("summary") if isinstance(data.get("summary"), dict) else None
    plan = data.get("research_plan") if isinstance(data.get("research_plan"), dict) else {}
    return {
        "name": name,
        "kind": "recorded output",
        "about": "A real run of the Research Studio, recorded on the author's computer. Third-party text is not "
                 "included: open each source through its link.",
        "removed": ["full source texts and pages", "passage and evidence-pool text", "verifier notes",
                    "raw search-error messages", "saving paths and local service addresses"],
        "input": data.get("input"), "mode": data.get("mode"), "status": data.get("status"),
        "outcome": data.get("outcome"), "started": data.get("started"), "finished": data.get("finished"),
        "depth": data.get("depth"),
        "model": {"model": model.get("model") or settings.get("model"),
                  "runtime": "local Ollama " + str(model.get("ollama_version") or model.get("service_version") or "")
                  if not model.get("provider") or model.get("provider") == "ollama" else model.get("provider_label")},
        "verifier": settings.get("verifier"),
        "stages": [{"name": s.get("name"), "status": s.get("status"), "seconds": s.get("seconds"),
                    "detail": _clean(s.get("detail"))} for s in data.get("stages") or [] if isinstance(s, dict)],
        "plan": {"subquestions": [q.get("text") for q in plan.get("subquestions") or [] if isinstance(q, dict)],
                 "queries": [q.get("query") for q in plan.get("queries") or [] if isinstance(q, dict)],
                 "counter_queries": [q.get("query") for q in plan.get("counter_queries") or [] if isinstance(q, dict)]},
        "sources": [{key: s.get(key) for key in SOURCE_FIELDS} for s in data.get("sources") or [] if isinstance(s, dict)],
        "claims": claims,
        "summary": {"direct_answer": (summary.get("direct_answer") or {}).get("text"),
                    "direct_answer_claims": (summary.get("direct_answer") or {}).get("claims") or [],
                    "points": [{"text": p.get("text"), "claims": p.get("claims") or []}
                               for p in summary.get("summary") or [] if isinstance(p, dict)]} if summary else None,
        "coverage": [{"question": r.get("text"), "status": r.get("status"), "evidence": r.get("evidence_state"),
                      "passages": len(r.get("passages") or []), "supporting": len(r.get("supporting") or []),
                      "contradicting": len(r.get("contradicting") or []),
                      "independent_publishers": r.get("independent_publishers")}
                     for r in data.get("ledger") or [] if isinstance(r, dict)],
        "stopping": {k: (data.get("stopping") or {}).get(k) for k in ("reason", "detail", "completeness")}
        if isinstance(data.get("stopping"), dict) else None,
        "source_quality": data.get("source_quality") if isinstance(data.get("source_quality"), dict) else None,
        "search_errors": len(data.get("search_errors") or []),
        "limitations": [_clean(x) for x in data.get("limitations") or []],
        "gaps": [_clean(x) for x in data.get("gaps") or []],
        "review": [{"kind": r.get("kind"), "detail": _clean(r.get("detail"))}
                   for r in data.get("review") or [] if isinstance(r, dict)],
    }


def main(paths: List[str]) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for path in map(Path, paths):
        data = json.loads(path.read_text(encoding="utf-8"))
        name = path.parent.name
        target = OUT / f"{name}.json"
        target.write_text(json.dumps(example(data, name), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"{target.relative_to(OUT.parent.parent.parent)}  ({target.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
