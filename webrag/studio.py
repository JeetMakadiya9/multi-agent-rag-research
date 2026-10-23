"""Data services for the Research Studio web interface.

Everything here reads what the research pipeline, the document assistant and the evaluation harness already
produce. Nothing in this module runs a model, changes a run record, re-scores an evaluation or alters research
logic. The document assistant is built (and questioned) only when the user asks for it, through ``docqa``'s own
``build_assistant``.

* saved research runs:    ``outputs/webrag/runs/<name>/run.json`` (written by ``Orchestrator._save``);
* recorded evaluations:   ``evaluation/webrag/runs/<name>/{manifest,metrics}.json``, ``summary.txt``, ``INCOMPLETE.md``
                          (written by the evaluation harness; per-item ``records.jsonl`` files are never read here);
* document library:       the folder ``docqa`` reads (``data/documents`` by default).
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "outputs" / "webrag" / "runs"
EVAL_DIR = ROOT / "evaluation" / "webrag"
DOCS_DIR = ROOT / "data" / "documents"

STUDIO_VERSION = "0.2.0"              # version of the studio interface, independent of the research pipeline
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
CONTEXT_CHARS = 1200
ACCESS_ORDER = ("FULL_TEXT", "ABSTRACT_ONLY", "SNIPPET_ONLY", "FAILED")
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
UPLOAD_SUFFIXES = (".txt", ".pdf")


class StudioError(Exception):
    """A request the studio cannot serve; ``status`` is the HTTP status to report."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def _child(folder: Path, name: str) -> Path:
    """``folder / name`` for a plain, single-component name; anything else is refused."""
    if not isinstance(name, str) or not _NAME.fullmatch(name) or ".." in name:
        raise StudioError(404, "unknown item")
    path = folder / name
    if path.parent.resolve() != folder.resolve():
        raise StudioError(404, "unknown item")
    return path


def _iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _count(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# ================================================================================================= research runs
def run_summary(name: str, data: Dict[str, Any]) -> Dict[str, Any]:
    claims = [c for c in data.get("claims") or [] if isinstance(c, dict)]
    labels = [c.get("label") for c in claims]
    status = data.get("provider_status") if isinstance(data.get("provider_status"), dict) else {}
    return {
        "name": name, "input": str(data.get("input", "")), "mode": data.get("mode"), "status": data.get("status"),
        "outcome": data.get("outcome"), "started": data.get("started"), "finished": data.get("finished"),
        "claims": len(claims), "supported": labels.count("SUPPORTED"), "contradicted": labels.count("CONTRADICTED"),
        "insufficient": labels.count("INSUFFICIENT"), "sources": len(data.get("sources") or []),
        "error": str(data.get("error") or "")[:300],
        "providers": {str(provider): {key: value.get(key) for key in ("queries", "succeeded", "failed", "hits")}
                      for provider, value in status.items() if isinstance(value, dict)},
    }


def _used_sources(data: Dict[str, Any]) -> set:
    used = set()
    for claim in data.get("claims") or []:
        if isinstance(claim, dict):
            for key in ("cited_sources", "supporting_sources", "contradicting_sources"):
                used.update(str(i) for i in claim.get(key) or [])
    return used


def light_run(data: Dict[str, Any]) -> Dict[str, Any]:
    """A copy of a run without the full extracted source texts (they are served on demand as context)."""
    light = copy.deepcopy(data)
    for source in light.get("sources") or []:
        if isinstance(source, dict):
            text, pages = source.pop("text", None), source.pop("pages", None)
            source["has_text"] = bool(text) or bool(pages)
            source["page_count"] = len(pages) if isinstance(pages, list) else (1 if text else 0)
            source.pop("extra", None)
    return light


class RunStore:
    """Saved research runs, read from disk; a run whose file cannot be read is listed as UNREADABLE."""

    def __init__(self, folder: Path = RUNS_DIR):
        self.folder = Path(folder)
        self._cache: Dict[str, Tuple[Tuple[int, int], Dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def _load(self, path: Path) -> Dict[str, Any]:
        stat = path.stat()
        key = (stat.st_mtime_ns, stat.st_size)
        with self._lock:
            cached = self._cache.get(str(path))
            if cached and cached[0] == key:
                return cached[1]
        data = _read_json(path)
        if not isinstance(data, dict):
            raise ValueError("run record is not a JSON object")
        with self._lock:
            self._cache[str(path)] = (key, data)
        return data

    def list(self) -> List[Dict[str, Any]]:
        if not self.folder.is_dir():
            return []
        runs = []
        for folder in sorted(self.folder.iterdir(), reverse=True):
            record = folder / "run.json"
            if not folder.is_dir() or not _NAME.fullmatch(folder.name) or not record.is_file():
                continue
            try:
                runs.append(run_summary(folder.name, self._load(record)))
            except (OSError, ValueError) as exc:
                runs.append({"name": folder.name, "input": "", "mode": None, "status": "UNREADABLE",
                             "outcome": None, "started": None, "finished": None, "claims": 0, "supported": 0,
                             "contradicted": 0, "insufficient": 0, "sources": 0,
                             "error": f"The run record could not be read ({type(exc).__name__})."})
        return runs

    def get(self, name: str) -> Dict[str, Any]:
        record = _child(self.folder, name) / "run.json"
        if not record.is_file():
            raise StudioError(404, "unknown run")
        try:
            return self._load(record)
        except (OSError, ValueError) as exc:
            raise StudioError(422, f"The run record could not be read ({type(exc).__name__}).") from None

    def sources(self) -> List[Dict[str, Any]]:
        """Every source recorded in the saved runs, merged across runs by its canonical key (URL, DOI or arXiv id as
        the web researcher computed it), with the runs that selected it. Nothing is fetched or re-read from the web."""
        merged: Dict[str, Dict[str, Any]] = {}
        for summary in self.list():
            if summary["status"] == "UNREADABLE":
                continue
            try:
                data = self.get(summary["name"])
            except StudioError:
                continue
            used = _used_sources(data)
            for source in data.get("sources") or []:
                if not isinstance(source, dict):
                    continue
                key = str(source.get("key") or source.get("url") or source.get("title") or "")
                if not key:
                    continue
                entry = merged.setdefault(key, {
                    "key": key, "title": source.get("title") or source.get("url") or key, "url": source.get("url"),
                    "kind": source.get("kind"), "provider": source.get("provider"),
                    "publisher": source.get("publisher") or source.get("domain") or "",
                    "published": source.get("published"), "access": source.get("access"), "runs": []})
                access = source.get("access")
                if access in ACCESS_ORDER and (entry["access"] not in ACCESS_ORDER
                                               or ACCESS_ORDER.index(access) < ACCESS_ORDER.index(entry["access"])):
                    entry["access"] = access
                entry["runs"].append({"name": summary["name"], "input": summary["input"], "mode": summary["mode"],
                                      "id": source.get("id"), "access": access,
                                      "used": str(source.get("id")) in used, "started": summary["started"]})
        entries = list(merged.values())
        for entry in entries:
            entry["used_in"] = sum(r["used"] for r in entry["runs"])
        entries.sort(key=lambda e: (-e["used_in"], -len(e["runs"]), str(e["title"]).lower()))
        return entries


# ================================================================================================= source context
def _normalized(text: str) -> Tuple[str, List[int]]:
    """Text as the passage chunker sees it, with the original index of every kept character: whitespace runs become
    one space, soft hyphens are dropped, and a word broken across a line ("incor- porating") is joined."""
    out: List[str] = []
    origin: List[int] = []
    index, length = 0, len(text)
    while index < length:
        char = text[index]
        if char == "­":
            index += 1
            continue
        if char == "-" and out and out[-1].isalpha():
            after = index + 1
            while after < length and text[after].isspace():
                after += 1
            if after > index + 1 and after < length and text[after].islower():
                index = after                            # a line-break hyphen: drop it and the break
                continue
        if char.isspace():
            if out and out[-1] != " ":
                out.append(" ")
                origin.append(index)
            index += 1
            continue
        out.append(char)
        origin.append(index)
        index += 1
    return "".join(out).rstrip(), origin


def locate_passage(source_text: str, passage_text: str) -> Optional[Tuple[int, int]]:
    """Where a passage sits in its source text, ignoring whitespace and line-break hyphenation; None if absent."""
    if not passage_text.strip() or not source_text:
        return None
    start = source_text.find(passage_text)
    if start >= 0:
        return start, start + len(passage_text)
    needle = _normalized(passage_text)[0].strip()
    haystack, origin = _normalized(source_text)
    position = haystack.find(needle)
    if position < 0 or not needle:
        return None
    return origin[position], origin[position + len(needle) - 1] + 1


def _find_passage(data: Dict[str, Any], key: str) -> Optional[Dict[str, Any]]:
    for passage in data.get("evidence_pool") or []:
        if isinstance(passage, dict) and passage.get("key") == key:
            return passage
    for claim in data.get("claims") or []:
        for passage in (claim or {}).get("passages") or []:
            if isinstance(passage, dict) and passage.get("key") == key:
                return passage
    return None


def source_context(data: Dict[str, Any], source_id: str, key: str, width: int = CONTEXT_CHARS) -> Dict[str, Any]:
    """The surrounding text of one retrieved passage in the source it came from, as recorded in the run."""
    source = next((s for s in data.get("sources") or [] if isinstance(s, dict) and s.get("id") == source_id), None)
    if source is None:
        raise StudioError(404, "unknown source")
    passage = _find_passage(data, key)
    if passage is None or passage.get("source_id") != source_id:
        raise StudioError(404, "unknown passage")
    meta = {k: source.get(k) for k in ("id", "title", "url", "provider", "kind", "access", "published", "publisher",
                                       "domain", "note", "read_url")}
    pages = source.get("pages") if isinstance(source.get("pages"), list) else None
    page = passage.get("page") or 1
    text = (pages[page - 1] if pages and 0 < page <= len(pages) else source.get("text")) or ""
    found = locate_passage(text, str(passage.get("text") or ""))
    result = {"source": meta, "page": page, "passage": passage.get("text"), "located": bool(found),
              "before": "", "match": "", "after": "", "truncated_before": False, "truncated_after": False}
    if found:
        start, end = found
        lo, hi = max(0, start - width), min(len(text), end + width)
        result.update(before=text[lo:start], match=text[start:end], after=text[end:hi],
                      truncated_before=lo > 0, truncated_after=hi < len(text))
    return result


# ================================================================================================= evaluations
_EVAL_DOCS = re.compile(r"(README|DECISION_RULE(_V[0-9]+)?)\.md")


def _frozen_hashes(folder: Path) -> Dict[str, str]:
    hashes: Dict[str, str] = {}
    for path in sorted(folder.glob("FROZEN*.json")):
        try:
            for dataset, digest in (_read_json(path).get("datasets") or {}).items():
                hashes[str(dataset)] = str(digest)
        except (OSError, ValueError, AttributeError):
            continue
    return hashes


def _evaluation(folder: Path, frozen: Dict[str, str]) -> Dict[str, Any]:
    manifest = _read_json(folder / "manifest.json")
    metrics_path, summary_path, incomplete_path = (folder / "metrics.json", folder / "summary.txt",
                                                   folder / "INCOMPLETE.md")
    metrics = _read_json(metrics_path) if metrics_path.is_file() else None
    dataset = manifest.get("dataset")
    expected = frozen.get(str(dataset)) if dataset else None
    verifiers = manifest.get("verifiers") or ([manifest["verifier"]] if manifest.get("verifier") else [])
    kind = ("claim verification" if isinstance(metrics, dict) and "classification" in metrics
            else "research" if isinstance(metrics, dict) and "completion_rate" in metrics else "unknown")
    return {
        "name": folder.name, "kind": kind, "dataset": dataset, "dataset_sha256": manifest.get("dataset_sha256"),
        "dataset_matches_frozen": (expected == manifest.get("dataset_sha256")) if expected else None,
        "snapshot": manifest.get("snapshot"), "snapshot_sha256": manifest.get("snapshot_sha256"),
        "verifiers": verifiers, "model": manifest.get("model"), "k": manifest.get("k"),
        "per_source": manifest.get("per_source"), "fresh_runner_policy": manifest.get("fresh_runner_policy"),
        "started": manifest.get("started"), "finished": manifest.get("finished"),
        "complete": bool(manifest.get("finished")) and not incomplete_path.is_file() and metrics is not None,
        "incomplete_note": incomplete_path.read_text(encoding="utf-8")[:2000] if incomplete_path.is_file() else "",
        "metrics": metrics, "summary": summary_path.read_text(encoding="utf-8") if summary_path.is_file() else "",
        "comparison_group": f"{manifest.get('dataset_sha256') or '-'}|{manifest.get('snapshot_sha256') or '-'}",
    }


class EvaluationStore:
    """Recorded evaluation runs (manifest, metrics and summary only), plus the method documents."""

    def __init__(self, folder: Path = EVAL_DIR):
        self.folder = Path(folder)

    def list(self) -> Dict[str, Any]:
        runs_dir = self.folder / "runs"
        if not runs_dir.is_dir():
            return {"available": False, "runs": [], "documents": self.documents()}
        frozen = _frozen_hashes(self.folder)
        runs = []
        for folder in sorted(runs_dir.iterdir()):
            if not folder.is_dir() or not _NAME.fullmatch(folder.name) or not (folder / "manifest.json").is_file():
                continue
            try:
                record = _evaluation(folder, frozen)
            except (OSError, ValueError, AttributeError) as exc:
                record = {"name": folder.name, "kind": "unknown", "complete": False, "metrics": None, "summary": "",
                          "error": f"The run files could not be read ({type(exc).__name__})."}
            record.pop("summary", None)
            runs.append(record)
        return {"available": True, "runs": runs, "documents": self.documents()}

    def get(self, name: str) -> Dict[str, Any]:
        folder = _child(self.folder / "runs", name)
        if not (folder / "manifest.json").is_file():
            raise StudioError(404, "unknown evaluation run")
        try:
            return _evaluation(folder, _frozen_hashes(self.folder))
        except (OSError, ValueError, AttributeError) as exc:
            raise StudioError(422, f"The run files could not be read ({type(exc).__name__}).") from None

    def documents(self) -> List[Dict[str, str]]:
        if not self.folder.is_dir():
            return []
        docs = []
        for path in sorted(self.folder.glob("*.md")):
            if _EVAL_DOCS.fullmatch(path.name):
                first = next((line.strip("# ").strip() for line in path.read_text(encoding="utf-8").splitlines()
                              if line.strip()), path.name)
                docs.append({"name": path.name, "title": first})
        return docs

    def document(self, name: str) -> Dict[str, str]:
        if not _EVAL_DOCS.fullmatch(name or ""):
            raise StudioError(404, "unknown document")
        path = self.folder / name
        if not path.is_file():
            raise StudioError(404, "unknown document")
        return {"name": name, "path": _relative(path), "text": path.read_text(encoding="utf-8")}


# ================================================================================================= document library
def folder_fingerprint(folder: Path) -> str:
    entries = []
    if folder.is_dir():
        for path in sorted(folder.iterdir()):
            if path.is_file():
                stat = path.stat()
                entries.append(f"{path.name}|{stat.st_size}|{stat.st_mtime_ns}")
    return hashlib.sha256("\n".join(entries).encode("utf-8")).hexdigest()


class Library:
    """The document folder ``docqa`` reads, its ingestion report, and the document assistant built from it."""

    def __init__(self, folder: Path = DOCS_DIR, assistant_factory: Optional[Callable[..., Any]] = None):
        self.folder = Path(folder)
        self.assistant_factory = assistant_factory
        self._lock = threading.Lock()
        self._report_cache: Optional[Tuple[str, Any]] = None
        self.assistant: Any = None
        self.index_state: Dict[str, Any] = {"state": "not_built", "fingerprint": None, "built_at": None,
                                            "documents": [], "pages": 0, "error": ""}

    # ------------------------------------------------------------------ listing
    def _report(self):
        from docqa.ingest import load_documents

        fingerprint = folder_fingerprint(self.folder)
        with self._lock:
            if self._report_cache and self._report_cache[0] == fingerprint:
                return fingerprint, self._report_cache[1]
        report = load_documents(self.folder)
        with self._lock:
            self._report_cache = (fingerprint, report)
        return fingerprint, report

    def listing(self) -> Dict[str, Any]:
        base = {"folder": _relative(self.folder), "exists": self.folder.is_dir(), "documents": [],
                "max_upload_bytes": MAX_UPLOAD_BYTES, "upload_types": list(UPLOAD_SUFFIXES)}
        if not self.folder.is_dir():
            return base | {"index": self.index_status(None)}
        fingerprint, report = self._report()
        pages: Dict[str, List[Any]] = {}
        for page in report.pages:
            pages.setdefault(page.document, []).append(page)
        skipped = dict(report.skipped)
        documents = []
        for path in sorted(self.folder.iterdir()):
            if not path.is_file() or path.name.startswith("."):
                continue
            stat = path.stat()
            doc_pages = pages.get(path.name, [])
            documents.append({
                "name": path.name, "type": path.suffix.lower().lstrip(".") or "file", "bytes": stat.st_size,
                "modified": _iso(stat.st_mtime), "pages": len(doc_pages), "chars": sum(len(p.text) for p in doc_pages),
                "readable": bool(doc_pages), "skipped_reason": skipped.get(path.name, ""),
                "preview": (doc_pages[0].text[:600] if doc_pages else ""),
            })
        return base | {"documents": documents, "index": self.index_status(fingerprint)}

    def index_status(self, fingerprint: Optional[str]) -> Dict[str, Any]:
        with self._lock:
            status = dict(self.index_state)
        if status["state"] == "ready" and fingerprint and status["fingerprint"] != fingerprint:
            status["state"] = "stale"
        status.pop("fingerprint", None)
        return status

    # ------------------------------------------------------------------ upload
    def save_upload(self, filename: str, data: bytes) -> str:
        name = Path(str(filename or "")).name.strip()
        if not name or name.startswith(".") or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 _.()+-]{0,150}", name):
            raise StudioError(400, "Use a plain file name (letters, digits, spaces, '.', '_', '-', '(', ')').")
        if Path(name).suffix.lower() not in UPLOAD_SUFFIXES:
            raise StudioError(415, "Only .txt and .pdf files can be added.")
        if not data:
            raise StudioError(400, "The file is empty.")
        if len(data) > MAX_UPLOAD_BYTES:
            raise StudioError(413, "The file is larger than the upload limit.")
        if name.lower().endswith(".pdf") and not data.startswith(b"%PDF-"):
            raise StudioError(415, "The file does not look like a PDF.")
        if name.lower().endswith(".txt"):
            try:
                data.decode("utf-8-sig")
            except UnicodeDecodeError:
                raise StudioError(415, "Text files must be UTF-8.") from None
        self.folder.mkdir(parents=True, exist_ok=True)
        target = self.folder / name
        if target.exists():
            raise StudioError(409, f"A document named {name!r} already exists; rename the file to add it.")
        handle, temp = tempfile.mkstemp(dir=self.folder, prefix=".upload-", suffix=".part")
        try:
            with os.fdopen(handle, "wb") as stream:
                stream.write(data)
            if target.exists():
                raise StudioError(409, f"A document named {name!r} already exists; rename the file to add it.")
            os.replace(temp, target)
        finally:
            if os.path.exists(temp):
                os.remove(temp)
        return name

    # ------------------------------------------------------------------ assistant
    def _build(self, stages: List[Dict[str, Any]], progress: Callable[[], None]) -> Any:
        fingerprint = folder_fingerprint(self.folder)
        with self._lock:
            if self.assistant is not None and self.index_state["fingerprint"] == fingerprint:
                return self.assistant
            self.index_state = {"state": "building", "fingerprint": fingerprint, "built_at": None,
                                "documents": [], "pages": 0, "error": ""}
        progress()
        started = time.perf_counter()
        log: List[str] = []
        try:
            factory = self.assistant_factory
            if factory is None:
                from docqa.cli import build_assistant as factory
            assistant, report = factory(str(self.folder), log=log.append)
        except SystemExit as exc:                    # build_assistant reports "no documents" with SystemExit
            message = str(exc.code) if exc.code not in (None, 0) else "The documents could not be indexed."
            with self._lock:
                self.index_state.update(state="failed", error=message)
            stages.append({"name": "load and index documents", "status": "failed",
                           "seconds": round(time.perf_counter() - started, 2), "detail": message})
            raise StudioError(422, message) from None
        except Exception as exc:
            message = f"Indexing failed: {type(exc).__name__}: {exc}"
            with self._lock:
                self.index_state.update(state="failed", error=message)
            stages.append({"name": "load and index documents", "status": "failed",
                           "seconds": round(time.perf_counter() - started, 2), "detail": message})
            raise StudioError(500, message) from None
        with self._lock:
            self.assistant = assistant
            self.index_state = {"state": "ready", "fingerprint": fingerprint, "built_at": _iso(time.time()),
                                "documents": list(report.documents), "pages": len(report.pages), "error": ""}
        stages.append({"name": "load and index documents", "status": "ok",
                       "seconds": round(time.perf_counter() - started, 2),
                       "detail": f"{_count(len(report.documents), 'document')}, {_count(len(report.pages), 'page')}"
                                 + (f"; skipped {len(report.skipped)}" if report.skipped else "")})
        return assistant

    def run(self, mode: str, text: str, progress: Callable[[Dict[str, Any]], None]) -> Dict[str, Any]:
        """One document job: ``documents-index``, ``documents`` (ask) or ``documents-verify``."""
        data: Dict[str, Any] = {"input": text, "mode": mode, "status": "RUNNING", "outcome": None, "stages": [],
                                "error": "", "result": None,
                                "started": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
        emit = lambda: progress(copy.deepcopy(data))
        try:
            assistant = self._build(data["stages"], emit)
            emit()
            if mode == "documents-index":
                data["status"], data["outcome"] = "COMPLETE", "INDEXED"
            elif mode == "documents":
                data["current"] = "answering from the documents and checking each statement"
                emit()
                started = time.perf_counter()
                result = qa_record(assistant.ask(text))
                data["stages"].append({"name": "answer and verify", "status": "failed" if result["error"] else "ok",
                                       "seconds": round(time.perf_counter() - started, 2),
                                       "detail": f"{len(result['claims'])} statements checked"})
                data.update(result=result, outcome=result["status"], error=result["error"],
                            status="FAILED" if result["status"] == "FAILED" else "COMPLETE")
            else:
                data["current"] = "checking the claim against the documents"
                emit()
                started = time.perf_counter()
                result = check_record(assistant.verify_claim(text))
                data["stages"].append({"name": "verify", "status": "failed" if result["error"] else "ok",
                                       "seconds": round(time.perf_counter() - started, 2),
                                       "detail": (result["verdict"] or {}).get("label", "no verdict")})
                data.update(result=result, error=result["error"],
                            outcome=(result["verdict"] or {}).get("label"),
                            status="FAILED" if result["error"] else "COMPLETE")
        except StudioError as exc:
            data.update(status="FAILED", error=str(exc))
        data.pop("current", None)
        data["finished"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        emit()
        return data


def _evidence_record(e: Any) -> Dict[str, Any]:
    return {"id": e.id, "document": e.document, "page": e.page, "text": e.text, "score": round(float(e.score), 4),
            "citation": e.citation}


def _verdict_record(v: Any) -> Optional[Dict[str, Any]]:
    if v is None:
        return None
    return {"label": v.label, "sub_state": v.sub_state, "stances": dict(v.stances), "supporting": v.supporting,
            "contradicting": v.contradicting, "parse_ok": v.parse_ok, "truncated": v.truncated,
            "error": v.error or "", "note": v.note}


def qa_record(result: Any) -> Dict[str, Any]:
    """``docqa.pipeline.QAResult`` as JSON (model outputs other than the answer are not included)."""
    return {"question": result.question, "answer": result.answer, "abstained": result.abstained,
            "status": result.status, "limitations": list(result.limitations), "error": result.error,
            "evidence": [_evidence_record(e) for e in result.evidence],
            "claims": [{"text": c.text, "cited": list(c.cited), "verdict": _verdict_record(c.verdict)}
                       for c in result.claims]}


def check_record(check: Any) -> Dict[str, Any]:
    """``docqa.pipeline.ClaimCheck`` as JSON."""
    return {"claim": check.claim, "error": check.error, "evidence": [_evidence_record(e) for e in check.evidence],
            "verdict": _verdict_record(check.verdict)}
