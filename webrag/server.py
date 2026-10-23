"""Local web interface (standard library only), bound to 127.0.0.1: the Multi-Agent Research Studio.

    python -m webrag.server            # then open http://127.0.0.1:8765

Runs are processed one at a time by a worker thread; the page polls for stage-by-stage progress. Besides the
research and verification jobs of the Evidence Research Assistant pipeline, the server reads saved runs, recorded
evaluation summaries and the document library (``webrag.studio``), and runs the ``docqa`` document assistant on
request. It never runs a model on its own initiative.

Cancellation: a job that is still waiting in the queue can be withdrawn (``POST /api/run/<id>/cancel``) and never
starts. A job that has started cannot be interrupted, because the pipelines have no safe stopping point; it runs to
the end and is saved as usual, and the request reports that instead of pretending to stop it.

Integrations (``webrag.integrations``) are reported from real configuration only; saved runs can be exported as
Markdown or JSON, and into an Obsidian vault chosen at launch (``--obsidian-vault``).

This server is for one person on this computer: it answers only requests addressed to 127.0.0.1 or localhost,
rejects cross-origin writes, and has no accounts. It must not be exposed to a network as it is.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import queue
import re
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import parse_qs, unquote, urlsplit

from dataclasses import asdict

from .config import ConfigError, ConfigStore, public_view
from .integrations import (ObsidianVault, describe as describe_integrations, obsidian_integration, ollama_integration,
                           run_markdown)
from .orchestrator import Orchestrator, Settings
from .studio import (DOCS_DIR, MAX_UPLOAD_BYTES, ROOT, RUNS_DIR, STUDIO_VERSION, EvaluationStore, Library, RunStore,
                     StudioError, light_run, source_context)

PAGE = Path(__file__).with_name("ui.html")
STATIC = Path(__file__).with_name("static")
STATIC_TYPES = {".css": "text/css; charset=utf-8", ".mjs": "text/javascript; charset=utf-8",
                ".js": "text/javascript; charset=utf-8", ".svg": "image/svg+xml"}
_STATIC_NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}\.(css|mjs|js|svg)")
_CANCEL = re.compile(r"/api/run/([A-Za-z0-9]{1,32})/cancel")
_PROVIDER_TEST = re.compile(r"/api/providers/([a-z]{2,20})/test")
TEST_QUERY = "renewable energy"            # a neutral query for a user-requested provider test


def check_provider(provider_id: str, config: Dict[str, Any], session: Any = None) -> Dict[str, Any]:
    """One real search with a neutral query, only when the user presses Test: it says whether the provider answers,
    how many results came back and how long it took. Errors are returned as text with secrets redacted."""
    from .net import PoliteSession, redact
    from .search import EXCLUSION_NOTE, PROVIDERS, SearchError, default_providers, is_evidence_provider
    if provider_id not in PROVIDERS:
        return {"id": provider_id, "ok": False, "error": "unknown provider"}
    if not is_evidence_provider(provider_id):                 # source policy: nothing is sent
        return {"id": provider_id, "ok": False, "excluded": True, "error": f"{EXCLUSION_NOTE}, so it is not searched."}
    try:
        provider = default_providers(session or PoliteSession(), [provider_id], config)[0]
    except ValueError as exc:
        return {"id": provider_id, "ok": False, "error": str(exc), "configured": False}
    started = time.perf_counter()
    try:
        hits = provider.search(TEST_QUERY, 2)
    except SearchError as exc:
        return {"id": provider_id, "ok": False, "error": redact(str(exc)), "seconds": round(time.perf_counter() - started, 2)}
    return {"id": provider_id, "ok": True, "results": len(hits), "query": TEST_QUERY,
            "seconds": round(time.perf_counter() - started, 2), "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
MAX_INPUT = 2000
RESEARCH_MODES = {"research", "verify"}
DOCUMENT_MODES = {"documents", "documents-verify", "documents-index"}
CSP = ("default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; "
       "base-uri 'none'; form-action 'none'; frame-ancestors 'none'")


class JobQueue:
    def __init__(self, orchestrator_factory, library: Optional[Library] = None):
        self.factory = orchestrator_factory
        self.library = library
        self.orchestrator: Optional[Orchestrator] = None
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
        self.queue: "queue.Queue[str]" = queue.Queue()
        self._sequence = 0
        threading.Thread(target=self._work, daemon=True).start()

    def submit(self, text: str, mode: str, options: Optional[Dict[str, Any]] = None) -> str:
        """``options`` (research and verification only): deep-research choices passed to ``Orchestrator.run``."""
        job_id = uuid.uuid4().hex[:12]
        with self.lock:
            self._sequence += 1
            shown = {k: v for k, v in (options or {}).items() if k != "config"}
            self.jobs[job_id] = {"id": job_id, "state": "queued", "sequence": self._sequence, "submitted": time.time(),
                                 "options": dict(options) if options else None,
                                 "data": {"input": text, "mode": mode, "stages": [], **({"options": shown} if shown else {})}}
        self.queue.put(job_id)
        return job_id

    def _position(self, job: Dict[str, Any]) -> Optional[int]:
        """1 for the next job to start; jobs withdrawn from the queue no longer count."""
        if job["state"] != "queued":
            return None
        return 1 + sum(j["state"] == "queued" and j["sequence"] < job["sequence"] for j in self.jobs.values())

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return None
            snapshot = copy.deepcopy({k: v for k, v in job.items() if k != "options"})   # the config stays server-side
            snapshot["position"] = self._position(job)
            return snapshot

    def cancel(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Withdraw a job that has not started; a started or finished job is left as it is. Returns the job's
        state afterwards (None for an unknown job)."""
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return None
            if job["state"] == "queued":
                job["state"] = "cancelled"
                job["cancelled"] = time.time()
                job["data"] = dict(job["data"], status="CANCELLED")
            return {"id": job_id, "state": job["state"]}

    def overview(self) -> List[Dict[str, Any]]:
        """Jobs of this server session, newest first, without their result payloads."""
        with self.lock:
            jobs = sorted(self.jobs.values(), key=lambda j: j.get("submitted", 0), reverse=True)
            return [{"id": j["id"], "state": j["state"], "position": self._position(j), "mode": j["data"].get("mode"),
                     "input": j["data"].get("input"), "status": j["data"].get("status"),
                     "outcome": j["data"].get("outcome"), "saved_to": j["data"].get("saved_to"),
                     "depth": (j["data"].get("options") or {}).get("depth")} for j in jobs]

    def _update(self, job_id: str, data: Dict[str, Any]) -> None:
        with self.lock:
            self.jobs[job_id]["data"] = copy.deepcopy(data)

    def _work(self) -> None:
        while True:
            job_id = self.queue.get()
            with self.lock:
                job = self.jobs[job_id]
                if job["state"] == "cancelled":         # withdrawn while it waited: never started
                    continue
                job["state"] = "running"
                job["started"] = time.time()
                text, mode, options = job["data"]["input"], job["data"]["mode"], job.get("options")
            progress = lambda d: self._update(job_id, d)
            try:
                if mode in DOCUMENT_MODES:
                    if self.library is None:
                        raise RuntimeError("the document assistant is not configured in this server")
                    data = self.library.run(mode, text, progress=progress)
                else:
                    if self.orchestrator is None:
                        self.orchestrator = self.factory()
                    data = (self.orchestrator.run(text, mode, progress=progress, options=options) if options
                            else self.orchestrator.run(text, mode, progress=progress))
            except Exception as exc:                 # keep the server alive and say what happened
                data = {"input": text, "mode": mode, "status": "FAILED", "stages": [],
                        "error": f"Could not start the research pipeline: {type(exc).__name__}: {exc}"}
            with self.lock:
                self.jobs[job_id].update(state="done", data=copy.deepcopy(data))


class StatusProbe:
    """What the studio can report about its own configuration; Ollama is asked only for its version and models."""

    def __init__(self, model: str, verifier: str, describe: Optional[Callable[[], Dict[str, Any]]] = None,
                 providers: Optional[Callable[[], List[str]]] = None, ttl: float = 15.0,
                 models: Optional[Callable[[], List[str]]] = None,
                 config: Optional[Callable[[], Dict[str, Any]]] = None):
        self.model, self.verifier, self.ttl = model, verifier, ttl
        self._describe, self._providers, self._models, self._config = describe, providers, models, config
        self._cache: Optional[tuple] = None
        self._lock = threading.Lock()

    def _ollama(self) -> Dict[str, Any]:
        if self._describe is None:
            from docqa.llm import OllamaChat
            return OllamaChat(model=self.model).describe()
        return self._describe()

    def _installed_models(self) -> List[str]:
        """Names of the models installed in the local Ollama (a GET of its model list, nothing is downloaded)."""
        if self._models is not None:
            return list(self._models())
        if self._describe is not None:          # a substituted runtime: never contact the real one
            return []
        from docqa.llm import OllamaChat
        try:
            with urllib.request.urlopen(OllamaChat(model=self.model).base_url + "/api/tags", timeout=5) as response:
                rows = json.loads(response.read()).get("models", [])
            return sorted(str(row["name"]) for row in rows if isinstance(row, dict) and row.get("name"))
        except (urllib.error.URLError, TimeoutError, ValueError, OSError):
            return []

    def _provider_names(self) -> Dict[str, Any]:
        try:
            if self._providers is None:
                from .net import PoliteSession
                from .search import default_providers
                config = self._config() if self._config is not None else None
                names = [p.name for p in default_providers(PoliteSession(), None, config)]
            else:
                names = list(self._providers())
            return {"providers": names, "providers_error": "",
                    "general_web_search": bool({"searxng", "brave"} & set(names))}
        except Exception as exc:                     # a misconfigured provider list is reported, not hidden
            return {"providers": [], "providers_error": f"{type(exc).__name__}: {exc}", "general_web_search": False}

    def invalidate(self) -> None:
        with self._lock:
            self._cache = None

    def get(self, refresh: bool = False) -> Dict[str, Any]:
        with self._lock:
            if not refresh and self._cache and time.monotonic() - self._cache[0] < self.ttl:
                return copy.deepcopy(self._cache[1])
        info = self._ollama()
        status = {"app": "Multi-Agent Research Studio", "studio_version": STUDIO_VERSION, "model": self.model,
                  "verifier": self.verifier, "document_verifier": "frozen_b", "ollama_version": info.get("ollama_version"),
                  "model_installed": bool(info.get("model_digest")),
                  "model_digest": info.get("model_digest"),
                  "models": self._installed_models() if info.get("ollama_version") else [],
                  "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        status.update(self._provider_names())
        with self._lock:
            self._cache = (time.monotonic(), status)
        return copy.deepcopy(status)


def _display_path(folder: Path) -> str:
    """A folder relative to the project when it is inside it, otherwise its full path (shown to the local user)."""
    resolved = Path(folder).resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _run_options(body: Dict[str, Any], config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Deep-research choices from a run request, checked against what is actually active. None when the request
    asks for none of them (the original single-pass pipeline then runs, as before)."""
    from .deep import DEPTHS, RECENCY
    from .search import SOURCE_CATEGORIES, provider_catalog
    if not any(key in body for key in ("depth", "providers", "categories", "recency")):
        return None
    depth = body.get("depth") or "standard"
    if depth not in DEPTHS:
        raise ValueError(f"depth must be one of {', '.join(DEPTHS)}")
    active = [row["id"] for row in provider_catalog(config) if row["active"]]
    providers = body.get("providers")
    if providers is not None:
        if not isinstance(providers, list) or not all(isinstance(p, str) for p in providers):
            raise ValueError("providers must be a list of provider ids")
        inactive = [p for p in providers if p not in active]
        if inactive:
            raise ValueError(f"not active: {', '.join(inactive)}; enable and set them up in Integrations")
        if not providers:
            raise ValueError("choose at least one source to search")
    elif not active:
        raise ValueError("no search provider is active; enable one in Integrations")
    categories = body.get("categories") or []
    if not isinstance(categories, list) or any(c not in SOURCE_CATEGORIES for c in categories):
        raise ValueError(f"categories must be among {', '.join(SOURCE_CATEGORIES)}")
    recency = body.get("recency") or "any"
    if recency not in RECENCY:
        raise ValueError(f"recency must be one of {', '.join(RECENCY)}")
    return {"depth": depth, "providers": providers or active, "categories": categories, "recency": recency,
            "config": config}


def make_handler(jobs: JobQueue, port: int, runs: Optional[RunStore] = None,
                 evaluations: Optional[EvaluationStore] = None, status: Optional[StatusProbe] = None,
                 vault: Optional[ObsidianVault] = None, config_store: Optional[ConfigStore] = None,
                 provider_test: Optional[Callable[[str, Dict[str, Any]], Dict[str, Any]]] = None):
    allowed_origins = {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}
    # Requests must name this computer: a page on another site that re-points its domain to 127.0.0.1 (DNS
    # rebinding) sends its own host name and is refused.
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"} | ({"127.0.0.1", "localhost"} if port == 80 else set())
    runs = runs or RunStore(RUNS_DIR)
    evaluations = evaluations or EvaluationStore()
    vault = vault or ObsidianVault(None)
    config_store = config_store or ConfigStore()
    provider_test = provider_test or check_provider

    def providers_payload() -> Dict[str, Any]:
        from .deep import BUDGETS, EVIDENCE_POLICY, RECENCY
        from .search import CATEGORY_PROVIDERS, provider_catalog
        config = config_store.load()
        recorded = {}
        for summary in runs.list():
            for name, value in (summary.get("providers") or {}).items():
                recorded.setdefault(name, {"run": summary["name"], "started": summary.get("started"), **value})
        catalog = provider_catalog(config)
        for row in catalog:
            row["last_recorded"] = recorded.get(row["id"])
        return {"providers": catalog, "config": public_view(config), "config_error": config_store.error,
                "depths": {name: asdict(b) for name, b in BUDGETS.items()}, "categories": CATEGORY_PROVIDERS,
                "recency": list(RECENCY), "evidence_policy": EVIDENCE_POLICY}

    def studio_status(refresh: bool = False) -> Dict[str, Any]:
        info = status.get(refresh=refresh) if status is not None else {"app": "Multi-Agent Research Studio", "available": False}
        return dict(info, studio_version=STUDIO_VERSION, runs_folder=_display_path(runs.folder),
                    docs_folder=_display_path(jobs.library.folder) if jobs.library is not None else None,
                    host="127.0.0.1", port=port)

    class Handler(BaseHTTPRequestHandler):
        server_version = "webrag"

        def log_message(self, fmt, *args):                       # keep the console quiet
            pass

        def _send(self, status_code: int, body: bytes, content_type: str, page: bool = False,
                  headers: Optional[Dict[str, str]] = None) -> None:
            self.send_response(status_code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            for name, value in (headers or {}).items():
                self.send_header(name, value)
            if page:
                self.send_header("Content-Security-Policy", CSP)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status_code: int, payload: Any) -> None:
            self._send(status_code, json.dumps(payload).encode("utf-8"), "application/json")

        def _host_ok(self) -> bool:
            if self.headers.get("Host", "") in allowed_hosts:
                return True
            self._json(421, {"error": "This server only answers requests addressed to 127.0.0.1 or localhost."})
            return False

        # ------------------------------------------------------------------------------------------ GET
        def do_GET(self):
            if not self._host_ok():
                return None
            parts = urlsplit(self.path)
            path, query = parts.path, parse_qs(parts.query)
            try:
                if path in {"/", "/index.html"}:
                    return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8", page=True)
                if path == "/favicon.ico":
                    return self._static("favicon.svg")
                if path.startswith("/static/"):
                    return self._static(path[len("/static/"):])
                if path == "/api/status":
                    return self._json(200, studio_status())
                if path == "/api/integrations":
                    listing = jobs.library.listing() if jobs.library is not None else None
                    return self._json(200, describe_integrations(studio_status(), runs.list(), listing, vault))
                if path == "/api/integrations/obsidian":
                    return self._json(200, obsidian_integration(vault))
                if path == "/api/providers":
                    return self._json(200, providers_payload())
                if path == "/api/sources":
                    return self._json(200, {"sources": runs.sources()})
                if path == "/api/jobs":
                    return self._json(200, {"jobs": jobs.overview()})
                if path.startswith("/api/run/"):
                    rest = path[len("/api/run/"):].split("/")
                    job = jobs.get(rest[0])
                    if not job:
                        return self._json(404, {"error": "unknown run"})
                    if len(rest) == 2 and rest[1] == "context":
                        return self._json(200, source_context(job.get("data") or {}, *self._context_args(query)))
                    return self._json(200, job) if len(rest) == 1 else self._json(404, {"error": "not found"})
                if path == "/api/runs":
                    return self._json(200, {"folder": RUNS_DIR.name, "runs": runs.list()})
                if path.startswith("/api/runs/"):
                    rest = [unquote(p) for p in path[len("/api/runs/"):].split("/")]
                    data = runs.get(rest[0])
                    if len(rest) == 1:
                        return self._json(200, {"name": rest[0], "data": light_run(data)})
                    if len(rest) == 2 and rest[1] == "context":
                        return self._json(200, source_context(data, *self._context_args(query)))
                    if len(rest) == 2 and rest[1] == "export":
                        return self._export(rest[0], data, (query.get("format") or ["md"])[0])
                    return self._json(404, {"error": "not found"})
                if path == "/api/evaluations":
                    return self._json(200, evaluations.list())
                if path.startswith("/api/evaluations/docs/"):
                    return self._json(200, evaluations.document(unquote(path[len("/api/evaluations/docs/"):])))
                if path.startswith("/api/evaluations/"):
                    return self._json(200, evaluations.get(unquote(path[len("/api/evaluations/"):])))
                if path == "/api/library":
                    if jobs.library is None:
                        return self._json(200, {"exists": False, "available": False, "documents": []})
                    return self._json(200, jobs.library.listing())
                return self._json(404, {"error": "not found"})
            except StudioError as exc:
                return self._json(exc.status, {"error": str(exc)})

        @staticmethod
        def _context_args(query: Dict[str, List[str]]):
            source, passage = (query.get("source") or [""])[0], (query.get("passage") or [""])[0]
            if not source or not passage:
                raise StudioError(400, "source and passage are required")
            return source, passage

        def _static(self, name: str) -> None:
            if not _STATIC_NAME.fullmatch(name):
                return self._json(404, {"error": "not found"})
            path = STATIC / name
            if not path.is_file() or path.parent.resolve() != STATIC.resolve():
                return self._json(404, {"error": "not found"})
            self._send(200, path.read_bytes(), STATIC_TYPES[path.suffix])

        def _export(self, name: str, data: Dict[str, Any], kind: str) -> None:
            if kind == "md":
                body, ctype = run_markdown(name, data).encode("utf-8"), "text/markdown; charset=utf-8"
            elif kind == "json":
                body, ctype = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8"
            else:
                return self._json(400, {"error": "format must be md or json"})
            self._send(200, body, ctype, headers={"Content-Disposition": f'attachment; filename="{name}.{kind}"'})

        # ------------------------------------------------------------------------------------------ POST
        def do_POST(self):
            path = urlsplit(self.path).path
            limit = MAX_UPLOAD_BYTES if path == "/api/library/upload" else 20_000
            length = int(self.headers.get("Content-Length") or 0)
            if length > limit:
                self.close_connection = True
                return self._json(413, {"error": "request too large"})
            raw = self.rfile.read(length)             # read before any reply, or Windows aborts the connection
            if not self._host_ok():
                return None
            cancel = _CANCEL.fullmatch(path)
            provider_check = _PROVIDER_TEST.fullmatch(path)
            if path not in {"/api/run", "/api/library/upload", "/api/integrations/ollama/test", "/api/export/obsidian",
                            "/api/providers/config"} and not cancel and not provider_check:
                return self._json(404, {"error": "not found"})
            origin = self.headers.get("Origin")
            if origin and origin not in allowed_origins:
                return self._json(403, {"error": "cross-origin requests are not accepted"})
            if cancel:
                return self._cancel(cancel.group(1))
            if path == "/api/library/upload":
                return self._upload(raw)
            if path == "/api/integrations/ollama/test":            # a fresh check of the local runtime
                return self._json(200, ollama_integration(studio_status(refresh=True)))
            if provider_check:                                   # one live search, only when the user asks
                from .search import EXCLUSION_NOTE, is_evidence_provider
                if not is_evidence_provider(provider_check.group(1)):          # source policy: nothing is sent
                    return self._json(200, {"id": provider_check.group(1), "ok": False, "excluded": True,
                                            "error": f"{EXCLUSION_NOTE}, so it is not searched."})
                return self._json(200, provider_test(provider_check.group(1), config_store.load()))
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                return self._json(415, {"error": "send JSON"})
            try:
                body = json.loads(raw or b"{}")
            except ValueError:
                return self._json(400, {"error": "malformed JSON"})
            if not isinstance(body, dict):
                return self._json(400, {"error": "malformed JSON"})
            if path == "/api/export/obsidian":
                return self._obsidian(str(body.get("run", "")))
            if path == "/api/providers/config":
                try:
                    config_store.update(body)
                except ConfigError as exc:
                    return self._json(400, {"error": str(exc)})
                if status is not None:
                    status.invalidate()                           # the provider list may have changed
                return self._json(200, providers_payload())
            text = str(body.get("input", "")).strip()
            mode = body.get("mode", "research")
            if mode == "documents-index":
                text = ""
            elif not text or len(text) > MAX_INPUT or mode not in RESEARCH_MODES | DOCUMENT_MODES:
                return self._json(400, {"error": f"enter a question or claim (1-{MAX_INPUT} characters)"})
            if mode in DOCUMENT_MODES and jobs.library is None:
                return self._json(503, {"error": "the document assistant is not configured in this server"})
            options = None
            if mode in RESEARCH_MODES:
                try:
                    options = _run_options(body, config_store.load())
                except ValueError as exc:
                    return self._json(400, {"error": str(exc)})
            self._json(202, {"id": jobs.submit(text, mode, options) if options else jobs.submit(text, mode)})

        def _cancel(self, job_id: str) -> None:
            outcome = jobs.cancel(job_id)
            if outcome is None:
                return self._json(404, {"error": "unknown run"})
            if outcome["state"] == "cancelled":
                return self._json(200, dict(outcome, cancelled=True))
            message = ("This run has already started. The pipeline cannot be interrupted safely, so it will finish "
                       "and be saved as usual." if outcome["state"] == "running" else "This run has already finished.")
            self._json(409, dict(outcome, cancelled=False, error=message))

        def _obsidian(self, name: str) -> None:
            try:
                data = runs.get(name)
                saved = vault.export(str(data.get("input") or name), str(data.get("finished") or data.get("started") or ""),
                                     run_markdown(name, data))
            except StudioError as exc:
                return self._json(exc.status, {"error": str(exc)})
            self._json(201, {"saved": saved, "vault": vault.check()["vault"]})

        def _upload(self, raw: bytes) -> None:
            if jobs.library is None:
                return self._json(503, {"error": "the document library is not configured in this server"})
            if not self.headers.get("Content-Type", "").startswith("application/octet-stream"):
                return self._json(415, {"error": "send the file as application/octet-stream"})
            try:
                name = jobs.library.save_upload(unquote(self.headers.get("X-Filename") or ""), raw)
            except StudioError as exc:
                return self._json(exc.status, {"error": str(exc)})
            self._json(201, {"saved": name, "library": jobs.library.listing()})

    return Handler


def studio_orchestrator(model: str, verifier: str, injection_guard: str = "off",
                        quote_attestation: bool = False) -> Orchestrator:
    """The studio's pipeline. A request without a research depth still runs the original single-pass pipeline, but
    never with a provider the source policy excludes (Wikipedia), even when WEBRAG_PROVIDERS names it. The command
    line and the evaluation build their own Orchestrator and are unchanged."""
    from .search import is_evidence_provider
    orchestrator = Orchestrator(Settings(model=model, verifier=verifier, injection_guard=injection_guard,
                                         quote_attestation=quote_attestation))
    orchestrator.providers = [p for p in orchestrator.providers if is_evidence_provider(p.name)]
    return orchestrator


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m webrag.server")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model", default="qwen3:4b")
    from docqa.verify import VERIFIERS
    parser.add_argument("--verifier", default="quote_stance", choices=sorted(VERIFIERS),
                        help="quote_stance (default, recommended); quote_stance_adjudicated (experimental opt-in: "
                             "adjudicates conflicting evidence; in exploratory runs it raised false support, see "
                             "evaluation/webrag/ADJUDICATED_VERIFIER.md); frozen_b; others are evaluation variants")
    parser.add_argument("--injection-guard", default="off", choices=("off", "flag", "exclude"),
                        help="screen retrieved passages for instruction-like text (prompt injection): off (default), "
                             "flag (record only) or exclude (withhold them from every model call); see "
                             "docs/reverse_engineering/EVALUATION_RESULTS.md")
    parser.add_argument("--quote-attestation", action="store_true",
                        help="record where each verifier quote occurs in its passage (record only: never changes a "
                             "verdict; needs a quote-based verifier)")
    parser.add_argument("--docs", default=str(DOCS_DIR), help="document folder for the docqa assistant")
    parser.add_argument("--obsidian-vault", default=os.environ.get("WEBRAG_OBSIDIAN_VAULT", ""),
                        help="Obsidian vault for Markdown export (new notes go to its 'Research Studio' folder)")
    args = parser.parse_args(argv)
    try:                                         # fail at start-up, not at the first research job
        Settings(model=args.model, verifier=args.verifier, injection_guard=args.injection_guard,
                 quote_attestation=args.quote_attestation)
    except ValueError as exc:
        parser.error(str(exc))
    library = Library(Path(args.docs))
    config_store = ConfigStore()
    jobs = JobQueue(lambda: studio_orchestrator(args.model, args.verifier, args.injection_guard,
                                                args.quote_attestation), library)
    probe = StatusProbe(args.model, args.verifier, config=config_store.load)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(jobs, args.port, status=probe,
                                                                         vault=ObsidianVault(args.obsidian_vault),
                                                                         config_store=config_store))
    print(f"Multi-Agent Research Studio running at http://127.0.0.1:{args.port}  (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
