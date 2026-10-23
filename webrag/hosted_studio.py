"""The Research Studio inside a hosted Streamlit app (``streamlit_cloud/streamlit_app.py``).

The Studio's own server (``webrag.server``: job queue, run store, library, status probe and every ``/api`` route)
runs unchanged on a loopback port inside the app's process. A gateway mounted on the Streamlit server at ``studio/``
serves it to the browser: it requires a sign-in, refuses cross-site requests, limits request sizes and forwards each
request to the loopback server with that server's own host and origin, so the Studio's checks keep applying.

The language model is chosen by server-side configuration (environment variables; on Streamlit Community Cloud,
top-level secrets):

    WEBRAG_LLM_PROVIDER         local (default: the local Ollama, as in the desktop Studio) | ollama | openai
    WEBRAG_LLM_BASE_URL         service address (https://...; for openai including the /v1 prefix)
    WEBRAG_LLM_API_KEY          service key, when the service needs one (kept on the server)
    WEBRAG_LLM_MODEL            model name (default qwen3:4b)
    WEBRAG_LLM_VERIFIER_MODEL   optional separate model for the verifier (same service)
    WEBRAG_LLM_JSON_MODE        openai only: schema (default) | object | none
    WEBRAG_LLM_TIMEOUT          seconds per model call (default 180)
    WEBRAG_VERIFIER             research verifier (default quote_stance)
    WEBRAG_INJECTION_GUARD      off (default) | flag | exclude
    STUDIO_ACCESS_TOKEN         required: the sign-in token (16 characters or more)
    WEBRAG_DATA_DIR             where uploaded documents and the provider settings are kept (runs are saved where the
                                orchestrator saves them, outputs/webrag/runs/, as in the desktop Studio)

Nothing here prints or returns the API key or the access token.
"""
from __future__ import annotations

import hashlib
import hmac
import html
import http.client
import os
import re
import socket
import threading
import time
from dataclasses import dataclass
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = ROOT / "outputs" / "hosted"
COOKIE = "studio_auth"
SESSION_SECONDS = 12 * 3600
MIN_TOKEN = 16
MAX_BODY = 20 * 1024 * 1024 + 4096          # the Studio's own upload limit plus room for headers of the form
FORWARD_TIMEOUT = 180
_PATH = re.compile(r"(?:|index\.html|favicon\.ico|static/[a-z0-9][a-z0-9_-]{0,63}\.(?:css|mjs|js|svg)|api/[A-Za-z0-9._~%/@:+-]{1,400})")
_FORWARD_HEADERS = ("content-type", "accept", "x-filename")
_RETURN_HEADERS = ("content-type", "content-disposition", "cache-control", "x-content-type-options", "referrer-policy",
                   "content-security-policy")


# ================================================================================================= configuration
@dataclass(frozen=True)
class HostedConfig:
    provider: str = "local"
    base_url: str = ""
    api_key: str = ""
    model: str = "qwen3:4b"
    verifier_model: str = ""
    json_mode: str = "schema"
    timeout: int = 180
    verifier: str = "quote_stance"
    injection_guard: str = "off"
    access_token: str = ""
    data_dir: Path = DEFAULT_DATA_DIR

    @classmethod
    def from_env(cls, env: Optional[Dict[str, str]] = None) -> "HostedConfig":
        env = os.environ if env is None else env
        get = lambda name, default="": str(env.get(name, default) or default).strip()
        try:
            timeout = int(get("WEBRAG_LLM_TIMEOUT", "180"))
        except ValueError:
            timeout = -1
        return cls(provider=get("WEBRAG_LLM_PROVIDER", "local").lower(), base_url=get("WEBRAG_LLM_BASE_URL"),
                   api_key=get("WEBRAG_LLM_API_KEY"), model=get("WEBRAG_LLM_MODEL", "qwen3:4b"),
                   verifier_model=get("WEBRAG_LLM_VERIFIER_MODEL"), json_mode=get("WEBRAG_LLM_JSON_MODE", "schema"),
                   timeout=timeout, verifier=get("WEBRAG_VERIFIER", "quote_stance"),
                   injection_guard=get("WEBRAG_INJECTION_GUARD", "off"), access_token=get("STUDIO_ACCESS_TOKEN"),
                   data_dir=Path(get("WEBRAG_DATA_DIR") or DEFAULT_DATA_DIR))

    def __repr__(self) -> str:                                     # never show the key or the token
        return (f"HostedConfig(provider={self.provider!r}, endpoint={self.endpoint!r}, model={self.model!r}, "
                f"verifier_model={self.verifier_model!r}, verifier={self.verifier!r})")

    @property
    def endpoint(self) -> str:
        if self.provider == "local":
            return "http://127.0.0.1:11434"
        from docqa.remote_llm import endpoint_label
        try:
            return endpoint_label(self.base_url)
        except ValueError:
            return ""

    def problems(self) -> List[str]:
        """Configuration errors, by setting name (never values of secrets)."""
        from docqa.remote_llm import JSON_MODES, PROVIDERS, check_base_url
        from docqa.verify import VERIFIERS
        out = []
        if self.provider not in ("local", *PROVIDERS):
            out.append(f"WEBRAG_LLM_PROVIDER must be local, {' or '.join(PROVIDERS)}")
        if self.provider in PROVIDERS:
            try:
                check_base_url(self.base_url)
            except ValueError as exc:
                out.append(f"WEBRAG_LLM_BASE_URL: {exc}")
        if self.provider == "openai" and self.json_mode not in JSON_MODES:
            out.append(f"WEBRAG_LLM_JSON_MODE must be one of {', '.join(JSON_MODES)}")
        if not 10 <= self.timeout <= 900:
            out.append("WEBRAG_LLM_TIMEOUT must be a number of seconds between 10 and 900")
        if self.verifier not in VERIFIERS:
            out.append(f"WEBRAG_VERIFIER must be one of {', '.join(sorted(VERIFIERS))}")
        if self.injection_guard not in ("off", "flag", "exclude"):
            out.append("WEBRAG_INJECTION_GUARD must be off, flag or exclude")
        if self.verifier_model and self.provider == "local":
            out.append("WEBRAG_LLM_VERIFIER_MODEL needs a configured model service (WEBRAG_LLM_PROVIDER ollama or openai)")
        return out

    def token_problem(self) -> str:
        if not self.access_token:
            return "STUDIO_ACCESS_TOKEN is not set, so the hosted Studio stays locked"
        if len(self.access_token) < MIN_TOKEN:
            return f"STUDIO_ACCESS_TOKEN is shorter than {MIN_TOKEN} characters, so the hosted Studio stays locked"
        return ""

    # ---------------------------------------------------------------------------------------------- model clients
    def client(self, model: Optional[str] = None):
        if self.provider == "local":
            from docqa.llm import OllamaChat
            return OllamaChat(model=model or self.model)
        from docqa.remote_llm import build_client
        return build_client(self.provider, model or self.model, self.base_url, self.api_key, self.json_mode,
                            timeout=self.timeout)

    def clients(self) -> Tuple[Any, Any]:
        """(model client, verifier client); the same object unless a separate verifier model is configured."""
        client = self.client()
        if self.verifier_model and self.verifier_model != self.model:
            return client, self.client(self.verifier_model)
        return client, client


RETRIEVAL_MODELS = ("sentence-transformers/all-MiniLM-L6-v2", "cross-encoder/ms-marco-MiniLM-L-6-v2")


def allow_model_downloads(env=None) -> bool:
    """Whether the retrieval models (rag.py) may be downloaded from the Hugging Face Hub. docqa.index keeps the Hub
    offline unless told otherwise; a hosted app starts without the models, so it must fetch them once. Set
    WEBRAG_MODEL_DOWNLOADS to 1 or 0 to decide; the default (auto) allows it only when a model is not cached yet.
    Call before docqa.index is imported."""
    env = os.environ if env is None else env
    choice = str(env.get("WEBRAG_MODEL_DOWNLOADS", "auto")).strip().lower()
    if choice in ("0", "1"):
        allow = choice == "1"
    else:
        cache = Path(env.get("HF_HUB_CACHE") or Path(env.get("HF_HOME") or Path.home() / ".cache" / "huggingface") / "hub")
        allow = not all((cache / ("models--" + name.replace("/", "--"))).is_dir() for name in RETRIEVAL_MODELS)
    if allow:
        env.setdefault("HF_HUB_OFFLINE", "0")
        env.setdefault("TRANSFORMERS_OFFLINE", "0")
    return allow


# ================================================================================================= studio backend
def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class StudioBackend:
    """The unchanged Studio server, built with the configured model clients and kept under the data folder."""

    def __init__(self, config: HostedConfig, port: Optional[int] = None):
        from docqa.verify import VERIFIERS
        from .config import ConfigStore
        from .integrations import ObsidianVault
        from .orchestrator import RUNS_DIR, Orchestrator, Settings
        from .search import is_evidence_provider
        from .server import JobQueue, StatusProbe, make_handler
        from .studio import Library, RunStore

        self.config = config
        self.problems = config.problems()
        data = Path(config.data_dir)
        for folder in (RUNS_DIR, data / "documents"):
            folder.mkdir(parents=True, exist_ok=True)
        self.port = port or _free_port()

        def orchestrator():
            if self.problems:
                raise RuntimeError("The model service is not configured correctly: " + "; ".join(self.problems))
            client, verifier_client = config.clients()
            built = Orchestrator(Settings(model=config.model, verifier=config.verifier,
                                          injection_guard=config.injection_guard),
                                 client=client, verifier=VERIFIERS[config.verifier](verifier_client))
            built.providers = [p for p in built.providers if is_evidence_provider(p.name)]
            return built

        def assistant(folder: str, log: Callable[[str], None] = print):
            from docqa.answer import AnswerGenerator
            from docqa.index import EvidenceIndex
            from docqa.ingest import load_documents
            from docqa.pipeline import Assistant
            report = load_documents(folder)
            for name, reason in report.skipped:
                log(f"[skipped] {name}: {reason}")
            if not report.pages:
                raise SystemExit("No readable TXT or PDF documents in the library.")
            client, verifier_client = config.clients()
            return Assistant(EvidenceIndex(report.pages), AnswerGenerator(client),
                             VERIFIERS["frozen_b"](verifier_client), k=4), report

        def describe() -> Dict[str, Any]:
            if self.problems:
                return {"model": config.model, "provider": config.provider if config.provider != "local" else "ollama",
                        "provider_label": "Model service", "endpoint": config.endpoint or "not configured",
                        "service_reachable": False, "model_available": False, "service_version": None,
                        "ollama_version": None, "model_digest": None, "error": "; ".join(self.problems)}
            client, verifier_client = config.clients()
            info = client.describe()
            if verifier_client is not client:
                info["verifier_model"] = verifier_client.describe()
            return info

        def models() -> List[str]:
            try:
                return config.clients()[0].list_models()
            except Exception:
                return []

        remote = config.provider != "local"
        self.jobs = JobQueue(orchestrator, Library(data / "documents", assistant_factory=assistant if remote else None))
        self.status = StatusProbe(config.model, config.verifier, describe=describe if remote or self.problems else None,
                                  models=models if remote else None)
        self.runs = RunStore(RUNS_DIR)                 # where Orchestrator saves every finished run
        handler = make_handler(self.jobs, self.port, runs=self.runs, status=self.status, vault=ObsidianVault(None),
                               config_store=ConfigStore(data / "studio-config.json"))
        self.server = ThreadingHTTPServer(("127.0.0.1", self.port), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, name="studio-backend", daemon=True)

    def start(self) -> "StudioBackend":
        self.thread.start()
        return self

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def forward(self, method: str, path: str, headers: Dict[str, str], body: bytes) -> Tuple[int, List[Tuple[str, str]], bytes]:
        """One request to the Studio server, addressed as the Studio expects (its own host and origin)."""
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=FORWARD_TIMEOUT)
        try:
            out = {"Host": f"127.0.0.1:{self.port}", **headers}
            if body or method == "POST":
                out["Content-Length"] = str(len(body))
            connection.request(method, path, body=body or None, headers=out)
            response = connection.getresponse()
            return response.status, response.getheaders(), response.read()
        finally:
            connection.close()


# ================================================================================================= gateway
def session_value(token: str) -> str:
    return hmac.new(token.encode("utf-8"), b"webrag-hosted-studio-session-v1", hashlib.sha256).hexdigest()


class LoginLimiter:
    """At most ``limit`` failed sign-ins per client address in ``window`` seconds."""

    def __init__(self, limit: int = 8, window: float = 900.0):
        self.limit, self.window = limit, window
        self._failures: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

    def blocked(self, client: str) -> bool:
        now = time.monotonic()
        with self._lock:
            recent = [t for t in self._failures.get(client, []) if now - t < self.window]
            self._failures[client] = recent
            return len(recent) >= self.limit

    def failed(self, client: str) -> None:
        with self._lock:
            self._failures.setdefault(client, []).append(time.monotonic())
            if len(self._failures) > 10_000:                   # bound memory
                self._failures.clear()


PAGE_CSP = "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; base-uri 'none'; frame-ancestors 'self'"


def _page(title: str, body: str, status: int = 200):
    from starlette.responses import HTMLResponse
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(title)}</title>
<style>body{{font:16px/1.5 system-ui,sans-serif;background:#f5f2ea;color:#1b1a17;margin:0;display:grid;place-items:center;min-height:100vh}}
main{{max-width:28rem;padding:2rem;background:#fffdf8;border:1px solid #ddd6c8;border-radius:12px}}
input{{width:100%;box-sizing:border-box;padding:.6rem;font:inherit;margin:.5rem 0 1rem;border:1px solid #bdb5a6;border-radius:8px}}
button{{padding:.6rem 1.2rem;font:inherit;background:#2f6f5e;color:#fff;border:0;border-radius:8px;cursor:pointer}}
.err{{color:#a3382b}} .muted{{color:#6b665c;font-size:.9rem}}
@media (prefers-color-scheme: dark){{body{{background:#1b1a17;color:#ece6da}}main{{background:#25231f;border-color:#3a362f}}input{{background:#1b1a17;color:#ece6da}}}}</style>
</head><body><main>{body}</main></body></html>"""
    return HTMLResponse(document, status_code=status, headers={"Content-Security-Policy": PAGE_CSP,
                                                                "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
                                                                "Referrer-Policy": "no-referrer"})


def _login_page(error: str = "", status: int = 200):
    message = f'<p class="err" role="alert">{html.escape(error)}</p>' if error else ""
    return _page("Sign in · Research Studio", f"""<h1>Research Studio</h1>
<p>This hosted Studio runs research with the operator's model service. Enter the access token to continue.</p>{message}
<form method="post" action="login"><label for="token">Access token</label>
<input id="token" name="token" type="password" autocomplete="current-password" required autofocus>
<button type="submit">Sign in</button></form>
<p class="muted">The token is set by the operator of this app (STUDIO_ACCESS_TOKEN).</p>""", status)


def _client_address(request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()[:64]
    return request.client.host if request.client else "unknown"


def _cross_site(request) -> bool:
    """A request another site made the browser send. Browsers mark these with Sec-Fetch-Site; the session cookie is
    SameSite=Strict as well."""
    return request.headers.get("sec-fetch-site", "") == "cross-site"


def _secure(request) -> bool:
    return request.url.scheme == "https" or request.headers.get("x-forwarded-proto", "") == "https"


def create_gateway(config: HostedConfig, backend: Optional[StudioBackend] = None,
                   limiter: Optional[LoginLimiter] = None):
    """The Starlette app mounted at ``studio``. ``backend`` is started lazily on the first signed-in request when not
    given."""
    import anyio
    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.requests import Request
    from starlette.responses import JSONResponse, RedirectResponse, Response
    from starlette.routing import Route

    limiter = limiter or LoginLimiter()
    state: Dict[str, Any] = {"backend": backend}
    lock = threading.Lock()

    def get_backend() -> StudioBackend:
        with lock:
            if state["backend"] is None:
                state["backend"] = StudioBackend(config).start()
            return state["backend"]

    def signed_in(request: Request) -> bool:
        if config.token_problem():
            return False
        cookie = request.cookies.get(COOKIE, "")
        return bool(cookie) and hmac.compare_digest(cookie, session_value(config.access_token))

    async def login(request: Request):
        if config.token_problem():
            return _page("Studio locked", f"<h1>Studio locked</h1><p>{html.escape(config.token_problem())}. "
                                          "The operator sets it in the app's secrets.</p>", 503)
        if request.method == "GET":
            return _login_page()
        if _cross_site(request):
            return _login_page("Sign-in from another site is not accepted.", 403)
        client = _client_address(request)
        if limiter.blocked(client):
            return _login_page("Too many failed attempts. Try again later.", 429)
        body = await request.body()
        if len(body) > 4096:
            return _login_page("Request too large.", 413)
        from urllib.parse import parse_qs
        token = (parse_qs(body.decode("utf-8", "replace")).get("token") or [""])[0]
        if not hmac.compare_digest(token.encode("utf-8"), config.access_token.encode("utf-8")):
            limiter.failed(client)
            return _login_page("That token is not correct.", 401)
        response = RedirectResponse("./", status_code=303)
        response.set_cookie(COOKIE, session_value(config.access_token), max_age=SESSION_SECONDS, httponly=True,
                            secure=_secure(request), samesite="strict", path="/")
        return response

    async def logout(request: Request):
        response = RedirectResponse("login", status_code=303)
        response.delete_cookie(COOKIE, path="/")
        return response

    async def health(request: Request):
        return JSONResponse({"ok": True, "locked": bool(config.token_problem()),
                             "configuration_problems": len(config.problems())},
                            headers={"Cache-Control": "no-store"})

    async def studio(request: Request):
        path = request.path_params.get("path", "")
        if not signed_in(request):
            if path.startswith("api/"):
                return JSONResponse({"error": "Sign in to the hosted Studio first."}, status_code=401)
            return RedirectResponse(_up(path) + "login", status_code=303)
        if not _PATH.fullmatch(path):
            return JSONResponse({"error": "not found"}, status_code=404)
        if request.method not in ("GET", "HEAD") and _cross_site(request):
            return JSONResponse({"error": "cross-site requests are not accepted"}, status_code=403)
        body = b""
        if request.method == "POST":
            declared = request.headers.get("content-length")
            if declared and declared.isdigit() and int(declared) > MAX_BODY:
                return JSONResponse({"error": "request too large"}, status_code=413)
            chunks, size = [], 0
            async for chunk in request.stream():
                size += len(chunk)
                if size > MAX_BODY:
                    return JSONResponse({"error": "request too large"}, status_code=413)
                chunks.append(chunk)
            body = b"".join(chunks)
        backend = get_backend()
        headers = {k: v for k, v in ((h, request.headers.get(h)) for h in _FORWARD_HEADERS) if v}
        if request.method == "POST":
            headers["Origin"] = f"http://127.0.0.1:{backend.port}"          # checked above: same site
        target = "/" + path + (f"?{request.url.query}" if request.url.query else "")
        try:
            status, raw_headers, content = await anyio.to_thread.run_sync(
                backend.forward, request.method, target, headers, body)
        except (OSError, http.client.HTTPException) as exc:
            return JSONResponse({"error": f"the Studio server did not answer ({type(exc).__name__})"}, status_code=502)
        out = {}
        for name, value in raw_headers:
            if name.lower() in _RETURN_HEADERS:
                out[name] = value.replace("frame-ancestors 'none'", "frame-ancestors 'self'") \
                    if name.lower() == "content-security-policy" else value
        return Response(content, status_code=status, headers=out)

    routes = [Route("/login", login, methods=["GET", "POST"]), Route("/logout", logout, methods=["POST", "GET"]),
              Route("/healthz", health), Route("/{path:path}", studio, methods=["GET", "HEAD", "POST"])]
    return Starlette(routes=routes, middleware=[Middleware(PassThroughStatus)])


# Streamlit Community Cloud's proxy replaces some error responses of an app (seen: 401 and 503) with its own "Internal
# server error." page, so the Studio's message would be lost. Statuses seen to pass unchanged go out as they are; any
# other error goes out as 400 with the real status in X-Studio-Status, which the Studio's API client reads
# (webrag/static/api.mjs).
PASSING_STATUSES = frozenset({200, 201, 202, 204, 206, 301, 302, 303, 304, 307, 308, 400, 404, 405})
STATUS_HEADER = "X-Studio-Status"


class PassThroughStatus:
    """ASGI middleware applying PASSING_STATUSES to every gateway response."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        async def send_wrapper(message):
            if message["type"] == "http.response.start" and message["status"] not in PASSING_STATUSES:
                headers = list(message.get("headers") or [])
                headers.append((STATUS_HEADER.lower().encode("latin-1"), str(message["status"]).encode("latin-1")))
                message = {**message, "status": 400, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_wrapper)


def _up(path: str) -> str:
    """Relative prefix from ``path`` back to the gateway root (redirects stay relative: the app may be served under a
    prefix such as Streamlit Cloud's)."""
    return "../" * path.count("/")
