"""Model clients for a configured model service other than this computer's Ollama: an Ollama server reached over the
network (native API) or an OpenAI-compatible chat-completions service (vLLM, LM Studio, Groq, OpenRouter, Together
and similar).

They have the interface of ``docqa.llm.OllamaChat`` that the pipeline uses: ``chat(messages, response_format=...,
temperature=..., num_predict=..., timeout=...) -> LLMResponse``, ``history``, ``model``, ``base_url`` and
``describe()``. ``docqa.llm`` is unchanged: the local command line and Studio keep using the local Ollama only. These
clients are built only from server-side configuration (``webrag/hosted_studio.py``); API keys are sent in request
headers and never returned by ``describe()`` or written to a run record.

No other model is ever substituted: when the configured model is not offered, ``describe()`` says so and the
orchestrator stops the run with that explanation.
"""
from __future__ import annotations

import ipaddress
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Sequence

from .llm import LLMError, LLMResponse

PROVIDERS = {"ollama": "Ollama server", "openai": "OpenAI-compatible API"}
JSON_MODES = ("schema", "object", "none")
_MODEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,199}")
_THINK = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)


def _loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def endpoint_label(base_url: str) -> str:
    """scheme://host[:port] of a service address: what may be shown to users (no path, query or credentials)."""
    parts = urllib.parse.urlsplit(base_url)
    host = parts.hostname or ""
    return f"{parts.scheme}://{host}" + (f":{parts.port}" if parts.port else "")


def check_base_url(base_url: str) -> str:
    """A service address the operator configured: https, or plain http only for this computer (loopback)."""
    base_url = str(base_url or "").strip().rstrip("/")
    try:
        parts = urllib.parse.urlsplit(base_url)
        host = parts.hostname or ""
        parts.port                                            # raises ValueError for an invalid port
    except ValueError:
        raise ValueError("the model service address is not a valid URL") from None
    if parts.scheme not in ("http", "https") or not host:
        raise ValueError("the model service address must start with https://")
    if parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError("put credentials in the API key setting, not in the model service address")
    if parts.scheme == "http" and not _loopback(host):
        raise ValueError("a model service on another computer must use https://")
    return base_url


class RemoteChat:
    """Shared request handling. Subclasses define the wire format."""

    provider = ""

    def __init__(self, model: str, base_url: str, api_key: str = "", num_predict: int = 512, timeout: int = 180,
                 retries: int = 2, think: bool = False):
        model = str(model or "").strip()
        if not _MODEL.fullmatch(model):
            raise ValueError("set a model name for the model service")
        self.model, self.base_url = model, check_base_url(base_url)
        self._key = str(api_key or "").strip()
        self.num_predict, self.timeout, self.retries, self.think = num_predict, timeout, max(0, int(retries)), think
        self.history: List[LLMResponse] = []
        self.endpoint = endpoint_label(self.base_url)

    def __repr__(self) -> str:                                # never show the key
        return f"{type(self).__name__}(model={self.model!r}, endpoint={self.endpoint!r})"

    # ---------------------------------------------------------------------------------------------- transport
    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._key:
            headers["Authorization"] = f"Bearer {self._key}"
        return headers

    def _request(self, method: str, path: str, payload: Optional[dict] = None, timeout: Optional[float] = None) -> dict:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        attempt = 0
        while True:
            request = urllib.request.Request(self.base_url + path, data=data, headers=self._headers(), method=method)
            try:
                with urllib.request.urlopen(request, timeout=timeout or self.timeout) as response:
                    raw = response.read()
                break
            except urllib.error.HTTPError as exc:
                detail = self._error_detail(exc)
                if exc.code in (429, 500, 502, 503, 504) and attempt < self.retries:
                    attempt += 1
                    time.sleep(self._retry_after(exc, attempt))
                    continue
                if exc.code == 429:
                    raise LLMError("http_error", f"the model service is rate limiting requests (HTTP 429){detail}") from None
                if exc.code in (401, 403):
                    raise LLMError("http_error", f"the model service refused the API key (HTTP {exc.code}){detail}") from None
                if exc.code == 404:
                    raise LLMError("http_error", f"HTTP 404 from the model service: is the model '{self.model}' "
                                                 f"offered there?{detail}") from None
                raise LLMError("http_error", f"HTTP {exc.code} from the model service{detail}") from None
            except urllib.error.URLError as exc:
                if isinstance(exc.reason, TimeoutError) or "timed out" in str(exc.reason):
                    raise LLMError("timeout", f"no response within {timeout or self.timeout}s") from None
                raise LLMError("unreachable", f"cannot reach the model service at {self.endpoint}") from None
            except TimeoutError:
                raise LLMError("timeout", f"no response within {timeout or self.timeout}s") from None
        try:
            body = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise LLMError("model_error", "the model service sent a response that is not JSON") from None
        if not isinstance(body, dict):
            raise LLMError("model_error", "the model service sent an unexpected response")
        return body

    @staticmethod
    def _retry_after(exc: urllib.error.HTTPError, attempt: int) -> float:
        try:
            return min(20.0, max(0.5, float(exc.headers.get("Retry-After", ""))))
        except (TypeError, ValueError):
            return min(20.0, 2.0 ** attempt)

    @staticmethod
    def _error_detail(exc: urllib.error.HTTPError) -> str:
        try:
            body = json.loads(exc.read(4096).decode("utf-8", "replace"))
        except Exception:
            return ""
        error = body.get("error") if isinstance(body, dict) else None
        message = error.get("message") if isinstance(error, dict) else error
        return f": {str(message)[:200]}" if message else ""

    def _finish(self, text: Optional[str], done_reason: Optional[str], eval_count, prompt_count, started: float) -> LLMResponse:
        text = _THINK.sub("", str(text or "")).strip()          # reasoning models may prepend their thinking
        result = LLMResponse(text=text, done_reason=done_reason, eval_count=eval_count, prompt_eval_count=prompt_count,
                             seconds=round(time.perf_counter() - started, 3))
        self.history.append(result)
        if not result.text:
            raise LLMError("empty", "the model returned no text")
        return result

    # ---------------------------------------------------------------------------------------------- interface
    def chat(self, messages: Sequence[Dict[str, str]], *, response_format: Any = None, temperature: float = 0.0,
             num_predict: Optional[int] = None, timeout: Optional[int] = None) -> LLMResponse:
        raise NotImplementedError

    def list_models(self) -> List[str]:
        raise NotImplementedError

    def describe(self) -> Dict[str, Any]:
        """What runs can rely on, for the status page and the run record (never the key). ``ollama_version`` and
        ``model_digest`` describe only a local Ollama and stay empty here."""
        info: Dict[str, Any] = {"model": self.model, "provider": self.provider,
                                "provider_label": PROVIDERS[self.provider], "endpoint": self.endpoint,
                                "service_reachable": False, "model_available": False, "service_version": None,
                                "ollama_version": None, "model_digest": None, "error": ""}
        try:
            models = self.list_models()
        except LLMError as exc:
            info["error"] = str(exc)
            return info
        info["service_reachable"] = True
        info["model_available"] = self.model in models
        if not info["model_available"]:
            info["error"] = f"'{self.model}' is not among the {len(models)} models the service lists"
        info.update(self._extra_description())
        return info

    def _extra_description(self) -> Dict[str, Any]:
        return {}


class RemoteOllamaChat(RemoteChat):
    """An Ollama server on another computer or a hosted Ollama service, through Ollama's own API (``/api/chat``,
    ``/api/tags``). JSON schemas are passed as Ollama's ``format``, exactly as the local client does."""

    provider = "ollama"

    def chat(self, messages, *, response_format=None, temperature=0.0, num_predict=None, timeout=None):
        payload: Dict[str, Any] = {"model": self.model, "messages": list(messages), "stream": False, "think": self.think,
                                   "options": {"temperature": temperature, "num_predict": num_predict or self.num_predict}}
        if response_format is not None:
            payload["format"] = response_format
        started = time.perf_counter()
        body = self._request("POST", "/api/chat", payload, timeout)
        if body.get("error"):
            raise LLMError("model_error", str(body["error"])[:300])
        message = body.get("message")
        if not isinstance(message, dict):
            raise LLMError("model_error", "the model service sent a response without a message")
        return self._finish(message.get("content"), body.get("done_reason"), body.get("eval_count"),
                            body.get("prompt_eval_count"), started)

    def list_models(self) -> List[str]:
        rows = self._request("GET", "/api/tags", timeout=10).get("models") or []
        return sorted({str(r.get(k)) for r in rows if isinstance(r, dict) for k in ("name", "model") if r.get(k)})

    def _extra_description(self) -> Dict[str, Any]:
        try:
            return {"service_version": self._request("GET", "/api/version", timeout=10).get("version")}
        except LLMError:
            return {}


class OpenAICompatibleChat(RemoteChat):
    """A ``/chat/completions`` service; ``base_url`` includes the version prefix (for example ``.../v1``).

    ``json_mode`` chooses how the pipeline's JSON schemas are requested: ``schema`` (``response_format`` type
    ``json_schema``), ``object`` (type ``json_object``; the schema is then given to the model as an instruction) or
    ``none`` (instruction only). Services differ in what they accept, so it is a setting rather than a guess."""

    provider = "openai"

    def __init__(self, *args, json_mode: str = "schema", **kwargs):
        super().__init__(*args, **kwargs)
        if json_mode not in JSON_MODES:
            raise ValueError(f"json mode must be one of {', '.join(JSON_MODES)}")
        self.json_mode = json_mode

    def _format(self, messages: List[Dict[str, str]], response_format: Any):
        if response_format is None:
            return messages, None
        schema = response_format if isinstance(response_format, dict) else None
        if self.json_mode == "schema" and schema is not None:
            return messages, {"type": "json_schema", "json_schema": {"name": "response", "schema": schema}}
        note = "Reply with one JSON object only" + (f" that matches this JSON schema: {json.dumps(schema)}"
                                                     if schema is not None else ".")
        messages = [*messages[:-1], {**messages[-1], "content": f"{messages[-1]['content']}\n\n{note}"}] if messages else messages
        return messages, ({"type": "json_object"} if self.json_mode in ("object", "schema") else None)

    def chat(self, messages, *, response_format=None, temperature=0.0, num_predict=None, timeout=None):
        messages, fmt = self._format(list(messages), response_format)
        payload: Dict[str, Any] = {"model": self.model, "messages": messages, "temperature": temperature,
                                   "max_tokens": num_predict or self.num_predict, "stream": False}
        if fmt is not None:
            payload["response_format"] = fmt
        started = time.perf_counter()
        body = self._request("POST", "/chat/completions", payload, timeout)
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise LLMError("model_error", "the model service sent a response without choices")
        message = choices[0].get("message") if isinstance(choices[0].get("message"), dict) else {}
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        reason = choices[0].get("finish_reason")
        return self._finish(message.get("content"), "length" if reason == "length" else reason,
                            usage.get("completion_tokens"), usage.get("prompt_tokens"), started)

    def list_models(self) -> List[str]:
        rows = self._request("GET", "/models", timeout=10).get("data") or []
        return sorted({str(r["id"]) for r in rows if isinstance(r, dict) and r.get("id")})


def build_client(provider: str, model: str, base_url: str, api_key: str = "", json_mode: str = "schema",
                 timeout: int = 180, think: bool = False) -> RemoteChat:
    if provider == "ollama":
        return RemoteOllamaChat(model, base_url, api_key, timeout=timeout, think=think)
    if provider == "openai":
        return OpenAICompatibleChat(model, base_url, api_key, timeout=timeout, json_mode=json_mode)
    raise ValueError(f"model provider must be one of {', '.join(PROVIDERS)}")
