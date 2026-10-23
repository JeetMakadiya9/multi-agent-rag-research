"""A small local-only Ollama chat client.

Unlike the generic client in ``llm.py`` it reports ``done_reason`` (so truncation at the output cap is visible),
token counts and latency, and it refuses non-local hosts and ``:cloud`` models so no data leaves the machine.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


class LLMError(RuntimeError):
    """Model or service failure. ``code`` is one of: unreachable, http_error, timeout, model_error, empty."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass
class LLMResponse:
    text: str
    done_reason: Optional[str]
    eval_count: Optional[int]
    prompt_eval_count: Optional[int]
    seconds: float

    @property
    def truncated(self) -> bool:
        return self.done_reason == "length"


class OllamaChat:
    def __init__(self, model: str = "qwen3:4b", base_url: str = "http://127.0.0.1:11434", num_predict: int = 512,
                 think: bool = False, timeout: int = 180):
        host = urllib.parse.urlparse(base_url).hostname
        if host not in LOCAL_HOSTS:
            raise ValueError(f"Only a local Ollama server is allowed, not {host!r}.")
        if model.endswith(":cloud"):
            raise ValueError("Cloud models are not allowed; choose a local model.")
        self.model, self.base_url = model, base_url.rstrip("/")
        self.num_predict, self.think, self.timeout = num_predict, think, timeout
        self.history: List[LLMResponse] = []            # every call, for truncation and latency accounting

    def chat(self, messages: Sequence[Dict[str, str]], *, response_format: Any = None, temperature: float = 0.0,
             num_predict: Optional[int] = None, timeout: Optional[int] = None) -> LLMResponse:
        payload: Dict[str, Any] = {"model": self.model, "messages": list(messages), "stream": True, "think": self.think,
                                   "options": {"temperature": temperature,
                                               "num_predict": num_predict or self.num_predict}}
        if response_format is not None:
            payload["format"] = response_format
        request = urllib.request.Request(self.base_url + "/api/chat", data=json.dumps(payload).encode("utf-8"),
                                         headers={"Content-Type": "application/json"}, method="POST")
        started, parts, final = time.perf_counter(), [], {}
        try:
            with urllib.request.urlopen(request, timeout=timeout or self.timeout) as response:
                for line in response:
                    if not line.strip():
                        continue
                    event = json.loads(line.decode("utf-8"))
                    if event.get("error"):
                        raise LLMError("model_error", str(event["error"]))
                    parts.append(event.get("message", {}).get("content", ""))
                    if event.get("done"):
                        final = event
                        break
        except LLMError:
            raise
        except urllib.error.HTTPError as exc:
            raise LLMError("http_error", f"HTTP {exc.code} from Ollama (is the model '{self.model}' pulled?)") from None
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError) or "timed out" in str(exc.reason):
                raise LLMError("timeout", f"no response within {timeout or self.timeout}s") from None
            raise LLMError("unreachable", f"cannot reach Ollama at {self.base_url} (is it running?)") from None
        except TimeoutError:
            raise LLMError("timeout", f"no response within {timeout or self.timeout}s") from None
        result = LLMResponse(text="".join(parts).strip(), done_reason=final.get("done_reason"),
                             eval_count=final.get("eval_count"), prompt_eval_count=final.get("prompt_eval_count"),
                             seconds=round(time.perf_counter() - started, 3))
        self.history.append(result)
        if not result.text:
            raise LLMError("empty", "the model returned no text")
        return result

    def describe(self) -> Dict[str, Optional[str]]:
        """Server version and the local model's digest, for run manifests (local GET requests only)."""
        info: Dict[str, Optional[str]] = {"model": self.model, "ollama_version": None, "model_digest": None}
        try:
            with urllib.request.urlopen(self.base_url + "/api/version", timeout=5) as response:
                info["ollama_version"] = json.loads(response.read()).get("version")
            with urllib.request.urlopen(self.base_url + "/api/tags", timeout=5) as response:
                for row in json.loads(response.read()).get("models", []):
                    if row.get("name") == self.model or row.get("model") == self.model:
                        info["model_digest"] = row.get("digest")
        except (urllib.error.URLError, TimeoutError, ValueError):
            pass
        return info


class JsonJudge:
    """``LLMProvider``-compatible adapter (``generate(messages, temperature, timeout) -> str``) so the frozen
    Experiment B verifier can run on this client with its own frozen settings: JSON output, thinking off, 256 tokens."""

    def __init__(self, client: OllamaChat, num_predict: int = 256):
        self.client, self.num_predict = client, num_predict

    def generate(self, messages: Sequence[Dict[str, str]], *, temperature: float = 0.0, timeout: int = 120) -> str:
        return self.client.chat(messages, response_format="json", temperature=temperature,
                                num_predict=self.num_predict, timeout=timeout).text
