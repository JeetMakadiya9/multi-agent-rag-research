"""Local Ollama chat client with a content-addressed replay cache and cost accounting.

Every request is deterministic (temperature 0, fixed seed, JSON mode, thinking off). The cache key is the SHA-256 of
the canonical request, so an identical request is answered from disk and costs nothing; ``Usage`` records whether a
call was live or replayed, so reported costs can count both (the protocol reports *logical* calls and tokens, i.e.
what the arm would cost without the cache).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

MODEL = "qwen3:4b"
EXPECTED_DIGEST = "359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7"
ENDPOINT = "http://127.0.0.1:11434"
# num_ctx 4096 keeps the whole model and its KV cache on the 4 GB laptop GPU; 8192 split it with the CPU and ran
# out of system memory (development run dev01, aborted). Every prompt here stays far below 4096 tokens.
OPTIONS = {"temperature": 0, "seed": 0, "num_ctx": 4096, "num_gpu": 99}     # num_gpu 99: every layer on the GPU


class LLMError(RuntimeError):
    """The endpoint failed or returned no usable message."""


@dataclass
class Usage:
    calls: int = 0
    live_calls: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    by_role: Dict[str, int] = field(default_factory=dict)

    def add(self, role: str, prompt_tokens: int, output_tokens: int, seconds: float, live: bool) -> None:
        self.calls += 1
        self.live_calls += int(live)
        self.prompt_tokens += prompt_tokens
        self.output_tokens += output_tokens
        self.seconds += seconds
        self.by_role[role] = self.by_role.get(role, 0) + 1

    def as_dict(self) -> Dict[str, Any]:
        return {"calls": self.calls, "live_calls": self.live_calls, "prompt_tokens": self.prompt_tokens,
                "output_tokens": self.output_tokens, "total_tokens": self.prompt_tokens + self.output_tokens,
                "model_seconds": round(self.seconds, 3), "by_role": dict(sorted(self.by_role.items()))}


def model_digest(endpoint: str = ENDPOINT) -> str:
    with urllib.request.urlopen(f"{endpoint}/api/tags", timeout=10) as response:
        tags = json.loads(response.read())
    for entry in tags.get("models", []):
        if entry.get("name") == MODEL or entry.get("model") == MODEL:
            return entry.get("digest", "")
    return ""


class ChatClient:
    """``complete(role, system, user, max_tokens)`` returns the reply text; costs go to the ``usage`` passed in."""

    def __init__(self, cache_dir: Optional[Path], endpoint: str = ENDPOINT, timeout: float = 240.0,
                 transport: Any = None):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.endpoint, self.timeout = endpoint, timeout
        self.transport = transport or self._post
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _post(self, body: Dict[str, Any]) -> Dict[str, Any]:
        request = urllib.request.Request(f"{self.endpoint}/api/chat", json.dumps(body).encode("utf-8"),
                                         {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read())
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise LLMError(f"{type(exc).__name__}") from None

    def complete(self, role: str, system: str, user: str, max_tokens: int, usage: Usage) -> str:
        body = {"model": MODEL, "stream": False, "think": False, "format": "json",
                "options": dict(OPTIONS, num_predict=max_tokens),
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        key = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        path = self.cache_dir / key[:2] / f"{key}.json" if self.cache_dir else None
        if path is not None and path.is_file():
            record = json.loads(path.read_text(encoding="utf-8"))
            usage.add(role, record["prompt_tokens"], record["output_tokens"], record["seconds"], live=False)
            return record["content"]
        start = time.perf_counter()
        try:
            reply = self.transport(body)
        except LLMError:                          # one retry for an endpoint error (infrastructure, not model output)
            time.sleep(5)
            reply = self.transport(body)
        seconds = time.perf_counter() - start
        content = (reply.get("message") or {}).get("content")
        if not isinstance(content, str):
            raise LLMError("NO_MESSAGE")
        record = {"content": content, "prompt_tokens": int(reply.get("prompt_eval_count") or 0),
                  "output_tokens": int(reply.get("eval_count") or 0), "seconds": seconds,
                  "done_reason": reply.get("done_reason")}
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, path)
        usage.add(role, record["prompt_tokens"], record["output_tokens"], seconds, live=True)
        return content


def parse_json(text: str) -> Optional[Any]:
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        start, end = (text or "").find("{"), (text or "").rfind("}")
        if 0 <= start < end:
            try:
                return json.loads(text[start:end + 1])
            except ValueError:
                return None
        return None
