"""
Provider-independent LLM interface for Multi-Agent RAG.

The research system should not depend on one model provider.  Agents call
this small interface, while the concrete provider (currently Ollama) lives
behind it.

The default path is local Ollama, so no paid API is required.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Sequence, Union


Message = Dict[str, str]


class LLMProvider(Protocol):
    """Minimal interface required by research agents."""

    def generate(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        timeout: int = 120,
    ) -> str:
        ...


@dataclass
class OllamaProvider:
    """Local Ollama chat provider."""

    model: str = "qwen3:4b"
    base_url: str = "http://127.0.0.1:11434/api/chat"
    think: Optional[bool] = None
    format: Optional[Union[str, Dict[str, Any]]] = None

    def generate(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        timeout: int = 120,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": list(messages),
            "stream": True,
            "options": {"temperature": temperature},
        }
        if self.think is not None:
            payload["think"] = self.think
        if self.format is not None:
            payload["format"] = self.format

        request = urllib.request.Request(
            self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                content_parts = []
                for line in response:
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line.decode("utf-8"))
                    except json.JSONDecodeError as exc:
                        raise RuntimeError("Ollama returned an invalid streaming JSON event.") from exc

                    if event.get("error"):
                        raise RuntimeError(f"Ollama returned an error: {event['error']}")

                    content = event.get("message", {}).get("content", "")
                    if content:
                        content_parts.append(str(content))
                    if event.get("done"):
                        break
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Could not reach Ollama at {self.base_url}. "
                "Make sure Ollama is running."
            ) from exc

        content = "".join(content_parts)
        if not content:
            raise RuntimeError("Ollama returned an empty response.")

        return str(content).strip()


class StaticProvider:
    """Deterministic provider useful for unit tests.

    A string is returned on every call, preserving the original behavior. A
    sequence supplies one response per call, which supports multi-step flows
    such as claim extraction followed by one or more claim verdicts.
    """

    def __init__(self, response: Union[str, Sequence[str]]):
        self.response = response
        self._responses = None if isinstance(response, str) else list(response)
        self._call_index = 0

    def generate(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        timeout: int = 120,
    ) -> str:
        if self._responses is None:
            return str(self.response)
        if self._call_index >= len(self._responses):
            raise RuntimeError("StaticProvider response sequence exhausted.")
        response = self._responses[self._call_index]
        self._call_index += 1
        return response


__all__ = ["Message", "LLMProvider", "OllamaProvider", "StaticProvider"]
