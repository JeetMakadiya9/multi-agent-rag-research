"""Local Ollama chat adapter pinned to CPU for repeatable smoke execution."""
from __future__ import annotations

import json
import urllib.request
from typing import Any, Sequence


class OllamaCPUProvider:
    """LLMProvider-compatible Ollama adapter with GPU offload disabled."""
    def __init__(self, model: str = "qwen3:4b",
                 base_url: str = "http://127.0.0.1:11434/api/chat",
                 num_predict: int = 256, think: bool = False,
                 response_format: str | None = None):
        self.model = model
        self.base_url = base_url
        self.num_predict = num_predict
        self.think = think
        self.response_format = response_format
        self.calls = 0

    def generate(self, messages: Sequence[dict[str, str]], *,
                 temperature: float = 0.0, timeout: int = 120) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "stream": True,
            "think": self.think,
            "options": {"temperature": temperature, "num_gpu": 0,
                        "num_predict": self.num_predict},
        }
        if self.response_format is not None:
            payload["format"] = self.response_format
        request = urllib.request.Request(
            self.base_url, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        parts = []
        self.calls += 1
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for line in response:
                if not line.strip():
                    continue
                event = json.loads(line.decode("utf-8"))
                if event.get("error"):
                    raise RuntimeError(f"Ollama returned an error: {event['error']}")
                content = event.get("message", {}).get("content", "")
                if content:
                    parts.append(str(content))
                if event.get("done"):
                    break
        output = "".join(parts).strip()
        if not output:
            raise RuntimeError("Ollama returned an empty response.")
        return output
