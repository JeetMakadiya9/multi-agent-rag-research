"""Remote model clients (docqa/remote_llm.py) against a stub service on this computer. Every response below is
SYNTHETIC TEST DATA; no real model service is contacted."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from docqa.llm import LLMError
from docqa.remote_llm import (OpenAICompatibleChat, RemoteOllamaChat, build_client, check_base_url, endpoint_label)

KEY = "sk-test-key-not-real-0123456789"


class Stub:
    """A loopback HTTP service answering from a per-path script of (status, body) pairs; records requests."""

    def __init__(self):
        self.script, self.requests = {}, []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _answer(self, body=None):
                stub.requests.append({"method": self.command, "path": self.path, "headers": dict(self.headers),
                                      "body": json.loads(body) if body else None})
                queue = stub.script.get(self.path) or [(404, {"error": "no such path"})]
                status, payload = queue.pop(0) if len(queue) > 1 else queue[0]
                data = json.dumps(payload).encode() if not isinstance(payload, bytes) else payload
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                if status == 429:
                    self.send_header("Retry-After", "0")
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self._answer()

            def do_POST(self):
                self._answer(self.rfile.read(int(self.headers.get("Content-Length") or 0)))

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def stub():
    service = Stub()
    yield service
    service.close()


SCHEMA = {"type": "object", "properties": {"label": {"type": "string"}}, "required": ["label"]}


def test_ollama_chat_sends_schema_as_format_and_key_as_header(stub):
    stub.script["/api/chat"] = [(200, {"message": {"content": '{"label": "SUPPORTS"}'}, "done_reason": "stop",
                                       "eval_count": 7, "prompt_eval_count": 30})]
    client = RemoteOllamaChat("qwen3:4b", stub.url, KEY)
    response = client.chat([{"role": "user", "content": "x"}], response_format=SCHEMA, num_predict=50)
    assert response.text == '{"label": "SUPPORTS"}' and response.eval_count == 7 and client.history == [response]
    sent = stub.requests[-1]
    assert sent["body"]["format"] == SCHEMA and sent["body"]["options"]["num_predict"] == 50
    assert sent["body"]["stream"] is False and sent["headers"]["Authorization"] == f"Bearer {KEY}"


def test_openai_chat_schema_mode(stub):
    stub.script["/v1/chat/completions"] = [(200, {"choices": [{"message": {"content": '{"label": "NEUTRAL"}'},
                                                                "finish_reason": "stop"}],
                                                   "usage": {"completion_tokens": 5, "prompt_tokens": 20}})]
    client = OpenAICompatibleChat("model-a", stub.url + "/v1", KEY)
    response = client.chat([{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
                           response_format=SCHEMA, num_predict=40)
    body = stub.requests[-1]["body"]
    assert body["response_format"] == {"type": "json_schema", "json_schema": {"name": "response", "schema": SCHEMA}}
    assert body["max_tokens"] == 40 and body["messages"][-1]["content"] == "u"
    assert response.text == '{"label": "NEUTRAL"}' and response.done_reason == "stop" and not response.truncated


def test_openai_object_mode_puts_the_schema_in_the_prompt(stub):
    stub.script["/v1/chat/completions"] = [(200, {"choices": [{"message": {"content": "{}"}, "finish_reason": "length"}]})]
    client = OpenAICompatibleChat("model-a", stub.url + "/v1", json_mode="object")
    response = client.chat([{"role": "user", "content": "u"}], response_format=SCHEMA)
    body = stub.requests[-1]["body"]
    assert body["response_format"] == {"type": "json_object"}
    assert body["messages"][-1]["content"].startswith("u\n\nReply with one JSON object only") and '"label"' in body["messages"][-1]["content"]
    assert response.truncated
    assert "Authorization" not in stub.requests[-1]["headers"]                       # no key configured, none sent


def test_thinking_is_removed_and_empty_answers_fail(stub):
    stub.script["/api/chat"] = [(200, {"message": {"content": "<think>hmm</think>\n{\"a\": 1}"}}),
                                (200, {"message": {"content": "<think>only thinking</think>"}})]
    client = RemoteOllamaChat("m", stub.url)
    assert client.chat([{"role": "user", "content": "x"}]).text == '{"a": 1}'
    with pytest.raises(LLMError) as error:
        client.chat([{"role": "user", "content": "x"}])
    assert error.value.code == "empty"


@pytest.mark.parametrize("status, code, words", [(401, "http_error", "refused the API key"),
                                                 (403, "http_error", "refused the API key"),
                                                 (404, "http_error", "offered there"),
                                                 (400, "http_error", "HTTP 400")])
def test_service_errors_are_explained(stub, status, code, words):
    stub.script["/v1/chat/completions"] = [(status, {"error": {"message": "detail from service"}})]
    client = OpenAICompatibleChat("m", stub.url + "/v1", KEY, retries=0)
    with pytest.raises(LLMError) as error:
        client.chat([{"role": "user", "content": "x"}])
    assert error.value.code == code and words in str(error.value) and "detail from service" in str(error.value)
    assert KEY not in str(error.value)


def test_rate_limits_are_retried_then_reported(stub):
    ok = (200, {"choices": [{"message": {"content": "fine"}, "finish_reason": "stop"}]})
    stub.script["/v1/chat/completions"] = [(429, {"error": "slow down"}), ok]
    client = OpenAICompatibleChat("m", stub.url + "/v1", retries=2)
    assert client.chat([{"role": "user", "content": "x"}]).text == "fine"
    stub.script["/v1/chat/completions"] = [(429, {"error": "slow down"})]
    with pytest.raises(LLMError, match="rate limiting"):
        OpenAICompatibleChat("m", stub.url + "/v1", retries=1).chat([{"role": "user", "content": "x"}])


def test_malformed_and_unreachable(stub):
    stub.script["/v1/chat/completions"] = [(200, b"not json")]
    with pytest.raises(LLMError) as error:
        OpenAICompatibleChat("m", stub.url + "/v1").chat([{"role": "user", "content": "x"}])
    assert error.value.code == "model_error"
    stub.script["/v1/chat/completions"] = [(200, {"choices": []})]
    with pytest.raises(LLMError, match="without choices"):
        OpenAICompatibleChat("m", stub.url + "/v1").chat([{"role": "user", "content": "x"}])
    port = stub.server.server_address[1]
    stub.close()
    with pytest.raises(LLMError) as error:
        RemoteOllamaChat("m", f"http://127.0.0.1:{port}", timeout=5).chat([{"role": "user", "content": "x"}])
    assert error.value.code == "unreachable"


def test_describe_reports_availability_without_the_key(stub):
    stub.script["/api/tags"] = [(200, {"models": [{"name": "qwen3:4b"}, {"name": "llama3.2:3b"}]})]
    stub.script["/api/version"] = [(200, {"version": "9.9.9"})]
    info = RemoteOllamaChat("qwen3:4b", stub.url, KEY).describe()
    assert info["service_reachable"] and info["model_available"] and info["service_version"] == "9.9.9"
    assert info["ollama_version"] is None and info["provider"] == "ollama" and info["endpoint"] == stub.url
    missing = RemoteOllamaChat("other:1b", stub.url, KEY).describe()
    assert missing["service_reachable"] and not missing["model_available"] and "not among the 2 models" in missing["error"]
    stub.script["/v1/models"] = [(401, {"error": "bad key"})]
    refused = OpenAICompatibleChat("m", stub.url + "/v1", KEY).describe()
    assert not refused["service_reachable"] and "refused the API key" in refused["error"]
    for row in (info, missing, refused):
        assert KEY not in json.dumps(row)
    assert KEY not in repr(OpenAICompatibleChat("m", stub.url + "/v1", KEY))


@pytest.mark.parametrize("url, ok", [("https://api.example.org/v1", True), ("http://127.0.0.1:11434", True),
                                     ("http://localhost:8000/v1", True), ("http://api.example.org/v1", False),
                                     ("https://user:pw@api.example.org", False), ("https://api.example.org/v1?key=x", False),
                                     ("ftp://api.example.org", False), ("", False), ("https://", False)])
def test_service_addresses(url, ok):
    if ok:
        assert check_base_url(url) == url.rstrip("/")
    else:
        with pytest.raises(ValueError):
            check_base_url(url)


def test_build_client_and_labels():
    assert isinstance(build_client("ollama", "m", "https://h.example"), RemoteOllamaChat)
    assert isinstance(build_client("openai", "m", "https://h.example/v1", json_mode="none"), OpenAICompatibleChat)
    with pytest.raises(ValueError):
        build_client("anthropic", "m", "https://h.example")
    with pytest.raises(ValueError):
        build_client("openai", "", "https://h.example")
    with pytest.raises(ValueError):
        OpenAICompatibleChat("m", "https://h.example", json_mode="xml")
    assert endpoint_label("https://h.example:8443/v1/x") == "https://h.example:8443"
