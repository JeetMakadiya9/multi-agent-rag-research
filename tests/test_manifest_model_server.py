"""Run manifests record the model server's version, the model digest and the loaded context length
(evaluation/studio_eval/run_eval.model_server_state). A fake Ollama server on 127.0.0.1 answers; nothing else is
contacted."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from evaluation.studio_eval import run_eval

DIGEST = "359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7"


@pytest.fixture
def fake_ollama():
    replies = {"/api/version": {"version": "0.40.11"},
               "/api/tags": {"models": [{"name": "qwen3:4b", "digest": DIGEST}]},
               "/api/ps": {"models": [{"name": "qwen3:4b", "context_length": 4096}]}}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps(replies.get(self.path, {})).encode()
            self.send_response(200 if self.path in replies else 404)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}", replies
    server.shutdown()


def test_version_digest_and_context_are_recorded(fake_ollama):
    from docqa.llm import OllamaChat
    base, _ = fake_ollama
    state = run_eval.model_server_state(OllamaChat(model="qwen3:4b", base_url=base))
    assert state == {"ollama_version": "0.40.11", "model_digest": DIGEST, "context_length": 4096,
                     "model": "qwen3:4b", "errors": []}


def test_a_model_that_is_not_loaded_yet_has_no_context_and_says_why(fake_ollama):
    from docqa.llm import OllamaChat
    base, replies = fake_ollama
    replies["/api/ps"] = {"models": []}
    state = run_eval.model_server_state(OllamaChat(model="qwen3:4b", base_url=base))
    assert state["context_length"] is None and state["model_digest"] == DIGEST
    assert any("not loaded" in e for e in state["errors"])


def test_an_unreachable_server_is_recorded_not_fatal():
    from docqa.llm import OllamaChat
    state = run_eval.model_server_state(OllamaChat(model="qwen3:4b", base_url="http://127.0.0.1:9"), timeout=1)
    assert state["ollama_version"] is None and state["context_length"] is None and state["errors"]


def test_clients_without_a_server_description_are_handled():
    class Stub:
        model = "stub"
    state = run_eval.model_server_state(Stub())
    assert state["model"] == "stub" and state["errors"] == ["the model client cannot describe its server"]


def test_a_remote_base_url_is_never_asked_for_its_loaded_models():
    class Remote:
        model, base_url = "big", "https://api.example.com/v1"

        def describe(self):
            return {"ollama_version": None, "model_digest": None}
    assert run_eval.model_server_state(Remote())["errors"] == []      # no api/ps request outside localhost
