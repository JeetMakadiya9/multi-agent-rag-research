"""Model configuration of the local Studio (webrag/model_service.py), the Studio server's reporting of a configured
model service, and the orchestrator's model check. Every record and response below is SYNTHETIC TEST DATA; no model
service or search provider is contacted."""
from __future__ import annotations

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from webrag.model_service import ModelConfig, allow_model_downloads, load_env_file, settings_overview

KEY = "sk-test-key-not-real-0123456789"


# ------------------------------------------------------------------------------------------------ configuration
def test_default_is_the_local_ollama():
    config = ModelConfig.from_env({})
    assert (config.provider, config.model, config.remote, config.problems()) == ("local", "qwen3:4b", False, [])
    assert type(config.client()).__name__ == "OllamaChat"
    assert ModelConfig.from_env({"WEBRAG_LLM_MODEL": "llama3.2:3b"}, model="qwen3:4b").model == "qwen3:4b"   # --model wins


def test_remote_service_configuration():
    config = ModelConfig.from_env({"WEBRAG_LLM_PROVIDER": "OpenAI", "WEBRAG_LLM_BASE_URL": "https://api.example.org/v1",
                                   "WEBRAG_LLM_API_KEY": KEY, "WEBRAG_LLM_MODEL": "m-large",
                                   "WEBRAG_LLM_VERIFIER_MODEL": "m-small"})
    assert config.provider == "openai" and config.remote and config.problems() == []
    assert config.endpoint == "https://api.example.org" and KEY not in repr(config)
    client, verifier = config.clients()
    assert client.model == "m-large" and verifier.model == "m-small" and client is not verifier


@pytest.mark.parametrize("env, words", [
    ({"WEBRAG_LLM_PROVIDER": "anthropic"}, "WEBRAG_LLM_PROVIDER"),
    ({"WEBRAG_LLM_PROVIDER": "openai", "WEBRAG_LLM_BASE_URL": "http://api.example.org/v1"}, "https://"),
    ({"WEBRAG_LLM_PROVIDER": "openai", "WEBRAG_LLM_BASE_URL": "https://a.example/v1", "WEBRAG_LLM_JSON_MODE": "xml"}, "JSON_MODE"),
    ({"WEBRAG_LLM_TIMEOUT": "abc"}, "TIMEOUT"),
    ({"WEBRAG_LLM_VERIFIER_MODEL": "x"}, "VERIFIER_MODEL"),
])
def test_configuration_problems_name_the_setting(env, words):
    problems = ModelConfig.from_env(env).problems()
    assert problems and any(words in p for p in problems)


def test_env_file(tmp_path):
    path = tmp_path / ".env"
    path.write_text('# comment\nWEBRAG_LLM_PROVIDER=openai\nexport WEBRAG_LLM_MODEL = "m-large"\n'
                    f"WEBRAG_LLM_API_KEY='{KEY}'\nnot a setting\n\nOPENALEX_API_KEY=\n"
                    "WEBRAG_LLM_JSON_MODE=object   # services without JSON schemas\nWEBRAG_CONTACT='a #b'\n",
                    encoding="utf-8")
    env = {"WEBRAG_LLM_PROVIDER": "ollama"}                                  # the real environment wins
    names = load_env_file(path, env)
    assert names == ["WEBRAG_LLM_PROVIDER", "WEBRAG_LLM_MODEL", "WEBRAG_LLM_API_KEY", "OPENALEX_API_KEY",
                     "WEBRAG_LLM_JSON_MODE", "WEBRAG_CONTACT"]
    assert env["WEBRAG_LLM_JSON_MODE"] == "object" and env["WEBRAG_CONTACT"] == "a #b"     # quoted values keep '#'
    assert env["WEBRAG_LLM_PROVIDER"] == "ollama" and env["WEBRAG_LLM_MODEL"] == "m-large" and env["WEBRAG_LLM_API_KEY"] == KEY
    assert load_env_file(tmp_path / "missing.env", {}) == []
    overview = settings_overview(env)
    assert overview["secrets"]["WEBRAG_LLM_API_KEY"] is True and overview["secrets"]["OPENALEX_API_KEY"] is False
    assert KEY not in json.dumps(overview)


def test_model_downloads_policy(tmp_path):
    env = {"HF_HUB_CACHE": str(tmp_path)}
    assert allow_model_downloads(env) is True and env["HF_HUB_OFFLINE"] == "0"           # nothing cached yet
    cached = {"HF_HUB_CACHE": str(tmp_path)}
    for name in ("models--sentence-transformers--all-MiniLM-L6-v2", "models--cross-encoder--ms-marco-MiniLM-L-6-v2"):
        (tmp_path / name).mkdir()
    assert allow_model_downloads(cached) is False and "HF_HUB_OFFLINE" not in cached
    assert allow_model_downloads({"WEBRAG_MODEL_DOWNLOADS": "0"}) is False
    forced = {"WEBRAG_MODEL_DOWNLOADS": "1", "HF_HUB_OFFLINE": "1"}
    assert allow_model_downloads(forced) is True and forced["HF_HUB_OFFLINE"] == "1"       # an explicit setting wins


# ------------------------------------------------------------------------------------------------ studio server
class StubClient:
    """A configured model service that answers describe() but never generates (no research is run here)."""

    def __init__(self, model, reachable=True, available=True):
        self.model, self.reachable, self.available = model, reachable, available
        self.history = []

    def describe(self):
        return {"model": self.model, "provider": "openai", "provider_label": "OpenAI-compatible API",
                "endpoint": "https://api.example.org", "service_reachable": self.reachable,
                "model_available": self.available, "service_version": None, "ollama_version": None,
                "model_digest": None, "error": "" if self.available else "not listed"}

    def list_models(self):
        return [self.model] if self.available else ["other"]

    def chat(self, *args, **kwargs):
        from docqa.llm import LLMError
        raise LLMError("unreachable", "stub")


def test_studio_reports_a_configured_service(tmp_path, monkeypatch):
    from webrag.config import ConfigStore
    from webrag.server import JobQueue, StatusProbe, make_handler
    from webrag.studio import RunStore
    config = ModelConfig.from_env({"WEBRAG_LLM_PROVIDER": "openai", "WEBRAG_LLM_BASE_URL": "https://api.example.org/v1",
                                   "WEBRAG_LLM_API_KEY": KEY, "WEBRAG_LLM_MODEL": "m-large"})
    stub = StubClient("m-large")
    monkeypatch.setattr(ModelConfig, "clients", lambda self: (stub, stub))
    jobs = JobQueue(config.orchestrator_factory("quote_stance"))
    probe = StatusProbe(config.model, "quote_stance", describe=config.describe, models=config.list_models)
    server = ThreadingHTTPServer(("127.0.0.1", 0), None)
    port = server.server_address[1]
    server.RequestHandlerClass = make_handler(jobs, port, runs=RunStore(tmp_path / "runs"), status=probe,
                                              config_store=ConfigStore(tmp_path / "config.json"))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        get = lambda path: json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=10).read())
        status = get("/api/status")
        assert status["runtime"]["provider"] == "openai" and status["runtime"]["model_available"] is True
        assert status["model_installed"] is True and status["models"] == ["m-large"] and status["ollama_version"] is None
        integrations = get("/api/integrations")
        ai = next(i for i in integrations["integrations"] if i["id"] == "ollama")
        assert ai["name"] == "Model service" and ai["status"]["state"] == "ready" and "api.example.org" in ai["data"]
        assert KEY not in json.dumps(status) + json.dumps(integrations)
    finally:
        server.shutdown()
        server.server_close()


# ------------------------------------------------------------------------------------------------ model check
@pytest.mark.parametrize("reachable, available, words", [(False, False, "could not be used"),
                                                         (True, False, "is not offered by https://api.example.org")])
def test_orchestrator_stops_when_the_service_cannot_run_the_model(reachable, available, words):
    from webrag.orchestrator import Orchestrator, Settings
    stub = StubClient("m-large", reachable, available)
    data = Orchestrator(Settings(save=False), client=stub, session=object(), providers=[], verifier=object()).run("Q?")
    assert data["status"] == "FAILED" and words in data["error"]
    assert ("no other model" in data["error"]) is (reachable and not available)
    assert data["stages"][0]["name"] == "model check" and data["stages"][0]["status"] == "failed"
    assert data["model"]["provider"] == "openai"


def test_orchestrator_checks_a_separate_verifier_model():
    from webrag.orchestrator import Orchestrator, Settings

    class Verifier:
        client = StubClient("m-small", True, False)

    data = Orchestrator(Settings(save=False), client=StubClient("m-large"), session=object(), providers=[],
                        verifier=Verifier()).run("Q?")
    assert data["status"] == "FAILED" and "verifier model 'm-small'" in data["error"]
    assert data["model"]["verifier_model"]["model"] == "m-small"


def test_orchestrator_records_the_service_when_the_check_passes():
    from webrag.orchestrator import Orchestrator, Settings

    class Verifier:
        client = StubClient("m-small")

    data = Orchestrator(Settings(save=False), client=StubClient("m-large"), session=object(), providers=[],
                        verifier=Verifier()).run("Q?")
    first = data["stages"][0]
    assert first["status"] == "ok" and first["detail"] == "m-large on OpenAI-compatible API (https://api.example.org); verifier m-small"
