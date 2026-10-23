"""Hosted Studio (webrag/hosted_studio.py): configuration, sign-in gateway and the forwarded Studio server, plus the
model check for a configured model service. Every record and response below is SYNTHETIC TEST DATA; no model service
or search provider is contacted."""
from __future__ import annotations

import pytest

from webrag.hosted_studio import HostedConfig, LoginLimiter, allow_model_downloads

TOKEN = "correct-horse-battery-staple-42"
KEY = "sk-test-key-not-real-0123456789"


# ------------------------------------------------------------------------------------------------ configuration
def test_config_from_env_and_problems(tmp_path):
    config = HostedConfig.from_env({"WEBRAG_LLM_PROVIDER": "OpenAI", "WEBRAG_LLM_BASE_URL": "https://api.example.org/v1",
                                    "WEBRAG_LLM_API_KEY": KEY, "WEBRAG_LLM_MODEL": "m-large",
                                    "WEBRAG_LLM_VERIFIER_MODEL": "m-small", "STUDIO_ACCESS_TOKEN": TOKEN,
                                    "WEBRAG_DATA_DIR": str(tmp_path)})
    assert config.provider == "openai" and config.problems() == [] and config.token_problem() == ""
    assert config.endpoint == "https://api.example.org" and KEY not in repr(config) and TOKEN not in repr(config)
    client, verifier = config.clients()
    assert client.model == "m-large" and verifier.model == "m-small" and client is not verifier


@pytest.mark.parametrize("env, words", [
    ({"WEBRAG_LLM_PROVIDER": "anthropic"}, "WEBRAG_LLM_PROVIDER"),
    ({"WEBRAG_LLM_PROVIDER": "openai", "WEBRAG_LLM_BASE_URL": "http://api.example.org/v1"}, "https://"),
    ({"WEBRAG_LLM_PROVIDER": "openai", "WEBRAG_LLM_BASE_URL": "https://a.example/v1", "WEBRAG_LLM_JSON_MODE": "xml"}, "JSON_MODE"),
    ({"WEBRAG_LLM_TIMEOUT": "abc"}, "TIMEOUT"),
    ({"WEBRAG_VERIFIER": "oracle"}, "WEBRAG_VERIFIER"),
    ({"WEBRAG_INJECTION_GUARD": "maybe"}, "INJECTION_GUARD"),
    ({"WEBRAG_LLM_VERIFIER_MODEL": "x"}, "VERIFIER_MODEL"),
])
def test_config_problems_name_the_setting(env, words):
    problems = HostedConfig.from_env(env).problems()
    assert problems and any(words in p for p in problems)


@pytest.mark.parametrize("token, locked", [("", True), ("short", True), (TOKEN, False)])
def test_access_token_rules(token, locked):
    assert bool(HostedConfig.from_env({"STUDIO_ACCESS_TOKEN": token}).token_problem()) is locked


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


def test_login_limiter():
    limiter = LoginLimiter(limit=2, window=60)
    assert not limiter.blocked("a")
    limiter.failed("a"), limiter.failed("a")
    assert limiter.blocked("a") and not limiter.blocked("b")


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
    assert first["status"] == "ok" and "m-large on OpenAI-compatible API (https://api.example.org); verifier m-small" == first["detail"]
