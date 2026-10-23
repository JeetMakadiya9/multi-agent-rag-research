"""The Studio's verification options (verifier, injection guard, quote attestation): resolved from the command line,
then WEBRAG_* settings (environment or .env), then the defaults; invalid values stop the start; the running Studio
reports what it was built with. Offline: no model, no network, no server is left running."""
from __future__ import annotations

import pytest

from webrag.model_service import studio_options

NAMES = ("WEBRAG_VERIFIER", "WEBRAG_INJECTION_GUARD", "WEBRAG_QUOTE_ATTESTATION")


def test_defaults():         # Arm C (Studio-adapted variant), injection guard exclude, attestation on (it quotes)
    assert studio_options(env={}) == {"verifier": "arm_c_studio", "injection_guard": "exclude",
                                      "quote_attestation": True,
                                      "sources": {"verifier": "default", "injection_guard": "default",
                                                  "quote_attestation": "default"}}


def test_attestation_defaults_follow_the_verifier():
    assert studio_options("quote_stance", env={})["quote_attestation"] is True
    options = studio_options("frozen_b", env={})                       # does not quote: off by default, no error
    assert options["quote_attestation"] is False
    assert options["sources"]["quote_attestation"] == "default (off: frozen_b does not quote)"
    assert studio_options(quote_attestation=False, env={})["quote_attestation"] is False


def test_settings_from_the_environment_and_their_sources():
    options = studio_options(env={"WEBRAG_VERIFIER": "quote_stance_ranked", "WEBRAG_INJECTION_GUARD": "Flag",
                                  "WEBRAG_QUOTE_ATTESTATION": "yes"})
    assert (options["verifier"], options["injection_guard"], options["quote_attestation"]) == \
        ("quote_stance_ranked", "flag", True)
    assert options["sources"]["verifier"] == "environment (WEBRAG_VERIFIER)"
    assert studio_options(env={"WEBRAG_QUOTE_ATTESTATION": "0"})["quote_attestation"] is False


def test_the_command_line_wins():
    options = studio_options("quote_stance_adjudicated", "exclude", False,
                             env={"WEBRAG_VERIFIER": "quote_stance_ranked", "WEBRAG_INJECTION_GUARD": "flag",
                                  "WEBRAG_QUOTE_ATTESTATION": "1"})
    assert (options["verifier"], options["injection_guard"], options["quote_attestation"]) == \
        ("quote_stance_adjudicated", "exclude", False)
    assert set(options["sources"].values()) == {"command line"}


@pytest.mark.parametrize("env, words", [({"WEBRAG_VERIFIER": "best"}, "verifier 'best' is unknown"),
                                        ({"WEBRAG_INJECTION_GUARD": "block"}, "injection guard 'block'"),
                                        ({"WEBRAG_QUOTE_ATTESTATION": "maybe"}, "WEBRAG_QUOTE_ATTESTATION 'maybe'"),
                                        ({"WEBRAG_VERIFIER": "frozen_b", "WEBRAG_QUOTE_ATTESTATION": "1"},
                                         "needs a quote-based verifier")])
def test_an_invalid_value_is_an_error_never_a_silent_default(env, words):
    with pytest.raises(ValueError, match=words):
        studio_options(env=env)


def test_status_reports_the_options_the_pipeline_was_built_with():
    from webrag.server import StatusProbe
    describe = lambda: {"ollama_version": None, "model_digest": None}      # noqa: E731
    options = studio_options(env={"WEBRAG_VERIFIER": "quote_stance_ranked", "WEBRAG_INJECTION_GUARD": "flag"})
    status = StatusProbe("qwen3:4b", options["verifier"], describe=describe, providers=lambda: [], models=lambda: [],
                         options=options).get()
    assert (status["verifier"], status["injection_guard"], status["quote_attestation"]) == \
        ("quote_stance_ranked", "flag", True)                  # it quotes, so attestation is on by default
    assert status["option_sources"]["injection_guard"] == "environment (WEBRAG_INJECTION_GUARD)"
    assert status["document_verifier"] == "frozen_b"
    plain = StatusProbe("qwen3:4b", "quote_stance", describe=describe, providers=lambda: [], models=lambda: []).get()
    assert (plain["injection_guard"], plain["quote_attestation"]) == ("off", False)


class _Stop(Exception):
    pass


def _start(monkeypatch, tmp_path, argv, env):
    """Run ``webrag.server.main`` up to the point of serving; return the orchestrator its job queue builds and the
    status probe. Nothing listens on a port."""
    from webrag import model_service, server
    for name in NAMES + ("WEBRAG_LLM_PROVIDER", "WEBRAG_LLM_BASE_URL", "WEBRAG_LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(model_service, "load_env_file", lambda *a, **k: [])      # never the developer's .env
    captured = {}

    class Queue:
        def __init__(self, factory, library):
            captured["factory"] = factory

    def handler(jobs, port, status=None, **kwargs):
        captured["probe"] = status
        raise _Stop

    monkeypatch.setattr(server, "JobQueue", Queue)
    monkeypatch.setattr(server, "make_handler", handler)
    with pytest.raises(_Stop):
        server.main(argv + ["--docs", str(tmp_path)])
    return captured["factory"](), captured["probe"]


def test_the_server_builds_the_pipeline_from_the_environment(monkeypatch, tmp_path):
    from docqa.attestation import AttestationMixin
    from docqa.ranked import RankedQuoteStanceVerifier
    orch, probe = _start(monkeypatch, tmp_path, [], {"WEBRAG_VERIFIER": "quote_stance_ranked",
                                                     "WEBRAG_INJECTION_GUARD": "flag",
                                                     "WEBRAG_QUOTE_ATTESTATION": "1"})
    assert isinstance(orch.verifier, RankedQuoteStanceVerifier) and isinstance(orch.verifier, AttestationMixin)
    assert (orch.settings.verifier, orch.settings.injection_guard, orch.settings.quote_attestation) == \
        ("quote_stance_ranked", "flag", True)
    assert probe.options["verifier"] == "quote_stance_ranked" and probe.verifier == "quote_stance_ranked"


def test_the_server_command_line_overrides_the_environment(monkeypatch, tmp_path):
    from docqa.verify import QuoteStanceVerifier
    orch, probe = _start(monkeypatch, tmp_path, ["--verifier", "quote_stance"], {"WEBRAG_VERIFIER": "quote_stance_ranked"})
    assert isinstance(orch.verifier, QuoteStanceVerifier) and orch.settings.injection_guard == "exclude"
    assert probe.options["sources"]["verifier"] == "command line"


def test_the_server_refuses_an_invalid_setting(monkeypatch, tmp_path, capsys):
    from webrag import model_service, server
    monkeypatch.setattr(model_service, "load_env_file", lambda *a, **k: [])
    monkeypatch.setenv("WEBRAG_INJECTION_GUARD", "block")
    for name in ("WEBRAG_LLM_PROVIDER", "WEBRAG_LLM_BASE_URL", "WEBRAG_VERIFIER", "WEBRAG_QUOTE_ATTESTATION"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(SystemExit):
        server.main(["--docs", str(tmp_path)])
    assert "injection guard 'block'" in capsys.readouterr().err


def test_the_launcher_reports_and_validates_the_options(monkeypatch, capsys):
    from dashboard import launch
    from webrag import model_service
    from webrag.model_service import ModelConfig
    monkeypatch.setattr(launch, "missing_packages", lambda: [])
    monkeypatch.setattr(model_service, "load_env_file", lambda *a, **k: [])
    for name in NAMES + ("WEBRAG_LLM_PROVIDER", "WEBRAG_LLM_BASE_URL"):
        monkeypatch.delenv(name, raising=False)

    class Offline:
        def describe(self):
            return {"ollama_version": None, "model_digest": None}

    monkeypatch.setattr(ModelConfig, "client", lambda self, model=None: Offline())
    monkeypatch.setattr(launch, "port_free", lambda port: True)
    assert launch.main(["--check"]) == 0
    assert ("Verifier: arm_c_studio (default: Arm C agentic verification, Studio-adapted variant); injection guard: "
            "exclude; quote attestation: on (record only).") in capsys.readouterr().out
    monkeypatch.setenv("WEBRAG_VERIFIER", "quote_stance_ranked")
    monkeypatch.setenv("WEBRAG_QUOTE_ATTESTATION", "1")
    assert launch.main(["--check"]) == 0
    assert "Verifier: quote_stance_ranked (chosen in .env)" in capsys.readouterr().out
    monkeypatch.setenv("WEBRAG_VERIFIER", "best")
    assert launch.main(["--check"]) == 1
    assert "verifier 'best' is unknown" in capsys.readouterr().out


def test_the_server_builds_the_agentic_default_without_any_setting(monkeypatch, tmp_path):
    from docqa.agentic import StudioArmCVerifier
    orch, probe = _start(monkeypatch, tmp_path, [], {})
    assert type(orch.verifier) is StudioArmCVerifier and orch.settings.verifier == "arm_c_studio"
    assert orch.verifier.attest is True and orch.settings.injection_guard == "exclude"
    assert orch.store_factory.__name__ == "guarded_store"                # every evidence store is screened
    assert probe.options["verifier"] == "arm_c_studio" and probe.options["sources"]["verifier"] == "default"
    status = probe.get()
    assert status["verifier"] == status["default_verifier"] == "arm_c_studio"
    assert status["verifier_description"].startswith("Arm C agentic verification, Studio-adapted variant")
    assert status["verifier_kind"] == "Studio-adapted Arm C (default)"
    assert (status["injection_guard"], status["quote_attestation"]) == ("exclude", True)


def test_the_server_can_turn_attestation_off(monkeypatch, tmp_path):
    orch, probe = _start(monkeypatch, tmp_path, ["--no-quote-attestation"], {})
    assert orch.verifier.attest is False and probe.options["quote_attestation"] is False


def test_the_documented_launch_starts_the_server_without_overriding_the_defaults(monkeypatch):
    """start_studio.bat runs dashboard/launch.py, which starts ``python -m webrag.server --port <port>``: no verifier,
    guard or attestation flag, so the server's own defaults (or .env) decide."""
    from dashboard import launch
    calls = []

    class Process:
        def __init__(self, args, **kwargs):
            calls.append(args)
            raise SystemExit(0)                      # stop before anything is started

    monkeypatch.setattr(launch.subprocess, "Popen", Process)
    monkeypatch.setattr(launch, "missing_packages", lambda: [])
    monkeypatch.setattr(launch, "port_free", lambda port: True)
    from webrag import model_service
    monkeypatch.setattr(model_service, "load_env_file", lambda *a, **k: [])
    for name in NAMES + ("WEBRAG_LLM_PROVIDER", "WEBRAG_LLM_BASE_URL"):
        monkeypatch.delenv(name, raising=False)

    class Offline:
        def describe(self):
            return {"ollama_version": None, "model_digest": None}

    monkeypatch.setattr(model_service.ModelConfig, "client", lambda self, model=None: Offline())
    with pytest.raises(SystemExit):
        launch.main(["--no-browser", "--studio-port", "8899"])
    assert calls and calls[0][1:] == ["-m", "webrag.server", "--port", "8899"]


def test_a_request_cannot_change_the_verification_settings():
    from webrag.server import _run_options
    for key in ("verifier", "injection_guard", "quote_attestation", "followup_rounds", "planner"):
        with pytest.raises(ValueError, match="fixed when the Studio starts"):
            _run_options({key: "off", "depth": "standard"}, {})
