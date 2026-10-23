"""Local dashboard (dashboard/): its client of the Studio API against a real Studio server on 127.0.0.1, the launcher's
checks, and the pages rendered headless. Every run record below is SYNTHETIC TEST DATA; no model or network is used."""
from __future__ import annotations

import json
import socket
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "dashboard"))

from client import StudioClient, StudioError, studio_url  # noqa: E402

RUN_NAME = "20261018T100000000000Z-when-did-the-bridge-open"
RUN = {         # synthetic: shaped like Orchestrator.run output, values invented for the test
    "input": "When did the bridge open?", "mode": "research", "status": "COMPLETE", "outcome": "ANSWERED",
    "started": "2026-10-18T10:00:00Z", "finished": "2026-10-18T10:03:00Z", "stages": [], "error": "",
    "model": {"model": "qwen3:4b", "ollama_version": "0.40.6"}, "settings": {"model": "qwen3:4b"},
    "sources": [{"id": "S1", "title": "Bridge (synthetic)", "url": "https://example.org/bridge", "provider": "openalex",
                 "kind": "scholarly", "access": "FULL_TEXT", "published": "2020-05-01",
                 "retrieved_at": "2026-10-18T10:01:00Z", "text": "Intro. The bridge opened in 1932. Closing."}],
    "claims": [{"id": "C1", "text": "It is 900 metres long.", "label": "INSUFFICIENT", "sub_state": "",
                "cited_sources": ["S1"], "supporting_sources": [], "contradicting_sources": [],
                "independent_support": 0, "independent_contradiction": 0, "passages": []},
               {"id": "C2", "text": "The bridge opened in 1932.", "label": "SUPPORTED", "sub_state": "",
                "cited_sources": ["S1"], "supporting_sources": ["S1"], "contradicting_sources": [],
                "independent_support": 1, "independent_contradiction": 0,
                "passages": [{"key": "S1:p1:a", "source_id": "S1", "url": "https://example.org/bridge",
                              "stance": "SUPPORTS", "text": "The bridge opened in 1932.", "page": 1}]}],
    "evidence_pool": [{"key": "S1:p1:a", "source_id": "S1", "page": 1, "text": "The bridge opened in 1932."}],
    "summary": {"available": True, "direct_answer": {"text": "It opened in 1932.", "claims": [1]}, "summary": []},
    "disagreements": [], "gaps": [], "limitations": [],
}


@pytest.fixture
def studio(tmp_path):
    """The real Studio server (webrag.server.make_handler) on a free loopback port, with one saved run."""
    from webrag.config import ConfigStore
    from webrag.server import JobQueue, StatusProbe, make_handler
    from webrag.studio import RunStore
    folder = tmp_path / "runs" / RUN_NAME
    folder.mkdir(parents=True)
    (folder / "run.json").write_text(json.dumps(RUN), encoding="utf-8")
    probe = StatusProbe("qwen3:4b", "quote_stance", describe=lambda: {"ollama_version": "0.40.6", "model_digest": "sha256:x"},
                        providers=lambda: ["openalex"], models=lambda: ["qwen3:4b"])
    server = ThreadingHTTPServer(("127.0.0.1", 0), None)
    port = server.server_address[1]
    server.RequestHandlerClass = make_handler(JobQueue(lambda: None), port, runs=RunStore(tmp_path / "runs"),
                                              status=probe, config_store=ConfigStore(tmp_path / "config.json"))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()
    server.server_close()


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


# ------------------------------------------------------------------------------------------------ client
@pytest.mark.parametrize("url, ok", [("http://127.0.0.1:8765", True), ("http://localhost:9000/", True),
                                     ("https://127.0.0.1:8765", False), ("http://192.168.1.5:8765", False),
                                     ("http://example.org", False), ("http://127.0.0.1:8765/api", False)])
def test_studio_url_must_be_on_this_computer(url, ok):
    if ok:
        assert studio_url({"WEBRAG_STUDIO_URL": url}) == url.rstrip("/")
    else:
        with pytest.raises(ValueError):
            studio_url({"WEBRAG_STUDIO_URL": url})
    assert studio_url({}) == "http://127.0.0.1:8765"


def test_client_reads_the_real_studio_api(studio):
    client = StudioClient(studio)
    status = client.status()
    assert status["model"] == "qwen3:4b" and status["model_installed"] is True
    runs = client.runs()
    assert [r["name"] for r in runs] == [RUN_NAME] and runs[0]["supported"] == 1 and runs[0]["claims"] == 2
    data = client.run(RUN_NAME)
    assert data["input"] == RUN["input"] and "text" not in data["sources"][0]          # source texts stay server-side
    context = client.context(RUN_NAME, "S1", "S1:p1:a")
    assert context["located"] and context["match"] == "The bridge opened in 1932."
    assert b"When did the bridge open?" in client.export(RUN_NAME, "md")
    assert json.loads(client.export(RUN_NAME, "json"))["input"] == RUN["input"]
    assert client.jobs() == []


def test_client_reports_studio_errors(studio):
    client = StudioClient(studio)
    with pytest.raises(StudioError) as error:
        client.run("20990101T000000000000Z-no-such-run")
    assert error.value.status == 404 and "unknown run" in str(error.value)
    with pytest.raises(StudioError) as error:
        client.submit("", "research")
    assert error.value.status == 400 and "enter a question" in str(error.value)
    with pytest.raises(StudioError) as error:
        StudioClient(f"http://127.0.0.1:{free_port()}", timeout=3).status()
    assert error.value.status == 0 and "not reachable" in str(error.value)


# ------------------------------------------------------------------------------------------------ launcher
def test_launcher_checks(monkeypatch, capsys):
    from dashboard import launch
    monkeypatch.setattr(launch, "missing_packages", lambda: ["streamlit"])
    assert launch.main(["--check"]) == 1 and "pip install -r requirements.txt" in capsys.readouterr().out
    monkeypatch.setattr(launch, "missing_packages", lambda: [])
    monkeypatch.setenv("WEBRAG_LLM_PROVIDER", "openai")
    monkeypatch.setenv("WEBRAG_LLM_BASE_URL", "http://api.example.org/v1")
    assert launch.main(["--check"]) == 1 and "must use https://" in capsys.readouterr().out


def test_launcher_refuses_a_port_used_by_another_program(monkeypatch, capsys):
    from dashboard import launch
    from webrag.model_service import ModelConfig
    monkeypatch.setattr(launch, "missing_packages", lambda: [])
    for name in ("WEBRAG_LLM_PROVIDER", "WEBRAG_LLM_BASE_URL"):
        monkeypatch.delenv(name, raising=False)

    class Offline:
        def describe(self):
            return {"ollama_version": None, "model_digest": None}

    monkeypatch.setattr(ModelConfig, "client", lambda self, model=None: Offline())
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        port = busy.getsockname()[1]
        assert launch.main(["--check", "--studio-port", str(port), "--dashboard-port", str(free_port())]) == 1
    out = capsys.readouterr().out
    assert "Ollama is not reachable" in out and f"Port {port} is used by another program" in out
    assert launch.main(["--check", "--studio-port", str(free_port()), "--dashboard-port", str(free_port())]) == 0
    assert launch.port_free(free_port()) is True


# ------------------------------------------------------------------------------------------------ pages
streamlit = pytest.importorskip("streamlit")


def test_finding_numbers_match_the_studio():
    import views
    claims = RUN["claims"]
    numbers = views.finding_numbers(claims)
    assert numbers == {1: 1, 0: 2}                          # supported first, as in the Studio and the report
    assert views.finding_refs([1, 0, 7, "x"], numbers) == "F1, F2"


def run_app(base: str):
    from streamlit.testing.v1 import AppTest
    import os
    os.environ["WEBRAG_STUDIO_URL"] = base
    try:
        return AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=60).run()
    finally:
        del os.environ["WEBRAG_STUDIO_URL"]


def test_dashboard_without_the_studio_says_so():
    app = run_app(f"http://127.0.0.1:{free_port()}")
    assert not app.exception
    assert any("backend is not running" in m.value for m in app.markdown)
    assert any("Studio offline" in m.value for m in app.sidebar.markdown)


def test_dashboard_command_center_with_the_studio(studio):
    app = run_app(studio)
    assert not app.exception
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["Studio backend"] == "Online" and metrics["Model"] == "qwen3:4b"
    assert metrics["Saved runs"] == "1" and metrics["Statements supported"] == "1 / 2"
