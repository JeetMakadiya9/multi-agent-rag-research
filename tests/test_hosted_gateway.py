"""Hosted Studio gateway (webrag/hosted_studio.py): sign-in, cross-site refusal, size limits and forwarding to the
real Studio server. Needs Starlette and httpx (installed with the hosted requirements). Every record and response
below is SYNTHETIC TEST DATA; no model service or search provider is contacted."""
from __future__ import annotations

import json

import pytest

pytest.importorskip("starlette")
pytest.importorskip("httpx")
from starlette.testclient import TestClient  # noqa: E402

from tests.test_hosted_studio import KEY, TOKEN, StubClient  # noqa: E402
from webrag.hosted_studio import (COOKIE, HostedConfig, LoginLimiter, StudioBackend, create_gateway,  # noqa: E402
                                  session_value)

class FakeBackend:
    port = 9

    def __init__(self):
        self.calls = []

    def forward(self, method, path, headers, body):
        self.calls.append((method, path, headers, body))
        csp = "default-src 'self'; frame-ancestors 'none'"
        return 200, [("Content-Type", "application/json"), ("Content-Security-Policy", csp),
                     ("Set-Cookie", "x=1"), ("Server", "webrag")], b'{"ok": true}'


def gateway(tmp_path, token=TOKEN, backend=None, **kwargs):
    config = HostedConfig.from_env({"STUDIO_ACCESS_TOKEN": token, "WEBRAG_DATA_DIR": str(tmp_path)})
    return TestClient(create_gateway(config, backend or FakeBackend(), **kwargs), base_url="https://testserver")


def signed(client):
    client.cookies.set(COOKIE, session_value(TOKEN))
    return client


def test_locked_without_a_token(tmp_path):
    client = gateway(tmp_path, token="")
    page = client.get("/login")
    assert page.status_code == 503 and "STUDIO_ACCESS_TOKEN is not set" in page.text
    assert client.get("/api/status").status_code == 401
    client.cookies.set(COOKIE, session_value(""))
    assert client.get("/api/status").status_code == 401                 # no token: no cookie is ever valid
    assert client.get("/healthz").json() == {"ok": True, "locked": True, "configuration_problems": 0}


def test_sign_in_flow(tmp_path):
    backend = FakeBackend()
    client = gateway(tmp_path, backend=backend)
    first = client.get("/", follow_redirects=False)
    assert first.status_code == 303 and first.headers["location"] == "login"
    assert client.get("/static/app.mjs", follow_redirects=False).headers["location"] == "../login"
    assert client.get("/api/status").status_code == 401 and backend.calls == []
    wrong = client.post("/login", data={"token": "nope"}, follow_redirects=False)
    assert wrong.status_code == 401 and COOKIE not in wrong.cookies
    right = client.post("/login", data={"token": TOKEN}, follow_redirects=False)
    assert right.status_code == 303 and right.headers["location"] == "./"
    cookie = right.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=strict" in cookie and "secure" in cookie and TOKEN.lower() not in cookie
    assert client.get("/api/status").json() == {"ok": True}
    assert backend.calls[-1][:2] == ("GET", "/api/status")


def test_failed_sign_ins_are_limited(tmp_path):
    client = gateway(tmp_path, limiter=LoginLimiter(limit=2, window=60))
    for _ in range(2):
        assert client.post("/login", data={"token": "nope"}).status_code == 401
    assert client.post("/login", data={"token": TOKEN}).status_code == 429


def test_cross_site_requests_are_refused(tmp_path):
    backend = FakeBackend()
    client = signed(gateway(tmp_path, backend=backend))
    refused = client.post("/api/run", json={"input": "q"}, headers={"Sec-Fetch-Site": "cross-site"})
    assert refused.status_code == 403 and backend.calls == []
    assert client.post("/login", data={"token": TOKEN}, headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403


def test_forwarding(tmp_path):
    backend = FakeBackend()
    client = signed(gateway(tmp_path, backend=backend))
    response = client.post("/api/run?x=1", json={"input": "q"}, headers={"Sec-Fetch-Site": "same-origin",
                                                                          "Origin": "https://evil.example"})
    method, path, headers, body = backend.calls[-1]
    assert (method, path, json.loads(body)) == ("POST", "/api/run?x=1", {"input": "q"})
    assert headers["Origin"] == "http://127.0.0.1:9" and headers["content-type"] == "application/json"
    assert response.headers["content-security-policy"] == "default-src 'self'; frame-ancestors 'self'"
    assert "set-cookie" not in response.headers and response.headers.get("server") != "webrag"
    assert client.get("/../../etc/passwd").status_code in (401, 404)
    assert client.get("/static/../server.py").status_code == 404
    assert client.get("/admin").status_code == 404 and len(backend.calls) == 1


def test_large_requests_are_refused(tmp_path, monkeypatch):
    import webrag.hosted_studio as hs
    monkeypatch.setattr(hs, "MAX_BODY", 100)
    backend = FakeBackend()
    client = signed(gateway(tmp_path, backend=backend))
    assert client.post("/api/library/upload", content=b"x" * 200,
                       headers={"Content-Type": "application/octet-stream"}).status_code == 413
    assert backend.calls == []


# ------------------------------------------------------------------------------------------------ real studio server
def test_real_studio_server_behind_the_gateway(tmp_path, monkeypatch):
    config = HostedConfig.from_env({"WEBRAG_LLM_PROVIDER": "openai", "WEBRAG_LLM_BASE_URL": "https://api.example.org/v1",
                                    "WEBRAG_LLM_API_KEY": KEY, "WEBRAG_LLM_MODEL": "m-large",
                                    "STUDIO_ACCESS_TOKEN": TOKEN, "WEBRAG_DATA_DIR": str(tmp_path)})
    stub = StubClient("m-large")
    monkeypatch.setattr(HostedConfig, "clients", lambda self: (stub, stub))
    monkeypatch.setattr("webrag.orchestrator.RUNS_DIR", tmp_path / "runs")      # where finished runs are saved
    backend = StudioBackend(config).start()
    try:
        client = signed(TestClient(create_gateway(config, backend), base_url="https://testserver"))
        page = client.get("/")
        assert page.status_code == 200 and 'src="static/app.mjs"' in page.text
        assert "frame-ancestors 'self'" in page.headers["content-security-policy"]
        assert client.get("/static/api.mjs").status_code == 200
        status = client.get("/api/status").json()
        assert status["runtime"]["provider"] == "openai" and status["runtime"]["model_available"] is True
        assert status["model_installed"] is True and status["models"] == ["m-large"] and status["ollama_version"] is None
        assert KEY not in json.dumps(status) and TOKEN not in json.dumps(status)
        integrations = client.get("/api/integrations").json()
        ai = next(i for i in integrations["integrations"] if i["id"] == "ollama")
        assert ai["name"] == "Model service" and ai["status"]["state"] == "ready" and "api.example.org" in ai["data"]
        assert KEY not in json.dumps(integrations)
        assert client.get("/api/runs").json()["runs"] == []
        bad = client.post("/api/run", json={"input": ""}, headers={"Sec-Fetch-Site": "same-origin"})
        assert bad.status_code == 400
        unsent = client.post("/api/run", content=b'{"input": "Q?"}', headers={"Content-Type": "text/plain"})
        assert unsent.status_code == 415                           # the Studio's own checks still apply
    finally:
        backend.stop()


