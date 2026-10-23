"""The dashboard's client for the Research Studio's own HTTP API (``webrag/server.py``).

Every number and record the dashboard shows comes from these calls; nothing is computed from a model here. The Studio
answers only requests addressed to 127.0.0.1 or localhost, so the address must be on this computer.
"""
from __future__ import annotations

import ipaddress
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

DEFAULT_URL = "http://127.0.0.1:8765"


class StudioError(Exception):
    """A Studio request that failed. ``status`` is the HTTP status, or 0 when the Studio could not be reached."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def studio_url(env: Optional[Dict[str, str]] = None) -> str:
    """WEBRAG_STUDIO_URL, else the default. Only a loopback http address is accepted."""
    env = os.environ if env is None else env
    url = str(env.get("WEBRAG_STUDIO_URL") or DEFAULT_URL).strip().rstrip("/")
    parts = urllib.parse.urlsplit(url)
    host = parts.hostname or ""
    try:
        loopback = host == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = False
    if parts.scheme != "http" or not loopback or parts.path not in ("", "/"):
        raise ValueError("WEBRAG_STUDIO_URL must be http://127.0.0.1:<port> (the Studio runs on this computer only)")
    return url


class StudioClient:
    def __init__(self, base: Optional[str] = None, timeout: float = 15.0):
        self.base = studio_url() if base is None else base.rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, body: Optional[dict] = None, raw: bool = False,
                 timeout: Optional[float] = None) -> Any:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {"Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(self.base + path, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=timeout or self.timeout) as response:
                payload = response.read()
        except urllib.error.HTTPError as exc:
            try:
                message = json.loads(exc.read().decode("utf-8")).get("error") or f"HTTP {exc.code}"
            except (ValueError, AttributeError):
                message = f"HTTP {exc.code}"
            raise StudioError(exc.code, str(message)) from None
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            raise StudioError(0, f"The Research Studio is not reachable at {self.base} ({reason}).") from None
        if raw:
            return payload
        try:
            return json.loads(payload.decode("utf-8"))
        except ValueError:
            raise StudioError(502, "The Studio sent a response that is not JSON.") from None

    # ---------------------------------------------------------------------------------------------- reads
    def status(self) -> Dict[str, Any]:
        return self._request("GET", "/api/status")

    def providers(self) -> Dict[str, Any]:
        return self._request("GET", "/api/providers", timeout=30)

    def integrations(self) -> Dict[str, Any]:
        return self._request("GET", "/api/integrations", timeout=60)

    def jobs(self) -> List[Dict[str, Any]]:
        return self._request("GET", "/api/jobs").get("jobs", [])

    def job(self, job_id: str) -> Dict[str, Any]:
        return self._request("GET", "/api/run/" + urllib.parse.quote(job_id, safe=""))

    def runs(self) -> List[Dict[str, Any]]:
        return self._request("GET", "/api/runs", timeout=30).get("runs", [])

    def run(self, name: str) -> Dict[str, Any]:
        return self._request("GET", "/api/runs/" + urllib.parse.quote(name, safe=""), timeout=30)["data"]

    def context(self, name: str, source_id: str, passage_key: str) -> Dict[str, Any]:
        query = urllib.parse.urlencode({"source": source_id, "passage": passage_key})
        return self._request("GET", f"/api/runs/{urllib.parse.quote(name, safe='')}/context?{query}")

    def export(self, name: str, kind: str) -> bytes:
        return self._request("GET", f"/api/runs/{urllib.parse.quote(name, safe='')}/export?format={kind}", raw=True)

    # ---------------------------------------------------------------------------------------------- actions
    def submit(self, text: str, mode: str, options: Optional[Dict[str, Any]] = None) -> str:
        return self._request("POST", "/api/run", {"input": text, "mode": mode, **(options or {})})["id"]

    def cancel(self, job_id: str) -> Dict[str, Any]:
        return self._request("POST", f"/api/run/{urllib.parse.quote(job_id, safe='')}/cancel", {})
