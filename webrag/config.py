"""Search configuration chosen in the studio: which providers are on, and the non-secret settings some need.

Stored as JSON in ``outputs/webrag/studio-config.json`` (``outputs/`` is not versioned). Secrets are never stored here:
API keys stay in environment variables (BRAVE_API_KEY, OPENALEX_API_KEY), which the studio reports only as set or not
set. Environment variables also take precedence (SEARXNG_URL, WEBRAG_PROVIDERS), so a launch configuration keeps
working exactly as before.

Stored fields:
* ``providers``: {provider id: {"enabled": bool}}; providers not listed keep their default (``search.PROVIDERS``);
  a provider the source policy excludes (Wikipedia) cannot be switched on;
* ``searxng``: {"url", "engines", "categories"} for a self-hosted SearXNG;
* ``feeds``: [{"url", "name", "category"}] with category news, official, podcast or web;
* ``contact_email``: optional; sent to Crossref as its documented ``mailto`` parameter, nowhere else.
"""
from __future__ import annotations

import copy
import json
import os
import re
import tempfile
import threading
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List

from .search import PROVIDERS

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "outputs" / "webrag" / "studio-config.json"
FEED_CATEGORIES = ("news", "official", "podcast", "web")
MAX_FEEDS = 30
SECRET_WORDS = re.compile(r"(key|token|secret|password|auth)", re.I)
EMAIL = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,190}\.[A-Za-z]{2,}$")


class ConfigError(ValueError):
    pass


def _http_url(value: Any, field: str) -> str:
    url = str(value or "").strip()
    if not url:
        return ""
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ConfigError(f"{field} must be an http(s) address without a user name or password")
    if SECRET_WORDS.search(parsed.query or ""):
        raise ConfigError(f"{field} must not carry a key or token; keys belong in environment variables")
    return url.rstrip("/") if field == "SearXNG address" else url


def validate(raw: Any, stored: bool = False) -> Dict[str, Any]:
    """A clean configuration from user input; anything unknown or secret-looking is refused, not silently kept.
    ``stored``: a file saved before the source policy may switch Wikipedia on; that switch is dropped rather than
    making the whole file unusable (the provider stays excluded either way)."""
    if not isinstance(raw, dict):
        raise ConfigError("the configuration must be an object")
    unknown = set(raw) - {"providers", "searxng", "feeds", "contact_email"}
    if unknown:
        raise ConfigError(f"unknown settings: {', '.join(sorted(unknown))}")
    clean: Dict[str, Any] = {"providers": {}, "searxng": {}, "feeds": [], "contact_email": ""}
    for pid, value in (raw.get("providers") or {}).items():
        if pid not in PROVIDERS:
            raise ConfigError(f"unknown provider {pid!r}")
        if not isinstance(value, dict) or set(value) - {"enabled"} or not isinstance(value.get("enabled"), bool):
            raise ConfigError(f"provider {pid!r}: only {{\"enabled\": true|false}} can be set here")
        if value["enabled"] and not PROVIDERS[pid].evidence:
            if stored:
                continue
            raise ConfigError(f"{PROVIDERS[pid].name} is not accepted as an evidence source and cannot be switched on")
        clean["providers"][pid] = {"enabled": value["enabled"]}
    searxng = raw.get("searxng") or {}
    if not isinstance(searxng, dict) or set(searxng) - {"url", "engines", "categories"}:
        raise ConfigError("searxng accepts url, engines and categories")
    clean["searxng"] = {"url": _http_url(searxng.get("url"), "SearXNG address"),
                        "engines": re.sub(r"[^A-Za-z0-9_ ,\-]", "", str(searxng.get("engines") or ""))[:200],
                        "categories": re.sub(r"[^A-Za-z0-9_ ,\-]", "", str(searxng.get("categories") or ""))[:200]}
    feeds = raw.get("feeds") or []
    if not isinstance(feeds, list) or len(feeds) > MAX_FEEDS:
        raise ConfigError(f"feeds must be a list of at most {MAX_FEEDS}")
    seen = set()
    for feed in feeds:
        if not isinstance(feed, dict) or set(feed) - {"url", "name", "category"}:
            raise ConfigError("each feed accepts url, name and category")
        url = _http_url(feed.get("url"), "Feed address")
        if not url:
            raise ConfigError("a feed needs an address")
        category = str(feed.get("category") or "news")
        if category not in FEED_CATEGORIES:
            raise ConfigError(f"feed category must be one of {', '.join(FEED_CATEGORIES)}")
        if url in seen:
            continue
        seen.add(url)
        clean["feeds"].append({"url": url, "name": re.sub(r"\s+", " ", str(feed.get("name") or "")).strip()[:80],
                               "category": category})
    email = str(raw.get("contact_email") or "").strip()
    if email and not EMAIL.match(email):
        raise ConfigError("the contact email is not a valid address")
    clean["contact_email"] = email
    return clean


class ConfigStore:
    """The configuration file, read on demand and written atomically. A missing or unreadable file means defaults."""

    def __init__(self, path: Path = CONFIG_PATH):
        self.path = Path(path)
        self._lock = threading.Lock()
        self.error = ""

    def load(self) -> Dict[str, Any]:
        with self._lock:
            if not self.path.is_file():
                self.error = ""
                return validate({})
            try:
                data = validate(json.loads(self.path.read_text(encoding="utf-8")), stored=True)
                self.error = ""
                return data
            except (OSError, ValueError) as exc:
                self.error = f"The search configuration file could not be used ({exc}); defaults apply."
                return validate({})

    def save(self, raw: Any) -> Dict[str, Any]:
        clean = validate(raw)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            handle, temp = tempfile.mkstemp(prefix=".studio-config-", suffix=".tmp", dir=self.path.parent)
            try:
                with os.fdopen(handle, "w", encoding="utf-8") as out:
                    json.dump(clean, out, indent=2)
                os.replace(temp, self.path)
            except BaseException:
                if os.path.exists(temp):
                    os.remove(temp)
                raise
            self.error = ""
        return copy.deepcopy(clean)

    def update(self, patch: Any) -> Dict[str, Any]:
        """Merge a partial update: provider switches are merged by id; other sections are replaced when given."""
        if not isinstance(patch, dict):
            raise ConfigError("the update must be an object")
        unknown = set(patch) - {"providers", "searxng", "feeds", "contact_email"}
        if unknown:                                     # e.g. an API key: refused, never silently dropped or stored
            raise ConfigError(f"unknown settings: {', '.join(sorted(unknown))}; keys belong in environment variables")
        current = self.load()
        merged = copy.deepcopy(current)
        if "providers" in patch:
            if not isinstance(patch["providers"], dict):
                raise ConfigError("providers must be an object")
            merged["providers"].update(patch["providers"])
        for key in ("searxng", "feeds", "contact_email"):
            if key in patch:
                merged[key] = patch[key]
        return self.save(merged)


def public_view(config: Dict[str, Any]) -> Dict[str, Any]:
    """What the browser may see: the stored settings (none of them secret) plus whether the environment overrides."""
    view = copy.deepcopy(config)
    view["environment"] = {"SEARXNG_URL": bool(os.environ.get("SEARXNG_URL", "").strip()),
                           "WEBRAG_PROVIDERS": os.environ.get("WEBRAG_PROVIDERS", "").strip() or "",
                           "BRAVE_API_KEY": bool(os.environ.get("BRAVE_API_KEY", "").strip()),
                           "OPENALEX_API_KEY": bool(os.environ.get("OPENALEX_API_KEY", "").strip())}
    return view


def feed_list(config: Dict[str, Any]) -> List[Dict[str, str]]:
    return list(config.get("feeds") or [])
