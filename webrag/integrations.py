"""Integrations of the Research Studio: what each one does, and whether it is actually set up.

Status comes only from real configuration (the environment variables and launch options the existing code reads) or
from a real local check. Nothing is reported as connected because someone clicked a button, and secret values are
never returned: for a key, only whether it is set.

* AI runtime: Ollama, the only model runtime the pipelines use. ``docqa.llm.OllamaChat`` accepts local hosts only and
  refuses ``:cloud`` models, so cloud model APIs (OpenAI, Anthropic) are reported as not available in this version.
* Research sources: the providers of ``webrag.search``, configured by environment variables as before.
* Knowledge tools: the document folder read by ``docqa``, and an Obsidian vault for Markdown export. The vault is
  chosen by whoever starts the server (``--obsidian-vault`` or ``WEBRAG_OBSIDIAN_VAULT``), never by a browser request;
  the studio only writes new notes into one sub-folder and never reads, changes or overwrites existing notes.
* Export: Markdown (``webrag.report.render_markdown``, the renderer the command line already uses) and the JSON run
  record, for saved runs.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .report import render_markdown
from .studio import StudioError

VAULT_FOLDER = "Research Studio"
_UNSAFE_NAME = re.compile(r'[\\/:*?"<>|#^\[\]\x00-\x1f]+')

# webrag/search.py: name, purpose, how it is enabled. Keyless providers are part of the default set.
SEARCH_PROVIDERS = [
    {"id": "wikipedia", "name": "Wikipedia", "purpose": "Encyclopedia articles through the MediaWiki API, read in full.",
     "kind": "Encyclopedia", "setup": "Not used by the studio: Wikipedia is not accepted as an evidence source. Pages on "
                                      "wikipedia.org found by other providers are left out unread.", "env": []},
    {"id": "openalex", "name": "OpenAlex", "purpose": "Scholarly works with abstracts and open-access full text where available.",
     "kind": "Scholarly works", "setup": "On by default. An optional free key (OPENALEX_API_KEY) gives its own daily request budget.",
     "env": ["OPENALEX_API_KEY"]},
    {"id": "arxiv", "name": "arXiv", "purpose": "Preprints with abstracts, and full text from open PDFs.",
     "kind": "Preprints", "setup": "On by default; no key needed.", "env": []},
    {"id": "searxng", "name": "SearXNG", "purpose": "General web search through a self-hosted SearXNG instance: news, institutional and government pages.",
     "kind": "Web pages", "setup": "Enter the address of your own SearXNG instance (JSON output enabled) under SearXNG settings, "
     "or set SEARXNG_URL, which takes precedence.", "env": ["SEARXNG_URL"]},
    {"id": "brave", "name": "Brave Search API", "purpose": "General web search through the Brave Search API.",
     "kind": "Web pages", "setup": "Set BRAVE_API_KEY and restart the server.", "env": ["BRAVE_API_KEY"]},
    {"id": "europepmc", "name": "Europe PMC", "purpose": "Biomedical and life-science articles, with open-access full text through its API.",
     "kind": "Scholarly works", "setup": "Opt-in: switch it on under Research sources (or add europepmc to WEBRAG_PROVIDERS).", "env": []},
    {"id": "crossref", "name": "Crossref", "purpose": "DOI metadata from publishers: titles, authors, dates, licences, and abstracts when deposited.",
     "kind": "Scholarly metadata", "setup": "Opt-in: switch it on under Research sources. An optional contact email is sent with requests.", "env": []},
    {"id": "feeds", "name": "RSS and Atom feeds", "purpose": "Feeds you list: news, official publications and podcasts with published transcripts.",
     "kind": "News, official, podcasts", "setup": "Add feed addresses under Research sources.", "env": []},
]
SECRET_ENV = {"OPENALEX_API_KEY", "BRAVE_API_KEY"}


def _env_set(name: str) -> bool:
    return bool(os.environ.get(name, "").strip())


def _status(state: str, label: str, detail: str = "") -> Dict[str, str]:
    """state: ready | active | not_configured | unavailable | error | not_supported"""
    return {"state": state, "label": label, "detail": detail}


# ================================================================================================= AI runtime
def model_service_integration(status: Dict[str, Any]) -> Dict[str, Any]:
    """The configured model service of a hosted studio (docqa.remote_llm) in place of the local Ollama."""
    runtime, model = status["runtime"], status.get("model") or ""
    where = f"{runtime.get('provider_label')} at {runtime.get('endpoint')}"
    if not runtime.get("service_reachable"):
        state = _status("unavailable", "Not reachable", f"{where} could not be used: {runtime.get('error') or 'no answer'}.")
    elif not runtime.get("model_available"):
        state = _status("error", f"{model} is not offered", f"{where} answered, but does not list the configured model.")
    else:
        state = _status("ready", "Ready", f"{where}, model {model}.")
    verifier = runtime.get("verifier_model") or {}
    return {
        "id": "ollama", "name": "Model service", "category": "ai",
        "purpose": "Runs the language model for planning, writing and verifying (set in the Studio's configuration).",
        "status": state, "testable": True, "checked_at": status.get("checked_at"),
        "facts": {"Service": where, "Model": model,
                  "Verifier model": verifier.get("model") or model,
                  "Models offered": ", ".join((status.get("models") or [])[:40]) or "unknown",
                  "Research verifier": status.get("verifier"), "Document assistant verifier": status.get("document_verifier"),
                  "Fallback": "None: if the model is not offered, a run stops with an explanation; no other model is substituted."},
        "capabilities": ["Research planning, synthesis and verification", "Document questions and claim checks"],
        "data": f"Prompts, questions and retrieved passages are sent to {runtime.get('endpoint')} for every model call. "
                "The API key stays on the server.",
        "setup": ["Set in the .env file at the repository root (copy .env.example): WEBRAG_LLM_PROVIDER, "
                  "WEBRAG_LLM_BASE_URL, WEBRAG_LLM_MODEL and, when the service needs one, WEBRAG_LLM_API_KEY; then "
                  "restart the Studio. Remove them to use the local Ollama again."],
        "limitations": ["The model is chosen in the configuration when the Studio starts; it cannot be switched from the interface.",
                        "Verdicts were evaluated with qwen3:4b on a local Ollama; another model's verdicts have not been evaluated."],
    }


def ollama_integration(status: Dict[str, Any]) -> Dict[str, Any]:
    if status.get("runtime"):
        return model_service_integration(status)
    version, installed = status.get("ollama_version"), bool(status.get("model_installed"))
    model = status.get("model") or ""
    if not version:
        state = _status("unavailable", "Not reachable", "No Ollama server answered at 127.0.0.1:11434.")
    elif not installed:
        state = _status("error", f"{model} is not installed", f"Ollama {version} is running, but the configured model is missing.")
    else:
        state = _status("ready", "Ready", f"Ollama {version}, model {model} installed.")
    return {
        "id": "ollama", "name": "Ollama", "category": "ai", "purpose": "Runs the language model on this computer for planning, writing and verifying.",
        "status": state, "testable": True, "checked_at": status.get("checked_at"),
        "facts": {"Default model": model, "Installed models": ", ".join(status.get("models") or []) or ("none found" if version else "unknown"),
                  "Research verifier": status.get("verifier"), "Document assistant verifier": status.get("document_verifier"),
                  "Fallback": "None: if the model is missing, a run stops with an explanation; no other model is substituted."},
        "capabilities": ["Research planning, synthesis and verification", "Document questions and claim checks"],
        "data": "Prompts and retrieved passages are sent only to the local Ollama server; nothing goes to a cloud model.",
        "setup": ["Install Ollama and start it (for example `ollama serve`).", f"Install the default model: `ollama pull {model or 'qwen3:4b'}`.",
                  "To use another installed model, restart the studio with `--model NAME`."],
        "limitations": ["The model is chosen when the server starts (`--model`); it cannot be switched per run from the interface.",
                        "Only a local Ollama address is accepted, and `:cloud` models are refused by the client."],
    }


def cloud_integration(status: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if status and status.get("runtime"):
        runtime = status["runtime"]
        return {
            "id": "cloud-models", "name": "Remote model services", "category": "ai",
            "purpose": "A model service other than this computer's Ollama.",
            "status": _status("ready" if runtime.get("service_reachable") and runtime.get("model_available") else "unavailable",
                              "In use", f"This server uses {runtime.get('provider_label')} at {runtime.get('endpoint')}."),
            "testable": False, "facts": {"Clients": "Ollama API (remote), OpenAI-compatible chat completions"},
            "capabilities": [], "data": f"Model calls go to {runtime.get('endpoint')}.", "setup": [],
            "limitations": ["Only services with an Ollama or OpenAI-compatible API are supported; there is no native "
                            "Anthropic or Gemini client."],
        }
    return {
        "id": "cloud-models", "name": "Cloud model APIs (OpenAI, Anthropic)", "category": "ai",
        "purpose": "Hosted models through their official APIs.",
        "status": _status("not_supported", "Not configured",
                          "This studio uses the local Ollama. An Ollama server elsewhere or an OpenAI-compatible API can be set instead in .env (see .env.example)."),
        "testable": False, "facts": {}, "capabilities": [],
        "data": "Nothing is sent to OpenAI or Anthropic.",
        "setup": [],
        "limitations": ["A ChatGPT or Claude subscription does not provide API access for other applications; the official APIs use separately issued keys and billing.",
                        "The verifier was evaluated with qwen3:4b on a local Ollama; other models have not been evaluated."],
    }


# ================================================================================================= research sources
def search_integrations(status: Dict[str, Any], runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    active = set(status.get("providers") or [])
    error = status.get("providers_error") or ""
    items = []
    for provider in SEARCH_PROVIDERS:
        last = next(({"run": r["name"], "started": r.get("started"), **r["providers"][provider["id"]]}
                     for r in runs if provider["id"] in (r.get("providers") or {})), None)
        configured_env = {name: _env_set(name) for name in provider["env"]}
        if provider["id"] in active:
            state = _status("active", "In use", "Searched in every research run.")
        elif error:
            state = _status("error", "Provider list is invalid", error)
        elif provider["env"] and not any(configured_env.values()) and provider["id"] in {"searxng", "brave"}:
            state = _status("not_configured", "Not set up", provider["setup"])
        else:
            state = _status("not_configured", "Not in use", provider["setup"])
        facts = {f"{name}{' (secret)' if name in SECRET_ENV else ''}": "set" if is_set else "not set"
                 for name, is_set in configured_env.items()}
        if provider["id"] == "searxng" and _env_set("SEARXNG_URL"):
            facts["SEARXNG_URL"] = os.environ["SEARXNG_URL"].strip()        # a local address, not a secret
        items.append({
            "id": provider["id"], "name": provider["name"], "category": "sources", "purpose": provider["purpose"],
            "status": state, "testable": False, "facts": facts, "last_recorded": last,
            "capabilities": [provider["kind"]],
            "data": "Search queries are sent to this service, and result pages are fetched from their own sites.",
            "setup": [provider["setup"]],
            "limitations": ["API keys are read from environment variables when the server starts; the studio reports only whether they are set "
                            "and never stores or shows them. Switches and addresses are changed under Research sources, unless "
                            "WEBRAG_PROVIDERS or SEARXNG_URL is set, which take precedence."],
        })
    return items


# ================================================================================================= knowledge tools
def library_integration(listing: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not listing or listing.get("available") is False:
        state = _status("unavailable", "Not configured", "This server was started without a document folder.")
    elif not listing.get("exists"):
        state = _status("not_configured", "No folder yet", "The folder is created when you add the first document.")
    else:
        readable = sum(1 for d in listing.get("documents", []) if d.get("readable"))
        state = _status("ready", "Ready", f"{len(listing.get('documents', []))} files, {readable} readable.")
    return {
        "id": "documents", "name": "Local document folder", "category": "knowledge",
        "purpose": "TXT and PDF files the document assistant answers from.", "status": state, "testable": False,
        "facts": {"Folder": (listing or {}).get("folder", "")},
        "capabilities": ["Add documents", "Ask questions", "Check claims against the documents"],
        "data": "Documents stay in this folder; questions about them use the local model only.",
        "setup": ["Add files in Library, or place them in the folder and reload.", "Start the server with `--docs PATH` to use another folder."],
        "limitations": ["Documents cannot be removed from the interface; delete the file in the folder instead.",
                        "Answers are held in server memory and are not saved."],
    }


class ObsidianVault:
    """An Obsidian vault chosen at launch. Exports are new Markdown notes in ``<vault>/Research Studio``; existing
    notes are never read, changed or overwritten."""

    def __init__(self, path: Optional[str]):
        self.path = Path(path).expanduser() if path and str(path).strip() else None

    def check(self) -> Dict[str, Any]:
        if self.path is None:
            return {"configured": False, "ok": False, "problem": "", "vault": "", "folder": VAULT_FOLDER}
        problem = ""
        if not self.path.is_dir():
            problem = "The configured vault folder does not exist."
        elif not (self.path / ".obsidian").is_dir():
            problem = "The configured folder is not an Obsidian vault (it has no .obsidian folder)."
        elif not os.access(self.path, os.W_OK):
            problem = "The studio cannot write to the configured vault folder."
        return {"configured": True, "ok": not problem, "problem": problem, "vault": self.path.name,
                "path": str(self.path), "folder": VAULT_FOLDER}

    def export(self, title: str, created: str, markdown: str) -> str:
        state = self.check()
        if not state["configured"]:
            raise StudioError(409, "No Obsidian vault is set up for this server.")
        if not state["ok"]:
            raise StudioError(409, state["problem"])
        folder = self.path / VAULT_FOLDER
        folder.mkdir(exist_ok=True)
        clean = re.sub(r"\s+", " ", _UNSAFE_NAME.sub(" ", title)).strip(" .")
        if len(clean) > 80:                                   # shorten at a word boundary
            clean = clean[:80].rsplit(" ", 1)[0].strip(" .")
        stem = f"{created[:10]} {clean}".strip() or "Research report"
        for attempt in range(1, 100):
            path = folder / (f"{stem}.md" if attempt == 1 else f"{stem} ({attempt}).md")
            if path.parent.resolve() != folder.resolve():
                raise StudioError(400, "The note name is not valid.")
            try:
                with open(path, "x", encoding="utf-8", newline="\n") as handle:     # "x": never overwrite a note
                    handle.write(markdown)
            except FileExistsError:
                continue
            return f"{VAULT_FOLDER}/{path.name}"
        raise StudioError(409, "Too many notes with this name already exist in the vault.")


def obsidian_integration(vault: ObsidianVault) -> Dict[str, Any]:
    state = vault.check()
    if not state["configured"]:
        status = _status("not_configured", "Not set up", "Start the server with --obsidian-vault PATH to enable exports.")
    elif not state["ok"]:
        status = _status("error", "Vault problem", state["problem"])
    else:
        status = _status("ready", "Ready", f"Reports are saved as new notes in {state['vault']}/{VAULT_FOLDER}.")
    return {
        "id": "obsidian", "name": "Obsidian vault", "category": "knowledge",
        "purpose": "Save research reports as Markdown notes in your vault.", "status": status, "testable": False,
        "facts": {"Vault": state.get("path", "") if state["configured"] else "", "Notes folder": f"{VAULT_FOLDER}/" if state["configured"] else ""},
        "capabilities": ["Export a saved report as a new note, with its findings, limitations and sources"],
        "data": "Writes Markdown files into the vault folder only. The studio never reads your notes.",
        "setup": ["Start the server with `--obsidian-vault \"D:\\path\\to\\vault\"` (or set WEBRAG_OBSIDIAN_VAULT).",
                  "The folder must be an Obsidian vault (it contains a .obsidian folder)."],
        "limitations": ["The vault is chosen when the server starts, not from the browser.",
                        "Existing notes are never overwritten; a second export gets a numbered name.",
                        "Reading notes into research is not implemented."],
    }


def export_integrations() -> List[Dict[str, Any]]:
    common = {"category": "export", "testable": False, "facts": {}, "setup": [], "status": _status("ready", "Available")}
    return [
        dict(common, id="markdown", name="Markdown file", purpose="Download a saved report as a Markdown document.",
             capabilities=["Findings, statements not established, disagreements, limitations, sources and stages"],
             data="The file is created by this server and downloaded by your browser.",
             limitations=["Available for saved runs only."]),
        dict(common, id="json", name="Run record (JSON)", purpose="Download the complete saved run record.",
             capabilities=["Everything the pipeline recorded, including extracted source text"],
             data="The file is the run.json saved on this computer.",
             limitations=["Available for saved runs only. The file can be large."]),
    ]


# ================================================================================================= export
def run_markdown(name: str, data: Dict[str, Any]) -> str:
    """The command line's Markdown report with a short property header (Obsidian reads it as note properties)."""
    try:
        body = render_markdown(data)
    except (KeyError, TypeError, AttributeError) as exc:
        raise StudioError(422, f"This run record cannot be rendered as Markdown ({type(exc).__name__}).") from None
    created = str(data.get("finished") or data.get("started") or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    header = ["---", f"title: {json.dumps(str(data.get('input', '')), ensure_ascii=False)}",
              f"created: {json.dumps(created)}", f"mode: {json.dumps('verification' if data.get('mode') == 'verify' else 'research')}",
              f"status: {json.dumps(str(data.get('status') or ''))}", f"outcome: {json.dumps(str(data.get('outcome') or ''))}",
              f"run: {json.dumps(name)}", "tags: [research-studio]", "---", ""]
    return "\n".join(header) + body + "\n"


def describe(status: Dict[str, Any], runs: List[Dict[str, Any]], listing: Optional[Dict[str, Any]], vault: ObsidianVault) -> Dict[str, Any]:
    items = [ollama_integration(status), cloud_integration(status), *search_integrations(status, runs),
             library_integration(listing), obsidian_integration(vault), *export_integrations()]
    return {"categories": [
        {"id": "ai", "title": "AI runtime", "description": "Where the language model runs."},
        {"id": "sources", "title": "Research sources", "description": "Services searched during research."},
        {"id": "knowledge", "title": "Knowledge tools", "description": "Your documents and notes."},
        {"id": "export", "title": "Export", "description": "Ways to take a saved report elsewhere."},
    ], "integrations": items}
