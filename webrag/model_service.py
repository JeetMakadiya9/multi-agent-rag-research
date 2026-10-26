"""Which language model the Research Studio uses, from local configuration.

The default is the local Ollama (``docqa.llm.OllamaChat``), exactly as before. Optionally the Studio can use an Ollama
server elsewhere or an OpenAI-compatible API (``docqa.remote_llm``), set in environment variables or in the ignored
``.env`` file at the repository root (copy ``.env.example``):

    WEBRAG_LLM_PROVIDER         local (default) | ollama | openai
    WEBRAG_LLM_BASE_URL         service address for ollama/openai (https://...; for openai including the /v1 prefix;
                                plain http only for this computer)
    WEBRAG_LLM_API_KEY          service key, when the service needs one (stays in the Studio server process)
    WEBRAG_LLM_MODEL            model name (default qwen3:4b; the server's --model option overrides it)
    WEBRAG_LLM_VERIFIER_MODEL   optional separate model for the verifier (same service)
    WEBRAG_LLM_JSON_MODE        openai only: schema (default) | object | none
    WEBRAG_LLM_TIMEOUT          seconds per model call (default 180)

A configured service that fails is reported; no other model or provider is ever substituted. Nothing here prints or
returns the API key.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"
RETRIEVAL_MODELS = ("sentence-transformers/all-MiniLM-L6-v2", "cross-encoder/ms-marco-MiniLM-L-6-v2")
SETTING_NAMES = ("WEBRAG_LLM_PROVIDER", "WEBRAG_LLM_BASE_URL", "WEBRAG_LLM_API_KEY", "WEBRAG_LLM_MODEL",
                 "WEBRAG_LLM_VERIFIER_MODEL", "WEBRAG_LLM_JSON_MODE", "WEBRAG_LLM_TIMEOUT", "WEBRAG_VERIFIER",
                 "WEBRAG_INJECTION_GUARD", "WEBRAG_QUOTE_ATTESTATION")
GUARD_MODES = ("off", "flag", "exclude")
# The Research Studio's defaults (the command-line tool and the evaluation harness keep Settings' own defaults):
# * verifier: Arm C agentic verification, Studio-adapted variant (docqa/agentic.py);
# * injection guard: exclude (withhold instruction-like retrieved passages from every model call; webrag/guarded.py);
# * quote attestation: on whenever the verifier quotes (record only; never changes a stance or verdict).
STUDIO_VERIFIER = "arm_c_studio"
STUDIO_INJECTION_GUARD = "exclude"
VERIFIER_DESCRIPTIONS = {
    "arm_c_studio": "Arm C agentic verification, Studio-adapted variant (default): one verifier pass over the retrieved "
                    "passages; conflicting evidence goes to an adjudicator; when no passage took a side, a planner "
                    "searches this run's evidence (at most 2 searches) before adjudication. Differs from the historical "
                    "Arm C in three fail-closed rules: no adjudicator decision without a passage for or against, no "
                    "adjudication citing a passage on the other side, and a passage given an invalid stance fails "
                    "the claim",
    "arm_c": "Arm C agentic verification exactly as studied (historical method), including adjudicator decisions made "
             "without any passage for or against",
    "arm_c0": "Historical arm C0: one verifier pass and an adjudicator for unresolved evidence, without planner or search",
    "quote_stance": "Legacy per-passage quoted stances (the previous default); conflicting evidence stays undecided",
    "quote_stance_ranked": "quote_stance, with conflicting evidence decided by the first-ranked passage with a stance "
                           "(not Arm C)",
    "quote_stance_adjudicated": "quote_stance, with conflicting evidence decided by an adjudicator asked in both "
                                "passage orders (not Arm C; not recommended)",
    "frozen_b": "The earlier Experiment B verifier",
}
VERIFIER_KINDS = {"arm_c_studio": "Studio-adapted Arm C (default)", "arm_c": "historical Arm C",
                  "arm_c0": "historical arm C0", "quote_stance": "legacy fallback",
                  "quote_stance_ranked": "alternative", "quote_stance_adjudicated": "alternative (not recommended)",
                  "frozen_b": "legacy"}
TRUE_WORDS, FALSE_WORDS = ("1", "true", "yes", "on"), ("0", "false", "no", "off")
SECRET_NAMES = ("WEBRAG_LLM_API_KEY", "BRAVE_API_KEY", "OPENALEX_API_KEY")
_LINE = re.compile(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*")


def load_env_file(path: Path = ENV_FILE, env: Optional[MutableMapping[str, str]] = None) -> List[str]:
    """Settings from a KEY=VALUE file into ``env`` (default os.environ). Values already set in the environment win.
    Returns the names read (never values). A missing file is not an error."""
    env = os.environ if env is None else env
    try:
        lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    except FileNotFoundError:
        return []
    names = []
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = _LINE.fullmatch(line)
        if not match:
            continue
        name, value = match.group(1), match.group(2)
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        else:                                    # unquoted: "KEY=value  # comment" keeps only the value
            value = re.sub(r"\s+#.*$", "", value)
        env.setdefault(name, value)
        names.append(name)
    return names


def allow_model_downloads(env: Optional[MutableMapping[str, str]] = None) -> bool:
    """Whether the retrieval models (rag.py) may be downloaded from the Hugging Face Hub. docqa.index keeps the Hub
    offline unless told otherwise, which fails on a first installation. WEBRAG_MODEL_DOWNLOADS: 1 or 0 decide; the
    default (auto) allows it only while a model is not in the local cache. Call before docqa.index is imported."""
    env = os.environ if env is None else env
    choice = str(env.get("WEBRAG_MODEL_DOWNLOADS", "auto")).strip().lower()
    if choice in ("0", "1"):
        allow = choice == "1"
    else:
        cache = Path(env.get("HF_HUB_CACHE") or Path(env.get("HF_HOME") or Path.home() / ".cache" / "huggingface") / "hub")
        allow = not all((cache / ("models--" + name.replace("/", "--"))).is_dir() for name in RETRIEVAL_MODELS)
    if allow:
        env.setdefault("HF_HUB_OFFLINE", "0")
        env.setdefault("TRANSFORMERS_OFFLINE", "0")
    return allow


@dataclass(frozen=True)
class ModelConfig:
    provider: str = "local"
    base_url: str = ""
    api_key: str = ""
    model: str = "qwen3:4b"
    verifier_model: str = ""
    json_mode: str = "schema"
    timeout: int = 180

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None, model: Optional[str] = None) -> "ModelConfig":
        env = os.environ if env is None else env
        get = lambda name, default="": str(env.get(name, default) or default).strip()
        try:
            timeout = int(get("WEBRAG_LLM_TIMEOUT", "180"))
        except ValueError:
            timeout = -1
        return cls(provider=get("WEBRAG_LLM_PROVIDER", "local").lower(), base_url=get("WEBRAG_LLM_BASE_URL"),
                   api_key=get("WEBRAG_LLM_API_KEY"), model=model or get("WEBRAG_LLM_MODEL", "qwen3:4b"),
                   verifier_model=get("WEBRAG_LLM_VERIFIER_MODEL"), json_mode=get("WEBRAG_LLM_JSON_MODE", "schema"),
                   timeout=timeout)

    def __repr__(self) -> str:                                     # never show the key
        return (f"ModelConfig(provider={self.provider!r}, endpoint={self.endpoint!r}, model={self.model!r}, "
                f"verifier_model={self.verifier_model!r})")

    @property
    def remote(self) -> bool:
        return self.provider != "local"

    @property
    def endpoint(self) -> str:
        if not self.remote:
            return "http://127.0.0.1:11434"
        from docqa.remote_llm import endpoint_label
        try:
            return endpoint_label(self.base_url)
        except ValueError:
            return ""

    def problems(self) -> List[str]:
        """Configuration errors, by setting name (never values of secrets)."""
        from docqa.remote_llm import JSON_MODES, PROVIDERS, check_base_url
        out = []
        if self.provider not in ("local", *PROVIDERS):
            out.append(f"WEBRAG_LLM_PROVIDER must be local, {' or '.join(PROVIDERS)}")
        if self.provider in PROVIDERS:
            try:
                check_base_url(self.base_url)
            except ValueError as exc:
                out.append(f"WEBRAG_LLM_BASE_URL: {exc}")
        if self.provider == "openai" and self.json_mode not in JSON_MODES:
            out.append(f"WEBRAG_LLM_JSON_MODE must be one of {', '.join(JSON_MODES)}")
        if not 10 <= self.timeout <= 900:
            out.append("WEBRAG_LLM_TIMEOUT must be a number of seconds between 10 and 900")
        if self.verifier_model and not self.remote:
            out.append("WEBRAG_LLM_VERIFIER_MODEL needs a configured model service (WEBRAG_LLM_PROVIDER ollama or openai)")
        return out

    # ---------------------------------------------------------------------------------------------- clients
    def client(self, model: Optional[str] = None):
        if not self.remote:
            from docqa.llm import OllamaChat
            return OllamaChat(model=model or self.model)
        from docqa.remote_llm import build_client
        return build_client(self.provider, model or self.model, self.base_url, self.api_key, self.json_mode,
                            timeout=self.timeout)

    def clients(self) -> Tuple[Any, Any]:
        """(model client, verifier client); the same object unless a separate verifier model is configured."""
        client = self.client()
        if self.verifier_model and self.verifier_model != self.model:
            return client, self.client(self.verifier_model)
        return client, client

    # ---------------------------------------------------------------------------------------------- studio parts
    def describe(self) -> Dict[str, Any]:
        """The configured service for the Studio's status probe (remote services only)."""
        client, verifier_client = self.clients()
        info = client.describe()
        if verifier_client is not client:
            info["verifier_model"] = verifier_client.describe()
        return info

    def list_models(self) -> List[str]:
        try:
            return self.clients()[0].list_models()
        except Exception:                       # an unreachable service is reported by describe(), not here
            return []

    def orchestrator_factory(self, verifier: str, injection_guard: str = "off",
                             quote_attestation: bool = False) -> Callable[[], Any]:
        """The Studio's pipeline with this service's clients (see webrag.server.studio_orchestrator)."""
        def build():
            from docqa.verify import VERIFIERS
            from .orchestrator import Orchestrator, Settings
            from .search import is_evidence_provider
            client, verifier_client = self.clients()
            orchestrator = Orchestrator(Settings(model=self.model, verifier=verifier, injection_guard=injection_guard,
                                                 quote_attestation=quote_attestation),
                                        client=client, verifier=VERIFIERS[verifier](verifier_client))
            orchestrator.providers = [p for p in orchestrator.providers if is_evidence_provider(p.name)]
            return orchestrator
        return build

    def assistant_factory(self) -> Callable[..., Any]:
        """The document assistant (docqa.cli.build_assistant) with this service's clients."""
        def build(folder: str, log: Callable[[str], None] = print):
            from docqa.answer import AnswerGenerator
            from docqa.index import EvidenceIndex
            from docqa.ingest import load_documents
            from docqa.pipeline import Assistant
            from docqa.verify import VERIFIERS
            report = load_documents(folder)
            for name, reason in report.skipped:
                log(f"[skipped] {name}: {reason}")
            if not report.pages:
                raise SystemExit(f"No readable TXT or PDF documents in {folder}.")
            client, verifier_client = self.clients()
            return Assistant(EvidenceIndex(report.pages), AnswerGenerator(client),
                             VERIFIERS["frozen_b"](verifier_client), k=4), report
        return build


def studio_options(verifier: Optional[str] = None, injection_guard: Optional[str] = None,
                   quote_attestation: Optional[bool] = None,
                   env: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """The Studio's verification options and where each came from: the command line wins, then WEBRAG_VERIFIER,
    WEBRAG_INJECTION_GUARD and WEBRAG_QUOTE_ATTESTATION (environment or .env), then the defaults (``STUDIO_VERIFIER``,
    ``STUDIO_INJECTION_GUARD``, and quote attestation on when the verifier quotes). An invalid value, or an explicit
    request for attestation with a verifier that does not quote, raises ValueError naming every problem; it is never
    replaced by a default."""
    from docqa.verify import VERIFIERS
    env = os.environ if env is None else env
    problems: List[str] = []
    sources: Dict[str, str] = {}

    def pick(key: str, cli: Any, name: str, default: Any) -> Any:
        if cli is not None:
            sources[key] = "command line"
            return cli
        raw = str(env.get(name, "")).strip()
        if raw:
            sources[key] = f"environment ({name})"
            return raw
        sources[key] = "default"
        return default

    chosen_verifier = pick("verifier", verifier, "WEBRAG_VERIFIER", STUDIO_VERIFIER)
    if chosen_verifier not in VERIFIERS:
        problems.append(f"verifier {chosen_verifier!r} is unknown; choose one of {', '.join(sorted(VERIFIERS))}")
    guard = pick("injection_guard", injection_guard, "WEBRAG_INJECTION_GUARD", STUDIO_INJECTION_GUARD)
    guard = guard.lower() if isinstance(guard, str) else guard
    if guard not in GUARD_MODES:
        problems.append(f"injection guard {guard!r} must be one of {', '.join(GUARD_MODES)}")
    from docqa.attestation import supports_attestation
    quotes = chosen_verifier in VERIFIERS and supports_attestation(VERIFIERS[chosen_verifier])
    attest = pick("quote_attestation", quote_attestation, "WEBRAG_QUOTE_ATTESTATION", quotes)
    if isinstance(attest, str):
        word = attest.lower()
        if word not in TRUE_WORDS + FALSE_WORDS:
            problems.append(f"WEBRAG_QUOTE_ATTESTATION {attest!r} must be one of {', '.join(TRUE_WORDS + FALSE_WORDS)}")
        attest = word in TRUE_WORDS
    if sources["quote_attestation"] == "default" and not quotes:
        sources["quote_attestation"] = f"default (off: {chosen_verifier} does not quote)"
    if attest and not problems and not quotes:
        problems.append(f"quote attestation needs a quote-based verifier; {chosen_verifier!r} does not quote")
    if problems:
        raise ValueError("; ".join(problems))
    return {"verifier": chosen_verifier, "injection_guard": guard, "quote_attestation": bool(attest),
            "sources": sources}


def settings_overview(env: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """Which model settings are set (names only; secrets as set / not set), for diagnostics."""
    env = os.environ if env is None else env
    return {"env_file": ENV_FILE.name, "env_file_exists": ENV_FILE.is_file(),
            "set": [name for name in SETTING_NAMES if str(env.get(name, "")).strip()],
            "secrets": {name: bool(str(env.get(name, "")).strip()) for name in SECRET_NAMES}}
