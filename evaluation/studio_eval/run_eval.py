"""Run the Research Studio's verify step on a frozen, independently labelled claim set (PROTOCOL.md).

The system under test is built exactly as ``webrag.server`` builds it:
* ``studio_options()`` resolves the default verifier (``arm_c_studio``), the injection screen and quote attestation;
* ``studio_orchestrator()`` builds the orchestrator;
* each claim's fixed evidence packet becomes the run's sources, indexed through the orchestrator's own (guarded) store
  factory;
* ``Orchestrator._verify_mode`` performs the Studio's claim check, restricted to that packet.

The harness refuses to run unless ``FREEZE.json`` records the claims file's SHA-256 and declares the annotation
independent of model output. Records stay private (they contain passage text): the default output folder is under
``outputs/``, which is not versioned.

Usage:
    python -m evaluation.studio_eval.run_eval --claims PATH --freeze PATH [--out DIR] [--verifier NAME]
    python -m evaluation.studio_eval.run_eval --claims PATH --check-overlap
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[2]
GOLD_LABELS = ("SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO")
MAX_PASSAGES = 8
MAX_CLAIM_CHARS = 400
USED_CLAIM_FILES = ("data/scifact/data/claims_train.jsonl", "data/scifact/data/claims_dev.jsonl",
                    "data/scifact/data/claims_test.jsonl")


class ProtocolError(ValueError):
    """The input or the freeze does not satisfy PROTOCOL.md; nothing is run."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_claims(path: Path) -> List[Dict[str, Any]]:
    """Read and validate the claims file (one JSON object per line; see PROTOCOL.md Section 2)."""
    claims, seen = [], set()
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ProtocolError(f"line {number}: not JSON ({exc.msg})") from None
        problems = []
        cid, claim, label, passages = item.get("claim_id"), item.get("claim"), item.get("label"), item.get("passages")
        if not isinstance(cid, str) or not cid.strip():
            problems.append("claim_id must be a non-empty string")
        elif cid in seen:
            problems.append(f"duplicate claim_id {cid!r}")
        if not isinstance(claim, str) or not claim.strip() or len(claim) > MAX_CLAIM_CHARS:
            problems.append(f"claim must be 1-{MAX_CLAIM_CHARS} characters")
        if label not in GOLD_LABELS:
            problems.append(f"label must be one of {', '.join(GOLD_LABELS)}")
        if not isinstance(passages, list) or not 1 <= len(passages) <= MAX_PASSAGES:
            problems.append(f"passages must be a list of 1-{MAX_PASSAGES} items")
        else:
            for p in passages:
                if not isinstance(p, dict) or not str(p.get("text") or "").strip() or not str(p.get("url") or "").strip():
                    problems.append("every passage needs non-empty text and url")
                    break
        if problems:
            raise ProtocolError(f"line {number}: " + "; ".join(problems))
        seen.add(cid)
        claims.append(item)
    if not claims:
        raise ProtocolError("the claims file has no claims")
    return claims


def check_freeze(claims_path: Path, freeze_path: Optional[Path]) -> Dict[str, Any]:
    """The labels must be frozen, unchanged since freezing, and independent of model output."""
    if freeze_path is None or not Path(freeze_path).is_file():
        raise ProtocolError("FREEZE.json is missing: freeze the adjudicated labels before any run (PROTOCOL.md §4)")
    freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    if freeze.get("claims_sha256") != sha256_file(claims_path):
        raise ProtocolError("the claims file does not match the SHA-256 recorded in FREEZE.json")
    if (freeze.get("annotation") or {}).get("independent_of_model_output") is not True:
        raise ProtocolError("FREEZE.json does not declare the annotation independent of model output")
    return freeze


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join("".join(ch if ch.isalnum() else " " for ch in text).split())


def check_overlap(claims: Iterable[Dict[str, Any]], used_files: Iterable[str] = USED_CLAIM_FILES) -> List[str]:
    """Claim IDs whose normalised text equals a claim in a set this project has already used."""
    used = set()
    for name in used_files:
        path = ROOT / name
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    used.add(_normalise(json.loads(line).get("claim", "")))
    return [c["claim_id"] for c in claims if _normalise(c["claim"]) in used]


def fingerprint(options: Dict[str, Any], verifier: Any) -> str:
    """SHA-256 of the resolved options and the verifier's own settings."""
    settings = getattr(verifier, "settings", None)
    payload = {"options": {k: options[k] for k in ("verifier", "injection_guard", "quote_attestation")},
               "verifier_class": type(verifier).__name__,
               "verifier_settings": getattr(settings, "__dict__", None) and dict(settings.__dict__)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def packet_sources(claim: Dict[str, Any]) -> List[Any]:
    """Each packet passage becomes one readable source of the run (its own origin, so none is merged away)."""
    from webrag.researcher import FULL_TEXT, Source
    out = []
    for i, p in enumerate(claim["passages"], 1):
        sid = f"S{i}"
        out.append(Source(sid, str(p.get("title") or p["url"]), str(p["url"]), "evaluation-packet", "web",
                          f"{claim['claim_id']}:{i}", FULL_TEXT, str(p["text"]), origin=sid))
    return out


def build_system(verifier: Optional[str] = None, model: Optional[str] = None):
    """The Studio's system exactly as ``webrag.server.main`` builds it for the local model."""
    from webrag.model_service import ModelConfig, studio_options
    from webrag.server import studio_orchestrator
    options = studio_options(verifier)
    config = ModelConfig.from_env(model=model)
    orch = studio_orchestrator(config.model, options["verifier"], options["injection_guard"],
                               options["quote_attestation"])
    return orch, options


def evaluate(claims: List[Dict[str, Any]], orch: Any, options: Dict[str, Any],
             progress: Callable[[str], None] = lambda message: None) -> List[Dict[str, Any]]:
    """Run the verify step for every claim; a failure is recorded, never dropped."""
    from webrag.orchestrator import Run
    expected = options["verifier"]
    records = []
    for n, claim in enumerate(claims, 1):
        started = time.perf_counter()
        record: Dict[str, Any] = {"claim_id": claim["claim_id"], "gold": claim["label"],
                                  "subgroups": list(claim.get("subgroups") or [])}
        try:
            orch._guarded_stores = []
            store = orch.store_factory(packet_sources(claim), orch.models)
            run = Run(claim["claim"], "verify", lambda event: None)
            orch._verify_mode(run, store, claim["claim"])
            orch._record_injection_screen(run.data)
            result = (run.data.get("claims") or [{}])[0]
            ran = result.get("verifier") or getattr(orch.verifier, "name", "")
            record.update(
                verdict=result.get("label"), sub_state=result.get("sub_state", ""), verifier=ran,
                verifier_class=type(orch.verifier).__name__, parse_ok=result.get("parse_ok", True),
                error=result.get("error", ""),
                stances=[p.get("stance") for p in result.get("passages") or []],
                searched_passages=sum(1 for p in result.get("passages") or [] if p.get("found_by_search")),
                injection_withheld=len((run.data.get("injection_screen") or {}).get("flagged") or []),
                attestation=[a.get("attestation", {}).get("status") for p in result.get("passages") or []
                             for a in p.get("quote_attestation") or []],
                passage_diagnostics=[p.get("verifier_diagnostics") for p in result.get("passages") or []],
                calls=(result.get("verifier_trace") or {}).get("usage", {}).get("calls"))
            if ran != expected:
                record.update(verdict=None, error=f"verifier {ran!r} ran instead of {expected!r}")
        except Exception as exc:                          # recorded as an operational failure, never dropped
            record.update(verdict=None, sub_state="HARNESS_ERROR", error=f"{type(exc).__name__}: {exc}")
        record["seconds"] = round(time.perf_counter() - started, 2)
        records.append(record)
        progress(f"{n}/{len(claims)} {claim['claim_id']}: {record.get('verdict')} {record.get('sub_state')}")
    return records


def _git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


# Code and settings files a verify run depends on; their hashes record the code state that actually ran.
RUN_FILES = ("rag.py", "requirements-lock.txt", "docqa/agentic.py", "docqa/agentic_candidates.py", "docqa/attestation.py",
             "docqa/index.py", "docqa/llm.py", "docqa/verify.py", "evaluation/experiments/agentic_verify/arms.py",
             "evaluation/experiments/agentic_verify/llm.py", "evaluation/experiments/agentic_verify/prompts.py",
             "evaluation/experiments/agentic_verify/FROZEN_CONFIG.json", "evaluation/studio_eval/alt_benchmarks.py",
             "evaluation/studio_eval/run_eval.py", "evaluation/studio_eval/score.py", "webrag/evidence.py",
             "webrag/guarded.py", "webrag/injection.py", "webrag/orchestrator.py",
             # imported on the same path: verdict constants, quote attestation, and how build_system builds the Studio
             "docqa/adjudicate.py", "docqa/ranked.py", "evidence_core/span_attestation.py", "webrag/model_service.py",
             "webrag/server.py")
OLLAMA_SETTINGS = ("OLLAMA_CONTEXT_LENGTH", "OLLAMA_FLASH_ATTENTION", "OLLAMA_HOST", "OLLAMA_KEEP_ALIVE",
                   "OLLAMA_KV_CACHE_TYPE", "OLLAMA_MAX_LOADED_MODELS", "OLLAMA_NUM_PARALLEL")
HF_SETTINGS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
# Installed distributions on the retrieval and ranking path (rag.py and sentence-transformers), as pinned in
# requirements-lock.txt. Their versions are read from the running interpreter's package metadata.
PACKAGES = ("faiss-cpu", "huggingface_hub", "langchain-text-splitters", "numpy", "rank-bm25", "safetensors",
            "scikit-learn", "scipy", "sentence-transformers", "tokenizers", "torch", "transformers")
MAX_DIRTY_PATHS = 200
MAX_TEXT = 300


def _home() -> Optional[Path]:
    """The home folder, or None when the environment does not define one."""
    try:
        return Path.home()
    except (RuntimeError, KeyError, OSError):
        return None


def _scrub(text: str) -> str:
    """The text with the project folder and the home folder replaced (no user name in records). On Windows the
    match ignores case. Not covered: 8.3 short names and paths reached through a link to elsewhere."""
    flags = re.IGNORECASE if os.name == "nt" else 0
    for path, label in ((ROOT, "<project>"), (_home(), "~")):
        if path is None or len(path.parts) < 2:              # never replace a bare root such as "/" or "C:\"
            continue
        # plain, forward-slash and repr-escaped forms (OS error messages quote paths with doubled backslashes)
        for form in sorted({str(path), path.as_posix(), str(path).replace("\\", "\\\\")}, key=len, reverse=True):
            text = re.sub(re.escape(form), lambda _: label, text, flags=flags)
    return text


def short_error(exc: BaseException, limit: int = 200) -> str:
    """A bounded one-line description of an error, for records and manifests (local paths are scrubbed)."""
    text = _scrub(f"{type(exc).__name__}: {exc}".replace("\n", " "))
    return text if len(text) <= limit else text[:limit - 3] + "..."


_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")


def safe_url(value: Any) -> Any:
    """An address for records: scheme, host, port and path only. User info becomes ``***``; a query or fragment is
    replaced by ``?[redacted]``. The path is kept (an address whose path holds a secret is not protected)."""
    if value is None or value == "":
        return value
    if not isinstance(value, str):
        return f"<{type(value).__name__}>"
    has_scheme = bool(_SCHEME.match(value))
    try:
        parsed = urllib.parse.urlsplit(value if has_scheme else "//" + value)
        host = parsed.hostname or ""
        port = parsed.port
    except ValueError:
        return "[unparseable address]"
    netloc = (f"[{host}]" if ":" in host else host) + (f":{port}" if port else "")
    if parsed.username is not None or parsed.password is not None:
        netloc = "***@" + netloc
    rebuilt = urllib.parse.urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))
    if not has_scheme:
        rebuilt = rebuilt[2:] if rebuilt.startswith("//") else rebuilt
    if parsed.query or parsed.fragment:
        rebuilt += "?[redacted]"
    return rebuilt[:MAX_TEXT]


def safe_path(value: str) -> str:
    """A local path for records: relative to the project, or with the home folder shown as ``~``. The path as given
    is tried before its resolved form (a virtual environment's interpreter can be a link to elsewhere)."""
    path = Path(value)
    for base, label in ((ROOT, "<project>"), (_home(), "~")):
        if base is None:
            continue
        for candidate, anchor in ((path, base), (path, base.resolve()), (path.resolve(), base.resolve())):
            try:
                return f"{label}/{candidate.relative_to(anchor).as_posix()}"
            except (ValueError, OSError, RuntimeError):
                continue
    return _scrub(path.as_posix())[:MAX_TEXT]


def _git_state(runner: Callable[..., Any] = subprocess.run) -> Dict[str, Any]:
    """The working tree's uncommitted changes, read without git's optional locks (no index refresh). Paths are
    relative to the repository and the list is capped at ``MAX_DIRTY_PATHS``."""
    try:
        status = runner(["git", "--no-optional-locks", "status", "--porcelain"], cwd=ROOT, capture_output=True,
                        text=True, encoding="utf-8", errors="replace", timeout=60, check=True).stdout
    except (OSError, subprocess.SubprocessError, ValueError) as exc:     # includes a timeout and a failed status
        return {"dirty": None, "error": type(exc).__name__}
    lines = [line for line in status.splitlines() if line.strip()]
    paths = [{"status": line[:2], "path": line[3:][:MAX_TEXT]} for line in lines[:MAX_DIRTY_PATHS]]
    return {"dirty": bool(lines), "changed_paths": len(lines), "paths": paths,
            "paths_truncated": len(lines) > MAX_DIRTY_PATHS}


def _package_versions(names=PACKAGES) -> Dict[str, Any]:
    """Installed versions from the interpreter's package metadata; a package without metadata is listed, not guessed."""
    from importlib import metadata
    versions: Dict[str, str] = {}
    unavailable: Dict[str, str] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            unavailable[name] = "no installed package metadata"
        except Exception as exc:                          # recorded, never raised: metadata must not stop a run
            unavailable[name] = short_error(exc)
    return {"versions": versions, "unavailable": unavailable}


def _configured_models(path: Optional[Path] = None) -> Dict[str, Any]:
    """The embedding and reranker names ``rag.py`` configures, read from its source (no import, no model loading).
    Names only: the downloaded revision lives in the Hugging Face cache, which is not read here."""
    import ast
    source = path or ROOT / "rag.py"
    try:
        tree = ast.parse(source.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError, RecursionError, MemoryError) as exc:
        return {"error": short_error(exc)}
    wanted = {"EMBEDDING_MODEL_NAME": "embedding_model", "RERANKER_MODEL_NAME": "reranker_model"}
    found: Dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in wanted:
                    found[wanted[target.id]] = node.value.value[:MAX_TEXT]
    if not found:
        return {"error": "the model names were not found in rag.py"}
    missing = sorted(name for name, key in wanted.items() if key not in found)
    return dict(found, source=source.name, **({"not_found": missing} if missing else {}))


def _model_options(orch: Any) -> Dict[str, Any]:
    """Which request options reach the model, as fixed by the code that sends them; nothing is asked of the server.
    Established only for the Studio's local client with an Arm C verifier (``StudioModel.complete``)."""
    from docqa.agentic import StudioModel
    from docqa.llm import OllamaChat
    client, verifier = getattr(orch, "client", None), getattr(orch, "verifier", None)
    if type(client) is not OllamaChat:                    # a subclass may send other options
        return {"client": type(client).__name__, "established": False,
                "reason": "the options sent are only recorded for the local Ollama client itself"}
    out: Dict[str, Any] = {"client": "OllamaChat", "established": True,
                           "sent_by_client": {"stream": True, "think": client.think,
                                              "num_predict_default": client.num_predict},
                           # docqa/llm.py: the request's "options" hold only temperature and num_predict
                           "not_sent": ["seed", "num_ctx", "num_gpu"],
                           "source": "docqa/llm.py OllamaChat.chat"}
    settings = getattr(verifier, "settings", None)
    if getattr(verifier, "model_class", None) is StudioModel and settings is not None:   # not a candidate wrapper
        out["sent_per_call"] = {"format": "json", "temperature": 0.0,
                                "num_predict_by_role": {role: getattr(settings, f"{role}_tokens", None)
                                                        for role in ("verifier", "planner", "adjudicator")}}
        out["source"] += "; docqa/agentic.py StudioModel.complete"
    else:
        out["sent_per_call"] = "set by this verifier's own calls; not recorded here"
    return out


def environment(orch: Any, git_state: Optional[Callable[[], Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Run-manifest metadata, collected without any request to the model server and without changing a setting.
    A listed file that exists but cannot be read is reported in ``file_errors`` instead of stopping the run."""
    client = getattr(orch, "client", None)
    hashes: Dict[str, str] = {}
    errors: Dict[str, str] = {}
    missing: List[str] = []
    for name in RUN_FILES:
        if not (ROOT / name).is_file():
            missing.append(name)
            continue
        try:
            hashes[name] = sha256_file(ROOT / name)
        except OSError as exc:
            errors[name] = short_error(exc)
    client_info = {"class": type(client).__name__,
                   **{key: getattr(client, key, None) for key in ("model", "base_url", "num_predict", "think",
                                                                   "timeout")}}
    client_info["base_url"] = safe_url(client_info["base_url"])
    return {
        "interpreter": {"executable": safe_path(sys.executable), "version": sys.version,
                        "implementation": platform.python_implementation(),
                        "virtual_environment": sys.prefix != sys.base_prefix},
        "platform": platform.platform(),
        "git": (git_state or _git_state)(),
        "file_sha256": hashes,
        "file_errors": errors,
        "file_missing": missing,
        "packages": _package_versions(),
        "configured_models": _configured_models(),
        "model_client": client_info,
        "model_options": _model_options(orch),
        "ollama_settings": {name: (safe_url(os.environ[name]) if name == "OLLAMA_HOST"
                                   else _scrub(os.environ[name])[:MAX_TEXT])
                            for name in OLLAMA_SETTINGS if name in os.environ},
        "hf_settings": {name: _scrub(os.environ[name])[:MAX_TEXT] for name in HF_SETTINGS if name in os.environ},
        "not_collected": {"model_digest": "asked of the model server at run start: the manifest's model_server field",
                          "loaded_context": "asked of the model server at run start: the manifest's model_server field",
                          "retrieval_model_revisions": "in the Hugging Face cache outside the project, which is not read"}}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evaluation.studio_eval.run_eval", description=__doc__.split("\n\n")[0])
    parser.add_argument("--claims", type=Path, required=True)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--verifier", help="a candidate verifier for a paired comparison (default: the Studio default)")
    parser.add_argument("--check-overlap", action="store_true", help="only report overlap with used claim sets")
    args = parser.parse_args(argv)
    try:
        claims = load_claims(args.claims)
        if args.check_overlap:
            overlap = check_overlap(claims)
            print(json.dumps({"claims": len(claims), "overlapping_claim_ids": overlap}, indent=1))
            return 0
        freeze = check_freeze(args.claims, args.freeze)
    except ProtocolError as exc:
        parser.error(str(exc))
    orch, options = build_system(args.verifier)
    from evaluation.studio_eval import records as integrity
    out = args.out or ROOT / "outputs" / "studio_eval" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True, exist_ok=True)
    if (out / "records.jsonl").exists():                # never overwrite or append to another run's records (F5c)
        parser.error(f"{out.name} already holds records.jsonl; choose a new output folder")
    lock = integrity.acquire_lock(out)
    try:
        return _write_run(claims, freeze, orch, options, out, integrity)
    finally:
        lock.unlink(missing_ok=True)


LOCAL_MODEL_HOSTS = {"127.0.0.1", "localhost", "::1"}


def model_server_state(client: Any, timeout: float = 5.0) -> Dict[str, Any]:
    """What the model server reports at run start: its version, the model's digest and the context length the model
    is loaded with. Unlike ``environment``, this asks the server (local GET requests only; added 2026-10-26 at the
    owner's request so manifests stop saying "not recorded"). Anything the server does not answer stays None, with
    the reason in ``errors``; a run never stops because of it."""
    out: Dict[str, Any] = {"ollama_version": None, "model_digest": None, "context_length": None,
                           "model": getattr(client, "model", None), "errors": []}
    describe = getattr(client, "describe", None)
    if callable(describe):
        try:
            info = describe() or {}
            out["ollama_version"] = info.get("ollama_version")
            out["model_digest"] = info.get("model_digest")
        except Exception as exc:                       # a remote or unusual client: recorded, never fatal
            out["errors"].append(f"describe: {short_error(exc)}")
    else:
        out["errors"].append("the model client cannot describe its server")
    base = str(getattr(client, "base_url", "") or "")
    if base and urllib.parse.urlparse(base).hostname in LOCAL_MODEL_HOSTS:
        try:
            with urllib.request.urlopen(base.rstrip("/") + "/api/ps", timeout=timeout) as response:
                for row in json.loads(response.read()).get("models", []):
                    if out["model"] in (row.get("name"), row.get("model")):
                        out["context_length"] = row.get("context_length")
            if out["context_length"] is None:
                out["errors"].append("the model was not loaded when the run started (no context length yet)")
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
            out["errors"].append(f"api/ps: {short_error(exc)}")
    return out


def _write_run(claims, freeze, orch, options, out: Path, integrity) -> int:
    run_id = invocation_id = integrity.new_id()         # this harness has no resume: one start is one run
    manifest = {"run_id": run_id, "invocation_id": invocation_id,
                "started_utc": datetime.now(timezone.utc).isoformat(), "git_head": _git_head(),
                "claims_sha256": freeze["claims_sha256"], "claims": len(claims), "options": options,
                "verifier_class": type(orch.verifier).__name__, "config_fingerprint": fingerprint(options, orch.verifier),
                "model": getattr(orch.client, "model", None), "protocol": "evaluation/studio_eval/PROTOCOL.md",
                "environment": environment(orch), "model_server": model_server_state(orch.client)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")
    records = [integrity.stamp(r, run_id, invocation_id) for r in evaluate(claims, orch, options, progress=print)]
    (out / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    print(f"{len(records)} records written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
