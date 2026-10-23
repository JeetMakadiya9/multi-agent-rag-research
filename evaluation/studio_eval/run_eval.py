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
import subprocess
import time
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
    out = args.out or ROOT / "outputs" / "studio_eval" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"started_utc": datetime.now(timezone.utc).isoformat(), "git_head": _git_head(),
                "claims_sha256": freeze["claims_sha256"], "claims": len(claims), "options": options,
                "verifier_class": type(orch.verifier).__name__, "config_fingerprint": fingerprint(options, orch.verifier),
                "model": getattr(orch.client, "model", None), "protocol": "evaluation/studio_eval/PROTOCOL.md"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")
    records = evaluate(claims, orch, options, progress=print)
    (out / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    print(f"{len(records)} records written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
