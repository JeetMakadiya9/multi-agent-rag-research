"""Durable, auditable helpers for Phase 10 evaluation runs.

This module is evaluation infrastructure only. It does not call a model or
change retrieval, verification, or controller behavior.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
import urllib.error
import urllib.request
import contextlib
import errno
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


class OutputState(StrEnum):
    EXECUTION_FAILED = "EXECUTION_FAILED"
    OUTPUT_UNAVAILABLE = "OUTPUT_UNAVAILABLE"
    VALID_EMPTY = "VALID_EMPTY"
    VALID_NONEMPTY = "VALID_NONEMPTY"
    VALID_OUTPUT_INVALID = "VALID_OUTPUT_INVALID"


def output_state(*, execution_succeeded: bool, output_available: bool,
                 schema_valid: bool | None, item_count: int | None) -> OutputState:
    """Classify one independently persisted output without conflating absence and empty."""
    if not execution_succeeded:
        return OutputState.EXECUTION_FAILED
    if not output_available:
        return OutputState.OUTPUT_UNAVAILABLE
    if schema_valid is False:
        return OutputState.VALID_OUTPUT_INVALID
    if item_count is None:
        return OutputState.OUTPUT_UNAVAILABLE
    return OutputState.VALID_EMPTY if item_count == 0 else OutputState.VALID_NONEMPTY


def retrieval_availability_summary(rows: Sequence[Mapping[str, Any]], *,
                                   gold_field: str = "gold_evidence_doc_ids",
                                   status_field: str = "retrieval_output_status") -> dict[str, Any]:
    eligible = [r for r in rows if r.get(gold_field)]
    status_counts = {state.value: 0 for state in OutputState}
    valid = []
    for row in eligible:
        raw = row.get(status_field, OutputState.OUTPUT_UNAVAILABLE.value)
        try:
            state = OutputState(raw)
        except ValueError:
            state = OutputState.VALID_OUTPUT_INVALID
        status_counts[state.value] += 1
        if state in (OutputState.VALID_EMPTY, OutputState.VALID_NONEMPTY):
            valid.append(row)
    gold_docs = sum(len(set(r.get(gold_field) or [])) for r in valid)
    hits: dict[str, int] = {}
    for k in (1, 5, 10, 20):
        hits[str(k)] = sum(
            len({c.get("scifact_doc_id") for c in (r.get("retrieved_chunks") or [])[:k]
                 if c.get("scifact_doc_id") is not None} & set(r.get(gold_field) or []))
            for r in valid
        )
    return {
        "eligible_claims_with_gold_evidence": len(eligible),
        "output_status_counts": status_counts,
        "valid_output_claims": len(valid),
        "valid_empty_output_claims": status_counts[OutputState.VALID_EMPTY.value],
        "unavailable_or_failed_claims": (status_counts[OutputState.EXECUTION_FAILED.value]
                                         + status_counts[OutputState.OUTPUT_UNAVAILABLE.value]),
        "invalid_output_claims": status_counts[OutputState.VALID_OUTPUT_INVALID.value],
        "gold_document_denominator_over_valid_outputs": gold_docs,
        "gold_document_hits_at_k": hits,
        "recall_at_k_over_valid_outputs": {k: hits[k] / gold_docs if gold_docs else None for k in hits},
        "metric_definition": "gold-document overlap is scored only for eligible claims with valid retrieval output; execution failure/unavailable/invalid output is never counted as a miss",
    }


class JsonlRecoveryError(ValueError):
    """A complete malformed record, schema error, or duplicate key blocks recovery."""


@dataclass(frozen=True)
class JsonlRecoveryReport:
    path: str
    records: int
    recovered: bool
    quarantined_path: str | None
    quarantined_sha256: str | None
    normalized_final_newline: bool


def recover_jsonl(path: str | Path, *, validator: Callable[[Any], None] | None = None,
                  unique_key: str | Callable[[Mapping[str, Any]], Any] | None = None) -> tuple[list[Any], JsonlRecoveryReport]:
    """Parse a JSONL log and quarantine only a malformed unterminated final tail.

    A malformed newline-terminated line is a complete-line corruption and is
    never discarded. Valid JSON without a final newline is kept and normalized.
    """
    path = Path(path)
    if not path.exists():
        return [], JsonlRecoveryReport(str(path), 0, False, None, None, False)
    raw = path.read_bytes()
    if not raw:
        return [], JsonlRecoveryReport(str(path), 0, False, None, None, False)
    chunks = raw.splitlines(keepends=True)
    records: list[Any] = []
    seen: set[Any] = set()
    valid_prefix = bytearray()
    tail: bytes | None = None
    for i, chunk in enumerate(chunks):
        terminated = chunk.endswith((b"\n", b"\r"))
        content = chunk.rstrip(b"\r\n") if terminated else chunk
        if not content:
            if terminated:
                valid_prefix.extend(chunk)
                continue
            tail = chunk
            break
        try:
            decoded = content.decode("utf-8")
            item = json.loads(decoded)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            if i == len(chunks) - 1 and not terminated and _is_incomplete_json_tail(content, exc):
                tail = chunk
                break
            raise JsonlRecoveryError(f"Malformed complete JSONL line {i + 1} in {path}: {exc}") from exc
        if validator is not None:
            try:
                validator(item)
            except Exception as exc:
                raise JsonlRecoveryError(f"Schema-invalid JSONL line {i + 1} in {path}: {exc}") from exc
        if unique_key is not None:
            key = unique_key(item) if callable(unique_key) else item.get(unique_key) if isinstance(item, Mapping) else None
            if key is None:
                raise JsonlRecoveryError(f"Missing unique key at JSONL line {i + 1} in {path}")
            if key in seen:
                raise JsonlRecoveryError(f"Duplicate JSONL key {key!r} at line {i + 1} in {path}")
            seen.add(key)
        records.append(item)
        valid_prefix.extend(chunk if terminated else chunk + b"\n")
    if tail is None:
        normalized = not raw.endswith((b"\n", b"\r"))
        if normalized:
            _atomic_bytes(path, bytes(valid_prefix))
        return records, JsonlRecoveryReport(str(path), len(records), False, None, None,
                                           normalized)
    digest = hashlib.sha256(tail).hexdigest()
    quarantine = path.with_name(f"{path.name}.truncated-{digest[:16]}.quarantine")
    if quarantine.exists() and quarantine.read_bytes() != tail:
        raise JsonlRecoveryError(f"Quarantine path collision: {quarantine}")
    if not quarantine.exists():
        _atomic_bytes(quarantine, tail)
    _atomic_bytes(path, bytes(valid_prefix))
    return records, JsonlRecoveryReport(str(path), len(records), True, str(quarantine), digest, False)


def _is_incomplete_json_tail(content: bytes, error: Exception) -> bool:
    """Only recognize parser errors whose location proves input ended early."""
    if isinstance(error, UnicodeDecodeError):
        return error.end == len(content) and error.reason == "unexpected end of data"
    if not isinstance(error, json.JSONDecodeError):
        return False
    if error.msg.startswith("Unterminated string"):
        return True
    return error.pos >= len(error.doc) and error.msg in {
        "Expecting value", "Expecting property name enclosed in double quotes",
        "Expecting ',' delimiter", "Expecting ':' delimiter", "Expecting '}'",
        "Expecting ']'",
    }


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def append_jsonl(path: str | Path, record: Mapping[str, Any]) -> None:
    """Append one complete JSON record and fsync the file before returning."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o666)
    try:
        with os.fdopen(fd, "ab", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)


def atomic_write_json(path: str | Path, value: Any) -> None:
    _atomic_bytes(Path(path), (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))


def canonical_fingerprint(configuration: Mapping[str, Any]) -> str:
    canonical = json.dumps(configuration, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def field_differences(expected: Any, actual: Any, prefix: str = "") -> list[dict[str, Any]]:
    """Return deterministic JSON-pointer-like field-level differences."""
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        diffs = []
        for key in sorted(set(expected) | set(actual), key=str):
            name = f"{prefix}.{key}" if prefix else str(key)
            if key not in expected:
                diffs.append({"field": name, "stored": "<missing>", "current": actual[key]})
            elif key not in actual:
                diffs.append({"field": name, "stored": expected[key], "current": "<missing>"})
            else:
                diffs.extend(field_differences(expected[key], actual[key], name))
        return diffs
    if isinstance(expected, list) and isinstance(actual, list):
        return [] if expected == actual else [{"field": prefix, "stored": expected, "current": actual}]
    return [] if expected == actual else [{"field": prefix, "stored": expected, "current": actual}]


def validate_resume_configuration(stored: Mapping[str, Any], current: Mapping[str, Any]) -> str:
    diffs = field_differences(stored, current)
    if diffs:
        detail = "; ".join(f"{d['field']}: {d['stored']!r} != {d['current']!r}" for d in diffs)
        raise ValueError(f"Incompatible resume configuration does not match stored run ({detail})")
    return canonical_fingerprint(current)


def load_resume_manifest(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise ValueError(f"Resume artifact missing: {path.name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Corrupted resume manifest {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Corrupted resume manifest {path.name}: root must be an object")
    return value


def interrupted_attempt_record(inflight: Mapping[str, Any]) -> dict[str, Any]:
    """Represent an invocation interrupted before its result was durably logged."""
    return {"attempt_id": inflight["attempt_id"], "example_id": inflight["example_id"],
        "system": inflight["system"], "split": "dev", "runtime_input": inflight["runtime_input"],
        "status": "INTERRUPTED", "latency_seconds": None, "predicted_label": None,
        "retrieved_chunks": [], "selected_evidence_doc_ids": [], "assessments": [],
        "unsupported_claims": None, "verification_calls": None, "llm_calls": None,
        "retrieval_calls": None, "controller_cycles": None, "raw_output": None,
        "error": "Process interruption; callback completion is unknown",
        "error_type": "InterruptedAttempt", "error_stage": "system_callback",
        "error_traceback": None, "retrieval_output_status": OutputState.OUTPUT_UNAVAILABLE.value,
        "evidence_output_status": OutputState.OUTPUT_UNAVAILABLE.value,
        "interruption_provenance": {"started_at_utc": inflight.get("started_at_utc"),
            "inference_started": "UNKNOWN", "retrieval_occurred": "UNKNOWN",
            "automatic_retry": False}}


def validate_attempt_record(record: Any) -> None:
    if not isinstance(record, dict):
        raise ValueError("record must be an object")
    required = {"attempt_id", "example_id", "system", "status", "runtime_input"}
    missing = sorted(required - set(record))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if record["system"] not in {"A", "B", "C"}:
        raise ValueError("system must be A, B, or C")
    if record["status"] not in {"SUCCESS", "FAILED", "TIMEOUT", "INTERRUPTED"}:
        raise ValueError("unsupported execution status")


def attempt_id_key(record: Mapping[str, Any]) -> str:
    return str(record["attempt_id"])


def validate_scored_record(record: Any) -> None:
    if not isinstance(record, dict):
        raise ValueError("record must be an object")
    required = {"example_id", "system", "status", "retrieval", "gold_label",
                "runtime_input", "retrieved_chunks"}
    missing = sorted(required - set(record))
    if missing:
        raise ValueError(f"missing required scored fields: {missing}")
    if record["system"] not in {"A", "B", "C"}:
        raise ValueError("system must be A, B, or C")


def claim_system_key(record: Mapping[str, Any]) -> tuple[Any, str]:
    return record["example_id"], record["system"]


def resume_pending_pairs(expected_pairs: Iterable[tuple[Any, str]],
                         attempts: Sequence[Mapping[str, Any]]) -> list[tuple[Any, str]]:
    """Return only never-persisted claim/system pairs; failures are terminal records."""
    keys = [(row.get("example_id"), row.get("system")) for row in attempts]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate claim/system attempt records prevent safe resume")
    completed_or_failed = set(keys)
    return [pair for pair in expected_pairs if pair not in completed_or_failed]


@contextlib.contextmanager
def exclusive_run_lock(path: str | Path):
    """Prevent two writers from mutating the same run directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        try:
            lock_record = json.loads(path.read_text(encoding="utf-8"))
            pid = int(lock_record["pid"])
        except Exception as read_exc:
            raise RuntimeError(f"Evaluation run lock is malformed; refusing to take over: {path}") from read_exc
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            # A dead writer left a stale lock after interruption. Remove only
            # this syntactically valid lock, then acquire with O_EXCL again.
            path.unlink()
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError as retry_exc:
                raise RuntimeError(f"Evaluation run is locked by another writer: {path}") from retry_exc
        except PermissionError:
            raise RuntimeError(f"Evaluation run is locked by another writer: {path}") from exc
        except OSError as process_error:
            if process_error.errno == errno.ESRCH or getattr(process_error, "winerror", None) == 87:
                path.unlink()
                try:
                    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                except FileExistsError as retry_exc:
                    raise RuntimeError(f"Evaluation run is locked by another writer: {path}") from retry_exc
            else:
                raise RuntimeError(f"Cannot verify evaluation lock owner {pid}; refusing takeover") from process_error
        else:
            raise RuntimeError(f"Evaluation run is locked by live process {pid}: {path}") from exc
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid(), "created_unix": time.time()}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        yield
    finally:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def make_configuration_lock(configuration: Mapping[str, Any]) -> dict[str, Any]:
    """Return the readable configuration identity plus its deterministic fingerprint."""
    material = json.loads(json.dumps(configuration, sort_keys=True, ensure_ascii=False))
    return {"schema_version": "phase10e-config-lock-v1", "configuration": material,
            "fingerprint_sha256": canonical_fingerprint(material)}


def model_health_preflight(*, model: str = "qwen3:4b",
                           base_url: str = "http://127.0.0.1:11434",
                           temperature: float = 0.0, timeout_seconds: float = 8.0,
                           provider_parameters: Mapping[str, Any] | None = None,
                           opener: Callable[..., Any] | None = None) -> dict[str, Any]:
    """Check local Ollama identity and one tiny deterministic JSON response.

    This is an infrastructure ping only. It does not use benchmark claims or
    SciFact evidence and sends no request when the tags/show checks fail.
    """
    started = time.time()
    result: dict[str, Any] = {
        "schema_version": "phase10e-model-preflight-v1",
        "model": model,
        "endpoint": base_url,
        "temperature": temperature,
        "timeout_seconds": timeout_seconds,
        "generation_parameters": {"num_predict": 16, "num_gpu": 0, "think": False,
                                   "format": "json", **dict(provider_parameters or {})},
        "endpoint_reachable": False,
        "model_present": False,
        "request_succeeded": False,
        "json_capability": False,
        "model_digest": None,
        "digest_status": "unavailable",
        "status": "FAIL",
        "errors": [],
        "elapsed_seconds": None,
    }
    # Bypass ambient proxy settings for a loopback-only health check. A proxy
    # can make a failed local endpoint appear to hang past its intended timeout.
    local_open = opener or urllib.request.build_opener(urllib.request.ProxyHandler({})).open

    def request(path: str, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        url = base_url.rstrip("/") + path
        if payload is None:
            req = urllib.request.Request(url, method="GET")
        else:
            req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}, method="POST")
        with local_open(req, timeout=timeout_seconds) as response:
            data = response.read()
        return json.loads(data.decode("utf-8"))

    try:
        tags = request("/api/tags")
        result["endpoint_reachable"] = True
        models = tags.get("models", [])
        match = next((m for m in models if m.get("name") == model or m.get("model") == model), None)
        if match is None:
            result["errors"].append(f"Requested model {model!r} is not listed by /api/tags")
            return result
        result["model_present"] = True
        digest = match.get("digest")
        if digest:
            result["model_digest"] = str(digest)
            result["digest_status"] = "captured"
        show = request("/api/show", {"name": model})
        result["model_identity"] = {"requested_name": model,
                                    "reported_name": match.get("name") or match.get("model"),
                                    "digest": result["model_digest"],
                                    "details": show.get("details", {}),
                                    "model_info": show.get("model_info", {})}
        chat = request("/api/chat", {
            "model": model,
            "messages": [{"role": "user", "content": "Return exactly this JSON object: {\"status\":\"ok\"}"}],
            "stream": False,
            "think": False,
            "format": "json",
            "options": {"temperature": temperature, "num_gpu": 0, "num_predict": 16},
        })
        result["request_succeeded"] = True
        content = chat.get("message", {}).get("content", "")
        result["response_content"] = content
        try:
            parsed = json.loads(content)
            result["json_capability"] = isinstance(parsed, dict) and parsed.get("status") == "ok"
        except (TypeError, json.JSONDecodeError):
            result["json_capability"] = False
        if not result["json_capability"]:
            result["errors"].append("Minimal model response did not satisfy the JSON response contract")
        else:
            result["status"] = "PASS"
    except Exception as exc:
        kind = "timeout" if isinstance(exc, TimeoutError) else "connection_refused" if isinstance(exc, ConnectionRefusedError) else type(exc).__name__
        result["errors"].append(f"{kind}: {exc}")
        result["failure_category"] = kind
    finally:
        result["elapsed_seconds"] = time.time() - started
    return result

