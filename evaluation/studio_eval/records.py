"""Record provenance (F5a) and duplicate-scoring integrity (F5c) for Studio evaluation records.

Provenance. Records written from now on carry three fields:
* ``run_id``: one per run folder (kept in the folder's ``run.json``), shared by every start and resume of that run;
* ``invocation_id``: one per start or resume of the runner (it also names that start's manifest);
* ``record_fingerprint``: ``rf1:`` + SHA-256 of the record's canonical fields (``FINGERPRINT_FIELDS``; see
  ``fingerprint``). It identifies the record's content. It is not proof of which run wrote the record, and it is not
  evidence of scientific validity.
Records written before this change have none of these fields. They are treated as legacy records; no run identity is
ever inferred or filled in for them.

Duplicate policy (one claim with more than one row in a scoring input):
A. Rows of one identified run whose scoring fields agree (``scoring_signature``): the claim counts once. All rows are
   kept in the file; other fields that differ (timings, notes, stances, quotes, evidence-safety counts) are reported
   per claim. The row used for descriptive counts is the one with the smallest fingerprint, so row order never
   changes any output.
B. Rows of one identified run whose scoring fields disagree: the input is rejected (``RecordIntegrityError``) with the
   conflicting claims and fields. No row is chosen.
C. Rows of different runs are never merged. An input holding more than one run is rejected unless one run is
   selected (``run_id=``); comparisons between runs load each run separately.
D. Duplicate rows without run identity (legacy), or an input that mixes identified and legacy rows, cannot be
   classified: the duplicate groups are reported as unresolved and the input is rejected. A legacy input without
   duplicates is scored as before.
A stored fingerprint that does not match the record's content also rejects the input.
"""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

POLICY = "F5c duplicate policy v1"
FINGERPRINT_VERSION = "rf1"
PROVENANCE_FIELDS = ("run_id", "invocation_id", "record_fingerprint")
# The canonical fields of a record's fingerprint: the claim, its gold label and the verdict as the scorer reads it,
# the verifier that produced it, and the per-passage stances. Timings, notes, quotes and traces are not included.
FINGERPRINT_FIELDS = ("claim_id", "gold", "verdict", "sub_state", "parse_ok", "verifier", "verifier_class", "error",
                      "stances", "subgroups")
LEGACY = "(no run identity)"


class RecordIntegrityError(ValueError):
    """A scoring input that the duplicate policy does not allow to be scored; ``diagnostics`` says why."""

    def __init__(self, message: str, diagnostics: Dict[str, Any]):
        super().__init__(message)
        self.diagnostics = diagnostics


def acquire_lock(out: Path) -> Path:
    """One writer per run folder: a second start or resume while one is writing is refused (F5c). The lock holds the
    process id and start time only. A lock left by a crashed run must be removed by hand after checking that no run
    is writing."""
    lock = Path(out) / "run.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            holder = json.loads(lock.read_text(encoding="utf-8"))
            owner = f"pid {holder.get('pid')}, started {holder.get('started_utc')}"
        except (OSError, ValueError, AttributeError):
            owner = "owner unreadable"
        raise SystemExit(f"another run is writing to {Path(out).name} ({lock.name} exists: {owner}); if that "
                         f"process is no longer running, the lock is stale: delete {lock.name} and start again") from None
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(json.dumps({"pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat()}))
    return lock


def set_run_status(out: Path, status: str, **fields: Any) -> None:
    """Records in the folder's ``run.json`` whether the run is ``in_progress`` (started or resumed, not finished: an
    interrupted run stays in this state) or ``complete`` (every claim of its claims file has a record)."""
    info = Path(out) / "run.json"
    data = json.loads(info.read_text(encoding="utf-8"))
    data.update(status=status, status_utc=datetime.now(timezone.utc).isoformat(), **fields)
    info.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8", newline="\n")


def check_run_folder(out: Path) -> Dict[str, Any]:
    """Whether a run folder's records, ``run.json`` and manifests agree (read only; nothing is repaired)."""
    out = Path(out)
    info = out / "run.json"
    if not info.is_file():
        return {"provenance": "legacy (no run.json)", "status": "unknown"}
    meta = json.loads(info.read_text(encoding="utf-8"))
    rows = _rows(out / "records.jsonl") if (out / "records.jsonl").is_file() else []
    manifests = []
    for path in sorted(out.glob("manifest_*.json")):
        try:
            manifests.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            manifests.append({})
    invocations = {m.get("invocation_id") for m in manifests if isinstance(m, dict)}
    return {"provenance": "run.json", "run_id": meta.get("run_id"), "status": meta.get("status", "unknown"),
            "records": len(rows), "lock_present": (out / "run.lock").exists(),
            "records_of_other_runs": sum(1 for r in rows if r.get("run_id") != meta.get("run_id")),
            "manifests_of_other_runs": sum(1 for m in manifests if isinstance(m, dict) and m.get("run_id") != meta.get("run_id")),
            "records_without_a_manifest": sum(1 for r in rows if r.get("invocation_id") not in invocations)}


def new_id() -> str:
    """A fresh run or invocation identifier (random; it carries no path, user or machine information)."""
    return uuid.uuid4().hex


def _normalise(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return [_normalise(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _normalise(v) for k, v in sorted(value.items())}
    return value


def fingerprint(record: Dict[str, Any]) -> str:
    """``rf1:`` + SHA-256 of the canonical JSON of ``FINGERPRINT_FIELDS``: a missing field is null, strings are
    stripped, keys are sorted, separators are compact and non-ASCII text is kept as is."""
    canonical = {"v": FINGERPRINT_VERSION, **{name: _normalise(record.get(name)) for name in FINGERPRINT_FIELDS}}
    text = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return f"{FINGERPRINT_VERSION}:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


def stamp(record: Dict[str, Any], run_id: str, invocation_id: str) -> Dict[str, Any]:
    """The record with its provenance fields set (the fingerprint is computed from the record as it is)."""
    record.update(run_id=run_id, invocation_id=invocation_id)
    record["record_fingerprint"] = fingerprint(record)
    return record


def scoring_signature(record: Dict[str, Any]) -> Dict[str, Any]:
    """The fields the governing scorer (score.py) reads to score a claim: gold label, normalised prediction, failure
    status and kind, sub-state (failure kinds and the ADJUDICATED count) and the verifier that ran."""
    from evaluation.studio_eval import score
    return {"gold": record.get("gold"), "prediction": score.predicted(record), "failure": score.is_failure(record),
            "sub_state": record.get("sub_state") or "", "verifier": record.get("verifier")}


def _rows(source: Union[Path, str, Iterable[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """The input's rows. A line that is not a JSON object with a non-empty string ``claim_id`` (for example the
    truncated last line of an interrupted write) rejects the input; no row is skipped silently."""
    if isinstance(source, (str, Path)):
        lines = [(n, line) for n, line in enumerate(Path(source).read_text(encoding="utf-8").splitlines(), 1)
                 if line.strip()]
    else:
        lines = [(n, row) for n, row in enumerate(source, 1)]
    rows, bad = [], []
    for n, line in lines:
        try:
            row = json.loads(line) if isinstance(line, str) else line
        except json.JSONDecodeError as exc:
            bad.append({"line": n, "problem": f"not JSON ({exc.msg})"})
            continue
        if not isinstance(row, dict):
            bad.append({"line": n, "problem": "not a JSON object"})
        elif not isinstance(row.get("claim_id"), str) or not row["claim_id"].strip():
            bad.append({"line": n, "problem": "missing or empty claim_id"})
        else:
            rows.append(row)
    if bad:
        diag = {"policy": POLICY, "rows": len(lines), "malformed_rows": bad, "run_ids": [], "selected_run_id": None,
                "legacy_rows_without_run_id": 0, "fingerprint_mismatches": [], "agreeing_duplicates": [],
                "conflicts": [], "unresolved": [], "rows_of_other_runs": 0}
        _reject(f"{len(bad)} row(s) are malformed (first at line {bad[0]['line']}: {bad[0]['problem']})", diag)
    return rows


def _differing(rows: List[Dict[str, Any]], skip=PROVENANCE_FIELDS) -> List[str]:
    keys = sorted({k for r in rows for k in r} - set(skip))
    return [k for k in keys if len({json.dumps(r.get(k), sort_keys=True, default=str) for r in rows}) > 1]


def load_records(source: Union[Path, str, Iterable[Dict[str, Any]]], run_id: Optional[str] = None):
    """The scoring view of one run's records (claim id -> record) and the integrity diagnostics. Raises
    ``RecordIntegrityError`` when the policy forbids scoring the input. The input itself is never changed."""
    rows = _rows(source)
    diag: Dict[str, Any] = {"policy": POLICY, "rows": len(rows), "selected_run_id": run_id,
                            "run_ids": sorted({r.get("run_id") or LEGACY for r in rows}),
                            "legacy_rows_without_run_id": sum(1 for r in rows if not r.get("run_id")),
                            "fingerprint_mismatches": [], "agreeing_duplicates": [], "conflicts": [],
                            "unresolved": [], "rows_of_other_runs": 0}
    for i, r in enumerate(rows):
        stored = r.get("record_fingerprint")
        if stored is not None and stored != fingerprint(r):
            diag["fingerprint_mismatches"].append({"row": i, "claim_id": r.get("claim_id")})
    if run_id is not None:
        selected = [r for r in rows if r.get("run_id") == run_id]
        diag["rows_of_other_runs"] = len(rows) - len(selected)
        if not selected:
            return _reject(f"no record of run {run_id!r} in the input", diag)
        rows = selected
    else:
        identified = {r.get("run_id") for r in rows if r.get("run_id")}
        if len(identified) > 1:
            return _reject("the input holds records of more than one run; select one run (run_id=)", diag)
        if identified and diag["legacy_rows_without_run_id"]:
            diag["mixed_provenance"] = True
            return _reject("the input mixes records with and without run identity", diag)
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(r["claim_id"], []).append(r)
    chosen: Dict[str, Dict[str, Any]] = {}
    for cid, group in groups.items():
        if len(group) == 1:
            chosen[cid] = group[0]
            continue
        if not all(r.get("run_id") for r in group):
            diag["unresolved"].append({"claim_id": cid, "rows": len(group),
                                       "reason": "duplicate rows without run identity"})
            continue
        signatures = [scoring_signature(r) for r in group]
        if any(s != signatures[0] for s in signatures):
            diag["conflicts"].append({"claim_id": cid, "rows": len(group), "fields": sorted(
                k for k in signatures[0] if len({json.dumps(s[k], default=str) for s in signatures}) > 1)})
            continue
        diag["agreeing_duplicates"].append({"claim_id": cid, "rows": len(group), "discrepant_fields": _differing(group)})
        chosen[cid] = min(group, key=fingerprint)
    diag.update(claims=len(groups), duplicate_groups=sum(1 for g in groups.values() if len(g) > 1),
                extra_rows=sum(len(g) - 1 for g in groups.values()))
    if diag["fingerprint_mismatches"]:
        return _reject("a stored record fingerprint does not match the record's content", diag)
    if diag["conflicts"]:
        return _reject(f"{len(diag['conflicts'])} claim(s) have duplicate rows that disagree on scoring fields", diag)
    if diag["unresolved"]:
        return _reject(f"{len(diag['unresolved'])} claim(s) have duplicate rows whose run cannot be established", diag)
    diag["decision"] = "scored" + (": agreeing duplicates counted once" if diag["agreeing_duplicates"] else "")
    return chosen, diag


def _reject(message: str, diag: Dict[str, Any]):
    diag["decision"] = "rejected: " + message
    raise RecordIntegrityError(message, diag)


def summary(diag: Dict[str, Any]) -> Dict[str, Any]:
    """The diagnostics without claim identifiers (for published aggregates)."""
    return {"policy": diag["policy"], "rows": diag["rows"], "claims": diag.get("claims"),
            "runs": len([r for r in diag["run_ids"] if r != LEGACY]),
            "legacy_rows_without_run_id": diag["legacy_rows_without_run_id"],
            "duplicate_groups": diag.get("duplicate_groups", 0), "extra_rows": diag.get("extra_rows", 0),
            "agreeing_duplicates": len(diag["agreeing_duplicates"]),
            "agreeing_duplicates_with_discrepant_fields": sum(1 for d in diag["agreeing_duplicates"]
                                                              if d["discrepant_fields"]),
            # which non-scoring fields differ among agreeing duplicates (the row used for descriptive counts is
            # chosen by fingerprint, so these differences are listed rather than resolved)
            "discrepant_field_counts": dict(sorted(Counter(f for d in diag["agreeing_duplicates"]
                                                           for f in d["discrepant_fields"]).items())),
            "malformed_rows": len(diag.get("malformed_rows", [])),
            "conflicts": len(diag["conflicts"]), "unresolved": len(diag["unresolved"]),
            "rows_of_other_runs": diag["rows_of_other_runs"], "decision": diag["decision"]}
