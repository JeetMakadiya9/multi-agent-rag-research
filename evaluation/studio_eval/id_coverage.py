"""Passage-id coverage of a run's verifier replies (``arm_c_studio_all_idfix`` and, for contrast, any Arm C run).

Coverage only: the record fields that carry the gold label or the verdict are never read (``ALLOWED`` below is the
complete list of fields used), so this cannot show accuracy.

    python -m evaluation.studio_eval.id_coverage --records RUNDIR/records.jsonl [--json-out FILE]

Measures:
* ``passages_presented``: passages the verifier judged (one ``passage_diagnostics`` entry each);
* ``valid_matched``: passages whose reply entry was matched to them exactly once with a valid stance (``reply_entry``
  and ``studio_check_entry`` both "valid");
* ``unmatched_or_invalid``: passages read without a valid matched entry, by kind ("missing", "invalid_stance", None
  when the replies could not be reconstructed);
* ``silent_neutral``: of those, passages in claims that were nevertheless decided (not ASSESSMENT_FAILED or another
  failure). These are verdicts that an invalid or unmatched reply turned into NEUTRAL. It must be 0 for the fix;
* ``replies``: verifier replies, by the wrapper's check ("ok", "rejected" by reason code, or "unchecked" when the run
  did not use the id check);
* ``assessment_failed`` and the other failure sub-states, per claim.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

ALLOWED = ("claim_id", "sub_state", "passage_diagnostics", "reply_identifiers")
FAILURES = ("ASSESSMENT_FAILED", "SERVICE_ERROR", "HARNESS_ERROR")


def load(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            rows.append({k: record.get(k) for k in ALLOWED})       # nothing else is kept
    return rows


def measure(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    claims = presented = matched = silent = 0
    unmatched: Counter = Counter()
    replies: Counter = Counter()
    rejected: Counter = Counter()
    sub_states: Counter = Counter()
    for record in records:
        claims += 1
        failed = record.get("sub_state") in FAILURES
        sub_states[record.get("sub_state") or "decided"] += 1
        diagnostics = record.get("passage_diagnostics") or []      # a list in run records, a dict on a Verdict
        for diag in (diagnostics.values() if isinstance(diagnostics, dict) else diagnostics):
            presented += 1
            if diag.get("reply_entry") == "valid" and diag.get("studio_check_entry") == "valid":
                matched += 1
                continue
            unmatched[str(diag.get("reply_entry"))] += 1
            if not failed:
                silent += 1
        for reply in record.get("reply_identifiers") or []:
            check = reply.get("id_check")
            if not check:
                replies["unchecked"] += 1
            elif check.get("id_check") == "ok":
                replies["ok"] += 1
            else:
                replies["rejected"] += 1
                rejected[check.get("code") or "unknown"] += 1
    return {"claims": claims, "passages_presented": presented, "valid_matched": matched,
            "unmatched_or_invalid": dict(unmatched), "silent_neutral": silent,
            "replies": dict(replies), "rejected_by_code": dict(rejected),
            "assessment_failed": sub_states.get("ASSESSMENT_FAILED", 0), "claim_sub_states": dict(sub_states)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)
    result = measure(load(args.records))
    text = json.dumps(result, indent=1)
    print(text)
    if args.json_out:
        args.json_out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
