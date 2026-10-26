"""Coverage measures for verifier passage-id matching (evaluation/studio_eval/id_coverage.py). Synthetic records."""
import json

from evaluation.studio_eval import id_coverage


def diag(reply_entry, check_entry=None):
    return {"reply_entry": reply_entry, "studio_check_entry": check_entry or reply_entry}


def test_measures_count_matched_unmatched_silent_and_rejected():
    records = [
        {"claim_id": "a", "sub_state": "", "passage_diagnostics": [diag("valid"), diag("missing")],
         "reply_identifiers": [{"returned_ids": ["P1"]}, {"returned_ids": ["P1"]}]},              # old candidate
        {"claim_id": "b", "sub_state": "ASSESSMENT_FAILED", "passage_diagnostics": [diag("valid")],
         "reply_identifiers": [{"id_check": {"id_check": "ok"}},
                               {"id_check": {"id_check": "rejected", "code": "unexpected_id"}},
                               {"id_check": {"id_check": "rejected", "code": "duplicate_id"}}]},
        {"claim_id": "c", "sub_state": "NO_DECISIVE_EVIDENCE",
         "passage_diagnostics": {"E1": diag("valid"), "E2": diag("invalid_stance")},
         "reply_identifiers": [{"id_check": {"id_check": "ok"}}, {"id_check": {"id_check": "ok"}}]},
    ]
    m = id_coverage.measure(records)
    assert (m["claims"], m["passages_presented"], m["valid_matched"]) == (3, 5, 3)
    assert m["unmatched_or_invalid"] == {"missing": 1, "invalid_stance": 1}
    assert m["silent_neutral"] == 2                       # claims a and c were decided with an unmatched passage
    assert m["replies"] == {"unchecked": 2, "ok": 3, "rejected": 2}
    assert m["rejected_by_code"] == {"unexpected_id": 1, "duplicate_id": 1} and m["assessment_failed"] == 1


def test_a_passage_matched_by_one_reading_only_is_not_counted_as_valid():
    m = id_coverage.measure([{"claim_id": "a", "sub_state": "", "reply_identifiers": [],
                              "passage_diagnostics": {"E1": diag("valid", "missing")}}])
    assert m["valid_matched"] == 0 and m["silent_neutral"] == 1


def test_gold_and_verdict_are_never_loaded(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text(json.dumps({"claim_id": "a", "gold": "SUPPORTED", "verdict": "SUPPORTED", "sub_state": "",
                                "passage_diagnostics": {}, "reply_identifiers": []}) + "\n", encoding="utf-8")
    [row] = id_coverage.load(path)
    assert set(row) == set(id_coverage.ALLOWED) and "gold" not in row and "verdict" not in row
