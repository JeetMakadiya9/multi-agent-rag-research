"""F5c duplicate-scoring integrity and F5a record provenance (evaluation/studio_eval/records.py and its callers).

Every record here is SYNTHETIC TEST DATA. No model, server, network or evaluation dataset is used; runs use a fake
orchestrator whose verifier is never called."""
import io
import json
import random
from contextlib import redirect_stdout

import pytest

from evaluation.studio_eval import alt_benchmarks as alt
from evaluation.studio_eval import records, run_eval, score
from evaluation.studio_eval.records import RecordIntegrityError

S, C, N = "SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO"
GOLD = {"c0": S, "c1": C, "c2": N}


def rec(cid, verdict, run_id="run-a", **extra):
    return {"claim_id": cid, "gold": GOLD[cid], "verdict": verdict, "sub_state": "", "verifier": "arm_c_studio",
            "seconds": 1.0, "stances": ["NEUTRAL"], **({"run_id": run_id} if run_id else {}), **extra}


def write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def claims_file(tmp_path):
    return write(tmp_path / "claims.jsonl", [{"claim_id": c, "claim": "x", "label": g,
                                              "passages": [{"url": "https://e.org", "text": "t"}]}
                                             for c, g in GOLD.items()])


AGREEING = [rec("c0", S), rec("c1", C), rec("c1", C, seconds=9.5, stances=["CONTRADICTS"], injection_withheld=1),
            rec("c2", "INSUFFICIENT")]


# 1 ----------------------------------------------------------------------------------------------------------------
def test_agreeing_duplicates_of_one_run_count_once(tmp_path):
    chosen, diag = records.load_records(AGREEING)
    assert sorted(chosen) == ["c0", "c1", "c2"] and diag["rows"] == 4 and diag["extra_rows"] == 1
    out = alt.score_records(write(tmp_path / "r.jsonl", AGREEING), None, claims_file(tmp_path))
    single = alt.score_records(write(tmp_path / "s.jsonl", [AGREEING[0], AGREEING[1], AGREEING[3]]), None,
                               claims_file(tmp_path))
    assert out["system"]["macro_f1"] == single["system"]["macro_f1"] == 1.0
    assert out["system"]["confusion"] == single["system"]["confusion"]


# 2 ----------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("second, field", [(rec("c1", S), "prediction"), (dict(rec("c1", C), gold=N), "gold"),
                                           (rec("c1", None, sub_state="SERVICE_ERROR"), "failure")])
def test_conflicting_duplicates_of_one_run_are_rejected(tmp_path, second, field):
    rows = [rec("c0", S), rec("c1", C), second]
    with pytest.raises(RecordIntegrityError) as caught:
        alt.score_records(write(tmp_path / "r.jsonl", rows), None, claims_file(tmp_path))
    [conflict] = caught.value.diagnostics["conflicts"]
    assert conflict["claim_id"] == "c1" and field in conflict["fields"]
    assert caught.value.diagnostics["decision"].startswith("rejected")


# 3 ----------------------------------------------------------------------------------------------------------------
def test_agreeing_duplicates_keep_and_report_their_other_differences():
    _, diag = records.load_records(AGREEING)
    [dup] = diag["agreeing_duplicates"]
    assert dup["claim_id"] == "c1" and dup["rows"] == 2
    assert set(dup["discrepant_fields"]) == {"seconds", "stances", "injection_withheld"}
    assert records.summary(diag)["agreeing_duplicates_with_discrepant_fields"] == 1


# 4 ----------------------------------------------------------------------------------------------------------------
def test_the_same_claims_from_different_runs_are_never_merged(tmp_path):
    rows = [rec("c0", S, "run-a"), rec("c1", C, "run-a"), rec("c0", C, "run-b"), rec("c1", S, "run-b")]
    with pytest.raises(RecordIntegrityError, match="more than one run"):
        records.load_records(rows)
    a, da = records.load_records(rows, run_id="run-a")
    b, db = records.load_records(rows, run_id="run-b")
    assert (a["c0"]["verdict"], b["c0"]["verdict"]) == (S, C)
    assert da["rows_of_other_runs"] == db["rows_of_other_runs"] == 2
    paired = alt.score_records(write(tmp_path / "a.jsonl", rows[:2]), write(tmp_path / "b.jsonl", rows[2:]),
                               claims_file(tmp_path))
    assert paired["paired"]["prediction_changed"] == 2                                 # compared, not merged


# 5 ----------------------------------------------------------------------------------------------------------------
def test_missing_or_mixed_run_identity_never_merges_silently():
    legacy = [rec("c0", S, None), rec("c0", S, None)]
    with pytest.raises(RecordIntegrityError) as caught:
        records.load_records(legacy)
    assert caught.value.diagnostics["unresolved"] == [{"claim_id": "c0", "rows": 2,
                                                       "reason": "duplicate rows without run identity"}]
    with pytest.raises(RecordIntegrityError, match="mixes records with and without run identity"):
        records.load_records([rec("c0", S), rec("c1", C, None)])


# 6 ----------------------------------------------------------------------------------------------------------------
def test_row_order_cannot_change_any_output(tmp_path):
    claims = claims_file(tmp_path)
    reference = alt.score_records(write(tmp_path / "r0.jsonl", AGREEING), None, claims)
    for seed in range(5):
        rows = AGREEING[:]
        random.Random(seed).shuffle(rows)
        out = alt.score_records(write(tmp_path / f"r{seed + 1}.jsonl", rows), None, claims)
        assert out["system"] == reference["system"] and out["duplicate_records"] == reference["duplicate_records"]


# 7 ----------------------------------------------------------------------------------------------------------------
def publish_base(tmp_path, rows):
    base = tmp_path / "base"
    (base / "sets" / "retro").mkdir(parents=True)
    claims_file(base / "sets" / "retro")
    (base / "sets" / "retro" / "FREEZE_ALT.json").write_text(json.dumps({"claims_sha256": "f" * 64}), encoding="utf-8")
    run_dir = base / "runs" / "retro_arm_c_studio"
    run_dir.mkdir(parents=True)
    write(run_dir / "records.jsonl", rows)
    (run_dir / "manifest_20260101T000000Z.json").write_text(json.dumps({
        "started_utc": "x", "git_head": "1" * 40, "claims_sha256": "f" * 64, "options": {"verifier": "arm_c_studio"},
        "config_fingerprint": "a" * 64, "model": "scripted", "resumed_with": 0}), encoding="utf-8")
    return base, run_dir


def test_score_load_score_records_and_publish_apply_one_policy(tmp_path):
    good, _ = publish_base(tmp_path / "good", AGREEING)
    chosen = score._load(good / "runs" / "retro_arm_c_studio" / "records.jsonl")
    assert chosen == records.load_records(AGREEING)[0]
    entry = alt.publish(good)["sets"]["retro"]
    assert entry["duplicate_records"]["agreeing_duplicates"] == 1 and entry["system"]["macro_f1"] == 1.0
    conflicting = [rec("c0", S), rec("c1", C), rec("c1", S), rec("c2", "INSUFFICIENT")]
    bad, run_dir = publish_base(tmp_path / "bad", conflicting)
    with pytest.raises(RecordIntegrityError):
        score._load(run_dir / "records.jsonl")
    with pytest.raises(RecordIntegrityError):
        alt.publish(bad)
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        assert score.main(["--records", str(run_dir / "records.jsonl")]) == 2
    assert json.loads(buffer.getvalue())["record_integrity"]["conflicts"][0]["claim_id"] == "c1"


# 8 ----------------------------------------------------------------------------------------------------------------
class FakeOrchestrator:
    """Stands in for the Studio: its verify step is replaced by ``fake_evaluate`` and never called."""
    client = type("Client", (), {"model": "scripted", "base_url": "", "num_predict": 0, "think": False, "timeout": 0})()
    verifier = None

    def _verify(self, *args, **kwargs):
        raise AssertionError("not used")


def fake_evaluate(claims, orch, options, progress=None):
    return [{"claim_id": c["claim_id"], "gold": c["label"], "verdict": "INSUFFICIENT", "sub_state": "",
             "verifier": "arm_c_studio", "seconds": 0.0, "stances": []} for c in claims]


@pytest.fixture
def fake_run(monkeypatch, tmp_path):
    options = {"verifier": "arm_c_studio", "injection_guard": "exclude", "quote_attestation": True}
    monkeypatch.setattr(run_eval, "build_system", lambda verifier=None: (FakeOrchestrator(), options))
    monkeypatch.setattr(run_eval, "evaluate", fake_evaluate)
    monkeypatch.setattr(run_eval, "environment", lambda orch: {"scripted": True})
    monkeypatch.setattr(run_eval, "_git_head", lambda: "0" * 40)
    return claims_file(tmp_path), tmp_path / "out"


def test_a_resumed_run_keeps_its_run_id_and_adds_no_duplicate(fake_run):
    claims, out = fake_run
    assert alt.run(claims, out, None, 2) == 0
    run_id = json.loads((out / "run.json").read_text(encoding="utf-8"))["run_id"]
    assert alt.run(claims, out, None, None) == 0                                      # resume: one claim left
    rows = alt._jsonl(out / "records.jsonl")
    assert [r["claim_id"] for r in rows] == ["c0", "c1", "c2"] and {r["run_id"] for r in rows} == {run_id}
    assert len({r["invocation_id"] for r in rows}) == 2 and not (out / "run.lock").exists()
    manifests = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out.glob("manifest_*.json"))]
    assert {m["run_id"] for m in manifests} == {run_id} and records.load_records(rows)[1]["extra_rows"] == 0


def test_a_second_writer_and_a_legacy_folder_are_refused(fake_run):
    claims, out = fake_run
    out.mkdir()
    (out / "run.lock").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="another run is writing"):
        alt.run(claims, out, None, None)
    assert not (out / "records.jsonl").exists()
    (out / "run.lock").unlink()
    write(out / "records.jsonl", [rec("c0", S, None)])                               # written before run IDs
    with pytest.raises(SystemExit, match="without run identity"):
        alt.run(claims, out, None, None)
    assert alt._jsonl(out / "records.jsonl") == [rec("c0", S, None)] and not (out / "run.lock").exists()


def test_the_protocol_harness_never_appends_to_or_overwrites_existing_records(fake_run, tmp_path):
    claims = tmp_path / "frozen_claims.jsonl"
    claims.write_text("".join(json.dumps({"claim_id": c, "claim": "x", "label": g,
                                          "passages": [{"url": "https://e.org", "text": "t"}]}) + "\n"
                              for c, g in GOLD.items()), encoding="utf-8")
    freeze = tmp_path / "FREEZE.json"
    freeze.write_text(json.dumps({"claims_sha256": run_eval.sha256_file(claims),
                                  "annotation": {"independent_of_model_output": True}}), encoding="utf-8")
    out = tmp_path / "protocol_run"
    assert run_eval.main(["--claims", str(claims), "--freeze", str(freeze), "--out", str(out)]) == 0
    rows = alt._jsonl(out / "records.jsonl")
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert {r["run_id"] for r in rows} == {manifest["run_id"]} and all(r["record_fingerprint"] for r in rows)
    before = (out / "records.jsonl").read_bytes()
    with pytest.raises(SystemExit):
        run_eval.main(["--claims", str(claims), "--freeze", str(freeze), "--out", str(out)])
    assert (out / "records.jsonl").read_bytes() == before


# 9 ----------------------------------------------------------------------------------------------------------------
def test_fingerprints_are_deterministic_over_the_documented_fields():
    a = rec("c1", C, stances=["CONTRADICTS"], verifier_note="note A")
    b = {k: a[k] for k in reversed(list(a))}                                          # another key order
    b.update(verdict=" CONTRADICTED ", seconds=99.0, verifier_note="note B", run_id="run-z")
    assert records.fingerprint(a) == records.fingerprint(b)                           # only canonical fields count
    assert records.fingerprint(a).startswith("rf1:") and len(records.fingerprint(a)) == 4 + 64
    for field, value in (("verdict", S), ("gold", S), ("stances", ["NEUTRAL"]), ("sub_state", "ADJUDICATED")):
        assert records.fingerprint(dict(a, **{field: value})) != records.fingerprint(a), field
    assert records.FINGERPRINT_FIELDS == ("claim_id", "gold", "verdict", "sub_state", "parse_ok", "verifier",
                                          "verifier_class", "error", "stances", "subgroups")


def test_a_stored_fingerprint_that_does_not_match_its_record_is_rejected():
    row = records.stamp(rec("c0", S, None), "run-a", "inv-1")
    assert records.load_records([row])[1]["decision"] == "scored"
    with pytest.raises(RecordIntegrityError, match="fingerprint"):
        records.load_records([dict(row, verdict=C)])


# 10 ---------------------------------------------------------------------------------------------------------------
def test_legacy_records_are_scored_as_before_and_get_no_invented_provenance(tmp_path):
    legacy = [rec("c0", S, None), rec("c1", C, None), rec("c2", "INSUFFICIENT", None)]
    path = write(tmp_path / "legacy.jsonl", legacy)
    before = path.read_bytes()
    chosen, diag = records.load_records(path)
    assert diag["legacy_rows_without_run_id"] == 3 and diag["run_ids"] == [records.LEGACY]
    assert all("run_id" not in r and "record_fingerprint" not in r for r in chosen.values())
    assert path.read_bytes() == before                                                # the file is not changed


# 11 ---------------------------------------------------------------------------------------------------------------
def test_diagnostics_report_counts_conflicts_unresolved_groups_and_the_decision():
    _, diag = records.load_records(AGREEING)
    assert records.summary(diag) == {
        "policy": records.POLICY, "rows": 4, "claims": 3, "runs": 1, "legacy_rows_without_run_id": 0,
        "duplicate_groups": 1, "extra_rows": 1, "agreeing_duplicates": 1, "agreeing_duplicates_with_discrepant_fields": 1,
        "discrepant_field_counts": {"injection_withheld": 1, "seconds": 1, "stances": 1}, "malformed_rows": 0,
        "conflicts": 0, "unresolved": 0, "rows_of_other_runs": 0,
        "decision": "scored: agreeing duplicates counted once"}
    with pytest.raises(RecordIntegrityError) as caught:
        records.load_records([rec("c0", S), rec("c0", C), rec("c1", C), rec("c1", C, seconds=3.0)])
    summary = records.summary(caught.value.diagnostics)
    assert (summary["conflicts"], summary["agreeing_duplicates"], summary["duplicate_groups"]) == (1, 1, 2)
    assert summary["decision"].startswith("rejected: 1 claim(s) have duplicate rows that disagree")


# 12 ---------------------------------------------------------------------------------------------------------------
def test_valid_single_record_inputs_score_exactly_as_before(tmp_path):
    rows = [rec("c0", S, None), rec("c1", "INSUFFICIENT", None), rec("c2", C, None)]
    out = alt.score_records(write(tmp_path / "r.jsonl", rows), None, claims_file(tmp_path))
    expected = score.summarise([S, C, N], rows)                                      # the scorer applied directly
    assert {k: out["system"][k] for k in expected} == expected
    assert out["record_integrity"]["decision"] == "scored"


# ------------------------------------------------------------------------------------------------ hardening (25 Oct)
@pytest.mark.parametrize("text, problem", [
    ('{"claim_id": "c0", "gold": "SUPPORTED", "verdict": "SUPPORTED"}\n{"claim_id": "c1", "gold"', "not JSON"),
    ('[1, 2]\n', "not a JSON object"),
    ('{"gold": "SUPPORTED", "verdict": "SUPPORTED"}\n', "missing or empty claim_id"),
    ('{"claim_id": "  ", "verdict": "SUPPORTED"}\n', "missing or empty claim_id")])
def test_malformed_or_truncated_rows_reject_the_input_with_their_line(tmp_path, text, problem):
    path = tmp_path / "r.jsonl"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(RecordIntegrityError, match=problem) as caught:
        records.load_records(path)
    assert caught.value.diagnostics["malformed_rows"][0]["problem"].startswith(problem.split(" (")[0])
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        assert score.main(["--records", str(path)]) == 2                              # stable rejection code


def test_an_empty_records_file_is_reported_not_divided_by_zero(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        assert score.main(["--records", str(path)]) == 2
    assert "no records" in json.loads(buffer.getvalue())["error"]
    out = alt.score_records(path, None, claims_file(tmp_path))                     # every claim missing
    assert out["missing_records"] == 3 and out["system"]["operational_failures"] == {"MISSING": 3}


def test_unknown_claims_and_gold_mismatches_are_reported(tmp_path):
    rows = [rec("c0", S), dict(rec("c1", C), gold=N), {"claim_id": "zz", "gold": S, "verdict": S, "run_id": "run-a"}]
    out = alt.score_records(write(tmp_path / "r.jsonl", rows), None, claims_file(tmp_path))
    assert out["records_outside_claim_set"] == 1 and out["gold_mismatches"] == 1
    assert out["claims"] == 3 and out["missing_records"] == 1                         # c2 missing; zz not scored


def test_a_stale_lock_names_its_owner_and_is_never_removed_automatically(fake_run):
    claims, out = fake_run
    out.mkdir()
    (out / "run.lock").write_text(json.dumps({"pid": 999999, "started_utc": "2026-10-25T00:00:00Z"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="pid 999999, started 2026-10-25T00:00:00Z.*stale"):
        alt.run(claims, out, None, None)
    assert (out / "run.lock").exists()


def test_an_interrupted_run_stays_in_progress_until_a_resume_completes_it(fake_run, monkeypatch):
    claims, out = fake_run
    calls = {"n": 0}

    def flaky(claims_, orch, options, progress=None):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt("simulated interruption")
        return fake_evaluate(claims_, orch, options)

    monkeypatch.setattr(run_eval, "evaluate", flaky)
    with pytest.raises(KeyboardInterrupt):
        alt.run(claims, out, None, None)
    state = records.check_run_folder(out)
    assert (state["status"], state["records"], state["lock_present"]) == ("in_progress", 1, False)
    monkeypatch.setattr(run_eval, "evaluate", fake_evaluate)
    assert alt.run(claims, out, None, None) == 0
    state = records.check_run_folder(out)
    assert (state["status"], state["records"]) == ("complete", 3)
    assert state["records_of_other_runs"] == state["manifests_of_other_runs"] == state["records_without_a_manifest"] == 0


def test_a_limited_run_is_not_marked_complete(fake_run):
    claims, out = fake_run
    assert alt.run(claims, out, None, 1) == 0
    assert records.check_run_folder(out)["status"] == "in_progress"


def test_a_legacy_folder_is_reported_as_legacy_without_invented_identity(tmp_path):
    folder = tmp_path / "legacy"
    folder.mkdir()
    write(folder / "records.jsonl", [rec("c0", S, None)])
    assert records.check_run_folder(folder) == {"provenance": "legacy (no run.json)", "status": "unknown"}
    assert not (folder / "run.json").exists()


def test_publish_reports_each_run_folder_state(tmp_path):
    base, run_dir = publish_base(tmp_path, AGREEING)
    assert alt.publish(base)["sets"]["retro"]["run_folder"] == {"provenance": "legacy (no run.json)",
                                                                "status": "unknown"}


def test_the_summary_names_the_differing_non_scoring_fields():
    _, diag = records.load_records(AGREEING)
    assert records.summary(diag)["discrepant_field_counts"] == {"injection_withheld": 1, "seconds": 1, "stances": 1}


def test_benchmark_metrics_match_a_hand_calculation(tmp_path):
    """3 claims per class. Hand calculation:
    SUPPORTED: tp 2, fp 1 (one NEI predicted S), fn 1  -> P 2/3, R 2/3, F1 2/3
    CONTRADICTED: tp 1, fp 0, fn 2                     -> P 1, R 1/3, F1 0.5
    NOT_ENOUGH_INFO: tp 2 (one by failure), fp 3, fn 1 -> P 2/5, R 2/3, F1 0.5
    macro-F1 = (2/3 + 0.5 + 0.5)/3 = 0.5556; accuracy 5/9 = 0.5556; false support (C or NEI -> S) 1/6."""
    gold = {"s1": S, "s2": S, "s3": S, "c1": C, "c2": C, "c3": C, "n1": N, "n2": N, "n3": N}
    verdicts = {"s1": S, "s2": S, "s3": "INSUFFICIENT", "c1": C, "c2": "INSUFFICIENT", "c3": "INSUFFICIENT",
                "n1": S, "n2": "INSUFFICIENT", "n3": None}
    rows = [{"claim_id": k, "gold": gold[k], "verdict": v, "sub_state": "SERVICE_ERROR" if v is None else ""}
            for k, v in verdicts.items()]
    claims = write(tmp_path / "claims.jsonl", [{"claim_id": k, "claim": "x", "label": g,
                                                "passages": [{"url": "https://e.org", "text": "t"}]}
                                               for k, g in gold.items()])
    s = alt.score_records(write(tmp_path / "r.jsonl", rows), None, claims)["system"]
    assert s["macro_f1"] == round((2 / 3 + 0.5 + 0.5) / 3, 4) and s["accuracy"] == round(5 / 9, 4)
    assert s["false_support"] == "1/6" and s["missed_contradiction"] == "2/3" and s["nei_decided"] == "1/3"
    assert s["confusion"][N] == {S: 1, C: 0, N: 2} and s["operational_failures"] == {"SERVICE_ERROR": 1}
    assert sum(v for row in s["confusion"].values() for v in row.values()) == 9


def test_the_integrity_code_never_refers_to_restricted_locations():
    for name in ("records.py", "score.py", "alt_benchmarks.py", "run_eval.py"):
        text = (run_eval.ROOT / "evaluation" / "studio_eval" / name).read_text(encoding="utf-8").lower()
        assert "multiagentrag_restricted" not in text and "hmac" not in text, name
