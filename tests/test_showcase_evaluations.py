"""The showcase's Evaluations dashboard: run selection rules (showcase/results.py), the registry builder
(showcase/make_evaluation_runs.py) and the page (showcase/pages.py). Offline; no model is used."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SHOWCASE = ROOT / "showcase"
sys.path.insert(0, str(SHOWCASE))

import make_evaluation_runs as builder  # noqa: E402
import results  # noqa: E402
from test_showcase import LOCAL_PATH, SECRET, no_network, page_text  # noqa: E402,F401

REGISTRY = SHOWCASE / "data" / "evaluation_runs.json"


def run(key, status="valid", set_name="retro", sha="aaa", three=0.6, two=0.7, date="2026-10-25T10:00:00",
        phase="Round 3", complete=True, integrity="ok", label=None):
    metrics = None if three is None else {"three_class_macro_f1": three, "two_class_macro_f1": two,
                                          "three_class_ci95": [three - 0.05, three + 0.05],
                                          "two_class_ci95": [two - 0.05, two + 0.05], "two_class_claims": 200}
    return {"key": key, "label": label or key, "phase": phase, "role": "development", "status": status,
            "set": set_name, "set_sha256": sha, "set_size": 300, "complete_set": complete,
            "last_invocation_utc": date, "metrics": metrics,
            "provenance": {"record_integrity": {"status": integrity}}}


# ------------------------------------------------------------------------------------------------ Phase 11 exclusion
def test_phase11_runs_are_excluded_from_every_selection():
    runs = [run("good", three=0.60), run("phase11b", phase="Phase 11B", three=0.99, two=0.99),
            run("p11", phase="phase11", three=0.98), run("evaluation/phase11_final", phase="Historical", three=0.97)]
    data = {"runs": runs}
    assert [r["key"] for r in results.visible_runs(data)] == ["good"]
    for metric in results.METRICS:
        assert [r["key"] for r in results.best_results(runs, metric)] == ["good"]
    assert [r["key"] for r in results.recent_runs(runs)] == ["good"]


def test_diagnostics_of_a_phase11_run_are_excluded_and_other_phases_kept():
    import content
    shown = results.visible_diagnostics(content.LATER_DIAGNOSTICS)
    assert [d["name"] for d in shown] == [d["name"] for d in content.LATER_DIAGNOSTICS if "Phase 11" not in d["name"]]
    assert any("Phase 13" in d["name"] for d in shown)                      # unrelated phases stay
    assert len(content.LATER_DIAGNOSTICS) == len(shown) + 1                  # the data itself is unchanged


def test_no_page_presents_phase11():
    code = (SHOWCASE / "pages.py").read_text(encoding="utf-8")
    assert "results.phase11(" not in code and "PHASE11" not in code and not re.search(r"Phase\s*11", code)
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    assert not any(results.is_excluded(r) for r in registry["runs"])


@pytest.mark.parametrize("page", ["evaluations", "results_page", "components", "workflow", "install", "use_cases"])
def test_rendered_pages_never_show_phase11(page, no_network):
    from test_showcase import render
    app = render(page)
    assert not app.exception, [e.value for e in app.exception]
    text = page_text(app) + " ".join(str(t.label) for t in app.tabs)
    assert not re.search(r"Phase\s*11", text), page


# ------------------------------------------------------------------------------------------------ rankings
def test_best_results_compare_only_the_same_claim_set_and_metric():
    runs = [run("a", sha="s1", three=0.60, two=0.80), run("b", sha="s1", three=0.65, two=0.70),
            run("c", sha="s2", set_name="confirm", three=0.50, two=0.67)]
    three = {r["key"] for r in results.best_results(runs, "three_class_macro_f1")}
    two = {r["key"] for r in results.best_results(runs, "two_class_macro_f1")}
    assert three == {"b", "c"} and two == {"a", "c"}        # each set keeps its own best; the metrics never mix
    with pytest.raises(ValueError):
        results.best_results(runs, "macro_f1")


@pytest.mark.parametrize("bad", [dict(status="rejected"), dict(status="invalid"), dict(status="incomplete"),
                                 dict(status="partial"), dict(status="coverage"), dict(complete=False),
                                 dict(integrity="rejected"), dict(three=None)])
def test_runs_that_are_not_valid_and_complete_are_never_ranked(bad):
    runs = [run("ok", three=0.55), run("bad", **{"three": 0.95, **bad})]
    assert [r["key"] for r in results.best_results(runs, "three_class_macro_f1")] == ["ok"]


def test_recent_runs_keep_failed_and_poor_runs_newest_first():
    runs = [run("old", date="2026-10-23T10:00"), run("failed", status="invalid", three=None, date="2026-10-25T09:00"),
            run("poor", three=0.10, date="2026-10-26T08:00")]
    assert [r["key"] for r in results.recent_runs(runs)] == ["poor", "failed", "old"]


def test_empty_registries_select_nothing():
    assert results.visible_runs(None) == [] and results.visible_runs({}) == []
    assert results.best_results([], "three_class_macro_f1") == [] and results.recent_runs([]) == []


# ------------------------------------------------------------------------------------------------ builder fields
def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def diag(entry, raw="NEUTRAL", grounded=True):
    return {"reply_entry": entry, "studio_check_entry": entry, "raw_stance": raw, "grounded": grounded}


@pytest.fixture
def run_folder(tmp_path, monkeypatch):
    claims = [{"claim_id": f"c{i}", "label": label} for i, label in
              enumerate(["SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO", "NOT_ENOUGH_INFO"])]
    write_jsonl(tmp_path / "sets" / "toy" / "claims.jsonl", claims)
    sha = hashlib.sha256((tmp_path / "sets" / "toy" / "claims.jsonl").read_bytes()).hexdigest()
    records = [
        {"claim_id": "c0", "gold": "SUPPORTED", "verdict": "SUPPORTED", "sub_state": "", "seconds": 10.0, "calls": 2,
         "passage_diagnostics": [diag("valid", "SUPPORTS"), diag("valid")], "reply_identifiers": []},
        {"claim_id": "c1", "gold": "CONTRADICTED", "verdict": "INSUFFICIENT", "sub_state": "NO_DECISIVE_EVIDENCE",
         "seconds": 20.0, "calls": 3, "passage_diagnostics": [diag("valid", "CONTRADICTS", grounded=False),
                                                              diag("missing")], "reply_identifiers": []},
        {"claim_id": "c2", "gold": "NOT_ENOUGH_INFO", "verdict": "INSUFFICIENT", "sub_state": "SERVICE_ERROR",
         "error": "timeout: no response within 180s", "seconds": 7212.0, "calls": 1, "parse_ok": False},
        {"claim_id": "c3", "gold": "NOT_ENOUGH_INFO", "verdict": "CONTRADICTED", "sub_state": "", "seconds": 30.0,
         "calls": 4, "passage_diagnostics": [diag("valid", "CONTRADICTS")], "reply_identifiers": []},
    ]
    for key in ("toy/valid_run", "toy/coverage_run"):
        folder = tmp_path / key
        write_jsonl(folder / "records.jsonl", records)
        (folder / "manifest_1.json").write_text(json.dumps({"claims_sha256": sha, "started_utc": "2026-10-26T01:00:00",
                                                            "git_head": "abc1234", "model": "qwen3:4b",
                                                            "options": {"verifier": "arm_c_studio"}}), encoding="utf-8")
    (tmp_path / "toy" / "valid_run.log").write_text('Ollama {"version":"0.40.11"}\nGUARD STOPPED THE RUN: free memory '
                                                    "fell below 0.8 GB (0.7 GB)\n", encoding="utf-8")
    monkeypatch.setattr(builder, "RUNS", {
        "toy/valid_run": ("candidate", "Round 9", "development", "plan.md", "valid", "note"),
        "toy/coverage_run": ("candidate", "Round 9", "coverage check", "plan.md", "coverage", "note")})
    monkeypatch.setattr(builder, "PAIRED", [])
    return tmp_path


def test_builder_computes_each_figure_from_its_field(run_folder):
    data = builder.build(run_folder)
    valid, coverage = data["runs"]
    e, x, c = valid["efficiency"], valid["reliability"], valid["coverage"]
    assert valid["set"] == "toy" and valid["complete_set"] is True and valid["set_size"] == 4
    assert e["seconds_per_claim_median"] == 25.0 and e["calls_per_claim_mean"] == 2.5     # median resists the outlier
    assert e["seconds_total"] == 7272.0 and e["tokens"] is None and e["gpu_memory"] is None
    assert x["failure_kinds"] == {"timeout": 1} and x["timeouts"] == 1
    assert x["infrastructure_failures"] == 0 and x["model_output_failures"] == 0
    assert x["memory_stops"] == 1 and valid["provenance"]["ollama_version"] == "0.40.11"
    assert c["passages_presented"] == 5 and c["valid_matched"] == 4 and c["silent_neutral"] == 1
    assert c["quotes_demoted"] == 1
    m = valid["metrics"]
    assert m["false_support"] == "0/3" and m["nei_decided"] == "1/2"
    assert m["confusion"]["NOT_ENOUGH_INFO"]["CONTRADICTED"] == 1                   # c3
    assert m["confusion"]["NOT_ENOUGH_INFO"]["NOT_ENOUGH_INFO"] == 1                # the failure counts as NEI
    assert coverage["metrics"] is None                                              # accuracy never computed
    p = valid["provenance"]
    assert p["model_digest"] is None and p["context_length"] is None and p["run_id"] is None   # never invented
    assert p["record_integrity"]["status"] == "ok"


def test_builder_applies_the_duplicate_policy(run_folder):
    path = run_folder / "toy" / "valid_run" / "records.jsonl"
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    conflicting = dict(rows[0], verdict="CONTRADICTED")
    write_jsonl(path, rows + [conflicting])
    assert builder.integrity(path, None)["status"] == "rejected"
    data = builder.build(run_folder)
    assert data["runs"][0]["provenance"]["record_integrity"]["status"] == "rejected"
    assert not results.rankable(data["runs"][0])


def test_builder_skips_runs_without_records(run_folder, monkeypatch):
    monkeypatch.setattr(builder, "RUNS", {"toy/missing": ("x", "Round 9", "development", "p", "valid", "")})
    assert builder.build(run_folder)["runs"] == []


# ------------------------------------------------------------------------------------------------ bundled registry
def test_bundled_registry_is_aggregates_only():
    raw = REGISTRY.read_text(encoding="utf-8")
    data = json.loads(raw)
    assert not re.search(r"(retro|prog|adv|confirm)-(train|negate|swap)-\d+|test-dev-\d+", raw)   # no claim ids
    for key in ('"claim":', '"claim_id":', '"passages":', '"text":', '"quote":', '"quotes":', '"gold":', '"verdict":'):
        assert key not in raw, key
    assert not SECRET.search(raw) and not LOCAL_PATH.search(raw) and "outputs/" not in raw
    assert len(raw) < 128_000 and data["target"]["metric"] == "three_class_macro_f1"
    coverage = [r for r in data["runs"] if r["status"] == "coverage"]
    assert coverage and all(r["metrics"] is None for r in coverage)
    assert all(r["provenance"]["model_digest"] is None for r in data["runs"])        # not recorded anywhere


def test_bundled_registry_target_is_not_met():
    data = json.loads(REGISTRY.read_text(encoding="utf-8"))
    best = results.best_results(results.visible_runs(data), "three_class_macro_f1")
    assert best and all(r["metrics"]["three_class_macro_f1"] < data["target"]["value"] for r in best)


# ------------------------------------------------------------------------------------------------ the page
def test_evaluations_page_separates_the_metrics_and_keeps_every_run(no_network):
    from test_showcase import render
    app = render("evaluations")
    assert not app.exception
    text = page_text(app).replace("\\", "")
    metrics = {m.label: m.value for m in app.metric}
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    assert metrics["Three-class target"] == str(registry["target"]["value"])
    assert "two-class and three-class scores are never ranked together" in text
    assert "coverage is not correctness" in text
    labels = " ".join(str(t.label) for t in app.tabs)
    for tab in ("Best results", "All runs", "Quality", "Operations", "Coverage and provenance"):
        assert tab in labels, tab
    frames = " ".join(df.value.to_string() for df in app.dataframe)
    assert "not recorded" in frames                                                    # missing metadata is shown
    assert "invalid" in frames and "incomplete" in frames and "rejected" in frames     # failures stay visible


def test_results_page_explains_the_two_metrics(no_network):
    from test_showcase import render
    app = render("results_page")
    assert not app.exception
    text = page_text(app).replace("\\", "")
    assert "Two-class" in text and "Three-class" in text and "never ranked against each other" in text
    assert "next milestone" in text


def test_every_table_on_the_page_converts_to_arrow_without_fixes(no_network):
    """Mixed numbers and text in one column make Streamlit fall back to automatic type fixes; every table must convert
    as it is."""
    pa = pytest.importorskip("pyarrow")
    import streamlit
    from test_showcase import render
    passed, real = [], streamlit.dataframe

    def capture(data, *args, **kwargs):                 # the frame as the page built it, before Streamlit's fixes
        passed.append(data)
        return real(data, *args, **kwargs)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(streamlit, "dataframe", capture)
    try:
        app = render("evaluations")
    finally:
        monkeypatch.undo()
    assert not app.exception and passed
    for frame in passed:
        pa.Table.from_pandas(frame)


def test_evaluations_page_with_no_registry_shows_an_empty_state(tmp_path, monkeypatch, no_network):
    from test_showcase import render
    monkeypatch.setattr(results, "DATA", tmp_path)          # restored after the test; the page shares this module
    app = render("evaluations")
    assert not app.exception
    assert any("No evaluation run is bundled" in i.value for i in app.info)
