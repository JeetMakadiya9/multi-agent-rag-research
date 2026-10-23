"""Public showcase (showcase/): it renders without a backend or network, publishes no third-party text or secrets,
cites real files, shows metrics that match their result files, and stays lightweight. Offline; no model is used."""
from __future__ import annotations

import ast
import json
import os
import re
import socket
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SHOWCASE = ROOT / "showcase"
sys.path.insert(0, str(SHOWCASE))

import architecture  # noqa: E402
import content  # noqa: E402
import results  # noqa: E402

STDLIB_OK = {"__future__", "ast", "csv", "functools", "html", "json", "pathlib", "re", "sys", "typing", "collections"}
THIRD_PARTY_OK = {"streamlit", "altair", "pandas"}
LOCAL = {"architecture", "content", "pages", "results"}
SECRET = re.compile(r"(sk-[A-Za-z0-9]{16,}|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|Bearer\s+[A-Za-z0-9._-]{12,}|"
                    r"api[_-]?key\s*[:=]\s*['\"][^'\"]{8,})", re.I)
LOCAL_PATH = re.compile(r"([A-Za-z]:[\\/](Users|Projects)|/home/[a-z]|/Users/[A-Za-z]|AppData)")


# ------------------------------------------------------------------------------------------------ offline and light
def test_showcase_code_imports_no_network_or_backend_module():
    for path in SHOWCASE.glob("*.py"):
        if path.name == "make_examples.py":                 # offline maintenance script, not part of the site
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
            for name in names:
                top = name.split(".")[0]
                assert top in STDLIB_OK | THIRD_PARTY_OK | LOCAL, f"{path.name} imports {name}"


def test_showcase_requirements_are_lightweight():
    lines = [l.split("#")[0].strip() for l in (SHOWCASE / "requirements.txt").read_text().splitlines()]
    packages = {re.split(r"[=<>@ ;]", l)[0].lower() for l in lines if l}
    assert packages == {"streamlit", "pyarrow"}


def test_no_service_address_is_required():
    """The site may mention 127.0.0.1 in instructions, but its code never connects anywhere."""
    for path in SHOWCASE.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "urlopen" not in text and "requests." not in text and "socket" not in text


# ------------------------------------------------------------------------------------------------ published data
def example_files():
    return sorted((SHOWCASE / "data" / "examples").glob("*.json"))


def test_examples_exist_and_contain_no_third_party_text():
    files = example_files()
    assert files, "the showcase should bundle at least one recorded example"
    for path in files:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        assert len(raw) < 64_000 and data["kind"] == "recorded output"
        for source in data["sources"]:
            assert not {"text", "pages", "read_url"} & set(source)
            assert str(source.get("url") or "").startswith(("http://", "https://"))
        for claim in data["claims"]:
            for passage in claim["passages"]:
                assert set(passage) == {"source_id", "stance", "page"}            # links only, no passage text
        assert "evidence_pool" not in data and "verifier_note" not in raw
        assert "saved_to" not in raw and "127.0.0.1" not in raw and "localhost" not in raw
        assert not SECRET.search(raw) and not LOCAL_PATH.search(raw)


def test_examples_match_their_originals_when_available():
    """When the original run records are available (SHOWCASE_SOURCE_RUNS), every 60-character window of their
    third-party text is absent from the published examples."""
    folder = os.environ.get("SHOWCASE_SOURCE_RUNS")
    if not folder or not Path(folder).is_dir():
        pytest.skip("original run records are not available here (they are not published)")
    for path in example_files():
        original = json.loads((Path(folder) / path.stem / "run.json").read_text(encoding="utf-8"))
        published = path.read_text(encoding="utf-8")
        texts = [s.get("text") or "" for s in original["sources"]]
        texts += [p for s in original["sources"] for p in s.get("pages") or []]
        texts += [p.get("text") or "" for c in original["claims"] for p in c.get("passages") or []]
        for text in texts:
            for start in range(0, max(0, len(text) - 60), 30):
                window = text[start:start + 60]
                assert not window.strip() or window not in published


def test_development_findings_are_aggregates_only():
    raw = (SHOWCASE / "data" / "development_findings.json").read_text(encoding="utf-8")
    data = json.loads(raw)
    assert not re.search(r"\b(ph|pz|pd|pe|dev2?)-\d+\b", raw)                 # no claim identifiers
    assert "cases" not in raw and not SECRET.search(raw) and not LOCAL_PATH.search(raw)
    assert "outputs/" not in raw and ".txt" not in raw and "results.json" not in raw    # no private file paths
    replay = data["quote_attestation_replay"]
    stances = replay["decisive_stances"]
    unattested = stances["SUPPORTS"]["unattested"] + stances["CONTRADICTS"]["unattested"]
    located = stances["SUPPORTS"]["located"] + stances["CONTRADICTS"]["located"]
    assert located + unattested == stances["total"] == 653
    assert unattested == replay["quotes_not_contiguous_but_accepted_by_legacy_check"] == 185
    assert sum(replay["disagreement_reasons"].values()) == 185
    assert sum(replay["hypothetical_verdict_changes"]["transitions"].values()) == 23
    assert sum(replay["hypothetical_verdict_changes"]["against_reference_labels"].values()) == 23


def test_development_findings_match_the_aggregate_findings_when_available():
    """When the replay's aggregate findings document is available (SHOWCASE_REPLAY_FINDINGS), every published count
    appears in it. Only that aggregate document is read; claim-level records are never opened."""
    path = os.environ.get("SHOWCASE_REPLAY_FINDINGS")
    if not path or not Path(path).is_file():
        pytest.skip("the replay's findings document is not in this repository")
    findings = Path(path).read_text(encoding="utf-8")
    replay = json.loads((SHOWCASE / "data" / "development_findings.json").read_text(encoding="utf-8"))["quote_attestation_replay"]
    stances = replay["decisive_stances"]
    expected = [f"{stances['total']} in total", f"**{replay['quotes_not_contiguous_but_accepted_by_legacy_check']} quotes**",
                f"{replay['records_with_an_unattested_decisive_stance']} mapped records",
                f"**{replay['hypothetical_verdict_changes']['count']} records**", f"`{replay['protocol_sha256_prefix']}",
                f"{replay['records']['runs']} runs, {replay['records']['eligible']} eligible records: "
                f"{replay['records']['mapped']} mapped, {replay['records']['excluded']} excluded",
                f"| SUPPORTS | {stances['SUPPORTS']['located']} | 0 | {stances['SUPPORTS']['unattested']} | 0 |",
                f"| CONTRADICTS | {stances['CONTRADICTS']['located']} | 0 | {stances['CONTRADICTS']['unattested']} | 0 |",
                f"`NO_CONTIGUOUS_MATCH` {replay['disagreement_reasons']['NO_CONTIGUOUS_MATCH']}",
                f"**{replay['quotes_with_number_tokens_absent_from_passage']} of the 185 quotes"]
    for text in expected:
        assert text in findings, text


# ------------------------------------------------------------------------------------------------ references
def cited_paths():
    paths = set()
    for c in content.COMPONENTS:
        paths.update(c["files"])
        paths.update(c["tests"])
    paths.update(edge[3] for edge in content.RUNTIME_EDGES)
    pages_source = (SHOWCASE / "pages.py").read_text(encoding="utf-8")
    paths.update(re.findall(r'src\("([^"]+)"', pages_source))
    paths.update(re.findall(r'results\.phase11_path\("([^"]+)"', pages_source) and
                 [results.phase11_path(p) for p in re.findall(r'results\.phase11_path\("([^"]+)"', pages_source)])
    return paths


def test_every_cited_path_exists():
    missing = [p for p in cited_paths() if not (ROOT / p.rstrip("/")).exists()]
    assert not missing, missing


def test_github_links_use_the_branch():
    assert content.source_url("webrag/server.py") == \
        "https://github.com/JeetMakadiya9/multi-agent-rag-research/blob/release/public-candidate-2/webrag/server.py"
    assert content.source_url("evaluation/experiments/results/phase11_final/").endswith(
        "/tree/release/public-candidate-2/evaluation/experiments/results/phase11_final")


# ------------------------------------------------------------------------------------------------ architecture
def test_code_edges_come_from_real_imports():
    edges = {(a, b): evidence for a, b, evidence in architecture.code_edges()}
    for expected in [("orchestrator", "verifier"), ("orchestrator", "agents"), ("docqa_index", "rag_engine"),
                     ("researcher", "network"), ("verifier", "frozen_b"), ("eval_systems", "research_backend")]:
        assert expected in edges, expected
    assert ("dashboard", "server") not in edges          # the dashboard reaches the Studio over HTTP, not by import
    for (a, b), evidence in edges.items():
        for line in evidence:
            path, _, module = line.partition(" imports ")
            assert path in next(c for c in content.COMPONENTS if c["id"] == a)["files"]
            source = (ROOT / path).read_text(encoding="utf-8")
            assert module.split(".")[-1] in source       # the imported name appears in that file


def test_every_runtime_edge_has_its_marker():
    assert len(architecture.runtime_edges()) == len(content.RUNTIME_EDGES)
    ids = set(architecture.nodes())
    for a, b, *_ in content.RUNTIME_EDGES:
        assert a in ids and b in ids


def test_views_and_dot():
    ids = set(architecture.nodes())
    for view in content.VIEWS.values():
        assert view is None or set(view) <= ids
        graph = architecture.dot(view, "orchestrator")
        assert graph.startswith("digraph") and graph.rstrip().endswith("}")


# ------------------------------------------------------------------------------------------------ metrics
def test_benchmark_metrics_match_the_result_files():
    data = results.phase11()
    summary = json.loads((ROOT / results.PHASE11 / "final_results_summary.json").read_text(encoding="utf-8"))
    rows = {r["system"]: r for r in data["accuracy"]}
    assert float(rows["B"]["macro_f1"]) == pytest.approx(summary["classification"]["B"]["macro_f1_present_gold_classes"])
    assert int(rows["B"]["accuracy_num"]) == summary["classification"]["B"]["correct"] == 87
    assert data["stats"]["p_value"] == summary["paired_BC"]["exact_McNemar_two_sided_p"] == 1.0
    assert data["stats"]["significant_at_0.05"] is False
    assert summary["accounting"]["claims"] == 300 and summary["accounting"]["pair_coverage"]["attempt_pairs"] == 900


def test_missing_result_files_give_none(tmp_path, monkeypatch):
    monkeypatch.setattr(results, "ROOT", tmp_path)
    monkeypatch.setattr(results, "DATA", tmp_path)
    assert results.phase11() is None and results.development_findings() is None and results.examples() == []


# ------------------------------------------------------------------------------------------------ install facts
def test_install_facts_match_the_implementation():
    code = "\n".join((ROOT / p).read_text(encoding="utf-8") for p in
                     ("webrag/model_service.py", ".env.example", "webrag/search.py", "webrag/net.py", "README.md"))
    for name, *_ in content.ENV_SETTINGS:
        assert name in code, name
    launch = (ROOT / "dashboard" / "launch.py").read_text(encoding="utf-8")
    for option in content.LAUNCHER_OPTIONS:
        assert option in launch
    assert "dashboard.launch" in (ROOT / "start_studio.bat").read_text(encoding="utf-8")
    pages_source = (SHOWCASE / "pages.py").read_text(encoding="utf-8")
    for needed in ("requirements.txt", "start_studio.bat", ".env.example"):
        assert needed in pages_source and (ROOT / needed).is_file()
    deep = (ROOT / "webrag" / "deep.py").read_text(encoding="utf-8")
    for reason in content.STOP_REASONS:
        assert f'"{reason}"' in deep, reason
    pipeline = deep + (ROOT / "webrag" / "orchestrator.py").read_text(encoding="utf-8")
    for stage in ("model check", "plan", "coverage", "evidence pool", "synthesize", "verify", "summary"):
        assert f'"{stage}' in pipeline or f"'{stage}" in pipeline or f": {stage}" in pipeline, stage


# ------------------------------------------------------------------------------------------------ pages render offline
@pytest.fixture
def no_network(monkeypatch):
    """Refuse every connection that leaves this computer. Loopback stays open: on Windows asyncio builds its internal
    socket pair over 127.0.0.1, and the showcase itself never connects anywhere (checked statically above)."""
    pytest.importorskip("streamlit")                      # the pages need Streamlit; the checks above do not
    attempts = []
    real_connect = socket.socket.connect

    def guarded(self, address, *args, **kwargs):
        host = address[0] if isinstance(address, tuple) else str(address)
        if host not in ("127.0.0.1", "::1", "localhost"):
            attempts.append(host)
            raise AssertionError(f"the showcase tried to connect to {host}")
        return real_connect(self, address, *args, **kwargs)

    def refuse(address, *args, **kwargs):
        attempts.append(address)
        raise AssertionError(f"the showcase tried to connect to {address}")

    monkeypatch.setattr(socket.socket, "connect", guarded)
    monkeypatch.setattr(socket, "create_connection", refuse)
    yield attempts
    assert not attempts, attempts


def render(page: str):
    def script(page_name, folder):
        import sys
        sys.path.insert(0, folder)
        import pages
        getattr(pages, page_name)()
    from streamlit.testing.v1 import AppTest
    return AppTest.from_function(script, args=(page, str(SHOWCASE)), default_timeout=60).run()


@pytest.mark.parametrize("page", ["architecture_page", "workflow", "components", "results_page", "install", "use_cases"])
def test_pages_render_offline(page, no_network):
    app = render(page)
    assert not app.exception, [e.value for e in app.exception]


def test_entry_point_renders_offline(no_network):
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(SHOWCASE / "app.py"), default_timeout=60).run()
    assert not app.exception
    assert app.title[0].value == "Multi-Agent RAG Research Studio"
    assert any("static guide" in c.value.lower() for c in app.sidebar.caption)


def test_recorded_run_view_and_results_text(no_network):
    app = render("workflow")
    app.segmented_control[0].set_value("A recorded run").run()
    assert not app.exception
    assert any("Nothing is running now" in w.value for w in app.warning)
    results_app = render("results_page")
    assert any("Not statistically significant" in e.value for e in results_app.error)


# ------------------------------------------------------------------------------------------------ evidence labelling
def showcase_text() -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in list(SHOWCASE.glob("*.py")) + list(SHOWCASE.rglob("*.json")))


def test_unpublished_and_unsupported_figures_are_absent():
    text = showcase_text()
    for figure in ("0.755", "0.715", "0.081", "0.0813"):
        assert figure not in text, figure
    assert not re.search(r"\b127\s+(targeted\s+)?tests?\b", text, re.I)


def test_disclosure_wording():
    text = showcase_text()
    assert "Not in this repository" not in text
    assert "publicly reproducible" not in text and "claim-level records are public" not in text
    data = json.loads((SHOWCASE / "data" / "development_findings.json").read_text(encoding="utf-8"))
    replay = data["quote_attestation_replay"]
    assert "not distributed publicly" in data["about"]
    assert replay["kind"].startswith("Engineering diagnostic")
    assert "not verified" in replay["hypothetical_verdict_changes"]["assumption"].lower()
    joined = " ".join(replay["does_not_establish"]).lower()
    assert "benchmark accuracy" in joined and "hallucination" in joined and "hypothetical" in joined
    for kind in ("targeted_tests", "full_suite"):
        assert data[kind]["kind"] == "Software test report"


def test_evidence_types_are_complete():
    labels = {label for label, _, _ in content.EVIDENCE_TYPES.values()}
    assert {"Benchmark result", "Engineering diagnostic", "Recorded demonstration", "Historical baseline",
            "Unpublished / not substantiated publicly", "Software test report"} == labels


def test_phase11_dates_come_from_the_run_manifest():
    manifest = json.loads((ROOT / content.PHASE11_RUN_MANIFEST).read_text(encoding="utf-8"))
    start, end = content.PHASE11_RUN_DATES
    assert manifest["created_at_utc"].startswith(start) and manifest["updated_at_utc"].startswith(end)
    assert manifest["run_id"] == json.loads((ROOT / results.PHASE11 / "final_results_summary.json")
                                            .read_text(encoding="utf-8"))["run_id"]


def test_paired_accuracy_matches_the_summary():
    summary = results.phase11()["summary"]
    assert summary["paired_BC"]["B_paired_accuracy"] == pytest.approx(0.48295, abs=1e-4)
    assert summary["paired_BC"]["C_paired_accuracy"] == pytest.approx(0.48295, abs=1e-4)
    assert summary["paired_BC"]["valid_labeled_pairs"] == 176


def page_text(app) -> str:
    parts = [m.value for m in app.markdown] + [c.value for c in app.caption] + [e.value for e in app.error]
    parts += [w.value for w in app.warning] + [i.value for i in app.info] + [s.value for s in app.subheader]
    return "\n".join(parts)


def test_overview_ranks_evidence_and_marks_phase11_historical(no_network):
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(SHOWCASE / "app.py"), default_timeout=60).run()
    assert not app.exception
    text = page_text(app)
    demo, latest, historical = (text.index(marker) for marker in
                                ("What the Research Studio workflow demonstrates", "Latest engineering finding",
                                 "Historical benchmark: Phase 11B"))
    assert demo < latest < historical
    assert "no public benchmark of the Studio exists yet" in text
    assert "is engineering evidence, not a new benchmark" in text
    assert "no statistically significant difference" in text
    assert "not a benchmark" in text and "does not show improved accuracy" in text
    assert "latest benchmark" not in text.lower()


def test_results_page_labels(no_network):
    app = render("results_page")
    assert not app.exception
    text = page_text(app)
    assert "Exploratory engineering diagnostic" in text and "not distributed publicly" in text
    assert "do not establish improved benchmark accuracy or reduced hallucination rates" in text
    assert "it is not an independent verification" in text
    assert "Historical baseline" in text and "Not statistically significant" in text
    assert "does **not** evaluate the current Research Studio pipeline" in text
    assert "do not, by themselves, establish the system's overall accuracy" in text
    assert "not re-run by this page" in text and "not research performance" in text
    assert "Not in this repository" not in text


def test_later_diagnostics_are_real_and_not_benchmarks():
    for item in content.LATER_DIAGNOSTICS:
        assert (ROOT / item["folder"]).is_dir() and (ROOT / item["date_source"]).is_file()
        assert item["date"][:10] in (ROOT / item["date_source"]).read_text(encoding="utf-8")
    phase13 = (ROOT / "evaluation/experiments/results/phase13_improvement/experiment_B3.md").read_text(encoding="utf-8")
    assert "engineering only; not a benchmark" in phase13
    phase12 = (ROOT / "evaluation/experiments/results/phase12_diagnosis/followup_experiment_design.md").read_text(encoding="utf-8")
    assert "No follow-up experiment or intervention was run in Phase 12" in phase12
