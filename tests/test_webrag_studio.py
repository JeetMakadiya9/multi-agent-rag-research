"""Research Studio back end: saved runs, source context, recorded evaluations, the document library and the new
server routes. Every record below is SYNTHETIC TEST DATA written into temporary folders; no model is called."""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from docqa.index import Evidence
from docqa.pipeline import CheckedClaim, ClaimCheck, QAResult
from docqa.verify import Verdict
from webrag import server as server_module
from webrag import studio
from webrag.server import JobQueue, StatusProbe, make_handler
from webrag.studio import (EvaluationStore, Library, RunStore, StudioError, folder_fingerprint, light_run,
                           locate_passage, source_context)

SOURCE_TEXT = ("Intro line.\nThe bridge opened in 1932 and carries road traffic.\nIts main span is 503 metres long, "
               "incor- porating a steel arch.\nClosing line.")
SAMPLE_RUN = {          # synthetic: shaped like Orchestrator.run output, values invented for the test
    "input": "When did the bridge open?", "mode": "research", "status": "COMPLETE", "outcome": "ANSWERED",
    "started": "2026-10-18T10:00:00Z", "finished": "2026-10-18T10:03:00Z", "stages": [],
    "sources": [{"id": "S1", "title": "Bridge (synthetic)", "url": "https://example.org/bridge", "provider": "wikipedia",
                 "kind": "encyclopedia", "access": "FULL_TEXT", "text": SOURCE_TEXT, "pages": None, "extra": {"k": "v"}}],
    "evidence_pool": [{"key": "S1:p1:aaaa", "source_id": "S1", "page": 1, "text": "The bridge opened in 1932",
                       "score": 0.9, "url": "https://example.org/bridge", "title": "Bridge (synthetic)", "access": "FULL_TEXT"}],
    "claims": [{"text": "The bridge opened in 1932.", "role": "finding", "label": "SUPPORTED", "sub_state": "",
                "passages": [{"key": "S1:p1:bbbb", "source_id": "S1", "page": 1, "stance": "SUPPORTS",
                              "text": "main span is 503 metres long, incorporating a steel arch.", "score": 0.7}],
                "cited_sources": ["S1"], "supporting_sources": ["S1"], "contradicting_sources": []},
               {"text": "It is the longest bridge.", "role": "finding", "label": "INSUFFICIENT",
                "sub_state": "NO_DECISIVE_EVIDENCE", "passages": []}],
    "error": "",
}


def write_run(folder, name, data):
    path = folder / name
    path.mkdir(parents=True)
    (path / "run.json").write_text(json.dumps(data), encoding="utf-8")


# ================================================================================================= runs
def test_run_store_lists_newest_first_and_reports_unreadable_records(tmp_path):
    write_run(tmp_path, "20261018T100000000000Z-bridge", SAMPLE_RUN)
    write_run(tmp_path, "20261019T100000000000Z-verify", dict(SAMPLE_RUN, mode="verify", outcome="SUPPORTED"))
    (tmp_path / "20261020T100000000000Z-broken").mkdir()
    (tmp_path / "20261020T100000000000Z-broken" / "run.json").write_text("{not json", encoding="utf-8")
    (tmp_path / ".hidden").mkdir()
    runs = RunStore(tmp_path).list()
    assert [r["name"] for r in runs] == ["20261020T100000000000Z-broken", "20261019T100000000000Z-verify",
                                         "20261018T100000000000Z-bridge"]
    assert runs[0]["status"] == "UNREADABLE" and "could not be read" in runs[0]["error"]
    bridge = runs[2]
    assert bridge["claims"] == 2 and bridge["supported"] == 1 and bridge["insufficient"] == 1 and bridge["sources"] == 1


def test_run_store_get_refuses_unknown_and_traversal_names(tmp_path):
    write_run(tmp_path, "run-a", SAMPLE_RUN)
    store = RunStore(tmp_path)
    assert store.get("run-a")["input"] == SAMPLE_RUN["input"]
    for name in ("run-b", "../run-a", "..", "run-a/../run-a", "", "a" * 200):
        with pytest.raises(StudioError) as caught:
            store.get(name)
        assert caught.value.status == 404


def test_light_run_drops_source_text_without_touching_the_original():
    light = light_run(SAMPLE_RUN)
    source = light["sources"][0]
    assert "text" not in source and "pages" not in source and "extra" not in source
    assert source["has_text"] is True and source["page_count"] == 1
    assert SAMPLE_RUN["sources"][0]["text"] == SOURCE_TEXT          # the input is not modified


# ================================================================================================= context
def test_locate_passage_ignores_whitespace_and_line_break_hyphenation():
    assert locate_passage("abc def", "abc def") == (0, 7)
    start, end = locate_passage(SOURCE_TEXT, "The bridge opened   in 1932\nand carries")
    assert SOURCE_TEXT[start:end] == "The bridge opened in 1932 and carries"
    start, end = locate_passage(SOURCE_TEXT, "503 metres long, incorporating a steel arch.")
    assert SOURCE_TEXT[start:end] == "503 metres long, incor- porating a steel arch."
    assert locate_passage(SOURCE_TEXT, "a passage that is not there") is None
    assert locate_passage("", "x") is None and locate_passage("text", "   ") is None
    assert locate_passage("well-known fact", "wellknown fact") is None       # a real hyphen is not a line break


def test_source_context_returns_the_surrounding_text():
    context = source_context(SAMPLE_RUN, "S1", "S1:p1:bbbb", width=20)
    assert context["located"] is True
    assert context["match"] == "main span is 503 metres long, incor- porating a steel arch."
    assert context["before"].endswith("Its ") and context["truncated_before"] is True
    assert context["source"]["title"] == "Bridge (synthetic)" and "text" not in context["source"]
    missing = dict(SAMPLE_RUN, evidence_pool=[dict(SAMPLE_RUN["evidence_pool"][0], text="Not in the source at all")])
    assert source_context(missing, "S1", "S1:p1:aaaa")["located"] is False


@pytest.mark.parametrize("source, key", [("S9", "S1:p1:aaaa"), ("S1", "S1:p1:zzzz")])
def test_source_context_refuses_unknown_sources_and_passages(source, key):
    with pytest.raises(StudioError) as caught:
        source_context(SAMPLE_RUN, source, key)
    assert caught.value.status == 404


# ================================================================================================= evaluations
def make_eval_folder(root):
    runs = root / "runs"
    good = runs / "eval-good"
    good.mkdir(parents=True)
    (good / "manifest.json").write_text(json.dumps({
        "run_name": "eval-good", "dataset": "evaluation/webrag/claims_x.jsonl", "dataset_sha256": "aa",
        "snapshot": "snap", "snapshot_sha256": "ss", "verifier": "quote_stance", "model": {"model": "m"},
        "started": "2026-10-18T10:00:00+0530", "finished": "2026-10-18T10:05:00+0530",
        "command": "D:\\private\\path python -m webrag.evaluate", "code": {"x.py": "hash"}}), encoding="utf-8")
    (good / "metrics.json").write_text(json.dumps({"n": 2, "classification": {"accuracy": {"value": 0.5, "num": 1, "den": 2}}}), encoding="utf-8")
    (good / "summary.txt").write_text("SYNTHETIC SUMMARY", encoding="utf-8")
    partial = runs / "eval-partial"
    partial.mkdir()
    (partial / "manifest.json").write_text(json.dumps({"dataset": "evaluation/webrag/claims_x.jsonl", "dataset_sha256": "bb"}), encoding="utf-8")
    (partial / "INCOMPLETE.md").write_text("Stopped early (synthetic).", encoding="utf-8")
    (runs / "no-manifest").mkdir()
    (root / "FROZEN.json").write_text(json.dumps({"datasets": {"evaluation/webrag/claims_x.jsonl": "aa"}}), encoding="utf-8")
    (root / "README.md").write_text("# Method (synthetic)\n", encoding="utf-8")
    (root / "DECISION_RULE_V4.md").write_text("# Rule four\n", encoding="utf-8")
    (root / "NOTES.md").write_text("not a method document", encoding="utf-8")
    return root


def test_evaluation_store_reports_recorded_runs_without_paths_or_code(tmp_path):
    store = EvaluationStore(make_eval_folder(tmp_path))
    payload = store.list()
    assert payload["available"] is True
    by_name = {r["name"]: r for r in payload["runs"]}
    assert set(by_name) == {"eval-good", "eval-partial"}
    good, partial = by_name["eval-good"], by_name["eval-partial"]
    assert good["kind"] == "claim verification" and good["complete"] is True and good["dataset_matches_frozen"] is True
    assert good["verifiers"] == ["quote_stance"] and "summary" not in good
    assert "command" not in good and "code" not in good
    assert partial["complete"] is False and partial["dataset_matches_frozen"] is False
    assert partial["incomplete_note"].startswith("Stopped early")
    assert store.get("eval-good")["summary"] == "SYNTHETIC SUMMARY"
    assert [d["name"] for d in payload["documents"]] == ["DECISION_RULE_V4.md", "README.md"]
    assert store.document("README.md")["text"].startswith("# Method")
    for bad in ("NOTES.md", "../README.md", "README.txt"):
        with pytest.raises(StudioError):
            store.document(bad)
    with pytest.raises(StudioError):
        store.get("no-manifest")


def test_evaluation_store_without_runs_is_unavailable(tmp_path):
    assert EvaluationStore(tmp_path).list() == {"available": False, "runs": [], "documents": []}


# ================================================================================================= library
def test_library_lists_ingestion_status_and_hides_temporary_files(tmp_path):
    (tmp_path / "notes.txt").write_text("Page one.\fPage two.", encoding="utf-8")
    (tmp_path / "broken.pdf").write_bytes(b"%PDF-1.4 not really a pdf")
    (tmp_path / "slides.md").write_text("# not supported", encoding="utf-8")
    (tmp_path / ".upload-x.part").write_bytes(b"partial")
    listing = Library(tmp_path).listing()
    docs = {d["name"]: d for d in listing["documents"]}
    assert set(docs) == {"notes.txt", "broken.pdf", "slides.md"}
    assert docs["notes.txt"]["readable"] is True and docs["notes.txt"]["pages"] == 2
    assert docs["broken.pdf"]["readable"] is False and docs["broken.pdf"]["skipped_reason"]
    assert docs["slides.md"]["skipped_reason"] == "unsupported file type"
    assert listing["index"]["state"] == "not_built"


def test_library_listing_for_a_missing_folder(tmp_path):
    listing = Library(tmp_path / "absent").listing()
    assert listing["exists"] is False and listing["documents"] == []


@pytest.mark.parametrize("name, data, status", [
    ("../evil.txt", b"x", None),                       # path components are dropped: saved as evil.txt
    (".hidden.txt", b"x", 400), ("report.docx", b"x", 415), ("empty.txt", b"", 400),
    ("fake.pdf", b"hello", 415), ("latin.txt", "caf\u00e9".encode("latin-1"), 415), ("a<b>.txt", b"x", 400),
])
def test_upload_validation(tmp_path, name, data, status):
    library = Library(tmp_path)
    if status is None:
        assert library.save_upload(name, data) == "evil.txt"
        assert (tmp_path / "evil.txt").read_bytes() == b"x" and not (tmp_path.parent / "evil.txt").exists()
        return
    with pytest.raises(StudioError) as caught:
        library.save_upload(name, data)
    assert caught.value.status == status
    assert [p.name for p in tmp_path.iterdir()] == []


def test_upload_never_overwrites_and_respects_the_size_limit(tmp_path, monkeypatch):
    library = Library(tmp_path)
    assert library.save_upload("paper.pdf", b"%PDF-1.4 synthetic") == "paper.pdf"
    with pytest.raises(StudioError) as caught:
        library.save_upload("paper.pdf", b"%PDF-1.4 other")
    assert caught.value.status == 409 and (tmp_path / "paper.pdf").read_bytes() == b"%PDF-1.4 synthetic"
    monkeypatch.setattr(studio, "MAX_UPLOAD_BYTES", 4)
    with pytest.raises(StudioError) as caught:
        library.save_upload("big.txt", b"12345")
    assert caught.value.status == 413


class FakeReport:
    documents = ["notes.txt"]
    pages = ["p1", "p2"]
    skipped = []


class FakeAssistant:
    """Returns docqa's own result types with SYNTHETIC content."""

    def ask(self, question):
        evidence = [Evidence("E1", "notes.txt", 1, "The bridge opened in 1932.", 0.8)]
        verdict = Verdict("SUPPORTED", stances={"E1": "SUPPORTS"})
        return QAResult(question, evidence=evidence, answer="It opened in 1932 [E1].", status="VERIFIED",
                        claims=[CheckedClaim("It opened in 1932.", ["E1"], verdict)])

    def verify_claim(self, claim):
        evidence = [Evidence("E1", "notes.txt", 2, "It was closed in 1990.", 0.5)]
        return ClaimCheck(claim, evidence=evidence, verdict=Verdict("INSUFFICIENT", "NO_DECISIVE_EVIDENCE", stances={"E1": "NEUTRAL"}))


def test_document_jobs_use_the_assistant_and_rebuild_only_when_documents_change(tmp_path):
    (tmp_path / "notes.txt").write_text("text", encoding="utf-8")
    calls = []

    def factory(folder, log):
        calls.append(folder)
        return FakeAssistant(), FakeReport()

    library = Library(tmp_path, assistant_factory=factory)
    updates = []
    asked = library.run("documents", "When did it open?", updates.append)
    assert asked["status"] == "COMPLETE" and asked["outcome"] == "VERIFIED"
    assert asked["result"]["claims"][0]["verdict"]["label"] == "SUPPORTED"
    assert asked["result"]["evidence"][0]["citation"] == "notes.txt, p. 1"
    assert [s["name"] for s in asked["stages"]] == ["load and index documents", "answer and verify"]
    assert updates and updates[-1]["status"] == "COMPLETE"
    checked = library.run("documents-verify", "It closed in 1990.", lambda d: None)
    assert checked["outcome"] == "INSUFFICIENT" and checked["result"]["verdict"]["sub_state"] == "NO_DECISIVE_EVIDENCE"
    assert len(calls) == 1                                   # same documents: the index is reused
    assert library.listing()["index"]["state"] == "ready"
    (tmp_path / "more.txt").write_text("new", encoding="utf-8")
    assert library.listing()["index"]["state"] == "stale"
    indexed = library.run("documents-index", "", lambda d: None)
    assert indexed["outcome"] == "INDEXED" and len(calls) == 2


def test_document_job_reports_build_assistant_refusals(tmp_path):
    def factory(folder, log):
        raise SystemExit("No readable TXT or PDF documents in the folder.")

    library = Library(tmp_path, assistant_factory=factory)
    data = library.run("documents", "Q?", lambda d: None)
    assert data["status"] == "FAILED" and "No readable" in data["error"]
    assert library.index_state["state"] == "failed"


def test_folder_fingerprint_changes_with_content(tmp_path):
    before = folder_fingerprint(tmp_path)
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    assert folder_fingerprint(tmp_path) != before


# ================================================================================================= server
class Instant:
    def run(self, text, mode, progress):
        return dict(SAMPLE_RUN, input=text, mode=mode)


def test_status_probe_caches_and_reports_provider_errors():
    calls = []

    def describe():
        calls.append(1)
        return {"model": "m", "ollama_version": "0.0-test", "model_digest": "digest"}

    def providers():
        raise ValueError("Provider 'brave' was requested but BRAVE_API_KEY is not set.")

    probe = StatusProbe("m", "quote_stance", describe=describe, providers=providers)
    first, second = probe.get(), probe.get()
    assert first == second and len(calls) == 1
    assert first["model_installed"] is True and first["providers"] == [] and "BRAVE_API_KEY" in first["providers_error"]


@pytest.fixture()
def live_server(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    write_run(runs, "20261018T100000000000Z-bridge", SAMPLE_RUN)
    docs = tmp_path / "docs"
    docs.mkdir()
    library = Library(docs, assistant_factory=lambda folder, log: (FakeAssistant(), FakeReport()))
    jobs = JobQueue(Instant, library)
    probe = StatusProbe("m", "quote_stance", describe=lambda: {"ollama_version": None, "model_digest": None},
                        providers=lambda: ["wikipedia", "arxiv"])
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(jobs, 0))
    port = httpd.server_address[1]
    httpd.RequestHandlerClass = make_handler(jobs, port, runs=RunStore(runs),
                                             evaluations=EvaluationStore(make_eval_folder(tmp_path / "eval")), status=probe)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{port}", jobs, docs
    httpd.shutdown()
    httpd.server_close()


def call(url, data=None, headers=None, method=None):
    request = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), exc.read()


def wait_done(jobs, job_id):
    for _ in range(200):
        job = jobs.get(job_id)
        if job["state"] == "done":
            return job
        threading.Event().wait(0.02)
    raise AssertionError("job did not finish")


def test_page_and_static_assets(live_server):
    base, _, _ = live_server
    status, headers, body = call(base + "/")
    assert status == 200 and b"Multi-Agent Research Studio" in body and "default-src 'self'" in headers["Content-Security-Policy"]
    status, headers, _ = call(base + "/static/app.mjs")
    assert status == 200 and headers["Content-Type"].startswith("text/javascript")
    assert call(base + "/static/studio.css")[1]["Content-Type"].startswith("text/css")
    for bad in ("/static/../server.py", "/static/missing.mjs", "/static/APP.MJS", "/static/app.py", "/static/"):
        assert call(base + bad)[0] == 404, bad


def test_status_runs_and_context_routes(live_server):
    base, _, _ = live_server
    status = json.loads(call(base + "/api/status")[2])
    assert status["ollama_version"] is None and status["providers"] == ["wikipedia", "arxiv"] and not status["general_web_search"]
    runs = json.loads(call(base + "/api/runs")[2])["runs"]
    assert [r["name"] for r in runs] == ["20261018T100000000000Z-bridge"]
    detail = json.loads(call(base + "/api/runs/20261018T100000000000Z-bridge")[2])
    assert "text" not in detail["data"]["sources"][0] and detail["data"]["claims"][0]["label"] == "SUPPORTED"
    context = json.loads(call(base + "/api/runs/20261018T100000000000Z-bridge/context?source=S1&passage=S1%3Ap1%3Aaaaa")[2])
    assert context["located"] is True and context["match"] == "The bridge opened in 1932"
    assert call(base + "/api/runs/20261018T100000000000Z-bridge/context?source=S1")[0] == 400
    assert call(base + "/api/runs/unknown-run")[0] == 404
    assert call(base + "/api/runs/..%2Fsecret")[0] == 404


def test_research_job_context_and_overview(live_server):
    base, jobs, _ = live_server
    status, _, body = call(base + "/api/run", json.dumps({"input": "Q?", "mode": "research"}).encode(),
                           {"Content-Type": "application/json"})
    assert status == 202
    job_id = json.loads(body)["id"]
    wait_done(jobs, job_id)
    context = json.loads(call(f"{base}/api/run/{job_id}/context?source=S1&passage=S1%3Ap1%3Abbbb")[2])
    assert context["located"] is True
    overview = json.loads(call(base + "/api/jobs")[2])["jobs"]
    assert overview[0]["id"] == job_id and overview[0]["state"] == "done" and "data" not in overview[0]
    assert call(base + "/api/run", json.dumps({"input": "Q?", "mode": "nonsense"}).encode(), {"Content-Type": "application/json"})[0] == 400


def test_document_jobs_through_the_server(live_server):
    base, jobs, _ = live_server
    headers = {"Content-Type": "application/json"}
    status, _, body = call(base + "/api/run", json.dumps({"input": "When did it open?", "mode": "documents"}).encode(), headers)
    assert status == 202
    job = wait_done(jobs, json.loads(body)["id"])
    assert job["data"]["result"]["answer"].startswith("It opened") and job["data"]["outcome"] == "VERIFIED"
    status, _, body = call(base + "/api/run", json.dumps({"input": "", "mode": "documents-index"}).encode(), headers)
    assert status == 202 and wait_done(jobs, json.loads(body)["id"])["data"]["outcome"] == "INDEXED"


def test_upload_route(live_server, monkeypatch):
    base, _, docs = live_server
    octet = {"Content-Type": "application/octet-stream", "X-Filename": "notes%20one.txt"}
    status, _, body = call(base + "/api/library/upload", b"Hello library", octet)
    assert status == 201 and json.loads(body)["saved"] == "notes one.txt" and (docs / "notes one.txt").exists()
    assert call(base + "/api/library/upload", b"Hello", dict(octet, **{"Content-Type": "text/plain"}))[0] == 415
    assert call(base + "/api/library/upload", b"Hello", dict(octet, Origin="https://evil.example"))[0] == 403
    assert call(base + "/api/library/upload", b"Hello again", octet)[0] == 409
    monkeypatch.setattr(server_module, "MAX_UPLOAD_BYTES", 8)
    assert call(base + "/api/library/upload", b"0123456789", dict(octet, **{"X-Filename": "big.txt"}))[0] == 413
    listing = json.loads(call(base + "/api/library")[2])
    assert [d["name"] for d in listing["documents"]] == ["notes one.txt"]


class Blocking:
    """A research runner that holds each run until the test releases it, so jobs can be observed while queued."""

    def __init__(self):
        self.release = threading.Event()
        self.started = []

    def __call__(self):                      # used as the orchestrator factory
        return self

    def run(self, text, mode, progress):
        self.started.append(text)
        self.release.wait(10)
        return dict(SAMPLE_RUN, input=text, mode=mode)


def wait_state(jobs, job_id, state):
    for _ in range(200):
        if jobs.get(job_id)["state"] == state:
            return
        threading.Event().wait(0.02)
    raise AssertionError(f"job did not reach {state}")


def test_cancel_withdraws_only_queued_jobs_and_they_never_start():
    runner = Blocking()
    jobs = JobQueue(runner)
    first, second, third = jobs.submit("first", "research"), jobs.submit("second", "research"), jobs.submit("third", "verify")
    wait_state(jobs, first, "running")
    assert jobs.get(second)["position"] == 1 and jobs.get(third)["position"] == 2
    assert jobs.cancel(first) == {"id": first, "state": "running", "cancel_requested": True}   # stops at a checkpoint
    assert jobs.cancel(second) == {"id": second, "state": "cancelled", "cancel_requested": False}
    assert jobs.cancel(second)["state"] == "cancelled"                      # repeating it is harmless
    assert jobs.cancel("unknown") is None
    cancelled = jobs.get(second)
    assert cancelled["data"]["status"] == "CANCELLED" and cancelled["position"] is None
    assert jobs.get(third)["position"] == 1                                 # withdrawn jobs no longer count
    runner.release.set()
    assert wait_done(jobs, third)["data"]["input"] == "third"
    assert runner.started == ["first", "third"]                             # the cancelled job never ran
    assert jobs.get(second)["state"] == "cancelled"
    assert jobs.cancel(first)["state"] == "done"                            # it reached no checkpoint: it finished
    overview = {j["id"]: j for j in jobs.overview()}
    assert overview[second]["state"] == "cancelled" and overview[second]["status"] == "CANCELLED"


def test_cancel_route(tmp_path):
    runner = Blocking()
    jobs = JobQueue(runner)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(jobs, 0))
    port = httpd.server_address[1]
    httpd.RequestHandlerClass = make_handler(jobs, port, runs=RunStore(tmp_path), evaluations=EvaluationStore(tmp_path))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    try:
        running, queued = jobs.submit("running", "research"), jobs.submit("queued", "research")
        wait_state(jobs, running, "running")
        status, _, body = call(f"{base}/api/run/{queued}/cancel", b"", method="POST")
        assert status == 200 and json.loads(body) == {"id": queued, "state": "cancelled", "cancel_requested": False,
                                                      "cancelled": True}
        assert json.loads(call(f"{base}/api/run/{queued}")[2])["state"] == "cancelled"
        status, _, body = call(f"{base}/api/run/{running}/cancel", b"", method="POST")
        payload = json.loads(body)
        assert status == 202 and payload["state"] == "running" and payload["cancel_requested"] is True
        assert payload["cancelled"] is False and "Stopping" in payload["message"]
        assert json.loads(call(f"{base}/api/run/{running}")[2])["cancel_requested"] is True
        assert call(f"{base}/api/run/unknown/cancel", b"", method="POST")[0] == 404
        assert call(f"{base}/api/run/{running}/cancel", b"", {"Origin": "https://evil.example"}, method="POST")[0] == 403
        assert call(f"{base}/api/run/{running}/cancel", method="GET")[0] == 404
        runner.release.set()
        wait_done(jobs, running)
        status, _, body = call(f"{base}/api/run/{running}/cancel", b"", method="POST")
        assert status == 409 and json.loads(body)["state"] == "done"
        assert runner.started == ["running"]
    finally:
        runner.release.set()
        httpd.shutdown()
        httpd.server_close()


class Stepping:
    """A research runner that reports a stage, then waits for the test before each next stage, like the pipeline
    reporting progress between model calls."""

    STAGES = ("model check", "plan", "search and fetch", "index", "verify")

    def __init__(self):
        self.go = threading.Event()
        self.free = False                       # True: run the remaining stages without waiting
        self.reached = []

    def __call__(self):
        return self

    def run(self, text, mode, progress):
        data = {"input": text, "mode": mode, "status": "RUNNING", "stages": []}
        for name in self.STAGES:
            if not self.free:
                self.go.wait(10)
                self.go.clear()
            self.reached.append(name)
            data["stages"].append({"name": name, "status": "ok", "seconds": 0.0, "detail": ""})
            data["current"] = f"{name}: working"
            progress(data)
        return dict(SAMPLE_RUN, input=text, mode=mode)


def test_cancelling_a_running_job_stops_it_at_the_next_checkpoint_and_the_queue_moves_on():
    runner = Stepping()
    jobs = JobQueue(runner)
    running, after = jobs.submit("running", "research"), jobs.submit("after", "research")
    wait_state(jobs, running, "running")
    runner.go.set()                                                        # model check
    for _ in range(200):
        if runner.reached == ["model check"] and jobs.get(running)["data"]["stages"]:
            break
        threading.Event().wait(0.02)
    assert jobs.cancel(running) == {"id": running, "state": "running", "cancel_requested": True}
    runner.go.set()                                                        # plan finishes, then the job stops
    wait_state(jobs, running, "cancelled")
    job = jobs.get(running)
    assert runner.reached[:2] == ["model check", "plan"] and "search and fetch" not in runner.reached[:2]
    assert job["interrupted"] is True and job["data"]["status"] == "CANCELLED" and job["data"]["outcome"] is None
    assert [s["name"] for s in job["data"]["stages"]] == ["model check", "plan"] and "current" not in job["data"]
    assert "saved_to" not in job["data"]
    runner.free = True                                                     # the next job still runs to the end
    runner.go.set()
    assert wait_done(jobs, after)["data"]["input"] == "after"
    assert jobs.cancel(running)["state"] == "cancelled"                    # repeating it is harmless


# ================================================================================================= stage 2: library, integrations, export
def test_run_summaries_record_provider_status_and_sources_merge_across_runs(tmp_path):
    first = dict(SAMPLE_RUN, provider_status={"wikipedia": {"queries": 3, "succeeded": 3, "failed": 0, "hits": 9, "failures": {}}},
                 sources=[dict(SAMPLE_RUN["sources"][0], key="example.org/bridge", access="ABSTRACT_ONLY")])
    second = dict(SAMPLE_RUN, input="Is the bridge long?", claims=[],
                  sources=[dict(SAMPLE_RUN["sources"][0], key="example.org/bridge", access="FULL_TEXT"),
                           {"id": "S2", "title": "Other (synthetic)", "url": "https://example.org/other", "key": "example.org/other",
                            "kind": "web", "provider": "searxng", "access": "FAILED"}])
    write_run(tmp_path, "20261018T100000000000Z-a", first)
    write_run(tmp_path, "20261019T100000000000Z-b", second)
    store = RunStore(tmp_path)
    summaries = {r["name"]: r for r in store.list()}
    assert summaries["20261018T100000000000Z-a"]["providers"] == {"wikipedia": {"queries": 3, "succeeded": 3, "failed": 0, "hits": 9}}
    assert summaries["20261019T100000000000Z-b"]["providers"] == {}
    merged = {s["key"]: s for s in store.sources()}
    bridge = merged["example.org/bridge"]
    assert bridge["access"] == "FULL_TEXT" and len(bridge["runs"]) == 2 and bridge["used_in"] == 1    # cited only in the first run
    assert [s["key"] for s in store.sources()][0] == "example.org/bridge"                              # used sources first
    assert merged["example.org/other"]["used_in"] == 0 and merged["example.org/other"]["access"] == "FAILED"


def test_integrations_report_configuration_and_never_secret_values(monkeypatch):
    from webrag.integrations import ObsidianVault, describe
    for name in ("SEARXNG_URL", "OPENALEX_API_KEY", "WEBRAG_PROVIDERS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("BRAVE_API_KEY", "synthetic-secret-value-123")
    status = {"model": "m", "verifier": "quote_stance", "document_verifier": "frozen_b", "ollama_version": None,
              "model_installed": False, "models": [], "providers": ["wikipedia", "openalex", "arxiv"], "providers_error": ""}
    runs = [{"name": "r1", "started": "2026-10-18T10:00:00Z", "providers": {"wikipedia": {"queries": 4, "succeeded": 4, "failed": 0, "hits": 12}}}]
    payload = describe(status, runs, {"exists": True, "folder": "data/documents", "documents": [{"readable": True}]}, ObsidianVault(None))
    assert "synthetic-secret-value-123" not in json.dumps(payload)
    items = {i["id"]: i for i in payload["integrations"]}
    assert items["ollama"]["status"]["state"] == "unavailable"
    assert items["cloud-models"]["status"]["state"] == "not_supported"
    assert items["wikipedia"]["status"]["state"] == "active" and items["wikipedia"]["last_recorded"]["hits"] == 12
    assert items["searxng"]["status"]["state"] == "not_configured" and items["searxng"]["facts"] == {"SEARXNG_URL": "not set"}
    assert items["brave"]["facts"] == {"BRAVE_API_KEY (secret)": "set"} and items["brave"]["status"]["state"] == "not_configured"
    assert items["documents"]["status"]["state"] == "ready" and items["obsidian"]["status"]["state"] == "not_configured"
    ready = describe(dict(status, ollama_version="0.0-test", model_installed=True, models=["m"]), [], None, ObsidianVault(None))
    ollama = next(i for i in ready["integrations"] if i["id"] == "ollama")
    assert ollama["status"]["state"] == "ready" and ollama["facts"]["Installed models"] == "m"
    broken = describe(dict(status, providers=[], providers_error="ValueError: unknown provider"), [], None, ObsidianVault(None))
    assert next(i for i in broken["integrations"] if i["id"] == "wikipedia")["status"]["state"] == "error"


def test_obsidian_vault_is_validated_and_never_overwritten(tmp_path):
    from webrag.integrations import ObsidianVault
    assert ObsidianVault(None).check()["configured"] is False
    assert "does not exist" in ObsidianVault(str(tmp_path / "missing")).check()["problem"]
    plain = tmp_path / "plain"
    plain.mkdir()
    assert "not an Obsidian vault" in ObsidianVault(str(plain)).check()["problem"]
    with pytest.raises(StudioError) as caught:
        ObsidianVault(str(plain)).export("t", "2026-10-18", "x")
    assert caught.value.status == 409 and not (plain / "Research Studio").exists()
    vault = tmp_path / "vault"
    (vault / ".obsidian").mkdir(parents=True)
    (vault / "My note.md").write_text("mine", encoding="utf-8")
    store = ObsidianVault(str(vault))
    assert store.check()["ok"] is True
    first = store.export('When did the bridge open? "a/b" <c>', "2026-10-18T10:00:00Z", "# Report one\n")
    second = store.export('When did the bridge open? "a/b" <c>', "2026-10-18T10:00:00Z", "# Report two\n")
    assert first == "Research Studio/2026-10-18 When did the bridge open a b c.md"
    assert second == "Research Studio/2026-10-18 When did the bridge open a b c (2).md"
    assert (vault / first).read_text(encoding="utf-8") == "# Report one\n"
    assert (vault / "My note.md").read_text(encoding="utf-8") == "mine"                      # other notes untouched
    assert sorted(p.name for p in vault.iterdir()) == [".obsidian", "My note.md", "Research Studio"]


def test_markdown_export_uses_the_report_renderer_with_a_property_header():
    from webrag.integrations import run_markdown
    full = dict(SAMPLE_RUN, sources=[dict(SAMPLE_RUN["sources"][0], key="k", published=None, retrieved_at="2026-10-18T10:00:00Z", note="")],
                claims=[dict(c, cited_sources=c.get("cited_sources", []), supporting_sources=c.get("supporting_sources", []),
                             contradicting_sources=c.get("contradicting_sources", []), independent_support=1, independent_contradiction=0)
                        for c in SAMPLE_RUN["claims"]],
                disagreements=[], gaps=[], limitations=["Only one source (synthetic)."])
    text = run_markdown("run-a", full)
    assert text.startswith('---\ntitle: "When did the bridge open?"\ncreated: "2026-10-18T10:03:00Z"\nmode: "research"')
    assert "# Research report" in text and "## Verified findings" in text and "Only one source (synthetic)." in text
    with pytest.raises(StudioError) as caught:
        run_markdown("bad", {"input": "x"})                                                   # no mode: cannot be rendered
    assert caught.value.status == 422


def make_server(tmp_path, **kwargs):
    runs = tmp_path / "runs"
    full = dict(SAMPLE_RUN, sources=[dict(SAMPLE_RUN["sources"][0], key="k", published=None, retrieved_at="2026-10-18T10:00:00Z", note="")],
                claims=[dict(c, cited_sources=c.get("cited_sources", []), supporting_sources=c.get("supporting_sources", []),
                             contradicting_sources=c.get("contradicting_sources", []), independent_support=1, independent_contradiction=0)
                        for c in SAMPLE_RUN["claims"]], disagreements=[], gaps=[], limitations=[])
    write_run(runs, "20261018T100000000000Z-bridge", full)
    jobs = JobQueue(Instant)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(jobs, 0))
    port = httpd.server_address[1]
    httpd.RequestHandlerClass = make_handler(jobs, port, runs=RunStore(runs), evaluations=EvaluationStore(tmp_path), **kwargs)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{port}", port


def test_export_integration_and_host_routes(tmp_path):
    calls = []

    def describe():
        calls.append(1)
        return {"model": "m", "ollama_version": "0.0-test", "model_digest": "d"}

    probe = StatusProbe("m", "quote_stance", describe=describe, providers=lambda: ["wikipedia"], models=lambda: ["m", "other:1b"])
    vault = tmp_path / "vault"
    (vault / ".obsidian").mkdir(parents=True)
    from webrag.integrations import ObsidianVault
    httpd, base, port = make_server(tmp_path, status=probe, vault=ObsidianVault(str(vault)))
    try:
        status, headers, body = call(base + "/api/runs/20261018T100000000000Z-bridge/export?format=md")
        assert status == 200 and headers["Content-Type"].startswith("text/markdown")
        assert headers["Content-Disposition"] == 'attachment; filename="20261018T100000000000Z-bridge.md"' and body.startswith(b"---\n")
        status, headers, body = call(base + "/api/runs/20261018T100000000000Z-bridge/export?format=json")
        assert status == 200 and json.loads(body)["sources"][0]["text"] == SOURCE_TEXT        # the full record, with source text
        assert call(base + "/api/runs/20261018T100000000000Z-bridge/export?format=xml")[0] == 400
        assert call(base + "/api/runs/..%2Fx/export?format=md")[0] == 404
        payload = json.loads(call(base + "/api/integrations")[2])
        assert [c["id"] for c in payload["categories"]] == ["ai", "sources", "knowledge", "export"]
        assert next(i for i in payload["integrations"] if i["id"] == "obsidian")["status"]["state"] == "ready"
        before = len(calls)
        tested = json.loads(call(base + "/api/integrations/ollama/test", b"", method="POST")[2])
        assert tested["status"]["state"] == "ready" and len(calls) == before + 1                 # a fresh check, not the cache
        assert json.loads(call(base + "/api/status")[2])["models"] == ["m", "other:1b"]
        status, _, body = call(base + "/api/export/obsidian", json.dumps({"run": "20261018T100000000000Z-bridge"}).encode(),
                               {"Content-Type": "application/json"})
        saved = json.loads(body)
        assert status == 201 and saved["vault"] == "vault" and (vault / saved["saved"]).read_text(encoding="utf-8").startswith("---\n")
        assert call(base + "/api/export/obsidian", json.dumps({"run": "missing"}).encode(), {"Content-Type": "application/json"})[0] == 404
        assert call(base + "/api/export/obsidian", b"{}", {"Content-Type": "application/json", "Origin": "https://evil.example"})[0] == 403
        assert json.loads(call(base + "/api/sources")[2])["sources"][0]["key"] == "k"
        status, headers, _ = call(base + "/favicon.ico")
        assert status == 200 and headers["Content-Type"] == "image/svg+xml"
        assert call(base + "/api/status", headers={"Host": f"evil.example:{port}"})[0] == 421          # DNS rebinding
        assert call(base + "/api/run", json.dumps({"input": "Q?"}).encode(), {"Content-Type": "application/json", "Host": "evil.example"})[0] == 421
        assert call(base + "/", headers={"Host": f"localhost:{port}"})[0] == 200
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_obsidian_export_without_a_vault_is_refused(tmp_path):
    httpd, base, _ = make_server(tmp_path)
    try:
        status, _, body = call(base + "/api/export/obsidian", json.dumps({"run": "20261018T100000000000Z-bridge"}).encode(),
                               {"Content-Type": "application/json"})
        assert status == 409 and "No Obsidian vault" in json.loads(body)["error"]
        assert json.loads(call(base + "/api/integrations/obsidian")[2])["status"]["state"] == "not_configured"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_status_probe_with_a_substituted_runtime_never_lists_real_models():
    probe = StatusProbe("m", "quote_stance", describe=lambda: {"ollama_version": "0.0-test", "model_digest": "d"}, providers=lambda: [])
    assert probe.get()["models"] == []


def test_evaluation_routes(live_server):
    base, _, _ = live_server
    payload = json.loads(call(base + "/api/evaluations")[2])
    assert {r["name"] for r in payload["runs"]} == {"eval-good", "eval-partial"}
    assert json.loads(call(base + "/api/evaluations/eval-good")[2])["summary"] == "SYNTHETIC SUMMARY"
    assert json.loads(call(base + "/api/evaluations/docs/README.md")[2])["text"].startswith("# Method")
    assert call(base + "/api/evaluations/docs/NOTES.md")[0] == 404
    assert call(base + "/api/evaluations/missing")[0] == 404
