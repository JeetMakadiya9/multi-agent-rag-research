"""Offline tests for webrag (mocked HTTP, search APIs and model; no network). The live test is separate."""
import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from docqa.llm import LLMError, LLMResponse
from docqa.pdf_utils import write_simple_pdf
from docqa.verify import CONTRADICTED, INSUFFICIENT, SUPPORTED, Verdict
from webrag import dedupe, extract
from webrag.agents import PLAN_PROMPT, Planner, Synthesizer
from webrag.evidence import Passage, merge_passages
from webrag.net import FetchError, Fetched, PoliteSession, _check_url
from webrag.orchestrator import Orchestrator, RunCancelled, Settings
from webrag.report import render_markdown
from webrag.researcher import ABSTRACT_ONLY, FAILED, FULL_TEXT, SNIPPET_ONLY, Source, WebResearcher
from webrag.search import ArxivSearch, OpenAlexSearch, SearchError, SearchHit, WikipediaSearch, keywords
from webrag.server import JobQueue, make_handler


# ============================================================================================ fakes

class FakeSession:
    def __init__(self, routes=()):
        self.routes, self.requests_made, self.calls = list(routes), 0, []

    def get(self, url, *, params=None, headers=None, check_robots=True, max_bytes=None):
        self.requests_made += 1
        self.calls.append({"url": url, "params": params or {}, "check_robots": check_robots})
        for match, result in self.routes:
            if match(url, params or {}):
                if isinstance(result, Exception):
                    raise result
                return result
        raise FetchError("http_404", url)


def ok(body, content_type="application/json", url="https://example.org/x"):
    data = body if isinstance(body, bytes) else (json.dumps(body) if not isinstance(body, str) else body).encode()
    return Fetched(url, 200, content_type, data)


class FakeClient:
    model, base_url = "qwen3:4b", "http://127.0.0.1:11434"

    def __init__(self, plan=None, synthesis=None, describe=None, fail=None):
        self.plan, self.synthesis, self.fail = plan, synthesis, fail
        self.info = describe or {"model": "qwen3:4b", "ollama_version": "0.40.5", "model_digest": "abc"}
        self.history = []

    def describe(self):
        return dict(self.info)

    def chat(self, messages, *, response_format=None, temperature=0.0, num_predict=None, timeout=None):
        if self.fail:
            raise self.fail
        text = self.plan if messages[0]["content"] == PLAN_PROMPT else self.synthesis
        response = LLMResponse(text if isinstance(text, str) else json.dumps(text), "stop", 1, 1, 0.01)
        self.history.append(response)
        return response


class FakeProvider:
    def __init__(self, name, hits, texts=None, kind="encyclopedia", error=None):
        self.name, self.hits, self.texts, self.kind, self.error = name, hits, texts or {}, kind, error

    def search(self, query, limit=3):
        if self.error:
            raise SearchError(self.error)
        return [SearchHit(self.name, query, i, title, url, snippet, kind=self.kind, abstract=abstract)
                for i, (title, url, snippet, abstract) in enumerate(self.hits[:limit], 1)]

    def fetch_text(self, hit):
        if hit.url not in self.texts:
            raise SearchError("article text unavailable")
        return {"text": self.texts[hit.url], "revised": "2026-09-01T00:00:00Z"}


class FakeStore:
    """Sentence-level passages ranked by word overlap (stands in for the rag.py index)."""

    def __init__(self, sources, models=None):
        self.sources = {s.id: s for s in sources}
        self.passages = [Passage(f"{s.id}:p1:{i}", s.id, 1, sentence.strip() + ".", 0.0)
                         for s in sources for i, sentence in enumerate(s.text.split(".")) if sentence.strip()]
        self.size = len(self.passages)

    def rank(self, query, k=6, per_source=2):
        words = set(query.lower().split())
        scored = sorted(self.passages, key=lambda p: -len(words & set(p.text.lower().split())))
        return scored[:k]


class FakeVerifier:
    def __init__(self, labels):
        self.labels, self.calls = labels, []

    def verify(self, claim, evidence):
        self.calls.append((claim, [e.text for e in evidence]))
        label = self.labels.get(claim, INSUFFICIENT)
        if label == "ERROR":
            raise LLMError("timeout", "slow")
        stance = {"SUPPORTED": "SUPPORTS", "CONTRADICTED": "CONTRADICTS"}.get(label, "NEUTRAL")
        if label == "BROKEN":
            return Verdict(INSUFFICIENT, "ASSESSMENT_FAILED", parse_ok=False)
        stances = {e.id: (stance if i == 0 else "NEUTRAL") for i, e in enumerate(evidence)}
        return Verdict(label, stances=stances)


# ============================================================================================ extraction

def test_html_extraction_skips_boilerplate_and_reads_metadata():
    html = """<html><head><title>Ignored</title><meta name="citation_title" content="GNNs for Fraud">
    <meta name="citation_publication_date" content="2023/05/01"><script>var x = "secret";</script></head>
    <body><nav>Home | About | Contact</nav><article><h1>Results</h1><p>The graph model reached an F1 of 0.91 on the
    held-out transactions, compared with 0.84 for gradient boosting.</p><p>Short</p></article>
    <footer>Copyright notice and other footer text that is long enough to keep.</footer></body></html>"""
    out = extract.html_to_text(html)
    assert out.title == "GNNs for Fraud" and out.published == "2023/05/01"
    assert "F1 of 0.91" in out.text and "secret" not in out.text and "Home | About" not in out.text
    assert "Copyright notice" not in out.text


def test_malformed_html_does_not_raise():
    assert "kept text that is long enough to survive." in extract.html_to_text(
        "<div><p>kept text that is long enough to survive.<b></div></span>").text


def test_pdf_extraction(tmp_path):
    write_simple_pdf(tmp_path / "a.pdf", [["Page one text about fraud detection."], ["Page two results."]])
    out = extract.pdf_to_text((tmp_path / "a.pdf").read_bytes())
    assert out.pages[0].startswith("Page one") and "Page two" in out.text


def test_openalex_abstract_reconstruction():
    assert extract.openalex_abstract({"neural": [1], "Graph": [0], "networks.": [2]}) == "Graph neural networks."
    assert extract.openalex_abstract(None) == ""


# ============================================================================================ identity

@pytest.mark.parametrize("a,b", [
    ("http://www.Example.com/path/?utm_source=x&b=2&a=1#frag", "https://example.com/path?a=1&b=2"),
    ("https://en.m.wikipedia.org/wiki/Graph", "https://en.wikipedia.org/wiki/Graph"),
])
def test_normalize_url(a, b):
    assert dedupe.normalize_url(a) == dedupe.normalize_url(b)


def test_canonical_key_unifies_arxiv_and_doi_forms():
    keys = {dedupe.canonical_key("https://arxiv.org/abs/2101.00001v2"),
            dedupe.canonical_key("https://arxiv.org/pdf/2101.00001.pdf"),
            dedupe.canonical_key("https://doi.org/10.48550/arXiv.2101.00001"),
            dedupe.canonical_key("https://openalex.org/W1", doi="https://doi.org/10.48550/arxiv.2101.00001")}
    assert keys == {"arxiv:2101.00001"}
    assert dedupe.canonical_key("https://publisher.org/a", doi="https://doi.org/10.1000/ABC") == "doi:10.1000/abc"
    assert dedupe.canonical_key("https://doi.org/10.1000/abc") == "doi:10.1000/abc"


def test_near_duplicates_are_grouped():
    base = " ".join(f"word{i}" for i in range(60))
    texts = [base, "Syndicated copy. " + base, " ".join(f"other{i}" for i in range(60))]
    assert dedupe.near_duplicate_groups(texts) == [0, 0, 2]


# ============================================================================================ search parsing

def test_wikipedia_search_and_text():
    session = FakeSession([
        (lambda u, p: p.get("list") == "search",
         ok({"query": {"search": [{"title": "Graph neural network", "pageid": 7, "snippet": "A <b>GNN</b> is"}]}})),
        (lambda u, p: p.get("prop") == "extracts|revisions",
         ok({"query": {"pages": {"7": {"extract": "Full article text.", "revisions": [{"timestamp": "2026-01-02"}]}}}})),
    ])
    provider = WikipediaSearch(session)
    hit = provider.search("graph neural network", 3)[0]
    assert (hit.url, hit.snippet, hit.kind) == ("https://en.wikipedia.org/wiki/Graph_neural_network", "A GNN is",
                                                "encyclopedia")
    assert provider.fetch_text(hit) == {"text": "Full article text.", "revised": "2026-01-02"}
    assert all(not c["check_robots"] for c in session.calls)          # documented API, not page scraping


def test_openalex_search_parses_abstract_and_oa_pdf():
    body = {"results": [{"id": "https://openalex.org/W1", "doi": "https://doi.org/10.1/x", "display_name": "GNN fraud",
                         "publication_date": "2022-03-01", "abstract_inverted_index": {"We": [0], "test.": [1]},
                         "best_oa_location": {"pdf_url": "https://repo.org/x.pdf"},
                         "primary_location": {"source": {"display_name": "J. Fraud"}}}]}
    hit = OpenAlexSearch(FakeSession([(lambda u, p: "openalex" in u, ok(body))])).search("gnn fraud")[0]
    assert (hit.abstract, hit.pdf_url, hit.url, hit.extra["venue"]) == ("We test.", "https://repo.org/x.pdf",
                                                                        "https://doi.org/10.1/x", "J. Fraud")


def test_arxiv_search_parses_atom():
    atom = """<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/2101.00001v1</id>
      <title>Graph  Fraud</title><summary>We study fraud.</summary><published>2021-01-01T00:00:00Z</published>
      <link title="pdf" href="http://arxiv.org/pdf/2101.00001v1"/></entry></feed>"""
    session = FakeSession([(lambda u, p: "arxiv" in u, ok(atom, "application/atom+xml"))])
    hit = ArxivSearch(session).search("graph neural networks for fraud detection")[0]
    assert (hit.title, hit.abstract, hit.published, hit.kind) == ("Graph Fraud", "We study fraud.", "2021-01-01",
                                                                  "preprint")
    assert session.calls[0]["params"]["search_query"] == "all:graph AND all:neural AND all:networks AND all:fraud " \
                                                         "AND all:detection"


@pytest.mark.parametrize("result", [FetchError("timeout", "x"), ok("<html>not json", "text/html")])
def test_search_failures_become_search_errors(result):
    with pytest.raises(SearchError):
        OpenAlexSearch(FakeSession([(lambda u, p: True, result)])).search("q")


def test_keywords_drop_stopwords():
    assert keywords("Do graph neural networks outperform the baselines for fraud?") == [
        "graph", "neural", "networks", "outperform", "baselines", "fraud"]


# ============================================================================================ HTTP safety

@pytest.mark.parametrize("url,code", [("ftp://x.org/a", "invalid_url"), ("http://localhost:8080/", "blocked_host"),
                                      ("http://192.168.1.4/admin", "blocked_host"), ("http://127.0.0.1/", "blocked_host")])
def test_unsafe_urls_are_refused(url, code):
    with pytest.raises(FetchError) as info:
        _check_url(url)
    assert info.value.code == code


class FakeResponse:
    def __init__(self, status=200, body=b"", headers=None, url="https://site.org/page"):
        self.status_code, self.body, self.headers, self.url = status, body, headers or {}, url

    def iter_content(self, size):
        yield self.body

    def close(self):
        pass


def public_resolver(host):
    return ["93.184.216.34"]                       # offline stand-in for DNS: every name looks public


def session_with(responses, **kwargs):
    kwargs.setdefault("resolver", public_resolver)
    session = PoliteSession(min_interval=0, **kwargs)
    queue = list(responses)
    session.calls = []

    def fake_get(url, **kw):
        session.calls.append({"url": url, **kw})
        return queue.pop(0)(url) if callable(queue[0]) else queue.pop(0)

    session.session.get = fake_get
    return session


def test_robots_txt_is_obeyed():
    robots = FakeResponse(body=b"User-agent: *\nDisallow: /private", headers={"Content-Type": "text/plain"})
    session = session_with([robots])
    with pytest.raises(FetchError) as info:
        session.get("https://site.org/private/page")
    assert info.value.code == "robots_disallowed"


def test_long_retry_after_pauses_the_host_without_new_requests():
    """Regression: OpenAlex answered 'Retry-After: 43560' and was still queried for every remaining claim."""
    session = session_with([FakeResponse(429, headers={"Retry-After": "43560"})], respect_robots=False)
    with pytest.raises(FetchError, match="retry after 43560"):
        session.get("https://api.example.org/a")
    with pytest.raises(FetchError, match="asked for a pause"):
        session.get("https://api.example.org/b")                     # no response queued: no request was made
    assert session.requests_made == 1


def test_short_retry_after_is_waited_out(monkeypatch):
    """Regression: a 34 s Wikipedia pause made the client skip every Wikipedia request until it ended."""
    import webrag.net as net
    slept = []
    monkeypatch.setattr(net.time, "sleep", lambda s: slept.append(s))
    session = session_with([FakeResponse(429, headers={"Retry-After": "34"}), FakeResponse(body=b"ok")],
                           respect_robots=False)
    assert session.get("https://en.wikipedia.org/w/api.php").content == b"ok"
    assert 34.0 in slept and session.requests_made == 2


def test_openalex_rate_limit_explains_the_fix(monkeypatch):
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    session = FakeSession([(lambda u, p: True, FetchError("rate_limited", "api.openalex.org asked to retry"))])
    with pytest.raises(SearchError, match="OPENALEX_API_KEY"):
        OpenAlexSearch(session).search("q")


def test_size_cap_and_rate_limit():
    session = session_with([FakeResponse(body=b"x" * 50)], respect_robots=False, max_bytes=10)
    with pytest.raises(FetchError, match="too_large"):
        session.get("https://site.org/a")
    limited = session_with([FakeResponse(429, headers={"Retry-After": "0"}), FakeResponse(429)], respect_robots=False)
    with pytest.raises(FetchError, match="rate_limited"):
        limited.get("https://site.org/a")


# ============================================================================================ researcher

def test_researcher_dedupes_and_records_access_levels():
    wiki = FakeProvider("wikipedia", [("GNN", "https://en.wikipedia.org/wiki/GNN", "snippet", "")],
                        texts={"https://en.wikipedia.org/wiki/GNN": "Graph neural networks learn on graphs. " * 5})
    scholarly = FakeProvider("openalex", [("Paper", "https://doi.org/10.48550/arXiv.2101.00001", "", "An abstract."),
                                          ("No content", "https://doi.org/10.1/none", "", "")], kind="scholarly")
    preprints = FakeProvider("arxiv", [("Paper", "https://arxiv.org/abs/2101.00001v1", "", "An abstract.")],
                             kind="preprint")
    corpus = WebResearcher([wiki, scholarly, preprints], FakeSession(), fetch_full_text=False).gather(["q1", "q2"])
    by_title = {s.title: s for s in corpus.sources}
    assert len(corpus.sources) == 3                                   # the arXiv copy merged with the OpenAlex record
    assert by_title["GNN"].access == FULL_TEXT and by_title["Paper"].access == ABSTRACT_ONLY
    assert by_title["Paper"].queries == ["q1", "q2"]
    assert by_title["No content"].access == FAILED and not by_title["No content"].usable


def test_researcher_full_text_and_failed_pdf(tmp_path):
    write_simple_pdf(tmp_path / "p.pdf", [["Full paper text. " * 20] * 8])
    pdf = (tmp_path / "p.pdf").read_bytes()
    session = FakeSession([(lambda u, p: u == "https://repo.org/ok.pdf", ok(pdf, "application/pdf", "https://repo.org/ok.pdf")),
                           (lambda u, p: u == "https://repo.org/bad.pdf", FetchError("http_403", "x"))])
    researcher = WebResearcher([], session, max_full_text=4)
    good = researcher.fetch("S1", SearchHit("openalex", "q", 1, "A", "https://doi.org/10.1/a", kind="scholarly",
                                            abstract="Abstract A.", pdf_url="https://repo.org/ok.pdf"), ["q"])
    bad = researcher.fetch("S2", SearchHit("openalex", "q", 2, "B", "https://doi.org/10.1/b", kind="scholarly",
                                           abstract="Abstract B.", pdf_url="https://repo.org/bad.pdf"), ["q"])
    assert good.access == FULL_TEXT and good.read_url == "https://repo.org/ok.pdf" and good.pages
    assert bad.access == ABSTRACT_ONLY and "http_403" in bad.note


def test_web_page_falls_back_to_snippet():
    session = FakeSession([(lambda u, p: True, FetchError("timeout", "x"))])
    source = WebResearcher([], session).fetch("S1", SearchHit("brave", "q", 1, "T", "https://news.org/a", "A snippet."),
                                               ["q"])
    assert source.access == SNIPPET_ONLY and "timeout" in source.note


# ============================================================================================ agents

def test_planner_validates_and_falls_back():
    plan = Planner(FakeClient(plan={"subquestions": ["What is X?", "what is x?"],
                                    "search_queries": ["x benchmark results", "x limitations"]})).plan("Is X good?",
                                                                                                       "research")
    assert plan.subquestions == ["What is X?"] and plan.queries == ["Is X good?", "x benchmark results",
                                                                     "x limitations"]
    assert Planner(FakeClient(plan="{bad")).plan("Is X good?", "research").fallback
    down = Planner(FakeClient(fail=LLMError("unreachable", "down"))).plan("Is X good?", "research")
    assert down.fallback and down.queries == ["Is X good?"]


def test_perspective_planner_is_opt_in_and_keeps_the_query_limit():
    from webrag.agents import PERSPECTIVE_PROMPT

    class Client(FakeClient):
        def chat(self, messages, **kw):
            assert messages[0]["content"] == PERSPECTIVE_PROMPT
            self.schema = kw["response_format"]
            return LLMResponse(self.plan if isinstance(self.plan, str) else json.dumps(self.plan), "stop", 1, 1, 0.0)

    rows = [{"direction": "supporting", "query": "x benchmark results"},
            {"direction": "challenging", "query": "x failure cases arXiv"},
            {"direction": "context", "query": "x evaluation methodology"},
            {"direction": "context", "query": "x dataset limitations"}]
    client = Client(plan={"subquestions": ["Where does X fail?"], "queries": rows})
    plan = Planner(client, style="perspective").plan("Is X good?", "research")
    assert plan.style == "perspective" and len(plan.queries) == 4                  # same limit as the default
    assert plan.queries == ["Is X good?", "x benchmark results", "x failure cases", "x evaluation methodology"]
    assert plan.directions == ["question", "supporting", "challenging", "context"]
    assert client.schema["properties"]["queries"]["items"]["properties"]["direction"]["enum"] == \
        ["supporting", "challenging", "context"]
    bad = Planner(Client(plan={"subquestions": [], "queries": ["not an object"]}), style="perspective")
    assert bad.plan("Is X good?", "research").fallback
    assert Planner(FakeClient(plan={"subquestions": [], "search_queries": ["x"]})).plan("Is X?", "research") \
        .directions == []                                                           # default planner unchanged
    with pytest.raises(ValueError):
        Planner(FakeClient(), style="storm")
    with pytest.raises(ValueError):
        Settings(planner="storm")
    assert Settings().planner == "default"


@pytest.mark.parametrize("query,expected", [("Guido van Rossum Python development OpenAlex",
                                              "Guido van Rossum Python development"),
                                             ("XG: XGBoost license arXiv", "XG: XGBoost license"),
                                             ("MNIST size on Wikipedia", "MNIST size"),
                                             ("graph fraud detection", "graph fraud detection")])
def test_provider_names_are_stripped_from_queries(query, expected):
    """Regression: the first dev collection sent queries such as '... OpenAlex' to the search APIs."""
    from webrag.agents import strip_provider_names
    assert strip_provider_names(query) == expected


def test_synthesizer_drops_unknown_citations_and_retries():
    passages = [Passage("S1:p1:a", "S1", 1, "X scored 0.9.", 0.5)]
    sources = {"S1": Source("S1", "T", "https://a.org", "wikipedia", "encyclopedia", "url:a", FULL_TEXT, "X scored 0.9.")}
    good = {"findings": [{"text": "X scored 0.9.", "evidence": ["E1", "E7"]}], "disagreements": [], "gaps": [],
            "insufficient": False}
    replies = iter(["{broken", json.dumps(good)])
    client = FakeClient()
    client.chat = lambda messages, **kw: LLMResponse(next(replies), "stop", 1, 1, 0.0)
    result = Synthesizer(client).synthesize("Q?", [], passages, sources)
    assert result.parse_ok and len(result.raw) == 2
    assert [p.key for p in result.findings[0].passages] == ["S1:p1:a"] and result.findings[0].unknown_ids == ["E7"]


def test_synthesizer_v2_hides_source_dates():
    """Regression: v1 showed '2023-...' in passage labels and findings copied the year ('a 2023 study')."""
    from webrag.agents import SYNTHESIS_PROMPT, SYNTHESIS_PROMPT_V2
    passages = [Passage("S1:p1:a", "S1", 1, "GNNs outperform logistic regression on the benchmark.", 0.5)]
    sources = {"S1": Source("S1", "GNN study", "https://a.org", "arxiv", "preprint", "url:a", FULL_TEXT, "x",
                            published="2023-05-01")}
    reply = json.dumps({"findings": [], "disagreements": [], "gaps": [], "insufficient": True})
    seen = {}

    def capture(version):
        client = FakeClient()
        client.chat = lambda messages, **kw: (seen.__setitem__(version, messages), LLMResponse(reply, "stop", 1, 1, 0))[1]
        Synthesizer(client, version=version).synthesize("Q?", [], passages, sources)

    capture(1), capture(2)
    assert "2023-05-01" in seen[1][1]["content"] and seen[1][0]["content"] == SYNTHESIS_PROMPT
    assert "2023" not in seen[2][1]["content"] and seen[2][0]["content"] == SYNTHESIS_PROMPT_V2


def test_merge_passages_dedupes_by_text():
    a, b = Passage("k1", "S1", 1, "same", 0), Passage("k2", "S2", 1, "same", 0)
    assert merge_passages([a], [b, Passage("k3", "S2", 1, "other", 0)], limit=5) == [a, Passage("k3", "S2", 1, "other", 0)]


# ============================================================================================ orchestration

WIKI = FakeProvider("wikipedia", [("Fraud detection", "https://en.wikipedia.org/wiki/Fraud", "s", "")],
                    texts={"https://en.wikipedia.org/wiki/Fraud": "Graph models reached F1 0.91 on transactions. "
                                                                  "Boosting reached F1 0.84 on transactions. "
                                                                  "Graph models are slower to train."})
PLAN = {"subquestions": ["How do graph models compare?"], "search_queries": ["graph fraud detection"]}
SYNTH = {"findings": [{"text": "Graph models reached F1 0.91.", "evidence": ["E1"]},
                      {"text": "Graph models are cheaper to train.", "evidence": ["E9"]}],
         "disagreements": [{"text": "Sources differ on cost.", "evidence": ["E1", "E2"]},
                           {"text": "No disagreements were found.", "evidence": []}], "gaps": ["No production data."],
         "insufficient": False}


def orchestrator(client, labels, providers=(WIKI,)):
    return Orchestrator(Settings(save=False), client=client, session=FakeSession(), providers=list(providers),
                        verifier=FakeVerifier(labels), store_factory=FakeStore)


def test_research_run_reports_only_verified_findings_with_real_sources():
    labels = {"Graph models reached F1 0.91.": SUPPORTED, "Graph models are cheaper to train.": CONTRADICTED}
    data = orchestrator(FakeClient(PLAN, SYNTH), labels).run("Do graph models beat boosting for fraud?")
    assert data["status"] == "COMPLETE" and data["outcome"] == "ANSWERED"
    by_text = {c["text"]: c for c in data["claims"]}
    supported = by_text["Graph models reached F1 0.91."]
    assert supported["label"] == SUPPORTED and supported["supporting_sources"] == ["S1"]
    assert by_text["Graph models are cheaper to train."]["label"] == CONTRADICTED
    urls = {s["url"] for s in data["sources"]}
    assert all(p["url"] in urls for c in data["claims"] for p in c["passages"])          # citation integrity
    assert any("E9" in x for x in data["limitations"])                                    # unknown citation disclosed
    report = render_markdown(data)
    assert "Graph models reached F1 0.91. — [S1](https://en.wikipedia.org/wiki/Fraud)" in report
    assert "## Statements not verified" in report and "not independently checked" in report
    assert [d["text"] for d in data["disagreements"]] == ["Sources differ on cost."]   # uncited filler dropped


def test_failed_verifier_never_approves_a_claim():
    labels = {"Graph models reached F1 0.91.": "ERROR", "Graph models are cheaper to train.": "BROKEN"}
    data = orchestrator(FakeClient(PLAN, SYNTH), labels).run("Q?")
    assert data["status"] == "PARTIAL" and data["outcome"] == "INSUFFICIENT_EVIDENCE"
    assert all(c["label"] == INSUFFICIENT and not c["parse_ok"] for c in data["claims"])


def test_verify_mode_outcome():
    claim = "Graph models reached F1 0.91 on transactions."
    data = orchestrator(FakeClient(PLAN, SYNTH), {claim: CONTRADICTED}).run(claim, "verify")
    assert data["outcome"] == CONTRADICTED and data["claims"][0]["contradicting_sources"] == ["S1"]
    assert "## Verdict: CONTRADICTED" in render_markdown(data)


def test_a_cancelling_progress_callback_stops_the_run_at_its_next_checkpoint_and_nothing_is_saved():
    saved, seen = [], []
    orch = Orchestrator(Settings(save=True), client=FakeClient(PLAN, SYNTH), session=FakeSession(), providers=[WIKI],
                        verifier=FakeVerifier({}), store_factory=FakeStore)
    orch._save = lambda data: saved.append(data) or "never"

    def progress(data):
        seen.append([s["name"] for s in data["stages"]])
        if "plan" in seen[-1]:
            raise RunCancelled("job")

    with pytest.raises(RunCancelled):
        orch.run("Q?", progress=progress)
    assert seen[-1] == ["model check", "plan"] and not saved          # stopped before searching; nothing saved


def test_a_cancel_that_arrives_after_the_last_stage_leaves_the_run_complete():
    labels = {"Graph models reached F1 0.91.": SUPPORTED, "Graph models are cheaper to train.": CONTRADICTED}

    def progress(data):
        if data.get("finished"):                                      # the final report, after the run is done
            raise RunCancelled("job")

    data = orchestrator(FakeClient(PLAN, SYNTH), labels).run("Q?", progress=progress)
    assert data["status"] == "COMPLETE" and data["outcome"] == "ANSWERED"


@pytest.mark.parametrize("client,expected", [
    (FakeClient(describe={"model": "qwen3:4b", "ollama_version": None, "model_digest": None}), "not reachable"),
    (FakeClient(describe={"model": "qwen3:4b", "ollama_version": "0.40", "model_digest": None}), "ollama pull"),
])
def test_model_problems_fail_clearly(client, expected):
    data = orchestrator(client, {}).run("Q?")
    assert data["status"] == "FAILED" and expected in data["error"] and not data["sources"]


def test_no_sources_and_search_outage():
    empty = orchestrator(FakeClient(PLAN, SYNTH), {}, providers=[FakeProvider("wikipedia", [])]).run("Q?")
    assert empty["status"] == "COMPLETE" and empty["outcome"] == "INSUFFICIENT_EVIDENCE"
    outage = orchestrator(FakeClient(PLAN, SYNTH), {}, providers=[FakeProvider("wikipedia", [], error="timeout")]).run("Q?")
    assert outage["status"] == "FAILED" and "No search provider" in outage["error"]
    assert any("failed for 2 of 2 queries" in x for x in outage["limitations"])          # input + planner query
    assert outage["provider_status"]["wikipedia"]["failures"] == {"timeout": 2}


def test_unparsable_synthesis_fails_without_findings():
    data = orchestrator(FakeClient(PLAN, "{not json"), {}).run("Q?")
    assert data["status"] == "FAILED" and data["claims"] == [] and "could not be parsed" in data["error"]


# ============================================================================================ web interface

def test_server_validates_requests():
    class Instant:
        def run(self, text, mode, progress):
            return {"input": text, "mode": mode, "status": "COMPLETE", "outcome": "ANSWERED", "stages": []}

    jobs = JobQueue(Instant)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(jobs, 0))
    port = server.server_address[1]
    server.RequestHandlerClass = make_handler(jobs, port)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"

    def post(body, headers):
        request = urllib.request.Request(base + "/api/run", data=json.dumps(body).encode(), headers=headers,
                                         method="POST")
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    try:
        assert post({"input": "Q?"}, {"Content-Type": "text/plain"})[0] == 415
        assert post({"input": "Q?"}, {"Content-Type": "application/json", "Origin": "https://evil.example"})[0] == 403
        assert post({"input": ""}, {"Content-Type": "application/json"})[0] == 400
        status, body = post({"input": "Q?", "mode": "verify"}, {"Content-Type": "application/json",
                                                                "Origin": base})
        assert status == 202
        for _ in range(50):
            job = jobs.get(body["id"])
            if job["state"] == "done":
                break
            threading.Event().wait(0.05)
        assert job["state"] == "done" and job["data"]["outcome"] == "ANSWERED"
        with urllib.request.urlopen(base + "/", timeout=5) as page:
            assert b"Evidence Research Assistant" in page.read()
    finally:
        server.shutdown()
        server.server_close()
