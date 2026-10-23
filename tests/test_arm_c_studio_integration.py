"""End-to-end, offline: a Studio run through the real Orchestrator, the registry-built default verifier (Arm C,
Studio-adapted variant) and the real HTTP API. The web, the model and the retrieval index are replaced by the fakes of
tests/test_webrag.py (word-overlap passages instead of rag.py); everything between them is the application's code.
The model replies are SCRIPTED: they test the wiring, not model quality."""
import json
import re
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from docqa.agentic import StudioArmCVerifier
from docqa.llm import LLMResponse
from docqa.verify import SUPPORTED
from evaluation.experiments.agentic_verify import prompts
from test_webrag import PLAN, SYNTH, FakeClient, FakeProvider, FakeSession, FakeStore
from webrag.orchestrator import Orchestrator, Settings
from webrag.server import JobQueue, StatusProbe, make_handler

TEXT = ("Graph models reached an F1 of 91 percent on transactions. Boosting reached an F1 of 84 percent on transactions. "
        "Fraud rings share devices and addresses. Card fraud losses grew in 2024. "
        "Teams often deploy graph models on transactions after shadow testing.")
PROVIDER = FakeProvider("wikipedia", [("Fraud detection", "https://en.wikipedia.org/wiki/Fraud", "s", "")],
                        texts={"https://en.wikipedia.org/wiki/Fraud": TEXT})
ROLES = {prompts.VERIFIER_SYSTEM["v1"]: "verifier", prompts.PLANNER_SYSTEM["v0"]: "planner",
         prompts.ADJUDICATOR_SYSTEM["v2"]: "adjudicator"}


class ScriptedModel(FakeClient):
    """The planner/synthesizer of tests/test_webrag.py plus scripted Arm C roles: ``stances`` maps a passage text to
    its stance; the planner searches once, then asks for adjudication; the adjudicator answers ``decision``."""

    def __init__(self, stances, decision="SUPPORTED", query="deploy shadow testing"):
        super().__init__(PLAN, SYNTH)
        self.stances, self.decision, self.query, self.roles = stances, decision, query, []

    def chat(self, messages, *, response_format=None, temperature=0.0, num_predict=None, timeout=None):
        role = ROLES.get(messages[0]["content"])
        if role is None:
            return super().chat(messages, response_format=response_format, num_predict=num_predict)
        self.roles.append(role)
        user = messages[1]["content"]
        if role == "verifier":
            body = user.split("Passages:\n", 1)[1].split("\n\nReturn valid JSON", 1)[0]
            entries = re.findall(r"\[(P\d+)\] (.*?)(?=\n\n\[P\d+\] |\Z)", body, re.S)
            reply = {"assessments": [{"id": pid, "stance": self.stances.get(text.strip(), "NEUTRAL"),
                                      "quote": text.strip()[:40]} for pid, text in entries]}
        elif role == "planner":
            reply = ({"action": "SEARCH", "query": self.query} if self.roles.count("planner") == 1
                     else {"action": "ADJUDICATE"})
        else:
            marked = (re.findall(r"\[(P\d+)\] \(marked SUPPORTS\) (.*)", user)
                      or re.findall(r"\[(P\d+)\] \(marked \w+\) (.*)", user))
            pid, text = marked[0]
            reply = {"verdict": self.decision, "passage": pid, "quote": text.strip()[:40]}
        return LLMResponse(json.dumps(reply), "stop", 10, 100, 0.01)


def studio(model, passages=8):
    """The Studio's pipeline with its defaults (verifier and quote attestation as studio_options resolves them)."""
    return Orchestrator(Settings(save=False, verifier="arm_c_studio", quote_attestation=True,
                                 question_passages=passages),
                        client=model, session=FakeSession(), providers=[PROVIDER], store_factory=FakeStore)


def test_conflicting_evidence_reaches_the_arm_c_adjudicator_in_a_studio_run():
    claim = "Graph models reached an F1 of 91 percent on transactions."
    model = ScriptedModel({"Graph models reached an F1 of 91 percent on transactions.": "SUPPORTS",
                           "Boosting reached an F1 of 84 percent on transactions.": "CONTRADICTS"})
    orch = studio(model)
    assert type(orch.verifier) is StudioArmCVerifier                        # built from the registry, not injected
    data = orch.run(claim, "verify")
    result = data["claims"][0]
    assert model.roles == ["verifier", "adjudicator"]                         # Arm C: conflict -> adjudicator
    assert (data["outcome"], result["sub_state"], result["verifier"]) == (SUPPORTED, "ADJUDICATED", "arm_c_studio")
    stances = {p["text"]: p["stance"] for p in result["passages"]}
    assert stances["Boosting reached an F1 of 84 percent on transactions."] == "CONTRADICTS"        # counter-evidence kept
    assert data["disagreements"] and "adjudication" in data["disagreements"][0]["text"]
    assert result["verifier_trace"]["adjudication"]["verdict"] == SUPPORTED
    assert all("quote_attestation" in p for p in result["passages"])               # record-only attestation on
    assert data["settings"]["verifier"] == "arm_c_studio"


def test_without_a_sided_passage_the_arm_c_planner_searches_and_judges_new_passages():
    claim = "Graph models are used in production."
    found = "Teams often deploy graph models on transactions after shadow testing."
    model = ScriptedModel({found: "SUPPORTS"})
    data = studio(model, passages=1).run(claim, "verify")             # the first retrieval misses that passage
    result = data["claims"][0]
    assert model.roles[:3] == ["verifier", "planner", "verifier"]             # planner, then the new passage judged
    assert result["verifier_trace"]["actions"][0] == {"turn": 0, "action": "SEARCH",
                                                      "query": "deploy shadow testing", "new_passages": 4}
    searched = [p for p in result["passages"] if p.get("found_by_search")]
    assert any(p["text"] == found and p["stance"] == "SUPPORTS" for p in searched)
    assert (result["label"], result["sub_state"]) == (SUPPORTED, "")          # unanimous after search
    assert result["supporting_sources"] == ["S1"]


def test_missing_evidence_stays_insufficient_and_is_never_decided():
    model = ScriptedModel({})
    data = studio(model).run("Quantum annealers detect card fraud in real time.", "verify")
    result = data["claims"][0]
    assert result["label"] == "INSUFFICIENT" and result["sub_state"] == "NO_DECISIVE_EVIDENCE"
    assert "not accepted without a passage judged for or against" in result["verifier_note"]


def test_the_decision_and_evidence_survive_the_http_api():
    model = ScriptedModel({"Graph models reached an F1 of 91 percent on transactions.": "SUPPORTS",
                           "Boosting reached an F1 of 84 percent on transactions.": "CONTRADICTS"})
    jobs = JobQueue(lambda: studio(model))
    options = {"verifier": "arm_c_studio", "injection_guard": "exclude", "quote_attestation": True,
               "sources": {"verifier": "default", "injection_guard": "default", "quote_attestation": "default"}}
    probe = StatusProbe("qwen3:4b", "arm_c_studio", describe=lambda: {"ollama_version": "0.40.6", "model_digest": "x"},
                        providers=lambda: [], models=lambda: [], options=options)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(jobs, 0, status=probe))
    port = server.server_address[1]
    server.RequestHandlerClass = make_handler(jobs, port, status=probe)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"

    def post(body):
        request = urllib.request.Request(base + "/api/run", data=json.dumps(body).encode(), method="POST",
                                         headers={"Content-Type": "application/json", "Origin": base})
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    try:
        status, body = post({"input": "Graph models reached an F1 of 91 percent on transactions.", "mode": "verify",
                             "verifier": "quote_stance"})
        assert status == 400 and "fixed when the Studio starts" in body["error"]      # no per-request override
        status, body = post({"input": "Graph models reached an F1 of 91 percent on transactions.", "mode": "verify"})
        assert status == 202
        for _ in range(100):
            with urllib.request.urlopen(f"{base}/api/run/{body['id']}", timeout=5) as response:
                job = json.loads(response.read())
            if job["state"] == "done":
                break
            threading.Event().wait(0.05)
        claim = job["data"]["claims"][0]
        assert (claim["label"], claim["sub_state"], claim["verifier"]) == (SUPPORTED, "ADJUDICATED", "arm_c_studio")
        assert sorted(p["stance"] for p in claim["passages"] if p["stance"] != "NEUTRAL") == ["CONTRADICTS", "SUPPORTS"]
        assert claim["verifier_trace"]["study_arm"] == "C"
        with urllib.request.urlopen(base + "/api/status", timeout=5) as response:
            status_body = json.loads(response.read())
        assert (status_body["verifier"], status_body["injection_guard"], status_body["quote_attestation"]) == \
            ("arm_c_studio", "exclude", True)
    finally:
        server.shutdown()
        server.server_close()
