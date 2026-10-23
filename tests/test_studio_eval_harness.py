"""The Studio evaluation harness and scorer (evaluation/studio_eval). Every claim, label and model reply here is
SYNTHETIC TEST DATA: these tests check the harness, not the verifier, and are not an evaluation."""
import hashlib
import json

import pytest

from evaluation.studio_eval import run_eval, score
from test_arm_c_studio_integration import ScriptedModel
from test_webrag import FakeStore
from webrag.orchestrator import Orchestrator, Settings

PASS_A = "The bridge opened to traffic in 1937 after four years of construction."
PASS_B = "Some accounts say the bridge did not open until 1939."
PASS_C = "The city is known for its fog and its hills."


def claim(cid, text, label, *passages):
    return {"claim_id": cid, "claim": text, "label": label,
            "passages": [{"url": f"https://example.org/{cid}/{i}", "title": "synthetic", "text": t}
                         for i, t in enumerate(passages, 1)]}


def write(tmp_path, claims, independent=True, tamper=False):
    path = tmp_path / "claims.jsonl"
    path.write_text("".join(json.dumps(c) + "\n" for c in claims), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    freeze = tmp_path / "FREEZE.json"
    freeze.write_text(json.dumps({"claims_sha256": "0" * 64 if tamper else digest,
                                  "annotation": {"independent_of_model_output": independent}}), encoding="utf-8")
    return path, freeze


CLAIMS = [claim("c1", "The bridge opened in 1937.", "SUPPORTED", PASS_A, PASS_C),
          claim("c2", "The bridge opened in 1939.", "CONTRADICTED", PASS_A, PASS_C),
          claim("c3", "The bridge is painted blue.", "NOT_ENOUGH_INFO", PASS_C)]


# ------------------------------------------------------------------------------------------- protocol enforcement
def test_the_harness_refuses_to_run_without_a_valid_freeze(tmp_path):
    path, freeze = write(tmp_path, CLAIMS)
    assert run_eval.check_freeze(path, freeze)["claims_sha256"]
    with pytest.raises(run_eval.ProtocolError, match="missing"):
        run_eval.check_freeze(path, tmp_path / "absent.json")
    with pytest.raises(run_eval.ProtocolError, match="does not match"):
        run_eval.check_freeze(*write(tmp_path, CLAIMS, tamper=True))
    with pytest.raises(run_eval.ProtocolError, match="independent of model output"):
        run_eval.check_freeze(*write(tmp_path, CLAIMS, independent=False))


@pytest.mark.parametrize("bad, words", [
    ({"claim_id": "", "claim": "x", "label": "SUPPORTED", "passages": [{"url": "u", "text": "t"}]}, "claim_id"),
    ({"claim_id": "a", "claim": "x" * 401, "label": "SUPPORTED", "passages": [{"url": "u", "text": "t"}]}, "1-400"),
    ({"claim_id": "a", "claim": "x", "label": "TRUE", "passages": [{"url": "u", "text": "t"}]}, "label"),
    ({"claim_id": "a", "claim": "x", "label": "SUPPORTED", "passages": []}, "passages"),
    ({"claim_id": "a", "claim": "x", "label": "SUPPORTED", "passages": [{"url": "u", "text": " "}]}, "text and url"),
])
def test_invalid_claims_are_rejected_before_anything_runs(tmp_path, bad, words):
    path = tmp_path / "c.jsonl"
    path.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    with pytest.raises(run_eval.ProtocolError, match=words):
        run_eval.load_claims(path)


def test_duplicate_ids_and_used_claims_are_caught(tmp_path):
    path = tmp_path / "c.jsonl"
    path.write_text(json.dumps(CLAIMS[0]) + "\n" + json.dumps(CLAIMS[0]) + "\n", encoding="utf-8")
    with pytest.raises(run_eval.ProtocolError, match="duplicate claim_id"):
        run_eval.load_claims(path)
    used = json.loads((run_eval.ROOT / run_eval.USED_CLAIM_FILES[1]).read_text(encoding="utf-8").splitlines()[0])
    reused = dict(CLAIMS[0], claim_id="r1", claim=used["claim"].upper() + "  ")
    assert run_eval.check_overlap([CLAIMS[0], reused]) == ["r1"]           # normalised match, never silently used


# ------------------------------------------------------------------------------------------- the real Studio path
def studio(model):
    orch = Orchestrator(Settings(save=False, verifier="arm_c_studio", quote_attestation=True),
                        client=model, providers=[], store_factory=FakeStore)
    options = {"verifier": "arm_c_studio", "injection_guard": "exclude", "quote_attestation": True}
    return orch, options


def test_the_harness_runs_the_studio_verify_path_and_records_what_ran():
    model = ScriptedModel({PASS_A: "SUPPORTS"})
    orch, options = studio(model)
    records = run_eval.evaluate(CLAIMS[:1], orch, options)
    [r] = records
    assert (r["verdict"], r["verifier"], r["verifier_class"]) == ("SUPPORTED", "arm_c_studio", "StudioArmCVerifier")
    assert "verifier" in model.roles and r["gold"] == "SUPPORTED" and r["error"] == ""
    assert run_eval.fingerprint(options, orch.verifier) == run_eval.fingerprint(options, orch.verifier)


def test_a_failure_is_recorded_and_scored_as_not_enough_info_never_dropped():
    class Broken:
        name = "arm_c_studio"
        uses_search = True

        def verify(self, claim, evidence, search=None):
            raise RuntimeError("boom")

    orch, options = studio(ScriptedModel({}))
    orch.verifier = Broken()
    [r] = run_eval.evaluate(CLAIMS[:1], orch, options)
    assert r["verdict"] is None and r["sub_state"] == "HARNESS_ERROR" and "boom" in r["error"]
    assert score.predicted(r) == "NOT_ENOUGH_INFO" and score.is_failure(r)


def test_a_different_verifier_than_declared_is_an_error_not_a_result():
    orch, options = studio(ScriptedModel({PASS_A: "SUPPORTS"}))
    [r] = run_eval.evaluate(CLAIMS[:1], orch, dict(options, verifier="quote_stance"))
    assert r["verdict"] is None and "ran instead of" in r["error"]


# ------------------------------------------------------------------------------------------- scorer and gate
def rec(cid, verdict, sub_state=""):
    return {"claim_id": cid, "verdict": verdict, "sub_state": sub_state, "parse_ok": sub_state not in score.FAILURE_STATES}


def test_scoring_counts_every_claim_and_reports_safety_measures():
    gold = ["SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO", "CONTRADICTED"]
    records = [rec("a", "SUPPORTED"), rec("b", "SUPPORTED"), rec("c", "INSUFFICIENT", "SERVICE_ERROR"), None]
    s = score.summarise(gold, records)
    assert s["claims"] == 4 and s["false_support"] == "1/3" and s["missed_contradiction"] == "2/2"
    assert s["operational_failures"] == {"SERVICE_ERROR": 1, "MISSING": 1}
    assert s["confusion"]["CONTRADICTED"]["NOT_ENOUGH_INFO"] == 1          # the missing record counts, as NEI


def test_the_gate_stays_closed_on_a_higher_score_with_more_false_support():
    gold = ["SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO"] * 40
    base = [rec(str(i), "INSUFFICIENT") for i in range(len(gold))]
    cand = [rec(str(i), "SUPPORTED" if g != "CONTRADICTED" else "SUPPORTED") for i, g in enumerate(gold)]
    gate = score.promotion_gate(gold, base, cand, adequate_set=True)
    assert gate["criteria"]["false_support_not_higher"] is False and gate["promote"] is False
    better = [rec(str(i), {"SUPPORTED": "SUPPORTED", "CONTRADICTED": "CONTRADICTED"}.get(g, "INSUFFICIENT"))
              for i, g in enumerate(gold)]
    assert score.promotion_gate(gold, base, better, adequate_set=True)["promote"] is True
    assert score.promotion_gate(gold, base, better, adequate_set=False)["promote"] is False   # too few claims
    assert score.adequate(gold) is False                                          # 120 claims: below 300
    assert score.adequate(["SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO"] * 100) is True
