"""The three Studio-specific rules of ``arm_c_studio`` through a real Studio run (Orchestrator, registry-built verifier,
evidence store), not only at the verifier level. Model replies are SCRIPTED (tests/test_arm_c_studio_integration.py):
these tests check that the rules act in the application path, not model quality.

Rule 1 (no decision without a sided passage) is covered at this level by
test_arm_c_studio_integration.test_missing_evidence_stays_insufficient_and_is_never_decided; rules 2 and 3 are
covered here."""
import json
import re

from docqa.llm import LLMResponse
from test_arm_c_studio_integration import ScriptedModel, studio

CLAIM = "Graph models reached an F1 of 91 percent on transactions."
FOR = "Graph models reached an F1 of 91 percent on transactions."
AGAINST = "Boosting reached an F1 of 84 percent on transactions."


class CitesTheOtherSide(ScriptedModel):
    """The adjudicator decides SUPPORTED while citing the passage the verifier judged CONTRADICTS."""

    def chat(self, messages, **kw):
        response = super().chat(messages, **kw)
        if self.roles and self.roles[-1] == "adjudicator":
            user = self.users[-1]
            pid, text = re.findall(r"\[(P\d+)\] \(marked CONTRADICTS\) (.*)", user)[0]
            return LLMResponse(json.dumps({"verdict": "SUPPORTED", "passage": pid, "quote": text.strip()[:40]}),
                               "stop", 10, 100, 0.01)
        return response


class InvalidStance(ScriptedModel):
    """The verifier gives one passage a label outside SUPPORTS / CONTRADICTS / NEUTRAL."""

    def chat(self, messages, **kw):
        response = super().chat(messages, **kw)
        if self.roles and self.roles[-1] == "verifier":
            reply = json.loads(response.text)
            reply["assessments"][0]["stance"] = "PROBABLY_TRUE"
            return LLMResponse(json.dumps(reply), "stop", 10, 100, 0.01)
        return response


def test_rule2_an_adjudication_citing_the_other_side_falls_back_to_the_rank_rule_in_a_studio_run():
    model = CitesTheOtherSide({FOR: "SUPPORTS", AGAINST: "CONTRADICTS"})
    data = studio(model).run(CLAIM, "verify")
    result = data["claims"][0]
    assert "adjudicator" in model.roles and result["verifier"] == "arm_c_studio"
    assert result["sub_state"] == "RANK_TIEBREAK" and result["label"] in ("SUPPORTED", "CONTRADICTED")
    assert "does not match its cited passage" in result["verifier_note"]
    assert result["verifier_trace"]["adjudication"]["verdict"] == "SUPPORTED"          # the proposal is recorded


def test_rule3_an_invalid_stance_fails_the_claim_closed_in_a_studio_run():
    model = InvalidStance({FOR: "SUPPORTS"})
    data = studio(model).run(CLAIM, "verify")
    result = data["claims"][0]
    assert (result["label"], result["sub_state"], result["parse_ok"]) == ("INSUFFICIENT", "ASSESSMENT_FAILED", False)
    assert "no valid stance" in result["error"] and data["outcome"] == "INSUFFICIENT"
    assert result["verifier_trace"]["invalid_stance_passages"]
