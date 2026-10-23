"""Numeric grounding of generated summaries in the application (agents._statement_ok / summarize). Offline: a fake
model client returns fixed JSON; no model, no network. The token contract itself: tests/test_numeric_tokens.py."""
import json
from types import SimpleNamespace

from webrag.agents import _statement_ok, summarize


# ------------------------------------------------------------------------------------------- _statement_ok
def test_statement_ok_rejects_digit_collisions_and_keeps_exact_numbers():
    finding = "Solar capacity grew 25 percent in 2024."
    assert _statement_ok("Solar capacity grew 25 percent in 2024.", [finding])
    assert not _statement_ok("Solar capacity grew 5 percent in 2024.", [finding])         # was accepted ("5" in "25")
    assert not _statement_ok("Solar capacity grew 2 percent.", [finding])                 # was accepted ("2" in "2024")
    assert _statement_ok("Solar capacity grew in 2024.", [finding])


# ------------------------------------------------------------------------------------------- summarize (integration)
class FixedClient:
    def __init__(self, reply):
        self.reply = reply

    def chat(self, messages, **kwargs):
        return SimpleNamespace(text=json.dumps(self.reply))


def test_summary_sentence_with_a_collided_number_is_dropped_and_counted():
    findings = ["Solar capacity grew 25 percent in 2024.", "Storage costs fell to 100 dollars per kilowatt hour."]
    reply = {"answered": True,
             "direct_answer": {"text": "Solar capacity grew 25 percent in 2024.", "findings": ["F1"]},
             "summary": [{"text": "Capacity grew 5 percent.", "findings": ["F1"]},                 # collision
                         {"text": "Storage costs fell to 10 dollars per kilowatt hour.", "findings": ["F2"]},   # 10 in 100
                         {"text": "Storage costs fell to 100 dollars per kilowatt hour.", "findings": ["F2"]}]}
    result = summarize(FixedClient(reply), "Q?", findings)
    assert result["available"] and result["direct_answer"]["text"] == "Solar capacity grew 25 percent in 2024."
    assert [p["text"] for p in result["summary"]] == ["Storage costs fell to 100 dollars per kilowatt hour."]
    assert result["dropped"] == 2


def test_a_summary_whose_every_sentence_is_dropped_is_not_used():
    reply = {"answered": True, "direct_answer": {"text": "Capacity grew 5 percent.", "findings": ["F1"]},
             "summary": [{"text": "Capacity grew 2 percent.", "findings": ["F1"]}]}
    result = summarize(FixedClient(reply), "Q?", ["Capacity grew 25 percent in 2024."])
    assert result["available"] is False and result["answered"] is False and result["dropped"] == 2
    assert "did not cite verified findings" in result["reason"]
