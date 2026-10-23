"""Quote attestation in the application: the adjudicated verifier, Settings validation, orchestrator recording and the
Markdown report. Offline fakes only. Standalone module tests: tests/test_quote_attestation.py."""
import json
from types import SimpleNamespace

import pytest

from docqa.adjudicate import AdjudicatedQuoteStanceVerifier
from docqa.attestation import AttestationMixin, attestations, supports_attestation, with_attestation
from docqa.verify import VERIFIERS, QuoteStanceVerifier
from test_quote_attestation import CLAIM, EVIDENCE, P1, P2, ScriptedClient, replies, stance, verdict_fields


def test_adjudicated_verifier_keeps_its_decision_and_records_passage_quotes():
    def scripted():                     # the adjudicator's prompt (listed first) also contains both passage markers
        return ScriptedClient({"Passages:": [{"decision": "UNRESOLVED", "passage": "NONE", "quote": ""}] * 2,
                               "Alpha": [stance("The bridge opened in 1932.", "SUPPORTS")],
                               "Beta": [stance("The bridge was not opened in 1932 but later.", "CONTRADICTS")]})
    plain_client, attested_client = scripted(), scripted()
    plain = AdjudicatedQuoteStanceVerifier(plain_client).verify(CLAIM, EVIDENCE)
    attested = with_attestation(AdjudicatedQuoteStanceVerifier(attested_client)).verify(CLAIM, EVIDENCE)
    assert verdict_fields(attested) == verdict_fields(plain)
    assert (attested.label, attested.sub_state) == ("INSUFFICIENT", "CONFLICTING")
    assert {e["attestation"]["status"] for v in attestations(attested).values() for e in v} == {"LOCATED"}


def test_with_attestation_keeps_name_and_settings():
    base = VERIFIERS["quote_stance_adjudicated_nocheck"](None)
    attested = with_attestation(base)
    assert isinstance(attested, AttestationMixin) and attested.name == base.name
    assert attested.order_check is False and with_attestation(attested) is attested
    assert not isinstance(base, AttestationMixin)                     # the original is not modified


def test_settings_validate_the_combination():
    from webrag.orchestrator import Settings
    assert Settings().quote_attestation is False
    assert Settings(quote_attestation=True).quote_attestation is True
    with pytest.raises(ValueError):
        Settings(verifier="frozen_b", quote_attestation=True)
    with pytest.raises(ValueError):
        Settings(verifier="no_such_verifier")
    with pytest.raises(ValueError):
        Settings(quote_attestation="yes")


def test_adjudicated_variants_qualify_for_attestation():
    assert supports_attestation(VERIFIERS["quote_stance_adjudicated_nocheck"])


# ------------------------------------------------------------------------------------------- orchestrator and report
def orchestrator(attest):
    from webrag.orchestrator import Orchestrator, Settings
    verifier = QuoteStanceVerifier(ScriptedClient(replies()))
    return Orchestrator(Settings(quote_attestation=attest), client=object(), session=object(), providers=[],
                        verifier=verifier)


def store():
    sources = {sid: SimpleNamespace(origin=sid, url=f"https://{sid}.example", title=sid, access="FULL_TEXT",
                                    page_labels=None) for sid in ("S1", "S2")}
    return SimpleNamespace(sources=sources)


def passages():
    from webrag.evidence import Passage
    return [Passage("S1:p1:a", "S1", 1, P1, 1.0), Passage("S2:p1:b", "S2", 1, P2, 0.9)]


def test_orchestrator_records_attestation_on_the_matching_passage_only_when_enabled():
    on, off = orchestrator(True), orchestrator(False)
    a = on._verify(CLAIM, "input claim", passages(), store())
    b = off._verify(CLAIM, "input claim", passages(), store())
    assert (a.label, a.sub_state, a.supporting_sources) == (b.label, b.sub_state, b.supporting_sources)
    assert [p["key"] for p in a.passages] == ["S1:p1:a", "S2:p1:b"]
    assert a.passages[0]["quote_attestation"][0]["attestation"]["status"] == "LOCATED"
    assert a.passages[1]["quote_attestation"][0]["attestation"]["status"] == "UNATTESTED"
    assert all("quote_attestation" not in p for p in b.passages)
    assert [{k: v for k, v in p.items() if k != "quote_attestation"} for p in a.passages] == b.passages


def test_report_shows_attestation_only_when_recorded_and_legacy_runs_render():
    from dataclasses import asdict

    from webrag.report import render_markdown
    claim = asdict(orchestrator(True)._verify(CLAIM, "input claim", passages(), store()))
    data = {"mode": "verify", "input": CLAIM, "status": "COMPLETE", "outcome": claim["label"], "claims": [claim],
            "sources": [], "limitations": [], "disagreements": [], "gaps": []}
    text = render_markdown(data)
    assert "## Quote attestation (record only)" in text and "| located (exact) | accepted |" in text
    assert "unattested" in text
    legacy = json.loads(json.dumps(data))
    for p in legacy["claims"][0]["passages"]:
        p.pop("quote_attestation")
    assert "Quote attestation" not in render_markdown(legacy)
