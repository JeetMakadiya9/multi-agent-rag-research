"""Retraction status contract (webrag/source_status.py). All assertions below are SYNTHETIC fixtures written for these
tests (example DOIs under 10.5555, a prefix reserved for examples); none comes from a real provider response."""
import pytest

from webrag.source_status import (CONFLICTING, LIVE_LOOKUP_AVAILABLE, LOOKUP_FAILED, NOT_RETRACTED, RETRACTED, STATES,
                                  UNKNOWN, resolve, valid_doi)

PAPER = "10.5555/example.paper.1"
NOTICE = "10.5555/example.notice.1"
PREPRINT = "10.5555/example.preprint.1"


def said(assertion, subject=PAPER, provider="fixture-a", **extra):
    return {"provider": provider, "subject": subject, "assertion": assertion, **extra}


def test_states_and_no_live_lookup():
    assert STATES == ("RETRACTED", "NOT_RETRACTED", "CONFLICTING", "LOOKUP_FAILED", "UNKNOWN")
    assert LIVE_LOOKUP_AVAILABLE is False and resolve(PAPER)["live_lookup"] is False


@pytest.mark.parametrize("doi, expected", [
    ("https://doi.org/10.5555/Example.Paper.1", "10.5555/example.paper.1"),
    ("doi:10.5555/example.paper.1", "10.5555/example.paper.1"),
    ("http://dx.doi.org/10.5555%2Fexample.paper.1", "10.5555/example.paper.1"),
    ("10.55/x", None),                      # registrant too short
    ("11.5555/x", None), ("10.5555/", None), ("not a doi", None), ("", None), (None, None), (42, None),
])
def test_doi_validation_and_normalisation(doi, expected):
    assert valid_doi(doi) == expected


def test_no_lookup_and_no_doi_are_unknown_never_not_retracted():
    assert resolve(PAPER)["state"] == UNKNOWN and "no retraction lookup" in resolve(PAPER)["reason"]
    assert resolve(None, [said("not_retracted", covers_retractions=True)])["state"] == UNKNOWN
    assert resolve(PAPER, [])["state"] == UNKNOWN


def test_a_retraction_assertion_about_this_doi_is_retracted():
    r = resolve(f"https://doi.org/{PAPER.upper()}", [said("retracted", subject=f"doi:{PAPER}")])
    assert r["state"] == RETRACTED and r["doi"] == PAPER and "fixture-a" in r["reason"]


def test_not_retracted_needs_a_provider_that_covers_retractions():
    assert resolve(PAPER, [said("not_retracted", covers_retractions=True)])["state"] == NOT_RETRACTED
    uncovered = resolve(PAPER, [said("not_retracted")])
    assert uncovered["state"] == UNKNOWN and any("covers_retractions" in p for p in uncovered["problems"])
    assert resolve(PAPER, [said("not_retracted", covers_retractions=False)])["state"] == UNKNOWN


def test_a_notice_does_not_retract_itself_and_versions_do_not_transfer():
    evidence = [said("retracted", subject=PAPER)]
    assert resolve(NOTICE, evidence)["state"] == UNKNOWN              # the notice's own DOI is not the retracted work
    assert resolve(PREPRINT, evidence)["state"] == UNKNOWN            # no inferred preprint / journal link
    r = resolve(PAPER, [said("retracted", subject=PREPRINT)])
    assert r["state"] == UNKNOWN and "about other DOIs" in r["problems"][0]


def test_corrections_and_concerns_are_notes_not_retractions():
    r = resolve(PAPER, [said("correction", detail="erratum to table 2"), said("expression_of_concern")])
    assert r["state"] == UNKNOWN and len(r["notes"]) == 2 and "erratum to table 2" in r["notes"][0]
    r = resolve(PAPER, [said("correction"), said("not_retracted", provider="fixture-b", covers_retractions=True)])
    assert r["state"] == NOT_RETRACTED and r["notes"]


def test_conflicting_providers_are_shown_not_voted():
    r = resolve(PAPER, [said("retracted"), said("not_retracted", provider="fixture-b", covers_retractions=True),
                        said("not_retracted", provider="fixture-c", covers_retractions=True)])
    assert r["state"] == CONFLICTING and "fixture-a" in r["reason"] and "fixture-b" in r["reason"]


def test_errors_and_malformed_records_are_lookup_failures():
    assert resolve(PAPER, [{"provider": "fixture-a", "assertion": "error", "detail": "timeout"}])["state"] == LOOKUP_FAILED
    malformed = resolve(PAPER, [None, {"assertion": "retracted"}, said("withdrawn?"), said("retracted", subject="x"),
                                said("not_retracted", covers_retractions="yes")])
    assert malformed["state"] == LOOKUP_FAILED and len(malformed["problems"]) == 5
    assert resolve(PAPER, "retracted")["state"] == LOOKUP_FAILED
    assert resolve(PAPER, {"assertion": "retracted"})["state"] == LOOKUP_FAILED


def test_a_usable_assertion_wins_over_a_failed_provider_and_the_failure_is_kept():
    r = resolve(PAPER, [{"provider": "fixture-b", "assertion": "error", "detail": "HTTP 503"}, said("retracted")])
    assert r["state"] == RETRACTED and any("HTTP 503" in p for p in r["problems"])


def test_record_lists_only_usable_assertions_about_this_doi():
    r = resolve(PAPER, [said("retracted", detail="notice 10.5555/example.notice.1"), said("retracted", subject=NOTICE)])
    assert r["assertions"] == [{"provider": "fixture-a", "subject": PAPER, "assertion": "retracted",
                                "detail": "notice 10.5555/example.notice.1"}]
