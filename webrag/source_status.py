"""Retraction status of a scholarly source: a provider-independent contract and its resolution (offline only).

NOT VALIDATED AGAINST ANY LIVE PROVIDER. The application does not look retraction status up: no provider adapter
exists, no source record carries a status, and nothing here is called by a research run. This module fixes what a
status means and how evidence about it is combined, so that a future adapter (for example one reading a scholarly
metadata service) can be added and validated against it. See webrag/README.md ("Retraction status") for the
external validation that must come first.

States (``resolve``):

* ``RETRACTED``       at least one usable assertion says the work with this DOI is retracted, and none says it is not.
* ``NOT_RETRACTED``   at least one usable assertion says it is not retracted AND that its provider covers retractions
                      (``covers_retractions``: the provider would list one if it existed); none says retracted. Missing
                      metadata, an empty answer or a provider that does not cover retractions never gives this state.
* ``CONFLICTING``     usable assertions disagree (retracted and covered not-retracted); shown, never resolved by vote.
* ``LOOKUP_FAILED``   a lookup was attempted but gave no usable assertion: provider errors or malformed records.
* ``UNKNOWN``         no lookup was made, the source has no valid DOI, or nothing usable was said about this DOI.

Evidence is a list of assertions, each about one work::

    {"provider": "name", "subject": "<DOI the assertion is about>",
     "assertion": "retracted" | "not_retracted" | "correction" | "expression_of_concern" | "error",
     "covers_retractions": bool,          # required for "not_retracted" to count
     "detail": "free text"}               # optional

Rules that keep the status honest:

* An assertion counts only for the DOI it names (``subject``), compared after ``dedupe.normalize_doi``. A retraction
  notice is its own work with its own DOI: the adapter must state the retracted paper's DOI as ``subject``. An
  assertion about a preprint, a journal version or another related work does not transfer: version links are not
  inferred here.
* Corrections, errata and expressions of concern are recorded as notes; they never make a work ``RETRACTED``.
* A malformed assertion is reported in ``problems`` and ignored; if nothing usable remains it counts as a failed
  lookup, not as absence of a retraction.
* The status is separate from claim verification: it changes no verdict, label or corroboration rule.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Optional

from .dedupe import normalize_doi

_DOI = re.compile(r"^10\.[0-9]{4,9}(?:\.[0-9]+)*/\S+$")       # directory "10.", registrant code, "/", suffix

SOURCE_STATUS_VERSION = "source-status-v1"
RETRACTED, NOT_RETRACTED, CONFLICTING, LOOKUP_FAILED, UNKNOWN = (
    "RETRACTED", "NOT_RETRACTED", "CONFLICTING", "LOOKUP_FAILED", "UNKNOWN")
STATES = (RETRACTED, NOT_RETRACTED, CONFLICTING, LOOKUP_FAILED, UNKNOWN)
ASSERTIONS = ("retracted", "not_retracted", "correction", "expression_of_concern", "error")
NOTE_ASSERTIONS = {"correction", "expression_of_concern"}
LIVE_LOOKUP_AVAILABLE = False          # no validated provider adapter exists


def valid_doi(doi: Optional[str]) -> Optional[str]:
    """The normalised DOI (``dedupe.normalize_doi``) when it has the DOI shape "10.<4+ digit registrant>/<suffix>"
    with no whitespace in the suffix, else None."""
    value = normalize_doi(doi) if isinstance(doi, str) else None
    return value if value and _DOI.match(value) else None


def _check(item: Any) -> Optional[str]:
    """Why an assertion is unusable, or None."""
    if not isinstance(item, dict):
        return "not an object"
    if not isinstance(item.get("provider"), str) or not item["provider"].strip():
        return "no provider name"
    if item.get("assertion") not in ASSERTIONS:
        return f"unknown assertion {item.get('assertion')!r}"
    if item["assertion"] != "error" and valid_doi(item.get("subject")) is None:
        return f"no valid subject DOI ({item.get('subject')!r})"
    if "covers_retractions" in item and not isinstance(item["covers_retractions"], bool):
        return "covers_retractions is not true or false"
    return None


def resolve(doi: Optional[str], evidence: Optional[Iterable[Any]] = None) -> Dict[str, Any]:
    """The retraction status of the work ``doi`` from ``evidence`` (None = no lookup was made)."""
    record: Dict[str, Any] = {"version": SOURCE_STATUS_VERSION, "doi": valid_doi(doi), "state": UNKNOWN,
                              "reason": "", "assertions": [], "notes": [], "problems": [],
                              "live_lookup": LIVE_LOOKUP_AVAILABLE}
    if record["doi"] is None:
        record["reason"] = "the source has no valid DOI, so its retraction status cannot be looked up"
        return record
    if evidence is None:
        record["reason"] = "no retraction lookup was made (no validated provider is available)"
        return record
    if isinstance(evidence, (str, bytes, dict)):
        record["problems"].append("the lookup result is not a list of assertions")
        record["state"], record["reason"] = LOOKUP_FAILED, "the lookup result could not be read"
        return record
    retracted, cleared, errors, unrelated, unusable = [], [], [], 0, 0
    for item in evidence:
        problem = _check(item)
        if problem:
            unusable += 1
            record["problems"].append(problem)
            continue
        if item["assertion"] == "error":
            errors.append(item)
            continue
        if valid_doi(item["subject"]) != record["doi"]:
            unrelated += 1                                   # about another work (notice, preprint, other version)
            continue
        kept = {k: item[k] for k in ("provider", "subject", "assertion", "covers_retractions", "detail") if k in item}
        record["assertions"].append(kept)
        if item["assertion"] in NOTE_ASSERTIONS:
            record["notes"].append(f"{item['provider']}: {item['assertion'].replace('_', ' ')}"
                                   + (f" ({item['detail']})" if item.get("detail") else ""))
        elif item["assertion"] == "retracted":
            retracted.append(item)
        elif item.get("covers_retractions") is True:
            cleared.append(item)
        else:
            record["problems"].append(f"{item['provider']}: 'not_retracted' without covers_retractions does not "
                                      f"count as evidence of no retraction")
    if unrelated:
        record["problems"].append(f"{unrelated} assertion(s) about other DOIs were ignored")
    record["problems"] += [f"{e['provider']}: lookup error: {e.get('detail') or 'no detail'}" for e in errors]
    names = lambda items: ", ".join(sorted({i["provider"] for i in items}))
    if retracted and cleared:
        record["state"] = CONFLICTING
        record["reason"] = f"reported retracted by {names(retracted)} but not retracted by {names(cleared)}"
    elif retracted:
        record["state"], record["reason"] = RETRACTED, f"reported retracted by {names(retracted)}"
    elif cleared:
        record["state"] = NOT_RETRACTED
        record["reason"] = f"no retraction listed by {names(cleared)}, which cover retractions"
    elif errors or unusable:
        record["state"] = LOOKUP_FAILED
        record["reason"] = ("the lookup failed" + (f" ({names(errors)})" if errors else "")
                            + (f"; {unusable} unusable record(s)" if unusable else ""))
    else:
        record["reason"] = "nothing usable was reported about this DOI"
    return record
