"""Record-only quote attestation for the quote-based verifiers (opt-in; no effect on any stance or verdict).

The frozen ``QuoteStanceVerifier`` grounds a SUPPORTS / CONTRADICTS stance with ``quote_in_passage``, which is
tolerant: the joined lowercase word tokens of the quote may occur anywhere in the passage's joined tokens, with no
word boundary at either end, or 90% of a quote of at least four words may be present in any order. Reordered words,
a dropped "not" and "effective" inside "ineffective" pass. ``docqa/verify.py`` is frozen, so that check is unchanged.

This module records, next to it, where the verifier's final quote actually occurs in the passage it judged, using
``evidence_core.span_attestation.attest_quote`` unchanged (EXACT / CANONICAL tiers; LOCATED, AMBIGUOUS, UNATTESTED,
EMPTY; offsets into the original passage). It is a description of text position only: a LOCATED quote can still fail
to support the claim, and nothing here rejects a quote, changes a stance, causes an abstention or alters a verdict.

How the record is tied to the right passage and result:

* ``AttestationMixin.judge_passage`` wraps the verifier's own ``judge_passage``, which already includes its retry on
  unusable output; the quote attested is the one returned with the final stance (after the deterministic checks).
* Each record is keyed by the evidence label the verifier was given (``E1``, ``E2`` ... in the order of the passages
  the orchestrator passed) and stored on the ``Verdict`` object of that one ``verify`` call, never on the verifier,
  so a verifier instance shared between runs holds no per-request state. A verifier that judges one passage several
  times (once per fact, ``CheckedStanceVerifier``) records one entry per judgement, in order.
* With conflict adjudication (``quote_stance_adjudicated``) the per-passage quotes are attested; the adjudicator's own
  deciding quote is not (it is validated inside the frozen-protocol code path and only its outcome is kept).

``with_attestation(verifier)`` returns a copy of a quote-based verifier whose class adds the mixin; the registry name
and every setting are kept. Verifiers that do not ground a stance in a quote are refused with ``ValueError``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from evidence_core.span_attestation import attest_quote

from .verify import QuoteStanceVerifier, Verdict, quote_in_passage

ATTESTATION_RECORD_VERSION = "quote-attestation-record-v1"
_ATTRIBUTE = "quote_attestations"


def attestation_record(quote: str, passage: str) -> Dict[str, Any]:
    """The span attestation of ``quote`` in ``passage`` as plain JSON. Every span satisfies
    ``passage[start:end] == text`` (the attestation module's own invariant)."""
    result = attest_quote(quote, passage)
    return {"status": result.status.value, "tier": result.tier.value if result.tier else None,
            "reason": result.reason.value,
            "spans": [{"start": s.start, "end": s.end, "text": s.text, "tier": s.tier.value,
                       "negation_cue": s.negation_cue} for s in result.spans],
            "attestation_version": result.version, "canonicalization": result.canonicalization}


class AttestationMixin:
    """Adds a record of where each judged quote occurs; the judgement returned is exactly the wrapped one."""

    def judge_passage(self, claim: str, e: Any, verdict: Verdict, overrides: List[str]) -> Optional[Tuple[str, str]]:
        judged = super().judge_passage(claim, e, verdict, overrides)
        entry: Dict[str, Any] = {"record_version": ATTESTATION_RECORD_VERSION, "checked_text": claim}
        if judged is None:
            entry["assessed"] = False                    # no usable verifier output after its retry: nothing to locate
        else:
            stance, quote = judged
            entry.update(assessed=True, stance=stance, quote=quote)
            try:
                entry["legacy_quote_check"] = quote_in_passage(quote, e.text)
                entry["attestation"] = attestation_record(quote, e.text)
            except Exception as exc:                     # recorded, never raised: attestation must not stop a verdict
                entry["attestation_error"] = f"{type(exc).__name__}: {exc}"
        verdict.__dict__.setdefault(_ATTRIBUTE, {}).setdefault(e.id, []).append(entry)
        return judged


def attestations(verdict: Any) -> Dict[str, List[Dict[str, Any]]]:
    """The records of one ``verify`` call by evidence label; empty for a verifier without attestation."""
    return dict(getattr(verdict, _ATTRIBUTE, None) or {})


def supports_attestation(factory: Any) -> bool:
    """Whether a ``VERIFIERS`` entry (a class, or a factory taking the model client) builds a quote-based verifier.
    Factories are called with no client: the registered ones only store it when they are built."""
    if getattr(factory, "records_quote_attestation", False):     # Arm C verifiers capture and attest their quotes
        return True
    if isinstance(factory, type):
        return issubclass(factory, QuoteStanceVerifier)
    return isinstance(factory(None), QuoteStanceVerifier)


_CLASSES: Dict[type, type] = {}


def with_attestation(verifier: Any) -> Any:
    """A copy of ``verifier`` that also records quote attestations. Only quote-based verifiers qualify."""
    if not isinstance(verifier, QuoteStanceVerifier):
        raise ValueError(f"quote attestation needs a quote-based verifier (quote_stance or one derived from it); "
                         f"{getattr(verifier, 'name', type(verifier).__name__)!r} does not ground stances in quotes")
    base = type(verifier)
    if isinstance(verifier, AttestationMixin):
        return verifier
    if base not in _CLASSES:
        _CLASSES[base] = type(f"Attesting{base.__name__}", (AttestationMixin, base), {})
    attested = object.__new__(_CLASSES[base])
    attested.__dict__.update(verifier.__dict__)
    return attested
