"""Rank tie-break quote-stance verification (EXPERIMENTAL opt-in verifier mode; the default stays ``quote_stance``).

Per-passage judgement is exactly ``QuoteStanceVerifier`` (one quoted stance per passage, its retry and its
deterministic checks). Only when the grounded stances genuinely conflict — at least one SUPPORTS and at least one
CONTRADICTS, which ``decide`` reports as INSUFFICIENT / CONFLICTING — the conflict is decided by the first sided
passage in the order the evidence was given (for a claim check, retrieval rank; for a research finding, the
finding's cited passages first, then ranked extras). No extra model call is made. Everything else is unchanged:
unanimous evidence keeps its verdict; missing, neutral or unassessable evidence stays INSUFFICIENT.

A decided conflict is recorded as sub-state RANK_TIEBREAK with a note naming the deciding passage; the per-passage
stances, including those on the other side, are kept unchanged, so the disagreement stays visible.

Research basis and its limits (agentic verification study, SciFact DEV, arm B1R): a rank tie-break on conflicting
evidence raised macro-F1 over a single-pass baseline that abstains on conflicts (0.715 against 0.579), with the same
false-support count. That study judged eight SciFact passages in one verifier call and ranked them with the frozen
hybrid retriever; this mode judges each web passage separately. Like B1R, empty or all-neutral evidence is never
decided. The rule has NOT been evaluated on the Studio's web evidence. A first-ranked passage is not a more reliable
passage, so a decided conflict is shown as a decided conflict, never as clean support or contradiction.
"""
from __future__ import annotations

from typing import Sequence

from .adjudicate import ADJUDICATED
from .index import Evidence
from .verify import CONTRADICTED, INSUFFICIENT, SUPPORTED, VERIFIERS, QuoteStanceVerifier, Verdict

RANK_TIEBREAK = "RANK_TIEBREAK"
# arm_c only: the adjudicator decided although no passage was judged for or against the claim. Not a
# disagreement, so it is not in DECIDED_CONFLICTS; it has its own label, caution and review entry.
ADJUDICATED_UNSIDED = "ADJUDICATED_UNSIDED"
DECIDES = {"SUPPORTS": SUPPORTED, "CONTRADICTS": CONTRADICTED}
# Conflicting evidence decided by a verifier mode (agentic or ranked), and how each mechanism is named in disagreements.
DECIDED_CONFLICTS = {ADJUDICATED: "adjudicated", RANK_TIEBREAK: "decided by rank"}


class RankedQuoteStanceVerifier(QuoteStanceVerifier):
    """``quote_stance`` whose conflicting verdicts are decided by the first sided passage in evidence order."""

    name = "quote_stance_ranked"

    def verify(self, claim: str, evidence: Sequence[Evidence]) -> Verdict:
        verdict = super().verify(claim, evidence)
        if verdict.label != INSUFFICIENT or verdict.sub_state != "CONFLICTING":
            return verdict
        for e in evidence:                                   # the order the caller ranked the evidence in
            stance = verdict.stances.get(e.id)
            if stance in DECIDES:
                verdict.label, verdict.sub_state = DECIDES[stance], RANK_TIEBREAK
                note = (f"conflicting evidence decided by rank: {e.id} (the first passage with a stance) "
                        f"{stance}; no extra model call")
                verdict.note = "; ".join(x for x in (verdict.note, note) if x)
                break
        return verdict


RANKED_VERIFIERS = {RankedQuoteStanceVerifier.name: RankedQuoteStanceVerifier}
VERIFIERS.update(RANKED_VERIFIERS)               # the registry used by the studio, the CLI and the evaluation harness
