"""Conflict-adjudicated quote-stance verification (EXPERIMENTAL opt-in verifier modes; the default stays
``quote_stance``). Not recommended where false support matters: in exploratory app runs the adjudicator chose
SUPPORTED in almost every conflict and raised false support (evaluation/webrag/ADJUDICATED_VERIFIER.md).

Per-passage judgement is exactly ``QuoteStanceVerifier`` (one quoted stance per passage, its retry and its
deterministic checks). Only when the stances genuinely conflict — at least one grounded SUPPORTS and at least one
grounded CONTRADICTS, which ``decide`` reports as INSUFFICIENT / CONFLICTING — one extra model call (the adjudicator)
is asked which side states the fact the claim asserts. Everything else is unchanged: unanimous evidence keeps its
verdict and no adjudicator is called; missing, neutral or unassessable evidence stays INSUFFICIENT.

The adjudicator's answer is accepted only if every check passes; otherwise the verdict stays INSUFFICIENT /
CONFLICTING (no verdict is forced):
* the decision is SUPPORTED, CONTRADICTED or UNRESOLVED (UNRESOLVED keeps the abstention);
* the cited passage is one of the passages shown to it (a label validated against the actual list, also enforced by
  the JSON schema's enum) — passage labels are neutral (no example label in the prompt);
* the cited passage's own stance matches the decision (a SUPPORTED decision must cite a passage judged SUPPORTS);
* the quote is found in that passage (``quote_in_passage``);
* the reply is valid JSON and not truncated (one retry); a service failure or timeout leaves the abstention;
* (order check, on by default) the same decision is reached again with the passages in reverse order.
An accepted decision is recorded as sub-state ADJUDICATED with a note naming the deciding passage and quote; the
per-passage stances, including those on the other side, are kept unchanged, so the disagreement stays visible.

Research basis and its limits (docs/thesis): on SciFact DEV, conflict adjudication raised macro-F1 over a
single-pass baseline that abstains on conflicting evidence, but not significantly over a rank tie-break, and it
increased false support. Here, unlike that study, empty evidence is never adjudicated and an unusable adjudication
abstains instead of falling back to a rank rule. Application-level results: evaluation/webrag/ADJUDICATED_VERIFIER.md.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence, Tuple

from .index import Evidence
from .llm import LLMError, OllamaChat
from .verify import (CONTRADICTED, INSUFFICIENT, SUPPORTED, QUOTE_SCHEMA_V2, VERIFIERS, QuoteStanceVerifier, Verdict,
                     quote_in_passage)

ADJUDICATED = "ADJUDICATED"
MAX_SHOWN = 6                      # passages shown to the adjudicator (at most 3 per side when more were sided)

# v3 stance prompt: QUOTE_PROMPT_V2 with the contradiction definitions made explicit and symmetric (direction,
# no effect / no association, negation) - the change that raised contradiction recall in the SciFact study.
QUOTE_PROMPT_V3 = (
    "You check one claim against one evidence passage, using only the passage.\n"
    "Step 1 - quote: copy, word for word, the single sentence of the passage that says the most about the claim's "
    "subject and the property or finding the claim asserts (for example who created it, when it happened, how many "
    "there are, where it is, or what effect was found). If no sentence addresses it, return an empty quote.\n"
    "Step 2 - same_subject_and_property: true only if the quote is about the same subject AND the same property or "
    "finding as the claim. A later version, a later event, a different part, a related product or a different "
    "population is a different subject or property.\n"
    "Step 3 - stance:\n"
    "SUPPORTS: the quote gives the same value or finding, in the same direction.\n"
    "CONTRADICTS: the quote shows the claim is false: a different, incompatible value (a different year, quantity, "
    "person, organisation or place), an effect in the opposite direction, no effect or no association where the "
    "claim asserts one, an effect where the claim asserts none, or a negation of the claim. A different value is a "
    "contradiction even if the passage never says the claim is false; a claim the passage shows to be false is "
    "CONTRADICTS, not NEUTRAL.\n"
    "NEUTRAL: anything else. Being on the same topic is not support.\n"
    "Return JSON only."
)

ADJUDICATOR_PROMPT = (
    "You resolve a disagreement about one claim. Each passage below was judged by an assessor: some as supporting "
    "the claim, some as contradicting it.\n"
    "Decide which side states the fact the claim asserts, for the same subject, the same property, the same time and "
    "the same scope as the claim. Passages about a different subject, version, time, population or scope do not "
    "decide the claim.\n"
    "SUPPORTED: a passage judged as supporting states the claim's fact, and the opposing passages are about something "
    "else or are less direct.\n"
    "CONTRADICTED: a passage judged as contradicting shows the claim's fact is false, and the supporting passages are "
    "about something else or are less direct.\n"
    "UNRESOLVED: the passages genuinely disagree about the same fact, or you cannot tell. Choosing UNRESOLVED is "
    "correct when the evidence does not settle the claim.\n"
    "For SUPPORTED or CONTRADICTED, give the label of the deciding passage exactly as it appears in brackets, and "
    "copy its deciding sentence word for word. Return JSON only."
)


def _schema(labels: Sequence[str]) -> dict:
    return {"type": "object",
            "properties": {"decision": {"type": "string", "enum": ["SUPPORTED", "CONTRADICTED", "UNRESOLVED"]},
                           "passage": {"type": "string", "enum": list(labels) + ["NONE"]},
                           "quote": {"type": "string"}},
            "required": ["decision", "passage", "quote"]}


def _shown(evidence: Sequence[Evidence], stances: Dict[str, str]) -> List[Tuple[str, Evidence, str]]:
    """The sided passages in retrieval order, at most 3 per side if more than MAX_SHOWN, labelled neutrally."""
    sides = {"SUPPORTS": [e for e in evidence if stances.get(e.id) == "SUPPORTS"],
             "CONTRADICTS": [e for e in evidence if stances.get(e.id) == "CONTRADICTS"]}
    if len(sides["SUPPORTS"]) + len(sides["CONTRADICTS"]) > MAX_SHOWN:
        keep = {e.id for side in sides.values() for e in side[:MAX_SHOWN // 2]}
    else:
        keep = {e.id for side in sides.values() for e in side}
    chosen = [e for e in evidence if e.id in keep]
    return [(f"P{i}", e, stances[e.id]) for i, e in enumerate(chosen, 1)]


class AdjudicatedQuoteStanceVerifier(QuoteStanceVerifier):
    """``quote_stance`` judgement plus adjudication of genuinely conflicting evidence (see the module docstring)."""

    name = "quote_stance_adjudicated"

    def __init__(self, client: OllamaChat, num_predict: int = 200, stance_version: int = 2,
                 adjudicator_tokens: int = 260, order_check: bool = True):
        super().__init__(client, num_predict=num_predict, version=2)
        if stance_version == 3:
            self.prompt, self.schema = QUOTE_PROMPT_V3, QUOTE_SCHEMA_V2
        elif stance_version != 2:
            raise ValueError("stance_version must be 2 or 3")
        self.stance_version, self.adjudicator_tokens = stance_version, adjudicator_tokens
        # order_check: ask a second time with the passages in reverse order and accept only the same decision (a
        # position-bias calibration, cf. Wang et al., ACL 2024). Exploratory app runs of the first version (no check)
        # gave SUPPORTED in 15 of 15 adjudications, 9 of them on claims labelled CONTRADICTED.
        self.order_check = order_check

    def verify(self, claim: str, evidence: Sequence[Evidence]) -> Verdict:
        verdict = super().verify(claim, evidence)
        if verdict.label != INSUFFICIENT or verdict.sub_state != "CONFLICTING":
            return verdict                                  # unanimous, missing or failed evidence: unchanged
        decided, note = self.adjudicate(claim, evidence, verdict)
        verdict.note = "; ".join(x for x in (verdict.note, note) if x)
        if decided:
            verdict.label, verdict.sub_state = decided, ADJUDICATED
        verdict.seconds = round(verdict.seconds, 3)
        return verdict

    def adjudicate(self, claim: str, evidence: Sequence[Evidence], verdict: Verdict) -> Tuple[Optional[str], str]:
        """Returns (SUPPORTED / CONTRADICTED, note) for an accepted decision, else (None, note): the verdict then
        stays INSUFFICIENT / CONFLICTING. With ``order_check`` the decision must be reached again with the passages
        in reverse order (relabelled), otherwise the conflict stays undecided."""
        shown = _shown(evidence, verdict.stances)
        decision, note = self._decide(claim, shown, verdict)
        if decision is None or not self.order_check:
            return decision, note
        reverse = [(f"P{i}", e, stance) for i, (_, e, stance) in enumerate(reversed(shown), 1)]
        again, again_note = self._decide(claim, reverse, verdict)
        if again != decision:
            return None, (f"adjudication rejected: the decision was not reproduced with the passages in reverse "
                          f"order ({decision} then {again or again_note.split(';')[0]})")
        return decision, note + " (reproduced with the passages in reverse order)"

    def _decide(self, claim: str, shown: Sequence[Tuple[str, Evidence, str]],
                verdict: Verdict) -> Tuple[Optional[str], str]:
        """One adjudication over ``shown`` (label, passage, stance), with its validation."""
        labels = {label: (e, stance) for label, e, stance in shown}
        body = "\n\n".join(f"[{label}] (judged {'supporting' if stance == 'SUPPORTS' else 'contradicting'}) {e.text}"
                           for label, e, stance in shown)
        messages = [{"role": "system", "content": ADJUDICATOR_PROMPT},
                    {"role": "user", "content": f"Claim: {claim}\n\nPassages:\n{body}"}]
        data = None
        for attempt in range(2):                            # one retry on unusable output
            try:
                response = self.client.chat(messages, response_format=_schema(list(labels)),
                                            num_predict=self.adjudicator_tokens)
            except LLMError as exc:
                return None, f"adjudication not completed ({exc.code}); conflicting evidence left undecided"
            verdict.calls += 1
            verdict.seconds += response.seconds
            verdict.truncated = verdict.truncated or response.truncated
            verdict.raw.append(response.text)
            if response.truncated:
                continue
            try:
                data = json.loads(response.text)
                if data.get("decision") not in ("SUPPORTED", "CONTRADICTED", "UNRESOLVED"):
                    raise ValueError("decision")
                break
            except (ValueError, TypeError, AttributeError, json.JSONDecodeError):
                data = None
        if data is None:
            return None, "adjudication output unusable after one retry; conflicting evidence left undecided"
        decision = data["decision"]
        if decision == "UNRESOLVED":
            return None, "adjudicator: unresolved; conflicting evidence left undecided"
        label = str(data.get("passage", ""))
        if label not in labels:
            return None, f"adjudication rejected: passage {label!r} was not among the passages shown"
        passage, stance = labels[label]
        expected = "SUPPORTS" if decision == SUPPORTED else "CONTRADICTS"
        if stance != expected:
            return None, (f"adjudication rejected: {decision} cites {passage.id}, which was judged {stance}")
        if not quote_in_passage(str(data.get("quote", "")), passage.text):
            return None, f"adjudication rejected: the quote is not in {passage.id}"
        return decision, f"adjudicated: {passage.id} judged decisive over conflicting evidence"


class AdjudicatedQuoteStanceV3Verifier(AdjudicatedQuoteStanceVerifier):
    """Experimental: the v3 stance prompt (explicit, symmetric contradiction definitions) plus adjudication."""

    name = "quote_stance_v3_adjudicated"

    def __init__(self, client: OllamaChat, num_predict: int = 200, adjudicator_tokens: int = 260):
        super().__init__(client, num_predict=num_predict, stance_version=3, adjudicator_tokens=adjudicator_tokens)


def _without_order_check(cls):
    """The first implemented version (one adjudication, no order check); reproduces the exploratory runs
    ``app-adj-*-quote-stance-adjudicated`` and ``app-adj-*-quote-stance-v3-adjudicated``."""
    def make(client: OllamaChat):
        verifier = cls(client)
        verifier.order_check = False
        return verifier
    return make


ADJUDICATED_VERIFIERS = {AdjudicatedQuoteStanceVerifier.name: AdjudicatedQuoteStanceVerifier,
                         AdjudicatedQuoteStanceV3Verifier.name: AdjudicatedQuoteStanceV3Verifier,
                         "quote_stance_adjudicated_nocheck": _without_order_check(AdjudicatedQuoteStanceVerifier),
                         "quote_stance_v3_adjudicated_nocheck": _without_order_check(AdjudicatedQuoteStanceV3Verifier)}
VERIFIERS.update(ADJUDICATED_VERIFIERS)          # the registry used by the studio, the CLI and the evaluation harness
