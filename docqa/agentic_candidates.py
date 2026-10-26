"""Candidate variants of the Studio's Arm C verifier (2026-10-24 improvement work; IMPROVEMENT_PLAN.md). None of them is
the default: ``arm_c_studio`` stays the Studio's verifier unless a candidate passes the frozen plan and the owner
switches it. The frozen study code (``evaluation/experiments/agentic_verify/``) is not changed.

``arm_c_studio_qr`` (quote repair). Diagnosis: on the development claims the verifier often took the right side but
mis-copied its quote, mostly by abbreviating words ("Drosoph." for "Drosophila"). The study's grounding check needs an
exact copy, so the stance was demoted to NEUTRAL and the claim ended INSUFFICIENT; the mis-copies also vary between
identical runs. Here a quote that is not exact text of its passage is replaced by the passage span it was evidently
copied from, before the study's code checks it, but only when all of these hold:
* the span matches the quote closely (character similarity of at least 0.90);
* the quote and the span contain the same numbers;
* the quote and the span contain the same negation words.
Otherwise the quote is left alone and the study's rule demotes it, as before. Every repair is recorded in the trace.
"""
from __future__ import annotations

import dataclasses
import difflib
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from evaluation.experiments.agentic_verify import arms
from evaluation.experiments.agentic_verify.llm import parse_json

from .agentic import StudioArmCVerifier, StudioModel
from .verify import VERIFIERS

MIN_SIMILARITY = 0.90
NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
NEGATION = re.compile(r"\b(not|no|never|none|neither|nor|without|cannot|lack(?:s|ed|ing)?|absen(?:t|ce)|fail(?:s|ed)?)\b|n't\b",
                      re.I)
PASSAGES = re.compile(r"\[(P\d+)\] (.*?)(?=\n\n\[P\d+\] |\Z)", re.S)


def _numbers(text: str) -> List[str]:
    return sorted(NUMBER.findall(text))


def _negations(text: str) -> List[str]:
    return sorted(m.group(0).lower() for m in NEGATION.finditer(text))


def closest_span(quote: str, passage: str) -> Tuple[float, str]:
    """The passage span (a run of whole words) most similar to the quote, and its similarity."""
    words, size = passage.split(), len(quote.split())
    best: Tuple[float, str] = (0.0, "")
    lowered = quote.lower()
    for length in range(max(1, size - 3), size + 4):
        for start in range(0, max(1, len(words) - length + 1)):
            span = " ".join(words[start:start + length])
            ratio = difflib.SequenceMatcher(None, lowered, span.lower(), autojunk=False).ratio()
            if ratio > best[0]:
                best = (ratio, span)
    return best


def repair(quote: str, passage: str) -> Optional[str]:
    """The exact passage span to use instead of ``quote``, or None when the quote must stay as it is."""
    if not quote or arms.grounded(quote, passage):
        return None
    ratio, span = closest_span(quote, passage)
    if ratio < MIN_SIMILARITY or not arms.grounded(span, passage):
        return None
    if _numbers(quote) != _numbers(span) or _negations(quote) != _negations(span):
        return None
    return span


class QuoteRepairStudioModel(StudioModel):
    """``StudioModel`` that repairs near-copy quotes in verifier replies before the study's code reads them."""

    def __init__(self, client: Any):
        super().__init__(client)
        self.repairs: List[Dict[str, Any]] = []

    def complete(self, role, system, user, max_tokens, usage) -> str:
        text = super().complete(role, system, user, max_tokens, usage)
        if role != "verifier":
            return text
        parsed = parse_json(text)
        entries = parsed.get("assessments") if isinstance(parsed, dict) else None
        if not isinstance(entries, list):
            return text
        passages = {pid: body.strip() for pid, body in PASSAGES.findall(user.split("Passages:\n", 1)[-1])}
        changed = False
        for entry in entries:
            if not isinstance(entry, dict) or str(entry.get("stance", "")).upper().strip() not in ("SUPPORTS",
                                                                                                  "CONTRADICTS"):
                continue
            pid = str(entry.get("id", "")).strip().strip("[]")
            quote = str(entry.get("quote") or "")
            span = repair(quote, passages.get(pid, ""))
            if span is not None:
                self.repairs.append({"passage": pid, "quote": quote, "repaired_to": span})
                entry["quote"], changed = span, True
                self.quotes[pid] = span
        return json.dumps(parsed) if changed else text


class StudioArmCQuoteRepairVerifier(StudioArmCVerifier):
    """``arm_c_studio`` with near-copy quotes repaired (candidate; not the default)."""

    name = "arm_c_studio_qr"
    variant = "candidate: arm_c_studio with near-copy quote repair"
    model_class = QuoteRepairStudioModel


THINK_EXTRA_TOKENS = 3000                         # room for the reasoning trace before the JSON answer
THINK_TIMEOUT = 600


class ThinkingStudioModel(StudioModel):
    """``StudioModel`` whose verifier calls use the local model's reasoning ("thinking") mode. Planner and adjudicator
    calls are unchanged. Only a local Ollama client supports it; any other client is used as it is."""

    def __init__(self, client: Any):
        super().__init__(client)
        from .llm import OllamaChat
        self.thinking = (OllamaChat(model=client.model, base_url=client.base_url, think=True, timeout=THINK_TIMEOUT)
                         if isinstance(client, OllamaChat) else None)

    def complete(self, role, system, user, max_tokens, usage) -> str:
        if role != "verifier" or self.thinking is None:
            return super().complete(role, system, user, max_tokens, usage)
        plain, self.client = self.client, self.thinking
        try:
            return super().complete(role, system, user, max_tokens + THINK_EXTRA_TOKENS, usage)
        finally:
            self.client = plain


VERIFIER_PROMPT_P2 = (
    "You assess scientific evidence. For each numbered passage, decide its stance toward the claim.\n"
    "First identify what the claim asserts: the intervention or exposure, the population, the outcome, the direction "
    "of the effect and any quantity.\n"
    "SUPPORTS: the passage reports a finding that shows the claim is true as stated: same intervention, population, "
    "outcome and direction, and a compatible quantity.\n"
    "CONTRADICTS: the passage reports a finding about the same question that shows the claim is false as stated: the "
    "opposite direction, no effect where the claim asserts one, an effect where the claim asserts none, or an "
    "incompatible quantity.\n"
    "NEUTRAL: the passage does not report a result for the claim's specific question. This includes passages about a "
    "related but different intervention, population or outcome, background statements, hypotheses, and findings that "
    "neither confirm nor refute the claim. Do not choose CONTRADICTS merely because the passage does not mention what "
    "the claim says; that is NEUTRAL.\n"
    "Judge each passage on what it reports, not on background knowledge.\n"
    "For SUPPORTS or CONTRADICTS give a short exact quote (at most 30 words) copied word for word from that passage, "
    "without abbreviating any word.\n"
    'Answer only with JSON: {"assessments": [{"id": "P1", "stance": "SUPPORTS|CONTRADICTS|NEUTRAL", "quote": "..."}]} '
    "with one entry per passage.")


class PromptP2StudioModel(StudioModel):
    """``StudioModel`` that gives the verifier role a revised system prompt (``VERIFIER_PROMPT_P2``)."""

    def complete(self, role, system, user, max_tokens, usage) -> str:
        return super().complete(role, VERIFIER_PROMPT_P2 if role == "verifier" else system, user, max_tokens, usage)


class StudioArmCThinkingVerifier(StudioArmCVerifier):
    """``arm_c_studio`` with the verifier's reasoning mode on (candidate; not the default)."""

    name = "arm_c_studio_think"
    variant = "candidate: arm_c_studio with the local model's reasoning mode for the verifier"
    model_class = ThinkingStudioModel


class StudioArmCPromptP2Verifier(StudioArmCVerifier):
    """``arm_c_studio`` with a verifier prompt that separates CONTRADICTS from NEUTRAL explicitly (candidate)."""

    name = "arm_c_studio_p2"
    variant = "candidate: arm_c_studio with verifier prompt P2"
    model_class = PromptP2StudioModel


class StudioArmCAllPassagesVerifier(StudioArmCVerifier):
    """Round 3 candidate (``IMPROVEMENT_PLAN_3.md``): the corrected judge. Two defects found by the architecture
    audit are removed, nothing else changes:
    * every retrieved passage is judged: the verifier is called once per passage (``verifier_batch`` 1 instead of the
      frozen 8), so the model can no longer leave passages out of a batched reply (on development, 332 of 691
      passages went unjudged and were read as NEUTRAL);
    * verify mode keeps up to 8 passages of one source instead of 2 (``verify_per_source``), so the whole cited text
      reaches the verifier.
    Prompts, model settings, the adjudicator, the planner, rules 1-3 and the scoring are unchanged. Not the default."""

    name = "arm_c_studio_all"
    variant = "candidate (round 3): every passage judged in its own call; no 2-passage-per-source cap when verifying"
    verify_per_source = 8

    def __init__(self, client: Any):
        super().__init__(client)
        self.settings = dataclasses.replace(self.settings, verifier_batch=1)


VERIFIER_RETRY_SUFFIX = "\n\nReturn valid JSON with one assessment per passage."   # appended by the study's judge
INVALID_REPLY = '{"reply_rejected": true}'       # no assessments list: the study's judge retries, then fails closed


def split_verifier_user(user: str) -> Optional[Tuple[str, List[Tuple[str, str]], str]]:
    """(claim, [(passage id, text)], suffix) of a verifier prompt built by ``prompts.verifier_user``, or None when the
    prompt cannot be parsed back exactly (it is rebuilt from the parts and compared)."""
    from evaluation.experiments.agentic_verify import prompts
    suffix = VERIFIER_RETRY_SUFFIX if user.endswith(VERIFIER_RETRY_SUFFIX) else ""
    body = user[:len(user) - len(suffix)]
    if not body.startswith("Claim: ") or "\n\nPassages:\n" not in body:
        return None
    claim, listing = body[len("Claim: "):].split("\n\nPassages:\n", 1)
    passages = PASSAGES.findall(listing)
    if not passages or prompts.verifier_user(claim, passages) != body:
        return None
    return claim, passages, suffix


def check_reply(text: str, expected: Sequence[str]) -> Tuple[Optional[List[Dict[str, Any]]], str]:
    """The reply's entries, one per expected id in a valid form, or (None, "<code>: <reason>") when it is rejected.
    Codes: ``no_list`` (no assessments list), ``malformed_entry`` (not an object with a string id), ``unexpected_id``
    (not one of the passages shown), ``duplicate_id``, ``invalid_stance`` (missing or not SUPPORTS/CONTRADICTS/NEUTRAL)
    and ``missing_entry`` (a shown passage has none). Nothing is read as NEUTRAL by default: NEUTRAL counts only when
    the model wrote it."""
    parsed = parse_json(text)
    entries = parsed.get("assessments") if isinstance(parsed, dict) else None
    if not isinstance(entries, list):
        return None, "no_list: no assessments list"
    seen: Dict[str, Dict[str, Any]] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            return None, f"malformed_entry: entry {index} is not an object with a string id"
        pid = entry["id"].strip().strip("[]")
        if pid not in expected:
            return None, f"unexpected_id: unexpected id {entry['id']!r} (shown: {', '.join(expected)})"
        if pid in seen:
            return None, f"duplicate_id: duplicate id {pid}"
        stance = entry.get("stance")
        if not isinstance(stance, str) or stance.upper().strip() not in arms.STANCES:
            return None, f"invalid_stance: missing or invalid stance for {pid}: {stance!r}"
        seen[pid] = entry
    missing = [pid for pid in expected if pid not in seen]
    if missing:
        return None, f"missing_entry: no entry for {', '.join(missing)}"
    return [seen[pid] for pid in expected], ""


class _CheckedVerifierClient:
    """Wraps one verifier call: the model's reply is checked against the passage ids it was shown and rewritten to the
    study's ids, or replaced by a reply without an assessments list (rejected) with the reason recorded."""

    def __init__(self, client: Any, shown_to_study: Dict[str, str], outcome: Dict[str, Any], refuse: str = ""):
        self.client, self.shown_to_study, self.outcome, self.refuse = client, shown_to_study, outcome, refuse

    def chat(self, messages, **kw):
        response = self.client.chat(messages, **kw)
        reason = self.refuse                           # the prompt's ids could not be trusted: rejected unread
        entries = None
        if not reason:
            entries, reason = check_reply(response.text, list(self.shown_to_study))
        if entries is None:
            self.outcome.update(id_check="rejected", code=reason.split(":", 1)[0], reason=reason)
            return dataclasses.replace(response, text=INVALID_REPLY)
        self.outcome.update(id_check="ok")
        rewritten = [dict(entry, id=self.shown_to_study[entry["id"].strip().strip("[]")]) for entry in entries]
        return dataclasses.replace(response, text=json.dumps({"assessments": rewritten}))


class IdCheckedStudioModel(StudioModel):
    """``StudioModel`` whose verifier calls show the passages as P1..Pn (the ids the prompt's JSON example uses) and
    map the reply back to the study's ids, with every passage required to have exactly one valid entry.

    Why: the study's judge labels passages with run-wide ids ([P2], [P3] ...) while the prompt's example answers
    ``"id": "P1"``; a reply that copies the example is not matched, and the judge silently reads the passage as
    NEUTRAL. On the round-3 confirmation set this happened to 64% of the passages after P1 under ``arm_c_studio_all``
    (517 of 668 single-passage replies said "P1"). Here such a reply is matched correctly, and any reply that cannot be
    matched is rejected: the study's judge then retries once and, if the retry is rejected too, the claim fails closed
    (ASSESSMENT_FAILED) instead of being decided with passages read as NEUTRAL."""

    def __init__(self, client: Any, max_passages: Optional[int] = None):
        super().__init__(client)
        self.reply_checks: List[Dict[str, Any]] = []
        # the most passages one verifier call can hold (the judge's batch size). A prompt that parses into more was
        # split inside a passage whose own text contains a blank line and "[P<n>] "; rebuilding the prompt cannot detect
        # that (it joins back to the same text), this bound can
        self.max_passages = max_passages

    def complete(self, role, system, user, max_tokens, usage) -> str:
        if role != "verifier":
            return super().complete(role, system, user, max_tokens, usage)
        from evaluation.experiments.agentic_verify import prompts
        parts = split_verifier_user(user)
        outcome: Dict[str, Any] = {}
        refuse = ""
        if parts is None:
            refuse = "unparsable_prompt: the verifier prompt could not be parsed into passages"
        elif self.max_passages and len(parts[1]) > self.max_passages:
            refuse = (f"ambiguous_prompt: the prompt parses into {len(parts[1])} passages, more than the "
                      f"{self.max_passages} one call holds (a passage's text contains a passage label)")
        if refuse:
            shown_user, shown_to_study = user, {}
        else:
            claim, passages, suffix = parts
            shown_to_study = {f"P{i}": pid for i, (pid, _) in enumerate(passages, 1)}
            shown_user = prompts.verifier_user(claim, [(f"P{i}", text) for i, (_, text) in enumerate(passages, 1)])
            shown_user += suffix
        outcome.update(shown=list(shown_to_study), study_ids=list(shown_to_study.values()))
        plain, self.client = self.client, _CheckedVerifierClient(self.client, shown_to_study, outcome, refuse)
        try:
            text = super().complete(role, system, shown_user, max_tokens, usage)
        finally:
            self.client = plain
        self.reply_checks.append(outcome)
        if self.reply_ids:                             # the record of this reply (as the study's judge reads it)
            self.reply_ids[-1]["id_check"] = dict(outcome)
        return text


class StudioArmCAllPassagesIdCheckedVerifier(StudioArmCAllPassagesVerifier):
    """Round-3 candidate with the passage-id defect fixed (``IdCheckedStudioModel``); everything else, including one
    passage per verifier call and up to 8 passages per source, is ``arm_c_studio_all``. Not the default."""

    name = "arm_c_studio_all_idfix"
    variant = ("candidate: arm_c_studio_all with passage ids matched to the prompt (shown as P1..Pn, mapped back) and "
               "unmatched or invalid verifier replies rejected instead of read as NEUTRAL")
    model_class = IdCheckedStudioModel

    def run(self, claim, retriever, model):
        model.max_passages = max(1, int(self.settings.verifier_batch))
        return super().run(claim, retriever, model)


# Retired candidates (2026-10-26): rejected by their frozen plans and superseded by docqa/verifier_v2.py. They stay
# importable ONLY so the recorded evaluation runs can be reproduced; the Studio never imports this module, so it never
# offers them, and no new work should build on them.
RETIRED = {
    "arm_c_studio_qr": "round 1: no gain on the held-out split and more false support (IMPROVEMENT_PLAN.md)",
    "arm_c_studio_think": "round 2: infeasible on a 4 GB GPU, 600-700 s per claim (IMPROVEMENT_PLAN_2.md)",
    "arm_c_studio_p2": "round 2: more false support at screening (IMPROVEMENT_PLAN_2.md)",
    "arm_c_studio_all": "round 3: 75% rule not met; passage-ID defect (IMPROVEMENT_PLAN_3.md)",
}

CANDIDATE_VERIFIERS = {cls.name: cls for cls in (StudioArmCQuoteRepairVerifier, StudioArmCThinkingVerifier,
                                                 StudioArmCPromptP2Verifier, StudioArmCAllPassagesVerifier,
                                                 StudioArmCAllPassagesIdCheckedVerifier)}
VERIFIERS.update(CANDIDATE_VERIFIERS)
