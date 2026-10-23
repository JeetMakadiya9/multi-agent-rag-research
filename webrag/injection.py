"""Deterministic screen for instruction-like text in retrieved sources (prompt-injection evidence gate).

Retrieved pages, PDFs and abstracts are untrusted data, but their passages are pasted into the synthesizer, the
verifier and (opt-in) the adjudicator prompts. Text that addresses an AI system, tries to override instructions,
fakes a chat role or pre-writes a verdict can steer those calls. This module recognises such text with a small set of
conservative patterns; it does not judge whether a source is true, and it cannot recognise every attack (paraphrased
or encoded instructions pass). ``GuardedEvidenceStore`` (webrag/guarded.py) applies it to every ranked passage, so
one gate covers every model call that reads source text (``Settings.injection_guard``: "off" (default), "flag"
(record only) or "exclude" (flagged passages are not given to any model and are recorded)).

Patterns favour precision: ordinary scientific prose about "instructions" or "supports" does not match; text that
discusses prompt injection by quoting an attack does match (a known false positive, reported, not hidden).
Design notes and the evaluation: docs/reverse_engineering/ (EVALUATION_RESULTS.md).
"""
from __future__ import annotations

import re
from typing import Dict, List, Pattern, Tuple

# Revision 2 (after the development false-positive check on the WebRAG snapshots, before any held-out check): generic
# role words (assistant, model, judge, agent) and "ignore all rules" matched ordinary prose ("as an assistant
# professor", "a model for three other bridges", Wikipedia's "Ignore all rules"), "return support," matched a list,
# and a paper quoting the "<|endoftext|>" token matched the role-marker rule. The AI terms, qualifiers, verdict words
# and markers below are narrowed accordingly.
_VERDICT = r"(?:supports|supported|contradicts|contradicted|refuted|neutral|insufficient)"
_AI = (r"(?:ai|a\.i\.|artificial intelligence(?: system| model| assistant)?|(?:large )?language models?|llms?|"
       r"chatbots?|chat ?gpt|ai (?:assistant|model|system|agent|assessor|judge|grader|verifier|evaluator)s?)")

RULES: Tuple[Tuple[str, Pattern[str]], ...] = (
    # "ignore all previous instructions", "disregard the above rules"
    ("override_instructions", re.compile(
        r"\b(?:ignore|disregard|forget|override|bypass)\b[^.\n]{0,40}?\b(?:previous|prior|above|earlier|preceding|"
        r"foregoing|system)\b[^.\n]{0,30}?\b(?:instructions?|prompts?|rules|directions|guidelines|constraints)\b",
        re.I)),
    # "Note to the AI assessor: ...", "If you are an AI language model ...", "LLMs reading this must ..."
    ("addresses_ai", re.compile(
        r"\b(?:note|message|instructions?|attention|notice)\s+(?:to|for)\s+(?:the |any |all )?" + _AI + r"\b"
        r"|\b(?:if you are|you are) an? " + _AI + r"\b"
        r"|\b" + _AI + r"\s+(?:reading|processing|evaluating|analy[sz]ing|summari[sz]ing|reviewing) this\b", re.I)),
    # "answer SUPPORTS", "respond with 'supported'", "mark this claim as supported"
    ("dictates_verdict", re.compile(
        r"\b(?:answer|respond|reply|output|return|mark|label|classify|rate|judge)\b(?:\s+(?:with|as|that|it|this|the "
        r"claim|this claim|the statement)){0,2}\s*(?:as\s+)?[:\"'`]*\s*" + _VERDICT + r"\b[\"'`]?(?=\s*(?:[.,;!\n]|$|"
        r"regardless|no matter|for this|for every|for all))", re.I)),
    # "this passage confirms the claim" (self-referential endorsement of an unspecified claim)
    ("self_endorsement", re.compile(
        r"\b(?:this|the following|the present) (?:passage|text|page|document|paragraph|source)\b[^.\n]{0,30}?"
        r"\b(?:supports|confirms|proves|verifies|validates)\b[^.\n]{0,15}?\b(?:the|this|any|every|your) "
        r"(?:claim|statement|assertion)s?\b", re.I)),
    # chat-template and role markers
    ("role_marker", re.compile(
        r"<\|(?:im_start|im_end|start_header_id|end_header_id)\|>|\[/?INST\]|<<SYS>>"
        r"|(?:^|\n)\s*###\s*(?:system|assistant|instruction)s?\s*(?:prompt)?\s*:", re.I)),
    # a pre-written verdict in the verifier's own JSON vocabulary
    ("fake_output", re.compile(
        r"[\"'](?:stance|verdict|decision|label|same_subject_and_property)[\"']\s*:\s*[\"']?(?:SUPPORTS|SUPPORTED|"
        r"CONTRADICTS|CONTRADICTED|NEUTRAL|INSUFFICIENT|true|false)\b", re.I)),
    # "new instructions:", "updated system prompt"
    ("new_instructions", re.compile(
        r"\b(?:new|updated|revised|real|actual|hidden|secret)\s+(?:instructions?|system prompt|directives?)\s*[:\-]"
        r"|\bsystem prompt\s*[:\-]", re.I)),
)


def scan(text: str) -> List[str]:
    """Rule identifiers matched in ``text`` (empty when nothing instruction-like was found)."""
    if not text:
        return []
    return [name for name, pattern in RULES if pattern.search(text)]


def scan_many(texts: Dict[str, str]) -> Dict[str, List[str]]:
    """``{key: rules}`` for the texts that matched at least one rule."""
    out = {}
    for key, text in texts.items():
        hits = scan(text)
        if hits:
            out[key] = hits
    return out


MODES = ("off", "flag", "exclude")
