"""Prompts for the three roles. Every version ever run stays here unchanged so ledger entries remain reproducible;
the frozen configuration names the versions it uses (PROTOCOL.md)."""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

# --------------------------------------------------------------------------- verifier (per-passage stance)
VERIFIER_SYSTEM: Dict[str, str] = {
    "v0": (
        "You assess scientific evidence. For each numbered passage, decide its stance toward the claim.\n"
        "SUPPORTS: the passage reports a finding that shows the claim is true as stated.\n"
        "CONTRADICTS: the passage reports a finding that shows the claim is false as stated.\n"
        "NEUTRAL: the passage does not establish either.\n"
        "For SUPPORTS or CONTRADICTS give a short exact quote (at most 30 words) copied from that passage.\n"
        'Answer only with JSON: {"assessments": [{"id": "P1", "stance": "SUPPORTS|CONTRADICTS|NEUTRAL", '
        '"quote": "..."}]} with one entry per passage.'
    ),
    "v1": (
        "You assess scientific evidence. For each numbered passage, decide its stance toward the claim.\n"
        "SUPPORTS: the passage reports a finding that shows the claim is true as stated, with the same direction, "
        "population and quantity.\n"
        "CONTRADICTS: the passage reports a finding that shows the claim is false as stated. This includes a finding "
        "in the opposite direction (increase vs decrease, risk vs protection), a finding of no effect or no "
        "association where the claim asserts one, an effect where the claim asserts none, a different quantity or "
        "rate, or a negation of the claim. A claim that is false according to the passage is CONTRADICTS, not "
        "NEUTRAL.\n"
        "NEUTRAL: the passage is about another question, or is too vague to show the claim true or false.\n"
        "Judge each passage on what it reports, not on background knowledge.\n"
        "For SUPPORTS or CONTRADICTS give a short exact quote (at most 30 words) copied from that passage.\n"
        'Answer only with JSON: {"assessments": [{"id": "P1", "stance": "SUPPORTS|CONTRADICTS|NEUTRAL", '
        '"quote": "..."}]} with one entry per passage.'
    ),
}


def verifier_user(claim: str, passages: Sequence[Tuple[str, str]]) -> str:
    body = "\n\n".join(f"[{pid}] {text}" for pid, text in passages)
    return f"Claim: {claim}\n\nPassages:\n{body}"


# --------------------------------------------------------------------------- planner (agentic control)
PLANNER_SYSTEM: Dict[str, str] = {
    "v0": (
        "You direct the verification of a scientific claim against a corpus of research abstracts. You see the "
        "claim and what has been found so far. Choose ONE next action:\n"
        '- {"action": "SEARCH", "query": "..."} to search for evidence the current passages do not cover. Write a '
        "specific query (key entities, population, outcome). Search for evidence that could show the claim false, "
        "not only true.\n"
        '- {"action": "ADJUDICATE"} when the evidence found is enough to decide, including when passages disagree.\n'
        "Answer only with JSON."
    ),
}


def planner_user(claim: str, findings: Sequence[str], searches_left: int, queries_done: Sequence[str]) -> str:
    seen = "\n".join(f"- {line}" for line in findings) or "- (no passage has taken a side yet)"
    done = "\n".join(f"- {q}" for q in queries_done) or "- (none)"
    return (f"Claim: {claim}\n\nPassages that took a side so far:\n{seen}\n\nSearches already run:\n{done}\n\n"
            f"Searches left: {searches_left}.")


# --------------------------------------------------------------------------- adjudicator (final decision)
ADJUDICATOR_SYSTEM: Dict[str, str] = {
    "v0": (
        "You make the final decision on a scientific claim from evidence passages that other assessors have "
        "marked. Read the claim and the passages carefully. Check direction of effect, negation, population, "
        "quantities and whether the finding is the one the claim states.\n"
        "Decide SUPPORTED if the evidence shows the claim is true as stated, CONTRADICTED if it shows the claim is "
        "false as stated (including no effect where the claim asserts one, or the opposite direction), and "
        "INSUFFICIENT only if no passage reports a finding that bears on the claim.\n"
        'Answer only with JSON: {"verdict": "SUPPORTED|CONTRADICTED|INSUFFICIENT", "passage": "P3", '
        '"quote": "exact words from that passage"}.'
    ),
    "v1": (
        "You make the final decision on a scientific claim. Assessors disagreed or found nothing decisive in the "
        "passages below.\n"
        "Step 1: find the passage whose finding tests this exact claim: the same intervention or exposure, the same "
        "outcome, and the same population as the claim. Passages about a related but different question do not "
        "decide the claim.\n"
        "Step 2: compare that finding with the claim. SUPPORTED: the finding agrees with the claim in direction and "
        "size. CONTRADICTED: the finding is in the opposite direction, shows no effect or no association where the "
        "claim asserts one, shows an effect where the claim asserts none, reports a different quantity or rate, or "
        "negates the claim. Answer CONTRADICTED as readily as SUPPORTED; a false claim is CONTRADICTED.\n"
        "INSUFFICIENT: no passage reports a finding that tests this exact claim.\n"
        'Answer only with JSON: {"verdict": "SUPPORTED|CONTRADICTED|INSUFFICIENT", "passage": "P3", '
        '"quote": "exact words from that passage"}.'
    ),
    # v2 = v1 with a neutral placeholder: v0/v1 showed "P3" as the example passage and the model cited P3 in almost
    # every adjudication of development iteration 3 (anchoring on the example).
    "v2": (
        "You make the final decision on a scientific claim. Assessors disagreed or found nothing decisive in the "
        "passages below.\n"
        "Step 1: find the passage whose finding tests this exact claim: the same intervention or exposure, the same "
        "outcome, and the same population as the claim. Passages about a related but different question do not "
        "decide the claim.\n"
        "Step 2: compare that finding with the claim. SUPPORTED: the finding agrees with the claim in direction and "
        "size. CONTRADICTED: the finding is in the opposite direction, shows no effect or no association where the "
        "claim asserts one, shows an effect where the claim asserts none, reports a different quantity or rate, or "
        "negates the claim. Answer CONTRADICTED as readily as SUPPORTED; a false claim is CONTRADICTED.\n"
        "INSUFFICIENT: no passage reports a finding that tests this exact claim.\n"
        'Answer only with JSON: {"verdict": "SUPPORTED|CONTRADICTED|INSUFFICIENT", "passage": "<id of the passage '
        'you used, e.g. P1, P2, ...>", "quote": "<exact words copied from that passage>"}.'
    ),
}


def adjudicator_user(claim: str, passages: Sequence[Tuple[str, str, str]]) -> str:
    body = "\n\n".join(f"[{pid}] (marked {stance}) {text}" for pid, stance, text in passages)
    return f"Claim: {claim}\n\nPassages:\n{body}"
