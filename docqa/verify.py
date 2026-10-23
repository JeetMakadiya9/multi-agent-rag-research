"""Claim verification against retrieved evidence.

Decision rules (shared by every verifier; each passage is judged SUPPORTS / CONTRADICTS / NEUTRAL for the claim):

* SUPPORTED     - at least one passage SUPPORTS the claim and none CONTRADICTS it;
* CONTRADICTED  - at least one passage CONTRADICTS the claim and none SUPPORTS it;
* INSUFFICIENT  - no passage is decisive (sub-state NO_DECISIVE_EVIDENCE or NO_EVIDENCE), or the passages disagree
                  (sub-state CONFLICTING: no source is trusted over another), or the assessment itself failed
                  (sub-state ASSESSMENT_FAILED / SERVICE_ERROR). A missing supporting passage is never treated as
                  a contradiction.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .index import Evidence
from .llm import JsonJudge, LLMError, OllamaChat

SUPPORTED, CONTRADICTED, INSUFFICIENT = "SUPPORTED", "CONTRADICTED", "INSUFFICIENT"
LABELS = (SUPPORTED, CONTRADICTED, INSUFFICIENT)
STANCES = ("SUPPORTS", "CONTRADICTS", "NEUTRAL")


@dataclass
class Verdict:
    label: str
    sub_state: str = ""
    stances: Dict[str, str] = field(default_factory=dict)     # evidence id -> SUPPORTS / CONTRADICTS / NEUTRAL
    parse_ok: bool = True
    truncated: bool = False
    calls: int = 0
    seconds: float = 0.0
    error: Optional[str] = None
    note: str = ""
    raw: List[str] = field(default_factory=list)                # model outputs, kept for error analysis

    @property
    def supporting(self) -> List[str]:
        return [e for e, s in self.stances.items() if s == "SUPPORTS"]

    @property
    def contradicting(self) -> List[str]:
        return [e for e, s in self.stances.items() if s == "CONTRADICTS"]


def decide(stances: Dict[str, str]) -> tuple:
    """Apply the decision rules above to per-passage stances; returns (label, sub_state)."""
    if not stances:
        return INSUFFICIENT, "NO_EVIDENCE"
    supports = any(s == "SUPPORTS" for s in stances.values())
    contradicts = any(s == "CONTRADICTS" for s in stances.values())
    if supports and contradicts:
        return INSUFFICIENT, "CONFLICTING"
    if contradicts:
        return CONTRADICTED, ""
    if supports:
        return SUPPORTED, ""
    return INSUFFICIENT, "NO_DECISIVE_EVIDENCE"


def _strongest(current: Optional[str], new: str) -> str:
    """One passage judged against several components of a claim: CONTRADICTS outranks SUPPORTS outranks NEUTRAL."""
    order = {"CONTRADICTS": 2, "SUPPORTS": 1, "NEUTRAL": 0}
    return new if current is None or order[new] > order[current] else current


class _RecordingJudge(JsonJudge):
    """Keeps service failures visible: the frozen verifier turns any judge exception into a pair error."""

    def __init__(self, client: OllamaChat, num_predict: int = 256):
        super().__init__(client, num_predict)
        self.service_errors: List[str] = []

    def generate(self, messages, *, temperature: float = 0.0, timeout: int = 120) -> str:
        try:
            return super().generate(messages, temperature=temperature, timeout=timeout)
        except LLMError as exc:
            if exc.code != "empty":
                self.service_errors.append(exc.code)
            raise


class FrozenBVerifier:
    """Baseline: the frozen Experiment B verifier (pairwise judge prompt, strict parser, one retry, deterministic
    aggregation), imported unchanged and run with its frozen settings (JSON, thinking off, 256 output tokens)."""

    name = "frozen_b"

    def __init__(self, client: OllamaChat):
        self.client = client

    def verify(self, claim: str, evidence: Sequence[Evidence]) -> Verdict:
        import experiment_b_verification as frozen_b
        from claim_verification import CONTRADICTED as B_CON, SUPPORTED as B_SUP, EvidenceItem

        if not evidence:
            return Verdict(INSUFFICIENT, "NO_EVIDENCE", note="no evidence was retrieved")
        items = [EvidenceItem(evidence_id=e.id, text=e.text, source=e.document, page=e.page, document_id=e.document)
                 for e in evidence]
        judge, start = _RecordingJudge(self.client), len(self.client.history)
        config = frozen_b.ExperimentBConfig(model=self.client.model, temperature=0.0, max_evidence_items=len(items),
                                            unit_specific_retrieval=False, max_targeted_retrieval_retries=0,
                                            answer_revision_enabled=False)
        report = frozen_b.run_experiment_b_verification(claim, claim, items, retrieval_provider=None,
                                                        llm_provider=judge, config=config)
        calls = self.client.history[start:]
        verdict = Verdict(INSUFFICIENT, calls=len(calls), seconds=round(sum(c.seconds for c in calls), 3),
                          truncated=any(c.truncated for c in calls))
        if judge.service_errors:
            verdict.sub_state, verdict.parse_ok = "SERVICE_ERROR", False
            verdict.error = judge.service_errors[0]
            return verdict
        units = report.verifications
        verdict.raw = [u.raw_judge_output or "" for u in units]
        positional = {f"E{i}": e.id for i, e in enumerate(evidence, 1)}      # the frozen judge numbers passages E1..
        for unit in units:
            for row in unit.evidence_assessments or []:
                eid = positional.get(str(row.get("evidence_index", "")))
                stance = str(row.get("stance", "")).upper()
                if eid and stance in STANCES:
                    verdict.stances[eid] = _strongest(verdict.stances.get(eid), stance)
        complete = all(u.assessment_complete for u in units) and bool(units)
        verdict.parse_ok = complete and not any(e.startswith("Invalid assessment") for u in units
                                                for e in u.validation_errors)
        labels = [u.verdict for u in units]
        if any(label == B_CON for label in labels):
            verdict.label = CONTRADICTED
        elif labels and all(label == B_SUP for label in labels):
            verdict.label = SUPPORTED
        if verdict.label == INSUFFICIENT:
            if not complete:
                verdict.sub_state = "ASSESSMENT_FAILED"
            elif any(u.conflicting_components for u in units):
                verdict.sub_state = "CONFLICTING"
            elif any(e.startswith("Uncovered required detail") for u in units for e in u.validation_errors):
                verdict.sub_state = "QUALIFIER_GAP"
            else:
                verdict.sub_state = "NO_DECISIVE_EVIDENCE"
        return verdict


QUOTE_SCHEMA = {
    "type": "object",
    "properties": {"quote": {"type": "string"},
                   "stance": {"type": "string", "enum": ["SUPPORTS", "CONTRADICTS", "NEUTRAL"]}},
    "required": ["quote", "stance"],
}

QUOTE_PROMPT = (
    "You check one claim against one evidence passage, using only the passage.\n"
    "Step 1 - quote: copy, word for word, the single sentence of the passage that says the most about the claim's "
    "subject and the specific detail the claim asserts (its date, number, name, place, organisation or outcome). "
    "If no sentence addresses that detail, return an empty quote.\n"
    "Step 2 - stance:\n"
    "SUPPORTS: the quote states the same detail the claim asserts.\n"
    "CONTRADICTS: the quote gives a different, incompatible detail for the same subject, for example a different "
    "year, quantity, person, organisation or place, or the opposite result. A different value is a contradiction "
    "even if the passage never says the claim is false.\n"
    "NEUTRAL: the passage does not state that detail, or it is about a different subject. Being on the same topic "
    "is not support.\n"
    "Return JSON only."
)

# v2 (after the dev run of v1): passages with the right value for a different property ("a redesign was carried out
# in 2005" for a launch claim) were judged SUPPORTS. The model now states whether the quote concerns the same subject
# and the same property before choosing a stance, and "no" forces NEUTRAL in code.
QUOTE_SCHEMA_V2 = {
    "type": "object",
    "properties": {"quote": {"type": "string"},
                   "same_subject_and_property": {"type": "boolean"},
                   "stance": {"type": "string", "enum": ["SUPPORTS", "CONTRADICTS", "NEUTRAL"]}},
    "required": ["quote", "same_subject_and_property", "stance"],
}

QUOTE_PROMPT_V2 = (
    "You check one claim against one evidence passage, using only the passage.\n"
    "Step 1 - quote: copy, word for word, the single sentence of the passage that says the most about the claim's "
    "subject and the property the claim asserts (for example who created it, when it first happened, how many there "
    "are, where it is). If no sentence addresses that property, return an empty quote.\n"
    "Step 2 - same_subject_and_property: true only if the quote is about the same subject AND the same property as "
    "the claim. A later version, a later event, a different part or a related product is a different property or "
    "subject.\n"
    "Step 3 - stance:\n"
    "SUPPORTS: the quote gives the same value for that property.\n"
    "CONTRADICTS: the quote gives a different, incompatible value for that property (a different year, quantity, "
    "person, organisation or place, or the opposite result). A different value is a contradiction even if the "
    "passage never says the claim is false.\n"
    "NEUTRAL: anything else. Being on the same topic is not support.\n"
    "Return JSON only."
)


def _words(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def quote_in_passage(quote: str, passage: str) -> bool:
    """The quote must come from the passage: a verbatim match after normalisation, or 90% of its words present for
    quotes of at least four words (tolerates a dropped comma or changed quote mark)."""
    q, p = " ".join(_words(quote)), " ".join(_words(passage))
    if not q:
        return False
    if q in p:
        return True
    words = q.split()
    passage_words = set(p.split())
    return len(words) >= 4 and sum(w in passage_words for w in words) / len(words) >= 0.9


def numbers_in(text: str) -> set:
    return {n.replace(",", "") for n in re.findall(r"\d[\d,]*(?:\.\d+)?", text)}


class QuoteStanceVerifier:
    """Candidate replacement for the stance step: one judgement per passage, grounded in a quoted sentence.

    Deterministic checks, applied after the model answers:
    * a SUPPORTS or CONTRADICTS stance whose quote is not in the passage becomes NEUTRAL (ungrounded);
    * a SUPPORTS stance whose passage lacks a number stated in the claim becomes NEUTRAL (support must state it).
    * (v2) a stance on a quote the model itself marks as not about the same subject and property becomes NEUTRAL.
    The claim label then follows ``decide`` (conflicting evidence -> INSUFFICIENT).
    """

    name = "quote_stance"

    def __init__(self, client: OllamaChat, num_predict: int = 200, version: int = 2):
        self.client, self.num_predict, self.version = client, num_predict, version
        self.prompt, self.schema = (QUOTE_PROMPT_V2, QUOTE_SCHEMA_V2) if version == 2 else (QUOTE_PROMPT, QUOTE_SCHEMA)

    def judge_passage(self, claim: str, e: Evidence, verdict: Verdict,
                      overrides: List[str]) -> Optional[Tuple[str, str]]:
        """One passage: the stance after the deterministic checks, and the model's quote; None if unusable."""
        claim_numbers = numbers_in(claim)
        messages = [{"role": "system", "content": self.prompt},
                    {"role": "user", "content": f"Claim: {claim}\n\nPassage:\n{e.text}"}]
        stance, same, quote = None, True, ""
        for _ in range(2):                                   # one retry on unusable output
            response = self.client.chat(messages, response_format=self.schema, num_predict=self.num_predict)
            verdict.calls += 1
            verdict.seconds += response.seconds
            verdict.truncated = verdict.truncated or response.truncated
            verdict.raw.append(response.text)
            try:
                data = json.loads(response.text)
                stance, quote = str(data["stance"]).upper(), str(data.get("quote", ""))
                if stance not in STANCES:
                    raise ValueError(stance)
                if self.version == 2:
                    if not isinstance(data["same_subject_and_property"], bool):
                        raise ValueError("same_subject_and_property must be a boolean")
                    same = data["same_subject_and_property"]
                break
            except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                stance = None
        if stance is None:
            return None
        if stance != "NEUTRAL" and not same:
            overrides.append(f"{e.id}: {stance} on a different subject or property -> NEUTRAL")
            stance = "NEUTRAL"
        if stance != "NEUTRAL" and not quote_in_passage(quote, e.text):
            overrides.append(f"{e.id}: {stance} without a quote from the passage -> NEUTRAL")
            stance = "NEUTRAL"
        if stance == "SUPPORTS" and not claim_numbers <= numbers_in(e.text):
            overrides.append(f"{e.id}: SUPPORTS but the passage lacks {sorted(claim_numbers - numbers_in(e.text))}"
                             f" -> NEUTRAL")
            stance = "NEUTRAL"
        return stance, quote

    def verify(self, claim: str, evidence: Sequence[Evidence]) -> Verdict:
        if not evidence:
            return Verdict(INSUFFICIENT, "NO_EVIDENCE", note="no evidence was retrieved")
        verdict = Verdict(INSUFFICIENT)
        failures, overrides = 0, []
        for e in evidence:
            judged = self.judge_passage(claim, e, verdict, overrides)
            if judged is None:
                failures += 1
                continue
            verdict.stances[e.id] = judged[0]
        verdict.seconds = round(verdict.seconds, 3)
        verdict.note = "; ".join(overrides)
        if failures:
            # An unassessed passage might be the decisive one: never let a failed check approve or reject a claim.
            verdict.parse_ok = False
            verdict.label, verdict.sub_state = INSUFFICIENT, "ASSESSMENT_FAILED"
            return verdict
        verdict.label, verdict.sub_state = decide(verdict.stances)
        return verdict


ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {"parts": {"type": "array", "items": {
        "type": "object",
        "properties": {"subject": {"type": "string"}, "property": {"type": "string"}, "value": {"type": "string"}},
        "required": ["subject", "property", "value"]}}},
    "required": ["parts"],
}

ANALYSIS_PROMPT = (
    "Split the claim into the separate facts it asserts. For each fact give: subject (what the fact is about), "
    "property (which aspect of the subject, stated precisely, for example 'year it opened to traffic', 'year "
    "construction began', 'total height', 'final score of the 2016 match', 'who created it') and value (the value the "
    "claim gives, copied exactly from the claim). A claim that asserts one fact has one part; never more than 3 "
    "parts. Return JSON only."
)

PROPERTY_SCHEMA = {
    "type": "object",
    "properties": {"quote": {"type": "string"},
                   "quote_states": {"type": "string"},
                   "same_subject_and_property": {"type": "boolean"},
                   "stance": {"type": "string", "enum": ["SUPPORTS", "CONTRADICTS", "NEUTRAL"]}},
    "required": ["quote", "quote_states", "same_subject_and_property", "stance"],
}

PROPERTY_PROMPT = (
    "You check one fact against one evidence passage, using only the passage.\n"
    "Step 1 - quote: copy, word for word, the single sentence of the passage that comes closest to stating the "
    "given property of the given subject. Empty if no sentence does.\n"
    "Step 2 - quote_states: in a few words, which property of which subject the quote gives a value for (for "
    "example 'year construction of the bridge began').\n"
    "Step 3 - same_subject_and_property: true only if quote_states is the same property of the same subject as the "
    "fact. Another event, another stage (announced versus released, launched versus landed, construction began "
    "versus opened), another version, a part of the subject or a related subject is NOT the same property.\n"
    "Step 4 - stance: SUPPORTS if the quote gives the fact's value for that property; CONTRADICTS if it gives a "
    "different value for that same property; NEUTRAL otherwise.\n"
    "Return JSON only."
)

CONFIRM_SCHEMA = {"type": "object", "properties": {"answer": {"type": "string", "enum": ["yes", "no"]}},
                  "required": ["answer"]}

CONFIRM_PROMPT = (
    "Answer yes or no. Does the sentence give the requested property of the requested subject? Answer yes if it "
    "gives that information, even together with other details. Answer no if it gives that kind of information only "
    "for a different event, stage, version, part or subject. Return JSON only."
)


# v3c: one focused yes/no question on the real passage sentence decides every non-neutral stance. Support must be
# stated (property AND value); a contradiction must concern the same property of the same subject.
CHECK_PROMPT = "Answer yes or no about the sentence, reading it literally. Return JSON only."


def check_question(stance: str, claim: str, part: Dict[str, str]) -> str:
    claim_text = claim.strip().rstrip(".")
    if stance == "SUPPORTS":
        if part["subject"]:
            return f"Does the sentence state that the {part['property']} of {part['subject']} is {part['value']}?"
        return f"Does the sentence state that \"{claim_text}\"?"
    if part["subject"]:
        return (f"Does the sentence give the {part['property']} of {part['subject']}? Answer yes even if the value "
                f"it gives is different from {part['value']}.")
    return f"Does the sentence state something that cannot be true if this claim is true: \"{claim_text}\"?"


def _value_in_claim(value: str, claim: str) -> bool:
    words, claim_words = _words(value), set(_words(claim))
    return bool(words) and all(w in claim_words for w in words)


def _specific_value(value: str) -> bool:
    """A usable value names something: a number, a proper noun, or at least two words (not just 'first')."""
    return bool(re.search(r"\d", value) or re.search(r"\b[A-Z][a-z]", value) or len(_words(value)) >= 2)


def locate_sentence(quote: str, passage: str) -> Optional[str]:
    """The passage sentence the model meant to quote. Matching uses letters only, because the model's copies can
    corrupt digits under JSON decoding ('1912' came back as '1:12'); checks then run on the real sentence."""
    quote_words = [w for w in _words(quote) if not w.isdigit()]
    if len(quote_words) < 3:
        return passage if quote_in_passage(quote, passage) else None
    best, best_score = None, 0.0
    for sentence in re.split(r"(?<=[.!?])\s+", passage):
        sentence_words = set(_words(sentence))
        score = sum(w in sentence_words for w in quote_words) / len(quote_words)
        if score > best_score:
            best, best_score = sentence, score
    return best if best_score >= 0.6 else None


class PropertyStanceVerifier:
    """Opt-in high-precision verifier (v3c; not the default - see evaluation/webrag/DECISION_RULE_V3.md: on the
    held-out property-trap set it made no false SUPPORTED verdicts versus 3 for v2, but recognised only 5 of 12 true
    claims versus 9, with lower macro-F1 and two truncated outputs).

    Design (follow-up to v2's property confusions): the claim is split into (subject, property, value)
    facts; every fact is judged against every passage with the property named explicitly; every non-neutral stance
    must survive a separate yes/no check that the quoted sentence states that property of that subject.

    Deterministic checks: a quote must come from the passage; support needs the fact's numbers in the quote itself;
    a contradiction of a numeric value needs a number in the quote; facts whose value is not in the claim are
    discarded (the whole claim is used as one fact if none remain). A claim is CONTRADICTED if any fact is
    contradicted and none conflicts, SUPPORTED if every fact is supported, otherwise INSUFFICIENT.
    """

    name = "property_stance"

    def __init__(self, client: OllamaChat, num_predict: int = 220, version: str = "c"):
        self.client, self.num_predict, self.version = client, num_predict, version

    def _ask(self, verdict: Verdict, system: str, user: str, schema: dict, num_predict: int) -> Optional[dict]:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        for _ in range(2):                                       # one retry on unusable output
            response = self.client.chat(messages, response_format=schema, num_predict=num_predict)
            verdict.calls += 1
            verdict.seconds += response.seconds
            verdict.truncated = verdict.truncated or response.truncated
            verdict.raw.append(response.text)
            try:
                data = json.loads(response.text)
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                pass
        return None

    def analyse(self, verdict: Verdict, claim: str) -> List[Dict[str, str]]:
        data = self._ask(verdict, ANALYSIS_PROMPT, f"Claim: {claim}", ANALYSIS_SCHEMA, 250) or {}
        parts, dropped = [], 0
        for row in (data.get("parts") or [])[:3] if isinstance(data.get("parts"), list) else []:
            part = {k: str(row.get(k, "")).strip() for k in ("subject", "property", "value")} \
                if isinstance(row, dict) else {}
            if part and part["subject"] and part["property"] and _value_in_claim(part["value"], claim) \
                    and _specific_value(part["value"]):
                parts.append(part)
            else:
                dropped += 1
        covered = set().union(*(numbers_in(p["value"]) for p in parts)) if parts else set()
        if not parts or dropped or not numbers_in(claim) <= covered:
            # A fact the split lost or garbled must still be checked: add the whole claim as one more fact.
            parts.append({"subject": "", "property": "", "value": claim})
        return parts

    def _judge(self, verdict: Verdict, claim: str, part: Dict[str, str], e: Evidence) -> Optional[str]:
        fact = (f"Subject: {part['subject']}\nProperty: {part['property']}\nValue: {part['value']}"
                if part["subject"] else f"The whole claim: {claim}")
        data = self._ask(verdict, PROPERTY_PROMPT, f"Claim: {claim}\nFact to check:\n{fact}\n\nPassage:\n{e.text}",
                         PROPERTY_SCHEMA, self.num_predict)
        if data is None:
            return None
        stance, quote = str(data.get("stance", "")).upper(), str(data.get("quote", ""))
        if stance not in STANCES or not isinstance(data.get("same_subject_and_property"), bool):
            return None
        notes = verdict.note.split("; ") if verdict.note else []
        value_numbers = numbers_in(part["value"])
        # v3b let the inline flag veto every stance; on dev it also vetoed real contradictions (a different value
        # read as a different property), so from v3c it gates support only and the yes/no check decides the rest.
        gated = {"SUPPORTS", "CONTRADICTS"} if self.version == "b" else {"SUPPORTS"}
        if stance in gated and not data["same_subject_and_property"]:
            notes.append(f"{e.id}: {stance} on a different subject or property -> NEUTRAL")
            stance = "NEUTRAL"
        sentence = locate_sentence(quote, e.text) if stance != "NEUTRAL" else None
        if stance != "NEUTRAL" and sentence is None:
            notes.append(f"{e.id}: {stance} without a quote from the passage -> NEUTRAL")
            stance = "NEUTRAL"
        if stance == "SUPPORTS" and not value_numbers <= numbers_in(sentence):
            notes.append(f"{e.id}: SUPPORTS but the sentence lacks {sorted(value_numbers - numbers_in(sentence))} "
                         f"-> NEUTRAL")
            stance = "NEUTRAL"
        if stance == "CONTRADICTS" and value_numbers and not numbers_in(sentence):
            notes.append(f"{e.id}: CONTRADICTS a number with a sentence that has none -> NEUTRAL")
            stance = "NEUTRAL"
        if stance != "NEUTRAL" and self.version == "b" and part["subject"]:
            answer = self._ask(verdict, CONFIRM_PROMPT,
                               f"Requested subject: {part['subject']}\nRequested property: {part['property']}\n"
                               f"Sentence: {sentence}", CONFIRM_SCHEMA, 20)
            if answer is None:
                return None
            if str(answer.get("answer", "")).lower() != "yes":
                notes.append(f"{e.id}: {stance} not confirmed as the same property -> NEUTRAL")
                stance = "NEUTRAL"
        elif stance != "NEUTRAL" and self.version != "b":
            question = check_question(stance, claim, part)
            answer = self._ask(verdict, CHECK_PROMPT, f"Sentence: {sentence}\nQuestion: {question}", CONFIRM_SCHEMA, 20)
            if answer is None:
                return None
            if str(answer.get("answer", "")).lower() != "yes":
                notes.append(f"{e.id}: {stance} failed the check '{question}' -> NEUTRAL")
                stance = "NEUTRAL"
        verdict.note = "; ".join(notes)
        return stance

    def verify(self, claim: str, evidence: Sequence[Evidence]) -> Verdict:
        if not evidence:
            return Verdict(INSUFFICIENT, "NO_EVIDENCE", note="no evidence was retrieved")
        verdict = Verdict(INSUFFICIENT)
        parts = self.analyse(verdict, claim)
        part_labels, failures = [], 0
        for part in parts:
            stances: Dict[str, str] = {}
            for e in evidence:
                stance = self._judge(verdict, claim, part, e)
                if stance is None:
                    failures += 1
                    continue
                stances[e.id] = stance
                verdict.stances[e.id] = _strongest(verdict.stances.get(e.id), stance)
            part_labels.append(decide(stances))
        verdict.seconds = round(verdict.seconds, 3)
        verdict.note = (f"facts checked: {[p['property'] or 'whole claim' for p in parts]}; " + verdict.note).strip("; ")
        if failures:
            verdict.parse_ok = False
            verdict.label, verdict.sub_state = INSUFFICIENT, "ASSESSMENT_FAILED"
            return verdict
        labels = [label for label, _ in part_labels]
        if any(sub == "CONFLICTING" for _, sub in part_labels):
            verdict.label, verdict.sub_state = INSUFFICIENT, "CONFLICTING"
        elif CONTRADICTED in labels:
            verdict.label = CONTRADICTED
        elif all(label == SUPPORTED for label in labels):
            verdict.label = SUPPORTED
        else:
            verdict.label = INSUFFICIENT
            verdict.sub_state = "PARTLY_SUPPORTED" if SUPPORTED in labels else "NO_DECISIVE_EVIDENCE"
        return verdict


# ---------------------------------------------------------------------------------------------------------- v4

COMPOUND_RE = re.compile(r"\b(and|but|while|whereas)\b|;", re.I)
SPLIT_SCHEMA = {"type": "object", "properties": {"facts": {"type": "array", "items": {"type": "string"}}},
                "required": ["facts"]}
SPLIT_PROMPT = (
    "Rewrite the claim as a list of short, independent sentences that each state exactly one fact and name its "
    "subject explicitly. Do not add, drop or change any detail (names, numbers, units, dates, negation). If the "
    "claim states only one fact, return it unchanged as the only item. Return JSON only."
)
YESNO_PROMPT = "Answer yes or no about the text, reading it literally and using nothing else. Return JSON only."
CONTENT_STOP = set("""a an the of in on at to for by with and or but while whereas its it is was were are be been has
have had that this these those as from than then into over about""".split())


def _content_words(text: str) -> set:
    return {w for w in _words(text) if w not in CONTENT_STOP and len(w) > 1}      # len: drop the "s" of "X's"


def split_is_faithful(claim: str, facts: Sequence[str]) -> bool:
    """A split may only restate the claim: every number and content word of every fact comes from the claim, and
    together the facts keep every number and content word of the claim."""
    claim_words, claim_numbers = _content_words(claim), numbers_in(claim)
    fact_words = [_content_words(f) for f in facts]
    if not facts or any(not w for w in fact_words):
        return False
    if any(not (w <= claim_words) or not (numbers_in(f) <= claim_numbers) for f, w in zip(facts, fact_words)):
        return False
    return set().union(*fact_words) >= claim_words and set().union(*(numbers_in(f) for f in facts)) >= claim_numbers


def context_sentence(quote: str, passage: str) -> Optional[str]:
    """The passage sentence matching the quote, preceded by the sentence before it (for pronouns such as 'It')."""
    sentence = locate_sentence(quote, passage)
    if sentence is None:
        return None
    sentences = re.split(r"(?<=[.!?])\s+", passage)
    index = next((i for i, s in enumerate(sentences) if s == sentence), None)
    return (sentences[index - 1] + " " + sentence) if index else sentence


def check_question_v4(stance: str, fact: str) -> str:
    fact = fact.strip().rstrip(".")
    if stance == "SUPPORTS":
        return (f"Does the text state that \"{fact}\"? Answer yes only if every detail of that statement - names, "
                f"numbers, units, dates, negation, and what each value refers to - is stated in the text.")
    return (f"Can the text and the statement \"{fact}\" both be true at the same time? Answer no only if the text "
            f"gives a different value for exactly the same thing the statement is about.")


# v4b (one development iteration after v4a): v4a's literal reading rejected true claims stated in other words, and
# its "both true?" question kept contradictions from a different event. Both checks are now asked so that "yes"
# keeps the stance.
YESNO_PROMPT_V4B = "Answer yes or no using only the text. Return JSON only."


def check_question_v4b(stance: str, fact: str) -> str:
    fact = fact.strip().rstrip(".")
    if stance == "SUPPORTS":
        return (f"Does the text say that \"{fact}\"? The same meaning in other words counts. Answer no if a name, "
                f"number, unit, date or negation differs, if the value belongs to a different event, person or "
                f"property, or if part of the statement is not in the text.")
    return (f"Does the text show that \"{fact}\" is false? Answer yes only if the text gives a different value for "
            f"exactly the same detail of the same thing. A value that belongs to a different event, person or "
            f"property does not count.")


class CheckedStanceVerifier(QuoteStanceVerifier):
    """Candidate v4: v2's per-passage judgement unchanged, then every non-neutral stance must pass one focused yes/no
    check on the real passage text (the located sentence and the one before it):
    * SUPPORTS stays only if the text states the statement with every detail (traps, dropped qualifiers);
    * CONTRADICTS stays only if the text and the statement cannot both be true (a different value for the same thing).
    Compound claims (joined by and/but/while/whereas/;) are split into single-fact sentences first; a split that adds,
    drops or changes a detail is rejected and the whole claim is checked as one statement. Facts aggregate as: any
    contradicted -> CONTRADICTED (unless one conflicts), all supported -> SUPPORTED, otherwise INSUFFICIENT
    (PARTLY_SUPPORTED when some are supported). An unusable check output fails the assessment (never approves).
    """

    name = "checked_stance"

    def __init__(self, client: OllamaChat, num_predict: int = 200, check_version: str = "a"):
        if check_version not in {"a", "b"}:
            raise ValueError("check_version must be 'a' or 'b'")
        super().__init__(client, num_predict=num_predict, version=2)
        self.check_version = check_version

    def check(self, verdict: Verdict, stance: str, fact: str, text: Optional[str]) -> Optional[bool]:
        """True = keep the stance, False = demote it to NEUTRAL, None = unusable check output. ``text`` is the
        located sentence (None if the quote could not be located)."""
        if self.check_version == "a":                      # v4a exactly as scored in the *-checked-stance-v4a runs
            answer = self._ask_yes(verdict, text, check_question_v4(stance, fact)) if text else False
            return None if answer is None else (answer if stance == "SUPPORTS" else not answer)
        if not text:
            return False
        return self._ask_yes(verdict, text, check_question_v4b(stance, fact), YESNO_PROMPT_V4B)

    def _ask_yes(self, verdict: Verdict, text: str, question: str, system: str = YESNO_PROMPT) -> Optional[bool]:
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": f"Text: {text}\nQuestion: {question}"}]
        for _ in range(2):
            response = self.client.chat(messages, response_format=CONFIRM_SCHEMA, num_predict=20)
            verdict.calls += 1
            verdict.seconds += response.seconds
            verdict.truncated = verdict.truncated or response.truncated
            verdict.raw.append(response.text)
            try:
                answer = str(json.loads(response.text)["answer"]).lower()
                if answer in {"yes", "no"}:
                    return answer == "yes"
            except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                pass
        return None

    def split(self, verdict: Verdict, claim: str, notes: List[str]) -> List[str]:
        if not COMPOUND_RE.search(claim):
            return [claim]
        messages = [{"role": "system", "content": SPLIT_PROMPT}, {"role": "user", "content": f"Claim: {claim}"}]
        response = self.client.chat(messages, response_format=SPLIT_SCHEMA, num_predict=200)
        verdict.calls += 1
        verdict.seconds += response.seconds
        verdict.truncated = verdict.truncated or response.truncated
        verdict.raw.append(response.text)
        try:
            facts = [str(f).strip() for f in json.loads(response.text)["facts"] if str(f).strip()][:4]
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            facts = []
        if len(facts) > 1 and split_is_faithful(claim, facts):
            notes.append(f"compound claim checked as {len(facts)} facts")
            return facts
        notes.append("compound claim checked as one statement (split rejected or not needed)")
        return [claim]

    def verify(self, claim: str, evidence: Sequence[Evidence]) -> Verdict:
        if not evidence:
            return Verdict(INSUFFICIENT, "NO_EVIDENCE", note="no evidence was retrieved")
        verdict, notes, failures = Verdict(INSUFFICIENT), [], 0
        facts = self.split(verdict, claim, notes)
        fact_labels = []
        for fact in facts:
            stances: Dict[str, str] = {}
            for e in evidence:
                judged = self.judge_passage(fact, e, verdict, notes)
                if judged is None:
                    failures += 1
                    continue
                stance, quote = judged
                if stance != "NEUTRAL":
                    text = context_sentence(quote, e.text)
                    keep = self.check(verdict, stance, fact, text)
                    if keep is None:
                        failures += 1
                        continue
                    if not keep:
                        notes.append(f"{e.id}: {stance} failed the focused check -> NEUTRAL")
                        stance = "NEUTRAL"
                stances[e.id] = stance
                verdict.stances[e.id] = _strongest(verdict.stances.get(e.id), stance)
            fact_labels.append(decide(stances))
        verdict.seconds = round(verdict.seconds, 3)
        verdict.note = "; ".join(notes)
        if failures:
            verdict.parse_ok = False
            verdict.label, verdict.sub_state = INSUFFICIENT, "ASSESSMENT_FAILED"
            return verdict
        labels = [label for label, _ in fact_labels]
        if any(sub == "CONFLICTING" for _, sub in fact_labels):
            verdict.label, verdict.sub_state = INSUFFICIENT, "CONFLICTING"
        elif CONTRADICTED in labels:
            verdict.label = CONTRADICTED
        elif all(label == SUPPORTED for label in labels):
            verdict.label = SUPPORTED
        else:
            verdict.label = INSUFFICIENT
            verdict.sub_state = "PARTLY_SUPPORTED" if SUPPORTED in labels else "NO_DECISIVE_EVIDENCE"
        return verdict


def _quote_stance_v1(client: OllamaChat) -> QuoteStanceVerifier:
    return QuoteStanceVerifier(client, version=1)


# "quote_stance" is v2. Run dev-candidate-quote-stance used v1 (the code then registered v1 under that name), which
# "quote_stance_v1" reproduces.
def _property_stance_v3b(client: OllamaChat) -> PropertyStanceVerifier:
    return PropertyStanceVerifier(client, version="b")


def _checked_stance_v4b(client: OllamaChat) -> CheckedStanceVerifier:
    return CheckedStanceVerifier(client, check_version="b")


# "checked_stance" is v4a (runs *-checked-stance-v4a); "checked_stance_v4b" is the one development iteration.
# "property_stance" is v3c. Dev run dev2-property-stance-v3b used v3b, which "property_stance_v3b" reproduces
# (dev2-property-stance-v3a-b used an earlier v3a, kept only as an iteration record).
VERIFIERS = {FrozenBVerifier.name: FrozenBVerifier, QuoteStanceVerifier.name: QuoteStanceVerifier,
             "quote_stance_v1": _quote_stance_v1, PropertyStanceVerifier.name: PropertyStanceVerifier,
             "property_stance_v3b": _property_stance_v3b, CheckedStanceVerifier.name: CheckedStanceVerifier,
             "checked_stance_v4b": _checked_stance_v4b}
