"""Phase 13 B+ variant components. Frozen modules are imported, never edited.

Design: every variant reuses the frozen System B adapter
(``evaluation.experiments.system_b_verified_rag.run_system_b``) and the frozen
Experiment B engine. Each variant changes exactly one factor:

* evidence limit (B1)        -> ``ExperimentConfig.verification_evidence_limit``
* evidence packaging (B2)    -> ``ReplayRetrievalProvider(context_window=1)``
* lexical guards (B3)        -> ``validate_judgement_plus(guard_version="v2")``
* judge prompt (B4)          -> ``judge_prompt_override(PROMPT_V2_SYSTEM)``

Retrieval is replayed from the frozen Phase 10F run (identical chunks, ranks
and scores), so retrieval is held constant by construction.
"""
from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass, field, replace
from typing import Any, Iterator, Mapping, Sequence

import experiment_b_verification as frozen_b
from claim_verification import CONTRADICTED, INSUFFICIENT_EVIDENCE, SUPPORTED, EvidenceItem
from research_retrieval import RetrievalEvidence, RetrievalResponse

# --------------------------------------------------------------- retrieval replay


def package_with_context(title: str, abstract: Sequence[str], sentence_index: int, window: int) -> str:
    """B2 evidence packaging: the retrieved sentence plus same-abstract neighbours.

    The retrieved sentence keeps its identity and is clearly marked; neighbours
    are labelled as context. Only corpus text is used (no gold information).
    """
    lines = [f"Title: {title}."]
    for offset in range(-window, 0):
        j = sentence_index + offset
        if 0 <= j < len(abstract):
            lines.append(f"Context sentence before (abstract sentence {j + 1}): {abstract[j]}")
    lines.append(f"Retrieved sentence (abstract sentence {sentence_index + 1}): {abstract[sentence_index]}")
    for offset in range(1, window + 1):
        j = sentence_index + offset
        if 0 <= j < len(abstract):
            lines.append(f"Context sentence after (abstract sentence {j + 1}): {abstract[j]}")
    return "\n".join(lines)


@dataclass
class ReplayRetrievalProvider:
    """Serves the frozen Phase 10F retrieved chunks for a claim (gold-free).

    ``retrieval_by_claim_text`` maps claim text -> frozen chunk refs. Only text,
    doc ID, sentence index and score are used — the same fields the frozen
    ``_runtime_evidence`` consumes. With ``context_window > 0`` the evidence text
    is re-packaged with neighbouring sentences from the same abstract.
    """
    retrieval_by_claim_text: Mapping[str, Sequence[Mapping[str, Any]]]
    corpus: Mapping[int, Any] = field(default_factory=dict)   # doc_id -> CorpusDocument
    context_window: int = 0
    name: str = "phase10f_frozen_retrieval_replay"
    calls: int = 0

    def retrieve(self, query: str, limit: int = 8) -> RetrievalResponse:
        self.calls += 1
        chunks = self.retrieval_by_claim_text[query]
        evidence = []
        for chunk in list(chunks)[:limit]:
            doc_id = int(chunk["scifact_doc_id"])
            indices = list(chunk.get("scifact_sentence_indices") or [])
            text = chunk["text"]
            context_indices: list[int] = []
            if self.context_window > 0:
                if len(indices) != 1:
                    raise ValueError("context packaging expects one sentence per chunk")
                doc = self.corpus[doc_id]
                i = indices[0]
                text = package_with_context(doc.title, doc.abstract, i, self.context_window)
                context_indices = [j for j in range(i - self.context_window, i + self.context_window + 1)
                                   if 0 <= j < len(doc.abstract) and j != i]
            evidence.append(RetrievalEvidence(
                evidence_id=str(chunk["chunk_id"]), text=text, source_id=f"scifact:{doc_id}",
                filename=f"scifact_doc_{doc_id}", page=(indices[0] + 1) if indices else 0,
                score=float(chunk.get("retrieval_score") or 0.0), document_id=str(doc_id),
                chunk_id=str(chunk["chunk_id"]), retrieval_score=chunk.get("retrieval_score"),
                retrieval_metadata={"chunk": {
                    "scifact_doc_id": doc_id, "scifact_sentence_indices": indices,
                    "scifact_sentence_texts": list(chunk.get("scifact_sentence_texts") or []),
                    "phase13_context_sentence_indices": context_indices,
                }},
            ))
        return RetrievalResponse(query=query, evidence=evidence, provider=self.name)


# --------------------------------------------------------------- B4 judge prompt

PROMPT_V1_SYSTEM = frozen_b._pairwise_judge_prompt(
    "q", frozen_b.analyze_verification_units("A uses B.")[0], "E1",
    EvidenceItem("x", "y"), frozen_b.analyze_verification_units("A uses B.")[0].components[0],
)[0]["content"]

PROMPT_V2_SYSTEM = (
    "Interpret evidence only; do NOT produce a final verdict. Assess only the single supplied evidence item "
    "against the single supplied verification component, and decide which one of three relations holds.\n"
    "SUPPORTS: the evidence entails the component. Someone who accepts the evidence must also accept the "
    "component, including its direction of effect, population, conditions and qualifiers.\n"
    "CONTRADICTS: the evidence directly conflicts with the component. It reports a finding about the same "
    "entities and relation that cannot be true at the same time as the component. Typical forms are: the "
    "opposite direction of effect (increases vs decreases, promotes vs inhibits, higher vs lower); a reported "
    "absence of an effect, difference or association where the component asserts one, or the reverse; an "
    "explicit denial of the component's relation; an incompatible quantity, timing or outcome for the same "
    "measurement. Different wording, terminology or level of detail does not prevent CONTRADICTS when the "
    "meanings are incompatible.\n"
    "NEUTRAL: the evidence is related but neither establishes nor refutes the component, for example because "
    "it concerns a different entity, population or outcome, gives only background, or is silent on the "
    "specific relation.\n"
    "Before answering NEUTRAL, explicitly check whether the evidence states the opposite of, or denies, the "
    "component's relation. Topical similarity alone is never SUPPORTS, and mere absence of information is "
    "NEUTRAL, not CONTRADICTS. Preserve dates, quantities, conditions, negation and relations exactly as "
    "included in the component. Return exactly one assessment for the supplied positional evidence and "
    "component indices. Allowed stance values: SUPPORTS, CONTRADICTS, NEUTRAL. Do not output a claim-level "
    "verdict. Return JSON only."
)


class PromptOverrideJudge:
    """Wraps the frozen judge provider and swaps ONLY the pairwise system message.

    The user message (schema + payload) and all generation settings are
    untouched. Any message that is not the exact frozen v1 system prompt raises.
    """

    def __init__(self, inner: Any, system_text: str):
        self.inner = inner
        self.system_text = system_text
        self.replacements = 0

    @property
    def calls(self) -> int:
        return self.inner.calls

    def generate(self, messages: Sequence[Mapping[str, str]], *, temperature: float = 0.0, timeout: int = 120) -> str:
        messages = [dict(m) for m in messages]
        if not messages or messages[0].get("role") != "system" or messages[0].get("content") != PROMPT_V1_SYSTEM:
            raise RuntimeError("PromptOverrideJudge received an unexpected system prompt; refusing to substitute.")
        messages[0]["content"] = self.system_text
        self.replacements += 1
        return self.inner.generate(messages, temperature=temperature, timeout=timeout)


@contextlib.contextmanager
def judge_prompt_override(system_text: str) -> Iterator[list[PromptOverrideJudge]]:
    """Make the frozen ``run_system_b`` build a judge whose system prompt is replaced.

    ``run_system_b`` imports ``OllamaCPUProvider`` from ``local_model`` at call
    time, so the factory is swapped for the duration of the block only.
    """
    from evaluation.experiments import local_model
    original = local_model.OllamaCPUProvider
    created: list[PromptOverrideJudge] = []

    def factory(*args: Any, **kwargs: Any) -> PromptOverrideJudge:
        judge = PromptOverrideJudge(original(*args, **kwargs), system_text)
        created.append(judge)
        return judge

    local_model.OllamaCPUProvider = factory  # type: ignore[assignment]
    try:
        yield created
    finally:
        local_model.OllamaCPUProvider = original  # type: ignore[assignment]


# --------------------------------------------------------------- B3 guards v2

_GREEK = {"α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "κ": "kappa", "μ": "mu", "ε": "epsilon"}
_NUMBER_WORDS = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen twenty".split())}
_UNIT_ALIASES = [
    (r"\b(\d[\d,.]*)\s*-?\s*(?:h|hr|hrs)\b", r"\1 hours"),
    (r"\b(\d[\d,.]*)\s*-?\s*(?:min|mins)\b", r"\1 minutes"),
    (r"\b(\d[\d,.]*)\s*-?\s*(?:s|sec|secs)\b", r"\1 seconds"),
    (r"\b(\d[\d,.]*)\s*(?:per\s*cent|percent)\b", r"\1%"),
    (r"\b(\d[\d,.]*)\s*-\s*(hours?|minutes?|seconds?|days?|weeks?|months?|years?)\b", r"\1 \2"),
]
_QUANTITY_SUFFIX = (r"(?:fold|times|years?|yrs?|days?|weeks?|months?|hours?|hrs?|h|minutes?|mins?|seconds?|"
                    r"secs?|doses?|mg|g|kg|µg|ug|ng|ml|l|mm|cm|mmol|nmol|µmol|umol|mol|iu|units?|percent|"
                    r"patients?|subjects?|participants?|people|persons?|cases?)\b")
_STOPWORDS = {"the", "a", "an", "of", "in", "on", "for", "to", "and", "or", "with", "by", "at", "from",
              "both", "either", "neither", "its", "their", "as", "into", "than"}
_CLAUSE_VERBS = re.compile(r"\b(?:is|are|was|were|be|been|being|has|have|had|does|do|did|can|could|may|"
                           r"might|will|would|should|shall)\b", re.I)
_CLAUSE_START = re.compile(r"^\s*(?:this|that|these|those|it|they|which|who|whom|whose)\b", re.I)
_LEADING_FUNCTION = re.compile(r"^(?:(?:the|a|an|with|in|of|on|for|by|both|either|and)\s+)+", re.I)


def _normalise_evidence_numbers(text: str) -> str:
    text = text.lower()
    for greek, name in _GREEK.items():
        text = text.replace(greek, name)
    text = re.sub(r"\b(" + "|".join(_NUMBER_WORDS) + r")\b", lambda m: _NUMBER_WORDS[m.group(1)], text)
    for pattern, repl in _UNIT_ALIASES:
        text = re.sub(pattern, repl, text)
    return text


def _embedded_number_anchors(text: str) -> set[str]:
    """G2: digits inside alphanumeric identifiers (e.g. MEK1/2, CIN2, Arg90, LATS1/2)."""
    anchors: set[str] = set()
    for token in re.findall(r"[A-Za-z]+\d[\w/]*|\d+[A-Za-z]+\w*", text):
        anchors.update(re.findall(r"\d+", token))
    return anchors


def _is_nomenclature_number(claim: str, start: int, end: int) -> bool:
    """G1: a number that is part of chemical/biological nomenclature in the claim.

    Matches locants and hyphen/prime-attached numbers (3,4-bisphosphate,
    4-phosphate, 0-UTRs, 5'-UTR, (3,4)P) and family indices following an
    uppercase gene/protein symbol (MEK 1/2, IL 6, P 2).
    """
    after = claim[end:end + 24]
    before = claim[max(0, start - 16):start]
    # Real quantities keep their requirement: 10-fold, 5-year, 3 hours, 10%, 75 nmol/liter ...
    if re.match(r"\s*-?\s*" + _QUANTITY_SUFFIX, after, re.I) or re.match(r"\s*(?:%|percent)", after, re.I):
        return False
    # Locants / hyphen- or prime-attached nomenclature: 3,4-bisphosphate, 4-phosphate, 0-UTRs, 5'-UTR.
    if re.match(r"(?:-|′|'|’)\s*[A-Za-z(]", after) or re.match(r"(?:-|′|'|’)\d", after):
        return True
    # Parenthesised locant followed by a symbol: (3,4)P
    if re.search(r"\(\s*$", before) and re.match(r"\)\s*[A-Za-z]", after):
        return True
    # Family index after an uppercase gene/protein symbol (>=2 chars) or a ")X" symbol: MEK 1/2, IL 6, (3,4)P 2.
    symbol = r"(?:(?:^|[\s(/-])[A-Z][A-Z0-9]{1,6}|\)[A-Z])"
    if re.search(symbol + r"\s+$", before) or re.search(symbol + r"\s+\d+/$", before):
        return True
    # Slash-joined family index glued to a symbol: LATS1/2, ERK1/2.
    if re.search(r"[A-Z][A-Za-z]{1,6}\d+/$", before):
        return True
    return False


def _value_occurrences(claim: str, value: str) -> list[tuple[int, int]]:
    core = re.match(r"\$?\s*\d[\d,]*(?:\.\d+)?", value.strip())
    if not core:
        return []
    pattern = r"(?<![\w.])" + re.escape(core.group(0).strip()) + r"(?![\d])"
    spans = []
    for m in re.finditer(pattern, claim):
        start, end = m.start(), m.end()
        # Judge a number inside a locant list ("(3,4)") as part of the whole list.
        while start > 0 and (claim[start - 1].isdigit() or (claim[start - 1] == "," and start > 1
                                                             and claim[start - 2].isdigit())):
            start -= 1
        while end < len(claim) and (claim[end].isdigit() or (claim[end] == "," and end + 1 < len(claim)
                                                             and claim[end + 1].isdigit())):
            end += 1
        spans.append((start, end))
    return spans


def hard_qualifier_gaps_v2(unit: Any, evidence: Sequence[EvidenceItem]) -> list[str]:
    """Frozen ``_hard_qualifier_gaps`` with rules G1-G4 (see validator_analysis.md)."""
    claim = unit.original_text
    corpus = "\n".join(item.text for item in evidence)
    corpus_norm_text = _normalise_evidence_numbers(corpus)
    corpus_norm = frozen_b._normalise(corpus_norm_text)
    evidence_numbers = frozen_b._numeric_anchors(corpus_norm_text) | _embedded_number_anchors(corpus)
    corpus_digits = re.sub(r"\D", "", corpus)
    missing: list[str] = []
    for date in unit.temporal_constraints:
        occurrences = _value_occurrences(claim, date)
        # G3: "1/2000" is a fraction denominator, not a year.
        if occurrences and all(claim[max(0, s - 1):s] == "/" for s, _ in occurrences):
            anchors = frozen_b._numeric_anchors(date)
            if anchors and not anchors.issubset(evidence_numbers):
                missing.append(f"quantity: {date}")
            continue
        date_norm = frozen_b._normalise(date)
        digits = re.sub(r"\D", "", date)
        if date_norm and date_norm not in corpus_norm and not (digits and digits in corpus_digits):
            missing.append(f"date/time qualifier: {date}")
    for quantity in unit.quantitative_constraints:
        occurrences = _value_occurrences(claim, quantity)
        if occurrences and all(_is_nomenclature_number(claim, s, e) for s, e in occurrences):
            continue  # G1
        anchors = frozen_b._numeric_anchors(_normalise_evidence_numbers(quantity))
        if anchors and not anchors.issubset(evidence_numbers):
            missing.append(f"quantity: {quantity}")
    return missing


def _content_tokens(text: str) -> set[str]:
    text = text.lower()
    for greek, name in _GREEK.items():
        text = text.replace(greek, name)
    tokens = re.findall(r"[a-z0-9]+", text)
    return {t[:-1] if len(t) > 3 and t.endswith("s") else t for t in tokens if t not in _STOPWORDS}


def coordinated_component_gaps_v2(unit: Any, evidence: Sequence[EvidenceItem]) -> list[str]:
    """G5: frozen coordinated-object presence check, applied to noun-phrase objects only.

    Clausal "objects" (a copula/auxiliary verb, or a leading pronoun or
    demonstrative) are left to the per-component judge stance instead of a
    lexical requirement. Noun-phrase objects still require ALL their content
    tokens (leading function words stripped, simple plural folding, Greek-letter
    names) to appear in the evidence.
    """
    if len(unit.objects) < 2:
        return []
    evidence_tokens = _content_tokens(" ".join(item.text for item in evidence))
    missing = []
    for obj in unit.objects:
        if _CLAUSE_START.search(obj) or _CLAUSE_VERBS.search(obj):
            continue
        anchor = _LEADING_FUNCTION.sub("", obj.strip())
        tokens = _content_tokens(anchor)
        if tokens and not tokens.issubset(evidence_tokens):
            missing.append(f"coordinated object: {obj}")
    return missing


_HARD_INVALID_PREFIXES = ("Invalid evidence reference", "Unknown evidence ID", "Invalid component reference",
                          "Assessment row", "Invalid stance")
_GUARD_PREFIX = "Uncovered required detail: "


def decide_verdict(base: frozen_b.ValidationResult, errors: Sequence[str]) -> tuple[str, list[str]]:
    """The frozen ``validate_judgement`` decision ladder, re-applied to a new error list."""
    n_components = len(base.coverage_matrix)
    all_supported = (len(base.supported_components) == n_components and not base.conflicting_components
                     and not errors)
    hard_invalid = any(str(e).startswith(_HARD_INVALID_PREFIXES) for e in errors)
    if hard_invalid or not base.assessment_complete:
        verdict = INSUFFICIENT_EVIDENCE
    elif base.conflicting_components:
        verdict = INSUFFICIENT_EVIDENCE
    elif base.contradicted_components:
        verdict = CONTRADICTED
    elif errors:
        verdict = INSUFFICIENT_EVIDENCE
    elif all_supported:
        verdict = SUPPORTED
    else:
        verdict = INSUFFICIENT_EVIDENCE
    overrides = list(errors)
    if base.conflicting_components:
        overrides.append("Evidence stances conflict on a component; verdict is INSUFFICIENT_EVIDENCE.")
    elif base.contradicted_components:
        overrides.append("At least one required component is contradicted by evidence.")
    elif not all_supported and not errors:
        overrides.append("Not every required component has supporting evidence.")
    return verdict, overrides


def validate_judgement_plus(unit: Any, data: Mapping[str, Any], evidence: Sequence[EvidenceItem], *,
                            guard_version: str = "v1") -> frozen_b.ValidationResult:
    """Frozen validator; with guard_version='v2' only the two lexical guards are replaced."""
    base = frozen_b.validate_judgement(unit, dict(data), list(evidence))
    if guard_version == "v1":
        return base
    if guard_version != "v2":
        raise ValueError(f"unknown guard_version {guard_version!r}")
    errors = [e for e in base.validation_errors if not str(e).startswith(_GUARD_PREFIX)]
    errors += [_GUARD_PREFIX + gap for gap in hard_qualifier_gaps_v2(unit, evidence)
               + coordinated_component_gaps_v2(unit, evidence)]
    verdict, overrides = decide_verdict(base, errors)
    return replace(base, verdict=verdict, overrides=overrides, validation_errors=errors)


def reaggregate_output(claim: str, assessments: Sequence[Mapping[str, Any]], judged_texts: Sequence[str], *,
                       guard_version: str) -> dict[str, Any]:
    """Recompute B's claim verdict from recorded pair stances (no inference).

    Valid because the lexical guards act after the judge: they never change
    which pairs are judged or what the judge returns. Mirrors B's use of the
    first verification unit.
    """
    unit = frozen_b.analyze_verification_units(claim)[0]
    evidence = [EvidenceItem(f"E{i}", text) for i, text in enumerate(judged_texts, start=1)]
    data = {"assessments": [dict(a) for a in assessments], "reason": ""}
    result = validate_judgement_plus(unit, data, evidence, guard_version=guard_version)
    return {"verdict": result.verdict, "validation_errors": list(result.validation_errors),
            "guard_messages": [e for e in result.validation_errors if str(e).startswith(_GUARD_PREFIX)]}
