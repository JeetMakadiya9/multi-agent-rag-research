"""Conservative, deterministic analysis into verification units.

The analyzer preserves each sentence as a semantic unit by default. It only
extracts fields when simple surface rules provide a plausible structure; the
original text is always retained and uncertain structure is left unset.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class VerificationComponent:
    """A deterministic coverage requirement within one verification unit."""

    component_id: str
    text: str
    role: str = "proposition"
    parent_unit_id: str = ""
    required: bool = True
    condition: Optional[str] = None
    qualifiers: list[str] = field(default_factory=list)
    relation: Optional[str] = None
    subject: Optional[str] = None
    object: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationUnit:
    """Smallest safe verification proposition, retaining dependencies."""

    id: str
    original_text: str
    subject: Optional[str] = None
    predicate: Optional[str] = None
    objects: list[str] = field(default_factory=list)
    qualifiers: list[str] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    temporal_constraints: list[str] = field(default_factory=list)
    quantitative_constraints: list[str] = field(default_factory=list)
    negation: Optional[str] = None
    causal_relation: Optional[str] = None
    comparison_relation: Optional[str] = None
    dependencies: list[str] = field(default_factory=list)
    required_evidence: list[str] = field(default_factory=list)
    status: str = "unverified"
    uncertainties: list[str] = field(default_factory=list)
    components: list[VerificationComponent] = field(default_factory=list)

    @property
    def text(self) -> str:
        """Compatibility alias for callers that expect claim-like text."""
        return self.original_text

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_DATE_PATTERNS = (
    re.compile(r"\b(?:19|20)\d{2}(?:[-/]\d{1,2}(?:[-/]\d{1,2})?)?\b"),
    re.compile(r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(?:19|20)\d{2}\b", re.I),
    re.compile(r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:,?\s+(?:19|20)\d{2})?\b", re.I),
)
_QUANTITY_RE = re.compile(
    r"(?<!\w)(?:\$\s*)?\d[\d,]*(?:\.\d+)?\s*(?:%|percent|percentage|"
    r"billion|million|thousand|ms|milliseconds?|seconds?|minutes?|hours?|"
    r"GB|MB|KB|TB|images?|pages?|documents?|samples?|records?|items?|"
    r"times|per\s+minute|per\s+hour|degrees?\s+Celsius)?\b", re.I
)
_CONDITION_RE = re.compile(
    r"\b(?:only\s+if|provided\s+that|as\s+long\s+as|if|unless|when)\b"
    r"[^,;.!?]*", re.I
)
_CAUSAL_RE = re.compile(
    r"\b(?:because|since|therefore|thus|so|as\s+a\s+result|leading\s+to|"
    r"resulting\s+in|due\s+to|thereby)\b", re.I
)
_COMPARISON_RE = re.compile(
    r"\b(?:higher|lower|greater|less|faster|slower|more\s+accurate|"
    r"less\s+accurate|more\s+efficient|less\s+efficient|fewer|more)\b"
    r"[^,;.!?]*?\bthan\b[^,;.!?]*", re.I
)
_PREDICATE_RE = re.compile(
    r"\b(uses|used|contains|contained|combines|combined|includes|included|"
    r"stores|stored|retrieves|retrieved|performs|performed|supports|supported|"
    r"acquired|acquires|partnered|partners|achieved|achieves|reached|reaches|"
    r"processes|processed|indexes|indexed|reranks|reranked|fuses|fused|"
    r"consumes|consume|requires|required|reduces|reduced|increases|increased|"
    r"decreases|decreased|improves|improved|causes|caused|returns|returned|"
    r"applies|applied|activates|activated|remains|remained)\b", re.I
)


def _sentences(text: str) -> list[str]:
    """Split at clear sentence punctuation; avoid decimals and abbreviations."""
    cleaned = text.strip()
    if not cleaned:
        return []
    return [piece.strip() for piece in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"“])", cleaned) if piece.strip()]


def _objects(value: str) -> list[str]:
    """Split a clearly coordinated object list while keeping predicate context."""
    value = re.sub(r"[.!?]+$", "", value).strip()
    # Split a comma list only if its final coordinator is present.
    if re.search(r",\s+and\s+|\s+and\s+", value, re.I):
        bits = re.split(r",\s*(?:and\s+)?|\s+and\s+", value, flags=re.I)
        cleaned = [b.strip(" ,") for b in bits if b.strip(" ,")]
        contains_new_predicate = any(_PREDICATE_RE.search(bit) for bit in cleaned[1:])
        if 2 <= len(cleaned) <= 8 and not contains_new_predicate:
            return cleaned
    return [value] if value else []


def analyze_verification_units(answer: str) -> list[VerificationUnit]:
    """Create safe sentence-level units and attach observable qualifiers.

    No LLM or external parser is used. Coordinated objects remain in one
    unit. Multi-clause sentences are conservatively retained together unless
    sentence punctuation already separates them.
    """
    units: list[VerificationUnit] = []
    for number, sentence in enumerate(_sentences(answer), start=1):
        unit = VerificationUnit(id=f"U{number}", original_text=sentence)
        for pattern in _DATE_PATTERNS:
            unit.temporal_constraints.extend(m.group(0) for m in pattern.finditer(sentence))
        unit.temporal_constraints = list(dict.fromkeys(unit.temporal_constraints))
        date_spans = [(m.start(), m.end()) for pattern in _DATE_PATTERNS for m in pattern.finditer(sentence)]
        quantities = []
        for match in _QUANTITY_RE.finditer(sentence):
            if any(match.start() < end and match.end() > start for start, end in date_spans):
                continue
            value = match.group(0).strip()
            suffix = re.match(r"\s*(%|percent|percentage|billion|million|thousand|ms|milliseconds?|seconds?|minutes?|hours?|GB|MB|KB|TB|images?|pages?|documents?|samples?|records?|items?|times|per\s+minute|per\s+hour|degrees?\s+Celsius)(?=\b|\s|$|[.,;!?])", sentence[match.end():], re.I)
            if suffix:
                value += suffix.group(0).strip()
            if value:
                quantities.append(value)
        unit.quantitative_constraints = list(dict.fromkeys(quantities))
        unit.conditions = list(dict.fromkeys(m.group(0).strip(" ,") for m in _CONDITION_RE.finditer(sentence)))

        negation = re.search(r"\b(?:does\s+not|do\s+not|did\s+not|is\s+not|are\s+not|was\s+not|were\s+not|never|neither|nor|without|no)\b[^,;.!?]*", sentence, re.I)
        if negation:
            unit.negation = negation.group(0).strip()

        causal = _CAUSAL_RE.search(sentence)
        if causal:
            unit.causal_relation = sentence
            unit.dependencies.append(f"causal dependency marked by '{causal.group(0)}'")

        comparison = _COMPARISON_RE.search(sentence)
        if comparison:
            unit.comparison_relation = comparison.group(0).strip()
            unit.dependencies.append("comparison direction must remain attached")

        if unit.conditions:
            unit.dependencies.append("proposition depends on the stated condition")

        # Safe shallow parse: only known predicate forms, with no attempt to
        # resolve pronouns or infer omitted subjects.
        match = _PREDICATE_RE.search(sentence)
        if match:
            prefix = sentence[:match.start()].strip(" ,")
            # For a leading conditional clause, inspect the consequent.
            if unit.conditions and "," in sentence and match.start() > sentence.find(","):
                prefix = sentence[sentence.find(",") + 1:match.start()].strip(" ,")
            subject = re.sub(r"^(?:the|a|an)\s+", "", prefix, flags=re.I).strip()
            predicate = match.group(0)
            tail = sentence[match.end():].strip(" ,")
            if subject and len(subject.split()) <= 10 and not re.search(r"\b(?:it|they|them|this|that|which|these|those)\b", subject, re.I):
                unit.subject = subject
                unit.predicate = predicate
                unit.objects = _objects(tail)
            else:
                unit.uncertainties.append("subject or predicate structure is ambiguous")
        else:
            unit.uncertainties.append("subject/predicate structure left unparsed")

        if re.search(r"\b(?:it|they|them|this|that|which|these|those)\b", sentence, re.I):
            unit.uncertainties.append("pronoun or anaphora requires contextual adjudication")
        if unit.conditions or unit.causal_relation or unit.comparison_relation or unit.negation:
            unit.qualifiers.extend(unit.conditions)
            unit.qualifiers.extend(unit.temporal_constraints)
            unit.qualifiers.extend(unit.quantitative_constraints)
            if unit.negation:
                unit.qualifiers.append(unit.negation)
            if unit.causal_relation:
                unit.qualifiers.append("causal relationship: " + unit.causal_relation)
            if unit.comparison_relation:
                unit.qualifiers.append("comparison: " + unit.comparison_relation)
        unit.required_evidence = list(dict.fromkeys(
            unit.objects + unit.temporal_constraints + unit.quantitative_constraints + unit.conditions
            + ([unit.negation] if unit.negation else [])
            + ([unit.comparison_relation] if unit.comparison_relation else [])
            + ([unit.causal_relation] if unit.causal_relation else [])
        ))
        # A coordinated object list creates explicit coverage requirements.
        # All unit-level qualifiers remain attached to each requirement so
        # dates, conditions, quantities and relations cannot be detached.
        component_objects = unit.objects if len(unit.objects) > 1 else [None]
        qualifier_context = list(dict.fromkeys(
            unit.conditions + unit.temporal_constraints + unit.quantitative_constraints
            + ([unit.negation] if unit.negation else [])
            + ([unit.causal_relation] if unit.causal_relation else [])
            + ([unit.comparison_relation] if unit.comparison_relation else [])
        ))
        for component_number, obj in enumerate(component_objects, start=1):
            if obj is None or not unit.subject or not unit.predicate:
                component_text = sentence
            else:
                component_text = f"{unit.subject} {unit.predicate} {obj}"
                if qualifier_context:
                    component_text += " (with required context: " + "; ".join(qualifier_context) + ")"
            unit.components.append(VerificationComponent(
                component_id=f"C{component_number}", text=component_text,
                role="coordinated_object" if obj is not None else "whole_proposition",
                parent_unit_id=unit.id, condition="; ".join(unit.conditions) or None,
                qualifiers=list(qualifier_context), relation=unit.predicate,
                subject=unit.subject, object=obj,
            ))
        units.append(unit)
    return units


__all__ = ["VerificationComponent", "VerificationUnit", "analyze_verification_units"]
