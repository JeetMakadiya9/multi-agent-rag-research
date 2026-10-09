"""Domain-general evidence-core contracts.

The central distinction encoded here:

    RetrievedCandidate  — "a retriever returned this span for this query"
    Evidence            — "this span was admitted as a candidate evidentiary unit
                           for a specific claim, for a recorded reason"
    Judgement           — "producer P assessed evidence E against claim C as stance S"
    ClaimVerdict        — "policy Q aggregated judgements into a claim-level state"

Retrieval never implies evidence, and evidence never implies support. Support
exists only as a Judgement stance produced by an identified judge, and only a
policy turns judgements into a verdict.

All objects are immutable, storage-independent value objects with stable,
content-derived IDs and lossless dict round-trips.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from typing import Any, Mapping, Optional, Tuple, Type, TypeVar

T = TypeVar("T")


def stable_id(kind: str, *parts: Any) -> str:
    """Deterministic ID from a kind prefix and canonical identity parts."""
    payload = json.dumps([kind, *parts], ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return f"{kind}_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:20]}"


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- enums

class SourceType(str, Enum):
    PRIMARY_RESEARCH = "PRIMARY_RESEARCH"
    SYSTEMATIC_REVIEW = "SYSTEMATIC_REVIEW"
    NARRATIVE_REVIEW = "NARRATIVE_REVIEW"
    BENCHMARK = "BENCHMARK"
    OFFICIAL_DOCUMENTATION = "OFFICIAL_DOCUMENTATION"
    GOVERNMENT = "GOVERNMENT"
    REGULATORY = "REGULATORY"
    STANDARD = "STANDARD"
    COMPANY_ANNOUNCEMENT = "COMPANY_ANNOUNCEMENT"
    NEWS = "NEWS"
    TECHNICAL_BLOG = "TECHNICAL_BLOG"
    FORUM = "FORUM"
    PREPRINT = "PREPRINT"
    DATASET_RECORD = "DATASET_RECORD"
    UNKNOWN = "UNKNOWN"


class SpanKind(str, Enum):
    SENTENCE = "SENTENCE"
    PARAGRAPH = "PARAGRAPH"
    PASSAGE = "PASSAGE"
    TITLE = "TITLE"
    TABLE = "TABLE"
    TABLE_ROW = "TABLE_ROW"
    FIGURE_CAPTION = "FIGURE_CAPTION"
    LIST_ITEM = "LIST_ITEM"
    CODE = "CODE"
    OTHER = "OTHER"


class EvidenceRole(str, Enum):
    """Why a span became candidate evidence for a claim. Never a stance."""
    LEGACY_TOP_N_PRESENTED_TO_JUDGE = "LEGACY_TOP_N_PRESENTED_TO_JUDGE"
    EXTRACTED_FOR_CLAIM = "EXTRACTED_FOR_CLAIM"
    MANUALLY_SUPPLIED = "MANUALLY_SUPPLIED"
    SYNTHETIC_TEST = "SYNTHETIC_TEST"


class ClaimOrigin(str, Enum):
    USER = "USER"
    PLAN = "PLAN"
    EXTRACTED = "EXTRACTED"
    BENCHMARK = "BENCHMARK"
    LEGACY_SENTENCE_UNIT = "LEGACY_SENTENCE_UNIT"
    LEGACY_COMPONENT = "LEGACY_COMPONENT"
    SYNTHETIC_TEST = "SYNTHETIC_TEST"


class Stance(str, Enum):
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    NEUTRAL = "NEUTRAL"
    INVALID = "INVALID"      # the producer failed to give a usable stance; never support


class JudgeKind(str, Enum):
    LLM = "LLM"
    NLI = "NLI"
    HUMAN = "HUMAN"
    DETERMINISTIC = "DETERMINISTIC"


class SlotStatus(str, Enum):
    MATCHED = "MATCHED"
    CONFLICTING = "CONFLICTING"
    ABSENT = "ABSENT"
    NOT_ASSESSED = "NOT_ASSESSED"


class VerdictState(str, Enum):
    """Rich production states (inferred by future policies; Slice 1+)."""
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    MIXED_EVIDENCE = "MIXED_EVIDENCE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    UNVERIFIED = "UNVERIFIED"


class LegacyVerdict(str, Enum):
    """The frozen Experiment B three-class output, kept separate from VerdictState."""
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class LegacyRunStatus(str, Enum):
    """Frozen run_system_b outcome: a verdict was emitted, or the attempt failed."""
    VERDICT = "VERDICT"
    FAILED_INCOMPLETE_ASSESSMENT = "FAILED_INCOMPLETE_ASSESSMENT"


# --------------------------------------------------------------------------- serialization

def _to_plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {f.name: _to_plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _to_plain(v) for k, v in value.items()}
    return value


def _freeze(value: Any) -> Any:
    """Normalise JSON-like metadata so equality survives a round trip."""
    if isinstance(value, Mapping):
        return {str(k): _freeze(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_freeze(v) for v in value]
    return value


class Contract:
    """Mixin: dict round trip with enum/tuple/nested-contract reconstruction."""

    def to_dict(self) -> dict[str, Any]:
        return {"__type__": type(self).__name__, **_to_plain(self)}

    @classmethod
    def from_dict(cls: Type[T], data: Mapping[str, Any]) -> T:
        data = dict(data)
        kind = data.pop("__type__", cls.__name__)
        if kind != cls.__name__:
            raise ValueError(f"Expected {cls.__name__}, got {kind}")
        kwargs = {}
        hints = {f.name: f for f in fields(cls)}
        for name, value in data.items():
            if name not in hints:
                raise ValueError(f"Unknown field {name!r} for {cls.__name__}")
            kwargs[name] = _CONVERTERS.get((cls.__name__, name), lambda v: v)(value)
        return cls(**kwargs)


def _tuple(conv=lambda v: v):
    return lambda v: tuple(conv(x) for x in (v or ()))


def _opt(conv):
    return lambda v: None if v is None else conv(v)


# --------------------------------------------------------------------------- contracts

@dataclass(frozen=True)
class Provenance(Contract):
    """Who/what created a record, from which input, when."""
    producer: str
    producer_version: Optional[str] = None
    input_reference: Optional[str] = None
    created_at: Optional[str] = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "details", _freeze(self.details))


@dataclass(frozen=True)
class Source(Contract):
    source_id: str
    canonical_identity: str
    source_type: SourceType = SourceType.UNKNOWN
    publisher: Optional[str] = None
    authors: Tuple[str, ...] = ()
    publication_date: Optional[str] = None
    retrieval_date: Optional[str] = None
    independence_group: Optional[str] = None   # sources sharing a group are NOT independent
    metadata: Mapping[str, Any] = field(default_factory=dict)
    provenance: Optional[Provenance] = None

    def __post_init__(self):
        object.__setattr__(self, "authors", tuple(self.authors))
        object.__setattr__(self, "metadata", _freeze(self.metadata))

    @staticmethod
    def make_id(canonical_identity: str) -> str:
        return stable_id("src", canonical_identity)


@dataclass(frozen=True)
class DocumentVersion(Contract):
    document_version_id: str
    source_id: str
    content_hash: Optional[str] = None
    retrieved_at: Optional[str] = None
    previous_version_id: Optional[str] = None
    title: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    structure: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "metadata", _freeze(self.metadata))
        object.__setattr__(self, "structure", _freeze(self.structure))

    @staticmethod
    def make_id(source_id: str, version_key: Optional[str]) -> str:
        return stable_id("docv", source_id, version_key)


@dataclass(frozen=True)
class Span(Contract):
    span_id: str
    document_version_id: str
    text: str
    kind: SpanKind = SpanKind.OTHER
    section_path: Tuple[str, ...] = ()
    paragraph_index: Optional[int] = None
    sentence_indices: Tuple[int, ...] = ()
    start_offset: Optional[int] = None
    end_offset: Optional[int] = None
    parent_span_id: Optional[str] = None
    previous_span_id: Optional[str] = None
    next_span_id: Optional[str] = None

    def __post_init__(self):
        object.__setattr__(self, "section_path", tuple(self.section_path))
        object.__setattr__(self, "sentence_indices", tuple(self.sentence_indices))

    @staticmethod
    def make_id(document_version_id: str, locator: Any) -> str:
        return stable_id("span", document_version_id, locator)


@dataclass(frozen=True)
class RetrievalProvenance(Contract):
    query: str
    method: str
    rank: int
    retrieval_run_id: str
    scores: Mapping[str, Optional[float]] = field(default_factory=dict)
    reason: Optional[str] = None
    provider: Optional[str] = None

    def __post_init__(self):
        object.__setattr__(self, "scores", _freeze(self.scores))


@dataclass(frozen=True)
class RetrievedCandidate(Contract):
    """A retrieval hit. Carries NO evidentiary or stance meaning."""
    candidate_id: str
    span_id: str
    document_version_id: str
    source_id: str
    presented_text: str            # exact text the retriever emitted (may include title prefixes)
    retrieval: RetrievalProvenance
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "metadata", _freeze(self.metadata))


@dataclass(frozen=True)
class AdmissionRecord(Contract):
    """Why a span became candidate evidence for a claim (not whether it supports it)."""
    role: EvidenceRole
    reason: str
    admitted_by: str
    presentation_index: Optional[str] = None   # e.g. legacy "E1"
    candidate_id: Optional[str] = None


@dataclass(frozen=True)
class Evidence(Contract):
    """A candidate evidentiary unit for a specific claim. Never a stance."""
    evidence_id: str
    claim_id: str
    span_id: str
    document_version_id: str
    source_id: str
    text: str                       # exact text a judge would see
    admission: AdmissionRecord
    context: Optional[str] = None
    retrieval: Optional[RetrievalProvenance] = None
    attributes: Mapping[str, Any] = field(default_factory=dict)   # entities, quantities, dates ... (future)
    extraction: Optional[Provenance] = None

    def __post_init__(self):
        object.__setattr__(self, "attributes", _freeze(self.attributes))

    @staticmethod
    def make_id(claim_id: str, span_id: str, admission_key: str) -> str:
        return stable_id("ev", claim_id, span_id, admission_key)


@dataclass(frozen=True)
class Claim(Contract):
    claim_id: str
    text: str
    origin: ClaimOrigin
    parent_claim_id: Optional[str] = None
    decomposition_reason: Optional[str] = None
    subject: Optional[str] = None
    predicate: Optional[str] = None
    object: Optional[str] = None
    comparison: Optional[str] = None
    quantity: Tuple[str, ...] = ()
    unit: Optional[str] = None
    time: Tuple[str, ...] = ()
    population: Optional[str] = None
    condition: Tuple[str, ...] = ()
    polarity: Optional[str] = None
    modality: Optional[str] = None
    causal_type: Optional[str] = None
    required_slots: Tuple[str, ...] = ()
    semantic_unity: bool = True     # True: qualifiers stay attached; do not split further
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        for name in ("quantity", "time", "condition", "required_slots"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "metadata", _freeze(self.metadata))

    @staticmethod
    def make_id(*identity: Any) -> str:
        return stable_id("clm", *identity)


@dataclass(frozen=True)
class JudgeIdentity(Contract):
    kind: JudgeKind
    name: str
    model: Optional[str] = None
    model_version: Optional[str] = None
    prompt_version: Optional[str] = None
    settings: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "settings", _freeze(self.settings))


@dataclass(frozen=True)
class Judgement(Contract):
    """One producer's stance on one (claim, evidence) pair."""
    judgement_id: str
    claim_id: str
    evidence_id: str
    stance: Stance
    judge: JudgeIdentity
    slot_coverage: Mapping[str, SlotStatus] = field(default_factory=dict)
    confidence: Optional[float] = None
    rationale: Optional[str] = None
    created_at: Optional[str] = None
    provenance: Optional[Provenance] = None

    def __post_init__(self):
        object.__setattr__(self, "slot_coverage",
                           {str(k): SlotStatus(v) for k, v in dict(self.slot_coverage).items()})

    @staticmethod
    def make_id(claim_id: str, evidence_id: str, judge: JudgeIdentity, run_key: str = "") -> str:
        return stable_id("jdg", claim_id, evidence_id, judge.kind.value, judge.name, judge.model,
                         judge.prompt_version, run_key)


@dataclass(frozen=True)
class ComponentCoverage(Contract):
    component_claim_id: str
    status: str                      # policy-specific (legacy: SUPPORTED/CONTRADICTED/CONFLICTING/UNSUPPORTED/INCOMPLETE)
    supporting_evidence_ids: Tuple[str, ...] = ()
    contradicting_evidence_ids: Tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "supporting_evidence_ids", tuple(self.supporting_evidence_ids))
        object.__setattr__(self, "contradicting_evidence_ids", tuple(self.contradicting_evidence_ids))


@dataclass(frozen=True)
class LegacyOutcome(Contract):
    run_status: LegacyRunStatus
    verdict: Optional[LegacyVerdict]   # None when the legacy run failed
    diagnostics: Tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))


@dataclass(frozen=True)
class ClaimVerdict(Contract):
    """Policy output. Rich state and legacy outcome are independent fields."""
    verdict_id: str
    claim_id: str
    policy: str
    policy_version: str
    judgement_ids: Tuple[str, ...]
    state: Optional[VerdictState] = None          # rich production state (Slice 1+)
    legacy: Optional[LegacyOutcome] = None        # frozen Experiment B outcome (legacy policy only)
    component_coverage: Tuple[ComponentCoverage, ...] = ()
    uncovered_slots: Tuple[str, ...] = ()
    notes: Tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "judgement_ids", tuple(self.judgement_ids))
        object.__setattr__(self, "component_coverage", tuple(self.component_coverage))
        object.__setattr__(self, "uncovered_slots", tuple(self.uncovered_slots))
        object.__setattr__(self, "notes", tuple(self.notes))
        if self.state is None and self.legacy is None:
            raise ValueError("A ClaimVerdict must carry a rich state, a legacy outcome, or both.")
        # Contract properties for rich states (future policies); the legacy outcome is untouched.
        if self.state is VerdictState.SUPPORTED and self.uncovered_slots:
            raise ValueError("A SUPPORTED verdict cannot have uncovered required slots.")
        if self.state in (VerdictState.SUPPORTED, VerdictState.PARTIALLY_SUPPORTED, VerdictState.CONTRADICTED,
                          VerdictState.MIXED_EVIDENCE) and not self.judgement_ids:
            raise ValueError(f"A {self.state.value} verdict requires at least one judgement.")


_CONVERTERS = {
    ("Source", "source_type"): SourceType,
    ("Source", "provenance"): _opt(Provenance.from_dict),
    ("Span", "kind"): SpanKind,
    ("RetrievedCandidate", "retrieval"): RetrievalProvenance.from_dict,
    ("AdmissionRecord", "role"): EvidenceRole,
    ("Evidence", "admission"): AdmissionRecord.from_dict,
    ("Evidence", "retrieval"): _opt(RetrievalProvenance.from_dict),
    ("Evidence", "extraction"): _opt(Provenance.from_dict),
    ("Claim", "origin"): ClaimOrigin,
    ("JudgeIdentity", "kind"): JudgeKind,
    ("Judgement", "stance"): Stance,
    ("Judgement", "judge"): JudgeIdentity.from_dict,
    ("Judgement", "provenance"): _opt(Provenance.from_dict),
    ("LegacyOutcome", "run_status"): LegacyRunStatus,
    ("LegacyOutcome", "verdict"): _opt(LegacyVerdict),
    ("ClaimVerdict", "state"): _opt(VerdictState),
    ("ClaimVerdict", "legacy"): _opt(LegacyOutcome.from_dict),
    ("ClaimVerdict", "component_coverage"): _tuple(ComponentCoverage.from_dict),
}

CONTRACT_TYPES = {cls.__name__: cls for cls in (
    Provenance, Source, DocumentVersion, Span, RetrievalProvenance, RetrievedCandidate, AdmissionRecord,
    Evidence, Claim, JudgeIdentity, Judgement, ComponentCoverage, LegacyOutcome, ClaimVerdict)}


def from_dict(data: Mapping[str, Any]):
    """Reconstruct any contract from its to_dict() output."""
    return CONTRACT_TYPES[data["__type__"]].from_dict(data)
