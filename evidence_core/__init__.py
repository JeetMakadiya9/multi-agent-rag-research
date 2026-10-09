"""Evidence core (Slice 0): domain-general contracts for the research co-pilot.

RETRIEVAL RESULT != EVIDENCE != SUPPORT.

See evidence_core/README.md. Nothing in this package changes retrieval,
prompts, models or the frozen Experiment B pipeline.
"""
from .contracts import (  # noqa: F401
    AdmissionRecord, Claim, ClaimOrigin, ClaimVerdict, ComponentCoverage, DocumentVersion, Evidence, EvidenceRole,
    JudgeIdentity, JudgeKind, Judgement, LegacyOutcome, LegacyRunStatus, LegacyVerdict, Provenance,
    RetrievalProvenance, RetrievedCandidate, SlotStatus, Source, SourceType, Span, SpanKind, Stance, VerdictState,
    from_dict, stable_id,
)
from .graph import EvidenceGraph, IntegrityError  # noqa: F401
