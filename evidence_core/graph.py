"""Minimal, storage-independent registry of evidence-core objects and their relations.

Relations are carried by IDs on the contracts themselves; this module only
indexes them and enforces referential integrity. It is not a graph database.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

from .contracts import (
    Claim, ClaimVerdict, Contract, DocumentVersion, Evidence, Judgement, RetrievedCandidate, Source, Span,
    from_dict,
)

_ID_FIELD = {
    Source: "source_id", DocumentVersion: "document_version_id", Span: "span_id",
    RetrievedCandidate: "candidate_id", Evidence: "evidence_id", Claim: "claim_id",
    Judgement: "judgement_id", ClaimVerdict: "verdict_id",
}


class IntegrityError(ValueError):
    pass


class EvidenceGraph:
    def __init__(self) -> None:
        self._objects: dict[type, dict[str, Contract]] = {cls: {} for cls in _ID_FIELD}

    # ----------------------------------------------------------------- mutation
    def add(self, obj: Contract) -> Contract:
        cls = type(obj)
        if cls not in _ID_FIELD:
            raise TypeError(f"Unsupported object type {cls.__name__}")
        key = getattr(obj, _ID_FIELD[cls])
        existing = self._objects[cls].get(key)
        if existing is not None:
            if existing != obj:
                raise IntegrityError(f"{cls.__name__} {key} already exists with different content")
            return existing
        self._objects[cls][key] = obj
        return obj

    def add_all(self, objects: Iterable[Contract]) -> None:
        for obj in objects:
            self.add(obj)

    # ----------------------------------------------------------------- lookup
    def get(self, cls: type, key: str):
        return self._objects[cls][key]

    def all(self, cls: type) -> list:
        return list(self._objects[cls].values())

    def children(self, claim_id: str) -> list[Claim]:
        return [c for c in self._objects[Claim].values() if c.parent_claim_id == claim_id]

    def evidence_for(self, claim_id: str) -> list[Evidence]:
        return [e for e in self._objects[Evidence].values() if e.claim_id == claim_id]

    def judgements_for(self, claim_id: str) -> list[Judgement]:
        return [j for j in self._objects[Judgement].values() if j.claim_id == claim_id]

    def source_of(self, evidence: Evidence) -> Source:
        return self._objects[Source][evidence.source_id]

    def independence_key(self, evidence: Evidence) -> str:
        """Evidence sharing this key is not independent (same group, else same source)."""
        source = self.source_of(evidence)
        return source.independence_group or source.source_id

    def evidence_by_independence(self, evidence: Iterable[Evidence]) -> Mapping[str, list[Evidence]]:
        groups: dict[str, list[Evidence]] = defaultdict(list)
        for item in evidence:
            groups[self.independence_key(item)].append(item)
        return dict(groups)

    # ----------------------------------------------------------------- integrity
    def validate(self) -> None:
        o = self._objects
        for dv in o[DocumentVersion].values():
            self._ref(Source, dv.source_id, dv)
            if dv.previous_version_id:
                self._ref(DocumentVersion, dv.previous_version_id, dv)
        for sp in o[Span].values():
            self._ref(DocumentVersion, sp.document_version_id, sp)
            for rel in (sp.parent_span_id, sp.previous_span_id, sp.next_span_id):
                if rel:
                    self._ref(Span, rel, sp)
        for cand in o[RetrievedCandidate].values():
            self._chain(cand.span_id, cand.document_version_id, cand.source_id, cand)
        for ev in o[Evidence].values():
            self._chain(ev.span_id, ev.document_version_id, ev.source_id, ev)
            self._ref(Claim, ev.claim_id, ev)
            if ev.admission.candidate_id:
                self._ref(RetrievedCandidate, ev.admission.candidate_id, ev)
        for cl in o[Claim].values():
            if cl.parent_claim_id:
                self._ref(Claim, cl.parent_claim_id, cl)
        for jd in o[Judgement].values():
            self._ref(Claim, jd.claim_id, jd)
            ev = self._ref(Evidence, jd.evidence_id, jd)
            if not self._claim_in_scope(ev.claim_id, jd.claim_id):
                raise IntegrityError(f"Judgement {jd.judgement_id} pairs claim {jd.claim_id} with evidence "
                                     f"admitted for unrelated claim {ev.claim_id}")
        for vd in o[ClaimVerdict].values():
            self._ref(Claim, vd.claim_id, vd)
            for jid in vd.judgement_ids:
                self._ref(Judgement, jid, vd)

    def _claim_in_scope(self, evidence_claim_id: str, judged_claim_id: str) -> bool:
        """Evidence admitted for a claim may be judged against that claim or its descendants."""
        current = judged_claim_id
        while current:
            if current == evidence_claim_id:
                return True
            current = self._objects[Claim][current].parent_claim_id
        return False

    def _chain(self, span_id, docv_id, source_id, owner):
        span = self._ref(Span, span_id, owner)
        docv = self._ref(DocumentVersion, docv_id, owner)
        self._ref(Source, source_id, owner)
        if span.document_version_id != docv_id or docv.source_id != source_id:
            raise IntegrityError(f"{type(owner).__name__} has inconsistent span/document/source references")

    def _ref(self, cls, key, owner):
        try:
            return self._objects[cls][key]
        except KeyError:
            raise IntegrityError(f"{type(owner).__name__} references missing {cls.__name__} {key}") from None

    # ----------------------------------------------------------------- serialization
    def to_dict(self) -> dict[str, Any]:
        return {cls.__name__: [obj.to_dict() for obj in objs.values()] for cls, objs in self._objects.items()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvidenceGraph":
        graph = cls()
        for items in data.values():
            for item in items:
                graph.add(from_dict(item))
        return graph
