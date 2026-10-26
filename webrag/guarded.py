"""``GuardedEvidenceStore``: the evidence store with the opt-in prompt-injection screen (``webrag.injection``).

A subclass, so ``webrag/evidence.py`` stays byte-identical to the version pinned by the heldout3 protocols. Only
``rank`` differs, and only in what it does with passages the screen matches: "flag" records them in ``flagged``;
"exclude" also withholds them from every caller of ``rank`` (so no model reads them) and refills the slot from the
same source by a wider search, never with another flagged passage. "off" ranks exactly as ``EvidenceStore``.
"""
from __future__ import annotations

import hashlib
from typing import Dict, List, Sequence

from docqa.index import EvidenceIndex

from .evidence import EvidenceStore, Passage, cap_per_origin
from .injection import MODES, scan
from .researcher import Source


class GuardedEvidenceStore(EvidenceStore):
    def __init__(self, sources: Sequence[Source], models=None, guard: str = "flag"):
        if guard not in MODES:
            raise ValueError(f"guard must be one of {MODES}")
        self.guard = guard
        self.flagged: Dict[str, Dict[str, object]] = {}      # passage key -> {"source_id", "page", "rules"}
        super().__init__(sources, models)

    def _search(self, source_id: str, index: EvidenceIndex, query: str, k: int) -> List[Passage]:
        out = []
        for evidence in index.search(query, k=k):                       # as in EvidenceStore.rank
            score = evidence.rerank_score if evidence.rerank_score is not None else evidence.score
            digest = hashlib.sha1(evidence.text.encode("utf-8")).hexdigest()[:8]
            out.append(Passage(f"{source_id}:p{evidence.page}:{digest}", source_id, evidence.page, evidence.text,
                               float(score)))
        return out

    def _screened(self, passages: List[Passage]) -> List[Passage]:
        """Record instruction-like passages; in "exclude" mode also drop them."""
        kept = []
        for p in passages:
            rules = scan(p.text)
            if rules:
                self.flagged[p.key] = {"source_id": p.source_id, "page": p.page, "rules": rules}
                if self.guard == "exclude":
                    continue
            kept.append(p)
        return kept

    def rank(self, query: str, k: int = 6, per_source: int = 2) -> List[Passage]:
        if self.guard == "off":
            return super().rank(query, k, per_source)
        candidates: List[Passage] = []
        for source_id, index in self.indexes.items():
            found = self._search(source_id, index, query, per_source)
            kept = self._screened(found)
            if self.guard == "exclude" and len(kept) < len(found):
                # refill from the same source, never with a flagged passage
                wider = self._screened(self._search(source_id, index, query, per_source + len(found) - len(kept)))
                kept = wider[:per_source]
            candidates.extend(kept)
        candidates.sort(key=lambda p: (-p.score, p.source_id))
        return cap_per_origin(candidates, self.sources, k, per_source)
