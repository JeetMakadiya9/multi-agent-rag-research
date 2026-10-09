"""Slice 0 contract tests: round trips, identity, provenance, and semantic boundaries."""
from __future__ import annotations

import json

import pytest

from evidence_core import (
    AdmissionRecord, Claim, ClaimOrigin, ClaimVerdict, DocumentVersion, Evidence, EvidenceGraph, EvidenceRole,
    IntegrityError, JudgeIdentity, JudgeKind, Judgement, LegacyOutcome, LegacyRunStatus, LegacyVerdict, Provenance,
    RetrievalProvenance, RetrievedCandidate, SlotStatus, Source, SourceType, Span, SpanKind, Stance, VerdictState,
    from_dict,
)
from evidence_core.adapters.retrieval import adapt_recorded_refs, adapt_retrieval_response
from evidence_core.policies import LegacyAnySupportsPolicy


# --------------------------------------------------------------------------- builders

def source(name, group=None, kind=SourceType.PRIMARY_RESEARCH):
    return Source(source_id=Source.make_id(name), canonical_identity=name, source_type=kind, independence_group=group,
                  authors=("A. Author",), metadata={"venue": "X"},
                  provenance=Provenance(producer="test", details={"k": [1, 2]}))


def docv(src, version="v1"):
    return DocumentVersion(document_version_id=DocumentVersion.make_id(src.source_id, version), source_id=src.source_id,
                           content_hash=version, title="T")


def span(dv, text, idx):
    return Span(span_id=Span.make_id(dv.document_version_id, ("s", idx)), document_version_id=dv.document_version_id,
                text=text, kind=SpanKind.SENTENCE, sentence_indices=(idx,), section_path=("Results",))


def claim(text, parent=None, **slots):
    return Claim(claim_id=Claim.make_id("test", text, parent), text=text, origin=ClaimOrigin.SYNTHETIC_TEST,
                 parent_claim_id=parent, **slots)


def evidence(cl, sp, dv, src, key="k"):
    return Evidence(evidence_id=Evidence.make_id(cl.claim_id, sp.span_id, key), claim_id=cl.claim_id,
                    span_id=sp.span_id, document_version_id=dv.document_version_id, source_id=src.source_id,
                    text=sp.text, admission=AdmissionRecord(EvidenceRole.SYNTHETIC_TEST, "test fixture", "test"),
                    retrieval=RetrievalProvenance(query="q", method="hybrid", rank=1, retrieval_run_id="r",
                                                  scores={"rerank": 1.5}),
                    extraction=Provenance(producer="test-extractor", details={"window": 0}))


JUDGES = {
    "llm": JudgeIdentity(JudgeKind.LLM, "pairwise", model="qwen3:4b", prompt_version="v1"),
    "nli": JudgeIdentity(JudgeKind.NLI, "nli-crossencoder", model="some-nli", model_version="rev1"),
    "human": JudgeIdentity(JudgeKind.HUMAN, "annotator-1"),
}


def judgement(cl, ev, stance, judge="human", **kw):
    j = JUDGES[judge]
    return Judgement(judgement_id=Judgement.make_id(cl.claim_id, ev.evidence_id, j), claim_id=cl.claim_id,
                     evidence_id=ev.evidence_id, stance=stance, judge=j,
                     provenance=Provenance(producer=j.name, details={"raw": "{}"}), **kw)


def world():
    s = source("doi:10/abc")
    d = docv(s)
    c = claim("Drug X reduces mortality.")
    sp = span(d, "Drug X reduced mortality in trial T.", 3)
    e = evidence(c, sp, d, s)
    g = EvidenceGraph()
    g.add_all([s, d, sp, c, e])
    return g, s, d, c, sp, e


# --------------------------------------------------------------------------- 1-5: round trips, ids, provenance

def test_every_contract_round_trips_through_json():
    g, s, d, c, sp, e = world()
    child = claim("Drug X reduces mortality in adults.", parent=c.claim_id, population="adults",
                  required_slots=("population",))
    j = judgement(child, e, Stance.SUPPORTS, slot_coverage={"population": SlotStatus.ABSENT}, confidence=0.4,
                  rationale="r")
    v = ClaimVerdict(verdict_id="v1", claim_id=child.claim_id, policy="p", policy_version="1",
                     judgement_ids=(j.judgement_id,), state=VerdictState.PARTIALLY_SUPPORTED,
                     uncovered_slots=("population",))
    cand = RetrievedCandidate(candidate_id="c1", span_id=sp.span_id, document_version_id=d.document_version_id,
                              source_id=s.source_id, presented_text="Title: T. " + sp.text,
                              retrieval=e.retrieval, metadata={"chunk_id": "9"})
    for obj in (s, d, sp, c, e, child, j, v, cand):
        assert from_dict(json.loads(json.dumps(obj.to_dict()))) == obj


def test_ids_are_stable_and_content_derived():
    _, s, d, c, sp, e = world()
    _, s2, d2, c2, sp2, e2 = world()
    assert (s.source_id, d.document_version_id, sp.span_id, c.claim_id, e.evidence_id) == \
           (s2.source_id, d2.document_version_id, sp2.span_id, c2.claim_id, e2.evidence_id)
    assert Source.make_id("doi:10/abc") != Source.make_id("doi:10/abd")


def test_evidence_provenance_survives_round_trip():
    _, _, _, _, _, e = world()
    back = Evidence.from_dict(json.loads(json.dumps(e.to_dict())))
    assert back.retrieval.scores == {"rerank": 1.5} and back.retrieval.rank == 1
    assert back.extraction.details == {"window": 0} and back.admission.role is EvidenceRole.SYNTHETIC_TEST


def test_claim_decomposition_relationships_survive_graph_round_trip():
    g, s, d, c, sp, e = world()
    kids = [claim(t, parent=c.claim_id) for t in ("Drug X reduces mortality at 1 year.", "Drug X reduces mortality in adults.")]
    g.add_all(kids)
    g.validate()
    back = EvidenceGraph.from_dict(json.loads(json.dumps(g.to_dict())))
    back.validate()
    assert {k.claim_id for k in back.children(c.claim_id)} == {k.claim_id for k in kids}


def test_judgement_provenance_survives():
    g, s, d, c, sp, e = world()
    j = judgement(c, e, Stance.NEUTRAL, judge="llm", rationale="topical only")
    back = Judgement.from_dict(json.loads(json.dumps(j.to_dict())))
    assert back.judge == JUDGES["llm"] and back.provenance.details == {"raw": "{}"} and back.rationale == "topical only"


# --------------------------------------------------------------------------- 6-7: duplicates and source identity

def test_duplicate_same_source_evidence_is_not_independent():
    g, s, d, c, sp, e = world()
    sp2 = span(d, "Mortality fell with drug X.", 7)
    e2 = evidence(c, sp2, d, s, key="k2")
    g.add_all([sp2, e2])
    groups = g.evidence_by_independence([e, e2])
    assert len(groups) == 1 and len(next(iter(groups.values()))) == 2


def test_independence_group_overrides_distinct_source_ids():
    a, b = source("press-release", group="company-z"), source("company-blog", group="company-z")
    assert a.source_id != b.source_id
    g = EvidenceGraph()
    c = claim("Z ships feature F.")
    items = []
    for src, text in ((a, "Z announced F."), (b, "F is shipping.")):
        d = docv(src); sp = span(d, text, 0); ev = evidence(c, sp, d, src)
        g.add_all([src, d, sp, ev]); items.append(ev)
    g.add(c)
    assert list(g.evidence_by_independence(items)) == ["company-z"]


def test_same_content_id_with_different_content_is_rejected():
    g, s, *_ = world()
    with pytest.raises(IntegrityError):
        g.add(Source(source_id=s.source_id, canonical_identity="other"))


def test_dangling_references_are_rejected():
    g = EvidenceGraph()
    s = source("x"); d = docv(s)
    g.add(d)
    with pytest.raises(IntegrityError):
        g.validate()


# --------------------------------------------------------------------------- 8-10: no implicit support

def test_evidence_has_no_stance_and_retrieval_is_not_support():
    assert not any(f in Evidence.__dataclass_fields__ for f in ("stance", "supports", "verdict"))
    assert not any(f in RetrievedCandidate.__dataclass_fields__ for f in ("stance", "supports", "verdict", "claim_id"))


def _legacy_unit_claim(text):
    from verification_units import analyze_verification_units
    unit = analyze_verification_units(text)[0].to_dict()
    root = claim(text)
    unit_claim = Claim(claim_id=Claim.make_id(root.claim_id, "unit"), text=text, origin=ClaimOrigin.LEGACY_SENTENCE_UNIT,
                       parent_claim_id=root.claim_id, metadata={"legacy_unit": unit})
    comps = [Claim(claim_id=Claim.make_id(unit_claim.claim_id, comp["component_id"]), text=comp["text"],
                   origin=ClaimOrigin.LEGACY_COMPONENT, parent_claim_id=unit_claim.claim_id,
                   metadata={"legacy_component_id": comp["component_id"]}) for comp in unit["components"]]
    return root, unit_claim, comps


def _presented(cl, texts):
    out = []
    for i, text in enumerate(texts, start=1):
        s = source(f"doc{i}"); d = docv(s); sp = span(d, text, 0)
        out.append(Evidence(evidence_id=Evidence.make_id(cl.claim_id, sp.span_id, f"E{i}"), claim_id=cl.claim_id,
                            span_id=sp.span_id, document_version_id=d.document_version_id, source_id=s.source_id,
                            text=text, admission=AdmissionRecord(EvidenceRole.LEGACY_TOP_N_PRESENTED_TO_JUDGE, "t",
                                                                 "t", presentation_index=f"E{i}")))
    return out


def test_missing_evidence_never_becomes_support():
    root, unit, comps = _legacy_unit_claim("Drug X reduces mortality.")
    v = LegacyAnySupportsPolicy().aggregate_unit(unit, comps, [], [])
    assert v.legacy.verdict is LegacyVerdict.INSUFFICIENT_EVIDENCE
    with pytest.raises(ValueError):
        ClaimVerdict(verdict_id="v", claim_id=root.claim_id, policy="p", policy_version="1", judgement_ids=(),
                     state=VerdictState.SUPPORTED)


def test_invalid_judgement_is_never_support():
    root, unit, comps = _legacy_unit_claim("Drug X reduces mortality.")
    evs = _presented(root, ["Drug X reduced mortality.", "Other."])
    js = [judgement(comps[0], evs[0], Stance.SUPPORTS), judgement(comps[0], evs[1], Stance.INVALID)]
    policy = LegacyAnySupportsPolicy()
    uv = policy.aggregate_unit(unit, comps, evs, js)
    assert uv.legacy.verdict is LegacyVerdict.INSUFFICIENT_EVIDENCE
    cv = policy.aggregate_claim(root, [uv], [js])
    assert cv.legacy.run_status is LegacyRunStatus.FAILED_INCOMPLETE_ASSESSMENT and cv.legacy.verdict is None


def test_rich_verdict_exists_without_legacy_collapse():
    v = ClaimVerdict(verdict_id="v", claim_id="c", policy="future", policy_version="0", judgement_ids=("j1", "j2"),
                     state=VerdictState.MIXED_EVIDENCE)
    assert v.legacy is None and v.state is VerdictState.MIXED_EVIDENCE
    assert ClaimVerdict.from_dict(v.to_dict()) == v
    legacy_only = ClaimVerdict(verdict_id="w", claim_id="c", policy="legacy", policy_version="0", judgement_ids=(),
                               legacy=LegacyOutcome(LegacyRunStatus.VERDICT, LegacyVerdict.INSUFFICIENT_EVIDENCE))
    assert legacy_only.state is None
    with pytest.raises(ValueError):
        ClaimVerdict(verdict_id="x", claim_id="c", policy="p", policy_version="0", judgement_ids=())


def test_supported_state_cannot_have_uncovered_required_slots():
    with pytest.raises(ValueError):
        ClaimVerdict(verdict_id="v", claim_id="c", policy="p", policy_version="0", judgement_ids=("j",),
                     state=VerdictState.SUPPORTED, uncovered_slots=("time",))


def test_conflicting_judgements_remain_observable():
    root, unit, comps = _legacy_unit_claim("Drug X reduces mortality.")
    evs = _presented(root, ["Drug X reduced mortality.", "Drug X increased mortality."])
    js = [judgement(comps[0], evs[0], Stance.SUPPORTS), judgement(comps[0], evs[1], Stance.CONTRADICTS)]
    uv = LegacyAnySupportsPolicy().aggregate_unit(unit, comps, evs, js)
    cov = uv.component_coverage[0]
    assert cov.status == "CONFLICTING"
    assert cov.supporting_evidence_ids == (evs[0].evidence_id,) and cov.contradicting_evidence_ids == (evs[1].evidence_id,)
    assert set(uv.judgement_ids) == {j.judgement_id for j in js}


# --------------------------------------------------------------------------- old vs new semantics (synthetic)

def test_example1_retrieved_but_not_support():
    g, s, d, c, sp, e = world()
    topical = span(d, "Drug X was studied in patients with cancer.", 1)
    cand = RetrievedCandidate(candidate_id="cand1", span_id=topical.span_id, document_version_id=d.document_version_id,
                              source_id=s.source_id, presented_text=topical.text,
                              retrieval=RetrievalProvenance(query=c.text, method="hybrid", rank=1, retrieval_run_id="r"))
    g.add_all([topical, cand])
    g.validate()
    assert g.evidence_for(c.claim_id) == [e]          # the candidate did not become evidence
    ev = evidence(c, topical, d, s, key="topical")
    j = judgement(c, ev, Stance.NEUTRAL, rationale="same topic, no entailment")
    g.add_all([ev, j]); g.validate()
    assert [x.stance for x in g.judgements_for(c.claim_id)] == [Stance.NEUTRAL]


def test_example2_numeric_mismatch_is_contradiction_with_slot_conflict():
    g, s, d, _, _, _ = world()
    c = claim("Method X achieves 92% accuracy.", quantity=("92%",), required_slots=("quantity",))
    sp = span(d, "Method X achieves 87% accuracy.", 2)
    ev = evidence(c, sp, d, s)
    j = judgement(c, ev, Stance.CONTRADICTS, slot_coverage={"quantity": SlotStatus.CONFLICTING})
    g.add_all([c, sp, ev, j]); g.validate()
    assert j.slot_coverage["quantity"] is SlotStatus.CONFLICTING


def test_example3_wrong_dataset_is_representable_as_not_support():
    g, s, d, _, _, _ = world()
    c = claim("Method X improves accuracy on Dataset A.", population="Dataset A", required_slots=("population",))
    sp = span(d, "Method X improves accuracy on Dataset B.", 4)
    ev = evidence(c, sp, d, s)
    j = judgement(c, ev, Stance.NEUTRAL, slot_coverage={"population": SlotStatus.CONFLICTING})
    g.add_all([c, sp, ev, j]); g.validate()
    with pytest.raises(ValueError):   # a future SUPPORTED verdict cannot ignore the uncovered dataset slot
        ClaimVerdict(verdict_id="v", claim_id=c.claim_id, policy="future", policy_version="0",
                     judgement_ids=(j.judgement_id,), state=VerdictState.SUPPORTED, uncovered_slots=("population",))


def test_example4_two_passages_same_source_keep_identity():
    g, s, d, c, sp, e = world()
    sp2 = span(d, "Mortality was lower with drug X.", 9)
    e2 = evidence(c, sp2, d, s, key="2")
    g.add_all([sp2, e2])
    assert g.source_of(e) == g.source_of(e2) == s
    assert len(g.evidence_by_independence([e, e2])) == 1


def test_example5_cross_source_conflict_survives_unresolved():
    g = EvidenceGraph()
    c = claim("Drug X reduces mortality.")
    g.add(c)
    js = []
    for name, stance in (("source-a", Stance.SUPPORTS), ("source-b", Stance.CONTRADICTS)):
        s = source(name); d = docv(s); sp = span(d, name, 0); ev = evidence(c, sp, d, s)
        j = judgement(c, ev, stance)
        g.add_all([s, d, sp, ev, j]); js.append(j)
    g.validate()
    assert {j.stance for j in g.judgements_for(c.claim_id)} == {Stance.SUPPORTS, Stance.CONTRADICTS}


# --------------------------------------------------------------------------- H5 readiness

def test_same_claim_same_evidence_different_judges():
    g, s, d, c, sp, e = world()
    before = (c.to_dict(), e.to_dict())
    js = [judgement(c, e, Stance.SUPPORTS, judge=k) for k in ("llm", "nli", "human")]
    g.add_all(js)
    g.validate()
    assert len({j.judgement_id for j in js}) == 3
    assert {j.judge.kind for j in g.judgements_for(c.claim_id)} == {JudgeKind.LLM, JudgeKind.NLI, JudgeKind.HUMAN}
    assert (c.to_dict(), e.to_dict()) == before


# --------------------------------------------------------------------------- retrieval adapter

REFS = [{"rank": 1, "evidence_id": "E1", "chunk_id": "27894", "scifact_doc_id": 21456232, "scifact_sentence_indices": [7],
         "scifact_sentence_texts": ["Sentence seven."], "text": "Title: T. Abstract sentence: Sentence seven.",
         "retrieval_score": 0.28},
        {"rank": 2, "evidence_id": "E2", "chunk_id": "23029", "scifact_doc_id": 16532419, "scifact_sentence_indices": [8],
         "scifact_sentence_texts": ["Sentence eight."], "text": "Title: U. Abstract sentence: Sentence eight.",
         "retrieval_score": 0.21}]


def test_recorded_refs_preserve_identity_rank_score_text_and_query():
    a = adapt_recorded_refs(REFS, query="the claim", run_key="run")
    g = EvidenceGraph(); g.add_all(a.objects()); g.validate()
    assert [c.retrieval.rank for c in a.candidates] == [1, 2]
    assert [c.retrieval.scores["retrieval_score"] for c in a.candidates] == [0.28, 0.21]
    assert [c.presented_text for c in a.candidates] == [r["text"] for r in REFS]
    assert {c.retrieval.query for c in a.candidates} == {"the claim"}
    spans = {s.span_id: s for s in a.spans}
    assert spans[a.candidates[0].span_id].sentence_indices == (7,) and spans[a.candidates[0].span_id].text == "Sentence seven."
    assert {s.canonical_identity for s in a.sources} == {"scifact:21456232", "scifact:16532419"}
    assert all(s.source_type is SourceType.UNKNOWN for s in a.sources)
    assert not g.all(Evidence)        # adaptation never creates evidence


def test_live_retrieval_response_adapter_preserves_all_scores():
    from research_retrieval import RetrievalEvidence, RetrievalResponse
    items = [RetrievalEvidence(evidence_id="9", text="Title: T. Abstract sentence: S.", source_id="scifact:1",
                               filename="scifact_doc_1", page=3, score=0.5, chunk_id="9", retrieval_score=0.5,
                               reranker_score=2.0, semantic_score=0.7, lexical_score=0.1,
                               retrieval_metadata={"chunk": {"scifact_doc_id": 1, "scifact_sentence_indices": [2],
                                                             "scifact_sentence_texts": ["S."]}}),
             RetrievalEvidence(evidence_id="10", text="A generic passage.", source_id="manual.pdf",
                               filename="manual.pdf", page=1, score=0.4, chunk_id="10", document_id="manual.pdf")]
    a = adapt_retrieval_response(RetrievalResponse(query="q", evidence=items, provider="existing_advanced_rag"),
                                 run_key="live")
    g = EvidenceGraph(); g.add_all(a.objects()); g.validate()
    first, second = a.candidates
    assert first.retrieval.scores["reranker_score"] == 2.0 and first.retrieval.scores["semantic_score"] == 0.7
    assert first.retrieval.provider == "existing_advanced_rag" and first.retrieval.rank == 1
    spans = {s.span_id: s for s in a.spans}
    assert spans[second.span_id].kind is SpanKind.PASSAGE
    assert {s.canonical_identity for s in a.sources} == {"scifact:1", "document:manual.pdf"}
