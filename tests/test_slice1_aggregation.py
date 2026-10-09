"""Slice 1: conservative component-level aggregation — unit, property and integrity tests."""
from __future__ import annotations

import itertools
import json
import random
import socket

import pytest

from evidence_core import (
    AdmissionRecord, Claim, ClaimOrigin, DocumentVersion, Evidence, EvidenceGraph, EvidenceRole, JudgeIdentity,
    JudgeKind, Judgement, Provenance, Source, Span, SpanKind, Stance, VerdictState,
)
from evidence_core.aggregation_v2 import (
    CONTRADICTED_C, MIXED_C, SUPPORTED_C, UNCORROBORATED_C, UNRESOLVED_C, UNVERIFIED_C, ConservativeComponentPolicy,
    IndependentCorroborationSensitivity, classify_failure,
)
from evidence_core.policies import LegacyAnySupportsPolicy

JUDGE = JudgeIdentity(JudgeKind.HUMAN, "fixture")
RANK = {Stance.INVALID: 0, Stance.NEUTRAL: 1, Stance.CONTRADICTS: 1, Stance.SUPPORTS: 2}
# Support strength only: MIXED_EVIDENCE is not "more support" than CONTRADICTED (it is less decisive, not more supportive).
STATE_SUPPORT_LEVEL = {VerdictState.SUPPORTED: 2, VerdictState.PARTIALLY_SUPPORTED: 1, VerdictState.MIXED_EVIDENCE: 0,
                       VerdictState.INSUFFICIENT_EVIDENCE: 0, VerdictState.UNVERIFIED: 0, VerdictState.CONTRADICTED: 0}


class World:
    """root claim -> one legacy unit -> N components; evidence spans in named sources (independence groups)."""

    def __init__(self, n_components=1, required_slots=()):
        self.graph = EvidenceGraph()
        self.root = Claim(claim_id="root", text="claim", origin=ClaimOrigin.SYNTHETIC_TEST)
        self.unit = Claim(claim_id="unit", text="claim", origin=ClaimOrigin.LEGACY_SENTENCE_UNIT, parent_claim_id="root",
                          required_slots=tuple(required_slots))
        self.components = [Claim(claim_id=f"C{i}", text=f"component {i}", origin=ClaimOrigin.LEGACY_COMPONENT,
                                 parent_claim_id="unit", metadata={"legacy_component_id": f"C{i}"})
                           for i in range(1, n_components + 1)]
        self.graph.add_all([self.root, self.unit, *self.components])
        self.evidence, self.judgements = [], []

    def ev(self, source_name, group=None, idx=None):
        idx = len(self.evidence) if idx is None else idx
        src = Source(source_id=Source.make_id(source_name), canonical_identity=source_name, independence_group=group)
        dv = DocumentVersion(document_version_id=DocumentVersion.make_id(src.source_id, "v"), source_id=src.source_id)
        sp = Span(span_id=Span.make_id(dv.document_version_id, idx), document_version_id=dv.document_version_id,
                  text=f"{source_name}-{idx}", kind=SpanKind.SENTENCE)
        e = Evidence(evidence_id=f"ev{idx}", claim_id="root", span_id=sp.span_id, document_version_id=dv.document_version_id,
                     source_id=src.source_id, text=sp.text,
                     admission=AdmissionRecord(EvidenceRole.SYNTHETIC_TEST, "fixture", "test", presentation_index=f"E{idx + 1}"))
        self.graph.add_all([src, dv, sp, e])
        self.evidence.append(e)
        return e

    def judge(self, evidence, component, stance, error=None):
        j = Judgement(judgement_id=f"j-{evidence.evidence_id}-{component.claim_id}", claim_id=component.claim_id,
                      evidence_id=evidence.evidence_id, stance=stance, judge=JUDGE,
                      provenance=Provenance(producer="fixture", details={"final_error": error}))
        self.graph.add(j)
        self.judgements.append(j)
        return j

    def decide(self, policy=None, judgements=None):
        self.graph.validate()
        policy = policy or ConservativeComponentPolicy()
        return policy.decide(self.root, [self.unit], [self.components], self.evidence,
                             self.judgements if judgements is None else judgements, self.graph)


def single(stances, sources=None, errors=None):
    w = World()
    for i, st in enumerate(stances):
        e = w.ev((sources or [f"s{i}" for i in range(len(stances))])[i])
        w.judge(e, w.components[0], st, (errors or {}).get(i))
    return w


# ------------------------------------------------------------------ 1-11 component semantics
@pytest.mark.parametrize("stances,component_state,claim_state", [
    ([Stance.SUPPORTS], SUPPORTED_C, VerdictState.SUPPORTED),                                   # 1
    ([Stance.CONTRADICTS], CONTRADICTED_C, VerdictState.CONTRADICTED),                          # 2
    ([Stance.SUPPORTS, Stance.CONTRADICTS], MIXED_C, VerdictState.MIXED_EVIDENCE),              # 3
    ([Stance.NEUTRAL, Stance.NEUTRAL], UNRESOLVED_C, VerdictState.INSUFFICIENT_EVIDENCE),       # 4
    ([Stance.INVALID], UNVERIFIED_C, VerdictState.UNVERIFIED),                                  # 5
    ([Stance.SUPPORTS, Stance.INVALID], SUPPORTED_C, VerdictState.SUPPORTED),                   # 6
    ([Stance.CONTRADICTS, Stance.INVALID], CONTRADICTED_C, VerdictState.CONTRADICTED),          # 7
    ([Stance.SUPPORTS, Stance.NEUTRAL], SUPPORTED_C, VerdictState.SUPPORTED),                   # 8
])
def test_component_and_claim_states(stances, component_state, claim_state):
    d = single(stances).decide()
    assert d.components[0].state == component_state and d.state is claim_state
    assert d.components[0].judgement_incomplete == (Stance.INVALID in stances)
    if Stance.INVALID in stances:
        assert "INVALID_JUDGEMENT_PRESENT" in d.reason_codes
    if Stance.NEUTRAL in stances and Stance.SUPPORTS in stances:
        assert d.components[0].neutral_judgement_ids          # neutral evidence stays visible


def test_same_source_duplicate_support_is_one_group():                                         # 9
    w = single([Stance.SUPPORTS, Stance.SUPPORTS], sources=["doc-a", "doc-a"])
    comp = w.decide().components[0]
    assert comp.supporting_groups == (w.graph.independence_key(w.evidence[0]),) and comp.single_group_support
    assert w.decide(IndependentCorroborationSensitivity()).components[0].state == UNCORROBORATED_C


def test_independence_group_metadata_merges_distinct_sources():
    w = World()
    for name in ("press-release", "company-blog"):
        w.judge(w.ev(name, group="company-z"), w.components[0], Stance.SUPPORTS)
    assert w.decide().components[0].supporting_groups == ("company-z",)
    assert w.decide(IndependentCorroborationSensitivity()).state is not VerdictState.SUPPORTED


def test_multiple_independent_groups_can_remain_supported():                                   # 10
    w = single([Stance.SUPPORTS, Stance.SUPPORTS], sources=["doc-a", "doc-b"])
    assert w.decide().state is VerdictState.SUPPORTED
    assert w.decide(IndependentCorroborationSensitivity()).state is VerdictState.SUPPORTED
    assert not w.decide().components[0].single_group_support


def test_missing_component_judgement_is_not_support():                                         # 11
    w = World(n_components=2)
    e = w.ev("a")
    w.judge(e, w.components[0], Stance.SUPPORTS)
    d = w.decide()
    assert d.state is VerdictState.PARTIALLY_SUPPORTED
    assert [c.state for c in d.components] == [SUPPORTED_C, UNVERIFIED_C]


# ------------------------------------------------------------------ 12-14 structure
def test_multi_component_requires_every_component():                                           # 12
    w = World(n_components=2)
    e1, e2 = w.ev("a"), w.ev("b")
    for e in (e1, e2):
        w.judge(e, w.components[0], Stance.SUPPORTS)
    w.judge(e1, w.components[1], Stance.NEUTRAL); w.judge(e2, w.components[1], Stance.NEUTRAL)
    assert w.decide().state is VerdictState.PARTIALLY_SUPPORTED
    w.graph = EvidenceGraph.from_dict(w.graph.to_dict())
    w2 = World(n_components=2)
    f1 = w2.ev("a")
    w2.judge(f1, w2.components[0], Stance.SUPPORTS); w2.judge(f1, w2.components[1], Stance.SUPPORTS)
    assert w2.decide().state is VerdictState.SUPPORTED


def test_multi_unit_uses_first_unit_by_default():                                              # 13
    w = World()
    second = Claim(claim_id="unit2", text="second", origin=ClaimOrigin.LEGACY_SENTENCE_UNIT, parent_claim_id="root")
    comp2 = Claim(claim_id="U2C1", text="c", origin=ClaimOrigin.LEGACY_COMPONENT, parent_claim_id="unit2")
    w.graph.add_all([second, comp2])
    e = w.ev("a")
    w.judge(e, w.components[0], Stance.SUPPORTS)
    w.judge(e, comp2, Stance.CONTRADICTS)
    p = ConservativeComponentPolicy()
    first = p.decide(w.root, [w.unit, second], [w.components, [comp2]], w.evidence, w.judgements, w.graph)
    every = p.decide(w.root, [w.unit, second], [w.components, [comp2]], w.evidence, w.judgements, w.graph, all_units=True)
    assert first.state is VerdictState.SUPPORTED and "FIRST_UNIT_ONLY" in first.notes
    assert every.state is VerdictState.CONTRADICTED and "ALL_UNITS_DIAGNOSTIC" in every.notes


def test_rich_verdict_is_independent_of_legacy():                                              # 14
    d = single([Stance.SUPPORTS, Stance.CONTRADICTS]).decide()
    assert d.verdict.state is VerdictState.MIXED_EVIDENCE and d.verdict.legacy is None


def test_required_slots_are_diagnostic_only():
    w = World(required_slots=("quantity",))
    w.judge(w.ev("a"), w.components[0], Stance.SUPPORTS)
    d = w.decide()
    assert d.state is VerdictState.SUPPORTED and d.slot_coverage == "NOT_ASSESSED"
    assert "SLOT_COVERAGE_NOT_ASSESSED" in d.reason_codes and d.verdict.uncovered_slots == ()


def test_failure_taxonomy_is_exact_and_never_guesses():
    assert classify_failure("URLError: <urlopen error [WinError 10061] No connection could be made because the target "
                            "machine actively refused it>") == "ENDPOINT_INFRASTRUCTURE"
    assert classify_failure("TimeoutError: timed out") == "TIMEOUT"
    assert classify_failure("ValueError: Legacy judgement verdict is invalid.") == "MODEL_OUTPUT_SCHEMA"
    assert classify_failure("ValueError: Pair assessment must contain a valid SUPPORTS, CONTRADICTS, or NEUTRAL stance.") \
        == "MODEL_OUTPUT_SCHEMA"
    assert classify_failure("ValueError: something new") == "UNKNOWN_FAILURE"
    assert classify_failure(None) == "MISSING_RECORD"


# ------------------------------------------------------------------ 15-16 + property tests
def _random_world(seed, n_ev=4, n_comp=2):
    rng = random.Random(seed)
    w = World(n_components=n_comp)
    evs = [w.ev(rng.choice(["a", "b", "c"]) + str(i if rng.random() < .5 else 0), idx=i) for i in range(n_ev)]
    for e in evs:
        for c in w.components:
            w.judge(e, c, rng.choice(list(Stance)))
    return w


@pytest.mark.parametrize("seed", range(40))
def test_permutation_invariance_and_determinism(seed):                                         # 15 + determinism
    w = _random_world(seed)
    base = w.decide().to_dict()
    for _ in range(5):
        shuffled = list(w.judgements)
        random.Random(seed).shuffle(shuffled)
        assert w.decide(judgements=shuffled).to_dict() == base
    assert json.dumps(w.decide().to_dict(), sort_keys=True) == json.dumps(base, sort_keys=True)


def _replace(w, judgement, stance):
    return [Judgement(**{**{f: getattr(j, f) for f in Judgement.__dataclass_fields__}, "stance": stance})
            if j.judgement_id == judgement.judgement_id else j for j in w.judgements]


@pytest.mark.parametrize("seed", range(40))
def test_invalid_and_neutral_never_improve_and_contradiction_never_strengthens(seed):           # 16 + properties
    w = _random_world(seed)
    base = STATE_SUPPORT_LEVEL[w.decide().state]
    for j in w.judgements:
        if j.stance is Stance.SUPPORTS:
            for weaker in (Stance.INVALID, Stance.NEUTRAL, Stance.CONTRADICTS):   # removing/weakening valid support
                assert STATE_SUPPORT_LEVEL[w.decide(judgements=_replace(w, j, weaker)).state] <= base
        else:
            for replacement in (Stance.INVALID, Stance.NEUTRAL):
                new = w.decide(judgements=_replace(w, j, replacement)).state
                assert not (new is VerdictState.SUPPORTED and base < STATE_SUPPORT_LEVEL[VerdictState.SUPPORTED] and
                            j.stance in (Stance.INVALID, Stance.NEUTRAL))


def test_invalid_alone_never_creates_support():
    for k in range(1, 4):
        d = single([Stance.INVALID] * k).decide()
        assert d.state is VerdictState.UNVERIFIED and not d.components[0].supporting_judgement_ids


def test_duplicate_same_group_support_never_creates_corroboration():
    for k in range(1, 5):
        w = single([Stance.SUPPORTS] * k, sources=["same-doc"] * k)
        assert len(w.decide().components[0].supporting_groups) == 1
        assert w.decide(IndependentCorroborationSensitivity()).state is not VerdictState.SUPPORTED


def test_documented_precedence_mixed_over_contradicted():
    """Specified ladder: any MIXED component outranks a CONTRADICTED one. Consequence (reported, not hidden):
    adding a contradiction to the supported component of a CONTRADICTED claim yields MIXED_EVIDENCE."""
    w = World(n_components=2)
    a, b = w.ev("a"), w.ev("b")
    w.judge(a, w.components[0], Stance.SUPPORTS)
    w.judge(a, w.components[1], Stance.CONTRADICTS)
    assert w.decide().state is VerdictState.CONTRADICTED
    w.judge(b, w.components[0], Stance.CONTRADICTS)
    assert w.decide().state is VerdictState.MIXED_EVIDENCE


def test_adding_contradiction_never_strengthens_support():
    for stances in itertools.product([Stance.SUPPORTS, Stance.NEUTRAL, Stance.INVALID], repeat=2):
        w = single(list(stances))
        before = STATE_SUPPORT_LEVEL[w.decide().state]
        w.judge(w.ev("extra"), w.components[0], Stance.CONTRADICTS)
        assert STATE_SUPPORT_LEVEL[w.decide().state] <= before


# ------------------------------------------------------------------ 17-18 integrity
def test_network_guard_blocks_connections():                                                   # 17
    from evidence_core.replay import network_forbidden
    with network_forbidden():
        with pytest.raises(RuntimeError):
            socket.create_connection(("127.0.0.1", 11434), timeout=1)


def test_legacy_policy_identity_unchanged():
    assert LegacyAnySupportsPolicy.name == "LegacyAnySupportsPolicy"
    assert LegacyAnySupportsPolicy.version == "experiment_b_frozen_66EA7B6B"


def test_offline_experiment_runs_and_slice0_replay_stays_exact():                              # 18
    from evaluation.experiments.run_slice1_aggregation import run
    from evidence_core.replay import network_forbidden, run_replay
    with network_forbidden():
        replay = run_replay()
        report, rows = run()
    assert replay["totals"]["exact_matches"] == 604 and replay["totals"]["mismatches"] == 0
    assert report["records"]["replayable_evaluated_by_all_policies"] == 604
    assert report["changes_legacy_to_primary"]["unexplained"] == 0
    assert all(per["frozen_definition_check"]["matches"] for per in report["gold_evaluation"]["per_run"].values())
    with network_forbidden():
        again, _ = run()
    assert again["decisions_sha256"] == report["decisions_sha256"]          # deterministic
