"""Slice 0 legacy replay: exactness on every recorded run, and proof the check can fail."""
from __future__ import annotations

import copy
import itertools
import json
import socket

import pytest

import experiment_b_verification as frozen_b
from claim_verification import EvidenceItem
from evidence_core import AdmissionRecord, Claim, ClaimOrigin, Evidence, EvidenceRole, Judgement, Stance
from evidence_core.adapters.experiment_b import NotReplayable, adapt_experiment_b_record
from evidence_core.policies import LegacyAnySupportsPolicy
from evidence_core.replay import (
    RESULTS, network_forbidden, recorded_sources, replay_record, run_replay, scifact_lookup,
)
from verification_units import analyze_verification_units

FROZEN = RESULTS / "phase10_full_dev/per_example.jsonl"


@pytest.fixture(scope="module")
def frozen_b_rows():
    with open(FROZEN, encoding="utf-8") as handle:
        return [r for r in map(json.loads, handle) if r["system"] == "B"]


@pytest.fixture(scope="module")
def judge():
    return recorded_sources()[0]["judge"]


def test_full_offline_replay_is_exact_for_every_replayable_record():
    with network_forbidden():
        report = run_replay()
    t = report["totals"]
    assert t["mismatches"] == 0 and t["blockers"] == 0
    assert t["exact_matches"] == t["replayed"] == t["records"] - t["not_replayable"]
    # The only non-replayable records are the two frozen INTERRUPTED attempts.
    assert {(e["source"], e["claim_id"]) for e in report["not_replayable"]} == {
        ("phase10f_frozen_full_dev_B", 343), ("phase10f_frozen_full_dev_B", 440)}
    assert report["per_source"]["phase10f_frozen_full_dev_B"]["EXACT_MATCH"] == 298


def _adapt(row, judge):
    return adapt_experiment_b_record(row, claim_key=f"t:{row['claim_id']}", run_key="t", judge=judge,
                                     lookup=scifact_lookup())


def test_flipping_one_recorded_stance_is_detected(frozen_b_rows, judge):
    row = next(r for r in frozen_b_rows if r["status"] == "SUCCESS" and r["predicted_label"] == "SUPPORTED"
               and [a["stance"] for a in r["output"]["assessments"]] == ["SUPPORTS", "NEUTRAL"])
    mutated = copy.deepcopy(row)
    call = mutated["output"]["raw_output"]["trace"][1]["judge_attempts"][0]["pair_assessment_calls"][0]
    call["assessment"]["stance"] = "NEUTRAL"
    root, _ = replay_record(_adapt(mutated, judge), "t")
    assert root.legacy.verdict.value == "INSUFFICIENT_EVIDENCE" != row["predicted_label"]


def test_dropping_a_recorded_pair_assessment_turns_a_verdict_into_a_failure(frozen_b_rows, judge):
    row = next(r for r in frozen_b_rows if r["status"] == "SUCCESS")
    mutated = copy.deepcopy(row)
    call = mutated["output"]["raw_output"]["trace"][1]["judge_attempts"][0]["pair_assessment_calls"][-1]
    call["assessment"], call["error"] = None, "ValueError: injected"
    root, _ = replay_record(_adapt(mutated, judge), "t")
    assert root.legacy.run_status.value == "FAILED_INCOMPLETE_ASSESSMENT" and root.legacy.verdict is None


def test_interrupted_records_are_not_replayable(frozen_b_rows, judge):
    row = next(r for r in frozen_b_rows if r["status"] == "INTERRUPTED")
    with pytest.raises(NotReplayable):
        _adapt(row, judge)


def test_replay_makes_no_network_calls():
    with network_forbidden():
        with pytest.raises(RuntimeError):
            socket.create_connection(("127.0.0.1", 11434), timeout=1)


@pytest.mark.parametrize("text,evidence_texts", [
    ("Drug X reduces mortality by 10%.", ["Drug X reduced mortality by 10%.", "Unrelated."]),
    ("Drug X reduces mortality by 10%.", ["Drug X reduced mortality.", "Unrelated."]),
    ("APOE4 increases AlphaBeta production and tau phosphorylation.", ["AlphaBeta production rose.", "tau phosphorylation"]),
    ("Drug X reduces mortality since 1999.", ["Since 1999 drug X reduced mortality.", "Other text."]),
])
def test_legacy_ladder_equals_frozen_validator_on_exhaustive_stance_grid(text, evidence_texts):
    unit = analyze_verification_units(text)[0]
    root = Claim(claim_id=Claim.make_id("grid", text), text=text, origin=ClaimOrigin.SYNTHETIC_TEST)
    unit_claim = Claim(claim_id=Claim.make_id(root.claim_id, "u"), text=text, origin=ClaimOrigin.LEGACY_SENTENCE_UNIT,
                       parent_claim_id=root.claim_id, metadata={"legacy_unit": unit.to_dict()})
    comps = [Claim(claim_id=Claim.make_id(unit_claim.claim_id, c.component_id), text=c.text,
                   origin=ClaimOrigin.LEGACY_COMPONENT, parent_claim_id=unit_claim.claim_id,
                   metadata={"legacy_component_id": c.component_id}) for c in unit.components]
    evs = [Evidence(evidence_id=f"ev{i}", claim_id=root.claim_id, span_id=f"s{i}", document_version_id="d",
                    source_id="src", text=t, admission=AdmissionRecord(EvidenceRole.SYNTHETIC_TEST, "grid", "test",
                                                                       presentation_index=f"E{i}"))
           for i, t in enumerate(evidence_texts, start=1)]
    pairs = [(e, c) for e in evs for c in comps]
    policy = LegacyAnySupportsPolicy()
    options = ("SUPPORTS", "CONTRADICTS", "NEUTRAL", "INVALID")
    for stances in itertools.product(options, repeat=len(pairs)):
        judgements = [Judgement(judgement_id=f"j{k}", claim_id=c.claim_id, evidence_id=e.evidence_id,
                                stance=Stance(s), judge=recorded_sources()[0]["judge"])
                      for k, ((e, c), s) in enumerate(zip(pairs, stances))]
        rows = [{"evidence_index": e.admission.presentation_index,
                 "component_indices": [c.metadata["legacy_component_id"]], "stance": s}
                for (e, c), s in zip(pairs, stances) if s != "INVALID"]
        frozen = frozen_b.validate_judgement(unit, {"assessments": rows, "reason": ""},
                                             [EvidenceItem(e.admission.presentation_index, e.text) for e in evs])
        mine = policy.aggregate_unit(unit_claim, comps, evs, judgements)
        assert mine.legacy.verdict.value == frozen.verdict, stances
        assert [c.status for c in mine.component_coverage] == [r["status"] for r in frozen.coverage_matrix], stances
