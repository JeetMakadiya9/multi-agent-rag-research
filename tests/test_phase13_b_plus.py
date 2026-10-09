"""Phase 13 B+ components: equivalence with the frozen baseline and guard safety."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from claim_verification import EvidenceItem
import experiment_b_verification as frozen_b
from evaluation.experiments.phase13 import b_plus
from llm import StaticProvider

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "evaluation/experiments/results/phase10_full_dev/per_example.jsonl"


def _unit(text):
    return frozen_b.analyze_verification_units(text)[0]


def _ev(*texts):
    return [EvidenceItem(f"E{i}", t) for i, t in enumerate(texts, start=1)]


def _data(*stances, components=("C1",)):
    rows = [{"evidence_index": f"E{i}", "component_indices": [c], "stance": s}
            for i, s in enumerate(stances, start=1) for c in components]
    return {"assessments": rows, "reason": ""}


# ------------------------------------------------------------ v1 equivalence

@pytest.mark.parametrize("claim,texts,stances", [
    ("Drug X reduces mortality by 10%.", ["Drug X reduced mortality by 10%."], ["SUPPORTS"]),
    ("Drug X reduces mortality by 10%.", ["Drug X reduced mortality."], ["SUPPORTS"]),
    ("Drug X reduces mortality.", ["Drug X increased mortality."], ["CONTRADICTS"]),
    ("Drug X reduces mortality.", ["a", "b"], ["SUPPORTS", "CONTRADICTS"]),
    ("Drug X reduces mortality.", ["a", "b"], ["NEUTRAL", "NEUTRAL"]),
])
def test_guard_v1_is_exactly_frozen_validator(claim, texts, stances):
    unit, evidence, data = _unit(claim), _ev(*texts), _data(*stances)
    frozen = frozen_b.validate_judgement(unit, data, evidence)
    plus = b_plus.validate_judgement_plus(unit, data, evidence, guard_version="v1")
    assert plus == frozen


def test_decision_ladder_reproduces_frozen_verdicts_on_synthetic_grid():
    claims = ["APOE4 increases AlphaBeta production and tau phosphorylation.",
              "Drug X reduces mortality by 10% since 1999.", "Drug X reduces mortality."]
    for claim in claims:
        unit = _unit(claim)
        comps = [c.component_id for c in unit.components]
        for s1 in ("SUPPORTS", "CONTRADICTS", "NEUTRAL"):
            for s2 in ("SUPPORTS", "CONTRADICTS", "NEUTRAL"):
                for texts in (["Drug X reduces mortality by 10% since 1999."], ["unrelated", "text"]):
                    evidence = _ev(*texts[:2], *([] if len(texts) == 2 else ["more"]))
                    rows = [{"evidence_index": e.evidence_id, "component_indices": [c], "stance": s}
                            for e, s in zip(evidence, (s1, s2)) for c in comps]
                    base = frozen_b.validate_judgement(unit, {"assessments": rows, "reason": ""}, evidence)
                    verdict, overrides = b_plus.decide_verdict(base, base.validation_errors)
                    assert verdict == base.verdict
                    assert overrides == base.overrides


def test_v1_reaggregation_reproduces_every_persisted_frozen_b_verdict():
    """B3 is computed by re-aggregation; prove the method is exact on all 280 frozen B outputs."""
    rows = [json.loads(line) for line in FROZEN.open(encoding="utf-8")]
    checked = 0
    for row in rows:
        if row["system"] != "B" or row["status"] != "SUCCESS":
            continue
        assessments = row["output"]["assessments"]
        n = len({a["evidence_index"] for a in assessments})
        texts = [c["text"] for c in row["retrieved_chunks"][:n]]
        out = b_plus.reaggregate_output(row["runtime_input"]["claim"], assessments, texts, guard_version="v1")
        assert out["verdict"] == row["predicted_label"], row["claim_id"]
        checked += 1
    assert checked == 280


# ------------------------------------------------------------ v2 guard behaviour

@pytest.mark.parametrize("claim,evidence_text", [
    ("Assembly of invadopodia is triggered by phosphatidylinositol-3,4-biphosphate.",
     "Phosphoinositide signalling triggers invadopodia."),
    ("Combining phosphatidylinositide 3-kinase and MEK 1/2 inhibitors is effective.", "PI3K and MEK inhibitors were effective."),
    ("Occupancy of ribosomes by IncRNAs mirror 5 0-UTRs", "Ribosome occupancy of lncRNAs mirrors 5 UTRs."),
    ("Inositol lipid 3-phosphatase PTEN converts Ptdlns(3,4)P 2 into phosphatidylinositol 4-phosphate.",
     "PTEN converts PI(3,4)P2 into PI(3)P and 2."),
    ("NF2 causes sequestration of YAP by activating LATS1/2 kinases.", "Merlin activates LATS kinases."),
])
def test_v2_does_not_treat_nomenclature_as_required_quantities(claim, evidence_text):
    unit = _unit(claim)
    assert frozen_b._hard_qualifier_gaps(unit, _ev(evidence_text))          # v1 fires
    assert b_plus.hard_qualifier_gaps_v2(unit, _ev(evidence_text)) == []     # v2 does not


@pytest.mark.parametrize("claim,evidence_text", [
    ("Incidence of heart failure decreased by 10% in women.", "Incidence of heart failure decreased in women."),
    ("A total of 1,000 people in the UK are carriers.", "Many people in the UK are carriers."),
    ("HNF4A mutations cause diabetes by the age of 14 years", "HNF4A mutations cause diabetes."),
    ("Treatment X gives a 10-fold increase.", "Treatment X gives an increase."),
    ("A 10% increase was observed.", "An increase was observed."),
    ("Calcium is unnecessary above 75 nmol/liter.", "Calcium is unnecessary above 50 nmol/liter."),
    ("Survival improved at 5-year follow-up.", "Survival improved at follow-up."),
    ("1/2000 in UK have abnormal PrP positivity.", "Abnormal PrP positivity was found in the UK."),
    ("The rate was 3 hours.", "The rate was 4 h."),
])
def test_v2_keeps_genuine_quantities_strict(claim, evidence_text):
    unit = _unit(claim)
    assert b_plus.hard_qualifier_gaps_v2(unit, _ev(evidence_text)), claim


@pytest.mark.parametrize("claim,evidence_text", [
    ("Symptoms began less than 3 hours before injury.", "Symptoms began less than three hours before injury."),
    ("Symptoms began less than 3 hours before injury.", "Symptoms began < 3 h before injury."),
    ("1/2000 in UK have abnormal PrP positivity.", "About 1/2000 in the UK had abnormal PrP positivity."),
    ("Screening detects cervical intraepithelial neoplasia grade 2.", "Detection of CIN2 was higher."),
    ("Combining PI3K and MEK 1/2 inhibitors is effective.", "MEK1/2 inhibition was effective."),
])
def test_v2_recognises_equivalent_numeric_notation(claim, evidence_text):
    unit = _unit(claim)
    assert b_plus.hard_qualifier_gaps_v2(unit, _ev(evidence_text)) == []


def test_v2_coordinated_objects_skip_clauses_but_keep_noun_phrases():
    unit = _unit("The risk of breast cancer increases with placental weight of pregnancies, and this association "
                 "is strongest for premenopausal breast cancer.")
    ev = _ev("Breast cancer risk increased with placental weight in pregnancies.")
    assert frozen_b._coordinated_component_gaps(unit, ev)
    assert b_plus.coordinated_component_gaps_v2(unit, ev) == []
    nps = _unit("Mutant mice lacking SVCT2 have greatly increased ascorbic acid levels in both brain and adrenals.")
    assert b_plus.coordinated_component_gaps_v2(nps, _ev("Ascorbic acid levels in brain were low.")) == [
        "coordinated object: adrenals"]


def test_v2_never_creates_support_without_judge_support():
    """Guards can only remove lexical false negatives; NEUTRAL stances stay INSUFFICIENT."""
    unit = _unit("Combining phosphatidylinositide 3-kinase and MEK 1/2 inhibitors is effective.")
    ev = _ev("PI3K and MEK inhibitors were effective.", "Unrelated.")
    for stances, expected in ((("NEUTRAL", "NEUTRAL"), "INSUFFICIENT_EVIDENCE"),
                              (("SUPPORTS", "NEUTRAL"), "SUPPORTED"),
                              (("SUPPORTS", "CONTRADICTS"), "INSUFFICIENT_EVIDENCE"),
                              (("CONTRADICTS", "NEUTRAL"), "CONTRADICTED")):
        result = b_plus.validate_judgement_plus(unit, _data(*stances), ev, guard_version="v2")
        assert result.verdict == expected, stances


# ------------------------------------------------------------ B2 packaging / replay

class _Doc:
    title = "T"
    abstract = ("s0.", "s1.", "s2.")


def test_context_packaging_marks_retrieved_sentence_and_neighbours():
    text = b_plus.package_with_context("T", _Doc.abstract, 0, 1)
    assert "Retrieved sentence (abstract sentence 1): s0." in text
    assert "Context sentence after (abstract sentence 2): s1." in text
    assert "before" not in text


def test_replay_preserves_identity_and_metadata_for_frozen_adapter():
    from evaluation.scifact.run_scifact_evaluation import _runtime_evidence
    chunks = [{"rank": 1, "chunk_id": "7", "scifact_doc_id": 5, "scifact_sentence_indices": [1],
               "scifact_sentence_texts": ["s1."], "text": "Title: T. Abstract sentence: s1.", "retrieval_score": 0.4}]
    plain = b_plus.ReplayRetrievalProvider({"c": chunks})
    refs, items = _runtime_evidence(plain.retrieve("c", limit=20).evidence)
    assert refs[0]["scifact_doc_id"] == 5 and refs[0]["scifact_sentence_indices"] == [1]
    assert refs[0]["chunk_id"] == "7" and items[0].text == chunks[0]["text"]
    ctx = b_plus.ReplayRetrievalProvider({"c": chunks}, corpus={5: _Doc}, context_window=1)
    refs2, items2 = _runtime_evidence(ctx.retrieve("c", limit=20).evidence)
    assert refs2[0]["scifact_sentence_indices"] == [1]          # scoring identity unchanged
    assert "Context sentence before (abstract sentence 1): s0." in items2[0].text


# ------------------------------------------------------------ B4 prompt override

def test_prompt_override_changes_only_the_system_message():
    unit = _unit("Drug X reduces mortality.")
    messages = frozen_b._pairwise_judge_prompt("q", unit, "E1", EvidenceItem("E1", "t"), unit.components[0])
    seen = []

    class Recorder:
        calls = 0

        def generate(self, msgs, *, temperature=0.0, timeout=120):
            seen.append(msgs)
            return "{}"

    judge = b_plus.PromptOverrideJudge(Recorder(), b_plus.PROMPT_V2_SYSTEM)
    judge.generate(messages)
    assert seen[0][0]["content"] == b_plus.PROMPT_V2_SYSTEM
    assert seen[0][1] == messages[1]
    with pytest.raises(RuntimeError):
        judge.generate([{"role": "system", "content": "other"}, messages[1]])


def test_prompt_v2_contains_no_benchmark_material():
    text = b_plus.PROMPT_V2_SYSTEM.lower()
    for forbidden in ("scifact", "gold", "rationale", "label", "supported", "contradicted", "insufficient"):
        assert forbidden not in text


def test_frozen_run_system_b_is_used_unchanged_with_override(monkeypatch):
    """With the override, the frozen adapter still runs; only the system message differs."""
    from evaluation.experiments import local_model
    from evaluation.experiments.experiment_config import ExperimentConfig
    from evaluation.experiments.system_b_verified_rag import run_system_b

    received = []

    class FakeCPU:
        def __init__(self, *a, **k):
            self.calls = 0

        def generate(self, msgs, *, temperature=0.0, timeout=120):
            self.calls += 1
            received.append(msgs[0]["content"])
            body = json.loads(msgs[1]["content"].split("Input:\n", 1)[1])
            idx = body["selected_evidence"]["evidence_index"]
            return json.dumps({"assessments": [{"evidence_index": idx, "component_indices": ["C1"],
                                                "stance": "SUPPORTS"}], "reason": "r"})

    monkeypatch.setattr(local_model, "OllamaCPUProvider", FakeCPU)
    chunks = [{"rank": i, "chunk_id": str(i), "scifact_doc_id": i, "scifact_sentence_indices": [0],
               "scifact_sentence_texts": ["x"], "text": f"Drug X reduces mortality {i}.", "retrieval_score": 1.0}
              for i in (1, 2, 3)]
    provider = b_plus.ReplayRetrievalProvider({"Drug X reduces mortality.": chunks})
    config = ExperimentConfig(dataset_root=".", verification_evidence_limit=2)
    runtime = {"claim_id": 1, "claim": "Drug X reduces mortality."}
    base = run_system_b(runtime, provider, config)
    with b_plus.judge_prompt_override(b_plus.PROMPT_V2_SYSTEM):
        over = run_system_b(runtime, provider, config)
    assert local_model.OllamaCPUProvider is FakeCPU            # restored
    assert base.predicted_label == over.predicted_label == "SUPPORTED"
    assert base.llm_calls == over.llm_calls == 2
    assert received[:2] == [b_plus.PROMPT_V1_SYSTEM] * 2
    assert received[2:] == [b_plus.PROMPT_V2_SYSTEM] * 2
