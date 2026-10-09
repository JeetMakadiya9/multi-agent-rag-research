"""Post-hoc NEI probe: configuration identity, sampling, isolation and resume safety."""
from __future__ import annotations

import json

import pytest

from evaluation.experiments.phase13 import run_nei_probe as probe
from evaluation.experiments.phase13.common import OUTPUT_DIR, RESULTS_DIR


def test_b1_spec_and_config_match_registered_b1():
    from evaluation.experiments.phase13.run_experiment import VARIANTS
    assert probe.b1_spec() == VARIANTS["B1"]
    assert (VARIANTS["B1"]["evidence_limit"], VARIANTS["B1"]["context_window"], VARIANTS["B1"]["prompt"]) == (8, 0, "v1")
    registered = json.loads((OUTPUT_DIR / "runs" / "B1" / "run_config.json").read_text(encoding="utf-8"))
    cfg = probe.b1_config([1, 2]).to_dict()
    for key in ("top_k", "verification_evidence_limit", "model", "temperature", "split", "seed",
                "hybrid_retrieval", "reranking", "claim_verification", "iterative_research", "critic"):
        assert cfg[key] == registered["experiment_config"][key], key


def test_sample_is_deterministic_30_claims_10_per_tercile():
    first, second = probe.build_manifest(), probe.build_manifest()
    assert first["claim_ids"] == second["claim_ids"]
    assert len(first["claim_ids"]) == 30 == len(set(first["claim_ids"]))
    assert {t: sum(c["tercile"] == t for c in first["claims"]) for t in ("low", "middle", "high")} == \
        {"low": 10, "middle": 10, "high": 10}
    assert first["pool"]["unlabeled_nei_claims"] == 112
    assert sum(first["pool"]["tercile_sizes"].values()) == 112


def test_sample_uses_only_nei_claims_and_ignores_predictions():
    from evaluation.experiments.phase13.common import gold_by_claim, load_dataset
    gold = gold_by_claim(load_dataset())
    m = probe.build_manifest()
    assert all(gold[c].label is None and not gold[c].rationales for c in m["claim_ids"])
    assert "any prediction" in m["inputs_not_consulted"]


def test_runtime_payload_is_gold_free():
    from evaluation.experiments.phase13.common import load_dataset
    from evaluation.scifact.scifact_adapter import runtime_claim_payload
    record = next(r for r in load_dataset().claims if r.runtime.claim_id == probe.build_manifest()["claim_ids"][0])
    assert set(runtime_claim_payload(record.runtime)) == {"claim_id", "claim"}


def test_output_paths_are_isolated_from_registered_runs():
    assert probe.PROBE_DIR == RESULTS_DIR / "phase13" / "runs_nei_probe"
    assert OUTPUT_DIR not in probe.PROBE_DIR.parents and probe.PROBE_DIR != OUTPUT_DIR
    for path in (probe.MANIFEST_PATH, probe.B1_DIR):
        assert probe.PROBE_DIR in path.parents or path == probe.PROBE_DIR


def test_resume_never_duplicates_completed_claims(tmp_path, monkeypatch):
    """With a fake system, two batches cover distinct claims and the file has no duplicates."""
    probe_dir = tmp_path / "runs_nei_probe"
    monkeypatch.setattr(probe, "PROBE_DIR", probe_dir)
    monkeypatch.setattr(probe, "MANIFEST_PATH", probe_dir / "nei_probe_manifest.json")
    monkeypatch.setattr(probe, "B1_DIR", probe_dir / "B1")
    monkeypatch.setattr(probe, "model_digest", lambda: probe.MODEL_DIGEST)
    manifest = probe.freeze_manifest()
    from evaluation.experiments import system_b_verified_rag
    from evaluation.experiments.evaluator import SystemOutput

    def fake_system(runtime, provider, config):
        assert set(runtime) == {"claim_id", "claim"}
        assert config.verification_evidence_limit == 8
        provider.retrieve(runtime["claim"], limit=config.top_k)
        return SystemOutput(predicted_label="INSUFFICIENT_EVIDENCE", llm_calls=8, retrieval_calls=1)

    monkeypatch.setattr(system_b_verified_rag, "run_system_b", fake_system)
    first = probe.run_batch(5)
    second = probe.run_batch(5)
    ids = probe.completed_claims()
    assert first["executed_this_batch"] == second["executed_this_batch"] == 5
    assert ids == manifest["claim_ids"][:10] and len(set(ids)) == 10
    with pytest.raises(SystemExit):
        probe.run_batch(6)
