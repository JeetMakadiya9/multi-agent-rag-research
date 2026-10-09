"""H5 protocol freeze + blinded gold-set package: determinism, protection, blinding, schema, tooling."""
from __future__ import annotations

import hashlib
import json
import shutil
import socket

import pytest

from evaluation.experiments.h5 import agreement as agr
from evaluation.experiments.h5 import build_package as bp

OUT = bp.OUT
ANNOTATOR_FILES = ("h5_annotator_a.jsonl", "h5_annotator_b.jsonl", "h5_pilot_annotator_a.jsonl",
                   "h5_pilot_annotator_b.jsonl")


def rows(name):
    return [json.loads(line) for line in (OUT / name).read_text(encoding="utf-8").splitlines() if line]


@pytest.fixture(scope="module")
def built():
    pool, inventory, protected = bp.build_pool()
    pilot = bp.select(pool, bp.PILOT_QUOTA, set())
    main = bp.select(pool, bp.MAIN_QUOTA, {r["claim_id"] for r in pilot})
    return pool, inventory, protected, pilot, main


@pytest.fixture(scope="module")
def manifest():
    return json.loads((OUT / "h5_sampling_manifest.json").read_text(encoding="utf-8"))


def _hash_rows(items):
    return bp.sha256_bytes("".join(bp.dumps(r) + "\n" for r in items).encode("utf-8"))


# 1-2, 15-16: determinism and stable hashes
def test_candidate_generation_is_deterministic(built, manifest):
    pool2, _, _ = bp.build_pool()
    assert _hash_rows(built[0]) == _hash_rows(pool2) == manifest["candidate_pool_hash"]


def test_sampling_is_deterministic_and_matches_frozen_manifests(built, manifest):
    _, _, _, pilot, main = built
    assert _hash_rows(bp.manifest_rows(main, "main")) == manifest["selected_manifest_hash"]
    assert _hash_rows(bp.manifest_rows(pilot, "pilot")) == manifest["pilot_manifest_hash"]


def test_full_rebuild_reproduces_every_file_byte_for_byte(tmp_path, monkeypatch, manifest):
    for name in ("h5_protocol.md", "h5_annotation_guidelines.md", "h5_nli_model_requirements.md",
                 "h5_data_dictionary.md", "h5_preanalysis.md"):
        shutil.copy(OUT / name, tmp_path / name)
    monkeypatch.setattr(bp, "OUT", tmp_path)
    bp.build()
    for path in OUT.iterdir():
        if path.suffix in (".jsonl", ".json") or path.name == "h5_integrity_report.md":
            assert (tmp_path / path.name).read_bytes() == path.read_bytes(), path.name


def test_recorded_file_hashes_match_disk(manifest):
    for name, digest in {**manifest["file_hashes"], **manifest["document_hashes"]}.items():
        assert bp.sha256_file(OUT / name) == digest, name


# 3-6: uniqueness and exact sizes
def test_no_duplicate_pair_ids_or_claim_evidence_pairs():
    for name in ("h5_gold_manifest.jsonl", "h5_pilot_manifest.jsonl", "h5_candidate_pool.jsonl"):
        data = rows(name)
        assert len({r["pair_id"] for r in data}) == len(data)
    for name in ("h5_gold_manifest.jsonl", "h5_pilot_manifest.jsonl"):
        data = rows(name)
        assert len({(r["claim_id"], bp.norm(r["evidence_text"])) for r in data}) == len(data)


def test_exact_300_main_and_60_pilot_balanced_and_disjoint():
    main, pilot = rows("h5_gold_manifest.jsonl"), rows("h5_pilot_manifest.jsonl")
    assert len(main) == 300 and len(pilot) == 60
    for data, q in ((main, 100), (pilot, 20)):
        assert {s: sum(r["sampling_stratum"] == s for r in data) for s in bp.SAMPLED_STRATA} == dict.fromkeys(bp.SAMPLED_STRATA, q)
    assert not {r["claim_id"] for r in main} & {r["claim_id"] for r in pilot}
    for data in (main, pilot):
        counts = {}
        for r in data:
            counts[r["claim_id"]] = counts.get(r["claim_id"], 0) + 1
        assert max(counts.values()) <= bp.CAP_PER_CLAIM


# 7-10: protected data and atomicity
def test_no_reserved_dev69_dev_or_test_claims(built):
    protected = built[2]
    sampled = rows("h5_gold_manifest.jsonl") + rows("h5_pilot_manifest.jsonl")
    texts = {bp.norm(r["claim_text"]) for r in sampled}
    assert not texts & protected["reserved_texts"] and not texts & protected["dev69_texts"]
    assert not texts & protected["dev_texts"] and not texts & protected["test_texts"]
    assert all(r["source_split"] == "train" and r["claim_id"].startswith("train:") for r in sampled)


def test_only_atomic_claims_sampled_and_others_inventoried():
    sampled = rows("h5_gold_manifest.jsonl") + rows("h5_pilot_manifest.jsonl")
    assert {r["claim_atomicity_status"] for r in sampled} == {"ATOMIC"}
    excluded = rows("h5_excluded_candidates.jsonl")
    assert {"COMPOUND", "UNCERTAIN"} <= {r["claim_atomicity_status"] for r in excluded}
    assert len(rows("h5_claim_inventory.jsonl")) == 809


@pytest.mark.parametrize("claim,status", [
    ("Drug X reduces mortality in adults.", "ATOMIC"),
    ("A and B both increase C.", "UNCERTAIN"),
    ("This effect is reduced in mice.", "UNCERTAIN"),
    ("Drug X reduces mortality. Drug Y increases it.", "COMPOUND"),
])
def test_atomicity_rules(claim, status):
    assert bp.atomicity(claim)[0] == status


# 11-14: blinding, blank labels, schema
FORBIDDEN_KEYS = {"sampling_stratum", "rationale_candidate", "non_rationale_candidate", "train_rationale_label_hidden",
                  "claim_id", "document_id", "sentence_index", "evidence_sentence_id", "sampling_rank_key", "score",
                  "rank", "retrieval_score", "prediction", "model", "system", "source_split", "label", "title"}
FORBIDDEN_VALUES = ("support_oriented", "contradiction_oriented", "neutral_oriented", "qwen", "ollama", "nli",
                    "rationale", "scifact", "train:")


@pytest.mark.parametrize("name", ANNOTATOR_FILES)
def test_annotator_files_are_blinded(name):
    for row in rows(name):
        assert set(row) == set(bp.ANNOTATOR_FIELDS)
        assert not set(row) & FORBIDDEN_KEYS
        blob = json.dumps({k: row[k] for k in ("pair_id", "gold_label", "annotator_note")}).lower()
        assert not any(v in blob for v in FORBIDDEN_VALUES)
        assert row["gold_label"] is None and row["annotator_note"] is None


def test_blinding_leak_detector_catches_metadata():
    leaky = {"pair_id": "h5_0000000000000000", "claim_text": "c", "evidence_text": "e", "gold_label": None,
             "annotator_note": None, "sampling_stratum": "support_oriented"}
    assert set(leaky) != set(bp.ANNOTATOR_FIELDS)
    with pytest.raises(ValueError):
        agr.validate_annotator_rows([leaky], require_labels=False)


def test_annotator_orderings_are_independent_and_cover_same_pairs():
    for a, b, m in (("h5_annotator_a.jsonl", "h5_annotator_b.jsonl", "h5_gold_manifest.jsonl"),
                    ("h5_pilot_annotator_a.jsonl", "h5_pilot_annotator_b.jsonl", "h5_pilot_manifest.jsonl")):
        ra, rb, rm = rows(a), rows(b), rows(m)
        assert {r["pair_id"] for r in ra} == {r["pair_id"] for r in rb} == {r["pair_id"] for r in rm}
        assert [r["pair_id"] for r in ra] != [r["pair_id"] for r in rb]
        assert [r["pair_id"] for r in ra] != [r["pair_id"] for r in rm]


def test_manifest_annotation_fields_blank():
    for name in ("h5_gold_manifest.jsonl", "h5_pilot_manifest.jsonl"):
        for r in rows(name):
            assert r["annotation_status"] == "NOT_STARTED" and r["gold_label"] is None
            assert r["annotator_a_label"] is None and r["annotator_b_label"] is None and r["adjudicated_label"] is None
            assert r["sampling_metadata_hidden"] is True


def test_schema_validates_annotator_rows():
    schema = json.loads((OUT / "h5_annotation_schema.json").read_text(encoding="utf-8"))
    assert schema["additionalProperties"] is False and set(schema["required"]) == set(bp.ANNOTATOR_FIELDS)
    assert schema["properties"]["gold_label"]["enum"] == [None, "SUPPORT", "CONTRADICT", "NEUTRAL"]
    for name in ANNOTATOR_FILES:
        agr.validate_annotator_rows(rows(name), require_labels=False)


def test_atomicity_review_shows_claims_only():
    review = rows("h5_atomicity_review.jsonl")
    sampled_claims = {r["claim_review_id"] for r in rows("h5_gold_manifest.jsonl") + rows("h5_pilot_manifest.jsonl")}
    assert {r["claim_review_id"] for r in review} == sampled_claims
    assert all(set(r) == {"claim_review_id", "claim_text", "reviewer_atomicity", "reviewer_note"} and
               r["reviewer_atomicity"] is None for r in review)


# 17-18: sources and offline
def test_source_hashes_verified(manifest):
    assert bp.verify_sources() == {k: v for k, v in bp.SOURCE_SHA256.items()}
    assert manifest["source_hashes_match_frozen"] is True


def test_build_is_offline():
    from evidence_core.replay import network_forbidden
    with network_forbidden():
        bp.build_pool()
        with pytest.raises(RuntimeError):
            socket.create_connection(("127.0.0.1", 11434), timeout=1)


# agreement tooling (synthetic labels only)
def _label(data, labels):
    return [{**r, "gold_label": lab} for r, lab in zip(data, labels)]


def test_agreement_kappa_confusion_and_adjudication():
    base = [{"pair_id": f"h5_{i:016x}", "claim_text": f"c{i}", "evidence_text": f"e{i}", "gold_label": None,
             "annotator_note": None} for i in range(6)]
    a = _label(base, ["SUPPORT", "SUPPORT", "NEUTRAL", "NEUTRAL", "CONTRADICT", "CONTRADICT"])
    b = _label(list(reversed(base)), list(reversed(["SUPPORT", "NEUTRAL", "NEUTRAL", "NEUTRAL", "CONTRADICT", "SUPPORT"])))
    res = agr.agreement(a, b)
    assert res["n"] == 6 and res["disagreements"] == 2 and abs(res["raw_agreement"] - 4 / 6) < 1e-12
    expected = (2 * 2 + 2 * 3 + 2 * 1) / 36
    assert abs(res["cohens_kappa"] - (4 / 6 - expected) / (1 - expected)) < 1e-12
    adj = agr.adjudication_rows(a, b)
    assert len(adj) == 2 and all(set(r) == {"pair_id", "claim_text", "evidence_text", "label_1", "label_2",
                                            "adjudicated_label", "adjudication_note"} for r in adj)
    manifest_rows = [{"pair_id": r["pair_id"], "gold_label": None} for r in base]
    pending = agr.final_gold(manifest_rows, a, b)
    assert sum(r["annotation_status"] == "AWAITING_ADJUDICATION" for r in pending) == 2
    done = agr.final_gold(manifest_rows, a, b, [{**r, "adjudicated_label": "NEUTRAL", "adjudication_note": "x"} for r in adj])
    assert all(r["gold_label"] in agr.LABELS for r in done)
    assert {r["annotator_a_label"] for r in done} == {"SUPPORT", "NEUTRAL", "CONTRADICT"}   # raw labels preserved


def test_quality_warning_is_preregistered_and_fires():
    base = [{"pair_id": f"h5_{i:016x}", "claim_text": "c", "evidence_text": f"e{i}", "gold_label": None,
             "annotator_note": None} for i in range(4)]
    res = agr.agreement(_label(base, ["SUPPORT", "NEUTRAL", "CONTRADICT", "NEUTRAL"]),
                        _label(base, ["NEUTRAL", "SUPPORT", "NEUTRAL", "CONTRADICT"]))
    assert res["GOLD_QUALITY_WARNING"] is True and agr.KAPPA_WARNING_BELOW == 0.60


def test_unlabelled_or_invalid_labels_rejected():
    row = {"pair_id": "h5_0000000000000001", "claim_text": "c", "evidence_text": "e", "gold_label": None, "annotator_note": None}
    with pytest.raises(ValueError):
        agr.validate_annotator_rows([row], require_labels=True)
    with pytest.raises(ValueError):
        agr.validate_annotator_rows([{**row, "gold_label": "SUPPORTS"}], require_labels=False)
