"""The alternative-benchmark builder and scorer (evaluation/studio_eval/alt_benchmarks.py). The sets are built from the
repository's SciFact files; the scoring tests use SYNTHETIC records. No model is called."""
import json

import pytest

from evaluation.studio_eval import alt_benchmarks as alt
from evaluation.studio_eval import run_eval

if not (alt.SCIFACT / "claims_train.jsonl").is_file():
    pytest.skip("SciFact files are not present", allow_module_level=True)

EXCLUDED = alt.excluded_train_ids(None)


@pytest.fixture(scope="module")
def retro():
    return alt.build_retro(EXCLUDED)


def test_the_exclusion_list_is_the_agentic_studys_160_train_claims():
    listed = json.loads((run_eval.ROOT / "evaluation/studio_eval/alt_excluded_train_ids.json").read_text("utf-8"))
    assert listed["count"] == len(EXCLUDED) == 160


def test_retro_is_balanced_unseen_and_labelled_by_scifact(retro):
    labels = [r["label"] for r in retro]
    assert len(retro) == 300 and {l: labels.count(l) for l in set(labels)} == {alt.S: 100, alt.C: 100, alt.N: 100}
    claims = {r["id"]: r for r in alt._jsonl(alt.SCIFACT / "claims_train.jsonl")}
    for row in retro:
        sid = row["source"]["scifact_id"]
        assert sid not in EXCLUDED and alt.gold_of(claims[sid]) == row["label"]          # SciFact's own label
        assert [p["scifact_doc_id"] for p in row["passages"]] == claims[sid]["cited_doc_ids"][:8]
    assert alt.build_retro(EXCLUDED) == retro                                            # deterministic


def test_programmatic_labels_follow_their_rules(retro):
    prog = alt.build_prog(EXCLUDED, {r["claim_id"] for r in retro})
    claims = {r["id"]: r for r in alt._jsonl(alt.SCIFACT / "claims_train.jsonl")}
    docs = alt.corpus()
    assert not {r["source"]["scifact_id"] for r in prog} & {r["source"]["scifact_id"] for r in retro}
    for row in prog:
        original = claims[row["source"]["scifact_id"]]
        if "negate" in row["subgroups"]:
            assert row["claim"].startswith("It is not true that ")
            assert row["label"] == {alt.S: alt.C, alt.C: alt.S}[alt.gold_of(original)]
        else:
            assert row["label"] == alt.N and row["claim"] == original["claim"]
            text = " ".join(docs[p["scifact_doc_id"]]["title"] + " " + " ".join(docs[p["scifact_doc_id"]]["abstract"])
                            for p in row["passages"])
            assert not alt._content(row["claim"]) & alt._content(text)                    # no shared content word


def test_adversarial_cases_keep_their_label_and_carry_the_injection(retro):
    adv = alt.build_adv(retro)
    assert len(adv) == 30 and all(r["passages"][-1]["text"] == alt.INJECTION for r in adv)
    assert {r["label"] for r in adv} == {alt.N, alt.C}


def test_scoring_counts_rules_failures_and_missing_records(tmp_path):
    claims = [{"claim_id": f"c{i}", "claim": "x", "label": label, "passages": [{"url": "https://e.org", "text": "t"}]}
              for i, label in enumerate([alt.S, alt.C, alt.N, alt.N])]
    (tmp_path / "claims.jsonl").write_text("".join(json.dumps(c) + "\n" for c in claims), encoding="utf-8")
    records = [{"claim_id": "c0", "gold": alt.S, "verdict": "SUPPORTED", "sub_state": "", "verifier_note": ""},
               {"claim_id": "c1", "gold": alt.C, "verdict": "INSUFFICIENT", "sub_state": "NO_DECISIVE_EVIDENCE",
                "verifier_note": "the adjudicator proposed CONTRADICTED, which is not accepted without a passage judged"},
               {"claim_id": "c2", "gold": alt.N, "verdict": None, "sub_state": "SERVICE_ERROR", "error": "timeout"}]
    (tmp_path / "r.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    out = alt.score_records(tmp_path / "r.jsonl", None, tmp_path / "claims.jsonl")
    assert out["claims"] == 4 and out["missing_records"] == 1
    assert out["system"]["rule_activations"]["rule1_unsided_declined"] == 1
    assert out["system"]["operational_failures"] == {"SERVICE_ERROR": 1, "MISSING": 1}
    assert out["system"]["accuracy"] == 0.75                    # failures and the missing record count as NEI
