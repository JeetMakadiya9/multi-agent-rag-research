"""The Studio evaluation's annotation and freeze workflow (evaluation/studio_eval/annotate.py). Every claim, passage
and label here is SYNTHETIC TEST DATA: these tests check the workflow's rules, not any verifier, and are not an
evaluation or a source of gold labels."""
import json

import pytest

from evaluation.studio_eval import annotate, run_eval
from evaluation.studio_eval.run_eval import ProtocolError

PROV = {"claim_author": "test author", "written_at": "2026-01-01", "source": "synthetic test fixture"}


def row(cid, text):
    return {"claim_id": cid, "claim": text, "provenance": PROV,
            "passages": [{"url": f"https://example.org/{cid}", "title": "synthetic", "licence": "CC0 (test)",
                          "text": f"Synthetic passage about {text.lower()}"}]}


DRAFT = [row("x1", "The tower is 300 metres tall."), row("x2", "The tower is made of glass."),
         row("x3", "The tower opened in 1889."), row("x4", "The tower is painted red.")]
A = {"x1": "SUPPORTED", "x2": "CONTRADICTED", "x3": "NOT_ENOUGH_INFO", "x4": "SUPPORTED"}
B = {"x1": "SUPPORTED", "x2": "NOT_ENOUGH_INFO", "x3": "NOT_ENOUGH_INFO", "x4": "CONTRADICTED"}


def jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def labels(path, mapping, packets=True):
    rows = {r["claim_id"]: r for r in DRAFT}
    return jsonl(path, [{"claim_id": k, "label": v, **({"packet_sha256": annotate.packet_sha256(rows[k])}
                                                      if packets else {})} for k, v in mapping.items()])


def test_a_draft_must_be_unlabelled_and_carry_provenance_and_licences(tmp_path):
    assert len(annotate.load_draft(jsonl(tmp_path / "d.jsonl", DRAFT))) == 4
    for broken, words in ((dict(DRAFT[0], label="SUPPORTED"), "must not carry a label"),
                          (dict(DRAFT[0], provenance={"source": "x"}), "provenance"),
                          (dict(DRAFT[0], passages=[{"url": "https://e.org", "text": "t"}]), "licence"),
                          (dict(DRAFT[0], passages=[DRAFT[0]["passages"][0]] * 9), "1-8")):
        with pytest.raises(ProtocolError, match=words):
            annotate.load_draft(jsonl(tmp_path / "bad.jsonl", [broken]))


def test_task_files_are_blind_and_ordered_differently():
    a, b = annotate.make_tasks(DRAFT, 1), annotate.make_tasks(DRAFT, 2)
    assert {r["claim_id"] for r in a} == {r["claim_id"] for r in b} == set(A)
    assert all(r["label"] is None and "provenance" not in r for r in a)
    assert [r["claim_id"] for r in a] != [r["claim_id"] for r in b] or len(DRAFT) < 3


def test_agreement_is_cohens_kappa_before_adjudication():
    stats = annotate.agreement(DRAFT, A, B)
    assert stats["raw_agreement"] == 0.5 and stats["disagreements"] == ["x2", "x4"]
    # observed 0.5; expected (2*1 + 1*1 + 1*2)/16 = 0.3125; kappa = 0.1875/0.6875
    assert stats["cohen_kappa"] == round(0.1875 / 0.6875, 4)


def test_labels_must_be_valid_complete_and_tied_to_the_packet(tmp_path):
    with pytest.raises(ProtocolError, match="not one of"):
        annotate.load_annotations(labels(tmp_path / "a.jsonl", {"x1": "TRUE"}), DRAFT, "annotator A")
    with pytest.raises(ProtocolError, match="different packet"):
        annotate.load_annotations(jsonl(tmp_path / "a.jsonl", [{"claim_id": "x1", "label": "SUPPORTED",
                                                                "packet_sha256": "0" * 64}]), DRAFT, "annotator A")
    partial = annotate.load_annotations(labels(tmp_path / "a.jsonl", {"x1": "SUPPORTED"}), DRAFT, "annotator A")
    with pytest.raises(ProtocolError, match="no label for 3"):
        annotate.require_complete(partial, DRAFT, "annotator A")


def test_freeze_needs_every_disagreement_adjudicated(tmp_path):
    with pytest.raises(ProtocolError, match="no adjudicated label"):
        annotate.freeze(DRAFT, A, B, {"x2": "CONTRADICTED"}, [], {}, tmp_path / "out")
    with pytest.raises(ProtocolError, match="agreed on"):
        annotate.freeze(DRAFT, A, B, {"x1": "SUPPORTED", "x2": "CONTRADICTED", "x4": "SUPPORTED"}, [], {},
                        tmp_path / "out")


def test_freeze_writes_a_claims_file_the_harness_accepts(tmp_path):
    draft = jsonl(tmp_path / "d.jsonl", DRAFT)
    record = annotate.freeze(DRAFT, A, B, {"x2": "CONTRADICTED", "x4": annotate.AMBIGUOUS}, [],
                             {"draft": draft}, tmp_path / "out")
    claims = run_eval.load_claims(tmp_path / "out" / "claims.jsonl")
    assert [c["claim_id"] for c in claims] == ["x1", "x2", "x3"]
    assert [c["label"] for c in claims] == ["SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO"]
    assert record["annotation"]["ambiguous_excluded"] == 1 and record["annotation"]["independent_of_model_output"]
    assert record["adequate_for_promotion"] is False                    # far below 300 claims, 100 per class
    assert run_eval.check_freeze(tmp_path / "out" / "claims.jsonl", tmp_path / "out" / "FREEZE.json")


def test_freeze_refuses_claims_from_used_sets(tmp_path):
    used = jsonl(tmp_path / "used.jsonl", [{"claim": "the tower opened in 1889"}])
    with pytest.raises(ProtocolError, match="match a claim set"):
        annotate.freeze(DRAFT, A, B, {"x2": "CONTRADICTED", "x4": "SUPPORTED"}, [used], {}, tmp_path / "out")


def test_the_cli_runs_the_three_steps(tmp_path):
    draft = jsonl(tmp_path / "d.jsonl", DRAFT)
    assert annotate.main(["tasks", "--draft", str(draft), "--out", str(tmp_path / "t")]) == 0
    assert (tmp_path / "t" / "tasks_annotator_a.jsonl").is_file()
    a, b = labels(tmp_path / "a.jsonl", A), labels(tmp_path / "b.jsonl", B)
    assert annotate.main(["agreement", "--draft", str(draft), "--a", str(a), "--b", str(b),
                          "--out", str(tmp_path / "g")]) == 0
    assert len((tmp_path / "g" / "to_adjudicate.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    adj = labels(tmp_path / "adj.jsonl", {"x2": "CONTRADICTED", "x4": "SUPPORTED"})
    assert annotate.main(["freeze", "--draft", str(draft), "--a", str(a), "--b", str(b), "--adjudication", str(adj),
                          "--out", str(tmp_path / "f")]) == 0
    freeze = json.loads((tmp_path / "f" / "FREEZE.json").read_text(encoding="utf-8"))
    assert freeze["claims"] == 4 and set(freeze["input_sha256"]) == {"draft", "annotator_a", "annotator_b",
                                                                      "adjudication"}
    with pytest.raises(SystemExit):
        annotate.main(["tasks", "--draft", str(draft), "--out", str(tmp_path / "t"), "--seed-a", "3", "--seed-b", "3"])
