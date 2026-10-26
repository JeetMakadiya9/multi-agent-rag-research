"""Record-only diagnostics of the Arm C verifiers and the Studio evaluation tooling (Candidate A record slice).

Every claim, passage and model reply here is SYNTHETIC TEST DATA, answered by a scripted fake model; nothing is sent
anywhere and no dataset is read. The expected verdicts are the existing behaviour, as captured from the code before the
diagnostics were added: these tests check that the new fields are recorded and that nothing else changed.
"""
import json
from pathlib import Path

import pytest

from docqa.agentic import ArmCVerifier, StudioArmCVerifier
from docqa.index import Evidence
from docqa.llm import LLMResponse
from docqa.ranked import ADJUDICATED
from docqa.verify import CONTRADICTED, INSUFFICIENT, SUPPORTED
from evaluation.experiments.agentic_verify import prompts
from evaluation.studio_eval import alt_benchmarks as alt
from evaluation.studio_eval import run_eval, score
from webrag.evidence import Passage
from webrag.orchestrator import Orchestrator, Settings

CLAIM = "Drug X lowers blood pressure in adults."
SUP = "In a randomised trial of 400 adults, drug X lowered systolic blood pressure by 9 mmHg."
CON = "In adults, drug X had no effect on blood pressure compared with placebo in a large trial."
NEU = "Drug X is metabolised in the liver and has a half-life of twelve hours in most patients."
NEU2 = "Blood pressure is measured with a cuff placed on the upper arm of the seated patient."
FOUND = "A meta-analysis found that drug X reduced blood pressure in adults across eleven trials."
ROLES = {prompts.VERIFIER_SYSTEM["v1"]: "verifier", prompts.PLANNER_SYSTEM["v0"]: "planner",
         prompts.ADJUDICATOR_SYSTEM["v2"]: "adjudicator"}
ADJ = {"action": "ADJUDICATE"}
NO_DECISION = {"verdict": "INSUFFICIENT", "passage": "P1", "quote": ""}


def q(text):
    return " ".join(text.split()[:6])


def e(pid, stance, quote=""):
    return {"id": pid, "stance": stance, "quote": quote}


def V(*entries):
    return {"assessments": list(entries)}


class Scripted:
    """Answers each chat call from a fixed queue (dict/list sent as JSON, str as is); records every request."""

    model = "scripted"

    def __init__(self, replies):
        self.replies, self.requests = list(replies), []

    def chat(self, messages, *, response_format=None, temperature=0.0, num_predict=None, timeout=None):
        self.requests.append({"role": ROLES.get(messages[0]["content"], "other"), "format": response_format,
                              "temperature": temperature, "num_predict": num_predict})
        item = self.replies.pop(0)
        return LLMResponse(item if isinstance(item, str) else json.dumps(item), "stop", 10, 100, 0.25)


class Store:
    """The run's passages in packet order; a planner search returns FOUND."""

    def __init__(self, texts):
        self.texts = list(texts)
        self.sources = {f"S{i}": type("Src", (), {"origin": f"S{i}", "url": "https://example.org", "title": "t",
                                                  "access": "full_text"})() for i in range(1, len(texts) + 2)}

    def rank(self, query, k=6, per_source=2):
        if query == CLAIM:
            return [Passage(f"S{i}:p1:{i}", f"S{i}", 1, t, 1.0) for i, t in enumerate(self.texts, 1)][:k]
        n = len(self.texts) + 1
        return [Passage(f"S{n}:p1:f", f"S{n}", 1, FOUND, 0.5)][:k]


class PacketStore(Store):
    """``store_factory`` for ``run_eval.evaluate``: the claim's packet sources, in order."""

    def __init__(self, sources, models=None):
        super().__init__([s.text for s in sources])


def run_verifier(cls, texts, replies, attest=True):
    model = Scripted(replies)
    verifier = cls(model)
    verifier.attest = attest
    evidence = [Evidence(f"E{i}", f"S{i}", 1, t) for i, t in enumerate(texts, 1)]
    return verifier.verify(CLAIM, evidence, search=Store(texts).rank), model


def diag(verdict):
    return verdict.__dict__["passage_diagnostics"]


def d(reply_entry, grounded, raw_stance, studio_check_entry=None):
    """One passage's diagnostics; the Studio check's reading equals the scoring reading unless given."""
    return {"reply_entry": reply_entry, "studio_check_entry": studio_check_entry or reply_entry,
            "grounded": grounded, "raw_stance": raw_stance}


# ------------------------------------------------------------------------------------- per-passage reply accounting
def test_a_complete_valid_reply_marks_every_passage_valid_and_the_verdict_is_unchanged():
    verdict, model = run_verifier(StudioArmCVerifier, [SUP, NEU], [V(e("P1", "SUPPORTS", q(SUP)), e("P2", "NEUTRAL"))])
    assert (verdict.label, verdict.sub_state, verdict.stances) == (SUPPORTED, "", {"E1": "SUPPORTS", "E2": "NEUTRAL"})
    assert diag(verdict) == {"E1": d("valid", True, "SUPPORTS"), "E2": d("valid", True, "NEUTRAL")}
    assert model.requests == [{"role": "verifier", "format": "json", "temperature": 0.0, "num_predict": 900}]
    assert verdict.agentic_trace["reply_reconstruction"] == {"consistent": True, "batches": 1, "verifier_replies": 1,
                                                             "replies_matched": 1}


def test_a_missing_reply_entry_is_recorded_as_missing_while_the_stance_stays_neutral():
    verdict, _ = run_verifier(StudioArmCVerifier, [SUP, NEU], [V(e("P1", "SUPPORTS", q(SUP)))])
    assert (verdict.label, verdict.stances) == (SUPPORTED, {"E1": "SUPPORTS", "E2": "NEUTRAL"})
    assert diag(verdict)["E2"] == d("missing", True, "NEUTRAL")
    assert verdict.agentic_trace["unlisted_passages"] == ["P2"] and "not listed" in verdict.note


def test_an_invalid_stance_is_recorded_and_each_variant_keeps_its_existing_outcome():
    replies = [V(e("P1", "SUPPORTS", q(SUP)), e("P2", "MAYBE"))]
    studio, _ = run_verifier(StudioArmCVerifier, [SUP, NEU], replies)
    exact, _ = run_verifier(ArmCVerifier, [SUP, NEU], replies)
    assert (studio.label, studio.sub_state, studio.parse_ok) == (INSUFFICIENT, "ASSESSMENT_FAILED", False)
    assert (exact.label, exact.sub_state) == (SUPPORTED, "")
    for verdict in (studio, exact):
        assert diag(verdict)["E2"] == d("invalid_stance", True, "NEUTRAL")


def test_identifiers_are_kept_as_written_and_unusable_entries_are_listed_without_changing_the_matching():
    reply = V({"id": "[P1]", "stance": "SUPPORTS", "quote": q(SUP)}, {"id": " P2 ", "stance": "NEUTRAL"},
              {"id": 3, "stance": "NEUTRAL"}, "junk", {"id": "p3", "stance": "CONTRADICTS", "quote": q(NEU2)})
    verdict, _ = run_verifier(StudioArmCVerifier, [SUP, NEU, NEU2], [reply])
    assert (verdict.label, verdict.stances) == (SUPPORTED, {"E1": "SUPPORTS", "E2": "NEUTRAL", "E3": "NEUTRAL"})
    assert [d["reply_entry"] for d in diag(verdict).values()] == ["valid", "valid", "missing"]  # "p3" never matched P3
    assert verdict.agentic_trace["reply_identifiers"] == [{
        "assessments_list": True, "returned_ids": ["[P1]", " P2 ", "p3"],
        "stance_valid_for_scoring": [True, True, True],
        "rejected_entries": [{"index": 2, "reason": "id is not a string", "id": 3},
                             {"index": 3, "reason": "not an object", "id": None}]}]


def test_an_ungrounded_quote_is_recorded_with_its_raw_stance_and_the_demotion_is_unchanged():
    replies = [V(e("P1", "SUPPORTS", "words that are not in the passage at all"), e("P2", "NEUTRAL")), ADJ,
               {"verdict": "SUPPORTED", "passage": "P1", "quote": SUP[:40]}]
    studio, _ = run_verifier(StudioArmCVerifier, [SUP, NEU], replies)
    exact, _ = run_verifier(ArmCVerifier, [SUP, NEU], replies)
    assert (studio.label, studio.sub_state) == (INSUFFICIENT, "NO_DECISIVE_EVIDENCE")
    assert (exact.label, exact.sub_state) == (SUPPORTED, "ADJUDICATED_UNSIDED")
    for verdict in (studio, exact):
        assert verdict.stances["E1"] == "NEUTRAL"
        assert diag(verdict)["E1"] == d("valid", False, "SUPPORTS")


def test_an_explicit_neutral_stance_is_told_apart_from_a_missing_entry():
    verdict, model = run_verifier(StudioArmCVerifier, [NEU], [V(e("P1", "NEUTRAL")), ADJ, NO_DECISION])
    assert (verdict.label, verdict.sub_state) == (INSUFFICIENT, "NO_DECISIVE_EVIDENCE")
    assert diag(verdict)["E1"] == d("valid", True, "NEUTRAL")
    assert [r["role"] for r in model.requests] == ["verifier", "planner", "adjudicator"]


def test_duplicate_identifiers_in_one_reply_record_both_readings():
    # Scoring (the frozen study code) reads the first entry; the Studio's stance check reads the last. Both unchanged.
    first_valid, _ = run_verifier(StudioArmCVerifier, [SUP], [V(e("P1", "SUPPORTS", q(SUP)), e("P1", "MAYBE"))])
    assert (first_valid.label, first_valid.sub_state) == (INSUFFICIENT, "ASSESSMENT_FAILED")
    assert diag(first_valid)["E1"] == d("valid", True, "SUPPORTS", studio_check_entry="invalid_stance")
    first_invalid, _ = run_verifier(StudioArmCVerifier, [SUP], [V(e("P1", "MAYBE"), e("P1", "SUPPORTS", q(SUP))),
                                                                 ADJ, NO_DECISION])
    assert (first_invalid.label, first_invalid.sub_state) == (INSUFFICIENT, "NO_DECISIVE_EVIDENCE")
    assert diag(first_invalid)["E1"] == d("invalid_stance", True, "NEUTRAL", studio_check_entry="valid")
    assert first_invalid.agentic_trace["reply_identifiers"][0]["returned_ids"] == ["P1", "P1"]


def test_a_malformed_reply_is_recorded_per_attempt_and_still_fails_closed():
    verdict, model = run_verifier(StudioArmCVerifier, [SUP], ["not json", "still not json"])
    assert (verdict.label, verdict.sub_state, verdict.parse_ok) == (INSUFFICIENT, "ASSESSMENT_FAILED", False)
    assert [r["assessments_list"] for r in verdict.agentic_trace["reply_identifiers"]] == [False, False]
    assert diag(verdict) == {} and len(model.requests) == 2
    assert verdict.agentic_trace["reply_reconstruction"] == {"batches": 1, "verifier_replies": 2, "replies_matched": 2,
                                                             "consistent": True, "stopped_at_batch": 0}


# ------------------------------------------------------------------------------------- the two readings diverge
FILLER = [f"Unrelated passage number {i} about hospital parking and cafeteria opening hours." for i in range(1, 9)]
NINE = [SUP] + FILLER                                   # 9 passages: two verifier batches, P1-P8 and P9


def first_batch(*extra):
    return V(e("P1", "SUPPORTS", q(SUP)), *[e(f"P{i}", "NEUTRAL") for i in range(2, 9)], *extra)


def test_more_than_eight_passages_are_matched_batch_by_batch():
    verdict, model = run_verifier(StudioArmCVerifier, NINE, [first_batch(), V(e("P9", "NEUTRAL"))])
    assert (verdict.label, len(verdict.stances)) == (SUPPORTED, 9)
    assert [r["num_predict"] for r in model.requests] == [900, 900] and len(model.requests) == 2
    assert {k: v["reply_entry"] for k, v in diag(verdict).items()} == {f"E{i}": "valid" for i in range(1, 10)}
    assert verdict.agentic_trace["reply_reconstruction"] == {"consistent": True, "batches": 2, "verifier_replies": 2,
                                                             "replies_matched": 2}


def test_reply_matching_follows_the_frozen_batch_size_it_depends_on():
    # ``reply_entry`` mirrors the frozen judge's batching and single retry (docqa/agentic.py, _scored_reply_entries).
    # It reads the batch size from the verifier's frozen settings, so a different size is still matched batch by batch;
    # a change to the judge's retry or batch order in the frozen code would need _scored_reply_entries updated too.
    import dataclasses
    from docqa.agentic import frozen_settings
    assert frozen_settings().verifier_batch == 8
    model = Scripted([V(e("P1", "SUPPORTS", q(SUP)), *[e(f"P{i}", "NEUTRAL") for i in range(2, 5)]),
                      V(*[e(f"P{i}", "NEUTRAL") for i in range(5, 9)]), V()])
    verifier = StudioArmCVerifier(model)
    verifier.settings = dataclasses.replace(verifier.settings, verifier_batch=4)
    verdict = verifier.verify(CLAIM, [Evidence(f"E{i}", f"S{i}", 1, t) for i, t in enumerate(NINE, 1)],
                              search=Store(NINE).rank)
    assert verdict.label == SUPPORTED and len(model.requests) == 3
    assert verdict.agentic_trace["reply_reconstruction"] == {"consistent": True, "batches": 3, "verifier_replies": 3,
                                                             "replies_matched": 3}
    assert [diag(verdict)[f"E{i}"]["reply_entry"] for i in (1, 8, 9)] == ["valid", "valid", "missing"]


def test_an_id_listed_in_another_batchs_reply_is_missing_for_scoring_but_listed_for_the_studio_check():
    verdict, _ = run_verifier(StudioArmCVerifier, NINE, [first_batch(e("P9", "SUPPORTS", q(SUP))), V()])
    assert (verdict.label, verdict.stances["E9"]) == (SUPPORTED, "NEUTRAL")       # existing outcome
    assert diag(verdict)["E9"] == d("missing", True, "NEUTRAL", studio_check_entry="valid")
    assert verdict.agentic_trace["unlisted_passages"] == []                       # the existing check's reading


def test_an_id_repeated_by_a_later_batch_keeps_its_scoring_reading_and_the_existing_fail_closed_outcome():
    replies = [first_batch(), V(e("P9", "NEUTRAL"), e("P1", "MAYBE"))]
    studio, _ = run_verifier(StudioArmCVerifier, NINE, replies)
    exact, _ = run_verifier(ArmCVerifier, NINE, replies)
    assert (studio.label, studio.sub_state) == (INSUFFICIENT, "ASSESSMENT_FAILED")  # existing: the check reads the last
    assert (exact.label, exact.sub_state) == (SUPPORTED, "")
    for verdict in (studio, exact):
        assert verdict.stances["E1"] == "SUPPORTS"
        assert diag(verdict)["E1"] == d("valid", True, "SUPPORTS", studio_check_entry="invalid_stance")


def test_a_search_batch_reply_repeating_a_claim_id_is_attributed_to_its_own_batch():
    replies = [V(e("P1", "NEUTRAL"), e("P2", "NEUTRAL")), {"action": "SEARCH", "query": "drug X meta-analysis"},
               V(e("P3", "SUPPORTS", q(FOUND)), e("P1", "MAYBE")), ADJ, NO_DECISION]
    studio, model = run_verifier(StudioArmCVerifier, [NEU, NEU2], replies)
    exact, _ = run_verifier(ArmCVerifier, [NEU, NEU2], replies)
    assert (studio.label, studio.sub_state) == (INSUFFICIENT, "ASSESSMENT_FAILED")
    assert (exact.label, exact.stances) == (SUPPORTED, {"E1": "NEUTRAL", "E2": "NEUTRAL", "S1": "SUPPORTS"})
    assert [r["role"] for r in model.requests] == ["verifier", "planner", "verifier", "planner"]
    for verdict in (studio, exact):
        assert diag(verdict)["E1"] == d("valid", True, "NEUTRAL", studio_check_entry="invalid_stance")
        assert diag(verdict)["S1"] == d("valid", True, "SUPPORTS")
        assert verdict.agentic_trace["reply_reconstruction"]["batches"] == 2


def test_a_retry_after_an_unusable_reply_and_an_omitted_entry():
    verdict, model = run_verifier(StudioArmCVerifier, [SUP, NEU], ["not json", V(e("P1", "SUPPORTS", q(SUP)))])
    assert (verdict.label, len(model.requests)) == (SUPPORTED, 2)
    assert diag(verdict)["E2"] == d("missing", True, "NEUTRAL")
    assert verdict.agentic_trace["reply_reconstruction"] == {"consistent": True, "batches": 1, "verifier_replies": 2,
                                                             "replies_matched": 2}


def test_nested_and_malformed_identifiers_are_recorded_by_type_and_a_missing_stance_is_read_two_ways():
    reply = V({"id": {"x": 1}, "stance": "SUPPORTS"}, {"id": ["P2"], "stance": "NEUTRAL"}, {"id": None},
              e("P1", "SUPPORTS", q(SUP)), {"id": "P2"})
    studio, _ = run_verifier(StudioArmCVerifier, [SUP, NEU], [reply])
    exact, _ = run_verifier(ArmCVerifier, [SUP, NEU], [reply])
    assert (studio.label, studio.sub_state) == (INSUFFICIENT, "ASSESSMENT_FAILED")  # existing: no stance -> invalid
    assert (exact.label, exact.sub_state) == (SUPPORTED, "")
    assert studio.agentic_trace["reply_identifiers"] == [{
        "assessments_list": True, "returned_ids": ["P1", "P2"], "stance_valid_for_scoring": [True, True],
        "rejected_entries": [{"index": 0, "reason": "id is not a string", "id": "dict"},
                             {"index": 1, "reason": "id is not a string", "id": "list"},
                             {"index": 2, "reason": "id is not a string", "id": None}]}]
    for verdict in (studio, exact):                                   # scoring reads the missing stance as NEUTRAL
        assert diag(verdict)["E2"] == d("valid", True, "NEUTRAL", studio_check_entry="invalid_stance")


def test_replies_that_cannot_be_matched_to_batches_give_no_reply_entry_and_say_why():
    from docqa.agentic import StudioModel

    class Unrecorded(StudioModel):                                    # a model wrapper that keeps no reply record
        def complete(self, *args, **kwargs):
            text = super().complete(*args, **kwargs)
            self.reply_ids.clear()
            return text

    class Verifier(StudioArmCVerifier):
        model_class = Unrecorded

    verdict, _ = run_verifier(Verifier, [SUP, NEU], [V(e("P1", "SUPPORTS", q(SUP)), e("P2", "NEUTRAL"))])
    assert (verdict.label, verdict.stances) == (SUPPORTED, {"E1": "SUPPORTS", "E2": "NEUTRAL"})
    assert [v["reply_entry"] for v in diag(verdict).values()] == [None, None]
    assert [v["studio_check_entry"] for v in diag(verdict).values()] == ["valid", "valid"]
    reconstruction = verdict.agentic_trace["reply_reconstruction"]
    assert reconstruction["consistent"] is False and "no usable recorded reply" in reconstruction["reason"]


def test_conflicting_stances_are_recorded_and_still_adjudicated():
    verdict, _ = run_verifier(StudioArmCVerifier, [SUP, CON], [V(e("P1", "SUPPORTS", q(SUP)),
                                                                 e("P2", "CONTRADICTS", q(CON))),
                                                               {"verdict": "CONTRADICTED", "passage": "P2",
                                                                "quote": CON[:40]}])
    assert (verdict.label, verdict.sub_state) == (CONTRADICTED, ADJUDICATED)
    assert [d["raw_stance"] for d in diag(verdict).values()] == ["SUPPORTS", "CONTRADICTS"]


def test_the_reply_text_given_to_the_study_code_is_the_model_reply_unchanged():
    from docqa.agentic import StudioModel
    from evaluation.experiments.agentic_verify.llm import Usage
    text = '  {"assessments": [{"id": " [P1] ", "stance": "supports", "quote": "x"}]}  '
    model = StudioModel(Scripted([text]))
    assert model.complete("verifier", prompts.VERIFIER_SYSTEM["v1"], "u", 900, Usage()) == text
    assert model.assessed == {"P1": True} and model.quotes == {"P1": "x"}            # the existing reading


# ------------------------------------------------------------------------------------- attestation records
def test_attestation_records_gain_reply_entry_and_keep_the_existing_assessed_field():
    verdict, _ = run_verifier(StudioArmCVerifier, [SUP, NEU], [V(e("P1", "SUPPORTS", q(SUP)))], attest=True)
    records = verdict.quote_attestations
    assert records["E1"][0]["reply_entry"] == "valid" and records["E2"][0]["reply_entry"] == "missing"
    assert records["E2"][0]["studio_check_entry"] == "missing"
    assert records["E1"][0]["assessed"] is True and records["E2"][0]["assessed"] is True       # unchanged by design
    plain, _ = run_verifier(StudioArmCVerifier, [SUP, NEU], [V(e("P1", "SUPPORTS", q(SUP)))], attest=False)
    assert "quote_attestations" not in plain.__dict__ and diag(plain) == diag(verdict)


# ------------------------------------------------------------------------------------- Studio records
def studio(replies, store_factory=None):
    model = Scripted(replies)
    kwargs = {"store_factory": store_factory} if store_factory else {}
    orch = Orchestrator(Settings(save=False, verifier="arm_c_studio", quote_attestation=True), client=model,
                        providers=[], session=object(), **kwargs)
    return orch, model


def test_passage_records_carry_the_diagnostics_including_a_searched_passage():
    orch, _ = studio([V(e("P1", "NEUTRAL")), {"action": "SEARCH", "query": "drug X meta-analysis"},
                      V(e("P3", "SUPPORTS", q(FOUND))), ADJ])
    store = Store([NEU, NEU2])
    result = orch._verify(CLAIM, "input claim", store.rank(CLAIM, k=8), store)
    assert (result.label, [p["stance"] for p in result.passages]) == (SUPPORTED, ["NEUTRAL", "NEUTRAL", "SUPPORTS"])
    assert [p["verifier_diagnostics"]["reply_entry"] for p in result.passages] == ["valid", "missing", "valid"]
    assert result.passages[2]["found_by_search"] == "drug X meta-analysis"
    assert result.passages[2]["verifier_diagnostics"] == d("valid", True, "SUPPORTS")


def packet_claim(cid, *texts):
    return {"claim_id": cid, "claim": CLAIM, "label": "SUPPORTED", "subgroups": [],
            "passages": [{"url": f"https://example.org/{cid}/{i}", "title": "t", "text": t}
                         for i, t in enumerate(texts, 1)]}


def test_evaluation_records_carry_aligned_diagnostics_and_claims_do_not_share_reply_state():
    orch, model = studio([V(e("P1", "SUPPORTS", q(SUP)), e("P2", "NEUTRAL")), V(e("P1", "SUPPORTS", q(SUP)))],
                         store_factory=PacketStore)
    alt.capture(orch)
    options = {"verifier": "arm_c_studio", "injection_guard": "off", "quote_attestation": True}
    records = []
    for claim in (packet_claim("c1", SUP, NEU), packet_claim("c2", SUP, NEU)):   # the same P1/P2 ids in both
        orch._alt_last_claim = None
        [record] = run_eval.evaluate([claim], orch, options)
        record.update(alt.trace_fields(orch))
        records.append(record)
    assert [r["verdict"] for r in records] == [SUPPORTED, SUPPORTED]
    assert [[x["reply_entry"] for x in r["passage_diagnostics"]] for r in records] == [["valid", "valid"],
                                                                                         ["valid", "missing"]]
    assert [len(r["reply_identifiers"]) for r in records] == [1, 1] and len(model.requests) == 2
    assert [x["reply_entry"] for x in records[1]["quotes"]] == ["valid", "missing"]
    assert [x["studio_check_entry"] for x in records[1]["quotes"]] == ["valid", "missing"]


# ------------------------------------------------------------------------------------- scoring and publishing
def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


GOLD = ["SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO", "NOT_ENOUGH_INFO"]
ROWS = [{"claim_id": "c0", "gold": "SUPPORTED", "verdict": "SUPPORTED", "sub_state": ""},
        {"claim_id": "c1", "gold": "CONTRADICTED", "verdict": "SUPPORTED", "sub_state": ""},
        {"claim_id": "c0", "gold": "SUPPORTED", "verdict": "SUPPORTED", "sub_state": ""},              # agrees
        {"claim_id": "c1", "gold": "CONTRADICTED", "verdict": "CONTRADICTED", "sub_state": ""},        # disagrees
        {"claim_id": "c2", "gold": "NOT_ENOUGH_INFO", "verdict": None, "sub_state": "SERVICE_ERROR"}]


def claims_file(tmp_path):
    return write_jsonl(tmp_path / "claims.jsonl", [
        {"claim_id": f"c{i}", "claim": "x", "label": g, "passages": [{"url": "https://e.org", "text": "t"}]}
        for i, g in enumerate(GOLD)])


def test_duplicate_rows_without_run_identity_are_unresolved_and_no_row_is_scored(tmp_path):
    """F5c (approved policy D): these legacy rows carry no run identity, so the duplicates cannot be attributed and
    the input is rejected. Before F5c the last row of each claim was scored silently."""
    from evaluation.studio_eval.records import RecordIntegrityError
    with pytest.raises(RecordIntegrityError) as caught:
        alt.score_records(write_jsonl(tmp_path / "r.jsonl", ROWS), None, claims_file(tmp_path))
    assert [u["claim_id"] for u in caught.value.diagnostics["unresolved"]] == ["c0", "c1"]


# One identified run whose duplicate rows agree on every scoring field (c1's second row differs only in timing).
ONE_RUN = [dict(r, run_id="run-a") for r in
           (ROWS[0], dict(ROWS[1], seconds=1.0), ROWS[2], dict(ROWS[1], seconds=2.0), ROWS[4])]


def publish_base(tmp_path):
    """A synthetic publish folder: a retro set, its records and two manifests that differ in fingerprint and head."""
    base = tmp_path / "base"
    (base / "sets" / "retro").mkdir(parents=True)
    claims_file(base / "sets" / "retro")
    (base / "sets" / "retro" / "FREEZE_ALT.json").write_text(json.dumps({"claims_sha256": "f" * 64}), encoding="utf-8")
    run_dir = base / "runs" / "retro_arm_c_studio"
    run_dir.mkdir(parents=True)
    write_jsonl(run_dir / "records.jsonl", ONE_RUN)
    for stamp, fp, head in (("20260101T000000Z", "a" * 64, "1" * 40), ("20260102T000000Z", "b" * 64, "2" * 40)):
        (run_dir / f"manifest_{stamp}.json").write_text(json.dumps({
            "started_utc": stamp, "git_head": head, "claims_sha256": "f" * 64, "options": {"verifier": "arm_c_studio"},
            "config_fingerprint": fp, "model": "scripted", "resumed_with": 0}), encoding="utf-8")
    return base, run_dir


def test_publish_lists_every_manifest_flags_differences_and_keeps_the_first_manifest_fields(tmp_path):
    base, run_dir = publish_base(tmp_path)
    entry = alt.publish(base)["sets"]["retro"]
    assert (entry["git_head"], entry["config_fingerprint"]) == ("1" * 40, "a" * 64)          # still the first
    assert [m["git_head"] for m in entry["manifests"]] == ["1" * 40, "2" * 40]
    assert entry["manifest_consistency"] == {"manifests": 2, "unreadable_manifests": 0,
                                             "config_fingerprints_differ": True, "git_heads_differ": True,
                                             "claims_sha256_differ": False, "verifiers_differ": False}
    scored = alt.score_records(run_dir / "records.jsonl", None, base / "sets" / "retro" / "claims.jsonl")
    assert entry["system"] == {k: scored["system"][k] for k in alt.AGGREGATE_KEYS}           # aggregates unchanged
    assert (entry["duplicate_records"]["duplicate_groups"], entry["duplicate_records"]["agreeing_duplicates"]) == (2, 2)


# ------------------------------------------------------------------------------------- run manifests
def test_environment_metadata_makes_no_model_call_and_changes_no_client_setting():
    class Client:
        model, base_url, num_predict, think, timeout = "qwen3:4b", "http://127.0.0.1:11434", 512, False, 180

        def chat(self, *args, **kwargs):
            raise AssertionError("the manifest must not call the model")

        def describe(self):
            raise AssertionError("the manifest must not query the model server")

    client = Client()
    env = run_eval.environment(type("Orch", (), {"client": client})(), git_state=lambda: {"dirty": True})
    assert env["git"] == {"dirty": True}
    assert env["model_client"] == {"class": "Client", "model": "qwen3:4b", "base_url": "http://127.0.0.1:11434",
                                   "num_predict": 512, "think": False, "timeout": 180}
    assert (client.num_predict, client.think, client.timeout) == (512, False, 180)
    assert set(env["not_collected"]) == {"model_digest", "loaded_context", "retrieval_model_revisions"}
    assert "docqa/agentic.py" in env["file_sha256"] and "rag.py" in env["file_sha256"]
    assert env["file_sha256"]["docqa/agentic.py"] == run_eval.sha256_file(run_eval.ROOT / "docqa/agentic.py")


def test_a_benchmark_run_writes_the_environment_and_makes_exactly_the_verify_calls(tmp_path, monkeypatch):
    replies = [V(e("P1", "SUPPORTS", q(SUP)), e("P2", "NEUTRAL"))]
    orch, model = studio(list(replies), store_factory=PacketStore)
    options = {"verifier": "arm_c_studio", "injection_guard": "off", "quote_attestation": True}
    monkeypatch.setattr(run_eval, "build_system", lambda verifier=None: (orch, options))
    monkeypatch.setattr(run_eval, "_git_state", lambda: {"dirty": False, "changed_paths": 0})
    claims = write_jsonl(tmp_path / "claims.jsonl", [packet_claim("c1", SUP, NEU)])
    assert alt.run(claims, tmp_path / "out", None, None) == 0
    [manifest] = [json.loads(p.read_text(encoding="utf-8")) for p in (tmp_path / "out").glob("manifest_*.json")]
    assert manifest["environment"]["git"] == {"dirty": False, "changed_paths": 0}
    assert manifest["environment"]["model_client"]["model"] == "scripted"
    [record] = alt._jsonl(tmp_path / "out" / "records.jsonl")
    assert (record["verdict"], record["stances"]) == (SUPPORTED, ["SUPPORTS", "NEUTRAL"])
    assert model.requests == [{"role": "verifier", "format": "json", "temperature": 0.0, "num_predict": 900}]


# ------------------------------------------------------------------------------------- error tolerance
@pytest.mark.parametrize("content", ["{not json", "[1, 2]"])
def test_an_unreadable_later_manifest_is_listed_with_its_error_and_publishing_continues(tmp_path, content):
    base, run_dir = publish_base(tmp_path)
    reference = alt.publish(base)["sets"]["retro"]
    (run_dir / "manifest_20260103T000000Z.json").write_text(content, encoding="utf-8")
    entry = alt.publish(base)["sets"]["retro"]
    assert entry["system"] == reference["system"] and entry["git_head"] == reference["git_head"]
    bad = entry["manifests"][2]
    assert bad["file"] == "manifest_20260103T000000Z.json" and set(bad) == {"file", "error"}
    assert bad["error"].startswith(("JSONDecodeError", "ValueError")) and len(bad["error"]) <= 200
    assert entry["manifest_consistency"] == dict(reference["manifest_consistency"], manifests=3,
                                                 unreadable_manifests=1)


def test_an_unreadable_first_manifest_still_stops_publishing_as_before(tmp_path):
    base, run_dir = publish_base(tmp_path)
    (run_dir / "manifest_20260101T000000Z.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):                        # unchanged: the entry is built from it
        alt.publish(base)


def test_an_unreadable_listed_file_is_reported_in_the_manifest_instead_of_stopping_the_run(monkeypatch):
    original = run_eval.sha256_file

    def failing(path):
        if str(path).replace("\\", "/").endswith("docqa/agentic.py"):
            raise PermissionError("access denied " + "x" * 500)
        return original(path)

    monkeypatch.setattr(run_eval, "sha256_file", failing)
    env = run_eval.environment(type("Orch", (), {"client": None})(), git_state=lambda: {"dirty": None})
    assert list(env["file_errors"]) == ["docqa/agentic.py"] and "docqa/agentic.py" not in env["file_sha256"]
    message = env["file_errors"]["docqa/agentic.py"]
    assert message.startswith("PermissionError: access denied") and len(message) == 200 and message.endswith("...")
    assert env["file_sha256"]["rag.py"] == original(run_eval.ROOT / "rag.py")       # the other files are still hashed


def test_environment_reports_no_file_errors_when_every_file_reads():
    env = run_eval.environment(type("Orch", (), {"client": None})(), git_state=lambda: {"dirty": None})
    assert env["file_errors"] == {} and set(env["file_sha256"]) <= set(run_eval.RUN_FILES)


# ------------------------------------------------------------------------------------- local reproducibility metadata
@pytest.mark.parametrize("raw, safe", [
    ("http://127.0.0.1:11434", "http://127.0.0.1:11434"),
    ("http://user:secret@host.example:11434/api?key=abc#frag", "http://***@host.example:11434/api?[redacted]"),
    ("https://token@service.example/v1", "https://***@service.example/v1"),
    ("0.0.0.0:11434", "0.0.0.0:11434"),
    ("user:secret@10.0.0.5:11434", "***@10.0.0.5:11434"),
    ("http://[::1]:11434", "http://[::1]:11434"),
    ("http://host.example:notaport", "[unparseable address]"),
    (None, None), ("", ""),
])
def test_addresses_are_recorded_without_credentials_query_or_fragment(raw, safe):
    assert run_eval.safe_url(raw) == safe


def test_credentials_in_the_client_address_or_ollama_host_never_reach_the_manifest(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://admin:hunter2@ollama.internal:11434")
    client = type("Client", (), {"model": "m", "base_url": "https://key-123@llm.example/v1?token=abc",
                                 "num_predict": 1, "think": False, "timeout": 1})()
    env = run_eval.environment(type("Orch", (), {"client": client})(), git_state=lambda: {"dirty": None})
    text = json.dumps(env)
    assert "hunter2" not in text and "key-123" not in text and "token=abc" not in text
    assert env["model_client"]["base_url"] == "https://***@llm.example/v1?[redacted]"
    assert env["ollama_settings"]["OLLAMA_HOST"] == "http://***@ollama.internal:11434"


def home_forms(home):
    """Every way the home folder can appear in a JSON-encoded record (plain, forward-slash, JSON-escaped)."""
    forms = {str(home), home.as_posix(), json.dumps(str(home))[1:-1], json.dumps(home.as_posix())[1:-1]}
    return {f.lower() for f in forms}


def test_the_manifest_holds_no_home_folder_path():
    env = run_eval.environment(type("Orch", (), {"client": None})(), git_state=lambda: {"dirty": None})
    home = Path.home()
    text = json.dumps(env).lower()
    assert not [form for form in home_forms(home) if form in text]
    assert run_eval.safe_path(str(home / "envs" / "python.exe")) == "~/envs/python.exe"
    assert run_eval.safe_path(str(run_eval.ROOT / "rag.py")) == "<project>/rag.py"
    message = run_eval.short_error(PermissionError(13, "denied", str(home / "secret" / "file.txt")))
    assert "~" in message and not [form for form in home_forms(home) if form in message.lower()]


@pytest.mark.skipif(__import__("os").name != "nt", reason="Windows paths are case-insensitive")
def test_a_differently_cased_home_path_is_scrubbed_on_windows():
    home = Path.home()
    assert run_eval._scrub(f"file {str(home).upper()}\\x.txt and {home.as_posix().lower()}/y") == "file ~\\x.txt and ~/y"


def test_a_missing_home_folder_does_not_stop_the_manifest(monkeypatch):
    def no_home():
        raise RuntimeError("Could not determine home directory.")
    monkeypatch.setattr(Path, "home", staticmethod(no_home))
    env = run_eval.environment(type("Orch", (), {"client": None})(), git_state=lambda: {"dirty": None})
    assert env["interpreter"]["executable"] and run_eval.short_error(OSError("x")) == "OSError: x"


def test_a_path_outside_the_project_and_home_is_still_scrubbed_and_bounded():
    assert run_eval.safe_path("Z:/elsewhere/python.exe") == "Z:/elsewhere/python.exe"
    assert len(run_eval.safe_path("Z:/" + "d" * 500)) == run_eval.MAX_TEXT


def test_other_ollama_and_hugging_face_values_are_scrubbed_and_bounded(monkeypatch):
    home = Path.home()
    monkeypatch.setenv("OLLAMA_KEEP_ALIVE", str(home / "notes") + "x" * 400)
    monkeypatch.setenv("HF_HUB_OFFLINE", str(home))
    env = run_eval.environment(type("Orch", (), {"client": None})(), git_state=lambda: {"dirty": None})
    assert env["ollama_settings"]["OLLAMA_KEEP_ALIVE"].startswith("~") and \
        len(env["ollama_settings"]["OLLAMA_KEEP_ALIVE"]) == run_eval.MAX_TEXT
    assert env["hf_settings"]["HF_HUB_OFFLINE"] == "~"


def test_an_address_without_a_scheme_but_with_one_in_its_query_keeps_its_password_out():
    assert run_eval.safe_url("user:pw@host.example/x?u=http://y") == "***@host.example/x?[redacted]"
    assert run_eval.safe_url(object()) == "<object>"


def test_dirty_paths_are_listed_relative_and_capped():
    class Done:
        def __init__(self, stdout):
            self.stdout = stdout

    out = run_eval._git_state(runner=lambda *a, **k: Done(" M docqa/agentic.py\n?? tests/new.py\nR  a.py -> b.py\n"))
    assert out == {"dirty": True, "changed_paths": 3, "paths_truncated": False,
                   "paths": [{"status": " M", "path": "docqa/agentic.py"}, {"status": "??", "path": "tests/new.py"},
                             {"status": "R ", "path": "a.py -> b.py"}]}
    many = run_eval._git_state(runner=lambda *a, **k: Done("".join(f"?? f{i}.py\n" for i in range(205))))
    assert (many["changed_paths"], len(many["paths"]), many["paths_truncated"]) == (205, run_eval.MAX_DIRTY_PATHS, True)

    def broken(*a, **k):
        raise OSError("git is not installed")
    assert run_eval._git_state(runner=broken) == {"dirty": None, "error": "OSError"}

    import subprocess
    calls = []

    def slow(*a, **k):
        calls.append(k)
        raise subprocess.TimeoutExpired(a[0], k["timeout"])
    assert run_eval._git_state(runner=slow) == {"dirty": None, "error": "TimeoutExpired"}
    assert (calls[0]["encoding"], calls[0]["errors"], calls[0]["timeout"]) == ("utf-8", "replace", 60)

    def undecodable(*a, **k):
        raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")
    assert run_eval._git_state(runner=undecodable) == {"dirty": None, "error": "UnicodeDecodeError"}


def test_package_versions_come_from_installed_metadata_and_a_missing_package_is_listed_not_guessed():
    from importlib import metadata
    out = run_eval._package_versions(("numpy", "no-such-package-for-this-test"))
    assert out["versions"] == {"numpy": metadata.version("numpy")}
    assert out["unavailable"] == {"no-such-package-for-this-test": "no installed package metadata"}


def test_the_configured_retrieval_models_are_read_from_rag_py_without_importing_it(tmp_path):
    assert run_eval._configured_models() == {"embedding_model": "all-MiniLM-L6-v2",
                                             "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                                             "source": "rag.py"}
    assert "error" in run_eval._configured_models(tmp_path / "absent.py")
    (tmp_path / "other.py").write_text("X = 1\n", encoding="utf-8")
    assert run_eval._configured_models(tmp_path / "other.py") == {"error": "the model names were not found in rag.py"}
    (tmp_path / "half.py").write_text('EMBEDDING_MODEL_NAME = "e"\n', encoding="utf-8")
    assert run_eval._configured_models(tmp_path / "half.py") == {"embedding_model": "e", "source": "half.py",
                                                                 "not_found": ["RERANKER_MODEL_NAME"]}


def test_the_options_sent_to_the_local_model_are_recorded_from_the_code_and_unsent_ones_are_named():
    from docqa.llm import OllamaChat
    client = OllamaChat()                                     # constructing it makes no request
    out = run_eval._model_options(type("Orch", (), {"client": client, "verifier": StudioArmCVerifier(client)})())
    assert out["established"] is True and out["not_sent"] == ["seed", "num_ctx", "num_gpu"]
    assert out["sent_per_call"] == {"format": "json", "temperature": 0.0,
                                    "num_predict_by_role": {"verifier": 900, "planner": 300, "adjudicator": 200}}
    assert out["sent_by_client"] == {"stream": True, "think": False, "num_predict_default": 512}
    other = run_eval._model_options(type("Orch", (), {"client": Scripted([]), "verifier": None})())
    assert other["established"] is False and "sent_per_call" not in other
    from docqa.agentic_candidates import StudioArmCPromptP2Verifier          # wraps StudioModel
    wrapped = run_eval._model_options(type("Orch", (), {"client": client,
                                                        "verifier": StudioArmCPromptP2Verifier(client)})())
    assert wrapped["sent_per_call"] == "set by this verifier's own calls; not recorded here"

    class Subclass(OllamaChat):
        pass
    sub = run_eval._model_options(type("Orch", (), {"client": Subclass(), "verifier": None})())
    assert sub["established"] is False


def test_a_listed_file_that_is_absent_is_reported_as_missing(monkeypatch):
    monkeypatch.setattr(run_eval, "RUN_FILES", run_eval.RUN_FILES + ("docqa/no_such_module.py",))
    env = run_eval.environment(type("Orch", (), {"client": None})(), git_state=lambda: {"dirty": None})
    assert env["file_missing"] == ["docqa/no_such_module.py"]
    assert {"docqa/ranked.py", "docqa/adjudicate.py"} <= set(env["file_sha256"])
