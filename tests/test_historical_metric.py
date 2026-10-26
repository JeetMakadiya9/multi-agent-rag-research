"""The historical two-class metric beside the three-class one (evaluation/studio_eval/historical_metric.py)."""
from evaluation.studio_eval import historical_metric as hm
from evaluation.studio_eval import score
from evaluation.studio_eval.score import C, N, S


def test_nei_claims_are_outside_the_two_class_denominator():
    gold = [S, C, N, N]
    pred = [S, C, C, S]                                    # both NEI claims wrongly decided
    assert hm.two_class_macro_f1(gold, pred) == 1.0        # the historical metric does not see them
    assert score.macro_f1(gold, pred) < 0.5                # the three-class metric does


def test_an_insufficient_answer_on_a_labelled_claim_is_an_error_for_its_class_only():
    gold, pred = [S, S, C, C], [S, N, C, C]
    # SUPPORTED: tp 1, fn 1 -> 2/3; CONTRADICTED: tp 2, no fp -> 1.0
    assert abs(hm.two_class_macro_f1(gold, pred) - (2 / 3 + 1.0) / 2) < 1e-12


def test_wrong_side_counts_against_both_classes():
    gold, pred = [S, C], [C, S]
    assert hm.two_class_macro_f1(gold, pred) == 0.0


def test_both_reports_each_metric_with_its_denominator():
    gold = [S, C, N, N, S]
    records = [{"verdict": S}, {"verdict": "INSUFFICIENT"}, {"verdict": C}, {"verdict": "INSUFFICIENT"},
               {"verdict": S, "sub_state": "SERVICE_ERROR"}]   # a failure counts as an error
    out = hm.both(gold, records, n=200)
    assert out["three_class"]["claims"] == 5 and out["two_class"]["claims"] == 3
    assert out["nei_claims_decided"] == "1/2"
    assert out["two_class"]["macro_f1"] == round(hm.two_class_macro_f1(gold, [score.predicted(r) for r in records]), 4)
    low, high = out["two_class"]["ci95"]
    assert 0.0 <= low <= high <= 1.0
