# Phase 13 — Experiment B3

**Hypothesis.** H3: deterministic qualifier/coordinated-object guards create false INSUFFICIENT_EVIDENCE decisions on biomedical language.

- Baseline: B v1 (frozen Phase 10F System B), re-run live on the dev subset as B0; reference for this single-factor comparison: **B0**
- Changed component (only): lexical guards v1 -> v2 (rules G1-G5); applied by exact deterministic re-aggregation of B0's recorded stances
- Evidence limit 2, context window 0, prompt v1 (`aa49d5e7dd47`), guards v2
- Dataset: SciFact DEV, Phase 13 development subset (engineering only; not a benchmark) — 69 claims

## Results (development subset only)

| Metric | Value |
|---|---|
| Accuracy | 23/68 = 33.8% |
| Macro F1 (2 gold classes) | 0.4088 |
| SUPPORTED P / R | 87.5% / 58.3% |
| CONTRADICTED P / R | 100.0% / 6.2% |
| INSUFFICIENT rate (valid labeled) | 42/68 = 61.8% |
| Failure rate | 1.4% `{'SUCCESS': 68, 'FAILED': 1}` |
| LLM calls total / mean | 153 / 2.22 |
| Latency mean / median (s) | 37.8 / 33.6 |
| Citation doc micro P / R / F1 | 83.3% / 32.5% / 0.4673 |
| Judge stances | `{'NEUTRAL': 103, 'SUPPORTS': 43, 'CONTRADICTS': 2}` |

Confusion matrix (rows gold, columns predicted SUP / CON / NEI):

- SUPPORTED: 21 / 0 / 15
- CONTRADICTED: 3 / 2 / 27

## Paired comparison vs B0

`{'pairs': 68, 'both_correct': 22, 'both_wrong': 45, 'reference_only_correct': 0, 'variant_only_correct': 1, 'net_gain_claims': 1, 'paired_accuracy_reference': 0.3235294117647059, 'paired_accuracy_variant': 0.3382352941176471, 'mcnemar_exact_p': 1.0, 'label_changes': 1, 'label_change_transitions': {'INSUFFICIENT_EVIDENCE->SUPPORTED (gold SUPPORTED)': 1}}`

Changed claim IDs: [1266]

## By retrieval bucket / rationale type

| Group | Valid | Correct | Accuracy | SUP acc | CON acc | NEI preds |
|---|---|---|---|---|---|---|
| A_top2 | 26 | 12 | 46.2% | 71.4% | 16.7% | 12 |
| B_rank3_8 | 21 | 7 | 33.3% | 63.6% | 0.0% | 13 |
| C_absent_top8 | 21 | 4 | 19.0% | 36.4% | 0.0% | 17 |
| single_sentence | 57 | 19 | 33.3% | 65.5% | 0.0% | 35 |
| multi_sentence | 11 | 4 | 36.4% | 28.6% | 50.0% | 7 |

## Failures (kept visible, excluded from accuracy denominators)

- claim 1339: FAILED — Experiment B returned an incomplete assessment: U1 semantic judge: E1-C1: ValueError: Legacy judgement verdict is invalid.

Development-subset results inform engineering decisions only; the subset is stratified and is not comparable to the 179-claim full-DEV accuracy and is not a benchmark.
