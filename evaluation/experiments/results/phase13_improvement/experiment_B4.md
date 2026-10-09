# Phase 13 — Experiment B4

**Hypothesis.** H4: the Qwen judge systematically under-detects contradiction.

- Baseline: B v1 (frozen Phase 10F System B), re-run live on the dev subset as B0; reference for this single-factor comparison: **B0**
- Changed component (only): pairwise judge SYSTEM message v1 -> v2 (explicit SUPPORTS/CONTRADICTS/NEUTRAL definitions); user payload unchanged
- Evidence limit 2, context window 0, prompt v2 (`825090298e51`), guards v1
- Dataset: SciFact DEV, Phase 13 development subset (engineering only; not a benchmark) — 69 claims

## Results (development subset only)

| Metric | Value |
|---|---|
| Accuracy | 29/66 = 43.9% |
| Macro F1 (2 gold classes) | 0.3656 |
| SUPPORTED P / R | 58.3% / 77.8% |
| CONTRADICTED P / R | 100.0% / 3.3% |
| INSUFFICIENT rate (valid labeled) | 17/66 = 25.8% |
| Failure rate | 4.3% `{'SUCCESS': 66, 'FAILED': 3}` |
| LLM calls total / mean | 153 / 2.22 |
| Latency mean / median (s) | 36.8 / 33.3 |
| Citation doc micro P / R / F1 | 65.7% / 58.7% / 0.6197 |
| Judge stances | `{'SUPPORTS': 101, 'NEUTRAL': 41, 'CONTRADICTS': 2}` |

Confusion matrix (rows gold, columns predicted SUP / CON / NEI):

- SUPPORTED: 28 / 0 / 8
- CONTRADICTED: 20 / 1 / 9

## Paired comparison vs B0

`{'pairs': 65, 'both_correct': 21, 'both_wrong': 35, 'reference_only_correct': 1, 'variant_only_correct': 8, 'net_gain_claims': 7, 'paired_accuracy_reference': 0.3384615384615385, 'paired_accuracy_variant': 0.4461538461538462, 'mcnemar_exact_p': 0.0390625, 'label_changes': 25, 'label_change_transitions': {'INSUFFICIENT_EVIDENCE->SUPPORTED (gold CONTRADICTED)': 16, 'INSUFFICIENT_EVIDENCE->SUPPORTED (gold SUPPORTED)': 8, 'CONTRADICTED->SUPPORTED (gold CONTRADICTED)': 1}}`

Changed claim IDs: [54, 115, 163, 208, 212, 219, 230, 239, 338, 415, 443, 452, 513, 597, 718, 728, 847, 1029, 1041, 1049, 1088, 1140, 1221, 1232, 1259]

## By retrieval bucket / rationale type

| Group | Valid | Correct | Accuracy | SUP acc | CON acc | NEI preds |
|---|---|---|---|---|---|---|
| A_top2 | 24 | 12 | 50.0% | 78.6% | 10.0% | 5 |
| B_rank3_8 | 21 | 9 | 42.9% | 81.8% | 0.0% | 6 |
| C_absent_top8 | 21 | 8 | 38.1% | 72.7% | 0.0% | 6 |
| single_sentence | 54 | 24 | 44.4% | 82.8% | 0.0% | 14 |
| multi_sentence | 12 | 5 | 41.7% | 57.1% | 20.0% | 3 |

## Failures (kept visible, excluded from accuracy denominators)

- claim 142: FAILED — Experiment B returned an incomplete assessment: U1 semantic judge: E2-C1: ValueError: Pair assessment must contain a valid SUPPORTS, CONTRADICTS, or NEUTRAL stance.
- claim 171: FAILED — Experiment B returned an incomplete assessment: U1 semantic judge: E2-C1: ValueError: Pair assessment must contain a valid SUPPORTS, CONTRADICTS, or NEUTRAL stance.
- claim 216: FAILED — Experiment B returned an incomplete assessment: U1 semantic judge: E1-C1: ValueError: Legacy judgement verdict is invalid.

Development-subset results inform engineering decisions only; the subset is stratified and is not comparable to the 179-claim full-DEV accuracy and is not a benchmark.
