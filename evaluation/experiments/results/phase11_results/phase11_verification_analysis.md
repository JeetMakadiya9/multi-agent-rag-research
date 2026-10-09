# Phase 11 — Verification Analysis

## Classification label metrics

Only B and C emit a verifier-derived label. Accuracy denominator is labeled claims with successful execution and a valid output; unlabelled claims, failures, interruptions, and invalid/missing labels are not treated as incorrect or correct. A is not applicable.

| System | Labeled claims | Valid predictions | Correct | Accuracy | Macro F1 | Invalid excluded | Failed execution excluded |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B | 136 | 128 | 58 | 0.453 | 0.465 | 0 | 8 |
| C | 136 | 126 | 56 | 0.444 | 0.461 | 0 | 10 |

## Persisted verifier output records

B persisted evidence-component stance assessments; C persisted claim-level verdict assessments and nested evidence assessments. Raw stance/verdict counts are provided in `phase11_verification_analysis.json`. These counts describe saved output records and are not a separate gold-scored metric.

The selected-evidence precision/recall/F1 values are document-level overlap scores. Their denominators differ by metric because some values are unavailable on individual rows. They do not establish entailment. The unsupported-claim rate is **NOT AVAILABLE**: the persisted field is not a populated rate with an established denominator. `INSUFFICIENT_EVIDENCE` as an output label is not equivalent to an unlabeled gold claim or an unavailable evidence artifact.
