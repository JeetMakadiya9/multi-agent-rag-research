# Classification analysis

A has no verifier-derived classification and is NOT APPLICABLE. B/C accuracy denominators include only labeled claims with SUCCESS and a valid persisted verifier label. Invalid outputs, execution failures and unlabeled claims are excluded and kept distinct.

| System | Correct / denominator | Accuracy | Macro F1 (2 gold-supported classes) | Failed labeled outputs |
|---|---:|---:|---:|---:|
| B | 87/179 | 48.603% | 0.4842 | 9 |
| C | 85/176 | 48.295% | 0.4834 | 12 |

Per-class measures/confusion matrices are in JSON. INSUFFICIENT_EVIDENCE has no gold support and is excluded from macro-F1 class denominator; its precision is still reported.
