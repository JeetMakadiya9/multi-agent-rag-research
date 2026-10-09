# Phase 11B — Final complete DEV results

The persisted run 02fb9c7a1268476ca1f9b12d9dbfb25e is COMPLETED: 900/900 attempts and 900/900 scored rows over 300/300 claims. Raw statuses are 840 SUCCESS, 55 FAILED, and 5 INTERRUPTED. All A/B/C claim pairs are present.

| System | Correct / valid labeled | Accuracy | Macro F1 |
|---|---:|---:|---:|
| B | 87/179 | 48.603% | 0.4842 |
| C | 85/176 | 48.295% | 0.4834 |

A is not applicable for classification. Paired B/C comparison uses 176 valid labeled pairs; both correct 84, both incorrect 90, B-only correct 1, C-only correct 1; exact McNemar p=1.0000. No statistically significant paired difference at 0.05.

| System | R@1 | R@5 | MRR including misses |
|---|---:|---:|---:|
| A | 69.856% (146/209) | 84.211% | 0.8305 |
| B | 70.192% (146/208) | 84.135% | 0.8322 |
| C | 70.244% (144/205) | 84.390% | 0.8286 |

Retrieval means annotated-document overlap, not entailment or factual correctness. B and C classification metrics use different valid-output denominators. The data do not establish B improves over A or C improves factual reliability over B. See component reports for metrics, denominators, failures and aggregation discrepancies.

Selected-evidence micro document-overlap results (not entailment): B precision 88.288%, recall 49.000%, F1 0.6302, denominator 179; C precision 86.486%, recall 49.231%, F1 0.6275, denominator 176.
Recall@10/@20 are NOT COMPUTED because no more than eight ranked chunks per example are persisted. Stored C failed-cycle counts have a 22-attempt discrepancy between top-level and embedded trace fields; see audit JSON.
