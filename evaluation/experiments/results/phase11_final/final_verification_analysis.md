# Verification and selected-evidence analysis

A has no verifier-derived classification. Verification execution success means a valid persisted verdict output per recorded verification call; accuracy is separately measured only against labeled gold claims.

| System | Valid verdict outputs / calls | Verdicts: SUPPORTED / CONTRADICTED / INSUFFICIENT | Labeled accuracy | Failed |
|---|---:|---|---:|---:|
| B | 280/298.0 (93.96%) | 108 / 8 / 164 | 87/179 (48.60%) | 9 labeled failures |
| C | 276/298.0 (92.62%) | 103 / 8 / 165 | 85/176 (48.30%) | 12 labeled failures |

Evidence assessment stance records (record-level, not claim-level correctness):

- B: {'CONTRADICTS': 10, 'NEUTRAL': 405, 'SUPPORTS': 157}
- C: {'CONTRADICTS': 9, 'NEUTRAL': 403, 'SUPPORTS': 152}

## Selected evidence document overlap

These compare selected document IDs with annotated gold evidence document IDs. They do not establish entailment. Empty selections contribute false negatives.

| System | TP / FP / FN | Precision | Recall | Micro F1 | Valid empty |
|---|---:|---:|---:|---:|---:|
| B | 98 / 13 / 102 | 88.288% | 49.000% | 0.6302 | 72 |
| C | 96 / 15 / 99 | 86.486% | 49.231% | 0.6275 | 71 |

B/C selected evidence sets match on 267 successful pairs and differ on 8. Retrieval relevance, evidence overlap, entailment, verifier stance and final correctness remain distinct.
