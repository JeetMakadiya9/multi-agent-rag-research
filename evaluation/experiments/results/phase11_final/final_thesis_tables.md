# Thesis-ready tables

## Table 1 — Dataset/configuration

| Dataset/scope | Claims | Corpus | Chunks | Run ID |
|---|---:|---:|---:|---|
| SciFact-Orig DEV | 300 | 5,183 docs | 45,972 | 02fb9c7a1268476ca1f9b12d9dbfb25e |

## Table 2 — Execution accounting

| System | Success | Failed | Interrupted | Planned |
|---|---:|---:|---:|---:|
| A | 284 | 15 | 1 | 300 |
| B | 280 | 18 | 2 | 300 |
| C | 276 | 22 | 2 | 300 |

## Table 3 — Classification

| System | Correct / valid labeled denominator | Accuracy | Macro F1 |
|---|---:|---:|---:|
| A | N/A | N/A | N/A |
| B | 87/179 | 48.603% | 0.4842 |
| C | 85/176 | 48.295% | 0.4834 |

## Table 4 — Retrieval

| System | R@1 | R@5 | MRR (misses zero) |
|---|---:|---:|---:|
| A | 69.856% | 84.211% | 0.8305 |
| B | 70.192% | 84.135% | 0.8322 |
| C | 70.244% | 84.390% | 0.8286 |

R@10/R@20 are NOT COMPUTED because per-claim persisted ranked lists contain at most eight chunks.

## Table 5 — Verification

| System | Correct / valid labeled | Accuracy | Failures |
|---|---:|---:|---:|
| B | 87/179 | 48.603% | 18 |
| C | 85/176 | 48.295% | 22 |

## Table 6 — Efficiency

| System | Success mean latency (s) | LLM | Retrieval | Verification | Cycles |
|---|---:|---:|---:|---:|---:|
| A | 26.49 | 299 | 299 | 0 | N/A |
| B | 39.64 | 645 | 298 | 298 | N/A |
| C | 38.58 | 608 | 596 | 298 | 596 |

## Table 7 — Failure categories

| Category | Count | % of 900 |
|---|---:|---:|
| interruption | 5 | 0.56% |
| verification_failure | 4 | 0.44% |
| controller_failure | 22 | 2.44% |
| local_model_endpoint_failure | 29 | 3.22% |

## Table 8 — Paired B/C correctness

| Both correct | Both wrong | B-only correct | C-only correct | McNemar p |
|---:|---:|---:|---:|---:|
| 84 | 90 | 1 | 1 | 1.0000 |

## Table 9 — C path validation

| Stage | Pass / audited successful traces |
|---|---:|
| phase_8a_validation | 276/276 |
| phase_8b_execution | 276/276 |
| phase_7b_dispatch | 276/276 |
| phase_6_retrieval | 276/276 |
| phase_6_verification | 276/276 |
| continuation_authorization | 276/276 |
| researcher_transition | 276/276 |
| completion | 276/276 |
