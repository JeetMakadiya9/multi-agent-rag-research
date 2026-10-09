# Phase 11 — Paired B vs C Analysis

Valid pair rule: both B and C have a persisted `SUCCESS`, each has a valid predicted label, and both share a gold label in `SUPPORTED` or `CONTRADICTED`. The denominator is **126 paired claims**.

| Measure | Count / value |
| --- | --- |
| Valid paired predictions | 126 |
| Same prediction | 124 |
| Different prediction | 2 |
| Both correct | 55 |
| Both incorrect | 69 |
| B-only correct | 1 |
| C-only correct | 1 |
| Descriptive C−B accuracy difference | 0.000 |

Agreement: **124/126**; disagreement: **2/126**. Correctness cells sum to the paired denominator: both correct 55, both incorrect 69, B-only correct 1, C-only correct 1. B/C class-wise paired confusion matrices are in the JSON artifact. The persisted paired bootstrap uses 10,000 resamples and seed 0; its interval is descriptive uncertainty only.

**This analysis does not establish statistical significance or superiority.** No definitive hypothesis test is performed.
