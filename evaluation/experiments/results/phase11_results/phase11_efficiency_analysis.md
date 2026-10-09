# Phase 11 — Efficiency Analysis

Durations are seconds from persisted attempt records. Successful and failed attempts are separated. Interrupted callbacks have no persisted timing. Totals count observed calls; unavailable call counts are not replaced with zero.

| System | Success latency mean (n) | Success median | Failure latency mean (n) | All timed latency mean (n) | LLM calls | Retrieval calls | Verification calls | Controller cycles |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 27.221 (n=205) | 27.216 | 9.580 (n=15) | 26.018 (n=220) | 220 | 220 | N/A | N/A |
| B | 40.834 (n=202) | 39.369 | 24.640 (n=17) | 39.577 (n=219) | 484 | 219 | 219 | N/A |
| C | 40.291 (n=199) | 38.639 | 23.964 (n=20) | 38.800 (n=219) | 448 | 438 | 219 | 438 |

Full mean/median/min/max, status-specific call counts, and denominators are retained in `phase11_efficiency_analysis.json`. The all-timed average combines successful and failed attempts with persisted timing; it is labeled accordingly. No full-run runtime projection is presented and these partial-run durations are not extrapolated to missing attempts.
