# Phase 13 — Offline Error Reconstruction (no inference)

Source: frozen Phase 10F per-example records (read-only) joined with SciFact DEV gold rationales after the fact. Buckets use retrieval ranks only, never predictions.

- **A_top2**: a gold rationale sentence is at rank 1–2 (the only ranks the frozen verifier judged).
- **B_rank3_8**: the first gold rationale sentence is at rank 3–8 (retrieved but never judged).
- **C_absent_top8**: no gold rationale sentence in the ≤8 retrieved chunks.

Reconciliation with Phase 11B/12 headline counts: **MATCH** (`{'B_valid_correct': [179, 87], 'expected_B': [179, 87], 'C_valid_correct': [176, 85], 'expected_C': [176, 85], 'B_errors_rationale_in_top8': 66, 'expected_phase12_B': 66, 'C_errors_rationale_in_top8': 65, 'expected_phase12_C': 65, 'matches': True}`).

## System B

### By rationale-rank bucket

| Group | Claims | Correct | Errors | Accuracy | SUPPORTED acc (n) | CONTRADICTED acc (n) |
|---|---|---|---|---|---|---|
| ALL | 179 | 87 | 92 | 48.6% | 68.4% (117) | 11.3% (62) |
| A_top2 | 108 | 63 | 45 | 58.3% | 76.7% (73) | 20.0% (35) |
| B_rank3_8 | 39 | 18 | 21 | 46.2% | 64.3% (28) | 0.0% (11) |
| C_absent_top8 | 32 | 6 | 26 | 18.8% | 37.5% (16) | 0.0% (16) |

### By rationale type

| Group | Claims | Correct | Errors | Accuracy | SUPPORTED acc (n) | CONTRADICTED acc (n) |
|---|---|---|---|---|---|---|
| single_sentence_rationale | 156 | 77 | 79 | 49.4% | 71.3% (101) | 9.1% (55) |
| multi_sentence_rationale | 23 | 10 | 13 | 43.5% | 50.0% (16) | 28.6% (7) |

### Bucket × rationale type

| Group | Claims | Correct | Errors | Accuracy | SUPPORTED acc (n) | CONTRADICTED acc (n) |
|---|---|---|---|---|---|---|
| A_top2|single | 95 | 56 | 39 | 58.9% | 79.7% (64) | 16.1% (31) |
| A_top2|multi | 13 | 7 | 6 | 53.8% | 55.6% (9) | 50.0% (4) |
| B_rank3_8|single | 37 | 18 | 19 | 48.6% | 69.2% (26) | 0.0% (11) |
| B_rank3_8|multi | 2 | 0 | 2 | 0.0% | 0.0% (2) | n/a (0) |
| C_absent_top8|single | 24 | 3 | 21 | 12.5% | 27.3% (11) | 0.0% (13) |
| C_absent_top8|multi | 8 | 3 | 5 | 37.5% | 60.0% (5) | 0.0% (3) |

### Complete rationale availability

| Group | Claims | Correct | Errors | Accuracy | SUPPORTED acc (n) | CONTRADICTED acc (n) |
|---|---|---|---|---|---|---|
| complete_in_top2 | 104 | 61 | 43 | 58.7% | 78.3% (69) | 20.0% (35) |
| partial_or_none_in_top2_but_complete_in_top8 | 40 | 19 | 21 | 47.5% | 65.5% (29) | 0.0% (11) |
| incomplete_in_top8 | 35 | 7 | 28 | 20.0% | 36.8% (19) | 0.0% (16) |

- Error bucket counts: `{'B_rank3_8': 21, 'A_top2': 45, 'C_absent_top8': 26}`
- Error transitions (gold->pred|bucket): `{'CONTRADICTED->INSUFFICIENT_EVIDENCE|A_top2': 20, 'CONTRADICTED->INSUFFICIENT_EVIDENCE|B_rank3_8': 10, 'CONTRADICTED->INSUFFICIENT_EVIDENCE|C_absent_top8': 13, 'CONTRADICTED->SUPPORTED|A_top2': 8, 'CONTRADICTED->SUPPORTED|B_rank3_8': 1, 'CONTRADICTED->SUPPORTED|C_absent_top8': 3, 'SUPPORTED->INSUFFICIENT_EVIDENCE|A_top2': 17, 'SUPPORTED->INSUFFICIENT_EVIDENCE|B_rank3_8': 10, 'SUPPORTED->INSUFFICIENT_EVIDENCE|C_absent_top8': 10}`
- INSUFFICIENT_EVIDENCE error causes: `{'guard_fired_no_supports': 6, 'all_neutral': 66, 'guard_overrode_supports': 8}`
- INSUFFICIENT_EVIDENCE causes over all valid labeled outputs: `{'guard_fired_no_supports': 6, 'all_neutral': 66, 'guard_overrode_supports': 8}`
- Judge stance counts (valid labeled): `{'NEUTRAL': 222, 'SUPPORTS': 138, 'CONTRADICTS': 8}`
- Stance on judged gold-rationale chunks (gold->pred|stance): `{'CONTRADICTED->CONTRADICTED|CONTRADICTS': 7, 'CONTRADICTED->CONTRADICTED|NEUTRAL': 1, 'CONTRADICTED->INSUFFICIENT_EVIDENCE|NEUTRAL': 19, 'CONTRADICTED->INSUFFICIENT_EVIDENCE|SUPPORTS': 3, 'CONTRADICTED->SUPPORTED|NEUTRAL': 4, 'CONTRADICTED->SUPPORTED|SUPPORTS': 6, 'SUPPORTED->INSUFFICIENT_EVIDENCE|NEUTRAL': 12, 'SUPPORTED->INSUFFICIENT_EVIDENCE|SUPPORTS': 10, 'SUPPORTED->SUPPORTED|NEUTRAL': 5, 'SUPPORTED->SUPPORTED|SUPPORTS': 56}`
- Claims where a lexical guard fired: all=14, errors=14, correct=0
- Failed labeled attempts (excluded from accuracy, still visible): 9 `{'InterruptedAttempt': 1, 'Experiment B returned an incomplete assessment': 8}` by bucket `{'retrieval_unavailable': 1, 'A_top2': 4, 'C_absent_top8': 2, 'B_rank3_8': 2}`

## System C

### By rationale-rank bucket

| Group | Claims | Correct | Errors | Accuracy | SUPPORTED acc (n) | CONTRADICTED acc (n) |
|---|---|---|---|---|---|---|
| ALL | 176 | 85 | 91 | 48.3% | 67.8% (115) | 11.5% (61) |
| A_top2 | 105 | 62 | 43 | 59.0% | 77.5% (71) | 20.6% (34) |
| B_rank3_8 | 39 | 17 | 22 | 43.6% | 60.7% (28) | 0.0% (11) |
| C_absent_top8 | 32 | 6 | 26 | 18.8% | 37.5% (16) | 0.0% (16) |

### By rationale type

| Group | Claims | Correct | Errors | Accuracy | SUPPORTED acc (n) | CONTRADICTED acc (n) |
|---|---|---|---|---|---|---|
| single_sentence_rationale | 154 | 74 | 80 | 48.1% | 69.7% (99) | 9.1% (55) |
| multi_sentence_rationale | 22 | 11 | 11 | 50.0% | 56.2% (16) | 33.3% (6) |

### Bucket × rationale type

| Group | Claims | Correct | Errors | Accuracy | SUPPORTED acc (n) | CONTRADICTED acc (n) |
|---|---|---|---|---|---|---|
| A_top2|single | 93 | 54 | 39 | 58.1% | 79.0% (62) | 16.1% (31) |
| A_top2|multi | 12 | 8 | 4 | 66.7% | 66.7% (9) | 66.7% (3) |
| B_rank3_8|single | 37 | 17 | 20 | 45.9% | 65.4% (26) | 0.0% (11) |
| B_rank3_8|multi | 2 | 0 | 2 | 0.0% | 0.0% (2) | n/a (0) |
| C_absent_top8|single | 24 | 3 | 21 | 12.5% | 27.3% (11) | 0.0% (13) |
| C_absent_top8|multi | 8 | 3 | 5 | 37.5% | 60.0% (5) | 0.0% (3) |

### Complete rationale availability

| Group | Claims | Correct | Errors | Accuracy | SUPPORTED acc (n) | CONTRADICTED acc (n) |
|---|---|---|---|---|---|---|
| complete_in_top2 | 101 | 60 | 41 | 59.4% | 79.1% (67) | 20.6% (34) |
| partial_or_none_in_top2_but_complete_in_top8 | 40 | 18 | 22 | 45.0% | 62.1% (29) | 0.0% (11) |
| incomplete_in_top8 | 35 | 7 | 28 | 20.0% | 36.8% (19) | 0.0% (16) |

- Error bucket counts: `{'B_rank3_8': 22, 'A_top2': 43, 'C_absent_top8': 26}`
- Error transitions (gold->pred|bucket): `{'CONTRADICTED->INSUFFICIENT_EVIDENCE|A_top2': 19, 'CONTRADICTED->INSUFFICIENT_EVIDENCE|B_rank3_8': 10, 'CONTRADICTED->INSUFFICIENT_EVIDENCE|C_absent_top8': 13, 'CONTRADICTED->SUPPORTED|A_top2': 8, 'CONTRADICTED->SUPPORTED|B_rank3_8': 1, 'CONTRADICTED->SUPPORTED|C_absent_top8': 3, 'SUPPORTED->INSUFFICIENT_EVIDENCE|A_top2': 16, 'SUPPORTED->INSUFFICIENT_EVIDENCE|B_rank3_8': 11, 'SUPPORTED->INSUFFICIENT_EVIDENCE|C_absent_top8': 10}`
- INSUFFICIENT_EVIDENCE error causes: `{'guard_fired_no_supports': 6, 'all_neutral': 65, 'guard_overrode_supports': 8}`
- INSUFFICIENT_EVIDENCE causes over all valid labeled outputs: `{'guard_fired_no_supports': 6, 'all_neutral': 65, 'guard_overrode_supports': 8}`
- Judge stance counts (valid labeled): `{'NEUTRAL': 219, 'SUPPORTS': 136, 'CONTRADICTS': 7}`
- Stance on judged gold-rationale chunks (gold->pred|stance): `{'CONTRADICTED->CONTRADICTED|CONTRADICTS': 7, 'CONTRADICTED->CONTRADICTED|NEUTRAL': 1, 'CONTRADICTED->INSUFFICIENT_EVIDENCE|NEUTRAL': 19, 'CONTRADICTED->INSUFFICIENT_EVIDENCE|SUPPORTS': 1, 'CONTRADICTED->SUPPORTED|NEUTRAL': 3, 'CONTRADICTED->SUPPORTED|SUPPORTS': 7, 'SUPPORTED->INSUFFICIENT_EVIDENCE|NEUTRAL': 11, 'SUPPORTED->INSUFFICIENT_EVIDENCE|SUPPORTS': 7, 'SUPPORTED->SUPPORTED|NEUTRAL': 4, 'SUPPORTED->SUPPORTED|SUPPORTS': 55}`
- Claims where a lexical guard fired: all=14, errors=14, correct=0
- Failed labeled attempts (excluded from accuracy, still visible): 12 `{'InterruptedAttempt': 2, 'SystemExecutionError': 10}` by bucket `{'retrieval_unavailable': 2, 'C_absent_top8': 3, 'A_top2': 5, 'B_rank3_8': 2}`

## Interpretation limits

- Bucket membership is an observational association, not a causal attribution. A gold sentence at rank 1–2 can still be insufficient alone (multi-sentence rationales), and a correct verdict in C_absent_top8 can come from non-annotated but valid evidence.
- 'Guard' causes are deterministic re-derivations of the frozen validator messages.
- C's per-pair stances are mapped to ranks by evaluation order (E1..En); C judged the same top-2 chunks.
- These diagnostics motivate H1–H4 but do not establish them; the controlled experiments do.
