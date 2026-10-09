# Phase 11B analysis audit

- Manifest: COMPLETED; run 02fb9c7a1268476ca1f9b12d9dbfb25e; fingerprint 875ee8247def6be3f440e34863168e1313f1a4cea33d9272f517a58c96000185.
- Coverage: 300 claims, 900 attempts, 900 score rows, complete A/B/C matrix without duplicate pairs.
- Attempt and score statuses/predictions reconcile. Runtime inputs contain claim ID and text; gold fields are post-execution score fields.
- Status counts: {'SUCCESS': 840, 'INTERRUPTED': 5, 'FAILED': 55}. The reported 60 non-success aggregate combines 55 FAILED and 5 INTERRUPTED.
- Preservation: Phase 10 raw artifacts True; Phase 11A True; provenance True; frozen hashes True; dataset hashes True.
- Metric definition differences are explicitly logged in JSON. No source values were overwritten.
- All package JSON/CSV parsed; full A/B/C coverage and persisted joins were validated. No inference or external calls were made.

- C failed-cycle count discrepancy: 22 failed attempts show differing top-level and embedded counts; both are retained.
- Recall@10 and Recall@20 are NOT COMPUTED because ranked lists contain no more than eight chunks.

- Deterministic package validation: 13 JSON reports parse and 8 CSV files parse; 900 records, 900 scores, 300 unique claims and full 900-pair matrix.
- Evidence metrics are document overlap; no entailment or factual-correctness claim is made.
