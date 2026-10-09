# Phase 11 chart data

All rows derive from the persisted Phase 10 full-DEV attempt/per-example JSONL and validated Phase 10 final-analysis metrics. No plot is provided because this output prioritizes data provenance and avoids encoding N/A as zero.

- `classification_and_retrieval.csv`: accuracy and macro F1 apply only to verifier systems B/C; A remains `N/A`. Recall@1 is micro gold-document overlap with denominator in its own column.
- `failure_category_counts.csv`: primary recorded failure categories, including interruption records.
- `latency_by_status.csv`: mean/median only when a persisted timing denominator exists; interruptions remain `NOT AVAILABLE`.
