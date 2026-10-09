# Phase 11A — Analysis Audit

## Scope and method

Post-hoc deterministic analysis of persisted JSON/JSONL files. No LLM/Ollama generation, external API/search, Phase 10 resume, or A/B/C execution occurred. No benchmark method or system behavior was changed. No new model-derived metric was generated. All Phase 11 calculations are derived from the identified persisted score/attempt/final-analysis artifacts and are denominator-labeled.

## Authoritative source set

- `phase10_full_dev/full_run_manifest.json`: run `02fb9c7a1268476ca1f9b12d9dbfb25e`, selected 300 DEV claims, 900 planned attempts, status value retained as `RUNNING`.
- `phase10_full_dev/attempt_records.jsonl`: 662 rows; status counts 606 SUCCESS, 52 FAILED, 4 INTERRUPTED.
- `phase10_full_dev/per_example.jsonl`: 660 rows; 220 complete A/B/C score triples.
- `phase10_full_dev/inflight_attempt.json`: one C marker for claim 1029; callback completion unknown.
- `phase10_full_dev/configuration_lock.json`, `dataset_identity.json`, `model_identity.json`.
- `phase10_final_analysis/phase10_final_report.json`, `phase10_final_metrics.json`, `phase10_final_failure_analysis.json`, `phase10_final_paired_comparison.json`, `phase10_final_reproducibility.json`.
- `authoritative_provenance/authoritative_provenance.json` and Phase 10G provenance inventory.

## Reconciliation

- 300 selected claims × 3 systems = 900 planned attempts.
- 662 durable attempt rows; status totals 606 + 52 + 4 = 662.
- 660 per-example rows = 220 complete scored claim triples.
- 220 triples have all three durable attempt rows; 197 are all-success.
- 238 planned pairs lack durable attempt rows; claim 1029/C is represented by the one inflight marker; 237 have neither a durable row nor marker.
- 136 labeled and 84 unlabeled among the scored 220. 80 selected claims have no scored triple and are not assigned label availability here.
- No malformed JSONL or duplicate attempt pair/ID was reported by the Phase 10 final artifact reconciliation.

## Scope distinctions and discrepancies

1. `phase10f_launch_validation/` is a separate earlier 9-attempt launch-validation run (run ID `3777c7b12d5e4cdd94403f222a86cc04`, status `COMPLETED`). Its counts/metrics are not combined with the current full-DEV run. The associated launch gate artifact reports its own `FAILED_GATE_STOPPED` gate status; this is not the final full-run status.
2. The full-DEV manifest says `RUNNING`; the experiment was intentionally closed early. The manifest is not edited. Phase 11 labels its outcome **PARTIAL** based on the persisted row counts and closure instruction.
3. Current Git HEAD `55a6abdb98625453b572503d4aab7390e59c51bd` differs from the `git_commit` recorded in authoritative provenance (`4e1f343a4d2e87c04393e9b5a473446b6e825231`); the Phase 10G baseline is `8e996b8db6c9a157bbb1f58009d34c148bc4782c`. This provenance discrepancy is reported, not resolved by assuming one is erroneous.
4. Earlier Phase 10C/10D results describe a separate 30-claim subset and post-hoc analysis; they are not merged into full-DEV metrics.
5. The same SciFact/test-file hashes appear in identity metadata for integrity validation. The current full-run selected input and per-example records are DEV only; no test claims are scored.

No numerical disagreement was found between the current full-DEV JSONL status reconstruction and its Phase 10 final-analysis counts. Values from the 9-attempt launch validation and the 30-claim Phase 10C artifacts are scope-distinct, not contradictory estimates of this run.

## Integrity and validation

- Current dataset hashes compared to Phase 10 dataset identity: **True**.
- Eight frozen source hashes compared to authoritative provenance: **True**.
- Phase 10 locked source hashes compared to current source: **True**.
- Pre-existing phase10_final_analysis raw attempt/per-example hashes are rechecked after Phase 11 creation.
- JSON/CSV parse and count assertions are performed by the deterministic builder; no test suite was run because no project analysis code was added.
- Existing untracked Phase 10 result directories remain outside this Phase 11 output. No commits are created.

## Scientific guardrails

Failure, interruption, invalid prediction, unlabeled gold, unavailable output, wrong valid prediction, retrieval overlap, evidence entailment, and final factual correctness remain separate concepts. The package does not claim full-run completion, test-set performance, significance, or B/C superiority.

## Post-write validation results

| Check | Result |
| --- | --- |
| attempt_rows | PASS |
| score_rows | PASS |
| status_totals | PASS |
| failure_categories_sum | PASS |
| paired_denominator | PASS |
| paired_agreement | PASS |
| full_success_triples | PASS |
| dataset_hashes | PASS |
| frozen_hashes | PASS |
| locked_source_hashes | PASS |
| phase10_raw_jsonl_unchanged | PASS |
| c_trace_stages | PASS |

Raw Phase 10 attempt/per-example JSONL hashes match their pre-analysis hashes. Dataset, frozen source, and locked evaluation source hashes match their recorded identities. Phase 10 raw artifacts were not modified. No automated test suite was run; these are deterministic data-integrity and arithmetic assertions in the analysis builder.
