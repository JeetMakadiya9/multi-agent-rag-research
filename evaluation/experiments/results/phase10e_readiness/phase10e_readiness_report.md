# Phase 10E — Full DEV Evaluation Readiness

## 1. Scope

Offline readiness audit only. No local model call, network/API call, inference, full run, dataset change, or production change occurred. **Readiness: NOT_READY.** The execution design is resumable for intact checkpoints, but retrieval/evidence denominators and crash-recovery safeguards need work before a full run.

## 2. Current Phase History

- **10A:** Initial framework; acceptance failed because the required real A/B/C smoke execution did not complete. Preserve its historical artifacts.
- **10B:** Accepted 3-claim A/B/C smoke; all systems completed 3/3 and System C used the real Phase 8C path.
- **10C Attempt #1:** 30-claim DEV run interrupted by laptop shutdown.
- **10C Attempt #2:** Distinct redesigned 30-claim subset; 90 system/claim attempts, 81 successful and 9 failed. It is not full DEV.
- **10D:** Deterministic post-hoc analysis of Attempt #2; no inference rerun.

The earlier checkpoint found the expected frozen metric bytes at `evaluation/scifact/metrics.py` under the exact expected hash. This audit reverified that actual path; no rename or copy was made.

## 3. Dataset Integrity

The current DEV and corpus SHA-256 values match the Attempt #2 manifest. The current adapter loaded DEV only. Applying the current frozen RAG chunker to the current corpus produced 45,972 chunks.

| Property | Verified current value |
|---|---:|
| DEV claims | 300 |
| Corpus documents | 5,183 |
| Retrieval chunks | 45,972 |
| SUPPORTED | 124 |
| CONTRADICTED | 64 |
| Unlabeled | 112 |
| Claims with nonempty rationale evidence | 188 |
| Gold claim-document evidence pairs | 209 |
| Rationale records | 338 |

Claims SHA-256: `86f0435d08fdb65d1aa41d1472684f57e6e71930626497bdf4d7a9ec1a632217`  
Corpus SHA-256: `b8d6c89624cb2ed74dee8938effc4f5d8bd2086887880af8110d64be4ceade62`

The recorded TEMP embedding cache exists; its SHA-256 matches the manifest and its shape is `[45972, 384]`. Both recorded local embedding and reranker snapshots exist. Provider setup reuses the embedding array but rebuilds FAISS and BM25 indexes in memory. No model download was attempted.

## 4. 10C Artifact Reconciliation

| Check | Reconstructed value |
|---|---:|
| Selected claims | 30 |
| Attempts | 90 |
| Successful attempts | 81 |
| Failed attempts | 9 |
| Successful A/B/C triples | 26 |
| Claims with all three pairs attempted | 30 |
| Per-example records | 90 |
| Selected labeled | 19 (13 SUPPORTED, 6 CONTRADICTED) |
| Selected unlabeled | 11 |

| System | Attempted | Success | Failed | Labeled valid predictions | Invalid predictions | Missing predictions |
|---|---:|---:|---:|---:|---:|---:|
| A | 30 | 27 | 3 | N/A | N/A | N/A |
| B | 30 | 27 | 3 | 17 | 0 | 3 |
| C | 30 | 27 | 3 | 16 | 0 | 3 |

Attempt log, per-example file, summary, and manifest agree on 30 attempted/27 successful/3 failed per system. There are no duplicate claim/system keys. A has no verifier classification. Among the 19 labeled claims, B has 17 valid predictions and C has 16. There were no malformed non-null predictions. Failed and missing predictions were not counted as correct.

A material metric-availability issue must be resolved before full DEV: Attempt #2’s C retrieval aggregate reports 14/19, treating three failed C rows without ranked chunks as misses. Ranked results are actually persisted for 16/19 eligible selected claims; 14 of those 16 contain a rank-1 gold-document hit. The existing C aggregate therefore conflates unavailable retrieval output with a retrieval miss. Evidence-selection rows also need an explicit availability status so failed execution is not treated as an explicit empty decision.

## 5. Failure Readiness

A and B failures cluster on claims 230, 249, and 268. All three A failure records contain `URLError` / Windows error 10061 (connection actively refused). The three matching B records are incomplete Experiment B assessments with endpoint-refused semantic-judge calls. C failed on claims 230, 249, and 859 with a Phase 8C/Phase 5 `TaskExecutionError`; the saved top-level traces do not establish its lower-level cause. Two C failures share claim IDs with A/B failures; this overlap does not prove common cause.

The records show repeated local endpoint refusal during Attempt #2, but do not establish current endpoint health or whether future failures would persist. No endpoint was contacted in this audit. Failure counts are A 3, B 3, C 3; failures remain failures and the runner has no automatic retry.

## 6. Resume Capability

**PARTIAL.** After a callback returns or is caught by `_safe_call`, the runner appends the gold-free attempt record, flushes it, and calls `os.fsync`. On resume, prior claim/system pairs are reconstructed and skipped, including failures; the runner does not retry them. It checks selection, config, and DEV/corpus hashes against the saved manifest. This supports continuation after a normal stop or machine shutdown when the manifest and JSONL records are intact.

It does not guarantee recovery from every power-loss point. A shutdown during an active callback can leave no attempt record, so the callback may run again even if local model work had already happened. JSONL parsing is strict and has no truncated-final-line recovery. The per-example log and manifest are not fsynced/atomically replaced. Resume does not compare current frozen/evaluation source hashes with the saved manifest, and no concurrent-run lock exists. Existing tests verify skipping persisted pairs and rejecting changed config/selection; they do not exercise torn writes or hard shutdown.

## 7. Runtime Projection

Only Attempt #2 callback timing was used. The following are successful attempts (`n=27` per system); p95 uses nearest-rank.

| System | Successful latency total (s) | Mean (s) | Median (s) | Min (s) | Max (s) | p95 (s) |
|---|---:|---:|---:|---:|---:|---:|
| A | 865.93 | 32.07 | 33.52 | 22.60 | 41.58 | 39.50 |
| B | 1,149.45 | 42.57 | 43.32 | 29.36 | 64.82 | 53.70 |
| C | 1,101.34 | 40.79 | 38.35 | 26.35 | 76.92 | 59.95 |

Successful callbacks sum to 3,116.72 seconds across 81 successful attempts. Scaling each system’s observed successful mean to 300 successful attempts gives a **PLANNING ESTIMATE** of 34,630.20 callback seconds (9.62 hours). Scaling all 90 observed attempt latencies, including failures, gives 32,797.45 seconds (9.11 hours). These are projections, not measured full-DEV runtimes; they exclude startup/index load, downtime, resume overhead, and future failures.

Observed Attempt #2 calls: A 30 LLM / 30 retrieval / 0 verification; B 66 / 30 / 30; C 54 / 54 / 27 and 54 controller cycles across 27 successful C attempts. C’s three failed top-level call counter sets are missing, not zero. Using observed rates, conditional projections for 300 successful attempts are A 300/300/0; B 660/300/300; C 600/600/300 plus 600 controller cycles. The C projection uses successful rows only. These are planning quantities, not guaranteed totals. The source counts and missing-counter denominators are in `runtime_projection.json`.

## 8. Local Model Reliability

Six failed A/B attempt records across the same three claims contain local endpoint-refused errors. B records two failed judge subcalls per incomplete assessment. C has three separate controlled-execution failures; the exact low-level cause is not recorded. This indicates an endpoint-refusal cluster in Attempt #2, not current service status. Require a separate local endpoint/model identity preflight before the full run. No Ollama process was launched and no model call was made here.

## 9. Gold Leakage Audit

The full runner hardcodes `split="dev"`; it builds callback input with `runtime_claim_payload`, which returns only `claim_id` and `claim`. Gold annotations are joined after A/B/C callbacks return. Tests inspect the callback fields, system parity, and join order; source inspection confirms them. The selection helper reads annotation strata, but at `count=300` it includes every DEV claim and returns original file order. No gold-based exclusion occurs. **Gold leakage audit: PASS for runtime execution.** The test split is not loaded.

## 10. System C Path Audit

The runner binds C directly to `run_system_c`. Current code performs a provenance preflight retrieval, then uses the actual autonomous controller, validated 8A decisions, bounded 8B execution, 7B capability dispatch, and Phase 6 local retrieval/evidence verification. It invokes explicit Phase 7C continuation authorization before the verification action and calls the controller again for completion. This is not an adapter simulation. **System C real path: PASS.**

## 11. Input Parity

A/B/C receive the same runtime claim ID/text, shared corpus-backed provider, and experiment config. Model tag and temperature are `qwen3:4b` and `0.0`; recorded setting is CPU (`num_gpu=0`), thinking disabled.

Intentional system differences: A produces free text and has no verifier label (generator `num_predict=160`); B verifies retrieved evidence with a two-item budget and JSON output; C uses the bounded controller, a provenance preflight retrieval, and explicit continuation. These distinctions should be recorded in the final manifest. **Input parity: PASS.**

## 12. Configuration Lock

`configuration_lock.json` captures the proposed 300-claim order, data hashes/counts, model/retrieval settings, local cache identity, source hashes, and all eight frozen hashes. The current embedding cache and local embedding/reranker snapshots are verified. **Configuration lock: PARTIAL:** Attempt #2 has no exact Ollama model digest, and current resume code does not enforce saved source/frozen hash equality. Record the digest locally and use an immutable pre-run/resume hash gate before starting.

## 13. Metric Denominators

The proposed rules are captured in `denominator_specification.json`.

- **Classification:** B/C primary accuracy denominator is all 188 labeled claims. Report valid-prediction accuracy separately and distinguish invalid, missing, and failed outputs. A is NOT APPLICABLE. The 112 unlabeled claims have no correctness score.
- **Retrieval:** 188 claims with evidence and 209 gold claim-document references are eligible. Score claims with persisted ranked results only. If retrieval ranks were persisted before later generation/verification failed, retain them. A failure without a ranked output is unavailable, not a retrieval miss.
- **Evidence selection:** B/C only. Score only when a selection result was persisted. An explicit empty selection is scoreable; a failed execution without a selection record is unavailable, not empty. A is NOT APPLICABLE. Document overlap is not entailment.
- **Efficiency:** Use nonmissing recorded latencies/counters, split success/failure, and report n and missing values. Never impute C failed counters as zero.
- **Execution:** Final completion requires 900 distinct persisted claim/system attempt rows. An interrupted run remains PARTIAL and lists pending pairs. Persisted failures are not retried or counted as successes.

The current aggregation does not preserve the retrieval/evidence availability distinction for failed C outputs. An evaluation-layer change is required to honor these denominators.

## 14. Safe Full-DEV Execution Strategy

Run claim-by-claim in DEV file order, invoking A/B/C per claim. Attempt records are appended and fsynced after each callback returns. Use one new output root for the entire run so its manifest and resume log cover all 900 pairs; do not point it at Attempt #2.

**Proposed initial command after all preconditions are satisfied. Do not execute now:**

```powershell
.\.venv\Scripts\python.exe -m evaluation.experiments.run_phase10_dev --count 300 --evidence-limit 2 --output-root evaluation\experiments\results\phase10e_full_dev_run
```

For a restart, repeat the exact command and add `--resume`. Run a single process per output root. Before initial launch and each resume, check the manifest/config, dataset hashes, all eight frozen hashes, evaluation source hashes, and model digest. Keep the machine on stable power and prevent sleep during the projected callback window. Fix truncated-log recovery before relying on hard-shutdown recovery.

## 15. Risks

1. Attempt #2 recorded repeated endpoint refusals across A/B and overlapping C controlled-execution failures; current endpoint health is unknown.
2. C failures can lose ranked retrieval output, and current aggregate counts that as a miss.
3. Failed executions can look like empty evidence selections without explicit availability fields.
4. Strict JSONL parsing has no truncated-tail recovery; score/manifest writes are not fsynced or atomic.
5. Resume does not enforce frozen/evaluation source hash equality and there is no concurrent-run lock.
6. Exact model digest is not recorded.
7. Runtime projections use 27 successful timing rows/system and do not account for full-run failure rate or downtime.

## 16. Required Preconditions Before Full Run

1. Add and test evaluation-layer fields that record ranked retrieval and evidence-selection availability separately from system completion; preserve ranks when a downstream generation or verification stage fails.
2. Add or approve tested recovery for torn JSONL/manifest writes, validate duplicate pair keys, and enforce single-run locking. Keep each durable failure distinct and never silently retry it.
3. Capture the exact local `qwen3:4b` digest and confirm endpoint readiness in a separate non-benchmark preflight.
4. Use an immutable run lock and check all frozen and evaluation source hashes before launch and each resume.
5. Use a new output root and confirm adequate disk space for per-attempt/per-example records.

No source code was created or modified, so no new code tests were added or run. Existing Phase 10C tests were inspected; they cover parity, leakage, persistence, and pair-skip resume behavior, not hard shutdown or torn-file recovery.

## 17. What This Phase Does NOT Establish

This audit did not execute or measure a full DEV run or check current endpoint health. Phase 10C Attempt #2 remains a 30-claim development subset and is not generalized to all DEV. This report does not rank A/B/C, claim superiority, or confirm/reject the hypothesis. The 300-claim evaluation was not started.

**PHASE 10E STATUS: NOT_READY.**

All eight requested frozen hashes match at their current expected paths, including `evaluation/scifact/metrics.py`. No production files were modified.

