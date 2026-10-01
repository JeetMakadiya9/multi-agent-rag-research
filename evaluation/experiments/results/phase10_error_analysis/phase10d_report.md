# Phase 10D — Failure, Error & Agreement Analysis

Generated from saved Phase 10C artifacts at 2026-09-29T17:29:59+00:00. Analysis version: phase10d-analysis-1.0.

## 1. Scope

Deterministic post-hoc analysis of the 30-claim SciFact DEV subset. No LLM, search, API, or inference calls were made. The Phase 10C artifacts were read only.

The requested evaluation/scifact/scifact_metrics.py file is absent from this checkout, so its protected hash could not be verified. The other seven requested frozen hashes matched before and after analysis.

## 2. Source Artifacts

| Artifact | SHA-256 |
|---|---|
| combined | FB330FC9A20B4C320EEE89D4289B0366A153B92059DB892F84FAB11276C8251F |
| per_example | 00D97C1CE6AE51C6E6BFA0CA8986B0E794A3AC4224CAD94AFA45C6C2B48E4639 |
| runtime_attempts | 5C7917F42077E9A28098D14ADE3E8B778F20212B208C0FBBE734274BD706F5D8 |
| manifest | 67E7425D405987061B9648D976360B4D7FF5FFB650936CDBE6A2025D447F5DD5 |
| summary | 7611FACD656131FE150FBA4BB954D6FF9C0F7252D3813076E4194A08715BA0D1 |

## 3. Phase 10C Reconstruction

Selected claims: 30; attempts: 90; successes: 81; failures: 9; complete A/B/C triples: 26; labeled: 19; unlabeled: 11.

| System | Attempts | Success | Failed | Valid predictions | Invalid | Missing | N/A |
|---|---|---|---|---|---|---|---|
| A | 30 | 27 | 3 | 0 | 0 | 0 | 30 |
| B | 30 | 27 | 3 | 27 | 0 | 0 | 0 |
| C | 30 | 27 | 3 | 27 | 0 | 0 | 0 |

Persisted retrieval records: 90; marked scored: 57; retrieved chunk lists: 90; selected-evidence fields: 90; latency records: 90.

Runtime attempts and per-example rows contain the same claim/system pairs. The discrepancy log lists the missing protected metrics source and C failure-counter observability gap. Phase 10C summary C counters describe successful returned outputs; failed serialized traces separately show failed controller cycles.

## 4. Artifact Reconciliation

[{"item": "C_retrieval_denominator_scope", "phase10c_saved_metric": {"hits": 14, "gold_document_denominator": 19, "recall": 0.7368421052631579}, "ranked_document_ids_unavailable_on_failed_claims": [230, 249, 859], "posthoc_ranked_document_denominator": 16, "severity": "representational_scope_difference", "explanation": "Phase 10C scored these failed rows with empty chunk lists as misses; Phase 10D does not call them retrieval misses because ranked document IDs were not persisted."}, {"item": "C_failed_attempt_call_counter_coverage", "phase10c_reported_controller_cycles": 54, "failed_controller_cycles_visible_in_error_traces": 3, "failed_top_level_call_counters_missing": 3, "severity": "observability_scope_difference", "explanation": "The saved C efficiency counters sum returned outputs; failed attempts have null component counters, although nested traces record failed dispatch cycles."}, {"item": "protected_file_missing", "path": "evaluation/scifact/scifact_metrics.py", "expected_sha256": "8BE4B90BB28090FC48F874AF399798B0F7E39C25AAD7A1FA17B864A002E55587", "severity": "unverifiable"}]

## 5. Failure Summary

| Claim | System | Category | Stage | Timestamp |
|---|---|---|---|---|
| 230 | A | model_endpoint_failure | system_callback | NOT RECORDED |
| 230 | B | model_endpoint_failure | experiment_b_verification | NOT RECORDED |
| 230 | C | controller_execution_failure | system_callback | 2026-09-29T15:38:32.392344+00:00, 2026-09-29T15:38:32.393345+00:00, 2026-09-29T15:38:36.523932+00:00 |
| 249 | A | model_endpoint_failure | system_callback | NOT RECORDED |
| 249 | B | model_endpoint_failure | experiment_b_verification | NOT RECORDED |
| 249 | C | controller_execution_failure | system_callback | 2026-09-29T15:39:09.498274+00:00, 2026-09-29T15:39:09.499275+00:00, 2026-09-29T15:39:13.616692+00:00, 2026-09-29T15:39:13.617692+00:00 |
| 268 | A | model_endpoint_failure | system_callback | NOT RECORDED |
| 268 | B | model_endpoint_failure | experiment_b_verification | NOT RECORDED |
| 859 | C | controller_execution_failure | system_callback | 2026-09-29T16:12:36.592264+00:00, 2026-09-29T16:12:36.593270+00:00, 2026-09-29T16:13:19.373921+00:00 |

Failure counts by system: A=3, B=3, C=3.

Failure categories: model_endpoint_failure=6, controller_execution_failure=3. Exact error messages and per-attempt execution details are in failure_records.jsonl.

A/B connection-refused records show that inference did not start. C failures are recorded as Phase 8C execution failures; whether inference started is not recorded. Attempt IDs and top-level attempt timestamps are absent.

## 6. Claim-Level Outcome Matrix

The complete 30-row matrix is in claim_outcome_matrix.jsonl. A has no verifier-derived classification; its prediction is N/A.

| Claim | Gold | A status | B status/pred | C status/pred | A/B/C hit@1 | A/B/C latency (s) |
|---|---|---|---|---|---|---|
| 53 | SUPPORTED | SUCCESS | SUCCESS / SUPPORTED | SUCCESS / SUPPORTED | True/True/True | 35.75616050005192/38.37548260000767/38.84251850005239 |
| 94 | UNLABELED | SUCCESS | SUCCESS / INSUFFICIENT_EVIDENCE | SUCCESS / INSUFFICIENT_EVIDENCE | None/None/None | 33.058148799987976/45.45033479999984/43.15121000004001 |
| 115 | CONTRADICTED | SUCCESS | SUCCESS / INSUFFICIENT_EVIDENCE | SUCCESS / INSUFFICIENT_EVIDENCE | True/True/True | 34.14057160000084/43.843625899986364/38.34959639998851 |
| 141 | SUPPORTED | SUCCESS | SUCCESS / SUPPORTED | SUCCESS / SUPPORTED | True/True/True | 34.965574700036086/43.3174695999478/37.80765470000915 |
| 230 | SUPPORTED | FAILED | FAILED / None | FAILED / None | True/True/None | 9.52284019999206/15.193615500000305/15.813735000003362 |
| 249 | CONTRADICTED | FAILED | FAILED / None | FAILED / None | True/True/None | 7.701961500002653/13.924232599994866/15.291035300004296 |
| 268 | UNLABELED | FAILED | FAILED / None | SUCCESS / SUPPORTED | None/None/None | 10.20459860000119/16.35247860000527/76.9197858999978 |
| 327 | SUPPORTED | SUCCESS | SUCCESS / SUPPORTED | SUCCESS / SUPPORTED | True/True/True | 34.057371000002604/43.4415302999987/36.800063400005456 |
| 384 | UNLABELED | SUCCESS | SUCCESS / SUPPORTED | SUCCESS / SUPPORTED | None/None/None | 34.269974199996796/50.437957600006484/42.94349109999894 |
| 443 | SUPPORTED | SUCCESS | SUCCESS / INSUFFICIENT_EVIDENCE | SUCCESS / INSUFFICIENT_EVIDENCE | True/True/True | 35.78354310001305/50.66743570000108/46.83894049999071 |
| 502 | UNLABELED | SUCCESS | SUCCESS / INSUFFICIENT_EVIDENCE | SUCCESS / INSUFFICIENT_EVIDENCE | None/None/None | 34.324286599992774/51.70298669999465/42.889403799999855 |
| 549 | SUPPORTED | SUCCESS | SUCCESS / SUPPORTED | SUCCESS / SUPPORTED | True/True/True | 37.3041191999946/48.6564728999947/49.15869959999691 |
| 554 | UNLABELED | SUCCESS | SUCCESS / INSUFFICIENT_EVIDENCE | SUCCESS / INSUFFICIENT_EVIDENCE | None/None/None | 37.48897630001011/50.92062239999359/56.65102960000513 |
| 660 | UNLABELED | SUCCESS | SUCCESS / INSUFFICIENT_EVIDENCE | SUCCESS / INSUFFICIENT_EVIDENCE | None/None/None | 33.51865369999723/44.73294509999687/40.504048200004036 |
| 674 | CONTRADICTED | SUCCESS | SUCCESS / CONTRADICTED | SUCCESS / CONTRADICTED | True/True/True | 39.49723369999265/39.9453913999896/38.92742009999347 |
| 723 | SUPPORTED | SUCCESS | SUCCESS / SUPPORTED | SUCCESS / SUPPORTED | True/True/True | 32.81531279999763/50.00707569999213/45.02952539999387 |
| 775 | UNLABELED | SUCCESS | SUCCESS / SUPPORTED | SUCCESS / SUPPORTED | None/None/None | 41.58324469999934/64.8171474999981/59.945245399998385 |
| 823 | SUPPORTED | SUCCESS | SUCCESS / SUPPORTED | SUCCESS / SUPPORTED | True/True/True | 39.430112500005635/47.38431899998977/42.972767100000056 |
| 859 | CONTRADICTED | SUCCESS | SUCCESS / INSUFFICIENT_EVIDENCE | FAILED / None | True/True/None | 38.481881100000464/53.703158699994674/59.022408200005884 |
| 870 | UNLABELED | SUCCESS | SUCCESS / INSUFFICIENT_EVIDENCE | SUCCESS / INSUFFICIENT_EVIDENCE | None/None/None | 30.709230199994636/32.290906299997005/26.776469300006283 |
| 957 | SUPPORTED | SUCCESS | SUCCESS / INSUFFICIENT_EVIDENCE | SUCCESS / INSUFFICIENT_EVIDENCE | False/False/False | 22.5987363000022/31.592859399999725/29.967225500004133 |
| 1024 | SUPPORTED | SUCCESS | SUCCESS / SUPPORTED | SUCCESS / SUPPORTED | True/True/True | 29.390801100002136/39.85796060001303/38.19508119999955 |
| 1088 | CONTRADICTED | SUCCESS | SUCCESS / INSUFFICIENT_EVIDENCE | SUCCESS / INSUFFICIENT_EVIDENCE | False/False/False | 24.84302290000778/39.828264700001455/37.62545940000564 |
| 1163 | SUPPORTED | SUCCESS | SUCCESS / SUPPORTED | SUCCESS / SUPPORTED | True/True/True | 26.728866699995706/29.35680229999707/28.5776165999996 |
| 1175 | UNLABELED | SUCCESS | SUCCESS / INSUFFICIENT_EVIDENCE | SUCCESS / INSUFFICIENT_EVIDENCE | None/None/None | 26.21373930000118/35.93620630000078/38.16589610000665 |
| 1213 | UNLABELED | SUCCESS | SUCCESS / INSUFFICIENT_EVIDENCE | SUCCESS / INSUFFICIENT_EVIDENCE | None/None/None | 27.198236199998064/36.920899299992016/36.74604650000401 |
| 1262 | SUPPORTED | SUCCESS | SUCCESS / SUPPORTED | SUCCESS / SUPPORTED | True/True/True | 25.435774700003094/31.904983100001118/31.4702338000061 |
| 1316 | UNLABELED | SUCCESS | SUCCESS / INSUFFICIENT_EVIDENCE | SUCCESS / INSUFFICIENT_EVIDENCE | None/None/None | 26.920115600005374/33.27166369999759/32.59588449999865 |
| 1320 | CONTRADICTED | SUCCESS | SUCCESS / INSUFFICIENT_EVIDENCE | SUCCESS / INSUFFICIENT_EVIDENCE | True/True/True | 24.582336300009047/38.595862899994245/37.14420270000119 |
| 1352 | SUPPORTED | SUCCESS | SUCCESS / SUPPORTED | SUCCESS / SUPPORTED | True/True/True | 24.831910999986576/32.4859584000078/26.34813950001262 |

## 7. Classification Error Analysis

| System | Correct/labeled | Accuracy | Valid n | Valid-only accuracy | Macro F1 |
|---|---|---|---|---|---|
| B | 11/19 | 0.5789473684210527 | 17 | 0.6470588235294118 | 0.5776397515527951 |
| C | 11/19 | 0.5789473684210527 | 16 | 0.6875 | 0.5776397515527951 |

Confusion tables distinguish SUPPORTED, CONTRADICTED, INSUFFICIENT_EVIDENCE, failed/invalid, and missing outputs. Macro F1 averages the two gold-supported classes only. No successful malformed predictions were persisted.

## 8. B/C Agreement Analysis

Valid pair agreement: 26 same and 0 different. Failed pairs remain separate: {"both_valid_same": 26, "both_failed": 2, "one_failed": 2}.

The disagreement detail file distinguishes genuine valid-prediction disagreement from one system failing while the other produced an output. Retrieval and selected-evidence comparisons are persisted per such claim.

## 9. Retrieval Error Analysis

Top-ranked document overlap with gold rationale document IDs measures retrieval relevance only. It is not evidence entailment, verification, or answer correctness.

| System | Hit numerator | Claim denominator | Recall@1 |
|---|---|---|---|
| A | 17 | 19 | 0.8947368421052632 |
| B | 17 | 19 | 0.8947368421052632 |
| C | 14 | 16 | 0.875 |

Claim lists for all-hit, all-miss, individual misses, A/B hit with C miss, and C hit with A/B miss are in retrieval_error_analysis.json, with retrieved and gold IDs.

The saved Phase 10C aggregate records C Recall@1 as 14/19. The persisted C rows for claims 230, 249, and 859 have no ranked document IDs because their callbacks failed. On the 16 C gold-evidence rows with ranked document IDs persisted, the post-hoc hit rate is 14/16. The saved 14/19 treats the three failed empty lists as misses; these are not identifiable retrieval misses from the C traces.

## 10. Evidence Analysis

B/C selected evidence IDs are compared against gold rationale document IDs. Per-claim precision, recall, F1, evidence counts, same/different selections, precision-1 incomplete recall, and zero recall are in the report JSON. These comparisons do not demonstrate entailment. A has no selected citation. Failed execution makes the selection unavailable.

The persisted Phase 10C citation aggregate uses 19 rows and reports recall 11/19 for B and C. Among successful outputs, the corresponding recall means are B 11/17 and C 11/16. Failed evidence-selection rows remain unavailable in the per-claim analysis.

## 11. System C Execution Analysis

Successful C traces: 27; accepted 8A validations: 54; completed 8B executions: 54; claims with continuation authorization: 27; failed dispatch cycles in error traces: 3.

The trace artifact records local_retrieval and evidence_verification dispatches for successful C runs. Failed C traces show the evidence_verification dispatch failure. Missing internal details are not reconstructed.

## 12. Efficiency Analysis

| System | Status | n | Mean (s) | Median (s) | Min (s) | Max (s) |
|---|---|---|---|---|---|---|
| A | SUCCESS | 27 | 32.07140499259576 | 33.51865369999723 | 22.5987363000022 | 41.58324469999934 |
| A | FAILED | 3 | 9.143133433331968 | 9.52284019999206 | 7.701961500002653 | 10.20459860000119 |
| B | SUCCESS | 27 | 42.572086033329406 | 43.3174695999478 | 29.35680229999707 | 64.8171474999981 |
| B | FAILED | 3 | 15.156775566666814 | 15.193615500000305 | 13.924232599994866 | 16.35247860000527 |
| C | SUCCESS | 27 | 40.790505733337746 | 38.34959639998851 | 26.34813950001262 | 76.9197858999978 |
| C | FAILED | 3 | 30.042392833337846 | 15.813735000003362 | 15.291035300004296 | 59.022408200005884 |

Persisted LLM/retrieval/verification/cycle counter sums, split by success/failure and showing missing fields, are in efficiency_analysis.json. C's three failed top-level counter sets are not imputed.

## 13. Failure Impact on Metrics

The JSON report contains a metric-by-system numerator/denominator table. B/C primary classification accuracy uses 19 labeled claims and counts failures as not correct; valid-only accuracy excludes them and uses different denominators. Unlabeled claims are excluded from correctness. Retrieval uses annotated gold documents; missing retrieved hits produce no overlap. Citation overlap is not entailment.

## 14. Unlabeled Claim Analysis

The 11 unlabeled claims include A 10 successes/1 failure, B 10 successes/1 failure, and C 11 successes. B/C outputs agree on 10 and differ on 1 (claim 268 has a B failure and a C prediction). Classification correctness is N/A because no gold label exists. Retrieval/evidence overlap with gold rationale documents is also N/A because annotations are explicitly empty; saved runtime retrieval IDs, evidence selections, latencies, and prediction validity remain listed per claim in the JSON report. Unlabeled is not incorrect.

## 15. Key Observations

- Six A/B failures are local endpoint connection refusals; three C failures are controller/Phase 5 execution failures.
- Execution failure, invalid prediction, missing prediction, and wrong prediction are distinct outcomes.
- Among paired valid B/C predictions, there are no different labels; one-failed and both-failed cases remain separate.
- Retrieval relevance, selected evidence overlap, verification verdict, and final answer correctness are different measures.

## 16. Limitations

- Results describe only the selected 30-claim DEV subset.
- Attempt IDs and root-level attempt timestamps are not persisted.
- Failed C top-level call counters are missing.
- One requested protected source file, scifact_metrics.py, was absent, preventing verification of that hash.
- No semantic evidence adjudication is possible from this deterministic analysis alone.

## 17. What Phase 10D Does NOT Establish

It does not rank systems, establish superiority, confirm or reject a hypothesis, generalize to all DEV claims or test, or treat retrieval overlap or citation precision as entailment.

## 18. Recommended Next Experimental Step

For any later evaluation, preregister the sample, failure/retry policy, and denominators; preserve per-attempt counters and traces; and report a larger development evaluation descriptively. This analysis does not recommend tuning systems toward a score.

### Additional artifacts

phase10d_report.json, failure_records.jsonl, claim_outcome_matrix.jsonl, agreement_analysis.json, retrieval_error_analysis.json, system_c_analysis.json, efficiency_analysis.json.
