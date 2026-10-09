# Phase 10 Full DEV Evaluation — Final Partial-Run Analysis

## Status and scope

**STATUS: PARTIAL — interrupted full-DEV execution.** This is a deterministic analysis of persisted records only. No inference, Ollama request, external API/search, run resume, test-set inference, or benchmark rerun was performed for this analysis. The manifest remains `RUNNING`; the planned full DEV run is not complete. Results describe the persisted execution and must not be generalized to all DEV claims or the test set.

## Source artifacts

| Artifact | SHA-256 |
|---|---|
| `evaluation\experiments\results\phase10_full_dev\full_run_manifest.json` | `F5C56FA90AD06A5D86C21A85BD7F303E4C84D8FD804EE7586240525B30ACE5CA` |
| `evaluation\experiments\results\phase10_full_dev\configuration_lock.json` | `A92D47A74B79D4420D4B21411B0BF009862D38FFAF2732A16A76CE2B76163F43` |
| `evaluation\experiments\results\phase10_full_dev\model_identity.json` | `F9C879D72398E8F98203CEE339B8F699F5982F4176F0356362235F7EB6295C12` |
| `evaluation\experiments\results\phase10_full_dev\dataset_identity.json` | `DF5B91D70DD61993CBB169208240271227FE5F0217C073C25EB465BAC7F23DDC` |
| `evaluation\experiments\results\phase10_full_dev\attempt_records.jsonl` | `CBCD6B3955322033E842062F85EA822ED2338EC54AFC4A0512C0F72B9FF81122` |
| `evaluation\experiments\results\phase10_full_dev\per_example.jsonl` | `19A0B23BC52EAF6C80EDC69EB9F6B6661290DF03CBEECE9833598660DF87E88A` |
| `evaluation\experiments\results\phase10_full_dev\inflight_attempt.json` | `48C61E5B1D9A37B61FABC5912F76148F5D72BED4156CF962F8A4204AD0D9B61D` |

## Phase 10 run reconstruction

- Selected claims: 300 of 300 DEV claims.
- Planned A/B/C attempts: 900; durable attempt rows: 662.
- Durable success / failure / interruption: 606 / 52 / 4.
- Scored per-example rows: 660 across 220 claim triples.
- Complete durable A/B/C triples: 220; all-success triples: 197.
- Planned pairs without durable attempt rows: 238. One of those pairs (claim 1029, C) has an inflight marker with unknown callback completion; the other 237 have neither a durable attempt row nor an inflight marker.
- Among the 220 claims with scored per-example records: 136 labeled and 84 unlabeled. The other 80 selected claims have no scored per-example record, so their label status is not inferred here. Labeled claims in classification evaluator: B 136, C 136.

## Artifact reconciliation

Attempt rows and per-example rows differ as expected: 662 durable attempt records versus 660 scored output rows. Claim 1029 has A/B durable attempts and an unresolved C inflight marker, but no scored three-system record. No malformed JSONL lines or duplicate attempt pair/ID detected. Full claim-by-claim discrepancy details are in `phase10_final_failure_analysis.json`.

## Failure summary

Failures by system (FAILED only): A 15, B 17, C 20. There are also 4 persisted interruption records.

| Primary recorded category | Count |
|---|---:|
| controller_capability_request_failure | 1 |
| local_model_endpoint_connection_refusal | 29 |
| phase5_execution_failure_via_controller | 19 |
| process_interruption | 4 |
| verification_invalid_verdict | 3 |

Exact error strings and per-attempt details are in `phase10_final_failure_analysis.json`. Failure stage and whether inference began are not consistently persisted; those fields are marked unknown rather than inferred.

## System results and denominators

- **A classification:** NOT APPLICABLE; no verifier-derived label.
- **A retrieval:** Recall@1 106/152 = 0.6974; MRR 0.8379 over 136 valid retrieval outputs, with valid misses scored 0. 1 unavailable/failed retrieval outputs excluded.
- **B classification:** accuracy 58/128 = 0.4531; valid predictions 128; labeled claims 136; unlabeled 84; failed labeled outputs excluded 8; invalid predictions excluded 0; macro F1 0.4650545149566676.
- **B retrieval:** Recall@1 106/151 = 0.7020; MRR 0.8404 over 135 valid retrieval outputs, with valid misses scored 0. 2 unavailable/failed retrieval outputs excluded.
  Persisted selected-evidence overlap means: precision 0.9054054054054054 (n=74), recall 0.511328125 (n=128), F1 0.9642156862745098 (n=68). These are document overlap scores, not entailment.
- **C classification:** accuracy 56/126 = 0.4444; valid predictions 126; labeled claims 136; unlabeled 84; failed labeled outputs excluded 10; invalid predictions excluded 0; macro F1 0.4611957459039954.
- **C retrieval:** Recall@1 105/151 = 0.6954; MRR 0.8367 over 135 valid retrieval outputs, with valid misses scored 0. 1 unavailable/failed retrieval outputs excluded.
  Persisted selected-evidence overlap means: precision 0.8888888888888888 (n=72), recall 0.4943121693121693 (n=126), F1 0.9404040404040404 (n=66). These are document overlap scores, not entailment.

## B/C paired comparison

Among 126 claims with both successful valid predictions and gold labels, B/C predictions agreed on 124 and differed on 2. Correctness: both correct 55, both wrong 69, B correct/C wrong 1, C correct/B wrong 1. Paired accuracy difference C-B is 0.0000. The persisted bootstrap interval is descriptive uncertainty only; no significance or superiority claim is made.

## System C traces

Persisted successful C trace audit: 199 records. Stage-by-stage audit details are in phase10_final_report.json. Failed attempts are outside this successful-trace audit; missing internal reasoning was not reconstructed.

## Efficiency

- **A:** 221 durable attempt records; latency mean/median/min/max = 26.0183176249999 / 26.84168100000079 / 5.601945499991416 / 37.542181299999356 sec (n=220); totals LLM/retrieval/verification calls = 220 / 220 / 0.
- **B:** 221 durable attempt records; latency mean/median/min/max = 39.576746099086826 / 38.38511010000002 / 11.760461000012583 / 110.98191030000089 sec (n=219); totals LLM/retrieval/verification calls = 484 / 219 / 219.
- **C:** 220 durable attempt records; latency mean/median/min/max = 38.799780178538725 / 38.18492049998895 / 11.385477399991942 / 89.1970246000019 sec (n=219); totals LLM/retrieval/verification calls = 448 / 438 / 219.

## Reproducibility and integrity

Configuration-lock fingerprint: `875ee8247def6be3f440e34863168e1313f1a4cea33d9272f517a58c96000185`; model identity and digest are reported from the persisted `model_identity.json`, not newly health-checked. Dataset integrity: {"claims_dev.jsonl": 300, "claims_test.jsonl": 300, "claims_train.jsonl": 809, "corpus.jsonl": 5183} claims/documents and 45972 chunks. Frozen hashes match expected: **True**. Persisted runtime inputs passed the gold-leakage schema scan: **True**. No DEV/test label flowed into runtime inputs according to those persisted records. Original result JSONL files are hash-checked after report writing.

## Limitations and interpretation

- This is partial full-DEV execution, not a completed 300-claim evaluation. Do not generalize beyond persisted records.
- Unavailable/failed outputs are not retrieval misses. Retrieval overlap, evidence overlap, entailment, verification, and final answer correctness are distinct.
- Unlabeled claims are not incorrect; A has no verifier label.
- Execution failures, malformed/invalid output, and wrong valid predictions remain distinct.
- Reported numbers are reconstructed from the current artifacts and must not be compared as a completed full-DEV result.
- No winner, rank, hypothesis conclusion, or superiority claim is made.
- Recommended next step: if a future full-DEV study is authorized, use a fresh documented run identity after verifying the runner can safely resume or restart, preserving this interrupted run unchanged.
