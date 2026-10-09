# Phase 11 — Thesis-Ready Tables

**Scope note:** Phase 10 full-DEV run is PARTIAL, not a completed benchmark. Values are descriptive persisted results only.

## Table 1 — Dataset and evaluation configuration

| Item | Persisted configuration |
| --- | --- |
| Dataset | SciFact-Orig |
| Split | DEV |
| DEV claims selected | 300 / 300 in canonical file order |
| Corpus documents | 5,183 |
| Retrieval chunks | 45,972 |
| Systems | A: Strong RAG; B: RAG + frozen verifier; C: bounded controller path |
| Model | Ollama qwen3:4b; digest recorded in reproducibility artifact |
| Selection | all-dev-file-order-v1; seed 0 |

## Table 2 — Overall execution coverage

| Coverage | Count |
| --- | --- |
| Planned system/claim attempts | 900 |
| Durable attempt rows | 662 |
| Successful | 606 |
| Failed | 52 |
| Interrupted | 4 |
| Scored rows | 660 |
| Complete A/B/C claim triples | 220 |
| All-success A/B/C triples | 197 |
| Missing durable attempt rows | 238 |
| Missing rows with inflight marker | 1 |
| Missing rows with no attempt or inflight marker | 237 |
| Scored claims labeled / unlabeled | 136 / 84 |
| Selected claims without scored records | 80 |

## Table 3 — Classification performance

| System | Correct / valid denominator | Accuracy | Macro F1 | Notes |
| --- | --- | --- | --- | --- |
| A | N/A | N/A | N/A | No verifier label |
| B | 58/128 | 0.453 | 0.465 | Failed/missing/invalid outputs excluded |
| C | 56/126 | 0.444 | 0.461 | Failed/missing/invalid outputs excluded |

## Table 4 — Retrieval performance

| System | Gold evidence docs hit at rank 1 | Gold-doc denominator | Recall@1 | Valid retrieval rows |
| --- | --- | --- | --- | --- |
| A | 106 | 152 | 0.697 | 136 | 0.838 |
| B | 106 | 151 | 0.702 | 135 | 0.840 |
| C | 105 | 151 | 0.695 | 135 | 0.837 |

## Table 5 — B vs C paired comparison

| Paired outcome | Count |
| --- | --- |
| Valid paired predictions | 126 |
| Same prediction | 124 |
| Different prediction | 2 |
| Both correct | 55 |
| Both incorrect | 69 |
| B-only correct | 1 |
| C-only correct | 1 |
| Descriptive C−B accuracy difference | 0.000 |

## Table 6 — Failure categories

| Category | Count | Affected systems |
| --- | --- | --- |
| controller_capability_request_failure | 1 | {"C": 1} |
| local_model_endpoint_connection_refusal | 29 | {"A": 15, "B": 14} |
| phase5_execution_failure_via_controller | 19 | {"C": 19} |
| process_interruption | 4 | {"A": 1, "B": 2, "C": 1} |
| verification_invalid_verdict | 3 | {"B": 3} |

## Table 7 — Efficiency

| System | Success mean latency | Failure mean latency | LLM calls | Retrieval calls | Verification calls | Controller cycles |
| --- | --- | --- | --- | --- | --- | --- |
| A | 27.221 (n=205) | 9.580 (n=15) | 220 | 220 | N/A | N/A |
| B | 40.834 (n=202) | 24.640 (n=17) | 484 | 219 | 219 | N/A |
| C | 40.291 (n=199) | 23.964 (n=20) | 448 | 438 | 219 | 438 |

## Table 8 — System C execution-path validation

| Stage | PASS | Denominator |
| --- | --- | --- |
| phase_8a_validation | 199 | 199 |
| phase_8b_execution | 199 | 199 |
| phase_7b_dispatch | 199 | 199 |
| phase_6_retrieval | 199 | 199 |
| phase_6_verification | 199 | 199 |
| continuation_authorization | 199 | 199 |
| researcher_transition | 199 | 199 |
| completion | 199 | 199 |

