# Phase 11 — Failure Analysis

The source is the persisted Phase 10 full-DEV attempt log and its post-hoc failure records. There are 56 non-success records: 52 FAILED and 4 INTERRUPTED. Failed/interrupted attempts are not predictions and are excluded from valid-prediction accuracy denominators. The exact individual errors remain in the source record; affected claims below are the unique IDs recorded for each category.

| Category | Count | Systems (count) | Affected claim IDs | Type | Scientific model error? | Interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| controller_capability_request_failure | 1 | C:1 | 300 | control-path/capability-dispatch | False | Persisted capability request violated the one-claim-unit constraint; it is a control-path failure. |
| local_model_endpoint_connection_refusal | 29 | A:15, B:14 | 350, 354, 362, 380, 384, 385, 386, 388, 399, 410, 411, 415, 437, 439, 440 | infrastructure/runtime | False | Recorded local endpoint connection refusals (WinError 10061); execution did not complete. Treat as infrastructure availability failures, not model-quality errors. |
| phase5_execution_failure_via_controller | 19 | C:19 | 232, 343, 350, 354, 362, 380, 384, 385, 386, 388, 399, 410, 411, 415, 436, 437, 439, 525, 993 | control-path/execution | False | Persisted Phase 5 TaskExecutionError surfaced through System C execution. This is a failed bounded execution path, not a wrong valid prediction. |
| process_interruption | 4 | C:1, B:2, A:1 | 118, 343, 421, 440 | runtime/experiment-interruption | False | Callback completion is explicitly unknown; no prediction is imputed. |
| verification_invalid_verdict | 3 | B:3 | 232, 525, 993 | model-output/verification-format | False | Experiment B returned an invalid legacy verdict; the attempt is an execution/verification-output failure, not an incorrect classified claim. |

## Summary by system

| System | Failures | Interruptions | Notes |
|---|---:|---:|---|
| A | 15 | 1 | Fifteen local endpoint refusals; interrupted callback outcome unknown. |
| B | 17 | 2 | Fourteen endpoint refusals, three invalid verifier verdicts; two interruptions. |
| C | 20 | 1 | Nineteen Phase 5 execution failures and one capability-request failure; one interruption. |

The 29 endpoint refusals are recorded as local Ollama connection refusals (`URLError` / `WinError 10061`, actively refused). They indicate endpoint availability failures in these persisted attempts, not scientific model errors. C execution failures and B malformed/invalid verification outputs are execution/output failures, not wrong valid labels. Failure stage or inference-began status is marked unknown when not persisted.
