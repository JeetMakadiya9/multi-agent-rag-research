# Phase 10F — Operational Launch Gate Report

## Status: FAILED GATE — FULL DEV RUN STOPPED

The predetermined launch validation executed the first three SciFact DEV claims in canonical file order: `[1, 3, 5]`. All 9 A/B/C attempts completed and were persisted. No full DEV attempts were started.

## Verified launch checks

- Dataset identity: SciFact DEV 300 claims; corpus 5,183 documents; 45,972 retrieval chunks.
- All three systems received identical gold-free runtime inputs per claim (`claim_id`, `claim`).
- All 9 attempt and scoring records parse; attempt IDs and claim/system pairs are unique. Each attempt contains the same configuration fingerprint and `qwen3:4b` digest `359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7`.
- No execution failures occurred in these 9 launch attempts.
- Persisted System C data shows the actual 8A decision/validation, 8B execution, 7B dispatch, Phase 6 retrieval/verification, and explicit continuation authorization: the second task execution has a linked `continuation_authorization_id`, and the loop records a researcher-authorized transition to EVIDENCE_REVIEW. The existing trace-audit helper incorrectly searched for the literal string `CONTINUE_SAME_TASK` in controller trace JSON, so its stored result is a false-negative/limited audit.
- Frozen hashes match; 43 historical Phase 10C/10D/10E artifact files retain their pre-run hashes.

## Blocking gate

The `resume-launch` check failed before entering the attempt loop. Exact error: `ValueError: Duplicate claim/system attempt records prevent safe resume`. The Phase 10F runner passes attempt rows containing `claim_id` to `resume_pending_pairs`, while that helper reads `example_id`; each system is consequently seen under a duplicate `(None, system)` key. The resume invocation made no inference calls and did not append attempts. Resume idempotence is therefore **not demonstrated**.

Per the launch stop condition, the full 300-claim run was not started. Its planned 900 attempts remain unexecuted. The launch subset is operational validation only and is not a benchmark result.

## Tests

All executed tests passed: Phase 10F 14; Phase 10E 49; Phase 10C 32; Phase 10D 20; Phase 9 43; Phase 8C integration 27; full unittest regression 754. The project and bundled Python do not have pytest installed; suites were run with `unittest`.

Historical limitation: `evaluation/scifact/scifact_metrics.py` is absent. Current `evaluation/scifact/metrics.py` bytes match the expected frozen hash; provenance of the historical filename is not verified.

Machine-readable detail: `launch_gate_validation.json`.
