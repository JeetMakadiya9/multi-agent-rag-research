PHASE 10F REMEDIATION STATUS: PARTIAL

# Phase 10F — Launch Gate Remediation

## Scope and result

Fixed the Phase 10F resume key mismatch and the System C trace-audit false negative. This was an infrastructure-only validation. No model inference was run and no full DEV evaluation was started.

- Canonical DEV prefix: claims 1, 3, and 5; 9 planned A/B/C attempt pairs.
- Existing attempts recognized: 9 successful, 0 failed, 0 missing, 0 duplicate.
- Two final-code resume invocations each scheduled 0 pairs, created 0 attempts, and made 0 inference calls. Configuration fingerprint remained `7c8b99e360727f05933f912ba5904a2e150a8311712e07bf832cc157f6c4dc78`.
- Structured System C audit: 3/3 successful traces PASS.
- Final regression: 779 passed, 0 failed, 0 errors.
- Frozen hashes: all 8 match. Phase 10C/10D/10E artifacts: 43/43 file hashes match. Original Phase 10F launch artifacts: 19/19 file hashes match.
- SciFact dataset file hashes and stored counts match; DEV 300, corpus 5,183, retrieval chunks 45,972.

## Defects and corrections

Resume had passed Phase 10F's `claim_id` rows to the generic Phase 10E helper, which reads legacy `example_id`. Phase 10F now plans persisted pairs by `(claim_id, system)` and validates run ID, IDs, status, and duplicates. Failed/interrupted attempts remain terminal and are never marked successful. Only the known prior runner hash may migrate, and only to this pinned remediation source; every other configuration/source field remains strict. The original run fingerprint is retained.

The old C audit searched for `CONTINUE_SAME_TASK`. The corrected audit uses accepted validation states, completed execution/dispatch statuses, expected capability names, execution IDs, continuation authorization IDs, and the researcher-authorized transition.

## Tests

See `test_results.json` for each required focused suite and the full regression. Phase 10E passed standalone (49/49); final full discovery passed (779/779). One earlier custom grouped-test harness invocation aborted on a Windows stale-lock `WinError 87`; it was not a recorded unittest failure, and the standalone Phase 10E suite and full discovery both passed.

## Preservation and limitations

No frozen architecture or historical inference output was modified. The launch validation operated on an isolated copy; the original launch directory is unchanged. The inherited historical path `evaluation/scifact/scifact_metrics.py` remains absent; `evaluation/scifact/metrics.py` matches its expected frozen bytes, but historical filename provenance remains unverified. This is the reason overall status is PARTIAL despite all operational launch gates passing.

No performance or scientific conclusion about A/B/C is made. The full 300-claim evaluation remains unstarted.

## Artifacts

- `resume_validation.json` / `.md`
- `trace_audit.json` / `.md`
- `preservation_check.json`
- `test_results.json`
- `defect_root_cause.md`
- isolated resume files under `resume_validation_run/`
