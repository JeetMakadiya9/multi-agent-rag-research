# Phase 10F Remediation Root Cause

## Defect 1: Phase 10F resume identity mismatch

- **Observed behavior:** Phase 10F attempt rows persist `claim_id`; the resume call raised a duplicate `(None, system)` error before scheduling work.
- **Root cause:** `run_phase10f.run_phase10f` builds expected keys as `(claim_id, system)` but delegates to the generic Phase 10E `resume_pending_pairs`, which intentionally reads `example_id` for legacy Phase 10C attempt schemas. Phase 10F therefore produced missing identifiers in the helper.
- **Affected functions:** `evaluation/experiments/run_phase10f.py::run_phase10f`; `evaluation/experiments/phase10e_infrastructure.py::resume_pending_pairs` (legacy `example_id` contract). Phase10C scorer records also retain `example_id` because `validate_scored_record` / `claim_system_key` use that generic score schema; attempt identity remains canonical `claim_id`.
- **Minimal fix:** Add a Phase10F-local resume planner keyed only by `(claim_id, system)`; explicitly validate identifiers, run identity, statuses, duplicates, and expected pairs. Keep the Phase10E helper and Phase10C schema unchanged. Permit only the known runner-file source hash difference on a resume after field-by-field comparison of all other configuration fields; retain the original lock/fingerprint for the persisted run. This exception is necessary to resume historical attempts using the corrected evaluation orchestration and does not relax dataset, model, system, prompt, retrieval, or frozen-source identity.

## Defect 2: System C trace audit false negative

- **Observed behavior:** The audit marked C traces LIMITED because it searched for the literal `CONTINUE_SAME_TASK`.
- **Root cause:** The persisted Phase8C schema represents continuation as `research_state.task_executions[].continuation_authorization_id` linked to a record in `research_state.research_loops[].continuation_authorizations[]`, with `previous_execution_id` linking back to the retrieval execution. Researcher authorization is stored in loop transitions (`authority=researcher`, `RESEARCH` -> `EVIDENCE_REVIEW`). It is not represented by that literal string.
- **Affected function:** `evaluation/experiments/run_phase10f.py::audit_c_traces`.
- **Minimal fix:** Validate actual structured fields and matching execution/authorization IDs, accepted 8A validation, completed 8B execution, expected 7B capability dispatch, Phase6 retrieval/verification actions, researcher-authorized transition, and explicit output completion. Return granular PASS / FAIL / NOT_APPLICABLE stage results and preserve evidence identifiers.

## Regression coverage

Add deterministic tests for resume identity/status/run compatibility/idempotence and structured C trace presence, absence, mismatches, optional stages, and malformed traces. Re-run the corrected resume against an isolated copy of the three-claim launch artifacts twice; never write to the original launch directory and never invoke inference.
