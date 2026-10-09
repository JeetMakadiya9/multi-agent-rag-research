# Architecture decision: version boundaries for the hybrid migration (M0)

Status: ACCEPTED for M0 (preservation). Scope: version boundaries and rules only; no implementation.

## Versions

* **V0 — Legacy system.** Inherited RAG (`rag.py`, `src/baseline_rag.py`), Experiment B verification
  (`claim_verification.py`, `verification_units.py`, `experiment_b_verification.py`, `research_pipeline.py`),
  the Phase 4–8 research/governance modules (`research_*.py`), Systems A/B/C, and all Phase 10–13 experiment code
  and results under `evaluation/experiments/`. **Frozen** for scientific comparability: no edits to its code,
  configurations or results.
* **V1 — Evidence core.** `evidence_core/` (contracts, graph, adapters, `LegacyAnySupportsPolicy`, replay, Slice 1
  aggregation), `evaluation/experiments/run_slice1_aggregation.py`, and the Slice 0/1 results. Extended
  **additively** unless a reviewed migration decision says otherwise; the exact legacy replay must keep passing.
* **V2 — Research co-pilot.** A future architecture, implemented **alongside** V0/V1 in its own package. Not created
  by M0.

These are logical boundaries; no files are moved or renamed.

## Rules

1. V2 must not depend on Experiment B aggregation or regex claim verification as its architecture. (The frozen
   Qwen pairwise prompt may be reused only as one pluggable stance provider.)
2. Existing hybrid retrieval may be reused through an interface without changing its historical implementation.
3. Every new experiment records configuration, model identity, dataset and split, hashes, failures (with failure
   category) and stopping conditions.
4. Improvements are evaluated against the appropriate frozen baseline on comparable inputs (same claims, same
   replayed retrieval where applicable; failures counted, never silently removed).
5. SciFact-specific assumptions stay in evaluation adapters and must not leak into domain-general contracts.
6. H5 v1.0 (`evaluation/experiments/results/h5/`, manual two-annotator design) remains a historical record. It is
   not marked superseded until a revised H5 protocol and the decision to adopt it are documented accurately.
7. M0 authorizes no model downloads, inference, retrieval tuning, architecture implementation or reserved-holdout
   evaluation. The 119 reserved labeled DEV claims (Phase 13 manifest) remain untouched.

## Provenance notes recorded at M0

* Phase 10H (`authoritative_provenance_closure/`), Phase 11A (`phase11_results/`) and `phase10_final_analysis/`
  describe the Phase 10F full run while it was incomplete and record hashes of that in-progress state. The run was
  later resumed and completed (`phase10_full_dev/resume_audit.json`); Phase 11B
  (`phase11_final/preservation_verification.json`) records the completed-run hashes, which match the committed
  files. The earlier documents are kept unchanged as historical records.
* At M0, all 268 recorded hashes found in the authoritative records (frozen sources, SciFact files, Phase 11B/12
  audits, Phase 10H closure documents, Phase 13 protected snapshot and manifests, NEI probe, H5 package) were
  recomputed and matched.
