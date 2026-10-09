# Phase 13 — Plan and Pre-Registration (written before any variant outcome was inspected)

Registered: 2026-10-05T14:23Z. At this point only the B0 control run had started; no B0–B5 outcome had been
scored or viewed.

## Objective

Identify, with controlled single-factor development experiments, which mechanisms suppress System B's
verification accuracy (frozen full-DEV baseline: 87/179 = 48.6%, macro-F1 0.4842), and implement only the
changes that are supported by measured evidence. The 90% figure is aspirational and is not a target that any
decision below optimises.

## Frozen baseline and protected artifacts

- B v1 = frozen Phase 10F System B (`system_b_verified_rag.run_system_b` + `experiment_b_verification`), unmodified.
- Never modified: `phase10_full_dev/`, `phase11_final/`, `phase11_results/` (Phase 11A), `phase12_diagnosis/`,
  SciFact data, frozen sources. All are SHA-256 snapshotted in `protected_hashes_before.json` (134 files) and
  re-verified at the end (`phase13_audit.json`).
- All Phase 13 code is additive under `evaluation/experiments/phase13/` (+ `tests/test_phase13_b_plus.py`).

## Development subset

`development_set_manifest.json`: 69 SciFact DEV labeled claims, seed 2026, stratified gold label × retrieval
bucket (10 per stratum) plus deterministic coverage top-ups (multi-sentence rationale ≥12, compound ≥8 → all 5
compound claims in the pool). Selection used only gold annotations, frozen retrieval ranks and claim-text
attributes — never predictions, correctness or failure status. 119 labeled claims are RESERVED and untouched.
The subset is for engineering decisions only and is not a benchmark.

## Variants (each changes exactly one factor relative to its reference)

| Variant | Reference | Single change | Hypothesis |
|---|---|---|---|
| B0 | — | none: live re-run of B v1 (control, reproducibility check) | — |
| B1 | B0 | verification evidence limit 2 → 8 | H1 evidence starvation |
| B2 | B1 | evidence text = retrieved sentence + same-abstract ±1 neighbours, clearly marked | H2 context loss |
| B3 | B0 | lexical guards v1 → v2 (G1–G5), exact re-aggregation of B0's recorded stances | H3 guard false negatives |
| B4 | B0 | judge system message v1 → v2 (explicit SUPPORTS/CONTRADICTS/NEUTRAL definitions) | H4 contradiction weakness |
| B5 | B0 | combination of only those components that pass the rule below | — |

Constant everywhere: replayed frozen retrieval (identical chunks/ranks/scores), claim decomposition, Experiment B
pairwise task and JSON schema, `max_judge_retries=1`, qwen3:4b (digest 359d7dd4…) on CPU (`num_gpu=0`,
`num_predict=256`, temperature 0, think=false), frozen label mapping and frozen 10F metric definitions.
Retrieval replay is justified because the frozen run's A and B retrieval orders are identical on all 297
comparable claims; replay holds retrieval constant by construction.

## Pre-registered decision rule ("independently demonstrated benefit")

A component is eligible for B5 only if, versus its reference on the same 69 claims:

1. net paired gain ≥ +3 correct claims (≈ +4.3 points), AND
2. neither SUPPORTED recall nor CONTRADICTED recall drops by more than 5 points, AND
3. failure rate does not increase by more than 2 points, AND
4. (B3 only) zero new SUPPORTED/CONTRADICTED verdicts on the 101 unlabeled frozen-B outputs.

A dramatic gain (≥ +10 claims) triggers a per-claim investigation before any combination. Results below the
threshold are reported, not discarded. With 69 claims, McNemar p-values are reported but are not expected to be
significant; the dev subset cannot establish efficacy.

H5 (NLI model) is investigated only if B1–B4 indicate the generative judge's stance assignment remains the
dominant error source.

## Known threats (declared in advance)

- H3 rules (G1–G5) were designed after inspecting the frozen catalogue of guard firings over all 300 DEV claims;
  H3 evidence is therefore in-sample on every DEV split, including the reserved claims.
- The dev subset is stratified (balanced labels and buckets), so its accuracy is not comparable to 48.6%.
- CPU inference ≈ 20 s per judge call; B1/B2 (~8 calls per claim) take several hours each.
- A later confirmation should use the 119 reserved labeled claims (paired B v1 vs B+), still reported as DEV
  development evidence because DEV informed diagnosis.

## Code identity at registration (SHA-256)

```
71e63b3a4978365769f80cbef241f9a2b7d6add969ae234c10dee4ef2b01b629  phase13/b_plus.py
bb80d9bb886936710708556ba2572c86765dee739e82a58bb562d53abb66b1cb  phase13/common.py
df296946f238a2cfcc2019c5cbbac56bc1b34530b3fdcef5f9be6032ae9464fb  phase13/devset.py
0ac1441d0ce1b82114803e34692cb3cc2f7e74a3904c6e9169600e491140c08e  phase13/run_experiment.py
8d4636fba32a9c09b6f7ed0a166d7892405b9d369c37e0f110546287e87ac606  phase13/scoring.py
83068c4dc2db63db54ddf8d493ced0771755f0dc4bf425838ef333cd6e1df7a0  development_set_manifest.json
```
