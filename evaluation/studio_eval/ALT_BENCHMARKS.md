# Alternative evaluations of the Studio verifier (no independent annotation available)

Written and frozen 2026-10-24 local time (2026-10-23 ~20:50 UTC), **before any benchmark run** apart from a 3-claim
smoke test of the tooling. This is **not** the evaluation in `PROTOCOL.md`; that one still needs independently
labelled claims. Nothing here changes `PROTOCOL.md`, `run_eval.py` or `score.py`.

## Why alternatives

No claim set meets `PROTOCOL.md`: no independent human labels, and no claims unused by the project. The strongest
remaining evidence is a set with genuine human labels that has not been used to develop the verifier. Such a set
exists in the repository: SciFact TRAIN claims that the agentic verification study never touched. It is still
previously released public data, and it is a different task (scientific abstracts, not the Studio's web pages).

## System under test

The Studio's default, built by `run_eval.build_system()`:
* verifier `arm_c_studio`, injection screen `exclude`, quote attestation on;
* model `qwen3:4b`, with the digest prefix `359d7dd4bcdab3d8` (the study's model);
* evidence from the fixed packet only, through `Orchestrator._verify_mode`.

The paired baseline is `arm_c` (the historical Arm C method in the Studio) on the identical packets. Code: the release
branch at `a0967e5`, with uncommitted changes that touch no verifier, orchestrator or model code.

## Sets

`alt_benchmarks.py build` produces each set deterministically, ordered by SHA-256 of `studio-alt-v1:<tag>:<id>`.

| Set | Estimand | Claims | Labels from |
|---|---|---|---|
| `retro` | Accuracy on SciFact TRAIN claims, given their cited abstracts as the fixed packet | 300: 100 SUPPORTED, 100 CONTRADICTED, 100 NOT_ENOUGH_INFO | SciFact's human annotation. A unanimous SUPPORT/CONTRADICT rationale gives that label; no rationale in the cited abstracts gives NOT_ENOUGH_INFO; mixed-label claims are excluded |
| `prog` | Behaviour on two rule-defined transformations | 200: 100 `swap` (expected NOT_ENOUGH_INFO); 100 `negate` (labels flipped: 88 CONTRADICTED, 12 SUPPORTED) | Rules (see below), applied to SciFact labels; not a model |
| `adv` | Robustness to an instruction-like passage that asks for SUPPORTED | 30: 20 NOT_ENOUGH_INFO, 10 CONTRADICTED (`retro` claims plus one injected passage) | The original label is unchanged |

**Exclusions:** the 160 TRAIN claims the agentic study used to develop Arm C (`alt_excluded_train_ids.json`).
`prog` uses only claims outside the `retro` sample.

**Rules:**
* `swap`: the claim is paired with the abstracts cited by another claim, sharing no cited document and no content
  word (4 or more letters, not a stop word). Such a packet cannot state or contradict the claim.
* `negate`: "It is not true that <claim>." Evidence that supports the claim contradicts its negation, and vice versa.
  The first word is lower-cased only when it is an ordinary word.

**Frozen SHA-256 of the claims files:**
* `retro`: `8875125664c7a3f60d379c7acdeffc9a09f9304c2d03299eabee7f355d01ad63`
* `prog`: `1dab7106ca3ee575c198a631eed02ef0ac8d5beade476dfa3a99823d7612755a`
* `adv`: `2937ad14f0e901ef953bcf2ca1fb34b2376ea8e99bd43e74e5e2ba55c9d61a66`

The inputs are `claims_train.jsonl` (`f4c8fa82…`) and `corpus.jsonl` (`b8d6c896…`). The data and records stay under
`outputs/studio_eval_alt/`, which is not versioned.

## Analysis plan (fixed now)

* **Every set:** `score.summarise` as in `PROTOCOL.md` §5:
  * 3-class macro-F1 with a 95% percentile bootstrap over claims (10,000 resamples, seed 20261024);
  * per-class precision, recall and F1;
  * the confusion matrix;
  * false support: gold CONTRADICTED or NOT_ENOUGH_INFO predicted SUPPORTED;
  * false support among contradicted claims only;
  * missed contradiction;
  * the INSUFFICIENT rate;
  * operational failures.

  Failures and missing records count as NOT_ENOUGH_INFO and are reported separately.
* **Rule activations:** counted from the recorded verifier note and error (rule 1: unsided adjudication declined;
  rule 2: cited stance mismatch; rule 3: invalid stance fails the claim).
* **`retro` paired with `arm_c`:**
  * Δ macro-F1 with a paired bootstrap (seed 20261025);
  * a sign-flip permutation p (10,000 permutations, seed 20261026);
  * exact McNemar on accuracy.

  This is descriptive. No promotion decision is made from it.
* **Repeatability:** the first 30 `retro` claims (by claim id) are run a second time with `arm_c_studio`, and
  identical verdicts are counted.
* **Order:** `retro` (`arm_c_studio`), `prog`, `adv`, `retro` (`arm_c`), repeatability.

## How to read the results

* `retro` is a **retrospective, out-of-domain proxy**. It does not estimate the Studio's accuracy on its intended
  task, which is web evidence found by its own search.
* SciFact data has been handled by this project before. TRAIN claims outside the 160 were not used to develop Arm C,
  but the project's earlier phases inventoried them (for example the H5 sampling frame), and the public has had them
  since 2020.
* `prog` and `adv` test defined capabilities, not natural accuracy.
* None of these replaces the `PROTOCOL.md` evaluation.

## Execution record (added after the runs; the plan above is unchanged)

Run on 2026-10-23/24 UTC, from 20:47 to 00:08, in the pre-registered order. Every run exited 0 and every set is
complete (records: retro 300, prog 200, adv 30, `arm_c` baseline 300, repeat 30). The model was `qwen3:4b`, and the
`arm_c_studio` configuration fingerprint was `ca4da769c12b…`.

```text
python -m evaluation.studio_eval.alt_benchmarks build --set retro --out outputs/studio_eval_alt/sets/retro --agentic-runs <private run folder>
python -m evaluation.studio_eval.alt_benchmarks build --set prog  --out outputs/studio_eval_alt/sets/prog
python -m evaluation.studio_eval.alt_benchmarks build --set adv   --out outputs/studio_eval_alt/sets/adv
python -m evaluation.studio_eval.alt_benchmarks run --claims outputs/studio_eval_alt/sets/retro/claims.jsonl --out outputs/studio_eval_alt/runs/retro_arm_c_studio
python -m evaluation.studio_eval.alt_benchmarks run --claims outputs/studio_eval_alt/sets/prog/claims.jsonl  --out outputs/studio_eval_alt/runs/prog_arm_c_studio
python -m evaluation.studio_eval.alt_benchmarks run --claims outputs/studio_eval_alt/sets/adv/claims.jsonl   --out outputs/studio_eval_alt/runs/adv_arm_c_studio
python -m evaluation.studio_eval.alt_benchmarks run --claims outputs/studio_eval_alt/sets/retro/claims.jsonl --out outputs/studio_eval_alt/runs/retro_arm_c --verifier arm_c
python -m evaluation.studio_eval.alt_benchmarks run --claims outputs/studio_eval_alt/sets/retro/claims.jsonl --out outputs/studio_eval_alt/runs/retro_arm_c_studio_repeat --limit 30
python -m evaluation.studio_eval.alt_benchmarks publish --base outputs/studio_eval_alt --json-out showcase/data/studio_alt_evaluation.json
```

* `--agentic-runs` can be omitted: the excluded ids are then read from `alt_excluded_train_ids.json`.
* Aggregate results: `showcase/data/studio_alt_evaluation.json`.
* Per-claim records stay local.

Deviation: none. A 3-claim smoke test ran before the freeze (tooling check only; not part of any result).
