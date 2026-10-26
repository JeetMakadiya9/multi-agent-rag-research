# Results: historical studies and the current Studio

The project's benchmark results are **historical**. Each describes the system and data it was run on, not the current
Research Studio.

* **Agentic verification study (the origin of Arm C), run 2026-10-23.** It was a pre-specified paired comparison on
  the 188 labelled SciFact development claims (124 supported, 64 contradicted), which earlier project systems had
  already used, so they are not an untouched test set. Model `qwen3:4b`, one run per arm. The metric is two-class
  macro-F1 over all 188 claims, with INSUFFICIENT counted as an error.
  * Arm C reached macro-F1 0.7551 against 0.7153 for the single-pass rank baseline (B1R).
  * **This is a two-class figure.** Claims whose gold label is NOT_ENOUGH_INFO were not in the denominator. It cannot
    be compared with the three-class macro-F1 used for every Studio evaluation below; see
    `evaluation/studio_eval/TWO_VS_THREE_CLASS.md`.
  * The difference was +0.0398, with a paired 95% bootstrap interval of −0.0041 to +0.0850 and a permutation p of
    0.0813. It was **not statistically significant** at the pre-specified α = 0.05.
  * Arm C0, without the planner's search, scored 0.7576.
  * Arm C confirmed more contradicted claims: false support was 13/64 (20.3%) against 10/64 (15.6%).
  * Re-scoring the same recorded Arm C outputs with `arm_c_studio`'s rules gives 0.7528. This is a post-hoc,
    exploratory re-scoring, not an evaluation of `arm_c_studio`.
  * The aggregate figures, their definitions and caveats are in `showcase/data/historical_agentic_verification.json`.
    The project recomputed every figure from per-claim records with an independent re-scorer (all matched). Those
    records are not published while the SciFact licence terms are clarified, so a reader cannot yet recompute the
    figures from this repository.
* **Phase 11B benchmark** (systems B and C, `evaluation/experiments/results/phase11_final/`). On 176 paired SciFact
  development claims, both systems had a paired accuracy of 0.483 (exact McNemar p = 1.0): no statistically
  significant difference.
* **The Research Studio's own accuracy with `arm_c_studio` has not been evaluated.** No accuracy figure is claimed for
  it.
  * A frozen evaluation protocol, harness and scorer for it are in `evaluation/studio_eval/`: fixed evidence packets,
    macro-F1 with a bootstrap interval, and a promotion gate.
  * The annotation and freeze workflow is in the same folder: `ANNOTATION_GUIDE.md`, `DATASET_CONSTRUCTION.md` and
    `annotate.py`. It covers blinded task files, Cohen's kappa, third-person adjudication, and a `FREEZE.json` that the
    harness verifies.
  * The evaluation has not been run. It needs about 360 new claims with fixed evidence packets, labelled
    independently by two human annotators with a third-person adjudication: at least 300 claims and 100 per class
    after exclusions. No existing project claim set qualifies, and no model may supply the labels.
* **Alternative evaluations of the current verifier (2026-10-24; proxy, not the Studio's accuracy on its task).**
  The real `arm_c_studio` path (local `qwen3:4b`, injection screen and quote attestation on) ran on fixed packets of
  SciFact abstracts, under a plan frozen before the runs (`evaluation/studio_eval/ALT_BENCHMARKS.md`). Aggregates are
  in `showcase/data/studio_alt_evaluation.json`.
  * **Retrospective benchmark:** 300 SciFact TRAIN claims never used to develop Arm C (100 per class, SciFact's human
    labels, cited abstracts as the packet). 3-class macro-F1 0.5878, 95% bootstrap interval 0.5305 to 0.6421;
    accuracy 0.5867. False support 35/200 (19/100 among contradicted claims); missed contradiction 42/100; 44/100
    not-enough-information claims were decided anyway; 1 operational failure, counted as an error.
  * **Against the historical `arm_c` on the same packets:** 0.6123. The difference (−0.0245, paired interval −0.0628
    to +0.0138, permutation p 0.2326) is not significant. `arm_c_studio` declined 45 decisions made without a sided
    passage (`arm_c` got 23 of them right) and had less false support (35/200 against 42/200).
  * **Programmatic cases:** unrelated abstracts were recognised as not enough information in 89/100 cases;
    "It is not true that …" negations got the expected opposite label in 58/100.
  * **Adversarial cases:** an injected instruction asking for SUPPORTED was withheld in 30/30 packets and changed no
    verdict's correctness (17/30 correct with and without it).
  * **Repeatability:** 30/30 identical verdicts in a second run.
  * **Held-out test of the default (official SciFact DEV split, 300 claims, cited abstracts):** 3-class macro-F1
    0.5644, 95% interval 0.5073 to 0.6191; false support 30/176.
  * **Improvement attempt, not adopted.** `arm_c_studio_qr` repairs near-copy quotes that the strict quote check
    rejected (`docqa/agentic_candidates.py`; opt-in, not the default). It raised development macro-F1 from 0.5878 to
    0.6179. On the held-out split it scored 0.5558 against 0.5644 (Δ −0.0087, paired interval −0.0391 to +0.0216),
    and raised false support from 30/176 to 38/176. A larger local model (qwen3.5) was rejected on development:
    more false support, model failures and 2.7 times the runtime. Plan and decisions:
    `evaluation/studio_eval/IMPROVEMENT_PLAN.md`.
  * **Second round, also not adopted** (`evaluation/studio_eval/IMPROVEMENT_PLAN_2.md`). The model's reasoning mode
    was infeasible on the 4 GB GPU (600–700 s per claim). A revised verifier prompt raised macro-F1 on 90
    development claims (+0.0228, interval −0.0978 to +0.1401), but increased false support from 7/60 to 11/60, so it
    stopped at screening.
  * **Simpler verifiers on the same 300 development claims tie on macro-F1.** `quote_stance` scored 0.5983,
    `arm_c0` 0.6076, `arm_c` 0.6123 and `arm_c_studio` 0.5878; no paired difference is significant. (`arm_c0` is the
    2026-10-25 re-run with run identity. The earlier 0.5850 came from a file with duplicate records, which the
    duplicate policy rejects.) They differ in
    their errors: `quote_stance` has the least false support (26/200) but misses 74/100 contradictions; `arm_c0`
    misses the fewest (33/100) but has 42/200 false support; `arm_c_studio` lies between (35/200 and 42/100). The
    multi-step design changes the error trade-off; it does not raise accuracy here.
  * **Two metrics.** Every figure above is three-class macro-F1 (SUPPORTED, CONTRADICTED, NOT_ENOUGH_INFO; all
    claims). The historical study's two-class metric (SUPPORTED and CONTRADICTED claims only; deciding a claim that
    has no evidence is not penalised) gives higher numbers on the same records:

    | Set and run | Three-class | Two-class | NEI claims decided |
    |---|---|---|---|
    | `retro`, `arm_c_studio` | 0.588 | 0.692 | 44/100 |
    | `retro`, `arm_c0` | 0.608 | 0.758 | 59/100 |
    | SciFact DEV (`test_dev`; its 188 S/C claims are the historical study's), `arm_c_studio` | 0.564 | 0.658 | 49/112 |
    | Round-3 confirmation (249 claims), `arm_c_studio` | 0.501 | 0.674 | 77/164 |

    * Most of the distance between the historical 0.7551 and today's 0.50–0.61 is the third class.
    * The verifier's prompt was written for the two-class study. It pushes borderline passages to CONTRADICTS, and its
      most common three-class error is a NOT_ENOUGH_INFO claim called CONTRADICTED.
    * The 0.75 target in the improvement plans is three-class. No valid three-class run has reached it.
    * Definitions, every run and the caveats are in `evaluation/studio_eval/TWO_VS_THREE_CLASS.md`. Recompute with
      `python -m evaluation.studio_eval.historical_metric`.
  * These are scientific abstracts, not the web pages the Studio reads, and SciFact is public data, so these figures
    describe the verifier on a proxy task only. They are not the Studio's accuracy on its task.

## What can be reproduced from this repository

| What | Status |
|---|---|
| Phase 11B metrics | Recomputable: result files and per-claim records are in `evaluation/experiments/results/` |
| Agentic verification study figures | Readable as aggregates; **not recomputable** here (per-claim records withheld) |
| Inference re-runs of either study | Not complete: the study's retriever builder and run records are not included, and model outputs may differ |
| Research Studio accuracy | Not measurable yet: `evaluation/studio_eval/` holds a frozen protocol and harness, but no labelled claim set exists |
| Alternative (proxy) evaluations | Re-runnable: `python -m evaluation.studio_eval.alt_benchmarks build/run/score` rebuilds the sets from `data/scifact/` and re-runs them (needs Ollama with `qwen3:4b`; about 3 hours); outputs may differ slightly between runs |
| Software behaviour | Re-runnable: the offline tests (see Tests) |
