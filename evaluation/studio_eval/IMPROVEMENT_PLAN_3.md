# Round 3: corrected Arm C judge (frozen 2026-10-25 17:30 UTC, before any round-3 run)

* **Authorised by the owner on 2026-10-25:** "implement the corrected Arm C judge (every passage judged; no 2-passage
  cap in verify mode) as a separate candidate with tests, freeze a plan that defines the blindfolded confirmation on
  the untouched TRAIN remainder, screen on development, then run the confirmation once."
* **This plan does not change** `PROTOCOL.md`, `IMPROVEMENT_PLAN.md` or `IMPROVEMENT_PLAN_2.md`. It is the round-3
  authorisation that round 2's rule 4 required.

## Candidate

**`arm_c_studio_all`** (`docqa/agentic_candidates.py`, `StudioArmCAllPassagesVerifier`):
* **Every passage judged:** one verifier call per passage (`verifier_batch` 1 instead of the frozen 8).
* **No source cap when verifying:** up to 8 passages per source in verify mode (`verify_per_source` 8, passed by
  `Orchestrator._verifier_rank_options`). The default verifiers pass nothing and keep the store's default of 2.
* **Unchanged:** prompts, model and its settings, planner, adjudicator, Studio rules 1–3, injection screen and quote
  attestation.

**Tests:** `tests/test_agentic_candidates.py`.
* The settings differ from the frozen ones only in `verifier_batch`.
* Each passage gets its own call, with no passage left unlisted.
* Verify mode passes `per_source` 8 only for this candidate.

The default is unchanged (370 related tests passed; frozen files unchanged).

**Code at freeze** (SHA-256, first 16 characters):

| File | Hash |
|---|---|
| `docqa/agentic_candidates.py` | `d706d7bad3505179` |
| `docqa/agentic.py` | `2696ef3b8d0ed598` |
| `webrag/orchestrator.py` | `e1885c97b05053d2` |
| `evaluation/studio_eval/alt_benchmarks.py` | `f464338d910886c8` |
| `evaluation/studio_eval/score.py` | `0e0f744416798043` |
| `evaluation/studio_eval/records.py` | `5834bf1175d4bc75` |

**Comparator:** the default `arm_c_studio`, run in the same session, on the same claims and under the same runtime.

**Runtime:**
* Ollama 0.40.11 with a 4096-token context;
* `qwen3:4b`, digest `359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7`, read from the local
  server and written in each run log.

## Data

| Role | Set | Claims | SHA-256 of `claims.jsonl` |
|---|---|---|---|
| Screening (development, used before) | `dev_sub90` | 90 (30 per class) | `dcf101a6e8fa6769bf21f26a37ffad1dd469162a2f8b0a3539396f6ccee01ce2` |
| **Blind confirmation** | `confirm`: every eligible TRAIN claim never evaluated by the project (not in the 160 Arm C development claims, `retro`, `prog`, `adv` or `dev_sub90`) | **249**: 64 SUPPORTED, 21 CONTRADICTED, 164 NOT_ENOUGH_INFO | `3e2ff5c1b926eeefd6cc4d06cbc5d7ab93c0ba0ec3973e594d90ba4face6b91d` |

* **Confirmation set:** overlap with all used sets is 0 (checked at build). Labels are SciFact's human annotations;
  packets are the cited abstracts, as in `retro`.

## Procedure

1. **Screening on `dev_sub90`.** Run `arm_c_studio` and then `arm_c_studio_all`, each once (fresh folders, guarded).
   * It is descriptive and a feasibility check, **not an accuracy gate**. There is one pre-specified candidate and
     nothing is tuned.
   * The confirmation is cancelled only on an operational failure: the candidate has more than 5 operational failures
     beyond the baseline's, or takes more than 4 times the baseline's time per claim.
2. **Blind confirmation on `confirm`.** Run `arm_c_studio` and then `arm_c_studio_all`, each exactly once.
   * Nobody reads predictions, records or run logs beyond operational status until both runs are complete.
   * Interrupted runs are resumed only in the same folder and run (guarded, memory-stop cycles). No claim is re-run
     selectively.
3. **Score once:** `alt_benchmarks score --records <candidate> --baseline <arm_c_studio> --claims sets/confirm`.
   * This is `score.summarise`: 3-class macro-F1 with a 95% claim bootstrap (seed 20261024), and paired Δ with a
     bootstrap (seed 20261025) and a permutation test (seed 20261026).
   * A failure counts as NOT_ENOUGH_INFO.

## Decision rules (fixed now)

**The 75% hypothesis** (target from `IMPROVEMENT_PLAN.md`). It is **SUPPORTED UNDER THE PREDECLARED RULE** only if,
on `confirm`, all three hold:
* (a) `arm_c_studio_all` 3-class macro-F1 ≥ 0.75 (point estimate, as `IMPROVEMENT_PLAN.md` states);
* (b) no regression in false support: the candidate's count of gold CONTRADICTED or NOT_ENOUGH_INFO claims predicted
  SUPPORTED is not higher than `arm_c_studio`'s on the same claims;
* (c) no regression in class recall: each class's recall is at least `arm_c_studio`'s.

Otherwise it is **NOT SUPPORTED UNDER THE PREDECLARED RULE**.
* `IMPROVEMENT_PLAN.md` never quantified "material". This plan fixes zero tolerance before any data, which is
  stricter, and follows the zero-tolerance false-support rule of `PROTOCOL.md` §6.
* The 95% interval is reported, but it is not part of rule (a).

**Adoption as the Studio default** is a separate decision. It requires all of the following, and the owner then
decides:
* the paired 95% interval of Δ macro-F1 is above 0;
* (b) and (c) above hold;
* operational failures are not higher.

## Limits, stated in advance

* `confirm` has only **21 contradicted claims**. The CONTRADICTED recall and F1 are imprecise, and so is macro-F1.
* These are scientific abstracts (SciFact), a stand-in for the Studio's web-evidence task. **Real-task accuracy stays
  unmeasured.**
* "Blind" means three things here: the claims were never evaluated, the labels are not used by the system, and no
  output is inspected before both runs end.
* It is **not** the `PROTOCOL.md` evaluation: there is no independent annotation and no `FREEZE.json`. It is not a
  protocol-level confirmation.
* The runtime (Ollama 0.40.11) differs from earlier rounds' (0.40.6). Comparisons with earlier records are
  descriptive only.
