# Round 4 (DRAFT, not frozen, not authorized): the passage-ID fix

**Status:** this is a draft for the owner.
* Nothing in it is authorized, and nothing has been run.
* The owner must approve inference, then freeze this file before any run.
* The decision rules below are proposals. They are copied from earlier rounds and are not new authorized margins.

## One variable

**Candidate:** `arm_c_studio_all_idfix`.

**Comparator:** `arm_c_studio_all`, the round-3 candidate.

**What differs:** only the verifier-reply wrapper (`IdCheckedStudioModel`). Everything else is the same:
* prompts and model settings;
* batch size 1 and 8 passages per source;
* rules 1–3.

**Also run:** the default `arm_c_studio`, for reference only. It is not part of the decision.

## Data

**Development only:** `retro` (300 claims; SciFact TRAIN; 100 per class). It has been used many times.
* `retro` is the only development set with enough claims. `dev_sub90` is a subset of it.
* A run on `retro` is a **development screen, not a confirmation**.

**Not eligible:** `confirm` (249) and `test_dev` (300) have each been used for a decision. Every eligible SciFact TRAIN
and DEV claim is already used, and SciFact TEST has no labels.

**Exclusion check:** at build time, the run's claim file must equal `sets/retro/claims.jsonl`, SHA-256 verified. No
claim from any other set is added.

## Procedure

1. **Coverage gate.** This is mechanical: no accuracy is read.
   * On the first 10 claims of `retro` (file order), run `arm_c_studio_all_idfix` once (`--limit 10`), with stdout
     discarded.
   * Measure with `python -m evaluation.studio_eval.id_coverage`. It loads no gold or verdict fields.
   * **Continue only if (proposed):**
     * `silent_neutral` is 0;
     * `valid_matched` equals `passages_presented` in every claim that is not `ASSESSMENT_FAILED`;
     * at most 1 of the 10 claims is `ASSESSMENT_FAILED`.
   * Report `rejected_by_code`. If the gate fails, stop and report.
2. **Development screen.** Run each of the three systems once on `retro` (fresh folders, guarded, run identity), then
   score once with `alt_benchmarks score`.
3. **Proposed descriptive gates** (from round 1; the owner may change them before freezing):
   * Δ macro-F1 ≥ +0.02 against `arm_c_studio_all`;
   * false support not higher;
   * operational failures (including fail-closed ID rejections) not higher;
   * runtime ≤ 2× per claim.
4. **No confirmation in this round.** Even if every gate passes, confirmation waits for a new labelled set:
   `PROTOCOL.md` annotation or an external corpus.

## Limits

* SciFact abstracts stand in for the Studio's task.
* `retro` is development data. Its result cannot support the 75% hypothesis, and cannot support adoption as the
  default.
