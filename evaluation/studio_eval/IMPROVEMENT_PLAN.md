# Improvement plan for the Studio verifier (frozen 2026-10-24, before any candidate run)

Goal: improve `arm_c_studio` toward 0.75–0.80 macro-F1 **without leaking the test set**. This plan fixes the splits,
the candidates, the decision rules and the final test before the candidates are run. It does not change
`PROTOCOL.md` or `ALT_BENCHMARKS.md`.

## Splits

| Role | Set | Claims | Status |
|---|---|---|---|
| Development (diagnosis, candidate selection) | `retro`: SciFact TRAIN, 100 per class (`ALT_BENCHMARKS.md`) | 300 | Already seen (baseline 0.5878) |
| Development subset for slow candidates | First 30 per class of `retro`, by SHA-256 of `studio-alt-v1:sub90:<id>` | 90 | Subset of development |
| **Held-out test** | `test_dev`: every single-label SciFact DEV claim with its cited abstracts (124 S, 64 C, 112 NEI) | 300 | **Frozen**: claims file SHA-256 `cff6c88cf8ec468e986b504042e20a5875649f2e9aa6e73936bad240c6f37a65`. No run has used it |

**About the test set:**
* SciFact DEV was used by the project's historical studies, in a different setup: two classes, retrieval packets, and
  the post-hoc choice of Studio rule 1. It has never been used in this packet format, or to develop any candidate.
* It is the standard SciFact evaluation split.
* It is **still a proxy for the Studio's task**: scientific abstracts, not web evidence from the Studio's own search.

## Diagnosis that motivates the candidates (development set, baseline records)

There are 124 errors out of 300:

| Error | Claims |
|---|---|
| The verifier took the correct side, but the quote was not exact text, so the stance was demoted (S→NEI 19, C→NEI 4) | 23 |
| No passage took a side although gold is S or C (S→NEI 10, C→NEI 19) | 29 |
| Gold NEI but decided (→C 28, →S 16) | 44 |
| Wrong side (C→S 19, S→C 8) | 27 |
| Operational failure | 1 |

* **Quote analysis:** 28 rejected sided quotes on 31 claims. 21 have character similarity of at least 0.90 to a passage
  span, mostly abbreviated words. One quote added a negation that is not in its passage.
* **Repeatability:** re-running those 31 claims changed 7 verdicts. The mis-copies vary between identical runs.

## Candidates (evaluated independently, in this order)

1. **`arm_c_studio_qr`: quote repair** (`docqa/agentic_candidates.py`). A sided quote that is not exact passage text
   is replaced by the passage span it was copied from, but only if:
   * the similarity is at least 0.90;
   * the quote and the span have the same numbers;
   * the quote and the span have the same negation words.

   Rules 1–3 are unchanged. Evaluated on the full development set (300).
2. **Larger local model:** `arm_c_studio` with `qwen3.5:latest` (9.7B, partly offloaded on the 4 GB GPU; about 60 s per
   claim). Evaluated on the 90-claim development subset against the baseline's records for the same claims.
3. **Rejected without a run:** disabling Studio rule 1.
   * It is a safety rule.
   * On development, `arm_c` (rule 1 off) gained +0.0245 macro-F1, which is not significant (CI −0.0628 to +0.0138
     for the reverse difference), and raised false support from 35 to 42 of 200.
   * It is kept.

## Decision rules (fixed now)

A candidate is **retained** on development only if:
* its macro-F1 is at least **+0.02 above the baseline** on the same claims (above the run-to-run noise seen in the
  repeat runs);
* its false support (gold C or NEI predicted S) is **not higher** than the baseline's;
* its operational failures are **not higher**;
* its runtime is feasible on this workstation (no more than 2× the baseline's time per claim for a default).

If more than one candidate qualifies and they are compatible, the combination is run once on development before
freezing.

## Final test (run once)

* **Runs:** the baseline `arm_c_studio` (`qwen3:4b`) and the frozen final candidate, if one is retained, each once on
  `test_dev`. Nothing is changed after these runs.
* **Analysis:** `score.summarise` as before, plus paired Δ macro-F1 with a bootstrap (seed 20261025), a sign-flip
  permutation (seed 20261026) and McNemar.
* **Outcome:**
  * **TARGET MET** only if the held-out macro-F1 is at least 0.75, with no material regression in false support or any
    class's recall.
  * Because `test_dev` is a proxy, the real-task status stays **UNMEASURABLE FOR REAL TASK** whatever the proxy result.

## Decision record (development, 2026-10-24; written before the held-out test)

| Candidate | Development data | Macro-F1 (base → cand) | Δ, paired 95% CI, p | False support | Failures | s/claim | Decision |
|---|---|---|---|---|---|---|---|
| `arm_c_studio_qr` (quote repair) | `retro`, 300 | 0.5878 → 0.6179 | +0.0302 [+0.0025, +0.0585], p 0.0417 | 35 → 35 /200 | 1 → 0 | 13.8 → 14.0 | **Retained** (all four rules hold) |
| `arm_c_studio` with `qwen3.5:latest` | `dev_sub90`, 90 | 0.6354 → 0.7044 | +0.069 [−0.0592, +0.1999], p 0.2802 | 7 → 9 /60 | 0 → 3 | 13.9 → 38.2 | **Rejected**: false support, failures and runtime rules fail |

The frozen final candidate is `arm_c_studio_qr` (`docqa/agentic_candidates.py`, SHA-256 prefix
`2c9b904543c72c23`), with model `qwen3:4b`. No combination is needed, because
only one candidate was retained. The held-out test runs next, once each for `arm_c_studio` and `arm_c_studio_qr`.

## Held-out test result (run once, 2026-10-24 11:53–14:08 UTC; `test_dev` SHA-256 verified before the run)

| System | Macro-F1 [95% CI] | Accuracy | False support (C/NEI → S) | Among contradicted | Missed contradiction | NEI decided | Failures |
|---|---|---|---|---|---|---|---|
| `arm_c_studio` (default) | **0.5644** [0.5073, 0.6191] | 0.5800 | 30/176 | 14/64 | 31/64 | 49/112 | 2 |
| `arm_c_studio_qr` (candidate) | **0.5558** [0.4986, 0.6094] | 0.5733 | 38/176 | 16/64 | 28/64 | 61/112 | 1 |

* **Paired:** Δ −0.0087, 95% CI [−0.0391, +0.0216] (seed 20261025), permutation p 0.5768, McNemar p 0.8388; 30
  verdicts differ.
* **Outcome:** the development gain did not generalise, and false support rose. Two development criteria fail on the
  held-out data: no improvement, and higher false support.
* **Decision:** the candidate is **not adopted**. `arm_c_studio` stays the default. `arm_c_studio_qr` stays
  available as an opt-in research variant only.
* **Target status:** NOT MET on the proxy (0.56, far from 0.75), and UNMEASURABLE FOR REAL TASK.
