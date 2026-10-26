# Improvement plan, round 2 (frozen 2026-10-24 ~15:45 UTC, before any round-2 screening result was seen)

This continues `IMPROVEMENT_PLAN.md`, which is unchanged. Round 1 ended with no adopted candidate. The default
`arm_c_studio` scored 0.5644 on the held-out SciFact DEV split, which has now been **used once for a decision** and is
therefore not reused as an untouched test.

## Bottleneck (development records, round 1)

* **Features:** errors are spread evenly across claim features. Accuracy is 0.61 with numbers vs 0.57 without; 0.58
  with negation or decrease words vs 0.59 without; 0.62 with comparatives vs 0.58 without.
* **Largest cluster:** CONTRADICTED ↔ NOT_ENOUGH_INFO confusion, 51 of 124 errors.
* **Other clusters:** wrong side (27), and correct side lost to quote mis-copies (23).
* **Context:** evidence is complete. Every chunk of the cited abstracts reaches the verifier (at most 8 passages; 98%
  of packets have 4 chunks or fewer).
* **Conclusion:** the bottleneck is the model's stance judgement, not retrieval, context construction or control flow.

## Candidates

| Candidate | Targets | Status |
|---|---|---|
| `arm_c_studio_think`: verifier calls with qwen3's reasoning mode | Stance judgement | **Rejected on feasibility before screening.** It forces a 16k context; Ollama then offloads most of the model to the CPU on the 4 GB GPU. The smoke test took 600–700 s per claim (about 45× the baseline), and 1 of 3 calls failed with HTTP 500 during the reload |
| `arm_c_studio_p2`: verifier prompt P2, with the same model and settings | The C↔NEI boundary, plus quote copying ("without abbreviating any word") | Screened below |

Prompt P2 is in `docqa/agentic_candidates.py`, `VERIFIER_PROMPT_P2`. The study's prompt v1 is unchanged.

## Rules

1. **Screen** on `dev_sub90` (90 development claims) against the baseline's records for the same claims. Continue only
   if:
   * Δ macro-F1 ≥ +0.02;
   * false support is not higher;
   * operational failures are not higher;
   * time per claim is no more than 2× the baseline's.
2. **Development:** the full `retro` set (300), with the same four criteria as round 1.
3. **Confirmation, only if step 2 passes:** the **untouched TRAIN remainder**. This is every eligible TRAIN claim not
   in the 160 Arm C development claims and not in `retro`, `prog`, `adv` or `dev_sub90`: 249 claims (64 SUPPORTED,
   21 CONTRADICTED, 164 NOT_ENOUGH_INFO), with their natural distribution. Run once for the baseline and once for the
   candidate.

   **Adopt as default only if:**
   * the paired 95% interval of Δ macro-F1 is above 0;
   * false support is not higher;
   * failures are not higher.

   Even then, the owner decides the switch.
4. **If the candidate stops at any step,** no further variants are tried on these data. Round 2 ends.

## Screening result and decision (2026-10-24 16:11 UTC)

`arm_c_studio_p2` was run on `dev_sub90` and compared with the baseline's records for the same 90 claims:

| | `arm_c_studio` | `arm_c_studio_p2` |
|---|---|---|
| Macro-F1 [95% CI] | 0.6354 [0.5331, 0.7290] | 0.6582 [0.5496, 0.7528] |
| Accuracy | 0.6333 | 0.6778 |
| False support (C/NEI → S) | 7/60 | **11/60** |
| Among contradicted | 3/30 | **8/30** |
| Missed contradiction | 12/30 | **19/30** |
| NEI decided | 13/30 | 4/30 |
| Failures | 0 | 0 |
| s/claim | 13.9 | 17.6 |

* **Paired:** Δ +0.0228 [−0.0978, +0.1401], permutation p 0.7256; 31 verdicts differ.
* **Decision: rejected at screening.** False support is higher. The prompt fixed over-decision on NEI claims, but made
  the model miss more contradictions and support more contradicted claims.
* **Under rule 4, round 2 ends:**
  * no further variant is tried on these data;
  * the untouched TRAIN remainder stays untouched;
  * the default stays `arm_c_studio`.
* **Runtime note:** during and after the thinking-mode smoke test, Ollama loaded qwen3:4b with a 16k context, partly on
  the CPU (2.2 of 5.4 GB on the GPU). The p2 screen ran in that state, so its runtime is not directly comparable with
  the baseline's 13.9 s.
