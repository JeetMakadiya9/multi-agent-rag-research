# Two-class and three-class macro-F1: what each measures, and both results

**The project reports two different metrics under the name "macro-F1".** They answer different questions and must
never be compared with each other.

| | Two-class (historical) | Three-class (current protocol) |
|---|---|---|
| Used by | The agentic verification study of 2026-10-23 (Arm C 0.7551, B1R 0.7153) | `PROTOCOL.md`, `score.summarise`, and every Studio evaluation since 2026-10-24 |
| Classes averaged | SUPPORTED, CONTRADICTED | SUPPORTED, CONTRADICTED, NOT_ENOUGH_INFO |
| Claims counted | Only claims whose gold label is SUPPORTED or CONTRADICTED | All claims |
| An INSUFFICIENT answer, or a failure | An error for that claim's class | Counted as NOT_ENOUGH_INFO: correct when gold is NOT_ENOUGH_INFO |
| Deciding a claim that has no evidence | **Not penalised**; reported separately as "NEI claims decided" | Penalised: an error, and it lowers the precision of the class predicted |
| Code | `evaluation/studio_eval/historical_metric.py` | `evaluation/studio_eval/score.py` |

## Why the two numbers differ

* **The three-class metric adds the hardest decision:** recognising that the evidence does not settle the claim.
  * In this verifier, the most common error is a NOT_ENOUGH_INFO claim called CONTRADICTED. On the 249-claim
    confirmation set this was 45% of all errors.
  * The two-class metric never sees those claims.
* **The two-class metric rewards deciding.** A system that always picks a side is never penalised for deciding
  NOT_ENOUGH_INFO claims. A cautious system is penalised for every INSUFFICIENT answer.
  * `quote_stance` declines the most (9/100 NEI decided): it scores 0.576 two-class, but 0.598 three-class.
  * `arm_c0` and `arm_c` decide the most (57–59/100): they score about 0.755 two-class, but about 0.61 three-class.
* **The verifier was designed under the two-class metric.** Prompt v1 tells the model that a claim false according
  to the passage "is CONTRADICTS, not NEUTRAL", and treats a different quantity, or an effect where the claim asserts
  none, as CONTRADICTS. That helps when every claim has evidence. It hurts when many claims have none.
* **Class mix moves the three-class number.** The confirmation set is 66% NOT_ENOUGH_INFO, so CONTRADICTED precision
  falls. The two-class number is unaffected by NOT_ENOUGH_INFO claims.

**Which one the Studio needs:** the three-class metric. Real web evidence often does not settle a claim, and saying so
is one of the Studio's answers. The two-class number is reported only to connect today's results to the historical
0.7551.

## Both metrics on the same records

Computed by `historical_metric.both` from the existing records (2026-10-26). There were no new model calls.
* CIs are 95% claim bootstraps, seed 20261024.
* "NEI decided" counts NOT_ENOUGH_INFO claims that were still called SUPPORTED or CONTRADICTED.
* Values in one row come from one run and can be compared with each other. Compare rows only within the same set.

| Set | Run | Three-class (n) | Two-class (n) | NEI decided |
|---|---|---|---|---|
| `retro`, development (300; 100 per class) | `arm_c_studio` (default) | 0.588 [0.531, 0.642] (300) | 0.692 [0.629, 0.752] (200) | 44/100 |
| | `arm_c0` (rerun5) | 0.608 [0.550, 0.663] | 0.758 [0.698, 0.815] | 59/100 |
| | `arm_c` (historical Arm C rules) | 0.612 [0.556, 0.668] | 0.755 [0.694, 0.812] | 57/100 |
| | `quote_stance` | 0.598 [0.543, 0.652] | 0.576 [0.508, 0.640] | 9/100 |
| | `arm_c_studio_qr` | 0.618 [0.561, 0.671] | 0.737 [0.676, 0.796] | 48/100 |
| `test_dev`, SciFact DEV (300; its 188 S/C claims are the historical study's 188) | `arm_c_studio` | 0.564 [0.507, 0.619] (300) | 0.658 [0.585, 0.726] (188) | 49/112 |
| | `arm_c_studio_qr` | 0.556 [0.499, 0.609] | 0.683 [0.611, 0.750] | 61/112 |
| `dev_sub90`, development (90) | `arm_c_studio` (round-3 screen) | 0.649 [0.549, 0.739] (90) | 0.796 [0.694, 0.878] (60) | 13/30 |
| | `arm_c_studio_all` | 0.658 [0.555, 0.752] | 0.776 [0.666, 0.868] | 13/30 |
| | `arm_c_studio`, qwen3.5 | 0.704 [0.605, 0.791] | 0.824 [0.730, 0.900] | 10/30 |
| | `arm_c_studio_p2` | 0.658 [0.550, 0.753] | 0.655 [0.525, 0.765] | 4/30 |
| `confirm`, round-3 blind set (249; 64 / 21 / 164) | `arm_c_studio` | 0.501 [0.436, 0.564] (249) | 0.674 [0.556, 0.776] (85) | 77/164 |
| | `arm_c_studio_all` | 0.526 [0.463, 0.587] | 0.629 [0.513, 0.733] | 67/164 |

## Reading it

1. **On the historical study's own 188 claims** (`test_dev`'s S/C claims), today's default scores **0.658**
   two-class. The historical Arm C scored 0.7551. They are not the same experiment:
   * the study used its own retrieval, while these runs use the cited abstracts as fixed packets;
   * the study ran historical Arm C, while `arm_c_studio` adds Studio rules 1–3;
   * the runtime differs.

   The gap is not attributed to any one of these.
2. **On development, systems that decide more score about 0.75 two-class.** Examples are `arm_c0` and `arm_c`, which
   accept adjudicator decisions without a sided passage. They pay for it in NOT_ENOUGH_INFO claims decided (57–59 of
   100). Their three-class score is about 0.61.
3. **The two metrics can rank systems differently.**
   * `quote_stance` is last two-class, but mid-table three-class.
   * `arm_c_studio_p2` loses about 0.12 two-class against the default on `dev_sub90`, yet is level three-class.
4. **The 0.75 target in the improvement plans is three-class** (`IMPROVEMENT_PLAN.md`, `IMPROVEMENT_PLAN_3.md`). No
   valid three-class run has reached it. The one at 0.70 (qwen3.5, 90 claims) was rejected for false support,
   failures and runtime.
5. **A two-class figure of at least 0.75 does not meet a three-class target.** Don't use a two-class figure as
   evidence for one.

## Recompute

```bash
python -m evaluation.studio_eval.historical_metric --records RUN/records.jsonl --claims SET/claims.jsonl
```

Tests: `tests/test_historical_metric.py`.
