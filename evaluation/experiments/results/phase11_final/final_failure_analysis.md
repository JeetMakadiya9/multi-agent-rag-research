# Failure analysis

Raw outcomes are 55 FAILED plus 5 INTERRUPTED (60 combined non-success). The 29 explicit local model endpoint failures are infrastructure/runtime failures, not scientific/model errors; they remain in operational totals and are excluded from valid-output denominators.

| System | Endpoint infrastructure | Verifier failure | Controller failure | Interrupted |
|---|---:|---:|---:|---:|
| A | 15 | 0 | 0 | 1 |
| B | 14 | 4 | 0 | 2 |
| C | 0 | 0 | 22 | 2 |

B has four verifier failures with invalid legacy-verdict errors. C has 22 controller/execution failures (21 Phase 5 task failures and one capability-contract error in the failure artifacts) and 2 interruptions. Exact raw errors and affected claims are in JSON. No failed attempt was retried or counted correct.
