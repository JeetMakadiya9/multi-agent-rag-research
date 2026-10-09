# Slice 0 — Legacy Replay Report

Policy: `LegacyAnySupportsPolicy@experiment_b_frozen_66EA7B6B`. Acceptance: every replayable record: exact equality of run status, verdict, unit verdicts and coverage. socket connections forbidden during replay.

| Records | Replayed | Exact matches | Mismatches | Not replayable | Blockers |
|---|---|---|---|---|---|
| 606 | 604 | 604 | 0 | 2 | 0 |

## Per source

| Source | Results |
|---|---|
| phase10f_frozen_full_dev_B | {'EXACT_MATCH': 298, 'NOT_REPLAYABLE': 2} |
| phase13_B0 | {'EXACT_MATCH': 69} |
| phase13_B1 | {'EXACT_MATCH': 69} |
| phase13_B2 | {'EXACT_MATCH': 69} |
| phase13_B4 | {'EXACT_MATCH': 69} |
| phase13_nei_probe_B1 | {'EXACT_MATCH': 30} |

## Replayed outcome distribution (status, verdict)

- VERDICT|INSUFFICIENT_EVIDENCE: 303
- VERDICT|SUPPORTED: 255
- VERDICT|CONTRADICTED: 18
- FAILED_INCOMPLETE_ASSESSMENT|None: 28

## Mismatches

- none

## Not replayable

- phase10f_frozen_full_dev_B claim 343: status='INTERRUPTED'; error='Process interruption; callback completion unknown': no recorded Experiment B judgement outcome
- phase10f_frozen_full_dev_B claim 440: status='INTERRUPTED'; error='Process interruption; callback completion unknown': no recorded Experiment B judgement outcome

## Blockers

- none
