# Slice 1 — Offline Aggregation Experiment

> Slice 1 does not evaluate slot-aware aggregation because the frozen judgements contain no slot-level coverage annotations. Claim aggregation is component-level because slot-level judgement coverage is absent from the frozen records.

Offline: no model inference, no retrieval, no network (socket connections forbidden during the run).

## Policies

- **control**: LegacyAnySupportsPolicy (Slice 0, unchanged)
- **primary**: ConservativeComponentPolicy@slice1-1
- **sensitivity**: IndependentCorroborationSensitivity (>=2 independence groups) — sensitivity analysis, not selected as primary policy

Records evaluated by all three policies: **604** (not replayable: 2).

## Distributions

| Policy | Distribution |
|---|---|
| legacy | {'CONTRADICTED': 18, 'FAILED': 28, 'INSUFFICIENT_EVIDENCE': 303, 'SUPPORTED': 255} |
| primary | {'CONTRADICTED': 19, 'INSUFFICIENT_EVIDENCE': 280, 'MIXED_EVIDENCE': 3, 'PARTIALLY_SUPPORTED': 4, 'SUPPORTED': 284, 'UNVERIFIED': 14} |
| sensitivity | {'CONTRADICTED': 19, 'INSUFFICIENT_EVIDENCE': 533, 'MIXED_EVIDENCE': 3, 'PARTIALLY_SUPPORTED': 1, 'SUPPORTED': 34, 'UNVERIFIED': 14} |

## Legacy → primary

- changed: 59 (9.8%); support-binary flips: 29
- by reason: `{'CONFLICT_MADE_EXPLICIT': 3, 'LEGACY_ATTEMPT_FAILED_ON_INVALID_PAIR__PRIMARY_DECIDES_FROM_VALID_JUDGEMENTS': 28, 'LEGACY_LEXICAL_GUARD_NOT_APPLIED': 24, 'PARTIAL_COMPONENT_SUPPORT_MADE_EXPLICIT': 4}`
- legacy SUPPORTED rejected by primary: 0; non-SUPPORTED → SUPPORTED: 29
- unexplained changes: 0

## Legacy → sensitivity (not selected)

`{'legacy_supported_rejected': 224, 'primary_supported_rejected': 250, 'by_reason_vs_primary': {'SINGLE_INDEPENDENCE_GROUP_SUPPORT_NOT_CORROBORATED': 254, 'UNCHANGED_FROM_PRIMARY': 350}}`

## Components, independence, INVALID

`{'components': 634, 'states': {'CONTRADICTED_COMPONENT': 19, 'MIXED_COMPONENT': 3, 'SUPPORTED_COMPONENT': 310, 'UNRESOLVED_COMPONENT': 288, 'UNVERIFIED_COMPONENT': 14}, 'judgement_incomplete_components': 28, 'invalid_only_components': 14, 'incomplete_with_valid_support': 5, 'conflict_within_independence_group': 2, 'supported_components_by_supporting_groups': {1: 273, 2: 30, 3: 7}, 'independence_note': 'independence key = Source.independence_group if present else source_id; SciFact sources carry no independence groups, so groups == documents here (UNKNOWN beyond document level)'}`

INVALID taxonomy: `{'total_invalid_judgements': 42, 'by_category': {'ENDPOINT_INFRASTRUCTURE': 28, 'MODEL_OUTPUT_SCHEMA': 13, 'TIMEOUT': 1}, 'by_component_state': {'CONTRADICTED_COMPONENT': 1, 'SUPPORTED_COMPONENT': 5, 'UNRESOLVED_COMPONENT': 8, 'UNVERIFIED_COMPONENT': 28}, 'note': 'infrastructure categories are execution failures, not reasoning errors'}`

Multi-unit records: `[{'source': 'phase10f_frozen_full_dev_B', 'claim_id': 300, 'legacy': 'SUPPORTED', 'primary_first_unit': 'SUPPORTED', 'diagnostic_all_units': 'SUPPORTED', 'note': 'diagnostic only; not used to choose the primary policy'}]`

NEI probe (descriptive, no gold): `{'legacy': {'CONTRADICTED': 1, 'FAILED': 1, 'INSUFFICIENT_EVIDENCE': 22, 'SUPPORTED': 6}, 'primary': {'CONTRADICTED': 1, 'INSUFFICIENT_EVIDENCE': 21, 'MIXED_EVIDENCE': 1, 'SUPPORTED': 7}, 'sensitivity': {'CONTRADICTED': 1, 'INSUFFICIENT_EVIDENCE': 27, 'MIXED_EVIDENCE': 1, 'SUPPORTED': 1}}`

## Gold evaluation (69 development claims only)

ONLY the 69 Phase 13 development claims (development evidence, not generalisation evidence); the 119 reserved labeled DEV claims are not scored

| Run | Denominator | Policy | n | Macro-F1 | SUP P / R | CON P / R | False support | Non-decisive | Accuracy (secondary) |
|---|---|---|---|---|---|---|---|---|---|
| phase13_B0 | legacy | legacy | 68 | 0.398 | 87.0% / 55.6% | 100.0% / 6.2% | 3 | 43 | 32.4% |
| phase13_B0 | legacy | primary | 68 | 0.414 | 84.6% / 61.1% | 100.0% / 6.2% | 4 | 40 | 35.3% |
| phase13_B0 | legacy | sensitivity | 68 | 0.086 | 100.0% / 2.8% | 100.0% / 6.2% | 0 | 65 | 4.4% |
| phase13_B0 | all | legacy | 69 | 0.396 | 87.0% / 55.6% | 100.0% / 6.1% | 3 | 44 | 31.9% |
| phase13_B0 | all | primary | 69 | 0.412 | 84.6% / 61.1% | 100.0% / 6.1% | 4 | 41 | 34.8% |
| phase13_B0 | all | sensitivity | 69 | 0.084 | 100.0% / 2.8% | 100.0% / 6.1% | 0 | 66 | 4.3% |
| phase13_B1 | legacy | legacy | 68 | 0.459 | 82.4% / 77.8% | 100.0% / 6.2% | 6 | 32 | 44.1% |
| phase13_B1 | legacy | primary | 68 | 0.478 | 81.6% / 86.1% | 100.0% / 6.2% | 7 | 28 | 48.5% |
| phase13_B1 | legacy | sensitivity | 68 | 0.293 | 100.0% / 30.6% | 100.0% / 6.2% | 0 | 55 | 19.1% |
| phase13_B1 | all | legacy | 69 | 0.457 | 82.4% / 77.8% | 100.0% / 6.1% | 6 | 33 | 43.5% |
| phase13_B1 | all | primary | 69 | 0.476 | 81.6% / 86.1% | 100.0% / 6.1% | 7 | 29 | 47.8% |
| phase13_B1 | all | sensitivity | 69 | 0.291 | 100.0% / 30.6% | 100.0% / 6.1% | 0 | 56 | 18.8% |
| phase13_B2 | legacy | legacy | 65 | 0.486 | 72.2% / 76.5% | 100.0% / 12.9% | 10 | 25 | 46.2% |
| phase13_B2 | legacy | primary | 65 | 0.495 | 73.0% / 79.4% | 100.0% / 12.9% | 10 | 24 | 47.7% |
| phase13_B2 | legacy | sensitivity | 65 | 0.217 | 80.0% / 11.8% | 100.0% / 12.9% | 1 | 56 | 12.3% |
| phase13_B2 | all | legacy | 69 | 0.469 | 72.2% / 72.2% | 100.0% / 12.1% | 10 | 29 | 43.5% |
| phase13_B2 | all | primary | 69 | 0.490 | 72.5% / 80.6% | 100.0% / 12.1% | 11 | 25 | 47.8% |
| phase13_B2 | all | sensitivity | 69 | 0.227 | 83.3% / 13.9% | 100.0% / 12.1% | 1 | 59 | 13.0% |
| phase13_B4 | legacy | legacy | 66 | 0.366 | 58.3% / 77.8% | 100.0% / 3.3% | 20 | 17 | 43.9% |
| phase13_B4 | legacy | primary | 66 | 0.388 | 59.3% / 88.9% | 100.0% / 3.3% | 22 | 11 | 50.0% |
| phase13_B4 | legacy | sensitivity | 66 | 0.202 | 72.7% / 22.2% | 100.0% / 3.3% | 3 | 54 | 13.6% |
| phase13_B4 | all | legacy | 69 | 0.363 | 58.3% / 77.8% | 100.0% / 3.0% | 20 | 20 | 42.0% |
| phase13_B4 | all | primary | 69 | 0.405 | 57.1% / 88.9% | 100.0% / 6.1% | 24 | 11 | 49.3% |
| phase13_B4 | all | sensitivity | 69 | 0.227 | 72.7% / 22.2% | 100.0% / 6.1% | 3 | 56 | 14.5% |

Decisions digest: `94ab7a9349289992fd7e64f12c242726027b1abebc0e0e9b7fbf488ca570b8ca`
