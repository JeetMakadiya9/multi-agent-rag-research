# Experiment B contract

## Scope

Experiment B consumes the user question, the already-generated strong-RAG answer,
and its initial retrieval results. Experiment A remains unchanged. Experiment B
does not plan research, call web search, or orchestrate agents.

## Unit semantics

`verification_units.VerificationUnit` keeps the original sentence text and adds
only shallow rule-derived fields when recognizable. Conditions, dates,
quantities, negation, causal language, comparisons, and coordinated objects
remain attached to that sentence. The current analyzer deliberately keeps a
whole multi-clause sentence together when it cannot safely isolate clauses;
ambiguous pronouns are marked in `uncertainties`.

## Evidence and judging

`run_experiment_b(..., retrieval_provider=...)` creates one query per unit from
the question context, original unit text, detected entities/relations, and
qualifiers. It calls the injected provider, merges its results with the initial
RAG results, and prioritizes the unit-specific evidence. An insufficient result
can trigger at most `max_targeted_retrieval_retries` focused retrieval calls.
Without a provider, the compatibility path uses the caller-supplied initial
evidence only and records this limitation in the trace.

The judge JSON includes three-way verdicts, selected support/refute evidence
IDs, supported/unsupported components, a reason, and per-evidence stance and
quality assessments. Invalid JSON/schema is retried only up to
`max_judge_retries`. Deterministic validation enforces evidence-ID membership,
support/refute requirements, explicit partial-support fields, explicit dates
and numbers, and presence of each parsed coordinated object. It cannot itself
prove semantic entailment or detect a model that omits an unrecognized
component; verification quality remains an empirical question.

Opposing evidence is left `INSUFFICIENT_EVIDENCE` unless explicit provenance
metadata decisively distinguishes a side (`source_reliability`, `authority_score`,
`directness_score`, `specificity_score`, or explicit supersession metadata).
Retrieval relevance scores are never used as entailment scores.

## Revision, trace, metrics

The report preserves the original answer and returns verification units, verdicts,
evidence, validation overrides, retries, errors, timing/call counts, and a
separate deterministic revised answer. `revise_answer=True` puts that revision
in `ResearchPipelineResult.final_answer`; the default preserves the prior API
behavior and returns the original answer there while retaining
`report.revised_answer`.

Gold-label accuracy/F1, evidence precision/recall, and citation quality remain
`null` until a manually annotated evaluation set is evaluated. Call counts,
latency, and observed verdict rates are measured directly. Ollama thinking and
format settings are explicit in `ExperimentBConfig`; the default is
`qwen3:4b`, `think=false`, JSON format, and temperature zero. This configuration
does not imply an accuracy guarantee.

## Ablation configuration

- B1: `unit_specific_retrieval=False`, `validator_enabled=False`, retries `0`, answer revision disabled.
- B2: `unit_specific_retrieval=False`, validator enabled, retries `0`, answer revision disabled.
- B3: `unit_specific_retrieval=True`, validator enabled, bounded targeted retry enabled, answer revision disabled.
- B4: `unit_specific_retrieval=True`, validator enabled, bounded targeted retry enabled, answer revision enabled.

Schema parsing and invalid evidence-ID checks remain safety requirements in all
configurations. No Experiment C components are included.

## Current limitations

- Rule extraction is sentence-level and conservative; this is not a dependency
  parser or coreference resolver.
- Coordinated-object coverage uses lexical presence as a necessary check, not
  proof of entailment.
- Conflict resolution is intentionally conservative and uses only explicit
  provenance metadata.
- No live Ollama generation or full hybrid retrieval run is claimed by the
  deterministic unit tests.
