# Improvement Candidates — Ranked, Not Implemented

## 1. Structured evidence-to-component verification / auditable stance aggregation

- Observed bottleneck: Most current wrong valid labels still have some annotated gold-document overlap; many wrong outputs are INSUFFICIENT_EVIDENCE. This locates the diagnostic priority downstream of retrieval hit, but does not prove verifier reasoning is causal.
- Expected benefit: Tests whether explicit traceable evidence coverage changes labels. No gain assumed.
- Complexity: Medium-high
- Risk: Changes verification mechanism and output distribution; must keep frozen B as comparator.
- Research hypothesis impact: No if mechanism test only; intervention arm is a changed B variant.
- A/B/C comparability: Preserved with unchanged B and same per-claim runtime inputs.

## 2. Evidence selection traceability / controlled selection-only B+

- Observed bottleneck: Top-8 gold document/passage overlap is more frequent than selected evidence overlap in errors; selected field semantics are limited.
- Expected benefit: Separates failure to surface a candidate from verifier interpretation.
- Complexity: Medium
- Risk: Selection definition can change the evidence exposure; log exact IDs and freeze other components.
- Research hypothesis impact: No, as an isolated selection arm.
- A/B/C comparability: Keep same retrieval candidates/model/config; B remains baseline.

## 3. Blinded manual error audit of stratified wrong predictions

- Observed bottleneck: Persisted artifacts do not prove entailment, claim interpretation, numerical mismatch, compoundness, or aggregation cause.
- Expected benefit: Provides human-coded diagnostic labels before any intervention choice.
- Complexity: Low-medium
- Risk: Reviewer subjectivity; use blind rubric, two annotators and adjudication.
- Research hypothesis impact: No.
- A/B/C comparability: Does not change systems; is an annotation study.

## 4. Atomic claim decomposition

- Observed bottleneck: Not established; lexical complexity features are only proxies.
- Expected benefit: Unknown until human audit indicates compound claims.
- Complexity: High
- Risk: Changes prompts/calls and can confound mechanism.
- Research hypothesis impact: Potentially; isolate.
- A/B/C comparability: Separate arm only.

## 5. Improve retrieval/reranking

- Observed bottleneck: Retrieval misses explain 14 B and 14 C wrong predictions by document overlap among valid labeled evaluable rows; misses remain but are minority of observed errors.
- Expected benefit: Could improve retrieval-miss cases; not expected to resolve errors when gold docs already present.
- Complexity: Medium-high
- Risk: Changes candidate pool and makes verifier comparisons less controlled.
- Research hypothesis impact: No if isolated retrieval arm, but broad architecture comparison changes.
- A/B/C comparability: Separate arm with B baseline.

## 6. Dedicated NLI/larger model

- Observed bottleneck: No model-size counterfactual or evidence to identify capacity as cause.
- Expected benefit: Unknown.
- Complexity: High
- Risk: Adds model/provider confounding and new identity/latency constraints.
- Research hypothesis impact: Could.
- A/B/C comparability: Separate controlled model arm.

## 7. Multi-agent orchestration changes

- Observed bottleneck: B/C retrieval chunk lists are mostly identical; paired labels/correctness are highly similar. No evidence broad orchestration is the bottleneck.
- Expected benefit: Not supported by this diagnosis.
- Complexity: High
- Risk: Changes C research architecture and causal attribution.
- Research hypothesis impact: Likely.
- A/B/C comparability: Only C+ arm, retain C.


Primary recommendation: test an auditable evidence-to-verdict intervention only after blind diagnosis, with frozen B retained. Secondary: selection-only traceability/intervention. Do not prioritize retriever, larger model, or controller rewrites first.
