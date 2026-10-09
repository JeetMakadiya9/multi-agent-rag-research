# Follow-up Experiment Design (Not Started)

Keep Phase 10F B frozen as the baseline. Proposed B+ changes exactly one diagnosed factor: structured, auditable evidence-to-component mapping and stance aggregation. Do not change model, retrieval/reranking, evidence budget, controller, prompts outside the predeclared intervention, and retry policy at the same time.

Before implementation, conduct a blinded two-reviewer audit of a deterministic stratified sample of persisted correct and wrong labeled examples. Annotate evidence selection, passage adequacy, support/contradiction, insufficiency, claim complexity, ambiguity; adjudicate disagreements. This is diagnosis, not verifier tuning.

If authorized later, use a seed-2026 proportional 60-claim labeled pilot (39 SUPPORTED, 21 CONTRADICTED) with exact IDs frozen before execution. It is exploratory feasibility only. A later paired B/B+ run across all 188 labeled DEV claims would remain development evidence because this split informed diagnosis. Reserve test generalization for a separately approved protocol. Keep unlabeled claims for operational/retrieval reporting only.

Prospective proposed GO rule: B+ must improve paired accuracy by at least 5 percentage points and the paired 95% CI lower bound must exceed zero on a confirmatory run. The 5-point margin is a proposed practical threshold (one extra correct claim per 20), not learned from results; pre-register it. Proposed guardrails are retrieval Recall@1 no worse than −5 points and invalid/unavailable outputs no more than +2 points; these are prospective design tolerances and must be locked before outputs. Report paired exact McNemar and paired bootstrap CI, plus coverage, latency and failures. A 60-case pilot is not efficacy confirmation.

No follow-up experiment or intervention was run in Phase 12. Full details and assumptions are machine-readable in `followup_experiment_design.json`.
