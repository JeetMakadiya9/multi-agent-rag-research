# Phase 11 — Experimental Results and Scientific Interpretation

## 1. Evaluation Objective

This package consolidates persisted Phase 10 artifacts into thesis-ready results and cautious scientific interpretation. It is a deterministic post-hoc analysis; no inference, evaluation resume, external API, or search was performed.

## 2. Dataset and Experimental Configuration

SciFact-Orig DEV; 300 claims selected in canonical DEV file order (selection algorithm `all-dev-file-order-v1`, seed 0). The corpus has 5,183 documents and the recorded retrieval index has 45,972 chunks. Systems: A is the existing RAG baseline; B is RAG with the frozen Experiment B verifier; C is bounded research execution using the real 8A/8B/8C controller, 7B dispatch, Phase 6 retrieval/verification, and explicit 7C continuation. Model/runtime identity and hashes are in `phase11_reproducibility.md`.

## 3. Evaluation Completion Status

The full-DEV run is **PARTIAL**. It planned 900 system/claim attempts, persisted 662 durable attempt rows (606 success, 52 failed, 4 interrupted), and produced 660 per-example rows for 220 full A/B/C claim triples. Of the planned pairs, 238 have no durable attempt row; one has an inflight marker for claim 1029/C with unknown callback completion, and 237 have neither attempt nor inflight marker. There are 197 all-success A/B/C triples. Among 220 scored claims, 136 have a gold label and 84 do not. The other 80 selected claims lack a scored per-example record; their label status is not inferred.

## 4. System A — Strong RAG

A produced 205 successful executions, 15 failed attempts, and 1 interrupted attempt. It has no verifier-derived classification label, so classification accuracy and F1 are **N/A**. Its document-level Recall@1 was 106/152 (0.697). Mean latency among successful attempts was 27.221 s (n=205). Persisted call totals were 220 LLM and 220 retrieval calls; verification calls are **N/A** for this system definition.

## 5. System B — RAG + Claim Verification

B produced 202 successful executions, 17 failures, and 2 interruptions. Among labeled claims with successful valid predictions, 58/128 were correct (accuracy 0.453; macro F1 0.465). Recall@1 was 106/151 (0.702). Successful-attempt mean latency was 40.834 s (n=202). Persisted call totals: 484 LLM, 219 retrieval, and 219 verification calls. Three persisted B attempts contained invalid legacy verifier verdicts; these are failures, not incorrect predictions.

## 6. System C — Bounded Multi-Agent Research

C produced 199 successful executions, 20 failures, and 1 interruption. Among labeled claims with successful valid predictions, 56/126 were correct (accuracy 0.444; macro F1 0.461). Recall@1 was 105/151 (0.695). Successful-attempt mean latency was 40.291 s (n=199). Persisted call totals: 448 LLM, 438 retrieval, 219 verification, and 438 controller cycles. The observed trace audit supports a bounded, controlled path; researcher transition/gate records remain present.

## 7. Classification Results

Accuracy denominators include only labeled claims with successful execution and a valid verifier-derived output. Unlabeled claims, failed attempts, interrupted attempts, and invalid/missing outputs are not treated as correct. A is not applicable. Macro F1 is reported using the class support in valid predictions as encoded by the Phase 10 metric artifact; details and confusion matrices are in the machine-readable JSON.

## 8. Retrieval Results

Recall@1 is micro document overlap between retrieved document IDs and annotated gold-evidence document IDs, among valid scored retrieval outputs. The denominator is the number of gold evidence documents (A 152, B 151, C 151); valid retrieval evaluation rows were A 136, B 135, C 135. Recall@1 is not entailment, claim verification, or final factual correctness. Unavailable outputs are excluded, not scored as misses.

## 9. Verification Results

B/C produce verifier-derived label categories, including `SUPPORTED`, `CONTRADICTED`, and `INSUFFICIENT_EVIDENCE`. B has persisted evidence-component stance records; C has claim verdict and nested evidence-assessment records. Evidence citation precision/recall/F1 are document-overlap statistics from the saved evaluator; their metric-specific denominators are reported in `phase11_verification_analysis.json`. The unsupported-claim rate is **NOT AVAILABLE** because the persisted field does not provide a populated rate with a defined denominator. Retrieval success must not be interpreted as evidence entailment.

## 10. Efficiency Results

Latency is reported separately for successful attempts, failed attempts, and all timed durable attempts. Interrupted attempts have no recorded latency. No full-run runtime projection is made. Call totals include persisted counts from both successful and failed attempts; missing counts remain missing.

## 11. System C Execution-Path Results

For 199 successful C executions with persisted trace audit records, each audited stage passed in 199/199 records: Phase 8A validation, Phase 8B execution, Phase 7B dispatch, Phase 6 retrieval, Phase 6 verification, continuation authorization, researcher transition, and completion. Continuation authorization is present and a researcher transition/gate remains in the trace. This supports the term **bounded autonomous execution path**, not “fully autonomous system.” Failed executions are separately retained and are not included in the successful-trace denominator.

## 12. Failure Analysis

The 56 persisted non-success records comprise 29 local endpoint connection refusals, 19 Phase 5 execution failures through C, 3 invalid B verifier verdicts, 1 C capability-request failure, and 4 interruptions. Endpoint refusals are infrastructure/runtime failures. Execution/control-path failures are not wrong valid predictions. See `phase11_failure_analysis.md` for affected claims, systems, and interpretation.

## 13. Paired B vs C Analysis

Among 126 valid paired predictions, B and C agreed on 124 and differed on 2. Both were correct on 55, both incorrect on 69, B-only correct on 1, and C-only correct on 1. The paired accuracy difference is 0.000. The existing bootstrap interval is included as descriptive uncertainty only.

**This analysis does not establish statistical significance or superiority.**

## 14. Scientific Interpretation

The persisted run demonstrates measurable execution of three system configurations and traceable bounded controller operation for successful C records. The partial data provide descriptive observations of classifier outputs, retrieval overlap, evidence selection, latency, calls, and failure behavior. B/C agreement is high on the valid paired subset but does not imply equivalence or superiority. The incomplete coverage and observed failures prevent strong comparative conclusions.

## 15. Limitations

The run is incomplete; 80 selected claims have no scored triple, failures are non-randomly concentrated in execution pathways, invalid and unavailable outputs have separate denominators, and the metrics describe only this persisted run. Retrieval/evidence overlap cannot establish entailment. No test-set evaluation occurred. Current Git HEAD differs from the commit recorded in the provenance snapshot; both are reported without an inferred resolution.

## 16. What Can Be Claimed

The study can state that an interrupted SciFact DEV run persisted 220 complete A/B/C score triples and 197 all-success triples, and that 199 successful C traces passed the recorded bounded controller-stage audit. It can report the observed denominators, metrics, and failure categories for this run.

## 17. What Cannot Be Claimed

Do not claim completion of the 300-claim benchmark, full-DEV generalization, test performance, statistical significance, B/C superiority, fully autonomous ungated research, or factual correctness from retrieval Recall@1.

## 18. Reproducibility Information

Run ID `02fb9c7a1268476ca1f9b12d9dbfb25e`; config fingerprint and full source/dataset/model identities are listed in `phase11_reproducibility.md`. Raw Phase 10 attempt/per-example JSONL SHA-256 values are checked before and after Phase 11 package creation.

## 19. Final Conclusion

Phase 11A consolidates persisted evidence into a reproducible results package. The Phase 10 full-DEV evaluation remains **PARTIAL**. Report the measurements descriptively and preserve failures and missing cases; the data do not establish comparative superiority or a completed full-DEV benchmark.

## Appendix — Compact results table

| System | Attempts | Success | Fail | Interrupted | Labeled | Valid predictions | Correct | Accuracy | Macro F1 | Recall@1 | Success mean latency (s) | LLM | Retrieval | Verification | Controller cycles |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 221 | 205 | 15 | 1 | 136 | N/A — no verifier-derived classification | N/A — no verifier-derived classification | N/A — no verifier-derived classification | N/A — no verifier-derived classification | 106/152 = 0.697368 | 27.221 | 220 | 220 | N/A — no verifier | N/A — not a controller system |
| B | 221 | 202 | 17 | 2 | 136 | 128 | 58 | 0.453 | 0.465 | 106/151 = 0.701987 | 40.834 | 484 | 219 | 219 | N/A — not a controller system |
| C | 220 | 199 | 20 | 1 | 136 | 126 | 56 | 0.444 | 0.461 | 105/151 = 0.695364 | 40.291 | 448 | 438 | 219 | 438 |
