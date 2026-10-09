# Phase 12 — Complete Error Diagnosis & Bottleneck Analysis

## 1. Scope

Post-hoc deterministic analysis of the completed Phase 10F SciFact DEV evaluation. This report uses persisted Phase 10/11 records and the unchanged DEV claim annotations. No model inference, Ollama, external API, search, resume, or benchmark rerun occurred. No system was modified or tuned.

Scope: 300 DEV claims; 900 system/claim attempts across A/B/C. This is development-set evidence only and does not generalize to SciFact test or establish a winner/hypothesis result.

## 2. Source Artifacts

Primary raw source: `evaluation/experiments/results/phase10_full_dev/` (per-example rows, attempt records, manifest, metrics, failure accounting, C trace audit). Authoritative inherited analysis: `evaluation/experiments/results/phase11_final/`. Phase 11A source package: `evaluation/experiments/results/phase11_results/`. Gold claim text and rationale sentence IDs were joined from `data/scifact/data/claims_dev.jsonl` only in this post-hoc process.

The per-example raw rows, Phase 11B summary metrics, paired analysis and final error analysis were inventoried with SHA-256 in `phase12_audit.json` and `phase12_diagnostic_report.json`.

## 3. Phase 10C Reconstruction

The Phase 10F full-dev raw run reconstructs to 300 claims × three systems = 900 per-example records and 900 attempt records. Statuses: 840 SUCCESS, 55 FAILED, 5 INTERRUPTED. Labeled claims: 188 ({'UNLABELED': 112, 'SUPPORTED': 124, 'CONTRADICTED': 64}); unlabeled: 112. Claims with rationale evidence: 188. Reconstructed valid labeled predictions: B 179, C 176; paired valid labeled claims 176.

The 900 planned attempts are all present. Per-claim A/B/C identity is complete. Failed/interrupted executions remain separate from invalid/missing prediction states.

## 4. Artifact Reconciliation

The raw attempt/per-example counts and status counts agree with the completed Phase 10F / Phase 11B accounting. The independently reconstructed classification counts are B 87/179 correct and C 85/176; Phase 11B reports the same. The 176 valid labeled pairs have 84 both correct, 90 both incorrect, 1 B-only correct, and 1 C-only correct. Discrepancy log: [].

One inherited interpretation issue remains documented by Phase 11B: MRR can be shown hit-conditional or with misses zero. This report cites the all-valid-output/misses-zero form. Top-10/top-20 are NOT COMPUTED because at most eight ranked chunks are persisted.

## 5. Failure Summary

Raw status counts: {'SUCCESS': 840, 'INTERRUPTED': 5, 'FAILED': 55}. Failures and interruptions total 60; they are operational execution outcomes, not scientific classification errors.

Failure categories use Phase 11B persisted primary-category fields joined by claim/system; exact raw errors and stages are preserved from Phase 10.
{
  "by_system": {
    "A": {
      "local_model_endpoint_failure": 15,
      "interruption": 1
    },
    "B": {
      "verification_failure": 4,
      "interruption": 2,
      "local_model_endpoint_failure": 14
    },
    "C": {
      "interruption": 2,
      "controller_failure": 22
    }
  },
  "by_category": {
    "interruption": 5,
    "verification_failure": 4,
    "controller_failure": 22,
    "local_model_endpoint_failure": 29
  }
}

Exact failure rows, claim IDs, system, recorded error, stage and counters are in `failure_analysis.json` and `failure_records.csv`. The rule classifier does not infer unrecorded inference/retrieval/verification stages. An absent field is left absent/unknown. The persisted failures include infrastructure endpoint failures, C controller/execution failures, B verification failures and interruptions; no retry or correction was performed here.

## 6. Claim-Level Outcome Matrix

`error_taxonomy.csv` and the per-claim record section of `phase12_diagnostic_report.json` provide the claim-level diagnostic table for all 300 claims, with A classification marked N/A. Each B/C status, valid prediction, correctness only when labeled, gold-document/passage overlap, selected-document overlap and latency is separately recorded. Unavailable retrieval outputs are never called misses.

## 7. Classification Error Analysis

Among labeled claims with successful execution and a valid verifier prediction:

- B: 87/179 (48.6%) accuracy; macro-F1 0.4842; labeled=188, failed labeled excluded=9, invalid labeled excluded=0.
- C: 85/176 (48.3%) accuracy; macro-F1 0.4834; labeled=188, failed labeled excluded=12, invalid labeled excluded=0.
- A classification: NOT APPLICABLE — A has no verifier-derived label.

For B the confusion matrix contains 37 SUPPORTED→INSUFFICIENT_EVIDENCE and 43 CONTRADICTED→INSUFFICIENT_EVIDENCE among its 92 incorrect valid predictions; 12 CONTRADICTED→SUPPORTED account for the other errors. For C, 37 SUPPORTED→INSUFFICIENT_EVIDENCE and 42 CONTRADICTED→INSUFFICIENT_EVIDENCE among 91 errors; 12 CONTRADICTED→SUPPORTED. These are observable label transitions, not proof that insufficient evidence was the internal reasoning failure. Failed labeled attempts excluded: B 9, C 12. Invalid labeled outputs excluded: B 0, C 0.

## 8. B/C Agreement Analysis

There are 176 paired valid labeled predictions: 84 both correct, 90 both incorrect, 1 B-only correct, 1 C-only correct. Phase 11B exact McNemar two-sided p=1.0 (descriptive paired outcome; no superiority conclusion).

Across 275 claims where both produced valid labels, final labels agree on 271 and disagree on 4. Persisted top-8 document sets are identical for 275/275 pairs; selected document sets match for 267; top-1 chunk matches for 275. The four all-claim disagreements are preserved in `bc_comparison_analysis.json`; paired labeled disagreements are separated from unlabeled disagreements.

What C adds: real bounded controller execution and traceable 8A validation → 8B execution → 7B dispatch → Phase 6 retrieval/verification → explicit continuation → completion. The persisted C audit has 276/276 successful C records passing the audit. It adds workflow/control/provenance and cycles, but this run’s paired correctness cells and similar retrieval/final labels show little measured classification difference. The artifacts do not establish that architectural complexity caused the similarity.

## 9. Retrieval Error Analysis

Phase 11B retrieval metrics (annotated gold-document overlap, not correctness):

- A: Recall@1 146/209 (69.9%); Recall@5 176/209 (84.2%); all-valid-output MRR 0.8305.
- B: Recall@1 146/208 (70.2%); Recall@5 175/208 (84.1%); all-valid-output MRR 0.8322.
- C: Recall@1 144/205 (70.2%); Recall@5 173/205 (84.4%); all-valid-output MRR 0.8286.

Among valid labeled predictions with gold evidence and valid retrieval output, the document-overlap × classification table is:

| system | eligible n | gold doc hit + correct | gold doc hit + wrong | gold doc miss + correct | gold doc miss + wrong |
|---|---|---|---|---|---|
| B | N/A | N/A | N/A | N/A | N/A |
| C | N/A | N/A | N/A | N/A | N/A |


For incorrect outputs, document-overlap misses are B 14/92 (15.22%) and C 14/91 (15.38%). Gold document present but prediction wrong occurs B 78/92 (84.78%) and C 77/91 (84.62%). This is a post-hoc diagnostic upper-bound proxy, NOT a perfect-retrieval experiment or a counterfactual: adding/altering retrieved evidence could change evidence selection and verifier behavior.

Exact sentence-index rationale overlap among wrong outputs is separately computed: B 66/92; C 65/91. Passage overlap remains a retrieval relevance proxy and does not prove entailment or model use.

## 10. Evidence Analysis

The Phase 10B adapters derive `selected_evidence_doc_ids` from verifier supporting/contradicting evidence references mapped to document IDs. Thus this measures cited verifier output, not independent input selection. The persisted `selected_evidence_doc_ids` overlap gold documents for B 17 of its wrong outputs and C 17 of its wrong outputs. Selected document overlap is not the same as selected annotated rationale passage, sufficient context, entailment, or factual support. Sentence-level selected-evidence provenance is not uniformly represented with compatible identifiers across B/C; where exact selection-to-rationale mapping is not persisted, that submetric is NOT COMPUTED. Full per-claim retrieval and selected ID observations are in `retrieval_vs_verification_analysis.csv`.

## 11. System C Execution Analysis

Successful C executions audited: 276; passed: 276; limited: 0. Persisted stage PASS counts: {"phase_8a_validation": 276, "phase_8b_execution": 276, "phase_7b_dispatch": 276, "phase_6_retrieval": 276, "phase_6_verification": 276, "continuation_authorization": 276, "researcher_transition": 276, "completion": 276}. Successful C controller cycle summary (Phase 11B): {"max": 2.0, "mean": 2.0, "median": 2.0, "min": 2.0, "n": 276, "sample_sd": 0.0, "sum": 552.0}. Failed-C top-level vs embedded cycle discrepancies: 22; both stored representations are preserved and failed cycle counts are not interpreted as completed cycles. Failures and labels are provided in `failure_analysis.json`, `bc_comparison_analysis.json`, and `phase12_diagnostic_report.json`. Where traces do not persist enough to explain a verdict, the proper conclusion is “Insufficient persisted trace information.”

## 12. Efficiency Analysis

Mean/median/min/max latency, LLM calls, retrieval calls, verification calls, and C controller cycles are split by system and execution status in `phase12_diagnostic_report.json`. Only persisted numeric values contribute; missing values are excluded from each statistic and its `n` is explicit. No cause is assigned to high-latency observations. C includes extra controller/retrieval work from its defined path, so call totals are architecture-specific rather than a direct quality measure.

## 13. Failure Impact on Metrics

Accuracy uses labeled claims with successful execution and a valid stance output. Failed executions, interruptions, and invalid/missing labels are excluded and separately counted; this is not a failure-as-miss convention. Retrieval Recall@1 denominators are Phase 11B gold-document denominator counts and exclude unavailable retrieval outputs; failed/unavailable counts are separately persisted. Valid `INSUFFICIENT_EVIDENCE` is a verifier class output and is counted as an incorrect prediction when gold is SUPPORT/CONTRADICTED. Unlabeled claims are excluded from correctness and are not treated as wrong. Thus B/C accuracy compares 179 vs 176 valid labeled outputs, not all 300 claims; its coverage differs and must accompany the accuracy.

## 14. Unlabeled Claim Analysis

There are 112 claims without a class label. By system, status, valid-prediction count/distribution, valid retrieval output count and latency are in `phase12_diagnostic_report.json`. Correctness is N/A — no gold label. Retrieval/evidence availability and B/C label agreement are measurable descriptively; they cannot be scored for correctness. An unlabeled claim is not an incorrect prediction.

## 15. Key Observations

1. The dominant *observed diagnostic location* is downstream of document retrieval: most incorrect B/C labels have some gold document among persisted top-8 chunks (78/92 B and 77/91 C). This does not prove verifier failure because document relevance is not entailment and selected evidence may differ.
2. Many errors are recorded as INSUFFICIENT_EVIDENCE labels on gold-labeled claims (B 80/92, C 79/91), showing a systematic output pattern, not a proven cause.
3. Exact gold rationale sentence overlap and selection overlap reduce the inference strength further. Reranking, evidence passage selection, and verifier interpretation cannot be causally separated from all persisted records.
4. Retrieval misses remain real and measurable, but constitute B 14/92 and C 14/91 wrong predictions among evaluable valid-label cases.
5. B/C share highly similar retrieval and classification outputs. C’s measured addition is a controlled multi-step execution path with provenance/cycle overhead, not a demonstrated classification gain.
6. The 4B model is not established as a bottleneck: no model-size counterfactual, randomized model comparison, or controlled capacity intervention exists. Do not blame Qwen3:4B from accuracy alone.
7. Claim text proxies are exploratory; semantic clause count, entities, entailment, and decomposition are NOT COMPUTED.

## 16. Limitations

- One completed DEV evaluation, with different valid accuracy denominators because execution failures differ.
- Annotated document/passage overlap is not evidence entailment or factual correctness.
- Persisted selected evidence may not encode every internal model input in a consistent way.
- Heuristic complexity variables are not validated linguistic annotations.
- Failure categories derive from recorded status/error text; unrecorded stages remain unknown.
- No causal intervention or independent test evaluation is included.

## 17. What Phase 12 Does NOT Establish

This phase does not establish that B or C is superior, that multi-agent orchestration improves verification, that retrieval is irrelevant, that the verifier/model is the proven cause, or that a hypothesis is confirmed/disproved. It does not establish generalization to test. It does not change the frozen Phase 10/11 baseline.

## 18. Recommended Next Experimental Step

First perform a blinded manual audit of stratified wrong and correct labeled examples to distinguish selection, evidence adequacy, insufficiency, contradiction, and claim-composition observations. Then, only after protocol freeze, test a single structured evidence-to-verdict intervention as B+ against frozen B with identical retrieval/model/runtime inputs and paired claims. The proposed 60-claim labeled pilot (seed 2026, proportional 39/21 class split) is exploratory; predefine IDs before execution. Keep the 188 labeled DEV claims for the later paired DEV evaluation, and characterize it as development evidence. A separately authorized test protocol is needed for generalization. Proposed prospective decision rule and guardrails are in `followup_experiment_design.json`; thresholds must be locked before B+ results are seen.

## Reproducibility and integrity

Run ID: `02fb9c7a1268476ca1f9b12d9dbfb25e`. Configuration fingerprint: `95184e04be10b963fd39433d1beb28e51e80667840f0f33bdc5dcc7d96b7f1fb`. Source hashes and before/after immutability checks are in `phase12_audit.json`. No production/system files changed, no raw Phase 10 or Phase 11 files changed, and no external/inference calls occurred.
