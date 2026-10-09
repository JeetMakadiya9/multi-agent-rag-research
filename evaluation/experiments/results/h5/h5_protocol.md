# H5 Protocol (frozen, version 1.0)

Status: FROZEN before any annotation, any model selection and any inference. Changes after this point are
limited to one guideline revision after the pilot (Section 9). No change may be made in response to final labels
or model outputs.

## 1. Research question

> Does a specialized local NLI model produce more accurate pair-level claim/evidence stance judgements than the
> current Qwen3:4b generative judge?

H5 tests whether pair-level stance quality differs between the incumbent generative judge and a specialized NLI
model. All four outcomes — NLI better, Qwen better, no meaningful difference, inconclusive — are valid results.

Scope: evidence sentence + claim → pair-level stance. H5 does **not** test retrieval, query planning, evidence
selection, claim decomposition, aggregation, orchestration, research planning or source discovery.

Motivation (hypothesis, not established fact): Slice 1 found that, with recorded judgements held fixed, no
aggregation rule separated correct from incorrect support decisions, which is consistent with pair-level stance
quality being the dominant unresolved bottleneck.

## 2. Experimental unit

One unit = **one corpus sentence (premise) + one single-component claim (hypothesis)**. No multi-sentence evidence,
titles, metadata, surrounding context, evidence bundles or multi-component claims.

## 3. Gold labels

`SUPPORT`, `CONTRADICT`, `NEUTRAL`, defined and governed by `h5_annotation_guidelines.md` (v1.0). Gold labels come
only from blinded human annotation (two independent annotators + third-person adjudication). Never from SciFact
labels, rationale status, Qwen, an NLI model or any other model.

## 4. Data source and sampling frame

* Source: SciFact **TRAIN** only. Excluded by construction and verified: the 69 Phase 13 dev claims, the 119
  reserved DEV claims, all other DEV claims (by text) and all TEST claims (by text).
* **TRAIN annotations are used for sampling-frame construction, not as pair-level gold labels.**
* Claim eligibility: `claim_atomicity_status = ATOMIC` under deterministic rule `h5-atomicity-v1` (no LLM):
  COMPOUND = frozen `verification_units` finds >1 sentence or >1 component; UNCERTAIN = a coordinator/connective
  (`and, or, but, whereas, while, although, as well as, both, either, neither, nor, versus, vs, respectively, ;`),
  a context-dependent start (`it, this, these, they, such, that, those, its, their`) or a non-restrictive relative
  clause (`, which/where/whereby`); ATOMIC otherwise. COMPOUND/UNCERTAIN claims are kept in
  `h5_excluded_candidates.jsonl`; duplicate claim texts keep the lowest TRAIN ID.
* Candidate strata (sampling signals only):
  * `support_oriented` — claim + a sentence in a SUPPORT rationale;
  * `contradiction_oriented` — claim + a sentence in a CONTRADICT rationale;
  * `neutral_oriented` — claim + a non-rationale sentence of the claim's own evidence documents (or of its cited
    documents when TRAIN has no evidence for the claim) — topically related, hence informative;
  * `neutral_unrelated_inventory` — one unrelated-document sentence per claim; inventoried, **not sampled**
    (trivially unrelated pairs carry little information about stance discrimination);
  * `conflicting_rationale_excluded` — sentences carrying both labels; not sampled.
* Per-claim de-duplication of identical evidence text; empty sentences excluded.

## 5. Samples

* Seed `h5-2026-v1`; rank = SHA-256(`seed|pair_id`) ascending; greedy fill per stratum quota; ≤ 2 pairs per claim.
* **Pilot**: 60 pairs (20 per stratum), drawn first.
* **Main (confirmatory)**: 300 pairs (100 per stratum), drawn second, excluding every pilot claim.
* 300 is a practical pre-registered target, not a power guarantee; results are reported with confidence intervals.
* Statistical unit is the pair; pairs share claims, so claim_id is preserved and all resampling is by claim.

## 6. Blinding

Annotators see only `pair_id`, claim text, evidence sentence, and fill `gold_label` + `annotator_note`.
Annotator A and B files have independent hash-based orderings. Hidden from annotators and adjudicator: system
predictions, model identity, SciFact labels and rationale status, sampling stratum, document title/ID, retrieval
rank/score, other annotators' labels (adjudicator sees two anonymised labels only).

## 7. Atomicity review (human)

`h5_atomicity_review.jsonl` lists every sampled claim (claim text only) for one human reviewer to mark
`ATOMIC | COMPOUND | UNCERTAIN`. The sample is **not** changed by this review. Pre-registered use: the primary
analysis uses all 300 pairs; a sensitivity analysis excludes pairs whose claim a human marked non-ATOMIC.

## 8. Annotation and adjudication

Two independent annotators label all pairs. Disagreements go to a third human adjudicator who sees claim, evidence
and the two labels in randomised anonymous order (`label_1`, `label_2`) and chooses a final label with a note.
Final gold = agreed label, else adjudicated label. Raw labels are never overwritten. Recorded per pair:
`annotation_status, annotator_a_label, annotator_b_label, adjudication_required, adjudicated_label,
adjudication_note, gold_label`. Tooling: `evaluation/experiments/h5/agreement.py`.

## 9. Pilot

The 60 pilot pairs are labelled by both annotators first. Report raw agreement, Cohen's kappa, per-class
agreement and the confusion matrix. Rules are not tuned to maximise kappa. At most **one** guideline revision is
allowed; every change is recorded and the revised guideline frozen as v1.1. The pilot is calibration only and is
never part of the confirmatory analysis.

## 10. Agreement and gold quality

Computed before adjudication on the main sample: raw agreement, Cohen's kappa, per-class agreement, confusion
matrix. Pre-registered quality flag (fixed now, before any label exists): `GOLD_QUALITY_WARNING` if Cohen's kappa
< 0.60 or any class's specific agreement < 0.50. A warning is reported with the disagreement structure; the gold set
is not silently treated as reliable.

## 11. Systems (comparison frozen, not run)

* **System Q**: Qwen3:4b via Ollama, Experiment B pairwise prompt v1 (system SHA-256 `aa49d5e7…`), temperature 0,
  `think=false`, `format=json`, `num_predict=256`, `num_gpu=0`, `max_judge_retries=1`, the existing project
  configuration. Input: `question_context = claim`, unit/component = claim, `selected_evidence.text` = the evidence
  sentence only (no title — this deliberately differs from the Phase 10/13 runtime presentation, which prefixed the
  title). Record model digest, Ollama version, prompt hash and generation configuration.
* **System N**: one local NLI model satisfying `h5_nli_model_requirements.md`, premise = evidence sentence,
  hypothesis = claim; entailment→SUPPORT, contradiction→CONTRADICT, neutral→NEUTRAL (argmax, no thresholds).
  Model selection is a separate gated step with an audited model card.

## 12. Endpoints

* **Primary**: pair-level macro-F1 over SUPPORT, CONTRADICT, NEUTRAL against adjudicated gold on all 300 eligible
  pairs; an invalid/error output counts as incorrect (never removed from the denominator).
* Secondary: per-class precision/recall/F1, accuracy, confusion matrix, invalid/error rate, abstention rate (if a
  system can abstain), latency, model calls, throughput; and valid-prediction-conditional metrics (reported in
  addition, never instead).

## 13. Statistical analysis

* Primary: Δ macro-F1 = NLI − Qwen; paired **claim-clustered bootstrap**, 10,000 resamples, 95% percentile CI,
  resampling unit `claim_id`, fixed bootstrap seed `h5-2026-v1-bootstrap`.
* Secondary: paired accuracy difference, per-class F1 differences, confusion matrices, exact McNemar on paired
  binary correctness (pair-level; reported with the clustering caveat).
* No significance claim beyond this pre-specified analysis; no post-hoc minimum effect size. The question is
  whether the observed difference is positive and how uncertain it is.
* Pre-registered sensitivity analyses: (a) excluding pairs whose claim a human marked non-ATOMIC; (b) restricted to
  pairs on which both annotators agreed before adjudication (agreement-only subset).

## 14. Contamination control

Before NLI inference, verify from the model card/source that the model was not trained or fine-tuned on SciFact
(any split), the H5 gold set, or the evaluation pairs. Unclear provenance → contamination status `UNKNOWN`, the
experiment is not called clean, and confirmatory inference does not proceed until resolved. General NLI training
(e.g. SNLI/MNLI/ANLI) is permitted and disclosed. Qwen3:4b pretraining data is undisclosed; this is reported.

## 15. Integrity

Hashes of source files, candidate pool, manifests, annotation package, guidelines and this protocol are recorded in
`h5_sampling_manifest.json` and `h5_integrity_report.md`; all generated files have deterministic ordering and no
timestamps. Model configuration hashes are added at the later model-selection gate.

## 16. Gate sequence

1. READY_FOR_H5_ANNOTATION (this package) → 2. pilot annotation + agreement → (optional single guideline
revision) → 3. human atomicity review → 4. main annotation + agreement + adjudication → gold frozen and hashed →
5. NLI model-card audit and selection (download only after approval) → 6. inference for Q and N on the frozen gold
set → 7. pre-registered analysis.

## 17. Stop conditions

Stop rather than improvise if: eligible pairs or any stratum fall below quota; claim/evidence mapping is
ambiguous; reserved DEV or TEST exclusion cannot be verified; sampling is not deterministic; blinding cannot be
guaranteed; source hashes differ; frozen artifacts would change; inference or network access would be needed
before its gate.
