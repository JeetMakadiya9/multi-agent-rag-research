# Research Studio verifier evaluation — protocol (frozen before any run)

**Status: NOT RUN.** No evaluation of the Studio verifier exists yet. Running one needs a claim set labelled by qualified
human annotators, and no such set is available yet. This protocol, the harness (`run_eval.py`) and the scorer
(`score.py`) were written and frozen on 2026-10-24, before any labels or results existed. Nothing in this folder is a
result.

## 1. Estimand

**Claim-verification accuracy of the Research Studio's verify step, conditional on a fixed evidence packet.**

* **The unit** is one claim together with its evidence packet: at most 8 passages, fixed before evaluation.
* **What is measured** is the verdict the Studio returns for that claim when its retrieval is restricted to that packet.
* **Retrieval and search quality are outside the estimand.** That includes web search, source fetching and source
  coverage. The Arm C planner may still search, but only within the packet; that is the Studio's own behaviour on a
  run's evidence.
* **End-to-end research performance is a different estimand.** It is not measured here and must not be inferred from
  this result.

**System under test:** the Studio's configured default.
* It is built exactly as `webrag.server` builds it: `webrag.model_service.studio_options()` → `webrag.server.studio_orchestrator()`.
* Defaults: the verifier is `arm_c_studio`, the injection screen is `exclude`, and quote attestation is on.
* Evidence goes through the guarded evidence store and `Orchestrator._verify_mode`. That is the same code a Studio
  claim check runs; only the retrieval step is restricted to the packet.
* The harness records the verifier name and class it actually ran. A record whose verifier differs from the declared
  system is an error, not a result.

## 2. Population, eligibility and labels

* **Target population.** English, checkable factual claims of the kind a Studio user verifies: scholarly, scientific,
  technical or encyclopedic statements.
* **Each claim comes with an evidence packet.** The packet holds 1 to 8 passages from openly accessible sources, each
  with its URL.
* **Eligible claims:**
  * a single verifiable proposition (compound claims are split before sampling);
  * at most 400 characters;
  * evidence available in English.

  Opinions, predictions and claims about the evaluation itself are excluded.
* **Label ontology** (the reference label is judged against the packet only, not world knowledge):
  * `SUPPORTED`: the packet states the claim.
  * `CONTRADICTED`: the packet states something incompatible with the claim.
  * `NOT_ENOUGH_INFO`: neither; this includes packets that only touch on the claim.
* **Generalisation limit.** Results apply only to claims and packets drawn like this set. They do not apply to other
  languages, other domains, or retrieval-dependent performance.

## 3. Independence and provenance

* **Claim source.** Claims and packets must come from a lawful source with documented provenance: the source URLs,
  their licence or terms, and who wrote each claim and when.
* **No reused claims.** No claim may come from a set this project has used for development or evaluation. That
  excludes the SciFact claims, the WebRAG claim sets, the Studio demonstrations and the agentic verification study.
* **Overlap check.** `run_eval.py --check-overlap` compares normalised claim text against the project's SciFact claim
  files. Any match is excluded and reported.
* **No tuning on this set.** No prompt, threshold or setting of the system under test may be changed after the
  claims are seen. If that becomes necessary, this set stops being a test set; the change is documented and a new set
  is collected.
* **Data stays private.** Evaluation data (claims, packets, labels, records) is not committed to the public
  repository; only aggregate results may be published.

## 4. Annotation

1. **Two annotators label independently.** Each labels every claim against its packet, without seeing any model
   output. Annotators must be able to read the claim's domain.
2. **Agreement is reported before adjudication:** Cohen's kappa over the three labels, plus the raw agreement rate.
3. **Disagreements are adjudicated** by a third person who sees both labels and the packet but no model output. The
   adjudicator may mark a claim `AMBIGUOUS`; such claims are excluded and their count is reported.
4. **The final labels are frozen before any run.** Their SHA-256 is written to `FREEZE.json` (template:
   `FREEZE.example.json`). `run_eval.py` refuses to run when that file is missing, when the hash does not match, or
   when the freeze does not declare the annotation independent of model output.

## 5. Statistical analysis plan

* **Primary endpoint: macro-F1 over the three classes** (SUPPORTED, CONTRADICTED, NOT_ENOUGH_INFO), computed over
  every eligible claim.
  * The Studio's `INSUFFICIENT` counts as `NOT_ENOUGH_INFO`.
  * The fixed denominator includes every failure. SERVICE_ERROR, ASSESSMENT_FAILED, a malformed result, a timeout or a
    missing record each count as `NOT_ENOUGH_INFO` for the primary endpoint and are reported separately.
  * The rationale for macro-F1: it weights the minority CONTRADICTED class equally, which matters because confirming a
    contradicted claim is the most harmful error.
* **Confidence interval:** a 95% percentile bootstrap over claims, with 10,000 resamples and seed 20261024.
* **Secondary measures:**
  * per-class precision and recall, and the confusion matrix;
  * accuracy;
  * **false support**: the share of gold CONTRADICTED or NOT_ENOUGH_INFO claims predicted SUPPORTED;
  * **missed contradiction**: the share of gold CONTRADICTED claims not predicted CONTRADICTED;
  * the INSUFFICIENT rate;
  * the operational-failure rate, by kind;
  * the share of `ADJUDICATED` verdicts and their accuracy;
  * the share of claims where the injection screen withheld a passage.
* **Prespecified subgroups** (descriptive only; no subgroup is used for a decision):
  * claims whose packet holds both supporting and contradicting passages (evidence conflict);
  * claims with a number in them;
  * claims whose packet holds an instruction-like passage.
* **Sample size: at least 300 eligible claims after exclusions, at least 100 per gold class.**
  * At a proportion near 0.7, a 95% interval of ±0.05 needs about 1.96² × 0.7 × 0.3 / 0.05² ≈ 323 claims.
  * At least 100 per class keeps each class's recall interval near ±0.1.
  * With fewer claims, the evaluation may still be run and reported, but it cannot support a promotion.

## 6. Comparator and promotion gate

* **Comparator.** A candidate verifier is compared with `arm_c_studio` on the same claims and packets, in the same
  session (a paired comparison).
* **Historical results are not comparators.** These include the SciFact agentic verification study and Phase 11B.
  They used other data, retrievers and conditions.
* **Paired statistics:**
  * Δ macro-F1 (candidate − `arm_c_studio`), with a paired bootstrap 95% interval (seed 20261025) and a paired
    sign-flip permutation p-value (10,000 permutations, seed 20261026);
  * exact McNemar tests on accuracy and on false support.

**A candidate replaces `arm_c_studio` as the default only if all of the following hold:**
1. The claim set meets Sections 2–4: independent, frozen and adequately sized (Section 5).
2. The lower bound of the paired 95% interval for Δ macro-F1 is above 0.
3. **No more false supports:** the candidate's false-support count is not higher than `arm_c_studio`'s on the same
   claims. Zero tolerance is used because more false support was this project's main documented harm (the historical
   Arm C study); it is not a tuned margin.
4. **No more operational failures:** the candidate's operational failures are not more numerous than
   `arm_c_studio`'s.
5. **Safeguards stay on:** the candidate runs with the injection screen and quote attestation enabled, through the
   Studio's orchestration path.
6. **Reproducible and tested:** its configuration fingerprint is recorded, and the full test suite passes.

If any criterion fails, or the evaluation is incomplete, `arm_c_studio` stays the default. A higher point estimate
alone never promotes a candidate.

**Multiplicity.** Only the primary endpoint and the gate above decide anything. Everything else is descriptive and
uncorrected.

## 7. Records and fingerprints

Every run records:
* the source commit and the configuration fingerprint (SHA-256 of the resolved options and the verifier's frozen
  settings);
* the model name and digest;
* the claim-set hash.

Per claim, it records:
* the verifier actually used;
* the verdict and sub-state;
* each passage's stance;
* the injection-screen decisions;
* the attestation summary;
* errors and timings.

Records stay private. They contain passage text, so they are kept under `outputs/`, which is not versioned.

## 8. Deviations

Any departure from this protocol is recorded here, with its date and reason, before the affected run.

*(none)*
