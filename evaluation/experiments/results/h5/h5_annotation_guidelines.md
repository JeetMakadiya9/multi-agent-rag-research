# H5 Annotation Guidelines (version 1.0)

You will see pairs of a **CLAIM** and one **EVIDENCE** sentence. Decide, using only these two texts, which label
applies. Do not use outside knowledge to fill gaps, do not search for the source, and do not consult anyone else.

## Labels

* **SUPPORT** — the evidence sentence *alone* entails the *entire* claim, including its important qualifiers
  (direction of effect, population, condition, quantity, time, causality).
* **CONTRADICT** — the evidence sentence *alone* makes the claim false or incompatible with what it reports.
* **NEUTRAL** — the evidence neither entails the entire claim nor contradicts it.

## Rules

1. Partial support = **NEUTRAL**.
2. Topic similarity is **not** support.
3. Mentioning the same entities is **not** enough for support.
4. A related mechanism is **not** enough for support.
5. Correlation does not entail causation: a correlational finding does not SUPPORT a causal claim.
6. Association does not entail causation.
7. Numerical mismatch = **NEUTRAL** unless the evidence explicitly makes the claim's number false (then CONTRADICT).
8. Date/time mismatch must be evaluated explicitly: SUPPORT only if the evidence covers the claim's time; CONTRADICT
   only if it explicitly excludes it; otherwise NEUTRAL.
9. Entity mismatch (different species, population, drug, protein, dataset…) = **NEUTRAL** unless the evidence
   directly contradicts the claim.
10. If the evidence supports only one clause of a compound claim, the label is **NEUTRAL**.
11. The evidence must support the **whole** claim for SUPPORT.
12. Hedged evidence ("may", "suggests") supports a claim only if the claim is equally hedged; otherwise NEUTRAL.
13. Negation: check polarity explicitly. Evidence reporting "no effect" CONTRADICTS a claim of an effect only if it
    concerns the same entities, outcome and setting.
14. Background or introductory statements that do not report the claim's finding are NEUTRAL.
15. If you are unsure between SUPPORT and NEUTRAL, or CONTRADICT and NEUTRAL, choose NEUTRAL and explain in the note.

The following are never available to you and must never be used: any system or model prediction, the source
dataset's labels, why a pair was selected, the document title or identity, or another annotator's label.

## What to write

For every row set `gold_label` to exactly one of `SUPPORT`, `CONTRADICT`, `NEUTRAL`. Use `annotator_note` for
uncertainty, a short reason for CONTRADICT, or `NON_ATOMIC` if the claim appears to contain more than one
independently verifiable proposition (the note never changes the label). Do not edit `pair_id`, `claim_text` or
`evidence_text`. Work alone; your file is compared with another annotator's only after both are complete.

## Worked illustrations (synthetic, not from the evaluation data)

| Claim | Evidence | Label | Why |
|---|---|---|---|
| Drug X reduces mortality in adults. | Drug X reduced all-cause mortality in an adult cohort. | SUPPORT | whole claim entailed |
| Drug X reduces mortality in adults. | Drug X was studied in adults with heart failure. | NEUTRAL | topic only |
| Drug X reduces mortality in adults. | Drug X had no effect on mortality in adults. | CONTRADICT | same entities/outcome, opposite finding |
| Drug X causes weight loss. | Drug X use was associated with lower body weight. | NEUTRAL | association ≠ causation |
| Method X reaches 92% accuracy. | Method X reaches 87% accuracy on the same benchmark. | CONTRADICT | explicit incompatible value |
| Method X improves accuracy on dataset A. | Method X improves accuracy on dataset B. | NEUTRAL | entity mismatch |

## Adjudicator instructions

You see the claim, the evidence and two anonymous labels (`label_1`, `label_2`, order randomised). Apply the same
rules and choose the final label with a short rationale. You do not see model outputs, dataset labels or why the
pair was sampled.

## Revision log

* v1.0 — initial frozen version. At most one revision (v1.1) may follow the pilot; every change will be listed here.
