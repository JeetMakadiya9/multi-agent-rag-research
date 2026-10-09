# Phase 11 — Retrieval Analysis

Recall@1 is the persisted **micro unique-gold-document overlap**: number of gold evidence document IDs retrieved at rank 1 divided by the total number of gold evidence document IDs for claims with valid scored retrieval output. This is a document-ID overlap measure.

| System | Gold-document hits | Gold-document denominator | Recall@1 | Valid retrieval evaluation rows | MRR (claim-level) |
| --- | --- | --- | --- | --- | --- |
| A | 106 | 152 | 0.697 | 136 | 0.838 |
| B | 106 | 151 | 0.702 | 135 | 0.840 |
| C | 105 | 151 | 0.695 | 135 | 0.837 |

## Interpretation boundary

- Retrieval overlap means a retrieved document ID matches an annotated evidence document ID.
- It does not establish that the retrieved passage entails or contradicts the claim.
- Entailment/evidence assessment is a separate verifier operation.
- A verifier label is distinct from final factual correctness.
- Unavailable and failed retrieval outputs are excluded from retrieval scoring, not recorded as misses.

Per-system valid scored retrieval row counts and MRR denominators are in `phase11_retrieval_analysis.json`. Recall denominators vary by system (A 152; B/C 151 gold documents) because the valid scored evidence sets differ. No extra retrieval metric was inferred beyond persisted Recall@1/MRR fields.
