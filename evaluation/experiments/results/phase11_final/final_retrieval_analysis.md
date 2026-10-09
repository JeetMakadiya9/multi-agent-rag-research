# Retrieval analysis

Metrics compare retrieved SciFact document IDs with annotated gold evidence document IDs. They measure document overlap, not entailment, evidence sufficiency, claim verification or answer correctness. R@k denominators count annotated gold documents in claims with valid retrieval output. The persisted ranked list has at most eight chunks per claim; therefore Recall@10 and Recall@20 are NOT COMPUTED. MRR uses persisted reciprocal rank and gives zero to misses.

| System | R@1 | R@5 | MRR (misses=0) | Valid claims |
|---|---:|---:|---:|---:|
| A | 146/209 (69.856%) | 84.211% | 0.8305 | 188 |
| B | 146/208 (70.192%) | 84.135% | 0.8322 | 187 |
| C | 144/205 (70.244%) | 84.390% | 0.8286 | 186 |

Inherited Phase 10 MRR is hit-conditional (denominators 170/169/168). This report also gives all-valid-output MRR, counting misses as zero; both are retained in JSON.
