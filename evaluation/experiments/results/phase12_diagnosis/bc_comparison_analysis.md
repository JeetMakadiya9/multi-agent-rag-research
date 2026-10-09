# B/C Comparison

On the 176 valid labeled paired predictions: 84 both correct, 90 both wrong, 1 B-only correct, 1 C-only correct. Exact McNemar p=1.0.

Across 275 valid prediction pairs, labels agree on 271 and differ on 4. Top-8 document sets match on 275; selected document sets on 267; top-1 chunks on 275.

C adds the audited bounded Phase 8 workflow and associated control/provenance trace. The results show no measured paired classification separation in this run; they do not identify the causal reason for B/C similarity. Four disagreements are enumerated in the JSON, with gold labels and stored retrieval/evidence fields.
