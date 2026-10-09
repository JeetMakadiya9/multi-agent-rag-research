# Phase 12 Bottleneck Summary

1. **Post-retrieval evidence-to-verdict path (highest-priority diagnostic target; causal mechanism unproven)**
   - B: 78/92 (84.78%) wrong B labels occurred despite at least one annotated gold document in persisted retrieval output.
   - C: 77/91 (84.62%) for C.
   - Confidence: Moderate for prioritizing diagnostic investigation; low for causal attribution. A retrieved document may be irrelevant to the specific rationale or not selected/used.
   - Next diagnostic action: Audit/alter one evidence-to-component-to-verdict step in a B+ arm; retain frozen B.

2. **Verifier-cited evidence / passage overlap (not an independent input-selection measure)**
   - B: Exact gold rationale sentence overlap among B wrong predictions: 66/92 (71.74%). Selected evidence gold-document overlap among wrong: 17/92.
   - C: C exact rationale sentence overlap: 65/91 (71.43%). Selected evidence overlap: 17/91.
   - Confidence: Moderate/low; artifact fields do not fully identify exactly what spans were actually passed into every internal verifier step.
   - Next diagnostic action: Persist stable candidate, selected chunk and sentence IDs plus verifier input/output mappings in future controlled run.

3. **Retrieval miss / ranked evidence**
   - B: 14/92 (15.22%) wrong predictions had no gold document among valid persisted retrieved chunks.
   - C: 14/91 (15.38%).
   - Confidence: High for document overlap accounting; it explains only these observed rows under the chosen criterion.
   - Next diagnostic action: Do not lead with retriever tuning; separately evaluate miss subset and reranking only after evidence-to-verdict audit.


The largest observed diagnostic group is wrong classification despite gold-document overlap. This points the next diagnostic toward the post-retrieval evidence-to-verdict path, but does not prove a verifier reasoning cause. Gold passage/cited evidence overlap is not entailment, and selected IDs are verifier output rather than independent input selection. No intervention is implemented here.
