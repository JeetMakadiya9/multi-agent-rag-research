# H5 — NLI Model Requirements (selection is a separate gated step; nothing downloaded)

A candidate NLI model is eligible only if every requirement is verified from its model card or source and recorded
in a model-selection audit before any download:

1. Runs locally (no API).
2. CPU-feasible on the project machine (i5-12450H, 8 cores, 15.6 GB RAM; venv torch is CPU-only).
3. Reproducible pinned revision (commit hash) and recorded weight-file hashes.
4. Native 3-way NLI output (entailment / neutral / contradiction).
5. Maximum sequence length ≥ 512 tokens preferred (H5 premises are single sentences).
6. Explicit, compatible license.
7. Documented training data.
8. No SciFact TRAIN/DEV/TEST contamination (no training or fine-tuning on SciFact or SciFact-derived data).
9. No fine-tuning on the H5 gold set or evaluation pairs.
10. Model card available.
11. Deterministic inference configuration (eval mode, no sampling, fixed max length/truncation, recorded).
12. Feasible memory footprint for CPU inference.

Mapping: entailment→SUPPORT, contradiction→CONTRADICT, neutral→NEUTRAL, by argmax; no thresholds; premise = evidence
sentence, hypothesis = claim. If training-data provenance is unclear, contamination status is `UNKNOWN` and
confirmatory inference does not proceed. Installing any new package (e.g. a tokenizer dependency) requires
separate approval.
