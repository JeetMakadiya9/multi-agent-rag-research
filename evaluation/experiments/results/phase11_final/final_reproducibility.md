# Reproducibility record

- Phase 10 full DEV run ID: 02fb9c7a1268476ca1f9b12d9dbfb25e
- Configuration fingerprint: 875ee8247def6be3f440e34863168e1313f1a4cea33d9272f517a58c96000185
- Git HEAD: 55a6abdb98625453b572503d4aab7390e59c51bd
- Dataset: SciFact-Orig / dev; counts {'claims_train.jsonl': 809, 'claims_dev.jsonl': 300, 'claims_test.jsonl': 300, 'corpus.jsonl': 5183}; chunk count 45972.
- Model: Ollama qwen3:4b; digest 359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7; temperature 0.0.
- Claims / attempts / scored: 300 / 900 / 900.
- Model, dataset and source hashes are in final_analysis_audit.json.
- Denominators: classification = valid labeled predictions; retrieval = valid retrieval output and annotated document references; MRR no-hit outputs contribute zero; evidence overlap = valid labeled predictions and gold/selected document IDs.
- Failure policy: stored failed/interrupted attempts remain visible; no failed attempt is credited as correct; no retries. Endpoint errors are classified as runtime infrastructure failures.
- Gold boundary: runtime input fields are ['claim', 'claim_id']; evaluation gold joins are in per-example records.
- Phase 11B made no inference, Ollama generation, search/API call or Phase 10 resume.
