# Phase 11 — Reproducibility

- Project: `D:\Projects\MultiAgent_RAG`
- Current Git HEAD: `55a6abdb98625453b572503d4aab7390e59c51bd`
- Commit recorded in `authoritative_provenance.json`: `4e1f343a4d2e87c04393e9b5a473446b6e825231`
- Phase 10G baseline commit: `8e996b8db6c9a157bbb1f58009d34c148bc4782c`
- Phase 10 full-DEV run ID: `02fb9c7a1268476ca1f9b12d9dbfb25e`
- Run manifest status: `RUNNING` (retained as persisted; Phase 10 was intentionally closed early and is treated as partial)
- Configuration fingerprint: `875ee8247def6be3f440e34863168e1313f1a4cea33d9272f517a58c96000185`
- Selection: 300 claims, `all-dev-file-order-v1`, seed `0`
- Dataset split: SciFact DEV; corpus 5,183 documents; train/dev/test file counts 809/300/300; corpus chunks 45,972.
- Dataset hash identity: `{"claims_dev.jsonl": "86F0435D08FDB65D1AA41D1472684F57E6E71930626497BDF4D7A9EC1A632217", "claims_test.jsonl": "558930D75215C73F84A28FE538307D6D397C9DE1EC7239514CF45F80D75D2CA3", "claims_train.jsonl": "F4C8FA82D8BD0653A9CC8D61A6EA48C25EACEA64E90AF5DBF390EBB1B74372F0", "corpus.jsonl": "B8D6C89624CB2ED74DEE8938EFFC4F5D8BD2086887880AF8110D64BE4CEADE62"}`
- Dataset hashes match current files: **True**.
- Model: Ollama `qwen3:4b`, digest `359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7` (`captured`). The model identity is persisted; this analysis did not query the endpoint.
- Generation settings: `{"base_url": "http://127.0.0.1:11434/api/chat", "format": "json", "num_gpu": 0, "num_predict": 16, "stream": false, "temperature": 0.0, "think": false, "timeout_seconds": 1.0}`
- Embedding: `all-MiniLM-L6-v2`; snapshot `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`; cache SHA-256 `623829baf5b38a36fdd359c1e3322dcfb71a2d9f37b6a14c4ec5629251d3247c`; shape `[45972, 384]`.
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`; snapshot `233902d25c440f23af6f7d6e94d2946bac0bee0a`.
- Retrieval: hybrid `True`, reranking `True`, parameters `{"BM25_TOP_K": 20, "CHILD_CHUNK_OVERLAP": 150, "CHILD_CHUNK_SIZE": 900, "FINAL_TOP_K": 8, "RERANK_TOP_K": 30, "RRF_K": 60, "VECTOR_TOP_K": 20}`, verifier evidence limit `2`.
- Verification: `{"answer_revision": false, "evidence_limit": 2, "implementation": "frozen Experiment B", "targeted_retrieval_retries": 0}`.
- Runtime: `{"platform": "Windows-11-10.0.26200-SP0", "python": "3.12.14 (main, Aug 25 2026, 14:01:42) [MSC v.1944 64 bit (AMD64)]", "python_implementation": "CPython", "random_seed": 0}`.
- Attempts: 900 planned; 662 durable (606 success, 52 failed, 4 interrupted); 238 missing durable rows including one inflight marker.
- Frozen hashes match the provenance expectations: **True**. Locked evaluation source hashes match current source: **True**.

## Artifact preservation

Phase 10 raw JSONL SHA-256 values before Phase 11 package writing:

| Artifact | SHA-256 |
| --- | --- |
| attempt_records.jsonl | CBCD6B3955322033E842062F85EA822ED2338EC54AFC4A0512C0F72B9FF81122 |
| per_example.jsonl | 19A0B23BC52EAF6C80EDC69EB9F6B6661290DF03CBEECE9833598660DF87E88A |

The final audit records the post-write comparison. No Phase 10 run data, SciFact file, or source implementation was intentionally edited by Phase 11.
