# Authoritative project provenance

This record establishes a forward-looking, reproducible provenance baseline for the Multi-Agent RAG research project. Historical provenance before the recorded Git state is not authoritative. The earlier `evaluation/scifact/scifact_metrics.py` path is intentionally not reconstructed; the canonical current implementation is `evaluation/scifact/metrics.py`.

The JSON record captures the verified source state at commit `4e1f343a4d2e87c04393e9b5a473446b6e825231` on `main`, based on the existing project baseline `8e996b8db6c9a157bbb1f58009d34c148bc4782c`. **Authoritative provenance begins from this Git commit onward.**

## What future evaluations should record

For every major evaluation, preserve a machine-readable manifest that records:

- the exact Git commit and branch used;
- dataset identity, split, file hashes, and record counts;
- model provider, model name, and provider-reported digest;
- a deterministic configuration fingerprint, selection rule, and random seed;
- hashes for per-example results and aggregate artifacts;
- start/end timestamps, runtime and environment information;
- execution failures, unavailable outputs, and metric denominators without converting missing results into successes or misses.

Do not overwrite prior run artifacts. Store each evaluation in a new run directory and retain the manifest with its results.

## Current verified identity

- Dataset: SciFact-Orig; train 809, DEV 300, TEST 300, corpus 5,183 documents.
- Recorded retrieval index: 45,972 chunks (existing Phase 10F identity record; not rebuilt for this provenance baseline).
- Model: `qwen3:4b`, digest `359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7`. Its local Ollama identity was checked using read-only tags/show metadata; no generation request was sent.
- Canonical metrics source: `evaluation/scifact/metrics.py`, SHA-256 `8BE4B90BB28090FC48F874AF399798B0F7E39C25AAD7A1FA17B864A002E55587`.

See `authoritative_provenance.json` for file-level hashes, runtime, configuration fingerprints, and repository identifiers.
