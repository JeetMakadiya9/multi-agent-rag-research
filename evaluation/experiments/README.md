# Phase 10 evaluation protocol and smoke run

`evaluator.run_experiment` pairs systems A, B, and C over the same ordered
SciFact split and writes a manifest, one JSONL record per claim, and aggregate
JSON. Callbacks receive the runtime `{claim_id, claim}` object plus a shared
corpus-only retrieval provider. Scoring annotations are joined after inference.
Existing SciFact adapters, metrics, and production RAG/verifier modules are
used without modifying them.

## Conditions

- **A — Strong RAG:** `system_a_rag.run_system_a`; existing hybrid RAG and
  reranking defaults, followed by the existing grounded answer prompt.
- **B — Strong RAG + claim verification:** `system_b_verified_rag.run_system_b`;
  same retriever and initial context, then frozen Experiment B verification.
  Claim-specific extra retrieval is disabled to isolate verification.
- **C — Bounded multi-agent:** `system_c_multi_agent.run_system_c` uses the
  existing 8C controller twice: an approved retrieval action followed by an
  explicitly continued claim-verification action. Existing 8A validation, 8B
  proposal-scoped approval, Phase 7B dispatch, Phase 6 retrieval/verification,
  and provenance recording remain authoritative. The smoke uses a deterministic
  8A action binding; the controller and validation/execution path are real.

Run the fixed three-example dev smoke with:

```powershell
python -m evaluation.experiments.run_phase10_smoke
```

The runner selects dev claim IDs `1`, `3`, and `42` (empty annotations,
unanimous SUPPORT, unanimous CONTRADICT), builds one provider from the complete
SciFact corpus, and passes that provider to all systems. A and B each retrieve
once. C performs a counted provenance preflight plus its controller-dispatched
retrieval and verification. Local Qwen inference is CPU-pinned with temperature
0; B and C use Ollama's JSON response mode to satisfy the frozen pairwise
verification output contract. Verification sees the first two retrieved items
in this bounded smoke; B and C use the same configured evidence limit. The
baseline generation uses the existing baseline prompt.

The runner writes `evaluation/experiments/results/phase10_smoke.json` plus a
manifest, per-example JSONL, and aggregate JSON in its run directory. A SHA-256
identity over the local embedding snapshot and ordered corpus keys caches the
full embedding array under the OS temporary directory; FAISS and BM25 indexes
are rebuilt from the cached array when rerunning. It refuses to overwrite
existing artifacts; move/archive a prior run before rerunning.

The general API remains available via an explicitly constructed
`ExperimentConfig` and an `ExistingRAGProvider`. Test split evaluation is
supported for all 300 rows;
the public test annotations are unavailable, so label/attribution accuracy
cannot be scored there. Dev rationale labels are mapped to a claim label only
when all annotated rationales agree. Empty and conflicting annotations remain
unscored; empty evidence is not automatically called NEI.

The deterministic unsupportedness measure is limited to a system prediction of
`SUPPORTED` that selects no gold-rationale document. It is undefined when gold
evidence is unavailable/empty. Citation P/R/F1 is document-level. Retrieved
rationale sentence recall uses the adapter's sentence mapping; official SciFact
rationale prediction scoring is not claimed. nDCG is not currently computed.

The smoke is execution validation, not a benchmark result or a superiority
claim. The public test split has no local gold annotations; accuracy and F1
remain unavailable until official test annotations are obtained through a
legitimate source.
