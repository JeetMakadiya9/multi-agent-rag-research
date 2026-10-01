# Experiment B SciFact evaluation infrastructure

This directory contains isolated SciFact-Orig loading, corpus adaptation,
retrieval/verification scoring, and a dev-only runner. It does not change the
retriever or verifier and it does not download data.

> **This infrastructure does not constitute the final thesis-level Experiment B result.**

## Task and dataset

The official SciFact-Orig task uses scientific claims and a corpus of scientific
abstracts. Corpus records contain `doc_id`, `title`, sentence-list `abstract`,
and `structured`. Labeled claim records contain an integer `id`, claim text,
evidence documents, rationale sentence-index sets, `SUPPORT` or `CONTRADICT`
labels, and `cited_doc_ids`. The official repository publishes `corpus.jsonl`
and `claims_train.jsonl`, `claims_dev.jsonl`, and `claims_test.jsonl`. Public
test labels are unavailable; the official repository directs test-set scoring
to its leaderboard. See the [official data schema](https://github.com/allenai/scifact/blob/master/doc/data.md),
[split information](https://github.com/allenai/scifact), and
[evaluation rules](https://github.com/allenai/scifact/blob/master/doc/evaluation.md).

The initial runner supports **dev only**. It takes an explicit dataset root
containing `corpus.jsonl` and `claims_dev.jsonl`; it never downloads files and
fails with the missing path(s) if they are absent. The data loader can parse
train/dev/test individually, but the runner intentionally rejects non-dev
splits in this initial version.

## Data flow and leakage controls

1. `scifact_adapter.load_scifact_dataset(root, split)` validates source JSONL.
2. Runtime claims expose only `claim_id` and claim text. Gold evidence,
   rationale indices, labels, and `cited_doc_ids` are held in a separate
   `ClaimAnnotations` value for scoring.
3. The corpus adapter receives only corpus documents. It converts each
   abstract sentence to one synthetic RAG page, adds the title to the searchable
   text, and calls the existing `rag.create_chunks()` function. The frozen
   chunker preserves filename/page/chunk ID but drops arbitrary source metadata.
   A small post-chunk wrapper maps each resulting chunk to `scifact_doc_id` and
   its original zero-based sentence index/text. This wrapper contains no claim
   or gold annotation fields.
4. The existing embedding, FAISS, BM25, and `ExistingRAGProvider` path builds
   and queries the corpus. Each retrieval query is exactly the claim text; no
   cited-document IDs are used as seeds.
5. Experiment A is optional and off by default. The adapter uses the existing
   baseline generation prompt and model with the retrieved context and records
   the output as `free_text_rag_output`. It does not parse it into SciFact
   labels or rationales.
6. Experiment B is on by default and uses the existing
   `run_experiment_b_verification()` implementation with the retrieved
   evidence. Unit-specific extra retrieval is disabled for this evaluation
   pass so the verification condition uses the same initial retrieval set
   whose ranks are measured. The pairwise verifier sees evidence text and
   stable IDs only; corpus sentence/document mapping stays in the runner’s
   scoring-side references. No gold fields are included in the evidence sent
   to the verifier.

The records intentionally retain scoring annotations separately under
`scoring_annotations`; those fields are never used to produce runtime inputs.
The public test split may omit the `evidence` field. The adapter preserves that
as unavailable annotations. An explicitly present empty evidence object is
also kept distinct; neither case is automatically converted to an Experiment B
verdict.

## Metrics and task mismatch

Retrieval metrics are unique-gold-document Recall@1/5/10/20 and first-gold rank
/ reciprocal rank. Duplicate chunks from one document do not inflate recall;
cutoffs beyond the number of returned chunks are safely calculated from all
available results. Empty gold evidence and unavailable labels produce null
recall with an explicit scoring status.

Experiment B records its deterministic verdict, evidence/component assessments,
coverage/conflicts/completeness, invalid or missing assessments, retries, call
counts, timing, errors, and raw trace. The scoring adapter checks stance
alignment only where the retrieved chunk maps to an annotated gold rationale
sentence. Multi-component cases, missing assessments, and opposing retrieved
gold labels are explicitly marked ambiguous or incomplete.

SciFact provides evidence-level `SUPPORT`/`CONTRADICT` annotations, while
Experiment B emits a claim-level `SUPPORTED`/`CONTRADICTED`/
`INSUFFICIENT_EVIDENCE` verdict. This initial infrastructure deliberately does
not invent a three-class gold label or calculate three-class accuracy/F1.
Claims with no annotated evidence do not automatically become gold
`INSUFFICIENT_EVIDENCE`. Conflicting retrieved gold labels are recorded rather
than collapsed into one stance.

The reported verification-coverage denominator contains claims for which a
complete gold rationale sentence was in Experiment B's evidence context, the
retrieved gold labels were not conflicting, and one component made a
claim-level stance comparison applicable. Coverage is the share of that
denominator with a complete assessment and a non-`INSUFFICIENT_EVIDENCE`
verdict; it is not a correctness score.

Official SciFact rationale scoring is not implemented. The current Experiment
B output does not select sentence-index rationale sets in SciFact's prediction
format, and a stance on an abstract sentence is not by itself a valid predicted
rationale set. A separate, predeclared rationale-selection method is needed
before those official metrics can be computed fairly.

## Run and outputs

From the project root, with the project's virtual environment active:

```powershell
python evaluation\scifact\run_scifact_evaluation.py `
  --dataset-root D:\path\to\scifact\data `
  --split dev `
  --output-dir evaluation\scifact\runs\dev-YYYYMMDD-HHMMSS `
  --top-k 1,5,10,20
```

Experiment B runs by default. Add `--run-experiment-a` to also produce and
store the raw free-text output. Use `--no-run-experiment-b` to disable B.
Existing output files are never overwritten. Running the CLI processes the
selected dev split in full; tests use temporary miniature fixtures and do not
run this full evaluation.

Each run writes:

- `run_manifest.json`: split and dataset paths/hashes, project revision when
  available, frozen `rag.py` hash, model/config values, prompt-source hashes,
  timestamp, Python and available package versions.
- `per_claim.jsonl`: separate runtime inputs and scoring annotations, ranked
  results, raw Experiment A output if requested, full Experiment B report and
  trace, and per-claim timings/calls.
- `summary.json`: retrieval recall/rank, evidence-level stance alignment and
  ambiguity counts, completeness/invalid rates, verdict counts, coverage, and
  efficiency. It does not report a fabricated three-class accuracy/F1.

The dataset root must contain the official files; nothing is acquired
automatically. Preserve dataset attribution and licenses: SciFact states that
claims/annotations are CC BY 4.0 and the S2ORC abstracts are ODC-By 1.0 in its
[license file](https://github.com/allenai/scifact/blob/master/LICENSE.md).

## Frozen components and scope

This infrastructure does not alter `rag.py`, retrieval algorithms or
thresholds, Experiment A's existing wrapper/prompt, Experiment B verification
or deterministic aggregation, or existing engineering cases. It does not
start Experiment C, tune settings on SciFact, or claim research-level results.
