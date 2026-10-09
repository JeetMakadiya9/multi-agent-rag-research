# H5 Pre-analysis (read-only audit before any H5 artifact was generated)

All facts below were established from the repository; nothing was inferred from memory.

1. **Frozen Qwen prompt.** `experiment_b_verification._pairwise_judge_prompt` (frozen file hash `66EA7B6B…`).
   System message v1 SHA-256 `aa49d5e7dd47d4a20cb7e68486c1e507fc4acfd74c079bb9d3e3f6b5eff0c853` (recorded in
   `results/phase13_improvement/runs/B1/run_config.json`). The user message carries `question_context`,
   `original_unit_text`, one `component` and one `selected_evidence {evidence_index, text}`; output JSON
   `{"assessments":[{"evidence_index","component_indices","stance"}],"reason"}`.
2. **Stance labels.** Judge: `SUPPORTS | CONTRADICTS | NEUTRAL` (invalid output = parse error, recorded as
   `Stance.INVALID` in evidence_core). H5 gold labels are `SUPPORT | CONTRADICT | NEUTRAL`; mapping
   SUPPORTS→SUPPORT, CONTRADICTS→CONTRADICT, NEUTRAL→NEUTRAL, INVALID→error.
3. **Pair-level records.** `evidence_core.contracts.Judgement` (claim_id, evidence_id, stance, JudgeIdentity
   kind/model/prompt, provenance) over `Claim` and `Evidence`; the same pair can carry LLM, NLI and HUMAN judgements.
4. **SciFact data available** (hashes match the frozen values in `phase10e_remediation/dataset_integrity.json`):
   TRAIN 809 claims (with rationale annotations), DEV 300, TEST 300 (no labels), corpus 5,183 abstracts.
5. **Reserved data.** `results/phase13_improvement/development_set_manifest.json`: 69 dev claims
   (`claim_ids`) and 119 reserved labeled claims (`reserved_for_later_validation_claim_ids`), all DEV IDs.
6. **Evidence sentence identity.** Corpus abstracts are lists of sentences; a sentence is identified by
   `(doc_id, 0-based index)`.
7. **TRAIN rationale mapping.** `evaluation.scifact.scifact_adapter.load_scifact_dataset(..., "train")` validates
   every rationale sentence index against the corpus (raises otherwise); mapping is deterministic.
8. **Feasibility.** Deterministic atomicity (rule `h5-atomicity-v1`): ATOMIC 673, UNCERTAIN 126, COMPOUND 10 of 809
   TRAIN claims; 3 TRAIN claim texts also occur in DEV/TEST and 2 are duplicate texts within TRAIN (all excluded).
   Eligible candidate pairs: support-oriented 571, contradiction-oriented 317, neutral-oriented 5,879 — each
   ≥ 120 (pilot 20 + main 100 per stratum).

Existing H5 material: the repository contains no H5 protocol file. H5 is referenced only in
`results/phase13_improvement/phase13_plan.md` (gate sentence, lines 59–60) and in `evidence_core/README.md`. The
earlier H5 design discussion was not persisted; this package is the first frozen H5 protocol.

Offline guarantees: stdlib + frozen project modules only; no package installation, Ollama call, model download or
network request is required.
