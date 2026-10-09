# H5 Data Dictionary

All JSONL files: one JSON object per line, keys sorted, UTF-8, `\n` line endings, deterministic row order.

## Hidden files (researchers only — never given to annotators or adjudicators)

**h5_candidate_pool.jsonl** (rows sorted by `pair_id`)
`pair_id` stable ID `h5_` + 16 hex (SHA-256 of `h5-pair-v1|claim_id|doc_id|sentence_index`) · `claim_id`
(`train:<id>`) · `claim_text` · `evidence_text` (one corpus sentence) · `document_id` · `sentence_index` (0-based)
· `evidence_sentence_id` (`doc:index`) · `source_split` (`train`) · `sampling_stratum` · `rationale_candidate` ·
`non_rationale_candidate` · `train_rationale_label_hidden` (TRAIN rationale label; sampling signal only, never gold)
· `claim_atomicity_status` · `sampling_rank_key` · `eligible_for_sampling` · `exclusion_reason`.

**h5_claim_inventory.jsonl / h5_excluded_candidates.jsonl** — every TRAIN claim (resp. the excluded ones) with
`claim_atomicity_status`, `atomicity_reasons`, `atomicity_rule_version`, `eligible`, `exclusion_reason`.

**h5_gold_manifest.jsonl** (main, 300) and **h5_pilot_manifest.jsonl** (pilot, 60) — the candidate fields above plus
`sample` (`main`/`pilot`), `claim_review_id`, `sampling_metadata_hidden=true`, and annotation-state fields filled
only by `agreement.py` after human labelling: `annotation_status`, `annotator_a_label`, `annotator_b_label`,
`adjudication_required`, `adjudicated_label`, `adjudication_note`, `gold_label` (all null now).

**h5_sampling_manifest.json** — seed, algorithm, quotas, atomicity rules, source hashes, file hashes, statistics.

## Blinded files (given to people)

**h5_annotator_a.jsonl / h5_annotator_b.jsonl / h5_pilot_annotator_a.jsonl / h5_pilot_annotator_b.jsonl** — exactly
`pair_id, claim_text, evidence_text, gold_label (null until labelled; SUPPORT|CONTRADICT|NEUTRAL), annotator_note`;
A and B have independent hash orderings. Validated by **h5_annotation_schema.json**.

**h5_atomicity_review.jsonl** — `claim_review_id, claim_text, reviewer_atomicity (null; ATOMIC|COMPOUND|UNCERTAIN),
reviewer_note`; one row per sampled claim; no evidence, no labels.

Adjudication files are generated later by `agreement.py` (claim, evidence, `label_1`, `label_2` in randomised
anonymous order, `adjudicated_label`, `adjudication_note`).
