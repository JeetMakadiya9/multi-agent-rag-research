# H5 Integrity Report

Generated deterministically by `evaluation/experiments/h5/build_package.py` (no timestamps). No inference, retrieval, downloads or network.

## Source hashes (verified against frozen SciFact hashes)

- `claims_dev.jsonl`: `86F0435D08FDB65D1AA41D1472684F57E6E71930626497BDF4D7A9EC1A632217`
- `claims_test.jsonl`: `558930D75215C73F84A28FE538307D6D397C9DE1EC7239514CF45F80D75D2CA3`
- `claims_train.jsonl`: `F4C8FA82D8BD0653A9CC8D61A6EA48C25EACEA64E90AF5DBF390EBB1B74372F0`
- `corpus.jsonl`: `B8D6C89624CB2ED74DEE8938EFFC4F5D8BD2086887880AF8110D64BE4CEADE62`

## Claims

`{"atomicity": {"ATOMIC": 673, "COMPOUND": 10, "UNCERTAIN": 126}, "eligible": 670, "exclusion_reasons": {"claim text also appears in DEV or TEST": 3, "claim_atomicity_status=COMPOUND": 10, "claim_atomicity_status=UNCERTAIN": 124, "duplicate claim text": 2, "eligible": 670}, "train_total": 809}`

## Candidate pool

`{"by_stratum": {"contradiction_oriented": 317, "neutral_oriented": 5881, "neutral_unrelated_inventory": 670, "support_oriented": 571}, "duplicate_evidence_excluded": 2, "eligible_by_stratum": {"contradiction_oriented": 317, "neutral_oriented": 5879, "support_oriented": 571}, "rows": 7439, "unique_claims": 670}`

## Samples

- pilot: `{"by_stratum": {"contradiction_oriented": 20, "neutral_oriented": 20, "support_oriented": 20}, "missing_fields": 0, "pairs": 60, "pairs_per_claim": {"1": 54, "2": 3}, "source_documents": 53, "unique_claim_evidence_pairs": 60, "unique_claims": 57, "unique_pair_ids": 60}`
- main: `{"by_stratum": {"contradiction_oriented": 100, "neutral_oriented": 100, "support_oriented": 100}, "missing_fields": 0, "pairs": 300, "pairs_per_claim": {"1": 168, "2": 66}, "source_documents": 198, "unique_claim_evidence_pairs": 300, "unique_claims": 234, "unique_pair_ids": 300}`
- pilot/main claim overlap: 0; pair overlap: 0

## Reserved-data protection

`{"dev": 300, "dev69": 69, "non_train_rows": 0, "reserved": 119, "sampled_claims_matching_any_dev": 0, "sampled_claims_matching_dev69": 0, "sampled_claims_matching_reserved": 0, "sampled_claims_matching_test": 0, "test": 300}`

## Hashes

- sampling seed: `h5-2026-v1`
- candidate pool: `E4C3798AD0A87A5DB54ED3C3D0538D9033D9D2778D07FAE85BC1163F46824743`
- selected (main) manifest: `CFC1613ECEC5BA30F03AB007296D2DD0ECB01CFF24A13978E493D3A6A17D8117`
- pilot manifest: `2B0881593349C2D3102C0608EC691C8193A73D074249B80908DF408154E2D32A`
- `h5_annotation_guidelines.md`: `44CC8E7070A05A26C6A98C4DF4E93FE7FEB116994D61EAA6D2073B1A1F69A27B`
- `h5_annotation_schema.json`: `2B54C29AA07A4672D117768AEFF3602CD030E7A64CFC3F5DFD64CDF9CC533CB2`
- `h5_annotator_a.jsonl`: `065A955D6AB8802851CA7D378F081F2288706A65F78871510FC42DF5FD5EFD5E`
- `h5_annotator_b.jsonl`: `BE3107CB1A1469CC09450447C5E927CC204D8715D91F989E7ACDB178436910AE`
- `h5_atomicity_review.jsonl`: `4B4FC473C9A57E4D7B2CE87E74639B77C63359E8E9DF7843DD5B25D4703C6CD4`
- `h5_candidate_pool.jsonl`: `E4C3798AD0A87A5DB54ED3C3D0538D9033D9D2778D07FAE85BC1163F46824743`
- `h5_claim_inventory.jsonl`: `4FD867F38C5B055256EFB83A719E44DB783E73AD622F8962A32C29DDF0FC9BE6`
- `h5_data_dictionary.md`: `E3AD592CE3F2228B97E18FD87A1098191128AD5E9968EC73FE1D6A0997D5FA95`
- `h5_excluded_candidates.jsonl`: `4A6631CF06AD48D4CB53953107F5955D6933FA4B9527646EC38CD6C638291E8F`
- `h5_gold_manifest.jsonl`: `CFC1613ECEC5BA30F03AB007296D2DD0ECB01CFF24A13978E493D3A6A17D8117`
- `h5_nli_model_requirements.md`: `DE2500D7D8A6EAA58299CC2EEE9B11673900936441CD2AC207C1D48831442F85`
- `h5_pilot_annotator_a.jsonl`: `63642062BF123B93A609DA5FE8DF60A9BFAA205C5D1EBAC8319C06214EC7119C`
- `h5_pilot_annotator_b.jsonl`: `8C33C58D7CD793E286D7D3F27A7FD60968579EE675181C11F887F21954B19B58`
- `h5_pilot_manifest.jsonl`: `2B0881593349C2D3102C0608EC691C8193A73D074249B80908DF408154E2D32A`
- `h5_preanalysis.md`: `43EE87F2957310BB518B9281B3EDD28E11C1BCF6C6C141FC1E6F4B0F1CD52750`
- `h5_protocol.md`: `C7A791E7C43252D21F7A6EC49CB644846D087C14539F49A9819D56060E41B4F8`

TRAIN annotations are used for sampling-frame construction, not as pair-level gold labels.
