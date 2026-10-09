# Phase 12 Reproducibility

Analysis version: 1.0. Run ID: `02fb9c7a1268476ca1f9b12d9dbfb25e`. Configuration identity: `95184e04be10b963fd39433d1beb28e51e80667840f0f33bdc5dcc7d96b7f1fb`.

This package was computed deterministically from persisted per-example and attempt JSONL, Phase 10 manifest/metrics, Phase 11A/B analyses, and SciFact DEV annotations. No random sampling was performed in Phase 12; no model was run. Timestamps and output artifact hashes are the only environment-dependent fields.

## Inputs and SHA-256

- `phase10_per_example` — `8B87A0A2EBE3D3DD0BDE8332C4A0B5766F51379DF80222A2234B0EE806E0242E`
- `phase10_attempt_records` — `941975A740159F1F89D44B8534B817D94B26C5164CAE64A868EFEA468165C372`
- `phase10_manifest` — `A01B3BE61D74737C9C2448B0FD2992B83AB6F99B5E4346DAEE059EB488D3BC36`
- `phase11b_final_audit` — `C4F7AD90ED9426E8AAFC4E9323CC4230F04769230400131E7D9F42DE22B46D12`
- `phase11b_classification` — `F06D34AFBB5FAB0711AE3C90B4418A3EE5D35B3AC01C1F527BB96453A009E440`
- `phase11b_retrieval` — `DEECA279A969D2F3068F569F79303B96320EC8804BA0B82B504945B6D8E19633`
- `phase11b_paired` — `937780758D5F4EB450F1037B0F94248A93274F6E5AFD936CCE762C330E049D92`
- `dev_claims` — `86F0435D08FDB65D1AA41D1472684F57E6E71930626497BDF4D7A9EC1A632217`
- `phase10_evaluator` — `7AAC677BA7BF054AC5CFE4827188394E2F6C6A08E5CCD51C9043A7C5A7776640`
- `system_a_adapter` — `8D8936A3FA0BADE165D5086225F5C5ED9AD0E01325089B22E5CCAEA491E2FC46`
- `system_b_adapter` — `79DD09885CFAB30CCCC2A0728F4E36AF7CCD7D4FD0CCBEBC0BD8AD570744C68F`
- `system_c_adapter` — `F30E1C5810D2BD15729950BD61BBC70173B1CB76F13D9064B615D1048684ABB2`
- `phase10_runner` — `C411C501BEB451E2221FB159BC48E73DCD0053894FD4429FCFA903E6A4CA1FC8`

## Join and denominator rules

- Runtime results remain one row per claim/system. Gold annotation is used only by this post-hoc analysis and is joined by claim ID.
- Valid classification requires SUCCESS plus one of the persisted stance labels. Accuracy denominator is labeled, successful, valid predictions. Failed, interrupted, invalid/missing, and unlabeled cases are counted separately.
- Retrieval overlap uses available persisted ranked chunks and gold document / sentence references. Unavailable or failed retrieval is excluded from retrieval-hit/miss cells, never mapped to miss.
- Exact gold rationale passage overlap requires matching SciFact document ID and sentence index. It is not entailment.
- Selected evidence is read from persisted selected document IDs; no missing selected IDs are inferred.
- Complexity features are hand-specified regular-expression/string rules, not semantic annotations.

## Reproduction

Run the deterministic Phase 12 analysis script in an isolated analysis environment using the pinned source artifacts and paths documented above. The generating script was ephemeral under the local TEMP directory and was not added to the project. Recompute results from original Phase 10/11 files; do not consume Phase 12 aggregates as input. Result source paths and full hashes are in `phase12_audit.json`.
