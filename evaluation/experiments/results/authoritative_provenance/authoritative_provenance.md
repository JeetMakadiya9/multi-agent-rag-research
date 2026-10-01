# Authoritative Project Provenance

- **Provenance version:** 1.0.0
- **Authoritative since (UTC):** 2026-10-01T16:07:09Z
- **Repository:** [JeetMakadiya9/multi-agent-rag-research](https://github.com/JeetMakadiya9/multi-agent-rag-research)
- **Branch:** `main`
- **Verified project commit:** `4e1f343a4d2e87c04393e9b5a473446b6e825231`
- **Git baseline commit:** `8e996b8db6c9a157bbb1f58009d34c148bc4782c`
- **Project path:** `D:\Projects\MultiAgent_RAG`

## Canonical SciFact metrics implementation

- Path: `evaluation/scifact/metrics.py`
- SHA-256: `8BE4B90BB28090FC48F874AF399798B0F7E39C25AAD7A1FA17B864A002E55587`

## Dataset identity

- Dataset: SciFact-Orig (authoritative evaluation split: dev)
- Counts: train 809; DEV 300; TEST 300; corpus 5183 documents.
- Retrieval index: 45972 chunks, as recorded by `evaluation/experiments/results/phase10f_launch_validation/dataset_identity.json`; it was not rebuilt for this record.

| Dataset file | Records | SHA-256 | Verification |
|---|---:|---|---|
| `claims_dev.jsonl` | 300 | `86F0435D08FDB65D1AA41D1472684F57E6E71930626497BDF4D7A9EC1A632217` | MATCH |
| `claims_test.jsonl` | 300 | `558930D75215C73F84A28FE538307D6D397C9DE1EC7239514CF45F80D75D2CA3` | MATCH |
| `claims_train.jsonl` | 809 | `F4C8FA82D8BD0653A9CC8D61A6EA48C25EACEA64E90AF5DBF390EBB1B74372F0` | MATCH |
| `corpus.jsonl` | 5183 | `B8D6C89624CB2ED74DEE8938EFFC4F5D8BD2086887880AF8110D64BE4CEADE62` | MATCH |

## Frozen source hashes

| Source file | SHA-256 |
|---|---|
| `rag.py` | `D2B0CF8930BC66FC66995AD63C1028AE3B126424E01E091DF22ED238AD5DBA19` |
| `research_pipeline.py` | `F55A29386AA13774A621B855E89CA8CA71FEAC0CFE2A124F90EF54DD7CD541D0` |
| `claim_verification.py` | `63FC8D73F55B7F48DBB340E4CD67C97975D0D587FB009CEAFA97E7FEF1C639A1` |
| `verification_units.py` | `BEFE07461C9B0DA76BF68226D86A96346D761C1AD3F127C554D4BB9C51644DDF` |
| `experiment_b_verification.py` | `66EA7B6BE6119E7A3E25A7BAD3AF3D078F08CBA4CC8AC4A462BAF6170548F3B4` |
| `evaluation/scifact/scifact_adapter.py` | `8F2BE845CFAF9ABB2210C88242F02E55EA8F47FF68E71BCDD27098287904BC9A` |
| `evaluation/scifact/metrics.py` | `8BE4B90BB28090FC48F874AF399798B0F7E39C25AAD7A1FA17B864A002E55587` |
| `evaluation/scifact/run_scifact_evaluation.py` | `21F540C534B5CFE8C8328C648C6E89AF6E4C4C16497CF78BC52DB51BC9629338` |

## Model identity

- Provider/model: Ollama / `qwen3:4b`
- Digest: `359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7`
- Live local identity check: captured and matched against live local /api/tags and /api/show metadata; endpoint reachable and model present. The check used local tags/show metadata only; no generation request was sent.
- Recorded generation settings:

  - `base_url`: `http://127.0.0.1:11434/api/chat`
  - `temperature`: `0.0`
  - `think`: `False`
  - `num_gpu`: `0`
  - `num_predict`: `16`
  - `format`: `json`
  - `stream`: `False`
  - `timeout_seconds`: `1.0`

## Evaluation framework identity

- Framework: Phase 10 SciFact A/B/C evaluation harness
- Evaluation schema: `phase10f-attempt-v1`
- Phase 10F configuration fingerprint: `7c8b99e360727f05933f912ba5904a2e150a8311712e07bf832cc157f6c4dc78`
- Phase 10E configuration fingerprint: `95184e04be10b963fd39433d1beb28e51e80667840f0f33bdc5dcc7d96b7f1fb`
- Runtime: Python 3.12.14 on `Windows-11-10.0.26200-SP0`.

### Framework source SHA-256

- `evaluation/experiments/run_phase10_dev.py`: `C411C501BEB451E2221FB159BC48E73DCD0053894FD4429FCFA903E6A4CA1FC8`
- `evaluation/experiments/run_phase10f.py`: `5F9D01126CC94D741AD5381E34FC25F06B7FD2EF38F99896744A416D333AE1D2`
- `evaluation/experiments/evaluator.py`: `7AAC677BA7BF054AC5CFE4827188394E2F6C6A08E5CCD51C9043A7C5A7776640`
- `evaluation/experiments/experiment_config.py`: `45D5BEA6D501C2EC15D3DE808A0002EDFF2DD2F816C9C4C806CF13C5F2BFBC64`
- `evaluation/experiments/local_model.py`: `7710E641F9C395DCA6B9CDECCC101FBE49B60E41F69623212C6456C120B679DB`
- `evaluation/experiments/system_a_rag.py`: `8D8936A3FA0BADE165D5086225F5C5ED9AD0E01325089B22E5CCAEA491E2FC46`
- `evaluation/experiments/system_b_verified_rag.py`: `79DD09885CFAB30CCCC2A0728F4E36AF7CCD7D4FD0CCBEBC0BD8AD570744C68F`
- `evaluation/experiments/system_c_multi_agent.py`: `F30E1C5810D2BD15729950BD61BBC70173B1CB76F13D9064B615D1048684ABB2`

## Scope statements

- Historical provenance before this baseline is not authoritative.
- Historical provenance for evaluation/scifact/scifact_metrics.py is not being reconstructed; no claim is made here that it did or did not exist or was renamed.
- The current canonical SciFact metrics implementation is evaluation/scifact/metrics.py.
- Authoritative provenance begins from this Git commit onward.

This record captures verified state only. Benchmark inference was not run; dataset and frozen files were not modified.
