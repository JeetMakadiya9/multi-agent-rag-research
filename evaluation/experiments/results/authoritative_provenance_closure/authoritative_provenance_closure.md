# Authoritative Provenance Closure

**Closure status: PASS**  
**Historical filename classification: UNRECOVERABLE_HISTORICAL_METADATA**  
**Current reproducibility: PASS**  
**Final full-DEV evaluation: SEPARATE / INCOMPLETE**

## Scope

This record closes the historical filename issue for current and future reproducibility. It does not reconstruct facts that the available repository history cannot establish.

## Historical filename issue

The repository currently contains `evaluation/scifact/metrics.py`. Its exact SHA-256 is `8BE4B90BB28090FC48F874AF399798B0F7E39C25AAD7A1FA17B864A002E55587` and matches the authoritative expected hash. The previously referenced `evaluation/scifact/scifact_metrics.py` path is absent. Available Git and repository evidence does not establish that the historical filename existed, was renamed, or was removed.

**The historical filename provenance cannot be reconstructed from the available repository history. This is treated as unresolved historical metadata rather than a current implementation or reproducibility defect.**

No claim is made that `evaluation/scifact/scifact_metrics.py` existed or was renamed. The filename classification is **UNRECOVERABLE_HISTORICAL_METADATA**.

## Authoritative provenance baseline

The Phase 10G baseline commit is `8e996b8db6c9a157bbb1f58009d34c148bc4782c`. The authoritative provenance-record commit is `4e1f343a4d2e87c04393e9b5a473446b6e825231`. Current HEAD is `55a6abdb98625453b572503d4aab7390e59c51bd` on branch `main`. The current HEAD directly descends from the recorded provenance commit; the Phase 10G baseline is an ancestor. The current `origin` URL matches the repository URL recorded in provenance, and local `origin/main` resolves to current HEAD. The Phase 10G point-in-time inventory recorded no configured remotes; the current origin is therefore recorded as a later environment state, not silently projected backward onto the baseline.

The current working tree has no tracked modifications. It contains these known, pre-existing untracked result directories: `evaluation/experiments/results/phase10_final_analysis/`, `evaluation/experiments/results/phase10_full_dev/`, `evaluation/experiments/results/phase11_results/`. The Phase 10H closure was not committed because the optional commit condition requires a clean tree apart from the new closure artifacts. No history was amended, rewritten, or pushed.

## Canonical implementation and reproducibility

The current canonical implementation is `evaluation/scifact/metrics.py`; its bytes are verified against SHA-256 `8BE4B90BB28090FC48F874AF399798B0F7E39C25AAD7A1FA17B864A002E55587`. The unresolved historical filename is not part of the current canonical implementation. The authoritative baseline establishes reproducibility from the verified current implementation and recorded configuration forward.

Dataset counts and hashes match the Phase 10G provenance record: train 809, DEV 300, test 300, corpus 5,183 documents, and 45,972 indexed chunks. All eight frozen source hashes match. The persisted full-DEV model/configuration identities are included in the JSON record; no endpoint health query or generation request was made for this closure.

## Preservation and scope

Existing historical reports and Phase 10 experiment data are preserved unchanged. The raw Phase 10 attempt/per-example JSONL hashes match their previously recorded hashes; Phase 10 final-analysis artifacts match previously sampled hashes where those were recorded.

**Phase status closure does not mean that the full 300-claim / 900-attempt evaluation is complete.** The full-DEV run remains **PARTIAL / INCOMPLETE**: 662 durable attempts, 606 successes, 52 failures, 4 interruptions, 660 scored rows, 220 complete A/B/C triples, 197 all-success triples, and 238 planned pairs without durable attempt rows. This closure is only about historical filename provenance.

## Superseding statement

1. The old historical filename cannot be verified.
2. The current canonical implementation is `evaluation/scifact/metrics.py`.
3. Its exact SHA-256 has been verified.
4. The authoritative baseline establishes reproducibility from that point forward.
5. The unresolved historical filename is not part of the current canonical implementation.
6. No claim is made that `evaluation/scifact/scifact_metrics.py` existed or was renamed.
7. Existing historical reports are preserved unchanged.
8. This closure supersedes the historical filename issue for current reproducibility and evaluation provenance.

Closure timestamp: `2026-10-05T00:42:35+05:30` (Asia/Kolkata); `2026-10-04T19:12:35Z` UTC.
