# Phase 10H — Authoritative Provenance Closure

## 1. Objective

Close the historical filename issue for reproducibility from the verified current canonical implementation forward, without reconstructing unsupported historical facts or changing experimental results.

## 2. Historical Filename Issue

The prior reference `evaluation/scifact/scifact_metrics.py` is absent. Available repository evidence does not establish that it existed or was renamed. Classification: **UNRECOVERABLE_HISTORICAL_METADATA**.

## 3. Evidence Available

The current repository, Phase 10G provenance record, Git commit lineage, authoritative dataset/frozen hashes, Phase 10 full-run identity and JSONL files, final Phase 10 analysis, and Phase 10 status artifacts were inspected. The Phase 10G baseline commit is `8e996b8db6c9a157bbb1f58009d34c148bc4782c`; authoritative provenance record commit `4e1f343a4d2e87c04393e9b5a473446b6e825231` is the direct parent of current HEAD `55a6abdb98625453b572503d4aab7390e59c51bd`. `origin/main` resolves to current HEAD and current origin URL matches the repository URL in the authoritative record.

## 4. Evidence Not Available

No repository/history evidence establishes historical existence, deletion, or rename of `evaluation/scifact/scifact_metrics.py`. No such claim is made. The Phase 10G inventory recorded no configured remotes at that point; an origin URL is configured now and is recorded as a later state.

## 5. Canonical Current Implementation

`evaluation/scifact/metrics.py` exists and its SHA-256 is `8BE4B90BB28090FC48F874AF399798B0F7E39C25AAD7A1FA17B864A002E55587`. It matches the authoritative expected value `8BE4B90BB28090FC48F874AF399798B0F7E39C25AAD7A1FA17B864A002E55587`. **Current implementation bytes: VERIFIED.**

## 6. New Authoritative Baseline

This closure records current HEAD, branch, origin URL, worktree state, canonical implementation hash, dataset identity, frozen hashes, model identity, configuration identity, and closure timestamp. It establishes reproducibility from this verified point forward. Historical filename provenance remains classified as unrecoverable metadata rather than a current defect.

## 7. Dataset Integrity

Counts and SHA-256 hashes match the authoritative provenance files: train 809; DEV 300; test 300; corpus 5,183 documents; full retrieval index 45,972 chunks. No SciFact file was modified.

## 8. Frozen Source Integrity

All eight frozen source hashes match their recorded values. No frozen or production source was modified.

## 9. Phase 10 Status Closure

See `phase10_status_closure.md` and `.json`. Phase 10A remains historical FAIL/SUPERSEDED. The scoped results/readiness/forensic phases 10B–10G are classified as specified in the closure record; original reports are not rewritten. Closure of the inherited filename issue does not elevate 10A.

## 10. Final Full-DEV Evaluation Status

The final full-DEV run remains **PARTIAL / INCOMPLETE**: 662 durable attempts; 606 successes; 52 failures; 4 interruptions; 660 scored rows; 220 complete A/B/C triples; 197 all-success triples; 238 planned pairs without durable attempt rows (one with an inflight marker). No counts were altered. **Phase status closure does not mean that the full 300-claim / 900-attempt evaluation is complete.**

## 11. Reproducibility Statement

The historical provenance of the previously referenced `evaluation/scifact/scifact_metrics.py` path cannot be reconstructed from the available repository history. This historical uncertainty is therefore closed as unrecoverable historical metadata rather than treated as a current implementation defect. The canonical implementation, `evaluation/scifact/metrics.py`, has been independently verified by exact SHA-256 hash and is incorporated into the authoritative reproducibility baseline. Existing experimental artifacts remain unchanged and historical records are preserved. The current full-DEV experiment remains an incomplete evaluation and is tracked separately from this provenance closure.

**CURRENT REPRODUCIBILITY: PASS.**  
**HISTORICAL FILENAME PROVENANCE: UNRECOVERABLE_HISTORICAL_METADATA.**  
**FINAL FULL-DEV EXPERIMENT: SEPARATE / INCOMPLETE.**

## 12. Limitations

The filename’s historical existence/rename cannot be reconstructed. The current worktree contains known untracked Phase 10/11 analysis output directories and is not clean; therefore no optional Git commit or push was performed. The full-DEV experimental run is separate and incomplete. This closure makes no performance or superiority claim.

## 13. Conclusion

The current canonical implementation and all checked dataset/frozen identities match the authoritative baseline. The historical filename issue is explicitly classified as **UNRECOVERABLE_HISTORICAL_METADATA**. This closes the historical provenance blocker for current reproducibility without changing prior reports or results. Phase 10A remains FAIL/SUPERSEDED, and the full-DEV evaluation remains PARTIAL / INCOMPLETE.

Closure time: 2026-10-05T00:42:35+05:30 (Asia/Kolkata), 2026-10-04T19:12:35Z UTC.
