# Paired B/C analysis

On 176 claims with both valid labels and gold labels: both correct 84; both incorrect 90; B-only correct 1; C-only correct 1. Paired accuracies: B 48.295%; C 48.295%.

Two-sided exact McNemar: 2 discordant pairs, p=1.0000; not significant at 0.05. This is descriptive and does not support superiority. Across all claims with valid B/C predictions, label agreement is 271/275 = 98.545%; 4 disagree.


## Evidence and retrieval agreement

Among successful B/C output pairs, selected evidence document sets were identical on 267 claims and different on 8. B/C top-1 retrieval agreement is in final_system_comparison.json; prediction disagreement claim details are in final_paired_analysis.json. No causal explanation is inferred from these differences.
