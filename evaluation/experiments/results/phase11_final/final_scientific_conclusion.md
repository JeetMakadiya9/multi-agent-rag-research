# Final scientific conclusion

The work built and evaluated three systems on the complete 300-claim SciFact DEV set: A (strong RAG), B (RAG plus claim/evidence verification), and C (bounded multi-agent research using Phase 8A/8B/8C). The authoritative run has 900/900 persisted attempts and scored rows: 840 SUCCESS, 55 FAILED, 5 INTERRUPTED.

Among labeled claims with valid verifier predictions, B was correct on 87/179 (48.6%); C on 85/176 (48.3%). In 176 paired valid labeled claims, both were correct on 84, both wrong on 90, with 1 B-only and 1 C-only correct cases. Exact McNemar p=1.000. A does not provide a verifier label, so the study cannot calculate whether B improves classification over A. The paired B/C data do not show a statistically significant correctness difference.

Document Recall@1 was A 146/209 (69.9%), B 146/208 (70.2%), C 144/205 (70.2%). Document overlap with annotated evidence does not establish entailment or factual correctness.

Successful mean latency was A 26.49s, B 39.64s, C 38.58s. Persisted total LLM calls were A 299, B 645, C 608; C recorded 596 controller cycles. C’s 276/276 successful traces passed the 8A, 8B, 7B, Phase 6, continuation authorization, researcher transition and completion audit. This demonstrates execution-path behavior, not improved correctness; the path retains a researcher gate.

The 60 non-success outcomes comprise 29 explicit endpoint infrastructure failures, 22 C controller/execution failures, 4 B verification failures and 5 interruptions. Failures remain in operational reliability and are excluded from valid-prediction denominators. They were not retried, erased or counted correct.

The empirical hypothesis is **not determined by this comparison**: A has no verifier-derived label, so verification cannot be compared against A on classification; paired B/C correctness shows no significant difference. The completed DEV measurements do not establish that verification improved over A or that C improved factual reliability over B. The architecture contribution is bounded, traceable execution with explicit verification/capability routing and researcher-gated continuation. Limitations include DEV-only scope, unequal valid denominators, runtime failures and inherited aggregate metric definition differences.
