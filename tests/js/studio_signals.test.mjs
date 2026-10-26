// Verdict signals in the Studio: why a statement stays undecided, agreement across passages, each passage's reason.
// Run with: node --test tests/js. Every record here is SYNTHETIC TEST DATA.
import assert from "node:assert/strict";
import { test } from "node:test";

import { CONFIDENCE_LEVELS, normalizeConfidence, normalizeRun } from "../../webrag/static/model.mjs";

const RUN = {
  input: "Drug X lowers blood pressure.", mode: "verify", status: "COMPLETE", outcome: "INSUFFICIENT_EVIDENCE",
  sources: [{ id: "S1", title: "Trial", url: "https://example.org/a", access: "FULL_TEXT" }],
  claims: [
    { text: "Drug X lowers blood pressure.", label: "INSUFFICIENT", sub_state: "NO_DECISIVE_EVIDENCE",
      abstain_reason: "No passage tested this statement: most concern a different population.",
      confidence: { level: "abstained", agreeing: 0, opposing: 0, judged: 1 },
      passages: [{ key: "S1:p1:a", source_id: "S1", page: 1, text: "children", stance: "NEUTRAL",
        verifier_reason: "does not test the claim (different population)" }] },
    { text: "It is safe.", label: "SUPPORTED", confidence: { level: "high", agreeing: 2, opposing: 0, judged: 3 } },
    { text: "Old record.", label: "SUPPORTED" },
  ],
};

test("a claim keeps its abstention reason, its confidence and each passage's reason", () => {
  const run = normalizeRun(RUN);
  const [undecided, supported, old] = run.claims;
  assert.equal(undecided.abstainReason, "No passage tested this statement: most concern a different population.");
  assert.equal(undecided.confidence.level, "abstained");
  assert.equal(undecided.passages[0].reason, "does not test the claim (different population)");
  assert.deepEqual(supported.confidence, { level: "high", label: "High confidence", agreeing: 2, opposing: 0, judged: 3 });
  assert.equal(old.confidence, null);                // records made before confidence existed show none
  assert.equal(old.abstainReason, "");
});

test("only known confidence levels are shown, and counts are never negative", () => {
  assert.equal(normalizeConfidence({ level: "certain", agreeing: 9 }), null);
  assert.equal(normalizeConfidence("high"), null);
  assert.deepEqual(Object.keys(CONFIDENCE_LEVELS), ["high", "medium", "low", "abstained"]);
  assert.equal(normalizeConfidence({ level: "low", agreeing: -3, opposing: 1, judged: 2 }).agreeing, 0);
});
