// The opt-in adjudicated verifier in the studio: its sub-state, evidence-quality label and caution are explained, and a
// record keeps both sides of the disagreement. Run with: node --test tests/js
// Every record here is SYNTHETIC TEST DATA shaped like webrag output; nothing was produced by a real run.
import assert from "node:assert/strict";
import { test } from "node:test";

import { CAUTION_KINDS, cautionMeta, normalizeRun, strengthMeta, subStateText } from "../../webrag/static/model.mjs";

test("the ADJUDICATED sub-state, the adjudicated label and the caution are known and explained", () => {
  assert.match(subStateText("ADJUDICATED"), /adjudication step judged one side decisive/);
  assert.match(subStateText("CONFLICTING"), /disagree/, "the existing sub-state is unchanged");
  const meta = strengthMeta("adjudicated");
  assert.equal(meta.label, "Adjudicated");
  assert.equal(meta.tone, "uncertain");
  assert.ok(CAUTION_KINDS.includes("adjudicated_conflict"));
  assert.equal(cautionMeta({ kind: "adjudicated_conflict", detail: "x" }, {}).known, true);
});

test("an adjudicated verification record keeps both sides and its verdict", () => {
  const run = normalizeRun({
    input: "The bridge opened in 1932.", mode: "verify", status: "COMPLETE", outcome: "SUPPORTED",
    claims: [{ text: "The bridge opened in 1932.", label: "SUPPORTED", sub_state: "ADJUDICATED",
      verifier_note: "adjudicated: E1 judged decisive over conflicting evidence",
      passages: [{ source_id: "S1", text: "Opened in 1932.", stance: "SUPPORTS" },
                 { source_id: "S2", text: "Opened in 1934.", stance: "CONTRADICTS" }] }],
    disagreements: [{ text: "Retrieved sources disagreed about this claim; an adjudication step judged one side decisive (SUPPORTED).",
      sources: ["S1", "S2"], verified: true }],
  });
  assert.equal(run.claims[0].label, "SUPPORTED");
  assert.equal(run.claims[0].subState, "ADJUDICATED");
  assert.deepEqual(run.claims[0].passages.map((p) => p.stance), ["SUPPORTS", "CONTRADICTS"]);
});
