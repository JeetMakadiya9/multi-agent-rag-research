// The opt-in rank tie-break verifier in the studio: its sub-state, evidence-quality label and caution are explained,
// and a record keeps both sides of the disagreement. Run with: node --test tests/js
// Every record here is SYNTHETIC TEST DATA shaped like webrag output; nothing was produced by a real run.
import assert from "node:assert/strict";
import { test } from "node:test";

import { CAUTION_KINDS, cautionMeta, normalizeRun, strengthMeta, subStateText } from "../../webrag/static/model.mjs";

test("the RANK_TIEBREAK sub-state, the rank decided label and the caution are known and explained", () => {
  assert.match(subStateText("RANK_TIEBREAK"), /rank tie-break took the first-ranked passage/);
  assert.match(subStateText("RANK_TIEBREAK"), /no extra model call/);
  assert.match(subStateText("ADJUDICATED"), /adjudication step/, "the adjudicated sub-state is unchanged");
  const meta = strengthMeta("rank decided");
  assert.equal(meta.label, "Decided by rank");
  assert.equal(meta.tone, "uncertain");
  assert.match(meta.description, /not a more reliable one/);
  assert.ok(CAUTION_KINDS.includes("rank_decided_conflict"));
  assert.equal(cautionMeta({ kind: "rank_decided_conflict", detail: "x" }, {}).known, true);
});

test("a rank-decided verification record keeps both sides and its verdict", () => {
  const run = normalizeRun({
    input: "The bridge opened in 1932.", mode: "verify", status: "COMPLETE", outcome: "CONTRADICTED",
    sources: [{ id: "S1", title: "Bridge A (synthetic)", url: "https://example.org/a" },
              { id: "S2", title: "Bridge B (synthetic)", url: "https://example.org/b" }],
    claims: [{ text: "The bridge opened in 1932.", label: "CONTRADICTED", sub_state: "RANK_TIEBREAK",
      verifier_note: "conflicting evidence decided by rank: E1 (the first passage with a stance) CONTRADICTS; no extra model call",
      passages: [{ source_id: "S2", text: "Opened in 1934.", stance: "CONTRADICTS" },
                 { source_id: "S1", text: "Opened in 1932.", stance: "SUPPORTS" }] }],
    disagreements: [{ text: "Retrieved sources disagreed about this claim; the rank tie-break took the first-ranked passage with a stance as decisive (CONTRADICTED).",
      sources: ["S2", "S1"], verified: true }],
  });
  assert.equal(run.claims[0].label, "CONTRADICTED");
  assert.equal(run.claims[0].subState, "RANK_TIEBREAK");
  assert.deepEqual(run.claims[0].passages.map((p) => p.stance), ["CONTRADICTS", "SUPPORTS"]);
  assert.deepEqual(run.warnings, []);
});
