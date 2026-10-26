// Agentic verification (the Studio default, study arm C) in the studio: its unsided sub-state, label and caution are
// explained, and a record with passages found by the planner's search and a workflow trace loads without warnings.
// Run with: node --test tests/js. Every record here is SYNTHETIC TEST DATA shaped like webrag output.
import assert from "node:assert/strict";
import { test } from "node:test";

import { CAUTION_KINDS, cautionMeta, normalizeRun, strengthMeta, subStateText } from "../../webrag/static/model.mjs";

test("the ADJUDICATED_UNSIDED sub-state, its label and caution are known and explained", () => {
  assert.match(subStateText("ADJUDICATED_UNSIDED"), /No passage was judged for or against/);
  assert.match(subStateText("ADJUDICATED"), /adjudication step judged one side decisive/);
  assert.doesNotMatch(subStateText("ADJUDICATED"), /opt-in/, "adjudication is part of the default workflow now");
  const meta = strengthMeta("adjudicated unsided");
  assert.equal(meta.tone, "uncertain");
  assert.ok(CAUTION_KINDS.includes("unsided_adjudication"));
  assert.equal(cautionMeta({ kind: "unsided_adjudication", detail: "x" }, {}).known, true);
});

test("an agentic verification record with a searched passage and a trace loads cleanly", () => {
  const run = normalizeRun({
    input: "Drug X lowers blood pressure.", mode: "verify", status: "COMPLETE", outcome: "SUPPORTED",
    settings: { verifier: "arm_c_studio" },
    sources: [{ id: "S1", title: "Trial (synthetic)", url: "https://example.org/a" },
              { id: "S2", title: "Meta-analysis (synthetic)", url: "https://example.org/b" }],
    claims: [{ text: "Drug X lowers blood pressure.", label: "SUPPORTED", sub_state: "", verifier: "arm_c_studio",
      verifier_note: "agentic verification (arm_c_studio, study arm C); 1 planner search(es) of this run's evidence, 1 new passage(s) judged; unanimous evidence; no adjudication",
      verifier_trace: { workflow: "arm_c_studio", study_arm: "C", actions: [{ turn: 0, action: "SEARCH", query: "drug X meta-analysis", new_passages: 1 }] },
      passages: [{ source_id: "S1", text: "Pharmacokinetics only.", stance: "NEUTRAL" },
                 { source_id: "S2", text: "Drug X reduced blood pressure.", stance: "SUPPORTS", found_by_search: "drug X meta-analysis" }] }],
  });
  assert.equal(run.verifier, "arm_c_studio");
  assert.equal(run.claims[0].label, "SUPPORTED");
  assert.deepEqual(run.claims[0].passages.map((p) => p.stance), ["NEUTRAL", "SUPPORTS"]);
  assert.equal(run.claims[0].passages[1].foundBySearch, "drug X meta-analysis");
  assert.deepEqual(run.warnings, []);
});
