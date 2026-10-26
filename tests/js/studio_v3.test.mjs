// deep-research-v3 records in the studio: the query ledger, provider coverage, the evidence review, statement IDs and
// several claims per verification run. Run with: node --test tests/js
// Every record here is SYNTHETIC TEST DATA shaped like webrag/deep.py output; nothing was produced by a real run.
import assert from "node:assert/strict";
import { test } from "node:test";

import { normalizeRun, outcomeMeta, providerStateMeta, queryStatusMeta } from "../../webrag/static/model.mjs";

const RUN = {
  input: "Do graph models detect fraud better?", mode: "research", status: "COMPLETE", outcome: "ANSWERED", depth: "quick",
  deep_version: "deep-research-v3",
  sources: [{ id: "S1", title: "Review", access: "FULL_TEXT" }],
  claims: [{ id: "C1", text: "Graph models detect fraud rings.", label: "SUPPORTED", cautions: [] }],
  queries: [
    { id: "Q1", round: 1, round_kind: "search", query: "graph fraud detection", category: "scholarly", purpose: "answer", subquestion: 0,
      status: "executed", reason: "", not_sent_to: [], searches: [{ provider: "openalex", requested: 10, returned: 7, error: "" }] },
    { id: "Q2", round: 1, round_kind: "search", query: "graph fraud overstated", category: "scholarly", purpose: "counter", subquestion: 0,
      status: "executed", reason: "", not_sent_to: [], searches: [{ provider: "openalex", requested: 10, returned: 0, error: "" }] },
    { id: "Q3", round: 1, round_kind: "search", query: "fraud graph limits", category: "scholarly", purpose: "answer", subquestion: 1,
      status: "skipped", reason: "the quick depth runs 4 queries per round", not_sent_to: [], searches: [] },
    { id: "Q4", round: 1, query: "agency statistics", category: "official", purpose: "answer", status: "no_provider",
      reason: "no active provider serves the 'official' source type", searches: "not a list" },
    "not a query",
  ],
  provider_coverage: [
    { id: "openalex", name: "OpenAlex", state: "searched", detail: "searched in this run", queries: 3, failed: 0, hits: 7, categories: ["scholarly"] },
    { id: "brave", name: "Brave Search API", state: "not_configured", detail: "switched on, but its settings are missing", missing: ["BRAVE_API_KEY environment variable"] },
    { id: "custom", name: "Custom", state: "mystery_state" },
  ],
  review: [
    { claim: "C1", kind: "abstract_only", detail: "Supported only by abstracts.", action: "needs_evidence", refs: ["S1:p1:a"] },
    { claim: null, kind: "query_not_run", detail: "A planned query was skipped.", action: "note", refs: ["Q3"] },
    "not an item",
  ],
};

test("the query ledger keeps every status, including zero results, skipped and no-provider queries", () => {
  const run = normalizeRun(RUN);
  assert.equal(run.queries.length, 4, "a malformed entry is dropped");
  assert.deepEqual(run.queries.map((q) => q.status), ["executed", "executed", "skipped", "no_provider"]);
  assert.equal(run.queries[1].purpose, "counter");
  assert.equal(run.queries[1].searches[0].returned, 0, "zero results is still an executed search");
  assert.equal(run.queries[2].reason, "the quick depth runs 4 queries per round");
  assert.deepEqual(run.queries[3].searches, [], "a malformed search list becomes empty");
  assert.equal(queryStatusMeta("skipped").label, "Not run");
  assert.equal(queryStatusMeta("failed").tone, "error");
  assert.deepEqual(queryStatusMeta("paused"), { key: "paused", label: "paused", tone: "neutral" }, "unknown statuses stay visible");
});

test("deep-research-v4 ledger fields are kept, and are null in older records", () => {
  const run = normalizeRun({ ...RUN, queries: [
    { id: "Q1", query: "graph fraud detection", status: "executed", allocated: true, tagged_by: "inferred", relevant_candidate: false,
      reason: "no result cleared the relevance floor, so none was read for this query in particular", searches: [] },
    { id: "Q2", query: "graph fraud comparison", status: "skipped", allocated: false, tagged_by: "planner", searches: [] },
    { id: "Q3", query: "old record", status: "executed", searches: [] },
  ] });
  assert.deepEqual(run.queries.map((q) => [q.allocated, q.taggedBy, q.relevantCandidate]),
    [[true, "inferred", false], [false, "planner", null], [null, "", null]]);
  assert.match(run.queries[0].reason, /relevance floor/);
});

test("provider coverage separates searched providers from ones that were not asked", () => {
  const run = normalizeRun(RUN);
  assert.deepEqual(run.providerCoverage.map((p) => p.state), ["searched", "not_configured", "mystery_state"]);
  assert.equal(run.providerCoverage[0].hits, 7);
  assert.deepEqual(run.providerCoverage[1].missing, ["BRAVE_API_KEY environment variable"]);
  assert.equal(providerStateMeta("not_configured").label, "Not configured");
  assert.equal(providerStateMeta("blocked_by_policy").label, "Blocked by network policy");
  assert.equal(providerStateMeta("mystery_state").label, "mystery state");
});

test("the evidence review and statement IDs are kept as recorded", () => {
  const run = normalizeRun(RUN);
  assert.equal(run.claims[0].id, "C1");
  assert.equal(run.review.length, 2);
  assert.deepEqual(run.review[0], { claim: "C1", kind: "abstract_only", detail: "Supported only by abstracts.", action: "needs_evidence", refs: ["S1:p1:a"] });
  assert.equal(run.review[1].claim, "", "a run-level item names no statement");
});

test("a run without v3 fields (deep-research-v1 or v2, or the single-pass pipeline) gets empty lists", () => {
  const run = normalizeRun({ input: "q", mode: "research", depth: "standard", claims: [{ text: "x", label: "SUPPORTED" }] });
  assert.deepEqual(run.queries, []);
  assert.deepEqual(run.providerCoverage, []);
  assert.deepEqual(run.review, []);
  assert.equal(run.claimsMode, null);
  assert.equal(run.claims[0].id, "");
});

test("several claims in one verification run keep their own verdicts and no overall verdict", () => {
  const run = normalizeRun({
    input: "Claim one.\nClaim two.", mode: "verify", status: "COMPLETE", outcome: "CLAIMS_ASSESSED", depth: "quick",
    claims: [{ id: "C1", text: "Claim one.", label: "SUPPORTED" }, { id: "C2", text: "Claim two.", label: "INSUFFICIENT", sub_state: "CONFLICTING" }],
    summary: { available: true, claims: 2, verdicts: { SUPPORTED: 1, INSUFFICIENT: 1 } },
    claims_mode: { claims: 2, limit: 8, dropped: 0 },
  });
  assert.deepEqual(run.claims.map((c) => [c.id, c.label]), [["C1", "SUPPORTED"], ["C2", "INSUFFICIENT"]]);
  assert.deepEqual(run.summary.verdicts, { SUPPORTED: 1, INSUFFICIENT: 1 });
  assert.deepEqual(run.claimsMode, { claims: 2, limit: 8, dropped: 0 });
  assert.equal(outcomeMeta("CLAIMS_ASSESSED").label, "Claims assessed");
  assert.match(outcomeMeta("CLAIMS_ASSESSED").description, /no overall verdict/);
  const single = normalizeRun({ mode: "verify", outcome: "SUPPORTED", summary: { available: true, verdict: "SUPPORTED" } });
  assert.equal(single.summary.verdicts, null);
});
