// Deep-research data logic of the studio. Run with: node --test tests/js
// Every record here is SYNTHETIC TEST DATA shaped like webrag/deep.py output; nothing was produced by a real run.
import assert from "node:assert/strict";
import { test } from "node:test";

import {
  budgetRows, completenessMeta, coverageMeta, describeBudget, findingsBySubquestion, independenceText, noAnswerState,
  normalizeRun, progressSteps, providerState, sourceJourney, stageLabel, strengthMeta,
} from "../../webrag/static/model.mjs";
import { normalizePreferences } from "../../webrag/static/prefs.mjs";

const LIMITS = { rounds: 2, queries_per_round: 6, searches: 24, searches_per_round: 14, sources: 16, full_text: 8, chars: 700000,
  seconds: 900, subquestions: 4, findings: 8, counter_evidence: 2 };

const DEEP = {
  input: "Do heat pumps cut household emissions?", mode: "research", status: "COMPLETE", outcome: "ANSWERED", depth: "standard",
  options: { depth: "standard", providers: ["wikipedia", "openalex"], categories: ["scholarly"], recency: "5y" },
  budget: { depth: "standard", limits: LIMITS, used: { rounds: 2, queries: 9, searches: 15, sources: 11, full_text: 3, chars: 123456 }, elapsed: 312.4 },
  sources: [
    { id: "S1", title: "Heat pump", access: "FULL_TEXT", category: "encyclopedia", discovered_by: ["wikipedia"] },
    { id: "S2", title: "Field trial", access: "ABSTRACT_ONLY", category: "scholarly", authors: ["A. Author"], identifiers: { doi: "10.1/x" } },
    { id: "S3", title: "Paywalled study", access: "METADATA_ONLY", category: "scholarly", retrieval_error: "no abstract or open-access copy", discovered_by: ["crossref", "openalex"] },
    { id: "S4", title: "Episode 12", access: "FULL_TEXT", category: "podcast", page_labels: ["0:00–1:00", "1:00–2:00"] },
  ],
  claims: [
    { text: "They cut emissions in cold climates.", label: "SUPPORTED", subquestion: 1, strength: "corroborated",
      supporting_sources: ["S1", "S2"], independent_support: 2, citations: { checked: 2, problems: [], unlocated: ["S2:p1:a"], ok: true },
      passages: [{ key: "S4:p2:a", source_id: "S4", page: 2, location: "1:00–2:00", text: "cut emissions", stance: "SUPPORTS" }] },
    { text: "Savings depend on the grid.", label: "INSUFFICIENT", sub_state: "CONFLICTING", subquestion: 2, strength: "contested",
      citations: { problems: ["S9:p1:a: unknown source S9"], unlocated: [] } },
    { text: "Main answer.", label: "SUPPORTED", subquestion: 0, strength: "single source" },
    { text: "Untagged.", label: "SUPPORTED" },
  ],
  ledger: [
    { index: 0, text: "Do heat pumps cut household emissions?", status: "partially answered", independent_origins: 1, evidence_state: "weak",
      categories: ["encyclopedia"], queries: ["heat pump emissions"], discovered: ["S1"], retrieved: ["S1"], supporting: ["Main answer."] },
    { index: 1, text: "In cold climates?", status: "answered", independent_origins: 2, evidence_state: "strong", next_action: "" },
    { index: 2, text: "Does the grid matter?", status: "contested", next_action: "Search official sources" },
  ],
  rounds: [
    { number: 1, searches: 8, hits: 30, new_sources: 7, indexed_sources: 4, errors: 1, filtered_by_date: 2, new_relevant_passages: 6,
      seconds: 41.2, requests: 19, queries: [{ query: "heat pump emissions", category: "scholarly", purpose: "answer", subquestion: 0, providers: ["openalex"] }],
      provider_status: { openalex: { queries: 4, succeeded: 3, failed: 1, hits: 20 }, broken: "not an object" },
      coverage: [{ subquestion: 0, state: "weak", origins: 1 }], access: { FULL_TEXT: 2, ABSTRACT_ONLY: 2, METADATA_ONLY: 1 } },
    { number: 2, purpose: "counter-evidence", queries: [], provider_status: null },
  ],
  stopping: { reason: "round_limit", detail: "The standard depth allows 2 search round(s).", completeness: "budget_limited" },
  summary: { available: true, answered: true, dropped: 1, direct_answer: { text: "Mostly yes.", claims: [0, 2, "x"] },
    summary: [{ text: "Cold climates benefit.", claims: [0] }, { text: "" }, "bad"] },
  counter_evidence: { skipped: "", queries: [{ query: "heat pump criticism", category: "web" }],
    changes: [{ text: "Main answer.", changed: true, reason: "SUPPORTED -> INSUFFICIENT", before: { label: "SUPPORTED" }, after: { label: "INSUFFICIENT", sub_state: "CONFLICTING" } }] },
  source_quality: { discovered: 4, read: 3, by_access: { FULL_TEXT: 2, ABSTRACT_ONLY: 1, SNIPPET_ONLY: 0, METADATA_ONLY: 1, FAILED: 0 } },
  research_plan: { central_question: "Do heat pumps cut household emissions?", subquestions: [{ text: "In cold climates?", categories: ["scholarly"] }], searched_categories: ["encyclopedia", "scholarly"] },
  evidence_policy: { FULL_TEXT: "Substantive text.", labels: { corroborated: "Two origins." } },
};

test("normalizeRun keeps the deep-research record and drops malformed parts", () => {
  const run = normalizeRun(DEEP);
  assert.equal(run.depth, "standard");
  assert.deepEqual(run.options.categories, ["scholarly"]);
  assert.equal(run.stopping.completeness, "budget_limited");
  assert.equal(run.ledger.length, 3);
  assert.equal(run.ledger[0].distinctTexts, 1, "an older record counted distinct texts, and is shown as such");
  assert.equal(run.ledger[0].publishers, 0);
  assert.deepEqual(run.ledger[0].queries, ["heat pump emissions"]);
  assert.equal(run.ledger[2].nextAction, "Search official sources");
  assert.equal(run.rounds[0].providerStatus.length, 1, "a provider status that is not an object is dropped");
  assert.deepEqual(run.rounds[0].providerStatus[0], { name: "openalex", queries: 4, succeeded: 3, failed: 1, hits: 20 });
  assert.equal(run.rounds[0].requests, 19);
  assert.equal(run.rounds[0].filtered, 2);
  assert.equal(run.rounds[1].purpose, "counter-evidence");
  assert.deepEqual(run.rounds[1].providerStatus, []);
  assert.deepEqual(run.summary.directAnswer.claims, [0, 2], "only integer claim indexes are kept");
  assert.equal(run.summary.points.length, 1, "empty and malformed summary points are dropped");
  assert.equal(run.summary.dropped, 1);
  assert.equal(run.counter.changes[0].before, "SUPPORTED");
  assert.equal(run.counter.changes[0].after, "INSUFFICIENT");
  assert.equal(run.counter.changes[0].afterSub, "CONFLICTING");
  assert.equal(run.quality.by_access.METADATA_ONLY, 1);
  assert.equal(run.evidencePolicy.FULL_TEXT, "Substantive text.");
  const [first, second, , untagged] = run.claims;
  assert.equal(first.strength, "corroborated");
  assert.equal(first.subquestion, 1);
  assert.deepEqual(first.unlocated, ["S2:p1:a"]);
  assert.equal(first.passages[0].location, "1:00–2:00");
  assert.deepEqual(second.citationProblems, ["S9:p1:a: unknown source S9"]);
  assert.equal(untagged.subquestion, -1);
  const s4 = run.sourceIndex.get("S4");
  assert.deepEqual(s4.pageLabels, ["0:00–1:00", "1:00–2:00"]);
  assert.equal(run.sourceIndex.get("S2").identifiers.doi, "10.1/x");
});

test("a legacy run has no deep fields", () => {
  const run = normalizeRun({ input: "q", mode: "research", claims: [{ text: "x", label: "SUPPORTED" }] });
  assert.equal(run.depth, "");
  assert.equal(run.stopping, null);
  assert.equal(run.summary, null);
  assert.equal(run.counter, null);
  assert.deepEqual(run.ledger, []);
  assert.deepEqual(run.rounds, []);
  assert.equal(run.claims[0].strength, "");
});

test("findings are grouped by the question they answer, untagged ones under the main question", () => {
  const groups = findingsBySubquestion(normalizeRun(DEEP));
  assert.deepEqual(groups.map((g) => g.index), [0, 1, 2]);
  assert.equal(groups[0].title, "The main question");
  assert.deepEqual(groups[0].claims.map((c) => c.text), ["Main answer.", "Untagged."]);
  assert.equal(groups[1].title, "In cold climates?");
  assert.equal(groups[2].title, "Does the grid matter?");
  const orphan = findingsBySubquestion(normalizeRun({ claims: [{ text: "x", label: "SUPPORTED", subquestion: 5 }] }));
  assert.equal(orphan[0].title, "Question 5", "a subquestion missing from the ledger keeps a neutral title");
});

test("budget rows show only what was reported", () => {
  const rows = budgetRows(DEEP.budget);
  assert.deepEqual(rows.map((r) => r.label), ["Search rounds", "Searches", "Sources", "Full-text attempts", "Elapsed seconds (search cut-off)"]);
  assert.deepEqual(rows[1], { label: "Searches", used: 15, limit: 24 });
  assert.equal(rows[4].used, 312);
  assert.deepEqual(budgetRows(null), []);
  assert.deepEqual(budgetRows({ limits: LIMITS }), [], "no used counters, no rows");
  assert.deepEqual(budgetRows({ limits: { rounds: 1 }, used: { rounds: "two" } }), [], "non-numbers are not shown");
  assert.match(describeBudget(LIMITS), /^2 rounds, up to 24 searches and 16 sources, 8 full-text attempts; searching stops after 15 min$/);
  assert.equal(describeBudget(null), "");
});

test("evidence, coverage and completeness labels fall back to the recorded value", () => {
  assert.equal(strengthMeta("corroborated").label, "Corroborated");
  assert.equal(strengthMeta("contradicted").tone, "contradict");
  assert.equal(strengthMeta("").label, "");
  assert.equal(strengthMeta("brand_new").label, "brand new");
  assert.equal(coverageMeta("not reached").tone, "error");
  assert.equal(coverageMeta("").label, "Unknown");
  assert.equal(completenessMeta("access_limited").label, "Access-limited");
  assert.equal(completenessMeta("").label, "");
});

test("provider states come only from the reported flags", () => {
  assert.equal(providerState({ active: true, enabled: true, configured: true }).state, "active");
  const setup = providerState({ active: false, enabled: true, configured: false, missing: ["BRAVE_API_KEY environment variable"] });
  assert.equal(setup.state, "needs_setup");
  assert.match(setup.detail, /BRAVE_API_KEY/);
  assert.match(providerState({ enabled: false, configured: true, controlled_by: "WEBRAG_PROVIDERS" }).detail, /WEBRAG_PROVIDERS/);
  assert.equal(providerState({ enabled: false, configured: true, controlled_by: "settings" }).detail, "Switched off.");
  assert.match(providerState({ enabled: false, configured: false, missing: ["a feed"] }).detail, /also needs a feed/);
  assert.equal(providerState(null).state, "unknown");
});

test("round stages get readable labels and a deep run has no invented step list", () => {
  assert.equal(stageLabel("round 2: search and read", "deep"), "Round 2: search and read sources");
  assert.equal(stageLabel("round 1: coverage", "deep"), "Round 1: assess coverage per question");
  assert.equal(stageLabel("round 3: follow-up queries", "deep"), "Round 3: plan follow-up searches");
  assert.equal(stageLabel("verify", "deep"), "Verify each finding");
  assert.equal(stageLabel("verify", "deep-verify"), "Judge passages against the claim");
  assert.equal(stageLabel("counter-evidence", "deep"), "Search for counter-evidence and re-check");
  const steps = progressSteps("deep", [{ name: "plan", status: "ok" }], { state: "running", current: "round 1" });
  assert.deepEqual(steps.map((s) => s.label), ["Understand the question", "In progress"]);
  assert.equal(steps[1].detail, "round 1");
  assert.deepEqual(progressSteps("deep", [{ name: "plan" }], { state: "done" }).map((s) => s.name), ["plan"]);
});

test("a metadata-only source is shown as found but not read, with the reason and every provider that found it", () => {
  const run = normalizeRun(DEEP);
  const journey = sourceJourney(run.sourceIndex.get("S3"), null);
  assert.equal(journey[0].detail, "search result from crossref, openalex");
  assert.equal(journey[1].label, "Metadata only");
  assert.equal(journey[1].state, "none");
  assert.match(journey[1].detail, /no abstract or open-access copy/);
});

test("the default depth preference accepts only the three depths", () => {
  assert.equal(normalizePreferences({}).defaultDepth, "standard");
  assert.equal(normalizePreferences({ defaultDepth: "deep" }).defaultDepth, "deep");
  assert.equal(normalizePreferences({ defaultDepth: "infinite" }).defaultDepth, "standard");
});

test("independence is read from the record and described without overstating it", () => {
  const run = normalizeRun({ mode: "research", depth: "standard", claims: [
    { text: "A.", label: "SUPPORTED", independence: { status: "established", independent: 2, groups: 3, sources: 4 } },
    { text: "B.", label: "SUPPORTED", independence: { status: "unknown", independent: 1, groups: 2, sources: 2 } },
    { text: "C.", label: "SUPPORTED", independence: { status: "single", independent: 1, groups: 1, sources: 2 } },
    { text: "D.", label: "SUPPORTED" }],
    ledger: [{ index: 0, text: "Q", status: "partially answered", independent_publishers: 1, independence: "unknown" }] });
  assert.deepEqual(run.claims[0].independence, { status: "established", independent: 2, groups: 3, sources: 4 });
  assert.equal(independenceText(run.claims[0].independence), "2 publishers established as independent");
  assert.equal(independenceText(run.claims[1].independence), "2 sources from 2 publishers, independence not established");
  assert.equal(independenceText(run.claims[2].independence), "one publisher (2 sources)");
  assert.equal(run.claims[3].independence, null, "a record without the assessment shows the original count instead");
  assert.equal(run.ledger[0].independence, "unknown");
  assert.equal(run.ledger[0].distinctTexts, null);
  assert.equal(strengthMeta("independence unknown").label, "Independence unknown");
  assert.notEqual(strengthMeta("independence unknown").tone, "support");
});

test("a completed run without readable evidence is not described as failed or answered", () => {
  const empty = normalizeRun({ mode: "research", depth: "standard", status: "COMPLETE", outcome: "INSUFFICIENT_EVIDENCE",
    error: "The searches returned no results for this input.", claims: [], summary: { available: false, reason: "No finding was verified." } });
  const state = noAnswerState(empty);
  assert.equal(state.failed, false);
  assert.equal(state.answered, false);
  assert.equal(state.insufficient, true);
  assert.equal(state.title, "No answer could be supported");
  assert.equal(state.tone, "warn");
  assert.equal(state.reason, "The searches returned no results for this input.");
  assert.equal(empty.summary.directAnswer, null, "no direct answer is invented");
  const verify = noAnswerState(normalizeRun({ mode: "verify", status: "COMPLETE", outcome: "INSUFFICIENT", error: "No readable source." }));
  assert.equal(verify.title, "No verdict could be supported");
  assert.equal(verify.failed, false);
  const failed = noAnswerState(normalizeRun({ mode: "research", status: "FAILED", error: "Unexpected error: RuntimeError: x" }));
  assert.equal(failed.failed, true);
  assert.equal(failed.title, "The run failed");
  assert.equal(failed.tone, "error");
  const answered = noAnswerState(normalizeRun({ mode: "research", status: "COMPLETE", outcome: "ANSWERED",
    claims: [{ text: "A.", label: "SUPPORTED" }] }));
  assert.equal(answered.answered, true);
  assert.equal(answered.insufficient, false);
});

test("an excluded provider is shown as not used as evidence, whatever its other flags", () => {
  const wiki = providerState({ id: "wikipedia", evidence: false, enabled: true, active: true, configured: true,
    policy: "Wikipedia is not accepted as an evidence source." });
  assert.equal(wiki.state, "excluded");
  assert.equal(wiki.label, "Not used as evidence");
  assert.match(wiki.detail, /not accepted as an evidence source/);
  assert.equal(providerState({ evidence: true, active: true }).state, "active");
});
