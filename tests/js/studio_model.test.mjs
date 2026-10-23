// Studio data logic and API client. Run with: node --test tests/js
// Every record here is SYNTHETIC TEST DATA.
import assert from "node:assert/strict";
import { afterEach, test } from "node:test";

import { ApiError, api, apiUrl, pollJob, request } from "../../webrag/static/api.mjs";
import {
  activeDuplicate, bibliography, citationsFor, comparability, elapsedSince, filterSources, formatBytes, formatDate,
  formatMetric, formatSeconds, groupByStance, headlineMetrics, normalizeDocumentJob, normalizeRun, outcomeMeta,
  progressSteps, safeUrl, sourceJourney, sourceKind, sourceUsage, statusMeta, subStateText, validateQuestion, verdictMeta,
} from "../../webrag/static/model.mjs";

const SAMPLE = {
  input: "When did the bridge open?", mode: "research", status: "PARTIAL", outcome: "ANSWERED",
  sources: [
    { id: "S1", title: "Bridge", url: "https://example.org/bridge", access: "FULL_TEXT" },
    { id: "S2", title: "Unsafe", url: "javascript:alert(1)", access: "SNIPPET_ONLY" },
    { title: "no id" },
  ],
  claims: [
    { text: "It opened in 1932.", label: "SUPPORTED", cited_sources: ["S1", "S1"], supporting_sources: ["S1"],
      passages: [{ key: "S1:p1:a", source_id: "S1", page: 1, text: "opened in 1932", stance: "SUPPORTS", score: 0.5 },
        { key: "S9:p1:b", source_id: "S9", page: 0, text: "orphan", stance: "NEUTRAL" }] },
    { text: "It is the longest.", label: "INSUFFICIENT", sub_state: "CONFLICTING", cited_sources: ["S2"],
      supporting_sources: ["S1"], contradicting_sources: ["S2"], parse_ok: false },
    "not an object",
  ],
  evidence_pool: [{ key: "S1:p1:a", source_id: "S1", text: "opened in 1932" }],
  limitations: ["Only Wikipedia was searched.", 42],
  settings: { verifier: "quote_stance", model: "m" },
};

test("verdict, outcome and sub-state vocabulary comes from the backend and unknown values stay visible", () => {
  assert.equal(verdictMeta("SUPPORTED").tone, "support");
  assert.equal(verdictMeta("CONTRADICTED").tone, "contradict");
  assert.equal(verdictMeta("INSUFFICIENT").tone, "uncertain");
  const unknown = verdictMeta("MAYBE");
  assert.equal(unknown.tone, "neutral");
  assert.match(unknown.rule, /MAYBE/);
  assert.equal(outcomeMeta("ANSWERED").label, "Findings supported");
  assert.equal(outcomeMeta("INSUFFICIENT_EVIDENCE").tone, "uncertain");
  assert.equal(outcomeMeta("VERIFIED").label, "Verified (docqa)");
  assert.match(subStateText("CONFLICTING"), /disagree/);
  assert.match(subStateText("SOMETHING_NEW"), /something new/);
  assert.equal(subStateText(""), "");
});

test("normalizeRun is defensive about malformed records", () => {
  const run = normalizeRun(SAMPLE);
  assert.equal(run.sources.length, 2);
  assert.equal(run.sources[1].url, null);                       // non-http URL is never linked
  assert.equal(run.claims.length, 2);
  assert.deepEqual(run.claims[0].cited, ["S1"]);
  assert.equal(run.claims[0].passages[1].page, 1);
  assert.equal(run.findings.length, 1);
  assert.equal(run.unsettled[0].parseOk, false);
  assert.deepEqual(run.limitations, ["Only Wikipedia was searched.", "42"]);
  assert.equal(run.verifier, "quote_stance");
  assert.ok(run.warnings.some((w) => w.includes("S9")));
  const empty = normalizeRun(null);
  assert.equal(empty.claims.length, 0);
  assert.equal(empty.status, "RUNNING");
  assert.equal(empty.mode, "research");
  assert.equal(normalizeRun({ mode: "verify" }).mode, "verify");
});

test("citation chips say what the model cited and where the verifier found support", () => {
  const run = normalizeRun(SAMPLE);
  const [first] = citationsFor(run.claims[0], run.sourceIndex);
  assert.deepEqual([first.id, first.cited, first.supports, first.contradicts, first.known], ["S1", true, true, false, true]);
  const chips = citationsFor(run.claims[1], run.sourceIndex);
  assert.deepEqual(chips.map((c) => [c.id, c.cited, c.supports, c.contradicts]), [["S2", true, false, true], ["S1", false, true, false]]);
  const groups = groupByStance(run.claims[0].passages);
  assert.equal(groups.SUPPORTS.length, 1);
  assert.equal(groups.OTHER.length, 1);
});

test("headline metrics include only recorded values, in a fixed order", () => {
  const metrics = { n: 30, classification: { accuracy: { value: 0.8667, num: 26, den: 30 }, macro_f1: 0.8644 },
    contradiction_recall: { value: 0.7, num: 7, den: 10 }, quote_stance: { answer_correctness: { value: 0.6, num: 3, den: 5 } } };
  const rows = headlineMetrics(metrics);
  assert.deepEqual(rows.map((r) => r.key), ["n", "accuracy", "macro_f1", "contradiction_recall", "quote_stance.answer_correctness"]);
  assert.equal(rows[1].display, "0.8667 (26/30)");
  assert.equal(rows[2].display, "0.8644");
  assert.deepEqual(headlineMetrics(null), []);
  assert.equal(formatMetric(undefined), "not recorded");
  assert.equal(formatMetric({ value: 1, num: 9, den: 9 }), "1 (9/9)");
});

test("runs are comparable only with the same dataset and snapshot, and both complete", () => {
  const a = { comparison_group: "d|s", complete: true };
  assert.equal(comparability(a, { comparison_group: "d|s", complete: true }).comparable, true);
  assert.equal(comparability(a, { comparison_group: "d|t", complete: true }).comparable, false);
  assert.equal(comparability(a, { comparison_group: "d|s", complete: false }).comparable, false);
  assert.equal(comparability(a, null).comparable, false);
});

test("document job results are normalised", () => {
  const job = normalizeDocumentJob({ mode: "documents", status: "COMPLETE", result: { answer: "A", status: "PARTIALLY_VERIFIED",
    evidence: [{ id: "E1", document: "a.txt", page: 2, text: "t", score: 0.1, citation: "a.txt, p. 2" }],
    claims: [{ text: "c", cited: ["E1"], verdict: { label: "SUPPORTED", stances: { E1: "SUPPORTS" } } }, { text: "d", verdict: null }] } });
  assert.equal(job.qaStatus, "PARTIALLY_VERIFIED");
  assert.equal(job.claims[0].verdict.stances.E1, "SUPPORTS");
  assert.equal(job.claims[1].verdict, null);
  assert.equal(normalizeDocumentJob(undefined).evidence.length, 0);
});

test("input validation and formatting", () => {
  assert.match(validateQuestion("   "), /Enter/);
  assert.match(validateQuestion("x".repeat(2001)), /2000/);
  assert.equal(validateQuestion("A question?"), "");
  assert.equal(formatSeconds(4.25), "4.3 s");
  assert.equal(formatSeconds(125), "2 min 05 s");
  assert.equal(formatSeconds(undefined), "");
  assert.equal(formatBytes(2048), "2.0 KB");
  assert.equal(formatDate("not a date"), "not a date");
  assert.notEqual(formatDate("2026-10-18T18:45:01+0530", "en-GB"), "2026-10-18T18:45:01+0530");
  assert.equal(safeUrl("https://example.org/a b"), null);
  assert.equal(safeUrl(" http://example.org/x "), "http://example.org/x");
});

// ------------------------------------------------------------------------------------------------ sources, bibliography, progress
const SOURCED = {
  input: "Q", mode: "research", status: "COMPLETE",
  sources: [
    { id: "S1", title: "Bridge.", url: "https://example.org/bridge", provider: "wikipedia", kind: "encyclopedia", access: "FULL_TEXT",
      publisher: "Wikipedia", published: "2026-10-06T04:59:14Z", retrieved_at: "2026-10-18T10:00:00Z", chars: 1200 },
    { id: "S2", title: "Paper", url: "https://doi.org/10.1/x", provider: "openalex", kind: "scholarly", access: "ABSTRACT_ONLY" },
    { id: "S3", title: "Gone", url: "", provider: "arxiv", kind: "preprint", access: "FAILED", note: "Content not retrieved (http_404)." },
    { id: "S4", title: "Only listed", url: "https://example.org/x", provider: "brave", kind: "web", access: "SNIPPET_ONLY" },
  ],
  evidence_pool: [{ key: "S1:p1:a", source_id: "S1", text: "a" }, { key: "S2:p1:b", source_id: "S2", text: "b" }],
  claims: [{ text: "c", label: "SUPPORTED", cited_sources: ["S1"], supporting_sources: ["S1"],
    passages: [{ key: "S1:p1:a", source_id: "S1", text: "a", stance: "SUPPORTS" }, { key: "S1:p1:z", source_id: "S1", text: "z", stance: "NEUTRAL" }] }],
};

test("source usage is counted from recorded passages and claims only", () => {
  const run = normalizeRun(SOURCED);
  const usage = sourceUsage(run);
  assert.deepEqual([usage.get("S1").retrieved, usage.get("S1").cited, usage.get("S1").supports, usage.get("S1").used], [2, 1, 1, true]);
  assert.deepEqual([usage.get("S2").retrieved, usage.get("S2").used], [1, false]);
  assert.deepEqual([usage.get("S4").retrieved, usage.get("S4").used], [0, false]);
  const [found, read, retrieved, used] = sourceJourney(run.sources[0], usage.get("S1"));
  assert.deepEqual([found.state, read.state, retrieved.state, used.state], ["done", "done", "done", "done"]);
  assert.match(read.detail, /1,200 characters/);
  assert.equal(sourceJourney(run.sources[1], usage.get("S2"))[1].state, "partial");
  const failed = sourceJourney(run.sources[2], usage.get("S3"));
  assert.deepEqual([failed[1].label, failed[1].state, failed[2].state], ["Not read", "failed", "none"]);
  assert.deepEqual(filterSources(run.sources, usage, { filter: "used" }).map((s) => s.id), ["S1"]);
  assert.deepEqual(filterSources(run.sources, usage, { filter: "partial" }).map((s) => s.id), ["S2", "S4"]);
  assert.deepEqual(filterSources(run.sources, usage, { filter: "unread" }).map((s) => s.id), ["S3"]);
  assert.deepEqual(filterSources(run.sources, usage, { query: "PAPER" }).map((s) => s.id), ["S2"]);
  assert.equal(sourceKind("encyclopedia").label, "Encyclopedia");
  assert.equal(sourceKind("podcast").label, "Podcast transcript");
  assert.equal(sourceKind("data_set").label, "data set");               // unknown kinds stay visible, never re-labelled
});

test("bibliography orders sources by use and never fills in missing fields", () => {
  const run = normalizeRun(SOURCED);
  const entries = bibliography(run);
  assert.deepEqual(entries.map((e) => [e.id, e.group]), [["S1", "cited"], ["S2", "consulted"], ["S3", "listed"], ["S4", "listed"]]);
  assert.equal(entries[0].citation, "Bridge. Wikipedia, 2026-10-06. https://example.org/bridge (accessed 2026-10-18)");
  assert.equal(entries[1].citation, "Paper. https://doi.org/10.1/x");     // no publisher, date or access date recorded
  assert.equal(entries[2].citation, "Gone.");
});

test("progress steps use reported stages and only name the ones not started", () => {
  const stages = [{ name: "model check", status: "ok", seconds: 0.1 }, { name: "plan", status: "partial", seconds: 3, detail: "fallback" }];
  const running = progressSteps("research", stages, { state: "running", current: "search: searching wikipedia" });
  assert.deepEqual(running.map((s) => s.state), ["ok", "partial", "active", "pending", "pending", "pending", "pending"]);
  assert.equal(running[2].name, "search and fetch");
  assert.equal(running[2].detail, "search: searching wikipedia");
  assert.ok(running.every((s) => !("percent" in s)));
  const queued = progressSteps("verify", [], { state: "queued" });
  assert.equal(queued.length, 6);
  assert.ok(queued.every((s) => s.state === "pending"));
  const done = progressSteps("research", stages, { state: "done" });
  assert.deepEqual(done.map((s) => s.name), ["model check", "plan"]);   // a finished run shows only what happened
  const extra = progressSteps("research", [...stages, { name: "search and fetch", status: "ok" }, { name: "index", status: "ok" },
    { name: "retrieve", status: "ok" }, { name: "synthesize", status: "ok" }, { name: "verify", status: "ok" }, { name: "follow-up", status: "ok" }], { state: "running" });
  assert.equal(extra.at(-2).label, "Follow-up search");
  assert.deepEqual([extra.at(-1).label, extra.at(-1).state], ["Finishing up", "active"]);
  assert.equal(statusMeta("CANCELLED").tone, "neutral");
});

test("duplicate detection and elapsed time", () => {
  const jobs = [{ id: "a", state: "running", mode: "research", input: "  When did the  Bridge open? " },
    { id: "b", state: "done", mode: "research", input: "Other" }, { id: "c", state: "cancelled", mode: "verify", input: "Claim" }];
  assert.equal(activeDuplicate(jobs, "when did the bridge open?", "research").id, "a");
  assert.equal(activeDuplicate(jobs, "when did the bridge open?", "verify"), null);
  assert.equal(activeDuplicate(jobs, "Claim", "verify"), null);           // cancelled jobs are not duplicates
  assert.equal(activeDuplicate(null, "x", "research"), null);
  assert.equal(elapsedSince(100, 160_000), 60);
  assert.equal(elapsedSince("2026-10-18T10:00:00Z", Date.parse("2026-10-18T10:01:30Z")), 90);
  assert.equal(elapsedSince("not a time"), null);
});

// ------------------------------------------------------------------------------------------------ API client
const realFetch = globalThis.fetch;
afterEach(() => { globalThis.fetch = realFetch; });
const reply = (status, body) => async () => ({ ok: status >= 200 && status < 300, status, text: async () => body });

test("request turns every failure into a readable ApiError", async () => {
  globalThis.fetch = reply(200, '{"ok": true}');
  assert.deepEqual(await request("/x"), { ok: true });
  globalThis.fetch = reply(400, '{"error": "enter a question"}');
  await assert.rejects(request("/x"), (e) => e instanceof ApiError && e.status === 400 && e.message === "enter a question");
  globalThis.fetch = reply(502, "<html>bad gateway</html>");
  await assert.rejects(request("/x"), (e) => e.kind === "format" && e.status === 502);
  globalThis.fetch = reply(200, "");
  await assert.rejects(request("/x"), (e) => e.kind === "format");
  globalThis.fetch = async () => { throw new TypeError("fetch failed"); };
  await assert.rejects(request("/x"), (e) => e.kind === "network" && /could not be reached/.test(e.message));
});

test("request reads the status a hosting gateway relays in X-Studio-Status", async () => {
  const relay = (status, real, body) => async () => ({ ok: false, status, text: async () => body,
    headers: { get: (name) => (name === "X-Studio-Status" ? real : null) } });
  globalThis.fetch = relay(400, "409", '{"error": "still running", "state": "running"}');
  await assert.rejects(request("/x"), (e) => e.status === 409 && e.message === "still running" && e.payload.state === "running");
  globalThis.fetch = relay(400, "not a number", '{"error": "bad input"}');
  await assert.rejects(request("/x"), (e) => e.status === 400 && e.message === "bad input");
});

test("apiUrl keeps root paths outside a browser and resolves them against the page inside one", () => {
  assert.equal(apiUrl("/api/status"), "/api/status");
  const saved = globalThis.location;
  globalThis.location = { pathname: "/~/+/studio/" };
  try {
    assert.equal(apiUrl("/api/runs/a"), "/~/+/studio/api/runs/a");
    assert.equal(apiUrl("relative"), "relative");
  } finally {
    globalThis.location = saved;
  }
});

test("pollJob follows a job to completion and stops on 404", async () => {
  const states = ["queued", "running", "done"];
  globalThis.fetch = async () => ({ ok: true, status: 200, text: async () => JSON.stringify({ id: "j", state: states.shift(), data: {} }) });
  const seen = [];
  const job = await pollJob("j", (j) => seen.push(j.state), { interval: 1 });
  assert.equal(job.state, "done");
  assert.deepEqual(seen, ["queued", "running", "done"]);
  globalThis.fetch = reply(404, '{"error": "unknown run"}');
  await assert.rejects(pollJob("gone", () => {}, { interval: 1 }), (e) => e.status === 404);
});

test("pollJob treats a cancelled job as final", async () => {
  globalThis.fetch = async () => ({ ok: true, status: 200, text: async () => JSON.stringify({ id: "j", state: "cancelled", data: {} }) });
  const seen = [];
  assert.equal((await pollJob("j", (j) => seen.push(j.state), { interval: 1 })).state, "cancelled");
  assert.deepEqual(seen, ["cancelled"]);
});

test("aborting pollJob stops it at once and drops a response that arrives late", async () => {
  const controller = new AbortController();
  let resolveFetch;
  globalThis.fetch = (url, { signal }) => new Promise((resolve, reject) => {
    resolveFetch = () => resolve({ ok: true, status: 200, text: async () => JSON.stringify({ id: "j", state: "running", data: {} }) });
    signal.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
  });
  const seen = [];
  const polling = pollJob("j", (j) => seen.push(j.state), { interval: 1, signal: controller.signal });
  controller.abort();
  resolveFetch();
  assert.equal(await polling, null);
  assert.deepEqual(seen, []);
});

test("cancel posts to the cancel route and a refusal keeps the server's explanation", async () => {
  const calls = [];
  globalThis.fetch = async (url, init) => { calls.push([url, init.method]);
    return { ok: false, status: 409, text: async () => JSON.stringify({ id: "j", state: "running", cancelled: false, error: "This run has already started." }) }; };
  await assert.rejects(api.cancel("j"), (e) => e.status === 409 && e.payload.state === "running" && e.message === "This run has already started.");
  assert.deepEqual(calls, [["/api/run/j/cancel", "POST"]]);
});
