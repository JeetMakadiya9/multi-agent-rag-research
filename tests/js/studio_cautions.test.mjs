// Evidence-quality cautions in the studio: kept through normalizeRun and shown beside their finding. Run with:
// node --test tests/js
// Every record here is SYNTHETIC TEST DATA shaped like webrag/deep.py output (claims[].cautions = [{ kind, detail }]);
// nothing was produced by a real run.
import assert from "node:assert/strict";
import { test } from "node:test";

// A minimal stand-in for the browser DOM, enough for dom.mjs `el` and `icon`: elements, text nodes and attributes.
class FakeNode {
  constructor(tag) { this.tagName = tag; this.children = []; this.attributes = {}; this.className = ""; this.dataset = {}; }
  append(...items) { for (const item of items) this.children.push(item instanceof FakeNode ? item : new FakeText(String(item))); }
  setAttribute(key, value) { this.attributes[key] = String(value); }
  getAttribute(key) { return key in this.attributes ? this.attributes[key] : null; }
  addEventListener() {}
  get textContent() { return this.children.map((c) => c.textContent).join(""); }
  find(tag) { return this.children.filter((c) => c.tagName === tag); }
}
class FakeText extends FakeNode {
  constructor(value) { super("#text"); this.value = value; }
  get textContent() { return this.value; }
}
globalThis.Node = FakeNode;
globalThis.document = { createElement: (tag) => new FakeNode(tag), createElementNS: (_, tag) => new FakeNode(tag) };

const { cautionBadge, cautionList } = await import("../../webrag/static/components.mjs");
const { CAUTION_KINDS, cautionMeta, normalizeRun } = await import("../../webrag/static/model.mjs");

const FRAGMENT = { kind: "fragment_subject", detail: "The best-matching supporting text (S2:p2:ab12) begins mid-word (\"els outperform traditional methods.\"), so the subject it describes is not in the retrieved passage." };
const HEDGE = { kind: "hedge_dropped", detail: "The supporting text is hedged ('may': \"Graph learning may improve early identification.\"), but the finding states it without a hedge." };
const CAUSAL = { kind: "causal_added", detail: "The finding says 'leads to', but its best-matching supporting text states no cause (it describes an association)." };
const POLICY = { cautions: { fragment_subject: "Policy text for fragments.", hedge_dropped: "Policy text for hedges.",
  causal_added: "Policy text for causes.", _rule: "A caution is a warning, not a verdict." } };

const RUN = {
  input: "Do graph neural networks detect fraud better?", mode: "research", status: "COMPLETE", outcome: "ANSWERED", depth: "quick",
  evidence_policy: POLICY,
  sources: [{ id: "S1", title: "Review", access: "FULL_TEXT" }, { id: "S2", title: "Study", access: "FULL_TEXT" }],
  claims: [
    { text: "Graph neural networks outperform traditional methods.", label: "SUPPORTED", strength: "single source", cautions: [FRAGMENT] },
    { text: "Graph learning improves early identification.", label: "SUPPORTED", cautions: [HEDGE] },
    { text: "Shared devices lead to higher fraud rates.", label: "SUPPORTED", cautions: [HEDGE, CAUSAL] },
    { text: "Graph neural networks model accounts as nodes.", label: "SUPPORTED", strength: "single source", cautions: [] },
    { text: "A record from deep-research-v1.", label: "SUPPORTED" },
  ],
  summary: { available: true, answered: true, direct_answer: { text: "Graph neural networks model accounts as nodes.", claims: [3] },
    summary: [{ text: "Graph neural networks model accounts as nodes.", claims: [3] }], dropped: 0 },
};

test("each recorded caution kind is kept through normalizeRun, in order and with its explanation", () => {
  const run = normalizeRun(RUN);
  assert.deepEqual(run.claims[0].cautions, [FRAGMENT]);
  assert.deepEqual(run.claims[1].cautions, [HEDGE]);
  assert.deepEqual(run.claims[2].cautions, [HEDGE, CAUSAL], "several cautions on one finding are all kept");
  assert.deepEqual(CAUTION_KINDS, ["fragment_subject", "hedge_dropped", "causal_added", "number_unsupported",
    "qualifier_dropped", "attribution_dropped", "support_unmatched", "adjudicated_conflict"]);   // last: opt-in adjudicated verifier
  assert.deepEqual(run.warnings, []);
});

test("missing or empty cautions give no caution and no record warning", () => {
  const run = normalizeRun(RUN);
  assert.deepEqual(run.claims[3].cautions, []);
  assert.deepEqual(run.claims[4].cautions, [], "a v1 record without the field");
  assert.equal(cautionList(run.claims[3].cautions, run.evidencePolicy), null);
  assert.equal(cautionList(run.claims[4].cautions, run.evidencePolicy), null);
  assert.deepEqual(run.warnings, []);
});

test("malformed caution data never crashes or invents a caution; unreadable entries are reported", () => {
  const run = normalizeRun({ ...RUN, claims: [
    { text: "Not a list.", label: "SUPPORTED", cautions: "fragment_subject" },
    { text: "Mixed entries.", label: "SUPPORTED", cautions: [null, 5, {}, { kind: "  " }, "hedge_dropped", FRAGMENT] },
    { text: "Explicit null.", label: "SUPPORTED", cautions: null },
  ] });
  assert.deepEqual(run.claims[0].cautions, []);
  assert.deepEqual(run.claims[1].cautions, [{ kind: "hedge_dropped", detail: "" }, FRAGMENT],
    "a bare kind string is kept with no explanation; null, numbers and empty objects are not cautions");
  assert.deepEqual(run.claims[2].cautions, []);
  assert.equal(run.warnings.filter((w) => w.includes('statement "Not a list."') && w.includes("not a list")).length, 1);
  assert.equal(run.warnings.filter((w) => w.includes('statement "Mixed entries."')).length, 1, "reported once per record problem text");
});

test("an unknown caution kind is kept and labelled from the record", () => {
  const run = normalizeRun({ ...RUN, claims: [{ text: "x", label: "SUPPORTED", cautions: [{ kind: "new_check", detail: "Recorded detail." }] }] });
  assert.deepEqual(run.claims[0].cautions, [{ kind: "new_check", detail: "Recorded detail." }]);
  const meta = cautionMeta(run.claims[0].cautions[0], run.evidencePolicy);
  assert.deepEqual(meta, { kind: "new_check", label: "new check", known: false, detail: "Recorded detail.", description: "" });
  const list = cautionList(run.claims[0].cautions, run.evidencePolicy);
  assert.equal(list.textContent, "Caution (new check): Recorded detail. Not used in the direct answer or summary.");
  const noKind = cautionMeta({ kind: "", detail: "Only a detail." }, POLICY);
  assert.equal(noKind.label, "unlabelled caution");
});

test("each caution kind is shown with its label and recorded explanation, not by colour alone", () => {
  const run = normalizeRun(RUN);
  for (const [claim, expected] of [[run.claims[0], FRAGMENT], [run.claims[1], HEDGE]]) {
    const label = expected.kind.replace("_", " ");
    const list = cautionList(claim.cautions, run.evidencePolicy);
    assert.equal(list.tagName, "ul");
    assert.equal(list.getAttribute("aria-label"), "Evidence-quality cautions");
    assert.equal(list.textContent, `Caution (${label}): ${expected.detail} Not used in the direct answer or summary.`);
    const badge = cautionBadge(claim.cautions[0], run.evidencePolicy);
    assert.equal(badge.textContent, `Caution: ${label}`, "the word and the kind are in the text");
    assert.ok(badge.className.includes("tone-uncertain") && badge.find("svg").length === 1, "an icon as well as a tone");
    assert.equal(badge.getAttribute("title"), POLICY.cautions[expected.kind]);
  }
  const both = cautionList(run.claims[2].cautions, run.evidencePolicy);
  assert.equal(both.find("li").length, 2, "every caution on a finding is rendered");
  assert.match(both.children[0].textContent, /^Caution \(hedge dropped\): The supporting text is hedged/);
  assert.match(both.children[1].textContent, /^Caution \(causal added\): The finding says 'leads to'/);
  assert.equal(cautionBadge(CAUSAL, {}).getAttribute("title"), CAUSAL.detail, "without a policy the recorded detail is the tooltip");
  assert.equal(cautionList([{ kind: "causal_added", detail: "" }], POLICY).textContent,
    "Caution (causal added): no explanation was recorded. Not used in the direct answer or summary.");
});

test("cautioned findings stay in the findings list and the recorded answer exclusion is unchanged", () => {
  const run = normalizeRun(RUN);
  assert.equal(run.findings.length, 5, "every supported statement is listed, cautioned or not");
  assert.ok(run.findings.includes(run.claims[0]) && run.findings.includes(run.claims[2]));
  assert.deepEqual(run.summary.directAnswer.claims, [3], "the direct answer cites only what the backend recorded");
  assert.deepEqual(run.summary.points.map((p) => p.claims), [[3]]);
  assert.ok(run.summary.directAnswer.claims.every((i) => run.claims[i].cautions.length === 0));
});

test("an uncautioned finding keeps its other fields exactly as before", () => {
  const run = normalizeRun(RUN);
  const plain = run.claims[3];
  assert.equal(plain.text, "Graph neural networks model accounts as nodes.");
  assert.equal(plain.label, "SUPPORTED");
  assert.equal(plain.strength, "single source");
  assert.deepEqual(plain.cautions, []);
  assert.equal(plain.id, "", "a record without a statement ID keeps an empty one");
  assert.deepEqual(Object.keys(plain).filter((k) => k !== "cautions" && k !== "id"),
    ["text", "role", "label", "subState", "passages", "cited", "supporting", "contradicting", "independentSupport",
      "independentContradiction", "parseOk", "error", "seconds", "strength", "subquestion", "citationProblems", "unlocated",
      "independence"]);
});
