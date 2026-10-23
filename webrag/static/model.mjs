// Pure data logic for the studio: vocabulary of the backend, defensive normalisation and formatting.
// No DOM access here, so it can be tested with `node --test`.

export const VERDICTS = {
  SUPPORTED: { label: "Supported", tone: "support",
    rule: "At least one retrieved passage was judged to support the statement and none to contradict it." },
  CONTRADICTED: { label: "Contradicted", tone: "contradict",
    rule: "At least one retrieved passage was judged to contradict the statement and none to support it." },
  INSUFFICIENT: { label: "Insufficient", tone: "uncertain",
    rule: "No retrieved passage was decisive, the passages disagree, or the assessment failed. A missing supporting passage is never treated as a contradiction." },
};

export const SUB_STATES = {
  NO_EVIDENCE: "No passages were available to judge.",
  NO_DECISIVE_EVIDENCE: "None of the retrieved passages was judged decisive.",
  CONFLICTING: "Retrieved passages disagree, and no source is trusted over another.",
  ADJUDICATED: "Retrieved passages disagreed; the opt-in adjudication step judged one side decisive. Both sides are kept.",
  ASSESSMENT_FAILED: "The verifier could not assess at least one passage, so the statement is not decided.",
  SERVICE_ERROR: "The model service failed during verification.",
};

export const STANCES = {
  SUPPORTS: { label: "Supports", tone: "support" },
  CONTRADICTS: { label: "Contradicts", tone: "contradict" },
  NEUTRAL: { label: "Neutral", tone: "neutral" },
  NOT_ASSESSED: { label: "Not assessed", tone: "neutral" },
};

export const RUN_STATUS = {
  COMPLETE: { label: "Complete", tone: "support", description: "Every pipeline stage ran." },
  PARTIAL: { label: "Partial", tone: "uncertain", description: "Some searches, sources or checks failed; see the limitations." },
  FAILED: { label: "Failed", tone: "error", description: "The run produced no usable result." },
  RUNNING: { label: "Running", tone: "info", description: "The run is still in progress." },
  CANCELLED: { label: "Cancelled", tone: "neutral", description: "Withdrawn from the queue before it started; nothing was searched or saved." },
  UNREADABLE: { label: "Unreadable", tone: "error", description: "The saved run record could not be read." },
};

export const OUTCOMES = {
  CLAIMS_ASSESSED: { label: "Claims assessed", tone: "neutral",
    description: "Several claims were checked in one run; each has its own verdict, and there is no overall verdict." },
  ANSWERED: { label: "Findings supported", tone: "support",
    description: "At least one model-written finding was judged SUPPORTED by the verifier against the retrieved passages." },
  INSUFFICIENT_EVIDENCE: { label: "Insufficient evidence", tone: "uncertain",
    description: "No model-written finding could be supported by the retrieved passages." },
  VERIFIED: { label: "Verified (docqa)", tone: "support",
    description: "docqa status: every checkable statement in the answer was judged SUPPORTED." },
  PARTIALLY_VERIFIED: { label: "Partially verified (docqa)", tone: "uncertain",
    description: "docqa status: some statements were judged SUPPORTED, others were not." },
  NOT_VERIFIED: { label: "Not verified (docqa)", tone: "uncertain",
    description: "docqa status: no statement was judged SUPPORTED, or the answer had nothing checkable." },
  ABSTAINED: { label: "Abstained", tone: "neutral",
    description: "The documents did not contain enough relevant evidence to answer." },
  INDEXED: { label: "Indexed", tone: "support", description: "The documents were loaded and indexed." },
  FAILED: { label: "Failed", tone: "error", description: "No result was produced." },
};

export const ACCESS = {
  FULL_TEXT: "Full text read",
  ABSTRACT_ONLY: "Abstract only",
  SNIPPET_ONLY: "Search snippet only",
  METADATA_ONLY: "Metadata only",
  FAILED: "Could not be read",
};

// How much of a source the web researcher could read, as recorded in its `access` field.
export const ACCESS_LEVEL = { FULL_TEXT: "full", ABSTRACT_ONLY: "partial", SNIPPET_ONLY: "partial", METADATA_ONLY: "none", FAILED: "none" };

// The source types the backend labels (webrag/search.py and its feeds). Unknown types are shown as recorded.
export const SOURCE_KINDS = {
  encyclopedia: { label: "Encyclopedia", hue: "ochre" },
  scholarly: { label: "Scholarly work", hue: "green" },
  preprint: { label: "Preprint", hue: "lavender" },
  web: { label: "Web page", hue: "stone" },
  news: { label: "News", hue: "stone" },
  official: { label: "Official source", hue: "green" },
  podcast: { label: "Podcast transcript", hue: "lavender" },
};

// Deep research (webrag/deep.py EVIDENCE_POLICY): what each evidence label means. Labels describe the evidence behind
// a verdict; they never change it.
export const STRENGTH = {
  corroborated: { label: "Corroborated", tone: "support", description: "Supported by passages from at least two publishers that their metadata establishes as independent, at least one of them full text, with no passage against it. Since deep-research-v4 a source counts only when one sentence of its passage contains most of the statement's content words (a lexical screen, not proof)." },
  "independence unknown": { label: "Independence unknown", tone: "uncertain", description: "Supported by more than one source, but their metadata does not establish that they are independent, so it is not called corroborated." },
  "single source": { label: "Single source", tone: "uncertain", description: "Supported, but all supporting passages come from one publisher, one work or copies of one text." },
  "abstract only": { label: "Abstract only", tone: "uncertain", description: "Supported only by abstracts; details may differ in the full paper." },
  contested: { label: "Contested", tone: "uncertain", description: "Passages were judged both for and against; both sides are kept." },
  adjudicated: { label: "Adjudicated", tone: "uncertain", description: "Passages were judged both for and against; the opt-in adjudication step judged one side decisive. Both sides are kept, and a supported statement is left out of the direct answer and summary." },
  contradicted: { label: "Contradicted", tone: "contradict", description: "The retrieved passages were judged to contradict it." },
  "not established": { label: "Not established", tone: "neutral", description: "No retrieved passage was decisive." },
  unverified: { label: "Unverified", tone: "error", description: "The verifier could not assess it." },
};

export const COVERAGE = {
  answered: { label: "Answered", tone: "support" },
  "answered with disagreement": { label: "Answered, with disagreement", tone: "uncertain" },
  "partially answered": { label: "Partially answered", tone: "uncertain" },
  contested: { label: "Contested", tone: "uncertain" },
  unanswered: { label: "Unanswered", tone: "neutral" },
  "not reached": { label: "Not reached", tone: "error" },
};

export const COMPLETENESS = {
  complete: { label: "Research complete", tone: "support", description: "Every question reached the evidence the stopping rule asks for." },
  partial: { label: "Partly complete", tone: "uncertain", description: "Searching ended with some questions still open." },
  budget_limited: { label: "Budget-limited", tone: "uncertain", description: "Searching stopped because the depth's budget was used up." },
  evidence_limited: { label: "Evidence-limited", tone: "uncertain", description: "Further searches were not finding new relevant evidence." },
  access_limited: { label: "Access-limited", tone: "uncertain", description: "Many sources found could not be read (paywalls, failures, metadata only)." },
  provider_limited: { label: "Provider-limited", tone: "error", description: "The search providers were failing." },
};

export const DEPTH_LABELS = { quick: "Quick", standard: "Standard", deep: "Deep" };
export const RECENCY_LABELS = { any: "Any time", year: "Past year", "5y": "Past 5 years" };
export const CATEGORY_LABELS = { encyclopedia: "Encyclopedia", scholarly: "Scholarly", web: "Web", news: "News", official: "Official", podcast: "Podcasts" };

export function sourceKind(kind) {
  return SOURCE_KINDS[kind] || { label: kind ? humanize(kind) : "Unlabelled", hue: "stone" };
}

const isObject = (value) => value !== null && typeof value === "object" && !Array.isArray(value);
const list = (value) => (Array.isArray(value) ? value : []);
const text = (value) => (typeof value === "string" ? value : value === null || value === undefined ? "" : String(value));
const count = (value) => (Number.isFinite(value) && value >= 0 ? Math.floor(value) : 0);
const humanize = (value) => text(value).replace(/_/g, " ").toLowerCase();

export function verdictMeta(label) {
  if (VERDICTS[label]) return { key: label, ...VERDICTS[label] };
  return { key: text(label) || "UNKNOWN", label: humanize(label) || "unknown", tone: "neutral",
    rule: `Reported by the backend as "${text(label) || "no verdict"}".` };
}

export function outcomeMeta(value) {
  if (OUTCOMES[value]) return { key: value, ...OUTCOMES[value] };
  if (VERDICTS[value]) return { key: value, label: VERDICTS[value].label, tone: VERDICTS[value].tone, description: VERDICTS[value].rule };
  return { key: text(value), label: value ? humanize(value) : "No outcome", tone: "neutral", description: "" };
}

export function statusMeta(value) {
  return RUN_STATUS[value] ? { key: value, ...RUN_STATUS[value] }
    : { key: text(value), label: value ? humanize(value) : "Unknown", tone: "neutral", description: "" };
}

export function stanceMeta(value) {
  return STANCES[value] ? { key: value, ...STANCES[value] } : { key: text(value) || "NOT_ASSESSED", label: humanize(value) || "not assessed", tone: "neutral" };
}

export function subStateText(subState) {
  if (!subState) return "";
  return SUB_STATES[subState] || `Sub-state reported by the backend: ${humanize(subState)}.`;
}

export function safeUrl(url) {
  const value = text(url).trim();
  return /^https?:\/\/[^\s]+$/i.test(value) ? value : null;
}

// ------------------------------------------------------------------------------------------------ research runs
function normalizeSource(raw) {
  return {
    id: text(raw.id), title: text(raw.title) || text(raw.url) || text(raw.id), url: safeUrl(raw.url),
    provider: text(raw.provider), kind: text(raw.kind), access: text(raw.access), published: text(raw.published),
    publisher: text(raw.publisher), domain: text(raw.domain), note: text(raw.note), origin: text(raw.origin),
    chars: count(raw.chars), hasText: raw.has_text === undefined ? Boolean(raw.text) : Boolean(raw.has_text),
    retrievedAt: text(raw.retrieved_at), readUrl: safeUrl(raw.read_url), queries: list(raw.queries).map(text).filter(Boolean),
    category: text(raw.category), retrieval: text(raw.retrieval), retrievalError: text(raw.retrieval_error),
    pageCount: count(raw.page_count) || (Array.isArray(raw.pages) ? raw.pages.length : 0),
    authors: list(raw.authors).map(text).filter(Boolean),
    identifiers: isObject(raw.identifiers) ? Object.fromEntries(Object.entries(raw.identifiers).map(([k, v]) => [text(k), text(v)])) : {},
    discoveredBy: list(raw.discovered_by).map(text).filter(Boolean), license: text(raw.license),
    pageLabels: list(raw.page_labels).map(text), mediaUrl: safeUrl(raw.media_url), round: count(raw.round) || 1,
    gate: isObject(raw.diagnostics) ? text(raw.diagnostics.gate) : "",
    imageOnly: isObject(raw.diagnostics) ? Boolean(raw.diagnostics.image_only) : false,
  };
}

function normalizePassage(raw, sources, warnings) {
  const passage = {
    key: text(raw.key), sourceId: text(raw.source_id), page: count(raw.page) || 1, text: text(raw.text),
    score: Number.isFinite(raw.score) ? raw.score : null, url: safeUrl(raw.url), title: text(raw.title),
    access: text(raw.access), stance: raw.stance ? text(raw.stance) : null, location: text(raw.location),
  };
  if (passage.sourceId && !sources.has(passage.sourceId)) warnings.add(`Passage ${passage.key || "?"} refers to an unknown source ${passage.sourceId}.`);
  return passage;
}

// Deep research (deep-research-v2): evidence-quality cautions on a verified finding, recorded as a list of
// { kind, detail } (webrag/deep.py `cautions`). A caution is a warning from fixed text checks, not a verdict; the
// backend keeps cautioned findings out of the direct answer and summary. Unknown kinds are kept and shown as recorded;
// an entry that cannot be read is reported as a record problem, never turned into a caution.
export const CAUTION_KINDS = ["fragment_subject", "hedge_dropped", "causal_added", "number_unsupported", "qualifier_dropped",
  "attribution_dropped", "support_unmatched", "adjudicated_conflict"];

function normalizeCautions(raw, statement, warnings) {
  if (raw === undefined || raw === null) return [];
  const where = `statement "${text(statement).slice(0, 60)}"`;
  if (!Array.isArray(raw)) {
    warnings.add(`The cautions recorded for ${where} are not a list and could not be read.`);
    return [];
  }
  const out = [];
  for (const c of raw) {
    if (typeof c === "string" && c.trim()) out.push({ kind: c.trim(), detail: "" });
    else if (isObject(c) && (text(c.kind).trim() || text(c.detail).trim())) out.push({ kind: text(c.kind).trim(), detail: text(c.detail).trim() });
    else warnings.add(`A caution recorded for ${where} could not be read.`);
  }
  return out;
}

// How a caution is labelled: its recorded kind (as the Markdown report writes it), its recorded explanation, and the
// evidence policy's description of that kind when the run recorded one. Nothing is filled in that was not recorded.
export function cautionMeta(caution, policy) {
  const kind = text(caution && caution.kind);
  const descriptions = isObject(policy) && isObject(policy.cautions) ? policy.cautions : {};
  return { kind, label: kind ? humanize(kind) : "unlabelled caution", known: CAUTION_KINDS.includes(kind),
    detail: text(caution && caution.detail), description: kind ? text(descriptions[kind]) : "" };
}

function normalizeClaim(raw, sources, warnings) {
  return {
    text: text(raw.text), role: text(raw.role), label: text(raw.label) || "INSUFFICIENT", subState: text(raw.sub_state),
    passages: list(raw.passages).filter(isObject).map((p) => normalizePassage(p, sources, warnings)),
    cited: [...new Set(list(raw.cited_sources).map(text).filter(Boolean))],
    supporting: [...new Set(list(raw.supporting_sources).map(text).filter(Boolean))],
    contradicting: [...new Set(list(raw.contradicting_sources).map(text).filter(Boolean))],
    independentSupport: count(raw.independent_support), independentContradiction: count(raw.independent_contradiction),
    parseOk: raw.parse_ok !== false, error: text(raw.error), seconds: Number.isFinite(raw.seconds) ? raw.seconds : null,
    strength: text(raw.strength), subquestion: Number.isInteger(raw.subquestion) ? raw.subquestion : -1,
    citationProblems: isObject(raw.citations) ? list(raw.citations.problems).map(text) : [],
    unlocated: isObject(raw.citations) ? list(raw.citations.unlocated).map(text) : [],
    independence: isObject(raw.independence) ? { status: text(raw.independence.status), independent: count(raw.independence.independent),
      groups: count(raw.independence.groups), sources: count(raw.independence.sources) } : null,
    cautions: normalizeCautions(raw.cautions, raw.text, warnings),
    id: text(raw.id),
  };
}

// Deep research: how independent a statement's supporting sources are, as judged from their metadata.
export function independenceText(independence) {
  if (!isObject(independence)) return "";
  if (independence.status === "established") return `${plural(independence.independent, "publisher")} established as independent`;
  if (independence.status === "unknown") return `${plural(independence.sources, "source")} from ${plural(independence.groups, "publisher")}, independence not established`;
  if (independence.status === "single") return `one publisher (${plural(independence.sources, "source")})`;
  return "";
}

// How a finished run with no supported finding is described: an execution failure, or a run that completed and could
// not support an answer. Only FAILED is a failure; the reason comes from the run record, never from the interface.
export function noAnswerState(run) {
  const failed = run.status === "FAILED";
  const supported = run.claims.some((c) => c.label === "SUPPORTED");
  const insufficient = ["INSUFFICIENT_EVIDENCE", "INSUFFICIENT"].includes(run.outcome);
  return {
    failed, answered: !failed && supported && (run.mode === "verify" || run.outcome === "ANSWERED"),
    insufficient: !failed && insufficient,
    title: failed ? (run.mode === "verify" ? "The verification did not complete" : "The run failed")
      : insufficient ? (run.mode === "verify" ? "No verdict could be supported" : "No answer could be supported") : "Problem reported by the pipeline",
    tone: failed ? "error" : insufficient ? "warn" : "error",
    reason: run.error,
  };
}

function normalizeLedger(raw) {
  return list(raw).filter(isObject).map((e) => ({
    index: count(e.index), text: text(e.text), status: text(e.status), reason: text(e.reason), nextAction: text(e.next_action),
    evidenceState: text(e.evidence_state), categories: list(e.categories).map(text),
    // Records written before independence was judged from metadata counted distinct texts ("independent_origins").
    publishers: count(e.independent_publishers), independence: text(e.independence),
    distinctTexts: e.independent_publishers === undefined ? count(e.independent_origins) : null,
    queries: list(e.queries).map(text), discovered: list(e.discovered).map(text), retrieved: list(e.retrieved).map(text),
    supporting: list(e.supporting).map(text), contradicting: list(e.contradicting).map(text), uncertain: list(e.uncertain).map(text),
  }));
}

function normalizeRounds(raw) {
  return list(raw).filter(isObject).map((r) => ({
    number: count(r.number), purpose: text(r.purpose), searches: count(r.searches), hits: count(r.hits),
    newSources: count(r.new_sources), indexed: count(r.indexed_sources), errors: count(r.errors),
    filtered: count(r.filtered_by_date), newPassages: count(r.new_relevant_passages), seconds: Number.isFinite(r.seconds) ? r.seconds : null,
    queries: list(r.queries).filter(isObject).map((q) => ({ query: text(q.query), category: text(q.category), purpose: text(q.purpose),
      subquestion: Number.isInteger(q.subquestion) ? q.subquestion : 0, providers: list(q.providers).map(text) })),
    coverage: list(r.coverage).filter(isObject).map((c) => ({ subquestion: count(c.subquestion), state: text(c.state),
      publishers: count(c.publishers), independence: text(c.independence) })),
    access: isObject(r.access) ? r.access : {}, requests: count(r.requests),
    providerStatus: isObject(r.provider_status) ? Object.entries(r.provider_status).filter(([, v]) => isObject(v))
      .map(([name, v]) => ({ name, queries: count(v.queries), succeeded: count(v.succeeded), failed: count(v.failed), hits: count(v.hits) })) : [],
  }));
}

function normalizeSummary(raw) {
  if (!isObject(raw)) return null;
  const part = (p) => (isObject(p) && text(p.text) ? { text: text(p.text), claims: list(p.claims).filter(Number.isInteger) } : null);
  return { available: Boolean(raw.available), answered: Boolean(raw.answered), reason: text(raw.reason), dropped: count(raw.dropped),
    directAnswer: part(raw.direct_answer), points: list(raw.summary).map(part).filter(Boolean),
    verdict: text(raw.verdict), strength: text(raw.strength),
    verdicts: isObject(raw.verdicts) ? Object.fromEntries(Object.entries(raw.verdicts).map(([k, v]) => [text(k), count(v)])) : null };
}

// deep-research-v3: the query ledger. Status vocabulary from webrag/deep.py; an unknown status is kept as recorded.
export const QUERY_STATUS = {
  executed: { label: "Executed", tone: "support" }, failed: { label: "Failed", tone: "error" },
  skipped: { label: "Not run", tone: "uncertain" }, no_provider: { label: "No provider", tone: "neutral" },
};
export function queryStatusMeta(value) {
  return QUERY_STATUS[value] ? { key: value, ...QUERY_STATUS[value] } : { key: text(value), label: value ? humanize(value) : "Not recorded", tone: "neutral" };
}

function normalizeQueries(raw) {
  return list(raw).filter(isObject).map((q) => ({
    id: text(q.id), round: count(q.round), roundKind: text(q.round_kind), query: text(q.query), category: text(q.category),
    purpose: text(q.purpose), subquestion: Number.isInteger(q.subquestion) ? q.subquestion : 0, status: text(q.status),
    reason: text(q.reason), notSentTo: list(q.not_sent_to).map(text).filter(Boolean),
    // deep-research-v4; absent in older records (null: not recorded)
    allocated: typeof q.allocated === "boolean" ? q.allocated : null, taggedBy: text(q.tagged_by),
    relevantCandidate: typeof q.relevant_candidate === "boolean" ? q.relevant_candidate : null,
    searches: list(q.searches).filter(isObject).map((s) => ({ provider: text(s.provider), requested: count(s.requested),
      returned: count(s.returned), error: text(s.error) })),
  }));
}

// deep-research-v3: what happened to each provider in the run. Only "searched" and "failed" mean searches were sent.
export const PROVIDER_STATES = {
  searched: { label: "Searched", tone: "support" }, failed: { label: "Failed", tone: "error" },
  blocked_by_policy: { label: "Blocked by network policy", tone: "uncertain" },
  not_used: { label: "Not used", tone: "neutral" }, not_selected: { label: "Not selected", tone: "neutral" },
  not_configured: { label: "Not configured", tone: "uncertain" }, disabled: { label: "Off", tone: "neutral" },
  excluded: { label: "Excluded by policy", tone: "neutral" },
};
export function providerStateMeta(value) {
  return PROVIDER_STATES[value] ? { key: value, ...PROVIDER_STATES[value] } : { key: text(value), label: value ? humanize(value) : "Unknown", tone: "neutral" };
}

function normalizeProviderCoverage(raw) {
  return list(raw).filter(isObject).map((p) => ({ id: text(p.id), name: text(p.name) || text(p.id), state: text(p.state),
    detail: text(p.detail), categories: list(p.categories).map(text), missing: list(p.missing).map(text),
    queries: count(p.queries), failed: count(p.failed), hits: count(p.hits) }));
}

// The model runtime the studio reports: the local Ollama, or a model service the server operator configured.
export function runtimeInfo(status) {
  const s = isObject(status) ? status : {};
  const remote = isObject(s.runtime) ? s.runtime : null;
  const reachable = remote ? Boolean(remote.service_reachable) : Boolean(s.ollama_version);
  return {
    remote: Boolean(remote), reachable, ready: reachable && Boolean(s.model_installed),
    name: remote ? `${text(remote.provider_label)} · ${text(remote.endpoint)}` : "Local · Ollama",
    version: remote ? text(remote.service_version) : text(s.ollama_version),
    error: remote ? text(remote.error) : "",
  };
}

export function normalizeRun(raw) {
  const data = isObject(raw) ? raw : {};
  const warnings = new Set();
  const sources = list(data.sources).filter((s) => isObject(s) && typeof s.id === "string").map(normalizeSource);
  const index = new Map(sources.map((s) => [s.id, s]));
  const claims = list(data.claims).filter(isObject).map((c) => normalizeClaim(c, index, warnings));
  const settings = isObject(data.settings) ? data.settings : {};
  const model = isObject(data.model) ? data.model : {};
  return {
    input: text(data.input), mode: data.mode === "verify" ? "verify" : "research", status: text(data.status) || "RUNNING",
    outcome: data.outcome ? text(data.outcome) : null, error: text(data.error), current: text(data.current),
    started: text(data.started), finished: text(data.finished), savedTo: text(data.saved_to),
    stages: list(data.stages).filter(isObject).map((s) => ({ name: text(s.name), status: text(s.status), seconds: Number.isFinite(s.seconds) ? s.seconds : null, detail: text(s.detail) })),
    plan: isObject(data.plan) ? { queries: list(data.plan.queries).map(text), subquestions: list(data.plan.subquestions).map(text), fallback: Boolean(data.plan.fallback) } : null,
    sources, sourceIndex: index, claims,
    findings: claims.filter((c) => c.label === "SUPPORTED"),
    unsettled: claims.filter((c) => c.label !== "SUPPORTED"),
    pool: list(data.evidence_pool).filter(isObject).map((p) => normalizePassage(p, index, warnings)),
    disagreements: list(data.disagreements).filter(isObject).map((d) => ({ text: text(d.text), sources: list(d.sources).map(text), verified: Boolean(d.verified) })),
    gaps: list(data.gaps).map(text).filter(Boolean),
    limitations: list(data.limitations).map(text).filter(Boolean),
    searchErrors: list(data.search_errors).filter(isObject).map((e) => ({ provider: text(e.provider), query: text(e.query), error: text(e.error) })),
    verifier: text(settings.verifier), model: text(model.model || settings.model), ollama: text(model.ollama_version),
    modelService: model.provider ? `${text(model.provider_label)} · ${text(model.endpoint)}` : "",
    verifierModel: isObject(model.verifier_model) ? text(model.verifier_model.model) : "",
    modelDigest: text(model.model_digest),
    providers: list(settings.providers).map(text), diversity: isObject(data.source_diversity) ? data.source_diversity : null,
    providerStatus: isObject(data.provider_status) ? Object.entries(data.provider_status).filter(([, v]) => isObject(v))
      .map(([name, v]) => ({ name, queries: count(v.queries), succeeded: count(v.succeeded), failed: count(v.failed), hits: count(v.hits) })) : [],
    requests: Number.isFinite(data.requests) ? data.requests : null,
    followup: isObject(data.followup) ? { skipped: text(data.followup.skipped_reason), queries: list(data.followup.queries).map(text),
      newSources: list(data.followup.new_sources).length, changes: list(data.followup.changes).length,
      changed: list(data.followup.changes).filter((c) => isObject(c) && c.changed).length } : null,
    settings: Object.fromEntries(Object.entries(settings).filter(([, v]) => ["string", "number", "boolean"].includes(typeof v))),
    depth: text(data.depth), options: isObject(data.options) ? data.options : null,
    budget: isObject(data.budget) ? data.budget : null, rounds: normalizeRounds(data.rounds), ledger: normalizeLedger(data.ledger),
    stopping: isObject(data.stopping) ? { reason: text(data.stopping.reason), detail: text(data.stopping.detail),
      completeness: text(data.stopping.completeness) } : null,
    summary: normalizeSummary(data.summary),
    counter: isObject(data.counter_evidence) ? { skipped: text(data.counter_evidence.skipped),
      queries: list(data.counter_evidence.queries).filter(isObject).map((q) => ({ query: text(q.query), category: text(q.category) })),
      changes: list(data.counter_evidence.changes).filter(isObject).map((c) => ({ text: text(c.text), changed: Boolean(c.changed), reason: text(c.reason),
        before: isObject(c.before) ? text(c.before.label) : "", after: isObject(c.after) ? text(c.after.label) : "",
        afterSub: isObject(c.after) ? text(c.after.sub_state) : "" })) } : null,
    quality: isObject(data.source_quality) ? data.source_quality : null,
    researchPlan: isObject(data.research_plan) ? data.research_plan : null,
    evidencePolicy: isObject(data.evidence_policy) ? data.evidence_policy : null,
    queries: normalizeQueries(data.queries), providerCoverage: normalizeProviderCoverage(data.provider_coverage),
    review: list(data.review).filter(isObject).map((r) => ({ claim: text(r.claim), kind: text(r.kind), detail: text(r.detail),
      action: text(r.action), refs: list(r.refs).map(text).filter(Boolean) })),
    claimsMode: isObject(data.claims_mode) ? { claims: count(data.claims_mode.claims), limit: count(data.claims_mode.limit),
      dropped: count(data.claims_mode.dropped) } : null,
    warnings: [...warnings],
  };
}

export function strengthMeta(value) {
  return STRENGTH[value] ? { key: value, ...STRENGTH[value] } : { key: text(value), label: value ? humanize(value) : "", tone: "neutral", description: "" };
}
export function coverageMeta(value) {
  return COVERAGE[value] ? { key: value, ...COVERAGE[value] } : { key: text(value), label: value ? humanize(value) : "Unknown", tone: "neutral" };
}
export function completenessMeta(value) {
  return COMPLETENESS[value] ? { key: value, ...COMPLETENESS[value] } : { key: text(value), label: value ? humanize(value) : "", tone: "neutral", description: "" };
}

// Findings in report order: by the subquestion they answer (deep research), then as drafted.
export function findingsBySubquestion(run) {
  const groups = new Map();
  for (const claim of run.claims) {
    const key = claim.subquestion >= 0 ? claim.subquestion : 0;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(claim);
  }
  const titles = new Map(run.ledger.map((e) => [e.index, e.text]));
  return [...groups.entries()].sort((a, b) => a[0] - b[0])
    .map(([index, claims]) => ({ index, title: index === 0 ? "The main question" : titles.get(index) || `Question ${index}`, claims }));
}

// Budget used versus the limits of the run's depth; nothing is shown that the backend did not report.
export function budgetRows(budget) {
  if (!isObject(budget) || !isObject(budget.used) || !isObject(budget.limits)) return [];
  const rows = [["Search rounds", budget.used.rounds, budget.limits.rounds], ["Searches", budget.used.searches, budget.limits.searches],
    ["Sources", budget.used.sources, budget.limits.sources], ["Full-text attempts", budget.used.full_text, budget.limits.full_text]];
  const out = rows.filter(([, used, limit]) => Number.isFinite(used) && Number.isFinite(limit)).map(([label, used, limit]) => ({ label, used, limit }));
  if (Number.isFinite(budget.elapsed) && Number.isFinite(budget.limits.seconds)) out.push({ label: "Elapsed seconds (search cut-off)", used: Math.round(budget.elapsed), limit: budget.limits.seconds });
  return out;
}

// The state of one search provider in the catalog the server reports (webrag/search.py provider_catalog): whether a
// new run would search it, and if not, why. Nothing is inferred beyond the reported flags.
export function providerState(p) {
  const missing = list(p && p.missing).map(text).filter(Boolean);
  if (!isObject(p)) return { state: "unknown", label: "Unknown", tone: "neutral", detail: "" };
  if (p.evidence === false) return { state: "excluded", label: "Not used as evidence", tone: "neutral", detail: text(p.policy) || "Excluded by the source policy." };
  if (p.active) return { state: "active", label: "Active", tone: "support", detail: "Searched in new research runs unless a run narrows the selection." };
  if (p.enabled && !p.configured) return { state: "needs_setup", label: "Needs setup", tone: "uncertain", detail: `Switched on, but missing ${missing.join(", ") || "part of its setup"}.` };
  if (p.controlled_by === "WEBRAG_PROVIDERS") return { state: "off", label: "Off", tone: "neutral", detail: "Not listed in the WEBRAG_PROVIDERS environment variable, which decides the providers." };
  return { state: "off", label: "Off", tone: "neutral", detail: missing.length ? `Switched off; it also needs ${missing.join(", ")}.` : "Switched off." };
}

export function describeBudget(limits) {
  if (!isObject(limits)) return "";
  return `${plural(limits.rounds, "round")}, up to ${limits.searches} searches and ${limits.sources} sources, ${limits.full_text} full-text attempts; searching stops after ${Math.round(limits.seconds / 60)} min`;
}

// Citation chips for one claim: what the synthesizer cited, plus where the verifier found support or contradiction.
export function citationsFor(claim, sourceIndex) {
  const ids = [...new Set([...claim.cited, ...claim.supporting, ...claim.contradicting])];
  return ids.map((id) => {
    const source = sourceIndex.get(id);
    return { id, cited: claim.cited.includes(id), supports: claim.supporting.includes(id),
      contradicts: claim.contradicting.includes(id), title: source ? source.title : id, url: source ? source.url : null,
      known: Boolean(source) };
  });
}

export function groupByStance(passages) {
  const groups = { SUPPORTS: [], CONTRADICTS: [], OTHER: [] };
  for (const passage of passages) (groups[passage.stance] || groups.OTHER).push(passage);
  return groups;
}

// How each source was used, counted from the recorded passages and claims only. Nothing is inferred from titles or
// search results: a source counts as "retrieved" only when one of its passages is in the record.
export function sourceUsage(run) {
  const usage = new Map(run.sources.map((s) => [s.id, { passages: new Set(), pool: 0, cited: 0, supports: 0, contradicts: 0, judged: 0 }]));
  const entry = (id) => {
    if (!usage.has(id)) usage.set(id, { passages: new Set(), pool: 0, cited: 0, supports: 0, contradicts: 0, judged: 0 });
    return usage.get(id);
  };
  for (const p of run.pool) if (p.sourceId) { const u = entry(p.sourceId); u.pool += 1; u.passages.add(p.key || p.text); }
  for (const claim of run.claims) {
    const judged = new Set();
    for (const p of claim.passages) if (p.sourceId) { entry(p.sourceId).passages.add(p.key || p.text); judged.add(p.sourceId); }
    for (const id of judged) entry(id).judged += 1;
    for (const id of claim.cited) entry(id).cited += 1;
    for (const id of claim.supporting) entry(id).supports += 1;
    for (const id of claim.contradicting) entry(id).contradicts += 1;
  }
  return new Map([...usage].map(([id, u]) => [id, { ...u, retrieved: u.passages.size, passages: undefined,
    used: u.cited > 0 || u.supports > 0 || u.contradicts > 0 }]));
}

// The recorded path of one source through the pipeline: selected from search results, read, retrieved, used.
export function sourceJourney(source, usage) {
  const u = usage || { retrieved: 0, cited: 0, supports: 0, contradicts: 0, used: false };
  const level = ACCESS_LEVEL[source.access] || "none";
  const used = [u.cited ? `cited in ${plural(u.cited, "statement")}` : "", u.supports ? `supports ${u.supports}` : "",
    u.contradicts ? `contradicts ${u.contradicts}` : ""].filter(Boolean).join(" · ");
  const metadataOnly = source.access === "METADATA_ONLY";
  const found = (source.discoveredBy && source.discoveredBy.length ? source.discoveredBy : [source.provider]).filter(Boolean);
  return [
    { key: "found", label: "Found", state: "done", detail: found.length ? `search result from ${found.join(", ")}` : "search result" },
    { key: "read", label: level === "full" ? "Read in full" : level === "partial" ? "Partly read" : metadataOnly ? "Metadata only" : "Not read",
      state: level === "full" ? "done" : level === "partial" ? "partial" : metadataOnly ? "none" : "failed",
      detail: (ACCESS[source.access] || source.access || "access not recorded") + (source.chars ? ` · ${source.chars.toLocaleString("en")} characters` : "")
        + (source.retrievalError && level !== "full" ? ` · ${source.retrievalError}` : "") },
    { key: "retrieved", label: "Retrieved", state: u.retrieved ? "done" : "none",
      detail: u.retrieved ? `${plural(u.retrieved, "passage")} in the record` : "no passage in the record" },
    { key: "used", label: "Used as evidence", state: u.used ? "done" : "none", detail: used || "not cited or judged decisive" },
  ];
}

export const SOURCE_FILTERS = {
  all: { label: "All", test: () => true },
  used: { label: "Used as evidence", test: (s, u) => Boolean(u && u.used) },
  retrieved: { label: "Retrieved", test: (s, u) => Boolean(u && u.retrieved) },
  full: { label: "Read in full", test: (s) => ACCESS_LEVEL[s.access] === "full" },
  partial: { label: "Partly read", test: (s) => ACCESS_LEVEL[s.access] === "partial" },
  unread: { label: "Not read", test: (s) => (ACCESS_LEVEL[s.access] || "none") === "none" },
};

export function filterSources(sources, usage, { filter = "all", query = "" } = {}) {
  const rule = (SOURCE_FILTERS[filter] || SOURCE_FILTERS.all).test;
  const needle = text(query).trim().toLowerCase();
  return sources.filter((s) => rule(s, usage.get(s.id)) && (!needle
    || [s.id, s.title, s.publisher, s.domain, s.provider].some((v) => v.toLowerCase().includes(needle))));
}

// Bibliography entries from the recorded source metadata. Sources cited by a statement come first, then sources whose
// passages were retrieved, then the remaining selected sources. Missing fields are left out, never filled in.
export function bibliography(run, usage = sourceUsage(run)) {
  const group = (s) => { const u = usage.get(s.id); return u && u.used ? "cited" : u && u.retrieved ? "consulted" : "listed"; };
  const order = { cited: 0, consulted: 1, listed: 2 };
  return run.sources.map((s, index) => {
    const date = s.published ? s.published.slice(0, 10) : "";
    const accessed = s.retrievedAt ? s.retrievedAt.slice(0, 10) : "";
    const publisher = s.publisher || s.domain;
    const where = [publisher, date].filter(Boolean).join(", ");
    const citation = [`${s.title.replace(/\.+$/, "")}.`, where ? `${where}.` : "", s.url || "", accessed ? `(accessed ${accessed})` : ""]
      .filter(Boolean).join(" ");
    return { id: s.id, group: group(s), index, title: s.title, publisher, date, accessed, url: s.url, kind: s.kind, access: s.access, citation };
  }).sort((a, b) => order[a.group] - order[b.group] || a.index - b.index);
}

// ------------------------------------------------------------------------------------------------ progress
// The stage names each pipeline reports, in order (webrag/orchestrator.py and webrag/studio.py). They are used only
// to name the steps that have not started yet; completed steps always come from the stages the backend reported.
export const EXPECTED_STAGES = {
  research: ["model check", "plan", "search and fetch", "index", "retrieve", "synthesize", "verify"],
  verify: ["model check", "plan", "search and fetch", "index", "retrieve", "verify"],
  documents: ["load and index documents", "answer and verify"],
  "documents-verify": ["load and index documents", "verify"],
  "documents-index": ["load and index documents"],
};

const STAGE_LABELS = {
  "model check": "Check the model", plan: "Understand the question", "search and fetch": "Discover and read sources",
  index: "Index passages", retrieve: "Retrieve evidence", synthesize: "Write cited findings", "follow-up": "Follow-up search",
  "load and index documents": "Load and index the documents", "answer and verify": "Answer and check each statement",
  "evidence pool": "Choose the evidence passages", "counter-evidence": "Search for counter-evidence and re-check",
  summary: "Write the answer from verified findings",
};
const ROUND_STAGES = [
  [/^round (\d+): search and read$/, (n) => `Round ${n}: search and read sources`],
  [/^round (\d+): coverage$/, (n) => `Round ${n}: assess coverage per question`],
  [/^round (\d+): follow-up queries$/, (n) => `Round ${n}: plan follow-up searches`],
];

export function stageLabel(name, mode) {
  if (name === "verify") return mode === "research" || mode === "deep" ? "Verify each finding" : "Judge passages against the claim";
  for (const [pattern, label] of ROUND_STAGES) {
    const match = pattern.exec(name || "");
    if (match) return label(match[1]);
  }
  return STAGE_LABELS[name] || (name ? name.charAt(0).toUpperCase() + name.slice(1) : "Working");
}

// Steps for a progress display: reported stages as reported, then (while the job is live) the next expected stage as
// active and the rest as not started. No percentage or completion is invented.
export function progressSteps(mode, stages, { state = "done", current = "" } = {}) {
  const reported = list(stages).map((s) => ({ name: s.name, label: stageLabel(s.name, mode), state: s.status || "ok",
    seconds: s.seconds, detail: s.detail || "" }));
  if (state !== "running" && state !== "queued") return reported;
  const seen = new Set(reported.map((s) => s.name));
  const expected = EXPECTED_STAGES[mode] || [];
  let last = -1;
  expected.forEach((name, i) => { if (seen.has(name)) last = i; });
  const upcoming = expected.slice(last + 1).filter((name) => !seen.has(name));
  const rows = [...reported];
  upcoming.forEach((name, i) => rows.push({ name, label: stageLabel(name, mode), seconds: null,
    state: state === "running" && i === 0 ? "active" : "pending", detail: state === "running" && i === 0 ? current : "" }));
  if (state === "running" && !upcoming.length) {
    rows.push({ name: "", label: expected.length ? "Finishing up" : "In progress", state: "active", seconds: null, detail: current });
  }
  return rows;
}

// Seconds since an ISO timestamp or a Unix time in seconds; null when it cannot be read.
export function elapsedSince(start, now = Date.now()) {
  const ms = typeof start === "number" ? start * 1000 : Date.parse(text(start));
  if (!Number.isFinite(ms)) return null;
  return Math.max(0, (now - ms) / 1000);
}

// A job of the same kind and text that is still waiting or running, so the same question is not submitted twice.
export function activeDuplicate(jobs, input, mode) {
  const wanted = text(input).trim().replace(/\s+/g, " ").toLowerCase();
  return list(jobs).find((j) => isObject(j) && (j.state === "queued" || j.state === "running") && j.mode === mode
    && text(j.input).trim().replace(/\s+/g, " ").toLowerCase() === wanted) || null;
}

// ------------------------------------------------------------------------------------------------ documents
function normalizeEvidence(raw) {
  return { id: text(raw.id), document: text(raw.document), page: count(raw.page) || 1, text: text(raw.text),
    score: Number.isFinite(raw.score) ? raw.score : null, citation: text(raw.citation) };
}

function normalizeVerdict(raw) {
  if (!isObject(raw)) return null;
  const stances = isObject(raw.stances) ? Object.fromEntries(Object.entries(raw.stances).map(([k, v]) => [text(k), text(v)])) : {};
  return { label: text(raw.label) || "INSUFFICIENT", subState: text(raw.sub_state), stances,
    supporting: list(raw.supporting).map(text), contradicting: list(raw.contradicting).map(text),
    parseOk: raw.parse_ok !== false, error: text(raw.error) };
}

export function normalizeDocumentJob(raw) {
  const data = isObject(raw) ? raw : {};
  const result = isObject(data.result) ? data.result : {};
  return {
    mode: text(data.mode), input: text(data.input), status: text(data.status) || "RUNNING", outcome: data.outcome ? text(data.outcome) : null,
    error: text(data.error), current: text(data.current),
    stages: list(data.stages).filter(isObject).map((s) => ({ name: text(s.name), status: text(s.status), seconds: Number.isFinite(s.seconds) ? s.seconds : null, detail: text(s.detail) })),
    answer: text(result.answer), qaStatus: text(result.status), abstained: Boolean(result.abstained),
    limitations: list(result.limitations).map(text).filter(Boolean),
    evidence: list(result.evidence).filter(isObject).map(normalizeEvidence),
    claims: list(result.claims).filter(isObject).map((c) => ({ text: text(c.text), cited: list(c.cited).map(text), verdict: normalizeVerdict(c.verdict) })),
    verdict: normalizeVerdict(result.verdict),
  };
}

// ------------------------------------------------------------------------------------------------ evaluations
const METRIC_LABELS = {
  accuracy: "Accuracy", macro_f1: "Macro-F1", contradiction_recall: "Contradiction recall",
  abstention_rate: "Abstention (INSUFFICIENT) rate", parse_failure_rate: "Parse failures", truncation_rate: "Truncations",
  service_error_rate: "Service errors", evidence_found_rate: "Evidence in snapshot", retrieval_hit_rate: "Anchor passage ranked",
  completion_rate: "Completed questions", citation_url_integrity: "Citation URL integrity",
  answer_correctness: "Answer correctness", unsupported_finding_rate: "Unsupported findings",
};

export function metricValue(value) {
  if (isObject(value) && Number.isFinite(value.value)) {
    return { value: value.value, num: Number.isFinite(value.num) ? value.num : null, den: Number.isFinite(value.den) ? value.den : null };
  }
  if (Number.isFinite(value)) return { value, num: null, den: null };
  return null;
}

export function formatMetric(value) {
  const metric = metricValue(value);
  if (!metric) return "not recorded";
  const shown = Number.isInteger(metric.value) ? String(metric.value) : metric.value.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
  return metric.num !== null && metric.den !== null ? `${shown} (${metric.num}/${metric.den})` : shown;
}

// Only metrics that the recorded file actually contains, in a fixed reading order; nothing is derived.
export function headlineMetrics(metrics) {
  if (!isObject(metrics)) return [];
  const rows = [];
  const add = (key, raw, label = METRIC_LABELS[key] || humanize(key)) => {
    const metric = metricValue(raw);
    if (metric) rows.push({ key, label, ...metric, display: formatMetric(raw) });
  };
  if (Number.isFinite(metrics.n)) rows.push({ key: "n", label: "Items", value: metrics.n, num: null, den: null, display: String(metrics.n) });
  const classification = isObject(metrics.classification) ? metrics.classification : null;
  if (classification) {
    add("accuracy", classification.accuracy);
    add("macro_f1", classification.macro_f1);
  }
  for (const key of ["contradiction_recall", "abstention_rate", "parse_failure_rate", "truncation_rate", "service_error_rate",
    "evidence_found_rate", "retrieval_hit_rate", "completion_rate", "citation_url_integrity"]) {
    if (key in metrics) add(key, metrics[key]);
  }
  for (const [name, value] of Object.entries(metrics)) {
    if (isObject(value) && ("answer_correctness" in value || "unsupported_finding_rate" in value)) {
      if ("answer_correctness" in value) add(`${name}.answer_correctness`, value.answer_correctness, `Answer correctness (${name})`);
      if ("unsupported_finding_rate" in value) add(`${name}.unsupported_finding_rate`, value.unsupported_finding_rate, `Unsupported findings (${name})`);
    }
  }
  return rows;
}

export function comparability(a, b) {
  if (!a || !b) return { comparable: false, reason: "Select two runs." };
  if (!a.comparison_group || a.comparison_group !== b.comparison_group) {
    return { comparable: false, reason: "These runs used a different dataset or evidence snapshot, so their metrics are not directly comparable." };
  }
  if (!a.complete || !b.complete) return { comparable: false, reason: "At least one run is marked incomplete." };
  return { comparable: true, reason: "Same dataset and evidence snapshot." };
}

// ------------------------------------------------------------------------------------------------ formatting
export function formatSeconds(seconds) {
  if (!Number.isFinite(seconds)) return "";
  if (seconds < 60) return `${seconds < 10 ? seconds.toFixed(1) : Math.round(seconds)} s`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes} min ${String(Math.round(seconds % 60)).padStart(2, "0")} s`;
}

export function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes < 0) return "";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function formatDate(value, locale) {
  const raw = text(value);
  if (!raw) return "";
  const normalized = /[+-]\d{4}$/.test(raw) ? raw.replace(/([+-]\d{2})(\d{2})$/, "$1:$2") : raw;
  const date = new Date(normalized);
  if (Number.isNaN(date.getTime())) return raw;
  return date.toLocaleString(locale, { day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit" });
}

export function plural(n, word, many = `${word}s`) {
  return `${n} ${n === 1 ? word : many}`;
}

export function truncate(value, length) {
  const s = text(value);
  return s.length > length ? `${s.slice(0, Math.max(0, length - 1)).trimEnd()}…` : s;
}

export function validateQuestion(value, max = 2000) {
  const s = text(value).trim();
  if (!s) return "Enter a question or claim first.";
  if (s.length > max) return `Keep it under ${max} characters (currently ${s.length}).`;
  return "";
}
