// Knowledge library: the document folder read by the docqa assistant, its ingestion state, and document questions.
import { api, pollJob } from "./api.mjs";
import {
  accessBadge, badge, callout, chips, disclosure, emptyState, errorCallout, externalLink, kindMark, loading, modal, outcomeBadge,
  provenance, searchField, sectionHead, stanceTag, statusBadge, tabbar, timeline, verdictBadge, verdictStamp,
} from "./components.mjs";
import { append, clear, el, icon } from "./dom.mjs";
import {
  ACCESS_LEVEL, formatBytes, formatDate, normalizeDocumentJob, plural, subStateText, truncate, validateQuestion, verdictMeta,
} from "./model.mjs";
import { progressView, sessionList } from "./workspace.mjs";

const INDEX_STATES = {
  not_built: ["Not indexed yet", "neutral", "The index is built the first time you ask a question or press Build index."],
  building: ["Indexing", "info", "The documents are being loaded and indexed."],
  ready: ["Indexed", "support", ""],
  stale: ["Index out of date", "uncertain", "The documents changed after indexing; the next question rebuilds the index."],
  failed: ["Indexing failed", "error", ""],
};
const JOB_TITLES = { documents: "Question about your documents", "documents-verify": "Claim checked against your documents", "documents-index": "Library index" };

const VIEWS = [
  { id: "reports", label: "Reports", icon: "report", lede: "Every saved investigation, newest first, with its findings, sources and verdicts." },
  { id: "sources", label: "Sources", icon: "book", lede: "Every source your saved investigations selected, gathered in one place, with the runs that used it." },
  { id: "documents", label: "Documents", icon: "file", lede: "Your own TXT and PDF files. The document assistant answers only from them and checks every statement against the retrieved pages." },
];

export function renderLibrary(view, ctx) {
  ctx.setTitle("Library");
  const wanted = ctx.query.get("view");
  const initial = VIEWS.some((v) => v.id === wanted) ? wanted : "reports";
  const lede = el("p", { class: "lede" });
  const panel = el("div", { id: "library-panel", role: "tabpanel", tabindex: "-1" });
  const bar = tabbar(VIEWS, initial, (id) => {
    history.replaceState(null, "", id === "reports" ? "#/library" : `#/library?view=${id}`);
    panel.setAttribute("aria-labelledby", `tab-${id}`);
    lede.textContent = VIEWS.find((v) => v.id === id).lede;
    clear(panel);
    ({ reports: reportsView, sources: sourcesLibraryView, documents: documentsView })[id](panel, ctx);
  }, { label: "Library views", panelId: "library-panel" });
  view.append(el("div", { class: "page" },
    el("header", { class: "page-header" }, el("div", {}, el("p", { class: "eyebrow" }, "Library"), el("h1", {}, "Your research library"), lede)),
    bar.node, panel));
  bar.select(initial);
}

// ------------------------------------------------------------------------------------------------ reports
const jobHref = (j) => (j.mode && j.mode.startsWith("documents") ? `#/library/job/${j.id}` : `#/${j.mode}/job/${j.id}`);
const JOB_KINDS = { documents: "Document question", "documents-verify": "Document claim check", "documents-index": "Library index", verify: "Verification", research: "Research" };

function reportsView(container, ctx) {
  let filter = "all";
  let query = "";
  let runs = [];
  const listArea = el("div", { class: "stack" }, loading("Loading saved reports"));
  const liveArea = el("div", {});
  const filterArea = el("div", {});
  const draw = () => {
    const needle = query.trim().toLowerCase();
    const shown = runs.filter((r) => (filter === "all" || r.mode === filter) && (!needle || `${r.input} ${r.name}`.toLowerCase().includes(needle)));
    append(clear(listArea), shown.length ? sessionList(shown)
      : emptyState({ iconName: "sessions", title: runs.length ? "No report matches" : "No saved reports yet",
        body: runs.length ? "Change the filter or the search text." : "Completed research runs and verifications are saved automatically and listed here.",
        actions: runs.length ? [] : [el("a", { class: "btn btn-primary", href: "#/research" }, "Start research")] }),
    runs.length ? el("p", { class: "xsmall muted" }, `Showing ${shown.length} of ${plural(runs.length, "saved report")}.`) : null);
  };
  container.append(el("div", { class: "stack-lg" }, liveArea, el("section", { "aria-label": "Saved reports" }, filterArea, listArea)));
  api.runs({ signal: ctx.signal }).then((payload) => {
    runs = payload.runs || [];
    const count = (mode) => runs.filter((r) => mode === "all" || r.mode === mode).length;
    clear(filterArea).append(el("div", { class: "toolbar" },
      chips([{ value: "all", label: "All", count: count("all") }, { value: "research", label: "Research", count: count("research") },
        { value: "verify", label: "Verification", count: count("verify") }], filter, (value) => { filter = value; draw(); }, "Filter reports"),
      searchField({ label: "Search reports", placeholder: "Search questions and claims", onInput: (v) => { query = v; draw(); } })));
    draw();
  }).catch((error) => { if (error.kind !== "aborted") clear(listArea).append(errorCallout(error, () => ctx.reload())); });
  api.jobs({ signal: ctx.signal }).then(({ jobs }) => {
    if (!(jobs || []).length) return;
    liveArea.append(el("section", { "aria-labelledby": "live-h" }, sectionHead("", "This server session", { id: "live-h", aside: "held in memory until the server stops" }),
      el("ul", { class: "session-list" }, jobs.map((j) => el("li", {}, el("a", { class: "session", href: jobHref(j) },
        el("span", { class: "session-icon" }, icon(j.mode && j.mode.startsWith("documents") ? "library" : j.mode === "verify" ? "verify" : "research")),
        el("span", { class: "session-q" }, j.input || (j.mode === "documents-index" ? "Index the document library" : j.mode)),
        el("span", {}, j.state === "done" ? (j.status ? statusBadge(j.status) : null) : j.state === "cancelled" ? statusBadge("CANCELLED")
          : badge(j.state === "queued" ? `queued${j.position ? ` · ${j.position}` : ""}` : "running", "info")),
        el("span", { class: "session-meta" }, el("span", {}, JOB_KINDS[j.mode] || j.mode), j.outcome ? outcomeBadge(j.outcome) : null)))))));
  }).catch(() => {});
}

// ------------------------------------------------------------------------------------------------ sources across runs
const LIBRARY_SOURCE_FILTERS = {
  all: { label: "All", test: () => true },
  used: { label: "Used as evidence", test: (s) => s.used_in > 0 },
  full: { label: "Read in full", test: (s) => ACCESS_LEVEL[s.access] === "full" },
  partial: { label: "Partly read", test: (s) => ACCESS_LEVEL[s.access] === "partial" },
  unread: { label: "Not read", test: (s) => (ACCESS_LEVEL[s.access] || "none") === "none" },
};

function sourcesLibraryView(container, ctx) {
  let filter = "all";
  let query = "";
  let sources = [];
  const toolbar = el("div", {});
  const list = el("div", {}, loading("Gathering sources from the saved reports"));
  container.append(el("div", { class: "stack" }, toolbar, list));
  const draw = () => {
    const needle = query.trim().toLowerCase();
    const rule = LIBRARY_SOURCE_FILTERS[filter].test;
    const shown = sources.filter((s) => rule(s) && (!needle || [s.title, s.publisher, s.provider, s.url].some((v) => String(v || "").toLowerCase().includes(needle))));
    append(clear(list), shown.length ? el("ul", { class: "ledger" }, shown.slice(0, 200).map((s) => el("li", {}, sourceSummary(s))))
      : emptyState({ iconName: "book", title: sources.length ? "No source matches" : "No sources yet", body: sources.length ? "Change the filter or the search text." : "Sources appear here once a research run has been saved." }),
    shown.length > 200 ? el("p", { class: "xsmall muted" }, `Showing the first 200 of ${shown.length}. Search to narrow the list.`) : null);
  };
  api.sources({ signal: ctx.signal }).then((payload) => {
    sources = payload.sources || [];
    const count = (key) => sources.filter(LIBRARY_SOURCE_FILTERS[key].test).length;
    clear(toolbar).append(el("p", { class: "small muted" }, "Merged from the saved run records by the address each source was read from. Nothing is fetched again; "
      + "a source shows as read only if its text was obtained in at least one run."),
    el("div", { class: "toolbar" },
      chips(Object.entries(LIBRARY_SOURCE_FILTERS).filter(([key]) => key === "all" || count(key)).map(([value, f]) => ({ value, label: f.label, count: count(value) })),
        filter, (v) => { filter = v; draw(); }, "Filter sources"),
      searchField({ label: "Search sources", placeholder: "Search titles, publishers, addresses", onInput: (v) => { query = v; draw(); } })));
    draw();
  }).catch((error) => { if (error.kind !== "aborted") clear(list).append(errorCallout(error, () => ctx.reload())); });
}

function sourceSummary(s) {
  const used = s.runs.filter((r) => r.used);
  return el("article", { class: "source-entry" },
    el("div", { class: "stack-sm" },
      el("div", { class: "source-title" }, externalLink(s.url, s.title)),
      el("div", { class: "source-meta" }, kindMark(s.kind), s.publisher ? el("span", {}, s.publisher) : null,
        s.published ? el("span", {}, String(s.published).slice(0, 10)) : null, s.provider ? el("span", {}, `via ${s.provider}`) : null)),
    el("div", { class: "source-actions" }, accessBadge(s.access)),
    el("div", { class: "source-runs" },
      el("span", { class: "xsmall muted" }, `Selected in ${plural(s.runs.length, "report")}${used.length ? `, used as evidence in ${used.length}` : ""}:`),
      el("ul", { class: "run-links" }, s.runs.slice(0, 6).map((r) => el("li", {},
        el("a", { href: `#/${r.mode === "verify" ? "verify" : "research"}/run/${encodeURIComponent(r.name)}?view=sources` }, truncate(r.input || r.name, 80)),
        r.used ? el("span", { class: "tag-used" }, "used") : null)),
      s.runs.length > 6 ? el("li", { class: "xsmall muted" }, `and ${s.runs.length - 6} more`) : null)));
}

// ------------------------------------------------------------------------------------------------ documents
function documentsView(container, ctx) {
  const listArea = el("div", { class: "stack-lg" }, loading("Reading the document folder"));
  const side = el("div", { class: "stack" });
  container.append(el("div", { class: "split" }, el("div", { class: "stack-lg" }, listArea), side));
  const load = () => api.library({ signal: ctx.signal }).then((lib) => draw(lib)).catch((error) => {
    if (error.kind !== "aborted") clear(listArea).append(errorCallout(error, load));
  });

  function draw(lib) {
    clear(listArea);
    clear(side);
    if (lib.available === false) {
      listArea.append(emptyState({ title: "The document library is not available", body: "This server was started without a document folder." }));
      return;
    }
    listArea.append(uploadZone(ctx, lib, load));
    if (!lib.exists) {
      listArea.append(emptyState({ iconName: "library", title: "No document folder yet", body: `Add a TXT or PDF file to create ${lib.folder}.` }));
    } else if (!lib.documents.length) {
      listArea.append(emptyState({ iconName: "file", title: "The library is empty", body: `Add TXT or PDF files here, or place them in ${lib.folder}.` }));
    } else {
      listArea.append(documentTable(lib));
    }
    side.append(assistantPanel(ctx, lib));
  }
  load();
}

function documentTable(lib) {
  let query = "";
  const tbody = el("tbody", {});
  const count = el("span", {});
  const row = (doc) => {
    const preview = el("button", { class: "btn btn-ghost btn-sm", type: "button", disabled: !doc.readable, "aria-label": `Preview ${doc.name}` }, icon("doc"), "Preview");
    preview.addEventListener("click", () => modal({ title: doc.name, subtitle: `page 1 of ${doc.pages} · first 600 characters as extracted`,
      body: el("div", { class: "stack" }, provenance("retrieved"), el("p", { class: "context" }, doc.preview || "(no text)")) }));
    return el("tr", {},
      el("td", {}, el("div", { class: "passage-source" }, doc.name), el("div", { class: "xsmall muted" }, `modified ${formatDate(doc.modified)}`)),
      el("td", {}, el("span", { class: "kind hue-ochre" }, doc.type.toUpperCase())),
      el("td", { class: "num" }, formatBytes(doc.bytes)),
      el("td", { class: "num" }, doc.readable ? String(doc.pages) : "—"),
      el("td", {}, doc.readable ? badge("Readable", "support", { iconName: "check" }) : badge("Skipped", "uncertain", { title: doc.skipped_reason, iconName: "alert" }),
        doc.skipped_reason ? el("div", { class: "xsmall muted" }, doc.skipped_reason) : null),
      el("td", {}, preview));
  };
  const draw = () => {
    const needle = query.trim().toLowerCase();
    const shown = lib.documents.filter((d) => !needle || d.name.toLowerCase().includes(needle));
    append(clear(tbody), shown.length ? shown.map(row) : el("tr", {}, el("td", { colspan: "6", class: "muted" }, "No document name matches the search.")));
    count.textContent = `${shown.length} of ${plural(lib.documents.length, "file")}`;
  };
  draw();
  return el("section", { "aria-labelledby": "docs-h" },
    sectionHead("", "Documents", { id: "docs-h", aside: el("span", { class: "mono" }, lib.folder) }),
    lib.documents.length > 1 ? el("div", { class: "toolbar" }, searchField({ label: "Search documents", placeholder: "Search file names", onInput: (v) => { query = v; draw(); } }),
      el("span", { class: "small muted" }, count)) : null,
    el("div", { class: "table-wrap" }, el("table", {},
      el("thead", {}, el("tr", {}, el("th", { scope: "col" }, "Name"), el("th", { scope: "col" }, "Type"), el("th", { scope: "col", class: "num" }, "Size"),
        el("th", { scope: "col", class: "num" }, "Pages"), el("th", { scope: "col" }, "Ingestion"), el("th", { scope: "col" }, el("span", { class: "visually-hidden" }, "Actions")))),
      tbody,
      el("caption", {}, "Ingestion status comes from docqa's own loader: a file is readable when text could be extracted from it. That is not the same as being indexed; see the index status."))));
}

function uploadZone(ctx, lib, reload) {
  const input = el("input", { type: "file", id: "upload-input", accept: ".txt,.pdf,text/plain,application/pdf", class: "visually-hidden" });
  const status = el("p", { class: "small", "aria-live": "polite" });
  const button = el("button", { class: "btn", type: "button" }, icon("upload"), "Add a document");
  const zone = el("div", { class: "dropzone" }, icon("upload"),
    el("p", {}, el("b", {}, "Drop a TXT or PDF file here"), " or ", button),
    el("p", { class: "xsmall muted" }, `Up to ${formatBytes(lib.max_upload_bytes || 0)}. Existing names are never overwritten. Files are stored in ${lib.folder}.`),
    input, status);
  button.addEventListener("click", () => input.click());
  const send = async (file) => {
    if (!file) return;
    const suffix = file.name.toLowerCase().slice(file.name.lastIndexOf("."));
    if (![".txt", ".pdf"].includes(suffix)) { status.className = "small form-error"; status.textContent = "Only .txt and .pdf files can be added."; return; }
    if (lib.max_upload_bytes && file.size > lib.max_upload_bytes) { status.className = "small form-error"; status.textContent = "The file is larger than the upload limit."; return; }
    button.disabled = true;
    status.className = "small";
    status.textContent = `Uploading ${file.name}…`;
    try {
      const result = await api.upload(file);
      ctx.toast(`Added ${result.saved}`);
      reload();
    } catch (error) {
      status.className = "small form-error";
      status.textContent = error.message;
      button.disabled = false;
    }
  };
  input.addEventListener("change", () => send(input.files && input.files[0]));
  zone.addEventListener("dragover", (event) => { event.preventDefault(); zone.classList.add("is-over"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("is-over"));
  zone.addEventListener("drop", (event) => { event.preventDefault(); zone.classList.remove("is-over"); send(event.dataTransfer.files && event.dataTransfer.files[0]); });
  return zone;
}

function assistantPanel(ctx, lib) {
  const readable = (lib.documents || []).filter((d) => d.readable).length;
  const index = lib.index || { state: "not_built" };
  const [label, tone, note] = INDEX_STATES[index.state] || [index.state, "neutral", ""];
  const disabled = !readable;
  const indexCard = el("section", { class: "card stack", "aria-labelledby": "index-h" },
    el("h2", { class: "card-title", id: "index-h" }, "Search index"),
    el("div", { class: "row" }, badge(label, tone)),
    note ? el("p", { class: "small muted" }, note) : null,
    index.state === "ready" ? el("p", { class: "small" }, `${plural(index.documents.length, "document")}, ${plural(index.pages, "page")} indexed ${formatDate(index.built_at)} in this server session.`) : null,
    index.error ? el("p", { class: "small form-error" }, index.error) : null,
    el("p", { class: "xsmall muted" }, "Indexing runs rag.py's hybrid index with the locally cached embedding models. The index lives in server memory and is rebuilt after a restart."),
    el("div", {}, submitButton(ctx, "documents-index", "", disabled, ["database", index.state === "ready" ? "Rebuild index" : "Build index"])));
  const form = (mode, title, placeholder, action) => {
    const area = el("textarea", { class: "textarea", rows: 3, placeholder, id: `docs-${mode}`, maxlength: "2000", disabled });
    const message = el("p", { class: "small", "aria-live": "polite" });
    const button = el("button", { class: "btn btn-primary", type: "submit", disabled }, action, icon("arrow"));
    const node = el("form", { class: "stack-sm", novalidate: true }, el("label", { class: "label", for: `docs-${mode}` }, title), area, message, el("div", { class: "row" }, button));
    node.addEventListener("submit", async (event) => {
      event.preventDefault();
      const problem = validateQuestion(area.value);
      if (problem) { message.className = "small form-error"; message.textContent = problem; return; }
      button.disabled = true;
      try {
        const { id } = await api.submit(area.value.trim(), mode);
        ctx.navigate(`#/library/job/${id}`);
      } catch (error) { message.className = "small form-error"; message.textContent = error.message; button.disabled = false; }
    });
    return node;
  };
  return el("aside", { class: "stack", "aria-label": "Document assistant" }, indexCard,
    el("section", { class: "card stack", "aria-labelledby": "ask-h" }, el("h2", { class: "card-title", id: "ask-h" }, "Ask the documents"),
      disabled ? callout("info", "No readable documents", el("p", {}, "Add a TXT or PDF file with extractable text to ask questions.")) : null,
      form("documents", "Question", "What does the report conclude about…?", "Ask"),
      el("hr", { class: "rule" }),
      form("documents-verify", "Claim to check against the documents", "The study found that…", "Check claim"),
      el("p", { class: "xsmall muted" }, "Answers use the local model through Ollama and the docqa default verifier (frozen_b). Document answers are kept in server memory only; docqa does not save them.")));
}

function submitButton(ctx, mode, input, disabled, [iconName, label]) {
  const button = el("button", { class: "btn", type: "button", disabled }, icon(iconName), label);
  button.addEventListener("click", async () => {
    button.disabled = true;
    try {
      const { id } = await api.submit(input, mode);
      ctx.navigate(`#/library/job/${id}`);
    } catch (error) { ctx.toast(error.message); button.disabled = false; }
  });
  return button;
}

export function renderLibraryJob(view, ctx, id) {
  ctx.setTitle("Document assistant");
  const progress = el("div", {}, loading("Connecting to the job"));
  const result = el("div", { class: "stack-lg" });
  view.append(el("div", { class: "page page-narrow" }, el("div", { class: "stack" },
    el("a", { class: "btn btn-ghost btn-sm", href: "#/library?view=documents" }, icon("back"), "Back to documents"),
    el("h1", { class: "visually-hidden" }, "Document assistant"), progress, result)));
  const follow = new AbortController();
  ctx.signal.addEventListener("abort", () => follow.abort(), { once: true });
  let stopped = false;
  let card = null;
  let lastData = null;

  async function cancel() {
    if (stopped) return;
    card.setCancelling(true);
    let outcome;
    try { outcome = await api.cancel(id); } catch (error) {
      outcome = error.status === 409 && error.payload ? error.payload : error.status === 404 ? { state: "gone" } : { state: "unreachable", error: error.message };
    }
    if (outcome.state === "done") { card.setCancelling(false); ctx.toast("The job had already finished, so there was nothing to cancel."); return; }
    stopped = true;
    follow.abort();
    card.destroy();
    clear(progress);
    clear(result).append(cancelledJob(ctx, outcome, lastData));
  }

  pollJob(id, (job) => {
    if (stopped) return;
    const data = normalizeDocumentJob(job.data);
    lastData = data;
    if (job.state === "queued" || job.state === "running") {
      if (!card) {
        card = progressView(data.mode, { onCancel: cancel, noun: "job", eyebrow: JOB_TITLES[data.mode] || "Document job", headingLevel: "h2",
          title: data.mode === "documents-index" ? "Indexing the document library" : "",
          runningNote: "This job has started and cannot be interrupted safely. Cancelling stops this page following it; the job still finishes in the server, and its result is held in memory, not saved." });
        clear(progress).append(card.node);
      }
      card.update(job, { input: data.input, current: data.current, sources: [], pool: [] });
      return;
    }
    stopped = true;
    if (card) card.destroy();
    clear(progress);
    clear(result).append(...(job.state === "cancelled" ? [cancelledJob(ctx, { state: "cancelled" }, data)] : documentResult(data)));
  }, { signal: follow.signal }).catch((error) => {
    if (stopped) return;
    if (card) card.destroy();
    clear(progress);
    clear(result).append(error.status === 404
      ? emptyState({ title: "This job is no longer held by the server", body: "Document jobs are kept in memory only while the server runs.", actions: [el("a", { class: "btn", href: "#/library?view=documents" }, "Back to documents")] })
      : errorCallout(error, () => ctx.reload()));
  });
}

function cancelledJob(ctx, outcome, data) {
  const texts = {
    cancelled: ["Cancelled before it started", "The job was withdrawn from the queue. Nothing was indexed or asked."],
    running: ["Stopped following this job", "It had already started and cannot be interrupted safely, so it is still running in the server. Its result is held in memory and appears in Library, under Reports, while the server runs."],
    gone: ["This job is no longer held by the server", "Document jobs are kept in memory only while the server runs."],
    unreachable: ["Stopped following this job", `The server could not be reached to cancel it (${outcome.error || "no response"}), so it may still be running.`],
  };
  const [title, body] = texts[outcome.state] || texts.gone;
  return el("section", { class: "sheet sheet-pad stack" },
    el("p", { class: "eyebrow" }, (data && JOB_TITLES[data.mode]) || "Document job"),
    el("h2", { class: "run-question" }, data && data.input ? data.input : data && data.mode === "documents-index" ? "Indexing the document library" : "Document job"),
    el("div", { class: "row" }, outcome.state === "cancelled" ? statusBadge("CANCELLED") : badge("Not followed", "neutral")),
    callout("note", title, el("p", {}, body)),
    el("div", { class: "row" }, outcome.state === "running" || outcome.state === "unreachable" ? el("button", { class: "btn", type: "button", on: { click: () => ctx.reload() } }, icon("refresh"), "Follow it again") : null,
      el("a", { class: "btn", href: "#/library?view=documents" }, "Back to documents")));
}

function documentResult(data) {
  const parts = [];
  if (data.mode === "documents-index") {
    parts.push(el("header", { class: "run-header" }, el("p", { class: "eyebrow" }, "Search index"), el("h2", { class: "run-question" }, data.status === "COMPLETE" ? "The library was indexed" : "Indexing failed")));
    parts.push(data.error ? callout("error", "Indexing failed", el("p", {}, data.error)) : callout("success", "Ready", el("p", {}, "Questions and claim checks now use this index until the documents change.")));
    parts.push(timeline(data.stages, data.mode));
    return parts;
  }
  const evidenceCard = (e, stance) => el("article", { class: `passage stance-${stance || "none"}`, id: `ev-${e.id}` },
    el("div", { class: "passage-head" }, el("span", { class: "passage-source" }, el("span", { class: "source-id" }, e.id), e.citation || `${e.document}, p. ${e.page}`), stance ? stanceTag(stance) : null),
    el("p", { class: "passage-text" }, e.text),
    el("div", { class: "passage-meta" }, provenance("retrieved"), e.score !== null ? el("span", { class: "adv-only" }, `score ${e.score.toFixed(3)}`) : null));
  if (data.mode === "documents") {
    parts.push(el("header", { class: "run-header" }, el("p", { class: "eyebrow" }, "Question about your documents"), el("h2", { class: "run-question" }, data.input),
      el("div", { class: "meta-line" }, statusBadge(data.status), data.qaStatus ? outcomeBadge(data.qaStatus) : null, el("span", {}, "docqa verifier frozen_b"))));
    if (data.error) parts.push(callout("error", "The assistant could not answer", el("p", {}, data.error)));
    if (data.answer) {
      parts.push(el("section", { class: "sheet sheet-pad stack", "aria-labelledby": "answer-h" },
        el("div", { class: "row-between" }, el("p", { class: "eyebrow", id: "answer-h" }, "Direct answer"), provenance("model")),
        el("p", { class: "finding-text" }, data.answer),
        el("p", { class: "xsmall muted" }, "Written by the local model from the evidence below. Its individual statements were checked separately; see their verdicts.")));
    }
    if (data.claims.length) {
      parts.push(el("section", { "aria-labelledby": "checked-h" }, sectionHead("01", "Checked statements", { id: "checked-h", aside: plural(data.claims.length, "statement") }),
        el("div", { class: "stack" }, data.claims.map((c, i) => el("article", { class: "finding is-static" },
          el("span", { class: "finding-num" }, String(i + 1).padStart(2, "0")),
          el("div", {}, el("p", { class: "finding-text" }, c.text),
            el("div", { class: "finding-meta" }, c.verdict ? verdictBadge(c.verdict.label) : badge("no verdict", "neutral"), provenance("verifier"),
              c.verdict && c.verdict.subState ? el("span", {}, subStateText(c.verdict.subState)) : null,
              c.cited.length ? el("span", {}, "cited: ", c.cited.map((cid) => el("a", { class: "cite", href: `#ev-${cid}`, on: { click: (e) => { e.preventDefault(); const t = document.getElementById(`ev-${cid}`); if (t) t.scrollIntoView({ block: "center" }); } } }, cid))) : el("span", {}, "no citation"))))))));
    }
    if (data.limitations.length) parts.push(el("section", {}, sectionHead("02", "Limitations"), el("ul", { class: "note-list tone-caution" }, data.limitations.map((l) => el("li", {}, l)))));
    parts.push(el("section", { "aria-labelledby": "evidence-h" }, sectionHead("03", "Retrieved evidence", { id: "evidence-h", aside: plural(data.evidence.length, "passage") }),
      data.evidence.length ? el("div", { class: "stack" }, data.evidence.map((e) => evidenceCard(e))) : emptyState({ title: "No passages were retrieved" })));
    return parts;
  }
  // documents-verify
  const verdict = data.verdict;
  const meta = verdict ? verdictMeta(verdict.label) : { tone: "error", label: "No verdict", rule: "" };
  parts.push(el("section", { class: "sheet verdict-hero" },
    el("div", { class: "stack" }, el("p", { class: "eyebrow" }, "Claim checked against your documents"), el("h2", { class: "verdict-claim" }, data.input),
      meta.rule ? el("p", { class: "verdict-rule" }, el("b", {}, "Rule applied: "), meta.rule) : null,
      verdict && verdict.subState ? el("p", { class: "verdict-rule" }, el("b", {}, "Why: "), subStateText(verdict.subState)) : null,
      el("div", { class: "meta-line" }, statusBadge(data.status), el("span", {}, "docqa verifier frozen_b"))),
    el("div", { class: "verdict-side" }, el("span", { class: "eyebrow" }, "Verdict"), verdictStamp(verdict ? verdict.label : null, { failed: Boolean(data.error) || !verdict }), provenance("verifier"))));
  if (data.error) parts.push(callout("error", "The check did not complete", el("p", {}, data.error)));
  parts.push(callout("info", "What this verdict means", el("p", {}, "The verdict is decided by fixed rules from per-passage judgements of the retrieved pages. It does not prove the claim correct.")));
  parts.push(el("section", { "aria-labelledby": "judged-h" }, sectionHead("01", "Judged passages", { id: "judged-h", aside: plural(data.evidence.length, "passage") }),
    data.evidence.length ? el("div", { class: "stack" }, data.evidence.map((e) => evidenceCard(e, verdict ? verdict.stances[e.id] || "NOT_ASSESSED" : null)))
      : emptyState({ title: "No passages were retrieved" })));
  if (data.limitations.length) parts.push(disclosure(`Limitations (${data.limitations.length})`, el("ul", { class: "prose-list" }, data.limitations.map((l) => el("li", {}, l))), true));
  return parts;
}
