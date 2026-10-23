// Research workspace and claim verification studio: composer, live progress with cancellation, report and evidence.
import { api, pollJob } from "./api.mjs";
import {
  badge, callout, cautionBadge, cautionList, chips, completenessBadge, copyButton, coverageBadge, disclosure, emptyState,
  errorCallout, externalLink, kv, loading, menuButton, modal, outcomeBadge, passageCard, provenance, searchField, sectionHead, segmented, skeleton,
  sourceEntry, statusBadge, stepper, strengthBadge, tabbar, timeline, verdictBadge, verdictStamp,
} from "./components.mjs";
import { append, clear, el, icon } from "./dom.mjs";
import { preferences } from "./prefs.mjs";
import {
  ACCESS, ACCESS_LEVEL, CATEGORY_LABELS, DEPTH_LABELS, RECENCY_LABELS, SOURCE_FILTERS, activeDuplicate, bibliography,
  budgetRows, citationsFor, completenessMeta, describeBudget, elapsedSince, filterSources, findingsBySubquestion,
  formatDate, formatSeconds, groupByStance, independenceText, noAnswerState, normalizeRun, outcomeMeta, plural, progressSteps,
  providerStateMeta, queryStatusMeta, sourceUsage, strengthMeta,
  subStateText, truncate, validateQuestion, verdictMeta,
} from "./model.mjs";

const MAX_INPUT = 2000;
const COPY = {
  research: {
    title: "Research workspace", eyebrow: "Research question", tab: "Research a question", noun: "research",
    hint: "The planner writes search queries, the web researcher reads the sources it can reach, the synthesizer drafts cited statements, and the verifier checks each one against the retrieved passages.",
    lede: "Ask a question and follow the investigation: which sources were found and read, which passages were retrieved, and which statements the verifier could support.",
    placeholder: "What architectural idea did ResNet introduce, and why did it help training?",
    action: "Start research", label: "Research question",
  },
  verify: {
    title: "Claim verification", eyebrow: "Claim under examination", tab: "Verify a claim", noun: "verification",
    hint: "Sources are searched and read, the most relevant passages are retrieved, and each passage is judged as supporting, contradicting or neutral before fixed rules decide the verdict.",
    lede: "Enter a factual claim, or several, one per line (up to 8). Each verdict comes with the passages judged for and against it, and with what could not be established.",
    placeholder: "The Eiffel Tower was completed in 1889.",
    action: "Verify", label: "Claims to verify, one per line",
  },
};

// ------------------------------------------------------------------------------------------------ composer
export function composer(ctx, { mode = "research", switchable = false, value = "", onMode = null } = {}) {
  let current = mode;
  const area = el("textarea", { class: "textarea", id: "composer-input", name: "input", rows: 3, maxlength: String(MAX_INPUT),
    placeholder: COPY[current].placeholder, "aria-describedby": "composer-intro composer-hint composer-message" });
  area.value = value;
  const label = el("label", { class: "visually-hidden", for: "composer-input" }, COPY[current].label);
  const intro = el("p", { class: "composer-intro", id: "composer-intro" }, COPY[current].hint);
  const counter = el("span", { class: "hint", id: "composer-hint" });
  const message = el("p", { class: "small", id: "composer-message", "aria-live": "polite" });
  const submitLabel = el("span", {}, COPY[current].action);
  const submit = el("button", { class: "btn btn-primary", type: "submit" }, submitLabel, icon("arrow"));
  const updateCounter = () => { counter.textContent = `${area.value.trim().length}/${MAX_INPUT} · Ctrl+Enter to submit`; };
  updateCounter();
  area.addEventListener("input", () => { updateCounter(); message.textContent = ""; });
  area.addEventListener("keydown", (event) => { if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) { event.preventDefault(); form.requestSubmit(); } });
  const setMode = (next) => {
    current = next;
    area.placeholder = COPY[next].placeholder;
    label.textContent = COPY[next].label;
    intro.textContent = COPY[next].hint;
    submitLabel.textContent = COPY[next].action;
    if (onMode) onMode(next);
  };
  const tabs = switchable
    ? el("div", { class: "notebook-tabs", role: "radiogroup", "aria-label": "What to do" }, ["research", "verify"].map((m) => el("label", { class: "notebook-tab" },
      el("input", { type: "radio", name: "composer-mode", value: m, checked: m === current, on: { change: () => setMode(m) } }),
      el("span", {}, icon(m), COPY[m].tab))))
    : el("div", { class: "notebook-tabs", "aria-hidden": "true" }, el("span", { class: "notebook-tab is-static" }, el("span", {}, icon(current), COPY[current].tab)));
  const queueNote = el("p", { class: "small muted", hidden: true });
  const controls = researchControls(ctx);
  let busy = false;
  const form = el("form", { class: "composer", novalidate: true },
    label, intro, area, controls.node,
    el("div", { class: "composer-bar" }, counter, submit));
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (busy) return;
    const problem = validateQuestion(area.value, MAX_INPUT) || controls.problem();
    if (problem) { message.textContent = problem; message.className = "small form-error"; area.focus(); return; }
    busy = true;
    submit.disabled = true;
    submitLabel.textContent = "Submitting…";
    message.textContent = "";
    const input = area.value.trim();
    try {
      const existing = await api.jobs({ timeout: 5000 }).then(({ jobs }) => activeDuplicate(jobs, input, current)).catch(() => null);
      if (existing) {
        ctx.toast("This question is already in progress; showing that run instead of starting another.");
        ctx.navigate(`#/${current}/job/${existing.id}`);
        return;
      }
      const { id } = await api.submit(input, current, controls.options());
      ctx.navigate(`#/${current}/job/${id}`);
    } catch (error) {
      message.textContent = error.message;
      message.className = "small form-error";
      busy = false;
      submit.disabled = false;
      submitLabel.textContent = COPY[current].action;
    }
  });
  api.jobs({ signal: ctx.signal }).then(({ jobs }) => {
    const active = (jobs || []).filter((j) => j.state === "queued" || j.state === "running");
    if (active.length) {
      queueNote.hidden = false;
      queueNote.textContent = `${plural(active.length, "run")} already waiting or running on this server. Runs are processed one at a time, so a new run waits its turn.`;
    }
  }).catch(() => {});
  return el("div", { class: "composer-wrap" }, tabs, form, el("div", { class: "stack-sm" }, message, queueNote));
}

// Research controls under the question box. Depth changes the real budget (shown from the server's presets); the
// source list holds only providers that are active now; source types are a soft preference for the planner; the time
// range is applied by providers that support it and to recorded dates afterwards.
function researchControls(ctx) {
  const state = { depth: ctx.prefs.defaultDepth || "standard", providers: null, categories: [], recency: "any", active: null, loaded: false };
  const node = el("div", { class: "composer-options" }, el("span", { class: "hint" }, "Loading search options…"));
  const hint = el("p", { class: "hint composer-budget", "aria-live": "polite" });
  api.providers({ signal: ctx.signal }).then((payload) => {
    state.loaded = true;
    state.depths = payload.depths;
    state.active = payload.providers.filter((p) => p.active);
    state.providers = state.active.map((p) => p.id);
    const categories = Object.keys(payload.categories || {}).filter((c) => (payload.categories[c] || []).some((id) => state.providers.includes(id)));
    const showBudget = () => {
      const limits = state.depths && state.depths[state.depth];
      hint.textContent = limits ? `${DEPTH_LABELS[state.depth]}: ${describeBudget(limits)}.` : "";
    };
    const depth = segmented("composer-depth", Object.keys(DEPTH_LABELS).map((d) => ({ value: d, label: DEPTH_LABELS[d] })), state.depth,
      (value) => { state.depth = value; showBudget(); }, "Research depth");
    const summary = el("span", {});
    const countSources = () => { summary.textContent = `Sources: ${state.providers.length} of ${state.active.length}`; };
    const picker = el("details", { class: "source-picker" }, el("summary", {}, icon("globe"), summary),
      el("div", { class: "source-picker-body" },
        state.active.length ? state.active.map((p) => el("label", { class: "check" },
          el("input", { type: "checkbox", checked: true, value: p.id, on: { change: (event) => {
            state.providers = event.target.checked ? [...new Set([...state.providers, p.id])] : state.providers.filter((x) => x !== p.id);
            countSources();
          } } }),
          el("span", {}, p.name, el("small", {}, ` ${CATEGORY_LABELS[p.category] || p.category}`))))
          : el("p", { class: "small form-error" }, "No search source is active."),
        el("p", { class: "hint" }, "Only sources that are switched on and set up are listed. ", el("a", { href: "#/integrations" }, "Set up more"), ".")));
    countSources();
    const prefer = el("div", { class: "chips", role: "group", "aria-label": "Preferred source types" }, categories.map((c) => {
      const button = el("button", { class: "chip", type: "button", "aria-pressed": "false" }, CATEGORY_LABELS[c] || c);
      button.addEventListener("click", () => {
        const on = button.getAttribute("aria-pressed") !== "true";
        button.setAttribute("aria-pressed", String(on));
        state.categories = on ? [...state.categories, c] : state.categories.filter((x) => x !== c);
      });
      return button;
    }));
    const recency = el("select", { class: "select select-sm", "aria-label": "Time range", on: { change: (event) => { state.recency = event.target.value; } } },
      (payload.recency || ["any"]).map((r) => el("option", { value: r }, RECENCY_LABELS[r] || r)));
    clear(node).append(
      el("div", { class: "composer-option" }, el("span", { class: "option-label" }, "Depth"), depth),
      el("div", { class: "composer-option" }, picker),
      el("div", { class: "composer-option" }, el("span", { class: "option-label", title: "A preference for the planner, not a filter" }, "Prefer"), prefer),
      el("div", { class: "composer-option" }, el("span", { class: "option-label" }, "Time"), recency),
      hint);
    showBudget();
  }).catch((error) => {
    if (error.kind === "aborted") return;
    clear(node).append(el("span", { class: "hint" }, `Search options could not be loaded (${error.message}); the server's defaults apply.`));
  });
  return {
    node,
    options: () => (state.loaded ? { depth: state.depth, providers: state.providers, categories: state.categories, recency: state.recency }
      : { depth: state.depth }),
    problem: () => (state.loaded && !state.providers.length ? "Choose at least one source to search, or set one up in Integrations." : ""),
  };
}

// ------------------------------------------------------------------------------------------------ start pages
// One page for both kinds of investigation; the tabs on the question box switch between them.
export function renderStart(view, ctx, mode) {
  ctx.setTitle("New investigation");
  const how = el("div", {}, mode === "research" ? howResearchWorks() : howVerificationWorks());
  const lede = el("p", { class: "lede" }, COPY[mode].lede);
  view.append(el("div", { class: "page page-narrow" },
    el("header", { class: "page-header" }, el("div", {}, el("p", { class: "eyebrow" }, "Research"), el("h1", {}, "New investigation"), lede)),
    el("div", { class: "stack-lg" },
      composer(ctx, { mode, switchable: true, onMode: (next) => {
        history.replaceState(null, "", `#/${next}`);
        lede.textContent = COPY[next].lede;
        clear(how).append(next === "research" ? howResearchWorks() : howVerificationWorks());
      } }),
      el("section", { "aria-labelledby": "how-h" }, sectionHead("", "How it works", { id: "how-h" }), how))));
}

function howResearchWorks() {
  return el("ol", { class: "steps", "aria-label": "How a research run works" },
    el("li", {}, el("strong", {}, "Plan by source type"), "The local model splits the question into sub-questions and writes queries for the source types that suit each one, sent to the active sources (by default OpenAlex and arXiv; Wikipedia is not accepted as evidence)."),
    el("li", {}, el("strong", {}, "Search in rounds"), "Sources read in full or as abstracts are indexed; snippets and metadata only point to sources. After each round a coverage check sends follow-up queries for open questions, until the evidence is corroborated or the depth's budget is used."),
    el("li", {}, el("strong", {}, "Write and verify"), "The model drafts cited findings per question and the verifier judges each against retrieved passages. At Standard and Deep depth, single-source findings are also searched for counter-evidence. The direct answer restates verified findings only."));
}

function howVerificationWorks() {
  return el("ol", { class: "steps", "aria-label": "How a verification works" },
    el("li", {}, el("strong", {}, "Gather sources"), "Queries are planned for the claim, including ones looking for contrary evidence, and the results are read; Standard and Deep runs add follow-up rounds."),
    el("li", {}, el("strong", {}, "Judge passages"), "Each retrieved passage is judged as supporting, contradicting or neutral, with a quotation that must come from the passage."),
    el("li", {}, el("strong", {}, "Apply the rules"), "SUPPORTED needs support and no contradiction, CONTRADICTED the reverse; anything else is INSUFFICIENT."));
}

export function sessionList(runs) {
  return el("ul", { class: "session-list" }, runs.map((run) => {
    const mode = run.mode === "verify" ? "verify" : "research";
    const href = run.status === "UNREADABLE" ? null : `#/${mode}/run/${encodeURIComponent(run.name)}`;
    const body = [
      el("span", { class: "session-icon", title: mode === "verify" ? "Verification" : "Research" }, icon(mode)),
      el("span", { class: "session-q" }, run.input || run.name),
      el("span", {}, run.status ? statusBadge(run.status) : null),
      el("span", { class: "session-meta" },
        el("span", {}, mode === "verify" ? "Verification" : "Research"),
        run.outcome ? outcomeBadge(run.outcome) : null,
        run.started ? el("span", {}, formatDate(run.started)) : null,
        mode === "research" && run.claims ? el("span", {}, `${run.supported} of ${plural(run.claims, "statement")} supported`) : null,
        run.sources ? el("span", {}, plural(run.sources, "source")) : null,
        run.error ? el("span", {}, truncate(run.error, 120)) : null),
    ];
    return el("li", {}, href ? el("a", { class: "session", href }, body) : el("div", { class: "session" }, body));
  }));
}

// ------------------------------------------------------------------------------------------------ run page
export function renderRunPage(view, ctx, { mode, jobId, runName }) {
  const copy = COPY[mode];
  ctx.setTitle(copy.title);
  const top = el("div", { class: "row-between" },
    el("a", { class: "btn btn-ghost btn-sm", href: `#/${mode}` }, icon("back"), mode === "verify" ? "New claim" : "New question"),
    el("span", { class: "small muted mono adv-only" }, jobId ? `live run ${jobId}` : runName));
  const progress = el("div", {});
  const result = el("div", {});
  view.append(el("div", { class: "page" }, el("div", { class: "stack" }, top, progress, result)));

  const contextLoader = (passage) => (jobId ? api.jobContext(jobId, passage.sourceId, passage.key)
    : api.runContext(runName, passage.sourceId, passage.key));
  const helpers = { openContext: (passage, source) => openContext(contextLoader, passage, source), ctx };

  if (runName) {
    append(result, el("h1", { class: "visually-hidden" }, copy.title), skeleton(6));
    api.run(runName, { signal: ctx.signal }).then(({ data }) => {
      const run = normalizeRun(data);
      if (run.mode !== mode) { location.replace(`#/${run.mode}/run/${encodeURIComponent(runName)}`); return; }
      clear(result);
      renderResult(result, run, helpers, { saved: runName });
    }).catch((error) => {
      if (error.kind === "aborted") return;
      clear(result).append(el("h1", {}, copy.title), error.status === 404
        ? emptyState({ iconName: "sessions", title: "This saved report was not found", body: "It may have been moved or deleted.", actions: [el("a", { class: "btn", href: "#/library" }, "Browse reports")] })
        : errorCallout(error, () => ctx.reload()));
    });
    return;
  }
  followJob(ctx, { jobId, mode, progress, result, helpers });
}

// Follow a live job until it finishes, offering cancellation. Polling stops at once when the user cancels or leaves the
// page, and no response that arrives after that is drawn.
function followJob(ctx, { jobId, mode, progress, result, helpers }) {
  const copy = COPY[mode];
  const follow = new AbortController();
  ctx.signal.addEventListener("abort", () => follow.abort(), { once: true });
  let stopped = false;
  let lastState = "";
  const card = progressView(mode, { onCancel: cancel });
  append(progress, el("h1", { class: "visually-hidden" }, copy.title), loading("Connecting to the run"));

  async function cancel() {
    if (stopped) return;
    card.setCancelling(true);
    let outcome;
    try {
      outcome = await api.cancel(jobId);
    } catch (error) {
      if (error.status === 409 && error.payload) outcome = error.payload;
      else if (error.status === 404) outcome = { state: "gone" };
      else outcome = { state: "unreachable", error: error.message };
    }
    if (outcome.state === "done") {                 // it finished in the meantime: let polling draw the result
      card.setCancelling(false);
      ctx.toast("The run had already finished, so there was nothing to cancel.");
      return;
    }
    stopped = true;
    follow.abort();
    card.destroy();
    clear(progress);
    clear(result).append(cancelledView(ctx, mode, outcome, card.lastRun, jobId));
    ctx.announce(outcome.state === "cancelled" ? "Run cancelled" : "Stopped following the run");
  }

  pollJob(jobId, (job) => {
    if (stopped) return;
    const run = normalizeRun(job.data);
    if (job.state === "queued" || job.state === "running") {
      if (!card.node.isConnected) clear(progress).append(card.node);
      card.update(job, run);
      if (job.state !== lastState) ctx.announce(job.state === "queued" ? "Run queued" : "Run started");
      lastState = job.state;
      return;
    }
    stopped = true;
    card.destroy();
    clear(progress);
    clear(result);
    if (job.state === "cancelled") {
      result.append(cancelledView(ctx, mode, { state: "cancelled" }, run, jobId));
      return;
    }
    if (job.data && job.data.mode && job.data.mode !== mode && !String(job.data.mode).startsWith("documents")) {
      location.replace(`#/${job.data.mode}/job/${jobId}`);
      return;
    }
    ctx.announce(`Run finished: ${run.status.toLowerCase()}`);
    const savedName = run.savedTo ? run.savedTo.split(/[\\/]/).pop() : "";
    renderResult(result, run, helpers, { live: true, savedName });
  }, { signal: follow.signal }).catch((error) => {
    if (stopped) return;
    card.destroy();
    clear(progress);
    clear(result).append(el("h1", {}, copy.title), error.status === 404
      ? emptyState({ iconName: "sessions", title: "This run is no longer held by the server",
        body: "Live runs are kept in memory only while the server runs. Completed runs are saved and listed in Library.",
        actions: [el("a", { class: "btn", href: "#/library" }, "Browse reports")] })
      : errorCallout(error, () => ctx.reload()));
  });
}

// The progress card is built once and updated in place, so the Cancel button keeps its focus between polls.
const RUNNING_NOTE = "This run has started and the pipeline cannot be interrupted safely. Cancelling stops this page following it; "
  + "the run still finishes on the server and is saved to your Library.";

export function progressView(mode, { onCancel, noun = COPY[mode] ? COPY[mode].noun : "job", eyebrow = COPY[mode] ? COPY[mode].eyebrow : "Job",
  title = "", runningNote = RUNNING_NOTE, headingLevel = "h1" } = {}) {
  const label = el("p", { class: "eyebrow" });
  const heading = el(headingLevel, { class: "run-question" });
  const state = el("span", {});
  const elapsed = el("span", { class: "mono" });
  const steps = el("div", {});
  const soFar = el("div", {});
  const note = el("p", { class: "cancel-note" });
  const buttonLabel = el("span", {}, `Cancel ${noun}`);
  const button = el("button", { class: "btn btn-danger", type: "button" }, icon("stop"), buttonLabel);
  button.addEventListener("click", () => onCancel());
  let since = null;
  const tick = () => { const s = elapsedSince(since); elapsed.textContent = s === null ? "" : formatSeconds(s); };
  const timer = setInterval(tick, 1000);
  const node = el("section", { class: "sheet sheet-pad progress-sheet", "aria-label": "Progress" },
    el("div", { class: "stack" },
      label,
      heading,
      el("div", { class: "meta-line" }, state, elapsed),
      el("div", { class: "indeterminate", "aria-hidden": "true" }),
      steps,
      el("p", { class: "xsmall muted" }, "The pipeline reports each stage when it finishes; there is no percentage for work inside a stage, so none is shown.")),
    el("aside", { class: "progress-side", "aria-label": "Run controls" },
      el("div", { class: "card-title" }, "Control"),
      button, note,
      el("p", { class: "cancel-note" }, "Leaving this page does not cancel anything."),
      soFar));
  const view = {
    node, lastRun: null,
    update(job, run) {
      view.lastRun = run;
      const queued = job.state === "queued";
      label.textContent = queued ? `${eyebrow} · waiting in the queue` : `${eyebrow} · in progress`;
      heading.textContent = title || run.input || (job.data && job.data.input) || "";
      since = queued ? job.submitted : job.started || job.submitted;
      clear(state).append(queued ? badge(job.position ? `Queued · position ${job.position}` : "Queued", "info")
        : badge("Running", "info"), " ", el("span", {}, queued ? "waiting for" : "running for"));
      tick();
      const options = (job.data && job.data.options) || {};
      const deepRun = Boolean((job.data && job.data.depth) || options.depth);
      const stepMode = deepRun ? (mode === "verify" ? "deep-verify" : "deep") : mode;
      clear(steps).append(stepper(progressSteps(stepMode, job.data ? job.data.stages || [] : [], { state: job.state, current: run.current }), { live: true }));
      note.textContent = queued ? "Withdraws it from the queue. It has not started, so nothing has been searched or saved." : runningNote;
      const read = run.sources.filter((s) => ACCESS_LEVEL[s.access] === "full").length;
      const partly = run.sources.filter((s) => ACCESS_LEVEL[s.access] === "partial").length;
      const rows = budgetRows(run.budget);
      append(clear(soFar),
        deepRun ? el("p", { class: "cancel-note" }, `${DEPTH_LABELS[options.depth || run.depth] || "Deep"} research`
          + (options.providers && options.providers.length ? ` · searching ${options.providers.join(", ")}` : "")) : null,
        rows.length ? el("div", { class: "stack-sm" }, el("div", { class: "card-title" }, "Budget used"),
          el("dl", { class: "kv budget" }, rows.map((r) => [el("dt", {}, r.label), el("dd", {}, `${r.used} of ${r.limit}`)]))) : null,
        run.sources.length ? el("div", { class: "stack-sm" }, el("div", { class: "card-title" }, "Recorded so far"),
          kv([["Sources found", String(run.sources.length)], ["Read in full", String(read)], ["Abstract or snippet only", String(partly)],
            ["Passages retrieved", run.pool.length ? String(run.pool.length) : "none yet"]]),
          el("p", { class: "cancel-note" }, "A source counts as read only when its text was obtained, not when it was found.")) : null);
    },
    setCancelling(on) {
      button.disabled = on;
      buttonLabel.textContent = on ? "Cancelling…" : `Cancel ${noun}`;
    },
    destroy() { clearInterval(timer); },
  };
  return view;
}

function cancelledView(ctx, mode, outcome, run, jobId) {
  const copy = COPY[mode];
  const input = run && run.input;
  const again = el("button", { class: "btn btn-primary", type: "button" }, icon("refresh"), "Run it again");
  again.addEventListener("click", async () => {
    again.disabled = true;
    try { const { id } = await api.submit(input, mode); ctx.navigate(`#/${mode}/job/${id}`); }
    catch (error) { ctx.toast(error.message); again.disabled = false; }
  });
  const follow = el("button", { class: "btn", type: "button", on: { click: () => ctx.reload() } }, icon("refresh"), "Follow it again");
  const texts = {
    cancelled: ["Cancelled before it started", "The run was withdrawn from the queue. Nothing was searched, read or saved.", [input ? again : null]],
    running: ["Stopped following this run", "It had already started, and the pipeline cannot be interrupted safely, so it is still running on the server. "
      + "It will finish and be saved to your Library. The stages below are what had finished when you cancelled; they are not a result.", [follow]],
    gone: ["This run is no longer held by the server", "Live runs are kept in memory only while the server runs.", []],
    unreachable: ["Stopped following this run", `The server could not be reached to cancel it (${outcome.error || "no response"}), so it may still be running.`, [follow]],
  };
  const [title, body, actions] = texts[outcome.state] || texts.gone;
  return el("section", { class: "sheet sheet-pad stack", "aria-labelledby": "cancel-h" },
    el("p", { class: "eyebrow" }, copy.eyebrow),
    el("h1", { class: "run-question", id: "cancel-h", tabindex: "-1" }, input || copy.title),
    el("div", { class: "row" }, outcome.state === "cancelled" ? statusBadge("CANCELLED") : badge("Not followed", "neutral"),
      outcome.state === "running" ? badge("Still running on the server", "info") : null),
    callout("note", title, el("p", {}, body)),
    outcome.state === "running" && run && run.stages.length ? el("div", { class: "stack-sm" }, el("div", { class: "card-title" }, "Stages finished so far"), stepper(progressSteps(mode, run.stages))) : null,
    el("div", { class: "row" }, ...actions, el("a", { class: "btn", href: `#/${mode}` }, mode === "verify" ? "New claim" : "New question"),
      el("a", { class: "btn btn-ghost", href: "#/library" }, "Library"),
      jobId && outcome.state === "running" ? el("span", { class: "small muted mono" }, `job ${jobId}`) : null));
}

// ------------------------------------------------------------------------------------------------ shared report pieces
function renderResult(container, run, helpers, meta) {
  if (run.mode === "verify" && run.claims.length > 1) renderClaims(container, run, helpers, meta);
  else if (run.mode === "verify") renderVerification(container, run, helpers, meta);
  else renderResearch(container, run, helpers, meta);
}

function metaLine(run, meta) {
  const read = run.sources.filter((s) => ACCESS_LEVEL[s.access] === "full").length;
  const parts = [
    statusBadge(run.status), run.outcome ? outcomeBadge(run.outcome) : null,
    run.depth ? badge(`${DEPTH_LABELS[run.depth] || run.depth} research`, "neutral", { title: describeBudget(run.budget && run.budget.limits) }) : null,
    run.stopping ? completenessBadge(run.stopping.completeness) : null,
    el("span", {}, el("strong", {}, plural(run.sources.length, "source")), ` · ${read} read in full`),
    run.model ? el("span", { class: "adv-only" }, el("strong", {}, run.model), run.modelService ? ` on ${run.modelService}` : run.ollama ? ` on Ollama ${run.ollama}` : "") : null,
    run.verifier ? el("span", { class: "adv-only" }, "verifier ", el("strong", {}, run.verifier)) : null,
    run.finished ? el("span", {}, `finished ${formatDate(run.finished)}`) : null,
    meta.savedName ? el("a", { href: `#/${run.mode}/run/${encodeURIComponent(meta.savedName)}` }, "permanent link")
      : meta.live ? el("span", {}, "not saved to disk") : null,
  ];
  return el("div", { class: "meta-line" }, parts.filter(Boolean));
}

function tabbed(container, run, tabs, defaultTab, draw) {
  const params = new URLSearchParams((location.hash.split("?")[1]) || "");
  const wanted = params.get("view");
  const initial = tabs.some((t) => t.id === wanted) ? wanted : defaultTab;
  const panel = el("div", { id: "run-panel", role: "tabpanel", tabindex: "-1" });
  const bar = tabbar(tabs, initial, (id) => {
    const path = location.hash.split("?")[0];
    history.replaceState(null, "", id === defaultTab ? path : `${path}?view=${id}`);
    panel.setAttribute("aria-labelledby", `tab-${id}`);
    clear(panel);
    draw(id, panel);
  }, { label: "Run views", panelId: "run-panel" });
  container.append(bar.node, panel);
  bar.select(initial);
  return bar;
}

function sourcesView(run, usage, { onPassages }) {
  let filter = "all";
  let query = "";
  const counts = Object.fromEntries(Object.keys(SOURCE_FILTERS).map((key) => [key, filterSources(run.sources, usage, { filter: key }).length]));
  const list = el("div", {});
  const draw = () => {
    const shown = filterSources(run.sources, usage, { filter, query });
    append(clear(list), shown.length ? el("ul", { class: "ledger" }, shown.map((s) => el("li", {}, sourceEntry(s, usage.get(s.id), { onPassages }))))
      : emptyState({ iconName: "book", title: run.sources.length ? "No source matches" : "No sources were selected",
        body: run.sources.length ? "Change the filter or the search text." : "The searches returned nothing that could be selected." }));
  };
  draw();
  return el("div", { class: "stack" },
    el("p", { class: "small muted" }, "Every source the web researcher selected from search results, with how much of it could be read and how it was used. "
      + "Search results that were not selected are not recorded, and no source is shown as read unless its text was actually obtained."),
    el("div", { class: "toolbar" },
      chips(Object.entries(SOURCE_FILTERS).filter(([key]) => key === "all" || counts[key]).map(([value, f]) => ({ value, label: f.label, count: counts[value] })),
        filter, (v) => { filter = v; draw(); }, "Filter sources"),
      searchField({ label: "Search sources", placeholder: "Search titles, publishers, IDs", onInput: (v) => { query = v; draw(); } })),
    list);
}

function evidenceView(run, helpers, initialSource = "") {
  let source = initialSource;
  const ids = [...new Set(run.pool.map((p) => p.sourceId).filter(Boolean))];
  const list = el("div", { class: "stack" });
  const draw = () => {
    const shown = run.pool.filter((p) => !source || p.sourceId === source);
    append(clear(list), shown.length ? shown.map((p) => passageCard(p, { source: run.sourceIndex.get(p.sourceId), showStance: false,
      onContext: (x) => helpers.openContext(x, run.sourceIndex.get(x.sourceId)) }))
      : emptyState({ iconName: "layers", title: source ? `No retrieved passage from ${source}` : "No passages were retrieved",
        body: source ? "This source was selected but none of its passages ranked among those retrieved for the question." : "Nothing was retrieved for this run." }));
  };
  draw();
  return el("div", { class: "stack" },
    el("p", { class: "small muted" }, run.mode === "verify"
      ? "Passages retrieved for the claim, in ranking order, before they were judged. Their stances are shown on the Evidence tab."
      : "Passages retrieved for the question and its sub-questions and given to the synthesizer, before any verification."),
    ids.length > 1 ? el("div", { class: "toolbar" }, chips([{ value: "", label: "All sources", count: run.pool.length },
      ...ids.map((id) => ({ value: id, label: id, count: run.pool.filter((p) => p.sourceId === id).length }))], source, (v) => { source = v; draw(); }, "Filter passages by source")) : null,
    list);
}

// How the investigation was carried out, in plain terms. Timings and internals are on the Diagnostics tab.
function methodView(run) {
  if (run.depth) return deepMethodView(run);
  const scope = [
    [run.mode === "verify" ? "Claim" : "Question", run.input],
    ["Sub-questions", run.plan && run.plan.subquestions.length ? el("ul", { class: "prose-list" }, run.plan.subquestions.map((q) => el("li", {}, q))) : null],
    ["Search queries", run.plan && run.plan.queries.length ? el("ul", { class: "prose-list" }, run.plan.queries.map((q) => el("li", {}, q))) : null],
    ["Searched", run.providers.length ? run.providers.join(", ") : null],
    ["Started", run.started ? formatDate(run.started) : null], ["Finished", run.finished ? formatDate(run.finished) : null],
  ];
  const failedSearches = run.providerStatus.reduce((n, p) => n + p.failed, 0) || run.searchErrors.length;
  const parts = [
    el("section", { "aria-labelledby": "scope-h" }, sectionHead("01", "Question and scope", { id: "scope-h", aside: provenance("model") }),
      run.plan && run.plan.fallback ? callout("warn", "Fallback plan", el("p", {}, "The planner's output could not be used, so a fallback plan was applied.")) : null,
      kv(scope),
      failedSearches ? el("p", { class: "small muted" }, `${plural(failedSearches, "search", "searches")} failed; the limitations in the report name the providers concerned.`) : null),
    el("section", { "aria-labelledby": "stages-h" }, sectionHead("02", "Steps", { id: "stages-h", aside: plural(run.stages.length, "step") }), timeline(run.stages, run.mode)),
  ];
  if (run.diversity) {
    const d = run.diversity;
    parts.push(el("section", { "aria-labelledby": "diversity-h" }, sectionHead("03", "Source diversity", { id: "diversity-h", aside: "readable sources only" }),
      kv([["Sources", d.sources], ["Domains", d.domains], ["Publishers", d.publishers], ["Independent origins", d.independent_origins]]
        .filter(([, v]) => Number.isFinite(v)).map(([k, v]) => [k, String(v)]))));
  }
  if (run.followup) {
    const f = run.followup;
    parts.push(el("section", {}, sectionHead("04", "Follow-up search"), f.skipped ? el("p", { class: "small" }, `Skipped: ${f.skipped}`)
      : kv([["Queries", String(f.queries.length)], ["New sources", String(f.newSources)], ["Verdicts changed", `${f.changed} of ${f.changes}`]])));
  }
  return el("div", { class: "stack-lg" }, parts);
}

const dataTable = (head, rows, caption) => el("div", { class: "table-wrap" }, el("table", {},
  el("thead", {}, el("tr", {}, head.map(([h, num]) => el("th", { scope: "col", class: num ? "num" : null }, h)))),
  el("tbody", {}, rows), caption ? el("caption", {}, caption) : null));
const numCell = (value) => el("td", { class: "num" }, value === null || value === undefined ? "—" : String(value));

const PURPOSES = { answer: "answer", "follow-up": "follow-up", counter: "counter-evidence" };
const STOPPING_RULE = "Searching stops as soon as every question has passages from two or more independent origins, including full text; "
  + "otherwise when the depth's rounds, searches, sources or time are used up, when every search in a round fails, when a follow-up "
  + "round finds no new relevant passage, or when no new query can be formed.";

// Deep research: the plan, each search round, the budget, the stopping rule and the evidence policy, as recorded.
function deepMethodView(run) {
  const plan = run.researchPlan || {};
  const options = run.options || {};
  const subquestions = Array.isArray(plan.subquestions) ? plan.subquestions.filter((s) => s && typeof s.text === "string") : [];
  const context = plan.context && typeof plan.context === "object" ? Object.entries(plan.context)
    .map(([k, v]) => [k.replace(/_/g, " "), Array.isArray(v) ? v.join(", ") : typeof v === "string" ? v : ""]).filter(([, v]) => v) : [];
  const list = (items) => (items && items.length ? items : null);
  const scope = [
    [run.mode === "verify" ? "Claim" : "Question", run.input],
    ["Depth", `${DEPTH_LABELS[run.depth] || run.depth}${run.budget ? ` — ${describeBudget(run.budget.limits)}` : ""}`],
    ["Deliverable", typeof plan.deliverable === "string" && plan.deliverable ? plan.deliverable : null],
    ["Sub-questions", subquestions.length ? el("ol", { class: "prose-list" }, subquestions.map((s) => el("li", {}, s.text,
      Array.isArray(s.categories) && s.categories.length ? el("span", { class: "muted" }, ` (${categoryNames(s.categories)})`) : null))) : null],
    ["Source types searched", list(plan.searched_categories) ? categoryNames(plan.searched_categories) : null],
    ["Preferred source types", list(options.categories) ? categoryNames(options.categories) : "none (all active types)"],
    ["Time range", RECENCY_LABELS[options.recency] || "Any time"],
    ["Providers", list(options.providers) ? options.providers.join(", ") : null],
    ["Started", run.started ? formatDate(run.started) : null], ["Finished", run.finished ? formatDate(run.finished) : null],
  ];
  const rows = budgetRows(run.budget);
  const policy = run.evidencePolicy || {};
  const labels = policy.labels && typeof policy.labels === "object" ? policy.labels : {};
  const parts = [
    el("section", { "aria-labelledby": "scope-h" }, sectionHead("01", "Question and research plan", { id: "scope-h", aside: provenance("model") }),
      run.plan && run.plan.fallback ? callout("warn", "Fallback plan", el("p", {}, "The planner's output could not be used, so the question was searched by its keywords instead.")) : null,
      kv(scope),
      context.length ? disclosure("Context the planner identified", kv(context)) : null),
    el("section", { "aria-labelledby": "rounds-h" }, sectionHead("02", "Search rounds", { id: "rounds-h", aside: plural(run.rounds.length, "round") }),
      run.rounds.length ? el("div", { class: "stack" }, run.rounds.map((r) => el("div", { class: "round-block" },
        el("div", { class: "row-between" }, el("h3", { class: "card-title" }, `Round ${r.number}${r.purpose ? ` — ${r.purpose}` : ""}`),
          el("span", { class: "xsmall muted" }, r.seconds === null ? "" : formatSeconds(r.seconds))),
        el("p", { class: "small" }, `${plural(r.searches, "search", "searches")}, ${plural(r.hits, "result")}, ${plural(r.newSources, "new source")}, `
          + `${r.indexed} with readable text, ${plural(r.newPassages, "new relevant passage")}`
          + (r.filtered ? `, ${r.filtered} outside the time range` : "") + (r.errors ? `, ${plural(r.errors, "failed search", "failed searches")}` : "") + "."),
        r.queries.length ? dataTable([["Query"], ["Source type"], ["Purpose"], ["For"], ["Providers"]], r.queries.map((q) => el("tr", {},
          el("td", {}, q.query), el("td", {}, CATEGORY_LABELS[q.category] || q.category), el("td", {}, PURPOSES[q.purpose] || q.purpose),
          el("td", {}, q.subquestion ? `question ${q.subquestion}` : "main question"), el("td", {}, q.providers.join(", ") || "none active")))) : null)))
        : el("p", { class: "small muted" }, "No search round was recorded.")),
    el("section", { "aria-labelledby": "budget-h" }, sectionHead("03", "Budget and stopping rule", { id: "budget-h" }),
      rows.length ? dataTable([["Budget"], ["Used", true], ["Limit", true]], rows.map((b) => el("tr", {}, el("td", {}, b.label), numCell(b.used), numCell(b.limit)))) : null,
      el("p", { class: "small" }, STOPPING_RULE), stoppingCallout(run)),
    el("section", { "aria-labelledby": "stages-h" }, sectionHead("04", "Steps", { id: "stages-h", aside: plural(run.stages.length, "step") }), timeline(run.stages, run.mode)),
  ];
  if (Object.keys(policy).length) {
    parts.push(el("section", { "aria-labelledby": "policy-h" }, sectionHead("05", "Evidence policy", { id: "policy-h" }),
      el("p", { class: "small muted" }, "Fixed rules for what counts as evidence. They decide what is indexed and how support is labelled; verdicts come from the verifier alone."),
      kv(Object.keys(ACCESS).filter((k) => typeof policy[k] === "string").map((k) => [ACCESS[k], policy[k]])),
      Object.keys(labels).length ? disclosure("Evidence labels on findings", kv(Object.entries(labels).map(([k, v]) => [strengthMeta(k).label || k, String(v)]))) : null));
  }
  return el("div", { class: "stack-lg" }, parts);
}

// Technical details for advanced users: only what the run record contains.
function diagnosticsView(run) {
  const table = dataTable;
  const parts = [
    el("section", { "aria-labelledby": "runtime-h" }, sectionHead("01", "Model and verifier", { id: "runtime-h" }),
      kv([["Model", run.model || "not recorded"], ...(run.modelService ? [["Model service", run.modelService], ["Verifier model", run.verifierModel || run.model]] : [["Ollama version", run.ollama || "not recorded"]]),
        ["Model digest", run.modelDigest ? el("code", {}, run.modelDigest.slice(0, 19)) : "not recorded"], ["Verifier", run.verifier || "not recorded"],
        ["Prompt version", run.settings.prompt_version || "not recorded"], ["HTTP requests", run.requests !== null ? String(run.requests) : "not recorded"],
        ["Token usage", "not recorded in run records"]])),
    el("section", { "aria-labelledby": "timing-h" }, sectionHead("02", "Stage timings", { id: "timing-h", aside: "seconds, as recorded" }),
      run.stages.length ? table([["Stage"], ["Status"], ["Seconds", true], ["Detail"]], run.stages.map((s) => el("tr", {}, el("td", {}, s.name), el("td", {}, s.status),
        el("td", { class: "num" }, s.seconds === null ? "—" : String(s.seconds)), el("td", {}, s.detail)))) : el("p", { class: "small muted" }, "No stages were recorded.")),
  ];
  if (run.claims.length) {
    parts.push(el("section", { "aria-labelledby": "verify-time-h" }, sectionHead("03", "Verification per statement", { id: "verify-time-h" }),
      table([["#"], ["Verdict"], ["Sub-state"], ["Passages", true], ["Seconds", true], ["Output complete"]], run.claims.map((c, i) => el("tr", {},
        el("td", {}, String(i + 1)), el("td", {}, c.label), el("td", {}, c.subState || "—"), el("td", { class: "num" }, String(c.passages.length)),
        el("td", { class: "num" }, c.seconds === null ? "—" : String(c.seconds)), el("td", {}, c.parseOk ? "yes" : `no${c.error ? ` (${truncate(c.error, 80)})` : ""}`))))));
  }
  if (run.providerStatus.length) {
    parts.push(el("section", { "aria-labelledby": "providers-h" }, sectionHead("04", "Search providers", { id: "providers-h" }),
      table([["Provider"], ["Queries", true], ["Succeeded", true], ["Failed", true], ["Hits", true]], run.providerStatus.map((p) => el("tr", {}, el("td", {}, p.name),
        el("td", { class: "num" }, String(p.queries)), el("td", { class: "num" }, String(p.succeeded)), el("td", { class: "num" }, String(p.failed)), el("td", { class: "num" }, String(p.hits)))),
      "Hits are search results; only some of them were selected as sources.")));
  }
  if (run.depth && run.rounds.length) {
    const status = run.rounds.flatMap((r) => r.providerStatus.map((p) => el("tr", {}, numCell(r.number), el("td", {}, p.name),
      numCell(p.queries), numCell(p.succeeded), numCell(p.failed), numCell(p.hits))));
    const access = (r, key) => numCell(Number.isFinite(r.access[key]) ? r.access[key] : null);
    parts.push(el("section", { "aria-labelledby": "rounds-diag-h" }, sectionHead("05", "Search rounds", { id: "rounds-diag-h", aside: "as recorded per round" }),
      status.length ? table([["Round", true], ["Provider"], ["Queries", true], ["Succeeded", true], ["Failed", true], ["Hits", true]], status,
        "Hits are search results; only some of them were selected as sources.") : null,
      table([["Round", true], ["Full text", true], ["Abstract", true], ["Snippet", true], ["Metadata", true], ["Failed", true], ["HTTP requests", true]],
        run.rounds.map((r) => el("tr", {}, numCell(r.number), access(r, "FULL_TEXT"), access(r, "ABSTRACT_ONLY"), access(r, "SNIPPET_ONLY"),
          access(r, "METADATA_ONLY"), access(r, "FAILED"), numCell(r.requests))), "New sources per round by how much of each could be read.")));
  }
  if (run.depth && run.claims.length) {
    parts.push(el("section", { "aria-labelledby": "cite-diag-h" }, sectionHead("06", "Citation checks", { id: "cite-diag-h" }),
      table([["#"], ["Evidence label"], ["Passages", true], ["Problems"], ["Not matched verbatim", true]], run.claims.map((c, i) => el("tr", {},
        el("td", {}, String(i + 1)), el("td", {}, strengthMeta(c.strength).label || "—"), numCell(c.passages.length),
        el("td", {}, c.citationProblems.length ? c.citationProblems.join("; ") : "none"), numCell(c.unlocated.length)))),
      el("p", { class: "xsmall muted" }, "Each passage behind a verdict must come from a recorded, readable source and sit on a page or time window that exists. "
        + "Passages not matched verbatim usually reflect the indexer's whitespace normalisation and are reported, not treated as failures.")));
  }
  if (run.depth && run.budget && run.budget.used && run.budget.limits) {
    const { used, limits } = run.budget;
    parts.push(el("section", { "aria-labelledby": "budget-diag-h" }, sectionHead("07", "Budget counters", { id: "budget-diag-h" }),
      kv([["Characters read", Number.isFinite(used.chars) ? `${used.chars.toLocaleString("en")} of ${Number(limits.chars).toLocaleString("en")}` : null],
        ["Queries sent", Number.isFinite(used.queries) ? String(used.queries) : null],
        ["Searches", Number.isFinite(used.searches) ? `${used.searches} of ${limits.searches} (query × provider)` : null],
        ["Elapsed", Number.isFinite(run.budget.elapsed) ? `${formatSeconds(run.budget.elapsed)} (searching stops after ${formatSeconds(limits.seconds)}; drafting and verification still finish)` : null]])));
  }
  if (run.searchErrors.length) {
    parts.push(disclosure(`Search errors (${run.searchErrors.length})`, el("ul", { class: "prose-list" },
      run.searchErrors.map((e) => el("li", {}, el("b", {}, e.provider), e.query ? ` for "${truncate(e.query, 80)}"` : "", ` — ${e.error}`))), true));
  }
  const settings = Object.entries(run.settings);
  if (settings.length) parts.push(disclosure("All recorded settings", kv(settings.map(([k, v]) => [k.replace(/_/g, " "), String(v)]))));
  return el("div", { class: "stack-lg" }, parts);
}

// Export of a saved run: Markdown and JSON downloads, and a new note in the Obsidian vault when one is set up.
function exportMenu(run, meta, ctx) {
  const name = meta.saved || meta.savedName;
  const slot = el("div", { class: "export-slot" });
  if (!name) return append(slot, el("span", { class: "xsmall muted" }, "Export becomes available once the run is saved."));
  const build = (obsidian) => {
    const ready = obsidian && obsidian.status.state === "ready";
    const items = [
      { label: "Markdown", note: "Findings, limitations and sources", href: api.exportUrl(name, "md"), download: `${name}.md`, iconName: "doc" },
      { label: "Run record (JSON)", note: "Everything the pipeline recorded", href: api.exportUrl(name, "json"), download: `${name}.json`, iconName: "database" },
      { separator: true },
      ready ? { label: "Save to Obsidian", note: `New note in ${obsidian.facts.Vault ? obsidian.facts.Vault.split(/[\\/]/).pop() : "your vault"}`, iconName: "file",
        onSelect: () => api.exportToObsidian(name).then((r) => ctx.toast(`Saved in ${r.vault}: ${r.saved}`)).catch((error) => ctx.toast(error.message)) }
        : { label: "Set up Obsidian export", note: obsidian && obsidian.status.state === "error" ? obsidian.status.detail : "Not set up for this server", href: "#/integrations", iconName: "plug" },
    ];
    clear(slot).append(menuButton({ content: [icon("download"), "Export", icon("chevrondown")], items, label: "Export this report" }));
  };
  api.obsidian().then(build).catch(() => build(null));
  return slot;
}

function runTabs(run, primary) {
  const tabs = [primary,
    ...(run.depth && run.ledger.length ? [{ id: "coverage", label: "Coverage", icon: "compare", count: run.ledger.length }] : []),
    { id: "sources", label: "Sources", icon: "book", count: run.sources.length },
    { id: "passages", label: "Passages", icon: "layers", count: run.pool.length },
    { id: "method", label: "How it was researched", icon: "list" }];
  if (preferences.get().advanced) tabs.push({ id: "diagnostics", label: "Diagnostics", icon: "settings" });
  return tabs;
}

function bibliographyView(run, usage) {
  const entries = bibliography(run, usage);
  if (!entries.length) return el("p", { class: "small muted" }, "No sources were selected, so there is nothing to cite.");
  const groups = [["cited", "Used as evidence"], ["consulted", "Retrieved but not used"], ["listed", "Selected but no passage retrieved"]];
  return el("div", {},
    groups.map(([key, title]) => {
      const items = entries.filter((e) => e.group === key);
      if (!items.length) return null;
      return [el("p", { class: "biblio-group" }, `${title} (${items.length})`),
        el("ol", { class: "biblio" }, items.map((e) => el("li", {},
          el("span", { class: "bib-id" }, e.id),
          el("span", {}, `${e.title.replace(/\.+$/, "")}. `, [e.publisher, e.date].filter(Boolean).join(", "), e.publisher || e.date ? ". " : "",
            e.url ? externalLink(e.url, e.url) : el("span", { class: "muted" }, "no URL recorded"),
            e.accessed ? el("span", { class: "muted" }, ` (accessed ${e.accessed})`) : null),
          copyButton(() => e.citation, "Copy"))))];
    }),
    el("p", { class: "xsmall muted" }, "Built from the metadata the web researcher recorded; missing fields are left out rather than guessed."));
}

// ------------------------------------------------------------------------------------------------ deep research
const EVIDENCE_STATE = {
  strong: "Strong: passages from two or more publishers established as independent, including full text",
  weak: "Weak: some relevant passages, not yet independently corroborated in full text",
  none: "None: no relevant readable passage was found",
};
const OPEN_STATUSES = new Set(["partially answered", "contested", "unanswered", "not reached"]);
const categoryNames = (list) => list.map((c) => CATEGORY_LABELS[c] || c).join(", ");

function stoppingCallout(run) {
  if (!run.stopping) return null;
  const meta = completenessMeta(run.stopping.completeness);
  return callout(run.stopping.completeness === "complete" ? "success" : "warn", "Why searching stopped",
    el("p", {}, meta.label ? el("strong", {}, `${meta.label}. `) : null, run.stopping.detail, meta.description ? ` ${meta.description}` : ""));
}

// The coverage ledger: for each question, what was searched, found, read and verified when research stopped. Its
// status follows fixed rules from the verified statements; it guides searching and reporting, never a verdict.
function coverageView(run) {
  const cards = run.ledger.map((e) => el("article", { class: "ledger-card", "aria-labelledby": `ledger-${e.index}` },
    el("div", { class: "row-between" }, el("span", { class: "eyebrow", id: `ledger-${e.index}` }, e.index === 0 ? "Main question" : `Question ${e.index}`),
      coverageBadge(e.status)),
    el("p", { class: "ledger-question" }, e.text),
    e.reason ? el("p", { class: "small" }, e.reason) : null,
    kv([
      ["Evidence found while searching", EVIDENCE_STATE[e.evidenceState] || e.evidenceState || null],
      e.distinctTexts !== null ? ["Distinct texts (older record)", String(e.distinctTexts)]
        : ["Independent publishers", e.independence === "established" ? String(e.publishers)
          : e.independence === "unknown" ? "not established from the metadata" : e.independence === "single" ? "one publisher" : "none"],
      ["Source types planned", e.categories.length ? categoryNames(e.categories) : null],
      ["Sources found / read", `${e.discovered.length} / ${e.retrieved.length}`],
      ["Supported statements", String(e.supporting.length)],
      ["Contested or contradicted", e.contradicting.length ? String(e.contradicting.length) : null],
      ["Not established", e.uncertain.length ? String(e.uncertain.length) : null],
      ["Suggested next step", e.nextAction || null],
    ]),
    e.queries.length ? disclosure(`Queries for this question (${e.queries.length})`, el("ul", { class: "prose-list" }, e.queries.map((q) => el("li", {}, q)))) : null));
  return el("div", { class: "stack-lg" },
    el("section", { "aria-labelledby": "coverage-h" },
      sectionHead("01", "Coverage by question", { id: "coverage-h", aside: provenance("recorded") }),
      el("p", { class: "small muted" }, "What was searched, found, read and verified for each question when research stopped. The status follows fixed rules "
        + "from the verified statements; it describes coverage and never changes a verdict."),
      stoppingCallout(run),
      el("div", { class: "coverage-ledger" }, cards)));
}

// Evidence quality and source diversity: counts the backend recorded; nothing is scored or weighted.
function qualitySection(run, number) {
  const q = run.quality;
  const obj = (v) => (v && typeof v === "object" && !Array.isArray(v) ? v : {});
  const num = (v) => (Number.isFinite(v) ? String(v) : null);
  const access = obj(q.by_access);
  const support = obj(q.supporting_passages_by_access);
  const types = Object.entries(obj(q.by_category)).filter(([, n]) => n > 0).map(([c, n]) => `${CATEGORY_LABELS[c] || c} ${n}`).join(", ");
  return el("section", { class: "report-section", "aria-labelledby": "quality-h" },
    sectionHead(number, "Evidence quality and source diversity", { id: "quality-h", aside: provenance("recorded") }),
    el("div", { class: "quality-grid" },
      kv([["Sources found", num(q.discovered)], ["Usable as evidence", Number.isFinite(q.read) ? `${q.read} (full text or abstract)` : null],
        ["Read in full", num(access.FULL_TEXT)], ["Abstract only", num(access.ABSTRACT_ONLY)],
        ["Search snippet only", Number.isFinite(access.SNIPPET_ONLY) ? `${access.SNIPPET_ONLY} (used to find sources, not as evidence)` : null],
        ["Metadata only", num(access.METADATA_ONLY)], ["Could not be retrieved", num(access.FAILED)]]),
      kv([["Distinct texts (near-copies merged)", num(q.distinct_texts ?? q.independent_origins)],
        ["Publisher groups", Number.isFinite(q.publisher_groups) ? `${q.publisher_groups} (independence is judged per finding)` : null],
        ["Domains", num(q.domains)], ["Publishers", num(q.publishers)],
        ["Source types read", types || null],
        ["Supporting passages", Number.isFinite(support.FULL_TEXT) ? `${support.FULL_TEXT} from full text, ${support.ABSTRACT_ONLY || 0} from abstracts` : null],
        ["Providers that returned sources", Array.isArray(q.providers_used) && q.providers_used.length ? q.providers_used.join(", ") : null]])),
    el("p", { class: "xsmall muted" }, "Counts from the run record. No overall confidence score is computed: the evidence label on each finding "
      + "(corroborated, single source, abstract only, contested …) states what its support rests on."));
}

// ------------------------------------------------------------------------------------------------ research report
// deep-research-v3: what was searched, as recorded. Queries that did not run are listed with the reason, so a planned
// counter-evidence search is never mistaken for one that was carried out.
function executionSection(run, number) {
  const counts = new Map();
  for (const q of run.queries) counts.set(q.status, (counts.get(q.status) || 0) + 1);
  const counter = run.queries.filter((q) => q.purpose === "counter");
  const notRun = run.queries.filter((q) => q.status !== "executed");
  const nothingRelevant = run.queries.filter((q) => q.status === "executed" && q.relevantCandidate === false);
  const searched = run.providerCoverage.filter((p) => p.state === "searched" || p.state === "failed");
  const others = run.providerCoverage.filter((p) => p.state !== "searched" && p.state !== "failed");
  return el("section", { class: "report-section", "aria-labelledby": "execution-h" },
    sectionHead(number, "Search execution", { id: "execution-h", aside: plural(run.queries.length, "planned query", "planned queries") }),
    el("div", { class: "stack" },
      run.queries.length ? el("div", { class: "row-tight", "aria-label": "Queries by status" },
        [...counts].map(([status, n]) => { const m = queryStatusMeta(status); return badge(`${m.label}: ${n}`, m.tone); }),
        counter.length ? el("span", { class: "small" }, `Counter-evidence: ${counter.filter((q) => q.status === "executed").length} of ${counter.length} run`) : null) : null,
      notRun.length ? el("ul", { class: "note-list tone-caution", "aria-label": "Queries that did not run" }, notRun.map((q) => el("li", {},
        el("div", { class: "row-tight" }, badge(queryStatusMeta(q.status).label, queryStatusMeta(q.status).tone), el("span", { class: "claim-id" }, q.id),
          el("span", {}, `“${q.query}”`), q.purpose === "counter" ? badge("counter-evidence", "neutral") : null),
        el("div", { class: "xsmall muted" }, q.reason || "No reason was recorded.")))) : null,
      nothingRelevant.length ? el("p", { class: "small" }, el("b", {}, "Ran, but no result was relevant enough to read: "),
        nothingRelevant.map((q) => `${q.id} “${q.query}”`).join("; ")) : null,
      searched.length ? el("p", { class: "small" }, el("b", {}, "Searched: "), searched.map((p) =>
        `${p.name} (${plural(p.queries, "search", "searches")}, ${plural(p.hits, "result")}${p.failed ? `, ${p.failed} failed` : ""})`).join("; ")) : null,
      others.length ? el("ul", { class: "prose-list", "aria-label": "Providers not searched" }, others.map((p) => el("li", {},
        el("b", {}, p.name), `: ${providerStateMeta(p.state).label.toLowerCase()}`, p.detail ? ` — ${p.detail}` : "",
        p.missing.length ? ` (needs ${p.missing.join(", ")})` : ""))) : null,
      el("p", { class: "xsmall muted" }, "Only providers marked Searched were queried in this run; the others were not asked anything.")));
}

// deep-research-v3: the evidence review restates the recorded checks per statement; it is not an independent check.
function reviewSection(run, number, onClaim) {
  const ACTIONS = { excluded: "kept out of the answer", needs_evidence: "needs further evidence", unresolved: "unresolved",
    not_reported: "not reported as a finding", note: "noted" };
  return el("section", { class: "report-section", "aria-labelledby": "review-h" },
    sectionHead(number, "Evidence review", { id: "review-h", aside: plural(run.review.length, "item") }),
    el("p", { class: "xsmall muted" }, "Fixed checks of the recorded evidence: what limits each statement's support, and planned work that was not done. They restate the run's own checks; no model was asked."),
    el("ul", { class: "note-list tone-caution" }, run.review.map((item) => el("li", {},
      el("div", { class: "row-tight" },
        item.claim ? el("button", { class: "chip", type: "button", title: "Show this statement's evidence", on: { click: () => onClaim(item.claim) } }, item.claim) : null,
        badge(ACTIONS[item.action] || item.action || "noted", item.action === "excluded" || item.action === "unresolved" ? "uncertain" : "neutral")),
      el("div", {}, item.detail),
      item.refs.length ? el("div", { class: "xsmall muted" }, `Refers to ${item.refs.slice(0, 4).join(", ")}${item.refs.length > 4 ? " …" : ""}`) : null))));
}

// A table of contents for a long report: buttons that scroll to each section (the URL hash is the app's route).
function reportNav(report) {
  const sections = [...report.querySelectorAll("section.report-section[aria-labelledby]")];
  if (sections.length < 3) return null;
  return el("nav", { class: "report-nav", "aria-label": "Report sections" }, el("ol", {}, sections.map((section) => {
    const head = report.querySelector(`[id="${section.getAttribute("aria-labelledby")}"]`);
    return el("li", {}, el("button", { class: "report-nav-link", type: "button",
      on: { click: () => { section.scrollIntoView({ block: "start" }); if (head) { head.setAttribute("tabindex", "-1"); head.focus({ preventScroll: true }); } } } },
    head ? head.textContent : "Section"));
  })));
}

// Claims workspace with several claims: a table of claims and their verdicts, and the evidence of the selected one.
function renderClaims(container, run, helpers, meta) {
  const usage = sourceUsage(run);
  container.append(el("div", { class: "row-between run-actions" }, el("span", {}), exportMenu(run, meta, helpers.ctx)));
  const tally = run.summary && run.summary.verdicts ? Object.entries(run.summary.verdicts) : [];
  container.append(el("section", { class: "sheet verdict-hero", "aria-label": "Claims" },
    el("div", { class: "stack" },
      el("p", { class: "eyebrow" }, "Claims under examination"),
      el("h1", { class: "verdict-claim" }, `${plural(run.claims.length, "claim")} checked`),
      el("div", { class: "row-tight", "aria-label": "Verdicts" }, tally.map(([label, n]) => el("span", { class: "row-tight" }, verdictBadge(label), el("span", { class: "small" }, `× ${n}`)))),
      metaLine(run, meta)),
    el("div", { class: "verdict-side" }, el("span", { class: "eyebrow" }, "Verdicts"), provenance("verifier"),
      el("p", { class: "small" }, "Each claim has its own verdict; there is no overall verdict."))));
  const notes = el("div", { class: "stack" }, callout("info", "What these verdicts mean",
    el("p", {}, "Each claim is judged on its own, by fixed rules, from per-passage judgements of the passages retrieved for it. A quotation found in a source shows only that the source contains that text; an Insufficient verdict is not evidence against a claim.")));
  if (run.claimsMode && run.claimsMode.dropped) {
    notes.append(callout("warn", "Some lines were not checked", el("p", {}, `A run checks at most ${run.claimsMode.limit} claims; ${plural(run.claimsMode.dropped, "further line")} ${run.claimsMode.dropped === 1 ? "was" : "were"} left out.`)));
  }
  if (run.error) { const state = noAnswerState(run); notes.append(callout(state.tone, state.title, el("p", {}, state.reason))); }
  container.append(notes, el("div", { class: "rule" }));

  const detail = el("div", { class: "stack-lg", id: "claim-detail", tabindex: "-1" });
  const rows = new Map();
  const show = (claim, focus = false) => {
    for (const [c, row] of rows) row.classList.toggle("is-selected", c === claim);
    clear(detail).append(
      el("div", { class: "stack-sm" },
        el("p", { class: "eyebrow" }, `Claim ${claim.id || ""}`.trim()),
        el("h2", { class: "claim-detail-title" }, claim.text),
        el("div", { class: "row-tight" }, verdictBadge(claim.label), claim.strength ? strengthBadge(claim.strength) : null,
          claim.independence ? el("span", { class: "small" }, independenceText(claim.independence)) : null),
        claim.subState ? el("p", { class: "small" }, subStateText(claim.subState)) : null,
        cautionList(claim.cautions, run.evidencePolicy),
        claim.citationProblems.length ? callout("error", "Citation check failed", el("ul", { class: "prose-list" }, claim.citationProblems.map((p) => el("li", {}, p)))) : null),
      ...claimEvidenceParts(run, claim, helpers));
    if (focus) detail.focus();
  };
  const body = el("tbody", {}, run.claims.map((claim) => {
    const groups = groupByStance(claim.passages);
    const row = el("tr", {},
      el("td", {}, el("span", { class: "claim-id" }, claim.id || ""), " ", claim.text,
        claim.cautions.length ? el("div", { class: "row-tight" }, claim.cautions.map((c) => cautionBadge(c, run.evidencePolicy))) : null),
      el("td", {}, verdictBadge(claim.label)),
      el("td", {}, claim.strength ? strengthBadge(claim.strength) : "—"),
      el("td", { class: "num" }, `${groups.SUPPORTS.length} / ${groups.CONTRADICTS.length}`),
      el("td", {}, el("button", { class: "btn btn-ghost btn-sm", type: "button", "aria-controls": "claim-detail",
        on: { click: () => show(claim, true) } }, "Show evidence")));
    rows.set(claim, row);
    return row;
  }));
  let bar = null;
  let passagesFor = "";
  bar = tabbed(container, run, runTabs(run, { id: "claims", label: "Claims", icon: "scale", count: run.claims.length }), "claims", (id, panel) => {
    if (id === "claims") {
      panel.append(el("div", { class: "table-wrap" }, el("table", { class: "claims-table" },
        el("caption", { class: "visually-hidden" }, "Claims, their verdicts and the number of passages judged for and against each"),
        el("thead", {}, el("tr", {}, ["Claim", "Verdict", "Evidence label", "Passages for / against", ""].map((h) => el("th", { scope: "col" }, h)))),
        body)), el("section", { class: "sheet claim-detail", "aria-label": "Evidence for the selected claim" }, detail),
      verificationTail(run, null, meta, []));
      if (run.claims.length) show(run.claims[0]);
    } else if (id === "coverage") panel.append(coverageView(run));
    else if (id === "sources") panel.append(sourcesView(run, usage, { onPassages: (s) => { passagesFor = s.id; bar.select("passages"); } }));
    else if (id === "passages") { panel.append(evidenceView(run, helpers, passagesFor)); passagesFor = ""; }
    else if (id === "diagnostics") panel.append(diagnosticsView(run));
    else panel.append(methodView(run));
  });
}

function renderResearch(container, run, helpers, meta) {
  const usage = sourceUsage(run);
  const header = el("header", { class: "run-header" },
    el("div", { class: "row-between" }, el("p", { class: "eyebrow" }, "Research question"), exportMenu(run, meta, helpers.ctx)),
    el("h1", { class: "run-question" }, run.input),
    metaLine(run, meta));
  container.append(header);
  if (run.error) {                                   // a completed run without an answer is not shown as a failure
    const state = noAnswerState(run);
    container.append(callout(state.tone, state.title, el("p", {}, state.reason)));
  }
  if (run.warnings.length) container.append(callout("warn", "Inconsistent record", el("ul", { class: "prose-list" }, run.warnings.map((w) => el("li", {}, w)))));
  if (run.error || run.warnings.length) container.append(el("div", { class: "rule" }));
  let bar = null;
  let evidenceSource = "";
  const showPassages = (source) => { evidenceSource = source.id; bar.select("passages"); };
  bar = tabbed(container, run, runTabs(run, { id: "report", label: "Report", icon: "report" }), "report", (id, panel) => {
    if (id === "report") panel.append(reportView(run, usage, helpers, meta));
    else if (id === "coverage") panel.append(coverageView(run));
    else if (id === "sources") panel.append(sourcesView(run, usage, { onPassages: showPassages }));
    else if (id === "passages") { panel.append(evidenceView(run, helpers, evidenceSource)); evidenceSource = ""; }
    else if (id === "diagnostics") panel.append(diagnosticsView(run));
    else panel.append(methodView(run));
  });
}

function reportView(run, usage, helpers, meta) {
  const state = { claim: run.findings[0] || run.unsettled[0] || null, source: null };
  const panel = el("aside", { class: "sheet evidence-panel", id: "evidence-panel", "aria-label": "Evidence for the selected statement" });
  const workspace = el("div", { class: "workspace" });
  const toggle = el("button", { class: "btn btn-quiet btn-sm", type: "button", "aria-expanded": "true", "aria-controls": "evidence-panel" }, icon("panel"), el("span", {}, "Hide evidence"));
  const setCollapsed = (collapsed) => {
    workspace.classList.toggle("panel-collapsed", collapsed);
    toggle.setAttribute("aria-expanded", String(!collapsed));
    toggle.lastChild.textContent = collapsed ? "Show evidence" : "Hide evidence";
  };
  toggle.addEventListener("click", () => setCollapsed(!workspace.classList.contains("panel-collapsed")));
  if (preferences.get().evidencePanel === "closed") setCollapsed(true);

  const cards = new Map();
  const select = (claim, source = null, focusPanel = false) => {
    state.claim = claim;
    state.source = source;
    for (const [c, card] of cards) card.setAttribute("aria-current", String(c === claim));
    drawPanel();
    if (focusPanel) {                                  // an explicit request for evidence opens the panel
      setCollapsed(false);
      panel.querySelector(".evidence-body").focus();
    }
  };

  const claimCard = (claim, number) => {
    const chipsNodes = citationsFor(claim, run.sourceIndex).map((c) => {
      const chip = el("button", { class: `cite${c.supports ? " is-support" : c.contradicts ? " is-contradict" : ""}`, type: "button",
        title: `${c.title}${c.cited ? " — cited by the model" : ""}${c.supports ? "; has a supporting passage" : ""}${c.contradicts ? "; has a contradicting passage" : ""}`,
        "aria-label": `Source ${c.id}: ${c.title}${c.supports ? ", supporting" : c.contradicts ? ", contradicting" : ""}. Show its passages` },
      c.supports ? icon("tick") : c.contradicts ? icon("x") : null, c.id);
      chip.addEventListener("click", (event) => { event.stopPropagation(); select(claim, c.id, true); });
      return chip;
    });
    const evidenceButton = el("button", { class: "btn btn-ghost btn-sm", type: "button", "aria-controls": "evidence-panel" },
      icon("layers"), `Evidence (${claim.passages.length})`);
    evidenceButton.addEventListener("click", (event) => { event.stopPropagation(); select(claim, null, true); });
    const card = el("article", { class: `finding${claim.label === "SUPPORTED" ? "" : " is-unsettled"}`, "aria-current": String(state.claim === claim),
      "aria-label": `Statement ${number}` },
    el("span", { class: "finding-num", "aria-hidden": "true" }, String(number).padStart(2, "0")),
    el("div", {},
      el("p", { class: "finding-text" }, claim.text, " ", chipsNodes),
      el("div", { class: "finding-meta" }, verdictBadge(claim.label), claim.strength ? strengthBadge(claim.strength) : null, provenance("model"),
        claim.label === "SUPPORTED" ? el("span", {}, claim.independence ? independenceText(claim.independence)
          : plural(claim.independentSupport, "independent supporting source")) : null,
        claim.citationProblems.length ? badge("citation check failed", "error", { title: claim.citationProblems.join("\n") }) : null,
        claim.unlocated.length ? el("span", { title: "These passages could not be matched word for word to the stored extract; this is usually the indexer's own normalisation." },
          `${plural(claim.unlocated.length, "passage")} not matched verbatim`) : null,
        claim.subState ? el("span", {}, subStateText(claim.subState)) : null,
        !claim.parseOk ? badge("verifier output incomplete", "error") : null,
        claim.cautions.map((c) => cautionBadge(c, run.evidencePolicy)),
        claim.id ? el("span", { class: "claim-id", title: "Statement ID in the run record and the evidence review" }, claim.id) : null,
        evidenceButton),
      cautionList(claim.cautions, run.evidencePolicy)));
    card.addEventListener("click", () => select(claim, null, false));
    cards.set(claim, card);
    return card;
  };

  const failed = run.status === "FAILED" && !run.claims.length;
  const supported = run.findings.length;
  const deep = Boolean(run.depth);
  // Statements are numbered supported first, as in the Markdown report, so "F3" means the same finding in both.
  const numberOf = new Map([...run.findings, ...run.unsettled].map((c, i) => [c, i + 1]));
  let section = 1;
  const no = (fixed) => (deep ? String(++section).padStart(2, "0") : fixed);
  const report = el("article", { class: "sheet report", "aria-label": "Report" });
  const answerLead = failed ? "The run did not produce any statements, so there is nothing to report."
    : !run.claims.length ? "No statements were written, so nothing could be checked."
      : supported ? `The verifier supported ${supported} of the ${plural(run.claims.length, "model-written statement")} against the retrieved passages.`
        : `None of the ${plural(run.claims.length, "model-written statement")} could be supported by the retrieved passages.`;
  const refs = (part) => part.claims.map((i) => run.claims[i]).filter(Boolean).map((claim) => {
    const ref = el("button", { class: "cite is-support", type: "button", title: truncate(claim.text, 200),
      "aria-label": `Finding ${numberOf.get(claim)}: show its evidence` }, `F${numberOf.get(claim)}`);
    ref.addEventListener("click", () => { select(claim, null, true); if (cards.has(claim)) cards.get(claim).scrollIntoView({ block: "nearest" }); });
    return ref;
  });
  const summary = deep ? run.summary : null;
  const direct = summary && summary.available && summary.directAnswer ? summary.directAnswer : null;
  const legacyNote = "Supported means at least one retrieved passage was judged to support the statement and none to contradict it — a judgement about the retrieved text, not proof that the statement is true.";
  const deepNote = direct
    ? ["The direct answer and summary were written by the local model from verified findings only; each sentence names the findings it restates (F1, F2 …). ",
      summary.dropped ? `${plural(summary.dropped, "sentence")} that cited no finding or added a number not in the findings ${summary.dropped === 1 ? "was" : "were"} removed. ` : "",
      legacyNote]
    : [summary && summary.reason ? `No direct answer is given: ${summary.reason} ` : "No direct answer is given; the answer is the set of findings below. ", legacyNote];
  report.append(el("section", { class: "report-section", "aria-labelledby": "answer-h" },
    el("div", { class: "row-between" }, el("p", { class: "eyebrow", id: "answer-h" }, direct ? "Direct answer" : "Answer"), toggle),
    el("div", { class: "answer-box" },
      el("div", { class: "stack-sm" },
        direct ? el("p", { class: "answer-lead" }, direct.text, " ", refs(direct)) : el("p", { class: "answer-lead" }, answerLead),
        direct && summary.points.length ? el("ul", { class: "answer-points", "aria-label": "Summary" }, summary.points.map((p) => el("li", {}, p.text, " ", refs(p)))) : null,
        direct ? el("p", { class: "small muted" }, answerLead) : null,
        el("div", { class: "row-tight" }, run.outcome ? outcomeBadge(run.outcome) : statusBadge(run.status),
          deep && run.stopping ? completenessBadge(run.stopping.completeness) : null,
          direct ? provenance("model") : null, run.claims.length ? provenance("verifier") : null)),
      run.claims.length ? el("div", { class: "answer-count" }, el("span", { class: "big" }, String(supported), el("small", {}, ` / ${run.claims.length}`)), el("span", { class: "xsmall muted" }, "verified of the model-drafted statements")) : null,
      deep && run.claims.length ? el("p", { class: "answer-note" }, deepNote)
        : run.claims.length ? el("p", { class: "answer-note" }, "This pipeline does not write a direct answer or an executive summary; its answer is the set of supported findings below. ", legacyNote) : null),
    deep ? stoppingCallout(run) : null));

  const emptyFindings = () => emptyState({ iconName: failed ? "alert" : "info", title: failed ? "No findings: the run did not complete" : "No finding was supported",
    body: failed ? "See the error above. Nothing was reported as a finding." : outcomeMeta(run.outcome || "INSUFFICIENT_EVIDENCE").description,
    actions: failed ? [el("a", { class: "btn btn-primary", href: "#/research" }, "Try another question")] : [] });
  if (deep) {
    // Every drafted statement, grouped by the question it answers; the verdict and evidence label sit on each card.
    const ledgerOf = new Map(run.ledger.map((e) => [e.index, e]));
    const findings = el("section", { class: "report-section", "aria-labelledby": "findings-h" },
      sectionHead("01", "Findings by question", { id: "findings-h", aside: `${supported} of ${plural(run.claims.length, "statement")} supported` }));
    if (!run.claims.length) findings.append(emptyFindings());
    for (const group of findingsBySubquestion(run)) {
      const entry = ledgerOf.get(group.index);
      const ordered = [...group.claims].sort((a, b) => numberOf.get(a) - numberOf.get(b));
      findings.append(el("div", { class: "question-group" },
        el("div", { class: "question-head" }, el("h3", {}, group.title), entry ? coverageBadge(entry.status) : null),
        el("div", {}, ordered.map((c) => claimCard(c, numberOf.get(c))))));
    }
    report.append(findings);
  } else {
    let n = 1;
    const findings = el("section", { class: "report-section", "aria-labelledby": "findings-h" },
      sectionHead("01", "Supported findings", { id: "findings-h", aside: `${supported} of ${plural(run.claims.length, "statement")}` }));
    if (supported) findings.append(el("div", {}, run.findings.map((c) => claimCard(c, n++))));
    else findings.append(emptyFindings());
    report.append(findings);

    if (run.unsettled.length) {
      report.append(el("section", { class: "report-section", "aria-labelledby": "unsettled-h" },
        sectionHead("02", "Statements not established", { id: "unsettled-h", aside: "drafted by the model, not reported as findings" }),
        el("div", {}, run.unsettled.map((c) => claimCard(c, n++)))));
    }
  }
  if (deep && run.counter && (run.counter.skipped || run.counter.changes.length)) {
    const c = run.counter;
    report.append(el("section", { class: "report-section", "aria-labelledby": "counter-h" },
      sectionHead(no(), "Counter-evidence", { id: "counter-h", aside: c.skipped ? "not searched" : `${c.changes.filter((x) => x.changed).length} of ${plural(c.changes.length, "finding")} changed` }),
      c.skipped ? el("p", { class: "small" }, `Not searched: ${c.skipped}.`)
        : [el("p", { class: "small muted" }, "Supported findings resting on a single origin were searched for opposing evidence, then verified again over their earlier and new passages. "
          + "Both verdicts are kept in the run record; a contradiction is never removed."),
        el("ul", { class: "note-list" }, c.changes.map((x) => el("li", {},
          el("div", {}, x.text), el("div", { class: "row-tight" }, verdictBadge(x.before), el("span", { "aria-hidden": "true" }, "→"),
            el("span", { class: "visually-hidden" }, "after re-verification:"), verdictBadge(x.after), x.changed ? badge("changed", "uncertain") : null),
          el("div", { class: "xsmall muted" }, x.reason))))]));
  }
  if (run.disagreements.length) {
    report.append(el("section", { class: "report-section", "aria-labelledby": "disagree-h" },
      sectionHead(no("03"), "Disagreements between sources", { id: "disagree-h" }),
      el("ul", { class: "note-list tone-caution" }, run.disagreements.map((d) => el("li", {}, d.text, " ",
        el("span", { class: "muted" }, d.verified ? "(detected by the verifier" : "(identified by the model, not independently checked",
          d.sources.length ? `; ${d.sources.join(", ")}` : "", ")"))))));
  }
  if (deep && run.quality) report.append(qualitySection(run, no()));
  if (deep && (run.queries.length || run.providerCoverage.length)) report.append(executionSection(run, no()));
  if (deep && run.review.length) {
    report.append(reviewSection(run, no(), (id) => { const c = run.claims.find((x) => x.id === id); if (c) select(c, null, true); }));
  }
  const open = deep ? run.ledger.filter((e) => OPEN_STATUSES.has(e.status)) : [];
  if (open.length) {
    report.append(el("section", { class: "report-section", "aria-labelledby": "open-h" },
      sectionHead(no(), "Unresolved questions", { id: "open-h", aside: `${open.length} of ${plural(run.ledger.length, "question")}` }),
      el("ul", { class: "note-list tone-lavender" }, open.map((e) => el("li", {},
        el("div", { class: "row-tight" }, el("span", {}, e.index === 0 ? "Main question: " : `Question ${e.index}: `, e.text), coverageBadge(e.status)),
        el("div", { class: "xsmall muted" }, e.reason, e.nextAction ? ` Next step: ${e.nextAction}.` : "")))),
      el("p", { class: "xsmall muted" }, "From the coverage ledger; the Coverage tab shows what was searched and found for each question.")));
  }
  if (run.gaps.length || run.limitations.length) {
    report.append(el("section", { class: "report-section", "aria-labelledby": "limits-h" },
      sectionHead(no("04"), "Uncertainty and limitations", { id: "limits-h" }),
      el("div", { class: "stack" },
        run.gaps.length ? el("div", { class: "stack-sm" }, el("div", { class: "row-between" }, el("span", { class: "card-title" }, "Gaps named by the model"), provenance("model")),
          el("ul", { class: "note-list tone-lavender" }, run.gaps.map((g) => el("li", {}, g)))) : null,
        run.limitations.length ? el("div", { class: "stack-sm" }, el("span", { class: "card-title" }, "Recorded by the pipeline"),
          el("ul", { class: "note-list tone-caution" }, run.limitations.map((g) => el("li", {}, g)))) : null)));
  }
  const entries = bibliography(run, usage);
  report.append(el("section", { class: "report-section", "aria-labelledby": "biblio-h" },
    sectionHead(no("05"), "Bibliography", { id: "biblio-h", aside: entries.length ? copyButton(() => entries.map((e) => `[${e.id}] ${e.citation}`).join("\n"), "Copy all") : null }),
    bibliographyView(run, usage)));

  function drawPanel() {
    const body = el("div", { class: "evidence-body", tabindex: "-1" });
    if (!state.claim) body.append(emptyState({ title: "No statement selected", body: "This run has no statements to inspect." }));
    else {
      const passages = state.claim.passages.filter((p) => !state.source || p.sourceId === state.source);
      const groups = groupByStance(passages);
      const card = (p) => passageCard(p, { source: run.sourceIndex.get(p.sourceId), onContext: (x) => helpers.openContext(x, run.sourceIndex.get(x.sourceId)) });
      const group = (title, list) => (list.length ? [el("p", { class: "biblio-group" }, `${title} (${list.length})`), ...list.map(card)] : []);
      append(body,
        el("p", { class: "evidence-statement" }, state.claim.text),
        el("div", { class: "row-tight" }, verdictBadge(state.claim.label), state.claim.strength ? strengthBadge(state.claim.strength) : null,
          state.source ? el("button", { class: "chip", type: "button", "aria-pressed": "true", on: { click: () => select(state.claim, null) } }, `Only ${state.source}`, icon("close")) : null),
        passages.length ? [...group("Supporting", groups.SUPPORTS), ...group("Contradicting", groups.CONTRADICTS), ...group("Neutral or not assessed", groups.OTHER)]
          : emptyState({ title: "No passages", body: state.source ? `No judged passage came from ${state.source}.` : "No passages were judged for this statement." }));
    }
    clear(panel).append(el("div", { class: "evidence-head" },
      el("div", { class: "row-between" }, el("h2", {}, "Evidence"), provenance("verifier")),
      el("p", { class: "xsmall muted" }, "Passages the verifier judged for the selected statement, grouped by its stance on each.")), body);
  }
  drawPanel();
  const nav = deep ? reportNav(report) : null;
  if (nav) report.prepend(nav);
  workspace.append(report, panel);
  if (!run.claims.length) {                         // nothing to inspect: no empty side panel
    workspace.classList.add("panel-collapsed");
    toggle.hidden = true;
  }
  return el("div", { class: "stack" }, workspace, savedNote(meta, "research"));
}

function savedNote(meta, mode) {
  if (meta.savedName) {
    return el("p", { class: "small muted" }, "Saved as ",
      el("a", { href: `#/${mode}/run/${encodeURIComponent(meta.savedName)}` }, meta.savedName), " (permanent link).");
  }
  if (meta.live) return el("p", { class: "small muted" }, "This run was not saved to disk; it is held in server memory only.");
  return null;
}

// ------------------------------------------------------------------------------------------------ verification report
function renderVerification(container, run, helpers, meta) {
  const usage = sourceUsage(run);
  const claim = run.claims[0] || null;
  const label = claim ? claim.label : run.outcome;
  const verdict = verdictMeta(label);
  const failed = run.status === "FAILED";          // a completed run with nothing readable keeps its Insufficient verdict
  container.append(el("div", { class: "row-between run-actions" }, el("span", {}), exportMenu(run, meta, helpers.ctx)));
  container.append(el("section", { class: "sheet verdict-hero", "aria-label": "Verdict" },
    el("div", { class: "stack" },
      el("p", { class: "eyebrow" }, "Claim under examination"),
      el("h1", { class: "verdict-claim" }, run.input),
      failed ? null : el("p", { class: "verdict-rule" }, el("b", {}, "Rule applied: "), verdict.rule),
      claim && claim.subState ? el("p", { class: "verdict-rule" }, el("b", {}, "Why: "), subStateText(claim.subState)) : null,
      metaLine(run, meta)),
    el("div", { class: "verdict-side" },
      el("span", { class: "eyebrow" }, "Verdict"),
      verdictStamp(label, { failed }),
      provenance("verifier"),
      claim && claim.strength ? strengthBadge(claim.strength) : null,
      claim ? el("div", { class: "tally" },
        el("span", {}, el("b", {}, String(claim.supporting.length)), "supporting sources"),
        el("span", {}, el("b", {}, String(claim.contradicting.length)), "contradicting"),
        claim.independence ? el("span", {}, independenceText(claim.independence) || "no supporting source")
          : el("span", {}, el("b", {}, `${claim.independentSupport}/${claim.independentContradiction}`), "independent for/against")) : null)));

  if (run.error) {
    const state = noAnswerState(run);
    container.append(callout(state.tone, state.title, el("p", {}, state.reason)));
  }
  const notes = el("div", { class: "stack" },
    callout("info", "What this verdict means",
      el("p", {}, "The verdict is decided by fixed rules from per-passage judgements of the retrieved text. A quotation found in a source shows only that the source contains that text; "
        + "it does not prove the claim is correct, and an Insufficient verdict is not evidence against the claim.")));
  if (claim && claim.subState === "CONFLICTING") {
    notes.append(callout("warn", "Sources disagree", el("p", {}, "At least one passage supports the claim and at least one contradicts it, so no verdict is given.")));
  }
  if (claim && claim.subState === "ADJUDICATED") {
    notes.append(callout("warn", "Sources disagreed — adjudicated", el("p", {}, "At least one passage supports the claim and at least one contradicts it. The adjudication step judged one side decisive and quoted it; the passages on the other side are still listed below. Treat this verdict with more caution than one on which the passages agree.")));
  }
  if (claim && claim.subState === "ADJUDICATED_UNSIDED") {
    notes.append(callout("warn", "Decided without evidence for or against", el("p", {}, "No retrieved passage was judged to support or contradict the claim. The adjudication step of the historical arm_c or arm_c0 decided it anyway, from passages the verifier judged neutral. Treat this verdict as unestablished.")));
  }
  if (claim && claim.subState === "RANK_TIEBREAK") {
    notes.append(callout("warn", "Sources disagreed — decided by rank", el("p", {}, "At least one passage supports the claim and at least one contradicts it. The rank tie-break took the first-ranked passage with a stance as decisive, without another model call; a first-ranked passage is not a more reliable one. The passages on the other side are still listed below. Treat this verdict with more caution than one on which the passages agree.")));
  }
  container.append(notes, el("div", { class: "rule" }));

  let bar = null;
  let passagesFor = "";
  bar = tabbed(container, run, runTabs(run, { id: "evidence", label: "Evidence", icon: "scale", count: claim ? claim.passages.length : 0 }), "evidence", (id, panel) => {
    if (id === "evidence") panel.append(verificationEvidence(run, claim, helpers, meta));
    else if (id === "coverage") panel.append(coverageView(run));
    else if (id === "sources") panel.append(sourcesView(run, usage, { onPassages: (s) => { passagesFor = s.id; bar.select("passages"); } }));
    else if (id === "passages") { panel.append(evidenceView(run, helpers, passagesFor)); passagesFor = ""; }
    else if (id === "diagnostics") panel.append(diagnosticsView(run));
    else panel.append(methodView(run));
  });
}

function verificationEvidence(run, claim, helpers, meta) {
  const parts = [];
  parts.push(...(claim ? claimEvidenceParts(run, claim, helpers) : noVerdictParts(run)));
  return verificationTail(run, claim, meta, parts);
}

// The passages judged for one claim, by stance, and what is missing when there is no decisive verdict.
function claimEvidenceParts(run, claim, helpers) {
  const parts = [];
  const groups = groupByStance(claim.passages);
  const column = (title, iconName, tone, list, note) => el("section", { class: `stance-column col-${tone}`, "aria-label": title },
    el("h3", {}, icon(iconName), title, el("span", { class: "count" }, String(list.length))),
    list.length ? el("div", { class: "stack" }, list.map((p) => passageCard(p, { source: run.sourceIndex.get(p.sourceId), onContext: (x) => helpers.openContext(x, run.sourceIndex.get(x.sourceId)) })))
      : el("p", { class: "small muted" }, note));
  parts.push(el("section", { "aria-labelledby": "judged-h" },
    sectionHead("01", "Judged passages", { id: "judged-h", aside: plural(claim.passages.length, "passage") }),
    el("div", { class: "stance-columns" },
      column("Supports", "check", "support", groups.SUPPORTS, "No passage was judged to support the claim."),
      column("Contradicts", "cross", "contradict", groups.CONTRADICTS, "No passage was judged to contradict the claim."),
      column("Neutral or not assessed", "minus", "neutral", groups.OTHER, "Every judged passage was decisive."))));
  if (claim.label !== "SUPPORTED" && claim.label !== "CONTRADICTED") {
    const missing = [];
    if (!claim.passages.length) missing.push("No passage was available to judge.");
    if (!groups.SUPPORTS.length) missing.push("No retrieved passage was judged to support the claim.");
    if (!groups.CONTRADICTS.length) missing.push("No retrieved passage was judged to contradict it.");
    if (groups.OTHER.some((p) => p.stance === "NOT_ASSESSED")) missing.push("At least one passage could not be assessed by the verifier.");
    parts.push(el("section", { "aria-labelledby": "missing-h" }, sectionHead("02", "Missing or insufficient evidence", { id: "missing-h" }),
      el("ul", { class: "note-list tone-caution" }, (claim.subState ? [subStateText(claim.subState)] : []).concat(missing).map((m) => el("li", {}, m)))));
  }
  return parts;
}

function noVerdictParts(run) {
  const parts = [];
  if (run.status === "FAILED") {
    parts.push(emptyState({ iconName: "alert", title: "No verdict was produced", body: run.error || "The run ended before the claim could be judged.",
      actions: [el("a", { class: "btn btn-primary", href: "#/verify" }, "Try another claim")] }));
  } else {
    parts.push(emptyState({ iconName: "info", title: "No passage could be judged",
      body: run.error || "No readable source was found, so the verdict is Insufficient. That is not evidence against the claim.",
      actions: [el("a", { class: "btn", href: "#/verify" }, "Try another claim")] }));
  }
  return parts;
}

function verificationTail(run, claim, meta, parts) {
  if (claim && claim.citationProblems.length) {
    parts.push(callout("error", "Citation check failed", el("ul", { class: "prose-list" }, claim.citationProblems.map((p) => el("li", {}, p)))));
  }
  if (run.depth && run.counter) {
    const change = run.counter.changes[0];
    parts.push(el("section", { "aria-labelledby": "counter-v-h" }, sectionHead("", "Counter-evidence", { id: "counter-v-h" }),
      run.counter.skipped ? el("p", { class: "small" }, `Not searched: ${run.counter.skipped}.`)
        : change ? el("div", { class: "stack-sm" }, el("div", { class: "row-tight" }, el("span", { class: "small" }, "Before:"), verdictBadge(change.before),
          el("span", { class: "small" }, "after re-verification:"), verdictBadge(change.after)), el("p", { class: "xsmall muted" }, change.reason))
          : el("p", { class: "small muted" }, "No counter-evidence change was recorded."),
      stoppingCallout(run)));
  }
  if (run.disagreements.length) {
    parts.push(el("section", {}, sectionHead("03", "Disagreements"), el("ul", { class: "note-list tone-caution" }, run.disagreements.map((d) => el("li", {}, d.text)))));
  }
  if (run.limitations.length) {
    parts.push(el("section", {}, sectionHead("04", "Limitations", { aside: "recorded by the pipeline" }),
      el("ul", { class: "note-list tone-caution" }, run.limitations.map((g) => el("li", {}, g)))));
  }
  parts.push(savedNote(meta, "verify"));
  return el("div", { class: "stack-lg" }, parts);
}

// ------------------------------------------------------------------------------------------------ source context
function openContext(loader, passage, source) {
  const body = el("div", { class: "stack" }, loading("Loading the source text"));
  modal({ title: source ? source.title : passage.sourceId, subtitle: `${passage.sourceId} · ${passage.location ? `at ${passage.location}` : `page ${passage.page}`}${source && source.access ? ` · ${ACCESS[source.access] || source.access}` : ""}`, body });
  loader(passage).then((context) => {
    clear(body);
    body.append(el("div", { class: "row-between" }, provenance("retrieved"), source && source.url ? el("span", { class: "small" }, externalLink(source.url, "Open the original source")) : null),
      el("p", { class: "small muted" }, "Text as extracted by the web researcher when the run was made; the original page may have changed since."));
    if (!context.located) {
      body.append(callout("warn", "Passage not located in the recorded source text",
        el("p", {}, "The passage could not be matched to the stored extract (it may have been normalised during indexing). The passage itself is shown below.")),
      el("p", { class: "context" }, context.passage || passage.text));
      return;
    }
    body.append(el("p", { class: "context" },
      context.truncated_before ? "… " : "", context.before, el("mark", {}, context.match), context.after, context.truncated_after ? " …" : ""));
  }).catch((error) => clear(body).append(errorCallout(error)));
}

export { COPY };
