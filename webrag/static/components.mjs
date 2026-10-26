// Reusable interface components. Pages compose these; none of them fetch data.
import { el, icon } from "./dom.mjs";
import {
  ACCESS, ACCESS_LEVEL, cautionMeta, completenessMeta, coverageMeta, formatSeconds, outcomeMeta, progressSteps, safeUrl,
  sourceJourney, sourceKind, stanceMeta, statusMeta, strengthMeta, verdictMeta,
} from "./model.mjs";

export const strengthBadge = (value) => { const m = strengthMeta(value); return m.label ? badge(m.label, m.tone, { title: m.description }) : null; };

// Evidence-quality cautions on a finding: a badge per caution (icon and the word "Caution", not colour alone) and the
// recorded explanations, worded as in the Markdown report. Warnings, not verdicts.
export const cautionBadge = (caution, policy) => {
  const m = cautionMeta(caution, policy);
  return badge(`Caution: ${m.label}`, "uncertain", { iconName: "alert", title: m.description || m.detail || undefined });
};
export function cautionList(cautions, policy) {
  if (!Array.isArray(cautions) || !cautions.length) return null;
  return el("ul", { class: "prose-list finding-cautions", "aria-label": "Evidence-quality cautions" }, cautions.map((c) => {
    const m = cautionMeta(c, policy);
    return el("li", {}, el("strong", {}, `Caution (${m.label})`), m.detail ? `: ${m.detail}` : ": no explanation was recorded.",
      " Not used in the direct answer or summary.");
  }));
}
export const coverageBadge = (value) => { const m = coverageMeta(value); return badge(m.label, m.tone); };
export const completenessBadge = (value) => { const m = completenessMeta(value); return m.label ? badge(m.label, m.tone, { title: m.description }) : null; };

// ------------------------------------------------------------------------------------------------ badges and marks
export function badge(text, tone = "neutral", { large = false, title, iconName } = {}) {
  return el("span", { class: `badge tone-${tone}${large ? " badge-lg" : ""}${iconName ? " has-icon" : ""}`, title },
    iconName ? icon(iconName) : null, text);
}

const VERDICT_ICONS = { support: "check", contradict: "cross", uncertain: "minus", error: "alert", neutral: "pending" };
const STANCE_ICONS = { SUPPORTS: "tick", CONTRADICTS: "x", NEUTRAL: "minus", NOT_ASSESSED: "pending" };

// "High confidence · 2 for, 0 against": agreement across the judged passages (a count, not a probability).
export const confidenceBadge = (confidence) => {
  if (!confidence || confidence.level === "abstained") return null;
  const tone = { high: "support", medium: "neutral", low: "uncertain" }[confidence.level] || "neutral";
  return badge(`${confidence.label} · ${confidence.agreeing} for, ${confidence.opposing} against`, tone, {
    title: `${confidence.agreeing} of ${confidence.judged} judged passages back this verdict and ${confidence.opposing} oppose it. `
      + "This counts passages; it is not a probability." });
};

export const verdictBadge = (label, opts) => {
  const m = verdictMeta(label);
  return badge(m.label, m.tone, { title: m.rule, iconName: VERDICT_ICONS[m.tone], ...opts });
};
export const outcomeBadge = (value, opts) => { const m = outcomeMeta(value); return badge(m.label, m.tone, { title: m.description, ...opts }); };
export const statusBadge = (value, opts) => { const m = statusMeta(value); return badge(m.label, m.tone, { title: m.description, ...opts }); };
export const stanceTag = (value) => {
  const m = stanceMeta(value);
  return el("span", { class: `stance ${m.key}` }, icon(STANCE_ICONS[m.key] || "pending"), m.label);
};

export function verdictStamp(label, { failed = false } = {}) {
  const m = failed ? { label: "No verdict", tone: "error" } : verdictMeta(label);
  return el("span", { class: `stamp tone-${m.tone}` }, icon(VERDICT_ICONS[m.tone] || "pending"), m.label);
}

export function kindMark(kind) {
  const meta = sourceKind(kind);
  return el("span", { class: `kind hue-${meta.hue}` }, meta.label);
}

export function accessBadge(access) {
  const level = ACCESS_LEVEL[access] || "none";
  const tone = access === "METADATA_ONLY" ? "neutral" : { full: "support", partial: "uncertain", none: "error" }[level];
  return badge(ACCESS[access] || access || "Access not recorded", access ? tone : "neutral",
    { iconName: access === "METADATA_ONLY" ? "file" : { full: "book", partial: "doc", none: "cross" }[level] });
}

export function provenance(kind) {
  const map = {
    model: ["pen", "Model-written"],
    retrieved: ["quote", "Retrieved source text"],
    verifier: ["scale", "Verifier judgement"],
    recorded: ["database", "Recorded result"],
  };
  const [name, label] = map[kind] || map.model;
  return el("span", { class: `provenance from-${map[kind] ? kind : "model"}` }, icon(name), label);
}

// ------------------------------------------------------------------------------------------------ messages
export function callout(kind, title, ...body) {
  const names = { info: "info", note: "pen", warn: "alert", error: "alert", success: "check" };
  return el("div", { class: `callout callout-${kind}`, role: kind === "error" ? "alert" : null },
    icon(names[kind] || "info"), el("div", {}, title ? el("strong", {}, title) : null, ...body));
}

export function errorCallout(error, retry) {
  return callout("error", "Something went wrong", el("p", {}, error && error.message ? error.message : String(error)),
    retry ? el("div", { class: "row" }, el("button", { class: "btn btn-sm", type: "button", on: { click: retry } }, icon("refresh"), "Try again")) : null);
}

export function emptyState({ iconName = "info", title, body, actions = [] }) {
  return el("div", { class: "empty" }, icon(iconName), el("h3", {}, title), body ? el("p", {}, body) : null,
    actions.length ? el("div", { class: "row" }, actions) : null);
}

export function skeleton(lines = 4) {
  return el("div", { "aria-hidden": "true" }, Array.from({ length: lines }, () => el("div", { class: "skeleton" })));
}

export function loading(label = "Loading") {
  return el("div", { class: "row muted small", role: "status" }, el("span", { class: "spinner" }), label);
}

export function kv(pairs) {
  return el("dl", { class: "kv" }, pairs.filter(([, v]) => v !== null && v !== undefined && v !== "")
    .map(([k, v]) => [el("dt", {}, k), el("dd", {}, v)]));
}

export function externalLink(url, label) {
  const href = safeUrl(url);
  if (!href) return el("span", {}, label);
  return el("a", { href, target: "_blank", rel: "noopener noreferrer" }, label, " ", icon("external", "inline-icon"));
}

export function sectionHead(number, title, { aside = null, id } = {}) {
  return el("header", { class: "section-head" }, number ? el("span", { class: "section-no", "aria-hidden": "true" }, number) : null,
    el("h2", { id }, title), aside ? el("span", { class: "section-aside" }, aside) : null);
}

// ------------------------------------------------------------------------------------------------ progress
const STEP_MARKS = { ok: "tick", partial: "bang", failed: "x" };
const STEP_STATE = { ok: "done", partial: "partial", failed: "failed", active: "in progress", pending: "not started" };

export function stepper(steps, { live = false } = {}) {
  return el("ol", { class: "stepper", "aria-live": live ? "polite" : null }, steps.map((s) => el("li", { class: `step is-${s.state}` },
    el("span", { class: "step-mark", "aria-hidden": "true" }, s.state === "active" ? el("span", { class: "spinner" }) : STEP_MARKS[s.state] ? icon(STEP_MARKS[s.state]) : null),
    el("div", {}, el("span", { class: "step-name" }, s.label),
      el("span", { class: s.state === "active" || s.state === "pending" ? "step-state" : "visually-hidden" }, STEP_STATE[s.state] || s.state),
      s.detail ? el("span", { class: "step-detail" }, s.detail) : null),
    el("span", { class: "step-time" }, formatSeconds(s.seconds)))));
}

// Recorded stages of a finished run, in the order the backend reported them.
export function timeline(stages, mode = "research") {
  if (!stages.length) return el("p", { class: "small muted" }, "No stages were recorded.");
  return stepper(progressSteps(mode, stages));
}

// ------------------------------------------------------------------------------------------------ evidence
export function passageCard(passage, { source, onContext, showStance = true, clamp = true } = {}) {
  const title = source ? source.title : passage.title || passage.sourceId;
  const url = (source && source.url) || passage.url;
  const long = clamp && passage.text.length > 420;
  const textNode = el("p", { class: `passage-text${long ? " clamped" : ""}` }, passage.text);
  const actions = [];
  if (long) {
    const toggle = el("button", { class: "btn btn-ghost btn-sm", type: "button", "aria-expanded": "false" }, "Show full passage");
    toggle.addEventListener("click", () => {
      const open = textNode.classList.toggle("clamped") === false;
      toggle.setAttribute("aria-expanded", String(open));
      toggle.textContent = open ? "Show less" : "Show full passage";
    });
    actions.push(toggle);
  }
  if (onContext && passage.key && passage.sourceId) {
    actions.push(el("button", { class: "btn btn-ghost btn-sm", type: "button", on: { click: () => onContext(passage) } },
      icon("book"), "Show in source"));
  }
  return el("article", { class: `passage stance-${showStance ? passage.stance || "NOT_ASSESSED" : "none"}` },
    el("div", { class: "passage-head" },
      el("span", { class: "passage-source" }, el("span", { class: "source-id" }, passage.sourceId), externalLink(url, title)),
      showStance && passage.stance ? stanceTag(passage.stance) : null),
    textNode,
    el("div", { class: "passage-meta" },
      provenance("retrieved"),
      passage.foundBySearch ? el("span", { title: "Found by the agentic verifier's planner search of this run's evidence" }, `found by search: ${passage.foundBySearch}`) : null,
      showStance && passage.reason ? el("span", { class: "passage-reason", title: "Why the verifier counted this passage, or did not" }, passage.reason) : null,
      source && (source.category || source.kind) ? kindMark(source.category || source.kind) : null,
      passage.location ? el("span", { title: "Time range of the transcript this passage comes from" }, `at ${passage.location}`)
        : !source || source.pageCount > 1 || passage.page > 1 ? el("span", {}, `page ${passage.page}`) : null,
      passage.access ? el("span", {}, ACCESS[passage.access] || passage.access) : null,
      passage.score !== null ? el("span", { class: "adv-only", title: "Retrieval score from rag.py (higher ranks first)" }, `score ${passage.score.toFixed(3)}`) : null),
    actions.length ? el("div", { class: "passage-actions" }, actions) : null);
}

const JOURNEY_ICONS = { done: "check", partial: "alert", failed: "cross", none: "pending" };

export function journeyTrack(steps) {
  return el("ol", { class: "journey", "aria-label": "What happened to this source" }, steps.map((s) => el("li", { class: `journey-step is-${s.state}` },
    el("span", { class: "journey-label" }, icon(JOURNEY_ICONS[s.state] || "pending"), s.label),
    el("span", { class: "journey-detail" }, s.detail))));
}

// One source with its metadata, how much of it was read and how it was used. `usage` comes from sourceUsage().
const ID_LINKS = { doi: (v) => `https://doi.org/${v}`, arxiv: (v) => `https://arxiv.org/abs/${v}`,
  pmid: (v) => `https://pubmed.ncbi.nlm.nih.gov/${v}/`, pmcid: (v) => `https://www.ncbi.nlm.nih.gov/pmc/articles/${v}/` };

export function sourceEntry(source, usage, { onPassages } = {}) {
  const u = usage || null;
  const authors = source.authors && source.authors.length ? `${source.authors.slice(0, 3).join(", ")}${source.authors.length > 3 ? " et al." : ""}` : "";
  const ids = Object.entries(source.identifiers || {}).filter(([k]) => ID_LINKS[k]);
  return el("article", { class: "source-entry" },
    el("div", { class: "stack-sm" },
      el("div", { class: "source-title" }, el("span", { class: "source-id" }, source.id), externalLink(source.url, source.title)),
      el("div", { class: "source-meta" },
        kindMark(source.category || source.kind),
        authors ? el("span", {}, authors) : null,
        source.publisher || source.domain ? el("span", {}, source.publisher || source.domain) : null,
        source.published ? el("span", { title: "Publication date as the source states it" }, `published ${source.published.slice(0, 10)}`) : null,
        source.retrievedAt ? el("span", { class: "adv-only", title: "When the studio retrieved it" }, `retrieved ${source.retrievedAt.slice(0, 10)}`) : null,
        source.discoveredBy && source.discoveredBy.length > 1 ? el("span", {}, `found by ${source.discoveredBy.join(", ")}`)
          : source.provider ? el("span", {}, `via ${source.provider}`) : null,
        ids.map(([k, v]) => el("span", {}, externalLink(ID_LINKS[k](v), `${k.toUpperCase()} ${v}`))),
        source.license ? el("span", {}, `licence ${source.license}`) : null,
        source.origin && source.origin !== source.id ? badge(`near-duplicate of ${source.origin}`, "uncertain") : null)),
    el("div", { class: "source-actions" }, accessBadge(source.access),
      onPassages && u && u.retrieved ? el("button", { class: "btn btn-quiet btn-sm", type: "button", on: { click: () => onPassages(source) } }, icon("layers"), "Passages") : null),
    journeyTrack(sourceJourney(source, u)),
    source.note ? el("p", { class: "source-note" }, source.note) : null);
}

export function sourceRow(source) {
  return sourceEntry(source, null);
}

// ------------------------------------------------------------------------------------------------ dialogs and disclosure
// A modal dialog. `dialog.dismiss()` closes it and cleans up at once (the browser's own close event, from Escape,
// does the same), so a closed dialog never lingers in the page.
export function modal({ title, subtitle, body, onClose }) {
  const dialog = el("dialog", { class: "modal", "aria-labelledby": "modal-title" });
  let finished = false;
  const finish = () => {
    if (finished) return;
    finished = true;
    dialog.remove();
    if (onClose) onClose();
  };
  dialog.dismiss = () => { if (dialog.open) dialog.close(); finish(); };
  const close = el("button", { class: "icon-btn", type: "button", "aria-label": "Close dialog" }, icon("close"));
  close.addEventListener("click", () => dialog.dismiss());
  dialog.addEventListener("close", finish);
  dialog.addEventListener("click", (event) => { if (event.target === dialog) dialog.dismiss(); });
  dialog.append(
    el("header", { class: "modal-head" }, el("div", {}, el("h2", { id: "modal-title" }, title), subtitle ? el("p", { class: "small muted" }, subtitle) : null), close),
    el("div", { class: "modal-body" }, body));
  document.body.append(dialog);
  dialog.showModal();
  return dialog;
}

export function disclosure(summary, body, open = false) {
  return el("details", { class: "disclosure", open }, el("summary", {}, summary), el("div", { class: "disclosure-body" }, body));
}

// ------------------------------------------------------------------------------------------------ controls
export function segmented(name, options, value, onChange, label) {
  return el("div", { class: "segmented", role: "radiogroup", "aria-label": label },
    options.map((o) => el("label", {},
      el("input", { type: "radio", name, value: o.value, "aria-label": o.label, checked: o.value === value, on: { change: () => onChange(o.value) } }),
      el("span", {}, o.icon ? icon(o.icon) : null, o.label))));
}

// Toggle chips for filters: exactly one is pressed.
export function chips(options, value, onChange, label) {
  const group = el("div", { class: "chips", role: "group", "aria-label": label });
  const buttons = options.map((o) => {
    const button = el("button", { class: "chip", type: "button", "aria-pressed": String(o.value === value) },
      o.label, o.count !== undefined ? el("span", { class: "count" }, String(o.count)) : null);
    button.addEventListener("click", () => {
      for (const b of buttons) b.setAttribute("aria-pressed", String(b === button));
      onChange(o.value);
    });
    return button;
  });
  group.append(...buttons);
  return group;
}

export function searchField({ label, placeholder = "Search", onInput, id }) {
  const input = el("input", { class: "input", type: "search", id, placeholder, autocomplete: "off", "aria-label": label });
  input.addEventListener("input", () => onInput(input.value));
  return el("div", { class: "search-field", role: "search" }, icon("research"), input);
}

// Tabs with arrow-key movement. `onSelect(id)` draws the panel; the caller owns the panel element.
export function tabbar(tabs, selected, onSelect, { label, panelId }) {
  const bar = el("div", { class: "tabbar", role: "tablist", "aria-label": label });
  const buttons = tabs.map((t) => {
    const button = el("button", { class: "tabbar-tab", role: "tab", type: "button", id: `tab-${t.id}`, "aria-controls": panelId,
      "aria-selected": String(t.id === selected), tabindex: t.id === selected ? "0" : "-1" },
    t.icon ? icon(t.icon) : null, t.label, t.count !== undefined ? el("span", { class: "count" }, String(t.count)) : null);
    button.addEventListener("click", () => choose(t.id));
    button.addEventListener("keydown", (event) => {
      const index = tabs.findIndex((x) => x.id === t.id);
      const move = { ArrowRight: 1, ArrowLeft: -1, Home: -index, End: tabs.length - 1 - index }[event.key];
      if (move === undefined) return;
      event.preventDefault();
      const next = tabs[(index + move + tabs.length) % tabs.length];
      choose(next.id);
      bar.querySelector(`#tab-${next.id}`).focus();
    });
    return button;
  });
  function choose(id) {
    for (const b of buttons) {
      const on = b.id === `tab-${id}`;
      b.setAttribute("aria-selected", String(on));
      b.setAttribute("tabindex", on ? "0" : "-1");
    }
    onSelect(id);
  }
  bar.append(...buttons);
  return { node: bar, select: choose };
}

// The asynchronous clipboard API first; where the browser refuses it, the older selection-based copy.
async function copyText(value) {
  try {
    await navigator.clipboard.writeText(value);
    return true;
  } catch {
    const area = el("textarea", { class: "visually-hidden", readonly: true, "aria-hidden": "true" });
    area.value = value;
    document.body.append(area);
    area.select();
    let copied = false;
    try { copied = document.execCommand("copy"); } catch { copied = false; }
    area.remove();
    return copied;
  }
}

export function copyButton(getText, label = "Copy") {
  const text = el("span", {}, label);
  const button = el("button", { class: "btn btn-ghost btn-sm", type: "button" }, icon("copy"), text);
  button.addEventListener("click", async () => {
    text.textContent = (await copyText(getText())) ? "Copied" : "Copy failed: the browser blocked the clipboard";
    setTimeout(() => { text.textContent = label; }, 2200);
  });
  return button;
}

// A button that opens a short menu of real links and actions. Arrow keys move, Escape or a click outside closes.
// items: { label, href, download, onSelect, iconName, disabled, note } | { heading, note } | { separator: true }
let menuCounter = 0;
export function menuButton({ content, items, className = "btn btn-quiet btn-sm", align = "end", label }) {
  menuCounter += 1;
  const id = `menu-${menuCounter}`;
  const button = el("button", { class: className, type: "button", "aria-haspopup": "menu", "aria-expanded": "false", "aria-controls": id, "aria-label": label }, content);
  const menu = el("div", { class: `menu menu-${align}`, role: "menu", id, hidden: true, "aria-label": label });
  const entries = [];
  for (const item of items) {
    if (item.separator) { menu.append(el("div", { class: "menu-sep", role: "separator" })); continue; }
    if (item.heading) { menu.append(el("div", { class: "menu-heading" }, el("strong", {}, item.heading), item.note ? el("span", {}, item.note) : null)); continue; }
    const inner = [item.iconName ? icon(item.iconName) : null, el("span", { class: "menu-label" }, item.label, item.note ? el("small", {}, item.note) : null)];
    const entry = item.href && !item.disabled
      ? el("a", { class: "menu-item", role: "menuitem", href: item.href, download: item.download || null, tabindex: "-1" }, inner)
      : el("button", { class: "menu-item", role: "menuitem", type: "button", tabindex: "-1", "aria-disabled": item.disabled ? "true" : null }, inner);
    entry.addEventListener("click", (event) => {
      if (item.disabled) { event.preventDefault(); return; }
      close(false);
      if (item.onSelect) item.onSelect(event);
    });
    entries.push(entry);
    menu.append(entry);
  }
  const outside = (event) => { if (!wrap.contains(event.target)) close(false); };
  function open() {
    menu.hidden = false;
    button.setAttribute("aria-expanded", "true");
    document.addEventListener("pointerdown", outside);
    (entries.find((e) => e.getAttribute("aria-disabled") !== "true") || entries[0]).focus();
  }
  function close(focusButton = true) {
    if (menu.hidden) return;
    menu.hidden = true;
    button.setAttribute("aria-expanded", "false");
    document.removeEventListener("pointerdown", outside);
    if (focusButton) button.focus();
  }
  button.addEventListener("click", () => (menu.hidden ? open() : close()));
  button.addEventListener("keydown", (event) => { if (event.key === "ArrowDown" && menu.hidden) { event.preventDefault(); open(); } });
  menu.addEventListener("keydown", (event) => {
    const index = entries.indexOf(document.activeElement);
    const move = { ArrowDown: 1, ArrowUp: -1 }[event.key];
    if (move) { event.preventDefault(); entries[(index + move + entries.length) % entries.length].focus(); }
    else if (event.key === "Home") { event.preventDefault(); entries[0].focus(); }
    else if (event.key === "End") { event.preventDefault(); entries[entries.length - 1].focus(); }
    else if (event.key === "Escape") { event.preventDefault(); close(); }
    else if (event.key === "Tab") close(false);
  });
  const wrap = el("div", { class: "menu-wrap" }, button, menu);
  return wrap;
}

// Opens a dialog asking to confirm an action; resolves true only when the confirm button was pressed.
export function confirmDialog({ title, body, confirm = "Confirm", danger = false }) {
  return new Promise((resolve) => {
    let answer = false;
    const yes = el("button", { class: `btn ${danger ? "btn-danger" : "btn-primary"}`, type: "button" }, confirm);
    const no = el("button", { class: "btn", type: "button" }, "Cancel");
    const dialog = modal({ title, body: el("div", { class: "stack" }, body, el("div", { class: "row" }, yes, no)), onClose: () => resolve(answer) });
    yes.addEventListener("click", () => { answer = true; dialog.dismiss(); });
    no.addEventListener("click", () => dialog.dismiss());
    no.focus();
  });
}

// Lets the browser save generated content as a file (no server round trip).
export function downloadText(filename, text, type = "application/json") {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const link = el("a", { href: url, download: filename, hidden: true });
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function stat(value, label) {
  return el("div", { class: "stat" }, el("span", { class: "stat-value" }, value), el("span", { class: "stat-label" }, label));
}
