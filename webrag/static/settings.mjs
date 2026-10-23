// Settings: profile, appearance, research preferences, the AI model, privacy and data, advanced options and about.
// Every control here changes real behaviour; options that belong to the server are shown read-only, with how to
// change them, because the interface cannot change them.
import { api } from "./api.mjs";
import { PRODUCT, logoMark } from "./brand.mjs";
import { badge, callout, confirmDialog, downloadText, emptyState, errorCallout, kv, loading, sectionHead } from "./components.mjs";
import { clear, el, icon } from "./dom.mjs";
import { DEPTH_LABELS, formatDate, plural, runtimeInfo } from "./model.mjs";
import { browserStorage, exportLocalData, normalizeProfile, preferences } from "./prefs.mjs";

const SECTIONS = [
  { id: "profile", label: "Profile", icon: "user" },
  { id: "appearance", label: "Appearance", icon: "sun" },
  { id: "research", label: "Research", icon: "research" },
  { id: "model", label: "AI model", icon: "chip" },
  { id: "privacy", label: "Privacy and data", icon: "lock" },
  { id: "advanced", label: "Advanced", icon: "settings" },
  { id: "about", label: "About", icon: "info" },
];

export function renderSettings(view, ctx, section = "profile") {
  const current = SECTIONS.some((s) => s.id === section) ? section : "profile";
  const meta = SECTIONS.find((s) => s.id === current);
  ctx.setTitle(`Settings · ${meta.label}`);
  const body = el("div", { class: "settings-body" });
  view.append(el("div", { class: "page" },
    el("header", { class: "page-header" }, el("div", {}, el("p", { class: "eyebrow" }, "Settings"), el("h1", {}, meta.label))),
    el("div", { class: "settings" },
      el("nav", { class: "settings-nav", "aria-label": "Settings sections" }, SECTIONS.map((s) => el("a", {
        class: "settings-link", href: `#/settings/${s.id}`, "aria-current": s.id === current ? "page" : null }, icon(s.icon), s.label))),
      body)));
  ({ profile, appearance, research, model, privacy, advanced, about })[current](body, ctx);
  const link = view.querySelector(".settings-link[aria-current]");
  const nav = link && link.parentElement;
  if (nav && nav.scrollWidth > nav.clientWidth) nav.scrollLeft = link.offsetLeft - (nav.clientWidth - link.offsetWidth) / 2;   // narrow screens
}

// A group of radio options laid out as cards; changing it calls onChange immediately.
function choiceGroup(name, legend, options, value, onChange, hint) {
  return el("fieldset", { class: "choice-group" }, el("legend", {}, legend), hint ? el("p", { class: "hint" }, hint) : null,
    el("div", { class: "choices" }, options.map((o) => el("label", { class: "choice" },
      el("input", { type: "radio", name, value: o.value, checked: o.value === value, on: { change: () => onChange(o.value) } }),
      el("span", { class: "choice-body" }, o.icon ? icon(o.icon) : null, el("span", { class: "choice-title" }, o.label),
        o.note ? el("span", { class: "choice-note" }, o.note) : null)))));
}

function toggle(label, note, checked, onChange) {
  const input = el("input", { type: "checkbox", role: "switch", checked, on: { change: () => onChange(input.checked) } });
  return el("label", { class: "toggle" }, input, el("span", { class: "toggle-track", "aria-hidden": "true" }),
    el("span", {}, el("span", { class: "choice-title" }, label), note ? el("span", { class: "choice-note" }, note) : null));
}

const saved = (ctx, ok) => ctx.toast(ok ? "Saved in this browser" : "This browser refused to store the setting (private mode?); it applies until you reload.");

function settingsCard(title, ...children) {
  return el("section", { class: "card stack settings-card" }, title ? el("h2", { class: "card-title" }, title) : null, ...children);
}

// ------------------------------------------------------------------------------------------------ profile
function profile(body, ctx) {
  const current = preferences.profile();
  const name = el("input", { class: "input", id: "profile-name", maxlength: "40", autocomplete: "nickname", value: current.name });
  const initials = el("input", { class: "input", id: "profile-initials", maxlength: "3", autocomplete: "off", value: current.initials, placeholder: current.display || "AB" });
  const preview = el("span", { class: "avatar avatar-lg", "aria-hidden": "true" });
  const previewName = el("strong", {});
  const update = () => {
    const value = { name: name.value, initials: initials.value };
    const shown = normalizeProfile(value).display;
    clear(preview).append(shown || icon("user"));
    previewName.textContent = name.value.trim() || "Local profile";
    return value;
  };
  update();
  name.addEventListener("input", update);
  initials.addEventListener("input", update);
  const form = el("form", { class: "stack", novalidate: true },
    el("div", { class: "row" }, preview, el("div", {}, previewName, el("p", { class: "hint" }, "Shown in the sidebar and on the home page."))),
    el("div", { class: "field" }, el("label", { class: "label", for: "profile-name" }, "Display name"), name, el("p", { class: "hint" }, "Optional. Up to 40 characters.")),
    el("div", { class: "field" }, el("label", { class: "label", for: "profile-initials" }, "Initials"), initials, el("p", { class: "hint" }, "Optional. Leave empty to use the initials of your name.")),
    el("div", { class: "row" }, el("button", { class: "btn btn-primary", type: "submit" }, "Save profile"),
      el("button", { class: "btn btn-ghost", type: "button", on: { click: () => { name.value = ""; initials.value = ""; update(); } } }, "Clear")));
  form.addEventListener("submit", (event) => { event.preventDefault(); saved(ctx, preferences.setProfile(update())); });
  body.append(
    settingsCard("Local profile", form),
    callout("note", "This is not an account", el("p", {}, "The profile lives only in this browser's storage. It does not sign you in anywhere, "
      + "is never sent to a search service or model, and another browser on this computer will not see it. The studio has no accounts.")));
}

// ------------------------------------------------------------------------------------------------ appearance
function appearance(body, ctx) {
  const prefs = preferences.get();
  const set = (patch) => saved(ctx, preferences.set(patch));
  body.append(settingsCard(null,
    choiceGroup("theme", "Theme", [
      { value: "light", label: "Light", icon: "sun", note: "Warm paper" },
      { value: "dark", label: "Dark", icon: "moon", note: "Warm charcoal" },
      { value: "system", label: "System", icon: "monitor", note: "Follows your operating system" }], prefs.theme, (v) => set({ theme: v })),
    choiceGroup("text-size", "Text size", [
      { value: "default", label: "Default" }, { value: "large", label: "Larger", note: "About 12% larger text everywhere" }], prefs.textSize, (v) => set({ textSize: v })),
    choiceGroup("motion", "Motion", [
      { value: "system", label: "Follow system", note: "Uses your operating system's reduced-motion setting" },
      { value: "reduce", label: "Reduce motion", note: "No spinners, sweeps or transitions" }], prefs.motion, (v) => set({ motion: v }))));
}

// ------------------------------------------------------------------------------------------------ research
function research(body, ctx) {
  const prefs = preferences.get();
  const set = (patch) => saved(ctx, preferences.set(patch));
  const server = el("div", {}, loading("Reading the server configuration"));
  body.append(
    settingsCard("Your defaults",
      choiceGroup("default-mode", "Start new investigations as", [
        { value: "research", label: "Research a question", icon: "research", note: "Cited findings, each verified" },
        { value: "verify", label: "Verify a claim", icon: "verify", note: "A verdict with passages for and against" }], prefs.defaultMode, (v) => set({ defaultMode: v })),
      choiceGroup("evidence-panel", "Evidence panel in reports", [
        { value: "open", label: "Open", note: "Show the passages beside the findings" },
        { value: "closed", label: "Closed", note: "More room for the report; open it when needed" }], prefs.evidencePanel, (v) => set({ evidencePanel: v })),
      choiceGroup("default-depth", "Default research depth", [
        { value: "quick", label: DEPTH_LABELS.quick, note: "One search round, no counter-evidence search" },
        { value: "standard", label: DEPTH_LABELS.standard, note: "Two rounds with follow-up queries and a counter-evidence check" },
        { value: "deep", label: DEPTH_LABELS.deep, note: "Three rounds, more sources and more counter-evidence checks" }], prefs.defaultDepth, (v) => set({ defaultDepth: v }),
      "Preselected in the question box; each run can still choose another depth.")),
    settingsCard("Depth budgets and search sources", server));
  api.providers({ signal: ctx.signal }).then((payload) => {
    const depths = payload.depths || {};
    const names = Object.keys(DEPTH_LABELS).filter((d) => depths[d]);
    const rows = [["Search rounds", "rounds"], ["Queries per round", "queries_per_round"], ["Searches (query × provider)", "searches"],
      ["Sources selected", "sources"], ["Full-text attempts", "full_text"], ["Sub-questions", "subquestions"], ["Findings drafted", "findings"],
      ["Counter-evidence checks", "counter_evidence"], ["Searching stops after (minutes)", "seconds"]];
    const active = (payload.providers || []).filter((p) => p.active);
    clear(server).append(
      el("div", { class: "table-wrap" }, el("table", {},
        el("thead", {}, el("tr", {}, el("th", { scope: "col" }, "Limit"), names.map((d) => el("th", { scope: "col", class: "num" }, DEPTH_LABELS[d])))),
        el("tbody", {}, rows.map(([label, key]) => el("tr", {}, el("th", { scope: "row" }, label),
          names.map((d) => el("td", { class: "num" }, key === "seconds" ? String(Math.round(depths[d][key] / 60)) : String(depths[d][key])))))),
        el("caption", {}, "Upper limits; research stops earlier when every question has corroborated full-text evidence or when a round finds nothing new."))),
      kv([["Active search sources", active.length ? active.map((p) => p.name).join(", ") : "none"],
        ["General web search", active.some((p) => p.id === "searxng" || p.id === "brave") ? "active" : "not set up (SearXNG or Brave)"]]),
      el("p", { class: "hint" }, "Budgets are fixed per depth by the research engine (webrag/deep.py) so runs stay comparable; they are not editable here. "
        + "Search sources are switched on, off and set up in Integrations."),
      el("a", { class: "btn btn-quiet btn-sm", href: "#/integrations" }, icon("plug"), "Research sources in Integrations"));
  }).catch((error) => { if (error.kind !== "aborted") clear(server).append(errorCallout(error)); });
}

// ------------------------------------------------------------------------------------------------ AI model
function model(body, ctx) {
  const area = el("div", { class: "stack" }, loading("Checking the model"));
  body.append(area);
  api.status({ signal: ctx.signal }).then((status) => {
    const info = runtimeInfo(status);
    const runtime = !info.reachable ? badge("Not reachable", "error") : status.model_installed ? badge("Ready", "support", { iconName: "check" }) : badge("Model missing", "uncertain");
    if (info.remote) {
      clear(area).append(settingsCard("Execution path",
        el("div", { class: "row-between" }, el("div", {}, el("strong", {}, info.name), el("p", { class: "hint" }, "Every model call goes to this model service, configured by the server operator. Prompts and retrieved passages are sent to it.")), runtime),
        kv([["Model in use", el("code", {}, status.model)], ["Verifier model", el("code", {}, (status.runtime.verifier_model && status.runtime.verifier_model.model) || status.model)],
          ["Service version", info.version || "not reported"], ["Problem", info.error || "none"],
          ["Fallback", "none — if the model is not offered, runs stop with an explanation"], ["Checked", status.checked_at ? formatDate(status.checked_at) : null]]),
        el("p", { class: "hint" }, "The model is chosen in the server configuration. Verdict quality was evaluated with qwen3:4b on a local Ollama; other models have not been evaluated."),
        el("a", { class: "btn btn-quiet btn-sm", href: "#/integrations" }, icon("plug"), "Test the connection in Integrations")));
      return;
    }
    clear(area).append(
      settingsCard("Execution path",
        el("div", { class: "row-between" }, el("div", {}, el("strong", {}, "Local · Ollama"), el("p", { class: "hint" }, "Every model call goes to the Ollama server on this computer.")), runtime),
        kv([["Model in use", el("code", {}, status.model)], ["Installed models", (status.models || []).length ? status.models.join(", ") : status.ollama_version ? "none" : "unknown (Ollama not reachable)"],
          ["Ollama version", status.ollama_version || "not reachable"], ["Fallback", "none — if the model is missing, runs stop with an explanation"],
          ["Checked", status.checked_at ? formatDate(status.checked_at) : null]]),
        el("p", { class: "hint" }, "The model is chosen when the server starts. To use another installed model, restart with ",
          el("code", {}, "--model NAME"), ". The interface does not offer a switch, because changing it here would not change the running pipeline."),
        el("a", { class: "btn btn-quiet btn-sm", href: "#/integrations" }, icon("plug"), "Test the connection in Integrations")),
      settingsCard("Cloud models",
        el("div", { class: "row-between" }, el("strong", {}, "OpenAI, Anthropic and other hosted models"), badge("Not available", "neutral")),
        el("p", { class: "small" }, "This studio runs the local model: its model client accepts a local Ollama address and refuses cloud models, so "
          + "no prompt, document or passage is sent to a cloud model provider. A hosted deployment can instead be configured by its operator with an "
          + "Ollama or OpenAI-compatible model service (streamlit_cloud/README.md). A chat subscription does not include API access for other applications.")));
  }).catch((error) => { if (error.kind !== "aborted") clear(area).append(errorCallout(error)); });
}

// ------------------------------------------------------------------------------------------------ privacy and data
function privacy(body, ctx) {
  const where = el("div", {}, loading("Reading where data is kept"));
  const network = el("div", {}, loading("Reading the configured services"));
  const credentials = el("div", {}, loading("Checking credentials"));
  const history = el("button", { class: "btn", type: "button" }, icon("download"), "Download research history (JSON)");
  history.addEventListener("click", async () => {
    history.disabled = true;
    try {
      const { runs } = await api.runs();
      downloadText(`research-history-${new Date().toISOString().slice(0, 10)}.json`, JSON.stringify({ exported: new Date().toISOString(),
        note: "Index of saved runs. Each full report can be downloaded from its page.", runs }, null, 2));
    } catch (error) { ctx.toast(error.message); }
    history.disabled = false;
  });
  const local = el("button", { class: "btn", type: "button", on: { click: () => downloadText("research-studio-preferences.json", JSON.stringify(exportLocalData(browserStorage()), null, 2)) } },
    icon("download"), "Download preferences and profile (JSON)");
  const reset = el("button", { class: "btn btn-danger", type: "button" }, "Reset preferences and profile");
  reset.addEventListener("click", async () => {
    const ok = await confirmDialog({ title: "Reset preferences and profile?", danger: true, confirm: "Reset",
      body: el("p", {}, "This removes the theme, text size, motion, research defaults, technical-details setting and local profile stored in this browser. "
        + "Saved research runs, documents and everything else on disk are not touched.") });
    if (!ok) return;
    ctx.toast(preferences.reset() ? "Preferences and profile were reset" : "This browser did not allow the reset");
    ctx.reload();
  });
  body.append(
    settingsCard("Where your data is kept", where),
    settingsCard("What leaves this computer", network),
    settingsCard("Credentials", credentials),
    settingsCard("Export and reset", el("div", { class: "row" }, history, local), el("div", { class: "row" }, reset),
      el("p", { class: "hint" }, "Each report can also be downloaded as Markdown or as its full run record from its own page. "
        + "The studio cannot delete saved runs or documents: remove their folders or files on disk instead. "
        + "Search services may keep their own logs of the queries sent to them; the studio cannot delete those.")));
  Promise.all([api.status({ signal: ctx.signal }), api.runs({ signal: ctx.signal }), api.library({ signal: ctx.signal }).catch(() => null)]).then(([status, runs, lib]) => {
    clear(where).append(el("ul", { class: "note-list" },
      el("li", {}, el("strong", {}, "Saved research runs: "), `${plural((runs.runs || []).length, "run")} in `, el("code", {}, status.runs_folder || "outputs/webrag/runs"),
        " — questions, sources with their extracted text, passages, verdicts and stages."),
      el("li", {}, el("strong", {}, "Documents: "), lib && lib.exists ? `${plural(lib.documents.length, "file")} in ` : "the folder ", el("code", {}, status.docs_folder || (lib && lib.folder) || "data/documents"), "."),
      el("li", {}, el("strong", {}, "Server memory only: "), "runs in progress, document answers and the document index; they are gone when the server stops."),
      el("li", {}, el("strong", {}, "This browser: "), "preferences and the local profile.")));
    const providers = status.providers || [];
    clear(network).append(el("ul", { class: "note-list tone-caution" },
      el("li", {}, el("strong", {}, "Research and claim verification: "), "search queries are sent to ", providers.length ? providers.join(", ") : "the configured providers",
        ", and the sources they return are fetched from their own web sites (robots.txt is respected for pages)."),
      el("li", {}, el("strong", {}, "The language model: "), runtimeInfo(status).remote
        ? `runs on ${runtimeInfo(status).name}; prompts and retrieved passages are sent to that service.`
        : "runs locally in Ollama; prompts and passages stay on this computer."),
      el("li", {}, el("strong", {}, "Document questions: "), "use your documents, local embedding models and the local model only; nothing is searched online.")),
    el("p", { class: "hint" }, "Research needs an internet connection; the document assistant and saved reports do not."));
  }).catch((error) => { if (error.kind !== "aborted") clear(where).append(errorCallout(error)); });
  api.integrations({ signal: ctx.signal }).then(({ integrations }) => {
    const rows = [];
    for (const item of integrations) for (const [key, value] of Object.entries(item.facts || {})) if (/^[A-Z_]+( \(secret\))?$/.test(key) && key !== "SEARXNG_URL") rows.push([key.replace(" (secret)", ""), value]);
    clear(credentials).append(rows.length ? kv(rows) : el("p", { class: "small" }, "No credentials are used."),
      el("p", { class: "hint" }, "Keys are read from environment variables when the server starts. The studio shows only whether each one is set — never its value — and does not write keys into run records or exports."));
  }).catch((error) => { if (error.kind !== "aborted") clear(credentials).append(errorCallout(error)); });
}

// ------------------------------------------------------------------------------------------------ advanced
function advanced(body, ctx) {
  const prefs = preferences.get();
  const server = el("div", {}, loading("Reading the server configuration"));
  body.append(
    settingsCard("Technical details",
      toggle("Show technical details", "Retrieval scores, stage timings, model and verifier names, a Diagnostics tab in reports and the Evaluation lab in the sidebar.",
        prefs.advanced, (on) => saved(ctx, preferences.set({ advanced: on }))),
      el("div", { class: "row" }, el("a", { class: "btn btn-quiet btn-sm", href: "#/lab" }, icon("lab"), "Open the evaluation lab"))),
    settingsCard("Server", server));
  api.status({ signal: ctx.signal }).then((status) => {
    clear(server).append(kv([["Address", `http://${status.host || "127.0.0.1"}:${status.port || ""}`], ["Research verifier", status.verifier], ["Document assistant verifier", status.document_verifier],
      ["Model", status.model], ["Saved runs folder", status.runs_folder], ["Document folder", status.docs_folder]]),
    callout("warn", "For use on this computer only", el("p", {}, "The server listens on 127.0.0.1, answers only requests addressed to this computer and refuses cross-site writes, "
      + "but it has no accounts or passwords. Do not expose it to a network or the internet as it is.")));
  }).catch((error) => { if (error.kind !== "aborted") clear(server).append(errorCallout(error)); });
}

// ------------------------------------------------------------------------------------------------ about
function about(body, ctx) {
  const facts = el("div", {}, loading("Reading version information"));
  body.append(settingsCard(null,
    el("div", { class: "about-brand" }, logoMark({ size: 64, label: `${PRODUCT.name} logo` }),
      el("div", {}, el("p", { class: "about-name" }, PRODUCT.full), el("p", { class: "muted" }, PRODUCT.tagline))),
    el("p", {}, "A local research workspace on a multi-agent retrieval-augmented pipeline: it plans searches, reads the sources it can reach, retrieves passages, "
      + "writes cited statements and reports only those a verifier judged supported by the retrieved text."),
    facts,
    el("p", { class: "hint" }, "The mark shows a source page with one line drawn out onto the next page: evidence carried from a source into your notes.")));
  api.status({ signal: ctx.signal }).then((status) => {
    clear(facts).append(kv([["Studio version", status.studio_version], ["Model runtime", runtimeInfo(status).remote ? runtimeInfo(status).name : status.ollama_version ? `Ollama ${status.ollama_version}` : "Ollama not reachable"],
      ["Documentation", el("code", {}, "webrag/README.md")]]));
  }).catch(() => clear(facts).append(emptyState({ title: "Version information unavailable" })));
}
