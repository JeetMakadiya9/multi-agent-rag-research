// Integrations: the model runtime, research sources, knowledge tools and export, each with its real status.
// Status comes from the server (configuration or a real local check); nothing here marks anything as connected.
import { api } from "./api.mjs";
import { badge, callout, disclosure, errorCallout, kv, loading, sectionHead } from "./components.mjs";
import { append, clear, el, icon } from "./dom.mjs";
import { CATEGORY_LABELS, formatDate, plural, providerState, sourceKind } from "./model.mjs";

export const STATUS_TONES = { ready: "support", active: "support", not_configured: "uncertain", unavailable: "error", error: "error", not_supported: "neutral" };
const CATEGORY_ICONS = { ai: "chip", sources: "globe", knowledge: "library", export: "download" };
const ITEM_ICONS = { ollama: "chip", "cloud-models": "cloud", documents: "library", obsidian: "file", markdown: "doc", json: "database" };
const PROVIDER_ICONS = { wikipedia: "book", openalex: "doc", arxiv: "doc", europepmc: "doc", crossref: "database", searxng: "globe", brave: "globe", feeds: "list" };
const FEED_CATEGORIES = ["news", "official", "podcast", "web"];

export function statusBadge(status) {
  const tone = STATUS_TONES[status.state] || "neutral";
  return badge(status.label, tone, { iconName: tone === "support" ? "check" : tone === "error" ? "alert" : null });
}

export function renderIntegrations(view, ctx) {
  ctx.setTitle("Integrations");
  const area = el("div", { class: "stack-lg" }, loading("Reading the integration status"));
  view.append(el("div", { class: "page" },
    el("header", { class: "page-header" }, el("div", {}, el("p", { class: "eyebrow" }, "Integrations"), el("h1", {}, "Connected services and tools"),
      el("p", { class: "lede" }, "What the studio can use, and whether each one is actually set up. A status comes from the server's configuration or a real local check — never from a button press."))),
    area));
  const load = () => api.integrations({ signal: ctx.signal }).then((payload) => draw(payload)).catch((error) => {
    if (error.kind !== "aborted") clear(area).append(errorCallout(error, load));
  });

  function draw({ categories, integrations }) {
    clear(area).append(...categories.map((category) => {
      const items = integrations.filter((i) => i.category === category.id);
      if (category.id === "sources") return researchSources(category, items, ctx);
      return el("section", { "aria-labelledby": `cat-${category.id}` },
        sectionHead("", category.title, { id: `cat-${category.id}`, aside: category.description }),
        el("div", { class: "integration-grid" }, items.map((item) => card(item))));
    }));
  }

  function card(item) {
    const node = el("article", { class: `integration is-${item.status.state}`, "aria-labelledby": `int-${item.id}` });
    const fill = (current) => {
      const facts = Object.entries(current.facts || {}).filter(([, v]) => v);
      const actions = [];
      if (current.testable) {
        const test = el("button", { class: "btn btn-sm test-button", type: "button" }, icon("refresh"), "Test connection");
        test.addEventListener("click", async () => {
          test.disabled = true;
          test.lastChild.textContent = "Testing…";
          try {
            const fresh = await api.testOllama();
            fill({ ...fresh, tested: new Date() });
            const again = node.querySelector(".test-button");
            if (again) again.focus();
            ctx.refreshStatus();
            ctx.announce(`Ollama connection test: ${fresh.status.label}`);
          } catch (error) {
            ctx.toast(error.message);
            test.disabled = false;
            test.lastChild.textContent = "Test connection";
          }
        });
        actions.push(test);
      }
      if (current.id === "documents") actions.push(el("a", { class: "btn btn-quiet btn-sm", href: "#/library?view=documents" }, icon("library"), "Open documents"));
      if (current.id === "obsidian" || current.id === "documents") {
        actions.push(el("button", { class: "btn btn-ghost btn-sm", type: "button", on: { click: () => load() } }, icon("refresh"), "Check again"));
      }
      const last = current.last_recorded;
      node.className = `integration is-${current.status.state}`;
      append(clear(node),
        el("header", { class: "integration-head" },
          el("span", { class: "integration-icon", "aria-hidden": "true" }, icon(ITEM_ICONS[current.id] || CATEGORY_ICONS[current.category] || "plug")),
          el("div", { class: "integration-title" }, el("h3", { id: `int-${current.id}` }, current.name), el("p", { class: "small muted" }, current.purpose))),
        el("div", { class: "integration-status" }, statusBadge(current.status), current.status.detail ? el("span", { class: "small" }, current.status.detail) : null),
        current.capabilities.length ? el("div", { class: "row-tight" }, current.capabilities.map((c) => el("span", { class: "chip-static" }, c))) : null,
        facts.length ? kv(facts) : null,
        last ? el("p", { class: "xsmall muted" }, `Last recorded in a saved run (${formatDate(last.started)}): ${last.queries ?? "?"} queries, `
          + `${last.failed ?? 0} failed, ${last.hits ?? 0} results.`) : null,
        el("p", { class: "xsmall muted integration-data" }, icon("lock", "inline-icon"), " ", current.data),
        current.setup.length && current.status.state !== "ready" && current.status.state !== "active"
          ? el("div", { class: "stack-sm" }, el("p", { class: "card-title" }, "How to set it up"), el("ol", { class: "prose-list" }, current.setup.map((s) => el("li", {}, s))))
          : current.setup.length ? disclosure("Setup", el("ol", { class: "prose-list" }, current.setup.map((s) => el("li", {}, s)))) : null,
        current.limitations.length ? disclosure("Limitations", el("ul", { class: "prose-list" }, current.limitations.map((s) => el("li", {}, s)))) : null,
        actions.length ? el("div", { class: "row" }, actions) : null,
        current.tested ? el("p", { class: "xsmall", role: "status" }, `Tested just now (${current.tested.toLocaleTimeString()}): ${current.status.label}.`)
          : current.checked_at ? el("p", { class: "xsmall muted" }, `Checked ${formatDate(current.checked_at)}`) : null);
    };
    fill(item);
    return node;
  }

  load();
}

// ------------------------------------------------------------------------------------------------ research sources
const field = (id, label, input, hint) => {
  input.id = id;
  if (hint) input.setAttribute("aria-describedby", `${id}-hint`);
  return el("div", { class: "field" }, el("label", { class: "label", for: id }, label), input, hint ? el("p", { class: "hint", id: `${id}-hint` }, hint) : null);
};
const textInput = (attrs) => el("input", { class: "input", autocomplete: "off", spellcheck: "false", ...attrs });

// Search providers with working controls, from /api/providers. A switch or address is saved on the server
// (outputs/webrag/studio-config.json) and applies to the next run; keys stay in environment variables.
function researchSources(category, items, ctx) {
  const body = el("div", { class: "stack" }, loading("Reading the search configuration"));
  const info = new Map(items.map((i) => [i.id, i]));
  const tested = new Map();
  let payload = null;

  const save = async (patch, message, focus) => {
    try {
      payload = await api.saveProviderConfig(patch);
      render();
      ctx.toast(message);
      ctx.refreshStatus();
      const target = focus && body.querySelector(focus);
      if (target) target.focus();
    } catch (error) {
      ctx.toast(error.message);
    }
  };

  function settingsFor(p, env) {
    const config = payload.config;
    if (p.id === "searxng") {
      const current = config.searxng || {};
      const url = textInput({ type: "url", value: current.url || "", placeholder: "http://127.0.0.1:8888", disabled: Boolean(env.SEARXNG_URL) });
      const engines = textInput({ value: current.engines || "", placeholder: "for example: duckduckgo, bing" });
      const categories = textInput({ value: current.categories || "", placeholder: "for example: general, news" });
      const form = el("form", { class: "stack-sm provider-form", novalidate: true },
        field("searxng-url", "Address of your SearXNG instance", url, env.SEARXNG_URL
          ? "Set by the SEARXNG_URL environment variable, which takes precedence over this field."
          : "Your own instance with JSON output enabled. Public instances usually refuse automated queries."),
        field("searxng-engines", "Engines (optional)", engines, "Comma-separated SearXNG engine names."),
        field("searxng-categories", "Categories (optional)", categories, "Comma-separated SearXNG categories."),
        el("div", { class: "row" }, el("button", { class: "btn btn-sm", type: "submit" }, "Save SearXNG settings")));
      form.addEventListener("submit", (event) => {
        event.preventDefault();
        save({ searxng: { url: env.SEARXNG_URL ? current.url || "" : url.value.trim(), engines: engines.value, categories: categories.value } },
          "SearXNG settings saved", "#searxng-url:not([disabled]), #searxng-engines");
      });
      return disclosure("SearXNG settings", form, !p.configured);
    }
    if (p.id === "feeds") {
      const feeds = (config.feeds || []).map((f) => ({ url: f.url, name: f.name, category: f.category }));
      const list = el("ul", { class: "feed-list", "aria-label": "Feeds" }, feeds.length ? feeds.map((f, i) => el("li", {},
        el("div", { class: "feed-text" }, el("strong", {}, f.name || f.url), el("span", { class: "xsmall muted" }, ` · ${CATEGORY_LABELS[f.category] || f.category}`),
          f.name ? el("span", { class: "xsmall muted feed-url" }, f.url) : null),
        el("button", { class: "btn btn-ghost btn-sm", type: "button", "aria-label": `Remove the feed ${f.name || f.url}`,
          on: { click: () => save({ feeds: feeds.filter((_, j) => j !== i) }, "Feed removed", "#feed-url") } }, icon("close"), "Remove")))
        : el("li", { class: "small muted" }, "No feeds yet."));
      const url = textInput({ type: "url", placeholder: "https://example.org/feed.xml" });
      const name = textInput({ maxlength: "80" });
      const kind = el("select", { class: "select" }, FEED_CATEGORIES.map((c) => el("option", { value: c }, CATEGORY_LABELS[c] || c)));
      const form = el("form", { class: "stack-sm provider-form", novalidate: true }, list,
        el("div", { class: "feed-add" }, field("feed-url", "Feed address", url), field("feed-name", "Name (optional)", name), field("feed-category", "Type", kind)),
        el("div", { class: "row" }, el("button", { class: "btn btn-sm", type: "submit" }, icon("plus"), "Add feed")),
        el("p", { class: "hint" }, "RSS or Atom. Feeds are searched by keyword in their titles and descriptions. Podcast episodes that publish a "
          + "transcript (Podcasting 2.0) are read from that transcript and quoted by timestamp; audio is never downloaded or transcribed."));
      form.addEventListener("submit", (event) => {
        event.preventDefault();
        if (!url.value.trim()) { ctx.toast("Enter the feed's address"); url.focus(); return; }
        save({ feeds: [...feeds, { url: url.value.trim(), name: name.value.trim(), category: kind.value }] }, "Feed added", "#feed-url");
      });
      return disclosure(`Feeds (${feeds.length})`, form, true);
    }
    if (p.id === "crossref") {
      const email = textInput({ type: "email", value: config.contact_email || "", placeholder: "you@example.org", autocomplete: "email" });
      const form = el("form", { class: "stack-sm provider-form", novalidate: true },
        field("contact-email", "Contact email (optional)", email, "Sent to Crossref as its documented mailto parameter, nowhere else. Leave empty to send none."),
        el("div", { class: "row" }, el("button", { class: "btn btn-sm", type: "submit" }, "Save contact email")));
      form.addEventListener("submit", (event) => { event.preventDefault(); save({ contact_email: email.value.trim() }, "Contact email saved", "#contact-email"); });
      return disclosure("Crossref contact", form, false);
    }
    return null;
  }

  function card(p, env) {
    const item = info.get(p.id) || {};
    const state = providerState(p);
    const excluded = p.evidence === false;              // source policy: cannot be switched on or tested
    const locked = excluded || p.controlled_by === "WEBRAG_PROVIDERS";
    const input = el("input", { type: "checkbox", role: "switch", id: `use-${p.id}`, checked: Boolean(p.enabled), disabled: locked });
    input.addEventListener("change", () => save({ providers: { [p.id]: { enabled: input.checked } } },
      `${p.name} ${input.checked ? "switched on" : "switched off"} for new runs`, `#use-${p.id}`));
    const result = tested.get(p.id);
    const test = excluded ? null : el("button", { class: "btn btn-sm", type: "button", id: `test-${p.id}`, disabled: !p.configured,
      title: p.configured ? null : "Set it up first" }, icon("refresh"), "Test search");
    if (test) test.addEventListener("click", async () => {
      test.disabled = true;
      test.lastChild.textContent = "Testing…";
      let outcome;
      try { outcome = await api.testProvider(p.id); } catch (error) { outcome = { ok: false, error: error.message }; }
      tested.set(p.id, { ...outcome, at: new Date() });
      render();
      const again = body.querySelector(`#test-${p.id}`);
      if (again) again.focus();
      ctx.announce(`${p.name} test search: ${outcome.ok ? `answered with ${plural(outcome.results, "result")}` : "failed"}`);
    });
    const secrets = Object.entries(p.secrets || {}).map(([name, set]) => [`${name} (secret)`, set ? "set" : "not set"]);
    const last = p.last_recorded;
    return el("article", { class: `integration provider-card is-${state.state}`, "aria-labelledby": `prov-${p.id}` },
      el("header", { class: "integration-head" },
        el("span", { class: "integration-icon", "aria-hidden": "true" }, icon(PROVIDER_ICONS[p.id] || "globe")),
        el("div", { class: "integration-title" }, el("h3", { id: `prov-${p.id}` }, p.name), el("p", { class: "small muted" }, item.purpose || p.returns))),
      el("div", { class: "integration-status" }, badge(state.label, state.tone, { iconName: state.tone === "support" ? "check" : null }),
        el("span", { class: "small" }, state.detail)),
      el("label", { class: "toggle" }, input, el("span", { class: "toggle-track", "aria-hidden": "true" }),
        el("span", {}, el("span", { class: "choice-title" }, "Use in research"),
          el("span", { class: "choice-note" }, excluded ? "Not accepted as an evidence source" : locked ? "Decided by WEBRAG_PROVIDERS"
            : p.default_on ? "On by default" : "Off by default"))),
      settingsFor(p, env),
      kv([["Source types", (p.categories && p.categories.length ? p.categories : [p.category]).map((c) => CATEGORY_LABELS[c] || sourceKind(c).label).join(", ")],
        ["Returns", p.returns], ["Content", p.content], ["Results per query (single-pass)", p.max_per_query ? String(p.max_per_query) : null], ["Candidates per search (deep research)", p.candidates_per_query ? String(p.candidates_per_query) : null],
        ["Time range", p.recency === "native" ? "filtered by the service itself" : p.recency], ["Limits", p.limits], ...secrets]),
      el("p", { class: "xsmall muted" }, last ? `Last used in a saved run (${formatDate(last.started)}): ${last.queries ?? "?"} queries, ${last.failed ?? 0} failed, ${last.hits ?? 0} results.`
        : "Not used in any saved run yet."),
      test ? el("div", { class: "row" }, test) : null,
      result ? el("p", { class: result.ok ? "xsmall" : "xsmall form-error", role: "status" }, result.ok
        ? `Tested ${result.at.toLocaleTimeString()}: answered in ${result.seconds} s with ${plural(result.results, "result")} for the test query “${result.query}”.`
        : `Tested ${result.at.toLocaleTimeString()}: ${result.error}`) : null,
      el("p", { class: "xsmall muted integration-data" }, icon("lock", "inline-icon"), " ",
        excluded ? "Nothing is sent to this service by the studio." : `${item.data || "Search queries are sent to this service."} Test search sends one neutral query (“renewable energy”) and nothing else.`),
      item.setup && item.setup.length ? disclosure("Setup", el("ol", { class: "prose-list" }, item.setup.map((s) => el("li", {}, s))), state.state === "needs_setup") : null,
      item.limitations && item.limitations.length ? disclosure("Limitations", el("ul", { class: "prose-list" }, item.limitations.map((s) => el("li", {}, s)))) : null);
  }

  function render() {
    const env = (payload.config && payload.config.environment) || {};
    const active = payload.providers.filter((p) => p.active);
    append(clear(body),
      payload.config_error ? callout("warn", "The configuration file could not be used", el("p", {}, payload.config_error)) : null,
      env.WEBRAG_PROVIDERS ? callout("info", "Providers chosen by an environment variable", el("p", {}, "WEBRAG_PROVIDERS is set to ", el("code", {}, env.WEBRAG_PROVIDERS),
        "; it decides which providers are used, so the switches below are disabled. Remove it and restart the server to use them.")) : null,
      el("p", { class: "small", role: "status" }, active.length ? `Active for new runs: ${active.map((p) => p.name).join(", ")}.`
        : "No source is active, so research cannot search. Switch one on below."),
      el("div", { class: "integration-grid" }, payload.providers.map((p) => card(p, env))));
  }

  api.providers({ signal: ctx.signal }).then((data) => { payload = data; render(); })
    .catch((error) => { if (error.kind !== "aborted") clear(body).append(errorCallout(error)); });
  return el("section", { "aria-labelledby": "cat-sources" },
    sectionHead("", category.title, { id: "cat-sources", aside: category.description }),
    el("p", { class: "small muted" }, "Switch a source on or off for every new research run; each run can narrow the selection in the question box. "
      + "These settings are saved on this computer in ", el("code", {}, "outputs/webrag/studio-config.json"), ". API keys are never entered here: they stay in "
      + "environment variables, and the studio shows only whether each one is set."),
    body);
}
