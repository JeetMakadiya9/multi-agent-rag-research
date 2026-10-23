// Research home: the question box first, then work in progress, recent investigations and the library shelf.
import { api } from "./api.mjs";
import { callout, emptyState, errorCallout, loading, sectionHead } from "./components.mjs";
import { clear, el, icon } from "./dom.mjs";
import { plural, runtimeInfo, truncate } from "./model.mjs";
import { preferences } from "./prefs.mjs";
import { composer, sessionList } from "./workspace.mjs";

const jobHref = (j) => (j.mode && j.mode.startsWith("documents") ? `#/library/job/${j.id}` : `#/${j.mode}/job/${j.id}`);
const isLive = (j) => j.state === "queued" || j.state === "running";

export function renderHome(view, ctx) {
  ctx.setTitle("");
  const profile = preferences.profile();
  const first = profile.name ? profile.name.split(/\s+/)[0] : "";
  const notices = el("div", { class: "stack" });
  const recent = el("div", {}, loading("Loading saved investigations"));
  const shelf = {
    reports: el("span", { class: "shelf-detail" }, "Counting saved reports…"),
    sources: el("span", { class: "shelf-detail" }, "Gathering sources…"),
    documents: el("span", { class: "shelf-detail" }, "Reading the document folder…"),
  };
  const guidance = el("div", {});
  const shelfItem = (href, iconName, hue, title, detail) => el("li", {}, el("a", { class: "shelf-item", href },
    el("span", { class: `shelf-icon${hue ? ` hue-${hue}` : ""}` }, icon(iconName)),
    el("span", {}, el("span", { class: "shelf-title" }, title), detail), icon("chevron")));
  view.append(el("div", { class: "page" },
    el("header", { class: "home-head" }, el("div", {},
      el("h1", { class: "home-title" }, "What are you ", el("em", {}, "investigating"), first ? el("span", {}, `, ${first}`) : null, "?"),
      el("p", { class: "home-sub" }, "Ask a question or test a claim. Sources are searched and read, and only statements a verifier could support with the retrieved text are reported as findings."))),
    el("div", { class: "stack-lg" },
      composer(ctx, { mode: ctx.prefs.defaultMode, switchable: true }),
      notices,
      el("div", { class: "split" },
        el("section", { "aria-labelledby": "recent-h" }, sectionHead("", "Recent investigations", { id: "recent-h", aside: el("a", { href: "#/library" }, "All reports") }), recent),
        el("section", { "aria-labelledby": "shelf-h" }, sectionHead("", "Library", { id: "shelf-h" }),
          el("ul", { class: "shelf" },
            shelfItem("#/library", "report", "", "Reports", shelf.reports),
            shelfItem("#/library?view=sources", "book", "ochre", "Sources", shelf.sources),
            shelfItem("#/library?view=documents", "file", "lavender", "Documents", shelf.documents)))),
      guidance)));

  ctx.refreshStatus().then((status) => {
    const runtime = runtimeInfo(status);
    if (runtime.ready) return;
    const title = runtime.remote
      ? (runtime.reachable ? `The model ${status.model} is not offered by the model service` : "The model service is not reachable")
      : (runtime.reachable ? `The model ${status.model} is not installed` : "The local model is not reachable");
    const hint = runtime.remote
      ? `${runtime.error || "No answer from the service."} Check the model service settings in .env. Saved reports and the library still work.`
      : runtime.reachable ? `Install it with "ollama pull ${status.model}" before starting research.` : "Start Ollama on this computer before starting research. Saved reports and the library still work.";
    notices.prepend(callout("warn", title,
      el("p", {}, hint),
      el("div", { class: "row" }, el("a", { class: "btn btn-sm", href: "#/integrations" }, icon("plug"), "Open Integrations"))));
  });

  api.jobs({ signal: ctx.signal }).then(({ jobs }) => {
    const live = (jobs || []).filter(isLive);
    if (live.length) {
      notices.append(callout("info", `${plural(live.length, "run")} in progress`,
        el("ul", { class: "prose-list" }, live.map((j) => el("li", {},
          el("a", { href: jobHref(j) }, truncate(j.input || (j.mode === "documents-index" ? "Index the document library" : j.mode), 110)),
          el("span", { class: "muted" }, ` — ${j.state === "queued" ? `queued${j.position ? `, position ${j.position}` : ""}` : "running"}`))))));
    }
  }).catch(() => {});

  api.runs({ signal: ctx.signal }).then(({ runs }) => {
    const all = runs || [];
    clear(recent).append(all.length ? sessionList(all.slice(0, 6))
      : emptyState({ iconName: "sessions", title: "No investigations yet", body: "Your first research run or claim verification is saved automatically and appears here." }));
    shelf.reports.textContent = all.length ? `${plural(all.length, "saved report")}` : "None saved yet";
    if (!all.length) guidance.append(firstUse());
  }).catch((error) => {
    if (error.kind === "aborted") return;
    clear(recent).append(errorCallout(error));
    shelf.reports.textContent = "Saved reports could not be read";
  });

  api.sources({ signal: ctx.signal }).then(({ sources }) => {
    const used = (sources || []).filter((s) => s.used_in > 0).length;
    shelf.sources.textContent = (sources || []).length ? `${plural(sources.length, "source")}, ${used} used as evidence` : "None yet";
  }).catch((error) => { if (error.kind !== "aborted") shelf.sources.textContent = "Sources could not be read"; });

  api.library({ signal: ctx.signal }).then((lib) => {
    if (lib.available === false) { shelf.documents.textContent = "Not configured in this server"; return; }
    if (!lib.exists) { shelf.documents.textContent = "No documents yet"; return; }
    const readable = lib.documents.filter((d) => d.readable).length;
    shelf.documents.textContent = `${plural(lib.documents.length, "file")}, ${readable} readable`;
  }).catch((error) => { if (error.kind !== "aborted") shelf.documents.textContent = "The document folder could not be read"; });
}

function firstUse() {
  return el("section", { "aria-labelledby": "start-h" },
    sectionHead("", "Getting started", { id: "start-h" }),
    el("ol", { class: "steps" },
      el("li", {}, el("strong", {}, "Start the local model"), "Run Ollama with ", el("code", {}, "qwen3:4b"), " installed. The sidebar shows whether it is reachable."),
      el("li", {}, el("strong", {}, "Ask or verify"), "A research question produces cited findings; a claim produces a verdict with the passages for and against it."),
      el("li", {}, el("strong", {}, "Inspect the evidence"), "Every statement links to the passages it was checked against, and each passage can be shown inside its source.")));
}
