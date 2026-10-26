// Application shell: navigation, routing, preferences, the local profile menu, backend status and notifications.
import { api } from "./api.mjs";
import { PRODUCT, brandLink, logoMark } from "./brand.mjs";
import { menuButton } from "./components.mjs";
import { clear, el, icon } from "./dom.mjs";
import { renderHome } from "./home.mjs";
import { renderIntegrations } from "./integrations.mjs";
import { renderLab, renderEvaluation, renderCompare } from "./lab.mjs";
import { renderLibrary, renderLibraryJob } from "./library.mjs";
import { CHANGE_EVENT, applyPreferences, preferences } from "./prefs.mjs";
import { runtimeInfo } from "./model.mjs";
import { renderSettings } from "./settings.mjs";
import { renderRunPage, renderStart } from "./workspace.mjs";

const APP = PRODUCT.full;

const NAV = [
  { id: "research", href: "#/", label: "Research", icon: "research" },
  { id: "library", href: "#/library", label: "Library", icon: "library" },
  { id: "integrations", href: "#/integrations", label: "Integrations", icon: "plug" },
  { id: "settings", href: "#/settings", label: "Settings", icon: "settings" },
  { group: "Advanced", advanced: true },
  { id: "lab", href: "#/lab", label: "Evaluation lab", icon: "lab", advanced: true },
];

const ROUTES = [
  [/^\/?$/, "research", (view, ctx) => renderHome(view, ctx)],
  [/^\/(research|verify)$/, "research", (view, ctx, [mode]) => renderStart(view, ctx, mode)],
  [/^\/(research|verify)\/job\/([A-Za-z0-9]+)$/, "research", (view, ctx, [mode, id]) => renderRunPage(view, ctx, { mode, jobId: id })],
  [/^\/(research|verify)\/run\/([^/]+)$/, "research", (view, ctx, [mode, name]) => renderRunPage(view, ctx, { mode, runName: decodeURIComponent(name) })],
  [/^\/sessions$/, "library", () => location.replace("#/library?view=reports")],
  [/^\/library$/, "library", (view, ctx) => renderLibrary(view, ctx)],
  [/^\/library\/job\/([A-Za-z0-9]+)$/, "library", (view, ctx, [id]) => renderLibraryJob(view, ctx, id)],
  [/^\/integrations$/, "integrations", (view, ctx) => renderIntegrations(view, ctx)],
  [/^\/settings(?:\/([a-z-]+))?$/, "settings", (view, ctx, [section]) => renderSettings(view, ctx, section)],
  [/^\/lab$/, "lab", (view, ctx) => renderLab(view, ctx)],
  [/^\/lab\/run\/([^/]+)$/, "lab", (view, ctx, [name]) => renderEvaluation(view, ctx, decodeURIComponent(name))],
  [/^\/lab\/compare\/([^/]+)\/([^/]+)$/, "lab", (view, ctx, [a, b]) => renderCompare(view, ctx, decodeURIComponent(a), decodeURIComponent(b))],
];

// ------------------------------------------------------------------------------------------------ shell
const shell = {};

function navigation() {
  const prefs = preferences.get();
  shell.links = new Map();
  return el("nav", { class: "nav", "aria-label": "Main" }, NAV.filter((item) => !item.advanced || prefs.advanced).map((item) => {
    if (item.group) return el("p", { class: "nav-group", "aria-hidden": "true" }, item.group);
    const link = el("a", { class: "nav-item", href: item.href }, icon(item.icon), item.label);
    link.addEventListener("click", closeMenu);
    shell.links.set(item.id, link);
    return link;
  }));
}

function avatar(profile, size = "") {
  return el("span", { class: `avatar${size ? ` avatar-${size}` : ""}`, "aria-hidden": "true" }, profile.display || icon("user"));
}

function profileMenu() {
  const profile = preferences.profile();
  return menuButton({
    className: "profile-button", align: "start", label: "Profile menu",
    content: [avatar(profile), el("span", { class: "profile-name" }, profile.name || "Local profile"), icon("chevrondown")],
    items: [
      { heading: profile.name || "Local profile", note: "Stored in this browser only. It is not an account and signs you in nowhere." },
      { separator: true },
      { label: "Profile and preferences", href: "#/settings/profile", iconName: "user" },
      { label: "Appearance", href: "#/settings/appearance", iconName: "sun" },
      { label: "Data and privacy", href: "#/settings/privacy", iconName: "lock" },
      { separator: true },
      { label: `About ${PRODUCT.name}`, href: "#/settings/about", iconName: "info" },
    ],
  });
}

function buildShell() {
  const root = document.getElementById("app");
  clear(root);
  shell.navSlot = el("div", {}, navigation());
  shell.status = el("a", { class: "status-line status-link", href: "#/integrations", title: "Open Integrations" },
    el("span", { class: "dot" }), el("span", { class: "small muted" }, "Checking the model…"));
  shell.profileSlot = el("div", { class: "profile-slot" }, profileMenu());
  shell.sidebar = el("aside", { class: "sidebar", id: "sidebar" },
    brandLink("#/"),
    el("a", { class: "btn btn-primary new-research", href: "#/research", on: { click: closeMenu } }, icon("plus"), "New research"),
    shell.navSlot,
    el("div", { class: "sidebar-foot" },
      el("div", {}, el("div", { class: "foot-label" }, "Engine"), shell.status),
      shell.profileSlot));
  shell.menuButton = el("button", { class: "icon-btn", type: "button", "aria-label": "Open navigation", "aria-expanded": "false", "aria-controls": "sidebar" }, icon("menu"));
  shell.menuButton.addEventListener("click", () => (shell.sidebar.classList.contains("is-open") ? closeMenu() : openMenu()));
  shell.view = el("main", { class: "view", id: "view", tabindex: "-1" });
  shell.live = el("div", { class: "visually-hidden", "aria-live": "polite", id: "announcer" });
  shell.toasts = el("div", { class: "toast-region", "aria-live": "polite" });
  root.append(
    el("a", { class: "skip-link", href: "#view", on: { click: (e) => { e.preventDefault(); shell.view.focus(); } } }, "Skip to content"),
    el("div", { class: "app" }, shell.sidebar,
      el("div", { class: "main" },
        el("header", { class: "topbar" }, shell.menuButton,
          el("a", { class: "topbar-brand", href: "#/" }, logoMark({ size: 24 }), el("span", { class: "wordmark-name" }, PRODUCT.name)),
          el("a", { class: "icon-btn", href: "#/research", "aria-label": "New research" }, icon("plus"))),
        shell.view)),
    shell.live, shell.toasts);
  document.addEventListener("keydown", (event) => { if (event.key === "Escape" && shell.sidebar.classList.contains("is-open")) closeMenu(); });
  window.addEventListener(CHANGE_EVENT, () => {
    clear(shell.navSlot).append(navigation());
    clear(shell.profileSlot).append(profileMenu());
    markCurrent();
  });
}

function openMenu() {
  shell.sidebar.classList.add("is-open");
  shell.menuButton.setAttribute("aria-expanded", "true");
  shell.scrim = el("div", { class: "scrim", on: { click: closeMenu } });
  document.body.append(shell.scrim);
  const first = shell.sidebar.querySelector("a");
  if (first) first.focus();
}
function closeMenu() {
  if (!shell.sidebar || !shell.sidebar.classList.contains("is-open")) return;
  shell.sidebar.classList.remove("is-open");
  shell.menuButton.setAttribute("aria-expanded", "false");
  if (shell.scrim) shell.scrim.remove();
}

function toast(message) {
  const node = el("div", { class: "toast" }, message);
  shell.toasts.append(node);
  setTimeout(() => node.remove(), 4500);
}

function announce(message) {
  shell.live.textContent = "";
  setTimeout(() => { shell.live.textContent = message; }, 30);
}

// ------------------------------------------------------------------------------------------------ status
let lastStatus = null;
async function refreshStatus() {
  const [dot, label] = shell.status.children;
  try {
    lastStatus = await api.status();
    const runtime = runtimeInfo(lastStatus);
    if (!runtime.reachable) {
      dot.className = "dot bad";
      label.textContent = runtime.remote ? "Model service not reachable" : "Local model not reachable";
    } else if (!lastStatus.model_installed) {
      dot.className = "dot warn";
      label.textContent = runtime.remote ? `${lastStatus.model} is not offered` : `${lastStatus.model} is not installed`;
    } else {
      dot.className = "dot ok";
      label.textContent = `${lastStatus.model} ready`;
    }
  } catch (error) {
    lastStatus = null;
    dot.className = "dot bad";
    label.textContent = error.message;
  }
  return lastStatus;
}

// ------------------------------------------------------------------------------------------------ router
let routeController = null;
let currentSection = null;

function markCurrent() {
  for (const [id, link] of shell.links) {
    if (id === currentSection) link.setAttribute("aria-current", "page"); else link.removeAttribute("aria-current");
  }
}

function route() {
  const hash = location.hash.replace(/^#/, "") || "/";
  const [path, search] = hash.split("?");
  if (routeController) routeController.abort();
  routeController = new AbortController();
  let match = null;
  for (const [pattern, section, render] of ROUTES) {
    const m = path.match(pattern);
    if (m) { match = { section, render, params: m.slice(1) }; break; }
  }
  currentSection = match ? match.section : null;
  markCurrent();
  clear(shell.view);
  const ctx = {
    api, signal: routeController.signal, toast, announce, refreshStatus,
    get status() { return lastStatus; },
    prefs: preferences.get(),
    query: new URLSearchParams(search || ""),
    navigate: (target) => { location.hash = target; },
    reload: () => route(),
    setTitle: (title) => { document.title = title ? `${title} · ${APP}` : APP; },
  };
  if (!match) {
    ctx.setTitle("Not found");
    shell.view.append(el("div", { class: "page page-narrow" }, el("h1", { tabindex: "-1" }, "Page not found"),
      el("p", { class: "lede" }, "This address does not match a page of the studio."), el("a", { class: "btn", href: "#/" }, "Go to Research")));
  } else {
    try {
      match.render(shell.view, ctx, match.params);
    } catch (error) {
      shell.view.append(el("div", { class: "page" }, el("h1", {}, "This page could not be shown"), el("p", { class: "lede" }, String(error.message || error))));
    }
  }
  const heading = shell.view.querySelector("h1");
  if (heading) { heading.setAttribute("tabindex", "-1"); heading.focus({ preventScroll: true }); }
  window.scrollTo(0, 0);
}

// ------------------------------------------------------------------------------------------------ start
applyPreferences(preferences.get());
buildShell();
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => applyPreferences(preferences.get()));
window.addEventListener("hashchange", route);
route();
refreshStatus();
