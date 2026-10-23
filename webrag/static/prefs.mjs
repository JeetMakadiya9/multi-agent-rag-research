// Preferences and the local profile. Both are stored in this browser only (localStorage), like the theme before
// them: they are personal settings, not an account, and they never leave this computer. Pure functions take the
// storage object as a parameter so they can be tested without a browser.

export const PREFS_KEY = "studio.preferences";
export const PROFILE_KEY = "studio.profile";
export const LEGACY_THEME_KEY = "studio.theme";
export const CHANGE_EVENT = "studio:preferences";

export const PREF_CHOICES = {
  theme: ["system", "light", "dark"],
  textSize: ["default", "large"],
  motion: ["system", "reduce"],
  defaultMode: ["research", "verify"],
  evidencePanel: ["open", "closed"],
  defaultDepth: ["standard", "quick", "deep"],
};
export const PREF_DEFAULTS = { theme: "system", textSize: "default", motion: "system", defaultMode: "research", evidencePanel: "open",
  defaultDepth: "standard", advanced: false };

const isObject = (v) => v !== null && typeof v === "object" && !Array.isArray(v);

export function normalizePreferences(raw) {
  const value = isObject(raw) ? raw : {};
  const prefs = { ...PREF_DEFAULTS };
  for (const [key, choices] of Object.entries(PREF_CHOICES)) if (choices.includes(value[key])) prefs[key] = value[key];
  prefs.advanced = value.advanced === true;
  return prefs;
}

const CONTROL = /[\u0000-\u001f\u007f]/g;

export function initialsFor(name) {
  const words = String(name || "").trim().split(/\s+/).filter(Boolean);
  if (!words.length) return "";
  const letters = words.length === 1 ? [...words[0]].slice(0, 2) : [[...words[0]][0], [...words[words.length - 1]][0]];
  return letters.join("").toUpperCase();
}

export function normalizeProfile(raw) {
  const value = isObject(raw) ? raw : {};
  const name = String(value.name || "").replace(CONTROL, "").trim().slice(0, 40);
  const custom = String(value.initials || "").replace(/[^\p{L}\p{N}]/gu, "").slice(0, 3).toUpperCase();
  return { name, initials: custom, display: custom || initialsFor(name) };
}

function read(storage, key) {
  try { const text = storage && storage.getItem(key); return text ? JSON.parse(text) : null; } catch { return null; }
}

function write(storage, key, value) {
  try { storage.setItem(key, JSON.stringify(value)); return true; } catch { return false; }
}

export function loadPreferences(storage) {
  const stored = read(storage, PREFS_KEY);
  if (stored) return normalizePreferences(stored);
  let legacy = null;
  try { legacy = storage && storage.getItem(LEGACY_THEME_KEY); } catch { legacy = null; }
  return normalizePreferences({ theme: legacy });      // the theme chosen before preferences existed is kept
}

export function savePreferences(storage, prefs) {
  const clean = normalizePreferences(prefs);
  const ok = write(storage, PREFS_KEY, clean);
  try { storage.removeItem(LEGACY_THEME_KEY); } catch { /* nothing to migrate */ }
  return { prefs: clean, saved: ok };
}

export function loadProfile(storage) { return normalizeProfile(read(storage, PROFILE_KEY)); }

export function saveProfile(storage, profile) {
  const clean = normalizeProfile(profile);
  const ok = clean.name || clean.initials ? write(storage, PROFILE_KEY, { name: clean.name, initials: clean.initials })
    : (() => { try { storage.removeItem(PROFILE_KEY); return true; } catch { return false; } })();
  return { profile: clean, saved: ok };
}

// Removes everything the studio stores in this browser. Files on disk are not touched.
export function resetLocalData(storage) {
  let ok = true;
  for (const key of [PREFS_KEY, PROFILE_KEY, LEGACY_THEME_KEY]) { try { storage.removeItem(key); } catch { ok = false; } }
  return ok;
}

export function exportLocalData(storage) {
  return { exported: new Date().toISOString(), note: "Preferences and profile stored by Research Studio in this browser.",
    preferences: loadPreferences(storage), profile: (({ name, initials }) => ({ name, initials }))(loadProfile(storage)) };
}

// ------------------------------------------------------------------------------------------------ browser side
export function browserStorage() {
  try { return window.localStorage; } catch { return null; }
}

export function resolvedTheme(choice, prefersDark) {
  return choice === "dark" || (choice === "system" && prefersDark) ? "dark" : "light";
}

export function applyPreferences(prefs, root = document.documentElement) {
  const dark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  root.setAttribute("data-theme", resolvedTheme(prefs.theme, dark));
  root.setAttribute("data-theme-choice", prefs.theme);
  root.setAttribute("data-text-size", prefs.textSize);
  root.setAttribute("data-motion", prefs.motion);
  root.setAttribute("data-advanced", prefs.advanced ? "on" : "off");
}

// The current preferences; updating them saves, applies and tells the rest of the page.
export const preferences = {
  get() { return loadPreferences(browserStorage()); },
  set(patch) {
    const { prefs, saved } = savePreferences(browserStorage(), { ...this.get(), ...patch });
    applyPreferences(prefs);
    window.dispatchEvent(new CustomEvent(CHANGE_EVENT, { detail: { prefs } }));
    return saved;
  },
  profile() { return loadProfile(browserStorage()); },
  setProfile(profile) {
    const result = saveProfile(browserStorage(), profile);
    window.dispatchEvent(new CustomEvent(CHANGE_EVENT, { detail: { profile: result.profile } }));
    return result.saved;
  },
  reset() {
    const ok = resetLocalData(browserStorage());
    applyPreferences(PREF_DEFAULTS);
    window.dispatchEvent(new CustomEvent(CHANGE_EVENT, { detail: { prefs: { ...PREF_DEFAULTS } } }));
    return ok;
  },
};
