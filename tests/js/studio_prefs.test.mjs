// Preferences and the local profile. Run with: node --test tests/js/studio_prefs.test.mjs
// Uses an in-memory stand-in for localStorage; every value is SYNTHETIC TEST DATA.
import assert from "node:assert/strict";
import { test } from "node:test";

import {
  LEGACY_THEME_KEY, PREFS_KEY, PREF_DEFAULTS, PROFILE_KEY, exportLocalData, initialsFor, loadPreferences, loadProfile,
  normalizePreferences, normalizeProfile, resetLocalData, resolvedTheme, savePreferences, saveProfile,
} from "../../webrag/static/prefs.mjs";
import { STATUS_TONES } from "../../webrag/static/integrations.mjs";

function memoryStorage(initial = {}) {
  const data = new Map(Object.entries(initial));
  return { getItem: (k) => (data.has(k) ? data.get(k) : null), setItem: (k, v) => data.set(k, String(v)), removeItem: (k) => data.delete(k), data };
}

test("preferences are validated and unknown values fall back to the defaults", () => {
  assert.deepEqual(normalizePreferences(null), PREF_DEFAULTS);
  assert.deepEqual(normalizePreferences({ theme: "dark", textSize: "huge", motion: "reduce", defaultMode: "verify", evidencePanel: "closed", advanced: "yes" }),
    { ...PREF_DEFAULTS, theme: "dark", motion: "reduce", defaultMode: "verify", evidencePanel: "closed", advanced: false });
  assert.equal(normalizePreferences({ advanced: true }).advanced, true);
});

test("the theme chosen before preferences existed is migrated, and saving replaces the old key", () => {
  const storage = memoryStorage({ [LEGACY_THEME_KEY]: "dark" });
  assert.equal(loadPreferences(storage).theme, "dark");
  const { prefs, saved } = savePreferences(storage, { ...loadPreferences(storage), textSize: "large" });
  assert.equal(saved, true);
  assert.deepEqual(JSON.parse(storage.getItem(PREFS_KEY)), prefs);
  assert.equal(storage.getItem(LEGACY_THEME_KEY), null);
  assert.equal(loadPreferences(memoryStorage({ [PREFS_KEY]: "{not json" })).theme, "system");
});

test("a storage that refuses writes is reported, not hidden", () => {
  const refusing = { getItem: () => null, setItem: () => { throw new Error("QuotaExceededError"); }, removeItem: () => {} };
  assert.equal(savePreferences(refusing, { theme: "light" }).saved, false);
  assert.equal(loadPreferences(null).theme, "system");
});

test("the profile keeps only a short display name and initials", () => {
  assert.equal(initialsFor("Ada Lovelace"), "AL");
  assert.equal(initialsFor("Ada Byron Lovelace"), "AL");
  assert.equal(initialsFor("Ada"), "AD");
  assert.equal(initialsFor("   "), "");
  assert.deepEqual(normalizeProfile({ name: "  Ada\u0007 Lovelace  ", initials: "" }), { name: "Ada Lovelace", initials: "", display: "AL" });
  assert.equal(normalizeProfile({ name: "x".repeat(60) }).name.length, 40);
  assert.equal(normalizeProfile({ initials: "a.b-cd" }).display, "ABC");
  const storage = memoryStorage();
  saveProfile(storage, { name: "Ada Lovelace", initials: "", email: "should-not-be-kept@example.org" });
  assert.deepEqual(JSON.parse(storage.getItem(PROFILE_KEY)), { name: "Ada Lovelace", initials: "" });
  assert.equal(loadProfile(storage).display, "AL");
  saveProfile(storage, { name: "", initials: "" });
  assert.equal(storage.getItem(PROFILE_KEY), null);                       // an empty profile is removed, not stored
});

test("reset removes only the studio's own keys and export describes them", () => {
  const storage = memoryStorage({ [PREFS_KEY]: JSON.stringify({ theme: "dark" }), [PROFILE_KEY]: JSON.stringify({ name: "Ada" }), other: "kept" });
  const exported = exportLocalData(storage);
  assert.equal(exported.preferences.theme, "dark");
  assert.deepEqual(exported.profile, { name: "Ada", initials: "" });
  assert.equal(resetLocalData(storage), true);
  assert.deepEqual([...storage.data.keys()], ["other"]);
});

test("theme resolution and integration status tones", () => {
  assert.equal(resolvedTheme("system", true), "dark");
  assert.equal(resolvedTheme("system", false), "light");
  assert.equal(resolvedTheme("light", true), "light");
  assert.equal(STATUS_TONES.ready, "support");
  assert.equal(STATUS_TONES.not_supported, "neutral");
  assert.equal(STATUS_TONES.unavailable, "error");
});
