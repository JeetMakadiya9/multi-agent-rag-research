/* Applies the saved appearance before the first paint (loaded synchronously in <head>): theme, text size, motion
   and whether technical details are shown. The same keys are managed by prefs.mjs; values are re-checked there. */
(function () {
  var prefs = {};
  try { prefs = JSON.parse(window.localStorage.getItem("studio.preferences") || "{}") || {}; } catch (error) { prefs = {}; }
  var choice = prefs.theme;
  if (!choice) { try { choice = window.localStorage.getItem("studio.theme"); } catch (error) { choice = null; } }
  if (choice !== "light" && choice !== "dark") choice = "system";
  var prefersDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  var root = document.documentElement;
  root.setAttribute("data-theme", choice === "dark" || (choice === "system" && prefersDark) ? "dark" : "light");
  root.setAttribute("data-theme-choice", choice);
  root.setAttribute("data-text-size", prefs.textSize === "large" ? "large" : "default");
  root.setAttribute("data-motion", prefs.motion === "reduce" ? "reduce" : "system");
  root.setAttribute("data-advanced", prefs.advanced === true ? "on" : "off");
})();
