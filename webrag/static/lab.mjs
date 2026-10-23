// Evaluation lab: recorded evaluation runs, shown as recorded. Nothing is re-run, re-scored or derived here.
import { api } from "./api.mjs";
import { badge, callout, chips, emptyState, errorCallout, kv, loading, modal, provenance, sectionHead } from "./components.mjs";
import { clear, el, icon } from "./dom.mjs";
import { comparability, formatDate, formatMetric, headlineMetrics, plural } from "./model.mjs";

const KINDS = { "claim verification": "Claim verification", research: "Research", unknown: "Unrecognised" };

export function renderLab(view, ctx) {
  ctx.setTitle("Evaluation lab");
  const area = el("div", {}, loading("Reading recorded evaluations"));
  const docs = el("div", {});
  view.append(el("div", { class: "page" },
    el("header", { class: "page-header" }, el("div", {}, el("p", { class: "eyebrow" }, "Evaluation lab"), el("h1", {}, "Recorded evaluations"),
      el("p", { class: "lede" }, "Results written by the webrag evaluation harness in evaluation/webrag/runs, shown exactly as recorded. Nothing is re-run or re-scored here, and per-item records are not opened."))),
    el("div", { class: "stack-lg" },
      callout("note", "Reading these numbers",
        el("p", {}, "Each run scores one verifier configuration on one frozen claim set and one frozen evidence snapshot. Runs are directly comparable only when both match; "
          + "small sets make single-run differences fragile. Rates are fractions between 0 and 1, followed by the recorded count (numerator/denominator) when there is one. "
          + "Metric definitions are in the harness README.")),
      docs, area)));
  api.evaluations({ signal: ctx.signal }).then((payload) => {
    clear(docs).append(documents(payload.documents || []));
    if (!payload.available) {
      clear(area).append(emptyState({ iconName: "lab", title: "No recorded evaluations", body: "evaluation/webrag/runs was not found, so there is nothing to show." }));
      return;
    }
    if (!payload.runs.length) { clear(area).append(emptyState({ iconName: "lab", title: "No recorded evaluations yet" })); return; }
    clear(area).append(runTable(ctx, payload.runs));
  }).catch((error) => { if (error.kind !== "aborted") clear(area).append(errorCallout(error)); });
}

function documents(list) {
  if (!list.length) return el("span", {});
  return el("section", { "aria-labelledby": "methods-h" }, sectionHead("", "Method documents", { id: "methods-h", aside: "evaluation/webrag" }),
    el("div", { class: "row" }, list.map((doc) => {
      const button = el("button", { class: "btn btn-quiet btn-sm", type: "button", title: doc.title }, icon("doc"), doc.name);
      button.addEventListener("click", () => {
        const body = el("div", {}, loading("Loading"));
        modal({ title: doc.title, subtitle: `evaluation/webrag/${doc.name}`, body });
        api.evaluationDoc(doc.name).then((d) => clear(body).append(el("pre", { class: "doc" }, d.text))).catch((e) => clear(body).append(errorCallout(e)));
      });
      return button;
    })));
}

function runTable(ctx, runs) {
  let kind = "all";
  const selected = new Set();
  const compare = el("button", { class: "btn btn-primary btn-sm", type: "button", disabled: true }, icon("compare"), "Compare selected");
  compare.addEventListener("click", () => { const [a, b] = [...selected]; ctx.navigate(`#/lab/compare/${encodeURIComponent(a)}/${encodeURIComponent(b)}`); });
  const tableArea = el("div", {});
  const draw = () => {
    const shown = runs.filter((r) => kind === "all" || r.kind === kind);
    const rows = shown.map((r) => {
      const head = headlineMetrics(r.metrics);
      const find = (key) => { const m = head.find((h) => h.key === key); return m ? m.display : "—"; };
      const box = el("input", { type: "checkbox", checked: selected.has(r.name), "aria-label": `Select ${r.name} for comparison` });
      box.addEventListener("click", (event) => event.stopPropagation());
      box.addEventListener("change", () => {
        if (box.checked) { if (selected.size >= 2) { box.checked = false; ctx.toast("Select at most two runs to compare."); return; } selected.add(r.name); }
        else selected.delete(r.name);
        compare.disabled = selected.size !== 2;
      });
      const open = () => ctx.navigate(`#/lab/run/${encodeURIComponent(r.name)}`);
      const row = el("tr", { class: "clickable", tabindex: "0", on: { click: open, keydown: (e) => { if (e.key === "Enter") open(); } } },
        el("td", {}, box),
        el("td", {}, el("div", { class: "passage-source" }, r.name), el("div", { class: "small muted" }, (r.dataset || "").split("/").pop())),
        el("td", {}, KINDS[r.kind] || r.kind),
        el("td", {}, (r.verifiers || []).join(", ") || "—"),
        el("td", { class: "num" }, find("n")),
        el("td", { class: "num" }, find("accuracy")),
        el("td", { class: "num" }, find("macro_f1")),
        el("td", {}, r.complete ? badge("Complete", "support") : badge("Incomplete", "uncertain"),
          r.dataset_matches_frozen === false ? badge("dataset hash differs", "error") : null, r.error ? badge("unreadable", "error") : null));
      return row;
    });
    clear(tableArea).append(el("div", { class: "table-wrap" }, el("table", {},
      el("thead", {}, el("tr", {}, el("th", { scope: "col" }, el("span", { class: "visually-hidden" }, "Select")), el("th", { scope: "col" }, "Run"), el("th", { scope: "col" }, "Kind"),
        el("th", { scope: "col" }, "Verifier"), el("th", { scope: "col", class: "num" }, "Items"), el("th", { scope: "col", class: "num" }, "Accuracy ", el("span", { class: "unit" }, "0–1")),
        el("th", { scope: "col", class: "num" }, "Macro-F1 ", el("span", { class: "unit" }, "0–1")), el("th", { scope: "col" }, "Record"))),
      el("tbody", {}, rows.length ? rows : el("tr", {}, el("td", { colspan: "8", class: "muted" }, "No run of this kind is recorded."))),
      el("caption", {}, "Values are copied from each run's metrics.json. A dash means the run did not record that value."))));
  };
  draw();
  const count = (k) => runs.filter((r) => k === "all" || r.kind === k).length;
  return el("section", { "aria-labelledby": "runs-h" },
    sectionHead("", "Recorded runs", { id: "runs-h", aside: plural(runs.length, "run") }),
    el("div", { class: "toolbar" },
      chips([{ value: "all", label: "All", count: count("all") }, { value: "claim verification", label: "Claims", count: count("claim verification") },
        { value: "research", label: "Research", count: count("research") }], kind, (v) => { kind = v; draw(); }, "Filter by kind"),
      el("div", { class: "row" }, el("span", { class: "xsmall muted" }, "Select two runs to compare them side by side."), compare)),
    tableArea);
}

function metaBlock(r) {
  const model = r.model && typeof r.model === "object" ? `${r.model.model || ""}${r.model.ollama_version ? ` · Ollama ${r.model.ollama_version}` : ""}` : r.model;
  return kv([
    ["Kind", KINDS[r.kind] || r.kind],
    ["Dataset", r.dataset ? el("span", {}, el("code", {}, r.dataset), r.dataset_matches_frozen === true ? " — matches its frozen hash"
      : r.dataset_matches_frozen === false ? el("span", { class: "form-error" }, " — differs from its frozen hash") : "") : null],
    ["Evidence snapshot", r.snapshot ? el("code", {}, r.snapshot) : null],
    ["Verifier", (r.verifiers || []).join(", ")],
    ["Model", model],
    ["Passages per claim", r.k], ["Per source", r.per_source], ["Fresh runner", r.fresh_runner_policy],
    ["Started", formatDate(r.started)], ["Finished", r.finished ? formatDate(r.finished) : "not recorded"],
  ]);
}

export function renderEvaluation(view, ctx, name) {
  ctx.setTitle(name);
  const area = el("div", {}, loading("Loading the run"));
  view.append(el("div", { class: "page page-narrow" }, el("div", { class: "stack" },
    el("a", { class: "btn btn-ghost btn-sm", href: "#/lab" }, icon("back"), "All evaluations"), area)));
  api.evaluation(name, { signal: ctx.signal }).then((r) => {
    clear(area).append(el("header", { class: "page-header" }, el("div", {}, el("p", { class: "eyebrow" }, "Recorded evaluation"), el("h1", {}, r.name),
      el("div", { class: "row" }, provenance("recorded"), r.complete ? badge("Complete", "support") : badge("Incomplete", "uncertain")))),
    el("div", { class: "stack-lg" },
      r.incomplete_note ? callout("warn", "Marked incomplete by the harness", el("p", {}, r.incomplete_note)) : null,
      el("section", { "aria-labelledby": "setup-h" }, sectionHead("", "Setup", { id: "setup-h", aside: "manifest.json" }), el("div", { class: "card" }, metaBlock(r))),
      metricsView(r.metrics),
      r.summary ? el("section", { "aria-labelledby": "summary-h" }, sectionHead("", "Harness summary", { id: "summary-h", aside: "summary.txt" }), el("pre", { class: "doc" }, r.summary)) : null));
  }).catch((error) => { if (error.kind !== "aborted") clear(area).append(error.status === 404 ? emptyState({ title: "Unknown evaluation run" }) : errorCallout(error)); });
}

function metricsView(metrics) {
  if (!metrics) return callout("warn", "No metrics recorded", el("p", {}, "This run has no metrics.json."));
  const head = headlineMetrics(metrics);
  const parts = [el("section", { "aria-labelledby": "headline-h" }, sectionHead("", "Headline metrics", { id: "headline-h", aside: "metrics.json" }),
    el("div", { class: "grid-3" }, head.map((m) => el("div", { class: "stat-card" },
      el("span", { class: "stat-label" }, m.label),
      el("span", { class: "stat-value" }, m.num !== null ? m.display.split(" ")[0] : m.display),
      el("span", { class: "stat-fraction" }, m.key === "n" ? "items scored" : m.num !== null ? `${m.num} of ${m.den}` : "fraction, no count recorded")))))];
  const classification = metrics.classification;
  if (classification && classification.per_class) {
    const labels = Object.keys(classification.per_class);
    parts.push(el("section", {}, sectionHead("", "Per class", { aside: "precision, recall and F1 are fractions; support and predicted are counts" }),
      el("div", { class: "table-wrap" }, el("table", {},
        el("thead", {}, el("tr", {}, ["Gold class", "Precision", "Recall", "F1", "Support", "Predicted"].map((h, i) => el("th", { scope: "col", class: i ? "num" : null }, h)))),
        el("tbody", {}, labels.map((label) => { const c = classification.per_class[label] || {};
          return el("tr", {}, el("td", {}, label), ...["precision", "recall", "f1", "support", "predicted"].map((k) => (c[k] === null
            ? el("td", { class: "num muted", title: "Recorded as null in metrics.json" }, "n/a")
            : el("td", { class: "num" }, c[k] === undefined ? "—" : String(c[k]))))); })),
        el("caption", {}, "n/a: recorded as null; —: not recorded.")))));
    if (classification.confusion) {
      const cell = (gold, predicted) => {
        const value = (classification.confusion[gold] || {})[predicted];
        return el("td", { class: gold === predicted ? "diag" : null }, value === undefined ? "—" : String(value));
      };
      const head = el("thead", {}, el("tr", {}, el("th", { scope: "col" }, "Gold \\ predicted"), labels.map((p) => el("th", { scope: "col" }, p))));
      const body = el("tbody", {}, labels.map((gold) => el("tr", {}, el("th", { scope: "row" }, gold), labels.map((p) => cell(gold, p)))));
      parts.push(el("section", {},
        sectionHead("", "Confusion matrix", { aside: "rows: gold label · columns: predicted · cells are item counts" }),
        el("div", { class: "table-wrap" }, el("table", { class: "matrix" }, head, body))));
    }
  }
  const tables = [];
  if (metrics.latency_seconds && typeof metrics.latency_seconds === "object") {
    tables.push(["Latency", "seconds", Object.entries(metrics.latency_seconds).map(([k, v]) => [k, typeof v === "number" ? `${v} s` : JSON.stringify(v)])]);
  }
  if (metrics.accuracy_by_tag) tables.push(["Accuracy by tag", "fraction (correct/total)", Object.entries(metrics.accuracy_by_tag).map(([k, v]) => [k, formatMetric(v)])]);
  if (metrics.error_categories) tables.push(["Error categories", "item counts", Object.entries(metrics.error_categories).map(([k, v]) => [k, String(v)])]);
  if (metrics.sources_by_access) tables.push(["Sources by access", "source counts", Object.entries(metrics.sources_by_access).map(([k, v]) => [k, String(v)])]);
  const scalars = Object.entries(metrics).filter(([k, v]) => typeof v === "number" && k !== "n").map(([k, v]) => [k.replace(/_/g, " "), String(v)]);
  if (scalars.length) tables.push(["Other recorded values", "as recorded", scalars]);
  for (const [title, unit, rows] of tables) {
    parts.push(el("section", {}, sectionHead("", title, { aside: unit }),
      el("div", { class: "table-wrap" }, el("table", {}, el("tbody", {}, rows.map(([k, v]) => el("tr", {}, el("td", {}, k), el("td", { class: "num" }, v))))))));
  }
  return el("div", { class: "stack-lg" }, parts);
}

export function renderCompare(view, ctx, a, b) {
  ctx.setTitle("Compare evaluations");
  const area = el("div", {}, loading("Loading both runs"));
  view.append(el("div", { class: "page page-narrow" }, el("div", { class: "stack" },
    el("a", { class: "btn btn-ghost btn-sm", href: "#/lab" }, icon("back"), "All evaluations"),
    el("header", { class: "page-header" }, el("div", {}, el("p", { class: "eyebrow" }, "Comparison"), el("h1", {}, "Side by side"))), area)));
  Promise.all([api.evaluation(a, { signal: ctx.signal }), api.evaluation(b, { signal: ctx.signal })]).then(([left, right]) => {
    const verdict = comparability(left, right);
    const lh = headlineMetrics(left.metrics), rh = headlineMetrics(right.metrics);
    const keys = [...new Set([...lh, ...rh].map((m) => m.key))];
    const label = (key) => (lh.find((m) => m.key === key) || rh.find((m) => m.key === key)).label;
    const value = (list, key) => { const m = list.find((x) => x.key === key); return m ? m.display : "not recorded"; };
    const perClass = (r, cls, k) => { const c = r.metrics && r.metrics.classification && r.metrics.classification.per_class && r.metrics.classification.per_class[cls]; return c && c[k] !== undefined ? String(c[k]) : "—"; };
    const classes = [...new Set([left, right].flatMap((r) => Object.keys((r.metrics && r.metrics.classification && r.metrics.classification.per_class) || {})))];
    clear(area).append(el("div", { class: "stack-lg" },
      verdict.comparable ? callout("success", "Comparable", el("p", {}, verdict.reason)) : callout("warn", "Not directly comparable", el("p", {}, verdict.reason)),
      el("p", { class: "small muted" }, "Values are shown as recorded; no differences, significance or rankings are computed."),
      el("div", { class: "grid-2" }, el("section", { class: "card" }, el("h2", { class: "card-title" }, left.name), metaBlock(left)),
        el("section", { class: "card" }, el("h2", { class: "card-title" }, right.name), metaBlock(right))),
      el("div", { class: "table-wrap" }, el("table", {},
        el("thead", {}, el("tr", {}, el("th", { scope: "col" }, "Metric"), el("th", { scope: "col", class: "num" }, left.name), el("th", { scope: "col", class: "num" }, right.name))),
        el("tbody", {}, keys.map((k) => el("tr", {}, el("td", {}, label(k)), el("td", { class: "num" }, value(lh, k)), el("td", { class: "num" }, value(rh, k)))),
          classes.map((cls) => el("tr", {}, el("td", {}, `F1 — ${cls}`), el("td", { class: "num" }, perClass(left, cls, "f1")), el("td", { class: "num" }, perClass(right, cls, "f1"))))))),
    ));
  }).catch((error) => { if (error.kind !== "aborted") clear(area).append(errorCallout(error)); });
}
