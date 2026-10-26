"""The showcase pages. They read only repository files and showcase/data/; nothing here starts a server, calls a
model or contacts a service."""
from __future__ import annotations

import html
from collections import Counter
from typing import Any, Dict, List

import altair as alt
import pandas as pd
import streamlit as st

import architecture
import content
import results

ACCENT = "#2bb3a3"
VERDICT = {"SUPPORTED": "green", "CONTRADICTED": "red", "INSUFFICIENT": "orange"}
STANCE = {"SUPPORTS": "green", "CONTRADICTS": "red", "NEUTRAL": "gray", "NOT_ASSESSED": "gray"}


def esc(text: Any) -> str:
    """Text from run records (model output, source titles) as plain markdown text."""
    return (html.escape(str(text or "")).replace("[", "\\[").replace("]", "\\]").replace("*", "\\*")
            .replace("_", "\\_").replace("#", "\\#"))


def kind_badge(kind: str) -> str:
    """Badges for page content that is not a result: explanations, and facts derived from the code."""
    return {"illustration": ":violet-badge[Explanation]", "code": ":green-badge[Derived from code]"}[kind]


def evidence_badge(kind: str) -> str:
    """The evidence type of a result (content.EVIDENCE_TYPES): benchmark, diagnostic, demonstration, ..."""
    label, color, _ = content.EVIDENCE_TYPES[kind]
    return f":{color}-badge[{label}]"


def status_badges(statuses: List[str]) -> str:
    return " ".join(f":{content.STATUS[s][1]}-badge[{content.STATUS[s][0]}]" for s in statuses)


def src(path: str, label: str = "") -> str:
    return f"[`{label or path}`]({content.source_url(path)})"


def link(key: str, label: str, icon: str) -> None:
    """A link to another page of the app (skipped when a page is rendered on its own, without the app's registry)."""
    page = st.session_state.get("pages", {}).get(key)
    if page is not None:
        st.page_link(page, label=label, icon=icon)


def header(title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(subtitle)


# ================================================================================================= 1. overview
PIPELINE = [(":material/edit_note:", "Plan", "A planner turns the question into targeted search queries."),
            (":material/travel_explore:", "Research", "A web researcher reads scholarly sources under guarded network access."),
            (":material/manage_search:", "Retrieve", "A hybrid engine ranks the passages that matter: dense, BM25 and a re-ranker."),
            (":material/draw:", "Synthesize", "A synthesizer drafts statements, each citing its evidence."),
            (":material/verified:", "Verify", "An agentic verifier checks every statement against every passage, with quotes.")]


def highlights() -> Dict[str, Any]:
    """The headline figures, read from the data files at page load (nothing typed in)."""
    registry = results.evaluation_runs() or {}
    runs = {r["key"]: r for r in results.visible_runs(registry)}
    study = results.historical_study()
    alt_data = results.studio_alt_evaluation() or {}
    out: Dict[str, Any] = {"study": study, "tests": results.test_inventory()}
    base, cand = runs.get("round3/confirm_arm_c_studio"), runs.get("round3/confirm_arm_c_studio_all")
    if base and cand and base.get("metrics") and cand.get("metrics"):
        before = int(base["metrics"]["false_support"].split("/")[0])
        after = int(cand["metrics"]["false_support"].split("/")[0])
        out["false_support"] = (before, after, cand["set_size"])
    fixed, old = runs.get("coverage_idfix/retro10"), runs.get("round3/confirm_arm_c_studio_all")
    if fixed and old and (fixed.get("coverage") or {}).get("passages_presented"):
        f, o = fixed["coverage"], old["coverage"]
        out["coverage"] = (f["valid_matched"], f["passages_presented"], o["valid_matched"] / o["passages_presented"])
    sets = alt_data.get("sets") or {}
    adv, prog = (sets.get("adv") or {}).get("system") or {}, (sets.get("prog") or {}).get("system") or {}
    if adv:
        out["injection"] = (adv.get("injection_withheld_claims"), adv["by_subgroup"]["injection"]["n"])
    if prog:
        out["swap"] = (prog["by_subgroup"]["swap"]["correct"], prog["by_subgroup"]["swap"]["n"])
    out["repeat"] = alt_data.get("repeatability")
    retro = [r for r in results.best_results(list(runs.values()), "three_class_macro_f1") if r.get("set") == "retro"]
    out["best_three"] = retro[0] if retro else None
    out["default_three"] = runs.get("runs/retro_arm_c_studio")
    return out


def headline_metrics(h: Dict[str, Any]) -> None:
    cards = []
    study = h["study"]
    if study:
        cards.append(dict(label="Arm C macro-F1", value=f"{study['arms']['C']['macro_f1']:.4f}",
                          delta=f"{study['primary']['delta_macro_f1']:+.4f} vs rank baseline",
                          help=f"Two-class macro-F1 on {study['dataset']['labelled_claims']} SciFact claims "
                               f"(agentic verification study, {study['run_date']})."))
    if h.get("false_support"):
        before, after, n = h["false_support"]
        cards.append(dict(label="False support", value=f"{after}", delta=f"{(after - before) / before:+.0%} ({before} → {after})",
                          delta_color="inverse", help=f"Unsupported claims wrongly confirmed, corrected judge vs default, "
                                                      f"on {n} held-out SciFact claims."))
    if h.get("coverage"):
        matched, shown, old = h["coverage"]
        cards.append(dict(label="Passage verdict coverage", value=f"{matched / shown:.0%}",
                          delta=f"+{(matched / shown - old) * 100:.0f} pts with the ID fix",
                          help=f"{matched} of {shown} passages received a matched verdict in a real-model check, against "
                               f"{old:.0%} before the fix."))
    if h.get("injection"):
        withheld, n = h["injection"]
        cards.append(dict(label="Injections withheld", value=f"{withheld}/{n}",
                          help="Passages carrying an injected instruction were recognised and withheld."))
    metric_row(cards)


def metric_row(cards: List[Dict[str, Any]]) -> None:
    """Metric cards in one aligned row."""
    if not cards:
        return
    for col, card in zip(st.columns(len(cards)), cards):
        with col:
            st.metric(border=True, height="stretch", **card)


def overview() -> None:
    h = highlights()
    st.markdown("<div class='eyebrow'>M.Tech research project · open source</div>", unsafe_allow_html=True)
    st.title("Multi-Agent RAG Research Studio")
    st.markdown("#### Research answers you can check: every statement verified against the evidence it cites")
    st.write("A team of cooperating agents plans the search, reads scholarly sources, ranks the evidence, drafts cited "
             "statements and then verifies each one, passage by passage. Only statements the evidence supports become "
             "findings, with the quotes, sources and conflicts in plain view.")
    with st.container(horizontal=True):
        link("results", label="See the results", icon=":material/insights:")
        link("install", label="Run it yourself", icon=":material/download:")
        st.page_link(f"{content.REPO_URL}/tree/{content.BRANCH}", label="Source on GitHub", icon=":material/code:")
    headline_metrics(h)

    st.subheader("How it works")
    cols = st.columns(len(PIPELINE))
    for col, (icon, title, text) in zip(cols, PIPELINE):
        with col.container(border=True, height="stretch"):
            st.markdown(f"**{icon} {title}**")
            st.caption(text)

    st.subheader("Why it stands out")
    cols = st.columns(3)
    with cols[0].container(border=True, height="stretch"):
        st.markdown("**:material/fact_check: Evidence first**")
        st.write("Every decisive judgement carries an exact quote from its passage, and nothing is reported as a finding "
                 "unless a passage supports it. Retrieved text that tries to give the model instructions is withheld.")
    with cols[1].container(border=True, height="stretch"):
        st.markdown("**:material/psychology: Agentic verification**")
        st.write("The default verifier is **Arm C agentic verification, in a Studio-adapted variant** (`arm_c_studio`): "
                 "a verifier, a planner that searches for missing evidence and an adjudicator for conflicts. Without a "
                 "passage for or against, the statement stays INSUFFICIENT instead of guessing.")
    with cols[2].container(border=True, height="stretch"):
        st.markdown("**:material/science: Research-grade evaluation**")
        st.write(f"Plans frozen before every run, blind confirmation sets, record provenance with run IDs and duplicate "
                 f"checks, and {h['tests']['python_functions']:,} automated test functions guarding the code.")

    st.subheader("Explore")
    cards = [("results", ":material/insights:", "Results", "Headline numbers, robustness and milestones."),
             ("evaluations", ":material/monitoring:", "Evaluation runs", "Every run with its quality, cost and provenance."),
             ("architecture", ":material/hub:", "Architecture", "The real components and their dependencies."),
             ("workflow", ":material/route:", "Research workflow", "A question's path through the pipeline."),
             ("components", ":material/widgets:", "Components", "Every module: purpose, files and tests."),
             ("install", ":material/download:", "Install and run", "Set it up on your own Windows computer.")]
    cols = st.columns(3)
    for index, (key, icon, title, text) in enumerate(cards):
        with cols[index % 3].container(border=True, height="stretch"):
            link(key, title, icon)
            st.caption(text)
    st.caption("Benchmarks use SciFact scientific claims with their cited abstracts; measuring accuracy on live web "
               f"evidence is the next milestone. Source: [{content.REPO}]({content.REPO_URL}/tree/{content.BRANCH}) · "
               "MIT licence for the project's own code.")


# ================================================================================================= 2. architecture
def architecture_page() -> None:
    header("Architecture explorer", "The system as the code builds it. Solid arrows are Python imports found in the "
                                    "source now; dashed arrows are runtime connections (HTTP, processes, files), each "
                                    "backed by a marker in a named file.")
    table = architecture.nodes()
    with st.container(horizontal=True, vertical_alignment="bottom"):
        view = st.segmented_control("View", list(content.VIEWS), default="Research execution", required=True)
        show_code = st.toggle("Code dependencies", value=True)
        show_runtime = st.toggle("Runtime connections", value=True)
    members = content.VIEWS[view] or list(table)
    selectable = [c["id"] for c in content.COMPONENTS if c["id"] in members]
    default = st.session_state.get("component") if st.session_state.get("component") in selectable else "orchestrator" \
        if "orchestrator" in selectable else selectable[0]
    selected = st.selectbox("Inspect component", selectable, index=selectable.index(default),
                            format_func=lambda c: f"{table[c]['name']}  ·  {table[c]['layer']}")
    st.session_state["component"] = selected

    with st.container(border=True):
        st.graphviz_chart(architecture.dot(content.VIEWS[view], selected, show_code, show_runtime), width="stretch")
        st.markdown(" &nbsp; ".join(f"<span style='color:{color}'>&#9632;</span> {layer}" for layer, color in
                                     architecture.LAYER_COLOR.items()), unsafe_allow_html=True)
        st.caption(f"{kind_badge('code')} Arrows point from the importing component to the one it uses; dashed "
                   "arrows are runtime connections. Highlighted: the selected component and its direct connections. "
                   "Use the diagram's fullscreen button to enlarge it.")
    with st.container(border=True):
        component_card(selected, table)


def component_card(component_id: str, table: Dict[str, dict]) -> None:
    c = table[component_id]
    st.markdown(f"### {c['name']}")
    st.markdown(status_badges(c["status"]) + f"  :gray-badge[{c['layer']}]")
    st.write(c["purpose"])
    left, right = st.columns(2, gap="large")
    with left:
        component_facts(component_id, table, c)
    with right:
        component_evidence(component_id, table)


def component_facts(component_id: str, table: Dict[str, dict], c: dict) -> None:
    st.markdown(f"**Inputs** — {c['inputs']}  \n**Outputs** — {c['outputs']}")
    st.markdown("**Source files** — " + ", ".join(src(p) for p in c["files"]))
    links = architecture.connections(component_id)
    if links["uses"]:
        st.markdown("**Uses (imports)** — " + ", ".join(table[b]["name"] for b, _ in links["uses"]))
    if links["used_by"]:
        st.markdown("**Used by** — " + ", ".join(table[a]["name"] for a, _ in links["used_by"]))
    runtime = [f"{label} → {table[b]['name']}" for b, label, _ in links["runtime_out"]] + \
              [f"{table[a]['name']} → {label}" for a, label, _ in links["runtime_in"]]
    if runtime:
        st.markdown("**Runtime** — " + "; ".join(runtime))
    if c["tests"]:
        st.markdown("**Tests** — " + ", ".join(src(t, t.split('/')[-1]) for t in c["tests"]))
    else:
        st.markdown("**Tests** — no public test file targets this component directly.")
    st.markdown(f"**Limits** — {c['limits']}")


def component_evidence(component_id: str, table: Dict[str, dict]) -> None:
    links = architecture.connections(component_id)
    st.markdown("**Evidence for each connection** — found in the source when this page loaded")
    rows = [(table[b]["name"], e) for b, ev in links["uses"] for e in ev]
    rows += [(table[a]["name"] + " (incoming)", e) for a, ev in links["used_by"] for e in ev]
    rows += [(table[b]["name"], ev) for b, _, ev in links["runtime_out"]]
    rows += [(table[a]["name"] + " (incoming)", ev) for a, _, ev in links["runtime_in"]]
    if rows:
        st.dataframe(pd.DataFrame(rows, columns=["Connected to", "Evidence in the source"]), hide_index=True,
                     width="stretch", height=min(420, 38 + 35 * len(rows)))
    else:
        st.caption("No connections in the code.")


# ================================================================================================= 3. workflow
def workflow() -> None:
    header("Research workflow", "How a question travels through the Studio's pipeline "
                                "(webrag/orchestrator.py and webrag/deep.py).")
    mode = st.segmented_control("Show", ["How a run works", "A recorded run"], default="How a run works",
                                required=True)
    if mode == "How a run works":
        workflow_general()
    else:
        workflow_recorded()


def workflow_general() -> None:
    st.markdown(kind_badge("illustration") + " The stages below are the ones a deep research run records, in order.")
    names = [s["stage"] for s in content.WORKFLOW]
    step = st.select_slider("Stage", names, value=names[0], label_visibility="collapsed")
    stage = content.WORKFLOW[names.index(step)]
    component = next(c for c in content.COMPONENTS if c["id"] == stage["component"])
    with st.container(border=True):
        st.markdown(f"### {names.index(step) + 1}. {stage['stage'].capitalize()}")
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"**Input**  \n{stage['input']}")
        c2.markdown(f"**What happens**  \n{stage['does']}")
        c3.markdown(f"**Output**  \n{stage['output']}")
        st.markdown(f"**Component** — {component['name']} ({', '.join(src(p) for p in component['files'][:2])})")
        if stage["stops"]:
            st.markdown(f"**When it stops or fails** — {stage['stops']}")
    st.dataframe(pd.DataFrame([{"#": i + 1, "Stage": s["stage"], "Output": s["output"]}
                               for i, s in enumerate(content.WORKFLOW)]), hide_index=True, width="stretch")
    with st.container(border=True):
        st.markdown("**Why a run stops** — every deep run records one of these reasons "
                    f"({src('webrag/deep.py')}):")
        st.dataframe(pd.DataFrame([{"Reason": k, "Meaning": v} for k, v in content.STOP_REASONS.items()]),
                     hide_index=True, width="stretch")
    st.markdown("**Verdict rule** — a statement is SUPPORTED when at least one passage supports it and none "
                "contradicts it; CONTRADICTED in the reverse case; INSUFFICIENT otherwise, including when passages "
                f"disagree ({src('docqa/verify.py')}).")


def workflow_recorded() -> None:
    runs = results.examples()
    if not runs:
        st.info("No recorded example is bundled with this copy of the showcase.", icon=":material/inbox:")
        return
    labels = {i: f"{r['input'][:80]} · {r['mode']}" for i, r in enumerate(runs)}
    pick = st.selectbox("Recorded run", list(labels), format_func=labels.get)
    run = runs[pick]
    verifier = run.get("verifier") or (run.get("settings") or {}).get("verifier") or "not recorded"
    st.warning(f"{evidence_badge('demonstration')} This is a run recorded on {str(run.get('started'))[:10]} on the author's "
               "computer and replayed from a file. Nothing is running now. Third-party text is not included: "
               "open each source through its link. "
               f"Verifier used for this run: `{verifier}`"
               + (" (the earlier default; the Studio now defaults to Arm C, `arm_c_studio`)."
                  if not str(verifier).startswith("arm_c") else "."), icon=":material/history:")
    show_run(run)


def show_run(run: Dict[str, Any]) -> None:
    claims = run.get("claims") or []
    sources = {s["id"]: s for s in run.get("sources") or []}
    with st.container(border=True):
        st.markdown(f"### {esc(run['input'])}")
        st.markdown(f":blue-badge[{'Claim verification' if run['mode'] == 'verify' else 'Research question'}] "
                    f":gray-badge[status {run.get('status')}] :gray-badge[{str(run.get('outcome')).replace('_', ' ').lower()}] "
                    f":violet-badge[{run.get('depth') or 'single pass'} depth]")
        model = run.get("model") or {}
        st.caption(f"Model {model.get('model')} on {model.get('runtime')} · verifier {run.get('verifier')} · "
                   f"{len(sources)} sources · {run.get('search_errors', 0)} failed provider searches (recorded)")
    with st.container(horizontal=True):
        counts = Counter(c.get("label") for c in claims)
        st.metric("Statements", len(claims), border=True)
        st.metric("Supported", counts.get("SUPPORTED", 0), border=True)
        st.metric("Not supported", len(claims) - counts.get("SUPPORTED", 0), border=True)
        quality = run.get("source_quality") or {}
        st.metric("Read in full", (quality.get("by_access") or {}).get("FULL_TEXT", 0), border=True,
                  help="Sources whose full text was read; the others were read as abstracts.")
    tabs = st.tabs(["Stages", "Plan", "Findings", "Sources", "Coverage and stopping", "Limitations"])
    with tabs[0]:
        st.dataframe(pd.DataFrame([{"Stage": s["name"], "Result": s["status"], "Seconds": s["seconds"],
                                    "Detail": s["detail"]} for s in run.get("stages") or []]),
                     hide_index=True, width="stretch")
    with tabs[1]:
        plan = run.get("plan") or {}
        for title, key in (("Subquestions", "subquestions"), ("Queries", "queries"),
                           ("Counter-evidence queries", "counter_queries")):
            if plan.get(key):
                st.markdown(f"**{title}** (model-written)")
                st.markdown("\n".join(f"- {esc(q)}" for q in plan[key]))
        if not any(plan.values()):
            st.caption("No plan recorded.")
    with tabs[2]:
        summary = run.get("summary") or {}
        if summary.get("direct_answer"):
            with st.container(border=True):
                st.markdown(f"**Direct answer** (model-written from verified findings)  \n{esc(summary['direct_answer'])}")
        for claim in claims:
            stances = Counter(p.get("stance") for p in claim.get("passages") or [])
            with st.expander(f":{VERDICT.get(claim.get('label'), 'gray')}-badge[{claim.get('label')}] "
                             f"{esc(claim.get('text'))}"):
                st.caption("Statement written by the model; verdict recorded by the verifier.")
                st.markdown(" ".join(f":{STANCE.get(k, 'gray')}-badge[{k} × {v}]" for k, v in stances.items()))
                support = claim.get("supporting_sources") or []
                if support:
                    st.markdown("**Supporting sources** — " + ", ".join(
                        f"[{esc(sources.get(s, {}).get('title') or s)}]({sources.get(s, {}).get('url')})"
                        if sources.get(s, {}).get("url") else esc(s) for s in support))
                for caution in claim.get("cautions") or []:
                    st.caption(f"Caution ({str(caution.get('kind')).replace('_', ' ')}): {esc(caution.get('detail'))}")
    with tabs[3]:
        frame = pd.DataFrame([{"ID": s["id"], "Title": s.get("title"), "Provider": s.get("provider"),
                               "Access": str(s.get("access") or "").replace("_", " ").lower(),
                               "Published": str(s.get("published") or "")[:10], "Link": s.get("url")}
                              for s in sources.values()])
        st.dataframe(frame, hide_index=True, width="stretch",
                     column_config={"Link": st.column_config.LinkColumn(display_text="open source"),
                                    "Title": st.column_config.TextColumn(width="large")})
        st.caption("Bibliographic metadata as recorded; texts belong to their authors and publishers.")
    with tabs[4]:
        if run.get("coverage"):
            st.dataframe(pd.DataFrame(run["coverage"]), hide_index=True, width="stretch")
        stopping = run.get("stopping") or {}
        if stopping.get("reason"):
            st.markdown(f"**Why searching stopped** — `{stopping['reason']}` ({stopping.get('completeness')}): "
                        f"{esc(stopping.get('detail'))}")
    with tabs[5]:
        st.markdown("\n".join(f"- {esc(x)}" for x in run.get("limitations") or []) or "None recorded.")
        st.caption("Removed before publishing: " + "; ".join(run.get("removed") or []))


# ================================================================================================= 4. components
def components() -> None:
    header("Component laboratory", "Every mapped component with its purpose, source files, tests, status and limits.")
    with st.container(horizontal=True, vertical_alignment="bottom"):
        query = st.text_input("Search", placeholder="Search components", label_visibility="collapsed",
                              icon=":material/search:")
        layers = st.multiselect("Layer", [l for l in content.LAYERS if l != "External"], placeholder="Any layer",
                                label_visibility="collapsed")
        wanted = st.multiselect("Status", list(content.STATUS), format_func=lambda s: content.STATUS[s][0],
                                placeholder="Any status", label_visibility="collapsed")
    with st.expander("What the status labels mean"):
        for key, (label, color, meaning) in content.STATUS.items():
            st.markdown(f":{color}-badge[{label}] {meaning}")
    rows = [c for c in content.COMPONENTS
            if (not query or query.lower() in (c["name"] + c["purpose"] + " ".join(c["files"])).lower())
            and (not layers or c["layer"] in layers) and (not wanted or set(wanted) & set(c["status"]))]
    st.caption(f"{len(rows)} of {len(content.COMPONENTS)} components")
    table = architecture.nodes()
    cols = st.columns(2)
    for index, c in enumerate(rows):
        with cols[index % 2].container(border=True):
            st.markdown(f"**{c['name']}** :gray-badge[{c['layer']}]")
            st.markdown(status_badges(c["status"]))
            st.write(c["purpose"])
            st.markdown("Files: " + ", ".join(src(p, p.split('/')[-1]) for p in c["files"]))
            links = architecture.connections(c["id"])
            if links["uses"]:
                st.caption("Uses: " + ", ".join(table[b]["name"] for b, _ in links["uses"]))
            st.caption(f"Tests: {', '.join(t.split('/')[-1] for t in c['tests']) or 'none targeted'} · Limits: {c['limits']}")
    if not rows:
        st.info("No component matches these filters.")


# ================================================================================================= 5. results
def fraction(count: Dict[str, int]) -> str:
    return f"{count['count']}/{count['denominator']}"


def ratio(text: str) -> str:
    """'a/b' with its percentage."""
    a, b = (int(x) for x in text.split("/"))
    return f"{text} ({a / b:.0%})" if b else text


MILESTONES = [
    ("Oct 23", "Agentic verification study", "Arm C leads the single-pass rank baseline on SciFact (two-class)."),
    ("Oct 24", "Proxy benchmark suite", "Retrospective, programmatic and adversarial SciFact sets; held-out test."),
    ("Oct 25", "Provenance and integrity", "Run IDs, record fingerprints, duplicate policy and guarded runs."),
    ("Oct 26", "Blind confirmation", "Corrected judge cuts false support on 249 untouched claims."),
    ("Oct 26", "Passage-ID fix", "Every passage now gets its own matched verdict, confirmed on the real model."),
]


def results_page() -> None:
    header("Results", "Benchmarks, robustness and milestones of the Research Studio's verifier. Every figure is read "
                      "from the project's result files when this page loads.")
    h = highlights()
    headline_metrics(h)
    cards = []
    best = h.get("best_three")
    if best:
        base = h.get("default_three")
        cards.append(dict(label="Best three-class macro-F1", value=f"{best['metrics']['three_class_macro_f1']:.4f}",
                          delta=(f"{best['metrics']['three_class_macro_f1'] - base['metrics']['three_class_macro_f1']:+.4f} "
                                 "vs default") if base and base.get("metrics") else None,
                          help=f"{best['label']} on the {best['set_size']}-claim SciFact development set (supported, "
                               "contradicted and not-enough-information claims)."))
    if h.get("swap"):
        cards.append(dict(label="Unrelated evidence caught", value=f"{h['swap'][0]}/{h['swap'][1]}",
                          help="Claims paired with unrelated abstracts were correctly left undecided."))
    if h.get("repeat"):
        cards.append(dict(label="Repeatable verdicts", value=f"{h['repeat']['identical_verdicts']}/{h['repeat']['claims']}",
                          help="Identical verdicts when the same claims were run again."))
    cards.append(dict(label="Automated test functions", value=f"{h['tests']['python_functions']:,}",
                      help=f"Counted now in {h['tests']['python_files']} Python test files, plus {h['tests']['js_files']} "
                           "interface test files."))
    metric_row(cards)

    left, right = st.columns(2)
    with left.container(border=True, height="stretch"):
        st.markdown("**Macro-F1 by verifier** · 300-claim SciFact development set")
        rows = [r for r in results.visible_runs(results.evaluation_runs()) if results.rankable(r) and r.get("set") == "retro"]
        if rows:
            frame = pd.DataFrame([{"verifier": r["verifier"], "metric": name, "macro_f1": r["metrics"][key]}
                                  for r in rows for key, name in (("two_class_macro_f1", "Two-class"),
                                                                  ("three_class_macro_f1", "Three-class"))])
            order = [r["verifier"] for r in sorted(rows, key=lambda r: -r["metrics"]["three_class_macro_f1"])]
            chart = alt.Chart(frame).mark_bar(cornerRadiusEnd=3).encode(
                y=alt.Y("verifier:N", sort=order, title=None), x=alt.X("macro_f1:Q", title="macro-F1",
                                                                       scale=alt.Scale(domain=[0, 1])),
                yOffset="metric:N", color=alt.Color("metric:N", scale=alt.Scale(range=["#4aa3df", ACCENT]), title=None,
                                                    legend=alt.Legend(orient="bottom")),
                tooltip=["verifier", "metric", alt.Tooltip("macro_f1:Q", format=".4f")]).properties(height=300)
            st.altair_chart(chart, width="stretch")
            st.caption("Two-class counts supported and contradicted claims; three-class also scores "
                       "not-enough-information claims, the stricter test.")
        else:
            st.info("No development runs are bundled.")
    with right.container(border=True, height="stretch"):
        study = h["study"]
        st.markdown("**Agentic verification study** · two-class macro-F1 by arm")
        if study:
            arms = study["arms"]
            frame = pd.DataFrame([{"arm": a, "macro_f1": arms[a]["macro_f1"], "highlight": a == "C"}
                                  for a in ("B1", "B1R", "C0", "C")])
            chart = alt.Chart(frame).mark_bar(cornerRadiusEnd=3).encode(
                x=alt.X("arm:N", sort=["B1", "B1R", "C0", "C"], title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("macro_f1:Q", title="macro-F1", scale=alt.Scale(domain=[0, 1])),
                color=alt.condition("datum.highlight", alt.value(ACCENT), alt.value("#4a5a62")),
                tooltip=["arm", alt.Tooltip("macro_f1:Q", format=".4f")]).properties(height=300)
            st.altair_chart(chart, width="stretch")
            st.caption(f"{study['dataset']['labelled_claims']} SciFact DEV claims · {study['model']['name']} · run "
                       f"{study['run_date']}. B1R: single-pass rank baseline; C0: Arm C without search; C: full Arm C.")
        else:
            st.info("The study's results file is not bundled.")

    st.subheader("Milestones")
    cols = st.columns(len(MILESTONES))
    for col, (date, title, text) in zip(cols, MILESTONES):
        with col.container(border=True, height="stretch"):
            st.caption(date)
            st.markdown(f"**{title}**")
            st.caption(text)

    with st.expander("How these results were measured"):
        st.markdown(
            "- **Data.** SciFact scientific claims labelled by human annotators, each checked against its cited "
            "abstracts. The Studio's accuracy on live web evidence is the next milestone.\n"
            "- **Two metrics.** *Two-class* macro-F1 averages the supported and contradicted classes (the agentic "
            "verification study's metric). *Three-class* macro-F1 also scores not-enough-information claims; it is the "
            "project's protocol metric, with a target of 0.75 on held-out claims as the next milestone. The two are "
            "never ranked against each other; see " + src("evaluation/studio_eval/TWO_VS_THREE_CLASS.md") + ".\n"
            "- **Discipline.** Plans and decision rules are frozen before each run, confirmation sets are used once, and "
            "every run carries a run ID and passes a duplicate check: " + src("evaluation/studio_eval/ALT_BENCHMARKS.md")
            + ", " + src("evaluation/studio_eval/PROTOCOL.md") + ".\n"
            "- **Records.** Figures come from aggregate files in " + src("showcase/data/") + ". Per-claim records stay "
            "private while the SciFact licence terms are clarified.")
    link("evaluations", label="Open every evaluation run",
                 icon=":material/arrow_forward:")


# ================================================================================================= evaluation runs
RUN_STATUS = {
    "valid": "Complete run that passed the record-integrity check",
    "rejected": "Failed the record-integrity check (duplicate records)",
    "invalid": "Infrastructure failure, not model behaviour",
    "incomplete": "Stopped before its claim set was covered",
    "partial": "Smoke or repeatability subset, not scored",
    "coverage": "Coverage check; accuracy not computed",
}
MISSING = "not recorded"
SET_NAMES = {"retro": "Development set", "dev_sub90": "Development subset", "test_dev": "SciFact DEV split",
             "confirm": "Blind confirmation", "prog": "Programmatic cases", "adv": "Adversarial cases"}


def set_name(run: Dict[str, Any]) -> str:
    return SET_NAMES.get(str(run.get("set")), str(run.get("set") or "?"))


def _num(value: Any, spec: str = ".4f") -> str:
    return MISSING if value is None else format(value, spec)


def _ci(ci: Any) -> str:
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "—"


def _date(run: Dict[str, Any]) -> str:
    return str(run.get("last_invocation_utc") or run.get("started_utc") or MISSING)[:16].replace("T", " ")


def _run_name(run: Dict[str, Any]) -> str:
    return f"{run['label']} · {set_name(run)} · {run['role']}"


def _metric(run: Dict[str, Any], key: str) -> str:
    return _num((run.get("metrics") or {}).get(key)) if run.get("metrics") else "—"


def evaluations() -> None:
    header("Evaluation runs", "Every recorded run of the Studio verifier and its candidates on SciFact: quality, cost, "
                              "reliability, evidence coverage and provenance.")
    registry = results.evaluation_runs()
    runs = results.visible_runs(registry)
    if not runs:
        st.info("No evaluation run is bundled with this copy of the showcase.", icon=":material/inbox:")
        return
    ranked = [r for r in runs if results.rankable(r)]
    latest = results.recent_runs(runs)[0]
    metric_row([
        dict(label="Valid runs", value=f"{len(ranked)}", help=f"{len(runs)} runs recorded in total."),
        dict(label="Claims scored", value=f"{sum(r['set_size'] or 0 for r in ranked):,}",
             help="Claims across all valid runs."),
        dict(label="Latest run", value=_date(latest)[:10], help=_run_name(latest)),
        dict(label="Three-class target", value=f"{registry['target']['value']}",
             help="Next milestone, on held-out claims. " + registry["target"]["note"])])
    tabs = st.tabs([":material/emoji_events: Best results", ":material/history: All runs",
                    ":material/target: Quality", ":material/speed: Operations",
                    ":material/fingerprint: Coverage and provenance"])
    with tabs[0]:
        best_valid_results(runs)
    with tabs[1]:
        recent_results(runs)
    with tabs[2]:
        predictive_quality(runs, registry)
    with tabs[3]:
        operations(runs, registry)
    with tabs[4]:
        evidence_coverage(runs)
        run_provenance(runs, registry)


def best_valid_results(runs: List[Dict[str, Any]]) -> None:
    st.caption("The best valid run per claim set, for each metric. Runs are compared only on the same claim set, and "
               "two-class and three-class scores are never ranked together.")
    for metric, label in results.METRICS.items():
        best = results.best_results(runs, metric)
        st.markdown(f"**{label}**")
        if not best:
            st.info("No valid run has this metric.")
            continue
        ci_key = metric.replace("macro_f1", "ci95")
        st.dataframe(pd.DataFrame([{
            "Claim set": set_name(r), "Claims": r["metrics"]["two_class_claims"] if metric.startswith("two") else r["set_size"],
            "Best run": r["label"], "Role": r["role"], "Macro-F1": _num(r["metrics"][metric]),
            "95% CI": _ci(r["metrics"].get(ci_key)), "Date (UTC)": _date(r)} for r in best]),
            hide_index=True, width="stretch")


def recent_results(runs: List[Dict[str, Any]]) -> None:
    ordered = results.recent_runs(runs)
    cols = st.columns(3)
    phases = cols[0].multiselect("Phase", sorted({r["phase"] for r in ordered}))
    sets = cols[1].multiselect("Claim set", sorted({set_name(r) for r in ordered}))
    statuses = cols[2].multiselect("Status", list(RUN_STATUS))
    shown = [r for r in ordered if (not phases or r["phase"] in phases) and (not sets or set_name(r) in sets)
             and (not statuses or r["status"] in statuses)]
    if not shown:
        st.info("No run matches these filters.")
        return
    st.dataframe(pd.DataFrame([{
        "Date (UTC)": _date(r), "Run": r["label"], "Phase": r["phase"], "Claim set": set_name(r), "Status": r["status"],
        "Three-class": _metric(r, "three_class_macro_f1"), "Two-class": _metric(r, "two_class_macro_f1"),
        "Note": r["note"]} for r in shown]), hide_index=True, width="stretch")
    st.caption("Newest first. Status: " + " · ".join(f"**{k}** {v}" for k, v in RUN_STATUS.items()) + ".")


def predictive_quality(runs: List[Dict[str, Any]], registry: Dict[str, Any]) -> None:
    ranked = [r for r in results.recent_runs(runs) if results.rankable(r)]
    if not ranked:
        st.info("No valid run has metrics.")
        return
    by_key = {r["key"]: r for r in ranked}
    run = by_key[st.selectbox("Run", list(by_key), format_func=lambda k: _run_name(by_key[k]), key="quality_run")]
    m = run["metrics"]
    metric_row([
        dict(label="Three-class macro-F1", value=_num(m["three_class_macro_f1"]),
             help=f"95% interval {_ci(m['three_class_ci95'])}; {run['set_size']} claims."),
        dict(label="Two-class macro-F1", value=_num(m["two_class_macro_f1"]),
             help=f"{m['two_class_claims']} supported and contradicted claims."),
        dict(label="Accuracy", value=_num(m["accuracy"], ".1%")),
        dict(label="False support", value=ratio(m["false_support"]),
             help="Contradicted or not-enough-information claims that were called SUPPORTED.")])
    left, right = st.columns(2)
    with left:
        st.markdown("**Per class**")
        st.dataframe(pd.DataFrame([{"Class": c, "Precision": _num(v["precision"]), "Recall": _num(v["recall"]),
                                    "F1": _num(v["f1"]), "Claims": v["support"]} for c, v in m["per_class"].items()]),
                     hide_index=True, width="stretch")
    with right:
        st.markdown("**Confusion matrix** (rows: label; columns: verdict)")
        st.dataframe(pd.DataFrame(m["confusion"]).T, width="stretch")
    names = {r["key"]: r for r in runs}
    rows = [{"System": names[p["system"]]["label"], "Baseline": names[p["baseline"]]["label"],
             "Claim set": set_name(names[p["system"]]), "Δ three-class": _num(p["delta_three_class_macro_f1"], "+.4f"),
             "Paired 95% CI": _ci(p["ci95"]), "Permutation p": _num(p["permutation_p"])}
            for p in registry.get("paired") or [] if p["system"] in names and p["baseline"] in names]
    if rows:
        st.markdown("**Paired comparisons** (same claims)")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _baseline_of(key: str, registry: Dict[str, Any]) -> Any:
    return next((p["baseline"] for p in registry.get("paired") or [] if p["system"] == key), None)


def operations(runs: List[Dict[str, Any]], registry: Dict[str, Any]) -> None:
    by_key = {r["key"]: r for r in runs}
    rows = []
    for r in results.recent_runs(runs):
        e, x = r["efficiency"], r["reliability"]
        base = by_key.get(_baseline_of(r["key"], registry) or "")
        relative = (e["seconds_per_claim_median"] / base["efficiency"]["seconds_per_claim_median"]
                    if base and e["seconds_per_claim_median"] and base["efficiency"]["seconds_per_claim_median"] else None)
        rows.append({"Run": _run_name(r), "Median s/claim": _num(e["seconds_per_claim_median"], ".1f"),
                     "vs baseline": f"{relative:.2f}×" if relative else "—",
                     "Model calls/claim": _num(e["calls_per_claim_mean"], ".2f"),
                     "Claims attempted": f"{x['claims_attempted']} / {r['set_size'] or '?'}",
                     "Infrastructure failures": str(x["infrastructure_failures"]),
                     "Model-output failures": str(x["model_output_failures"]), "Timeouts": str(x["timeouts"]),
                     "Resumes": str(max(0, (r["provenance"]["invocations"] or 1) - 1)),
                     "Memory stops": MISSING if x["memory_stops"] is None else str(x["memory_stops"])})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Wall-clock seconds per claim on a local qwen3:4b (median, robust to the occasional stalled claim). "
               "Infrastructure failures (server unreachable, out of memory) are separated from failures of the model's "
               f"own output; {registry['scorer']['failures']}. Token counts and GPU use were not recorded.")


def evidence_coverage(runs: List[Dict[str, Any]]) -> None:
    measured = [r for r in results.recent_runs(runs) if (r.get("coverage") or {}).get("passages_presented") is not None]
    st.markdown("**Passage verdict coverage**")
    if not measured:
        st.info("No run recorded passage diagnostics.")
        return
    st.dataframe(pd.DataFrame([{
        "Run": _run_name(r), "Passages": c["passages_presented"], "Matched verdicts": c["valid_matched"],
        "Coverage": f"{c['valid_matched'] / c['passages_presented']:.0%}" if c["passages_presented"] else "—",
        "Read as NEUTRAL without a verdict": c["silent_neutral"],
        "Rejected replies": ", ".join(f"{k} {v}" for k, v in c["rejected_by_code"].items()) or "none"}
        for r in measured for c in [r["coverage"]]]), hide_index=True, width="stretch")
    st.caption("Whether each passage shown to the verifier received its own matched verdict; coverage is not "
               "correctness. Measured with " + src("evaluation/studio_eval/id_coverage.py") + ".")


def run_provenance(runs: List[Dict[str, Any]], registry: Dict[str, Any]) -> None:
    st.markdown("**Provenance**")
    rows = []
    for r in results.recent_runs(runs):
        p = r["provenance"]
        rows.append({"Run": _run_name(r), "Run ID": (p["run_id"] or MISSING)[:12],
                     "Claim set SHA-256": (r["set_sha256"] or MISSING)[:12],
                     "Source revision": (p["git_head"] or MISSING)[:7]
                     + {True: " + local changes", False: ""}.get(p["working_tree_dirty"], ""),
                     "Model": p["model"] or MISSING, "Model digest": p["model_digest"] or MISSING,
                     "Ollama": p["ollama_version"] or MISSING, "Record integrity": p["record_integrity"]["status"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Built offline by " + src("showcase/make_evaluation_runs.py") + " into "
               + src("showcase/data/evaluation_runs.json") + ". Fields a run never recorded are shown as not recorded.")


# ================================================================================================= 6. install
def install() -> None:
    header("Install and run", "Set up the full Research Studio on your own computer. This website is only a guide; "
                              "research runs happen on your machine.")
    st.markdown(f"{kind_badge('code')} Commands and setting names below are taken from the project's files "
                f"({src('README.md')}, {src('start_studio.bat')}, {src('dashboard/launch.py')}, "
                f"{src('webrag/model_service.py')}, {src('.env.example')}).")
    with st.container(border=True):
        st.markdown("**Supported platform**")
        st.markdown("- **Windows 10/11, 64-bit, Python 3.12** — the tested setup; `requirements-lock.txt` pins every "
                    "package for it.\n- Linux and macOS: not tested. `requirements.txt` and `python -m dashboard.launch` "
                    "may work, but the lock file and `start_studio.bat` are Windows-only.\n"
                    "- About 4 GB of disk for the Python packages, Ollama's `qwen3:4b` (2.5 GB) and the retrieval models.")
    steps = [
        ("1 · Get the code", "required",
         "git clone --branch release/public-candidate-2 https://github.com/JeetMakadiya9/multi-agent-rag-research.git\n"
         "cd multi-agent-rag-research", "Downloads the repository on the branch that contains the application."),
        ("2 · Create a virtual environment", "required", "py -3.12 -m venv .venv",
         "Keeps the project's packages separate from the rest of your system. Activation is optional: the commands "
         "below call .venv\\Scripts\\python.exe directly."),
        ("3 · Install the packages", "required", ".venv\\Scripts\\python.exe -m pip install -r requirements.txt",
         "Installs the Studio, the retrieval stack (CPU) and the dashboard. For exactly the tested versions use "
         "requirements-lock.txt instead."),
        ("4 · Install Ollama and the model", "required for research",
         "ollama pull qwen3:4b", "Install Ollama from https://ollama.com first. The model runs on your computer; "
         "without it the Studio still opens and shows saved reports."),
        ("5 · Configure (optional)", "optional", "Copy-Item .env.example .env\nnotepad .env",
         "Only needed to change the model or add provider keys. .env is ignored by git; never commit it."),
        ("6 · Start", "required", ".\\start_studio.bat",
         "Checks the setup, starts the Studio backend and the dashboard, and opens the dashboard. Stop with Ctrl+C."),
    ]
    for title, need, command, why in steps:
        with st.container(border=True):
            st.markdown(f"**{title}** :{'green' if need == 'required' else 'gray'}-badge[{need}]")
            st.code(command, language="powershell")
            st.caption(why)

    with st.container(border=True):
        st.markdown("**First success**")
        st.markdown("1. The launcher prints `Dashboard: http://127.0.0.1:8501` and `Research Studio: http://127.0.0.1:8765`.\n"
                    "2. The dashboard's sidebar shows **Studio online** and **Model ready**.\n"
                    "3. In **Research Workspace**, verify a short claim at *Quick* depth; a run takes a few minutes.\n"
                    "4. Open it in **Evidence Explorer** — verdict, passages with stances, and source links.")
        st.markdown(f"Launcher options: {', '.join(f'`{o}`' for o in content.LAUNCHER_OPTIONS)} — for example "
                    "`.\\start_studio.bat --check` only checks the setup.")
    with st.container(border=True):
        st.markdown("**Settings (.env)** — all optional; without them the Studio uses the local Ollama with qwen3:4b.")
        st.dataframe(pd.DataFrame(content.ENV_SETTINGS, columns=["Setting", "Meaning", "Needed"]), hide_index=True,
                     width="stretch")
        st.caption("A remote model service receives your questions and the retrieved passages, and may charge per "
                   "use. Its key stays in the local Studio process. If the configured model is unavailable, a run "
                   "stops with the reason; nothing is substituted.")
    with st.container(border=True):
        st.markdown("**Troubleshooting**")
        st.dataframe(pd.DataFrame([
            ("No virtual environment found", "Run step 2, then step 3."),
            ("Missing packages: …", ".venv\\Scripts\\python.exe -m pip install -r requirements.txt"),
            ("Ollama is not reachable", "Start the Ollama app; `ollama list` should answer."),
            ("qwen3:4b is not installed", "ollama pull qwen3:4b, or set another installed model in .env."),
            ("Port 8765/8501 is used by another program", ".\\start_studio.bat --studio-port 8766 --dashboard-port 8502"),
            ("Configuration problem: …", "Fix the named setting in .env (see .env.example)."),
            ("Dashboard says the backend is not running", "Start with .\\start_studio.bat, or python -m webrag.server."),
            ("First run is slow", "The two retrieval models (about 180 MB) are downloaded once."),
            ("A provider fails (e.g. OpenAlex 429)", "The run continues with the others; OPENALEX_API_KEY avoids the shared budget."),
        ], columns=["Message or symptom", "What to do"]), hide_index=True, width="stretch")
    with st.container(border=True):
        st.markdown("**Security and privacy**")
        st.markdown("- Both services listen on 127.0.0.1 only and have no accounts: do not expose them to a network.\n"
                    "- Search queries go to the search providers you enable; source pages are fetched from their sites.\n"
                    "- With the default local model, prompts and passages stay on your computer.\n"
                    "- Saved runs (outputs/webrag/runs/) contain text fetched from the web and are not versioned.")
        st.markdown(f"Tests: `.venv\\Scripts\\python.exe -m pytest -q` and `node --test tests/js/*.test.mjs` (offline). "
                    f"Repository map: {src('docs/REPOSITORY_STRUCTURE.md')}.")


# ================================================================================================= 7. use cases
def use_cases() -> None:
    header("Use cases, limits and contributing", "What the current system is good for, what it cannot do, and what it "
                                                 "needs from you.")
    left, right = st.columns(2)
    with left.container(border=True, height="stretch"):
        st.markdown("**Good fits**")
        st.markdown("- Checking a specific scientific claim against open scholarly sources, with the passages shown\n"
                    "- Gathering cited, verifier-checked findings on a focused research question\n"
                    "- Seeing where evidence is missing, single-source or conflicting\n"
                    "- Asking questions about your own TXT and PDF documents, locally\n"
                    "- Studying a transparent RAG pipeline: every stage is recorded")
    with right.container(border=True, height="stretch"):
        st.markdown("**Not a fit (yet)**")
        st.markdown("- Final answers on medical, legal or safety questions: verdicts are a 4B model's judgements\n"
                    "- Paywalled literature: most sources are read as abstracts unless open access\n"
                    "- Exhaustive literature reviews: searches are budget-bounded\n"
                    "- Shared or multi-user deployment: no accounts or authentication\n"
                    "- Retraction checking: only a data contract exists")
    with st.container(border=True):
        st.markdown("**What you need, and what it costs**")
        st.dataframe(pd.DataFrame([
            ("Local model (default)", "Ollama + qwen3:4b on your computer", "Free; uses your CPU/GPU and about 2.5 GB disk"),
            ("Remote model API (optional)", "An Ollama server or OpenAI-compatible API and its key", "Provider charges may apply per run"),
            ("OpenAlex, arXiv, Crossref, Europe PMC", "Internet access", "Free; OpenAlex has a shared anonymous daily budget (free key available)"),
            ("SearXNG (optional)", "Your own instance", "Free to self-host"),
            ("Brave Search API (optional)", "API key", "Paid"),
            ("Retrieval models", "One-time download from the Hugging Face Hub", "Free, about 180 MB"),
        ], columns=["Dependency", "Needs", "Cost"]), hide_index=True, width="stretch")
        st.caption("Browsing this website triggers none of these: it calls no model and no search service.")
    with st.container(border=True):
        st.markdown("**Research limits**")
        st.markdown("- On SciFact proxy sets, no valid run of the current verifier or its candidates reaches the "
                    "three-class macro-F1 target (see Evaluations); its accuracy on web evidence is not measured.\n"
                    "- The verifier's quotes are sometimes not contiguous passage text (see the quote-attestation replay); "
                    "attestation is recorded, never used to change verdicts.\n"
                    "- Verdict quality with models other than qwen3:4b has not been evaluated.")
    with st.container(border=True):
        st.markdown("**Inspect and contribute**")
        st.markdown(f"- Code: [{content.REPO}]({content.REPO_URL}/tree/{content.BRANCH})\n"
                    f"- Report problems: [GitHub issues]({content.REPO_URL}/issues)\n"
                    f"- Before changing code, read {src('ARCHITECTURE_DECISION_VERSION_BOUNDARIES.md')}: several "
                    "research modules are frozen and must stay byte-identical.\n"
                    "- Run the offline tests before opening a pull request.")
