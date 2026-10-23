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


def header(title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(subtitle)


# ================================================================================================= 1. overview
def overview() -> None:
    st.markdown("<div class='eyebrow'>M.Tech research project · open source</div>", unsafe_allow_html=True)
    st.title("Multi-Agent RAG Research Studio")
    st.markdown("#### Research and claim verification that reports only what its evidence supports")
    st.write("Language models answer research questions fluently, including when no source supports the answer. This "
             "project splits research into separate, inspectable steps: a **planner** writes search queries, a **web "
             "researcher** reads scholarly sources, a **retrieval engine** ranks passages, a **synthesizer** drafts cited "
             "statements, and a **verifier** checks every statement against every retrieved passage. Only statements the "
             "verifier supports are reported as findings, and the sources, passages, conflicts and limits stay visible.")
    st.info("This showcase is a static guide to the project. It runs no model, starts no research and needs no "
            "account. To run research yourself, install the application on your computer (see **Install and run**).",
            icon=":material/info:")

    demos = results.examples()
    tests = results.test_inventory()
    with st.container(horizontal=True):
        st.metric("Components mapped", len(content.COMPONENTS), border=True,
                  help="Each with real source files; dependencies between them are computed from the code.")
        st.metric("Import edges found", len(architecture.code_edges()), border=True,
                  help="Computed now by parsing the import statements of the listed files.")
        st.metric("Recorded demonstrations", len(demos), border=True,
                  help="Saved Research Studio runs, published as links only (see Research workflow).")
        st.metric("Software test files", tests["python_files"] + tests["js_files"], border=True,
                  help=f"{tests['python_files']} Python files with {tests['python_functions']} test functions and "
                       f"{tests['js_files']} interface test files in the repository. Presence only, not a pass count, "
                       "and not a measure of research performance.")

    st.subheader("What the evidence shows")
    st.caption("Newest first. Each item says what kind of evidence it is; details and caveats are on Results and evidence.")
    findings = results.development_findings()
    with st.container(border=True):
        st.markdown(f"**1 · What the Research Studio workflow demonstrates** {evidence_badge('demonstration')}")
        st.write(f"{len(demos)} recorded runs (October 2026) show the full pipeline on real questions: planning, search, "
                 "per-passage verification with stances, cited findings and the recorded reason for stopping. They "
                 "illustrate behaviour; they do not measure accuracy.")
    if findings:
        replay = findings["quote_attestation_replay"]
        with st.container(border=True):
            st.markdown(f"**2 · Latest engineering finding ({replay['date']})** {evidence_badge('diagnostic')} "
                        ":orange-badge[exploratory]")
            st.write(f"A replay of recorded verifier outputs found that {replay['quotes_not_contiguous_but_accepted_by_legacy_check']} "
                     f"of {replay['decisive_stances']['total']} decisive quotes were not contiguous text of their "
                     "passage, often numbers mis-copied by the model. Quote attestation therefore stays record-only. "
                     "This is a diagnostic of evidence handling, not a benchmark, and it does not show improved accuracy "
                     "or fewer hallucinations.")
    with st.container(border=True):
        start, end = content.PHASE11_RUN_DATES
        st.markdown(f"**3 · Historical benchmark: Phase 11B (run {start} to {end})** {evidence_badge('historical')}")
        st.write("On 300 SciFact DEV claims, the project's earlier research backend was evaluated as three systems. The "
                 "two verification variants (B and C) showed **no statistically significant difference**. This "
                 "evaluated the earlier backend, not the current Research Studio pipeline. It is the most recent "
                 "public benchmark in this repository; later work (the Phase 12 and 13 diagnostics and the replay above) "
                 "is engineering evidence, not a new benchmark, and no public benchmark of the Studio exists yet.")
    st.page_link(st.session_state["pages"]["results"], label="See all results, methods and caveats",
                 icon=":material/arrow_forward:")

    st.subheader("Explore")
    cards = [("architecture", ":material/hub:", "Architecture explorer",
              "The real components and how they depend on each other, from the code."),
             ("workflow", ":material/route:", "Research workflow",
              "How a question moves through the pipeline, with recorded runs."),
             ("components", ":material/widgets:", "Component laboratory", "Every module: purpose, files, tests, limits."),
             ("results", ":material/analytics:", "Results and evidence",
              "Diagnostics, demonstrations, the historical benchmark, and what is unpublished."),
             ("install", ":material/download:", "Install and run", "Set it up on your own Windows computer."),
             ("use", ":material/menu_book:", "Use cases and limits", "What it can and cannot do, costs, privacy.")]
    cols = st.columns(3)
    for index, (key, icon, title, text) in enumerate(cards):
        with cols[index % 3].container(border=True, height="stretch"):
            st.page_link(st.session_state["pages"][key], label=title, icon=icon)
            st.caption(text)

    left, right = st.columns(2)
    with left.container(border=True):
        st.markdown("**Demonstrated capabilities**")
        st.markdown("- Research and claim verification with deep research (quick, standard, deep budgets)\n"
                    "- Per-passage verification with quoted evidence; verified-only findings\n"
                    "- Guarded network access (address checks, connection pinning, verified TLS)\n"
                    "- Original Studio interface and a local dashboard; saved runs and exports\n"
                    "- Local Ollama by default; optional remote model APIs")
    with right.container(border=True):
        st.markdown("**Experimental, unverified or not established**")
        st.markdown("- Default Arm C agentic verification (Studio-adapted): not yet evaluated on the Studio's web evidence\n"
                    "- Rank tie-break of conflicting evidence: experimental opt-in, not evaluated in the Studio\n"
                    "- Quote attestation (record only) and injection screening (fixed patterns): on by default, partial\n"
                    "- Retraction checks: contract only, no live lookup\n"
                    "- No public benchmark of the current Studio pipeline; unpublished figures are not shown\n"
                    "- Verdicts are judgements of a 4B model about retrieved text, not proof")

    with st.container(border=True):
        st.markdown("**How the Research Studio verifies a statement now**")
        st.markdown("The default verifier is **Arm C agentic verification, in a Studio-adapted variant** "
                    "(`arm_c_studio`):\n"
                    "1. One verifier call judges the retrieved passages: supports, contradicts or neutral, each "
                    "decisive judgement backed by an exact quote.\n"
                    "2. If the passages agree, that is the verdict.\n"
                    "3. If they conflict, an adjudicator decides which side tests the exact claim; the statement is "
                    "marked *adjudicated*, both sides stay listed, and it is kept out of the direct answer.\n"
                    "4. If no passage takes a side, a planner may search the evidence already collected (at most two "
                    "searches) before adjudication.\n"
                    "5. Without a passage for or against, nothing is decided: the statement stays INSUFFICIENT. "
                    "This rule, and two more (an adjudication citing a passage on the other side is not accepted; a "
                    "passage given an invalid stance fails the claim), are what differ from the historical Arm C, "
                    "which is still selectable as `arm_c`.\n\n"
                    "Retrieved passages that look like injected instructions are withheld whole (which can also "
                    "withhold genuine evidence in the same passage), and each quote's position in its passage is "
                    "recorded. The earlier per-passage verifier, `quote_stance`, stays available.")
        st.caption("Arm C comes from an unpublished, pre-specified study on SciFact development claims that earlier "
                   "project systems had already used; its result there was not significantly better than a simpler "
                   "baseline, and it confirmed more false claims. The Studio's accuracy with Arm C has not been "
                   "evaluated, so no accuracy figure is claimed for it here.")
    st.caption(f"Source code: [{content.REPO}]({content.REPO_URL}/tree/{content.BRANCH}) · branch `{content.BRANCH}` · "
               "MIT licence for the project's own code")


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
def results_page() -> None:
    header("Results and evidence", "What has been measured, how, and what it does and does not show. Benchmark "
                                   "numbers are read from the repository's result files when this page loads.")
    evidence_guide()
    tabs = st.tabs([":material/biotech: Engineering diagnostic", ":material/history: Recorded demonstrations",
                    ":material/analytics: Historical baseline (Phase 11B)", ":material/block: Unpublished claims",
                    ":material/fact_check: Software tests"])
    with tabs[0]:
        development()
    with tabs[1]:
        demonstrations()
    with tabs[2]:
        benchmark()
    with tabs[3]:
        unsubstantiated()
    with tabs[4]:
        software_tests()


def evidence_guide() -> None:
    findings = results.development_findings() or {}
    replay = findings.get("quote_attestation_replay") or {}
    start, end = content.PHASE11_RUN_DATES
    rows = [
        (replay.get("date", "—"), content.EVIDENCE_TYPES["diagnostic"][0] + " (exploratory)", "Quote-attestation replay",
         "the verifier's quoted evidence, in recorded runs", "aggregate counts only; records not distributed"),
        (", ".join(sorted({str(r.get("started"))[:10] for r in results.examples()})) or "—",
         content.EVIDENCE_TYPES["demonstration"][0], f"{len(results.examples())} Research Studio runs",
         "the Studio pipeline (recorded with the earlier quote_stance verifier)", "showcase/data/examples/ (links only)"),
    ]
    rows += [(d["date"], content.EVIDENCE_TYPES["diagnostic"][0], d["name"], "the earlier research backend", d["folder"])
             for d in content.LATER_DIAGNOSTICS]
    rows.append((f"{start} to {end}", content.EVIDENCE_TYPES["historical"][0] + " · " +
                 content.EVIDENCE_TYPES["benchmark"][0], "Phase 11B, SciFact DEV",
                 "the earlier research backend (systems A, B, C)", results.PHASE11 + "/"))
    with st.container(border=True):
        st.markdown("**How to read this page** — newest first; every result is labelled with its kind of evidence.")
        st.dataframe(pd.DataFrame(rows, columns=["Date", "Kind of evidence", "Finding", "Concerns", "Source"]),
                     hide_index=True, width="stretch")
        st.markdown(" ".join(f":{color}-badge[{label}]" for label, color, _ in content.EVIDENCE_TYPES.values()))
        st.caption(" · ".join(f"**{label}**: {meaning}" for label, _, meaning in content.EVIDENCE_TYPES.values()))


def benchmark() -> None:
    data = results.phase11()
    if not data:
        st.warning("The Phase 11B result files are not present in this copy of the repository.")
        return
    s, stats = data["summary"], data["stats"]
    acc = s["accounting"]
    paired = s.get("paired_BC") or {}
    start, end = content.PHASE11_RUN_DATES
    st.markdown(f"{evidence_badge('historical')} {evidence_badge('benchmark')} Phase 11B: a post-hoc analysis of one "
                f"completed run (created {start}, completed by {end}; {src(content.PHASE11_RUN_MANIFEST, 'run manifest')}) "
                f"on the **SciFact DEV** set — {acc['claims']} claims, {acc['pair_coverage']['attempt_pairs']} attempts, "
                f"{acc['claims_with_gold_evidence']} claims with gold labels.")
    with st.container(border=True):
        st.markdown("**What was compared — and what was not.** Three systems of the project's *earlier research "
                    "backend*: **A** strong RAG, **B** RAG plus claim/evidence verification, **C** bounded multi-agent "
                    "research. It does **not** evaluate the current Research Studio pipeline, and no newer public "
                    "benchmark exists. A gives no verifier label, so only B and C have classification scores. "
                    + src("evaluation/experiments/system_b_verified_rag.py", "system_b") + ", " +
                    src("evaluation/experiments/system_c_multi_agent.py", "system_c"))

    rows = {r["system"]: r for r in data["accuracy"]}
    b, c = rows.get("B"), rows.get("C")
    with st.container(horizontal=True):
        for name, row in (("B", b), ("C", c)):
            if row:
                st.metric(f"System {name} macro-F1", f"{float(row['macro_f1']):.4f}", border=True,
                          help=f"Accuracy {row['accuracy_num']}/{row['accuracy_den']} valid labelled outputs "
                               f"({float(row['accuracy']):.1%}). Denominators differ between systems.")
        st.metric("Paired claims", stats["valid_paired_labeled_n"], border=True,
                  help="Claims where both B and C produced a valid label.")
        if paired.get("B_paired_accuracy") is not None:
            st.metric("Paired accuracy, B and C", f"{paired['B_paired_accuracy']:.4f} / {paired['C_paired_accuracy']:.4f}",
                      border=True, help="Accuracy of each system on the same paired claims.")
        st.metric("Exact McNemar p", f"{stats['p_value']:.4f}", border=True, help=stats["test"])
    gap = abs(float(b["macro_f1"]) - float(c["macro_f1"])) if b and c else None
    st.error(f"**Not statistically significant.** Of the {stats['valid_paired_labeled_n']} paired claims, B alone was "
             f"right on {stats['B_only_correct']} and C alone on {stats['C_only_correct']} ({stats['test']}, "
             f"p = {stats['p_value']:.2f}). This does not show that the two are equivalent, and it does not show that "
             "either is better."
             + (f" The macro-F1 difference of {gap:.4f} is not evidence of a difference: the paired test found none, "
                f"and the two scores use different denominators ({b['accuracy_den']} and {c['accuracy_den']} valid "
                "outputs)." if gap is not None else ""),
             icon=":material/do_not_disturb_on:")

    left, right = st.columns(2)
    with left.container(border=True):
        st.markdown("**Paired outcomes, B vs C** (" + src(results.phase11_path("charts/paired_outcomes.csv"), "paired_outcomes.csv") + ")")
        frame = pd.DataFrame(data["paired"]).assign(count=lambda f: f["count"].astype(int))
        chart = alt.Chart(frame).mark_bar(color=ACCENT, cornerRadiusEnd=3).encode(
            x=alt.X("count:Q", title="claims"), y=alt.Y("outcome:N", sort="-x", title=None),
            tooltip=["outcome", "count"]).properties(height=180)
        st.altair_chart(chart, width="stretch")
    with right.container(border=True):
        st.markdown("**Document retrieval, Recall@k** (" + src(results.phase11_path("charts/retrieval_recall.csv"), "retrieval_recall.csv") + ")")
        frame = pd.DataFrame([r for r in data["retrieval"] if r["value"] not in ("NOT COMPUTED", "")])
        frame = frame.assign(value=lambda f: f["value"].astype(float), k=lambda f: "R@" + f["k"])
        chart = alt.Chart(frame).mark_bar(cornerRadiusEnd=3).encode(
            x=alt.X("system:N", title=None, axis=alt.Axis(labelAngle=0)), xOffset="k:N",
            y=alt.Y("value:Q", title=None, axis=alt.Axis(format="%")),
            color=alt.Color("k:N", scale=alt.Scale(range=[ACCENT, "#4aa3df"]), title=None),
            tooltip=["system", "k", alt.Tooltip("value:Q", format=".1%"), "numerator", "denominator"]).properties(height=180)
        st.altair_chart(chart, width="stretch")
        st.caption("Overlap with annotated evidence documents — not answer correctness or entailment. R@10/R@20 were "
                   "not computed (at most eight ranked chunks were stored).")

    left, right = st.columns(2)
    with left.container(border=True):
        st.markdown("**Execution outcomes** (" + src(results.phase11_path("charts/execution_outcomes.csv"), "execution_outcomes.csv") + ")")
        frame = pd.DataFrame(data["execution"]).assign(count=lambda f: f["count"].astype(int))
        chart = alt.Chart(frame).mark_bar().encode(
            x=alt.X("count:Q", stack="zero", title="attempts"), y=alt.Y("system:N", title=None),
            color=alt.Color("status:N", scale=alt.Scale(domain=["SUCCESS", "FAILED", "INTERRUPTED"],
                                                        range=[ACCENT, "#e5646e", "#d6a24e"]), title=None),
            tooltip=["system", "status", "count"]).properties(height=150)
        st.altair_chart(chart, width="stretch")
        st.caption("Failed and interrupted attempts were not retried, erased or counted as correct; they are left "
                   "out of the valid-prediction denominators.")
    with right.container(border=True):
        st.markdown("**Mean latency of successful attempts** (" + src(results.phase11_path("charts/latency.csv"), "latency.csv") + ")")
        frame = pd.DataFrame([r for r in data["latency"] if r["status"] == "SUCCESS" and r["mean"]])
        frame = frame.assign(seconds=lambda f: f["mean"].astype(float))
        chart = alt.Chart(frame).mark_bar(color="#4aa3df", cornerRadiusEnd=3).encode(
            x=alt.X("seconds:Q", title="seconds"), y=alt.Y("system:N", title=None),
            tooltip=["system", alt.Tooltip("seconds:Q", format=".1f"), "n"]).properties(height=150)
        st.altair_chart(chart, width="stretch")
        st.caption("On the author's hardware with a local 4B model; a measure of cost, not quality.")

    with st.container(border=True):
        st.markdown("**Recorded conclusion** (" + src(results.phase11_path("final_scientific_conclusion.md"),
                                                       "final_scientific_conclusion.md") + ")")
        st.write(s.get("scientific_conclusion"))
        st.markdown("**Recorded limitations**\n" + "\n".join(f"- {x}" for x in s.get("limitations") or []))
    st.caption("All files: " + src(results.PHASE11 + "/", results.PHASE11))


def development() -> None:
    data = results.development_findings()
    if not data:
        st.info("No development findings are bundled with this copy of the showcase.")
        return
    replay = data["quote_attestation_replay"]
    with st.container(border=True):
        st.markdown(f"### Exploratory engineering diagnostic {evidence_badge('diagnostic')}")
        st.caption(f"Quote-attestation replay · recorded {replay['date']}")
        st.write("An internal quote-attestation replay examined quote-contiguity issues and their potential effect on "
                 "recorded verdicts. This page presents aggregate diagnostic counts only; individual claim-level "
                 "records are not distributed publicly. The findings are exploratory and do not establish improved "
                 "benchmark accuracy or reduced hallucination rates.")
        st.markdown(f"**What was done** — {replay['what_was_done']}")
        d = replay["decisive_stances"]
        with st.container(horizontal=True):
            st.metric("Decisive quotes checked", d["total"], border=True)
            st.metric("Not contiguous text", replay["quotes_not_contiguous_but_accepted_by_legacy_check"], border=True,
                      help="Accepted by the earlier tolerant check, but not found as contiguous text of the passage.")
            st.metric("Records affected", replay["records_with_an_unattested_decisive_stance"], border=True)
            st.metric("Hypothetical verdict changes", replay["hypothetical_verdict_changes"]["count"], border=True,
                      help=replay["hypothetical_verdict_changes"]["assumption"])
        frame = pd.DataFrame([{"stance": stance, "status": status, "quotes": d[stance][status]}
                              for stance in ("SUPPORTS", "CONTRADICTS") for status in ("located", "unattested")])
        chart = alt.Chart(frame).mark_bar().encode(
            x=alt.X("quotes:Q", stack="zero", title="decisive quotes"), y=alt.Y("stance:N", title=None),
            color=alt.Color("status:N", scale=alt.Scale(domain=["located", "unattested"], range=[ACCENT, "#d6a24e"]),
                            title=None), tooltip=["stance", "status", "quotes"]).properties(height=120)
        st.altair_chart(chart, width="stretch")
        left, right = st.columns(2)
        with left:
            st.markdown("**What it measures**")
            st.markdown("\n".join(f"- {x}" for x in replay["interpretation"]))
        with right:
            st.markdown("**What it does not establish**")
            st.markdown("\n".join(f"- {x}" for x in replay["does_not_establish"]))
        st.markdown(f"**Decision recorded after the replay** — {replay['decision']}")
        st.caption(f"Limitations: {' '.join(replay['limitations'])} Protocol identifier: SHA-256 beginning "
                   f"{replay['protocol_sha256_prefix']}…, which identifies the recorded diagnostic protocol; it is not "
                   f"an independent verification. Source: {replay['source']}")
    with st.container(border=True):
        st.markdown(f"**Earlier public diagnostics of the research backend** {evidence_badge('diagnostic')}")
        for item in content.LATER_DIAGNOSTICS:
            st.markdown(f"- **{item['name']}** ({item['date']}; {src(item['date_source'], 'date source')}) — "
                        f"{item['what']} Files: {src(item['folder'], item['folder'])}")
        st.caption("These files are in this repository, so their details can be inspected there. They are listed for "
                   "completeness; they are not new benchmark results and are not summarised as such here.")


def demonstrations() -> None:
    runs = results.examples()
    st.markdown(f"{evidence_badge('demonstration')} Saved runs of the Research Studio, published as links only. "
                "Each lists the verifier it was recorded with; runs recorded before the Arm C default used "
                "`quote_stance`.")
    st.write("Each run shows the recorded question or claim, the workflow stages with their timings, the model-written "
             "plan and statements, the verifier's verdicts and per-passage stances, the sources' titles, links and "
             "access level, coverage, and the recorded reason for stopping. Third-party text is not included.")
    if not runs:
        st.info("No recorded demonstration is bundled with this copy of the showcase.")
        return
    for run in runs:
        claims = run.get("claims") or []
        supported = sum(1 for c in claims if c.get("label") == "SUPPORTED")
        with st.container(border=True):
            st.markdown(f"**{esc(run['input'])}**")
            st.caption(f"{'Claim verification' if run['mode'] == 'verify' else 'Research question'} · "
                       f"{run.get('depth') or 'single pass'} depth · recorded {str(run.get('started'))[:10]} · "
                       f"verifier {run.get('verifier') or 'not recorded'} · "
                       f"{supported} of {len(claims)} statements supported · {len(run.get('sources') or [])} sources")
    st.warning("Recorded examples illustrate how the workflow behaves; two runs do not, by themselves, establish the "
               "system's overall accuracy. Open them in **Research workflow → A recorded run**.",
               icon=":material/info:")


def unsubstantiated() -> None:
    st.markdown(f"{evidence_badge('unpublished')} Figures that have been cited about this project but are **not "
                "presented here as public performance claims**, and why.")
    with st.container(border=True):
        st.markdown("**An earlier-cited macro-F1 comparison from the thesis analysis**")
        st.write("This comparison (188 labelled SciFact DEV claims, a verification variant against a retrieval-tuned "
                 "baseline) comes from unpublished thesis analysis whose result files are not in this repository. By "
                 "its own record the difference was not statistically significant at the pre-specified α = 0.05. Its "
                 "figures are therefore not shown here.")
    with st.container(border=True):
        st.markdown("**An earlier-cited count of targeted tests**")
        st.write("No recorded test report supports that count, so it is not shown. The recorded test reports are "
                 "listed under Software tests.")
    st.caption("Methodology: results are shown as public performance claims only when their underlying evidence and "
               "evaluation protocol are available in this repository and appropriate to publish.")


def software_tests() -> None:
    data = results.development_findings()
    st.markdown(f"{evidence_badge('software')} Reports of the project's automated software tests. They check that "
                "the code behaves as specified, offline and with stubbed models; they are not research performance "
                "and do not show that this website or a deployment works.")
    if not data:
        return
    tests = data["targeted_tests"]
    full = data["full_suite"]
    with st.container(border=True):
        st.markdown(f"**Recorded reports** — {tests['date']}; recorded by the author, not re-run by this page")
        for run in tests["runs"]:
            st.markdown(f"- {run['scope']}: **{run['result']}**")
        st.markdown(f"- Full offline suite on commit `{full['commit']}`: **{full['result']}**")
        st.caption(f"Source: {tests['source']}")
    inventory = results.test_inventory()
    st.caption(f"Counted in this repository now: {inventory['python_files']} Python test files with "
               f"{inventory['python_functions']} test functions, and {inventory['js_files']} interface test files "
               "(presence, not pass status).")


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
        st.markdown("- The historical Phase 11B benchmark evaluated the earlier research backend on SciFact DEV only "
                    "and found no significant difference "
                    "between the verification variants it compared.\n"
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
