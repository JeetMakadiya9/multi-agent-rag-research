"""The dashboard's pages. Everything shown comes from the Research Studio's API (``client.StudioClient``); when the
Studio cannot answer, the page says so instead of showing placeholders."""
from __future__ import annotations

import html
import time
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional

import altair as alt
import pandas as pd
import streamlit as st

from client import StudioClient, StudioError

VERDICT_COLOR = {"SUPPORTED": "green", "CONTRADICTED": "red", "INSUFFICIENT": "orange"}
VERDICT_HEX = {"SUPPORTED": "#3fb68b", "CONTRADICTED": "#e5646e", "INSUFFICIENT": "#d6a24e"}
STANCE_COLOR = {"SUPPORTS": "green", "CONTRADICTS": "red", "NEUTRAL": "gray", "NOT_ASSESSED": "gray"}
STATE_COLOR = {"queued": "gray", "running": "blue", "done": "green", "cancelled": "gray"}
DEPTH_LABELS = {None: "Single pass", "quick": "Quick", "standard": "Standard", "deep": "Deep"}
MODE_LABELS = {"research": "Research", "verify": "Verification"}


def client() -> StudioClient:
    return st.session_state.setdefault("_client", StudioClient())


@st.cache_data(ttl=5, show_spinner=False)
def cached_status(base: str) -> Dict[str, Any]:
    return StudioClient(base).status()


@st.cache_data(ttl=10, show_spinner=False)
def cached_runs(base: str) -> List[Dict[str, Any]]:
    return StudioClient(base).runs()


@st.cache_data(ttl=300, show_spinner=False)
def cached_run(base: str, name: str) -> Dict[str, Any]:
    return StudioClient(base).run(name)


@st.cache_data(ttl=30, show_spinner=False)
def cached_providers(base: str) -> Dict[str, Any]:
    return StudioClient(base).providers()


@st.cache_data(ttl=30, show_spinner=False)
def cached_integrations(base: str) -> Dict[str, Any]:
    return StudioClient(base).integrations()


# ================================================================================================= shared pieces
def studio_unreachable(error: StudioError) -> None:
    with st.container(border=True):
        st.markdown("#### :material/power_off: Research Studio backend is not running")
        st.write(str(error))
        st.markdown("Start everything with the launcher in the project folder:")
        st.code("start_studio.bat", language="text")
        st.caption("Or start the backend alone with `.venv\\Scripts\\python.exe -m webrag.server` and reload this page.")


def load(fn, *args):
    """Call a cached Studio read; on failure show why and stop this page."""
    try:
        return fn(client().base, *args)
    except StudioError as exc:
        if exc.status == 0:
            studio_unreachable(exc)
        else:
            st.error(f"The Studio answered with an error: {exc}")
        st.stop()


def runtime(status: Dict[str, Any]) -> Dict[str, Any]:
    """Where the model runs and whether it is ready, from the Studio's status (local Ollama or a configured service)."""
    remote = status.get("runtime") if isinstance(status.get("runtime"), dict) else None
    reachable = bool(remote.get("service_reachable")) if remote else bool(status.get("ollama_version"))
    ready = reachable and bool(status.get("model_installed"))
    if remote:
        where = f"{remote.get('provider_label')} · {remote.get('endpoint')}"
        problem = remote.get("error") or ("" if ready else "the model is not offered by the service")
    else:
        where = f"Local Ollama {status.get('ollama_version')}" if reachable else "Local Ollama"
        problem = ("" if ready else f"{status.get('model')} is not installed (run: ollama pull {status.get('model')})"
                   if reachable else "Ollama is not reachable at 127.0.0.1:11434 (start the Ollama app)")
    return {"remote": bool(remote), "reachable": reachable, "ready": ready, "where": where, "problem": problem}


def studio_link(label: str = "Open Research Studio", fragment: str = "", key: Optional[str] = None) -> None:
    st.link_button(label, client().base + "/" + (f"#{fragment}" if fragment else ""), icon=":material/open_in_new:",
                   key=key)


def fmt_time(value: Optional[str]) -> str:
    if not value:
        return "—"
    try:
        stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return stamp.astimezone().strftime("%d %b %Y, %H:%M")
    except ValueError:
        return str(value)


SUB_STATE_WARNINGS = {
    "CONFLICTING": "Sources disagree: passages were judged both for and against, and no verdict is given.",
    "ADJUDICATED": "Sources disagreed; an adjudicator judged one side decisive. Both sides are listed below. Treat "
                   "this verdict with more caution than one on which the passages agree.",
    "RANK_TIEBREAK": "Sources disagreed; the first-ranked passage with a stance was taken as decisive. A first-ranked "
                     "passage is not a more reliable one. Both sides are listed below.",
    "ADJUDICATED_UNSIDED": "No passage was judged for or against, yet the historical Arm C adjudicator decided this "
                           "statement. Treat it as unestablished.",
    "ASSESSMENT_FAILED": "The verifier could not assess this statement; it is not counted as verified.",
    "SERVICE_ERROR": "The model service failed during verification; this statement is not verified.",
}


def page_header(title: str, subtitle: str, link: bool = True) -> None:
    left, right = st.columns([5, 1.4], vertical_alignment="bottom")
    with left:
        st.title(title)
        st.caption(subtitle)
    if link:
        with right:
            studio_link()


def open_run(name: str) -> None:
    st.session_state["run_name"] = name
    st.switch_page(st.session_state["pages"]["evidence"])


def esc(text: Any) -> str:
    """Untrusted run text (questions, model statements, web passages) for markdown: no HTML, no markdown links."""
    return html.escape(str(text or "")).replace("[", "\\[").replace("]", "\\]").replace("*", "\\*").replace("_", "\\_")


# ================================================================================================= A. command center
def command_center() -> None:
    page_header("Command Center", "Health of the local Research Studio, its model and your saved research.")
    status = load(cached_status)
    runs = load(cached_runs)
    info = runtime(status)

    with st.container(horizontal=True):
        st.metric("Studio backend", "Online", border=True, help=f"{client().base} · version {status.get('studio_version')}")
        st.metric("Model", status.get("model") or "—", border=True,
                  help=info["where"] + (f" — {info['problem']}" if info["problem"] else ""))
        st.metric("Model status", "Ready" if info["ready"] else "Not ready", border=True,
                  help="Checked by the Studio when it last reported status (every 15 s).")
        st.metric("Saved runs", len(runs), border=True, help=f"Records in {status.get('runs_folder')}")
        claims = sum(r.get("claims") or 0 for r in runs)
        supported = sum(r.get("supported") or 0 for r in runs)
        st.metric("Statements supported", f"{supported} / {claims}", border=True,
                  help="Verifier verdicts across all saved runs (model judgements about retrieved text, not proof).")

    if not info["ready"]:
        st.warning(f"**Research cannot start yet:** {info['problem']}. Saved reports remain available.",
                   icon=":material/warning:")

    left, right = st.columns([1, 1.25])
    with left:
        with st.container(border=True):
            st.markdown("**Verdicts across saved runs**")
            counts = {label: sum(r.get(label.lower()) or 0 for r in runs) for label in VERDICT_HEX}
            if sum(counts.values()):
                frame = pd.DataFrame({"verdict": list(counts), "statements": list(counts.values())})
                chart = alt.Chart(frame).mark_bar(cornerRadiusEnd=3).encode(
                    x=alt.X("statements:Q", title=None), y=alt.Y("verdict:N", sort=list(VERDICT_HEX), title=None),
                    color=alt.Color("verdict:N", scale=alt.Scale(domain=list(VERDICT_HEX), range=list(VERDICT_HEX.values())),
                                    legend=None),
                    tooltip=["verdict", "statements"]).properties(height=170)
                st.altair_chart(chart, width="stretch")
            else:
                st.caption("No verified statements yet.")
    with right:
        with st.container(border=True):
            st.markdown("**Research activity**")
            days = Counter(str(r.get("started") or "")[:10] for r in runs if r.get("started"))
            if days:
                frame = pd.DataFrame(sorted(days.items())[-30:], columns=["day", "runs"])
                frame["label"] = pd.to_datetime(frame["day"]).dt.strftime("%d %b")
                chart = alt.Chart(frame).mark_bar(color="#2bb3a3", cornerRadiusEnd=3, size=26).encode(
                    x=alt.X("label:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("runs:Q", title="runs started", axis=alt.Axis(tickMinStep=1)),
                    tooltip=[alt.Tooltip("day:N", title="day"), "runs"]).properties(height=170)
                st.altair_chart(chart, width="stretch")
            else:
                st.caption("No saved runs yet. Start one in the Research Workspace.")

    with st.container(border=True):
        st.markdown("**Recent investigations**")
        if runs:
            picked = runs_table(runs[:8], key="recent")
            if picked:
                open_run(picked)
        else:
            st.caption("Nothing saved yet.")

    try:
        jobs = client().jobs()
    except StudioError:
        jobs = []
    active = [j for j in jobs if j["state"] in ("queued", "running")]
    if active:
        with st.container(border=True):
            st.markdown(f"**Running now** · {len(active)} job(s)")
            for job in active:
                st.markdown(f":{STATE_COLOR[job['state']]}-badge[{job['state']}] {esc(job.get('input'))}")


def runs_table(runs: List[Dict[str, Any]], key: str) -> Optional[str]:
    """Saved runs as a table; returns the name of a selected run."""
    frame = pd.DataFrame([{
        "Question or claim": r.get("input") or "(unreadable record)",
        "Mode": MODE_LABELS.get(r.get("mode"), r.get("mode") or "—"),
        "Status": r.get("status") or "—", "Outcome": (r.get("outcome") or "—").replace("_", " ").title(),
        "Supported": f"{r.get('supported', 0)} / {r.get('claims', 0)}", "Sources": r.get("sources", 0),
        "Started": fmt_time(r.get("started")), "name": r["name"]} for r in runs])
    event = st.dataframe(frame, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
                         key=key, column_order=[c for c in frame.columns if c != "name"],
                         column_config={"Question or claim": st.column_config.TextColumn(width="large"),
                                        "Sources": st.column_config.NumberColumn(format="%d")})
    rows = event.selection.rows if event and event.selection else []
    return frame.iloc[rows[0]]["name"] if rows else None


# ================================================================================================= B. workspace
def workspace() -> None:
    page_header("Research Workspace", "Ask a research question or verify a claim. Runs execute in the Studio's real "
                                      "pipeline: planner, web researcher, retrieval, verifier and synthesizer.")
    status = load(cached_status)
    payload = load(cached_providers)
    info = runtime(status)
    active = [row for row in payload.get("providers", []) if row.get("active")]

    form_col, jobs_col = st.columns([1.1, 1], gap="large")
    with form_col:
        with st.container(border=True):
            mode = st.segmented_control("Task", ["research", "verify"], default="research", required=True,
                                        format_func={"research": "Research a question", "verify": "Verify a claim"}.get)
            text = st.text_area("Question" if mode == "research" else "Claim", height=110, max_chars=2000,
                                placeholder="What architectural idea did ResNet introduce, and why did it help training?"
                                if mode == "research" else "Residual connections make very deep networks easier to train.")
            depth = st.segmented_control("Depth", [None, "quick", "standard", "deep"], default="quick", required=True,
                                         format_func=DEPTH_LABELS.get)
            options: Dict[str, Any] = {}
            if depth:
                budget = (payload.get("depths") or {}).get(depth) or {}
                if budget:
                    st.caption(f"{DEPTH_LABELS[depth]}: up to {budget.get('rounds')} round(s), {budget.get('searches')} "
                               f"searches, {budget.get('sources')} sources, {budget.get('full_text')} full-text reads; "
                               f"stops after {round((budget.get('seconds') or 0) / 60)} min.")
                names = {row["id"]: row["name"] for row in active}
                c1, c2 = st.columns(2)
                chosen = c1.multiselect("Sources", list(names), default=list(names), format_func=names.get)
                recency = c2.selectbox("Published", payload.get("recency") or ["any"],
                                       format_func={"any": "Any time", "year": "Past year", "5y": "Past 5 years"}.get)
                categories = st.multiselect("Prefer source types", list(payload.get("categories") or {}))
                options = {"depth": depth, "providers": chosen, "recency": recency, "categories": categories}
            else:
                st.caption("Single pass: the original pipeline, one round over the active providers.")
            if not info["ready"]:
                st.warning(f"The model is not ready: {info['problem']}. A run would stop at its model check.",
                           icon=":material/warning:")
            start = st.button("Start research" if mode == "research" else "Verify claim", type="primary",
                              icon=":material/play_arrow:", width="stretch")
        if start:
            if not text.strip():
                st.toast("Enter a question or claim first.", icon=":material/info:")
            elif depth and not options["providers"]:
                st.toast("Choose at least one source.", icon=":material/info:")
            else:
                try:
                    job_id = client().submit(text.strip(), mode, options or None)
                    st.session_state.setdefault("my_jobs", []).insert(0, job_id)
                    st.toast("Submitted to the Studio's job queue.", icon=":material/check:")
                except StudioError as exc:
                    st.error(f"The Studio refused the run: {exc}")
        st.caption(f"Model: **{esc(status.get('model'))}** on {esc(info['where'])} · verifier {esc(status.get('verifier'))}. "
                   "One job runs at a time; queued jobs wait their turn.")

    with jobs_col:
        my_jobs = st.session_state.get("my_jobs", [])
        if not my_jobs:
            with st.container(border=True):
                st.markdown("**Progress**")
                st.caption("Runs you start here appear with their live stages. Runs started in the Studio "
                           "itself are listed in the Library when they finish.")
        else:
            live = any(job_state(j) in ("queued", "running", None) for j in my_jobs[:5])
            st.fragment(run_every=2 if live else None)(job_panel)()


def job_state(job_id: str) -> Optional[str]:
    return st.session_state.get("job_states", {}).get(job_id)


def job_panel() -> None:
    for job_id in st.session_state.get("my_jobs", [])[:5]:
        try:
            job = client().job(job_id)
        except StudioError as exc:
            st.warning(f"Job {job_id}: {exc}" if exc.status else str(exc))
            st.session_state.setdefault("job_states", {})[job_id] = "gone"
            continue
        st.session_state.setdefault("job_states", {})[job_id] = job["state"]
        data = job.get("data") or {}
        with st.container(border=True):
            head = f":{STATE_COLOR.get(job['state'], 'gray')}-badge[{job['state']}]"
            if job.get("position"):
                head += f" :gray-badge[position {job['position']} in queue]"
            if data.get("status") and job["state"] == "done":
                head += f" :{'red' if data['status'] == 'FAILED' else 'green'}-badge[{data['status']}]"
            st.markdown(head)
            st.markdown(f"**{esc(data.get('input'))}**")
            stages = data.get("stages") or []
            if stages:
                frame = pd.DataFrame([{"Stage": s.get("name"), "Result": s.get("status"), "Seconds": s.get("seconds"),
                                       "Detail": s.get("detail")} for s in stages])
                st.dataframe(frame, hide_index=True, width="stretch")
            if job["state"] == "running" and data.get("current"):
                st.caption(f":material/progress_activity: {esc(data['current'])}")
            if data.get("error"):
                st.error(data["error"])
            if job["state"] == "running" and job.get("cancel_requested"):
                st.caption("Stopping after the step under way finishes. Nothing will be saved.")
            elif job["state"] in ("queued", "running"):
                if st.button("Cancel", key=f"cancel-{job_id}", icon=":material/close:"):
                    try:
                        client().cancel(job_id)
                    except StudioError as exc:
                        st.error(str(exc))
                    st.rerun()
                if job["state"] == "running":
                    st.caption("Cancelling stops the run after the step under way finishes; nothing is saved.")
            if job["state"] == "done" and data.get("saved_to"):
                name = str(data["saved_to"]).replace("\\", "/").rstrip("/").split("/")[-1]
                if st.button("Open in Evidence Explorer", key=f"open-{job_id}", icon=":material/fact_check:"):
                    cached_runs.clear()
                    open_run(name)


# ================================================================================================= C. library
def library() -> None:
    page_header("Research Library", "Every saved investigation, newest first. Select one to open its evidence.")
    runs = load(cached_runs)
    if not runs:
        st.info("No saved runs yet. Finished runs are saved by the Studio and appear here.", icon=":material/inbox:")
        return
    with st.container(horizontal=True, vertical_alignment="bottom"):
        query = st.text_input("Search", placeholder="Search questions and claims", label_visibility="collapsed",
                              icon=":material/search:")
        modes = st.segmented_control("Mode", ["research", "verify"], selection_mode="multi",
                                     format_func=MODE_LABELS.get, label_visibility="collapsed")
        statuses = sorted({r.get("status") or "—" for r in runs})
        status_pick = st.multiselect("Status", statuses, placeholder="Any status", label_visibility="collapsed")
        if st.button("Refresh", icon=":material/refresh:"):
            cached_runs.clear()
            st.rerun()
    rows = [r for r in runs
            if (not query or query.lower() in str(r.get("input", "")).lower())
            and (not modes or r.get("mode") in modes)
            and (not status_pick or (r.get("status") or "—") in status_pick)]
    st.caption(f"Showing {len(rows)} of {len(runs)} saved runs.")
    picked = runs_table(rows, key="library") if rows else None
    if picked:
        open_run(picked)


# ================================================================================================= D. evidence
def evidence() -> None:
    page_header("Evidence Explorer", "Findings, verdicts, passages and sources of one saved run, exactly as recorded.",
                link=False)
    runs = load(cached_runs)
    readable = [r for r in runs if r.get("status") != "UNREADABLE"]
    if not readable:
        st.info("No saved runs to explore yet.", icon=":material/inbox:")
        return
    names = [r["name"] for r in readable]
    labels = {r["name"]: f"{str(r.get('input') or '')[:90]} · {fmt_time(r.get('started'))}" for r in readable}
    wanted = st.session_state.get("run_name")
    name = st.selectbox("Saved run", names, index=names.index(wanted) if wanted in names else 0, format_func=labels.get)
    st.session_state["run_name"] = name
    data = load(cached_run, name)
    claims = [c for c in data.get("claims") or [] if isinstance(c, dict)]
    sources = {s["id"]: s for s in data.get("sources") or [] if isinstance(s, dict) and "id" in s}

    with st.container(border=True):
        top, actions = st.columns([4, 1.6], vertical_alignment="center")
        with top:
            st.markdown(f"### {esc(data.get('input'))}")
            badges = [f":blue-badge[{MODE_LABELS.get(data.get('mode'), data.get('mode'))}]",
                      f":{'red' if data.get('status') == 'FAILED' else 'green'}-badge[{data.get('status')}]"]
            if data.get("outcome"):
                badges.append(f":gray-badge[{str(data['outcome']).replace('_', ' ').lower()}]")
            if data.get("depth"):
                badges.append(f":violet-badge[{data['depth']} research]")
            st.markdown(" ".join(badges))
            model = data.get("model") if isinstance(data.get("model"), dict) else {}
            where = (f"{model.get('provider_label')} {model.get('endpoint')}" if model.get("provider")
                     else f"Ollama {model.get('ollama_version')}" if model.get("ollama_version") else "not recorded")
            st.caption(f"Started {fmt_time(data.get('started'))} · finished {fmt_time(data.get('finished'))} · "
                       f"model {esc(model.get('model') or (data.get('settings') or {}).get('model') or '—')} on {esc(where)}")
        with actions:
            studio_link("Open in Studio", f"/{data.get('mode', 'research')}/run/{name}", key="open-studio-run")
            c1, c2 = st.columns(2)
            for col, kind in ((c1, "md"), (c2, "json")):
                try:
                    payload = client().export(name, kind)
                    col.download_button(kind.upper(), payload, file_name=f"{name}.{kind}", icon=":material/download:",
                                        key=f"export-{kind}", width="stretch")
                except StudioError as exc:
                    col.caption(f"{kind}: {exc}")
    if data.get("error"):
        st.error(data["error"])

    counts = Counter(c.get("label") for c in claims)
    with st.container(horizontal=True):
        st.metric("Statements", len(claims), border=True)
        st.metric("Supported", counts.get("SUPPORTED", 0), border=True)
        st.metric("Contradicted", counts.get("CONTRADICTED", 0), border=True)
        st.metric("Insufficient evidence", counts.get("INSUFFICIENT", 0), border=True)
        quality = data.get("source_quality") if isinstance(data.get("source_quality"), dict) else {}
        st.metric("Sources", len(sources), border=True,
                  help=f"{(quality.get('by_access') or {}).get('FULL_TEXT', 0)} read in full" if quality else None)

    tabs = st.tabs([":material/fact_check: Findings", ":material/description: Answer",
                    ":material/library_books: Sources", ":material/checklist: Coverage",
                    ":material/report: Limits"])
    numbers = finding_numbers(claims)
    with tabs[0]:
        findings(name, claims, sources, numbers)
    with tabs[1]:
        answer(data, numbers)
    with tabs[2]:
        sources_table(sources)
    with tabs[3]:
        coverage(data)
    with tabs[4]:
        limits(data)


def findings(name: str, claims: List[Dict[str, Any]], sources: Dict[str, Dict[str, Any]],
             numbers: Dict[int, int]) -> None:
    if not claims:
        st.caption("This run has no statements.")
        return
    st.caption("SUPPORTED: at least one passage supports the statement and none contradicts it. CONTRADICTED: the "
               "reverse. INSUFFICIENT: anything else, including disagreeing passages.")
    for position in sorted(range(len(claims)), key=numbers.get):
        claim = claims[position]
        label = claim.get("label", "")
        title = f"**F{numbers[position]}** :{VERDICT_COLOR.get(label, 'gray')}-badge[{label}] {esc(claim.get('text'))}"
        with st.expander(title, expanded=False):
            meta = [x for x in (claim.get("strength"), (claim.get("sub_state") or "").replace("_", " ").lower()) if x]
            if meta:
                st.caption(" · ".join(meta))
            if SUB_STATE_WARNINGS.get(claim.get("sub_state")):
                st.warning(SUB_STATE_WARNINGS[claim["sub_state"]])
            for caution in claim.get("cautions") or []:
                st.warning(f"Caution ({str(caution.get('kind', '')).replace('_', ' ')}): {caution.get('detail', '')}")
            if claim.get("verifier_note"):
                st.caption(f"Verifier ({esc(claim.get('verifier') or 'not recorded')}): {esc(claim['verifier_note'])}")
            support = ", ".join(claim.get("supporting_sources") or []) or "none"
            against = ", ".join(claim.get("contradicting_sources") or []) or "none"
            st.markdown(f"Supporting sources: **{esc(support)}** · contradicting sources: **{esc(against)}**")
            for index, passage in enumerate(claim.get("passages") or []):
                stance = passage.get("stance", "")
                source = sources.get(passage.get("source_id"), {})
                with st.container(border=True):
                    st.markdown(f":{STANCE_COLOR.get(stance, 'gray')}-badge[{stance}] "
                                f"**{esc(passage.get('source_id'))}** · {esc(source.get('title') or passage.get('title'))}")
                    if passage.get("found_by_search"):
                        st.caption(f"Found by the Arm C planner's search: {esc(passage['found_by_search'])}")
                    st.markdown(f"> {esc(str(passage.get('text') or '')[:900])}")
                    cols = st.columns([1, 1, 3])
                    url = passage.get("url") or source.get("url")
                    if url and str(url).startswith(("http://", "https://")):
                        cols[0].link_button("Source", url, icon=":material/link:")
                    key = f"ctx-{claim.get('id')}-{index}"
                    if passage.get("key") and cols[1].button("In context", key=key, icon=":material/zoom_in:"):
                        st.session_state[key + "-open"] = True
                    if st.session_state.get(key + "-open"):
                        show_context(name, passage)


def show_context(name: str, passage: Dict[str, Any]) -> None:
    try:
        ctx = client().context(name, str(passage.get("source_id")), str(passage.get("key")))
    except StudioError as exc:
        st.caption(f"Context unavailable: {exc}")
        return
    if not ctx.get("located"):
        st.caption("The passage could not be located in the recorded source text.")
        return
    text = lambda value: html.escape(str(value or ""))               # inside an HTML block: escape HTML only
    before = ("…" if ctx.get("truncated_before") else "") + text(ctx.get("before"))
    after = text(ctx.get("after")) + ("…" if ctx.get("truncated_after") else "")
    st.html(f"<div style='font-size:0.92rem;line-height:1.6;color:#c9d1d6;border-left:2px solid #2bb3a3;padding:0.4rem 0.9rem'>"
            f"{before}<mark style='background:#1f4f49;color:#e6fffb;padding:0 2px'>{text(ctx.get('match'))}</mark>{after}</div>")
    st.caption(f"Page {ctx.get('page')} of {esc((ctx.get('source') or {}).get('title'))}")


def finding_numbers(claims: List[Dict[str, Any]]) -> Dict[int, int]:
    """F-numbers as the Studio and the Markdown report assign them: supported statements first, then the others.
    Keyed by the claim's position in the run record (what summaries refer to)."""
    order = ([i for i, c in enumerate(claims) if c.get("label") == "SUPPORTED"]
             + [i for i, c in enumerate(claims) if c.get("label") != "SUPPORTED"])
    return {index: number for number, index in enumerate(order, 1)}


def finding_refs(refs: Any, numbers: Dict[int, int]) -> str:
    return ", ".join(f"F{numbers[r]}" for r in refs or [] if isinstance(r, int) and r in numbers)


def answer(data: Dict[str, Any], numbers: Dict[int, int]) -> None:
    summary = data.get("summary") if isinstance(data.get("summary"), dict) else None
    if summary and summary.get("available"):
        direct = summary.get("direct_answer") or {}
        if direct.get("text"):
            with st.container(border=True):
                st.markdown("**Direct answer**")
                st.write(direct["text"])
                st.caption("Written by the model from verified findings only; it restates "
                       + (finding_refs(direct.get("claims"), numbers) or "no findings"))
        for item in summary.get("summary") or []:
            st.markdown(f"- {esc(item.get('text'))} — {esc(finding_refs(item.get('claims'), numbers))}")
    elif summary and summary.get("reason"):
        st.info(f"No summary: {summary['reason']}")
    else:
        st.caption("This run has no model-written summary; see Findings for the verified statements.")
    stopping = data.get("stopping") if isinstance(data.get("stopping"), dict) else None
    if stopping:
        with st.container(border=True):
            st.markdown(f"**Why searching stopped** · :gray-badge[{stopping.get('completeness', '—')}]")
            st.write(stopping.get("detail") or stopping.get("reason"))


def sources_table(sources: Dict[str, Dict[str, Any]]) -> None:
    if not sources:
        st.caption("No sources were recorded.")
        return
    frame = pd.DataFrame([{"ID": s["id"], "Title": s.get("title") or "", "Provider": s.get("provider") or "",
                           "Access": str(s.get("access") or "").replace("_", " ").lower(),
                           "Published": str(s.get("published") or "")[:10], "Link": s.get("url") or ""}
                          for s in sources.values()])
    st.dataframe(frame, hide_index=True, width="stretch",
                 column_config={"Title": st.column_config.TextColumn(width="large"),
                                "Link": st.column_config.LinkColumn(display_text="open")})


def coverage(data: Dict[str, Any]) -> None:
    ledger = [row for row in data.get("ledger") or [] if isinstance(row, dict)]
    if not ledger:
        st.caption("Coverage by question is recorded for deep research runs (quick, standard or deep).")
        return
    frame = pd.DataFrame([{"Question": row.get("text"), "Status": str(row.get("status") or "").replace("_", " "),
                           "Evidence": str(row.get("evidence_state") or "").replace("_", " "),
                           "Passages": len(row.get("passages") or []), "Supporting": len(row.get("supporting") or []),
                           "Contradicting": len(row.get("contradicting") or []),
                           "Independent publishers": row.get("independent_publishers")} for row in ledger])
    st.dataframe(frame, hide_index=True, width="stretch", column_config={"Question": st.column_config.TextColumn(width="large")})
    providers = [p for p in data.get("provider_coverage") or [] if isinstance(p, dict)]
    if providers:
        st.markdown("**Search providers**")
        st.dataframe(pd.DataFrame([{"Provider": p.get("name"), "State": p.get("state"), "Queries": p.get("queries"),
                                    "Failed": p.get("failed"), "Hits": p.get("hits"), "Detail": p.get("detail")}
                                   for p in providers]), hide_index=True, width="stretch")


def limits(data: Dict[str, Any]) -> None:
    shown = False
    for title, key in (("Limitations", "limitations"), ("Gaps", "gaps")):
        items = [x for x in data.get(key) or [] if x]
        if items:
            shown = True
            st.markdown(f"**{title}**")
            for item in items:
                st.markdown(f"- {esc(item)}")
    disagreements = [d for d in data.get("disagreements") or [] if isinstance(d, dict)]
    if disagreements:
        shown = True
        st.markdown("**Disagreements between sources**")
        for d in disagreements:
            tag = "detected by the verifier" if d.get("verified") else "identified by the model, not independently checked"
            st.markdown(f"- {esc(d.get('text'))} ({tag})")
    errors = [e for e in data.get("search_errors") or [] if isinstance(e, dict)]
    if errors:
        shown = True
        st.markdown(f"**Search errors** ({len(errors)})")
        st.dataframe(pd.DataFrame([{"Provider": e.get("provider"), "Query": e.get("query"), "Error": e.get("error")}
                                   for e in errors]), hide_index=True, width="stretch")
    if not shown:
        st.caption("No limitations, gaps, disagreements or search errors were recorded.")


# ================================================================================================= E. models
def models() -> None:
    page_header("Model & Integrations", "Where the language model runs and which services the Studio uses.")
    status = load(cached_status)
    info = runtime(status)
    with st.container(border=True):
        left, right = st.columns([3, 1], vertical_alignment="center")
        with left:
            st.markdown(f"### {esc(status.get('model'))}")
            st.markdown(f"{':orange-badge[Remote service]' if info['remote'] else ':green-badge[Local]'} "
                        f"{':green-badge[Ready]' if info['ready'] else ':red-badge[Not ready]'} {esc(info['where'])}")
            if info["problem"]:
                st.caption(info["problem"])
        with right:
            if st.button("Check again", icon=":material/refresh:", width="stretch"):
                cached_status.clear()
                cached_integrations.clear()
                st.rerun()
        if info["remote"]:
            st.caption("Every model call sends prompts, the question and retrieved passages to this service. The API key "
                       "stays in the Studio server process and is never shown here.")
        else:
            st.caption("Every model call goes to Ollama on this computer; nothing is sent to a cloud model.")
        st.caption("The Studio's verdict quality was evaluated with qwen3:4b on a local Ollama only, and only for the "
                   "legacy quote_stance verifier; the Arm C default has not been evaluated in the Studio. No other "
                   "model is substituted when the configured one fails.")

    verification_panel(status)

    payload = load(cached_integrations)
    groups = {c["id"]: c for c in payload.get("categories", [])}
    for category in groups.values():
        items = [i for i in payload.get("integrations", []) if i.get("category") == category["id"]]
        if not items:
            continue
        st.subheader(category["title"])
        cols = st.columns(2)
        for index, item in enumerate(items):
            state = (item.get("status") or {})
            tone = {"ready": "green", "error": "red", "unavailable": "orange"}.get(state.get("state"), "gray")
            with cols[index % 2].container(border=True):
                st.markdown(f"**{esc(item.get('name'))}** :{tone}-badge[{esc(state.get('label'))}]")
                if state.get("detail"):
                    st.caption(state["detail"])
                facts = {k: v for k, v in (item.get("facts") or {}).items() if v not in (None, "")}
                if facts:
                    st.dataframe(pd.DataFrame({"": list(facts), "Value": [str(v) for v in facts.values()]}),
                                 hide_index=True, width="stretch")
                if item.get("setup"):
                    with st.expander("Setup"):
                        for step in item["setup"]:
                            st.markdown(f"- {step}")

    with st.expander("Use a different model", icon=":material/tune:"):
        st.markdown("The model is chosen when the Studio starts, from `.env` in the project folder "
                    "(copy `.env.example` to `.env`). After editing it, restart the launcher.")
        st.code('# local Ollama (default): leave the provider unset, or\nWEBRAG_LLM_MODEL=qwen3:4b\n\n'
                '# an OpenAI-compatible API\nWEBRAG_LLM_PROVIDER=openai\nWEBRAG_LLM_BASE_URL=https://api.example.com/v1\n'
                'WEBRAG_LLM_MODEL=model-name\nWEBRAG_LLM_API_KEY=your-key', language="bash")


def verification_panel(status: Dict[str, Any]) -> None:
    """The verification workflow the running Studio reports (``/api/status``), what it does and its limits. Every value
    shown comes from the backend; nothing here can change it (the options are fixed when the Studio starts)."""
    sources = status.get("option_sources") or {}
    verifier = status.get("verifier") or "not reported"
    guard, attest = status.get("injection_guard"), status.get("quote_attestation")
    with st.container(border=True):
        st.markdown("**Verification workflow (as the running Studio reports it)**")
        st.markdown(f"Active: **{esc(verifier)}** :blue-badge[{esc(status.get('verifier_kind') or 'not reported')}]")
        if status.get("verifier_description"):
            st.caption(esc(status["verifier_description"]) + ".")
        rows = [("Claim verifier", verifier, sources.get("verifier", "—")),
                ("Injection guard", guard if guard is not None else "not reported", sources.get("injection_guard", "—")),
                ("Quote attestation", ("on (record only)" if attest else "off") if attest is not None else "not reported",
                 sources.get("quote_attestation", "—")),
                ("Document verifier", status.get("document_verifier") or "—", "fixed")]
        st.dataframe(pd.DataFrame(rows, columns=["Option", "Value", "Set by"]), hide_index=True, width="stretch")
        if str(verifier).startswith("arm_c"):
            st.markdown("**What Arm C does with each statement**\n"
                        "1. One verifier call judges the retrieved passages (supports / contradicts / neutral, each "
                        "decisive stance with an exact quote from its passage).\n"
                        "2. If the passages agree, that is the verdict.\n"
                        "3. If they conflict, an adjudicator decides which side tests the exact claim. The statement "
                        "is marked *adjudicated*, both sides stay listed, and a supported statement is cautioned and kept "
                        "out of the direct answer and summary.\n"
                        "4. If no passage took a side, a planner may run up to 2 searches of this run's evidence; new "
                        "passages are judged the same way before adjudication.\n"
                        "5. If nothing is decisive, the statement stays INSUFFICIENT. Service failures and unusable "
                        "output are shown as failures and are never replaced by another verifier.")
        if verifier == "arm_c_studio":
            st.caption("Studio-adapted variant: the historical Arm C (arm_c) lets the adjudicator decide even when no "
                       "passage was judged for or against, accepts an adjudication citing a passage on the other side, "
                       "and reads an invalid stance as neutral; this variant does not accept such decisions and fails "
                       "those claims closed. Passages the verifier's reply leaves out are read as neutral in both.")
        with st.expander("Alternatives (set WEBRAG_VERIFIER in .env, then restart)"):
            st.markdown("- `arm_c`: the historical Arm C exactly.\n"
                        "- `arm_c0`: historical arm C0 (adjudication, no planner or search).\n"
                        "- `quote_stance`: the previous default; conflicting evidence stays INSUFFICIENT.\n"
                        "- `quote_stance_ranked`: quote_stance, conflicts decided by the first-ranked passage with a "
                        "stance. Not Arm C.\n"
                        "- `quote_stance_adjudicated`: quote_stance with an adjudicator. Not Arm C; not recommended.")
        with st.expander("Historical research results (not Studio performance)"):
            st.markdown("Arm C comes from a pre-specified study on 188 SciFact development claims that earlier project "
                        "systems had already used, so it is not an untouched test set. Historical results there: "
                        "macro-F1 0.7551 for Arm C against 0.7153 for the single-pass rank baseline (difference +0.040, "
                        "95% interval -0.004 to +0.085, permutation p = 0.081: **not significant**); arm C0, without "
                        "the planner's search, scored 0.7576. False support rose from 10/64 (15.6%) to 13/64 (20.3%), "
                        "and commitment on claims without gold evidence from 0.571 to 0.705.")
            st.caption("These numbers describe SciFact abstracts with the study's retriever. The Studio judges web "
                       "passages; its accuracy with Arm C has not been evaluated and may be lower or higher.")


# ================================================================================================= F. diagnostics
def diagnostics() -> None:
    page_header("Settings & Diagnostics", "Local configuration, data locations and troubleshooting.", link=False)
    from webrag.model_service import ModelConfig, settings_overview
    overview = settings_overview()
    config = ModelConfig.from_env()
    problems = config.problems()
    try:
        status = cached_status(client().base)
        reachable, error = True, ""
    except StudioError as exc:
        status, reachable, error = {}, False, str(exc)

    with st.container(horizontal=True):
        st.metric("Studio backend", "Online" if reachable else "Offline", border=True, help=client().base)
        st.metric("Dashboard", f"{st.get_option('server.address')}:{st.get_option('server.port')}", border=True,
                  help="Streamlit serves this page on this computer only.")
        st.metric(".env file", "Found" if overview["env_file_exists"] else "Not created", border=True,
                  help="Optional. Copy .env.example to .env to change the model.")
        st.metric("Configuration", "Valid" if not problems else f"{len(problems)} problem(s)", border=True)
    for problem in problems:
        st.error(problem)
    if error:
        st.warning(error)

    left, right = st.columns(2)
    with left.container(border=True):
        st.markdown("**Configuration (as read by this dashboard)**")
        rows = [("Model provider", "local Ollama" if not config.remote else config.provider),
                ("Model", config.model), ("Service", config.endpoint or "—"),
                ("Settings set", ", ".join(overview["set"]) or "none (defaults)")]
        rows += [(name, "set" if value else "not set") for name, value in overview["secrets"].items()]
        st.dataframe(pd.DataFrame(rows, columns=["Setting", "Value"]), hide_index=True, width="stretch")
        st.caption("Secrets are shown only as set / not set. The Studio reads the same .env when it starts; restart "
                   "it after changes.")
    with right.container(border=True):
        st.markdown("**Data locations**")
        rows = [("Saved runs", status.get("runs_folder") or "—"), ("Documents", status.get("docs_folder") or "—"),
                ("Studio", client().base),
                ("Dashboard", f"http://{st.get_option('server.address')}:{st.get_option('server.port')}")]
        st.dataframe(pd.DataFrame(rows, columns=["", "Location"]), hide_index=True, width="stretch")
        st.caption("Runs contain text fetched from the web; the outputs/ folder is not versioned.")

    with st.container(border=True):
        st.markdown("**Troubleshooting**")
        st.markdown(
            "- **Studio backend offline** — run `start_studio.bat`, or `.venv\\Scripts\\python.exe -m webrag.server`.\n"
            "- **Ollama not reachable** — start the Ollama app; check with `ollama list`.\n"
            "- **Model not installed** — `ollama pull qwen3:4b`.\n"
            "- **Port in use** — `start_studio.bat --studio-port 8766 --dashboard-port 8502`.\n"
            "- **Missing packages** — `.venv\\Scripts\\python.exe -m pip install -r requirements.txt`.\n"
            "- **First run is slow** — the two retrieval models (about 180 MB) are downloaded once.")
        st.markdown("**Tests**")
        st.code(".venv\\Scripts\\python.exe -m pytest -q\nnode --test tests/js/*.test.mjs", language="text")
    st.caption(f"Checked {time.strftime('%H:%M:%S')}")
