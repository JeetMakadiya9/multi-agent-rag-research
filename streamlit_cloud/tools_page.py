"""Quick source search and a saved-report viewer that need no model (a secondary page; the Research Studio is the
main page)."""
from __future__ import annotations

import streamlit as st

from webrag import hosted
from webrag.report import render_markdown

VERDICT_COLOR = {"SUPPORTED": "green", "CONTRADICTED": "red", "INSUFFICIENT": "orange"}


@st.cache_data(ttl=600, show_spinner=False)
def search(query: str, providers: tuple, limit: int, since: str | None) -> dict:
    return hosted.search_sources(query, list(providers), limit, since)


def show_run(data: dict) -> None:
    counts = hosted.label_counts(data)
    cols = st.columns(4)
    cols[0].metric("Status", data["status"])
    cols[1].metric("Supported", counts["SUPPORTED"])
    cols[2].metric("Not supported", counts["CONTRADICTED"] + counts["INSUFFICIENT"])
    cols[3].metric("Sources", len(data.get("sources", [])))
    if data.get("error"):
        st.error(f"The run recorded an error: {data['error']}")
    report, statements, sources = st.tabs(["Report", "Statements", "Sources"])
    with report:
        try:
            st.markdown(render_markdown(data))
        except Exception as exc:
            st.warning(f"This record could not be rendered as a report ({type(exc).__name__}: {exc}).")
    with statements:
        for claim in data.get("claims", []):
            with st.expander(f":{VERDICT_COLOR.get(claim['label'], 'gray')}[{claim['label']}] {claim['text']}"):
                for passage in claim.get("passages") or []:
                    st.markdown(f"**{passage.get('stance', '')}** · [{passage.get('source_id', '')}]({passage.get('url', '')})")
                    st.text(str(passage.get("text", ""))[:1200])
    with sources:
        rows = hosted.source_rows(data)
        if rows:
            st.dataframe(rows, hide_index=True, width="stretch", column_config={"url": st.column_config.LinkColumn("url")})


st.title("Quick source search")
st.write("Searches the project's scholarly providers directly, without a model. Results are shown as the providers "
         "return them: leads, not findings. For planned research with verified findings, use the Research Studio page.")

providers = [p for p in hosted.search_providers() if p["configured"]]
names = {p["id"]: p["name"] for p in providers}
with st.form("search"):
    query = st.text_input("Search query", max_chars=hosted.MAX_QUERY_CHARS)
    chosen = st.multiselect("Sources", list(names), default=[p for p in ("openalex", "arxiv") if p in names],
                            format_func=names.get)
    c1, c2 = st.columns(2)
    limit = c1.slider("Results per source", 1, hosted.MAX_RESULTS, 5)
    since_year = c2.number_input("Published since (year, 0 = any)", min_value=0, max_value=2100, value=0, step=1)
    submitted = st.form_submit_button("Search", type="primary")
if submitted:
    try:
        with st.spinner("Searching…"):
            st.session_state["search_result"] = search(query, tuple(chosen), int(limit),
                                                       f"{int(since_year)}-01-01" if since_year else None)
    except ValueError as exc:
        st.warning(str(exc).capitalize())
result = st.session_state.get("search_result")
if result:
    for name, error in result["errors"].items():
        st.error(f"{names.get(name, name)} failed: {error}")
    if result["since_ignored"]:
        st.info("Filtered by the recorded publication date after the search: "
                + ", ".join(names.get(n, n) for n in result["since_ignored"]) + ". Results without a date are kept.")
    count = len(result["hits"])
    st.subheader(f"{count} result{'' if count == 1 else 's'} for “{result['query']}”")
    for hit in result["hits"]:
        with st.container(border=True):
            st.markdown(f"**[{hit['title'] or '(untitled)'}]({hit['url']})**")
            meta = [names.get(hit["provider"], hit["provider"]), hit["kind"]]
            meta += [x for x in (hit.get("published"), hit.get("publisher")) if x]
            st.caption(" · ".join(meta))
            text = hit.get("abstract") or hit.get("snippet")
            if text:
                with st.expander("Abstract as returned by the provider" if hit.get("abstract") else "Snippet"):
                    st.write(text)

st.divider()
st.subheader("Open a saved run record")
st.caption("A run.json exported from a Studio. It is shown as saved, not re-run, and not stored here.")
upload = st.file_uploader("Run record", type=["json"])
if upload is not None:
    try:
        show_run(hosted.load_run(upload.getvalue()))
    except ValueError as exc:
        st.error(str(exc).capitalize())
