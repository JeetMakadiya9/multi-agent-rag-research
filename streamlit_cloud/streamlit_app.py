"""Multi-Agent RAG Research Studio — hosted front end (Streamlit).

Run from the repository root:  streamlit run streamlit_cloud/streamlit_app.py
On Streamlit Community Cloud set the main file path to  streamlit_cloud/streamlit_app.py  (see streamlit_cloud/README.md).

It imports the project as packages (webrag, docqa) and shows only what this machine can actually run: source search
and saved reports work anywhere; research and claim verification need the local model and the retrieval models, and
are switched off, with the reason, where those are missing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent          # the repository root, so `webrag` and `docqa` import as packages
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="Multi-Agent RAG Research Studio", page_icon=":material/fact_check:", layout="wide")

try:
    from webrag import hosted
    from webrag.report import render_markdown
except Exception as exc:                                  # a broken install must say so, not show an empty page
    st.error(f"The application could not load its modules: {type(exc).__name__}: {exc}")
    st.stop()

MODEL = "qwen3:4b"
VERDICT_COLOR = {"SUPPORTED": "green", "CONTRADICTED": "red", "INSUFFICIENT": "orange"}


@st.cache_data(ttl=60, show_spinner=False)
def capabilities() -> dict:
    return hosted.capabilities(MODEL)


@st.cache_data(ttl=600, show_spinner=False)
def search(query: str, providers: tuple, limit: int, since: str | None) -> dict:
    return hosted.search_sources(query, list(providers), limit, since)


def show_run(data: dict, key: str) -> None:
    """A run record: summary, report, statements, sources and the raw record. Shown as recorded, never re-judged."""
    counts = hosted.label_counts(data)
    cols = st.columns(5)
    cols[0].metric("Mode", "Claim verification" if data["mode"] == "verify" else "Research")
    cols[1].metric("Status", data["status"])
    cols[2].metric("Supported", counts["SUPPORTED"])
    cols[3].metric("Not supported", counts["CONTRADICTED"] + counts["INSUFFICIENT"])
    cols[4].metric("Sources", len(data.get("sources", [])))
    st.caption(f"Model: {data.get('model') or 'not recorded'} · started {data.get('started') or 'not recorded'} · "
               f"outcome: {data.get('outcome') or 'not recorded'}")
    if data.get("error"):
        st.error(f"The run recorded an error: {data['error']}")
    report, statements, sources, raw = st.tabs(["Report", "Statements", "Sources", "Raw record"])
    with report:
        try:
            st.markdown(render_markdown(data))
        except Exception as exc:
            st.warning(f"This record could not be rendered as a report ({type(exc).__name__}: {exc}). "
                       "The other tabs show its contents.")
    with statements:
        if not data.get("claims"):
            st.info("This run has no statements.")
        for claim in data.get("claims", []):
            label = claim["label"]
            title = f":{VERDICT_COLOR.get(label, 'gray')}[{label}] {claim['text']}"
            with st.expander(title):
                row = hosted.claim_rows({"claims": [claim]})[0]
                st.write({k: v for k, v in row.items() if v and k != "statement"})
                for passage in claim.get("passages") or []:
                    stance = passage.get("stance", "")
                    st.markdown(f"**{stance}** · [{passage.get('source_id', '')}]({passage.get('url', '')}) "
                                f"{passage.get('title', '')}")
                    st.text(str(passage.get("text", ""))[:1200])
    with sources:
        rows = hosted.source_rows(data)
        if rows:
            st.dataframe(rows, hide_index=True, width="stretch",
                         column_config={"url": st.column_config.LinkColumn("url")})
        else:
            st.info("This run has no sources.")
    with raw:
        st.download_button("Download run record (JSON)", json.dumps(data, ensure_ascii=False, indent=2),
                           file_name="run.json", mime="application/json", key=f"download-{key}")
        st.json(data, expanded=False)


caps = capabilities()

with st.sidebar:
    st.header("What runs here")
    active = [p for p in caps["providers"] if p["configured"]]
    st.markdown(f":green[●] **Source search**: {len(active)} of {len(caps['providers'])} providers ready")
    st.markdown(":green[●] **Saved reports**: open a run record exported from the Studio")
    llm = caps["model"]
    if caps["full_research"]:
        st.markdown(f":green[●] **Research and verification**: local model `{MODEL}` and retrieval stack found")
    else:
        st.markdown(":orange[●] **Research and verification**: not available here")
        reasons = []
        if not llm["reachable"]:
            reasons.append("no Ollama server answers on 127.0.0.1:11434")
        elif not llm["installed"]:
            reasons.append(f"Ollama does not have `{MODEL}`")
        if caps["retrieval_missing"]:
            reasons.append("retrieval packages not installed: " + ", ".join(caps["retrieval_missing"]))
        st.caption("; ".join(reasons) + ".")
    if st.button("Check again", width="stretch"):
        capabilities.clear()
        st.rerun()
    st.divider()
    st.caption("Verdicts are judgements by a 4B local model about retrieved text, not proof that a statement is true. "
               "Source code: github.com/JeetMakadiya9/multi-agent-rag-research")

st.title("Multi-Agent RAG Research Studio")
st.write("Evidence-grounded research and claim verification: every drafted statement is checked passage by passage "
         "against retrieved text, and only what that check supports is reported.")

tab_search, tab_report, tab_research, tab_about = st.tabs(
    ["Search sources", "Open a saved report", "Research and verify", "About this deployment"])

with tab_search:
    st.write("Searches the project's scholarly providers directly. Results are shown as the providers return them: "
             "no model has read or judged them, so they are leads, not findings.")
    ready = [p for p in caps["providers"] if p["configured"]]
    names = {p["id"]: p["name"] for p in ready}
    with st.form("search"):
        query = st.text_input("Search query", max_chars=hosted.MAX_QUERY_CHARS,
                              placeholder="e.g. graph neural networks drug discovery")
        chosen = st.multiselect("Sources", list(names), default=[p for p in ("openalex", "arxiv") if p in names],
                                format_func=names.get)
        c1, c2 = st.columns(2)
        limit = c1.slider("Results per source", 1, hosted.MAX_RESULTS, 5)
        since_year = c2.number_input("Published since (year, 0 = any)", min_value=0, max_value=2100, value=0, step=1)
        submitted = st.form_submit_button("Search", type="primary")
    unready = [p for p in caps["providers"] if not p["configured"]]
    if unready:
        st.caption("Not set up here: " + "; ".join(f"{p['name']} (needs {', '.join(p['missing'])})" for p in unready))
    if submitted:
        since = f"{int(since_year)}-01-01" if since_year else None
        try:
            with st.spinner("Searching…"):
                result = search(query, tuple(chosen), int(limit), since)
        except ValueError as exc:
            st.warning(str(exc).capitalize())
        else:
            st.session_state["search_result"] = result
    result = st.session_state.get("search_result")
    if result:
        for name, error in result["errors"].items():
            st.error(f"{names.get(name, name)} failed: {error}")
        if result["since_ignored"]:
            st.info("Filtered by the recorded publication date after the search (no date filter in the service): "
                    + ", ".join(names.get(n, n) for n in result["since_ignored"]) + ". Results without a date are kept.")
        count = len(result["hits"])
        st.subheader(f"{count} result{'' if count == 1 else 's'} for “{result['query']}”")
        if not result["hits"] and not result["errors"]:
            st.write("The providers returned no results.")
        for hit in result["hits"]:
            with st.container(border=True):
                st.markdown(f"**[{hit['title'] or '(untitled)'}]({hit['url']})**")
                meta = [names.get(hit["provider"], hit["provider"]), hit["kind"]]
                meta += [x for x in (hit.get("published"), hit.get("publisher")) if x]
                if hit.get("doi"):
                    meta.append(f"DOI {hit['doi'].removeprefix('https://doi.org/')}")
                st.caption(" · ".join(meta))
                if hit.get("authors"):
                    st.caption(", ".join(hit["authors"][:8]) + (" et al." if len(hit["authors"]) > 8 else ""))
                if hit.get("excluded_host"):
                    st.caption(f"Not accepted as evidence by the project's source policy ({hit['excluded_host']}).")
                text = hit.get("abstract") or hit.get("snippet")
                if text:
                    with st.expander("Abstract as returned by the provider" if hit.get("abstract") else "Snippet"):
                        st.write(text)
                else:
                    st.caption("The provider returned no abstract.")
        st.download_button("Download results (JSON)", json.dumps(result, ensure_ascii=False, indent=2),
                           file_name="search-results.json", mime="application/json")

with tab_report:
    st.write("Open a run record (`run.json`) saved by the Research Studio or the command line, under "
             "`outputs/webrag/runs/` on the machine that ran it. The record is shown as it was saved; nothing is "
             "re-run, and the file is not stored here.")
    upload = st.file_uploader("Run record", type=["json"])
    if upload is not None:
        try:
            run = hosted.load_run(upload.getvalue())
        except ValueError as exc:
            st.error(str(exc).capitalize())
        else:
            st.subheader(run["input"])
            show_run(run, "upload")

with tab_research:
    if not caps["full_research"]:
        st.warning("Research and claim verification are not available on this machine.")
        st.write(f"They need a local Ollama server with `{MODEL}` on 127.0.0.1:11434, the retrieval packages "
                 "(sentence-transformers, faiss-cpu, rank-bm25, langchain-text-splitters) and the retrieval models in "
                 "the local Hugging Face cache. A hosting service such as Streamlit Community Cloud provides none of "
                 "these, and the project never sends text to a remote model.")
        st.write("To run them, clone the repository and follow the README:")
        st.code("pip install -r requirements.txt\nollama pull qwen3:4b\npython -m webrag.server", language="text")
        st.write("Then use **Open a saved report** here to show a run record it saved.")
    else:
        st.write(f"Runs the full pipeline on this machine with `{MODEL}`. A quick run reads a few sources and takes "
                 "several minutes. The run is not saved under `outputs/`; download it below.")
        with st.form("research"):
            mode = st.radio("Mode", ["research", "verify"], horizontal=True,
                            format_func={"research": "Research a question", "verify": "Verify a claim"}.get)
            text = st.text_area("Question or claim", max_chars=2000)
            depth = st.selectbox("Depth", ["quick", "standard", "deep", None],
                                 format_func=lambda d: d.capitalize() if d else "Single pass (original pipeline)")
            go = st.form_submit_button("Run", type="primary")
        if go:
            if not text.strip():
                st.warning("Enter a question or claim.")
            else:
                with st.status("Running…", expanded=True) as status:
                    line = st.empty()

                    def progress(data: dict) -> None:
                        line.write(data.get("current") or (data["stages"][-1]["name"] if data.get("stages") else ""))

                    try:
                        run = hosted.run_pipeline(text, mode, depth, progress, MODEL)
                    except Exception as exc:
                        status.update(label="The run failed", state="error")
                        st.error(f"{type(exc).__name__}: {exc}")
                        run = None
                    else:
                        status.update(label=f"Finished: {run.get('status')}",
                                      state="complete" if not run.get("error") else "error")
                if run:
                    st.session_state["last_run"] = run
        if st.session_state.get("last_run"):
            show_run(st.session_state["last_run"], "local")

with tab_about:
    st.markdown("""
**What this deployment does**

* **Search sources**: live searches of OpenAlex, arXiv, Crossref and Europe PMC (SearXNG and Brave when configured)
  through the project's guarded network session. Wikipedia is never searched: the project does not accept it as
  evidence.
* **Open a saved report**: renders a run record with the project's own report code.
* **Research and verify**: only where the local model and retrieval models exist (see the sidebar).

**What it does not do**

* It runs no language model of its own and calls no remote model, so on a hosting service it drafts and verifies
  nothing. No result on this page is invented: a search shows provider output, a report shows a saved record.
* Searches are not saved. Without `OPENALEX_API_KEY`, OpenAlex shares a daily request budget per network address,
  which a shared host can use up (the error is shown when it happens).

Verdicts in reports are SUPPORTED (a passage supports the statement and none contradicts it), CONTRADICTED (the
reverse) or INSUFFICIENT (anything else, including disagreeing passages): judgements by a 4B model about the retrieved
text, not proof.
""")
