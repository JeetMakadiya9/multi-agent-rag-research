"""Multi-Agent RAG — public project showcase (Streamlit).

Streamlit Community Cloud main file path: showcase/app.py (dependencies: showcase/requirements.txt).
Locally: streamlit run showcase/app.py

A static, self-contained guide to the project: it reads repository files and showcase/data/ only. It starts no server,
calls no model or search service, and needs no account. The application itself runs on the user's own computer
(start_studio.bat; see the Install page).
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import content  # noqa: E402
import pages  # noqa: E402

st.set_page_config(page_title="Multi-Agent RAG · Project showcase", page_icon=":material/biotech:", layout="wide")
st.logo(str(HERE.parent / "webrag" / "static" / "favicon.svg"), size="large", link=content.REPO_URL)
st.html("""<style>
[data-testid="stMainBlockContainer"] { padding-top: 2.4rem; max-width: 1320px; }
h1 { font-weight: 650 !important; letter-spacing: -0.015em; }
h3, h4 { font-weight: 600 !important; }
.eyebrow { color: #2bb3a3; text-transform: uppercase; letter-spacing: 0.12em; font-size: 0.72rem; font-weight: 600; }
[data-testid="stMetricValue"] { font-size: 1.5rem; font-weight: 600; }
[data-testid="stMetricLabel"] p { text-transform: uppercase; letter-spacing: 0.06em; font-size: 0.7rem; opacity: 0.75; }
</style>""")

registry = {
    "overview": st.Page(pages.overview, title="Overview", icon=":material/space_dashboard:", default=True),
    "architecture": st.Page(pages.architecture_page, title="Architecture explorer", icon=":material/hub:",
                            url_path="architecture"),
    "workflow": st.Page(pages.workflow, title="Research workflow", icon=":material/route:", url_path="workflow"),
    "components": st.Page(pages.components, title="Component laboratory", icon=":material/widgets:",
                          url_path="components"),
    "results": st.Page(pages.results_page, title="Results and evidence", icon=":material/analytics:", url_path="results"),
    "install": st.Page(pages.install, title="Install and run", icon=":material/download:", url_path="install"),
    "use": st.Page(pages.use_cases, title="Use cases and limits", icon=":material/menu_book:", url_path="use-cases"),
}
st.session_state["pages"] = registry

with st.sidebar:
    st.markdown("**Multi-Agent RAG** · project showcase")
    st.caption("Static guide: no backend, no model calls, no account. The full application runs on your own computer.")
    st.link_button("Source on GitHub", f"{content.REPO_URL}/tree/{content.BRANCH}", icon=":material/code:",
                   width="stretch")

navigation = st.navigation({"Project": [registry["overview"], registry["architecture"], registry["workflow"],
                                        registry["components"]],
                            "Evidence": [registry["results"]],
                            "Get started": [registry["install"], registry["use"]]})
navigation.run()
