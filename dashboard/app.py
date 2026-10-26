"""Multi-Agent RAG Research Studio — local dashboard (Streamlit).

Start everything with ``start_studio.bat`` (or ``python -m dashboard.launch``). To run only the dashboard, with the
Studio backend already running on http://127.0.0.1:8765:

    streamlit run dashboard/app.py

The dashboard is a client of the Research Studio's own API (``webrag/server.py``): it starts real runs, follows their
progress and reads saved runs; the full Studio interface opens from every page.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for path in (str(HERE), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from webrag.model_service import load_env_file  # noqa: E402

load_env_file()                                    # the same optional .env the Studio reads

import views  # noqa: E402
from client import StudioError  # noqa: E402

st.set_page_config(page_title="Research Studio · Dashboard", page_icon=":material/biotech:", layout="wide",
                   initial_sidebar_state="expanded")

st.logo(str(ROOT / "webrag" / "static" / "favicon.svg"), size="large")
st.html("""<style>
[data-testid="stMainBlockContainer"] { padding-top: 2.2rem; max-width: 1400px; }
h1 { font-weight: 600 !important; letter-spacing: -0.01em; font-size: 1.85rem !important; }
h3 { font-weight: 600 !important; }
[data-testid="stMetricValue"] { font-size: 1.45rem; font-weight: 600; }
[data-testid="stMetricLabel"] p { text-transform: uppercase; letter-spacing: 0.06em; font-size: 0.72rem; opacity: 0.75; }
[data-testid="stSidebar"] .brand { font-weight: 600; font-size: 1.05rem; letter-spacing: 0.01em; }
[data-testid="stSidebar"] .brand span { color: #2bb3a3; }
</style>""")

pages = {
    "command": st.Page(views.command_center, title="Command Center", icon=":material/space_dashboard:", default=True),
    "workspace": st.Page(views.workspace, title="Research Workspace", icon=":material/science:", url_path="workspace"),
    "library": st.Page(views.library, title="Research Library", icon=":material/collections_bookmark:", url_path="library"),
    "evidence": st.Page(views.evidence, title="Evidence Explorer", icon=":material/fact_check:", url_path="evidence"),
    "models": st.Page(views.models, title="Model & Integrations", icon=":material/memory:", url_path="models"),
    "diagnostics": st.Page(views.diagnostics, title="Settings & Diagnostics", icon=":material/tune:",
                           url_path="diagnostics"),
}
st.session_state["pages"] = pages

with st.sidebar:
    st.html("<div class='brand'>Multi-Agent RAG <span>Research Studio</span></div>")
    try:
        status = views.cached_status(views.client().base)
        info = views.runtime(status)
        st.markdown(":green-badge[Studio online]  " + (":green-badge[Model ready]" if info["ready"]
                                                         else ":orange-badge[Model not ready]"))
        st.caption(f"{status.get('model')} · {info['where']}")
    except StudioError:
        st.markdown(":red-badge[Studio offline]")
        st.caption("Start it with start_studio.bat")
    views.studio_link("Open full Studio", key="sidebar-studio")

navigation = st.navigation({"Research": [pages["command"], pages["workspace"], pages["library"], pages["evidence"]],
                            "System": [pages["models"], pages["diagnostics"]]})
navigation.run()
