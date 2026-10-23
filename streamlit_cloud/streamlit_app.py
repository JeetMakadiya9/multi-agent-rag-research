"""Multi-Agent RAG Research Studio — hosted entry point (Streamlit).

Run from the repository root:  streamlit run streamlit_cloud/streamlit_app.py
On Streamlit Community Cloud the main file path is  streamlit_cloud/streamlit_app.py  (see streamlit_cloud/README.md).

This file only assembles the server: Streamlit serves the pages in ``studio_page.py`` and, at ``studio/``, the
original Research Studio (``webrag/ui.html`` and its ``/api``), whose own server runs inside this process behind a
signed-in gateway (``webrag/hosted_studio.py``). The model service is chosen by server-side settings.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from starlette.routing import Mount

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:                                  # top-level secrets become environment variables once they are loaded
    st.secrets.load_if_toml_exists()
except Exception:                     # malformed secrets are reported by Streamlit itself; settings then use defaults
    pass

from webrag.hosted_studio import HostedConfig, allow_model_downloads, create_gateway  # noqa: E402

allow_model_downloads()               # before anything imports docqa.index
app = st.App(str(HERE / "studio_page.py"), routes=[Mount("/studio", app=create_gateway(HostedConfig.from_env()))])
