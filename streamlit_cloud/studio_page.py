"""Pages of the hosted app: the Research Studio itself, and a quick source search that needs no model."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

st.set_page_config(page_title="Multi-Agent RAG Research Studio", page_icon=":material/fact_check:", layout="wide")

# The Studio is served by this app at studio/, next to this page. The frame resolves that address against the page's
# own location, so it also works under a hosting prefix (Streamlit Community Cloud serves apps below /~/+/).
STUDIO_FRAME = """<!doctype html><meta charset="utf-8"><title>Research Studio</title>
<p style="font:15px system-ui,sans-serif;color:#6b665c">Opening the Research Studio…</p>
<script>
  var page = window.parent && window.parent.location ? window.parent.location.href : document.baseURI;
  location.replace(new URL("studio/", page).href);
</script>"""


def studio() -> None:
    st.html("<style>[data-testid=\"stMainBlockContainer\"]{padding:1rem 1.5rem 0;max-width:100%}</style>")
    st.iframe(STUDIO_FRAME, height=920, alt="Multi-Agent RAG Research Studio")
    st.caption("Research runs use the operator's model service and the project's search providers. Verdicts are "
               "judgements by a language model about retrieved text, not proof. Source code: "
               "github.com/JeetMakadiya9/multi-agent-rag-research")


page = st.navigation([st.Page(studio, title="Research Studio", icon=":material/fact_check:", default=True),
                      st.Page(str(HERE / "tools_page.py"), title="Quick source search", icon=":material/search:")])
page.run()
