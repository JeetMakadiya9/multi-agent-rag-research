"""The dashboard shows the verification workflow the running Studio reports (never a hard-coded mode), Arm C's steps,
the alternatives, the historical results labelled as historical, and per-statement conflict warnings, the verifier's
note and passages found by the Arm C planner. Rendered headless; every record here is SYNTHETIC TEST DATA."""
import json
from pathlib import Path

import pytest

streamlit = pytest.importorskip("streamlit")
ROOT = Path(__file__).resolve().parent.parent


def render(body: str):
    from streamlit.testing.v1 import AppTest
    script = (f"import sys\nsys.path.insert(0, {str(ROOT)!r})\nsys.path.insert(0, {str(ROOT / 'dashboard')!r})\n"
              f"import json, views\n{body}\n")
    return AppTest.from_string(script, default_timeout=60).run()


def text_of(app) -> str:
    parts = [m.value for m in app.markdown] + [c.value for c in app.caption] + [w.value for w in app.warning]
    parts += [str(df.value) for df in app.dataframe]
    return "\n".join(parts).replace("\\", "")          # without Markdown escapes (arm\_c), for comparison


STATUS = {"verifier": "arm_c_studio", "verifier_kind": "Studio-adapted Arm C (default)",
          "verifier_description": "Arm C agentic verification, Studio-adapted variant (default): one verifier pass",
          "default_verifier": "arm_c_studio", "injection_guard": "exclude", "quote_attestation": True,
          "option_sources": {"verifier": "default", "injection_guard": "default", "quote_attestation": "default"},
          "document_verifier": "frozen_b"}


def test_the_model_page_shows_the_reported_workflow_and_labels_history_as_history():
    app = render(f"views.verification_panel(json.loads({json.dumps(json.dumps(STATUS))}))")
    assert not app.exception
    text = text_of(app)
    assert "Active: **arm_c_studio**" in text and "Studio-adapted Arm C (default)" in text
    assert "If they conflict, an adjudicator decides" in text and "up to 2 searches" in text
    assert "does not accept such decisions" in text                       # the variant's difference is stated
    assert "exclude" in text and "on (record only)" in text
    assert "not significant" in text and "has not been evaluated" in text  # historical, not Studio performance


def test_the_panel_follows_the_backend_not_a_hard_coded_label():
    legacy = dict(STATUS, verifier="quote_stance", verifier_kind="legacy fallback",
                  verifier_description="Legacy per-passage quoted stances")
    app = render(f"views.verification_panel(json.loads({json.dumps(json.dumps(legacy))}))")
    text = text_of(app)
    assert "Active: **quote_stance**" in text and "legacy fallback" in text
    assert "What Arm C does" not in text and "Studio-adapted variant:" not in text


def test_findings_show_conflict_warnings_the_verifier_note_and_search_found_passages():
    claims = [{"id": "C1", "text": "Drug X lowers blood pressure.", "label": "SUPPORTED", "sub_state": "ADJUDICATED",
               "verifier": "arm_c_studio", "strength": "adjudicated",
               "verifier_note": "Arm C agentic verification (arm_c_studio); conflicting evidence; the adjudicator judged",
               "cautions": [{"kind": "adjudicated_conflict", "detail": "kept out of the answer"}],
               "supporting_sources": ["S1"], "contradicting_sources": ["S2"],
               "passages": [{"source_id": "S1", "text": "Lowered it.", "stance": "SUPPORTS"},
                            {"source_id": "S2", "text": "No effect.", "stance": "CONTRADICTS",
                             "found_by_search": "drug X no effect"}]}]
    body = (f"views.findings('run', json.loads({json.dumps(json.dumps(claims))}), {{}}, {{0: 1}})\n"
            f"for e in __import__('streamlit').session_state: pass")
    app = render(body.replace("\nfor e in __import__('streamlit').session_state: pass", ""))
    assert not app.exception
    text = text_of(app)
    assert "Sources disagreed; an adjudicator judged one side decisive" in text
    assert "Verifier (arm_c_studio)" in text
    assert "Found by the Arm C planner's search: drug X no effect" in text
    assert "contradicting sources: **S2**" in text
