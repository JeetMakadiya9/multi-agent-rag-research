"""Retired candidate verifiers (docqa/agentic_candidates.RETIRED) stay importable for reproducing recorded runs, but
the Studio never offers them; verifier_v2 is offered instead."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_every_retired_candidate_is_a_known_candidate_with_a_reason():
    from docqa.agentic_candidates import CANDIDATE_VERIFIERS, RETIRED
    assert set(RETIRED) <= set(CANDIDATE_VERIFIERS) and all(reason for reason in RETIRED.values())
    assert set(RETIRED) == {"arm_c_studio_qr", "arm_c_studio_think", "arm_c_studio_p2", "arm_c_studio_all"}


def test_the_studio_offers_verifier_v2_and_no_candidate():
    """A fresh interpreter imports the Studio exactly as it starts, then lists the verifiers it can choose from."""
    code = ("import json, webrag.server, webrag.model_service; from docqa.verify import VERIFIERS; "
            "print(json.dumps(sorted(VERIFIERS)))")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=300,
                         env={**__import__("os").environ, "WEBRAG_NLI_MODEL": "off"})
    assert out.returncode == 0, out.stderr[-2000:]
    offered = set(json.loads(out.stdout.strip().splitlines()[-1]))
    from docqa.agentic_candidates import CANDIDATE_VERIFIERS
    assert "verifier_v2" in offered and "arm_c_studio" in offered
    assert not offered & set(CANDIDATE_VERIFIERS)
