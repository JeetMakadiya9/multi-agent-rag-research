"""Release regressions: the default verifier and safeguards agree across code, configuration, documentation and the
showcase; the historical Arm C and C0 modes keep the study's behaviour; recorded demonstrations keep their own verifier
labels; the evaluation harness is documented as not run."""
import json
import re
from pathlib import Path

from docqa.agentic import ArmC0Verifier, ArmCVerifier, StudioArmCVerifier
from webrag.model_service import STUDIO_INJECTION_GUARD, STUDIO_VERIFIER, studio_options

ROOT = Path(__file__).resolve().parent.parent


def read(path):
    return (ROOT / path).read_text(encoding="utf-8")


def test_the_default_is_the_same_in_code_configuration_and_documentation():
    assert STUDIO_VERIFIER == "arm_c_studio" and studio_options(env={})["verifier"] == "arm_c_studio"
    assert "WEBRAG_VERIFIER: arm_c_studio (default)" in read(".env.example")
    assert "Default verifier: Arm C agentic verification, Studio-adapted variant (`arm_c_studio`)" in read("README.md")
    assert "Studio default `arm_c_studio`" in read("webrag/README.md")
    assert "Arm C agentic verification, in a Studio-adapted variant** (`arm_c_studio`)" in read("showcase/pages.py")
    assert "quote_stance (default" not in read(".env.example")            # the previous default is not presented as it


def test_the_safeguards_are_on_by_default_everywhere():
    options = studio_options(env={})
    assert STUDIO_INJECTION_GUARD == "exclude" and options["injection_guard"] == "exclude"
    assert options["quote_attestation"] is True
    assert "WEBRAG_INJECTION_GUARD: exclude (default" in read(".env.example")


def test_the_historical_modes_keep_the_studys_behaviour():
    assert (ArmCVerifier.accept_unsided, ArmCVerifier.require_complete_assessment, ArmCVerifier.check_cited_stance) == \
        (True, False, False)
    assert (ArmC0Verifier.accept_unsided, ArmC0Verifier.require_complete_assessment,
            ArmC0Verifier.check_cited_stance) == (True, False, False)
    assert (StudioArmCVerifier.accept_unsided, StudioArmCVerifier.require_complete_assessment,
            StudioArmCVerifier.check_cited_stance) == (False, True, True)


def test_recorded_demonstrations_keep_their_own_verifier_labels():
    for path in (ROOT / "showcase" / "data" / "examples").glob("*.json"):
        assert json.loads(path.read_text(encoding="utf-8"))["verifier"] == "quote_stance", path.name


def test_the_studio_evaluation_is_documented_as_not_run_and_has_no_result():
    protocol = read("evaluation/studio_eval/PROTOCOL.md")
    assert protocol.startswith("# Research Studio verifier evaluation") and "**Status: NOT RUN.**" in protocol
    assert "has not been run" in read("README.md")
    folder = ROOT / "evaluation" / "studio_eval"
    assert not list(folder.glob("*.jsonl")) and not (folder / "FREEZE.json").exists()   # no data or labels published
    assert not re.search(r"macro-F1[^\n]{0,40}\b0\.\d{3,4}\b", protocol)              # no result figure
