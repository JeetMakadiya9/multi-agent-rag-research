"""Read the evidence the showcase displays. Benchmark numbers are read from the public Phase 11B result files in the
repository at page load (not copied); the agentic verification study's aggregates, development findings and run
examples come from showcase/data/."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
PHASE11 = "evaluation/experiments/results/phase11_final"
DATA = Path(__file__).resolve().parent / "data"


def phase11_path(name: str) -> str:
    return f"{PHASE11}/{name}"


def _json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None


def _csv(path: Path) -> Optional[List[Dict[str, str]]]:
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    except OSError:
        return None


def phase11() -> Optional[Dict[str, Any]]:
    """The Phase 11B package, or None when its files are missing (the page then says so)."""
    base = ROOT / PHASE11
    summary = _json(base / "final_results_summary.json")
    stats = _json(base / "final_statistical_analysis.json")
    accuracy = _csv(base / "charts" / "accuracy_macro_f1.csv")
    if not (summary and stats and accuracy):
        return None
    return {
        "summary": summary, "stats": stats, "accuracy": accuracy,
        "retrieval": _csv(base / "charts" / "retrieval_recall.csv") or [],
        "paired": _csv(base / "charts" / "paired_outcomes.csv") or [],
        "execution": _csv(base / "charts" / "execution_outcomes.csv") or [],
        "latency": _csv(base / "charts" / "latency.csv") or [],
        "conclusion_md": (base / "final_scientific_conclusion.md").read_text(encoding="utf-8")
        if (base / "final_scientific_conclusion.md").is_file() else "",
    }


def development_findings() -> Optional[Dict[str, Any]]:
    return _json(DATA / "development_findings.json")


def historical_study() -> Optional[Dict[str, Any]]:
    """Aggregate results of the agentic verification study (the origin of Arm C), or None when the file is missing.
    The per-claim records behind them are not in the repository; the file says how the figures were checked."""
    return _json(DATA / "historical_agentic_verification.json")


def studio_alt_evaluation() -> Optional[Dict[str, Any]]:
    """Aggregate results of the alternative (proxy, programmatic, adversarial) evaluations of the Studio verifier, or
    None when they have not been run. They are not the protocol evaluation of evaluation/studio_eval/PROTOCOL.md."""
    return _json(DATA / "studio_alt_evaluation.json")


def examples() -> List[Dict[str, Any]]:
    """Link-only recorded runs (showcase/make_examples.py), newest first."""
    folder = DATA / "examples"
    rows = [_json(p) for p in sorted(folder.glob("*.json"), reverse=True)] if folder.is_dir() else []
    return [r for r in rows if r]


def test_inventory() -> Dict[str, int]:
    """Test files and test functions present in the repository (presence only; says nothing about passing)."""
    import ast
    files = sorted((ROOT / "tests").glob("test_*.py"))
    functions = 0
    for path in files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        functions += sum(1 for node in ast.walk(tree)
                         if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"))
    js = sorted((ROOT / "tests" / "js").glob("*.test.mjs"))
    return {"python_files": len(files), "python_functions": functions, "js_files": len(js)}
