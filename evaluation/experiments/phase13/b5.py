"""B5: combination of components that independently passed the pre-registered rule.

The composition is read from results/phase13_improvement/b5_definition.json,
which is written only after B1-B4 were scored. The guard component (if any) is
applied by deterministic re-aggregation in scoring, like B3.
"""
from __future__ import annotations

import json

from evaluation.experiments.phase13.common import OUTPUT_DIR
from evaluation.experiments.phase13.run_experiment import register_variant


def load_definition() -> dict:
    return json.loads((OUTPUT_DIR / "b5_definition.json").read_text(encoding="utf-8"))


def register() -> None:
    spec = load_definition()
    register_variant("B5", {"evidence_limit": spec["evidence_limit"], "context_window": spec["context_window"],
                            "prompt": spec["prompt"], "description": spec["changed_component"]})
