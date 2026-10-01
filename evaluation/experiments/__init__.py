"""Phase 10 evaluation protocol for controlled A/B/C research experiments."""

from .experiment_config import ExperimentConfig
from .evaluator import run_experiment

__all__ = ["ExperimentConfig", "run_experiment"]
