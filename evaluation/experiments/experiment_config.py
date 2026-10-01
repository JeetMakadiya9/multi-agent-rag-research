"""Reproducible Phase 10 settings and ablation-ready system controls."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    dataset_root: str
    split: str = "dev"
    example_limit: int | None = None
    example_ids: tuple[int, ...] | None = None
    seed: int = 0
    top_k: int = 20
    verification_evidence_limit: int = 8
    model: str = "qwen3:4b"
    temperature: float = 0.0
    maximum_controller_cycles: int = 3
    maximum_retrieval_actions: int = 2
    maximum_verification_actions: int = 2
    hybrid_retrieval: bool = True
    reranking: bool = True
    claim_verification: bool = True
    iterative_research: bool = False
    critic: bool = False
    additional_research_cycles: int = 0
    threshold_source: str = "existing frozen RAG defaults; no Phase 10 calibration"
    notes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.split not in {"train", "dev", "test"}:
            raise ValueError("split must be train, dev, or test")
        if self.example_limit is not None and self.example_limit < 1:
            raise ValueError("example_limit must be positive when supplied")
        if self.example_ids is not None:
            if not self.example_ids or any(not isinstance(value, int) or isinstance(value, bool)
                                           for value in self.example_ids):
                raise ValueError("example_ids must be a non-empty tuple of integer SciFact IDs")
            if len(self.example_ids) != len(set(self.example_ids)):
                raise ValueError("example_ids must be unique")
            if self.example_limit is not None:
                raise ValueError("Use either example_ids or example_limit, not both")
        if self.top_k < 1 or self.verification_evidence_limit < 1:
            raise ValueError("retrieval and verification limits must be positive")
        if self.maximum_controller_cycles < 1:
            raise ValueError("maximum_controller_cycles must be positive")
        if min(self.maximum_retrieval_actions, self.maximum_verification_actions,
               self.additional_research_cycles) < 0:
            raise ValueError("action and cycle limits cannot be negative")
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
