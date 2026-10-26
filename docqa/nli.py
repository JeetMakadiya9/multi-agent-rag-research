"""Optional natural-language-inference (NLI) pre-filter for ``verifier_v2``.

A small cross-encoder scores a (passage, claim) pair as entailment, neutral or contradiction. It is fast and
deterministic, so ``verifier_v2`` uses it next to the language model:

* a confidently neutral passage is skipped without a model call;
* a contradiction the NLI model does not see is kept but counted as weak.

The model is loaded from the local Hugging Face cache only, unless downloads are explicitly allowed
(``WEBRAG_NLI_DOWNLOAD=1``). When it is not available, ``verifier_v2`` runs without it and says so in its trace.

Settings: ``WEBRAG_NLI_MODEL`` (default ``cross-encoder/nli-deberta-v3-xsmall``; ``off`` disables the filter).
"""
from __future__ import annotations

import os
import threading
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-xsmall"
NLI_LABELS = ("entailment", "neutral", "contradiction")


@dataclass(frozen=True)
class NLIScore:
    entailment: float
    neutral: float
    contradiction: float

    @property
    def label(self) -> str:
        return max(NLI_LABELS, key=lambda name: getattr(self, name))

    def as_dict(self) -> Dict[str, float]:
        return {k: round(v, 4) for k, v in asdict(self).items()}


class CrossEncoderNLI:
    """A sentence-transformers cross-encoder trained for NLI, loaded lazily on first use."""

    def __init__(self, model_name: str = DEFAULT_NLI_MODEL, allow_download: bool = False):
        self.model_name, self.allow_download = model_name, allow_download
        self.error = ""
        self._model: Any = None
        self._labels: List[str] = []
        self._lock = threading.Lock()

    def _load(self) -> bool:
        with self._lock:
            if self._model is not None:
                return True
            if self.error:
                return False
            try:
                from sentence_transformers import CrossEncoder
                model = CrossEncoder(self.model_name, local_files_only=not self.allow_download)
                id2label = getattr(getattr(model, "config", None), "id2label", None) or \
                    getattr(getattr(model, "model", None), "config", None).id2label
                labels = [str(id2label[i]).lower() for i in range(len(id2label))]
                if sorted(labels) != sorted(NLI_LABELS):
                    raise ValueError(f"not an NLI model (labels {labels})")
                self._model, self._labels = model, labels
                return True
            except Exception as exc:                      # missing from the cache, no library, not an NLI model
                self.error = f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"
                return False

    @property
    def available(self) -> bool:
        return self._load()

    def score(self, pairs: Sequence[Tuple[str, str]]) -> List[NLIScore]:
        """Scores for (premise, hypothesis) pairs, here (passage, claim). Raises RuntimeError when unavailable."""
        if not pairs:
            return []
        if not self._load():
            raise RuntimeError(f"NLI model unavailable: {self.error}")
        probs = self._model.predict(list(pairs), apply_softmax=True)
        out = []
        for row in probs:
            values = {label: float(p) for label, p in zip(self._labels, row)}
            out.append(NLIScore(values["entailment"], values["neutral"], values["contradiction"]))
        return out

    def describe(self) -> Dict[str, Any]:
        return {"model": self.model_name, "available": self.available, "error": self.error}


def nli_from_env(env: Optional[Mapping[str, str]] = None) -> Optional[CrossEncoderNLI]:
    """The configured NLI filter, or None when it is switched off. Availability is checked on first use."""
    env = os.environ if env is None else env
    name = str(env.get("WEBRAG_NLI_MODEL", DEFAULT_NLI_MODEL) or DEFAULT_NLI_MODEL).strip()
    if name.lower() in ("off", "none", "0", ""):
        return None
    return CrossEncoderNLI(name, allow_download=str(env.get("WEBRAG_NLI_DOWNLOAD", "0")).strip() == "1")
