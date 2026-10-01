"""
Claim-level verification layer for the Multi-Agent RAG research project.

Pipeline:
    generated answer
        -> atomic claim extraction
        -> claim-specific evidence assessment
        -> entailment / contradiction analysis
        -> SUPPORTED / CONTRADICTED / INSUFFICIENT_EVIDENCE

This module is independent of Streamlit and can consume retrieval results
from the existing advanced RAG system.

Default LLM: local Ollama (qwen3:4b).
No paid API is required.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from llm import LLMProvider, OllamaProvider


SUPPORTED = "SUPPORTED"
CONTRADICTED = "CONTRADICTED"
INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
DEFAULT_MODEL = "qwen3:4b"


@dataclass
class Claim:
    claim_id: str
    text: str


@dataclass
class EvidenceItem:
    evidence_id: str
    text: str
    source: str = ""
    page: Optional[int] = None
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_id: str = ""
    document_id: str = ""
    chunk_id: str = ""
    retrieval_method: str = ""
    retrieval_score: Optional[float] = None
    reranker_score: Optional[float] = None
    semantic_score: Optional[float] = None
    lexical_score: Optional[float] = None
    vector_distance: Optional[float] = None
    bm25_score: Optional[float] = None
    exact_score: Optional[float] = None
    rrf_score: Optional[float] = None
    interpretation: str = "NEUTRAL"
    relevance: Optional[float] = None
    entailment: Optional[float] = None
    contradiction: Optional[float] = None
    coverage: Optional[float] = None
    supported_components: List[str] = field(default_factory=list)
    unsupported_components: List[str] = field(default_factory=list)


@dataclass
class ClaimVerification:
    claim_id: str
    claim: str
    verdict: str
    confidence: float
    supporting_evidence: List[EvidenceItem] = field(default_factory=list)
    explanation: str = ""
    contradicting_evidence: List[EvidenceItem] = field(default_factory=list)
    supported_components: List[str] = field(default_factory=list)
    unsupported_components: List[str] = field(default_factory=list)
    evidence_assessments: List[Dict[str, Any]] = field(default_factory=list)
    llm_verdict: Optional[str] = None
    validator_overrides: List[str] = field(default_factory=list)
    retry_events: List[Dict[str, Any]] = field(default_factory=list)
    coverage_matrix: List[Dict[str, Any]] = field(default_factory=list)
    contradicted_components: List[str] = field(default_factory=list)
    conflicting_components: List[str] = field(default_factory=list)
    validation_errors: List[str] = field(default_factory=list)
    raw_judge_output: str = ""
    assessment_complete: bool = True
    missing_assessments: List[str] = field(default_factory=list)


@dataclass
class VerificationReport:
    original_answer: str
    claims: List[Claim]
    verifications: List[ClaimVerification]
    verification_units: List[Dict[str, Any]] = field(default_factory=list)
    trace: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    revised_answer: Optional[str] = None

    @property
    def supported_count(self) -> int:
        return sum(v.verdict == SUPPORTED for v in self.verifications)

    @property
    def contradicted_count(self) -> int:
        return sum(v.verdict == CONTRADICTED for v in self.verifications)

    @property
    def insufficient_count(self) -> int:
        return sum(v.verdict == INSUFFICIENT_EVIDENCE for v in self.verifications)

    @property
    def support_rate(self) -> float:
        if not self.verifications:
            return 0.0
        return self.supported_count / len(self.verifications)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_answer": self.original_answer,
            "claims": [asdict(c) for c in self.claims],
            "verifications": [asdict(v) for v in self.verifications],
            "verification_units": list(self.verification_units),
            "trace": list(self.trace),
            "errors": list(self.errors),
            "metrics": dict(self.metrics),
            "revised_answer": self.revised_answer,
            "summary": {
                "supported": self.supported_count,
                "contradicted": self.contradicted_count,
                "insufficient_evidence": self.insufficient_count,
                "support_rate": self.support_rate,
            },
        }


def _extract_json_object(text: str) -> Dict[str, Any]:
    """Extract the first valid JSON object from an LLM response."""
    text = (text or "").strip()

    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    for match in re.finditer(r"\{", text):
        start = match.start()
        depth = 0
        in_string = False
        escaped = False

        for index in range(start, len(text)):
            char = text[index]

            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : index + 1]
                    try:
                        value = json.loads(candidate)
                        if isinstance(value, dict):
                            return value
                    except json.JSONDecodeError:
                        break

    raise ValueError("No valid JSON object found in model response.")


def ollama_chat(
    messages: Sequence[Dict[str, str]],
    model: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    temperature: float = 0.0,
    timeout: int = 120,
    llm_provider: Optional[LLMProvider] = None,
) -> str:
    """Generate text through an injected provider or the default Ollama one."""
    provider = llm_provider or OllamaProvider(model=model, base_url=ollama_url)
    return provider.generate(
        messages,
        temperature=temperature,
        timeout=timeout,
    ).strip()


def extract_claims(
    answer: str,
    model: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    llm_provider: Optional[LLMProvider] = None,
) -> List[Claim]:
    """Extract atomic, externally verifiable factual claims from an answer."""
    if not answer or not answer.strip():
        return []

    prompt = f"""
You are the claim extraction component of a research fact-verification system.

Extract only atomic factual claims from the answer below.

Rules:
- Split compound statements into separate claims when needed.
- Keep claims specific and independently verifiable.
- Do not create claims that are not stated or clearly implied by the answer.
- Exclude greetings, opinions, recommendations, instructions, and meta-text.
- Preserve important names, dates, numbers, and relationships exactly.
- Return JSON only.

Required JSON schema:
{{
  "claims": [
    {{"claim_id": "C1", "text": "..."}},
    {{"claim_id": "C2", "text": "..."}}
  ]
}}

ANSWER:
{answer}
"""

    raw = ollama_chat(
        [
            {"role": "system", "content": "Extract factual claims. Return JSON only."},
            {"role": "user", "content": prompt},
        ],
        model=model,
        ollama_url=ollama_url,
        llm_provider=llm_provider,
    )

    data = _extract_json_object(raw)
    raw_claims = data.get("claims", [])

    claims: List[Claim] = []
    for index, item in enumerate(raw_claims, start=1):
        if isinstance(item, str):
            text = item.strip()
            claim_id = f"C{index}"
        elif isinstance(item, dict):
            text = str(item.get("text", "")).strip()
            claim_id = str(item.get("claim_id", f"C{index}")).strip()
        else:
            continue

        if text:
            claims.append(Claim(claim_id=claim_id or f"C{index}", text=text))

    return claims


def evidence_from_retrieval_results(
    results: Sequence[Any],
    max_items: int = 8,
) -> List[EvidenceItem]:
    """Convert existing retrieval results into a stable evidence representation."""
    evidence: List[EvidenceItem] = []

    for index, result in enumerate(list(results)[:max_items], start=1):
        # Make conversion idempotent: callers may supply already-normalized
        # EvidenceItems, which should retain their text and provenance.
        if isinstance(result, EvidenceItem):
            evidence.append(result)
            continue

        if isinstance(result, dict):
            chunk = result.get("chunk", result)
            get_value = result.get
        elif hasattr(result, "text") and not hasattr(result, "chunk"):
            # Duck-type retrieval adapter records without importing the
            # adapter module (which would create a circular dependency).
            chunk = {
                "text": getattr(result, "text", ""),
                "filename": getattr(result, "filename", ""),
                "page": getattr(result, "page", None),
                "chunk_id": getattr(result, "chunk_id", "") or getattr(result, "evidence_id", ""),
            }
            get_value = lambda key, default=None: getattr(result, key, default)
        else:
            chunk = getattr(result, "chunk", {}) or {}
            get_value = lambda key, default=None: getattr(result, key, default)

        if not isinstance(chunk, dict):
            chunk = {}

        text = str(chunk.get("text", "")).strip()
        if not text:
            continue

        source = str(
            get_value("source", "")
            or get_value("source_id", "")
            or get_value("filename", "")
            or chunk.get("filename")
            or chunk.get("source_id")
            or chunk.get("source")
            or get_value("source", "")
            or ""
        )

        page_value = chunk.get("page", get_value("page", None))
        try:
            page = int(page_value) if page_value is not None else None
        except (TypeError, ValueError):
            page = None

        score_value = get_value("evidence_score", None)
        if score_value is None:
            score_value = get_value("score", None)
        if score_value is None:
            score_value = get_value("final_score", 0.0)
        try:
            score = float(score_value or 0.0)
        except (TypeError, ValueError):
            score = 0.0

        metadata = get_value("metadata", None)
        if not isinstance(metadata, dict):
            metadata = get_value("retrieval_metadata", None)
        if not isinstance(metadata, dict):
            to_dict = get_value("to_dict", None)
            if callable(to_dict):
                try:
                    metadata = to_dict()
                except Exception:
                    metadata = None
        if not isinstance(metadata, dict):
            metadata = {}
        reranker_value = get_value("reranker_score", None)
        if reranker_value is None:
            reranker_value = get_value("rerank_score", None)
        metadata = {
            **metadata,
            "chunk_id": chunk.get("chunk_id", ""),
            "section": chunk.get("section", ""),
            "parent_id": chunk.get("parent_id", ""),
        }

        evidence.append(
            EvidenceItem(
                evidence_id=str(
                    get_value("evidence_id", None)
                    or chunk.get("chunk_id")
                    or f"E{index}"
                ),
                text=text,
                source=source,
                page=page,
                score=score,
                metadata=metadata,
                source_id=str(get_value("source_id", None) or source),
                document_id=str(get_value("document_id", None) or chunk.get("document_id") or metadata.get("document_id", "")),
                chunk_id=str(chunk.get("chunk_id") or metadata.get("chunk_id", "")),
                retrieval_method=str(get_value("retrieval_method", None) or metadata.get("retrieval_method", "")),
                retrieval_score=_optional_float(get_value("retrieval_score", None), score),
                reranker_score=_optional_float(reranker_value, None),
                semantic_score=_optional_float(get_value("semantic_score", None), _optional_float(metadata.get("semantic_score"), None)),
                lexical_score=_optional_float(get_value("lexical_score", None), _optional_float(metadata.get("lexical_score"), None)),
                vector_distance=_optional_float(get_value("vector_distance", None), _optional_float(metadata.get("vector_distance"), None)),
                bm25_score=_optional_float(get_value("bm25_score", None), _optional_float(metadata.get("bm25_score"), None)),
                exact_score=_optional_float(get_value("exact_score", None), _optional_float(metadata.get("exact_score"), None)),
                rrf_score=_optional_float(get_value("rrf_score", None), _optional_float(metadata.get("rrf_score"), None)),
            )
        )

    return evidence


def _optional_float(value: Any, fallback: Optional[float]) -> Optional[float]:
    if value is None:
        return fallback
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _format_evidence(evidence: Sequence[EvidenceItem]) -> str:
    blocks = []
    for item in evidence:
        location = item.source or "Unknown source"
        if item.page is not None:
            location += f", page {item.page}"
        blocks.append(
            f"[{item.evidence_id}] {location}\n"
            f"Retrieval score: {item.score:.4f}\n"
            f"{item.text}"
        )
    return "\n\n".join(blocks)


def verify_claim(
    claim: Claim,
    evidence: Sequence[EvidenceItem],
    model: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    llm_provider: Optional[LLMProvider] = None,
) -> ClaimVerification:
    """Classify one claim against retrieved evidence."""
    if not evidence:
        return ClaimVerification(
            claim_id=claim.claim_id,
            claim=claim.text,
            verdict=INSUFFICIENT_EVIDENCE,
            confidence=1.0,
            explanation="No evidence was supplied for this claim.",
        )

    prompt = f"""
You are a strict claim-level fact verifier.

Evaluate the CLAIM using ONLY the supplied EVIDENCE.

Important distinction:
- Evidence being topically relevant is NOT enough to support a claim.
- SUPPORTED requires that the evidence entails the claim.
- CONTRADICTED requires evidence that conflicts with the claim.
- If the evidence is relevant but does not establish or contradict the claim,
  use INSUFFICIENT_EVIDENCE.
- Do not use outside knowledge.
- Do not infer missing facts.
- Confidence must describe confidence in the verdict, not source quality.

Return JSON only with this schema:
{{
  "verdict": "SUPPORTED | CONTRADICTED | INSUFFICIENT_EVIDENCE",
  "confidence": 0.0,
  "evidence_ids": ["E1"],
  "explanation": "short evidence-grounded explanation"
}}

CLAIM:
{claim.text}

EVIDENCE:
{_format_evidence(evidence)}
"""

    raw = ollama_chat(
        [
            {"role": "system", "content": "Verify claims strictly from evidence. Return JSON only."},
            {"role": "user", "content": prompt},
        ],
        model=model,
        ollama_url=ollama_url,
        llm_provider=llm_provider,
    )

    data = _extract_json_object(raw)

    verdict = str(data.get("verdict", INSUFFICIENT_EVIDENCE)).strip().upper()
    if verdict not in {SUPPORTED, CONTRADICTED, INSUFFICIENT_EVIDENCE}:
        verdict = INSUFFICIENT_EVIDENCE

    try:
        confidence = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0

    selected_ids = data.get("evidence_ids", [])
    if not isinstance(selected_ids, list):
        selected_ids = []
    selected_ids = {str(item) for item in selected_ids}

    selected_evidence = [
        item for item in evidence if item.evidence_id in selected_ids
    ]

    explanation = str(data.get("explanation", "")).strip()

    return ClaimVerification(
        claim_id=claim.claim_id,
        claim=claim.text,
        verdict=verdict,
        confidence=confidence,
        supporting_evidence=selected_evidence,
        explanation=explanation,
    )


def build_verification_report(
    answer: str,
    retrieval_results: Sequence[Any],
    model: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    max_evidence: int = 8,
    llm_provider: Optional[LLMProvider] = None,
) -> VerificationReport:
    """Extract claims and verify them against one supplied evidence set.

    Retrieval results are normalized exactly once here. Already-normalized
    EvidenceItems remain unchanged, preserving compatibility with callers
    that explicitly normalize evidence before constructing the report.
    """
    claims = extract_claims(
        answer,
        model=model,
        ollama_url=ollama_url,
        llm_provider=llm_provider,
    )

    evidence = evidence_from_retrieval_results(
        retrieval_results,
        max_items=max_evidence,
    )

    verifications = [
        verify_claim(
            claim,
            evidence,
            model=model,
            ollama_url=ollama_url,
            llm_provider=llm_provider,
        )
        for claim in claims
    ]

    return VerificationReport(
        original_answer=answer,
        claims=claims,
        verifications=verifications,
    )


def filter_unverified_claims(
    answer_or_report: str | VerificationReport,
    report: Optional[VerificationReport] = None,
) -> str:
    """Return supported claims; accept the legacy report-only call form.

    The optional original answer is accepted to match Experiment B's public
    call shape. Filtering remains conservative and emits only supported
    factual claims.
    """
    if report is None:
        if not isinstance(answer_or_report, VerificationReport):
            raise TypeError("A VerificationReport is required.")
        report = answer_or_report

    supported = [
        verification.claim
        for verification in report.verifications
        if verification.verdict == SUPPORTED
    ]

    if not supported:
        return (
            "The retrieved evidence was insufficient to verify the factual "
            "claims in the generated answer."
        )

    return "\n".join(f"- {claim}" for claim in supported)


__all__ = [
    "SUPPORTED",
    "CONTRADICTED",
    "INSUFFICIENT_EVIDENCE",
    "DEFAULT_OLLAMA_URL",
    "DEFAULT_MODEL",
    "Claim",
    "EvidenceItem",
    "ClaimVerification",
    "VerificationReport",
    "ollama_chat",
    "extract_claims",
    "evidence_from_retrieval_results",
    "verify_claim",
    "build_verification_report",
    "filter_unverified_claims",
]
