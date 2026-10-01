"""
Research-layer orchestration for the Multi-Agent RAG project.

This module does NOT replace the existing RAG engine.
It wraps an already-generated RAG answer with claim-level verification.

Experiment A:
    Existing advanced RAG answer is used as-is.

Experiment B:
    Existing advanced RAG answer
        -> atomic claim extraction
        -> claim-level evidence verification
        -> verification report

The module is intentionally independent of Streamlit so it can be used by
both the application and later benchmark/evaluation scripts.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, Sequence

from claim_verification import (
    Claim,
    ClaimVerification,
    EvidenceItem,
    VerificationReport,
    _extract_json_object,
    build_verification_report,
    evidence_from_retrieval_results,
    filter_unverified_claims,
)
from llm import LLMProvider
from research_state import Evidence, ResearchClaim, ResearchState
from experiment_b_verification import ExperimentBConfig, run_experiment_b_verification


@dataclass
class ResearchPipelineResult:
    """Complete result for one query."""

    question: str
    baseline_answer: str
    final_answer: str
    report: Optional[VerificationReport] = None
    retrieval_count: int = 0
    verification_latency_seconds: float = 0.0
    verification_enabled: bool = False
    error: Optional[str] = None
    retrieval_latency_seconds: float = 0.0
    retrieval_calls: int = 0
    llm_calls: int = 0

    @property
    def original_answer(self) -> str:
        """Compatibility-friendly name for the supplied baseline answer."""
        return self.baseline_answer

    @property
    def verification_metrics(self) -> Dict[str, Any]:
        """Claim counts and support rate for the attached report, if any."""
        if self.report is None:
            return {
                "claim_count": 0,
                "verified_claim_count": 0,
                "supported_claims": 0,
                "contradicted_claims": 0,
                "insufficient_claims": 0,
                "support_rate": 0.0,
                "unsupported_claim_rate": 0.0,
                "contradiction_rate": 0.0,
                "retrieval_calls": self.retrieval_calls,
                "llm_calls": self.llm_calls,
            }
        return {
            "claim_count": len(self.report.claims),
            "verified_claim_count": len(self.report.verifications),
            "supported_claims": self.report.supported_count,
            "contradicted_claims": self.report.contradicted_count,
            "insufficient_claims": self.report.insufficient_count,
            "support_rate": self.report.support_rate,
            "unsupported_claim_rate": self.report.metrics.get("unsupported_claim_rate"),
            "contradiction_rate": self.report.metrics.get("contradiction_rate"),
            "retrieval_calls": self.report.metrics.get("retrieval_calls", self.retrieval_calls),
            "llm_calls": self.report.metrics.get("llm_calls", self.llm_calls),
            "evidence_precision": self.report.metrics.get("evidence_precision"),
            "evidence_recall": self.report.metrics.get("evidence_recall"),
            "evidence_coverage": self.report.metrics.get("evidence_coverage"),
            "citation_correctness": self.report.metrics.get("citation_correctness"),
            "citation_completeness": self.report.metrics.get("citation_completeness"),
            "evidence_alignment": self.report.metrics.get("evidence_alignment"),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the result while retaining the former ``original_answer`` key."""
        return {
            "question": self.question,
            "baseline_answer": self.baseline_answer,
            "original_answer": self.original_answer,
            "final_answer": self.final_answer,
            "report": self.report.to_dict() if self.report is not None else None,
            "retrieval_count": self.retrieval_count,
            "verification_latency_seconds": self.verification_latency_seconds,
            "verification_enabled": self.verification_enabled,
            "error": self.error,
            "retrieval_latency_seconds": self.retrieval_latency_seconds,
            "retrieval_calls": self.retrieval_calls,
            "llm_calls": self.llm_calls,
            "verification_metrics": self.verification_metrics,
        }


class _LegacyProviderCompatibilityAdapter:
    """Adapt the former extraction-then-verdict provider call sequence.

    The current verifier continues to own verification-unit boundaries and
    consumes only the current judge schema. This adapter recognizes an
    extraction-shaped response from providers that still return the former
    first call, retains it as diagnostic context, requests the actual judgment,
    and normalizes the former verdict field names at the provider boundary.
    """

    def __init__(self, provider: LLMProvider):
        self.provider = provider
        self.call_count = 0
        self.extraction_responses: list[list[dict[str, str]]] = []
        self.normalization_count = 0

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float = 0.0,
        timeout: int = 120,
    ) -> str:
        raw = self._generate(messages, temperature=temperature, timeout=timeout)
        payload = self._json_object(raw)

        # The legacy pipeline made claim extraction its first provider call.
        # Keep that response and supply it as compatibility context to the
        # subsequent judge call without allowing it to replace/split units.
        if payload is not None and isinstance(payload.get("claims"), list) and "verdict" not in payload:
            claims = [
                {"claim_id": str(item.get("claim_id", "")), "text": str(item.get("text", ""))}
                for item in payload["claims"]
                if isinstance(item, dict) and str(item.get("text", "")).strip()
            ]
            if claims:
                self.extraction_responses.append(claims)
                compatibility_message = {
                    "role": "user",
                    "content": (
                        "Compatibility context from the legacy claim-extraction call (retained for "
                        "call-sequence compatibility): " + json.dumps(claims, ensure_ascii=False) +
                        ". Judge the existing verification unit unchanged; do not replace, split, merge, "
                        "or omit that unit. This context is not evidence."
                    ),
                }
                raw = self._generate(
                    [*messages, compatibility_message],
                    temperature=temperature,
                    timeout=timeout,
                )
                payload = self._json_object(raw)

        if payload is None or "verdict" not in payload:
            return raw

        normalized = self._normalize_legacy_verdict(payload)
        if normalized != payload:
            self.normalization_count += 1
            return json.dumps(normalized, ensure_ascii=False)
        return raw

    def _generate(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float,
        timeout: int,
    ) -> str:
        self.call_count += 1
        return self.provider.generate(messages, temperature=temperature, timeout=timeout)

    @staticmethod
    def _json_object(raw: str) -> Optional[dict[str, Any]]:
        try:
            value = _extract_json_object(raw)
        except (TypeError, ValueError):
            return None
        return value if isinstance(value, dict) else None

    @staticmethod
    def _normalize_legacy_verdict(payload: dict[str, Any]) -> dict[str, Any]:
        result = dict(payload)
        verdict = str(result.get("verdict", "")).upper()
        evidence_ids = result.get("evidence_ids", [])
        if not isinstance(evidence_ids, list):
            evidence_ids = []

        if "supporting_evidence_ids" not in result:
            result["supporting_evidence_ids"] = (
                [str(item) for item in evidence_ids]
                if verdict == "SUPPORTED" else []
            )
        if "contradicting_evidence_ids" not in result:
            result["contradicting_evidence_ids"] = (
                [str(item) for item in evidence_ids]
                if verdict == "CONTRADICTED" else []
            )
        result.setdefault("supported_components", [])
        result.setdefault("unsupported_components", [])
        result.setdefault("evidence_assessments", [])
        if "reason" not in result and "explanation" in result:
            result["reason"] = result["explanation"]
        return result

    def compatibility_trace(self, units: Sequence[dict[str, Any]]) -> dict[str, Any]:
        """Return auditable extraction-to-unit associations, without rewriting units."""
        normalized_units = [
            (str(unit.get("id", "")), self._normalize_text(str(unit.get("original_text", ""))))
            for unit in units
        ]
        associations = []
        for response in self.extraction_responses:
            for claim in response:
                text = self._normalize_text(claim["text"])
                matched = [
                    unit_id for unit_id, unit_text in normalized_units
                    if text and unit_text and (text in unit_text or unit_text in text)
                ]
                associations.append({
                    "legacy_claim_id": claim["claim_id"],
                    "legacy_claim_text": claim["text"],
                    "matched_verification_unit_ids": matched,
                })
        return {
            "stage": "legacy_provider_compatibility",
            "extraction_responses_consumed": len(self.extraction_responses),
            "legacy_verdicts_normalized": self.normalization_count,
            "extraction_to_unit_associations": associations,
            "verification_units_rewritten": False,
        }

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join("".join(char.lower() if char.isalnum() else " " for char in text).split())

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["original_answer"] = self.original_answer
        data["verification_metrics"] = self.verification_metrics
        if self.report is not None:
            data["report"] = self.report.to_dict()
        return data


@dataclass
class EvaluationSummary:
    """Aggregate counters for a collection of pipeline runs."""

    total_questions: int = 0
    verified_questions: int = 0
    supported_claims: int = 0
    contradicted_claims: int = 0
    insufficient_claims: int = 0
    verification_errors: int = 0
    total_verification_latency_seconds: float = 0.0

    @property
    def average_verification_latency_seconds(self) -> float:
        if self.verified_questions == 0:
            return 0.0
        return self.total_verification_latency_seconds / self.verified_questions

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_questions": self.total_questions,
            "verified_questions": self.verified_questions,
            "supported_claims": self.supported_claims,
            "contradicted_claims": self.contradicted_claims,
            "insufficient_claims": self.insufficient_claims,
            "verification_errors": self.verification_errors,
            "total_verification_latency_seconds": self.total_verification_latency_seconds,
            "average_verification_latency_seconds": self.average_verification_latency_seconds,
        }


def run_experiment_a(
    question: str,
    answer: str,
    retrieval_results: Sequence[Any],
) -> ResearchPipelineResult:
    """Return the existing RAG result without changing the answer."""

    return ResearchPipelineResult(
        question=question,
        baseline_answer=answer,
        final_answer=answer,
        retrieval_count=len(retrieval_results),
        verification_enabled=False,
    )


def run_experiment_b(
    question: str,
    answer: str,
    retrieval_results: Sequence[Any],
    *,
    model: str = "qwen3:4b",
    ollama_url: str = "http://127.0.0.1:11434/api/chat",
    max_evidence_items: int = 8,
    remove_unverified_claims: bool = False,
    llm_provider: Optional[LLMProvider] = None,
    state: Optional[ResearchState] = None,
    retrieval_provider: Any = None,
    verification_config: Optional[ExperimentBConfig] = None,
    revise_answer: Optional[bool] = None,
) -> ResearchPipelineResult:
    """
    Experiment B contract for an existing strong-RAG answer:

    Inputs are the user question, an already-generated baseline answer, and
    baseline retrieval results. Verification units preserve qualifiers and
    dependencies. When ``retrieval_provider`` is supplied, each unit triggers
    claim-specific retrieval and may receive one bounded targeted retry.

    Experiment B never generates or changes the baseline RAG answer in place.
    The resulting report contains a separate verified/revised answer. Pass a
    retrieval provider to enable mandatory unit-specific retrieval; without
    one, the legacy API remains usable with the supplied initial evidence and
    records that targeted retrieval was unavailable.

    By default the original answer remains ``final_answer`` for API
    compatibility; ``report.revised_answer`` carries the verified version.
    Set ``revise_answer=True`` or enable it in ``verification_config`` to use
    the verified version as ``final_answer``. The legacy
    ``remove_unverified_claims`` option remains supported.
    """

    start = perf_counter()

    try:
        compatibility_provider = (
            _LegacyProviderCompatibilityAdapter(llm_provider)
            if llm_provider is not None else None
        )
        config = replace(verification_config) if verification_config is not None else ExperimentBConfig(
            model=model, ollama_url=ollama_url, max_evidence_items=max_evidence_items,
            answer_revision_enabled=True,
        )
        if revise_answer is not None:
            config.answer_revision_enabled = revise_answer
        report = run_experiment_b_verification(
            question, answer, retrieval_results,
            retrieval_provider=retrieval_provider,
            llm_provider=compatibility_provider or llm_provider,
            config=config,
        )

        if compatibility_provider is not None:
            report.metrics["llm_calls"] = compatibility_provider.call_count
            if compatibility_provider.extraction_responses or compatibility_provider.normalization_count:
                report.trace.append(compatibility_provider.compatibility_trace(report.verification_units))

        if state is not None:
            _record_report_in_state(state, report)

        use_revision = config.answer_revision_enabled and (
            revise_answer is True or verification_config is not None
        )
        final_answer = report.revised_answer if use_revision else answer
        if remove_unverified_claims:
            final_answer = filter_unverified_claims(answer, report)

        elapsed = perf_counter() - start

        return ResearchPipelineResult(
            question=question,
            baseline_answer=answer,
            final_answer=final_answer,
            report=report,
            retrieval_count=len(retrieval_results),
            verification_latency_seconds=elapsed,
            verification_enabled=True,
            retrieval_latency_seconds=float(report.metrics.get("retrieval_latency_seconds") or 0.0),
            retrieval_calls=int(report.metrics.get("retrieval_calls") or 0),
            llm_calls=int(report.metrics.get("llm_calls") or 0),
        )

    except Exception as exc:
        elapsed = perf_counter() - start
        if state is not None:
            state.errors.append(f"Experiment B: {type(exc).__name__}: {exc}")
            state.touch()
        return ResearchPipelineResult(
            question=question,
            baseline_answer=answer,
            final_answer=answer,
            retrieval_count=len(retrieval_results),
            verification_latency_seconds=elapsed,
            verification_enabled=True,
            error=f"{type(exc).__name__}: {exc}",
            retrieval_calls=0,
            llm_calls=0,
        )


def _record_report_in_state(
    state: ResearchState,
    report: VerificationReport,
) -> None:
    """Store report claims, verdicts, selected evidence, and follow-up needs."""
    verifications_by_id = {
        item.claim_id: item for item in report.verifications
    }
    recorded_evidence_ids = {item.evidence_id for item in state.evidence}

    for claim in report.claims:
        verification = verifications_by_id.get(claim.claim_id)
        if verification is None:
            continue

        evidence_ids = []
        source_ids = []
        selected_evidence = verification.supporting_evidence + verification.contradicting_evidence
        for item in selected_evidence:
            evidence_ids.append(item.evidence_id)
            if item.source:
                source_ids.append(item.source)
            if item.evidence_id not in recorded_evidence_ids:
                state.add_evidence(
                    Evidence(
                        evidence_id=item.evidence_id,
                        text=item.text,
                        source_id=item.source or "unknown",
                        relevance_score=item.score,
                        page=item.page,
                        chunk_id=str(item.metadata.get("chunk_id") or "") or None,
                        metadata=dict(item.metadata),
                    )
                )
                recorded_evidence_ids.add(item.evidence_id)

        state.add_claim(
            ResearchClaim(
                claim_id=claim.claim_id,
                text=claim.text,
                status=verification.verdict,
                confidence=verification.confidence,
                evidence_ids=evidence_ids,
                source_ids=list(dict.fromkeys(source_ids)),
                verification_reason=verification.explanation,
            )
        )

        if verification.verdict == "CONTRADICTED":
            detail = verification.explanation or "Evidence contradicts the claim."
            state.add_contradiction(f"{claim.text}: {detail}")
        elif verification.verdict == "INSUFFICIENT_EVIDENCE":
            state.request_more_research(claim.text)

    for event in report.trace:
        if hasattr(state, "log"):
            import json
            state.log("experiment_b", "verification_unit", json.dumps(event, ensure_ascii=False, default=str))


def summarize_results(results: Sequence[ResearchPipelineResult]) -> EvaluationSummary:
    """Aggregate claim-level counts and verification latency."""

    summary = EvaluationSummary()
    summary.total_questions = len(results)

    for result in results:
        if result.report is None:
            if result.error:
                summary.verification_errors += 1
            continue

        summary.verified_questions += 1
        summary.total_verification_latency_seconds += result.verification_latency_seconds
        summary.supported_claims += result.report.supported_count
        summary.contradicted_claims += result.report.contradicted_count
        summary.insufficient_claims += result.report.insufficient_count

    return summary


__all__ = [
    "ResearchPipelineResult",
    "EvaluationSummary",
    "run_experiment_a",
    "run_experiment_b",
    "summarize_results",
]
