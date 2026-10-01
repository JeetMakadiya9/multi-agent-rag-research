"""System B adapter: same strong RAG path plus frozen Experiment B."""
from __future__ import annotations

from typing import Any

from evaluation.experiments.evaluator import SystemOutput
from evaluation.experiments.phase10e_infrastructure import OutputState
from evaluation.scifact.run_scifact_evaluation import _runtime_evidence


def run_system_b(runtime: dict[str, Any], provider: Any, config: Any) -> SystemOutput:
    from experiment_b_verification import ExperimentBConfig, run_experiment_b_verification
    from .local_model import OllamaCPUProvider

    try:
        response = provider.retrieve(runtime["claim"], limit=config.top_k)
    except Exception as exc:
        return SystemOutput(retrieval_calls=1, verification_calls=0,
                            llm_calls=0, controller_cycles=0,
                            error=f"{type(exc).__name__}: {exc}")
    refs, evidence = _runtime_evidence(response.evidence)
    retrieval_status = OutputState.VALID_NONEMPTY.value if refs else OutputState.VALID_EMPTY.value
    verification_config = ExperimentBConfig(
        model=config.model,
        temperature=config.temperature,
        max_evidence_items=config.verification_evidence_limit,
        unit_specific_retrieval=False,
        max_targeted_retrieval_retries=0,
        answer_revision_enabled=False,
    )
    # Experiment B's pairwise judge has a strict JSON assessment contract.
    judge = OllamaCPUProvider(model=config.model, response_format="json")
    try:
        report = run_experiment_b_verification(
            runtime["claim"], runtime["claim"], evidence[:config.verification_evidence_limit],
            retrieval_provider=None, llm_provider=judge, config=verification_config,
        )
    except Exception as exc:
        return SystemOutput(retrieved_chunks=refs, retrieval_calls=1,
                            verification_calls=1, llm_calls=judge.calls,
                            controller_cycles=0, retrieval_output_status=retrieval_status,
                            evidence_output_status=OutputState.OUTPUT_UNAVAILABLE.value,
                            error=f"{type(exc).__name__}: {exc}")
    report_dict = report.to_dict()
    verification = (report_dict.get("verifications") or [{}])[0]
    if report_dict.get("errors") or not verification.get("assessment_complete", False):
        details = report_dict.get("errors") or verification.get("missing_assessments") or [
            "no complete evidence assessment was returned"]
        return SystemOutput(
            retrieved_chunks=refs, retrieval_calls=1,
            verification_calls=1, llm_calls=judge.calls,
            controller_cycles=0,
            retrieval_output_status=retrieval_status,
            evidence_output_status=OutputState.OUTPUT_UNAVAILABLE.value,
            raw_output={"verification": verification, "trace": report_dict.get("trace", [])},
            error="Experiment B returned an incomplete assessment: " + "; ".join(map(str, details)),
            error_stage="experiment_b_verification",
        )
    selected_ids = {
        str(item.get("evidence_id"))
        for key in ("supporting_evidence", "contradicting_evidence")
        for item in verification.get(key, []) or []
        if isinstance(item, dict) and item.get("evidence_id")
    }
    id_to_doc = {row["evidence_id"]: row["scifact_doc_id"] for row in refs}
    selected_docs = list(dict.fromkeys(id_to_doc[item] for item in selected_ids
                                       if item in id_to_doc and id_to_doc[item] is not None))
    metrics = report_dict.get("metrics", {})
    return SystemOutput(
        predicted_label=verification.get("verdict"),
        retrieved_chunks=refs,
        selected_evidence_doc_ids=selected_docs,
        assessments=verification.get("evidence_assessments", []),
        retrieval_calls=1 + int(metrics.get("retrieval_calls", 0) or 0),
        verification_calls=1,
        llm_calls=judge.calls,
        controller_cycles=0,
        raw_output={"verification": verification, "trace": report_dict.get("trace", [])},
        retrieval_output_status=retrieval_status,
        evidence_output_status=(OutputState.VALID_NONEMPTY.value if selected_docs
                                else OutputState.VALID_EMPTY.value),
    )
