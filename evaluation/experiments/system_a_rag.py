"""System A adapter around the existing strong RAG retrieval and generator."""
from __future__ import annotations

from typing import Any

from evaluation.experiments.evaluator import SystemOutput
from evaluation.experiments.phase10e_infrastructure import OutputState
from evaluation.scifact.run_scifact_evaluation import _runtime_evidence


def run_system_a(runtime: dict[str, Any], provider: Any, config: Any) -> SystemOutput:
    """Generate a free-text baseline answer from top-k existing RAG passages.

    System A deliberately has no verifier-derived predicted label. Its answer
    is stored verbatim; classification requires a separately preregistered
    output parser and is not fabricated here.
    """
    try:
        response = provider.retrieve(runtime["claim"], limit=config.top_k)
    except Exception as exc:
        return SystemOutput(retrieval_calls=1, llm_calls=0, verification_calls=0,
                            controller_cycles=0,
                            error=f"{type(exc).__name__}: {exc}")
    refs, _ = _runtime_evidence(response.evidence)
    retrieval_status = OutputState.VALID_NONEMPTY.value if refs else OutputState.VALID_EMPTY.value
    # Keep the project's baseline answer prompt and wrapper while selecting
    # CPU-only local inference to avoid device-dependent CUDA initialization.
    from langchain_core.documents import Document
    from research_pipeline import run_experiment_a
    from .local_model import OllamaCPUProvider
    from src.baseline_rag import create_prompt, format_context

    docs = []
    for item in response.evidence[:3]:
        metadata = getattr(item, "retrieval_metadata", {}) or {}
        chunk = metadata.get("chunk", {}) if isinstance(metadata, dict) else {}
        doc_id = chunk.get("scifact_doc_id", getattr(item, "document_id", "unknown"))
        docs.append(Document(page_content=str(getattr(item, "text", "") or ""),
                             metadata={"source": f"SciFact abstract {doc_id}"}))
    context = format_context(docs)
    messages = create_prompt().format_messages(context=context, question=runtime["claim"])
    llm = OllamaCPUProvider(model=config.model, num_predict=160)
    prompt_messages = [{"role": {"human": "user", "ai": "assistant"}.get(
                            getattr(message, "type", "user"), getattr(message, "type", "user")),
                        "content": str(getattr(message, "content", ""))}
                       for message in messages]
    try:
        generated = llm.generate(prompt_messages, temperature=config.temperature, timeout=120)
    except Exception as exc:
        return SystemOutput(
            retrieved_chunks=refs, retrieval_calls=1, llm_calls=1,
            verification_calls=0, controller_cycles=0,
            retrieval_output_status=retrieval_status,
            error=f"{type(exc).__name__}: {exc}",
        )
    answer = run_experiment_a(runtime["claim"], generated, response.evidence)
    return SystemOutput(
        retrieved_chunks=refs,
        raw_output={"output_type": "free_text_rag_output", "raw_output": answer.final_answer,
                    "prompt_identifier": "src.baseline_rag.create_prompt",
                    "generation_device": "CPU"},
        retrieval_calls=1,
        verification_calls=0,
        llm_calls=llm.calls,
        controller_cycles=0,
        retrieval_output_status=retrieval_status,
    )
