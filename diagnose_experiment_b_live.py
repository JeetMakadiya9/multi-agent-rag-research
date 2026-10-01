"""Temporary, in-process timing probe for a minimal live Experiment B run."""

from __future__ import annotations

from time import perf_counter

import claim_verification as verification
import research_pipeline as pipeline
from llm import OllamaProvider
from research_state import ResearchState


def timed_call(label, function, *args, **kwargs):
    started = perf_counter()
    print(f"{label} START", flush=True)
    try:
        value = function(*args, **kwargs)
    except Exception as exc:
        print(
            f"{label} FAIL seconds={perf_counter() - started:.3f} "
            f"error={type(exc).__name__}: {exc}",
            flush=True,
        )
        raise
    print(f"{label} END seconds={perf_counter() - started:.3f}", flush=True)
    return value


def main() -> None:
    originals = {
        "generate": OllamaProvider.generate,
        "parse": verification._extract_json_object,
        "extract": verification.extract_claims,
        "verify": verification.verify_claim,
        "report": pipeline.build_verification_report,
        "filter": pipeline.filter_unverified_claims,
    }

    def timed_generate(self, messages, **kwargs):
        size = sum(len(item.get("content", "")) for item in messages)
        return timed_call(
            f"OLLAMA_GENERATE timeout={kwargs.get('timeout', 120)} prompt_chars={size}",
            originals["generate"],
            self,
            messages,
            **kwargs,
        )

    def timed_parse(text):
        return timed_call(f"JSON_PARSE input_chars={len(text)}", originals["parse"], text)

    def timed_extract(*args, **kwargs):
        result = timed_call("CLAIM_EXTRACTION", originals["extract"], *args, **kwargs)
        print(f"CLAIM_EXTRACTION claims={len(result)}", flush=True)
        return result

    def timed_verify(claim, *args, **kwargs):
        result = timed_call(
            f"CLAIM_VERIFICATION claim_id={claim.claim_id}",
            originals["verify"],
            claim,
            *args,
            **kwargs,
        )
        print(f"CLAIM_VERIFICATION verdict={result.verdict}", flush=True)
        return result

    def timed_report(*args, **kwargs):
        return timed_call(
            "REPORT_CONSTRUCTION",
            originals["report"],
            *args,
            **kwargs,
        )

    def timed_filter(*args, **kwargs):
        return timed_call("FILTERING", originals["filter"], *args, **kwargs)

    OllamaProvider.generate = timed_generate
    verification._extract_json_object = timed_parse
    verification.extract_claims = timed_extract
    verification.verify_claim = timed_verify
    pipeline.build_verification_report = timed_report
    pipeline.filter_unverified_claims = timed_filter

    answer = "Water freezes at 0 degrees Celsius."
    evidence = [
        {
            "chunk": {
                "text": "Pure water freezes at 0 degrees Celsius at standard atmospheric pressure.",
                "filename": "synthetic.txt",
                "page": 1,
            },
            "evidence_score": 0.9,
        }
    ]
    state = ResearchState(user_request="Verify one short claim")
    provider = OllamaProvider(model="qwen3:4b")
    total_started = perf_counter()
    print("EXPERIMENT_B START", flush=True)
    try:
        result = pipeline.run_experiment_b(
            "Verify these statements.",
            answer,
            evidence,
            model="qwen3:4b",
            ollama_url="http://127.0.0.1:11434/api/chat",
            remove_unverified_claims=True,
            llm_provider=provider,
            state=state,
        )
        print(
            f"EXPERIMENT_B END seconds={perf_counter() - total_started:.3f} "
            f"error={result.error!r} report={result.report is not None} "
            f"metrics={result.verification_metrics} state_claims={len(state.claims)} "
            f"contradictions={len(state.contradictions)} "
            f"unresolved={len(state.unresolved_questions)}",
            flush=True,
        )
        print(f"FINAL_ANSWER: {result.final_answer}", flush=True)
        print(
            f"REPORT: {result.report.to_dict() if result.report else None}",
            flush=True,
        )
    finally:
        OllamaProvider.generate = originals["generate"]
        verification._extract_json_object = originals["parse"]
        verification.extract_claims = originals["extract"]
        verification.verify_claim = originals["verify"]
        pipeline.build_verification_report = originals["report"]
        pipeline.filter_unverified_claims = originals["filter"]


if __name__ == "__main__":
    main()
