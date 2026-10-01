"""Repeat only the live contradiction case, preserving Ollama's raw stream."""
from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

import llm as llm_module
import rag
from experiment_b_verification import ExperimentBConfig
from llm import OllamaProvider
from research_pipeline import run_experiment_b
from research_retrieval import ExistingRAGProvider


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "baseline_experiment_b_case2_judge_diagnostic.json"
QUESTION = "Does BM25 handle image generation?"
ANSWER = "The system uses BM25 for image generation."
EVIDENCE = "BM25 is not used for image generation; it is used for lexical document retrieval."
SOURCE = "bm25_image_generation.txt"


class CapturingResponse:
    """Transparent HTTP response proxy recording exact NDJSON bytes and events."""

    def __init__(self, response: Any, record: dict[str, Any]):
        self.response = response
        self.record = record

    def __enter__(self):
        self.response.__enter__()
        self.record["http_status"] = getattr(self.response, "status", None) or self.response.getcode()
        self.record["http_reason"] = getattr(self.response, "reason", None)
        self.record["response_headers"] = dict(self.response.headers.items())
        return self

    def __exit__(self, exc_type, exc, tb):
        return self.response.__exit__(exc_type, exc, tb)

    def __iter__(self):
        try:
            for line in self.response:
                raw = bytes(line)
                self.record["raw_http_stream_chunks_base64"].append(base64.b64encode(raw).decode("ascii"))
                self.record["raw_http_stream_text"] += raw.decode("utf-8", errors="replace")
                try:
                    event = json.loads(raw.decode("utf-8"))
                    self.record["stream_events"].append(event)
                    message = event.get("message", {})
                    content = message.get("content", "") if isinstance(message, dict) else ""
                    if content:
                        self.record["content_fragments"].append(str(content))
                    if event.get("done"):
                        self.record["done_event"] = event
                except Exception as exc:
                    self.record["transport_event_parse_error"] = f"{type(exc).__name__}: {exc}"
                yield line
        except Exception as exc:
            self.record["stream_exception"] = f"{type(exc).__name__}: {exc}"
            raise

    def __getattr__(self, name: str) -> Any:
        return getattr(self.response, name)


class CapturingOllamaProvider:
    def __init__(self, provider: OllamaProvider):
        self.provider = provider
        self.records: list[dict[str, Any]] = []
        self.active_record: dict[str, Any] | None = None
        self.pipeline_retrieval_calls = 0
        self._original_urlopen = llm_module.urllib.request.urlopen

    def _urlopen(self, request, timeout=...):
        record = self.active_record
        try:
            response = self._original_urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            if record is not None:
                record["http_status"] = exc.code
                record["http_reason"] = str(exc.reason)
                record["http_error_body_base64"] = base64.b64encode(exc.read()).decode("ascii")
            raise
        if record is None:
            return response
        record["http_status"] = getattr(response, "status", None) or response.getcode()
        return CapturingResponse(response, record)

    def generate(self, messages, *, temperature=0.0, timeout=120):
        record: dict[str, Any] = {
            "attempt_number": len(self.records) + 1,
            "retrieval_calls_completed_before_attempt": self.pipeline_retrieval_calls,
            "attempt_stage": (
                "after_targeted_retrieval" if self.pipeline_retrieval_calls >= 2 else "initial_verification"
            ),
            "temperature": temperature,
            "timeout_seconds": timeout,
            "raw_http_stream_chunks_base64": [],
            "raw_http_stream_text": "",
            "stream_events": [],
            "content_fragments": [],
            "http_status": None,
            "done_event": None,
            "stream_exception": None,
            "provider_exception": None,
        }
        same_stage_prior = sum(x["attempt_stage"] == record["attempt_stage"] for x in self.records)
        if record["attempt_stage"] == "initial_verification":
            record["attempt_type"] = "initial_judge_attempt" if same_stage_prior == 0 else "initial_judge_retry"
        else:
            record["attempt_type"] = "post_targeted_retrieval_judge_attempt" if same_stage_prior == 0 else "post_targeted_retrieval_judge_retry"
        self.records.append(record)
        self.active_record = record
        started = time.perf_counter()
        previous_urlopen = llm_module.urllib.request.urlopen
        llm_module.urllib.request.urlopen = self._urlopen
        try:
            returned = self.provider.generate(messages, temperature=temperature, timeout=timeout)
            record["provider_returned_content"] = returned
        except Exception as exc:
            record["provider_exception"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            record["elapsed_seconds"] = time.perf_counter() - started
            record["raw_response_text"] = "".join(record["content_fragments"])
            record["response_length_characters"] = len(record["raw_response_text"])
            record["stream_completed"] = bool(record["done_event"] and record["done_event"].get("done"))
            record["streamed_successfully"] = bool(
                record["http_status"] == 200
                and record["stream_completed"]
                and not record["stream_exception"]
                and not record["provider_exception"]
            )
            try:
                record["parsed_json"] = json.loads(record["raw_response_text"])
                record["json_parse_exception"] = None
            except Exception as exc:
                record["parsed_json"] = None
                record["json_parse_exception"] = f"{type(exc).__name__}: {exc}"
            parsed = record["parsed_json"]
            record["verdict_field_present"] = isinstance(parsed, dict) and "verdict" in parsed
            record["verdict_value"] = parsed.get("verdict") if isinstance(parsed, dict) else None
            record["returned_fields"] = sorted(parsed.keys()) if isinstance(parsed, dict) else []
            self.active_record = None
            llm_module.urllib.request.urlopen = previous_urlopen
        return returned


def main() -> int:
    started_at = datetime.now(timezone.utc).isoformat()
    print(f"Started UTC: {started_at}", flush=True)
    print("Loading local embedding and reranker models...", flush=True)
    embedding_model = SentenceTransformer(rag.EMBEDDING_MODEL_NAME)
    reranker = CrossEncoder(rag.RERANKER_MODEL_NAME)

    chunks = rag.create_chunks([{"page": 1, "text": EVIDENCE}], SOURCE)
    texts = [str(chunk.get("text", "")) for chunk in chunks]
    vectors = embedding_model.encode(
        texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")
    faiss_index = faiss.IndexFlatIP(vectors.shape[1])
    faiss_index.add(vectors)
    bm25_index = BM25Okapi([rag.tokenize_text(text) for text in texts])
    retrieval = ExistingRAGProvider(
        chunks=chunks, embedding_model=embedding_model, reranker=reranker,
        faiss_index=faiss_index, bm25_index=bm25_index, retrieval_module=rag,
    )

    initial_started = time.perf_counter()
    initial = retrieval.retrieve(QUESTION, limit=8)
    initial_retrieval_seconds = time.perf_counter() - initial_started
    config = ExperimentBConfig()
    config.timeout_seconds = 60
    base_provider = OllamaProvider(
        model=config.model, base_url=config.ollama_url,
        think=config.think, format=config.response_format,
    )
    capture = CapturingOllamaProvider(base_provider)
    original_retrieve = retrieval.retrieve

    def retrieve_counted(query, limit=8):
        response = original_retrieve(query, limit)
        capture.pipeline_retrieval_calls += 1
        return response

    retrieval.retrieve = retrieve_counted
    pipeline_started = time.perf_counter()
    result = run_experiment_b(
        QUESTION, ANSWER, initial.evidence,
        retrieval_provider=retrieval, llm_provider=capture,
        verification_config=config, revise_answer=True,
    )
    total_seconds = time.perf_counter() - pipeline_started
    unit_traces = [row for row in (result.report.trace if result.report else []) if row.get("unit")]
    judge_groups = [row.get("judge_attempts", []) for row in unit_traces]
    flat_group_meta = [meta for group in judge_groups for meta in group]
    errors = result.report.errors if result.report else ([result.error] if result.error else [])

    for index, record in enumerate(capture.records):
        group_index = 0 if record["attempt_stage"] == "initial_verification" else 1
        meta = flat_group_meta[group_index] if group_index < len(flat_group_meta) else {}
        record["pipeline_parse_exception"] = meta.get("parse_error")
        record["pipeline_validator_overrides"] = meta.get("validator_overrides", [])
        record["report_errors"] = errors

    output = {
        "diagnostic": "Case 2 only; raw judge transport capture before verifier parsing",
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "case_id": "case_02_direct_contradiction",
        "question": QUESTION,
        "original_answer": ANSWER,
        "expected_verdict": "CONTRADICTED",
        "evidence_corpus": [{"source_id": SOURCE, "page": 1, "text": EVIDENCE}],
        "configuration": {
            "model": config.model, "think": config.think, "format": config.response_format,
            "temperature": config.temperature, "timeout_seconds": config.timeout_seconds,
            "max_judge_retries": config.max_judge_retries,
            "max_targeted_retrieval_retries": config.max_targeted_retrieval_retries,
            "ollama_endpoint": config.ollama_url,
        },
        "initial_retrieval_seconds": initial_retrieval_seconds,
        "initial_retrieval_evidence": [item.to_dict() for item in initial.evidence],
        "verification_units": result.report.verification_units if result.report else [],
        "unit_retrieval_attempts": [
            {"queries": trace.get("retrieval_queries", []), "attempts": trace.get("retrieval_attempts", [])}
            for trace in unit_traces
        ],
        "raw_judge_attempts": capture.records,
        "final_verdicts": [
            {"unit_id": item.claim_id, "llm_verdict": item.llm_verdict, "verdict": item.verdict,
             "validator_overrides": item.validator_overrides, "retry_events": item.retry_events}
            for item in (result.report.verifications if result.report else [])
        ],
        "validator_reached": any(meta.get("parse_error") is None for meta in flat_group_meta),
        "retry_occurred": any(
            event.get("stage") in {"semantic_judge", "targeted_retrieval"}
            for item in (result.report.verifications if result.report else []) for event in item.retry_events
        ),
        "final_answer": result.final_answer,
        "errors": errors,
        "total_experiment_b_seconds": total_seconds,
    }
    OUTPUT.write_text(json.dumps(output, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps({
        "artifact": OUTPUT.name,
        "final_verdicts": output["final_verdicts"],
        "attempts": [{k: attempt.get(k) for k in (
            "attempt_number", "attempt_stage", "http_status", "streamed_successfully",
            "response_length_characters", "parsed_json", "json_parse_exception", "verdict_value",
            "returned_fields", "pipeline_parse_exception", "elapsed_seconds",
        )} for attempt in capture.records],
        "errors": errors,
        "validator_reached": output["validator_reached"],
        "total_experiment_b_seconds": total_seconds,
    }, indent=2, ensure_ascii=False, default=str), flush=True)
    return 0 if result.report is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
