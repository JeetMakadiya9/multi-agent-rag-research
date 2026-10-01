"""Run a frozen-pipeline SciFact-Orig evaluation (dev split only for now).

No data is downloaded. Gold annotations are held in the scoring-side claim
record and are not supplied to the retriever, Experiment A, or Experiment B.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:  # Package import in tests vs. direct script execution from its directory.
    from .scifact_adapter import corpus_to_rag_chunks, load_scifact_dataset, runtime_claim_payload  # noqa: E402
    from .metrics import (  # noqa: E402
        DEFAULT_TOP_K, build_run_manifest, retrieval_metrics_for_claim,
        summarize_records, verification_metrics_for_claim,
    )
except ImportError:  # pragma: no cover - direct ``python path/to/script.py`` mode.
    from scifact_adapter import corpus_to_rag_chunks, load_scifact_dataset, runtime_claim_payload  # noqa: E402
    from metrics import (  # noqa: E402
        DEFAULT_TOP_K, build_run_manifest, retrieval_metrics_for_claim,
        summarize_records, verification_metrics_for_claim,
    )


def parse_top_k(value: str) -> tuple[int, ...]:
    try:
        values = tuple(sorted(set(int(part.strip()) for part in value.split(",") if part.strip())))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("top-k must be comma-separated positive integers") from exc
    if not values or any(value < 1 for value in values):
        raise argparse.ArgumentTypeError("top-k must contain positive integers")
    return values


def _scoring_annotations(claim_record: Any) -> dict[str, Any]:
    annotations = claim_record.annotations
    return {
        "labels_available": annotations.labels_available,
        "evidence_field_present": annotations.evidence_field_present,
        "evidence": None if annotations.evidence is None else [
            {
                "doc_id": document.doc_id,
                "rationales": [
                    {"label": rationale.label, "sentence_indices": list(rationale.sentence_indices)}
                    for rationale in document.rationales
                ],
            }
            for document in annotations.evidence
        ],
        "cited_doc_ids": list(annotations.cited_doc_ids),
    }


def _safe_runtime_record(claim_record: Any) -> dict[str, Any]:
    """Runtime input helper deliberately drops the annotations object."""
    return runtime_claim_payload(claim_record.runtime)


def _retrieve_runtime_claim(provider: Any, runtime_claim: dict[str, Any], limit: int) -> Any:
    """Retrieve from runtime-only claim fields; annotation fields are absent."""
    return provider.retrieve(runtime_claim["claim"], limit=limit)


def _runtime_evidence(response_evidence: Sequence[Any]) -> tuple[list[dict[str, Any]], list[Any]]:
    """Create scoring refs and metadata-free EvidenceItems for Experiment B."""
    from claim_verification import EvidenceItem

    refs: list[dict[str, Any]] = []
    evidence_items: list[EvidenceItem] = []
    for rank, item in enumerate(response_evidence, start=1):
        metadata = getattr(item, "retrieval_metadata", {}) or {}
        raw_chunk = metadata.get("chunk", {}) if isinstance(metadata, dict) else {}
        chunk = dict(raw_chunk) if isinstance(raw_chunk, dict) else {}
        doc_id = chunk.get("scifact_doc_id", getattr(item, "document_id", None))
        try:
            doc_id = int(doc_id)
        except (TypeError, ValueError):
            doc_id = None
        sentence_indices = chunk.get("scifact_sentence_indices", [])
        if not isinstance(sentence_indices, list):
            sentence_indices = []
        sentence_indices = [x for x in sentence_indices if isinstance(x, int) and not isinstance(x, bool)]
        sentence_texts = chunk.get("scifact_sentence_texts", [])
        if not isinstance(sentence_texts, list):
            sentence_texts = []
        chunk_id = str(getattr(item, "chunk_id", "") or getattr(item, "evidence_id", "") or f"rank-{rank}")
        evidence_id = f"E{rank}"
        text = str(getattr(item, "text", "") or "")
        retrieval_score = getattr(item, "retrieval_score", None)
        if retrieval_score is None:
            retrieval_score = getattr(item, "score", 0.0)
        refs.append({
            "rank": rank,
            "evidence_id": evidence_id,
            "chunk_id": chunk_id,
            "scifact_doc_id": doc_id,
            "scifact_sentence_indices": sentence_indices,
            "scifact_sentence_texts": sentence_texts,
            "text": text,
            "retrieval_score": retrieval_score,
        })
        # Document/sentence mapping remains in refs only. The pairwise judge
        # receives text; the verifier EvidenceItem has no gold or index metadata.
        evidence_items.append(EvidenceItem(
            evidence_id=evidence_id,
            text=text,
            source=f"SciFact abstract {doc_id}" if doc_id is not None else "SciFact abstract",
            page=None,
            score=float(retrieval_score or 0.0),
            metadata={},
            source_id=f"scifact:{doc_id}" if doc_id is not None else "scifact:unknown",
            document_id=str(doc_id) if doc_id is not None else "",
            chunk_id=chunk_id,
            retrieval_method="existing_advanced_rag",
            retrieval_score=float(retrieval_score or 0.0),
        ))
    return refs, evidence_items


def _run_experiment_a(claim_text: str, response_evidence: Sequence[Any]) -> dict[str, Any]:
    """Generate and retain the existing baseline prompt's raw free-text output.

    The existing ``run_experiment_a`` API wraps an already-generated answer;
    it does not generate one. This adapter uses its existing baseline prompt
    and Ollama generator with the same frozen retriever's result context, then
    passes that raw answer through the unchanged Experiment A wrapper.
    """
    from langchain_core.documents import Document
    from src.baseline_rag import create_llm, create_prompt, format_context
    from research_pipeline import run_experiment_a

    docs = []
    for item in response_evidence[:3]:
        meta = getattr(item, "retrieval_metadata", {}) or {}
        chunk = meta.get("chunk", {}) if isinstance(meta, dict) else {}
        doc_id = chunk.get("scifact_doc_id", getattr(item, "document_id", "unknown")) if isinstance(chunk, dict) else "unknown"
        docs.append(Document(
            page_content=str(getattr(item, "text", "") or ""),
            metadata={"source": f"SciFact abstract {doc_id}"},
        ))
    context = format_context(docs)
    messages = create_prompt().format_messages(context=context, question=claim_text)
    raw_answer = create_llm().invoke(messages)
    wrapped = run_experiment_a(claim_text, str(raw_answer), response_evidence)
    return {
        "output_type": "free_text_rag_output",
        "raw_output": wrapped.final_answer,
        "retrieval_count": wrapped.retrieval_count,
        "prompt_identifier": "src.baseline_rag.create_prompt (source hash in manifest)",
    }


def run_evaluation(
    *,
    dataset_root: str | Path,
    split: str,
    output_dir: str | Path,
    top_k: Sequence[int] = DEFAULT_TOP_K,
    run_experiment_a: bool = False,
    run_experiment_b: bool = True,
    evidence_limit: int = 8,
) -> dict[str, Any]:
    if split != "dev":
        raise ValueError("The first SciFact runner version supports only --split dev.")
    if evidence_limit < 1:
        raise ValueError("evidence_limit must be >= 1")
    dataset = load_scifact_dataset(dataset_root, split)

    import rag
    from experiment_b_verification import ExperimentBConfig, run_experiment_b_verification
    from research_retrieval import ExistingRAGProvider
    from sentence_transformers import CrossEncoder, SentenceTransformer

    chunks = corpus_to_rag_chunks(dataset.corpus, rag)
    if not chunks:
        raise ValueError("SciFact corpus adaptation produced no RAG chunks.")
    embedding_model = SentenceTransformer(rag.EMBEDDING_MODEL_NAME)
    _, faiss_index, bm25_index = rag.build_search_database(chunks, embedding_model)
    reranker = CrossEncoder(rag.RERANKER_MODEL_NAME)
    provider = ExistingRAGProvider(
        chunks=chunks,
        embedding_model=embedding_model,
        reranker=reranker,
        faiss_index=faiss_index,
        bm25_index=bm25_index,
        parents=None,
        retrieval_module=rag,
    )
    verification_config = ExperimentBConfig(
        max_evidence_items=evidence_limit,
        unit_specific_retrieval=False,
        max_targeted_retrieval_retries=0,
        answer_revision_enabled=False,
    )
    config_dict = asdict(verification_config)
    manifest = build_run_manifest(
        dataset_root=dataset.dataset_root,
        split=split,
        corpus_path=dataset.corpus_path,
        claims_path=dataset.claims_path,
        output_dir=output_dir,
        project_root=PROJECT_ROOT,
        rag_module=rag,
        verification_config=config_dict,
        top_k=top_k,
        run_experiment_a=run_experiment_a,
        run_experiment_b=run_experiment_b,
    )

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    output_files = [output_path / "run_manifest.json", output_path / "per_claim.jsonl", output_path / "summary.json"]
    existing = [str(path) for path in output_files if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite existing evaluation artifact(s): " + ", ".join(existing))
    (output_path / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    records: list[dict[str, Any]] = []
    with (output_path / "per_claim.jsonl").open("w", encoding="utf-8") as stream:
        for record in dataset.claims:
            runtime_claim = _safe_runtime_record(record)
            claim_text = record.runtime.text
            end_to_end_start = time.perf_counter()
            retrieval_start = time.perf_counter()
            response = _retrieve_runtime_claim(
                provider, runtime_claim, limit=max(max(top_k), evidence_limit)
            )
            retrieval_elapsed = time.perf_counter() - retrieval_start
            retrieval_refs, verifier_evidence = _runtime_evidence(response.evidence)
            diagnostics = dict(response.diagnostics or {})
            retrieval_record = {
                **retrieval_metrics_for_claim(record.annotations, retrieval_refs, top_k),
                "query": claim_text,
                "ranked_chunks": retrieval_refs,
                "retrieved_chunk_count": len(retrieval_refs),
                "unique_retrieved_document_count": len({x["scifact_doc_id"] for x in retrieval_refs if x["scifact_doc_id"] is not None}),
                "empty_retrieval": not bool(retrieval_refs),
                "rejected": bool(diagnostics.get("rejected", False)),
                "rejection_reason": diagnostics.get("rejection_reason", ""),
                "diagnostics": diagnostics,
            }
            experiment_a = None
            if run_experiment_a:
                experiment_a = _run_experiment_a(claim_text, response.evidence)
            experiment_b = None
            verification_elapsed = None
            if run_experiment_b:
                verification_start = time.perf_counter()
                report = run_experiment_b_verification(
                    claim_text,
                    claim_text,
                    verifier_evidence,
                    retrieval_provider=None,
                    config=verification_config,
                )
                verification_elapsed = time.perf_counter() - verification_start
                report_dict = report.to_dict()
                verification_record = verification_metrics_for_claim(
                    record.annotations,
                    retrieval_refs,
                    report_dict,
                    evidence_limit=evidence_limit,
                )
                experiment_b = {
                    "verification": verification_record,
                    "verifications": report_dict.get("verifications", []),
                    "verification_units": report_dict.get("verification_units", []),
                    "trace": report_dict.get("trace", []),
                    "errors": report_dict.get("errors", []),
                    "report_metrics": report_dict.get("metrics", {}),
                }
            elapsed = time.perf_counter() - end_to_end_start
            report_metrics = (experiment_b or {}).get("report_metrics", {})
            retry_count = (experiment_b or {}).get("verification", {}).get("assessment_retry_count", 0)
            row = {
                "claim_id": record.runtime.claim_id,
                "split": split,
                "runtime_input": runtime_claim,
                "scoring_annotations": _scoring_annotations(record),
                "retrieval": retrieval_record,
                "experiment_a": experiment_a,
                "experiment_b": experiment_b,
                "efficiency": {
                    "rag_calls": 1,
                    "llm_calls": int(report_metrics.get("llm_calls", 0) or 0) + int(experiment_a is not None),
                    "experiment_a_llm_calls": int(experiment_a is not None),
                    "retries": int(retry_count or 0),
                    "retrieval_latency_seconds": retrieval_elapsed,
                    "verification_latency_seconds": verification_elapsed,
                    "end_to_end_latency_seconds": elapsed,
                },
            }
            records.append(row)
            stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    summary = summarize_records(records, top_k)
    summary.update({
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(output_path / "run_manifest.json"),
        "per_claim_path": str(output_path / "per_claim.jsonl"),
        "full_benchmark_completed": True,
        "note": "Running this CLI processes the selected split in full; smoke checks do not invoke it.",
    })
    (output_path / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path, help="Directory containing corpus.jsonl and claims_dev.jsonl")
    parser.add_argument("--split", choices=("dev",), default="dev", help="Only the labeled dev split is enabled in this initial runner.")
    parser.add_argument("--output-dir", required=True, type=Path, help="New or empty run directory; existing artifacts are never overwritten.")
    parser.add_argument("--top-k", type=parse_top_k, default=DEFAULT_TOP_K, help="Comma-separated retrieval cutoffs (default: 1,5,10,20).")
    parser.add_argument("--evidence-limit", type=int, default=8, help="Maximum retrieved items supplied to Experiment B.")
    parser.add_argument("--run-experiment-a", action=argparse.BooleanOptionalAction, default=False,
                        help="Generate a raw free-text output using the existing baseline prompt/generator (default: off).")
    parser.add_argument("--run-experiment-b", action=argparse.BooleanOptionalAction, default=True,
                        help="Run the existing deterministic Experiment B verifier (default: on).")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = run_evaluation(
            dataset_root=args.dataset_root,
            split=args.split,
            output_dir=args.output_dir,
            top_k=args.top_k,
            run_experiment_a=args.run_experiment_a,
            run_experiment_b=args.run_experiment_b,
            evidence_limit=args.evidence_limit,
        )
    except Exception as exc:
        parser.exit(2, f"SciFact evaluation failed: {type(exc).__name__}: {exc}\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
