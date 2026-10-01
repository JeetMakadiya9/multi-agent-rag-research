from __future__ import annotations

import math
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Sequence

import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

import rag


@dataclass(frozen=True)
class QueryCase:
    group: str
    query: str
    history: Optional[list[dict[str, str]]] = None


CORPUS = [
    {
        "page": 1,
        "text": """Retrieval-Augmented Generation

Retrieval-Augmented Generation (RAG) combines information retrieval with language generation. A retriever first identifies relevant documents or passages, and a language model then generates an answer grounded in the retrieved evidence.""",
    },
    {
        "page": 2,
        "text": """Hybrid Retrieval

A hybrid retrieval system can combine dense semantic retrieval with lexical retrieval such as BM25. Dense retrieval is useful when the query and evidence use different wording, while lexical retrieval can preserve exact technical terminology and identifiers.""",
    },
    {
        "page": 3,
        "text": """Claim Verification

Retrieval relevance does not prove that a claim is factually supported. Claim-level verification should compare a generated claim against retrieved evidence and distinguish supported, contradicted, and insufficiently supported claims.""",
    },
    {
        "page": 4,
        "text": """Unrelated Topic

Photosynthesis converts light energy into chemical energy in plants. This page is intentionally unrelated to the RAG retrieval questions.""",
    },
]


CASES = [
    QueryCase("IN", "What does retrieval-augmented generation combine?"),
    QueryCase("IN", "How can a retriever find useful passages when wording differs?"),
    QueryCase("IN", "What role does BM25 play in hybrid retrieval?"),
    QueryCase("IN", "Which details can lexical retrieval preserve exactly?"),
    QueryCase("IN", "Why is relevant retrieved evidence not proof that a claim is true?"),
    QueryCase("IN", "How should claim-level verification assess a generated claim?"),
    QueryCase("IN", "Compare dense semantic search with BM25 lexical search."),
    QueryCase("IN", "Explain hybrid retrieval and why relevance alone cannot verify a factual claim."),
    QueryCase(
        "IN",
        "Why is that important?",
        [
            {"role": "user", "content": "What is claim-level verification in RAG?"},
            {"role": "assistant", "content": "It compares a generated claim against retrieved evidence."},
        ],
    ),
    QueryCase(
        "IN",
        "What does it compare, and which outcomes can it distinguish?",
        [
            {"role": "user", "content": "How does claim verification work?"},
            {"role": "assistant", "content": "It checks a generated claim against retrieved evidence."},
        ],
    ),
    QueryCase("OUT", "What is the capital city of France?"),
    QueryCase("OUT", "Who invented the telephone?"),
    QueryCase("OUT", "What is the boiling point of water at sea level?"),
    QueryCase("OUT", "Who wrote the play Hamlet?"),
    QueryCase("OUT", "What is the population of Japan?"),
    QueryCase("OUT", "Which planet is the largest in our solar system?"),
    QueryCase("OUT", "In what year did humans first land on the Moon?"),
    QueryCase("OUT", "What is the chemical symbol for gold?"),
    QueryCase("OUT", "What is the tallest mountain above sea level?"),
    QueryCase("OUT", "What currency is used in the United Kingdom?"),
]


def build_indexes(chunks: Sequence[dict], embedding_model: SentenceTransformer):
    texts = [str(chunk.get("text", "")) for chunk in chunks]
    embeddings = embedding_model.encode(
        texts,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    bm25 = BM25Okapi([rag.tokenize_text(text) for text in texts])
    return index, bm25


def quantile(sorted_values: list[float], probability: float) -> float:
    """Linear-interpolated quantile, matching NumPy's default percentile method."""
    if not sorted_values:
        return math.nan
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def summarize(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p25": quantile(ordered, 0.25),
        "p75": quantile(ordered, 0.75),
    }


def fmt(value: object) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def main() -> None:
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    chunks = rag.create_chunks(CORPUS, "baseline_rejection_calibration.txt")
    print(f"[INFO] Synthetic corpus produced {len(chunks)} chunks.")

    print(f"[INFO] Loading embedding model: {rag.EMBEDDING_MODEL_NAME}")
    embedding_model = SentenceTransformer(rag.EMBEDDING_MODEL_NAME)
    faiss_index, bm25_index = build_indexes(chunks, embedding_model)
    print(f"[INFO] Loading reranker: {rag.RERANKER_MODEL_NAME}")
    reranker = CrossEncoder(rag.RERANKER_MODEL_NAME)

    rows: list[dict[str, object]] = []
    original_gate = rag.relevance_gate
    gate_candidates: list[rag.RetrievalResult] = []

    def observe_existing_gate(candidates):
        # Capture pre-gate results for offline scoring, then call the original
        # gate unchanged so the baseline's rejection decision remains intact.
        gate_candidates[:] = list(candidates)
        return original_gate(candidates)

    rag.relevance_gate = observe_existing_gate
    try:
        for case in CASES:
            diagnostics = rag.RetrievalDiagnostics(query=case.query)
            gate_candidates.clear()
            results = rag.retrieve_chunks_advanced(
                question=case.query,
                chunks=chunks,
                embedding_model=embedding_model,
                reranker=reranker,
                faiss_index=faiss_index,
                bm25_index=bm25_index,
                history=case.history,
                diagnostics=diagnostics,
            )

            observed = list(gate_candidates)
            rows.append(
                {
                    "query": case.query,
                    "group": case.group,
                    "rejected": diagnostics.rejected,
                    "rejection_reason": diagnostics.rejection_reason,
                    "best_evidence_score": max(
                        (float(result.evidence_score) for result in observed),
                        default=0.0,
                    ),
                    "best_vector_distance": diagnostics.best_vector_distance,
                    "best_rerank_score": diagnostics.best_rerank_score,
                    "best_bm25_score": diagnostics.best_bm25_score,
                    "best_exact_score": diagnostics.best_exact_score,
                    "top_pages": [
                        int(result.chunk.get("page", 0))
                        for result in observed[:5]
                    ],
                }
            )
    finally:
        rag.relevance_gate = original_gate

    print("\n=== INTERPRETER AND DATASET ===")
    print(f"Python interpreter: {sys.executable}")
    print(f"Timestamp (UTC): {timestamp}")
    print(f"Total queries: {len(rows)} (IN={sum(r['group'] == 'IN' for r in rows)}, OUT={sum(r['group'] == 'OUT' for r in rows)})")
    print("IN query list:")
    for row in rows:
        if row["group"] == "IN":
            print(f"  - {row['query']}")
    print("OUT query list:")
    for row in rows:
        if row["group"] == "OUT":
            print(f"  - {row['query']}")

    print("\n=== PER-QUERY RESULTS ===")
    fields = (
        "query", "group", "rejected", "rejection_reason", "best_evidence_score",
        "best_vector_distance", "best_rerank_score", "best_bm25_score",
        "best_exact_score", "top_pages",
    )
    for index, row in enumerate(rows, 1):
        print(f"[{index:02d}] " + " | ".join(f"{key}={fmt(row[key])}" for key in fields))

    print("\n=== SUMMARY STATISTICS ===")
    metrics = (
        ("evidence_score", "best_evidence_score"),
        ("vector_distance", "best_vector_distance"),
        ("reranker_score", "best_rerank_score"),
        ("bm25_score", "best_bm25_score"),
    )
    group_summaries: dict[str, dict[str, dict[str, float | int]]] = {}
    for group in ("IN", "OUT"):
        group_rows = [row for row in rows if row["group"] == group]
        group_summaries[group] = {}
        print(f"[{group}] count={len(group_rows)}")
        for label, key in metrics:
            values = [float(row[key]) for row in group_rows if row[key] is not None]
            stats = summarize(values)
            group_summaries[group][label] = stats
            print(
                f"  {label}: count={stats['count']} min={stats['min']:.6f} "
                f"max={stats['max']:.6f} mean={stats['mean']:.6f} "
                f"median={stats['median']:.6f} P25={stats['p25']:.6f} "
                f"P75={stats['p75']:.6f}"
            )

    positives = [row for row in rows if row["group"] == "IN"]
    negatives = [row for row in rows if row["group"] == "OUT"]
    print("\n=== EVIDENCE-SCORE THRESHOLD SWEEP ===")
    print("threshold TP FP TN FN precision recall false_accept_rate false_reject_rate")
    threshold_rows: list[dict[str, float | int]] = []
    for step in range(1, 17):
        threshold = step * 0.05
        tp = sum(float(row["best_evidence_score"]) >= threshold for row in positives)
        fn = len(positives) - tp
        fp = sum(float(row["best_evidence_score"]) >= threshold for row in negatives)
        tn = len(negatives) - fp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / len(positives) if positives else 0.0
        far = fp / len(negatives) if negatives else 0.0
        frr = fn / len(positives) if positives else 0.0
        threshold_rows.append(
            {
                "threshold": threshold, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                "precision": precision, "recall": recall,
                "far": far, "frr": frr,
            }
        )
        print(
            f"{threshold:.2f} {tp:2d} {fp:2d} {tn:2d} {fn:2d} "
            f"{precision:.4f} {recall:.4f} {far:.4f} {frr:.4f}"
        )

    print("\n=== THRESHOLD ERROR CASES ===")
    for result in threshold_rows:
        if result["fp"] or result["fn"]:
            threshold = float(result["threshold"])
            false_accepts = [
                str(row["query"]) for row in negatives
                if float(row["best_evidence_score"]) >= threshold
            ]
            false_rejects = [
                str(row["query"]) for row in positives
                if float(row["best_evidence_score"]) < threshold
            ]
            print(
                f"threshold={threshold:.2f} false_accepts={false_accepts} "
                f"false_rejects={false_rejects}"
            )

    print("\n=== END CALIBRATION RESULTS ===")


if __name__ == "__main__":
    main()
