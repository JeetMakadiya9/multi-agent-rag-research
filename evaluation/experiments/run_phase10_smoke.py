"""Execute a deterministic three-example A/B/C SciFact development smoke run."""
from __future__ import annotations

import json
import hashlib
import os
import platform
import sys
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Keep CPU inference predictable on hosts whose default BLAS/OpenMP thread
# count is much larger than the local smoke workload.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.experiments.evaluator import run_experiment
from evaluation.experiments.experiment_config import ExperimentConfig
from evaluation.experiments.system_a_rag import run_system_a
from evaluation.experiments.system_b_verified_rag import run_system_b
from evaluation.experiments.system_c_multi_agent import run_system_c
from evaluation.scifact.scifact_adapter import corpus_to_rag_chunks, load_scifact_dataset


# First examples encountered for empty, SUPPORT, and CONTRADICT rationale labels
# in the existing dev-file order. Fixed before running any system.
SCIFACT_DEV_SMOKE_IDS = (1, 3, 42)


def build_provider(dataset_root: Path):
    """Build one shared instance of the project's existing corpus retriever."""
    import rag
    from research_retrieval import ExistingRAGProvider
    from sentence_transformers import CrossEncoder, SentenceTransformer
    import torch

    torch.set_num_threads(4)

    def cached_snapshot(model_id: str) -> Path:
        hub = Path.home() / ".cache" / "huggingface" / "hub"
        folder = "models--" + model_id.replace("/", "--")
        candidates = [hub / folder]
        if "/" not in model_id:
            candidates.extend(hub.glob(f"models--*--{model_id}"))
        snapshots = sorted(
            (snapshot for candidate in candidates
             if (candidate / "snapshots").is_dir()
             for snapshot in (candidate / "snapshots").iterdir()
             if snapshot.is_dir() and (snapshot / "config.json").is_file()),
            key=lambda item: (item.parent.parent.name, item.name),
        )
        if not snapshots:
            raise FileNotFoundError(f"Required local model cache is unavailable: {model_id} ({root})")
        return snapshots[-1]

    dataset = load_scifact_dataset(dataset_root, "dev")
    chunks = corpus_to_rag_chunks(dataset.corpus, rag)
    embedding_path = cached_snapshot(rag.EMBEDDING_MODEL_NAME)
    reranker_path = cached_snapshot(rag.RERANKER_MODEL_NAME)
    print("Loading local embedding model", flush=True)
    embedding = SentenceTransformer(str(embedding_path), device="cpu")
    digest = hashlib.sha256()
    digest.update(embedding_path.name.encode("utf-8"))
    for chunk in chunks:
        digest.update(str(chunk.get("chunk_id", "")).encode("utf-8"))
        digest.update(str(chunk.get("text", "")).encode("utf-8"))
    cache_path = Path(tempfile.gettempdir()) / f"phase10_scifact_dev_embeddings_{digest.hexdigest()[:20]}.npy"
    if cache_path.is_file():
        import numpy as np
        print(f"Loading cached corpus embeddings from {cache_path}", flush=True)
        embeddings = np.load(cache_path, mmap_mode="r")
        if len(embeddings) != len(chunks):
            raise ValueError("Cached corpus embedding count does not match the corpus")
        faiss_index = rag.build_faiss_index(embeddings)
        bm25_index = rag.build_bm25_index(chunks)
    else:
        print(f"Building embeddings for {len(chunks)} corpus chunks", flush=True)
        embeddings, faiss_index, bm25_index = rag.build_search_database(chunks, embedding)
        import numpy as np
        np.save(cache_path, embeddings)
    print("Loading local reranker", flush=True)
    reranker = CrossEncoder(str(reranker_path), device="cpu")
    print("Retrieval provider is ready", flush=True)
    provider = ExistingRAGProvider(
        chunks=chunks, embedding_model=embedding, reranker=reranker,
        faiss_index=faiss_index, bm25_index=bm25_index, parents=None,
        retrieval_module=rag,
    )
    provider.phase10_model_snapshots = {
        "embedding": {"model": rag.EMBEDDING_MODEL_NAME, "snapshot": embedding_path.name},
        "reranker": {"model": rag.RERANKER_MODEL_NAME, "snapshot": reranker_path.name},
        "corpus_chunks": len(chunks), "embedding_cache_file": cache_path.name,
        "embedding_cache_sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest(),
    }
    return provider


def _ollama_model_details(model: str) -> dict[str, Any]:
    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/show",
        data=json.dumps({"name": model}).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            info = json.loads(response.read().decode("utf-8"))
        return {"name": model, "details": info.get("details", {}),
                "model_info": info.get("model_info", {})}
    except Exception as exc:
        return {"name": model, "version": None,
                "status": f"unavailable: {type(exc).__name__}: {exc}"}


def run_smoke(*, dataset_root: str | Path | None = None,
              results_dir: str | Path | None = None,
              provider: Any = None) -> dict[str, Any]:
    root = Path(dataset_root or ROOT / "data" / "scifact" / "data").resolve()
    output_root = Path(results_dir or ROOT / "evaluation" / "experiments" / "results").resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / "phase10_smoke_run"
    combined_path = output_root / "phase10_smoke.json"
    if combined_path.exists() or run_dir.exists():
        raise FileExistsError(f"Smoke artifacts already exist: {combined_path} or {run_dir}")

    dataset = load_scifact_dataset(root, "dev")
    available = {row.runtime.claim_id for row in dataset.claims}
    missing = set(SCIFACT_DEV_SMOKE_IDS) - available
    if missing:
        raise ValueError(f"Fixed smoke IDs missing from SciFact dev split: {sorted(missing)}")
    if provider is None:
        provider = build_provider(root)
    config = ExperimentConfig(
        dataset_root=str(root), split="dev", example_ids=SCIFACT_DEV_SMOKE_IDS,
        top_k=20, verification_evidence_limit=2, model="qwen3:4b", temperature=0.0,
        maximum_controller_cycles=1, maximum_retrieval_actions=1,
        maximum_verification_actions=1,
        notes={
            "experiment": "Phase 10B deterministic dev smoke",
            "system_c_decision_binding": "deterministic 8A action provider; real 8A validation/8B/Phase 7B/Phase 6/8C controller",
            "system_c_retrieval_preflight": "one additional retrieval to bind rank-to-version provenance; counted as system cost",
            "threshold_tuning": "none",
        },
    )
    result = run_experiment(
        config, provider,
        {"A": run_system_a, "B": run_system_b, "C": run_system_c},
        run_dir,
    )
    per_example_path = run_dir / "per_example.jsonl"
    rows = [json.loads(line) for line in per_example_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    counts = {}
    for system in ("A", "B", "C"):
        entries = [row["systems"][system] for row in rows]
        counts[system] = {
            "attempted": len(entries),
            "completed": sum(entry.get("status") == "SUCCESS" for entry in entries),
            "failures": sum(entry.get("error") is not None for entry in entries),
            "timeouts": sum(entry.get("status") == "TIMEOUT" for entry in entries),
        }
    combined = {
        "experiment_name": "Phase 10B Complete A/B/C Smoke Experiment",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {"name": "SciFact", "split": "dev", "root": str(root),
                    "selected_example_ids": list(SCIFACT_DEV_SMOKE_IDS),
                    "test_split_used": False, "gold_annotations_passed_to_execution": False},
        "corpus_index_identity": getattr(provider, "phase10_model_snapshots", None),
        "systems": {
            "A": {"definition": "Existing strong RAG retrieval plus baseline grounded answer generator",
                  "results": {row["example_id"]: {
                      **row["systems"]["A"], "example_id": row["example_id"],
                      "input_claim": row["runtime_input"]["claim"],
                      "gold_label": row["scoring"]["gold_label"]}
                      for row in rows}},
            "B": {"definition": "Same strong RAG retrieval plus frozen Experiment B claim verification",
                  "results": {row["example_id"]: {
                      **row["systems"]["B"], "example_id": row["example_id"],
                      "input_claim": row["runtime_input"]["claim"],
                      "gold_label": row["scoring"]["gold_label"]}
                      for row in rows}},
            "C": {"definition": "Real bounded Phase 8C controller; deterministic 8A smoke action binding; scoped approval and explicit continuation; Phase 7B dispatch to Phase 6 retrieval and verification",
                  "results": {row["example_id"]: {
                      **row["systems"]["C"], "example_id": row["example_id"],
                      "input_claim": row["runtime_input"]["claim"],
                      "gold_label": row["scoring"]["gold_label"]}
                      for row in rows}},
        },
        "execution_counts": counts,
        "aggregate_metrics": result["summary"],
        "per_example_scoring": {row["example_id"]: row["scoring"] for row in rows},
        "reproducibility": {
            "configuration": config.to_dict(),
            "model": _ollama_model_details(config.model),
            "retriever": "existing rag.py hybrid retrieval + reranking; one provider instance shared by A/B/C",
            "local_huggingface_snapshots": getattr(provider, "phase10_model_snapshots", None),
            "python": sys.version,
            "platform": platform.platform(),
            "run_manifest_path": str(run_dir / "run_manifest.json"),
            "per_example_jsonl_path": str(per_example_path),
            "summary_path": str(run_dir / "summary.json"),
        },
        "results": {"A": counts["A"], "B": counts["B"], "C": counts["C"]},
        "completion_status": (
            "COMPLETED" if all(value["attempted"] == 3 and value["completed"] == 3
                                for value in counts.values())
            else "PARTIAL" if any(value["attempted"] for value in counts.values())
            else "FAILED"),
        "scientific_note": "Development smoke only. It validates execution and metric plumbing; it is not the final benchmark and does not establish superiority.",
    }
    combined_path.write_text(json.dumps(combined, indent=2, ensure_ascii=False, default=str) + "\n",
                             encoding="utf-8")
    return combined


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    print(json.dumps(run_smoke(), indent=2, ensure_ascii=False, default=str))
