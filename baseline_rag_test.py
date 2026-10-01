from __future__ import annotations
import sys, time, traceback

def fail(message: str) -> None:
    print(f"\n[FAIL] {message}")
    sys.exit(1)

def main() -> None:
    print("=" * 72)
    print("BASELINE RAG v0.1 — REAL RUNTIME TEST")
    print("=" * 72)
    try:
        import faiss
        from rank_bm25 import BM25Okapi
        from sentence_transformers import CrossEncoder, SentenceTransformer
        import rag
    except Exception as exc:
        print("\n[FAIL] Required import failed.")
        print(f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        sys.exit(1)
    print("[PASS] Imported rag.py and retrieval dependencies.")
    print(f"[INFO] Embedding model: {rag.EMBEDDING_MODEL_NAME}")
    print(f"[INFO] Reranker model:  {rag.RERANKER_MODEL_NAME}")

    pages = [
        {"page": 1, "text": """Retrieval-Augmented Generation\n\nRetrieval-Augmented Generation (RAG) combines information retrieval with language generation. A retriever first identifies relevant documents or passages, and a language model then generates an answer grounded in the retrieved evidence."""},
        {"page": 2, "text": """Hybrid Retrieval\n\nA hybrid retrieval system can combine dense semantic retrieval with lexical retrieval such as BM25. Dense retrieval is useful when the query and evidence use different wording, while lexical retrieval can preserve exact technical terminology and identifiers."""},
        {"page": 3, "text": """Claim Verification\n\nRetrieval relevance does not prove that a claim is factually supported. Claim-level verification should compare a generated claim against retrieved evidence and distinguish supported, contradicted, and insufficiently supported claims."""},
        {"page": 4, "text": """Unrelated Topic\n\nPhotosynthesis converts light energy into chemical energy in plants. This page is intentionally unrelated to the RAG retrieval questions."""},
    ]
    chunks = rag.create_chunks(pages, "synthetic_rag_test.txt")
    if not chunks: fail("rag.create_chunks() returned no chunks.")
    print(f"[PASS] Created {len(chunks)} chunks.")
    texts = [str(c.get("text", "")) for c in chunks]
    if not all(texts): fail("At least one generated chunk has empty text.")

    print("\n[INFO] Loading embedding model...")
    started = time.perf_counter()
    embedding_model = SentenceTransformer(rag.EMBEDDING_MODEL_NAME)
    embeddings = embedding_model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks): fail(f"Embedding shape inconsistent: {embeddings.shape}")
    faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss_index.add(embeddings)
    print(f"[PASS] Built FAISS IndexFlatIP with {faiss_index.ntotal} vectors in {time.perf_counter()-started:.2f}s.")

    tokenized = [rag.tokenize_text(t) for t in texts]
    bm25_index = BM25Okapi(tokenized)
    print("[PASS] Built BM25 index.")

    print("\n[INFO] Loading cross-encoder reranker...")
    started = time.perf_counter()
    reranker = CrossEncoder(rag.RERANKER_MODEL_NAME)
    print(f"[PASS] Reranker loaded in {time.perf_counter()-started:.2f}s.")

    question = "How does hybrid retrieval combine dense retrieval and BM25?"
    print("\n" + "-" * 72)
    print(f"QUERY: {question}")
    print("-" * 72)
    diagnostics = rag.RetrievalDiagnostics(query=question)
    started = time.perf_counter()
    results = rag.retrieve_chunks_advanced(question=question, chunks=chunks, embedding_model=embedding_model, reranker=reranker, faiss_index=faiss_index, bm25_index=bm25_index, diagnostics=diagnostics)
    print(f"[INFO] Retrieval completed in {time.perf_counter()-started:.2f}s.")
    print(f"[INFO] Results returned: {len(results)}")
    if not results: fail("retrieve_chunks_advanced() returned no results.")
    top_pages = [int(r.chunk.get("page", 0)) for r in results[:3]]
    for rank, result in enumerate(results[:5], 1):
        c = result.chunk
        print(f"\n#{rank} | page={c.get('page')} | evidence={result.evidence_score:.4f} | rerank={result.rerank_score!r} | vector_distance={result.vector_distance!r} | bm25={result.bm25_score!r} | exact={result.exact_score!r}")
        print(str(c.get("text", ""))[:500].replace("\n", " "))
    if 2 in top_pages: print("\n[PASS] Relevant hybrid-retrieval page appeared in top 3.")
    else: print("\n[WARN] Page 2 did not appear in top 3; investigate ranking.")

    prepared = rag.prepare_retrieval_for_generation(question, results)
    required = {"question", "context", "sources", "visual_pages", "evidence", "has_evidence"}
    missing = required - set(prepared)
    if missing: fail(f"prepare_retrieval_for_generation() missing keys: {sorted(missing)}")
    if not prepared["context"].strip(): fail("Prepared retrieval context is empty.")
    if not prepared["has_evidence"]: fail("Prepared retrieval incorrectly reports has_evidence=False.")
    print("[PASS] prepare_retrieval_for_generation() contract passed.")

    exact_question = "What does BM25 do in hybrid retrieval?"
    exact_results = rag.retrieve_chunks_advanced(question=exact_question, chunks=chunks, embedding_model=embedding_model, reranker=reranker, faiss_index=faiss_index, bm25_index=bm25_index)
    exact_pages = [int(r.chunk.get("page", 0)) for r in exact_results[:3]]
    print("\n" + "-" * 72)
    print(f"LEXICAL CHECK: {exact_question}")
    print("-" * 72)
    print(f"Top-3 pages: {exact_pages}")
    if 2 in exact_pages: print("[PASS] BM25/technical-term query retrieved the expected page.")
    else: print("[WARN] BM25/technical-term query missed the expected page in top 3.")

    print("\n" + "=" * 72)
    print("BASELINE RAG v0.1 TEST COMPLETE")
    print("=" * 72)
    print("Production files modified: NO")
    print(f"Chunks: {len(chunks)}")
    print(f"FAISS vectors: {faiss_index.ntotal}")
    print(f"Primary results: {len(results)}")
    print(f"Primary top pages: {top_pages}")
    print(f"Diagnostics: {diagnostics}")
    print("=" * 72)

if __name__ == "__main__":
    main()
