from __future__ import annotations
import time
import traceback
from dataclasses import dataclass
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
import rag

@dataclass
class AuditCase:
    name: str
    query: str
    expected_pages: list[int]
    description: str
    history: list[dict[str, str]] | None = None

def build_test_corpus():
    return [
        {"page": 1, "text": """Retrieval-Augmented Generation\n\nRetrieval-Augmented Generation (RAG) combines information retrieval with language generation. A retriever first identifies relevant documents or passages, and a language model then generates an answer grounded in the retrieved evidence."""},
        {"page": 2, "text": """Hybrid Retrieval\n\nA hybrid retrieval system can combine dense semantic retrieval with lexical retrieval such as BM25. Dense retrieval is useful when the query and evidence use different wording, while lexical retrieval can preserve exact technical terminology and identifiers."""},
        {"page": 3, "text": """Claim Verification\n\nRetrieval relevance does not prove that a claim is factually supported. Claim-level verification should compare a generated claim against retrieved evidence and distinguish supported, contradicted, and insufficiently supported claims."""},
        {"page": 4, "text": """Unrelated Topic\n\nPhotosynthesis converts light energy into chemical energy in plants. This page is intentionally unrelated to the RAG retrieval questions."""},
    ]

def build_indexes(chunks, embedding_model):
    texts = [str(c.get("text", "")) for c in chunks]
    embeddings = embedding_model.encode(texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    bm25 = BM25Okapi([rag.tokenize_text(t) for t in texts])
    return index, bm25

def main():
    print("=" * 78)
    print("BASELINE RAG v0.1 — CONTROLLED RETRIEVAL AUDIT")
    print("=" * 78)
    chunks = rag.create_chunks(build_test_corpus(), "baseline_rag_audit.txt")
    print(f"[INFO] Chunks: {len(chunks)}")
    print("[INFO] Loading embedding model...")
    embedding_model = SentenceTransformer(rag.EMBEDDING_MODEL_NAME)
    print("[INFO] Building FAISS + BM25...")
    faiss_index, bm25_index = build_indexes(chunks, embedding_model)
    print("[INFO] Loading reranker...")
    reranker = CrossEncoder(rag.RERANKER_MODEL_NAME)
    cases = [
        AuditCase("DIRECT FACTUAL", "What is retrieval-augmented generation?", [1], "Direct question."),
        AuditCase("SEMANTIC PARAPHRASE", "How can a retrieval system find passages even when the wording differs?", [2], "Semantic retrieval."),
        AuditCase("TECHNICAL TERM", "What role does BM25 play in hybrid retrieval?", [2], "Lexical/exact matching."),
        AuditCase("CLAIM VERIFICATION", "Why does retrieving relevant evidence not prove a claim is true?", [3], "Verification section."),
        AuditCase("MULTI-PART", "Explain hybrid retrieval and why retrieval relevance alone does not prove factual support.", [2, 3], "Multiple sections."),
        AuditCase("IRRELEVANT", "What is the chemical process by which plants convert light energy?", [4], "Intentionally unrelated to RAG."),
        AuditCase("NOISE / OUT-OF-CORPUS", "What is the capital city of France?", [], "No supporting answer in corpus."),
        AuditCase("FOLLOW-UP", "Why is that important?", [3], "Follow-up resolution.", [{"role":"user","content":"What is claim-level verification in RAG?"},{"role":"assistant","content":"It compares a generated claim against retrieved evidence."}]),
    ]
    rows=[]
    for i, case in enumerate(cases, 1):
        d=rag.RetrievalDiagnostics(query=case.query); started=time.perf_counter()
        try:
            results=rag.retrieve_chunks_advanced(question=case.query,chunks=chunks,embedding_model=embedding_model,reranker=reranker,faiss_index=faiss_index,bm25_index=bm25_index,history=case.history,diagnostics=d)
        except Exception as exc:
            print(f"\n[{i}] {case.name}\nERROR: {type(exc).__name__}: {exc}"); traceback.print_exc(); rows.append((case,[],d,None,"ERROR")); continue
        elapsed=(time.perf_counter()-started)*1000
        top=[int(r.chunk.get("page",0)) for r in results[:5]]
        if case.expected_pages:
            matched=bool(set(case.expected_pages)&set(top[:3]))
            if case.name=="MULTI-PART": matched=set(case.expected_pages).issubset(set(top[:5]))
        else:
            matched=(len(results)==0 or d.rejected)
        status="PASS" if matched else "INVESTIGATE"
        rows.append((case,top,d,elapsed,status))
        print(f"\n[{i}] {case.name}\nQuery: {case.query}\nExpected: {case.expected_pages or 'NONE'}\nTop-5: {top}\nReturned: {len(results)}\nRejected: {d.rejected}\nReason: {d.rejection_reason!r}\nEvidence: {results[0].evidence_score:.4f}" if results else f"\n[{i}] {case.name}\nQuery: {case.query}\nExpected: {case.expected_pages or 'NONE'}\nTop-5: {top}\nReturned: 0\nRejected: {d.rejected}\nReason: {d.rejection_reason!r}\nEvidence: N/A")
        print(f"Latency: {elapsed:.2f} ms\nSTATUS: {status}")
    print("\n"+"="*78+"\nAUDIT SUMMARY\n"+"="*78)
    print(f"Cases: {len(rows)}")
    print(f"PASS: {sum(r[4]=='PASS' for r in rows)}")
    print(f"INVESTIGATE: {sum(r[4]=='INVESTIGATE' for r in rows)}")
    print(f"ERROR: {sum(r[4]=='ERROR' for r in rows)}")
    print("\nPASS means expected behavior was observed. INVESTIGATE exposes baseline behavior that may need analysis; it is not automatically a code failure.")
    print("No production files were modified by this audit.")
    print("="*78+"\nBASELINE RAG v0.1 AUDIT COMPLETE\n"+"="*78)

if __name__ == "__main__": main()
