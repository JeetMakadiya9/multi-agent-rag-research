from pathlib import Path

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"

OLLAMA_MODEL = "qwen3:4b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 3


# ============================================================
# 1. LOAD DOCUMENTS
# ============================================================

def load_documents():
    documents = []

    for file_path in DOCUMENTS_DIR.glob("*.txt"):
        text = file_path.read_text(encoding="utf-8")

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": file_path.name
                }
            )
        )

    if not documents:
        raise FileNotFoundError(
            f"No .txt files found in: {DOCUMENTS_DIR}"
        )

    print(f"\nLoaded documents: {len(documents)}")

    return documents


# ============================================================
# 2. SPLIT DOCUMENTS INTO CHUNKS
# ============================================================

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = splitter.split_documents(documents)

    print(f"Created chunks: {len(chunks)}")

    return chunks


# ============================================================
# 3. CREATE EMBEDDINGS + FAISS VECTOR STORE
# ============================================================

def create_vector_store(chunks):
    print("\nLoading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    print("Creating FAISS vector store...")

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    print("FAISS vector store ready.")

    return vector_store


# ============================================================
# 4. CREATE LOCAL LLM
# ============================================================

def create_llm():
    return OllamaLLM(
        model=OLLAMA_MODEL
    )


# ============================================================
# 5. CREATE RAG PROMPT
# ============================================================

def create_prompt():
    return ChatPromptTemplate.from_template(
        """
You are a helpful question-answering assistant.

Answer the user's question using ONLY the provided context.

If the context does not contain enough information to answer the
question, clearly say that the available context is insufficient.

Do not invent facts.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
    )


# ============================================================
# 6. FORMAT RETRIEVED DOCUMENTS
# ============================================================

def format_context(documents):
    formatted = []

    for i, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "unknown")

        formatted.append(
            f"[Evidence {i} | Source: {source}]\n"
            f"{document.page_content}"
        )

    return "\n\n".join(formatted)


# ============================================================
# 7. ASK QUESTION
# ============================================================

def ask_question(question, vector_store, llm, prompt):

    retrieved_docs = vector_store.similarity_search(
        question,
        k=TOP_K
    )

    print("\n" + "=" * 70)
    print("RETRIEVED EVIDENCE")
    print("=" * 70)

    for i, doc in enumerate(retrieved_docs, start=1):
        print(f"\n--- Evidence {i} ---")
        print(f"Source: {doc.metadata.get('source', 'unknown')}")
        print(doc.page_content)

    context = format_context(retrieved_docs)

    messages = prompt.format_messages(
        context=context,
        question=question
    )

    answer = llm.invoke(messages)

    print("\n" + "=" * 70)
    print("GENERATED ANSWER")
    print("=" * 70)
    print(answer)

    print("\n" + "=" * 70)
    print("SOURCES")
    print("=" * 70)

    for i, doc in enumerate(retrieved_docs, start=1):
        print(
            f"{i}. {doc.metadata.get('source', 'unknown')}"
        )

    return answer, retrieved_docs


# ============================================================
# 8. MAIN
# ============================================================

def main():

    print("=" * 70)
    print("MULTI-AGENT RAG PROJECT")
    print("BASELINE RAG v0.1")
    print("=" * 70)

    documents = load_documents()

    chunks = split_documents(documents)

    vector_store = create_vector_store(chunks)

    llm = create_llm()

    prompt = create_prompt()

    print("\nSystem ready.")

    while True:

        question = input(
            "\nAsk a question (type 'exit' to quit): "
        ).strip()

        if question.lower() == "exit":
            print("\nExiting...")
            break

        if not question:
            continue

        ask_question(
            question,
            vector_store,
            llm,
            prompt
        )


if __name__ == "__main__":
    main()