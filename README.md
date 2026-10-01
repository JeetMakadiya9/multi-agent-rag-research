# Multi-Agent RAG — Backend Milestone 1

This package is the first backend milestone for the thesis project:

**A Multi-Agent Retrieval-Augmented Generation Framework for Evidence-Grounded Research and Fact Verification**

## Included

- `research_state.py` — structured research state and provenance records.
- `research_tools.py` — provider-independent search boundary + deterministic test provider.
- `research_retrieval.py` — adapter boundary around the existing advanced RAG engine.
- `llm.py` — provider-independent local LLM interface with Ollama and a static test provider.
- `claim_verification.py` — claim extraction and claim-level verification using an LLM provider.
- `research_pipeline.py` — Experiment A/B wrappers for baseline RAG vs claim-level verification.
- `rag.py` — the existing advanced retrieval engine used by the adapter.

## Important

This milestone does **not** yet contain the LangGraph autonomous research loop or Streamlit Research Lab UI. Those are later milestones.

The included `rag.py` is a copy of the existing retrieval source used as the integration dependency. Keep your original project copy backed up before replacing anything in your Windows project.

## Recommended installation

Open `D:\MultiAgent_RAG` in VS Code and activate the project's Python 3.12 environment.

Install the dependencies already used by the project, including:

```text
langchain
langchain-community
langchain-ollama
langchain-huggingface
langchain-text-splitters
sentence-transformers
faiss-cpu
pypdf
python-dotenv
rank-bm25
```

## Local LLM

The default LLM provider uses Ollama at:

`http://127.0.0.1:11434`

Default model: `qwen3:4b`.

Make sure Ollama is running and the model is available before using LLM-backed verification.

## Verification performed for this package

- Python compilation passed for all included Python modules.
- Research state round-trip test previously passed (`RESEARCH_STATE_OK`).
- Search provider deterministic test passed (`STATIC SEARCH TEST: PASS`).
- Retrieval adapter deterministic integration test passed after packaging preparation.

## Next milestone

The next engineering step is to connect this backend into an autonomous research loop, then move orchestration to LangGraph and add evaluation hooks before building the Streamlit Research Lab UI.
