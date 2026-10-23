# Hosted Research Studio (Streamlit)

The hosted app serves the **original Research Studio** (`webrag/ui.html`, `webrag/static/`) and runs its real
backend: the Studio server (`webrag/server.py`: job queue, run store, document library, provider settings and every
`/api` route), the orchestrator, planner, web researcher, retrieval and ranking (`rag.py`), verifier and report code.
Nothing is reimplemented for the web.

```
browser ── Streamlit server (streamlit_cloud/streamlit_app.py, st.App)
             ├── /                Streamlit pages: the Studio in a frame, and a quick source search
             └── /studio/         gateway (webrag/hosted_studio.py): sign-in, cross-site refusal, size limits
                                    └── Studio server on 127.0.0.1 inside the same process (webrag/server.py)
                                          ├── search providers (OpenAlex, arXiv, Crossref, Europe PMC, ...)
                                          ├── retrieval models (MiniLM embeddings + cross-encoder, CPU)
                                          └── model service: Ollama API or OpenAI-compatible API (server-side key)
```

Why this shape: the Studio is a browser application that talks to its own HTTP API, and Streamlit Community Cloud
exposes only Streamlit's port. `st.App` (in the pinned Streamlit 1.65) can mount extra routes on that port, so the Studio and its
API are served by the same process, and no second hosted service is needed. A language model cannot run on Community
Cloud, so the model is a separate service that you choose.

## Deploying on Streamlit Community Cloud

| Setting | Value |
|---|---|
| Repository / branch | `JeetMakadiya9/multi-agent-rag-research`, the branch with this folder |
| Main file path | `streamlit_cloud/streamlit_app.py` |
| Python version (Advanced settings) | **3.12** (the CPU-only torch wheel in `requirements.txt` is for Python 3.12 on Linux) |
| Dependencies | `streamlit_cloud/requirements.txt` (used automatically: it sits next to the main file) |

### Secrets (Settings → Secrets, top-level TOML keys)

Required:

| Name | Meaning |
|---|---|
| `STUDIO_ACCESS_TOKEN` | the sign-in token for the Studio, at least 16 characters. Without it the Studio stays locked. |
| `WEBRAG_LLM_PROVIDER` | `ollama` (an Ollama server reached over the network, native API) or `openai` (an OpenAI-compatible chat-completions API) |
| `WEBRAG_LLM_BASE_URL` | the service address, `https://...`; for `openai` include the version prefix, e.g. `https://api.example.com/v1` |
| `WEBRAG_LLM_MODEL` | the model name exactly as the service lists it |
| `WEBRAG_LLM_API_KEY` | the service's API key, when it needs one |

Optional: `WEBRAG_LLM_VERIFIER_MODEL` (a separate model for the verifier, same service), `WEBRAG_LLM_JSON_MODE`
(`schema` default, `object` or `none` for services that do not accept JSON schemas), `WEBRAG_LLM_TIMEOUT` (seconds,
default 180), `WEBRAG_VERIFIER` (default `quote_stance`), `WEBRAG_INJECTION_GUARD` (`off`, `flag`, `exclude`),
`OPENALEX_API_KEY`, `WEBRAG_CONTACT`, `SEARXNG_URL`, `BRAVE_API_KEY`, `WEBRAG_ALLOW_PROXY`.

Example (values are placeholders):

```toml
STUDIO_ACCESS_TOKEN = "a-long-random-token-you-keep-private"
WEBRAG_LLM_PROVIDER = "openai"
WEBRAG_LLM_BASE_URL = "https://api.example.com/v1"
WEBRAG_LLM_MODEL = "provider-model-name"
WEBRAG_LLM_API_KEY = "..."
```

Keys stay on the server: the browser never receives them, `/api/status` reports only the service address
(scheme and host), and run records store the model name and service address, not the key. `.streamlit/secrets.toml`
is ignored by git.

### Choosing a model service

* **Your own Ollama with `qwen3:4b`** (`WEBRAG_LLM_PROVIDER = "ollama"`): the configuration the verifier was evaluated
  with. The server must be reachable over HTTPS from the internet (for example behind a reverse proxy or tunnel that
  adds authentication); put that proxy's bearer token in `WEBRAG_LLM_API_KEY`.
* **An OpenAI-compatible service** (`openai`): any service with `/v1/chat/completions` and `/v1/models`. Verdict quality
  with other models has **not** been evaluated; the Studio says so on its model settings page.

The model is never substituted: if the service does not list the configured model, runs stop with that explanation.

## What works, and its limits

| Feature | Hosted status |
|---|---|
| Studio interface: home, research and verification, progress, report, Evidence panel, Sources, Passages, Coverage, Library, Integrations, Settings, export | served from the original files |
| Research and claim verification, deep research (quick/standard/deep) | runs when the model service is configured and reachable |
| Saved runs (Library) | saved under `outputs/webrag/runs/` on the app's disk, as in the desktop Studio: **lost when Streamlit Cloud restarts or redeploys the app** (Community Cloud provides no persistent storage); export a report (Markdown/JSON) to keep it |
| Document library (upload TXT/PDF, ask questions) | works; files under `outputs/hosted/documents/` (`WEBRAG_DATA_DIR`), same storage limit as runs |
| Obsidian export | not available (needs a local vault); the Studio shows it as unavailable |
| Quick source search page (no model) | works |

Other limits:

* **Resources.** Community Cloud apps have limited memory and a shared CPU (see Streamlit's current resource limits).
  Retrieval runs on the CPU (torch, the two retrieval models and faiss take several hundred MB of memory); a quick
  research run takes minutes, and one job runs at a time (the Studio's own queue).
* **First run** downloads the two retrieval models (about 180 MB) from the Hugging Face Hub
  (`WEBRAG_MODEL_DOWNLOADS`: `auto` default, `1`, `0`).
* **Data sent elsewhere.** Every model call sends prompts, the question and retrieved passages to the configured model
  service. Searches go to the configured providers. Without `OPENALEX_API_KEY`, OpenAlex shares a daily budget per
  network address.
* **Access control** is a single shared token: everyone with it can run research (using your model service and its
  costs), see every saved run and change the Studio's provider settings. There are no per-user accounts.
* The sign-in session lasts 12 hours. Failed sign-ins are limited per address.

## Running it locally

```text
pip install -r requirements.txt streamlit==1.65.0
set STUDIO_ACCESS_TOKEN=a-long-random-token-you-keep-private
streamlit run streamlit_cloud/streamlit_app.py
```

Without `WEBRAG_LLM_PROVIDER` the hosted Studio uses the local Ollama exactly like `python -m webrag.server`, so the
local Studio is unchanged; `python -m webrag.server` remains the simplest way to run it on your own computer.
