# Hosted front end (Streamlit)

`streamlit_app.py` is a Streamlit front end for the Research Studio. It imports the project as packages (`webrag`,
`docqa`) and shows only what the machine it runs on can do. The full Studio (`python -m webrag.server`) is unchanged
and remains the main local application.

## Deploying on Streamlit Community Cloud

| Setting | Value |
|---|---|
| Repository | `JeetMakadiya9/multi-agent-rag-research` |
| Branch | the branch that contains this folder |
| Main file path | `streamlit_cloud/streamlit_app.py` |
| Python version (Advanced settings) | 3.12 |
| Dependencies | `streamlit_cloud/requirements.txt`, used automatically because it sits next to the main file |

Do not use `webrag/server.py` as the main file: it is the local Studio's HTTP server, a module of the `webrag`
package, and fails when run as a script (`attempted relative import with no known parent package`). If the main
file path of an existing app cannot be changed in its settings, delete the app and deploy it again with the path
above.

The entry point lives in its own folder so that Streamlit Cloud installs the light dependency file next to it
(Streamlit, requests, urllib3, pypdf) and not the root `requirements.txt`, which brings in sentence-transformers,
torch and faiss for the local pipeline.

Optional secrets (Settings → Secrets, as top-level TOML keys; Streamlit exposes them as environment variables):

| Name | Effect |
|---|---|
| `OPENALEX_API_KEY` | OpenAlex requests use your own free key instead of the shared anonymous daily budget |
| `WEBRAG_CONTACT` | contact address added to the user agent of outgoing requests |
| `SEARXNG_URL` | enables SearXNG web search through your own instance |
| `BRAVE_API_KEY` | enables the Brave Search API (paid) |
| `WEBRAG_ALLOW_PROXY` | set to `1` only if the host routes outbound traffic through a proxy and searches fail with `proxy_policy` |

`.streamlit/secrets.toml` is ignored by git; never commit keys.

## What works where

| Feature | Streamlit Community Cloud | Local machine with Ollama and `requirements.txt` |
|---|---|---|
| Search sources (OpenAlex, arXiv, Crossref, Europe PMC; SearXNG/Brave when configured) | yes | yes |
| Open a saved run record (`run.json`) and render its report, statements and sources | yes | yes |
| Research a question / verify a claim | no: switched off, with the reason | yes |

Research and verification need a local Ollama server with `qwen3:4b` on `127.0.0.1:11434`, the retrieval packages
and the retrieval models in the local Hugging Face cache. The project never sends text to a remote model, so a
hosting service cannot run them. The sidebar shows which of these were found.

## Limits

* Search results are shown as the providers return them; no model reads or judges them on the hosted app.
* Without `OPENALEX_API_KEY`, OpenAlex requests share a daily budget per network address, which a shared host can
  use up; the error is shown when it happens.
* Searches are cached for ten minutes per query and are not saved. Uploaded run records are read in memory and not
  stored; files over 20 MB are refused.
* Wikipedia is never searched (the project's source policy does not accept it as evidence); RSS/Atom feeds need a
  feed list that only the local Studio keeps.

## Running it locally

```text
pip install -r streamlit_cloud/requirements.txt
streamlit run streamlit_cloud/streamlit_app.py
```

With the full root `requirements.txt` installed and Ollama serving `qwen3:4b`, the *Research and verify* tab runs
the Studio's pipeline on the local machine. Such a run is not saved under `outputs/`; download its record from the
page.
