# Multi-Agent RAG — Research Studio

![Multi-Agent RAG — Intelligence under evidence](docs/assets/research-studio-poster.jpg)

Multi-Agent RAG is an M.Tech research project on retrieval-augmented generation that checks its own claims. Language
models answer research questions fluently even when no source supports the answer. This project splits research into
separate, inspectable steps:
* a **planner** writes search queries;
* a **web researcher** reads scholarly and web sources;
* a **retrieval engine** ranks passages;
* a **synthesizer** drafts cited statements;
* a **verifier** checks every statement against the retrieved passages.

Only statements the verifier supports are reported as findings, and the sources, passages, conflicts and search limits
stay visible.

The software runs as a **local application on your own computer**, by default with the `qwen3:4b` model in Ollama.
Its verdicts are judgements by a language model about the retrieved text, not proof that a statement is true.

* **Public showcase:** https://multi-agent-rag-research.streamlit.app. It is a static website about the project: an
  overview, an architecture explorer computed from the source code, the research workflow with recorded
  demonstrations, the historical results with their methodology, the current evaluation status, what can be
  reproduced, installation instructions, and use cases and limits. It runs no model and starts no research.
* **The application:** the Research Studio and its dashboard, installed and run locally (below).

## Quick start (Windows)

```text
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
ollama pull qwen3:4b
.\start_studio.bat
```

The launcher checks the setup, starts both parts and opens the dashboard:

| | Address | What it is |
|---|---|---|
| **Dashboard** | http://127.0.0.1:8501 | Streamlit command center: Command Center, Research Workspace, Research Library, Evidence Explorer, Model & Integrations, Settings & Diagnostics |
| **Research Studio** | http://127.0.0.1:8765 | the original Studio interface and the backend (API, job queue, pipeline) that the dashboard talks to |

In PowerShell keep the leading `.\`; in `cmd` it is optional. Both listen on `127.0.0.1` only. Stop both with **Ctrl+C**
in the launcher window.

## How a run works

```mermaid
flowchart LR
    Q[Question or claim] --> P[Planner<br/>subquestions, queries]
    P --> S[Search providers<br/>OpenAlex, arXiv, Crossref,<br/>Europe PMC, web, feeds]
    S --> R[Reader<br/>SSRF-guarded fetch,<br/>HTML/PDF extraction]
    R --> I[Per-source index<br/>BM25 + dense + rerank]
    I --> E[Evidence ranking<br/>injection screen: exclude]
    E --> V[Verifier: Arm C, arm_c_studio<br/>stances with quotes, adjudicator for conflicts,<br/>planner search when nothing decides]
    V --> D[Deep research loop<br/>coverage, counter-evidence,<br/>quality labels, stopping record]
    D --> O[Report and Studio<br/>verified findings only,<br/>numbers checked in summaries]
```

* **Research questions.** The planner splits the question into subquestions and writes queries. The web researcher
  searches the active providers and reads the sources it can reach. Network access is guarded against private
  addresses and pinned to checked addresses. Wikipedia is not accepted as an evidence source.
* **Retrieval.** Each source is indexed separately (BM25, dense embeddings, cross-encoder reranking), and passages are
  ranked per question or claim.
* **Verification.** The synthesizer drafts cited statements, and the verifier judges each one against its passages
  (SUPPORTS / CONTRADICTS / NEUTRAL, with the sentence it relies on).
* **Verdicts.** SUPPORTED needs a supporting passage, CONTRADICTED a contradicting one, and anything else is
  INSUFFICIENT.
* **Deep research** (quick, standard and deep budgets) adds bounded search rounds, coverage per subquestion,
  counter-evidence searches, evidence-quality labels (single source, abstract only, independence unknown),
  deterministic cautions and a recorded reason for stopping.
* **Claim verification** (`--verify` or the Studio's verify mode) retrieves passages for one claim and returns its
  verdict with every passage's stance.
* **Reports.** Runs are saved with their stages, sources, passages, stances and limits, and can be exported as
  Markdown or JSON.

## Verification in the Research Studio

**Default verifier: Arm C agentic verification, Studio-adapted variant (`arm_c_studio`).** It runs the frozen code of
the project's agentic verification study, arm C (`docqa/agentic.py` over `evaluation/experiments/agentic_verify/`).
For each statement:
1. One verifier call judges the retrieved passages. Each decisive stance needs an exact quote from its passage.
2. If the passages agree, that is the verdict.
3. If passages conflict, an adjudicator decides which side tests the exact claim. The statement is then marked
   *adjudicated*, both sides stay listed, and a supported statement stays out of the direct answer and summary.
4. If no passage took a side, a planner may search the evidence already collected for this run (at most two searches,
   with no new web requests). New passages are judged the same way before adjudication.
5. Without a passage for or against the claim, nothing is decided: the statement stays INSUFFICIENT.
6. Failures are reported, never replaced by another verifier: a model-service error, unusable output, an invalid
   stance, or a failed or impossible search.

`arm_c_studio` differs from the historical method (`arm_c`, also selectable) in three fail-closed rules:
* no adjudicator decision is accepted without a passage for or against the claim (rule 5);
* an adjudication that cites a passage judged for the other side is not accepted; the study's rank rule decides
  instead;
* if the verifier gives a passage an invalid stance, the claim is not assessed, rather than read as neutral as in the
  study.

A passage the verifier's reply leaves out is still read as neutral, as in the study, because the model routinely omits
passages it finds irrelevant. Each omission is recorded with the verdict. An omitted passage that would have
contradicted the claim cannot be detected.

The last two rules were added on 2026-10-24 after an adversarial test campaign showed incomplete or contradictory
model output producing SUPPORTED. No evaluation data was involved.

Every claim records which verifier ran and a trace of its steps. Passages found by the planner's search are marked.

**Safeguards, on by default:**
* **Injection screen** (`exclude`). Retrieved passages that look like injected instructions are withheld from every
  model call, and the run records which passages were withheld.
* **Quote attestation** (record only). The position of each verifier quote in its passage is recorded. It never
  changes a verdict.
* **Fixed settings.** The verification settings are fixed when the Studio starts; a run request cannot change them.

**Alternative verifiers** (set `WEBRAG_VERIFIER`):
| Verifier | What it does |
|---|---|
| `arm_c` | the historical Arm C exactly, including adjudicator decisions made without any passage for or against |
| `arm_c0` | historical arm C0: adjudication without the planner's search |
| `quote_stance` | the previous default: one model call per passage; conflicting evidence stays INSUFFICIENT |
| `quote_stance_ranked` | `quote_stance`, with conflicting evidence decided by the first-ranked passage with a stance (not Arm C) |
| `quote_stance_adjudicated` | `quote_stance` with an adjudicator asked in both passage orders (not Arm C; it raised false support in exploratory runs, so it is not recommended) |
| `frozen_b` | the earlier Experiment B verifier |
| `verifier_v2` | the new owned verifier (opt-in, not yet evaluated): a relevance gate before any stance, one structured-output call per passage, calibrated aggregation (one weak or unconfirmed contradiction no longer decides a claim) and an optional NLI pre-filter; see `docqa/verifier_v2.py` |

**Signals on every verdict.** Each statement shows a confidence level with the count of passages for and against it
(a count, not a probability). An INSUFFICIENT statement says why nothing was decided, for example that no passage
tested the claim's population. With `verifier_v2`, each passage also shows why it counted or did not.

**Caching.** Documents already read, and `verifier_v2`'s passage judgments, are kept in memory for the running Studio,
so a follow-up or re-run does not fetch or judge them again (`WEBRAG_READ_CACHE=0` switches page caching off).

**A larger model.** `WEBRAG_LLM_PROVIDER` (`ollama` or `openai`) with `WEBRAG_LLM_BASE_URL`, `WEBRAG_LLM_API_KEY` and
`WEBRAG_LLM_VERIFIER_MODEL` sends verification to a larger model on another server or API while the local 4B model
stays the private default. `verifier_v2`'s NLI filter uses `WEBRAG_NLI_MODEL` (default
`cross-encoder/nli-deberta-v3-xsmall`, from the local Hugging Face cache; `off` disables it).

## Results

* **Agentic verification study (2026-10-23):** Arm C reached a two-class macro-F1 of 0.7551 on 188 SciFact claims,
  +0.0398 over the single-pass rank baseline (not statistically significant, p = 0.0813).
* **Current verifier on SciFact proxy sets (2026-10-24 to 26):** three-class macro-F1 0.50–0.62 depending on the set;
  the project's three-class target of 0.75 is not met yet. On the blind confirmation set, the corrected judge cut
  false support from 32 to 10. Every run is listed, with its provenance, on the showcase's Evaluation runs page.
* **The Research Studio on live web evidence has not been evaluated.**

Full results, definitions, caveats and what can be reproduced: `docs/RESULTS.md`. Two-class and three-class
macro-F1 are explained in `evaluation/studio_eval/TWO_VS_THREE_CLASS.md`.

## Prerequisites

* Windows 10/11 x64 and **Python 3.12** (`py -3.12 --version`). The pinned packages are for Python 3.12 on Windows.
* **Ollama** (https://ollama.com) with the model: `ollama pull qwen3:4b`. Without Ollama, the Studio and dashboard still
  start, show saved reports and say that the model is not reachable.
* Internet access for the search providers. The first research run downloads the two retrieval models
  (`all-MiniLM-L6-v2`, `cross-encoder/ms-marco-MiniLM-L-6-v2`, about 180 MB) into the Hugging Face cache.
* Node.js, only to run the interface tests.

## Install

```text
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements.txt` lists the direct dependencies. `requirements-lock.txt` pins every package for Windows x64 /
Python 3.12; `pip install -r requirements-lock.txt` reproduces the tested environment exactly.

## Configure (optional)

Nothing needs configuring for the defaults. To change them, copy `.env.example` to `.env` (ignored by git), edit it,
and restart the launcher:

```text
copy .env.example .env
notepad .env
```

* **Another local model:** `WEBRAG_LLM_MODEL=llama3.2:3b` (pull it in Ollama first).
* **An OpenAI-compatible API or an Ollama server elsewhere:** set `WEBRAG_LLM_PROVIDER` (`openai` or `ollama`),
  `WEBRAG_LLM_BASE_URL` (`https://...`), `WEBRAG_LLM_MODEL` and, if needed, `WEBRAG_LLM_API_KEY`. Prompts, questions
  and retrieved passages are then sent to that service. The key stays in the Studio server process; the interfaces
  show only whether it is set.
* **Unavailable model:** the run stops with that reason. No other model or provider is substituted.
* **Verification:**
  * `WEBRAG_VERIFIER`: `arm_c_studio` by default; see the alternatives above.
  * `WEBRAG_INJECTION_GUARD`: `exclude` by default, or `flag` or `off`.
  * `WEBRAG_QUOTE_ATTESTATION`: on by default when the verifier quotes; `0` turns it off.

  The launcher prints what it will use, and an invalid value stops the start. A `python -m webrag.server` flag wins
  over the setting.
* **Search providers:** `OPENALEX_API_KEY`, `WEBRAG_CONTACT`, `SEARXNG_URL` and `BRAVE_API_KEY`; see `.env.example`.

The dashboard's *Model & Integrations* page shows the model, the active verification workflow and where each setting
came from, as the running Studio reports them. *Settings & Diagnostics* shows the configuration and data locations.

## Launch and stop

```text
.\start_studio.bat                                      # both parts, opens http://127.0.0.1:8501
.\start_studio.bat --no-browser                         # without opening a browser
.\start_studio.bat --studio-port 8766 --dashboard-port 8502
.\start_studio.bat --check                              # only check the setup
```

`start_studio.bat` runs `.venv\Scripts\python.exe -m dashboard.launch`. Stop it with **Ctrl+C** in its window; the
launcher stops both parts. Each part can also run on its own:

```text
.venv\Scripts\python.exe -m webrag.server                     # Research Studio only, http://127.0.0.1:8765
.venv\Scripts\python.exe -m streamlit run dashboard/app.py    # dashboard only (needs the Studio running)
.venv\Scripts\python.exe -m webrag "a research question"      # command line, no interface
.venv\Scripts\python.exe -m webrag --verify "a claim"
.venv\Scripts\python.exe -m streamlit run showcase/app.py     # the public showcase, locally
```

* **Saved runs** go under `outputs/webrag/runs/`. They are not versioned, because they contain fetched web text.
* **Command-line defaults.** The command-line tool keeps its own defaults (`quote_stance`, screen and attestation off),
  so earlier results stay reproducible.
* **More detail** (Studio options, deep research, network safety, evaluation): `webrag/README.md`.

## Troubleshooting

| Problem | What to do |
|---|---|
| `No virtual environment found` | create it: `py -3.12 -m venv .venv`, then install the requirements |
| `Missing packages: ...` | `.venv\Scripts\python.exe -m pip install -r requirements.txt` |
| `Ollama is not reachable` | start the Ollama app; `ollama list` should answer |
| `qwen3:4b is not installed` | `ollama pull qwen3:4b` (or set another installed model in `.env`) |
| `Port 8765/8501 is used by another program` | close that program, or `.\start_studio.bat --studio-port 8766 --dashboard-port 8502` |
| `Configuration problem: ...` | fix the named setting in `.env` (see `.env.example`) |
| dashboard shows *Studio backend is not running* | start with `.\start_studio.bat`, or start `python -m webrag.server` and reload |
| first run is slow or fails while loading models | the retrieval models are downloaded once; check the internet connection |
| a search provider fails (for example OpenAlex `429`) | the run continues with the others and lists the error; `OPENALEX_API_KEY` in `.env` avoids the shared budget |

## Tests

Offline, with no model and no network:

```text
.venv\Scripts\python.exe -m pytest -q
node --test tests/js/*.test.mjs
```

The tests cover:
* the pipeline, deep research and network safety;
* the Arm C integration: the real orchestration and HTTP API with scripted model replies;
* configuration, the dashboard and the showcase.

## Limitations

* **Model errors.** A 4B local model drafts and judges the statements, and it makes mistakes.
* **Strict quoting can miss true claims.** Arm C demotes a stance whose quote is not text of its passage. In a small
  live check, a true claim ended INSUFFICIENT because the model mis-copied its quote.
* **Adjudicated verdicts are less reliable.** In the historical study, the adjudicator raised false support. Here,
  adjudicated statements are marked, keep both sides and stay out of the direct answer, which reduces but does not
  remove this risk.
* **The injection screen withholds whole passages.** That includes any genuine evidence in the same passage, so
  evidence for or against a claim can be missing; the run's limitations say so. The screen matches fixed patterns and
  misses paraphrased attacks.
* **Quote checks in `quote_stance`.** Its quote check is tolerant. An exploratory replay found accepted quotes that
  were not contiguous text of their passage, for example mis-copied numbers. Quote attestation records this and does
  not change verdicts.
* **Search coverage** depends on what the providers return and what can be read; many sources are abstract-only.
* **Not for networks.** The Studio and dashboard bind to 127.0.0.1, have no authentication and are not meant to be
  exposed to a network.

## Repository layout

| Path | What it is |
|---|---|
| `webrag/` | the Research Studio: server, pipeline, deep research, sources, retrieval, reports, interface |
| `docqa/` | verifiers (including Arm C in `agentic.py`), model clients, document indexing and the document assistant |
| `dashboard/` | the local Streamlit dashboard and the launcher (`start_studio.bat` runs it) |
| `showcase/` | the public static showcase |
| `evaluation/` | experiment code, protocols and recorded results, including the frozen Arm C code |
| `evidence_core/` | evidence contracts, including span attestation |
| `tests/` | offline Python tests and interface tests (`tests/js/`) |
| root `*.py` files | the earlier research backend and benchmark scripts with their recorded results; they stay at the root because recorded manifests, provenance inventories and tests refer to them by path |

What each file is and which ones the application needs: `docs/REPOSITORY_STRUCTURE.md`. Frozen modules:
`ARCHITECTURE_DECISION_VERSION_BOUNDARIES.md`.

## Not included in this repository

These are kept private, and some documents refer to them by path:
* the thesis drafts;
* the agentic verification study's protocol, per-claim records and analysis (its frozen code and aggregate results
  are included);
* the evaluation command line and its held-out claim sets;
* evaluation run records;
* other unpublished research material.

## License

The project's own code and documentation are released under the MIT License (see `LICENSE`). The MIT License does not
cover third-party material in this repository, which keeps its own terms:
* `data/scifact/`: the SciFact dataset (claims and a corpus of scientific abstracts) by its original authors. Use it
  under the dataset's own licence and citation requirements, published with the dataset; this repository has not
  verified those terms.
* Titles, abstracts and other text of third-party publications that appear in data files or test fixtures remain the
  property of their authors and publishers.
* The Python packages in `requirements.txt` and `requirements-lock.txt` are not bundled; each is distributed under its
  own licence.
