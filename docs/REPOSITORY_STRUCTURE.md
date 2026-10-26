# Repository structure

What each part of the repository is, and which parts the local application needs. Launch instructions: `README.md`.

## The application (needed to run it)

| Path | What it is |
|---|---|
| `start_studio.bat` | Windows launcher: runs `dashboard/launch.py` with the project's `.venv` |
| `dashboard/` | the local dashboard: `app.py` (Streamlit entry point), `views.py` (pages), `client.py` (client of the Studio API), `launch.py` (starts and stops the Studio backend and the dashboard on 127.0.0.1) |
| `.streamlit/config.toml` | dashboard settings: loopback address, port, dark theme |
| `webrag/` | the Research Studio: `server.py` (HTTP server, job queue, API), `orchestrator.py` and `deep.py` (pipeline), `agents.py` (planner, synthesizer), `researcher.py`, `search.py`, `net.py`, `extract.py` (sources), `evidence.py` (retrieval), `report.py`, `studio.py`, `integrations.py`, `model_service.py` (model configuration), `ui.html` and `static/` (the original Studio interface) |
| `docqa/` | verifiers (the Studio default, Arm C, is in `agentic.py`), the local Ollama client (`llm.py`), the optional remote model clients (`remote_llm.py`), document indexing and the document assistant |
| `rag.py` | retrieval engine (embeddings, BM25, reranking) used through `docqa/index.py` and `webrag/evidence.py` |
| `claim_verification.py`, `experiment_b_verification.py`, `verification_units.py`, `llm.py` | the earlier verifier imported by the `frozen_b` verifier (`docqa/verify.py`) |
| `evidence_core/` | evidence contracts; the application uses `span_attestation.py` (quote attestation, on by default in the Studio for quoting verifiers, through `docqa/attestation.py`) |
| `requirements.txt`, `requirements-lock.txt` | dependencies: direct ones, and every package pinned for Windows x64 / Python 3.12 |
| `.env.example` | template for the optional `.env` (model service, provider keys); `.env` itself is ignored by git |

Several modules are frozen research baselines and must stay byte-identical; `ARCHITECTURE_DECISION_VERSION_BOUNDARIES.md`
lists them.

## Public showcase (separate website; not needed to run the application)

| Path | What it is |
|---|---|
| `showcase/app.py` | entry point of the static showcase (Streamlit Community Cloud main file) |
| `showcase/pages.py` | its eight pages: overview, architecture, workflow, components, results, evaluation runs, install, use cases. Results leads with the headline figures, two charts and the milestones; Evaluation runs lists every run with its quality, cost, reliability, evidence coverage and provenance |
| `showcase/content.py` | component, workflow and installation facts, with real file paths (checked by the tests) |
| `showcase/architecture.py` | builds the architecture diagram from the imports in the source code |
| `showcase/results.py` | reads the showcase data and selects what the pages may show. **Phase 11B is not presented**: `EXCLUDED_PHASES` removes Phase 11 runs, and diagnostics whose subject is a Phase 11 run, from every selection, ranking and aggregate. Its files stay in `evaluation/experiments/results/` (tested in `tests/test_showcase_evaluations.py`) |
| `showcase/data/` | link-only recorded runs, aggregate development findings, the agentic verification study's aggregate results, the alternative evaluations' aggregates and the evaluation run registry `evaluation_runs.json` (no per-claim records) |
| `showcase/make_examples.py` | turns a saved run into a link-only example (no third-party text) |
| `showcase/make_evaluation_runs.py` | builds `showcase/data/evaluation_runs.json` offline from the private run folders: run-level aggregates only, with every field's source documented in the script |
| `showcase/requirements.txt` | the showcase's only dependencies (Streamlit) |

## Tests

| Path | What it is |
|---|---|
| `tests/*.py` | offline Python tests (`python -m pytest -q`); no model or network is used. `tests/test_showcase.py` checks the showcase; `tests/test_arm_c_studio_integration.py` runs the default Arm C verifier through the real orchestration and HTTP API with scripted model replies |
| `tests/js/*.test.mjs` | tests of the Studio interface modules (`node --test tests/js/*.test.mjs`) |

## Research material (kept; not needed to run the application)

| Path | What it is |
|---|---|
| `research_*.py` | the project's earlier research-agent modules (planning, autonomy, capabilities, loops), with their tests |
| root location of the research files | the root `research_*.py`, `baseline_*`, `experiment_b_*` and frozen modules stay at the root: recorded configuration locks, run manifests, provenance inventories and tests refer to them by path, and the modules import one another as top-level modules |
| `baseline_*.py`, `experiment_b_*.py`, `diagnose_experiment_b_live.py` and their `*_results.jsonl` / `*_summary.json` | benchmark scripts and their recorded results |
| `src/baseline_rag.py` | the first LangChain baseline; its packages (`langchain_huggingface`, `langchain_ollama`) are not in `requirements.txt` |
| `evaluation/` | experiment code, protocols and recorded results, including provenance inventories (for example `evaluation/experiments/results/phase10g_provenance/`). `evaluation/experiments/agentic_verify/` holds the frozen Arm C code (`arms.py`, `prompts.py`, `llm.py`, `FROZEN_CONFIG.json`) that the Studio's default verifier runs; the study's per-claim records are not included (its aggregate results are in `showcase/data/historical_agentic_verification.json`). `evaluation/studio_eval/` holds the frozen protocol, harness and scorer for evaluating the Studio verifier, the annotation and freeze workflow (`ANNOTATION_GUIDE.md`, `DATASET_CONSTRUCTION.md`, `annotate.py`), and the alternative proxy evaluations run without independent labels (`ALT_BENCHMARKS.md`, `alt_benchmarks.py`, `alt_excluded_train_ids.json`; aggregates in `showcase/data/studio_alt_evaluation.json`). The protocol evaluation is not yet run, because it needs an independently labelled claim set, which is kept private |
| `data/scifact/` | the SciFact dataset used by the evaluations (its own licence; see `README.md`). The `._*` files are macOS archive metadata; they are kept because the recorded provenance inventory lists them |
| `EXPERIMENT_B_CONTRACT.md`, `ARCHITECTURE_DECISION_VERSION_BOUNDARIES.md` | research decisions and the frozen-module list |

Run logs under `evaluation/` are part of the recorded results and are kept.

## Not in the repository

| Path | Why |
|---|---|
| `.venv/` | the local virtual environment |
| `.env`, `.streamlit/secrets.toml` | local settings and keys |
| `outputs/` | saved runs (they contain text fetched from the web), uploaded documents and Studio settings |
| Hugging Face model cache | the retrieval models are downloaded to the user's cache on first use |

## Other files

| Path | What it is |
|---|---|
| `README.md` | what the project is, how a run and verification work, historical results, setup, launch, configuration, limitations |
| `webrag/README.md` | Studio options, network safety, known gaps |
| `LICENSE` | MIT licence for the project's own code and documentation |
| `docs/assets/` | images used by the README |
| `.gitignore`, `.gitattributes` | ignored local files; exact bytes kept on checkout |
| `.devcontainer/devcontainer.json` | Dev Container / GitHub Codespaces configuration (Python 3.11 image) that installs the requirements and starts the dashboard; the dashboard needs the Studio backend running, and the project itself is tested on Python 3.12 |
