"""What the showcase says about the project, in one place. Every file path here is a real repository path and is
checked by tests/test_showcase.py; code dependencies between components are not written here at all: they are
computed from the source by showcase/architecture.py."""
from __future__ import annotations

REPO = "JeetMakadiya9/multi-agent-rag-research"
BRANCH = "release/public-candidate-2"
REPO_URL = f"https://github.com/{REPO}"


def source_url(path: str) -> str:
    kind = "tree" if path.endswith("/") else "blob"
    return f"{REPO_URL}/{kind}/{BRANCH}/{path.rstrip('/')}"


# Status categories, from strongest to weakest claim.
STATUS = {
    "implemented": ("Implemented", "green", "In the code and used by the application."),
    "tested": ("Tested (offline)", "blue", "Covered by offline tests with stubbed models; not a measure of research quality."),
    "evaluated": ("Evaluated", "violet", "Measured in a recorded benchmark run (see Results)."),
    "experimental": ("Experimental", "orange", "Opt-in or exploratory; not recommended as a default."),
    "proposed": ("Proposed", "gray", "Specified or contract-only; the capability itself does not exist yet."),
}

# What kind of evidence a result is. Used for every result shown on the site.
EVIDENCE_TYPES = {
    "benchmark": ("Benchmark result", "violet", "A quantitative evaluation against a defined dataset and protocol."),
    "diagnostic": ("Engineering diagnostic", "orange",
                   "An investigation of one implementation or evidence-handling behaviour; not a measure of accuracy."),
    "demonstration": ("Recorded demonstration", "blue", "A saved example of the workflow's behaviour."),
    "historical": ("Historical baseline", "gray", "An earlier comparison kept for context; not the newest evaluation."),
    "unpublished": ("Unpublished / not substantiated publicly", "red",
                    "Not presented as a public performance claim: its evidence is not available here."),
    "software": ("Software test report", "green",
                 "Checks that code behaves as specified (stubbed models, no network); not research performance."),
}

# Sources of the dates shown for each result (repository files a visitor can open, or a stated private record).
PHASE11_RUN_DATES = ("2026-10-01", "2026-10-05")    # evaluation/experiments/results/phase10_full_dev/full_run_manifest.json
PHASE11_RUN_MANIFEST = "evaluation/experiments/results/phase10_full_dev/full_run_manifest.json"

# Later public work on the same research backend. Their own files describe them as diagnostics or engineering
# experiments, not benchmarks; the dates come from the named files.
LATER_DIAGNOSTICS = [
    {"name": "Phase 12 diagnosis of the Phase 11B run", "date": "2026-10-05",
     "date_source": "evaluation/experiments/results/phase12_diagnosis/phase12_audit.json",
     "folder": "evaluation/experiments/results/phase12_diagnosis/",
     "what": "Error and bottleneck analysis of the same run: most wrong B and C labels occurred even though a gold "
             "document had been retrieved, which points at the evidence-to-verdict step; the files state that the "
             "causal mechanism is unproven and that no intervention was run."},
    {"name": "Phase 13 single-factor experiments (B variants)", "date": "2026-10-05 to 2026-10-06",
     "date_source": "evaluation/experiments/results/phase13_improvement/chain_status.log",
     "folder": "evaluation/experiments/results/phase13_improvement/",
     "what": "Controlled one-change variants of System B on a 69-claim SciFact DEV development subset, which the "
             "result files themselves label as engineering only, not a benchmark."},
]

LAYERS = ["Entry points", "Studio server", "Research pipeline", "Sources", "Evidence", "Verification", "Models",
          "Research backend", "Evaluation", "External"]

# ------------------------------------------------------------------------------------------------ components
COMPONENTS = [
    {"id": "launcher", "name": "Launcher", "layer": "Entry points",
     "files": ["start_studio.bat", "dashboard/launch.py"],
     "purpose": "Checks the local setup and starts the Studio backend and the dashboard on 127.0.0.1; stops both together.",
     "inputs": "Command-line options, the optional .env file", "outputs": "Two local processes and their URLs",
     "status": ["implemented", "tested"], "tests": ["tests/test_dashboard.py"],
     "limits": "Windows batch launcher; the Python launcher itself is untested on Linux and macOS."},
    {"id": "dashboard", "name": "Local dashboard", "layer": "Entry points",
     "files": ["dashboard/app.py", "dashboard/views.py", "dashboard/client.py"],
     "purpose": "Command center, research workspace, library and evidence explorer for a local installation; a client of the Studio API.",
     "inputs": "Studio API responses", "outputs": "Pages in the browser; new research jobs submitted to the Studio",
     "status": ["implemented", "tested"], "tests": ["tests/test_dashboard.py"],
     "limits": "Needs the Studio backend running on the same computer; shows an offline notice otherwise."},
    {"id": "studio_ui", "name": "Research Studio interface", "layer": "Entry points",
     "files": ["webrag/ui.html", "webrag/static/app.mjs", "webrag/static/api.mjs", "webrag/static/workspace.mjs",
               "webrag/static/model.mjs"],
     "purpose": "The original browser interface: research and verification, report, evidence panel, sources, passages, coverage, library, integrations.",
     "inputs": "Studio API responses", "outputs": "Rendered reports; research requests",
     "status": ["implemented", "tested"], "tests": ["tests/js/studio_model.test.mjs", "tests/js/studio_deep.test.mjs"],
     "limits": "Local use only: no accounts or authentication."},
    {"id": "server", "name": "Studio server", "layer": "Studio server",
     "files": ["webrag/server.py", "webrag/studio.py", "webrag/integrations.py", "webrag/config.py"],
     "purpose": "HTTP API, single-worker job queue, run store, document library and status probe; answers only requests addressed to 127.0.0.1 or localhost.",
     "inputs": "HTTP requests from the interfaces", "outputs": "Jobs for the orchestrator; saved runs, exports, status",
     "status": ["implemented", "tested"], "tests": ["tests/test_webrag_studio.py", "tests/test_webrag.py"],
     "limits": "One job at a time; a started run cannot be interrupted safely."},
    {"id": "model_service", "name": "Model configuration", "layer": "Studio server",
     "files": ["webrag/model_service.py"],
     "purpose": "Chooses the language model from the environment or .env: local Ollama by default, or a remote Ollama / OpenAI-compatible service.",
     "inputs": "WEBRAG_LLM_* settings", "outputs": "Model clients for the pipeline and status probe",
     "status": ["implemented", "tested"], "tests": ["tests/test_model_service.py"],
     "limits": "Remote services are tested only against stubs; their verdict quality is not evaluated."},
    {"id": "orchestrator", "name": "Orchestrator and deep research", "layer": "Research pipeline",
     "files": ["webrag/orchestrator.py", "webrag/deep.py", "webrag/followup.py"],
     "purpose": "Runs a research or verification job: model check, planning, bounded search rounds, coverage, counter-evidence, synthesis, verification, summary, and the recorded reason for stopping.",
     "inputs": "A question or claim, depth and provider options", "outputs": "A run record (run.json)",
     "status": ["implemented", "tested"], "tests": ["tests/test_webrag_deep.py", "tests/test_webrag.py"],
     "limits": "Budgets (rounds, searches, sources, time) bound the search; results can be budget-limited."},
    {"id": "agents", "name": "Planner and synthesizer", "layer": "Research pipeline",
     "files": ["webrag/agents.py", "webrag/numeric.py"],
     "purpose": "Model-driven planner (subquestions, source types, queries) and synthesizer (cited statements, summary), with a whole-token number check on summary sentences.",
     "inputs": "Question; ranked passages", "outputs": "Research plan; cited statements; summary",
     "status": ["implemented", "tested"], "tests": ["tests/test_numeric_tokens.py", "tests/test_numeric_grounding.py"],
     "limits": "A 4B model writes the plan and statements; it makes mistakes, which is why every statement is verified."},
    {"id": "researcher", "name": "Web researcher", "layer": "Sources",
     "files": ["webrag/researcher.py", "webrag/dedupe.py"],
     "purpose": "Runs the planned queries across providers, selects and de-duplicates sources, and reads open-access full text where available.",
     "inputs": "Planned queries", "outputs": "Sources with text, access level and metadata",
     "status": ["implemented", "tested"], "tests": ["tests/test_webrag.py", "tests/test_webrag_deep.py"],
     "limits": "Many sources are abstract-only; coverage depends on what providers return."},
    {"id": "search", "name": "Search providers", "layer": "Sources",
     "files": ["webrag/search.py", "webrag/feeds.py"],
     "purpose": "Clients for OpenAlex, arXiv, Crossref, Europe PMC, SearXNG, Brave and RSS/Atom feeds; Wikipedia is excluded as evidence.",
     "inputs": "Queries", "outputs": "Search hits", "status": ["implemented", "tested"],
     "tests": ["tests/test_webrag_providers.py"],
     "limits": "OpenAlex without a key shares a daily budget; SearXNG and Brave need configuration."},
    {"id": "network", "name": "Guarded network and extraction", "layer": "Sources",
     "files": ["webrag/net.py", "webrag/extract.py"],
     "purpose": "Fetches pages and PDFs with scheme, port and address checks on every DNS answer, connection pinning, verified TLS and robots.txt; extracts text.",
     "inputs": "URLs", "outputs": "Extracted text", "status": ["implemented", "tested"],
     "tests": ["tests/test_webrag_netguard.py", "tests/test_webrag_redirects.py", "tests/test_webrag_tls_pinning.py"],
     "limits": "Requests through an environment proxy are refused unless WEBRAG_ALLOW_PROXY=1."},
    {"id": "evidence", "name": "Evidence store", "layer": "Evidence",
     "files": ["webrag/evidence.py", "webrag/guarded.py", "webrag/injection.py"],
     "purpose": "Splits sources into traceable passages and ranks them per question or claim; in the Studio a screen withholds passages that look like injected instructions (on by default).",
     "inputs": "Sources", "outputs": "Ranked passages with source and page", "status": ["implemented", "tested"],
     "tests": ["tests/test_injection_guard.py"],
     "limits": "The injection screen matches fixed patterns and misses paraphrased attacks; its own evaluation had recommended keeping it opt-in."},
    {"id": "docqa_index", "name": "Passage index", "layer": "Evidence",
     "files": ["docqa/index.py", "docqa/ingest.py", "docqa/pdf_utils.py"],
     "purpose": "Builds the per-source index over passages using the retrieval engine; also indexes local documents.",
     "inputs": "Pages of text", "outputs": "Retrievable passages", "status": ["implemented"], "tests": [],
     "limits": "Retrieval models must be downloaded once (about 180 MB)."},
    {"id": "rag_engine", "name": "Retrieval engine", "layer": "Evidence",
     "files": ["rag.py"],
     "purpose": "Hybrid retrieval: BM25 plus dense embeddings (all-MiniLM-L6-v2) and a cross-encoder reranker; a frozen research module.",
     "inputs": "Passages and a query", "outputs": "Ranked passages", "status": ["implemented", "tested", "evaluated"],
     "tests": ["tests/test_rag_retrieval_lineage.py"],
     "limits": "Runs on the CPU; frozen, so changes go through new modules."},
    {"id": "verifier", "name": "Claim verifier", "layer": "Verification",
     "files": ["docqa/agentic.py", "evaluation/experiments/agentic_verify/arms.py", "docqa/verify.py", "docqa/adjudicate.py", "docqa/ranked.py"],
     "purpose": "Judges each statement against each retrieved passage (SUPPORTS / CONTRADICTS / NEUTRAL with the quoted sentence) and decides SUPPORTED, CONTRADICTED or INSUFFICIENT; the default is Arm C agentic verification in a Studio-adapted variant (an adjudicator for conflicting evidence, planner searches when no passage takes a side, and no decision without a passage for or against); the historical Arm C and the previous per-passage verifier stay selectable.",
     "inputs": "A statement and its passages", "outputs": "A verdict with per-passage stances",
     "status": ["implemented", "tested"], "tests": ["tests/test_agentic_verifier.py", "tests/test_adjudicated_verifier.py", "tests/test_ranked_verifier.py", "tests/test_webrag.py"],
     "limits": "Verdicts are judgements of a 4B model about retrieved text, not proof. The Arm C default has not been evaluated on the Studio's web evidence; its study result is a historical, unpublished SciFact comparison on previously used claims, not significantly better than a simpler baseline, in which it also raised false support."},
    {"id": "attestation", "name": "Quote attestation", "layer": "Verification",
     "files": ["docqa/attestation.py", "evidence_core/span_attestation.py"],
     "purpose": "Records where each verifier quote occurs in its passage (located, ambiguous or unattested); on by default in the Studio, record only.",
     "inputs": "Verifier quotes and passages", "outputs": "Attestation records in the run",
     "status": ["experimental", "tested"],
     "tests": ["tests/test_quote_attestation.py", "tests/test_quote_attestation_integration.py",
               "tests/test_evidence_core_span_attestation.py"],
     "limits": "Opt-in and record-only: it never changes a stance or verdict (see the replay in Results)."},
    {"id": "frozen_b", "name": "Experiment B verifier", "layer": "Verification",
     "files": ["experiment_b_verification.py", "claim_verification.py", "verification_units.py", "llm.py"],
     "purpose": "The earlier claim/evidence verifier: System B in the benchmark, and the frozen_b verifier of the document assistant.",
     "inputs": "Claim and evidence items", "outputs": "Verification report", "status": ["implemented", "tested", "evaluated"],
     "tests": ["tests/test_experiment_b.py", "tests/test_experiment_b_verification.py"],
     "limits": "Frozen research module."},
    {"id": "model_clients", "name": "Model clients", "layer": "Models",
     "files": ["docqa/llm.py", "docqa/remote_llm.py"],
     "purpose": "Local Ollama client (only local addresses, no :cloud models) and optional clients for a remote Ollama or an OpenAI-compatible API.",
     "inputs": "Prompts and JSON schemas", "outputs": "Model responses", "status": ["implemented", "tested"],
     "tests": ["tests/test_remote_llm.py"],
     "limits": "No model is substituted when the configured one fails; the run stops with the reason."},
    {"id": "report", "name": "Report", "layer": "Research pipeline",
     "files": ["webrag/report.py"],
     "purpose": "Renders a run as Markdown: verified findings, statements not verified, disagreements, gaps, limitations and sources.",
     "inputs": "Run record", "outputs": "Markdown report", "status": ["implemented", "tested"],
     "tests": ["tests/test_webrag.py"], "limits": "Only statements the verifier supported appear as findings."},
    {"id": "doc_assistant", "name": "Document assistant", "layer": "Research pipeline",
     "files": ["docqa/pipeline.py", "docqa/answer.py", "docqa/cli.py"],
     "purpose": "Questions and claim checks over the user's own TXT and PDF documents in the Studio library.",
     "inputs": "Local documents and a question", "outputs": "Answer with checked claims", "status": ["implemented"],
     "tests": [], "limits": "Its public tests were withheld from the release because their fixtures contain held-out claims."},
    {"id": "source_status", "name": "Retraction-status contract", "layer": "Verification",
     "files": ["webrag/source_status.py"],
     "purpose": "Data contract for a source's retraction status (retracted, not retracted, conflicting, lookup failed, unknown).",
     "inputs": "DOI and lookup results", "outputs": "Status record", "status": ["proposed", "tested"],
     "tests": ["tests/test_source_status.py"], "limits": "Offline contract only: no live retraction lookup exists."},
    {"id": "research_backend", "name": "Bounded multi-agent research backend", "layer": "Research backend",
     "files": ["research_autonomous_controller.py", "research_planning.py", "research_loop.py",
               "research_bounded_loop.py", "research_capabilities.py", "research_pipeline.py"],
     "purpose": "The earlier research-agent backend (planning, capability dispatch, bounded loop, researcher-gated continuation): System A's pipeline and System C in the benchmark.",
     "inputs": "Research tasks", "outputs": "Executed, audited research steps", "status": ["implemented", "tested", "evaluated"],
     "tests": ["tests/test_research_autonomous_controller.py", "tests/test_research_loop.py"],
     "limits": "Separate from the Studio pipeline; the Studio does not use it."},
    {"id": "evidence_core", "name": "Evidence contracts", "layer": "Research backend",
     "files": ["evidence_core/contracts.py", "evidence_core/graph.py", "evidence_core/policies.py",
               "evidence_core/replay.py", "evidence_core/aggregation_v2.py"],
     "purpose": "Typed evidence contracts, an evidence graph, aggregation policies and replay used by the research backend and its evaluation.",
     "inputs": "Verification outputs", "outputs": "Aggregated evidence records", "status": ["implemented", "tested"],
     "tests": ["tests/test_evidence_core_contracts.py", "tests/test_evidence_core_replay.py", "tests/test_slice1_aggregation.py"],
     "limits": "Research infrastructure; not used by the Studio except span attestation."},
    {"id": "eval_systems", "name": "Benchmark systems A, B, C", "layer": "Evaluation",
     "files": ["evaluation/experiments/system_a_rag.py", "evaluation/experiments/system_b_verified_rag.py",
               "evaluation/experiments/system_c_multi_agent.py", "evaluation/experiments/evaluator.py",
               "evaluation/experiments/run_phase10f.py", "evaluation/experiments/local_model.py", "src/baseline_rag.py"],
     "purpose": "Adapters that run System A (strong RAG), B (RAG plus claim verification) and C (bounded multi-agent research) on SciFact, with execution accounting.",
     "inputs": "SciFact claims", "outputs": "Attempts, predictions and scores", "status": ["implemented", "tested", "evaluated"],
     "tests": ["tests/test_phase10f_full_dev.py", "tests/test_phase10f_remediation.py"],
     "limits": "System A's baseline needs packages not in requirements.txt (langchain_huggingface, langchain_ollama)."},
    {"id": "scifact_harness", "name": "SciFact harness", "layer": "Evaluation",
     "files": ["evaluation/scifact/scifact_adapter.py", "evaluation/scifact/metrics.py",
               "evaluation/scifact/run_scifact_evaluation.py"],
     "purpose": "Loads SciFact claims and corpus and computes the classification and retrieval metrics.",
     "inputs": "data/scifact/", "outputs": "Metrics", "status": ["implemented", "tested", "evaluated"],
     "tests": ["tests/test_scifact_infrastructure.py"], "limits": "DEV split only in the public results."},
]

# Things outside the code: services, stores and data the components talk to.
EXTERNAL = [
    {"id": "browser", "name": "Browser (user)", "kind": "user"},
    {"id": "ollama", "name": "Local Ollama (qwen3:4b)", "kind": "service"},
    {"id": "remote_api", "name": "Optional model API", "kind": "service"},
    {"id": "providers", "name": "Search providers and source sites", "kind": "service"},
    {"id": "hf_hub", "name": "Hugging Face Hub (retrieval models, first run)", "kind": "service"},
    {"id": "run_store", "name": "outputs/webrag/runs/ (saved runs)", "kind": "store"},
    {"id": "scifact_data", "name": "SciFact dataset (data/scifact/)", "kind": "store"},
]

# Runtime connections that are not Python imports (HTTP, processes, files). Each is backed by a marker string that
# must occur in the named file; tests/test_showcase.py checks every one.
RUNTIME_EDGES = [
    ("browser", "dashboard", "HTTP 127.0.0.1:8501", "dashboard/launch.py", "--server.address"),
    ("browser", "studio_ui", "HTTP 127.0.0.1:8765", "webrag/server.py", "PAGE.read_bytes"),
    ("launcher", "server", "starts process", "dashboard/launch.py", "webrag.server"),
    ("launcher", "dashboard", "starts process", "dashboard/launch.py", '"app.py"'),
    ("dashboard", "server", "HTTP /api", "dashboard/client.py", "/api/run"),
    ("studio_ui", "server", "HTTP /api", "webrag/static/api.mjs", "/api/run"),
    ("model_clients", "ollama", "HTTP 127.0.0.1:11434", "docqa/llm.py", "127.0.0.1:11434"),
    ("model_clients", "remote_api", "HTTPS (optional)", "docqa/remote_llm.py", "/chat/completions"),
    ("search", "providers", "HTTPS", "webrag/search.py", "api.openalex.org"),
    ("network", "providers", "HTTPS fetch", "webrag/net.py", "class PoliteSession"),
    ("orchestrator", "run_store", "writes run.json", "webrag/orchestrator.py", "run.json"),
    ("server", "run_store", "reads run.json", "webrag/studio.py", "run.json"),
    ("rag_engine", "hf_hub", "model download", "rag.py", "all-MiniLM-L6-v2"),
    ("eval_systems", "ollama", "HTTP :11434", "evaluation/experiments/local_model.py", "11434"),
    ("scifact_harness", "scifact_data", "reads", "evaluation/scifact/scifact_adapter.py", "scifact"),
]

VIEWS = {
    "Complete system": None,
    "Research execution": ["browser", "studio_ui", "dashboard", "server", "orchestrator", "agents", "researcher",
                           "search", "network", "evidence", "docqa_index", "rag_engine", "verifier", "model_clients",
                           "report", "providers", "ollama", "remote_api", "run_store", "hf_hub"],
    "Evidence and verification": ["orchestrator", "agents", "evidence", "docqa_index", "rag_engine", "verifier",
                                  "attestation", "frozen_b", "model_clients", "source_status", "ollama"],
    "UI and backend": ["browser", "launcher", "dashboard", "studio_ui", "server", "model_service", "orchestrator",
                       "doc_assistant", "run_store"],
    "Evaluation and diagnostics": ["eval_systems", "scifact_harness", "research_backend", "evidence_core", "frozen_b",
                                   "rag_engine", "docqa_index", "verifier", "ollama", "scifact_data"],
}

# ------------------------------------------------------------------------------------------------ workflow
# The stages of a deep research run (webrag/orchestrator.py, webrag/deep.py), in the order a run records them.
WORKFLOW = [
    {"stage": "model check", "component": "orchestrator", "input": "Configured model",
     "does": "Asks the model service whether it answers and offers the configured model (and the verifier's).",
     "output": "Model and version recorded with the run",
     "stops": "Service unreachable or model missing: the run fails with the reason; no other model is used."},
    {"stage": "plan", "component": "agents", "input": "Question or claim",
     "does": "The planner writes subquestions, preferred source types, search queries and counter-evidence queries.",
     "output": "Research plan", "stops": "If planning fails, a fallback plan uses the question itself (recorded as such)."},
    {"stage": "search and read (rounds)", "component": "researcher", "input": "Planned queries",
     "does": "Searches the active providers, selects and de-duplicates sources, reads open-access full text where possible.",
     "output": "Sources with access level (full text, abstract, snippet)",
     "stops": "Provider failures are recorded per provider; budgets bound searches, sources and time."},
    {"stage": "coverage", "component": "orchestrator", "input": "Retrieved passages per subquestion",
     "does": "Checks whether each question has keyword-matched passages from independent publishers; decides whether another round is needed.",
     "output": "Coverage ledger", "stops": "Stops when coverage is met, budgets are used, or a round finds nothing new."},
    {"stage": "evidence pool", "component": "evidence", "input": "Sources",
     "does": "Splits text into traceable passages (source, page) and ranks them with BM25, embeddings and a reranker.",
     "output": "Ranked passages", "stops": "No readable source: the run ends with insufficient evidence."},
    {"stage": "synthesize", "component": "agents", "input": "Ranked passages",
     "does": "The synthesizer drafts statements, each citing the passages it relies on.",
     "output": "Cited statements (model-written)", "stops": "Malformed model output is recorded, not repaired silently."},
    {"stage": "verify", "component": "verifier", "input": "Each statement and its passages",
     "does": "Judges every passage: SUPPORTS, CONTRADICTS or NEUTRAL, with the quoted sentence. Then decides SUPPORTED, CONTRADICTED or INSUFFICIENT.",
     "output": "Verdicts with per-passage stances and cautions",
     "stops": "Conflicting passages give INSUFFICIENT (CONFLICTING) adjudicated by the default Arm C verifier, which keeps both sides listed and the statement out of the answer; with quote_stance they stay INSUFFICIENT (CONFLICTING)."},
    {"stage": "summary", "component": "agents", "input": "Verified findings only",
     "does": "Writes the direct answer and summary from supported findings; sentences whose numbers are not in the cited findings are dropped.",
     "output": "Direct answer and summary", "stops": "Without supported findings there is no summary."},
    {"stage": "save", "component": "orchestrator", "input": "The run",
     "does": "Saves the complete run record, including why searching stopped.",
     "output": "outputs/webrag/runs/<time>-<question>/run.json", "stops": ""},
]

# Reasons a deep run records for stopping (webrag/deep.py); checked against the code by the tests.
STOP_REASONS = {
    "retrieval_coverage_met": "every question had keyword-matched passages from independent publishers",
    "round_limit": "the depth allows no more search rounds",
    "search_budget": "all searches of the depth budget were used",
    "source_budget": "the source limit was reached",
    "time_budget": "the time budget was used up",
    "little_new_evidence": "the last round found no new relevant passage",
    "no_new_queries": "no new query could be formed for the open questions",
    "providers_failing": "every search in the round failed",
}

# ------------------------------------------------------------------------------------------------ installation
# Names and commands taken from the implementation; tests/test_showcase.py checks them against the code.
ENV_SETTINGS = [
    ("WEBRAG_LLM_PROVIDER", "local (default) | ollama | openai", "optional"),
    ("WEBRAG_LLM_MODEL", "model name, default qwen3:4b", "optional"),
    ("WEBRAG_LLM_BASE_URL", "https address of an Ollama server or an OpenAI-compatible API (with /v1)", "only for ollama/openai"),
    ("WEBRAG_LLM_API_KEY", "key for that service; stays in the Studio process", "only if the service needs one"),
    ("WEBRAG_LLM_JSON_MODE", "schema | object | none (openai only)", "optional"),
    ("WEBRAG_LLM_VERIFIER_MODEL", "separate verifier model on the same service", "optional"),
    ("WEBRAG_LLM_TIMEOUT", "seconds per model call, default 180", "optional"),
    ("OPENALEX_API_KEY", "free OpenAlex key; avoids the shared anonymous daily budget", "optional"),
    ("WEBRAG_CONTACT", "contact address sent with requests", "optional"),
    ("SEARXNG_URL", "your own SearXNG instance (general web search)", "optional"),
    ("BRAVE_API_KEY", "Brave Search API (paid)", "optional"),
    ("WEBRAG_MODEL_DOWNLOADS", "auto | 0 | 1: whether the retrieval models may be downloaded", "optional"),
]

LAUNCHER_OPTIONS = ["--studio-port", "--dashboard-port", "--no-browser", "--check"]
