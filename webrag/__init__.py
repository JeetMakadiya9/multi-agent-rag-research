"""Internet research and fact verification.

Agents (each a separate component with its own inputs and outputs):
    planner.Planner              A  question -> subquestions and search queries (never findings)
    researcher.WebResearcher     B  queries -> search hits -> fetched, extracted, de-duplicated sources
    evidence.EvidenceStore       C  sources -> traceable passages, ranked per question or claim (frozen rag.py engine)
    verifier (docqa.verify)      D  claim + passages -> SUPPORTED / CONTRADICTED / INSUFFICIENT with per-passage stances
    synthesis.Synthesizer        E  passages -> cited findings; report.compose builds the report from verified findings
orchestrator.run_research / run_verification coordinate them and log every stage.

Search, fetching, ranking and the language model are independent and replaceable. Sources are only ever created from
real search results; the model never supplies a URL.
"""
