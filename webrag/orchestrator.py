"""Coordinates the agents, logs every stage, and assembles the report from verified claims only.

Investigation status (how completely the pipeline ran) is kept separate from the outcome (what the evidence shows):
* status:  COMPLETE | PARTIAL (some searches, sources or checks failed; disclosed) | FAILED (no usable result)
* outcome: research -> ANSWERED (at least one finding verified) | INSUFFICIENT_EVIDENCE
           verify   -> SUPPORTED | CONTRADICTED | INSUFFICIENT
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from docqa.llm import LLMError, OllamaChat
from docqa.adjudicate import ADJUDICATED
from docqa.ranked import DECIDED_CONFLICTS, RANK_TIEBREAK
from docqa.verify import CONTRADICTED, INSUFFICIENT, SUPPORTED, VERIFIERS

from .agents import PERSPECTIVE_VERSION, PROMPT_VERSION, CitedStatement, Planner, Synthesizer
from .evidence import EvidenceStore, Passage, cap_per_origin, merge_passages
from .net import PoliteSession
from .researcher import ABSTRACT_ONLY, FAILED, FULL_TEXT, SNIPPET_ONLY, Corpus, Source, WebResearcher
from .search import default_providers

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "outputs" / "webrag" / "runs"


@dataclass
class Settings:
    model: str = "qwen3:4b"
    verifier: str = "quote_stance"     # adopted over frozen_b by the eval comparison (evaluation/webrag/DECISION_RULE.md)
    providers: Optional[List[str]] = None
    per_query: int = 3
    max_sources: int = 10
    max_full_text: int = 4
    question_passages: int = 8
    claim_passages: int = 5
    synthesizer_version: int = 2       # adopted over v1 offline (evaluation/webrag/DECISION_RULE_V3.md)
    followup_rounds: int = 0           # opt-in verifier-driven follow-up research (0 or 1; see webrag/followup.py)
    planner: str = "default"           # "perspective" = opt-in planner experiment (webrag/agents.py)
    save: bool = True
    # None = the original single-pass pipeline (unchanged). "quick" | "standard" | "deep" = bounded deep research
    # (webrag/deep.py) with that budget; a run can also choose its depth through ``run(options=...)``.
    depth: Optional[str] = None
    # Prompt-injection evidence gate (webrag/injection.py): "off" (default, unchanged), "flag" (record passages with
    # instruction-like text) or "exclude" (also withhold them from every model call). See docs/reverse_engineering/.
    injection_guard: str = "off"
    # Record-only quote attestation (docqa/attestation.py): where each verifier quote occurs in its passage, recorded
    # per passage; never changes a stance or verdict. Needs a quote-based verifier. Off by default.
    quote_attestation: bool = False

    def __post_init__(self) -> None:
        if self.verifier not in VERIFIERS:
            raise ValueError(f"unknown verifier {self.verifier!r}; choose one of {', '.join(sorted(VERIFIERS))}")
        if self.injection_guard not in ("off", "flag", "exclude"):
            raise ValueError("injection_guard must be 'off', 'flag' or 'exclude'")
        if not isinstance(self.quote_attestation, bool):
            raise ValueError("quote_attestation must be true or false")
        if self.quote_attestation:
            from docqa.attestation import supports_attestation
            if not supports_attestation(VERIFIERS[self.verifier]):
                raise ValueError(f"quote attestation needs a quote-based verifier; {self.verifier!r} does not quote")
        if self.followup_rounds not in (0, 1):
            raise ValueError("followup_rounds must be 0 (off) or 1")
        if self.planner not in ("default", "perspective"):
            raise ValueError("planner must be 'default' or 'perspective'")
        if self.depth not in (None, "quick", "standard", "deep"):
            raise ValueError("depth must be None, 'quick', 'standard' or 'deep'")


@dataclass
class ClaimResult:
    text: str
    role: str                                   # "finding" or "input claim"
    label: str = INSUFFICIENT
    sub_state: str = ""
    passages: List[Dict[str, Any]] = field(default_factory=list)   # passage, its source and the verifier's stance
    cited_sources: List[str] = field(default_factory=list)          # what the synthesizer cited
    supporting_sources: List[str] = field(default_factory=list)
    contradicting_sources: List[str] = field(default_factory=list)
    independent_support: int = 0                # distinct origins among supporting sources
    independent_contradiction: int = 0
    parse_ok: bool = True
    error: str = ""
    seconds: float = 0.0
    verifier_note: str = ""                     # the verifier's own note (checks applied, adjudication outcome)
    verifier: str = ""                          # the verifier that produced this verdict
    verifier_trace: Dict[str, Any] = field(default_factory=dict)   # agentic verifiers: steps, adjudication, usage
    abstain_reason: str = ""                    # INSUFFICIENT: why nothing was decided, in plain words
    confidence: Dict[str, Any] = field(default_factory=dict)       # level and agreement across the judged passages


# Why a statement stays INSUFFICIENT, for verifiers that do not say so themselves (verifier_v2 does).
ABSTAIN_REASONS = {
    "NO_EVIDENCE": "No evidence was retrieved for this statement.",
    "NO_DECISIVE_EVIDENCE": "No passage showed this statement true or false.",
    "CONFLICTING": "The passages disagree, and nothing decided between them.",
    "ASSESSMENT_FAILED": "The verifier's output was unusable, so nothing was decided.",
    "SERVICE_ERROR": "The model service failed, so nothing was decided.",
}


def confidence_from_stances(label: str, stances: Dict[str, str]) -> Dict[str, Any]:
    """Agreement across the judged passages, for verifiers that report no confidence of their own: "high" when two or
    more passages back the verdict and none opposes it, "medium" with one, "low" when some passage opposes it, and
    "abstained" for INSUFFICIENT. A count of passages, not a probability."""
    want = {SUPPORTED: "SUPPORTS", CONTRADICTED: "CONTRADICTS"}.get(label)
    agreeing = sum(1 for s in stances.values() if want and s == want)
    opposing = sum(1 for s in stances.values() if want and s in ("SUPPORTS", "CONTRADICTS") and s != want)
    level = ("abstained" if not want else "high" if agreeing >= 2 and not opposing else "medium" if not opposing
             else "low")
    judged = sum(1 for s in stances.values() if s != "NOT_ASSESSED")
    return {"level": level, "agreeing": agreeing, "opposing": opposing, "judged": judged}


class RunCancelled(Exception):
    """Raised by a progress callback to stop a run at its next checkpoint (a stage or progress note). Nothing is saved:
    ``Orchestrator.run`` lets it through, and the caller decides how to report the cancelled run."""


class Run:
    def __init__(self, text: str, mode: str, progress: Callable[[Dict[str, Any]], None]):
        self.data: Dict[str, Any] = {
            "input": text, "mode": mode, "status": "RUNNING", "outcome": None,
            "started": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "stages": [], "plan": None,
            "sources": [], "search_errors": [], "claims": [], "disagreements": [], "gaps": [], "limitations": [],
            "evidence_pool": [], "error": "",
        }
        self.progress = progress

    def stage(self, name: str, status: str, seconds: float, detail: str = "") -> None:
        self.data["stages"].append({"name": name, "status": status, "seconds": round(seconds, 2), "detail": detail})
        self.progress(self.data)

    def note(self, name: str, detail: str) -> None:
        self.data["current"] = f"{name}: {detail}"
        self.progress(self.data)


class Orchestrator:
    def __init__(self, settings: Optional[Settings] = None, client: Optional[OllamaChat] = None,
                 session: Optional[PoliteSession] = None, providers: Optional[list] = None, models=None,
                 verifier: Any = None, store_factory: Callable[..., Any] = EvidenceStore):
        self.settings = settings or Settings()
        self.store_factory = store_factory
        self._guarded_stores: List[Any] = []
        if self.settings.injection_guard != "off" and store_factory is EvidenceStore:
            def guarded_store(*args, **kwargs):
                from .guarded import GuardedEvidenceStore
                store = GuardedEvidenceStore(*args, guard=self.settings.injection_guard, **kwargs)
                self._guarded_stores.append(store)
                return store
            self.store_factory = guarded_store
        self.client = client or OllamaChat(model=self.settings.model)
        self.session = session or PoliteSession()
        self.providers = providers if providers is not None else default_providers(self.session,
                                                                                     self.settings.providers)
        self.models = models
        self.verifier = verifier or VERIFIERS[self.settings.verifier](self.client)
        if self.settings.quote_attestation:
            if getattr(self.verifier, "records_quote_attestation", False):
                self.verifier.attest = True                          # Arm C: records its own quotes' attestation
            else:
                from docqa.attestation import with_attestation
                self.verifier = with_attestation(self.verifier)      # refuses verifiers that do not quote

    # ---------------------------------------------------------------------------------------------- entry points
    def run(self, text: str, mode: str = "research",
            progress: Callable[[Dict[str, Any]], None] = lambda data: None,
            options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """``options`` (all optional) choose deep research for this run: ``depth`` (quick | standard | deep),
        ``providers`` (ids to use, a subset of the active ones), ``categories`` (preferred source types),
        ``recency`` (any | year | 5y) and ``config`` (the studio's search configuration)."""
        if mode not in {"research", "verify"}:
            raise ValueError("mode must be 'research' or 'verify'")
        options = dict(options or {})
        if self.settings.depth and not options.get("depth"):
            options["depth"] = self.settings.depth
        run = Run(text.strip(), mode, progress)
        providers = self.providers
        self._guarded_stores = []
        try:
            if options.get("depth"):
                providers = self._providers_for(options)
            self._run(run, options, providers)
            self._record_injection_screen(run.data)
        except RunCancelled:                                     # stopped on request: no result, nothing saved
            raise
        except LLMError as exc:
            run.data["status"], run.data["error"] = "FAILED", f"Local model error: {exc}"
        except Exception as exc:                                 # report, never fabricate
            run.data["status"], run.data["error"] = "FAILED", f"Unexpected error: {type(exc).__name__}: {exc}"
        run.data["finished"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        planner = "" if self.settings.planner == "default" else f"+{PERSPECTIVE_VERSION}"
        if options.get("depth"):
            from .agents import SOURCE_PLANNER_VERSION
            from .deep import DEEP_VERSION
            planner = f"+{SOURCE_PLANNER_VERSION}+{DEEP_VERSION}"
        run.data["settings"] = asdict(self.settings) | {"prompt_version": f"{PROMPT_VERSION}+synthesizer-v"
                                                                           f"{3 if options.get('depth') else self.settings.synthesizer_version}"
                                                                           f"{planner}",
                                                        "providers": [p.name for p in providers]}
        if options.get("depth"):
            run.data["settings"]["depth"] = options["depth"]
        run.data["model"] = getattr(self, "_model_info", {"model": self.settings.model})
        if self.settings.save:
            run.data["saved_to"] = self._save(run.data)
        try:
            progress(run.data)
        except RunCancelled:                                     # asked too late: the run is complete and saved
            pass
        return run.data

    def _providers_for(self, options: Dict[str, Any]) -> list:
        """The providers of a deep-research run: from the studio configuration when one is given (else the providers
        this orchestrator was built with), narrowed to ``options['providers']`` when the run names some."""
        if options.get("config") is not None:
            providers = default_providers(self.session, None, options["config"])
        else:
            providers = list(self.providers)
        wanted = options.get("providers")
        if wanted:
            unknown = [n for n in wanted if n not in {p.name for p in providers}]
            if unknown:
                raise ValueError(f"Provider(s) {unknown} are not active; enable and configure them first.")
            providers = [p for p in providers if p.name in wanted]
        return providers

    def _service_check(self, run: Run, info: Dict[str, Any], started: float) -> bool:
        """Model check for a configured model service: the service must answer and offer the configured model (and
        the verifier's model, when the verifier has its own). Nothing is substituted; a failed check stops the run."""
        checks = [("model", info)]
        verifier_client = getattr(self.verifier, "client", None)
        if verifier_client is not None and verifier_client is not self.client and hasattr(verifier_client, "describe"):
            verifier_info = verifier_client.describe()
            info["verifier_model"] = verifier_info                # recorded with the run
            checks.append(("verifier model", verifier_info))
        for role, row in checks:
            if not row.get("service_reachable"):
                problem = (f"The model service at {row.get('endpoint')} could not be used"
                           + (f": {row['error']}" if row.get("error") else "") + ".")
            elif not row.get("model_available"):
                problem = (f"The {role} '{row.get('model')}' is not offered by {row.get('endpoint')}"
                           + (f" ({row['error']})" if row.get("error") else "")
                           + "; no other model is substituted automatically.")
            else:
                continue
            run.stage("model check", "failed", time.perf_counter() - started)
            run.data["status"], run.data["error"] = "FAILED", problem
            return False
        detail = f"{info['model']} on {info.get('provider_label') or info['provider']} ({info.get('endpoint')})"
        if len(checks) > 1:
            detail += f"; verifier {checks[1][1].get('model')}"
        run.stage("model check", "ok", time.perf_counter() - started, detail)
        return True

    def _run(self, run: Run, options: Optional[Dict[str, Any]] = None, providers: Optional[list] = None) -> None:
        data, text = run.data, run.data["input"]
        if not text:
            data["status"], data["error"] = "FAILED", "Empty input."
            return
        # 0. local model available?
        started = time.perf_counter()
        info = self.client.describe()
        self._model_info = info
        if info.get("provider"):                                  # a configured model service (docqa.remote_llm)
            if not self._service_check(run, info, started):
                return
        elif not info.get("ollama_version"):
            run.stage("model check", "failed", time.perf_counter() - started)
            data["status"] = "FAILED"
            data["error"] = (f"Ollama is not reachable at {self.client.base_url}. Start Ollama (for example "
                             f"'ollama serve') and try again.")
            return
        elif not info.get("model_digest"):
            run.stage("model check", "failed", time.perf_counter() - started)
            data["status"] = "FAILED"
            data["error"] = (f"The model '{self.client.model}' is not installed in the local Ollama. Install it with "
                             f"'ollama pull {self.client.model}' or choose an installed model; no other model is "
                             f"substituted automatically.")
            return
        else:
            run.stage("model check", "ok", time.perf_counter() - started,
                      f"{info['model']} on Ollama {info['ollama_version']}")

        if options and options.get("depth"):                    # bounded deep research (webrag/deep.py)
            from .deep import run_deep
            run_deep(self, run, text, options, providers if providers is not None else self.providers)
            return

        # A. plan
        started = time.perf_counter()
        run.note("plan", "planning subquestions and search queries")
        plan = Planner(self.client, style=self.settings.planner).plan(text, data["mode"])
        data["plan"] = asdict(plan)
        run.stage("plan", "partial" if plan.fallback else "ok", time.perf_counter() - started,
                  f"{len(plan.queries)} queries, {len(plan.subquestions)} subquestions")
        if plan.note:
            data["limitations"].append(plan.note)

        # B. search, fetch, extract
        started = time.perf_counter()
        run.note("search", f"searching {', '.join(p.name for p in self.providers)}")
        researcher = WebResearcher(self.providers, self.session, per_query=self.settings.per_query,
                                   max_sources=self.settings.max_sources, max_full_text=self.settings.max_full_text)
        corpus = researcher.gather(plan.queries, lambda message: run.note("search", message))
        data["sources"] = [self._source_record(s) for s in corpus.sources]
        data["search_errors"] = [{"provider": p, "query": q, "error": e} for p, q, e in corpus.search_errors]
        data["requests"] = corpus.requests_made
        usable = [s for s in corpus.sources if s.usable]
        failed_searches = len(corpus.search_errors)
        status = "ok" if usable and not failed_searches else ("partial" if usable else "failed")
        run.stage("search and fetch", status, time.perf_counter() - started,
                  f"{len(corpus.hits)} hits, {len(corpus.sources)} sources selected, {len(usable)} with content, "
                  f"{corpus.requests_made} HTTP requests")
        self._source_limitations(corpus, data)
        if not usable:
            data["status"] = "FAILED" if failed_searches and not corpus.hits else "COMPLETE"
            data["outcome"] = "INSUFFICIENT_EVIDENCE" if data["mode"] == "research" else INSUFFICIENT
            data["error"] = ("No search provider could be reached." if data["status"] == "FAILED"
                             else "The searches returned no readable sources for this input.")
            return

        # C. index and rank
        started = time.perf_counter()
        run.note("evidence", "indexing passages")
        store = self.store_factory(usable, self.models)
        run.stage("index", "ok", time.perf_counter() - started, f"{store.size} passages indexed")

        if data["mode"] == "verify":
            self._verify_mode(run, store, text)
        else:
            self._research_mode(run, store, text, plan.subquestions)
        partial = any(s["status"] in {"partial", "failed"} for s in data["stages"])
        data["status"] = "FAILED" if data["error"] else ("PARTIAL" if partial else "COMPLETE")

    # ---------------------------------------------------------------------------------------------- modes
    def _research_mode(self, run: Run, store: EvidenceStore, question: str, subquestions: List[str]) -> None:
        data = run.data
        started = time.perf_counter()
        groups = [store.rank(question, k=self.settings.question_passages)]
        groups += [store.rank(q, k=3) for q in subquestions]
        pool = cap_per_origin(merge_passages(*groups, limit=100), store.sources,
                              k=self.settings.question_passages + 2, per_source=2)
        data["evidence_pool"] = [self._passage_record(p, store) for p in pool]
        run.stage("retrieve", "ok" if pool else "failed", time.perf_counter() - started,
                  f"{len(pool)} passages from {len({p.source_id for p in pool})} sources")
        if not pool:
            data["outcome"] = "INSUFFICIENT_EVIDENCE"
            return

        started = time.perf_counter()
        run.note("synthesize", "drafting cited findings")
        synthesis = Synthesizer(self.client, version=self.settings.synthesizer_version).synthesize(
            question, subquestions, pool, store.sources)
        if not synthesis.parse_ok:
            run.stage("synthesize", "failed", time.perf_counter() - started, synthesis.error)
            data["error"] = "The local model's synthesis could not be parsed; no findings are reported."
            data["outcome"] = "INSUFFICIENT_EVIDENCE"
            return
        run.stage("synthesize", "partial" if synthesis.truncated else "ok", time.perf_counter() - started,
                  f"{len(synthesis.findings)} findings, {len(synthesis.disagreements)} disagreements")
        unknown = sorted({i for s in synthesis.findings + synthesis.disagreements for i in s.unknown_ids})
        if unknown:
            data["limitations"].append(f"The model cited passage IDs that were not supplied ({', '.join(unknown)}); "
                                       f"those citations were discarded.")

        started = time.perf_counter()
        claims = []
        for number, finding in enumerate(synthesis.findings, 1):
            run.note("verify", f"checking finding {number} of {len(synthesis.findings)}")
            extra = store.rank(finding.text, k=self.settings.claim_passages)
            passages = merge_passages(finding.passages, extra, limit=self.settings.claim_passages)
            claims.append(self._verify(finding.text, "finding", passages, store,
                                       cited=[p.source_id for p in finding.passages]))
        data["claims"] = [asdict(c) for c in claims]
        failures = sum(not c.parse_ok for c in claims)
        run.stage("verify", "partial" if failures else "ok", time.perf_counter() - started,
                  f"{sum(c.label == SUPPORTED for c in claims)} supported, "
                  f"{sum(c.label == CONTRADICTED for c in claims)} contradicted, "
                  f"{sum(c.label == INSUFFICIENT for c in claims)} not established"
                  + (f", {failures} checks failed" if failures else ""))
        # A disagreement needs at least two cited passages (one per side); the model otherwise fills the slot with
        # statements such as "no disagreements were found" (seen in the first live run).
        data["disagreements"] = [{"text": d.text, "sources": list(dict.fromkeys(p.source_id for p in d.passages)),
                                  "verified": False} for d in synthesis.disagreements if len(d.passages) >= 2]
        data["gaps"] = synthesis.gaps
        data["outcome"] = "ANSWERED" if any(c.label == SUPPORTED for c in claims) else "INSUFFICIENT_EVIDENCE"
        if synthesis.insufficient:
            data["limitations"].append("The synthesizer reported that the evidence does not fully answer the question.")
        for c in claims:
            if c.label == INSUFFICIENT and c.sub_state == "CONFLICTING":
                data["disagreements"].append({"text": f"Sources disagree on: {c.text}",
                                              "sources": c.supporting_sources + c.contradicting_sources,
                                              "verified": True})
            elif c.sub_state in DECIDED_CONFLICTS:      # a conflict-deciding verifier: the disagreement stays visible
                data["disagreements"].append({"text": f"Sources disagreed on: {c.text} "
                                                      f"({DECIDED_CONFLICTS[c.sub_state]} as {c.label})",
                                              "sources": c.supporting_sources + c.contradicting_sources,
                                              "verified": True})
        if self.settings.followup_rounds:
            from .followup import run_followup
            started = time.perf_counter()
            run.note("follow-up", "searching for evidence the verifier found missing")
            updated = run_followup(data, client=self.client, session=self.session, providers=self.providers,
                                   verify_fn=self._verify, settings=self.settings, store=store,
                                   source_record=self._source_record)
            data.clear()
            data.update(updated)
            record = data["followup"]
            detail = record["skipped_reason"] or (
                f"{len(record['queries'])} queries, {len(record['new_sources'])} new sources, "
                f"{sum(c['changed'] for c in record['changes'])} of {len(record['changes'])} verdicts changed")
            run.stage("follow-up", "ok", time.perf_counter() - started, detail)

    def _verifier_rank_options(self) -> Dict[str, Any]:
        """Ranking options a verifier asks for. Only candidate verifiers set ``verify_per_source`` (round 3: no
        2-passage-per-source cap when verifying); the default verifiers set nothing, so the store's default applies."""
        per_source = getattr(self.verifier, "verify_per_source", None)
        return {"per_source": per_source} if per_source else {}

    def _verify_mode(self, run: Run, store: EvidenceStore, claim: str) -> None:
        data = run.data
        started = time.perf_counter()
        passages = store.rank(claim, k=self.settings.question_passages, **self._verifier_rank_options())
        data["evidence_pool"] = [self._passage_record(p, store) for p in passages]
        run.stage("retrieve", "ok" if passages else "failed", time.perf_counter() - started,
                  f"{len(passages)} passages from {len({p.source_id for p in passages})} sources")
        started = time.perf_counter()
        run.note("verify", "judging each passage against the claim")
        result = self._verify(claim, "input claim", passages, store)
        data["claims"] = [asdict(result)]
        run.stage("verify", "ok" if result.parse_ok else "partial", time.perf_counter() - started,
                  f"{result.label}" + (f" ({result.sub_state})" if result.sub_state else ""))
        data["outcome"] = result.label
        if result.sub_state == "CONFLICTING":
            data["disagreements"].append({"text": "Retrieved sources disagree about this claim.",
                                          "sources": result.supporting_sources + result.contradicting_sources,
                                          "verified": True})
        elif result.sub_state in DECIDED_CONFLICTS:
            how = ("an adjudication step judged one side decisive" if result.sub_state == ADJUDICATED
                   else "the rank tie-break took the first-ranked passage with a stance as decisive")
            data["disagreements"].append({"text": f"Retrieved sources disagreed about this claim; {how} "
                                                  f"({result.label}).",
                                          "sources": result.supporting_sources + result.contradicting_sources,
                                          "verified": True})

    # ---------------------------------------------------------------------------------------------- agent D
    def _verify(self, claim: str, role: str, passages: List[Passage], store: EvidenceStore,
                cited: Optional[List[str]] = None) -> ClaimResult:
        result = ClaimResult(claim, role, cited_sources=list(dict.fromkeys(cited or [])))
        started = time.perf_counter()
        evidence = [p.as_evidence(f"E{i}") for i, p in enumerate(passages, 1)]
        result.verifier = getattr(self.verifier, "name", "")
        try:
            if getattr(self.verifier, "uses_search", False):   # agentic verification: its planner searches this store
                options = self._verifier_rank_options()
                verdict = self.verifier.verify(claim, evidence,
                                               search=lambda query, k: store.rank(query, k=k, **options))
            else:
                verdict = self.verifier.verify(claim, evidence)
        except LLMError as exc:
            result.error, result.parse_ok, result.sub_state = str(exc), False, "SERVICE_ERROR"
            result.abstain_reason = ABSTAIN_REASONS["SERVICE_ERROR"]
            result.confidence = {"level": "abstained", "agreeing": 0, "opposing": 0, "judged": 0}
            result.seconds = round(time.perf_counter() - started, 2)
            return result
        result.label, result.sub_state, result.parse_ok = verdict.label, verdict.sub_state, verdict.parse_ok
        result.error = verdict.error or ""
        result.verifier_note = verdict.note or ""
        attested = verdict.__dict__.get("quote_attestations") or {}      # only with Settings.quote_attestation
        diagnostics = verdict.__dict__.get("passage_diagnostics") or {}   # agentic verifiers; record only
        reasons = verdict.__dict__.get("passage_reasons") or {}           # verifier_v2: why each passage counted or not
        for e, p in zip(evidence, passages):
            stance = verdict.stances.get(e.id, "NOT_ASSESSED")
            record = self._passage_record(p, store)
            record["stance"] = stance
            if e.id in reasons:
                record["verifier_reason"] = reasons[e.id]
            if e.id in attested:
                record["quote_attestation"] = attested[e.id]
            if e.id in diagnostics:
                record["verifier_diagnostics"] = dict(diagnostics[e.id])
            result.passages.append(record)
            if stance == "SUPPORTS":
                result.supporting_sources.append(p.source_id)
            elif stance == "CONTRADICTS":
                result.contradicting_sources.append(p.source_id)
        for found in verdict.__dict__.get("searched_passages") or []:   # judged passages the planner's searches found
            p = found["passage"]
            record = self._passage_record(p, store)
            record.update(stance=found["stance"], found_by_search=found["query"])
            if found["id"] in attested:
                record["quote_attestation"] = attested[found["id"]]
            if found["id"] in diagnostics:
                record["verifier_diagnostics"] = dict(diagnostics[found["id"]])
            result.passages.append(record)
            if found["stance"] == "SUPPORTS":
                result.supporting_sources.append(p.source_id)
            elif found["stance"] == "CONTRADICTS":
                result.contradicting_sources.append(p.source_id)
        result.verifier_trace = dict(verdict.__dict__.get("agentic_trace") or verdict.__dict__.get("verifier_trace") or {})
        if result.label == INSUFFICIENT:
            result.abstain_reason = (verdict.__dict__.get("abstain_reason")
                                     or ABSTAIN_REASONS.get(result.sub_state, "No decisive evidence was found."))
        result.confidence = dict(verdict.__dict__.get("confidence")
                                 or confidence_from_stances(result.label, {p["source_id"] + str(i): p["stance"]
                                                                           for i, p in enumerate(result.passages)}))
        result.supporting_sources = list(dict.fromkeys(result.supporting_sources))
        result.contradicting_sources = list(dict.fromkeys(result.contradicting_sources))
        origin = lambda sid: store.sources[sid].origin or sid
        result.independent_support = len({origin(s) for s in result.supporting_sources})
        result.independent_contradiction = len({origin(s) for s in result.contradicting_sources})
        result.seconds = round(time.perf_counter() - started, 2)
        return result

    # ---------------------------------------------------------------------------------------------- records
    @staticmethod
    def _source_record(s: Source) -> Dict[str, Any]:
        record = asdict(s)
        record["chars"] = len(s.text)
        return record

    def _record_injection_screen(self, data: Dict[str, Any]) -> None:
        """With the injection guard (the Studio default is exclude), list every passage the screen matched and what was
        done with it. An excluded passage is withheld whole, including any genuine evidence it contains; the
        limitation says so, so a missing counter-argument is not mistaken for an absent one."""
        mode = self.settings.injection_guard
        if mode == "off":
            return
        flagged = {}
        for store in self._guarded_stores:
            flagged.update(getattr(store, "flagged", {}))
        action = "excluded from every model call" if mode == "exclude" else "flagged only (still used)"
        data["injection_screen"] = {"mode": mode, "flagged": [{"key": key, **info, "action": action}
                                                              for key, info in sorted(flagged.items())]}
        if flagged:
            sources = sorted({str(info["source_id"]) for info in flagged.values()})
            data.setdefault("limitations", []).append(
                f"{len(flagged)} retrieved passage(s) from {', '.join(sources)} contained instruction-like text (possible "
                f"prompt injection) and were {action}. "
                + ("Each was withheld whole, including any genuine evidence in the same passage, so evidence for or "
                   "against a claim may be missing from these results. " if mode == "exclude" else "")
                + "The screen matches fixed patterns; it cannot recognise every attack.")

    @staticmethod
    def _passage_record(p: Passage, store: EvidenceStore) -> Dict[str, Any]:
        s = store.sources[p.source_id]
        record = {"key": p.key, "source_id": p.source_id, "page": p.page, "text": p.text, "score": round(p.score, 4),
                  "url": s.url, "title": s.title, "access": s.access}
        labels = getattr(s, "page_labels", None)
        if labels and 1 <= p.page <= len(labels):                # transcripts are cited by time, not page
            record["location"] = labels[p.page - 1]
        flagged = getattr(store, "flagged", {}).get(p.key)
        if flagged:                                              # "flag" mode: used, but marked
            record["injection_flags"] = list(flagged["rules"])
        return record

    @staticmethod
    def _source_limitations(corpus: Corpus, data: Dict[str, Any]) -> None:
        sources = corpus.sources
        by_access = {a: [s.id for s in sources if s.access == a] for a in (FULL_TEXT, ABSTRACT_ONLY, SNIPPET_ONLY, FAILED)}
        data["limitations"].append(
            f"Of {len(sources)} sources, {len(by_access[FULL_TEXT])} were read in full, "
            f"{len(by_access[ABSTRACT_ONLY])} only as abstracts, {len(by_access[SNIPPET_ONLY])} only as search "
            f"snippets, and {len(by_access[FAILED])} could not be read.")
        for name, status in corpus.provider_status.items():
            if status["failed"]:
                example = next((e for p, _, e in corpus.search_errors if p == name), "")
                data["limitations"].append(f"Search provider '{name}' failed for {status['failed']} of "
                                           f"{status['queries']} queries ({example})")
        duplicates = [s for s in sources if s.origin and s.origin != s.id]
        if duplicates:
            data["limitations"].append("Near-duplicate sources were counted once: " +
                                       ", ".join(f"{s.id} copies {s.origin}" for s in duplicates))
        names = list(corpus.provider_status)
        if not ({"brave", "searxng"} & set(names)):
            data["limitations"].append(f"No general web search provider is configured (set SEARXNG_URL or "
                                       f"BRAVE_API_KEY); sources come from {', '.join(names) or 'no provider'} only.")
        usable = [s for s in sources if s.access != FAILED]
        data["source_diversity"] = {"sources": len(usable),
                                    "domains": len({s.domain for s in usable if s.domain}),
                                    "publishers": len({s.publisher for s in usable if s.publisher}),
                                    "independent_origins": len({s.origin or s.id for s in usable}),
                                    "by_provider": {n: sum(s.provider == n for s in usable) for n in names}}
        data["provider_status"] = corpus.provider_status

    @staticmethod
    def _save(data: Dict[str, Any]) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", data["input"].lower())[:40].strip("-") or "run"
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        folder = RUNS_DIR / f"{stamp}-{slug}"
        folder.mkdir(parents=True, exist_ok=False)
        (folder / "run.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return str(folder.relative_to(ROOT))
