"""Arm C agentic verification (the agentic verification study's arm C) as Studio verifiers. The Research Studio's
default is ``arm_c_studio``, a Studio-adapted variant of Arm C; ``arm_c`` is the historical method unchanged.

The workflow is the study's own code, not a re-implementation: ``evaluation/experiments/agentic_verify/arms.py``,
``prompts.py`` and ``llm.py`` are byte-identical to the frozen study (their SHA-256 values are in the copied
``FROZEN_CONFIG.json`` and checked by the tests), and the settings are read from that file: verifier prompt v1 (one call
judges up to 8 passages, each stance needs an exact quote), planner v0, adjudicator v2, trigger ``unresolved``,
``search_when`` ``empty``, at most 2 searches of 4 new passages in 3 planner turns, 6 passages shown to the
adjudicator, fallback to the rank rule when an adjudication is unusable. Per claim (``arms.run_agentic``):

1. the verifier judges the passages the Studio retrieved (a claim check: the 8 best-ranked; a research finding: its
   cited passages first, then ranked extras);
2. unanimous evidence keeps its verdict and nothing else runs;
3. conflicting evidence (passages for and against) goes to the adjudicator;
4. when no passage took a side, the planner may search; new passages are judged by the same verifier, and evidence
   that is still unresolved goes to the adjudicator.

Adaptations to the Studio (both modes). These do not change the decision logic:
* **Corpus.** The study's planner searched the SciFact corpus with the frozen Phase 10 retriever. Here it searches the
  evidence this run fetched (``EvidenceStore.rank``, the same ``rag.py`` hybrid engine; with the injection guard it is
  the guarded store). No new web requests are made.
* **Model client.** Model calls go through the Studio's client (local Ollama or a configured service): JSON mode,
  temperature 0, each role's output cap. The study also fixed seed 0 and ``num_ctx`` 4096 on its own client.
* **Recording.** The result is translated to the Studio's verdict with every stance kept. Passages found by search are
  returned so they are recorded. A trace records each step, the adjudication and the usage.
* **Sub-states.** An adjudicated conflict is sub-state ADJUDICATED and a rank fall-back RANK_TIEBREAK, so the Studio's
  safeguards apply: both sides stay listed, and a supported statement is cautioned and kept out of the direct answer.
* **Failures fail closed.** A model-service failure is raised as the Studio's ``LLMError`` (SERVICE_ERROR). A verifier
  reply unusable after its retry, or a planner search that fails or has no evidence store to search, gives
  ASSESSMENT_FAILED. Neither is replaced by another verifier.
* **Quote attestation** (record only, ``Settings.quote_attestation``). Each verifier quote is located in its passage
  with ``evidence_core.span_attestation`` and recorded. Stances and verdicts are unchanged.

The changes to the decision logic, in ``arm_c_studio`` only:
* An adjudicator decision made when no passage was judged for or against the claim, even after search, is not
  accepted. The verdict stays INSUFFICIENT and the proposal is recorded.
* (Added 2026-10-24 after the adversarial campaign, ADV-1 and ADV-2; no evaluation set was involved.) If the
  verifier gives a passage an invalid stance, the study reads it as NEUTRAL; here the claim fails closed
  (ASSESSMENT_FAILED). A passage the reply leaves out is still read as NEUTRAL, as in the study, because the model
  routinely omits passages it finds irrelevant (observed with qwen3:4b). Such omissions are recorded in the note and
  the trace. An omitted passage that would have contradicted the claim cannot be told apart from an irrelevant one;
  that risk remains.
* (Same date.) An adjudication whose cited passage the verifier judged for the other side (or neutral) is not
  accepted. The study's rank-rule fall-back decides instead.
* Reason: the Studio's rule that an adjudicator alone never makes a claim SUPPORTED or CONTRADICTED.
* In the study's DEV records those decisions were right 2 times in 4 and added 8 commitments on the 112 claims without
  gold evidence.
* Declining them changes Arm C's DEV macro-F1 from 0.7551 to 0.7528, and the commitment rate on claims without gold
  evidence from 0.705 to 0.634. This is an exploratory re-scoring of previously used data, not a new evaluation.
* ``arm_c`` accepts them (sub-state ADJUDICATED_UNSIDED), as the study did.

Limits. Arm C's macro-F1 of 0.7551 (SciFact DEV, previously used claims; not significantly above the rank baseline
B1R, p = 0.081) was measured on abstracts with the study's retriever. The Studio judges web passages, and its accuracy
with Arm C has not been evaluated. In the study, Arm C raised false support from 10/64 to 13/64: its adjudicator
decided SUPPORTED on three conflicting claims that were contradicted.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from evaluation.experiments.agentic_verify import arms
from evaluation.experiments.agentic_verify.llm import LLMError as StudyLLMError
from evaluation.experiments.agentic_verify.llm import Usage, parse_json

from .adjudicate import ADJUDICATED
from .index import Evidence
from .llm import LLMError
from .ranked import ADJUDICATED_UNSIDED, RANK_TIEBREAK
from .verify import CONTRADICTED, INSUFFICIENT, SUPPORTED, VERIFIERS, Verdict

FROZEN_CONFIG = Path(arms.__file__).with_name("FROZEN_CONFIG.json")
LABELS = {arms.SUPPORTED: SUPPORTED, arms.CONTRADICTED: CONTRADICTED, arms.INSUFFICIENT: INSUFFICIENT}
Search = Callable[[str, int], Sequence[Any]]          # (query, k) -> Studio passages (``webrag.evidence.Passage``)


def frozen_settings() -> arms.Settings:
    """The study's frozen settings, read from the copied FROZEN_CONFIG.json (never re-typed)."""
    return arms.Settings(**json.loads(FROZEN_CONFIG.read_text(encoding="utf-8"))["settings"])


def _text_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


class StudioModel:
    """The study's ``ChatClient.complete`` over the Studio's model client. A service failure is kept so it can be
    re-raised to the Studio; the verifier's quotes are kept (last usable reply per passage) for quote attestation."""

    def __init__(self, client: Any):
        self.client = client
        self.service_error: Optional[LLMError] = None
        self.quotes: Dict[str, str] = {}              # passage id (P1, P2 ...) -> the verifier's quote
        self.assessed: Dict[str, bool] = {}           # passage id -> the latest reply gave it a valid stance

    def complete(self, role: str, system: str, user: str, max_tokens: int, usage: Usage) -> str:
        try:
            response = self.client.chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                        response_format="json", temperature=0.0, num_predict=max_tokens)
        except LLMError as exc:
            self.service_error = exc
            raise StudyLLMError("SERVICE_ERROR") from None
        usage.add(role, int(response.prompt_eval_count or 0), int(response.eval_count or 0), response.seconds, live=True)
        if role == "verifier":
            parsed = parse_json(response.text)
            entries = parsed.get("assessments") if isinstance(parsed, dict) else None
            for entry in entries if isinstance(entries, list) else []:
                if isinstance(entry, dict) and isinstance(entry.get("id"), str):
                    pid = entry["id"].strip().strip("[]")
                    self.quotes[pid] = str(entry.get("quote") or "")
                    self.assessed[pid] = str(entry.get("stance", "")).upper().strip() in arms.STANCES
        return response.text


class StudioRetriever:
    """The study's ``Retriever.search`` over the Studio's evidence. The claim retrieval returns the passages the Studio
    already ranked (all of them, in order); a planner search ranks the run's evidence store with the planner's query.
    A failed search, or a search with no store to search, fails closed (the claim is not assessed)."""

    def __init__(self, evidence: Sequence[Evidence], search: Optional[Search]):
        self.evidence, self.search_fn, self.calls = list(evidence), search, 0
        self.found: Dict[str, Any] = {}               # text id -> Studio passage returned by a planner search
        self.queries: Dict[str, str] = {}             # text id -> the query that found it
        self.error = ""

    def search(self, query: str, limit: int, source: str) -> List[arms.Passage]:
        self.calls += 1
        if source == "claim":
            return [arms.Passage("", None, _text_id(e.text), e.text, "claim", rank)
                    for rank, e in enumerate(self.evidence, 1)]
        if self.search_fn is None:
            self.error = "the planner asked for a search, but no evidence store was available to search"
            raise StudyLLMError("RETRIEVAL_UNAVAILABLE")
        try:
            results = list(self.search_fn(query, limit))
        except Exception as exc:                      # an evidence-store failure is reported, never skipped silently
            self.error = f"planner search failed: {type(exc).__name__}: {exc}"
            raise StudyLLMError("RETRIEVAL_ERROR") from None
        out = []
        for rank, passage in enumerate(results, 1):
            tid = _text_id(passage.text)
            self.found.setdefault(tid, passage)
            self.queries.setdefault(tid, query)
            out.append(arms.Passage("", None, tid, passage.text, source, rank))
        return out


class ArmCVerifier:
    """Arm C exactly as studied (``arms.run_agentic`` with the frozen settings). An adjudication made without any sided
    passage is accepted (sub-state ADJUDICATED_UNSIDED), as in the study."""

    name = "arm_c"
    arm = "C"
    variant = "historical method"
    accept_unsided = True
    require_complete_assessment = False               # the study read an unknown stance as NEUTRAL
    check_cited_stance = False                        # the study did not check the adjudicator's cited stance
    uses_search = True                                # Orchestrator._verify passes the run's evidence search
    records_quote_attestation = True                  # quotes are captured and attested (record only) when asked
    attest = False

    def __init__(self, client: Any):
        self.client = client
        self.settings = frozen_settings()

    def run(self, claim: str, retriever: StudioRetriever, model: StudioModel) -> arms.Result:
        return arms.run_agentic(claim, retriever, model, self.settings)

    def verify(self, claim: str, evidence: Sequence[Evidence], search: Optional[Search] = None) -> Verdict:
        if not evidence:
            return Verdict(INSUFFICIENT, "NO_EVIDENCE", note="no evidence was retrieved")
        model, retriever = StudioModel(self.client), StudioRetriever(evidence, search)
        result = self.run(claim, retriever, model)
        if model.service_error is not None:           # truthful failure: the Studio records SERVICE_ERROR
            raise model.service_error
        verdict = Verdict(INSUFFICIENT, calls=int(result.usage.get("calls", 0)),
                          seconds=float(result.usage.get("model_seconds", 0.0)))
        searched: List[Dict[str, Any]] = []
        attested: Dict[str, List[Dict[str, Any]]] = {}
        for st in result.stances:
            passage = next((p for p in result.passages if p["pid"] == st["pid"]), None)
            if passage is None:
                continue
            tid = passage["chunk_id"]
            position = int(st["pid"][1:]) - 1 if st["pid"][1:].isdigit() else -1
            if passage["source"] == "claim" and 0 <= position < len(evidence):
                eid, text = evidence[position].id, evidence[position].text
            else:                                      # found by a planner search: a new evidence label
                eid = f"S{len(searched) + 1}"
                found = retriever.found.get(tid)
                text = found.text if found is not None else ""
                searched.append({"id": eid, "passage": found, "stance": st["stance"],
                                 "query": retriever.queries.get(tid, ""), "search": passage["source"]})
            verdict.stances[eid] = st["stance"]
            if self.attest:
                attested[eid] = [self._attestation(claim, st, model.quotes.get(st["pid"], ""), text)]
        verdict.__dict__["searched_passages"] = [s for s in searched if s["passage"] is not None]
        if self.attest:
            verdict.__dict__["quote_attestations"] = attested
        verdict.__dict__["agentic_trace"] = {
            "workflow": self.name, "study_arm": self.arm, "variant": self.variant, "status": result.status,
            "first_judgement": result.aggregate_verdict, "actions": result.actions,
            "adjudication": result.adjudication, "usage": result.usage, "retrieval_calls": result.retrieval_calls,
            "invalid_outputs": result.invalid_outputs, "ungrounded_quotes": result.ungrounded_quotes,
            "retrieval_error": retriever.error}
        notes = [f"Arm C agentic verification ({self.name}: {self.variant})"]
        if result.status != "ok":
            verdict.parse_ok, verdict.error = False, retriever.error or result.status
            verdict.label, verdict.sub_state = INSUFFICIENT, "ASSESSMENT_FAILED"
            reason = retriever.error or f"the verifier's output was unusable after its retry ({result.status})"
            verdict.note = "; ".join(notes + [reason, "not assessed (fails closed)"])
            return verdict
        invalid = [st["pid"] for st in result.stances if model.assessed.get(st["pid"]) is False]
        unlisted = [st["pid"] for st in result.stances if st["pid"] not in model.assessed]
        verdict.__dict__["agentic_trace"].update(invalid_stance_passages=invalid, unlisted_passages=unlisted)
        if invalid and self.require_complete_assessment:
            verdict.parse_ok, verdict.label, verdict.sub_state = False, INSUFFICIENT, "ASSESSMENT_FAILED"
            verdict.error = f"passage(s) {', '.join(invalid)} received no valid stance from the verifier"
            verdict.note = "; ".join(notes + [f"the verifier gave passage(s) {', '.join(invalid)} an invalid stance",
                                              "not assessed (fails closed)"])
            return verdict
        if unlisted:                                   # the model leaves out passages it finds irrelevant (observed)
            notes.append(f"passage(s) {', '.join(unlisted)} not listed in the verifier's reply; read as NEUTRAL, as "
                         f"in the study")
        sides = set(verdict.stances.values()) - {"NEUTRAL"}
        searches = [a for a in result.actions if a.get("action") == "SEARCH"]
        if searches:
            notes.append(f"{len(searches)} planner search(es) of this run's evidence, "
                         f"{sum(a.get('new_passages', 0) for a in searches)} new passage(s) judged")
        label = LABELS.get(result.prediction, INSUFFICIENT)
        adjudication = result.adjudication
        if adjudication is None:                       # unanimous evidence (first judgement or after search)
            verdict.label, verdict.sub_state = label, ("" if label != INSUFFICIENT else "NO_DECISIVE_EVIDENCE")
            notes.append("unanimous evidence; no adjudication")
        elif adjudication.get("valid") and adjudication.get("grounded") and len(sides) == 2 and \
                self._cited_stance_mismatch(adjudication, result):
            label = self._rank_rule(result)
            verdict.label = label
            verdict.sub_state = RANK_TIEBREAK if label != INSUFFICIENT and len(sides) == 2 else (
                "CONFLICTING" if len(sides) == 2 else "NO_DECISIVE_EVIDENCE")
            notes.append(f"the adjudicator's decision {adjudication.get('verdict')} does not match its cited passage "
                         f"{adjudication.get('passage')}; the study's rank-rule fall-back is used instead")
        elif adjudication.get("valid") and adjudication.get("grounded"):
            proposed = LABELS.get(adjudication.get("verdict"), INSUFFICIENT)
            if proposed == INSUFFICIENT:
                verdict.label = INSUFFICIENT
                verdict.sub_state = "CONFLICTING" if len(sides) == 2 else "NO_DECISIVE_EVIDENCE"
                notes.append("the adjudicator found the evidence insufficient")
            elif len(sides) == 2:
                verdict.label, verdict.sub_state = proposed, ADJUDICATED
                notes.append(f"conflicting evidence; the adjudicator judged {proposed} citing passage "
                             f"{adjudication.get('passage')} of those shown to it")
            elif self.accept_unsided:
                verdict.label, verdict.sub_state = proposed, ADJUDICATED_UNSIDED
                notes.append(f"no passage took a side; the adjudicator judged {proposed} (accepted, as in the study)")
            else:
                verdict.label, verdict.sub_state = INSUFFICIENT, "NO_DECISIVE_EVIDENCE"
                notes.append(f"no passage took a side; the adjudicator proposed {proposed}, which is not accepted "
                             f"without a passage judged for or against the claim")
        else:                                          # unusable adjudication: the study's rank-rule fall-back
            verdict.label = label
            if label != INSUFFICIENT and len(sides) == 2:
                verdict.sub_state = RANK_TIEBREAK
                notes.append("the adjudication was unusable; conflicting evidence decided by rank (study fall-back)")
            else:
                verdict.label, verdict.sub_state = INSUFFICIENT, ("CONFLICTING" if len(sides) == 2
                                                                  else "NO_DECISIVE_EVIDENCE")
                notes.append("the adjudication was unusable; no decision")
        if result.ungrounded_quotes:
            notes.append(f"{result.ungrounded_quotes} stance(s) without a quote from their passage set to NEUTRAL")
        verdict.note = "; ".join(notes)
        return verdict

    def _cited_stance_mismatch(self, adjudication: Dict[str, Any], result: arms.Result) -> bool:
        """True when the adjudicator decides one side while citing a passage the verifier judged otherwise (only
        checked where ``check_cited_stance`` is set; the historical modes keep the study's behaviour)."""
        decided = LABELS.get(adjudication.get("verdict"), INSUFFICIENT)
        if not self.check_cited_stance or decided == INSUFFICIENT:
            return False
        cited = next((st["stance"] for st in result.stances if st["pid"] == adjudication.get("passage")), None)
        return cited != {SUPPORTED: "SUPPORTS", CONTRADICTED: "CONTRADICTS"}[decided]

    @staticmethod
    def _rank_rule(result: arms.Result) -> str:
        """The study's B1R rule on the recorded stances (first-retrieval passages before search results)."""
        passages = [arms.Passage(p["pid"], None, p["chunk_id"], "", p["source"], p["rank"]) for p in result.passages]
        stances = [arms.Stance(st["pid"], st["stance"], st["grounded"], st["raw_stance"]) for st in result.stances]
        return LABELS.get(arms.rank_rule(passages, stances), INSUFFICIENT)

    @staticmethod
    def _attestation(claim: str, st: Dict[str, Any], quote: str, text: str) -> Dict[str, Any]:
        """Where the verifier's quote occurs in its passage (record only; ``docqa.attestation`` record format)."""
        from .attestation import ATTESTATION_RECORD_VERSION, attestation_record
        entry: Dict[str, Any] = {"record_version": ATTESTATION_RECORD_VERSION, "checked_text": claim, "assessed": True,
                                 "stance": st["stance"], "raw_stance": st.get("raw_stance"), "quote": quote}
        try:
            entry["legacy_quote_check"] = arms.grounded(quote, text)      # the study's own check, as applied
            entry["attestation"] = attestation_record(quote, text)
        except Exception as exc:                       # recorded, never raised: attestation must not stop a verdict
            entry["attestation_error"] = f"{type(exc).__name__}: {exc}"
        return entry


class StudioArmCVerifier(ArmCVerifier):
    """The Studio's default: Arm C with one safeguard (see the module docstring). An adjudication made without any
    sided passage is declined; everything else is ``arm_c``."""

    name = "arm_c_studio"
    variant = "Studio-adapted variant: no decision without a passage for or against"
    accept_unsided = False
    require_complete_assessment = True                # a passage given an invalid stance fails the claim closed
    check_cited_stance = True                         # an adjudication citing the other side is not accepted


class ArmC0Verifier(ArmCVerifier):
    """Study arm C0 as studied: the same first judgement and adjudicator, without planner or search (an adjudication
    without any sided passage is accepted, as in the study)."""

    name = "arm_c0"
    arm = "C0"
    variant = "historical arm C0: adjudication without planner or search"
    uses_search = False

    def run(self, claim: str, retriever: StudioRetriever, model: StudioModel) -> arms.Result:
        return arms.run_single_pass(claim, retriever, model, self.settings, adjudicated=True)


ARM_C_VERIFIERS = {cls.name: cls for cls in (StudioArmCVerifier, ArmCVerifier, ArmC0Verifier)}
VERIFIERS.update(ARM_C_VERIFIERS)                # the registry used by the studio, the CLI and the evaluation harness
