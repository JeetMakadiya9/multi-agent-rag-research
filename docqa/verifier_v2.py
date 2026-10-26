"""verifier_v2: the Research Studio's three-class claim verifier, owned and maintained here.

The Arm C study code (``evaluation/experiments/agentic_verify/``) stays frozen for the historical records; this module
takes over its role with a design built around what the evaluations found:

1. **One passage per call, structured output.** Each passage is judged in its own model call, with a JSON schema passed
   as the response format (Ollama ``format``; OpenAI-compatible ``json_schema``). There are no passage labels to copy or
   mismatch, so the passage-ID defect of the study code cannot occur; a reply that still fails the schema is retried
   once and then fails the claim closed. A missing or invalid verdict is never read as NEUTRAL.
2. **Relevance gate first.** The schema asks, before any stance, whether the passage reports a finding about the same
   population, intervention or exposure, and outcome. A passage that does not test the claim is NEUTRAL, whatever
   stance the model gives. On-topic but indecisive passages were the main source of false contradictions.
3. **Calibrated aggregation.** One passage no longer decides a claim on any stance. SUPPORTED needs a *strong*,
   grounded supporting passage that tests the claim, and no strong contradiction; CONTRADICTED the reverse, and, when the
   NLI filter is available, an NLI model that also sees the contradiction. Everything else is INSUFFICIENT, with an
   explicit reason.
4. **Optional NLI pre-filter** (``docqa.nli``): a confidently neutral passage is skipped without a model call.
5. **Cached judgments**: a passage already judged for the same claim, model and version is not judged again.

Every verdict carries ``abstain_reason`` (why nothing was decided) and ``confidence`` (agreement across passages).
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .index import Evidence
from .llm import LLMError
from .verify import CONTRADICTED, INSUFFICIENT, SUPPORTED, VERIFIERS, Verdict

VERSION = "2.0"
STANCES = ("SUPPORTS", "CONTRADICTS", "NEUTRAL")
STRENGTHS = ("strong", "weak")
GATE_FIELDS = ("same_population", "same_intervention", "same_outcome")
GATE_NAMES = {"same_population": "population", "same_intervention": "intervention or exposure",
              "same_outcome": "outcome"}

SYSTEM = (
    "You check one passage against one claim.\n"
    "Step 1 - relevance. Decide whether the passage reports a finding about the same population, the same "
    "intervention or exposure, and the same outcome as the claim. tests_claim is true only if all three match and the "
    "passage reports a result for that question. A passage on a related topic, background, methods or a hypothesis "
    "does not test the claim.\n"
    "Step 2 - stance. SUPPORTS: the finding shows the claim is true as stated. CONTRADICTS: a finding about the same "
    "question shows the claim is false as stated (opposite direction, no effect where the claim asserts one, an effect "
    "where it asserts none, or an incompatible quantity). NEUTRAL: anything else, including every passage that does "
    "not test the claim. A passage that simply does not mention the claim's result is NEUTRAL, never CONTRADICTS.\n"
    "strength: strong if the passage states the finding directly; weak if it is indirect, hedged or partial.\n"
    "quote: for SUPPORTS or CONTRADICTS, the sentence that shows it, copied exactly as written (at most 40 words); "
    "otherwise an empty string.\n"
    "Answer only with the JSON object.")

SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "same_population": {"type": "boolean"},
        "same_intervention": {"type": "boolean"},
        "same_outcome": {"type": "boolean"},
        "tests_claim": {"type": "boolean"},
        "stance": {"type": "string", "enum": list(STANCES)},
        "strength": {"type": "string", "enum": list(STRENGTHS)},
        "quote": {"type": "string"},
    },
    "required": [*GATE_FIELDS, "tests_claim", "stance", "strength", "quote"],
    "additionalProperties": False,
}
RETRY_SUFFIX = "\n\nReturn only the JSON object with every field."

_ALNUM = re.compile(r"[^a-z0-9]+")


def grounded(quote: str, text: str) -> bool:
    """The quote occurs in the passage, ignoring case, spacing and punctuation (at least 12 letters or digits)."""
    q = _ALNUM.sub("", (quote or "").lower())
    return len(q) >= 12 and q in _ALNUM.sub("", (text or "").lower())


def check_reply(text: str) -> Optional[Dict[str, Any]]:
    """The reply as a dict with every field valid, or None. Nothing is defaulted."""
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        match = re.search(r"\{.*\}", text or "", re.S)          # a model that wraps the object in prose
        try:
            data = json.loads(match.group(0)) if match else None
        except ValueError:
            data = None
    if not isinstance(data, dict):
        return None
    if any(not isinstance(data.get(name), bool) for name in (*GATE_FIELDS, "tests_claim")):
        return None
    stance = str(data.get("stance", "")).upper().strip()
    strength = str(data.get("strength", "")).lower().strip()
    if stance not in STANCES or strength not in STRENGTHS or not isinstance(data.get("quote"), str):
        return None
    return {**{name: data[name] for name in (*GATE_FIELDS, "tests_claim")}, "stance": stance, "strength": strength,
            "quote": data["quote"].strip()}


@dataclass
class Settings:
    num_predict: int = 400
    nli_skip_neutral: float = 0.90          # NLI neutral probability at or above which no model call is made
    nli_confirm_contradiction: float = 0.50  # NLI contradiction probability a CONTRADICTS stance needs to count
    max_passages: int = 8


@dataclass
class Judgment:
    """One passage's judgment and how it counts toward the verdict."""
    eid: str
    stance: str = "NEUTRAL"                 # after the relevance gate and quote grounding
    raw_stance: str = ""                    # as the model gave it
    strength: str = ""
    tests_claim: bool = False
    gate: Dict[str, bool] = field(default_factory=dict)
    quote: str = ""
    grounded: bool = False
    counts: str = ""                        # "support" or "contradict" when it counts toward a decision
    reason: str = ""
    source: str = "model"                   # model, nli, cache
    valid: bool = True
    nli: Optional[Dict[str, float]] = None


class VerdictCache:
    """Per-process cache of passage judgments, keyed by verifier version, model, claim and passage text."""

    def __init__(self, max_entries: int = 4096):
        self.max_entries = max_entries
        self._data: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._lock = threading.Lock()
        self.hits = self.misses = 0

    @staticmethod
    def key(model: str, claim: str, text: str, nli: bool) -> str:
        return hashlib.sha256(json.dumps([VERSION, model, claim, text, nli]).encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            value = self._data.get(key)
            if value is None:
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return dict(value)

    def put(self, key: str, value: Dict[str, Any]) -> None:
        with self._lock:
            self._data[key] = dict(value)
            self._data.move_to_end(key)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self.hits = self.misses = 0


DEFAULT_CACHE = VerdictCache()
_UNSET = object()


class VerifierV2:
    """Three-class verifier: relevance gate, one structured call per passage, calibrated aggregation."""

    name = "verifier_v2"
    variant = f"v{VERSION}: relevance gate, one structured call per passage, calibrated aggregation, optional NLI filter"
    uses_search = False
    records_quote_attestation = False
    attest = False

    def __init__(self, client: Any, nli: Any = _UNSET, cache: Optional[VerdictCache] = None,
                 settings: Optional[Settings] = None):
        self.client = client
        if nli is _UNSET:                                   # the configured filter (WEBRAG_NLI_MODEL), if any
            from .nli import nli_from_env
            nli = nli_from_env()
        self.nli = nli
        self.cache = DEFAULT_CACHE if cache is None else cache
        self.settings = settings or Settings()

    # ------------------------------------------------------------------------------------------------ verify
    def verify(self, claim: str, evidence: Sequence[Evidence], search: Any = None) -> Verdict:
        started = time.perf_counter()
        evidence = list(evidence)[:self.settings.max_passages]
        if not evidence:
            verdict = Verdict(INSUFFICIENT, "NO_EVIDENCE", note="no evidence was retrieved")
            self._extras(verdict, [], "No evidence was retrieved for this statement.", {"nli": "not used"})
            return verdict
        nli_scores, nli_state = self._nli(claim, evidence)
        model = str(getattr(self.client, "model", "") or "")
        calls, seconds = 0, 0.0
        judgments: List[Judgment] = []
        for e, score in zip(evidence, nli_scores):
            key = self.cache.key(model, claim, e.text, score is not None)
            cached = self.cache.get(key)
            if cached is not None:
                judgment = Judgment(**cached)
                judgment.eid, judgment.source = e.id, "cache"
            elif score is not None and score.neutral >= self.settings.nli_skip_neutral:
                judgment = Judgment(e.id, reason="the NLI check found the passage does not bear on the claim",
                                    source="nli", nli=score.as_dict())
            else:
                reply, used, spent = self._ask(claim, e.text)
                calls, seconds = calls + used, seconds + spent
                judgment = self._judge(e, reply, score)
            if judgment.valid and judgment.source != "cache":
                self.cache.put(key, asdict(judgment))
            judgments.append(judgment)
        verdict = self._decide(claim, judgments)
        verdict.calls, verdict.seconds = calls, round(seconds or (time.perf_counter() - started), 3)
        verdict.__dict__["verifier_trace"]["nli"] = nli_state
        verdict.__dict__["verifier_trace"]["cache"] = {"hits": sum(j.source == "cache" for j in judgments),
                                                       "skipped_by_nli": sum(j.source == "nli" for j in judgments)}
        return verdict

    def _nli(self, claim: str, evidence: Sequence[Evidence]):
        if self.nli is None:
            return [None] * len(evidence), "off"
        try:
            scores = self.nli.score([(e.text, claim) for e in evidence])
            return scores, f"on ({getattr(self.nli, 'model_name', 'NLI model')})"
        except Exception as exc:                            # unavailable: run without it, and say so
            return [None] * len(evidence), f"unavailable: {str(exc)[:200]}"

    def _ask(self, claim: str, text: str):
        """(valid reply or None, calls made, model seconds). A service failure raises LLMError to the Studio."""
        user = f"Claim: {claim}\n\nPassage:\n{text}"
        calls, seconds = 0, 0.0
        for suffix in ("", RETRY_SUFFIX):
            response = self.client.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user + suffix}],
                                        response_format=SCHEMA, temperature=0.0, num_predict=self.settings.num_predict)
            calls += 1
            seconds += float(getattr(response, "seconds", 0.0) or 0.0)
            reply = check_reply(response.text)
            if reply is not None:
                return reply, calls, seconds
        return None, calls, seconds

    def _judge(self, e: Evidence, reply: Optional[Dict[str, Any]], score: Any) -> Judgment:
        nli = score.as_dict() if score is not None else None
        if reply is None:
            return Judgment(e.id, valid=False, reason="the model's reply was not valid JSON for the schema after a retry",
                            nli=nli)
        j = Judgment(e.id, stance=reply["stance"], raw_stance=reply["stance"], strength=reply["strength"],
                     tests_claim=reply["tests_claim"], gate={k: reply[k] for k in GATE_FIELDS}, quote=reply["quote"],
                     nli=nli)
        if j.stance == "NEUTRAL":
            j.reason = "the passage does not show the claim true or false"
            return j
        if not j.tests_claim or not all(j.gate.values()):
            missing = [GATE_NAMES[k] for k, ok in j.gate.items() if not ok] or ["question"]
            j.stance, j.reason = "NEUTRAL", f"does not test the claim (different {', '.join(missing)})"
            return j
        j.grounded = grounded(j.quote, e.text)
        if not j.grounded:
            j.stance, j.reason = "NEUTRAL", "its quote was not found in the passage"
            return j
        if j.strength != "strong":
            j.reason = "only weak or indirect evidence"
            return j
        if j.stance == "CONTRADICTS" and score is not None and \
                score.contradiction < self.settings.nli_confirm_contradiction:
            j.reason = "a contradiction the NLI check did not confirm"
            return j
        j.counts = "support" if j.stance == "SUPPORTS" else "contradict"
        j.reason = "strong, grounded evidence that tests the claim"
        return j

    # ------------------------------------------------------------------------------------------------ decide
    def _decide(self, claim: str, judgments: List[Judgment]) -> Verdict:
        invalid = [j for j in judgments if not j.valid]
        stances = {j.eid: j.stance for j in judgments if j.valid}
        if invalid:
            verdict = Verdict(INSUFFICIENT, "ASSESSMENT_FAILED", stances=stances, parse_ok=False,
                              error=f"passage(s) {', '.join(j.eid for j in invalid)} got no valid judgment",
                              note="the verifier's output was unusable; not assessed (fails closed)")
            self._extras(verdict, judgments, "The verifier's output was unusable, so nothing was decided.")
            return verdict
        support = [j for j in judgments if j.counts == "support"]
        contra = [j for j in judgments if j.counts == "contradict"]
        if support and not contra:
            verdict, reason = Verdict(SUPPORTED, stances=stances), ""
        elif contra and not support:
            verdict, reason = Verdict(CONTRADICTED, stances=stances), ""
        elif support and contra:
            verdict = Verdict(INSUFFICIENT, "CONFLICTING", stances=stances)
            reason = (f"Strong evidence points both ways: {len(support)} passage(s) support the statement and "
                      f"{len(contra)} contradict it.")
        else:
            verdict, reason = Verdict(INSUFFICIENT, "NO_DECISIVE_EVIDENCE", stances=stances), self._why(judgments)
        verdict.note = (f"verifier_v2: {len(judgments)} passage(s) judged; {len(support)} strong support, "
                        f"{len(contra)} strong contradiction" + (f"; {reason}" if reason else ""))
        self._extras(verdict, judgments, reason)
        return verdict

    @staticmethod
    def _why(judgments: List[Judgment]) -> str:
        """Why nothing was decided, from the passages' own outcomes."""
        tested = [j for j in judgments if j.tests_claim and all(j.gate.values())]
        if not tested:
            mismatch: Dict[str, int] = {}
            for j in judgments:
                for k, ok in j.gate.items():
                    if not ok:
                        mismatch[GATE_NAMES[k]] = mismatch.get(GATE_NAMES[k], 0) + 1
            if mismatch:
                top = max(mismatch, key=mismatch.get)
                return f"No passage tested this statement: most concern a different {top}."
            return "No passage reported a finding on this statement."
        reasons = [j.reason for j in tested]
        if any("weak" in r for r in reasons):
            return "The passages that test this statement give only weak or indirect evidence."
        if any("quote was not found" in r for r in reasons):
            return "The evidence the verifier quoted could not be found in the passages, so it was not used."
        if any("did not confirm" in r for r in reasons):
            return "A possible contradiction was not confirmed by the NLI check."
        return "The passages that test this statement show it neither true nor false."

    @staticmethod
    def confidence(label: str, judgments: List[Judgment]) -> Dict[str, Any]:
        tested = [j for j in judgments if j.valid and j.tests_claim and all(j.gate.values())]
        want = {SUPPORTED: "SUPPORTS", CONTRADICTED: "CONTRADICTS"}.get(label)
        agreeing = sum(1 for j in judgments if want and j.counts and j.stance == want)
        opposing = sum(1 for j in judgments if want and j.stance in ("SUPPORTS", "CONTRADICTS") and j.stance != want)
        if label == INSUFFICIENT:
            level = "abstained"
        elif agreeing >= 2 and opposing == 0:
            level = "high"
        elif opposing == 0:
            level = "medium"
        else:
            level = "low"
        return {"level": level, "agreeing": agreeing, "opposing": opposing, "tested": len(tested),
                "judged": len(judgments)}

    def _extras(self, verdict: Verdict, judgments: List[Judgment], abstain_reason: str,
                trace: Optional[Dict[str, Any]] = None) -> None:
        verdict.__dict__["abstain_reason"] = abstain_reason if verdict.label == INSUFFICIENT else ""
        verdict.__dict__["confidence"] = self.confidence(verdict.label, judgments)
        verdict.__dict__["passage_reasons"] = {j.eid: j.reason for j in judgments}
        verdict.__dict__["verifier_trace"] = {
            "workflow": self.name, "version": VERSION, **(trace or {}),
            "judgments": [{k: v for k, v in asdict(j).items() if k != "quote"} | {"quote_chars": len(j.quote)}
                          for j in judgments]}


VERIFIERS[VerifierV2.name] = VerifierV2
