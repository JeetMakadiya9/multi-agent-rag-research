"""The end-to-end workflow: retrieve evidence, answer with citations, verify each claim, report limitations."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

from .answer import AnswerGenerator, GeneratedAnswer
from .index import Evidence, EvidenceIndex
from .llm import LLMError
from .verify import CONTRADICTED, INSUFFICIENT, SUPPORTED, Verdict


class RetrievalError(RuntimeError):
    pass


@dataclass
class CheckedClaim:
    text: str
    cited: List[str]
    verdict: Verdict


@dataclass
class QAResult:
    question: str
    evidence: List[Evidence] = field(default_factory=list)
    answer: str = ""
    abstained: bool = False
    claims: List[CheckedClaim] = field(default_factory=list)
    generation: Optional[GeneratedAnswer] = None
    status: str = ""                 # VERIFIED / PARTIALLY_VERIFIED / NOT_VERIFIED / ABSTAINED / FAILED
    limitations: List[str] = field(default_factory=list)
    error: str = ""


@dataclass
class ClaimCheck:
    claim: str
    evidence: List[Evidence] = field(default_factory=list)
    verdict: Optional[Verdict] = None
    error: str = ""


class Assistant:
    def __init__(self, index: EvidenceIndex, generator: AnswerGenerator, verifier: Any, k: int = 4):
        self.index, self.generator, self.verifier, self.k = index, generator, verifier, k

    def _search(self, query: str) -> List[Evidence]:
        try:
            return self.index.search(query, self.k)
        except Exception as exc:                      # any failure inside the retrieval engine
            raise RetrievalError(f"retrieval failed: {type(exc).__name__}: {exc}") from exc

    def verify_claim(self, claim: str) -> ClaimCheck:
        check = ClaimCheck(claim)
        if not claim.strip():
            check.error = "empty claim"
            return check
        try:
            check.evidence = self._search(claim)
            check.verdict = self.verifier.verify(claim, check.evidence)
        except (LLMError, RetrievalError) as exc:
            check.error = str(exc)
        return check

    def ask(self, question: str) -> QAResult:
        result = QAResult(question)
        if not question.strip():
            result.status, result.error = "FAILED", "empty question"
            return result
        try:
            result.evidence = self._search(question)
        except RetrievalError as exc:
            result.status, result.error = "FAILED", str(exc)
            result.limitations.append("Evidence retrieval failed; no answer is given.")
            return result
        if not result.evidence:
            result.abstained, result.status = True, "ABSTAINED"
            result.answer = "The documents do not contain passages relevant to this question."
            result.limitations.append("No relevant evidence was retrieved.")
            return result
        try:
            generation = self.generator.generate(question, result.evidence)
        except LLMError as exc:
            result.status, result.error = "FAILED", str(exc)
            result.limitations.append(f"The language model could not answer: {exc}")
            return result
        result.generation = generation
        if not generation.parse_ok:
            result.status, result.error = "FAILED", generation.error
            result.limitations.append("The model's answer could not be parsed; no answer is shown.")
            return result
        result.answer = generation.answer
        if generation.insufficient and not generation.claims:
            result.abstained, result.status = True, "ABSTAINED"
            result.limitations.append("The model reported that the evidence is insufficient.")
            return result
        for claim in generation.claims:
            try:
                verdict = self.verifier.verify(claim.text, result.evidence)
            except LLMError as exc:
                verdict = Verdict(INSUFFICIENT, "SERVICE_ERROR", parse_ok=False, error=str(exc))
            result.claims.append(CheckedClaim(claim.text, claim.evidence, verdict))
            if claim.unknown_ids:
                result.limitations.append(f"Ignored citations to passages that were not provided: {claim.unknown_ids}")
            if not claim.evidence:
                result.limitations.append(f"Uncited statement: \"{claim.text}\"")
        labels = [c.verdict.label for c in result.claims]
        if generation.insufficient:
            result.limitations.append("The model reported that the evidence is only partly sufficient.")
        if not labels:
            result.status = "NOT_VERIFIED"
            result.limitations.append("The answer contains no checkable statements.")
        elif all(label == SUPPORTED for label in labels):
            result.status = "VERIFIED"
        elif any(label == SUPPORTED for label in labels):
            result.status = "PARTIALLY_VERIFIED"
        else:
            result.status = "NOT_VERIFIED"
        for c in result.claims:
            if c.verdict.label == CONTRADICTED:
                result.limitations.append(f"Contradicted by {', '.join(c.verdict.contradicting)}: \"{c.text}\"")
            elif c.verdict.label == INSUFFICIENT:
                why = c.verdict.sub_state.replace("_", " ").lower() or "not established"
                result.limitations.append(f"Not established by the evidence ({why}): \"{c.text}\"")
        return result


def render_qa(result: QAResult) -> str:
    by_id = {e.id: e for e in result.evidence}
    lines = [f"QUESTION: {result.question}", "", f"ANSWER [{result.status}]:", result.answer or "(no answer)"]
    if result.claims:
        lines += ["", "CLAIMS:"]
        for i, c in enumerate(result.claims, 1):
            cites = ", ".join(f"{e} ({by_id[e].citation})" for e in c.cited if e in by_id) or "no citation"
            lines.append(f"  {i}. [{c.verdict.label}] {c.text}")
            lines.append(f"     cited: {cites}")
    lines += ["", "EVIDENCE:"]
    for e in result.evidence:
        snippet = e.text if len(e.text) <= 300 else e.text[:297] + "..."
        lines.append(f"  [{e.id}] {e.citation}: {snippet}")
    if result.limitations or result.error:
        lines += ["", "LIMITATIONS:"] + [f"  - {x}" for x in result.limitations]
        if result.error:
            lines.append(f"  - Error: {result.error}")
    return "\n".join(lines)


def render_check(check: ClaimCheck) -> str:
    lines = [f"CLAIM: {check.claim}"]
    if check.error:
        return "\n".join(lines + [f"ERROR: {check.error}"])
    v = check.verdict
    sub = f" ({v.sub_state})" if v.sub_state else ""
    lines.append(f"VERDICT: {v.label}{sub}")
    for e in check.evidence:
        snippet = e.text if len(e.text) <= 300 else e.text[:297] + "..."
        lines.append(f"  [{e.id}] {v.stances.get(e.id, 'not assessed')} - {e.citation}: {snippet}")
    if not v.parse_ok:
        lines.append("NOTE: the verification output was incomplete or invalid; the verdict defaults to INSUFFICIENT.")
    return "\n".join(lines)
