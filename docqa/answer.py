"""Answer generation in which every factual statement must cite the evidence passages that state it."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from .index import Evidence
from .llm import LLMResponse, OllamaChat

ANSWER_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "claims": {"type": "array", "items": {
            "type": "object",
            "properties": {"text": {"type": "string"}, "evidence": {"type": "array", "items": {"type": "string"}}},
            "required": ["text", "evidence"]}},
        "insufficient": {"type": "boolean"},
    },
    "required": ["answer", "claims", "insufficient"],
}

SYSTEM_PROMPT = (
    "You are a careful research assistant. Answer the question using ONLY the numbered evidence passages.\n"
    "Rules:\n"
    "1. List every factual statement of your answer in 'claims', each with the IDs of the passages that state it "
    "(for example [\"E2\"]).\n"
    "2. Do not use outside knowledge.\n"
    "3. If the passages do not contain the information needed, set 'insufficient' to true, say what is missing, "
    "and do not guess.\n"
    "4. If a passage contradicts an assumption in the question, say so and cite that passage.\n"
    "5. Keep the answer short. Return JSON only."
)


def format_evidence(evidence: Sequence[Evidence]) -> str:
    return "\n\n".join(f"[{e.id}] (source: {e.citation})\n{e.text}" for e in evidence)


@dataclass
class AnswerClaim:
    text: str
    evidence: List[str]                       # valid passage IDs only
    unknown_ids: List[str] = field(default_factory=list)


@dataclass
class GeneratedAnswer:
    answer: str = ""
    claims: List[AnswerClaim] = field(default_factory=list)
    insufficient: bool = False
    parse_ok: bool = True
    truncated: bool = False
    calls: int = 0
    seconds: float = 0.0
    error: str = ""
    raw: List[str] = field(default_factory=list)


def parse_answer(text: str, valid_ids: Sequence[str]) -> GeneratedAnswer:
    """Strict parse: a JSON object with a string answer, a claims list and a boolean insufficient flag."""
    data = json.loads(text)
    if not isinstance(data, dict) or not isinstance(data.get("answer"), str) \
            or not isinstance(data.get("claims"), list) or not isinstance(data.get("insufficient"), bool):
        raise ValueError("answer JSON does not match the schema")
    valid = set(valid_ids)
    claims = []
    for row in data["claims"]:
        if not isinstance(row, dict) or not isinstance(row.get("text"), str) or not row["text"].strip():
            raise ValueError("claim without text")
        ids = [str(x).strip().strip("[]") for x in row.get("evidence") or [] if str(x).strip()]
        claims.append(AnswerClaim(row["text"].strip(), [i for i in dict.fromkeys(ids) if i in valid],
                                  [i for i in dict.fromkeys(ids) if i not in valid]))
    return GeneratedAnswer(answer=data["answer"].strip(), claims=claims, insufficient=data["insufficient"])


class AnswerGenerator:
    def __init__(self, client: OllamaChat, num_predict: int = 512):
        self.client, self.num_predict = client, num_predict

    def generate(self, question: str, evidence: Sequence[Evidence]) -> GeneratedAnswer:
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Question: {question}\n\nEvidence passages:\n{format_evidence(evidence)}"}]
        responses: List[LLMResponse] = []
        result = GeneratedAnswer(parse_ok=False, error="no attempt")
        for _ in range(2):                                       # one retry on an unusable response
            response = self.client.chat(messages, response_format=ANSWER_SCHEMA, num_predict=self.num_predict)
            responses.append(response)
            try:
                result = parse_answer(response.text, [e.id for e in evidence])
                break
            except (ValueError, json.JSONDecodeError) as exc:
                result = GeneratedAnswer(parse_ok=False, error=f"unparsable answer ({type(exc).__name__})")
        result.calls, result.seconds = len(responses), round(sum(r.seconds for r in responses), 3)
        result.truncated = any(r.truncated for r in responses)
        result.raw = [r.text for r in responses]
        return result
