"""Controlled verification engine for Experiment B.

This module is independent of FAISS/Ollama imports. Retrieval and language
model providers are injected, allowing deterministic, offline tests.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Any, Optional, Sequence

from claim_verification import (
    CONTRADICTED, INSUFFICIENT_EVIDENCE, SUPPORTED, Claim, ClaimVerification,
    EvidenceItem, VerificationReport, _extract_json_object,
    evidence_from_retrieval_results,
)
from llm import LLMProvider, OllamaProvider
from verification_units import VerificationUnit, analyze_verification_units


VALID_VERDICTS = {SUPPORTED, CONTRADICTED, INSUFFICIENT_EVIDENCE}


@dataclass
class ExperimentBConfig:
    """Explicit, bounded controls; values support later ablation runs."""

    model: str = "qwen3:4b"
    ollama_url: str = "http://127.0.0.1:11434/api/chat"
    think: Optional[bool] = False
    response_format: Optional[str] = "json"
    temperature: float = 0.0
    timeout_seconds: int = 120
    max_evidence_items: int = 8
    max_judge_retries: int = 1
    unit_specific_retrieval: bool = True
    max_targeted_retrieval_retries: int = 1
    validator_enabled: bool = True
    answer_revision_enabled: bool = True

    def __post_init__(self) -> None:
        if self.max_evidence_items < 1:
            raise ValueError("max_evidence_items must be >= 1")
        if self.max_judge_retries < 0 or self.max_targeted_retrieval_retries < 0:
            raise ValueError("Retry limits cannot be negative")
        if self.timeout_seconds < 1:
            raise ValueError("timeout_seconds must be >= 1")


@dataclass
class ValidationResult:
    verdict: str
    overrides: list[str] = field(default_factory=list)
    conflict_resolution: Optional[str] = None
    coverage_matrix: list[dict[str, Any]] = field(default_factory=list)
    supported_components: list[str] = field(default_factory=list)
    unsupported_components: list[str] = field(default_factory=list)
    contradicted_components: list[str] = field(default_factory=list)
    conflicting_components: list[str] = field(default_factory=list)
    validation_errors: list[str] = field(default_factory=list)
    assessment_complete: bool = True
    missing_assessments: list[str] = field(default_factory=list)


def build_unit_retrieval_query(question: str, unit: VerificationUnit,
                               missing_components: Sequence[str] = ()) -> str:
    """Combine question context with unit semantics and any missing details."""
    terms: list[str] = [question.strip(), unit.original_text]
    if unit.subject:
        terms.append(unit.subject)
    if unit.predicate:
        terms.append(unit.predicate)
    terms.extend(unit.objects)
    terms.extend(unit.temporal_constraints)
    terms.extend(unit.quantitative_constraints)
    terms.extend(unit.conditions)
    if unit.negation:
        terms.append(unit.negation)
    if unit.causal_relation:
        terms.append(unit.causal_relation)
    if unit.comparison_relation:
        terms.append(unit.comparison_relation)
    terms.extend(str(item) for item in missing_components if str(item).strip())
    return " ".join(dict.fromkeys(term for term in terms if term))


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9$%]+", " ", text.lower()).strip()


def _numeric_anchors(text: str) -> set[str]:
    return {re.sub(r"[,.]", "", token.lower()) for token in re.findall(
        r"\$?\d[\d,.]*(?:\s?%|\s?(?:billion|million|thousand|ms|seconds?|minutes?|hours?|gb|mb|kb|tb|images?|pages?|samples?|records?|degrees?\s+celsius))?",
        text, re.I,
    )}


def _hard_qualifier_gaps(unit: VerificationUnit, evidence: Sequence[EvidenceItem]) -> list[str]:
    """Conservatively require explicit dates/numbers when present in a unit."""
    corpus = "\n".join(item.text for item in evidence)
    corpus_norm = _normalise(corpus)
    missing: list[str] = []
    for date in unit.temporal_constraints:
        date_norm = _normalise(date)
        digits = re.sub(r"\D", "", date)
        if date_norm and date_norm not in corpus_norm and not (digits and digits in re.sub(r"\D", "", corpus)):
            missing.append(f"date/time qualifier: {date}")
    evidence_numbers = _numeric_anchors(corpus)
    for quantity in unit.quantitative_constraints:
        anchors = _numeric_anchors(quantity)
        if anchors and not anchors.issubset(evidence_numbers):
            missing.append(f"quantity: {quantity}")
    return missing


def _coordinated_component_gaps(unit: VerificationUnit, evidence: Sequence[EvidenceItem]) -> list[str]:
    """Require explicit coverage of each deterministically parsed list item.

    Presence is necessary but not sufficient for entailment; final semantic
    support still requires the judge. This specifically prevents a compound
    list from being marked fully supported when a material object is absent.
    """
    if len(unit.objects) < 2:
        return []
    text = _normalise(" ".join(item.text for item in evidence))
    missing = []
    for obj in unit.objects:
        anchor = re.sub(r"^(?:the|a|an)\s+", "", obj.strip(), flags=re.I)
        if _normalise(anchor) not in text:
            missing.append(f"coordinated object: {obj}")
    return missing


def _metadata_strength(item: EvidenceItem) -> Optional[float]:
    """Read explicit, auditable provenance-quality fields only."""
    meta = item.metadata or {}
    values = []
    for key in ("source_reliability", "authority_score", "directness_score", "specificity_score", "consistency_score"):
        try:
            if meta.get(key) is not None:
                values.append(max(0.0, min(1.0, float(meta[key]))))
        except (TypeError, ValueError):
            continue
    return sum(values) / len(values) if values else None


def _resolve_conflict(unit: VerificationUnit, evidence_by_id: dict[str, EvidenceItem],
                      supporting_ids: Sequence[str], contradicting_ids: Sequence[str]) -> tuple[Optional[str], Optional[str]]:
    """Resolve only when explicit provenance strongly distinguishes sides.

    Evidence without auditable quality metadata is not ranked by retrieval
    relevance, because relevance is not entailment or source authority.
    """
    support = [evidence_by_id[x] for x in supporting_ids if x in evidence_by_id]
    refute = [evidence_by_id[x] for x in contradicting_ids if x in evidence_by_id]
    if not support or not refute:
        return None, None
    target_years = set(re.findall(r"\b(?:19|20)\d{2}\b", " ".join(unit.temporal_constraints)))
    if target_years:
        def event_years(item: EvidenceItem) -> set[str]:
            metadata = item.metadata or {}
            temporal_values = [metadata.get(key, "") for key in
                               ("event_date", "effective_at", "valid_at", "temporal_scope")]
            return set(re.findall(r"\b(?:19|20)\d{2}\b", " ".join(map(str, temporal_values))))
        support_years = set().union(*(event_years(x) for x in support))
        refute_years = set().union(*(event_years(x) for x in refute))
        support_matches = bool(target_years & support_years)
        refute_matches = bool(target_years & refute_years)
        support_conflicts = bool(support_years and not support_matches)
        refute_conflicts = bool(refute_years and not refute_matches)
        if support_matches and refute_conflicts:
            return SUPPORTED, "Resolved by explicit event/effective-date metadata matching the unit's temporal constraint."
        if refute_matches and support_conflicts:
            return CONTRADICTED, "Resolved by explicit event/effective-date metadata matching the unit's temporal constraint."
    support_strength = max((_metadata_strength(x) for x in support), default=None)
    refute_strength = max((_metadata_strength(x) for x in refute), default=None)
    if support_strength is not None and refute_strength is not None:
        gap = support_strength - refute_strength
        if abs(gap) >= 0.35:
            side = SUPPORTED if gap > 0 else CONTRADICTED
            return side, f"Resolved by explicit provenance-quality metadata (strength gap {abs(gap):.2f})."
    # Explicit supersession and effective-date metadata may resolve a dated unit.
    for winning, losing, verdict in ((support, refute, SUPPORTED), (refute, support, CONTRADICTED)):
        for item in winning:
            supersedes = set(map(str, item.metadata.get("supersedes_evidence_ids", [])))
            if supersedes.intersection(x.evidence_id for x in losing):
                return verdict, f"Resolved because {item.evidence_id} explicitly supersedes conflicting evidence."
    support_sources = {x.source_id or x.document_id or x.source for x in support if x.source_id or x.document_id or x.source}
    refute_sources = {x.source_id or x.document_id or x.source for x in refute if x.source_id or x.document_id or x.source}
    source_note = f" Observed source identities: support={len(support_sources)}, refute={len(refute_sources)}."
    return None, "Conflict unresolved after checking explicit provenance, temporal scope, specificity/directness, supersession, and source identities." + source_note


def validate_judgement(unit: VerificationUnit, data: dict[str, Any],
                       evidence: Sequence[EvidenceItem], *, enabled: bool = True) -> ValidationResult:
    """Validate stances and deterministically aggregate component coverage.

    A legacy answer-level decision is never used for multi-component units.
    For one-component units, legacy evidence IDs are accepted as a migration
    path and converted to evidence/component relations before aggregation.
    """
    components = list(unit.components)
    if not components:
        from verification_units import VerificationComponent
        components = [VerificationComponent("C1", unit.original_text, parent_unit_id=unit.id)]
    evidence_by_index = {f"E{i}": item for i, item in enumerate(evidence, start=1)}
    component_by_index = {component.component_id: component for component in components}
    legacy_contract = "assessments" not in data
    rows = data.get("assessments", data.get("evidence_assessments", []))
    rows = rows if isinstance(rows, list) else []
    normalized_rows: list[dict[str, Any]] = []
    errors: list[str] = []
    legacy_invalid_evidence: set[str] = set()
    # Migration support is intentionally limited to a single requirement.
    if legacy_contract and not rows and len(components) == 1:
        legacy_support = set(map(str, data.get("supporting_evidence_ids", [])))
        legacy_refute = set(map(str, data.get("contradicting_evidence_ids", [])))
        allowed_legacy_ids = {item.evidence_id for item in evidence}
        legacy_invalid_evidence.update((legacy_support | legacy_refute) - allowed_legacy_ids)
        for index, item in enumerate(evidence, start=1):
            stance = "SUPPORTS" if item.evidence_id in legacy_support else (
                "CONTRADICTS" if item.evidence_id in legacy_refute else "NEUTRAL")
            if stance != "NEUTRAL":
                normalized_rows.append({"evidence_index": f"E{index}", "component_indices": ["C1"], "stance": stance})
    else:
        for row in rows:
            if not isinstance(row, dict):
                errors.append("Assessment row is not an object.")
                continue
            evidence_index = str(row.get("evidence_index", ""))
            # Accept the old evidence_id shape only for one-component units.
            legacy_row = not evidence_index and "evidence_id" in row and len(components) == 1
            if legacy_row:
                legacy_id = str(row.get("evidence_id", ""))
                evidence_index = next((idx for idx, item in evidence_by_index.items() if item.evidence_id == legacy_id), "")
            component_indices = row.get("component_indices", [])
            if not component_indices and legacy_row:
                component_indices = ["C1"]
            if not isinstance(component_indices, list) or not component_indices:
                errors.append("Invalid component reference list.")
                continue
            stance = str(row.get("stance", row.get("interpretation", "NEUTRAL"))).upper()
            if stance in {"REFUTES", "CONTRADICT", "CONTRADICTS"}:
                stance = "CONTRADICTS"
            if stance not in {"SUPPORTS", "CONTRADICTS", "NEUTRAL"}:
                errors.append(f"Invalid stance: {stance}")
                continue
            normalized_rows.append({"evidence_index": evidence_index,
                                    "component_indices": [str(x) for x in component_indices], "stance": stance})

    stances: dict[str, dict[str, set[str]]] = {
        key: {"SUPPORTS": set(), "CONTRADICTS": set()} for key in component_by_index
    }
    invalid_evidence: set[str] = set()
    invalid_components: set[str] = set()
    assessed_pairs: set[tuple[str, str]] = set()
    for row in normalized_rows:
        eidx = row["evidence_index"]
        if eidx not in evidence_by_index:
            invalid_evidence.add(eidx or "<missing>")
            continue
        for cidx in row["component_indices"]:
            if cidx not in component_by_index:
                invalid_components.add(cidx or "<missing>")
                continue
            assessed_pairs.add((eidx, cidx))
            if row["stance"] in {"SUPPORTS", "CONTRADICTS"}:
                stances[cidx][row["stance"]].add(eidx)
    if invalid_evidence:
        errors.append("Invalid evidence reference(s): " + ", ".join(sorted(invalid_evidence)))
    if legacy_invalid_evidence:
        errors.append("Unknown evidence ID(s): " + ", ".join(sorted(legacy_invalid_evidence)))
    if invalid_components:
        errors.append("Invalid component reference(s): " + ", ".join(sorted(invalid_components)))

    required_pairs = {
        (eidx, cidx) for eidx in evidence_by_index for cidx in component_by_index
    }
    # Even the single-component legacy adapter must not let unreferenced
    # supplied evidence silently stand in for a completed assessment.
    missing_pair_set = required_pairs - assessed_pairs
    missing_assessments = [f"{eidx}-{cidx}" for eidx, cidx in sorted(missing_pair_set)]
    assessment_complete = not missing_assessments
    if missing_assessments:
        errors.append("Missing assessment(s): " + ", ".join(missing_assessments))

    coverage: list[dict[str, Any]] = []
    supported: list[str] = []
    unsupported: list[str] = []
    contradicted: list[str] = []
    conflicting: list[str] = []
    for cidx, component in component_by_index.items():
        support = sorted(stances[cidx]["SUPPORTS"])
        refute = sorted(stances[cidx]["CONTRADICTS"])
        component_incomplete = any(pair_component == cidx for _, pair_component in missing_pair_set)
        if component_incomplete:
            state = "INCOMPLETE"
            unsupported.append(component.text)
        elif support and refute:
            state = "CONFLICTING"
            conflicting.append(component.text)
        elif refute:
            state = "CONTRADICTED"
            contradicted.append(component.text)
        elif support:
            state = "SUPPORTED"
            supported.append(component.text)
        else:
            state = "UNSUPPORTED"
            unsupported.append(component.text)
        coverage.append({"component_index": cidx, "component": component.text, "status": state,
                         "supporting_evidence_indices": support, "contradicting_evidence_indices": refute})

    # Preserve exact numeric/date and coordinated-item guardrails even if the
    # judge omits a requirement or returns a stance that is too permissive.
    gaps = _hard_qualifier_gaps(unit, evidence) + _coordinated_component_gaps(unit, evidence)
    if gaps:
        errors.extend("Uncovered required detail: " + gap for gap in gaps)
    all_supported = len(supported) == len(components) and not conflicting and not errors
    hard_invalid = bool(invalid_evidence or legacy_invalid_evidence or invalid_components) or any(
        error.startswith(("Assessment row", "Invalid component reference", "Invalid component reference list", "Invalid stance"))
        for error in errors
    )
    if hard_invalid or not assessment_complete:
        verdict = INSUFFICIENT_EVIDENCE
    elif conflicting:
        verdict = INSUFFICIENT_EVIDENCE
    elif contradicted:
        verdict = CONTRADICTED
    elif errors:
        verdict = INSUFFICIENT_EVIDENCE
    elif all_supported:
        verdict = SUPPORTED
    else:
        verdict = INSUFFICIENT_EVIDENCE
    overrides = list(errors)
    if conflicting:
        overrides.append("Evidence stances conflict on a component; verdict is INSUFFICIENT_EVIDENCE.")
    elif contradicted:
        overrides.append("At least one required component is contradicted by evidence.")
    elif not all_supported and not errors:
        overrides.append("Not every required component has supporting evidence.")
    return ValidationResult(
        verdict=verdict, overrides=overrides, conflict_resolution=None,
        coverage_matrix=coverage, supported_components=supported,
        unsupported_components=unsupported, contradicted_components=contradicted,
        conflicting_components=conflicting, validation_errors=errors,
        assessment_complete=assessment_complete,
        missing_assessments=missing_assessments,
    )


def _judge_prompt(question: str, unit: VerificationUnit, evidence: Sequence[EvidenceItem]) -> list[dict[str, str]]:
    evidence_rows = [{"evidence_index": f"E{i}", "text": x.text} for i, x in enumerate(evidence, start=1)]
    component_rows = [{"component_index": component.component_id, "text": component.text,
                       "required": component.required} for component in unit.components]
    schema = {"assessments": [{"evidence_index": "E1", "component_indices": ["C1"],
                               "stance": "SUPPORTS | CONTRADICTS | NEUTRAL"}], "reason": "brief explanation"}
    payload = {"question_context": question, "verification_unit": unit.to_dict(),
               "original_unit_text": unit.original_text, "components": component_rows,
               "selected_evidence": evidence_rows}
    system = (
        "Interpret evidence only; do NOT produce a final verdict. Assess each evidence item against explicit components. "
        "An evidence item may SUPPORT one component and be NEUTRAL toward another. Example: for FAISS and BM25, "
        "a FAISS-only passage is SUPPORTS C1 and NEUTRAL C2, never support for the whole unit. "
        "Topical relevance is not entailment; absence is NEUTRAL, not contradiction. Preserve dates, quantities, "
        "conditions, negation and relations exactly as included in component text. Return one assessment per evidence/component "
        "pair, with positional E and C indices only. Allowed stance values: SUPPORTS, CONTRADICTS, NEUTRAL. "
        "Do not output SUPPORTED, CONTRADICTED, or INSUFFICIENT_EVIDENCE. Return JSON only."
    )
    user = "Schema:\n" + json.dumps(schema, ensure_ascii=False) + "\nInput:\n" + json.dumps(payload, ensure_ascii=False)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _pairwise_judge_prompt(question: str, unit: VerificationUnit, evidence_index: str,
                           evidence: EvidenceItem, component: Any) -> list[dict[str, str]]:
    """Ask for one stance on exactly one positional evidence/component pair."""
    schema = {
        "assessments": [{"evidence_index": evidence_index,
                         "component_indices": [component.component_id],
                         "stance": "SUPPORTS | CONTRADICTS | NEUTRAL"}],
        "reason": "brief explanation",
    }
    payload = {
        "question_context": question,
        "original_unit_text": unit.original_text,
        "component": {"component_index": component.component_id,
                      "text": component.text, "required": component.required},
        "selected_evidence": {"evidence_index": evidence_index, "text": evidence.text},
    }
    system = (
        "Interpret evidence only; do NOT produce a final verdict. Assess only the single supplied evidence item "
        "against the single supplied verification component. SUPPORTS means this evidence affirms the component; "
        "CONTRADICTS means this evidence directly negates or is incompatible with the component; NEUTRAL means "
        "it neither affirms nor contradicts it. Topical relevance or absence is NEUTRAL, not contradiction. "
        "Preserve dates, quantities, conditions, negation and relations exactly as included in the component. "
        "Return exactly one assessment for the supplied positional evidence and component indices. Allowed stance "
        "values: SUPPORTS, CONTRADICTS, NEUTRAL. Do not output a claim-level verdict. Return JSON only."
    )
    user = "Schema:\n" + json.dumps(schema, ensure_ascii=False) + "\nInput:\n" + json.dumps(payload, ensure_ascii=False)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _parse_pairwise_assessment(raw: str, evidence_index: str, component: Any,
                               evidence: EvidenceItem, unit: VerificationUnit) -> tuple[dict[str, Any], str]:
    """Validate one pair response; never fill an omitted stance with NEUTRAL."""
    data = _parse_judgement(raw)
    if "assessments" in data:
        if "verdict" in data:
            raise ValueError("Pairwise stance output must not contain a final verdict.")
        rows = data["assessments"]
        if len(rows) != 1:
            raise ValueError(f"Expected exactly one pair assessment; received {len(rows)}.")
        row = rows[0]
        if str(row.get("evidence_index", "")) != evidence_index:
            raise ValueError(f"Expected evidence index {evidence_index!r}.")
        component_indices = row.get("component_indices")
        if component_indices != [component.component_id]:
            raise ValueError(f"Expected component indices [{component.component_id!r}].")
        stance = row.get("stance")
        if not isinstance(stance, str) or stance.upper() not in {"SUPPORTS", "CONTRADICTS", "NEUTRAL"}:
            raise ValueError("Pair assessment must contain a valid SUPPORTS, CONTRADICTS, or NEUTRAL stance.")
        assessment = {"evidence_index": evidence_index,
                      "component_indices": [component.component_id], "stance": stance.upper()}
        return assessment, str(data["reason"]).strip()

    # Preserve the existing one-component legacy response path only when it
    # explicitly names this evidence as supporting/refuting (or gives an
    # explicit evidence_assessments stance). An unmentioned ID is not NEUTRAL.
    if len(unit.components) != 1:
        raise ValueError("Legacy evidence-ID responses are supported only for one-component units.")
    explicit_stances: set[str] = set()
    legacy_rows = data.get("evidence_assessments", [])
    if isinstance(legacy_rows, list):
        for legacy_row in legacy_rows:
            if not isinstance(legacy_row, dict) or str(legacy_row.get("evidence_id", "")) != evidence.evidence_id:
                continue
            value = str(legacy_row.get("stance", legacy_row.get("interpretation", ""))).upper()
            aliases = {"SUPPORT": "SUPPORTS", "REFUTES": "CONTRADICTS", "CONTRADICT": "CONTRADICTS"}
            value = aliases.get(value, value)
            if value in {"SUPPORTS", "CONTRADICTS", "NEUTRAL"}:
                explicit_stances.add(value)
    support_ids = set(map(str, data.get("supporting_evidence_ids", [])))
    refute_ids = set(map(str, data.get("contradicting_evidence_ids", [])))
    if evidence.evidence_id in support_ids:
        explicit_stances.add("SUPPORTS")
    if evidence.evidence_id in refute_ids:
        explicit_stances.add("CONTRADICTS")
    if len(explicit_stances) != 1:
        raise ValueError("Legacy response did not explicitly provide exactly one stance for this evidence item.")
    return ({"evidence_index": evidence_index,
             "component_indices": [component.component_id],
             "stance": next(iter(explicit_stances))},
            str(data.get("reason", data.get("explanation", ""))).strip())


def _parse_judgement(raw: str) -> dict[str, Any]:
    data = _extract_json_object(raw)
    # New contract: stance matrix and a brief reason, with no model verdict.
    if "assessments" in data:
        if not isinstance(data["assessments"], list) or any(not isinstance(x, dict) for x in data["assessments"]):
            raise ValueError("assessments must be a list of objects.")
        if not isinstance(data.get("reason"), str):
            raise ValueError("reason must be a string.")
        return data
    # Transitional legacy payload validation; final verdict is ignored.
    if not isinstance(data.get("verdict"), str) or data["verdict"].upper() not in VALID_VERDICTS:
        raise ValueError("Legacy judgement verdict is invalid.")
    data.setdefault("supporting_evidence_ids", [])
    data.setdefault("contradicting_evidence_ids", [])
    data.setdefault("reason", data.get("explanation", ""))
    data.setdefault("evidence_assessments", [])
    return data


def _normalise_retrieval_response(value: Any, limit: int) -> list[EvidenceItem]:
    if value is None:
        return []
    evidence = getattr(value, "evidence", None)
    if evidence is None:
        evidence = value
    return evidence_from_retrieval_results(list(evidence)[:limit], max_items=limit)


def _merge_evidence(*groups: Sequence[EvidenceItem], limit: int) -> list[EvidenceItem]:
    result: list[EvidenceItem] = []
    seen: set[str] = set()
    for group in groups:
        for item in group:
            if not item.text.strip():
                continue
            key = item.evidence_id or f"{item.source_id}:{item.page}:{item.text[:80]}"
            if key in seen:
                continue
            seen.add(key)
            result.append(item)
            if len(result) >= limit:
                return result
    return result


def _evidence_ids(data: dict[str, Any], key: str) -> list[str]:
    value = data.get(key, [])
    ids = [str(x) for x in value] if isinstance(value, list) else []
    stance = "SUPPORTS" if key == "supporting_evidence_ids" else "REFUTES"
    for row in data.get("evidence_assessments", []):
        if not isinstance(row, dict):
            continue
        interpretation = str(row.get("interpretation", "")).upper()
        matching = interpretation in ({"SUPPORTS", "SUPPORT"} if stance == "SUPPORTS" else {"REFUTES", "CONTRADICTS", "CONTRADICT"})
        evidence_id = str(row.get("evidence_id", ""))
        if matching and evidence_id and evidence_id not in ids:
            ids.append(evidence_id)
    return ids


def _verify_unit(unit: VerificationUnit, question: str, evidence: list[EvidenceItem],
                 provider: LLMProvider, config: ExperimentBConfig,
                 llm_calls: dict[str, int]) -> tuple[ClaimVerification, dict[str, Any]]:
    retry_events: list[dict[str, Any]] = []
    assessments: list[dict[str, Any]] = []
    reasons: list[str] = []
    pair_calls: list[dict[str, Any]] = []
    pair_errors: dict[str, str] = {}
    elapsed = 0.0

    # Keep each pair independent. A malformed response only invalidates its
    # own pair; successful assessments remain available to deterministic
    # coverage and the final verdict validator.
    for evidence_number, item in enumerate(evidence, start=1):
        evidence_index = f"E{evidence_number}"
        for component in unit.components:
            pair_id = f"{evidence_index}-{component.component_id}"
            pair_record: dict[str, Any] = {
                "pair_id": pair_id, "evidence_index": evidence_index,
                "component_index": component.component_id,
                "attempts": [], "assessment": None,
            }
            pair_started = perf_counter()
            pair_error: Optional[str] = None
            for attempt in range(config.max_judge_retries + 1):
                raw_response = ""
                attempt_record: dict[str, Any] = {"attempt": attempt + 1}
                llm_calls["llm"] += 1
                try:
                    raw_response = provider.generate(
                        _pairwise_judge_prompt(question, unit, evidence_index, item, component),
                        temperature=config.temperature, timeout=config.timeout_seconds,
                    )
                    assessment, pair_reason = _parse_pairwise_assessment(
                        raw_response, evidence_index, component, item, unit)
                    attempt_record.update({"raw_response": raw_response, "parse_error": None})
                    pair_record["attempts"].append(attempt_record)
                    pair_record["assessment"] = assessment
                    assessments.append(assessment)
                    if pair_reason:
                        reasons.append(f"{pair_id}: {pair_reason}")
                    if attempt:
                        retry_events.append({"stage": "semantic_judge", "pair": pair_id,
                                             "reason": "prior pair output was invalid", "attempt": attempt + 1})
                    pair_error = None
                    break
                except Exception as exc:
                    pair_error = f"{type(exc).__name__}: {exc}"
                    attempt_record.update({"raw_response": raw_response, "parse_error": pair_error})
                    pair_record["attempts"].append(attempt_record)
                    retry_events.append({"stage": "semantic_judge", "pair": pair_id,
                                         "reason": pair_error, "attempt": attempt + 1})
            elapsed += perf_counter() - pair_started
            if pair_error:
                pair_errors[pair_id] = pair_error
            pair_record["error"] = pair_error
            pair_calls.append(pair_record)

    data = {"assessments": assessments,
            "reason": "; ".join(reasons) or "No valid evidence/component assessment was returned."}
    # The existing raw-output field carries the exact responses in a JSON
    # envelope, retaining every pair and retry without losing successful rows.
    raw = json.dumps({"pairwise_assessment_calls": pair_calls,
                      "assembled_assessments": assessments}, ensure_ascii=False)
    validation = validate_judgement(unit, data, evidence, enabled=config.validator_enabled)
    for pair_id, error in pair_errors.items():
        message = f"Invalid assessment for {pair_id}: {error}"
        validation.validation_errors.append(message)
        validation.overrides.append(message)
    evidence_by_index = {f"E{i}": item for i, item in enumerate(evidence, start=1)}
    supporting_indices = {idx for row in validation.coverage_matrix for idx in row["supporting_evidence_indices"]}
    contradicting_indices = {idx for row in validation.coverage_matrix for idx in row["contradicting_evidence_indices"]}
    selected_support = [evidence_by_index[idx] for idx in sorted(supporting_indices) if idx in evidence_by_index]
    selected_refute = [evidence_by_index[idx] for idx in sorted(contradicting_indices) if idx in evidence_by_index]
    verdict = validation.verdict
    # Keep compatibility fields derived from deterministic coverage.
    supported_components = list(validation.supported_components)
    unsupported_components = list(validation.unsupported_components)
    reason = str(data.get("reason", data.get("explanation", ""))).strip()
    assessments = data.get("assessments", data.get("evidence_assessments", []))
    for item in evidence:
        item.supported_components = []
        item.unsupported_components = []
        item.interpretation = "NEUTRAL"
    for row in assessments if isinstance(assessments, list) else []:
        if not isinstance(row, dict):
            continue
        eidx = str(row.get("evidence_index", ""))
        if not eidx and len(unit.components) == 1:
            old_id = str(row.get("evidence_id", ""))
            eidx = next((idx for idx, ev in evidence_by_index.items() if ev.evidence_id == old_id), "")
        ev = evidence_by_index.get(eidx)
        if ev is None:
            continue
        stance = str(row.get("stance", row.get("interpretation", "NEUTRAL"))).upper()
        ev.interpretation = stance
        component_indices = row.get("component_indices", ["C1"] if len(unit.components) == 1 else [])
        for cidx in component_indices if isinstance(component_indices, list) else []:
            component = next((x for x in unit.components if x.component_id == str(cidx)), None)
            if component and stance in {"SUPPORTS", "SUPPORT"}:
                ev.supported_components.append(component.text)
            elif component:
                ev.unsupported_components.append(component.text)
    confidence = 1.0 if verdict != INSUFFICIENT_EVIDENCE else 0.0
    result = ClaimVerification(
        claim_id=unit.id, claim=unit.original_text, verdict=verdict, confidence=confidence,
        supporting_evidence=selected_support, explanation=data["reason"],
        contradicting_evidence=selected_refute,
        supported_components=supported_components,
        unsupported_components=unsupported_components,
        evidence_assessments=list(assessments) if isinstance(assessments, list) else [],
        llm_verdict=None, validator_overrides=validation.overrides,
        retry_events=retry_events,
        coverage_matrix=validation.coverage_matrix,
        contradicted_components=validation.contradicted_components,
        conflicting_components=validation.conflicting_components,
        validation_errors=validation.validation_errors,
        raw_judge_output=raw,
        assessment_complete=validation.assessment_complete,
        missing_assessments=validation.missing_assessments,
    )
    return result, {"raw_response": raw, "latency_seconds": elapsed,
                    "parse_error": "; ".join(f"{pair}: {error}" for pair, error in pair_errors.items()) or None,
                    "pair_assessment_calls": pair_calls,
                    "invalid_assessments": dict(pair_errors),
                    "validator_overrides": list(validation.overrides),
                    "validation_errors": list(validation.validation_errors),
                    "assessment_complete": validation.assessment_complete,
                    "missing_assessments": list(validation.missing_assessments),
                    "coverage_matrix": validation.coverage_matrix,
                    "conflict_resolution": validation.conflict_resolution}


def revise_answer(answer: str, verifications: Sequence[ClaimVerification]) -> str:
    """Revise deterministically and cite only selected evidence IDs.

    Supported units are retained. Unsupported assertions are qualified;
    contradicted assertions are replaced by direct evidence quotations.
    """
    sections: list[str] = []
    for item in verifications:
        if item.verdict == SUPPORTED:
            refs = " ".join(f"[{e.evidence_id}]" for e in item.supporting_evidence)
            sections.append(item.claim + (f" {refs}" if refs else ""))
        elif item.verdict == CONTRADICTED:
            if item.contradicting_evidence:
                ev = item.contradicting_evidence[0]
                sections.append(f"The evidence contradicts the original statement. Evidence: “{ev.text}” [{ev.evidence_id}]")
            else:
                sections.append("A statement in the original answer was contradicted by retrieved evidence and has been omitted.")
        elif item.conflicting_components:
            refs = list(dict.fromkeys(
                evidence.evidence_id
                for evidence in [*item.supporting_evidence, *item.contradicting_evidence]
            ))
            citation_text = " ".join(f"[{evidence_id}]" for evidence_id in refs)
            sections.append(
                "The available sources conflict about: "
                + "; ".join(item.conflicting_components)
                + (f" {citation_text}" if citation_text else "")
                + ". The claim remains unresolved."
            )
        else:
            if item.supported_components and item.unsupported_components:
                refs = " ".join(f"[{e.evidence_id}]" for e in item.supporting_evidence)
                sections.append(
                    "Evidence supports part of the statement: "
                    + "; ".join(item.supported_components)
                    + (f" {refs}" if refs else "")
                    + ". The available evidence does not establish: "
                    + "; ".join(item.unsupported_components)
                    + "."
                )
            else:
                sections.append(f"The available evidence does not establish: {item.claim}")
    if not sections and answer.strip():
        return "The available evidence did not establish any verification units in the original answer."
    return "\n".join(sections)


def run_experiment_b_verification(
    question: str,
    answer: str,
    retrieval_results: Sequence[Any],
    *,
    retrieval_provider: Any = None,
    llm_provider: Optional[LLMProvider] = None,
    config: Optional[ExperimentBConfig] = None,
) -> VerificationReport:
    """Verify conservative units, optionally retrieving evidence per unit."""
    config = config or ExperimentBConfig()
    start_total = perf_counter()
    judge = llm_provider or OllamaProvider(model=config.model, base_url=config.ollama_url,
                                           think=config.think, format=config.response_format)
    initial = evidence_from_retrieval_results(retrieval_results, max_items=config.max_evidence_items)
    units = analyze_verification_units(answer)
    claims = [Claim(unit.id, unit.original_text) for unit in units]
    verifications: list[ClaimVerification] = []
    trace: list[dict[str, Any]] = [{
        "stage": "pipeline_input", "question": question, "original_answer": answer,
        "initial_evidence": [asdict(item) for item in initial], "configuration": asdict(config),
    }]
    errors: list[str] = []
    call_counts = {"llm": 0, "retrieval": 0}
    retrieval_seconds = 0.0
    verification_seconds = 0.0

    for unit in units:
        unit_start = perf_counter()
        retrieval_query = build_unit_retrieval_query(question, unit)
        targeted: list[EvidenceItem] = []
        unit_trace: dict[str, Any] = {"unit": unit.to_dict(), "retrieval_queries": [], "retrieval_attempts": [],
                                     "judge_attempts": [], "validator_overrides": [], "retry_reasons": []}
        if retrieval_provider is not None and config.unit_specific_retrieval:
            began = perf_counter()
            call_counts["retrieval"] += 1
            unit_trace["retrieval_queries"].append(retrieval_query)
            try:
                response = retrieval_provider.retrieve(retrieval_query, limit=config.max_evidence_items)
                targeted = _normalise_retrieval_response(response, config.max_evidence_items)
                unit_trace["retrieval_attempts"].append({"query": retrieval_query,
                    "provider": getattr(response, "provider", getattr(retrieval_provider, "name", "unknown")),
                    "evidence": [asdict(x) for x in targeted], "retry": 0})
            except Exception as exc:
                errors.append(f"{unit.id} targeted retrieval: {type(exc).__name__}: {exc}")
                unit_trace["retry_reasons"].append("initial targeted retrieval failed: " + str(exc))
            retrieval_seconds += perf_counter() - began
        elif retrieval_provider is None:
            unit_trace["retry_reasons"].append("No retrieval provider supplied; used only the caller's initial evidence set.")
        else:
            unit_trace["retry_reasons"].append("Unit-specific retrieval disabled by the selected ablation configuration.")
        # Put claim-specific results first so a full initial RAG context cannot
        # crowd out evidence retrieved for this verification unit.
        evidence = _merge_evidence(targeted, initial, limit=config.max_evidence_items)

        if not evidence:
            verification = ClaimVerification(unit.id, unit.original_text, INSUFFICIENT_EVIDENCE, 1.0,
                                             explanation="No evidence was supplied or retrieved for this verification unit.")
            judge_meta = {"skipped": "empty evidence"}
        else:
            began = perf_counter()
            verification, judge_meta = _verify_unit(unit, question, evidence, judge, config, call_counts)
            verification_seconds += perf_counter() - began
            unit_trace["judge_attempts"].append(judge_meta)
            unit_trace["validator_overrides"].extend(verification.validator_overrides)
            if judge_meta.get("parse_error"):
                errors.append(f"{unit.id} semantic judge: {judge_meta['parse_error']}")

        new_retrieval_retries = 0
        while (verification.verdict == INSUFFICIENT_EVIDENCE and retrieval_provider is not None
               and config.unit_specific_retrieval
               and new_retrieval_retries < config.max_targeted_retrieval_retries):
            new_retrieval_retries += 1
            missing = verification.unsupported_components or unit.required_evidence or ["claim relationship and qualifiers"]
            focused_query = build_unit_retrieval_query(question, unit, missing_components=missing)
            if _normalise(focused_query) == _normalise(retrieval_query):
                focused_query = "Missing evidence detail: " + " ".join(missing) + " Context: " + question
            unit_trace["retry_reasons"].append("Initial verification was insufficient; retrieve the missing component(s): " + ", ".join(missing))
            began = perf_counter()
            call_counts["retrieval"] += 1
            unit_trace["retrieval_queries"].append(focused_query)
            additional: list[EvidenceItem] = []
            try:
                response = retrieval_provider.retrieve(focused_query, limit=config.max_evidence_items)
                additional = _normalise_retrieval_response(response, config.max_evidence_items)
                unit_trace["retrieval_attempts"].append({"query": focused_query,
                    "provider": getattr(response, "provider", getattr(retrieval_provider, "name", "unknown")),
                    "evidence": [asdict(x) for x in additional], "retry": new_retrieval_retries})
                if additional:
                    evidence = _merge_evidence(additional, evidence, limit=config.max_evidence_items)
            except Exception as exc:
                errors.append(f"{unit.id} targeted retrieval retry: {type(exc).__name__}: {exc}")
                unit_trace["retry_reasons"].append("targeted retrieval retry failed: " + str(exc))
            retrieval_seconds += perf_counter() - began
            began = perf_counter()
            if additional and evidence:
                verification, judge_meta = _verify_unit(unit, question, evidence, judge, config, call_counts)
                verification.retry_events.append({"stage": "targeted_retrieval", "attempt": new_retrieval_retries,
                                                  "reason": "insufficient evidence on prior pass"})
                verification_seconds += perf_counter() - began
                unit_trace["judge_attempts"].append(judge_meta)
                unit_trace["validator_overrides"].extend(verification.validator_overrides)
                if judge_meta.get("parse_error"):
                    errors.append(f"{unit.id} semantic judge retry: {judge_meta['parse_error']}")
            else:
                verification.retry_events.append({"stage": "targeted_retrieval", "attempt": new_retrieval_retries,
                                                  "reason": "retrieval returned no evidence; judge call skipped"})

        unit_trace["selected_evidence"] = {
            "supporting": [asdict(x) for x in verification.supporting_evidence],
            "contradicting": [asdict(x) for x in verification.contradicting_evidence],
        }
        unit_trace["llm_verdict"] = verification.llm_verdict
        unit_trace["final_verdict"] = verification.verdict
        unit_trace["validator_result"] = {"verdict": verification.verdict,
                                           "overrides": list(verification.validator_overrides),
                                           "assessment_complete": verification.assessment_complete,
                                           "missing_assessments": list(verification.missing_assessments)}
        unit_trace["reason"] = verification.explanation
        unit_trace["elapsed_seconds"] = perf_counter() - unit_start
        trace.append(unit_trace)
        verifications.append(verification)

    revised = revise_answer(answer, verifications) if config.answer_revision_enabled else answer
    total_seconds = perf_counter() - start_total
    report = VerificationReport(
        original_answer=answer, claims=claims, verifications=verifications,
        verification_units=[u.to_dict() for u in units], trace=trace, errors=errors,
        revised_answer=revised,
        metrics={"verification_accuracy": None, "macro_f1": None, "supported_f1": None,
                 "contradicted_f1": None, "insufficient_evidence_f1": None,
                 "evidence_precision": None, "evidence_recall": None, "evidence_coverage": None,
                 "entailment_correctness": None, "contradiction_correctness": None,
                 "citation_correctness": None, "citation_completeness": None, "evidence_alignment": None,
                 "supported_claim_rate": sum(v.verdict == SUPPORTED for v in verifications) / len(verifications) if verifications else None,
                 "unsupported_claim_rate": sum(v.verdict != SUPPORTED for v in verifications) / len(verifications) if verifications else None,
                 "insufficient_claim_rate": sum(v.verdict == INSUFFICIENT_EVIDENCE for v in verifications) / len(verifications) if verifications else None,
                 "contradiction_rate": sum(v.verdict == CONTRADICTED for v in verifications) / len(verifications) if verifications else None,
                 "total_latency_seconds": total_seconds, "retrieval_latency_seconds": retrieval_seconds,
                 "verification_latency_seconds": verification_seconds, "retrieval_calls": call_counts["retrieval"],
                 "llm_calls": call_counts["llm"], "tokens": None,
                 "note": "Gold-label metrics are None until an annotated evaluation dataset is supplied."},
    )
    return report


__all__ = ["ExperimentBConfig", "ValidationResult", "build_unit_retrieval_query",
           "validate_judgement", "revise_answer", "run_experiment_b_verification"]
