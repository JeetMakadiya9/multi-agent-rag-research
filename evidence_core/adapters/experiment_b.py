"""Adapt recorded Experiment B outputs into evidence-core contracts (read-only, no LLM calls).

Input: one recorded ``run_system_b`` output record, either a frozen Phase 10F
per-example row (fields under ``output``) or a Phase 13 attempt row (fields at
top level). The recorded Experiment B trace is the source of truth:

* trace[0] (``pipeline_input``)  -> the exact evidence texts the judge saw
* trace[1:] (one per unit)       -> the recorded VerificationUnit and every
                                    pairwise judge call (raw responses, parse
                                    errors, final assessment)

Mapping:
* claim text            -> root Claim (origin BENCHMARK)
* each recorded unit    -> child Claim (LEGACY_SENTENCE_UNIT); the recorded unit
                           dict is kept verbatim in metadata["legacy_unit"]
* each unit component   -> grandchild Claim (LEGACY_COMPONENT)
* retrieved chunks      -> RetrievedCandidates (adapters.retrieval)
* judged top-N chunks   -> Evidence admitted for the root claim with role
                           LEGACY_TOP_N_PRESENTED_TO_JUDGE (positional admission,
                           explicitly not an evidentiary judgement)
* each pair call        -> Judgement; a pair with no valid final assessment
                           becomes stance INVALID (never NEUTRAL, never SUPPORTS)

Anything the recorded data cannot determine unambiguously raises
``NotReplayable`` / ``AmbiguousRecord`` instead of being guessed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Tuple

from verification_units import analyze_verification_units

from ..contracts import (
    AdmissionRecord, Claim, ClaimOrigin, Evidence, EvidenceRole, JudgeIdentity, Judgement, LegacyRunStatus,
    LegacyVerdict, Provenance, Stance,
)
from .retrieval import AdaptedRetrieval, DocumentLookup, adapt_recorded_refs

ADAPTER = "evidence_core.adapters.experiment_b"
ADAPTER_VERSION = "slice0-1"
INCOMPLETE_PREFIX = "Experiment B returned an incomplete assessment"


class NotReplayable(Exception):
    """The record contains no recorded judgements (e.g. an interrupted attempt)."""


class AmbiguousRecord(Exception):
    """The record is internally inconsistent; replay must stop rather than guess."""


@dataclass(frozen=True)
class RecordedLegacyOutcome:
    """What the frozen pipeline recorded — used only for comparison, never as input."""
    run_status: LegacyRunStatus
    verdict: Optional[LegacyVerdict]
    unit_verdicts: Tuple[str, ...]
    unit_coverage: Tuple[Tuple[Mapping[str, Any], ...], ...]


@dataclass(frozen=True)
class AdaptedExperimentB:
    root_claim: Claim
    unit_claims: Tuple[Claim, ...]
    component_claims: Tuple[Tuple[Claim, ...], ...]      # per unit, in recorded component order
    evidence: Tuple[Evidence, ...]                        # in presentation order (E1..En)
    judgements: Tuple[Judgement, ...]
    retrieval: AdaptedRetrieval
    recorded: RecordedLegacyOutcome
    unit_skipped_empty_evidence: Tuple[bool, ...] = field(default=())

    def objects(self):
        comps = [c for group in self.component_claims for c in group]
        return (*self.retrieval.objects(), self.root_claim, *self.unit_claims, *comps, *self.evidence, *self.judgements)


def _fields(record: Mapping[str, Any]) -> Mapping[str, Any]:
    return record["output"] if "output" in record and isinstance(record["output"], Mapping) else record


def recorded_outcome(record: Mapping[str, Any]) -> tuple[LegacyRunStatus, Optional[LegacyVerdict]]:
    status = record.get("status")
    if status == "SUCCESS":
        return LegacyRunStatus.VERDICT, LegacyVerdict(record["predicted_label"] if "predicted_label" in record
                                                      else _fields(record)["predicted_label"])
    error = str(record.get("error") or _fields(record).get("error") or "")
    if status == "FAILED" and error.startswith(INCOMPLETE_PREFIX):
        return LegacyRunStatus.FAILED_INCOMPLETE_ASSESSMENT, None
    raise NotReplayable(f"status={status!r}; error={error[:120]!r}: no recorded Experiment B judgement outcome")


def _slots(unit: Mapping[str, Any]) -> dict[str, Any]:
    required = []
    for name, present in (("object", len(unit.get("objects") or []) > 0), ("quantity", bool(unit.get("quantitative_constraints"))),
                          ("time", bool(unit.get("temporal_constraints"))), ("condition", bool(unit.get("conditions"))),
                          ("polarity", bool(unit.get("negation"))), ("comparison", bool(unit.get("comparison_relation"))),
                          ("causal_type", bool(unit.get("causal_relation")))):
        if present:
            required.append(name)
    objects = unit.get("objects") or []
    return dict(subject=unit.get("subject"), predicate=unit.get("predicate"),
                object=objects[0] if len(objects) == 1 else None,
                comparison=unit.get("comparison_relation"), quantity=tuple(unit.get("quantitative_constraints") or ()),
                time=tuple(unit.get("temporal_constraints") or ()), condition=tuple(unit.get("conditions") or ()),
                polarity="negated" if unit.get("negation") else None,
                causal_type="causal" if unit.get("causal_relation") else None, required_slots=tuple(required))


def _rationale(attempts: Sequence[Mapping[str, Any]]) -> Optional[str]:
    for attempt in reversed(list(attempts)):
        if attempt.get("parse_error"):
            continue
        try:
            reason = json.loads(attempt.get("raw_response") or "").get("reason")
            return str(reason) if reason is not None else None
        except (ValueError, AttributeError):
            return None
    return None


def adapt_experiment_b_record(record: Mapping[str, Any], *, claim_key: str, run_key: str, judge: JudgeIdentity,
                              lookup: Optional[DocumentLookup] = None) -> AdaptedExperimentB:
    out = _fields(record)
    run_status, verdict = recorded_outcome(record)
    raw = out.get("raw_output")
    trace = raw.get("trace") if isinstance(raw, Mapping) else None
    if not trace:
        raise NotReplayable("record has no recorded Experiment B trace")
    claim_text = record["runtime_input"]["claim"]
    pipeline = trace[0]
    if pipeline.get("stage") != "pipeline_input" or pipeline.get("original_answer") != claim_text:
        raise AmbiguousRecord("trace[0] is not the pipeline input for this claim")
    unit_traces = [t for t in trace[1:] if "unit" in t]
    if len(unit_traces) != len(trace) - 1:
        raise AmbiguousRecord("unexpected non-unit trace stage")

    retrieval = adapt_recorded_refs(out.get("retrieved_chunks") or [], query=claim_text, run_key=run_key, lookup=lookup)
    cand_by_rank = {c.retrieval.rank: c for c in retrieval.candidates}

    root = Claim(claim_id=Claim.make_id("benchmark", claim_key), text=claim_text, origin=ClaimOrigin.BENCHMARK,
                 metadata={"claim_key": claim_key})

    # Evidence = exactly the texts presented to the judge, in presentation order.
    evidence = []
    for position, item in enumerate(pipeline.get("initial_evidence") or [], start=1):
        index = f"E{position}"
        if item.get("evidence_id") != index:
            raise AmbiguousRecord(f"initial evidence {position} has id {item.get('evidence_id')!r}, expected {index}")
        cand = cand_by_rank.get(position)
        if cand is None or cand.metadata.get("chunk_id") != str(item.get("chunk_id")):
            raise AmbiguousRecord(f"{index} does not correspond to retrieved rank {position}")
        transformed = item["text"] != cand.presented_text
        evidence.append(Evidence(
            evidence_id=Evidence.make_id(root.claim_id, cand.span_id, f"{run_key}:{index}"),
            claim_id=root.claim_id, span_id=cand.span_id, document_version_id=cand.document_version_id,
            source_id=cand.source_id, text=item["text"], retrieval=cand.retrieval,
            admission=AdmissionRecord(
                role=EvidenceRole.LEGACY_TOP_N_PRESENTED_TO_JUDGE, admitted_by="frozen run_system_b",
                reason=(f"retrieval ranks 1..{len(pipeline['initial_evidence'])} were presented to the judge "
                        f"(max_evidence_items={pipeline['configuration']['max_evidence_items']}); positional "
                        "admission, not an evidentiary assessment"),
                presentation_index=index, candidate_id=cand.candidate_id),
            extraction=Provenance(producer=ADAPTER, producer_version=ADAPTER_VERSION,
                                  details={"judged_text_differs_from_retrieved_text": transformed})))
    ev_by_index = {e.admission.presentation_index: e for e in evidence}

    unit_claims, component_claims, judgements, skipped = [], [], [], []
    unit_verdicts, unit_coverage = [], []
    reanalysed = analyze_verification_units(claim_text)
    if len(reanalysed) != len(unit_traces):
        raise AmbiguousRecord("recorded unit count differs from deterministic unit analysis")
    for position, unit_trace in enumerate(unit_traces):
        unit = unit_trace["unit"]
        if reanalysed[position].to_dict() != unit:
            raise AmbiguousRecord(f"recorded unit {unit.get('id')} differs from deterministic unit analysis")
        unit_claim = Claim(claim_id=Claim.make_id(root.claim_id, "unit", unit["id"]), text=unit["original_text"],
                           origin=ClaimOrigin.LEGACY_SENTENCE_UNIT, parent_claim_id=root.claim_id,
                           decomposition_reason="legacy sentence split (verification_units)", semantic_unity=True,
                           metadata={"legacy_unit": unit, "legacy_unit_id": unit["id"]}, **_slots(unit))
        comps = tuple(Claim(
            claim_id=Claim.make_id(unit_claim.claim_id, "component", comp["component_id"]), text=comp["text"],
            origin=ClaimOrigin.LEGACY_COMPONENT, parent_claim_id=unit_claim.claim_id,
            decomposition_reason=f"legacy component ({comp['role']})", subject=comp.get("subject"),
            predicate=comp.get("relation"), object=comp.get("object"), condition=tuple(filter(None, [comp.get("condition")])),
            semantic_unity=True, metadata={"legacy_component_id": comp["component_id"], "legacy_component": comp})
            for comp in unit["components"])
        comp_by_index = {c.metadata["legacy_component_id"]: c for c in comps}
        unit_claims.append(unit_claim)
        component_claims.append(comps)

        attempts = unit_trace.get("judge_attempts") or []
        if len(attempts) > 1:
            raise AmbiguousRecord("multiple judge passes per unit (targeted retrieval) are not part of the recorded protocol")
        meta = attempts[0] if attempts else {}
        if meta.get("skipped") == "empty evidence":
            if evidence:
                raise AmbiguousRecord("unit skipped for empty evidence although evidence was presented")
            skipped.append(True)
            unit_coverage.append(())
        else:
            if "pair_assessment_calls" not in meta:
                raise AmbiguousRecord("unit trace has no pair-level judge record")
            skipped.append(False)
            seen = set()
            for call in meta["pair_assessment_calls"]:
                ev, comp = ev_by_index.get(call["evidence_index"]), comp_by_index.get(call["component_index"])
                if ev is None or comp is None:
                    raise AmbiguousRecord(f"pair {call['pair_id']} references unknown evidence or component")
                if (ev.evidence_id, comp.claim_id) in seen:
                    raise AmbiguousRecord(f"duplicate pair record {call['pair_id']}")
                seen.add((ev.evidence_id, comp.claim_id))
                assessment = call.get("assessment")
                stance = Stance(str(assessment["stance"]).upper()) if assessment else Stance.INVALID
                judgements.append(Judgement(
                    judgement_id=Judgement.make_id(comp.claim_id, ev.evidence_id, judge, run_key), claim_id=comp.claim_id,
                    evidence_id=ev.evidence_id, stance=stance, judge=judge, rationale=_rationale(call["attempts"]),
                    provenance=Provenance(producer=ADAPTER, producer_version=ADAPTER_VERSION, input_reference=run_key,
                                          details={"pair_id": call["pair_id"], "attempts": call["attempts"],
                                                   "final_error": call.get("error"),
                                                   "recorded_assessment": assessment})))
            expected_pairs = {(e.evidence_id, c.claim_id) for e in evidence for c in comps}
            if seen != expected_pairs:
                raise AmbiguousRecord("recorded pair calls do not cover exactly evidence x components")
            unit_coverage.append(tuple(meta.get("coverage_matrix") or ()))
        unit_verdicts.append(unit_trace["final_verdict"])

    recorded = RecordedLegacyOutcome(run_status=run_status, verdict=verdict, unit_verdicts=tuple(unit_verdicts),
                                     unit_coverage=tuple(unit_coverage))
    return AdaptedExperimentB(root, tuple(unit_claims), tuple(component_claims), tuple(evidence), tuple(judgements),
                              retrieval, recorded, tuple(skipped))
