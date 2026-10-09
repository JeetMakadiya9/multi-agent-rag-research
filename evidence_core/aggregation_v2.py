"""Slice 1: conservative component-level aggregation (offline, deterministic, rule-based).

Research question: what changes when the legacy any-support rule is replaced by
conservative component-level semantics while every recorded judgement is held
fixed?

Scope decisions (fixed before results; see README "Slice 1"):
* Component-level only. The frozen judgements contain no slot-level coverage, so
  ``Claim.required_slots`` is diagnostic metadata (slot coverage NOT_ASSESSED);
  no word-presence or lexical slot checks are used (the legacy lexical guards
  are therefore NOT applied by this policy).
* INVALID contributes no stance; it never counts as support or contradiction.
  It is kept visible as JUDGEMENT_INCOMPLETE with a failure category derived
  from the recorded error string.
* Source independence is diagnostic in the primary policy. The separate
  ``IndependentCorroborationSensitivity`` requires >=2 independence groups for a
  supported component and is a sensitivity analysis only.
* First verification unit only (legacy comparability); an all-units diagnostic
  is available separately.
* No scores, weights, thresholds, retrieval signals or model calls.

Component states (from VALID judgements only):
    SUPPORTS only (+/- NEUTRAL)      -> SUPPORTED_COMPONENT
    CONTRADICTS only (+/- NEUTRAL)   -> CONTRADICTED_COMPONENT
    SUPPORTS and CONTRADICTS         -> MIXED_COMPONENT
    NEUTRAL only                     -> UNRESOLVED_COMPONENT
    no valid judgement (all INVALID) -> UNVERIFIED_COMPONENT
    no evidence presented            -> NO_EVIDENCE_COMPONENT
  (+ judgement_incomplete flag when any pair of the component is INVALID)

Claim state (first unit; existing VerdictState enum, no new states):
    any MIXED component                               -> MIXED_EVIDENCE
    else any CONTRADICTED component                   -> CONTRADICTED
    else every component SUPPORTED                    -> SUPPORTED
    else some SUPPORTED, rest UNRESOLVED/UNVERIFIED   -> PARTIALLY_SUPPORTED
    else every component UNVERIFIED                   -> UNVERIFIED
    else                                              -> INSUFFICIENT_EVIDENCE
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Sequence

from .contracts import Claim, ClaimVerdict, ComponentCoverage, Evidence, Judgement, Stance, VerdictState, stable_id
from .graph import EvidenceGraph

POLICY_VERSION = "slice1-1"

SUPPORTED_C = "SUPPORTED_COMPONENT"
CONTRADICTED_C = "CONTRADICTED_COMPONENT"
MIXED_C = "MIXED_COMPONENT"
UNRESOLVED_C = "UNRESOLVED_COMPONENT"
UNVERIFIED_C = "UNVERIFIED_COMPONENT"
NO_EVIDENCE_C = "NO_EVIDENCE_COMPONENT"
UNCORROBORATED_C = "SUPPORTED_UNCORROBORATED_COMPONENT"   # sensitivity policy only

# ------------------------------------------------------------------ failure taxonomy
_FAILURE_PATTERNS = (
    ("ENDPOINT_INFRASTRUCTURE", re.compile(r"^URLError: .*(WinError 10061|Connection refused|actively refused)", re.I)),
    ("TIMEOUT", re.compile(r"^(TimeoutError|socket\.timeout)\b", re.I)),
    ("MODEL_OUTPUT_SCHEMA", re.compile(
        r"^ValueError: (Legacy judgement verdict is invalid|Pair assessment must contain|Expected (exactly one|evidence index|component indices)"
        r"|No valid JSON object|Pairwise stance output must not|Legacy response did not|assessments must be|reason must be)")),
)


def classify_failure(error: Optional[str]) -> str:
    """Deterministic category for an INVALID judgement's recorded error; never guesses."""
    if not error:
        return "MISSING_RECORD"
    for category, pattern in _FAILURE_PATTERNS:
        if pattern.search(str(error)):
            return category
    return "UNKNOWN_FAILURE"


# ------------------------------------------------------------------ diagnostics
@dataclass(frozen=True)
class ComponentAssessment:
    component_claim_id: str
    legacy_component_id: Optional[str]
    state: str
    supporting_judgement_ids: tuple
    contradicting_judgement_ids: tuple
    neutral_judgement_ids: tuple
    invalid_judgement_ids: tuple
    invalid_categories: tuple
    judgement_incomplete: bool
    supporting_groups: tuple            # independence keys of supporting evidence (sorted, unique)
    contradicting_groups: tuple
    supporting_documents: tuple
    groups_with_internal_conflict: tuple
    single_group_support: bool


@dataclass(frozen=True)
class Decision:
    policy: str
    claim_id: str
    unit_claim_id: str
    state: VerdictState
    reason_codes: tuple
    components: tuple
    required_slots: tuple
    slot_coverage: str = "NOT_ASSESSED"
    verdict: Optional[ClaimVerdict] = None
    notes: tuple = field(default=())

    def to_dict(self) -> dict[str, Any]:
        return {"policy": self.policy, "claim_id": self.claim_id, "unit_claim_id": self.unit_claim_id,
                "state": self.state.value, "reason_codes": list(self.reason_codes),
                "components": [asdict(c) for c in self.components], "required_slots": list(self.required_slots),
                "slot_coverage": self.slot_coverage, "notes": list(self.notes)}


# ------------------------------------------------------------------ policies
class ConservativeComponentPolicy:
    """PRIMARY Slice 1 policy."""
    name = "ConservativeComponentPolicy"
    version = POLICY_VERSION
    min_support_groups = 1      # primary: independence is diagnostic only

    def assess_component(self, component: Claim, evidence: Sequence[Evidence], judgements: Sequence[Judgement],
                         graph: EvidenceGraph) -> ComponentAssessment:
        evidence_by_id = {e.evidence_id: e for e in evidence}
        own = sorted((j for j in judgements if j.claim_id == component.claim_id), key=lambda j: j.judgement_id)
        by = {s: tuple(j.judgement_id for j in own if j.stance is s) for s in Stance}
        group_of = {e.evidence_id: graph.independence_key(e) for e in evidence}
        sup_groups = tuple(sorted({group_of[j.evidence_id] for j in own if j.stance is Stance.SUPPORTS}))
        con_groups = tuple(sorted({group_of[j.evidence_id] for j in own if j.stance is Stance.CONTRADICTS}))
        sup_docs = tuple(sorted({evidence_by_id[j.evidence_id].document_version_id for j in own
                                 if j.stance is Stance.SUPPORTS}))
        categories = tuple(sorted(classify_failure((j.provenance.details or {}).get("final_error") if j.provenance else None)
                                  for j in own if j.stance is Stance.INVALID))
        valid = [j for j in own if j.stance is not Stance.INVALID]
        has_s, has_c = bool(by[Stance.SUPPORTS]), bool(by[Stance.CONTRADICTS])
        if not evidence:
            state = NO_EVIDENCE_C
        elif not valid:
            state = UNVERIFIED_C
        elif has_s and has_c:
            state = MIXED_C
        elif has_c:
            state = CONTRADICTED_C
        elif has_s:
            state = SUPPORTED_C if len(sup_groups) >= self.min_support_groups else UNCORROBORATED_C
        else:
            state = UNRESOLVED_C
        return ComponentAssessment(
            component_claim_id=component.claim_id,
            legacy_component_id=(component.metadata or {}).get("legacy_component_id"), state=state,
            supporting_judgement_ids=by[Stance.SUPPORTS], contradicting_judgement_ids=by[Stance.CONTRADICTS],
            neutral_judgement_ids=by[Stance.NEUTRAL], invalid_judgement_ids=by[Stance.INVALID],
            invalid_categories=categories, judgement_incomplete=bool(by[Stance.INVALID]),
            supporting_groups=sup_groups, contradicting_groups=con_groups, supporting_documents=sup_docs,
            groups_with_internal_conflict=tuple(sorted(set(sup_groups) & set(con_groups))),
            single_group_support=len(sup_groups) == 1)

    def aggregate_unit(self, root: Claim, unit: Claim, components: Sequence[Claim], evidence: Sequence[Evidence],
                       judgements: Sequence[Judgement], graph: EvidenceGraph, *, run_key: str = "") -> Decision:
        comps = tuple(self.assess_component(c, evidence, judgements, graph) for c in components)
        states = [c.state for c in comps]
        supported_like = {SUPPORTED_C}
        unsettled = {UNRESOLVED_C, UNVERIFIED_C, UNCORROBORATED_C, NO_EVIDENCE_C}
        codes: list[str] = []
        if any(s == MIXED_C for s in states):
            state = VerdictState.MIXED_EVIDENCE; codes.append("MIXED_SUPPORT_CONTRADICTION")
        elif any(s == CONTRADICTED_C for s in states):
            state = VerdictState.CONTRADICTED; codes.append("CONTRADICTION_ON_COMPONENT")
        elif states and all(s in supported_like for s in states):
            state = VerdictState.SUPPORTED; codes.append("ALL_COMPONENTS_SUPPORTED")
        elif any(s in supported_like for s in states) and all(s in supported_like | unsettled for s in states):
            state = VerdictState.PARTIALLY_SUPPORTED; codes.append("PARTIAL_COMPONENT_COVERAGE")
        elif states and all(s == UNVERIFIED_C for s in states):
            state = VerdictState.UNVERIFIED; codes.append("NO_VALID_JUDGEMENT")
        else:
            state = VerdictState.INSUFFICIENT_EVIDENCE
            if any(s == NO_EVIDENCE_C for s in states):
                codes.append("NO_EVIDENCE_PRESENTED")
            elif any(s == UNCORROBORATED_C for s in states):
                codes.append("SUPPORT_NOT_CORROBORATED")
            else:
                codes.append("NO_SUPPORT_OR_CONTRADICTION")
        if any(s == UNCORROBORATED_C for s in states):
            codes.append("UNCORROBORATED_SUPPORT")
        if any(c.judgement_incomplete for c in comps):
            codes.append("INVALID_JUDGEMENT_PRESENT")
        if any(c.state in (SUPPORTED_C, UNCORROBORATED_C) and c.single_group_support for c in comps):
            codes.append("SINGLE_INDEPENDENCE_GROUP_SUPPORT")
        if any(c.groups_with_internal_conflict for c in comps):
            codes.append("CONFLICT_WITHIN_INDEPENDENCE_GROUP")
        if unit.required_slots:
            codes.append("SLOT_COVERAGE_NOT_ASSESSED")
        judgement_ids = tuple(sorted(j.judgement_id for j in judgements
                                     if j.claim_id in {c.claim_id for c in components}))
        verdict = ClaimVerdict(
            verdict_id=stable_id("vrd", root.claim_id, unit.claim_id, self.name, self.version, run_key),
            claim_id=root.claim_id, policy=self.name, policy_version=self.version, judgement_ids=judgement_ids,
            state=state, legacy=None,
            component_coverage=tuple(ComponentCoverage(
                component_claim_id=c.component_claim_id, status=c.state,
                supporting_evidence_ids=tuple(sorted({j.evidence_id for j in judgements if j.judgement_id in c.supporting_judgement_ids})),
                contradicting_evidence_ids=tuple(sorted({j.evidence_id for j in judgements if j.judgement_id in c.contradicting_judgement_ids})))
                for c in comps),
            notes=tuple(codes) + ("slot coverage NOT_ASSESSED (no slot-level judgements recorded)",))
        return Decision(self.name, root.claim_id, unit.claim_id, state, tuple(codes), comps, tuple(unit.required_slots),
                        verdict=verdict)

    def decide(self, root: Claim, units: Sequence[Claim], components: Sequence[Sequence[Claim]],
               evidence: Sequence[Evidence], judgements: Sequence[Judgement], graph: EvidenceGraph, *,
               run_key: str = "", all_units: bool = False) -> Decision:
        """First unit only (legacy comparability) unless ``all_units`` (diagnostic only)."""
        if not units:
            raise ValueError("claim has no verification units")
        first = self.aggregate_unit(root, units[0], components[0], evidence, judgements, graph, run_key=run_key)
        if not all_units or len(units) == 1:
            return first if len(units) == 1 else Decision(**{**first.__dict__, "notes": ("FIRST_UNIT_ONLY",)})
        flat = [c for group in components for c in group]
        merged_unit = units[0]
        combined = self.aggregate_unit(root, merged_unit, flat, evidence, judgements, graph, run_key=run_key + ":all-units")
        return Decision(**{**combined.__dict__, "notes": ("ALL_UNITS_DIAGNOSTIC",)})


class IndependentCorroborationSensitivity(ConservativeComponentPolicy):
    """SENSITIVITY ANALYSIS — not selected as primary policy.

    Identical to the primary policy except that a component counts as supported
    only when its SUPPORTS judgements come from >= 2 independence groups.
    Contradiction is not subject to corroboration. The threshold is fixed at 2
    (the minimal meaning of "independent corroboration"); it is never tuned.
    """
    name = "IndependentCorroborationSensitivity"
    min_support_groups = 2


def state_counts(decisions: Sequence[Decision]) -> dict[str, int]:
    return dict(sorted(Counter(d.state.value for d in decisions).items()))
