"""Aggregation policies: Judgements -> ClaimVerdict.

Slice 0 contains exactly one implemented policy, ``LegacyAnySupportsPolicy``,
whose sole purpose is historical reproducibility of the frozen Experiment B
verdicts. It is deliberately NOT improved. ``AggregationPolicy`` is the
interface Slice 1 policies will implement (INTERFACE ONLY here).

Legacy semantics (frozen ``experiment_b_verification.validate_judgement`` +
``run_experiment_b_verification`` + ``system_b_verified_rag.run_system_b``):

Per verification unit, over evidence E1..En x components C1..Cm:
  1. a pair whose judge produced no valid stance (INVALID) is a missing
     assessment; any missing assessment makes the unit INCOMPLETE;
  2. per component: support and refute sets from SUPPORTS / CONTRADICTS stances;
     status = INCOMPLETE (missing pair) > CONFLICTING (both) > CONTRADICTED >
     SUPPORTED > UNSUPPORTED;
  3. errors = missing-assessment message + lexical guard gaps (frozen
     ``_hard_qualifier_gaps`` and ``_coordinated_component_gaps`` over the
     presented evidence texts — delegated to the frozen functions);
  4. verdict ladder: incomplete -> INSUFFICIENT; any conflicting component ->
     INSUFFICIENT; any contradicted component -> CONTRADICTED; any error ->
     INSUFFICIENT; every component supported -> SUPPORTED; else INSUFFICIENT.
     ("any SUPPORTS and no contradiction -> SUPPORTED" per component.)
  5. no presented evidence -> INSUFFICIENT with no judge calls.

Per claim (run_system_b):
  6. any INVALID judgement in ANY unit (recorded pair error) or an incomplete
     first unit -> the attempt FAILED (no verdict);
  7. otherwise the verdict is the FIRST unit's verdict (later units ignored).

Experiment B never resolves conflicts (the frozen ``_resolve_conflict`` is
never called), so neither does this policy.
"""
from __future__ import annotations

from typing import Optional, Protocol, Sequence

import experiment_b_verification as frozen_b
from claim_verification import EvidenceItem
from verification_units import VerificationComponent, VerificationUnit

from .contracts import (
    Claim, ClaimVerdict, ComponentCoverage, Evidence, Judgement, LegacyOutcome, LegacyRunStatus, LegacyVerdict,
    Stance, stable_id,
)


class AggregationPolicy(Protocol):
    """INTERFACE ONLY in Slice 0. Implementations map judgements to a ClaimVerdict."""
    name: str
    version: str

    def aggregate(self, claim: Claim, components: Sequence[Claim], evidence: Sequence[Evidence],
                  judgements: Sequence[Judgement], *, run_key: str = "") -> ClaimVerdict: ...


def legacy_unit_from_claim(unit_claim: Claim) -> VerificationUnit:
    """Rebuild the recorded frozen VerificationUnit (needed only by the frozen lexical guards)."""
    data = dict(unit_claim.metadata["legacy_unit"])
    components = [VerificationComponent(**dict(c)) for c in data.pop("components")]
    return VerificationUnit(**data, components=components)


class LegacyAnySupportsPolicy:
    name = "LegacyAnySupportsPolicy"
    version = "experiment_b_frozen_66EA7B6B"   # frozen experiment_b_verification.py hash prefix

    # ------------------------------------------------------------------ unit level
    def aggregate_unit(self, unit_claim: Claim, components: Sequence[Claim], evidence: Sequence[Evidence],
                       judgements: Sequence[Judgement], *, run_key: str = "") -> ClaimVerdict:
        component_ids = {c.claim_id for c in components}
        evidence_ids = {e.evidence_id for e in evidence}
        index_of = {e.evidence_id: e.admission.presentation_index for e in evidence}
        if not evidence:
            return self._verdict(unit_claim, LegacyVerdict.INSUFFICIENT_EVIDENCE, (), (), run_key,
                                 ("no evidence presented; judge not called",), complete=True)
        final: dict[tuple[str, str], Judgement] = {}
        for j in judgements:
            if j.claim_id not in component_ids or j.evidence_id not in evidence_ids:
                raise ValueError(f"judgement {j.judgement_id} is outside this unit's evidence x components")
            key = (j.evidence_id, j.claim_id)
            if key in final:
                raise ValueError(f"duplicate judgement for pair {key}")
            final[key] = j
        missing = [(e, c) for e in evidence for c in components
                   if final.get((e.evidence_id, c.claim_id)) is None
                   or final[(e.evidence_id, c.claim_id)].stance is Stance.INVALID]
        complete = not missing
        errors = []
        if missing:
            labels = sorted(f"{index_of[e.evidence_id]}-{c.metadata.get('legacy_component_id', c.claim_id)}"
                            for e, c in missing)
            errors.append("Missing assessment(s): " + ", ".join(labels))

        coverage, supported, conflicting, contradicted = [], [], [], []
        for comp in components:
            stances = [final[(e.evidence_id, comp.claim_id)] for e in evidence
                       if (e.evidence_id, comp.claim_id) in final]
            # The frozen validator orders indices as sorted strings ("E1" < "E2" ...).
            support = sorted((index_of[j.evidence_id] for j in stances if j.stance is Stance.SUPPORTS))
            refute = sorted((index_of[j.evidence_id] for j in stances if j.stance is Stance.CONTRADICTS))
            incomplete = any(c.claim_id == comp.claim_id for _, c in missing)
            if incomplete:
                status = "INCOMPLETE"
            elif support and refute:
                status = "CONFLICTING"; conflicting.append(comp)
            elif refute:
                status = "CONTRADICTED"; contradicted.append(comp)
            elif support:
                status = "SUPPORTED"; supported.append(comp)
            else:
                status = "UNSUPPORTED"
            id_of = {e.admission.presentation_index: e.evidence_id for e in evidence}
            coverage.append(ComponentCoverage(component_claim_id=comp.claim_id, status=status,
                                              supporting_evidence_ids=tuple(id_of[i] for i in support),
                                              contradicting_evidence_ids=tuple(id_of[i] for i in refute)))

        unit = legacy_unit_from_claim(unit_claim)
        items = [EvidenceItem(e.admission.presentation_index, e.text) for e in evidence]
        gaps = frozen_b._hard_qualifier_gaps(unit, items) + frozen_b._coordinated_component_gaps(unit, items)
        errors += ["Uncovered required detail: " + gap for gap in gaps]

        all_supported = len(supported) == len(components) and not conflicting and not errors
        if not complete:
            verdict = LegacyVerdict.INSUFFICIENT_EVIDENCE
        elif conflicting:
            verdict = LegacyVerdict.INSUFFICIENT_EVIDENCE
        elif contradicted:
            verdict = LegacyVerdict.CONTRADICTED
        elif errors:
            verdict = LegacyVerdict.INSUFFICIENT_EVIDENCE
        elif all_supported:
            verdict = LegacyVerdict.SUPPORTED
        else:
            verdict = LegacyVerdict.INSUFFICIENT_EVIDENCE
        return self._verdict(unit_claim, verdict, tuple(coverage), tuple(j.judgement_id for j in judgements), run_key,
                             tuple(errors), complete=complete)

    # ------------------------------------------------------------------ claim level
    def aggregate_claim(self, root: Claim, unit_verdicts: Sequence[ClaimVerdict],
                        unit_judgements: Sequence[Sequence[Judgement]], *, run_key: str = "") -> ClaimVerdict:
        if not unit_verdicts:
            raise ValueError("a claim with no verification units has no legacy verdict")
        any_invalid = any(j.stance is Stance.INVALID for group in unit_judgements for j in group)
        first = unit_verdicts[0]
        first_complete = "incomplete" not in first.notes
        ids = tuple(jid for v in unit_verdicts for jid in v.judgement_ids)
        if any_invalid or not first_complete:
            outcome = LegacyOutcome(LegacyRunStatus.FAILED_INCOMPLETE_ASSESSMENT, None,
                                    ("at least one pair has no valid recorded stance",))
        else:
            outcome = LegacyOutcome(LegacyRunStatus.VERDICT, first.legacy.verdict,
                                    ("verdict of the first verification unit (legacy behaviour)",))
        return ClaimVerdict(verdict_id=stable_id("vrd", root.claim_id, self.name, self.version, run_key),
                            claim_id=root.claim_id, policy=self.name, policy_version=self.version, judgement_ids=ids,
                            state=None, legacy=outcome, component_coverage=first.component_coverage,
                            notes=("rich VerdictState intentionally not inferred by the legacy policy",))

    def _verdict(self, claim: Claim, verdict: LegacyVerdict, coverage, judgement_ids, run_key, diagnostics, *,
                 complete: bool) -> ClaimVerdict:
        return ClaimVerdict(verdict_id=stable_id("vrd", claim.claim_id, self.name, self.version, run_key),
                            claim_id=claim.claim_id, policy=self.name, policy_version=self.version,
                            judgement_ids=judgement_ids, state=None,
                            legacy=LegacyOutcome(LegacyRunStatus.VERDICT, verdict, diagnostics),
                            component_coverage=coverage,
                            notes=("complete",) if complete else ("incomplete",))
