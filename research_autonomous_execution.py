"""Single-proposal bridge from 8A decisions into the controlled Phase 5–7 APIs.

This module validates a proposal against current state, requires explicit
proposal-scoped researcher authorization, and delegates at most one action.
It contains no autonomous decision loop and does not implement capabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from collections.abc import Mapping
from typing import Optional
from uuid import uuid4

from research_autonomy import (
    ResearchActionType, ResearchDecision, ResearchDecisionTrace, ResearcherOverride,
    ResearcherOverrideAction, ValidationStatus, validate_research_decision,
)
from research_bounded_loop import BoundedExecutionPolicy
from research_capabilities import CapabilityType, LocalRetrievalRequest
from research_capability_dispatch import (
    CapabilityRequest, DispatchStatus, ResearchCapabilityDispatchResult,
    dispatch_research_capability,
)
from research_continuation import ResearchContinuationAction, apply_research_continuation
from research_experiment_planning_capability import ExperimentPlanningRequest
from research_improvement_capability import ImprovementGenerationRequest
from research_loop import LoopStage, ResearchLoop, ResearchLoopStatus
from research_plan_review import PlanReview
from research_planning import ResearchPlan, ResearchTask
from research_prior_work_capability import PriorWorkInvestigationRequest
from research_state import DocumentVersion, PriorWorkSearchScope, ResearchRun, ResearchState
from research_synthesis_critique import (
    ResearchCritique, ResearchSynthesis, critique_research_synthesis,
    synthesize_research_state,
)
from research_verification_capability import EvidenceVerificationRequest


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:16]}"


def _enum(enum_type, value, label):
    try:
        return value if isinstance(value, enum_type) else enum_type(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unsupported {label}: {value!r}.") from exc


def _nonempty(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string.")
    return value


def _timestamp(value, label="timestamp"):
    _nonempty(value, label)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be timezone-aware ISO-8601.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware ISO-8601.")


def _jsonable(value):
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if is_dataclass(value):
        return {item.name: _jsonable(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _digest(value) -> str:
    payload = json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ApprovalStatus(str, Enum):
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    REVOKED = "REVOKED"
    SUPERSEDED = "SUPERSEDED"


@dataclass(frozen=True)
class AutonomousExecutionApproval:
    """Human authorization bound to one decision and its exact execution scope."""

    approval_id: str
    decision_id: str
    run_id: str
    loop_id: str
    iteration_id: str
    plan_id: str
    plan_revision: int
    action_type: ResearchActionType
    task_id: Optional[str]
    scope_digest: str
    reviewer: str
    status: ApprovalStatus = ApprovalStatus.APPROVED
    created_at: str = ""

    def __post_init__(self):
        for name in ("approval_id", "decision_id", "run_id", "loop_id", "iteration_id", "plan_id", "scope_digest", "reviewer"):
            _nonempty(getattr(self, name), name)
        if not isinstance(self.plan_revision, int) or isinstance(self.plan_revision, bool) or self.plan_revision < 1:
            raise ValueError("plan_revision must be a positive integer.")
        object.__setattr__(self, "action_type", _enum(ResearchActionType, self.action_type, "action_type"))
        object.__setattr__(self, "status", _enum(ApprovalStatus, self.status, "approval status"))
        if self.task_id is not None:
            _nonempty(self.task_id, "task_id")
        if not self.created_at:
            object.__setattr__(self, "created_at", _now())
        _timestamp(self.created_at, "created_at")

    @classmethod
    def for_request(cls, request: "AutonomousExecutionRequest", *, reviewer: str,
                    status: ApprovalStatus = ApprovalStatus.APPROVED, approval_id: Optional[str] = None,
                    created_at: Optional[str] = None) -> "AutonomousExecutionApproval":
        """Create a scope-bound approval from the exact immutable request."""
        decision = request.decision
        proposal = decision.proposal
        return cls(
            approval_id or _id("approval"), decision.trace.decision_id,
            request.run_id, request.loop_id, request.iteration_id, request.plan_id,
            request.plan_revision, proposal.action_type, proposal.task_id,
            request.scope_digest, reviewer, status, created_at or _now(),
        )

    def to_dict(self):
        return {
            "approval_id": self.approval_id, "decision_id": self.decision_id,
            "run_id": self.run_id, "loop_id": self.loop_id, "iteration_id": self.iteration_id,
            "plan_id": self.plan_id, "plan_revision": self.plan_revision,
            "action_type": self.action_type.value, "task_id": self.task_id,
            "scope_digest": self.scope_digest, "reviewer": self.reviewer,
            "status": self.status.value, "created_at": self.created_at,
        }

    def to_json(self):
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, data):
        expected = {"approval_id", "decision_id", "run_id", "loop_id", "iteration_id", "plan_id",
                    "plan_revision", "action_type", "task_id", "scope_digest", "reviewer", "status", "created_at"}
        if not isinstance(data, dict) or set(data) != expected:
            raise ValueError("Malformed AutonomousExecutionApproval payload.")
        try:
            return cls(**data)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Malformed AutonomousExecutionApproval: {exc}") from exc

    @classmethod
    def from_json(cls, payload):
        try:
            return cls.from_dict(json.loads(payload))
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Malformed AutonomousExecutionApproval JSON: {exc}") from exc


@dataclass(frozen=True)
class AutonomousExecutionRequest:
    """One typed proposal, its prior 8A validation, and the controlled request."""

    decision: ResearchDecision
    prior_validation: ResearchDecisionTrace
    run_id: str
    loop_id: str
    iteration_id: str
    plan_id: str
    plan_revision: int
    policy_digest: str
    capability_request: Optional[CapabilityRequest] = None
    synthesis: Optional[ResearchSynthesis] = None
    researcher_override: Optional[ResearcherOverride] = None
    request_id: str = ""
    retry_of_execution_id: Optional[str] = None
    continuation_authorization_id: Optional[str] = None
    provider_refs: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.decision, ResearchDecision) or not isinstance(self.prior_validation, ResearchDecisionTrace):
            raise ValueError("Typed ResearchDecision and prior 8A validation trace are required.")
        for name in ("run_id", "loop_id", "iteration_id", "plan_id"):
            _nonempty(getattr(self, name), name)
        if not isinstance(self.plan_revision, int) or isinstance(self.plan_revision, bool) or self.plan_revision < 1:
            raise ValueError("plan_revision must be a positive integer.")
        _nonempty(self.policy_digest, "policy_digest")
        if self.prior_validation.validation_status is not ValidationStatus.ACCEPTED:
            raise ValueError("8B requires a previously accepted 8A validation trace.")
        if self.prior_validation.loop_id != self.loop_id or self.prior_validation.run_id != self.run_id:
            raise ValueError("Prior validation must refer to this run and loop.")
        if self.decision.trace.decision_id != self.prior_validation.decision_id:
            raise ValueError("Prior validation must refer to this exact decision.")
        for name in ("context_id", "loop_id", "run_id", "iteration_id", "action_type", "rationale",
                     "provider", "model", "created_at", "provenance_ids"):
            if getattr(self.prior_validation, name) != getattr(self.decision.trace, name):
                raise ValueError(f"Prior validation field {name} does not match the original decision trace.")
        if self.researcher_override is not None:
            if not isinstance(self.researcher_override, ResearcherOverride):
                raise ValueError("researcher_override must be typed.")
            if self.researcher_override.decision_id != self.decision.trace.decision_id:
                raise ValueError("ResearcherOverride refers to another decision.")
        if self.synthesis is not None and not isinstance(self.synthesis, ResearchSynthesis):
            raise ValueError("synthesis must be a typed ResearchSynthesis.")
        if not self.request_id:
            object.__setattr__(self, "request_id", _id("autonomous_request"))
        _nonempty(self.request_id, "request_id")
        for name in ("retry_of_execution_id", "continuation_authorization_id"):
            value = getattr(self, name)
            if value is not None:
                _nonempty(value, name)
        if self.retry_of_execution_id is not None and self.continuation_authorization_id is not None:
            raise ValueError("A proposal execution cannot be both an explicit retry and continuation.")
        refs = tuple(self.provider_refs)
        if any(not isinstance(value, str) or not value.strip() for value in refs) or len(refs) != len(set(refs)):
            raise ValueError("provider_refs must contain unique non-empty provider references.")
        object.__setattr__(self, "provider_refs", refs)
        action = self.decision.proposal.action_type
        is_capability = action in _ACTION_CAPABILITY
        if is_capability != (self.capability_request is not None):
            raise ValueError("Capability actions require exactly one typed capability_request; other actions reject it.")
        if is_capability and not isinstance(self.capability_request, _ACTION_REQUEST_CLASSES[action]):
            raise ValueError(f"{action.value} requires a typed {_ACTION_REQUEST_CLASSES[action].__name__}.")
        if action is ResearchActionType.CRITIQUE and self.synthesis is None:
            raise ValueError("CRITIQUE requires the typed synthesis to critique.")
        if action is not ResearchActionType.CRITIQUE and self.synthesis is not None:
            raise ValueError("Only CRITIQUE may include a synthesis input.")

    @property
    def scope_digest(self) -> str:
        return _digest({
            "decision": self.decision.to_dict(), "run_id": self.run_id, "loop_id": self.loop_id,
            "iteration_id": self.iteration_id, "plan_id": self.plan_id,
            "plan_revision": self.plan_revision,
            "policy_digest": self.policy_digest,
            "capability_request": _request_to_dict(self.capability_request) if self.capability_request else None,
            "synthesis": self.synthesis.to_dict() if self.synthesis else None,
            "retry_of_execution_id": self.retry_of_execution_id,
            "continuation_authorization_id": self.continuation_authorization_id,
            "provider_refs": list(self.provider_refs),
        })

    def to_dict(self):
        return {
            "request_id": self.request_id, "decision": self.decision.to_dict(),
            "prior_validation": self.prior_validation.to_dict(), "run_id": self.run_id,
            "loop_id": self.loop_id, "iteration_id": self.iteration_id, "plan_id": self.plan_id,
            "plan_revision": self.plan_revision, "policy_digest": self.policy_digest,
            "capability_request": _request_to_dict(self.capability_request) if self.capability_request else None,
            "synthesis": self.synthesis.to_dict() if self.synthesis else None,
            "researcher_override": self.researcher_override.to_dict() if self.researcher_override else None,
            "retry_of_execution_id": self.retry_of_execution_id,
            "continuation_authorization_id": self.continuation_authorization_id,
            "provider_refs": list(self.provider_refs),
        }

    def to_json(self):
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, data):
        keys = {"request_id", "decision", "prior_validation", "run_id", "loop_id", "iteration_id", "plan_id",
                "plan_revision", "policy_digest", "capability_request", "synthesis", "researcher_override"}
        keys |= {"retry_of_execution_id", "continuation_authorization_id", "provider_refs"}
        if not isinstance(data, dict) or set(data) != keys:
            raise ValueError("Malformed AutonomousExecutionRequest payload.")
        try:
            return cls(
                ResearchDecision.from_dict(data["decision"]), ResearchDecisionTrace.from_dict(data["prior_validation"]),
                data["run_id"], data["loop_id"], data["iteration_id"], data["plan_id"], data["plan_revision"], data["policy_digest"],
                _request_from_dict(data["capability_request"]) if data["capability_request"] else None,
                ResearchSynthesis.from_dict(data["synthesis"]) if data["synthesis"] else None,
                _override_from_dict(data["researcher_override"]) if data["researcher_override"] else None,
                data["request_id"],
                data["retry_of_execution_id"], data["continuation_authorization_id"],
                tuple(data["provider_refs"]),
            )
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError(f"Malformed AutonomousExecutionRequest: {exc}") from exc

    @classmethod
    def from_json(cls, payload):
        try:
            return cls.from_dict(json.loads(payload))
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Malformed AutonomousExecutionRequest JSON: {exc}") from exc


class AutonomousExecutionStatus(str, Enum):
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    INVALID = "INVALID"
    REJECTED = "REJECTED"
    SUPERSEDED = "SUPERSEDED"
    ALREADY_EXECUTED = "ALREADY_EXECUTED"
    MODIFICATION_REQUIRED = "MODIFICATION_REQUIRED"
    CONTINUATION_REQUIRED = "CONTINUATION_REQUIRED"
    RETRY_REQUIRED = "RETRY_REQUIRED"


@dataclass(frozen=True)
class AutonomousExecutionValidation:
    status: ValidationStatus
    reasons: tuple[str, ...]
    validated_at: str
    scope_digest: str

    def __post_init__(self):
        object.__setattr__(self, "status", _enum(ValidationStatus, self.status, "validation status"))
        reasons = tuple(self.reasons)
        if any(not isinstance(value, str) or not value.strip() for value in reasons):
            raise ValueError("validation reasons must be non-empty strings.")
        object.__setattr__(self, "reasons", reasons)
        _timestamp(self.validated_at, "validated_at")
        _nonempty(self.scope_digest, "scope_digest")
        if (self.status is ValidationStatus.REJECTED) != bool(reasons):
            raise ValueError("Rejected validation requires reasons; accepted validation has none.")

    def to_dict(self):
        return {"status": self.status.value, "reasons": list(self.reasons),
                "validated_at": self.validated_at, "scope_digest": self.scope_digest}

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict) or set(data) != {"status", "reasons", "validated_at", "scope_digest"}:
            raise ValueError("Malformed AutonomousExecutionValidation payload.")
        return cls(data["status"], tuple(data["reasons"]), data["validated_at"], data["scope_digest"])


@dataclass(frozen=True)
class AutonomousExecutionTrace:
    trace_id: str
    request_id: str
    decision_id: str
    approval_id: Optional[str]
    run_id: str
    loop_id: str
    iteration_id: str
    plan_id: str
    plan_revision: int
    task_id: Optional[str]
    action_type: ResearchActionType
    capability: Optional[CapabilityType]
    rationale: str
    proposal_digest: str
    provider_refs: tuple[str, ...]
    status: AutonomousExecutionStatus
    validation: AutonomousExecutionValidation
    created_at: str
    completed_at: str
    controlled_execution_id: Optional[str] = None
    dispatch_id: Optional[str] = None
    artifact_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    provenance_ids: tuple[str, ...] = ()
    reason: Optional[str] = None
    output: Optional[dict] = None

    def __post_init__(self):
        for name in ("trace_id", "request_id", "decision_id", "run_id", "loop_id", "iteration_id", "plan_id"):
            _nonempty(getattr(self, name), name)
        if self.approval_id is not None:
            _nonempty(self.approval_id, "approval_id")
        if not isinstance(self.plan_revision, int) or isinstance(self.plan_revision, bool) or self.plan_revision < 1:
            raise ValueError("plan_revision must be positive.")
        if self.task_id is not None:
            _nonempty(self.task_id, "task_id")
        object.__setattr__(self, "action_type", _enum(ResearchActionType, self.action_type, "action_type"))
        _nonempty(self.rationale, "rationale")
        _nonempty(self.proposal_digest, "proposal_digest")
        provider_refs = tuple(self.provider_refs)
        if any(not isinstance(value, str) or not value.strip() for value in provider_refs) or len(provider_refs) != len(set(provider_refs)):
            raise ValueError("provider_refs must contain unique non-empty strings.")
        object.__setattr__(self, "provider_refs", provider_refs)
        object.__setattr__(self, "status", _enum(AutonomousExecutionStatus, self.status, "execution status"))
        if self.capability is not None:
            object.__setattr__(self, "capability", _enum(CapabilityType, self.capability, "capability"))
        if not isinstance(self.validation, AutonomousExecutionValidation):
            raise ValueError("validation must be typed.")
        for name in ("created_at", "completed_at"):
            _timestamp(getattr(self, name), name)
        for name in ("artifact_ids", "evidence_ids", "provenance_ids"):
            values = tuple(getattr(self, name))
            if any(not isinstance(value, str) or not value.strip() for value in values) or len(set(values)) != len(values):
                raise ValueError(f"{name} must contain unique non-empty strings.")
            object.__setattr__(self, name, values)
        for name in ("controlled_execution_id", "dispatch_id", "reason"):
            value = getattr(self, name)
            if value is not None:
                _nonempty(value, name)
        if self.output is not None and not isinstance(self.output, dict):
            raise ValueError("output must be a JSON-compatible object.")

    def to_dict(self):
        return {"trace_id": self.trace_id, "request_id": self.request_id, "decision_id": self.decision_id,
                "approval_id": self.approval_id, "run_id": self.run_id, "loop_id": self.loop_id,
                "iteration_id": self.iteration_id, "plan_id": self.plan_id, "plan_revision": self.plan_revision,
                "task_id": self.task_id, "action_type": self.action_type.value,
                "capability": self.capability.value if self.capability else None,
                "rationale": self.rationale, "proposal_digest": self.proposal_digest,
                "provider_refs": list(self.provider_refs), "status": self.status.value,
                "validation": self.validation.to_dict(), "created_at": self.created_at,
                "completed_at": self.completed_at, "controlled_execution_id": self.controlled_execution_id,
                "dispatch_id": self.dispatch_id, "artifact_ids": list(self.artifact_ids),
                "evidence_ids": list(self.evidence_ids), "provenance_ids": list(self.provenance_ids),
                "reason": self.reason, "output": self.output}

    def to_json(self):
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, data):
        keys = {"trace_id", "request_id", "decision_id", "approval_id", "run_id", "loop_id", "iteration_id",
                "plan_id", "plan_revision", "task_id", "action_type", "capability", "rationale",
                "proposal_digest", "provider_refs", "status", "validation",
                "created_at", "completed_at", "controlled_execution_id", "dispatch_id", "artifact_ids",
                "evidence_ids", "provenance_ids", "reason", "output"}
        if not isinstance(data, dict) or set(data) != keys:
            raise ValueError("Malformed AutonomousExecutionTrace payload.")
        values = dict(data)
        for key in ("artifact_ids", "evidence_ids", "provenance_ids"):
            values[key] = tuple(values[key])
        values["provider_refs"] = tuple(values["provider_refs"])
        values["validation"] = AutonomousExecutionValidation.from_dict(values["validation"])
        try:
            return cls(**values)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Malformed AutonomousExecutionTrace: {exc}") from exc

    @classmethod
    def from_json(cls, payload):
        try:
            return cls.from_dict(json.loads(payload))
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Malformed AutonomousExecutionTrace JSON: {exc}") from exc


@dataclass(frozen=True)
class AutonomousExecutionResult:
    status: AutonomousExecutionStatus
    validation: AutonomousExecutionValidation
    trace: AutonomousExecutionTrace
    dispatch: Optional[ResearchCapabilityDispatchResult] = None
    synthesis: Optional[ResearchSynthesis] = None
    critique: Optional[ResearchCritique] = None

    def __post_init__(self):
        object.__setattr__(self, "status", _enum(AutonomousExecutionStatus, self.status, "execution status"))
        if not isinstance(self.validation, AutonomousExecutionValidation) or not isinstance(self.trace, AutonomousExecutionTrace):
            raise ValueError("Result validation and trace must be typed.")
        if self.trace.status is not self.status or self.trace.validation != self.validation:
            raise ValueError("Result and trace must describe the same status and validation.")

    def to_dict(self):
        return {"status": self.status.value, "validation": self.validation.to_dict(), "trace": self.trace.to_dict(),
                "dispatch": self.dispatch.to_dict() if self.dispatch else None,
                "synthesis": self.synthesis.to_dict() if self.synthesis else None,
                "critique": self.critique.to_dict() if self.critique else None}

    def to_json(self):
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict) or set(data) != {"status", "validation", "trace", "dispatch", "synthesis", "critique"}:
            raise ValueError("Malformed AutonomousExecutionResult payload.")
        return cls(
            data["status"], AutonomousExecutionValidation.from_dict(data["validation"]),
            AutonomousExecutionTrace.from_dict(data["trace"]),
            ResearchCapabilityDispatchResult.from_dict(data["dispatch"]) if data["dispatch"] else None,
            ResearchSynthesis.from_dict(data["synthesis"]) if data["synthesis"] else None,
            ResearchCritique.from_dict(data["critique"]) if data["critique"] else None,
        )

    @classmethod
    def from_json(cls, payload):
        try:
            return cls.from_dict(json.loads(payload))
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Malformed AutonomousExecutionResult JSON: {exc}") from exc


_ACTION_CAPABILITY = {
    ResearchActionType.RETRIEVE_EVIDENCE: CapabilityType.LOCAL_RETRIEVAL,
    ResearchActionType.VERIFY_CLAIM: CapabilityType.EVIDENCE_VERIFICATION,
    ResearchActionType.GENERATE_IMPROVEMENT: CapabilityType.IMPROVEMENT_GENERATION,
    ResearchActionType.INVESTIGATE_PRIOR_WORK: CapabilityType.PRIOR_WORK_INVESTIGATION,
    ResearchActionType.PLAN_EXPERIMENT: CapabilityType.EXPERIMENT_PLANNING,
}

_ACTION_REQUEST_CLASSES = {
    ResearchActionType.RETRIEVE_EVIDENCE: LocalRetrievalRequest,
    ResearchActionType.VERIFY_CLAIM: EvidenceVerificationRequest,
    ResearchActionType.GENERATE_IMPROVEMENT: ImprovementGenerationRequest,
    ResearchActionType.INVESTIGATE_PRIOR_WORK: PriorWorkInvestigationRequest,
    ResearchActionType.PLAN_EXPERIMENT: ExperimentPlanningRequest,
}


def execute_autonomous_proposal(
    plan: ResearchPlan, run: ResearchRun, review: PlanReview, loop: ResearchLoop,
    state: ResearchState, policy: BoundedExecutionPolicy, request: AutonomousExecutionRequest,
    approval: Optional[AutonomousExecutionApproval] = None, *, retrieval_provider=None, llm_provider=None, search_provider=None,
) -> AutonomousExecutionResult:
    """Revalidate, authorize, and execute no more than one accepted proposal."""
    if not isinstance(request, AutonomousExecutionRequest):
        raise ValueError("A typed AutonomousExecutionRequest is required.")
    created = _now()
    if not isinstance(approval, AutonomousExecutionApproval):
        validation = AutonomousExecutionValidation(ValidationStatus.REJECTED,
            ("proposal-scoped researcher approval is required",), created, request.scope_digest)
        return _make_result(request, approval if isinstance(approval, AutonomousExecutionApproval) else None,
                            validation, AutonomousExecutionStatus.INVALID, created,
                            reason="Proposal-scoped researcher approval is required.")
    if not isinstance(state, ResearchState):
        validation = AutonomousExecutionValidation(ValidationStatus.REJECTED,
            ("a current ResearchState is required",), created, request.scope_digest)
        return _make_result(request, approval, validation, AutonomousExecutionStatus.INVALID, created,
                            reason="A current ResearchState is required.")
    if approval.status is ApprovalStatus.SUPERSEDED:
        return _make_result(request, approval,
            AutonomousExecutionValidation(ValidationStatus.REJECTED, ("approval was superseded",), created, request.scope_digest),
            AutonomousExecutionStatus.SUPERSEDED, created, reason="Approval was superseded.")
    if approval.status is ApprovalStatus.REJECTED:
        return _make_result(request, approval,
            AutonomousExecutionValidation(ValidationStatus.REJECTED, ("researcher rejected this proposal",), created, request.scope_digest),
            AutonomousExecutionStatus.REJECTED, created, reason="Researcher rejected this proposal.")
    if approval.status is ApprovalStatus.REVOKED or _approval_mismatch(approval, request):
        reason = "researcher approval was revoked" if approval.status is ApprovalStatus.REVOKED else "approval is stale or bound to a different scope"
        validation = AutonomousExecutionValidation(ValidationStatus.REJECTED, (reason,), created, request.scope_digest)
        return _make_result(request, approval, validation, AutonomousExecutionStatus.INVALID, created, reason=reason)
    prior_trace = _prior_trace(state, request.decision.trace.decision_id)
    if prior_trace is not None:
        validation = AutonomousExecutionValidation(ValidationStatus.ACCEPTED, (), created, request.scope_digest)
        return _make_result(request, approval, validation, AutonomousExecutionStatus.ALREADY_EXECUTED,
                            created, reason="This proposal already has a controlled execution trace.")

    reasons = []
    fresh = None
    try:
        if not isinstance(policy, BoundedExecutionPolicy):
            raise ValueError("A typed BoundedExecutionPolicy is required.")
        if request.plan_id != plan.plan_id or request.plan_revision != plan.revision:
            reasons.append("request plan ID/revision is stale")
        if request.run_id != run.run_id or request.loop_id != loop.loop_id:
            reasons.append("request run/loop identity is stale")
        if request.iteration_id != loop.current_iteration_id:
            reasons.append("request iteration is stale")
        if request.policy_digest != _digest(_policy_payload(policy)):
            reasons.append("execution policy changed after researcher approval")
        if approval.status is ApprovalStatus.REVOKED:
            reasons.append("researcher approval was revoked")
        if _approval_mismatch(approval, request):
            reasons.append("approval does not bind this exact proposal, run, revision, action, task, and scope")
        proposal = request.decision.proposal
        if request.researcher_override is not None:
            override = request.researcher_override
            if override.action is ResearcherOverrideAction.REJECT:
                return _make_result(request, approval,
                    AutonomousExecutionValidation(ValidationStatus.REJECTED, ("researcher rejected this proposal",), created, request.scope_digest),
                    AutonomousExecutionStatus.REJECTED, created, reason="Researcher rejected this proposal.")
            if override.action is ResearcherOverrideAction.MODIFY:
                return _make_result(request, approval,
                    AutonomousExecutionValidation(ValidationStatus.REJECTED, ("modified proposal must be issued and approved as a new decision",), created, request.scope_digest),
                    AutonomousExecutionStatus.MODIFICATION_REQUIRED, created,
                    reason="Modified proposal must receive a new decision and scoped approval.")
            if override.action is ResearcherOverrideAction.STOP and proposal.action_type is not ResearchActionType.STOP:
                reasons.append("researcher STOP override requires an explicit STOP proposal")
            if override.action is ResearcherOverrideAction.ACCEPT and override.decision_id != request.decision.trace.decision_id:
                reasons.append("researcher acceptance refers to another decision")
        if reasons:
            validation = AutonomousExecutionValidation(ValidationStatus.REJECTED, tuple(dict.fromkeys(reasons)), created, request.scope_digest)
            return _make_result(request, approval, validation, AutonomousExecutionStatus.INVALID, created, reason="; ".join(validation.reasons))
        fresh = validate_research_decision(
            request.decision, plan=plan, run=run, review=review, loop=loop, state=state,
            policy=policy, researcher_override=request.researcher_override,
        )
        if fresh.validation_status is not ValidationStatus.ACCEPTED:
            validation = AutonomousExecutionValidation(ValidationStatus.REJECTED, fresh.rejection_reasons, created, request.scope_digest)
            return _make_result(request, approval, validation, AutonomousExecutionStatus.INVALID, created,
                                reason="; ".join(validation.reasons))
        _validate_action_request(request, plan, run, loop, state)
        _validate_provider_inputs(request, retrieval_provider, llm_provider, search_provider)
    except Exception as exc:
        reasons.append(f"current-state validation failed: {type(exc).__name__}: {exc}")
        validation = AutonomousExecutionValidation(ValidationStatus.REJECTED, tuple(reasons), created, request.scope_digest)
        return _make_result(request, approval, validation, AutonomousExecutionStatus.INVALID, created, reason="; ".join(reasons))

    validation = AutonomousExecutionValidation(ValidationStatus.ACCEPTED, (), created, request.scope_digest)
    if request.decision.proposal.action_type in _ACTION_CAPABILITY:
        task = next(item for item in plan.tasks if item.task_id == request.decision.proposal.task_id)
        prior = [item for item in state.task_executions
                 if item.run_id == run.run_id and item.task_id == task.task_id]
        if prior and prior[-1].status.value == "COMPLETED" and request.continuation_authorization_id is None:
            return _make_result(request, approval, validation, AutonomousExecutionStatus.CONTINUATION_REQUIRED,
                                created, reason="Phase 7C must explicitly authorize continuation before another execution.")
        if prior and prior[-1].status.value == "FAILED" and request.retry_of_execution_id is None:
            return _make_result(request, approval, validation, AutonomousExecutionStatus.RETRY_REQUIRED,
                                created, reason="Phase 5 explicit retry authorization is required for a failed execution.")
        try:
            dispatch = dispatch_research_capability(
                plan, run, task, review, loop, state,
                capability=_ACTION_CAPABILITY[request.decision.proposal.action_type],
                capability_request=request.capability_request, iteration_id=request.iteration_id,
                retrieval_provider=retrieval_provider, llm_provider=llm_provider, search_provider=search_provider,
                retry_of_execution_id=request.retry_of_execution_id,
                continuation_authorization_id=request.continuation_authorization_id,
            )
            status = AutonomousExecutionStatus.COMPLETED if dispatch.status is DispatchStatus.COMPLETED else AutonomousExecutionStatus.FAILED
            trace = _trace(request, approval, validation, status, created, dispatch=dispatch,
                           reason=dispatch.error)
            _record_trace(state, trace)
            return AutonomousExecutionResult(status, validation, trace, dispatch=dispatch)
        except Exception as exc:
            failed_validation = validation
            trace = _trace(request, approval, failed_validation, AutonomousExecutionStatus.FAILED, created,
                           reason=f"{type(exc).__name__}: {exc}")
            _record_trace(state, trace)
            return AutonomousExecutionResult(AutonomousExecutionStatus.FAILED, failed_validation, trace)

    action = request.decision.proposal.action_type
    synthesis = critique = None
    dispatch = None
    try:
        if action is ResearchActionType.SYNTHESIZE:
            synthesis = synthesize_research_state(plan, run, review, loop, state, iteration_id=request.iteration_id,
                                                  artifact_ids=request.decision.proposal.artifact_ids or None)
        elif action is ResearchActionType.CRITIQUE:
            critique = critique_research_synthesis(plan, run, review, loop, state, request.synthesis,
                                                   iteration_id=request.iteration_id)
        else:
            continuation_action = {
                ResearchActionType.STOP: ResearchContinuationAction.STOP,
                ResearchActionType.REQUEST_RESEARCHER_REVIEW: ResearchContinuationAction.WAIT_FOR_RESEARCHER,
                ResearchActionType.RETURN_TO_PLANNING: ResearchContinuationAction.RETURN_TO_PLANNING,
            }.get(action)
            if continuation_action is None:
                raise ValueError(f"Unsupported 8A action for controlled execution: {action.value}.")
            continuation_result = apply_research_continuation(plan, run, review, loop, state, action=continuation_action)
        status = AutonomousExecutionStatus.COMPLETED
        output = (synthesis.to_dict() if synthesis else critique.to_dict() if critique
                  else _jsonable(continuation_result) if action in {
                      ResearchActionType.STOP, ResearchActionType.REQUEST_RESEARCHER_REVIEW,
                      ResearchActionType.RETURN_TO_PLANNING} else None)
        trace = _trace(request, approval, validation, status, created, output=output)
        _record_trace(state, trace)
        return AutonomousExecutionResult(status, validation, trace, dispatch, synthesis, critique)
    except Exception as exc:
        trace = _trace(request, approval, validation, AutonomousExecutionStatus.FAILED, created,
                       reason=f"{type(exc).__name__}: {exc}")
        _record_trace(state, trace)
        return AutonomousExecutionResult(AutonomousExecutionStatus.FAILED, validation, trace)


def _validate_action_request(request, plan, run, loop, state):
    action = request.decision.proposal.action_type
    proposal = request.decision.proposal
    if proposal.task_id is not None:
        if proposal.task_id != loop.current_task_id:
            raise ValueError("Proposal task is not the currently selected ResearchLoop task.")
        active = next((item for item in loop.loop_iterations if item.iteration_id == request.iteration_id), None)
        if active is None or proposal.task_id not in active.task_ids:
            raise ValueError("Proposal task is not part of the active loop iteration.")
    if action in _ACTION_CAPABILITY:
        expected = _ACTION_CAPABILITY[action]
        capability = request.capability_request
        if expected is CapabilityType.LOCAL_RETRIEVAL:
            if capability.query != proposal.query:
                raise ValueError("LOCAL_RETRIEVAL request query must match the proposal query.")
            if capability.question_id is not None and capability.question_id not in proposal.question_ids:
                raise ValueError("LOCAL_RETRIEVAL question_id must be referenced by the proposal.")
        elif expected is CapabilityType.EVIDENCE_VERIFICATION:
            claims = {item.claim_id: item for item in state.claims}
            if len(proposal.claim_ids) != 1 or proposal.claim_ids[0] not in claims:
                raise ValueError("VERIFY_CLAIM requires exactly one existing ResearchClaim.")
            if capability.claim != claims[proposal.claim_ids[0]].text:
                raise ValueError("Verification request claim must match the stored claim text.")
            if tuple(capability.evidence_ids) != tuple(proposal.evidence_ids):
                raise ValueError("Verification request evidence_ids must match the proposal exactly.")
        elif expected is CapabilityType.IMPROVEMENT_GENERATION:
            if capability.problem_statement != proposal.problem_statement or tuple(capability.evidence_ids) != tuple(proposal.evidence_ids):
                raise ValueError("Improvement request problem/evidence must match the proposal.")
            if not set(capability.verified_claim_ids).issubset(proposal.claim_ids):
                raise ValueError("Improvement request claims must be referenced by the proposal.")
            if not set(capability.verification_artifact_ids).issubset(proposal.artifact_ids):
                raise ValueError("Improvement verification artifacts must be referenced by the proposal.")
        elif expected is CapabilityType.PRIOR_WORK_INVESTIGATION:
            if (capability.candidate_id != proposal.candidate_id
                    or capability.candidate_artifact_id not in proposal.artifact_ids):
                raise ValueError("Prior-work request must target the proposed stored candidate artifact.")
            if capability.question_id is not None and capability.question_id not in proposal.question_ids:
                raise ValueError("Prior-work request question_id must be referenced by the proposal.")
        elif expected is CapabilityType.EXPERIMENT_PLANNING:
            if (capability.candidate_id != proposal.candidate_id
                    or capability.improvement_artifact_id not in proposal.artifact_ids):
                raise ValueError("Experiment request must target the proposed stored improvement artifact.")
            if not set(capability.prior_work_artifact_ids + capability.verification_artifact_ids).issubset(proposal.artifact_ids):
                raise ValueError("Experiment input artifacts must be referenced by the proposal.")
            if not set(capability.evidence_ids).issubset(proposal.evidence_ids):
                raise ValueError("Experiment evidence must be referenced by the proposal.")
    elif action is ResearchActionType.CRITIQUE:
        if request.synthesis.run_id != request.run_id or request.synthesis.plan_revision != request.plan_revision:
            raise ValueError("Critique synthesis must belong to this run and exact plan revision.")
    elif action not in {ResearchActionType.SYNTHESIZE, ResearchActionType.STOP,
                        ResearchActionType.REQUEST_RESEARCHER_REVIEW, ResearchActionType.RETURN_TO_PLANNING}:
        raise ValueError(f"Action {action.value!r} is not supported by the controlled bridge.")


def provider_reference(provider):
    """Return a stable, non-secret provider label suitable for scoped approval."""
    if provider is None:
        raise ValueError("A provider is required to form an execution scope.")
    for attribute in ("name", "model"):
        value = getattr(provider, attribute, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return f"{type(provider).__module__}.{type(provider).__qualname__}"


def provider_bindings_for_action(action, *, retrieval_provider=None, llm_provider=None, search_provider=None):
    """Build the ordered provider labels that must be bound into approval."""
    action = _enum(ResearchActionType, action, "action_type")
    capability = _ACTION_CAPABILITY.get(action)
    if capability is CapabilityType.LOCAL_RETRIEVAL:
        return (f"retrieval:{provider_reference(retrieval_provider)}",)
    if capability in {CapabilityType.EVIDENCE_VERIFICATION, CapabilityType.IMPROVEMENT_GENERATION,
                      CapabilityType.EXPERIMENT_PLANNING}:
        return (f"llm:{provider_reference(llm_provider)}",)
    if capability is CapabilityType.PRIOR_WORK_INVESTIGATION:
        return (f"search:{provider_reference(search_provider)}", f"llm:{provider_reference(llm_provider)}")
    return ()


def _validate_provider_inputs(request, retrieval_provider, llm_provider, search_provider):
    action = request.decision.proposal.action_type
    capability = _ACTION_CAPABILITY.get(action)
    if capability is CapabilityType.LOCAL_RETRIEVAL and retrieval_provider is None:
        raise ValueError("LOCAL_RETRIEVAL requires an explicit retrieval_provider.")
    if capability in {CapabilityType.EVIDENCE_VERIFICATION, CapabilityType.IMPROVEMENT_GENERATION,
                      CapabilityType.EXPERIMENT_PLANNING} and llm_provider is None:
        raise ValueError(f"{capability.value} requires an explicit llm_provider.")
    if capability is CapabilityType.PRIOR_WORK_INVESTIGATION and (llm_provider is None or search_provider is None):
        raise ValueError("PRIOR_WORK_INVESTIGATION requires explicit llm_provider and search_provider.")
    actual = provider_bindings_for_action(action, retrieval_provider=retrieval_provider,
                                          llm_provider=llm_provider, search_provider=search_provider)
    if request.provider_refs != actual:
        raise ValueError("Runtime providers do not match the provider references approved for this proposal.")


def _approval_mismatch(approval, request):
    proposal = request.decision.proposal
    return any((
        approval.decision_id != request.decision.trace.decision_id,
        approval.run_id != request.run_id, approval.loop_id != request.loop_id,
        approval.iteration_id != request.iteration_id, approval.plan_id != request.plan_id,
        approval.plan_revision != request.plan_revision, approval.action_type is not proposal.action_type,
        approval.task_id != proposal.task_id, approval.scope_digest != request.scope_digest,
    ))


def _policy_payload(policy):
    return {
        "maximum_iterations": policy.maximum_iterations,
        "maximum_capability_invocations": policy.maximum_capability_invocations,
        "maximum_task_executions": policy.maximum_task_executions,
        "allowed_capabilities": [item.value for item in policy.allowed_capabilities],
        "allowed_stages": [item.value for item in policy.allowed_stages],
        "allowed_task_ids": list(policy.allowed_task_ids),
        "allow_continuation": policy.allow_continuation,
        "stop_on_failure": policy.stop_on_failure,
        "stop_on_unresolved_blocker": policy.stop_on_unresolved_blocker,
    }


def execution_policy_digest(policy: BoundedExecutionPolicy) -> str:
    """Stable fingerprint used to bind researcher approval to current policy."""
    if not isinstance(policy, BoundedExecutionPolicy):
        raise ValueError("A typed BoundedExecutionPolicy is required.")
    return _digest(_policy_payload(policy))


def _trace(request, approval, validation, status, created, *, dispatch=None, output=None, reason=None):
    proposal = request.decision.proposal
    capability = _ACTION_CAPABILITY.get(proposal.action_type)
    return AutonomousExecutionTrace(
        _id("autonomous_trace"), request.request_id, request.decision.trace.decision_id,
        approval.approval_id if approval is not None else None,
        request.run_id, request.loop_id, request.iteration_id,
        request.plan_id, request.plan_revision, proposal.task_id, proposal.action_type,
        capability, proposal.rationale, _digest(request.decision.to_dict()), request.provider_refs,
        status, validation, created, _now(),
        controlled_execution_id=dispatch.execution_id if dispatch else None,
        dispatch_id=dispatch.dispatch_id if dispatch else None,
        artifact_ids=dispatch.artifact_ids if dispatch else (),
        evidence_ids=dispatch.evidence_ids if dispatch else (),
        provenance_ids=proposal.provenance_ids, reason=reason, output=output,
    )


def _make_result(request, approval, validation, status, created, *, reason=None):
    trace = _trace(request, approval, validation, status, created, reason=reason)
    return AutonomousExecutionResult(status, validation, trace)


def _record_trace(state, trace):
    """Use the existing event log only after execution/control handling occurs."""
    state.log("autonomous_execution", "proposal_execution_trace", trace.to_json())


def _prior_trace(state, decision_id):
    for event in state.events:
        if event.agent == "autonomous_execution" and event.action == "proposal_execution_trace":
            try:
                trace = AutonomousExecutionTrace.from_json(event.message)
            except ValueError:
                continue
            if trace.decision_id == decision_id:
                return trace
    return None


def _request_to_dict(request):
    if request is None:
        return None
    return {"type": type(request).__name__, "values": _jsonable(request)}


_REQUEST_CLASSES = {
    cls.__name__: cls for cls in (LocalRetrievalRequest, EvidenceVerificationRequest,
        ImprovementGenerationRequest, PriorWorkInvestigationRequest, ExperimentPlanningRequest)
}


def _request_from_dict(data):
    if not isinstance(data, dict) or set(data) != {"type", "values"} or data["type"] not in _REQUEST_CLASSES:
        raise ValueError("Unknown or malformed typed capability request.")
    cls = _REQUEST_CLASSES[data["type"]]
    values = data["values"]
    if not isinstance(values, dict):
        raise ValueError("Capability request values must be an object.")
    values = dict(values)
    if cls is LocalRetrievalRequest:
        versions = values.get("evidence_document_versions", {})
        values["evidence_document_versions"] = {int(k): DocumentVersion.from_dict(v) for k, v in versions.items()}
    elif cls is PriorWorkInvestigationRequest:
        values["scope"] = PriorWorkSearchScope.from_dict(values["scope"])
    for key, value in tuple(values.items()):
        if isinstance(value, list):
            values[key] = tuple(value)
    return cls(**values)


def _override_from_dict(data):
    if not isinstance(data, dict):
        raise ValueError("researcher_override must be an object.")
    values = dict(data)
    proposal = values.get("modified_proposal")
    if proposal is not None:
        values["modified_proposal"] = ResearchActionProposal.from_dict(proposal)
    return ResearcherOverride(**values)


__all__ = [
    "ApprovalStatus", "AutonomousExecutionApproval", "AutonomousExecutionRequest",
    "AutonomousExecutionStatus", "AutonomousExecutionValidation", "AutonomousExecutionTrace",
    "AutonomousExecutionResult", "execute_autonomous_proposal",
    "provider_reference", "provider_bindings_for_action", "execution_policy_digest",
]
