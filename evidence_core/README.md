# Evidence Core (Slice 0)

Domain-general foundation for the research co-pilot. The single architectural
correction it encodes:

> **RETRIEVAL RESULT ≠ EVIDENCE ≠ SUPPORT**

Slice 0 changes **no behaviour**. Retrieval (`rag.py`), prompts, models, the frozen
Experiment B pipeline and every Phase 10–13 artifact are untouched. The package
only *reads* existing outputs, re-expresses them in new contracts, and proves —
by exact replay — that the new representation preserves the old behaviour.

## 1. Why a retrieval candidate is not evidence

A retriever answers "which spans are *about* this query?" Its scores (BM25, dense
similarity, cross-encoder relevance, RRF) measure topical relevance. The inherited
pipeline converted every retrieved rank directly into an `Evidence` record and a
relevance composite was named `evidence_score`. That conflation is the root cause
of "topic relevance → SUPPORTS" errors observed in Phase 12/13.

Here a retrieval hit is a `RetrievedCandidate`. It records the query, method,
rank, scores, provider and the exact emitted text, and **has no claim and no
stance**. Adapters never create `Evidence` from retrieval alone.

## 2. Why evidence is not support

`Evidence` is a *candidate evidentiary unit for a specific claim*, created by an
explicit `AdmissionRecord` (role, reason, admitter, presentation index, source
candidate). Admission says *why the span is being considered*, never *what it
shows*. `Evidence` has no stance field.

Support exists only as a `Judgement`: producer P (LLM, NLI, human, deterministic)
assessed evidence E against claim C as `SUPPORTS | CONTRADICTS | NEUTRAL | INVALID`.
`INVALID` (the producer gave no usable answer) is never treated as support.

## 3. Why Claim, Judgement and Verdict are separate

* **Claim** — what is asserted, with optional slots (subject, predicate, object,
  comparison, quantity, unit, time, population, condition, polarity, modality,
  causal type), `required_slots`, a decomposition parent and `semantic_unity`
  (qualifiers that must stay attached are kept attached, not split off).
* **Judgement** — one producer's pairwise opinion, with judge identity
  (kind/model/version/prompt), optional per-slot coverage, confidence, rationale
  and raw provenance. Many judges can judge the *same* claim and evidence — the
  H5 experimental design (same Claim + same Evidence, different producer) needs
  no change to Claim or Evidence.
* **ClaimVerdict** — a *policy's* aggregation of judgements. It carries a rich
  `state` (`SUPPORTED, PARTIALLY_SUPPORTED, CONTRADICTED, MIXED_EVIDENCE,
  INSUFFICIENT_EVIDENCE, UNVERIFIED`) and, separately, a `LegacyOutcome` (the frozen
  3-class verdict + run status). The two are never collapsed into each other.

Relations are IDs on the contracts (Source → DocumentVersion → Span → Candidate /
Evidence; Claim parent → child; Judgement → (Claim, Evidence); Verdict →
Judgements). `EvidenceGraph` indexes them, enforces referential integrity and
exposes source independence (`independence_group`, else `source_id`). It is
storage-independent; every object round-trips through plain JSON with stable,
content-derived IDs.

## 4. Why the legacy aggregation policy is preserved

`LegacyAnySupportsPolicy` reproduces frozen Experiment B *exactly* — including its
known flaws (one SUPPORTS on a component is enough; support + contradiction
collapses to INSUFFICIENT; lexical guards; only the first unit counts; any failed
pair fails the attempt). It exists so that every later change can be measured
against a byte-exact baseline expressed in the new contracts. It does **not**
infer a rich `VerdictState`. The two frozen lexical guards are delegated to the
frozen functions rather than re-implemented, so they cannot drift.

## 5. How the adapters work

* `adapters/retrieval.py` — recorded `retrieved_chunks` rows or live
  `RetrievalResponse` objects → Source / DocumentVersion / Span /
  RetrievedCandidate. Preserves document identity, rank, all available scores,
  method, provider, query and the exact emitted text. Source type is `UNKNOWN`
  (retrieval does not say what kind of source a document is). With an optional
  document lookup, DocumentVersions get a content hash.
* `adapters/experiment_b.py` — one recorded `run_system_b` output → root Claim,
  legacy unit Claims, component Claims, Evidence for the judged top-N (role
  `LEGACY_TOP_N_PRESENTED_TO_JUDGE`, text = exactly what the judge saw), and one
  Judgement per recorded pair (failed pairs → `INVALID`, raw attempts kept in
  provenance). It never calls Ollama and raises `NotReplayable` / `AmbiguousRecord`
  instead of guessing (e.g. it re-derives units deterministically and refuses a
  record whose recorded unit differs).

## 6. How replay proves compatibility

`python -m evidence_core.replay` loads every recorded Experiment B run (frozen
Phase 10F full DEV, Phase 13 B0/B1/B2/B4, the NEI probe), adapts each record,
validates the graph, applies the legacy policy and requires **exact equality** of
run status, verdict, every unit verdict and every unit's per-component coverage
matrix. Socket connections are forbidden during replay. Records with no recorded
judgements (interrupted attempts) are listed as not replayable, never as matches.
Tests additionally show the check is discriminating (a flipped or dropped recorded
stance changes the replayed outcome) and that the legacy ladder equals the frozen
validator on exhaustive stance grids.

Result (see `evaluation/experiments/results/slice0_evidence_core/`): 606 records,
604 replayed, 604 exact, 0 mismatches, 2 not replayable (interrupted).

## 7. What Slice 1 will change

A new policy implementing `AggregationPolicy` (not the legacy one) that may use
source independence, slot coverage and rich states, evaluated offline against the
recorded judgements and the legacy baseline.

## 8. Status

| Item | Status |
|---|---|
| Contracts (Source, DocumentVersion, Span, RetrievedCandidate, AdmissionRecord, Evidence, Claim, JudgeIdentity, Judgement, ComponentCoverage, LegacyOutcome, ClaimVerdict, Provenance) | IMPLEMENTED |
| JSON round trip, stable IDs, referential integrity, independence grouping | IMPLEMENTED |
| Retrieval adapter (recorded refs, live RetrievalResponse) | IMPLEMENTED |
| Experiment B adapter (recorded outputs, read-only) | IMPLEMENTED |
| LegacyAnySupportsPolicy + offline exact replay | IMPLEMENTED |
| Rich-state contract properties (SUPPORTED needs judgements and no uncovered slots) | IMPLEMENTED (validation only) |
| `AggregationPolicy` for rich states | INTERFACE ONLY |
| `Judgement.slot_coverage`, `Evidence.attributes`, `Claim` slots beyond legacy | INTERFACE ONLY (populated only from legacy data / tests) |
| Source types, independence groups from real metadata | INTERFACE ONLY (adapters set UNKNOWN / none) |
| Rich aggregation, conflict analysis, NLI/H5 judges, claim decomposition, evidence extraction, ingestion, research loop, synthesis | FUTURE |

---

# Slice 1: aggregation semantics

Module: `evidence_core/aggregation_v2.py` · runner: `evaluation/experiments/run_slice1_aggregation.py` ·
results: `evaluation/experiments/results/slice1_aggregation/`. Offline and deterministic: no model, retrieval or
network (socket connections are forbidden during the run).

**Scope statement.** Slice 1 does not evaluate slot-aware aggregation because the frozen judgements contain no
slot-level coverage annotations (0 of 2,350 recorded judgements carry slot coverage). Claim aggregation is
component-level because slot-level judgement coverage is absent from the frozen records. `Claim.required_slots`
is diagnostic metadata only (`SLOT_COVERAGE_NOT_ASSESSED`); it is never treated as covered or as uncovered.

## Why the legacy rule is insufficient

Legacy Experiment B marks a component supported if *any one* presented evidence item is judged SUPPORTS, folds a
support/contradiction conflict into INSUFFICIENT_EVIDENCE, fails a whole attempt when any single pair has no
valid judgement (including endpoint outages), and blocks support with lexical number/object guards.

## What the primary policy (`ConservativeComponentPolicy`) considers

* component state from **valid** judgements only: SUPPORTS-only → supported, CONTRADICTS-only → contradicted,
  both → mixed, NEUTRAL-only → unresolved, no valid judgement → unverified;
* INVALID contributes no stance (never support, never contradiction); it sets `judgement_incomplete` and is
  classified from the recorded error into ENDPOINT_INFRASTRUCTURE / TIMEOUT / MODEL_OUTPUT_SCHEMA /
  MISSING_RECORD / UNKNOWN_FAILURE — infrastructure failures are not reasoning errors;
* claim (first unit, legacy-comparable): any mixed → MIXED_EVIDENCE; else any contradicted → CONTRADICTED; else
  all supported → SUPPORTED; else supported + unresolved/unverified → PARTIALLY_SUPPORTED; else all unverified →
  UNVERIFIED; else INSUFFICIENT_EVIDENCE. Precedence note: a MIXED component outranks a CONTRADICTED one, so adding
  a contradiction to the supported component of a CONTRADICTED claim yields MIXED_EVIDENCE (less decisive, not
  more supportive); this never occurs in the recorded data.

## What it deliberately does not consider

Slot coverage, lexical/word-presence checks (the legacy guards are **not** applied), retrieval or reranker scores,
confidence values, weights, numerical thresholds, models.

## Source independence

Independence key = `Source.independence_group`, else `source_id` (Slice 0). SciFact sources have no group
metadata, so independence is document-level and anything beyond that is UNKNOWN. In the primary policy
independence is diagnostic only. `IndependentCorroborationSensitivity` (≥2 independence groups per supported
component) is a **sensitivity analysis — not selected as primary policy**; its threshold is not tuned.

## Relationship to H5

Slice 1 holds every recorded pair-level judgement fixed and varies only aggregation. If aggregation cannot separate
correct from incorrect support decisions, the remaining error must be addressed at the pair-level stance layer
(H5). See the Slice 1 report for the decision; Slice 1 itself does not run any judge.

| Item | Status |
|---|---|
| ConservativeComponentPolicy, IndependentCorroborationSensitivity, failure taxonomy | IMPLEMENTED |
| Offline paired experiment (legacy vs primary vs sensitivity), gold metrics on the 69 dev claims only | IMPLEMENTED |
| Slot-aware aggregation | NOT POSSIBLE with frozen data (no slot-level judgements) |
| Independence beyond document level | UNKNOWN (no metadata) |
