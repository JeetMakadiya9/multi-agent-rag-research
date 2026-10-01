"""Phase 10F full DEV runner with streaming, resumable A/B/C attempts.

All system callbacks receive a gold-free runtime payload. Gold annotations are
loaded for scoring only after all three claim/system attempts have been
durably recorded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import statistics
import sys
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.experiments.evaluator import (  # noqa: E402
    LABELS, SYSTEM_NAMES, SystemOutput, _attribution_metrics,
    _safe_call, gold_claim_label,
)
from evaluation.experiments.experiment_config import ExperimentConfig  # noqa: E402
from evaluation.experiments.phase10e_infrastructure import (  # noqa: E402
    OutputState, append_jsonl, atomic_write_json, canonical_fingerprint,
    claim_system_key, exclusive_run_lock,
    load_resume_manifest, make_configuration_lock, recover_jsonl,
    validate_attempt_record, validate_resume_configuration,
    validate_scored_record,
)
from evaluation.experiments.run_phase10_dev import (  # noqa: E402
    SYSTEMS, _json_safe, _output_from_record, _output_to_record, _provenance, build_provider, paired_accuracy_bootstrap, sha256_file,
)
from evaluation.scifact.metrics import retrieval_metrics_for_claim  # noqa: E402
from evaluation.scifact.scifact_adapter import (  # noqa: E402
    corpus_to_rag_chunks, load_scifact_dataset, runtime_claim_payload,
)


FROZEN_EXPECTED = {
    "rag.py": "D2B0CF8930BC66FC66995AD63C1028AE3B126424E01E091DF22ED238AD5DBA19",
    "research_pipeline.py": "F55A29386AA13774A621B855E89CA8CA71FEAC0CFE2A124F90EF54DD7CD541D0",
    "claim_verification.py": "63FC8D73F55B7F48DBB340E4CD67C97975D0D587FB009CEAFA97E7FEF1C639A1",
    "verification_units.py": "BEFE07461C9B0DA76BF68226D86A96346D761C1AD3F127C554D4BB9C51644DDF",
    "experiment_b_verification.py": "66EA7B6BE6119E7A3E25A7BAD3AF3D078F08CBA4CC8AC4A462BAF6170548F3B4",
    "evaluation/scifact/scifact_adapter.py": "8F2BE845CFAF9ABB2210C88242F02E55EA8F47FF68E71BCDD27098287904BC9A",
    "evaluation/scifact/metrics.py": "8BE4B90BB28090FC48F874AF399798B0F7E39C25AAD7A1FA17B864A002E55587",
    "evaluation/scifact/run_scifact_evaluation.py": "21F540C534B5CFE8C8328C648C6E89AF6E4C4C16497CF78BC52DB51BC9629338",
}
PHASE10E_DIR = ROOT / "evaluation/experiments/results/phase10e_remediation"
DATA_DIR = ROOT / "data/scifact/data"
EVAL_DIR = ROOT / "evaluation/experiments"
SOURCE_FILES = (
    "evaluation/experiments/run_phase10f.py",
    "evaluation/experiments/phase10e_infrastructure.py",
    "evaluation/experiments/evaluator.py",
    "evaluation/experiments/experiment_config.py",
    "evaluation/experiments/system_a_rag.py",
    "evaluation/experiments/system_b_verified_rag.py",
    "evaluation/experiments/system_c_multi_agent.py",
    "evaluation/experiments/local_model.py",
    "src/baseline_rag.py",
    "research_capability_dispatch.py", "research_continuation.py",
    "research_autonomous_controller.py", "research_autonomous_execution.py",
    "research_verification_capability.py",
)

# One-time, narrowly scoped compatibility for the historical launch lock. The
# stored launch used this runner hash; the remediation changes orchestration /
# auditing only. Every other config and source identity remains strict.
PHASE10F_PRE_REMEDIATION_RUNNER_SHA256 = "863CE82B53586E013412279E90A56E3DD1C16E76F4A5882D14AEC229EABF9818"
PHASE10F_REMEDIATION_SOURCE_IDENTITY = "A8B45924659F646DA28E8318C7929A01E486AA55C6FC1CBA81727BB9B822A95F"
PHASE10F_RUNNER_SOURCE_PATH = "evaluation/experiments/run_phase10f.py"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_hash(value: Any) -> str:
    return canonical_fingerprint(value)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def historical_result_hashes() -> dict[str, str]:
    """Hash prior Phase 10C/10D/10E artifacts without changing them."""
    base = ROOT / "evaluation/experiments/results"
    roots = ("phase10_dev.json", "phase10_dev_run", "phase10_dev_preflight", "phase10_dev_preflight_2e", "phase10_error_analysis", "phase10e_readiness", "phase10e_remediation")
    result: dict[str, str] = {}
    for item in roots:
        path = base / item
        candidates = [path] if path.is_file() else sorted(path.rglob("*")) if path.is_dir() else []
        for candidate in candidates:
            if candidate.is_file():
                result[candidate.relative_to(ROOT).as_posix()] = sha256_file(candidate).upper()
    return result


def verify_historical_result_hashes(expected: Mapping[str, str]) -> bool:
    return dict(expected) == historical_result_hashes()

def validate_prerequisites(*, dataset_root: str | Path = DATA_DIR,
                            phase10e_dir: str | Path = PHASE10E_DIR) -> dict[str, Any]:
    """Fail closed on identity, frozen hash, dataset, or model-lock mismatch."""
    droot, e10 = Path(dataset_root).resolve(), Path(phase10e_dir).resolve()
    lock = _read_json(e10 / "configuration_lock.json")
    lock_config = lock.get("configuration")
    if not isinstance(lock_config, dict) or _json_hash(lock_config) != lock.get("fingerprint_sha256"):
        raise ValueError("Phase 10E configuration lock fingerprint is invalid")
    model = _read_json(e10 / "model_preflight.json")
    if model.get("status") != "PASS" or not model.get("model_digest"):
        raise ValueError("Phase 10E local model health/digest preflight is not PASS")
    if model.get("model") != lock_config.get("model", {}).get("name") or \
            model.get("model_digest") != lock_config.get("model", {}).get("digest"):
        raise ValueError("Phase 10E model name or digest differs between lock and preflight")
    prior_integrity = _read_json(e10 / "dataset_integrity.json")
    paths = {name: droot / name for name in
        ("claims_train.jsonl", "claims_dev.jsonl", "claims_test.jsonl", "corpus.jsonl")}
    counts = {name: sum(1 for line in path.open("rb") if line.strip()) for name, path in paths.items()}
    hashes = {name: sha256_file(path).upper() for name, path in paths.items()}
    expected_counts = {"claims_train.jsonl": 809, "claims_dev.jsonl": 300,
        "claims_test.jsonl": 300, "corpus.jsonl": 5183}
    if counts != expected_counts or hashes != prior_integrity.get("sha256"):
        raise ValueError(f"SciFact dataset identity changed: counts={counts}, hashes={hashes}")
    actual_frozen = {name: sha256_file(ROOT / name).upper() for name in FROZEN_EXPECTED}
    if actual_frozen != FROZEN_EXPECTED:
        mismatch = {name: {"expected": FROZEN_EXPECTED[name], "actual": actual_frozen[name]}
                    for name in FROZEN_EXPECTED if actual_frozen[name] != FROZEN_EXPECTED[name]}
        raise ValueError(f"Frozen file hash mismatch: {mismatch}")
    # Phase 10E source hashes must still match exactly before beginning 10F.
    for name, expected in lock_config.get("source_sha256", {}).items():
        current = sha256_file(ROOT / name).upper()
        if current != expected.upper():
            raise ValueError(f"Phase 10E source identity changed: {name}: {expected} != {current}")
    dev = load_scifact_dataset(droot, "dev")
    chunks = corpus_to_rag_chunks(dev.corpus, __import__("rag"))
    if len(dev.claims) != 300 or len(dev.corpus) != 5183 or len(chunks) != 45972:
        raise ValueError(f"SciFact loaded counts mismatch: {len(dev.claims)}, {len(dev.corpus)}, {len(chunks)}")
    parent = _read_json(e10 / "phase10e_remediation_report.json")
    return {"phase10e_configuration_fingerprint": lock["fingerprint_sha256"],
        "phase10e_readiness_status": parent.get("status"),
        "phase10e_historical_filename_provenance": parent.get("historical_filename_limitation", {}).get(
            "historical_filename_provenance"),
        "model_identity": {"provider": model.get("provider"), "endpoint": model.get("endpoint"),
            "model": model["model"], "digest": model["model_digest"],
            "digest_status": model.get("digest_status"), "temperature": model.get("temperature"),
            "generation_parameters": model.get("provider_configuration")},
        "dataset": {"name": "SciFact-Orig", "split": "dev", "counts": counts,
            "sha256": hashes, "retrieval_chunk_count": len(chunks)},
        "frozen_sha256": actual_frozen,
        "historical_result_sha256": historical_result_hashes(),
        "historical_scifact_metrics_path": "evaluation/scifact/scifact_metrics.py absent; current metrics.py verified",
        "git_metadata": "unavailable"}


def select_dev_claim_ids(claims: Sequence[Any], *, full_dev: bool) -> list[int]:
    """Select canonical DEV file order: all rows or exactly the first three."""
    chosen = claims if full_dev else claims[:3]
    expected = 300 if full_dev else 3
    if len(chosen) != expected:
        raise ValueError(f"Expected {expected} DEV claims, found {len(chosen)}")
    return [row.runtime.claim_id for row in chosen]

def _build_config(dataset_root: Path, claim_ids: Sequence[int], *, full_dev: bool) -> ExperimentConfig:
    return ExperimentConfig(dataset_root=str(dataset_root), split="dev", example_ids=tuple(claim_ids),
        seed=0, top_k=20, verification_evidence_limit=2, model="qwen3:4b", temperature=0.0,
        maximum_controller_cycles=1, maximum_retrieval_actions=1, maximum_verification_actions=1,
        notes={"phase": "10F", "selection_rule": "all 300 DEV claims in canonical claims_dev.jsonl order"
            if full_dev else "exact first three DEV claims in canonical claims_dev.jsonl order; launch validation only",
            "selection_algorithm": "canonical-file-prefix-v1" if not full_dev else "all-dev-file-order-v1",
            "threshold_tuning": "none", "automatic_retries": False})


def _configuration(prerequisites: Mapping[str, Any], dataset: Any, claim_ids: Sequence[int],
                   config: ExperimentConfig, provider: Any, run_id: str) -> dict[str, Any]:
    source = {name: sha256_file(ROOT / name).upper() for name in SOURCE_FILES}
    retrieval_id = {"top_k": config.top_k, "hybrid": config.hybrid_retrieval,
        "reranking": config.reranking, "verification_evidence_limit": config.verification_evidence_limit,
        "index_identity": getattr(provider, "phase10_model_snapshots", None)}
    return {"experiment": "Phase 10F full SciFact DEV A/B/C evaluation", "phase": "10F",
        "run_id": run_id, "parent_phase10e_fingerprint": prerequisites["phase10e_configuration_fingerprint"],
        "dataset": prerequisites["dataset"], "selection": {"claim_ids": list(claim_ids),
            "count": len(claim_ids), "seed": config.seed, "selection_rule": config.notes["selection_rule"],
            "selection_algorithm": config.notes["selection_algorithm"]},
        "systems": {"A": "Strong existing RAG; free-text answer; no verifier-derived classification",
            "B": "Existing RAG plus frozen Experiment B claim-level verification",
            "C": "Bounded research through real 8A/8B/8C controller, 7B dispatch, Phase 6 and explicit 7C continuation"},
        "model": prerequisites["model_identity"], "frozen_source_sha256": prerequisites["frozen_sha256"],
        "historical_result_sha256": prerequisites["historical_result_sha256"], "retrieval": retrieval_id,
        "verification": {"implementation": "frozen Experiment B", "evidence_limit": config.verification_evidence_limit,
            "targeted_retrieval_retries": 0, "answer_revision": False},
        "evaluation_schema": "phase10f-attempt-v1", "scoring_schema": "phase10f-per-example-v1",
        "failure_policy": {"automatic_retries": False, "unavailable_is_not_miss": True,
            "failed_predictions_excluded_from_valid_prediction_accuracy": True},
        "source_sha256": source, "runtime": {"python": sys.version,
            "python_implementation": platform.python_implementation(), "platform": platform.platform(),
            "random_seed": config.seed}}


def _validate_attempt(row: Any) -> None:
    if not isinstance(row, dict):
        raise ValueError("attempt must be a JSON object")
    required = {"run_id", "attempt_id", "system", "claim_id", "runtime_input",
        "configuration_fingerprint", "model_identity", "retrieval_configuration_identity",
        "execution_status", "output_availability_status", "started_at_utc", "completed_at_utc", "output"}
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(f"attempt schema missing {missing}")
    if row["system"] not in SYSTEM_NAMES:
        raise ValueError("invalid system name")
    forbidden_gold_fields = {"gold_label", "gold_evidence_doc_ids", "annotations", "rationales", "gold"}
    leaked = sorted(forbidden_gold_fields & set(row))
    if leaked:
        raise ValueError(f"gold fields present in runtime attempt record: {leaked}")
    if set(row["runtime_input"]) != {"claim_id", "claim"}:
        raise ValueError("runtime input contains unexpected fields")
    if row["runtime_input"]["claim_id"] != row["claim_id"]:
        raise ValueError("claim identity mismatch")


def _failure_category(row: Mapping[str, Any]) -> str:
    error = str(row.get("error") or "").lower()
    error_type = str(row.get("error_type") or "").lower()
    stage = str(row.get("error_stage") or "").lower()
    if row.get("execution_status") == "INTERRUPTED": return "interruption"
    if "connectionrefused" in error_type or "winerror 10061" in error or "connection refused" in error:
        return "local_model_endpoint_failure"
    if "timeout" in error_type or "timed out" in error: return "timeout"
    if "jsondecode" in error_type or "malformed" in error or "invalid json" in error: return "malformed_model_output"
    if "controller" in stage or "phase8" in stage: return "controller_failure"
    if "retrieval" in stage: return "retrieval_failure"
    if "verification" in stage: return "verification_failure"
    if row.get("execution_status") != "SUCCESS": return "other_recorded_execution_failure"
    return "none"


def _classification(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labeled = [r for r in rows if r.get("gold_label") in LABELS]
    valid = [r for r in labeled if r.get("status") == "SUCCESS" and r.get("predicted_label") in LABELS]
    matrix = {gold: {pred: 0 for pred in LABELS} for gold in LABELS}
    for row in valid:
        matrix[row["gold_label"]][row["predicted_label"]] += 1
    per_class: dict[str, Any] = {}
    for label in LABELS:
        tp = matrix[label][label]
        fp = sum(matrix[g][label] for g in LABELS if g != label)
        fn = sum(matrix[label][p] for p in LABELS if p != label)
        pden, rden = tp + fp, tp + fn
        precision = tp / pden if pden else None
        recall = tp / rden if rden else None
        f1 = (2 * precision * recall / (precision + recall) if precision is not None and recall is not None
              and precision + recall else (0.0 if precision == 0 or recall == 0 else None))
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1,
            "gold_support_valid_prediction_denominator": rden, "predicted_support": pden}
    present = [per_class[label]["f1"] for label in LABELS if per_class[label]["gold_support_valid_prediction_denominator"]]
    return {"accuracy": {"numerator": sum(r["gold_label"] == r["predicted_label"] for r in valid),
            "denominator": len(valid), "value": (sum(r["gold_label"] == r["predicted_label"] for r in valid) / len(valid)) if valid else None,
            "denominator_definition": "labeled claims with successful execution and a valid verifier label"},
        "labeled_claims": len(labeled), "valid_predictions": len(valid),
        "failed_execution_predictions_excluded": sum(r.get("status") != "SUCCESS" for r in labeled),
        "invalid_predictions_excluded": sum(r.get("status") == "SUCCESS" and r.get("predicted_label") not in LABELS for r in labeled),
        "unlabeled_claims": sum(r.get("gold_label") not in LABELS for r in rows),
        "per_class": per_class, "macro_f1": {"value": statistics.mean(present) if present else None,
            "class_denominator": len(present), "definition": "mean F1 over classes present in valid-prediction gold denominator"},
        "confusion_matrix": matrix}


def _aggregate(all_attempts: Sequence[Mapping[str, Any]], scores: Sequence[Mapping[str, Any]],
               planned_claims: int, *, resumed_attempt_count: int) -> dict[str, Any]:
    systems: dict[str, Any] = {}
    for system in SYSTEM_NAMES:
        attempts = [r for r in all_attempts if r["system"] == system]
        rows = [r for r in scores if r["system"] == system]
        failed = [r for r in attempts if r["execution_status"] != "SUCCESS"]
        labels = _classification(rows) if system != "A" else "NOT APPLICABLE"
        avail = Counter()
        for row in attempts:
            for key in ("retrieval", "evidence"):
                value = row["output_availability_status"].get(key, "OUTPUT_UNAVAILABLE")
                avail[f"{key}:{value}"] += 1
        latency = [r["latency_seconds"] for r in attempts if isinstance(r.get("latency_seconds"), (int, float))
                   and math.isfinite(r["latency_seconds"])]
        llm_values = [r["llm_calls"] for r in attempts if isinstance(r.get("llm_calls"), int)]
        success_call_values = [r["llm_calls"] for r in attempts if r["execution_status"] == "SUCCESS"
                               and isinstance(r.get("llm_calls"), int)]
        failure_categories = Counter(_failure_category(r) for r in failed)
        systems[system] = {"planned": planned_claims, "executed": len(attempts),
            "successful": len(attempts) - len(failed), "failed": len(failed),
            "execution_failure_categories": dict(failure_categories),
            "output_availability_counts": dict(avail),
            "latency_seconds": {"mean": statistics.mean(latency) if latency else None,
                "median": statistics.median(latency) if latency else None,
                "min": min(latency) if latency else None, "max": max(latency) if latency else None,
                "valid_timing_denominator": len(latency), "missing_timing_count": len(attempts) - len(latency)},
            "llm_calls": {"total": sum(llm_values) if llm_values else None,
                "reported_attempt_denominator": len(llm_values),
                "mean_per_successful_execution": statistics.mean(success_call_values) if success_call_values else None,
                "successful_execution_call_denominator": len(success_call_values)},
            "retrieval_calls": {"total": sum(r["retrieval_calls"] for r in attempts if isinstance(r.get("retrieval_calls"), int)),
                "available_count_denominator": sum(isinstance(r.get("retrieval_calls"), int) for r in attempts)},
            "verification_calls": {"total": sum(r["verification_calls"] for r in attempts if isinstance(r.get("verification_calls"), int)),
                "available_count_denominator": sum(isinstance(r.get("verification_calls"), int) for r in attempts)},
            "controller_cycles": {"total": sum(r["controller_cycles"] for r in attempts if isinstance(r.get("controller_cycles"), int)),
                "available_count_denominator": sum(isinstance(r.get("controller_cycles"), int) for r in attempts)} if system == "C" else "N/A",
            "classification": labels}
    failures = Counter(_failure_category(r) for r in all_attempts if r["execution_status"] != "SUCCESS")
    # Retrieval uses only valid retrieval records and gold evidence claims.
    retrieval: dict[str, Any] = {}
    for system in SYSTEM_NAMES:
        rows = [r for r in scores if r["system"] == system]
        valid = [r for r in rows if r["retrieval"].get("retrieval_scoring_status") == "scored"]
        gold_doc_count = sum(r["retrieval"].get("gold_evidence_document_count", 0) for r in valid)
        top1_hits, mrr = 0, []
        precision_values = []
        for row in valid:
            gold = set(row.get("gold_evidence_doc_ids") or [])
            docs = [c.get("scifact_doc_id") for c in row.get("retrieved_chunks", [])]
            rank = next((i + 1 for i, doc in enumerate(docs) if doc in gold), None)
            if rank is not None:
                mrr.append(1 / rank)
            topdocs = {doc for doc in docs[:1] if doc is not None}
            top1_hits += len(topdocs & gold)
            if topdocs:
                precision_values.append(len(topdocs & gold) / len(topdocs))
        retrieval[system] = {"gold_evidence_claims_with_valid_output": len(valid),
            "gold_document_denominator": gold_doc_count,
            "Recall@1": {"numerator": top1_hits, "denominator": gold_doc_count,
                "value": top1_hits / gold_doc_count if gold_doc_count else None},
            "MRR": {"value": statistics.mean(mrr) if mrr else None, "claim_denominator": len(mrr)},
            "document_precision@1_macro": {"value": statistics.mean(precision_values) if precision_values else None,
                "claim_denominator": len(precision_values)},
            "unavailable_or_failed_retrievals_excluded": sum(
                r.get("output_availability_status", {}).get("retrieval", OutputState.OUTPUT_UNAVAILABLE.value)
                not in {OutputState.VALID_EMPTY.value, OutputState.VALID_NONEMPTY.value}
                for r in all_attempts if r["system"] == system),
            "interpretation": "document overlap with annotated gold evidence; not entailment or factual correctness"}
    evidence: dict[str, Any] = {}
    for system in ("B", "C"):
        vals = [r["citation"] for r in scores if r["system"] == system and r.get("citation") is not None]
        evidence[system] = {"status": "scored" if vals else "N/A",
            "valid_scored_claims": len(vals),
            **{key: statistics.mean(v[key] for v in vals if v.get(key) is not None)
               if any(v.get(key) is not None for v in vals) else None for key in ("precision", "recall", "f1")},
            "definition": "document-level evidence overlap against annotated gold evidence; not entailment"}
    common_valid = [r for r in scores if r["system"] in {"B", "C"} and r.get("gold_label") in LABELS
                    and r.get("status") == "SUCCESS" and r.get("predicted_label") in LABELS]
    paired = paired_accuracy_bootstrap(common_valid, seed=0)
    return {"planned_claims": planned_claims, "planned_system_claim_attempts": planned_claims * 3,
        "executed_attempts": len(all_attempts), "successful_attempts": sum(r["execution_status"] == "SUCCESS" for r in all_attempts),
        "failed_attempts": sum(r["execution_status"] != "SUCCESS" for r in all_attempts),
        "unlabeled_claims": sum(r.get("gold_label") is None for r in scores if r["system"] == "A"),
        "claims_with_gold_evidence": sum(bool(r.get("gold_evidence_doc_ids")) for r in scores if r["system"] == "A"),
        "systems": systems, "retrieval": retrieval, "evidence_citation": evidence,
        "failure_categories": dict(failures), "resumed_attempts_this_invocation": resumed_attempt_count,
        "retries": 0, "automatic_retries": False,
        "paired_accuracy_bootstrap": paired,
        "classification_denominator_rule": "valid predictions only; failed, missing, invalid, and unlabeled outcomes are separately reported, never counted as wrong predictions",
        "coverage_metric": "N/C — no registered definition"}


def _phase10f_resume_plan(expected_pairs: Sequence[tuple[Any, str]],
                          attempts: Sequence[Mapping[str, Any]], *,
                          expected_run_id: str) -> dict[str, Any]:
    """Classify persisted Phase10F attempts using the canonical claim_id key.

    The Phase10C generic helper still consumes its historical example_id schema;
    Phase10F runtime attempt records use claim_id and never alias it to example_id.
    Failed/interrupted rows are terminal persisted outcomes under the no-retry policy.
    """
    expected = list(expected_pairs)
    if len(expected) != len(set(expected)):
        raise ValueError("Duplicate expected claim/system pairs")
    expected_set = set(expected)
    seen: dict[tuple[Any, str], Mapping[str, Any]] = {}
    duplicate_pairs: list[tuple[Any, str]] = []
    for row in attempts:
        if not isinstance(row, Mapping):
            raise ValueError("Malformed attempt record: expected an object")
        missing = [name for name in ("claim_id", "system", "run_id", "attempt_id", "execution_status")
                   if row.get(name) is None]
        if missing:
            raise ValueError(f"Malformed Phase10F attempt missing identifiers/status: {missing}")
        if row["run_id"] != expected_run_id:
            raise ValueError(f"Incompatible run identity for attempt {row['attempt_id']}: {row['run_id']!r}")
        pair = (row["claim_id"], row["system"])
        if pair not in expected_set:
            raise ValueError(f"Unexpected claim/system pair in attempt {row['attempt_id']}: {pair!r}")
        if row["system"] not in SYSTEM_NAMES:
            raise ValueError(f"Invalid system in attempt {row['attempt_id']}: {row['system']!r}")
        if pair in seen:
            duplicate_pairs.append(pair)
        else:
            seen[pair] = row
    if duplicate_pairs:
        raise ValueError(f"Duplicate Phase10F claim/system attempts: {duplicate_pairs!r}")
    successful = [pair for pair in expected if pair in seen and seen[pair].get("execution_status") == "SUCCESS"]
    failed = [pair for pair in expected if pair in seen and seen[pair].get("execution_status") != "SUCCESS"]
    missing_pairs = [pair for pair in expected if pair not in seen]
    # No retry is authorized. Only never-persisted pairs are scheduled.
    return {"planned_pairs": expected, "completed_pairs": successful, "failed_terminal_pairs": failed,
        "missing_pairs": missing_pairs, "scheduled_pairs": missing_pairs,
        "skipped_pairs": successful + failed, "duplicate_pairs": [],
        "existing_attempt_count": len(attempts)}


def _phase10f_normalized_runner_identity() -> str:
    """Stable source identity with only its own embedded digest normalized."""
    source = Path(__file__).read_bytes()
    pattern = rb'PHASE10F_REMEDIATION_SOURCE_IDENTITY = "[A-Fa-f0-9]{64}"'
    normalized, count = re.subn(pattern,
        b'PHASE10F_REMEDIATION_SOURCE_IDENTITY = "<normalized>"', source)
    if count != 1:
        raise ValueError("Cannot establish normalized Phase10F runner source identity")
    return hashlib.sha256(normalized).hexdigest().upper()


def _phase10f_resume_configuration(stored_lock: Mapping[str, Any],
                                   current_lock: Mapping[str, Any]) -> dict[str, Any]:
    """Compare locks strictly, with a pinned migration for the known old runner."""
    stored_config = stored_lock.get("configuration")
    current_config = current_lock.get("configuration")
    stored_fingerprint = stored_lock.get("fingerprint_sha256")
    current_fingerprint = current_lock.get("fingerprint_sha256")
    if not isinstance(stored_config, dict) or not isinstance(current_config, dict):
        raise ValueError("Incompatible resume configuration (configuration_lock missing configuration)")
    if canonical_fingerprint(stored_config) != stored_fingerprint:
        raise ValueError("Corrupt stored configuration lock fingerprint")
    if canonical_fingerprint(current_config) != current_fingerprint:
        raise ValueError("Corrupt current configuration lock fingerprint")
    stored_runner = stored_config.get("source_sha256", {}).get(PHASE10F_RUNNER_SOURCE_PATH)
    current_runner = current_config.get("source_sha256", {}).get(PHASE10F_RUNNER_SOURCE_PATH)
    if not current_runner or current_runner.upper() != hashlib.sha256(Path(__file__).read_bytes()).hexdigest().upper():
        raise ValueError("Incompatible resume configuration (current runner source hash does not match code)")
    if stored_runner == current_runner:
        validate_resume_configuration(stored_config, current_config)
        return {"compatible": True, "stored_fingerprint": stored_fingerprint,
            "stored_runner_sha256": stored_runner, "current_runner_sha256": current_runner,
            "runner_source_migration_only": False, "configuration_identity_preserved": True}
    if stored_runner != PHASE10F_PRE_REMEDIATION_RUNNER_SHA256:
        raise ValueError("Incompatible resume configuration (unrecognized stored runner source identity)")
    if _phase10f_normalized_runner_identity() != PHASE10F_REMEDIATION_SOURCE_IDENTITY:
        raise ValueError("Incompatible resume configuration (remediation runner source identity changed)")
    stored_rest, current_rest = json.loads(json.dumps(stored_config)), json.loads(json.dumps(current_config))
    stored_rest.get("source_sha256", {}).pop(PHASE10F_RUNNER_SOURCE_PATH, None)
    current_rest.get("source_sha256", {}).pop(PHASE10F_RUNNER_SOURCE_PATH, None)
    # Fail field-by-field on every dataset, selection, model, system, prompt,
    # retrieval, frozen-source, and runtime difference other than this pinned
    # orchestration/audit runner migration.
    validate_resume_configuration(stored_rest, current_rest)
    if stored_config.get("run_id") != current_config.get("run_id"):
        raise ValueError("Incompatible resume configuration (run_id differs)")
    return {"compatible": True, "stored_fingerprint": stored_fingerprint,
        "stored_runner_sha256": stored_runner, "current_runner_sha256": current_runner,
        "runner_source_migration_only": True, "configuration_identity_preserved": True}


def _phase10f_interrupted_attempt(inflight: Mapping[str, Any], *, run_id: str,
                                  configuration_fingerprint: str,
                                  model_identity: Mapping[str, Any],
                                  retrieval_identity: Mapping[str, Any]) -> dict[str, Any]:
    """Create a native Phase10F interrupted row without an example_id alias."""
    return {"run_id": run_id, "attempt_id": inflight["attempt_id"],
        "system": inflight["system"], "claim_id": inflight["claim_id"],
        "runtime_input": inflight["runtime_input"],
        "configuration_fingerprint": configuration_fingerprint,
        "model_identity": dict(model_identity),
        "retrieval_configuration_identity": dict(retrieval_identity),
        "execution_status": "INTERRUPTED",
        "output_availability_status": {"retrieval": OutputState.OUTPUT_UNAVAILABLE.value,
            "evidence": OutputState.OUTPUT_UNAVAILABLE.value},
        "started_at_utc": inflight["started_at_utc"], "completed_at_utc": _now(),
        "latency_seconds": None, "llm_calls": None, "retrieval_calls": None,
        "verification_calls": None, "controller_cycles": None,
        "selected_evidence_doc_ids": [], "retrieved_evidence_ids": [], "ranked_retrieval_ids": [],
        "provenance": [], "error": "Process interruption; callback completion unknown",
        "error_type": "InterruptedAttempt", "error_stage": "system_callback",
        "failure_category": "interruption",
        "output": {"predicted_label": None, "retrieved_chunks": [],
            "selected_evidence_doc_ids": [], "assessments": [], "unsupported_claims": None,
            "verification_calls": None, "llm_calls": None, "retrieval_calls": None,
            "controller_cycles": None, "raw_output": None,
            "error": "Process interruption; callback completion unknown", "status": "INTERRUPTED",
            "error_type": "InterruptedAttempt", "error_stage": "system_callback",
            "error_traceback": None,
            "retrieval_output_status": OutputState.OUTPUT_UNAVAILABLE.value,
            "evidence_output_status": OutputState.OUTPUT_UNAVAILABLE.value}}


def audit_c_traces(scores: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Audit Phase8 traces from persisted validation, dispatch, and continuation IDs."""
    crows = [r for r in scores if r.get("system") == "C" and r.get("status") == "SUCCESS"]
    audits: list[dict[str, Any]] = []

    def as_items(value: Any) -> list[Mapping[str, Any]]:
        if isinstance(value, Mapping):
            return [value]
        if isinstance(value, list):
            return [v for v in value if isinstance(v, Mapping)]
        return []

    for row in crows:
        raw = ((row.get("output") or {}).get("raw_output") or {})
        raw = raw if isinstance(raw, Mapping) else {}
        trace_runs = as_items(raw.get("controller_traces"))
        cycles: list[Mapping[str, Any]] = []
        for trace_run in trace_runs:
            trace = trace_run.get("trace")
            if isinstance(trace, Mapping):
                cycles.extend(c for c in trace.get("cycles", []) if isinstance(c, Mapping))
        actions = [((c.get("decision") or {}).get("proposal") or {}).get("action_type") for c in cycles]
        state = raw.get("research_state") if isinstance(raw.get("research_state"), Mapping) else {}
        executions = as_items(state.get("task_executions"))
        loops = as_items(state.get("research_loops"))
        stage: dict[str, str] = {}
        evidence: dict[str, Any] = {"actions": actions, "controller_cycle_count": len(cycles),
            "execution_ids": [e.get("execution_id") for e in executions if e.get("execution_id")],
            "dispatch_capabilities": [], "continuation_authorization_ids": []}
        if not cycles:
            for key in ("phase_8a_validation", "phase_8b_execution", "phase_7b_dispatch"):
                stage[key] = "FAIL"
        else:
            validations = [c.get("validation") or {} for c in cycles]
            stage["phase_8a_validation"] = "PASS" if all(v.get("validation_status") == "ACCEPTED" for v in validations) else "FAIL"
            stage["phase_8b_execution"] = "PASS" if all(
                (c.get("execution") or {}).get("status") == "COMPLETED" and c.get("approval_id")
                for c in cycles) else "FAIL"
            dispatches = [((c.get("execution") or {}).get("dispatch") or {}) for c in cycles]
            expected_capabilities = []
            for c, dispatch in zip(cycles, dispatches):
                action = ((c.get("decision") or {}).get("proposal") or {}).get("action_type")
                expected_capability = {"RETRIEVE_EVIDENCE": "local_retrieval",
                    "VERIFY_CLAIM": "evidence_verification"}.get(action)
                expected_capabilities.append(expected_capability)
                if dispatch.get("capability"):
                    evidence["dispatch_capabilities"].append(dispatch["capability"])
            stage["phase_7b_dispatch"] = "PASS" if all(
                d.get("status") == "COMPLETED" and d.get("capability") and
                (expected is None or d.get("capability") == expected)
                for d, expected in zip(dispatches, expected_capabilities)) else "FAIL"
        retrieve_actions = [c for c in cycles if ((c.get("decision") or {}).get("proposal") or {}).get("action_type") == "RETRIEVE_EVIDENCE"]
        verify_actions = [c for c in cycles if ((c.get("decision") or {}).get("proposal") or {}).get("action_type") == "VERIFY_CLAIM"]
        stage["phase_6_retrieval"] = ("PASS" if retrieve_actions and all(
            (((c.get("execution") or {}).get("dispatch") or {}).get("capability") == "local_retrieval" and
             ((c.get("execution") or {}).get("dispatch") or {}).get("status") == "COMPLETED")
            for c in retrieve_actions) else "FAIL") if retrieve_actions else "FAIL"
        evidence["phase_6_retrieval_required"] = bool(retrieve_actions)
        if verify_actions:
            stage["phase_6_verification"] = "PASS" if all(
                (((c.get("execution") or {}).get("dispatch") or {}).get("capability") == "evidence_verification" and
                 ((c.get("execution") or {}).get("dispatch") or {}).get("status") == "COMPLETED")
                for c in verify_actions) else "FAIL"
        elif row.get("selected_evidence_doc_ids"):
            stage["phase_6_verification"] = "FAIL"
        else:
            stage["phase_6_verification"] = "NOT_APPLICABLE"
        evidence["phase_6_verification_required"] = stage["phase_6_verification"] != "NOT_APPLICABLE"
        if verify_actions:
            retrieve_execs = [e for e in executions if e.get("status") == "COMPLETED" and
                              any(cycle is not None for cycle in retrieve_actions) and
                              e.get("execution_id")]
            # The actual continuation relation is represented by execution IDs
            # and authorization entries in persisted research state.
            verify_execs = [e for e in executions if e.get("status") == "COMPLETED" and
                            e.get("continuation_authorization_id")]
            links = []
            for later in verify_execs:
                for loop in loops:
                    for auth in as_items(loop.get("continuation_authorizations")):
                        if (auth.get("authorization_id") == later.get("continuation_authorization_id") and
                            auth.get("previous_execution_id") == later.get("continuation_of_execution_id") and
                            later.get("task_id") == auth.get("task_id") and
                            later.get("run_id") == auth.get("run_id")):
                            previous = next((e for e in retrieve_execs if e.get("execution_id") ==
                                             later.get("continuation_of_execution_id")), None)
                            if previous is not None:
                                links.append({"authorization_id": auth["authorization_id"],
                                    "previous_execution_id": previous["execution_id"],
                                    "continued_execution_id": later["execution_id"],
                                    "task_id": later.get("task_id"), "run_id": later.get("run_id")})
            stage["continuation_authorization"] = "PASS" if links else "FAIL"
            evidence["continuation_links"] = links
            evidence["continuation_authorization_ids"] = [x["authorization_id"] for x in links]
            transition_ok = any(
                t.get("from_stage") == "RESEARCH" and t.get("to_stage") == "EVIDENCE_REVIEW" and
                t.get("authority") == "researcher"
                for loop in loops for t in as_items(loop.get("transitions")))
            stage["researcher_transition"] = "PASS" if transition_ok else "FAIL"
            evidence["researcher_transition"] = "RESEARCH -> EVIDENCE_REVIEW, authority=researcher" if transition_ok else None
        else:
            stage["continuation_authorization"] = "NOT_APPLICABLE"
            stage["researcher_transition"] = "NOT_APPLICABLE"
            evidence["continuation_links"] = []
        required_task_statuses = [e.get("status") for e in executions]
        completion_ok = row.get("status") == "SUCCESS" and bool(cycles) and bool(executions) and all(
            value == "COMPLETED" for value in required_task_statuses) and stage.get("phase_7b_dispatch") == "PASS"
        stage["completion"] = "PASS" if completion_ok else "FAIL"
        overall = "PASS" if all(value in {"PASS", "NOT_APPLICABLE"} for value in stage.values()) else "FAIL"
        audits.append({"claim_id": row.get("claim_id"),
            "status": "PASS" if overall == "PASS" else "LIMITED", "overall": overall,
            "stages": stage, "evidence": evidence,
            "continuation_authorization_required": bool(verify_actions),
            "continuation_authorization_trace_present": stage["continuation_authorization"] == "PASS",
            "controller_runs": len(trace_runs), "cycles": len(cycles),
            "task_execution_count": len(executions)})
    return {"successful_c_executions": len(crows), "audited": len(audits),
        "pass": sum(a["status"] == "PASS" for a in audits),
        "limited": sum(a["status"] != "PASS" for a in audits), "records": audits,
        "definition": "Structured audit of persisted 8A validation, 8B execution/approval, 7B capability dispatch, Phase6 retrieval/verification, linked 7C continuation authorization, researcher transition, and completed task/output state."}


def run_phase10f(*, output_dir: str | Path, full_dev: bool,
                 dataset_root: str | Path = DATA_DIR,
                 phase10e_dir: str | Path = PHASE10E_DIR,
                 resume: bool = False, provider: Any = None) -> dict[str, Any]:
    prerequisites = validate_prerequisites(dataset_root=dataset_root, phase10e_dir=phase10e_dir)
    dataset = load_scifact_dataset(dataset_root, "dev")
    claims = list(dataset.claims)
    claim_ids = select_dev_claim_ids(claims, full_dev=full_dev)
    by_id = {row.runtime.claim_id: row for row in claims}
    config = _build_config(Path(dataset_root).resolve(), claim_ids, full_dev=full_dev)
    out = Path(output_dir).resolve()
    manifest_path = out / ("full_run_manifest.json" if full_dev else "launch_manifest.json")
    attempts_path, scores_path = out / "attempt_records.jsonl", out / "per_example.jsonl"
    inflight_path = out / "inflight_attempt.json"
    if provider is None:
        provider = build_provider(dataset)
    run_id = None
    prior_manifest = None
    if resume:
        prior_manifest = load_resume_manifest(manifest_path)
        run_id = prior_manifest.get("run_id")
        if not run_id:
            raise ValueError("Incompatible resume configuration (run_id: <missing>)")
    else:
        if out.exists() and any(out.iterdir()):
            raise FileExistsError(f"Refusing to overwrite existing Phase 10F output: {out}")
        out.mkdir(parents=True, exist_ok=True)
        run_id = uuid.uuid4().hex
    config_data = _configuration(prerequisites, dataset, claim_ids, config, provider, run_id)
    lock = make_configuration_lock(config_data)
    fingerprint = lock["fingerprint_sha256"]
    recovery = []
    existing: list[dict[str, Any]] = []
    scored: list[dict[str, Any]] = []
    resume_compatibility: dict[str, Any] | None = None
    resume_plan: dict[str, Any] | None = None
    if resume:
        stored_lock = prior_manifest.get("configuration_lock", {})
        resume_compatibility = _phase10f_resume_configuration(stored_lock, lock)
        if prior_manifest.get("mode") != ("full_dev" if full_dev else "launch_validation"):
            raise ValueError("Incompatible resume configuration (run mode differs)")
        # Historical run identity and configuration fingerprint are retained.
        lock = stored_lock
        fingerprint = stored_lock["fingerprint_sha256"]
        existing, r = recover_jsonl(attempts_path, validator=_validate_attempt,
                                    unique_key="attempt_id")
        recovery.append(r.__dict__)
        scored, r = recover_jsonl(scores_path, validator=validate_scored_record,
                                  unique_key=claim_system_key)
        recovery.append(r.__dict__)
        if inflight_path.exists():
            pending = _read_json(inflight_path)
            if pending.get("run_id") != run_id:
                raise ValueError("Incompatible run identity in inflight marker")
            pending_pair = (pending["claim_id"], pending["system"])
            existing_pairs = {(item.get("claim_id"), item.get("system")) for item in existing}
            if pending_pair not in existing_pairs:
                interrupted = _phase10f_interrupted_attempt(pending, run_id=run_id,
                    configuration_fingerprint=fingerprint,
                    model_identity=prerequisites["model_identity"],
                    retrieval_identity=lock["configuration"]["retrieval"])
                _validate_attempt(interrupted)
                append_jsonl(attempts_path, interrupted)
                existing.append(interrupted)
            inflight_path.unlink()
    else:
        if manifest_path.exists() or attempts_path.exists() or scores_path.exists():
            raise FileExistsError("Phase 10F output files already exist")
    resumed_count = len(existing)
    if prior_manifest is None:
        manifest = {"phase": "10F", "mode": "full_dev" if full_dev else "launch_validation",
            "run_id": run_id, "created_at_utc": _now(), "status": "RUNNING",
            "dataset": prerequisites["dataset"], "selection": lock["configuration"]["selection"],
            "configuration_lock": lock,
            "phase10e_configuration_fingerprint": prerequisites["phase10e_configuration_fingerprint"],
            "historical_result_sha256": prerequisites["historical_result_sha256"],
            "model_identity": prerequisites["model_identity"],
            "system_identity": lock["configuration"]["systems"],
            "retrieval_configuration_identity": lock["configuration"]["retrieval"],
            "gold_leakage": {"runtime_input_fields": ["claim_id", "claim"],
                "gold_join": "after all A/B/C attempts for claim are durable", "gold_in_runtime_attempts": False},
            "retry_policy": {"automatic_retries": False, "failed_attempts_are_terminal": True},
            "planned_attempts": len(claim_ids) * len(SYSTEM_NAMES), "recovery": recovery}
        atomic_write_json(manifest_path, manifest)
        atomic_write_json(out / "configuration_lock.json", lock)
        atomic_write_json(out / "model_identity.json", prerequisites["model_identity"])
        atomic_write_json(out / "dataset_identity.json", prerequisites["dataset"])
    else:
        manifest = prior_manifest
    expected_pairs = [(cid, system) for cid in claim_ids for system in SYSTEM_NAMES]
    resume_plan = _phase10f_resume_plan(expected_pairs, existing, expected_run_id=run_id)
    pending_pairs = set(resume_plan["scheduled_pairs"])
    by_pair = {(row["claim_id"], row["system"]): row for row in existing}
    # The generic scored-record schema still calls its claim key example_id;
    # runtime attempts and resume identity are canonical claim_id throughout.
    score_pairs = {(row["example_id"], row["system"]): row for row in scored}
    executed_this_call = 0
    for claim_id in claim_ids:
        record = by_id[claim_id]
        runtime = runtime_claim_payload(record.runtime)
        if set(runtime) != {"claim_id", "claim"}:
            raise AssertionError("Runtime payload is not gold-free minimal input")
        outputs: dict[str, SystemOutput] = {}
        for system in SYSTEM_NAMES:
            old = by_pair.get((claim_id, system))
            if old is not None:
                outputs[system] = _output_from_record(old["output"])
                outputs[system].status = old["execution_status"]
                outputs[system].error = old.get("error")
                continue
            if (claim_id, system) not in pending_pairs:
                continue
            started = _now()
            inflight = {"run_id": run_id, "attempt_id": uuid.uuid4().hex,
                "claim_id": claim_id, "system": system, "runtime_input": runtime,
                "started_at_utc": started, "configuration_fingerprint": fingerprint}
            atomic_write_json(inflight_path, inflight)
            output, elapsed = _safe_call(SYSTEMS[system], runtime, provider, config)
            completed = _now()
            output_row = _output_to_record(output)
            attempt = {"run_id": run_id, "attempt_id": inflight["attempt_id"],
                "system": system, "claim_id": claim_id, "runtime_input": runtime,
                "configuration_fingerprint": fingerprint,
                "model_identity": prerequisites["model_identity"],
                "retrieval_configuration_identity": lock["configuration"]["retrieval"],
                "execution_status": output.status,
                "output_availability_status": {"retrieval": output.retrieval_output_status,
                    "evidence": output.evidence_output_status},
                "started_at_utc": started, "completed_at_utc": completed,
                "latency_seconds": elapsed, "llm_calls": output.llm_calls,
                "retrieval_calls": output.retrieval_calls,
                "verification_calls": output.verification_calls,
                "controller_cycles": output.controller_cycles,
                "selected_evidence_doc_ids": output.selected_evidence_doc_ids,
                "retrieved_evidence_ids": [c.get("evidence_id") for c in output.retrieved_chunks],
                "ranked_retrieval_ids": [{"rank": c.get("rank"), "evidence_id": c.get("evidence_id"),
                    "chunk_id": c.get("chunk_id"), "scifact_doc_id": c.get("scifact_doc_id")}
                    for c in output.retrieved_chunks],
                "provenance": _provenance(output.raw_output), "error": output.error,
                "error_type": output.error_type, "error_stage": output.error_stage,
                "output": output_row, "failure_category": _failure_category({
                    "execution_status": output.status, "error": output.error,
                    "error_type": output.error_type, "error_stage": output.error_stage})}
            append_jsonl(attempts_path, attempt)
            inflight_path.unlink(missing_ok=True)
            existing.append(attempt)
            by_pair[(claim_id, system)] = attempt
            outputs[system] = output
            executed_this_call += 1
            print(f"Phase 10F {claim_id} {system}: {output.status} ({elapsed:.1f}s)", flush=True)
        if len(outputs) != 3:
            continue
        # Gold is joined after all three system outputs have been recorded.
        gold_label, gold_status = gold_claim_label(record.annotations)
        gold_docs = sorted({e.doc_id for e in record.annotations.evidence or ()}) \
            if record.annotations.evidence is not None else None
        for system in SYSTEM_NAMES:
            if (claim_id, system) in score_pairs:
                continue
            output = outputs[system]
            retrieval_ok = output.retrieval_output_status in {OutputState.VALID_EMPTY.value,
                                                               OutputState.VALID_NONEMPTY.value}
            retrieval = retrieval_metrics_for_claim(record.annotations, output.retrieved_chunks) if retrieval_ok else {
                "retrieval_scoring_status": output.retrieval_output_status,
                "reciprocal_rank": None, "recall_at_k": {},
                "gold_evidence_document_count": len(gold_docs or []),
                "gold_evidence_documents_retrieved": None}
            attribution = _attribution_metrics(record.annotations, output.selected_evidence_doc_ids,
                                               output.retrieved_chunks)
            evidence_ok = output.evidence_output_status in {OutputState.VALID_EMPTY.value,
                                                             OutputState.VALID_NONEMPTY.value}
            if not evidence_ok:
                attribution["citation"] = None
                attribution["status"] = "evidence_output_unavailable_or_failed"
            citation = attribution.get("citation") if evidence_ok else None
            score = {"example_id": claim_id, "claim_id": claim_id, "system": system,
                "split": "dev", "runtime_input": runtime, "status": output.status,
                "output": _output_to_record(output), "predicted_label": output.predicted_label,
                "gold_label": gold_label, "gold_label_status": gold_status,
                "gold_evidence_doc_ids": gold_docs,
                "retrieval_output_status": output.retrieval_output_status,
                "evidence_output_status": output.evidence_output_status,
                "retrieved_chunks": _json_safe(output.retrieved_chunks),
                "retrieved_evidence_ids": [c.get("evidence_id") for c in output.retrieved_chunks],
                "selected_evidence_doc_ids": output.selected_evidence_doc_ids,
                "latency_seconds": by_pair[(claim_id, system)].get("latency_seconds"),
                "llm_calls": output.llm_calls, "retrieval_calls": output.retrieval_calls,
                "verification_calls": output.verification_calls, "controller_cycles": output.controller_cycles,
                "provenance": _provenance(output.raw_output), "error": output.error,
                "error_type": output.error_type, "error_stage": output.error_stage,
                "scorable": {"classification": gold_label is not None,
                    "classification_reason": None if gold_label is not None else gold_status,
                    "retrieval": retrieval.get("retrieval_scoring_status") == "scored",
                    "retrieval_reason": retrieval.get("retrieval_scoring_status")},
                "retrieval": retrieval, "attribution": attribution, "citation": citation}
            append_jsonl(scores_path, score)
            scored.append(score)
            score_pairs[(claim_id, system)] = score
    # Re-read and validate durable files before computing aggregates.
    attempts, attempt_report = recover_jsonl(attempts_path, validator=_validate_attempt,
                                              unique_key="attempt_id")
    scored, scored_report = recover_jsonl(scores_path, validator=validate_scored_record,
        unique_key=claim_system_key)
    if len(attempts) != len(set((r["claim_id"], r["system"]) for r in attempts)):
        raise ValueError("Duplicate claim/system attempt discovered after persistence")
    metrics = _aggregate(attempts, scored, len(claim_ids), resumed_attempt_count=resumed_count)
    trace_audit = audit_c_traces(scored)
    done = len(attempts) == len(expected_pairs) and len(scored) == len(expected_pairs)
    manifest.update({"status": "COMPLETED" if done else "PARTIAL",
        "updated_at_utc": _now(), "executed_this_invocation": executed_this_call,
        "attempt_count": len(attempts), "scored_count": len(scored),
        "execution_counts": {s: {"planned": len(claim_ids),
            "executed": sum(r["system"] == s for r in attempts),
            "successful": sum(r["system"] == s and r["execution_status"] == "SUCCESS" for r in attempts),
            "failed": sum(r["system"] == s and r["execution_status"] != "SUCCESS" for r in attempts)}
            for s in SYSTEM_NAMES},
        "jsonl_integrity": {"attempts": attempt_report.__dict__, "scores": scored_report.__dict__},
        "gold_leakage_audit": {"runtime_attempt_gold_fields": False,
            "gold_join_after_system_execution": True},
        "trace_audit_summary": {"successful_c": trace_audit["successful_c_executions"],
            "audited_c": trace_audit["audited"], "pass": trace_audit["pass"],
            "limited": trace_audit["limited"]}})
    manifest["historical_result_hashes_unchanged"] = verify_historical_result_hashes(prerequisites["historical_result_sha256"])
    if not manifest["historical_result_hashes_unchanged"]:
        manifest["status"] = "INTEGRITY_FAILURE"
    atomic_write_json(manifest_path, manifest)
    atomic_write_json(out / "per_system_summaries.json", metrics["systems"])
    atomic_write_json(out / "retrieval_metrics.json", metrics["retrieval"])
    atomic_write_json(out / "classification_metrics.json", {
        s: ("NOT APPLICABLE" if s == "A" else _classification([r for r in scored if r["system"] == s]))
        for s in SYSTEM_NAMES})
    atomic_write_json(out / "evidence_citation_metrics.json", metrics["evidence_citation"])
    atomic_write_json(out / "efficiency_metrics.json", {s: metrics["systems"][s] for s in SYSTEM_NAMES})
    atomic_write_json(out / "failure_accounting.json", {"planned": len(expected_pairs),
        "executed": len(attempts), "successful": metrics["successful_attempts"],
        "failed": metrics["failed_attempts"], "by_category": metrics["failure_categories"],
        "by_system": {s: {"planned": len(claim_ids), "executed": metrics["systems"][s]["executed"],
            "successful": metrics["systems"][s]["successful"], "failed": metrics["systems"][s]["failed"],
            "output_availability": metrics["systems"][s]["output_availability_counts"]}
            for s in SYSTEM_NAMES}})
    atomic_write_json(out / "c_trace_audit.json", trace_audit)
    atomic_write_json(out / "resume_audit.json", {"resume_invocation": bool(resume),
        "attempts_present_at_start": resumed_count, "attempts_executed_this_invocation": executed_this_call,
        "pending_attempts_after_run": len(expected_pairs) - len(attempts),
        "resume_plan": resume_plan, "configuration_compatibility": resume_compatibility,
        "duplicate_pairs": [], "automatic_retries": False,
        "new_inference_calls": executed_this_call})
    atomic_write_json(out / "final_metrics.json", metrics)
    report = {"phase": "10F", "status": manifest["status"],
        "scope": "full SciFact DEV" if full_dev else "3-claim operational launch validation",
        "run_id": run_id, "configuration_fingerprint": fingerprint,
        "planned_attempts": len(expected_pairs), "attempts": len(attempts), "scored": len(scored),
        "metrics": metrics, "trace_audit": trace_audit,
        "limitations": ["Historical filename provenance for evaluation/scifact/scifact_metrics.py is NOT VERIFIED; current metrics.py bytes are verified."]}
    atomic_write_json(out / "final_report.json", report)
    (out / "final_report.md").write_text("# Phase 10F Run Report\n\n" +
        json.dumps({k: v for k, v in report.items() if k not in {"metrics", "trace_audit"}}, indent=2) + "\n",
        encoding="utf-8")
    return {"status": manifest["status"], "run_id": run_id,
        "configuration_fingerprint": fingerprint, "planned_attempts": len(expected_pairs),
        "attempts": len(attempts), "scored": len(scored),
        "executed_this_invocation": executed_this_call, "metrics": metrics,
        "trace_audit": trace_audit, "manifest_path": str(manifest_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("launch", "full", "resume-launch", "resume-full"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dataset-root", default=str(DATA_DIR))
    parser.add_argument("--phase10e-dir", default=str(PHASE10E_DIR))
    args = parser.parse_args()
    full = args.mode.endswith("full")
    resume = args.mode.startswith("resume")
    out = Path(args.output_dir or ROOT / "evaluation/experiments/results" /
        ("phase10_full_dev" if full else "phase10f_launch_validation"))
    out.parent.mkdir(parents=True, exist_ok=True)
    with exclusive_run_lock(out.parent / f".{out.name}.writer.lock"):
        result = run_phase10f(output_dir=out, full_dev=full,
            dataset_root=args.dataset_root, phase10e_dir=args.phase10e_dir, resume=resume)
    print(json.dumps({k: v for k, v in result.items() if k not in {"metrics", "trace_audit"}}, indent=2))


if __name__ == "__main__":
    main()
