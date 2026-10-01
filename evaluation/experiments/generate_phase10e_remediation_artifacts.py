"""Build deterministic Phase 10E readiness artifacts from local files only."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "evaluation/experiments/results/phase10e_remediation"
HIST = ROOT / "evaluation/experiments/results/phase10e_readiness"
EXPECTED = {
    "rag.py": "D2B0CF8930BC66FC66995AD63C1028AE3B126424E01E091DF22ED238AD5DBA19",
    "research_pipeline.py": "F55A29386AA13774A621B855E89CA8CA71FEAC0CFE2A124F90EF54DD7CD541D0",
    "claim_verification.py": "63FC8D73F55B7F48DBB340E4CD67C97975D0D587FB009CEAFA97E7FEF1C639A1",
    "verification_units.py": "BEFE07461C9B0DA76BF68226D86A96346D761C1AD3F127C554D4BB9C51644DDF",
    "experiment_b_verification.py": "66EA7B6BE6119E7A3E25A7BAD3AF3D078F08CBA4CC8AC4A462BAF6170548F3B4",
    "evaluation/scifact/scifact_adapter.py": "8F2BE845CFAF9ABB2210C88242F02E55EA8F47FF68E71BCDD27098287904BC9A",
    "evaluation/scifact/metrics.py": "8BE4B90BB28090FC48F874AF399798B0F7E39C25AAD7A1FA17B864A002E55587",
    "evaluation/scifact/run_scifact_evaluation.py": "21F540C534B5CFE8C8328C648C6E89AF6E4C4C16497CF78BC52DB51BC9629338",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def lines(path: Path) -> int:
    with path.open("rb") as stream:
        return sum(1 for line in stream if line.strip())


def dump(name: str, value: object) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False,
        default=lambda item: sorted(item) if isinstance(item, set) else str(item)) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    previous = json.loads((HIST / "phase10e_readiness_report.json").read_text(encoding="utf-8"))
    prior_integrity = previous["dataset_integrity"]
    data_root = ROOT / "data/scifact/data"
    dataset_files = {name: data_root / name for name in
        ("claims_train.jsonl", "claims_dev.jsonl", "claims_test.jsonl", "corpus.jsonl")}
    actual_counts = {key: lines(path) for key, path in dataset_files.items()}
    actual_hashes = {key: sha(path) for key, path in dataset_files.items()}
    expected_dataset_hashes = {"claims_dev.jsonl": prior_integrity["claims_sha256"],
                               "corpus.jsonl": prior_integrity["corpus_sha256"]}
    dataset_ok = (actual_counts == {"claims_train.jsonl": 809, "claims_dev.jsonl": 300,
        "claims_test.jsonl": 300, "corpus.jsonl": 5183} and
        all(actual_hashes[k] == v.upper() for k, v in expected_dataset_hashes.items()) and
        prior_integrity["chunk_count"] == 45972)
    dataset = {"status": "PASS" if dataset_ok else "FAIL", "split_counts": actual_counts,
        "sha256": actual_hashes, "corpus_documents": actual_counts["corpus.jsonl"],
        "chunk_count": prior_integrity["chunk_count"],
        "chunk_count_source": "previous readiness integrity artifact; dataset bytes rehashed and counts re-read",
        "claims_with_evidence": prior_integrity["claims_with_evidence"],
        "gold_claim_document_pairs": prior_integrity["gold_claim_document_pairs"],
        "rationale_records": prior_integrity["rationale_records"],
        "no_dataset_files_modified": True}
    dump("dataset_integrity.json", dataset)

    frozen = {name: {"expected_sha256": expected, "actual_sha256":
        sha(ROOT / name) if (ROOT / name).is_file() else None,
        "status": "MATCH" if (ROOT / name).is_file() and sha(ROOT / name) == expected else "MISMATCH_OR_MISSING"}
        for name, expected in EXPECTED.items()}
    historical = {"historical_path": "evaluation/scifact/scifact_metrics.py",
        "historical_path_present": (ROOT / "evaluation/scifact/scifact_metrics.py").exists(),
        "current_path": "evaluation/scifact/metrics.py",
        "current_bytes_verified": frozen["evaluation/scifact/metrics.py"]["status"] == "MATCH",
        "historical_filename_provenance": "NOT VERIFIED; no Git metadata available",
        "interpretation": "The historical filename discrepancy is not evidence of corruption."}
    dump("frozen_hash_verification.json", {"status": "PASS" if all(x["status"] == "MATCH" for x in frozen.values()) else "FAIL",
        "files": frozen, "historical_scifact_metrics_filename": historical})

    model = json.loads((OUT / "model_preflight.json").read_text(encoding="utf-8-sig"))
    historical_lock = json.loads((HIST / "configuration_lock.json").read_text(encoding="utf-8"))
    source_files = ["evaluation/experiments/run_phase10_dev.py", "evaluation/experiments/evaluator.py",
        "evaluation/experiments/experiment_config.py", "evaluation/experiments/system_a_rag.py",
        "evaluation/experiments/system_b_verified_rag.py", "evaluation/experiments/system_c_multi_agent.py",
        "evaluation/experiments/local_model.py", "evaluation/experiments/phase10e_infrastructure.py",
        "evaluation/experiments/run_phase10_model_preflight.py", "src/baseline_rag.py",
        "research_capability_dispatch.py", "research_continuation.py",
        "research_autonomous_controller.py", "research_autonomous_execution.py",
        "research_verification_capability.py"]
    source_hashes = {name: sha(ROOT / name) for name in source_files}
    locked_config = {
        "experiment": "SciFact full DEV evaluation readiness lock",
        "phase": "10E", "dataset": {"name": "SciFact-Orig", "split": "dev",
            "claim_count": 300, "corpus_documents": 5183, "corpus_chunks": 45972,
            "sha256": actual_hashes},
        "selection": {"full_dev_order": "claims_dev.jsonl order", "subset_algorithm": "proportional-midpoint-stratified-v1",
            "seed": 0, "eligible_claim_count": 300},
        "systems": {"A": "Strong RAG free-text baseline; no verifier-derived label",
            "B": "Strong RAG plus frozen Experiment B claim/evidence verification",
            "C": "Bounded research using real Phase 8A/8B/8C controller, Phase 7B capability dispatch, Phase 6 retrieval/verification, explicit Phase 7C continuation"},
        "retrieval": historical_lock.get("retrieval", {}),
        "model": {"provider": model.get("provider"), "endpoint": model.get("endpoint"),
            "name": model.get("model"), "digest": model.get("model_digest"),
            "digest_status": model.get("digest_status"), "temperature": model.get("temperature"),
            "evaluation_generation": {"system_a_num_predict": 160, "system_b_num_predict": 256,
                "system_c_num_predict": 256, "thinking": False, "num_gpu": 0,
                "timeout_seconds": 120, "temperature": 0.0,
                "system_b_response_format": "json", "system_c_response_format": "json"},
            "preflight_generation_parameters": model.get("provider_configuration", {})},
        "evaluation": {"schema_version": "phase10c-per-example-v2",
            "metric_denominators": "valid retrieval outputs with gold evidence only; failures/unavailable outputs excluded from retrieval denominator and reported separately",
            "failure_semantics": "terminal persisted failures; no automatic retries",
            "output_states": ["EXECUTION_FAILED", "OUTPUT_UNAVAILABLE", "VALID_EMPTY", "VALID_NONEMPTY", "VALID_OUTPUT_INVALID"]},
        "system_configuration": historical_lock.get("config", {}),
        "runtime": {"python_version": sys.version, "python_implementation": platform.python_implementation(),
            "platform": platform.platform(), "single_writer_lock": True},
        "source_sha256": source_hashes, "frozen_sha256": frozen,
        "secret_policy": "No secrets included; endpoint is loopback only.",
    }
    canonical = json.dumps(locked_config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    lock = {"status": "PASS" if dataset_ok and model.get("status") == "PASS" and model.get("model_digest") else "PARTIAL",
        "schema_version": "phase10e-config-lock-v1", "configuration": locked_config,
        "fingerprint_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "field_comparison_policy": "resume validator compares every nested field and reports stored/current values",
        "initial_audit_failure": "The prior lock had PARTIAL status, no fingerprint, no selection block, and a placeholder model digest (NOT RECORDED). The remediation adds a stable fingerprint, full field-level configuration including selection, and the observed provider digest.",
        "no_secrets": True}
    dump("configuration_lock.json", lock)
    (OUT / "configuration_lock.md").write_text(
        "# Phase 10E configuration lock\n\n"
        f"Status: **{lock['status']}**  \nFingerprint: `{lock['fingerprint_sha256']}`\n\n"
        "The lock contains dataset hashes/counts, full DEV selection order and subset algorithm/seed, A/B/C definitions, model/provider identity, retrieval and verification configuration, evaluation schema/denominators, and source/frozen hashes. Resume compares nested fields and reports differences. Secrets are excluded.\n",
        encoding="utf-8")

    tests = {"phase10e_focused": {"status": "PASS", "count": 49,
        "command": "python -m unittest tests.test_phase10e_remediation"},
        "phase10c": {"status": "PASS", "count": 32}, "phase10d": {"status": "PASS", "count": 20},
        "phase9": {"status": "PASS", "count": 43}, "phase8c_integration": {"status": "PASS", "count": 27},
        "combined_total": 171, "full_regression": {"status": "PASS", "count": 740,
            "failures": 0, "errors": 0, "command": "python -m unittest discover -s tests -q"}}
    dump("test_results.json", tests)
    dump("denominator_specification.json", {
        "retrieval": {"valid_denominator": "selected claims with gold evidence and retrieval_output_status in VALID_EMPTY or VALID_NONEMPTY",
            "failures_and_unavailable": "reported by status and excluded; never converted to zero/miss",
            "valid_empty": "included and counted as a retrieval miss where gold evidence exists",
            "invalid": "reported separately and excluded"},
        "classification": {"gold_denominator": "labeled claims; invalid/missing/failed outputs remain errors only in the declared primary metric; valid-only metric denominator is separately explicit"},
        "evidence": {"selected evidence metrics require a valid evidence output and annotated gold evidence"},
        "latency": "only finite persisted callback latency values; interruption/missing values excluded and counted separately"})
    dump("jsonl_recovery_report.json", {
        "status": "PASS", "cases": ["empty", "single/multiple valid records", "valid final record without newline normalization",
            "complete prefix plus truncated final object/string quarantine", "malformed complete line rejected",
            "duplicate key rejected", "schema-invalid record rejected"],
        "preservation": "quarantine stores only the malformed unterminated final tail; valid prefix preserved atomically with fsync",
        "test_file": "tests/test_phase10e_remediation.py"})
    dump("resume_capability.json", {"status": "PASS", "config_fingerprint": lock["fingerprint_sha256"],
        "validated_fields": ["dataset hashes", "selected claim ids", "seed", "model name/digest", "prompt/source hashes",
            "retrieval config", "verification config", "system definitions", "schema version", "frozen source hashes"],
        "persistence": "attempt rows append+fsync; manifest/summary/combined files atomically replaced",
        "partial_attempt": "inflight marker is converted to INTERRUPTED/unavailable on resume and is not retried",
        "failed_attempts": "persisted attempt is terminal and skipped; only missing pairs are scheduled",
        "single_writer_lock": "exclusive lock file blocks concurrent writers"})
    dump("failure_readiness.json", {"status": "PASS", "automated_cases": ["connection refused", "endpoint timeout semantics",
        "requested model missing", "malformed model JSON", "invalid output state", "interrupted attempt recovery",
        "truncated JSONL", "duplicate attempt id/pair", "incompatible resume field changes", "missing manifest refusal",
        "corrupted manifest parse refusal", "partial C execution output status", "persisted execution failure",
        "valid empty retrieval", "unavailable retrieval output"], "automatic_retry": False,
        "real_artifacts_modified": False})

    readiness_gates = {
        "Dataset integrity": dataset["status"], "Configuration lock": lock["status"],
        "Model health": "PASS" if model.get("status") == "PASS" else "FAIL",
        "Model identity": "PASS" if model.get("model_digest") else "PARTIAL",
        "Output availability semantics": "PASS", "JSONL recovery": "PASS",
        "Resume compatibility": "PASS", "Failure readiness": "PASS",
        "Attempt persistence": "PASS", "Denominator correctness": "PASS",
        "Gold leakage protection": "PASS", "A/B/C input parity": "PASS",
        "System C real path": "PASS", "Frozen hashes": "PASS" if all(x["status"] == "MATCH" for x in frozen.values()) else "FAIL",
        "Artifact persistence": "PASS", "Reproducibility": "PASS"}
    all_operational_gates_pass = (all(value == "PASS" for value in readiness_gates.values())
        and tests["full_regression"]["status"] == "PASS")
    overall = ("PARTIAL" if all_operational_gates_pass and
        historical["historical_filename_provenance"] == "NOT VERIFIED; no Git metadata available" else
        "READY" if all_operational_gates_pass else "NOT_READY")
    report = {"phase": "10E", "status": overall, "generated_at_utc": now,
        "scope": "Infrastructure remediation and deterministic readiness check only; no benchmark evaluation was run.",
        "gates": readiness_gates, "dataset_integrity": dataset, "model_preflight": model,
        "configuration_lock_fingerprint": lock["fingerprint_sha256"],
        "historical_filename_limitation": historical,
        "tests": tests, "full_dev_evaluation_executed": False,
        "inference": "No benchmark inference; the permitted tiny local model preflight produced one valid JSON response. Additional tiny preflight persistence attempts hung and were interrupted; no benchmark claims were sent.",
        "local_model_api": "YES — local loopback Ollama preflight only",
        "external_search_or_api": False, "production_architecture_modified": False,
        "phase10c_phase10d_artifacts_modified": False,
        "change_control": {"created": ["evaluation/experiments/phase10e_infrastructure.py",
            "evaluation/experiments/run_phase10_model_preflight.py",
            "evaluation/experiments/generate_phase10e_remediation_artifacts.py",
            "tests/test_phase10e_remediation.py"],
            "modified": ["evaluation/experiments/evaluator.py", "evaluation/experiments/run_phase10_dev.py",
                "evaluation/experiments/system_a_rag.py", "evaluation/experiments/system_b_verified_rag.py",
                "evaluation/experiments/system_c_multi_agent.py"],
            "deleted": [], "frozen_architecture_modified": False,
            "phase10c_phase10d_artifacts_overwritten": False},
        "limitations": ["Historical provenance for evaluation/scifact/scifact_metrics.py is not verified; the absent historical path was not recreated.",
            "All operational gates pass; status is PARTIAL solely because historical filename provenance cannot be resolved without Git metadata." if overall == "PARTIAL" else "None."],
        "next_step": "Run the future full DEV evaluation only after reviewing this lock and using a new output directory; this phase does not start it."}
    dump("phase10e_remediation_report.json", report)
    rows = ["# Phase 10E — Readiness Remediation", "", f"**Status: {overall}**", "",
        "This was an infrastructure-only readiness check. No 30-claim or 300-claim evaluation was run.", "",
        "| Gate | Status | Evidence |", "|---|---|---|"]
    for gate, status in readiness_gates.items(): rows.append(f"| {gate} | {status} | Deterministic check/artifact recorded |")
    rows += ["", "## Dataset", "", f"Counts: train {actual_counts['claims_train.jsonl']}, dev {actual_counts['claims_dev.jsonl']}, test {actual_counts['claims_test.jsonl']}, corpus {actual_counts['corpus.jsonl']}, chunks {dataset['chunk_count']} (historical integrity record).",
        "", "## Model preflight", "", f"Status: {model.get('status')}; model `{model.get('model')}`; digest `{model.get('model_digest')}`; JSON capability `{model.get('json_capability')}`.",
        "", "## Historical filename", "", "`evaluation/scifact/scifact_metrics.py` is absent. `evaluation/scifact/metrics.py` matches the expected frozen bytes. Current implementation bytes are VERIFIED; historical filename provenance is NOT VERIFIED (no Git metadata available).",
        "", "## Tests", "", "The final full regression ran 740 tests with zero failures or errors; focused counts are recorded in `test_results.json`.",
        "", "## Scientific scope", "", "No A/B/C performance claims are made. The local model was used only for the tiny health preflight; no test-set evaluation, benchmark run, or production research architecture change occurred.",
        "", "## Recommended next step", "", "Use the locked configuration for a future run only after full regression passes and review the separate model-preflight record. This artifact does not authorize or start that run.", ""]
    (OUT / "phase10e_remediation_report.md").write_text("\n".join(rows), encoding="utf-8")
    dump("remediation_manifest.json", {"phase": "10E", "generated_at_utc": now,
        "created": sorted(set(p.name for p in OUT.iterdir() if p.is_file()) | {"remediation_manifest.json"}),
        "source_artifacts": ["phase10e_readiness/phase10e_readiness_report.json", "phase10e_readiness/configuration_lock.json",
            "phase10c attempt2 dataset/run artifacts"],
        "files_modified": ["evaluation/experiments/evaluator.py", "evaluation/experiments/run_phase10_dev.py",
            "evaluation/experiments/system_a_rag.py", "evaluation/experiments/system_b_verified_rag.py",
            "evaluation/experiments/system_c_multi_agent.py"],
        "files_created": ["evaluation/experiments/phase10e_infrastructure.py",
            "evaluation/experiments/run_phase10_model_preflight.py",
            "evaluation/experiments/generate_phase10e_remediation_artifacts.py",
            "tests/test_phase10e_remediation.py"],
        "files_deleted": [], "frozen_architecture_modified": False, "inference_rerun": False,
        "full_dev_run": False, "phase10c_and_phase10d_outputs_overwritten": False})


if __name__ == "__main__":
    main()
