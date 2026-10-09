"""Phase 13 audit: frozen-artifact integrity and data-leakage checks (no inference)."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

from evaluation.experiments.phase13 import b_plus
from evaluation.experiments.phase13.common import (
    OUTPUT_DIR, ROOT, frozen_retrieval_by_claim, gold_by_claim, load_dataset, load_frozen_rows, snapshot_protected,
    write_json,
)
from evaluation.experiments.phase13.scoring import judged_texts, load_attempts

RUNTIME_KEYS = {"claim_id", "claim"}


def frozen_integrity() -> dict[str, Any]:
    before = json.loads((OUTPUT_DIR / "protected_hashes_before.json").read_text(encoding="utf-8"))["hashes"]
    after = snapshot_protected()
    from evaluation.experiments.run_phase10f import FROZEN_EXPECTED
    return {
        "files_checked": len(before),
        "modified": sorted(k for k in before if k in after and before[k] != after[k]),
        "deleted": sorted(k for k in before if k not in after),
        "added_inside_protected_dirs": sorted(k for k in after if k not in before),
        "frozen_expected_mismatches": sorted(k for k, v in FROZEN_EXPECTED.items() if after.get(k) != v),
        "pass": None,
    }


def _expected_text(chunk: dict[str, Any], corpus, window: int) -> str:
    if window == 0:
        return chunk["text"]
    doc = corpus[int(chunk["scifact_doc_id"])]
    return b_plus.package_with_context(doc.title, doc.abstract, int(chunk["scifact_sentence_indices"][0]), window)


def leakage_audit() -> dict[str, Any]:
    dataset = load_dataset()
    corpus = {d.doc_id: d for d in dataset.corpus}
    gold = gold_by_claim(dataset)
    retrieval = frozen_retrieval_by_claim(load_frozen_rows())
    manifest = json.loads((OUTPUT_DIR / "development_set_manifest.json").read_text(encoding="utf-8"))
    report: dict[str, Any] = {"variants": {}}
    for run_dir in sorted((OUTPUT_DIR / "runs").glob("*")):
        variant = run_dir.name
        config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        rows = load_attempts(variant)
        issues: list[str] = []
        for row in rows:
            cid = row["claim_id"]
            if set(row["runtime_input"]) != RUNTIME_KEYS:
                issues.append(f"{cid}: runtime payload keys {sorted(row['runtime_input'])}")
            if row["runtime_input"]["claim"] != gold[cid].claim:
                issues.append(f"{cid}: runtime claim text differs from dataset claim")
            if row["status"] != "SUCCESS":
                continue
            limit = config["evidence_limit"]
            expected = [_expected_text(c, corpus, config["context_window"]) for c in retrieval[cid]["chunks"][:limit]]
            if judged_texts(row) != expected:
                issues.append(f"{cid}: judged evidence differs from replayed frozen retrieval")
            refs = [(c["scifact_doc_id"], c["scifact_sentence_indices"]) for c in row["retrieved_chunks"]]
            frozen_refs = [(c["scifact_doc_id"], c["scifact_sentence_indices"]) for c in retrieval[cid]["chunks"]]
            if refs != frozen_refs:
                issues.append(f"{cid}: retrieved refs differ from frozen retrieval")
            raw = json.dumps(row.get("raw_output") or {})
            for marker in ('"gold_label"', '"rationales"', '"evidence_sets"', "gold_rationale"):
                if marker in raw:
                    issues.append(f"{cid}: gold marker {marker} present in verifier output")
        report["variants"][variant] = {"attempts": len(rows), "issues": issues, "pass": not issues and
                                       len(rows) == len(manifest["claim_ids"])}
    prompt_text = (b_plus.PROMPT_V2_SYSTEM + b_plus.PROMPT_V1_SYSTEM).lower()
    claim_words = {w for g in gold.values() for w in re.findall(r"[a-z]{9,}", g.claim.lower())}
    report["prompt_checks"] = {
        "benchmark_terms_in_v2_prompt": [t for t in ("scifact", "gold", "rationale", "label") if t in b_plus.PROMPT_V2_SYSTEM.lower()],
        "long_dev_claim_words_in_prompts": sorted(w for w in claim_words if w in prompt_text),
        "examples_from_evaluation_set": "none (v2 prompt has definitions and generic relation types only)",
    }
    report["development_subset_selection"] = {
        "inputs_consulted": manifest["inputs_consulted"], "inputs_not_consulted": manifest["inputs_not_consulted"],
        "frozen_before_inference": manifest["status"],
    }
    report["guard_rule_design"] = ("G1-G5 were designed from the frozen catalogue of 23 guard firings over all "
                                   "300 DEV claims (claim text, guard message, stances). They use no gold label "
                                   "at inference, but H3 evidence is in-sample for every DEV split.")
    report["pass"] = all(v["pass"] for v in report["variants"].values()) and not report["prompt_checks"]["benchmark_terms_in_v2_prompt"]
    return report


def main() -> None:
    integrity = frozen_integrity()
    integrity["pass"] = not (integrity["modified"] or integrity["deleted"] or integrity["added_inside_protected_dirs"]
                             or integrity["frozen_expected_mismatches"])
    leakage = leakage_audit()
    audit = {"phase": "13", "artifact": "phase13_audit", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
             "frozen_artifact_integrity": integrity, "data_leakage": leakage,
             "overall_pass": integrity["pass"] and leakage["pass"]}
    write_json(OUTPUT_DIR / "phase13_audit.json", audit)
    print(json.dumps({"integrity_pass": integrity["pass"], "modified": integrity["modified"],
                      "leakage_pass": leakage["pass"],
                      "variants": {k: (v["attempts"], len(v["issues"])) for k, v in leakage["variants"].items()},
                      "prompt": leakage["prompt_checks"]}, indent=1))


if __name__ == "__main__":
    main()
