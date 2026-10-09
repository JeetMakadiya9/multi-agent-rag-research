"""Build the frozen H5 candidate pool, samples and blinded annotation packages.

Deterministic, offline, stdlib-only (plus the frozen SciFact adapter and the
frozen verification-unit analyzer). No model, retrieval or network use.

TRAIN annotations are used for sampling-frame construction, not as pair-level
gold labels. Every selected pair receives blinded human annotation.

Usage: python -m evaluation.experiments.h5.build_package
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data/scifact/data"
OUT = ROOT / "evaluation/experiments/results/h5"
DEV_MANIFEST = ROOT / "evaluation/experiments/results/phase13_improvement/development_set_manifest.json"

SEED = "h5-2026-v1"
PILOT_QUOTA = 20            # per sampling stratum -> 60
MAIN_QUOTA = 100            # per sampling stratum -> 300
CAP_PER_CLAIM = 2           # max pairs per claim within each sample (limits clustering)
SAMPLED_STRATA = ("support_oriented", "contradiction_oriented", "neutral_oriented")
ATOMICITY_RULE_VERSION = "h5-atomicity-v1"
SOURCE_SHA256 = {           # frozen SciFact hashes (phase10e_remediation/dataset_integrity.json)
    "claims_train.jsonl": "F4C8FA82D8BD0653A9CC8D61A6EA48C25EACEA64E90AF5DBF390EBB1B74372F0",
    "claims_dev.jsonl": "86F0435D08FDB65D1AA41D1472684F57E6E71930626497BDF4D7A9EC1A632217",
    "claims_test.jsonl": "558930D75215C73F84A28FE538307D6D397C9DE1EC7239514CF45F80D75D2CA3",
    "corpus.jsonl": "B8D6C89624CB2ED74DEE8938EFFC4F5D8BD2086887880AF8110D64BE4CEADE62",
}
QWEN_PROMPT_V1_SHA256 = "aa49d5e7dd47d4a20cb7e68486c1e507fc4acfd74c079bb9d3e3f6b5eff0c853"

ANNOTATOR_FIELDS = ("pair_id", "claim_text", "evidence_text", "gold_label", "annotator_note")
LABELS = ("SUPPORT", "CONTRADICT", "NEUTRAL")

# Deterministic atomicity rules (no LLM). ATOMIC only when every rule passes.
COORDINATOR = re.compile(r"\b(and|or|but|whereas|while|although|as well as|both|either|neither|nor|versus|vs\.?|"
                         r"respectively)\b|;", re.I)
CONTEXT_DEPENDENT_START = re.compile(r"^(it|this|these|they|such|that|those|its|their)\b", re.I)
RELATIVE_CLAUSE = re.compile(r",\s*(which|where|whereby)\b", re.I)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def dumps(row: Mapping[str, Any]) -> str:
    return json.dumps(row, sort_keys=True, ensure_ascii=False)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    data = "".join(dumps(r) + "\n" for r in rows).encode("utf-8")
    path.write_bytes(data)
    return sha256_bytes(data)


def write_json(path: Path, value: Any) -> str:
    data = (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=1) + "\n").encode("utf-8")
    path.write_bytes(data)
    return sha256_bytes(data)


def atomicity(claim_text: str) -> tuple[str, list[str]]:
    from verification_units import analyze_verification_units
    reasons = []
    units = analyze_verification_units(claim_text)
    if len(units) != 1:
        reasons.append("multiple sentences")
    elif len(units[0].components) != 1:
        reasons.append("frozen analyzer found coordinated components")
    if reasons:
        return "COMPOUND", reasons
    if COORDINATOR.search(claim_text):
        reasons.append("coordinator/connective present: " + COORDINATOR.search(claim_text).group(0).lower())
    if CONTEXT_DEPENDENT_START.search(claim_text.strip()):
        reasons.append("starts with context-dependent pronoun/determiner")
    if RELATIVE_CLAUSE.search(claim_text):
        reasons.append("non-restrictive relative clause")
    return ("UNCERTAIN", reasons) if reasons else ("ATOMIC", [])


def pair_id(claim_id: str, doc_id: int, sentence_index: int) -> str:
    return "h5_" + hashlib.sha256(f"h5-pair-v1|{claim_id}|{doc_id}|{sentence_index}".encode()).hexdigest()[:16]


def rank_key(pid: str) -> str:
    return hashlib.sha256(f"{SEED}|{pid}".encode()).hexdigest()


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def verify_sources() -> dict[str, Any]:
    actual = {name: sha256_file(DATA_DIR / name) for name in SOURCE_SHA256}
    mismatches = [n for n in SOURCE_SHA256 if actual[n] != SOURCE_SHA256[n]]
    if mismatches:
        raise SystemExit(f"STOP: SciFact source hash mismatch: {mismatches}")
    return actual


def protected_sets() -> dict[str, Any]:
    """DEV (incl. Phase 13 dev-69 and reserved-119) and TEST claim ids/texts that must never be sampled."""
    from evaluation.scifact.scifact_adapter import load_scifact_dataset
    dev = load_scifact_dataset(DATA_DIR, "dev")
    test = [json.loads(line) for line in (DATA_DIR / "claims_test.jsonl").read_text(encoding="utf-8").splitlines() if line]
    manifest = json.loads(DEV_MANIFEST.read_text(encoding="utf-8"))
    dev_ids = {r.runtime.claim_id for r in dev.claims}
    dev69 = set(manifest["claim_ids"])
    reserved = set(manifest["reserved_for_later_validation_claim_ids"])
    if not dev69 <= dev_ids or not reserved <= dev_ids or dev69 & reserved:
        raise SystemExit("STOP: reserved/dev-69 membership cannot be verified against DEV")
    return {"dev_texts": {norm(r.runtime.text) for r in dev.claims}, "test_texts": {norm(c["claim"]) for c in test},
            "dev69_texts": {norm(r.runtime.text) for r in dev.claims if r.runtime.claim_id in dev69},
            "reserved_texts": {norm(r.runtime.text) for r in dev.claims if r.runtime.claim_id in reserved},
            "counts": {"dev": len(dev_ids), "dev69": len(dev69), "reserved": len(reserved), "test": len(test)}}


def build_pool() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    from evaluation.scifact.scifact_adapter import load_scifact_dataset
    train = load_scifact_dataset(DATA_DIR, "train")          # validates every rationale index against the corpus
    corpus = {d.doc_id: d for d in train.corpus}
    sorted_doc_ids = sorted(corpus)
    protected = protected_sets()
    claims_inventory, pool = [], []
    seen_text: dict[str, str] = {}
    for record in sorted(train.claims, key=lambda r: r.runtime.claim_id):
        cid = f"train:{record.runtime.claim_id}"
        text = record.runtime.text
        status, reasons = atomicity(text)
        entry = {"claim_id": cid, "claim_text": text, "claim_atomicity_status": status, "atomicity_reasons": reasons,
                 "atomicity_rule_version": ATOMICITY_RULE_VERSION, "eligible": False, "exclusion_reason": None}
        key = norm(text)
        if key in protected["dev_texts"] or key in protected["test_texts"]:
            entry["exclusion_reason"] = "claim text also appears in DEV or TEST"
        elif key in seen_text:
            entry["exclusion_reason"] = f"duplicate claim text of {seen_text[key]}"
        elif status != "ATOMIC":
            entry["exclusion_reason"] = f"claim_atomicity_status={status}"
        else:
            entry["eligible"] = True
        seen_text.setdefault(key, cid)
        claims_inventory.append(entry)
        if not entry["eligible"]:
            continue
        ann = record.annotations
        rationale: dict[tuple[int, int], set[str]] = defaultdict(set)
        for doc in ann.evidence or ():
            for r in doc.rationales:
                for s in r.sentence_indices:
                    rationale[(doc.doc_id, s)].add(r.label)
        related_docs = sorted({d.doc_id for d in ann.evidence or ()} or set(ann.cited_doc_ids))
        candidates = []
        for (doc_id, s), labels in sorted(rationale.items()):
            stratum = ("conflicting_rationale_excluded" if len(labels) > 1 else
                       "support_oriented" if labels == {"SUPPORT"} else "contradiction_oriented")
            candidates.append((doc_id, s, stratum, True, "/".join(sorted(labels))))
        for doc_id in related_docs:
            for s in range(len(corpus[doc_id].abstract)):
                if (doc_id, s) not in rationale:
                    candidates.append((doc_id, s, "neutral_oriented", False, None))
        # Inventory only (not sampled): one unrelated-document sentence per claim, chosen by stable hash.
        unrelated = [d for d in sorted_doc_ids if d not in set(related_docs) | set(ann.cited_doc_ids)]
        u_doc = min(unrelated, key=lambda d: hashlib.sha256(f"{SEED}|unrelated|{cid}|{d}".encode()).hexdigest())
        u_sent = int(hashlib.sha256(f"{SEED}|unrelated-sent|{cid}".encode()).hexdigest(), 16) % len(corpus[u_doc].abstract)
        candidates.append((u_doc, u_sent, "neutral_unrelated_inventory", False, None))
        seen_evidence: dict[str, str] = {}
        for doc_id, s, stratum, is_rationale, hidden_label in candidates:
            evidence = corpus[doc_id].abstract[s].strip()
            pid = pair_id(cid, doc_id, s)
            row = {"pair_id": pid, "claim_id": cid, "claim_text": text, "evidence_text": evidence,
                   "document_id": doc_id, "sentence_index": s, "evidence_sentence_id": f"{doc_id}:{s}",
                   "source_split": "train", "sampling_stratum": stratum, "rationale_candidate": is_rationale,
                   "non_rationale_candidate": not is_rationale, "train_rationale_label_hidden": hidden_label,
                   "claim_atomicity_status": status, "sampling_rank_key": rank_key(pid),
                   "eligible_for_sampling": stratum in SAMPLED_STRATA, "exclusion_reason": None}
            if not evidence:
                row.update(eligible_for_sampling=False, exclusion_reason="empty evidence sentence")
            elif norm(evidence) in seen_evidence:
                row.update(eligible_for_sampling=False,
                           exclusion_reason=f"duplicate evidence text for this claim ({seen_evidence[norm(evidence)]})")
            else:
                seen_evidence[norm(evidence)] = pid
            if stratum not in SAMPLED_STRATA and row["exclusion_reason"] is None:
                row["exclusion_reason"] = f"stratum {stratum} is not sampled"
            pool.append(row)
    pool.sort(key=lambda r: r["pair_id"])
    if len({r["pair_id"] for r in pool}) != len(pool):
        raise SystemExit("STOP: pair_id collision in candidate pool")
    return pool, claims_inventory, protected


def select(pool: Sequence[Mapping[str, Any]], quota: int, excluded_claims: set[str]) -> list[dict[str, Any]]:
    counts, per_claim, chosen = Counter(), Counter(), []
    for row in sorted((r for r in pool if r["eligible_for_sampling"]), key=lambda r: r["sampling_rank_key"]):
        stratum, cid = row["sampling_stratum"], row["claim_id"]
        if cid in excluded_claims or counts[stratum] >= quota or per_claim[cid] >= CAP_PER_CLAIM:
            continue
        chosen.append(dict(row))
        counts[stratum] += 1
        per_claim[cid] += 1
    if any(counts[s] != quota for s in SAMPLED_STRATA):
        raise SystemExit(f"STOP: quota {quota} not reachable per stratum: {dict(counts)}")
    return sorted(chosen, key=lambda r: r["pair_id"])


def manifest_rows(selected: Sequence[Mapping[str, Any]], sample: str) -> list[dict[str, Any]]:
    return [{**{k: r[k] for k in ("pair_id", "claim_id", "claim_text", "evidence_text", "source_split",
                                  "sampling_stratum", "claim_atomicity_status", "evidence_sentence_id", "document_id",
                                  "sentence_index", "rationale_candidate", "non_rationale_candidate",
                                  "train_rationale_label_hidden", "sampling_rank_key")},
             "sample": sample, "claim_review_id": claim_review_id(r["claim_id"]), "sampling_metadata_hidden": True,
             "annotation_status": "NOT_STARTED", "annotator_a_label": None, "annotator_b_label": None,
             "adjudication_required": None, "adjudicated_label": None, "adjudication_note": None, "gold_label": None}
            for r in selected]


def claim_review_id(claim_id: str) -> str:
    return "h5c_" + hashlib.sha256(f"h5-claim-v1|{claim_id}".encode()).hexdigest()[:12]


def annotator_rows(selected: Sequence[Mapping[str, Any]], annotator: str) -> list[dict[str, Any]]:
    ordered = sorted(selected, key=lambda r: hashlib.sha256(f"{SEED}|annotator-{annotator}|{r['pair_id']}".encode()).hexdigest())
    return [{"pair_id": r["pair_id"], "claim_text": r["claim_text"], "evidence_text": r["evidence_text"],
             "gold_label": None, "annotator_note": None} for r in ordered]


def build() -> dict[str, Any]:
    source_hashes = verify_sources()
    pool, claims_inventory, protected = build_pool()
    pilot = select(pool, PILOT_QUOTA, excluded_claims=set())
    main = select(pool, MAIN_QUOTA, excluded_claims={r["claim_id"] for r in pilot})
    for sample in (pilot, main):
        texts = {norm(r["claim_text"]) for r in sample}
        if texts & (protected["dev_texts"] | protected["test_texts"]) or any(r["source_split"] != "train" for r in sample):
            raise SystemExit("STOP: protected DEV/TEST claim in sample")
    OUT.mkdir(parents=True, exist_ok=True)
    hashes = {
        "h5_candidate_pool.jsonl": write_jsonl(OUT / "h5_candidate_pool.jsonl", pool),
        "h5_claim_inventory.jsonl": write_jsonl(OUT / "h5_claim_inventory.jsonl", claims_inventory),
        "h5_excluded_candidates.jsonl": write_jsonl(OUT / "h5_excluded_candidates.jsonl",
                                                    [c for c in claims_inventory if not c["eligible"]]),
        "h5_gold_manifest.jsonl": write_jsonl(OUT / "h5_gold_manifest.jsonl", manifest_rows(main, "main")),
        "h5_pilot_manifest.jsonl": write_jsonl(OUT / "h5_pilot_manifest.jsonl", manifest_rows(pilot, "pilot")),
        "h5_annotator_a.jsonl": write_jsonl(OUT / "h5_annotator_a.jsonl", annotator_rows(main, "A")),
        "h5_annotator_b.jsonl": write_jsonl(OUT / "h5_annotator_b.jsonl", annotator_rows(main, "B")),
        "h5_pilot_annotator_a.jsonl": write_jsonl(OUT / "h5_pilot_annotator_a.jsonl", annotator_rows(pilot, "A")),
        "h5_pilot_annotator_b.jsonl": write_jsonl(OUT / "h5_pilot_annotator_b.jsonl", annotator_rows(pilot, "B")),
    }
    review_claims = sorted({(claim_review_id(r["claim_id"]), r["claim_text"]) for r in pilot + main},
                           key=lambda x: hashlib.sha256(f"{SEED}|review|{x[0]}".encode()).hexdigest())
    hashes["h5_atomicity_review.jsonl"] = write_jsonl(OUT / "h5_atomicity_review.jsonl", [
        {"claim_review_id": cid, "claim_text": text, "reviewer_atomicity": None, "reviewer_note": None}
        for cid, text in review_claims])
    hashes["h5_annotation_schema.json"] = write_json(OUT / "h5_annotation_schema.json", annotation_schema())
    doc_names = ("h5_protocol.md", "h5_annotation_guidelines.md", "h5_nli_model_requirements.md",
                 "h5_data_dictionary.md", "h5_preanalysis.md")
    missing = [n for n in doc_names if not (OUT / n).exists()]
    if missing:
        raise SystemExit(f"STOP: frozen protocol documents missing: {missing}")
    docs = {name: sha256_file(OUT / name) for name in doc_names}
    stats = statistics(pool, claims_inventory, pilot, main, protected)
    sampling_manifest = {
        "protocol": "H5 pair-level stance gold set", "sampling_seed": SEED,
        "sampling_algorithm": ("candidates of ATOMIC TRAIN claims; eligible strata support_oriented / "
                               "contradiction_oriented / neutral_oriented; ranked by SHA-256(seed|pair_id) ascending; "
                               "greedy fill per stratum quota with at most CAP_PER_CLAIM pairs per claim; pilot first "
                               "(20/stratum), main second (100/stratum) excluding all pilot claims"),
        "quotas": {"pilot_per_stratum": PILOT_QUOTA, "main_per_stratum": MAIN_QUOTA, "cap_per_claim": CAP_PER_CLAIM},
        "atomicity_rule_version": ATOMICITY_RULE_VERSION,
        "atomicity_rules": {"COMPOUND": "frozen verification_units finds >1 sentence or >1 component",
                            "UNCERTAIN": [COORDINATOR.pattern, CONTEXT_DEPENDENT_START.pattern, RELATIVE_CLAUSE.pattern],
                            "ATOMIC": "neither COMPOUND nor UNCERTAIN"},
        "source_hashes_sha256": source_hashes, "source_hashes_match_frozen": True,
        "candidate_pool_hash": hashes["h5_candidate_pool.jsonl"],
        "selected_manifest_hash": hashes["h5_gold_manifest.jsonl"],
        "pilot_manifest_hash": hashes["h5_pilot_manifest.jsonl"],
        "file_hashes": hashes, "document_hashes": docs,
        "qwen_prompt_v1_sha256": QWEN_PROMPT_V1_SHA256,
        "train_annotations_role": "TRAIN annotations are used for sampling-frame construction, not as pair-level gold labels.",
        "statistics": stats,
    }
    hashes["h5_sampling_manifest.json"] = write_json(OUT / "h5_sampling_manifest.json", sampling_manifest)
    (OUT / "h5_integrity_report.md").write_bytes(integrity_report(sampling_manifest, hashes).encode("utf-8"))
    return sampling_manifest


def annotation_schema() -> dict[str, Any]:
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "title": "H5 annotator row", "type": "object",
            "additionalProperties": False, "required": list(ANNOTATOR_FIELDS),
            "properties": {"pair_id": {"type": "string", "pattern": "^h5_[0-9a-f]{16}$"},
                           "claim_text": {"type": "string", "minLength": 1},
                           "evidence_text": {"type": "string", "minLength": 1},
                           "gold_label": {"enum": [None, *LABELS]},
                           "annotator_note": {"type": ["string", "null"]}}}


def statistics(pool, claims_inventory, pilot, main, protected) -> dict[str, Any]:
    def sample_stats(rows):
        return {"pairs": len(rows), "unique_pair_ids": len({r["pair_id"] for r in rows}),
                "unique_claims": len({r["claim_id"] for r in rows}),
                "unique_claim_evidence_pairs": len({(r["claim_id"], norm(r["evidence_text"])) for r in rows}),
                "by_stratum": dict(sorted(Counter(r["sampling_stratum"] for r in rows).items())),
                "pairs_per_claim": dict(sorted(Counter(Counter(r["claim_id"] for r in rows).values()).items())),
                "source_documents": len({r["document_id"] for r in rows}),
                "missing_fields": sum(1 for r in rows for k in ("claim_text", "evidence_text", "claim_id")
                                      if not r.get(k))}
    return {
        "claims": {"train_total": len(claims_inventory),
                   "atomicity": dict(sorted(Counter(c["claim_atomicity_status"] for c in claims_inventory).items())),
                   "eligible": sum(c["eligible"] for c in claims_inventory),
                   "exclusion_reasons": dict(sorted(Counter((c["exclusion_reason"] or "eligible").split(" of ")[0]
                                                            for c in claims_inventory).items()))},
        "candidate_pool": {"rows": len(pool), "by_stratum": dict(sorted(Counter(r["sampling_stratum"] for r in pool).items())),
                           "eligible_by_stratum": dict(sorted(Counter(r["sampling_stratum"] for r in pool
                                                                      if r["eligible_for_sampling"]).items())),
                           "duplicate_evidence_excluded": sum(1 for r in pool if (r["exclusion_reason"] or "").startswith("duplicate")),
                           "unique_claims": len({r["claim_id"] for r in pool})},
        "pilot": sample_stats(pilot), "main": sample_stats(main),
        "pilot_main_claim_overlap": len({r["claim_id"] for r in pilot} & {r["claim_id"] for r in main}),
        "pilot_main_pair_overlap": len({r["pair_id"] for r in pilot} & {r["pair_id"] for r in main}),
        "protected": {**protected["counts"],
                      "sampled_claims_matching_dev69": len({norm(r["claim_text"]) for r in pilot + main} & protected["dev69_texts"]),
                      "sampled_claims_matching_reserved": len({norm(r["claim_text"]) for r in pilot + main} & protected["reserved_texts"]),
                      "sampled_claims_matching_any_dev": len({norm(r["claim_text"]) for r in pilot + main} & protected["dev_texts"]),
                      "sampled_claims_matching_test": len({norm(r["claim_text"]) for r in pilot + main} & protected["test_texts"]),
                      "non_train_rows": sum(r["source_split"] != "train" for r in pilot + main)},
    }


def integrity_report(manifest: Mapping[str, Any], hashes: Mapping[str, str]) -> str:
    st = manifest["statistics"]
    lines = ["# H5 Integrity Report", "", "Generated deterministically by `evaluation/experiments/h5/build_package.py` "
             "(no timestamps). No inference, retrieval, downloads or network.", "",
             "## Source hashes (verified against frozen SciFact hashes)", ""]
    lines += [f"- `{k}`: `{v}`" for k, v in sorted(manifest["source_hashes_sha256"].items())]
    lines += ["", "## Claims", "", f"`{json.dumps(st['claims'], sort_keys=True)}`", "", "## Candidate pool", "",
              f"`{json.dumps(st['candidate_pool'], sort_keys=True)}`", "", "## Samples", "",
              f"- pilot: `{json.dumps(st['pilot'], sort_keys=True)}`", f"- main: `{json.dumps(st['main'], sort_keys=True)}`",
              f"- pilot/main claim overlap: {st['pilot_main_claim_overlap']}; pair overlap: {st['pilot_main_pair_overlap']}",
              "", "## Reserved-data protection", "", f"`{json.dumps(st['protected'], sort_keys=True)}`", "",
              "## Hashes", "", f"- sampling seed: `{manifest['sampling_seed']}`",
              f"- candidate pool: `{manifest['candidate_pool_hash']}`",
              f"- selected (main) manifest: `{manifest['selected_manifest_hash']}`",
              f"- pilot manifest: `{manifest['pilot_manifest_hash']}`"]
    lines += [f"- `{k}`: `{v}`" for k, v in sorted({**hashes, **manifest["document_hashes"]}.items())
              if k not in ("h5_sampling_manifest.json",)]
    lines += ["", "TRAIN annotations are used for sampling-frame construction, not as pair-level gold labels.", ""]
    return "\n".join(lines)


def main() -> None:
    from evidence_core.replay import network_forbidden
    with network_forbidden():
        manifest = build()
    print(json.dumps({k: manifest[k] for k in ("candidate_pool_hash", "selected_manifest_hash", "pilot_manifest_hash")},
                     indent=1))
    print(json.dumps(manifest["statistics"], indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
