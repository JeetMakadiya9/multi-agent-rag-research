# Building the evaluation set: claims, packets, labels, freeze

How to produce the claim set that `PROTOCOL.md` requires, using `annotate.py` and `run_eval.py`. Nothing here changes
the protocol.

**Status: no claim set exists.** Everything below is the procedure. No step has been carried out.

## 1. Claims (people write or select them)

* **Target.** At least 300 eligible claims after exclusions, with at least 100 per gold class (Section 5). Plan for
  about 360 drafts, because some will be excluded as `AMBIGUOUS` or as overlaps. The final class balance is known
  only after labelling, so drafts should aim for roughly a third of each.
* **Kind.** Single, checkable, English factual claims of up to 400 characters, of the kind a Studio user verifies:
  scholarly, scientific, technical or encyclopedic statements. Compound claims are split before sampling. Opinions,
  predictions and statements about the evaluation are excluded.
* **Sampling.**
  1. Fix a list of topic areas and a quota per area before writing.
  2. For each claim, the writer first chooses a source passage. They then write a claim that the passage supports,
     contradicts (by changing one detail: a number, date, direction or scope), or leaves open.
  3. The writer's intended label is **not** recorded in the draft and is not shown to annotators. Gold labels come
     only from annotation.
* **Independence.**
  * No claim may come from SciFact, the WebRAG claim sets, the Studio demonstrations or the agentic verification
    study.
  * No claim may be written by looking at any verifier's output.
  * No setting of the system may change after the claims are seen.
* **Provenance** (required by the draft format): `claim_author`, `written_at` and `source` for every claim.

## 2. Fixed evidence packets

* 1 to 8 passages per claim, each with `url`, `title`, `text` and `licence`: the source's licence or terms of use.
  Only openly accessible sources whose terms allow this use.
* Passages are copied once, at drafting time, and never refreshed. The packet the annotators see is the packet the
  system sees. `packet_sha256` ties each label to its packet.
* **Mix within a packet:** the passage the claim was written from, plus 0 to 7 related passages (same topic, other
  sources). Some packets should hold both supporting and contradicting passages; that is the protocol's
  evidence-conflict subgroup.
* A few packets may include an instruction-like passage (the injection subgroup). Mark it in the claim's
  `subgroups` list (`["injection"]`), together with `["conflict"]` and `["number"]` where they apply.
* **Data stays private** (`PROTOCOL.md` Section 3). The draft, task, label and claims files live outside the public
  repository, for example under `outputs/studio_eval_data/`, which is git-ignored.

Draft line format (`draft.jsonl`):

```json
{"claim_id": "s0001", "claim": "...", "subgroups": ["number"],
 "provenance": {"claim_author": "...", "written_at": "2026-11-02", "source": "..."},
 "passages": [{"url": "https://...", "title": "...", "text": "...", "licence": "CC BY 4.0"}]}
```

## 3. Annotation, adjudication and freeze

```text
python -m evaluation.studio_eval.annotate tasks --draft draft.jsonl --out work/
  # give tasks_annotator_a.jsonl and tasks_annotator_b.jsonl to two people (ANNOTATION_GUIDE.md)
python -m evaluation.studio_eval.annotate agreement --draft draft.jsonl --a a.jsonl --b b.jsonl --out work/
  # reports kappa and raw agreement; give to_adjudicate.jsonl to a third person
python -m evaluation.studio_eval.annotate freeze --draft draft.jsonl --a a.jsonl --b b.jsonl \
    --adjudication adj.jsonl --out frozen/ --exclude-claims <used claim sets>
python -m evaluation.studio_eval.run_eval --claims frozen/claims.jsonl --check-overlap
```

`freeze` stops when:
* a label is missing or invalid;
* a label was given against another packet;
* a disagreement is not adjudicated;
* a claim matches a used set.

Otherwise it writes `claims.jsonl` (in the format `run_eval.py` validates) and `FREEZE.json`. `FREEZE.json` holds:
* the claims file's SHA-256;
* kappa, raw agreement and the number of disagreements;
* the `AMBIGUOUS` count;
* the class counts;
* whether the set is large enough for a promotion decision;
* the SHA-256 of every input file.

## 4. Run and score (only after the freeze)

```text
python -m evaluation.studio_eval.run_eval --claims frozen/claims.jsonl --freeze frozen/FREEZE.json --out outputs/studio_eval/<run>
python -m evaluation.studio_eval.score --records outputs/studio_eval/<run>/records.jsonl
```

`run_eval.py` refuses to run without a matching freeze that declares the labels independent of model output.

## What remains human work

1. Writing or selecting about 360 claims, with their packets, provenance and licences.
2. Two independent annotation passes over every claim.
3. Adjudication of the disagreements by a third person.
4. A pilot of about 20 items before step 2, after which `ANNOTATION_GUIDE.md` is frozen.

No model may stand in for any of these.
