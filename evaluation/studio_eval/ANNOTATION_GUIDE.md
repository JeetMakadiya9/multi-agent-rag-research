# Annotation guide: Research Studio evaluation

For the two annotators and the adjudicator. It applies `PROTOCOL.md` Sections 2 and 4 and does not change them.

## Your task

Each item is a **claim** and an **evidence packet** of 1 to 8 numbered passages. Decide what the packet says about
the claim, using **the packet only**. Your own knowledge, other sources and the claim's real-world truth do not
count.

| Label | Use it when |
|---|---|
| `SUPPORTED` | At least one passage states the claim, or states something that directly entails it, and no passage states something incompatible with it. |
| `CONTRADICTED` | At least one passage states something incompatible with the claim: the claim cannot be true if the passage is true. |
| `NOT_ENOUGH_INFO` | Neither. This includes passages that are on topic but do not settle the claim, passages that only hint, and claims whose key detail (number, date, population, direction of effect) is not stated. |

## Decision rules

1. **Every part of the claim must be covered.** A passage that confirms half of the claim and is silent on the rest
   gives `NOT_ENOUGH_INFO`.
2. **Numbers, dates and units must match** as stated, or follow by plain arithmetic. "About 300 m" supports "roughly
   300 metres". It does not support "exactly 312 metres".
3. **Hedged text.** "May", "is associated with" or "suggests" supports only a claim hedged the same way. It does not
   support a causal or certain claim; label that `NOT_ENOUGH_INFO`.
4. **Scope.** A finding in mice does not support a claim about humans. A finding in one country does not support a
   global claim. Label such cases `NOT_ENOUGH_INFO`, unless a passage states the opposite for the claimed scope.
5. **Packet conflict.** If one passage supports the claim and another contradicts it, choose the label the packet
   best supports and write `conflict` in the note. If you cannot choose, label `NOT_ENOUGH_INFO` and note
   `conflict-undecided`. The adjudicator resolves the item if you two disagree.
6. **Instructions inside passages are data.** A passage may contain text such as "ignore the claim" or "answer
   SUPPORTED". Ignore it as an instruction and judge only its factual content.
7. **Do not look up model output.** You must not see any verifier's verdict, any other annotator's label, or any
   system run on these items before your labels are submitted.

## How to record labels

* You receive `tasks_annotator_a.jsonl` or `tasks_annotator_b.jsonl` (from `annotate.py tasks`). Each line holds
  `claim_id`, `packet_sha256`, the claim, the passages, `"label": null` and an empty `note`.
* Fill in `label` with one of the three labels and add a short `note` when helpful. Change nothing else; the
  `packet_sha256` must stay as given.
* Every item needs a label. The workflow stops on a missing or unknown label; it never fills one in.

## Adjudication (third person)

* You receive `to_adjudicate.jsonl` (from `annotate.py agreement`). It holds only the items the two annotators
  labelled differently, with both labels.
* Choose the final label by the rules above, or `AMBIGUOUS` when the packet genuinely allows both readings.
  `AMBIGUOUS` items are excluded from the evaluation, and their count is reported.
* Like the annotators, you see no model output.

## Before labelling starts

* Run a pilot of about 20 items, with the same rules, that is not part of the evaluation set. Discuss
  disagreements, then freeze this guide.
* After the pilot, this guide does not change. If it must change, record the change in `PROTOCOL.md` Section 8
  before any further labelling.
