"""Held-out, offline claim-structure benchmark.

Uses analyzer functions imported from the frozen first benchmark. This file
does not tune or change those functions. Ground truth below is hand-authored.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
FIRST = ROOT / "baseline_nlp_claim_analysis_benchmark.py"
RAW = ROOT / "baseline_nlp_claim_analysis_heldout_v2_results.jsonl"
SUMMARY = ROOT / "baseline_nlp_claim_analysis_heldout_v2_summary.json"

spec = importlib.util.spec_from_file_location("frozen_claim_benchmark", FIRST)
frozen = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = frozen
spec.loader.exec_module(frozen)

def Q(kind: str, name: str, *alts: str) -> dict[str, Any]:
    return {"type": kind, "name": name, "alternatives": list(alts)}

def R(s: str, p: str, o: str) -> dict[str, list[str]]:
    return {"subject": [s], "predicate": [p], "object": [o]}

def E(text: str, *groups: str, q: list[dict[str, Any]] | None = None,
      rel: dict[str, list[str]] | None = None) -> dict[str, Any]:
    return {"text": text, "required_components": [[g] for g in groups],
            "qualifiers": q or [], "relationship": rel}

def C(id_: str, category: str, original: str, *claims: dict[str, Any], review: str | None = None) -> dict[str, Any]:
    expected = list(claims)
    qualifiers = [q for claim in expected for q in claim["qualifiers"]]
    return {"id": id_, "category": category, "original_text": original,
            "expected_claim_count": len(expected), "expected_claims": expected,
            "required_components": [g for x in expected for g in x["required_components"]],
            "dates": [q for q in qualifiers if q["type"] == "date"],
            "quantities": [q for q in qualifiers if q["type"] == "quantity"],
            "conditions": [q for q in qualifiers if q["type"] == "condition"],
            "negations": [q for q in qualifiers if q["type"] == "negation"],
            "comparisons": [q for q in qualifiers if q["type"] == "comparison"],
            "causal_links": [q for q in qualifiers if q["type"] == "causal"],
            "relationships": [x["relationship"] for x in expected if x["relationship"]],
            "technical_components": [g[0] for g in [g for x in expected for g in x["required_components"]]
                                     if any(t in g[0].lower() for t in ("faiss", "bm25", "embedding", "encoder", "retrieval", "index", "tokenizer", "vector", "rerank", "sqlite", "gpu"))],
            "manual_review_reason": review}

# New material authored for this held-out set. Each expected claim and anchor
# was annotated independently of analyzer output.
DATASET = [
 # 1 coordinated noun/entity lists
 C("H01","coordinated_lists","The ingest service handles scanned invoices, audio transcripts, and CAD drawings.", E("The ingest service handles scanned invoices.","ingest service","handles","scanned invoices"),E("The ingest service handles audio transcripts.","ingest service","handles","audio transcripts"),E("The ingest service handles CAD drawings.","ingest service","handles","CAD drawings")),
 C("H02","coordinated_lists","The dashboard displays queue depth and worker health.", E("The dashboard displays queue depth.","dashboard","displays","queue depth"),E("The dashboard displays worker health.","dashboard","displays","worker health")),
 C("H03","coordinated_lists","The archive stores prompts, model responses, reviewer notes, and source links.", E("The archive stores prompts.","archive","stores","prompts"),E("The archive stores model responses.","archive","stores","model responses"),E("The archive stores reviewer notes.","archive","stores","reviewer notes"),E("The archive stores source links.","archive","stores","source links")),
 C("H04","coordinated_lists","The gateway validates OAuth tokens, checks rate limits, and records request IDs.", E("The gateway validates OAuth tokens.","gateway","validates","OAuth tokens"),E("The gateway checks rate limits.","gateway","checks","rate limits"),E("The gateway records request IDs.","gateway","records","request IDs")),
 C("H05","coordinated_lists","The experiment records CPU throughput, GPU throughput, memory consumption, and power draw.", E("The experiment records CPU throughput.","experiment","records","CPU throughput"),E("The experiment records GPU throughput.","experiment","records","GPU throughput"),E("The experiment records memory consumption.","experiment","records","memory consumption"),E("The experiment records power draw.","experiment","records","power draw")),
 C("H06","coordinated_lists","The package exports a tokenizer, a vector index, and a citation formatter.", E("The package exports a tokenizer.","package","exports","tokenizer"),E("The package exports a vector index.","package","exports","vector index"),E("The package exports a citation formatter.","package","exports","citation formatter")),
 # 2 causal scope
 C("H07","causal_scope","Queueing fell because the workers were provisioned in larger pools.", E("Queueing fell because the workers were provisioned in larger pools.","queueing","fell","workers","provisioned","larger pools",q=[Q("causal","pool provisioning causes lower queueing","because") ])),
 C("H08","causal_scope","The cache avoided repeated parsing, thereby shortening document preparation.", E("The cache avoided repeated parsing, thereby shortening document preparation.","cache","avoided","repeated parsing","shortening","document preparation",q=[Q("causal","avoided parsing causes shorter preparation","thereby") ])),
 C("H09","causal_scope","The alert fired, so the operator paused the rollout.", E("The alert fired, so the operator paused the rollout.","alert","fired","operator","paused","rollout",q=[Q("causal","alert led to rollout pause","so") ])),
 C("H10","causal_scope","The index was compacted; as a result, disk reads became less frequent.", E("The index was compacted; as a result, disk reads became less frequent.","index","compacted","disk reads","less frequent",q=[Q("causal","compaction led to fewer reads","as a result") ])),
 C("H11","causal_scope","The report was delayed due to a failed checksum validation.", E("The report was delayed due to a failed checksum validation.","report","delayed","failed checksum validation",q=[Q("causal","checksum failure caused delay","due to") ])),
 C("H12","causal_scope","The shard count increased, leading to more parallel retrieval requests.", E("The shard count increased, leading to more parallel retrieval requests.","shard count","increased","more parallel retrieval requests",q=[Q("causal","increased shard count leads to parallel requests","leading to") ])),
 # 3 conditional scope
 C("H13","conditional_scope","If the signing key expires, new uploads are rejected.", E("If the signing key expires, new uploads are rejected.","signing key","expires","new uploads","rejected",q=[Q("condition","key expiration","if the signing key expires") ])),
 C("H14","conditional_scope","Unless a citation can be resolved, the draft remains marked provisional.", E("Unless a citation can be resolved, the draft remains marked provisional.","citation","resolved","draft","remains","provisional",q=[Q("condition","citation unresolved","unless a citation can be resolved") ])),
 C("H15","conditional_scope","When a request carries a tenant ID, the service applies tenant-specific filtering.", E("When a request carries a tenant ID, the service applies tenant-specific filtering.","request","tenant ID","service","applies","tenant-specific filtering",q=[Q("condition","request carries tenant ID","when a request carries a tenant ID") ])),
 C("H16","conditional_scope","The backup runs only if the primary snapshot is complete.", E("The backup runs only if the primary snapshot is complete.","backup","runs","primary snapshot","complete",q=[Q("condition","snapshot completeness","only if the primary snapshot is complete") ])),
 C("H17","conditional_scope","Provided that the checksum matches, the importer accepts the bundle.", E("Provided that the checksum matches, the importer accepts the bundle.","checksum","matches","importer","accepts","bundle",q=[Q("condition","checksum match","provided that the checksum matches") ])),
 C("H18","conditional_scope","As long as the lease remains valid, the client may reuse its session.", E("As long as the lease remains valid, the client may reuse its session.","lease","remains valid","client","may reuse","session",q=[Q("condition","valid lease","as long as the lease remains valid") ])),
 # 4 qualifier attachment
 C("H19","qualifier_attachment","Atlas reached 88% recall on the HarborQA set in 2026.", E("Atlas reached 88% recall on the HarborQA set in 2026.","Atlas","reached","88%","recall","HarborQA set","2026",q=[Q("quantity","recall","88%"),Q("date","year","2026")])),
 C("H20","qualifier_attachment","Northstar acquired Lumen for $740 million in 2022.", E("Northstar acquired Lumen for $740 million in 2022.","Northstar","acquired","Lumen","$740 million","2022",q=[Q("quantity","purchase price","$740 million"),Q("date","year","2022")],rel=R("Northstar","acquired","Lumen"))),
 C("H21","qualifier_attachment","The parser processed 31,500 pages during the March evaluation.", E("The parser processed 31,500 pages during the March evaluation.","parser","processed","31,500 pages","March evaluation",q=[Q("quantity","pages","31,500 pages"),Q("date","period","March evaluation")])),
 C("H22","qualifier_attachment","In the Toronto lab, version 3.4 indexed 6.2 million records.", E("Version 3.4 indexed 6.2 million records in the Toronto lab.","version 3.4","indexed","6.2 million records","Toronto lab",q=[Q("quantity","record count","6.2 million records"),Q("version","release","version 3.4"),Q("location","site","Toronto lab")])),
 C("H23","qualifier_attachment","During 2021–2023, the coastal sensor logged 14 readings per hour.", E("The coastal sensor logged 14 readings per hour during 2021–2023.","coastal sensor","logged","14 readings per hour","2021–2023",q=[Q("quantity","rate","14 readings per hour"),Q("date","period","2021–2023","2021-2023")])),
 C("H24","qualifier_attachment","Release 5.2 classified 97 of 120 samples as valid in the Alpine dataset.", E("Release 5.2 classified 97 of 120 samples as valid in the Alpine dataset.","release 5.2","classified","97 of 120 samples","valid","Alpine dataset",q=[Q("quantity","sample result","97 of 120 samples"),Q("version","release","release 5.2")])),
 # 5 negation + relationship
 C("H25","negation_relationship","Orchid never licensed the codec from Pine Labs.", E("Orchid never licensed the codec from Pine Labs.","Orchid","licensed","codec","Pine Labs",q=[Q("negation","never licensed","never")],rel=R("Orchid","licensed","Pine Labs"))),
 C("H26","negation_relationship","The renderer does not call the billing API.", E("The renderer does not call the billing API.","renderer","call","billing API",q=[Q("negation","does not call","does not") ])),
 C("H27","negation_relationship","Neither the cache nor the proxy was enabled for external traffic.", E("The cache was not enabled for external traffic.","cache","enabled","external traffic",q=[Q("negation","cache disabled","neither","not")]),E("The proxy was not enabled for external traffic.","proxy","enabled","external traffic",q=[Q("negation","proxy disabled","nor","not")])),
 C("H28","negation_relationship","Mira did not transfer the dataset to Sol during 2020.", E("Mira did not transfer the dataset to Sol during 2020.","Mira","transfer","dataset","Sol","2020",q=[Q("negation","no transfer","did not"),Q("date","year","2020")],rel=R("Mira","transfer","Sol"))),
 C("H29","negation_relationship","The collector operates without contacting the telemetry vendor.", E("The collector operates without contacting the telemetry vendor.","collector","operates","contacting","telemetry vendor",q=[Q("negation","no vendor contact","without") ])),
 C("H30","negation_relationship","Cedar did not merge with Rowan; it acquired Vale instead.", E("Cedar did not merge with Rowan.","Cedar","merge","Rowan",q=[Q("negation","no merger","did not")],rel=R("Cedar","merge","Rowan")),E("Cedar acquired Vale.","Cedar","acquired","Vale",rel=R("Cedar","acquired","Vale")),review="Pronoun 'it' refers to Cedar; any reconstructed output that resolves this must be manually checked."),
 # 6 comparison structure
 C("H31","comparison_structure","Engine P showed lower variance than Engine Q.", E("Engine P showed lower variance than Engine Q.","Engine P","lower variance","Engine Q",q=[Q("comparison","variance direction","lower variance than Engine Q")],rel=R("Engine P","lower variance","Engine Q"))),
 C("H32","comparison_structure","Protocol R used more memory than Protocol S.", E("Protocol R used more memory than Protocol S.","Protocol R","more memory","Protocol S",q=[Q("comparison","memory direction","more memory than Protocol S")],rel=R("Protocol R","more memory","Protocol S"))),
 C("H33","comparison_structure","Relative to the legacy parser, the new parser produced fewer warnings.", E("The new parser produced fewer warnings than the legacy parser.","new parser","fewer warnings","legacy parser",q=[Q("comparison","warning count","fewer warnings than the legacy parser","relative to the legacy parser")],rel=R("new parser","fewer warnings","legacy parser"))),
 C("H34","comparison_structure","The revised index was less expensive to rebuild compared with the previous index.", E("The revised index was less expensive to rebuild compared with the previous index.","revised index","less expensive","rebuild","previous index",q=[Q("comparison","rebuild cost","less expensive","compared with")],rel=R("revised index","less expensive","previous index"))),
 C("H35","comparison_structure","System K achieved a greater compression ratio than System J.", E("System K achieved a greater compression ratio than System J.","System K","greater compression ratio","System J",q=[Q("comparison","compression ratio","greater compression ratio than System J")],rel=R("System K","greater compression ratio","System J"))),
 C("H36","comparison_structure","Compared with batch mode, streaming mode returned results sooner but consumed more bandwidth.", E("Streaming mode returned results sooner than batch mode.","streaming mode","returned results sooner","batch mode",q=[Q("comparison","response time","sooner than batch mode")],rel=R("streaming mode","sooner","batch mode")),E("Streaming mode consumed more bandwidth than batch mode.","streaming mode","consumed more bandwidth","batch mode",q=[Q("comparison","bandwidth","more bandwidth than batch mode")],rel=R("streaming mode","more bandwidth","batch mode"))),
 # 7 multi-entity relationships
 C("H37","multi_entity_relationships","Aster licensed its map tiles to Brio and supplied routing data to Coda.", E("Aster licensed its map tiles to Brio.","Aster","licensed","map tiles","Brio",rel=R("Aster","licensed","Brio")),E("Aster supplied routing data to Coda.","Aster","supplied","routing data","Coda",rel=R("Aster","supplied","Coda"))),
 C("H38","multi_entity_relationships","Dr. Imani designed the protocol, and the Delta group audited it.", E("Dr. Imani designed the protocol.","Dr. Imani","designed","protocol",rel=R("Dr. Imani","designed","protocol")),E("The Delta group audited the protocol.","Delta group","audited","protocol",rel=R("Delta group","audited","protocol")),review="Pronoun 'it' refers to the protocol and requires coreference-aware adjudication."),
 C("H39","multi_entity_relationships","Morrow acquired Finch while Vale invested in Quill.", E("Morrow acquired Finch.","Morrow","acquired","Finch",rel=R("Morrow","acquired","Finch")),E("Vale invested in Quill.","Vale","invested","Quill",rel=R("Vale","invested","Quill"))),
 C("H40","multi_entity_relationships","The observatory trained the detector, whereas the city agency deployed it.", E("The observatory trained the detector.","observatory","trained","detector",rel=R("observatory","trained","detector")),E("The city agency deployed the detector.","city agency","deployed","detector",rel=R("city agency","deployed","detector")),review="Pronoun 'it' refers to the detector."),
 C("H41","multi_entity_relationships","Aster sent the blueprint to Brio before Coda reviewed the prototype.", E("Aster sent the blueprint to Brio.","Aster","sent","blueprint","Brio",rel=R("Aster","sent","Brio")),E("Coda reviewed the prototype.","Coda","reviewed","prototype",rel=R("Coda","reviewed","prototype"))),
 C("H42","multi_entity_relationships","Nile supplied batteries to Orin, which delivered them to Pax.", E("Nile supplied batteries to Orin.","Nile","supplied","batteries","Orin",rel=R("Nile","supplied","Orin")),E("Orin delivered batteries to Pax.","Orin","delivered","batteries","Pax",rel=R("Orin","delivered","Pax")),review="Relative pronoun 'which' and object pronoun 'them' require coreference adjudication."),
 # 8 technical multi-component
 C("H43","technical_components","The indexing stack applies OCR, layout parsing, and table extraction before storage.", E("The indexing stack applies OCR.","indexing stack","applies","OCR"),E("The indexing stack applies layout parsing.","indexing stack","applies","layout parsing"),E("The indexing stack applies table extraction.","indexing stack","applies","table extraction")),
 C("H44","technical_components","The ranker applies query expansion, reciprocal-rank fusion, and diversity filtering.", E("The ranker applies query expansion.","ranker","applies","query expansion"),E("The ranker applies reciprocal-rank fusion.","ranker","applies","reciprocal-rank fusion"),E("The ranker applies diversity filtering.","ranker","applies","diversity filtering")),
 C("H45","technical_components","The verifier checks source entailment, quotation alignment, and citation coverage.", E("The verifier checks source entailment.","verifier","checks","source entailment"),E("The verifier checks quotation alignment.","verifier","checks","quotation alignment"),E("The verifier checks citation coverage.","verifier","checks","citation coverage")),
 C("H46","technical_components","The serving path performs tokenization, embedding lookup, and approximate-neighbor search.", E("The serving path performs tokenization.","serving path","performs","tokenization"),E("The serving path performs embedding lookup.","serving path","performs","embedding lookup"),E("The serving path performs approximate-neighbor search.","serving path","performs","approximate-neighbor search")),
 C("H47","technical_components","The multimodal retriever fuses audio embeddings with frame-level visual features.", E("The multimodal retriever fuses audio embeddings with frame-level visual features.","multimodal retriever","fuses","audio embeddings","frame-level visual features")),
 C("H48","technical_components","The RAG service combines OCR-derived text search with a graph-based entity lookup.", E("The RAG service combines OCR-derived text search.","RAG service","combines","OCR-derived text search"),E("The RAG service combines graph-based entity lookup.","RAG service","combines","graph-based entity lookup")),
 # 9 pronoun/anaphora; expected safe output may retain discourse context
 C("H49","pronoun_anaphora","Juniper released the command-line tool in 2025. It later added signed package updates.", E("Juniper released the command-line tool in 2025.","Juniper","released","command-line tool","2025",q=[Q("date","year","2025")]),E("Juniper later added signed package updates.","Juniper","added","signed package updates"),review="Second sentence uses 'It' with Juniper/tool antecedent ambiguity; safe normalization requires review."),
 C("H50","pronoun_anaphora","The crawler found two mirrors and indexed them overnight.", E("The crawler found two mirrors.","crawler","found","two mirrors"),E("The crawler indexed the mirrors overnight.","crawler","indexed","mirrors","overnight"),review="Pronoun 'them' refers to the mirrors; output splitting without an explicit antecedent needs review."),
 C("H51","pronoun_anaphora","A new cache sits beside the broker, which forwards misses to storage.", E("A new cache sits beside the broker.","cache","sits","broker"),E("The broker forwards misses to storage.","broker","forwards","misses","storage"),review="Relative 'which' could attach to broker or cache; annotation assumes broker from intended reading."),
 C("H52","pronoun_anaphora","The recorder encrypted each segment before it uploaded those files.", E("The recorder encrypted each segment.","recorder","encrypted","each segment"),E("The recorder uploaded the files.","recorder","uploaded","files"),review="'it' likely refers to recorder and 'those files' to segments; coreference is not expected."),
 C("H53","pronoun_anaphora","Aster opened a support ticket. This triggered a review by the security team.", E("Aster opened a support ticket.","Aster","opened","support ticket"),E("Opening the support ticket triggered a security-team review.","support ticket","triggered","review","security team"),review="Demonstrative 'This' refers to the prior event, not a simple noun phrase."),
 C("H54","pronoun_anaphora","Three replicas were restarted, after which they rejoined the cluster.", E("Three replicas were restarted.","three replicas","restarted"),E("The replicas rejoined the cluster.","replicas","rejoined","cluster"),review="Pronoun 'they' refers to the replicas; reconstructed standalone claim requires review."),
 # 10 causal + qualifier
 C("H55","causal_qualifier","Because the router cache was warmed in April 2026, median lookup time fell by 22 ms.", E("Warming the router cache in April 2026 caused median lookup time to fall by 22 ms.","router cache","warmed","April 2026","median lookup time","fell","22 ms",q=[Q("date","month and year","April 2026"),Q("quantity","latency change","22 ms"),Q("causal","cache warming caused lookup reduction","because") ])),
 C("H56","causal_qualifier","The codec reduced archive size by 17% through dictionary sharing on the Northwind corpus.", E("Dictionary sharing in the codec reduced archive size by 17% on the Northwind corpus.","codec","reduced","archive size","17%","dictionary sharing","Northwind corpus",q=[Q("quantity","size reduction","17%"),Q("causal","dictionary sharing caused reduction","through") ])),
 C("H57","causal_qualifier","Since the node count doubled, nightly indexing completed 40 minutes earlier.", E("Doubling the node count caused nightly indexing to complete 40 minutes earlier.","node count","doubled","nightly indexing","completed","40 minutes earlier",q=[Q("quantity","time difference","40 minutes earlier"),Q("causal","node count caused earlier completion","since") ])),
 C("H58","causal_qualifier","The classifier reached 93% precision on the Birch set after hard-negative training was added.", E("Adding hard-negative training caused the classifier to reach 93% precision on the Birch set.","classifier","reached","93%","precision","Birch set","hard-negative training","added",q=[Q("quantity","precision","93%"),Q("causal","hard-negative training associated with precision","after") ])),
 C("H59","causal_qualifier","As a result of the May patch, upload failures dropped by 11% across European regions.", E("The May patch caused upload failures to drop by 11% across European regions.","May patch","upload failures","dropped","11%","European regions",q=[Q("date","month","May"),Q("quantity","failure reduction","11%"),Q("causal","patch caused fewer failures","as a result") ])),
 C("H60","causal_qualifier","Additional replicas were added in 2024, resulting in 2.5 times the ingest rate.", E("Adding replicas in 2024 resulted in 2.5 times the ingest rate.","replicas","added","2024","2.5 times","ingest rate",q=[Q("date","year","2024"),Q("quantity","rate multiplier","2.5 times"),Q("causal","replicas resulted in higher ingest rate","resulting in") ])),
 # 11 comparison + qualifier/condition
 C("H61","comparison_qualifier","With both encoders evaluated on the same clips, Falcon produced 9% more matches than Heron.", E("Falcon produced 9% more matches than Heron when both encoders used the same clips.","Falcon","9% more matches","Heron","both encoders","same clips",q=[Q("comparison","match count","more matches than Heron"),Q("quantity","match difference","9%"),Q("condition","same clip set","when both encoders evaluated on the same clips")],rel=R("Falcon","more matches","Heron"))),
 C("H62","comparison_qualifier","During the 2026 trial, parser M had 80 ms lower p95 latency than parser N.", E("Parser M had 80 ms lower p95 latency than parser N during the 2026 trial.","parser M","80 ms lower","p95 latency","parser N","2026 trial",q=[Q("comparison","latency direction","lower p95 latency than parser N"),Q("quantity","latency gap","80 ms"),Q("date","trial period","2026 trial")],rel=R("parser M","lower p95 latency","parser N"))),
 C("H63","comparison_qualifier","Method T needed fewer database reads than Method U while keeping the same error rate.", E("Method T needed fewer database reads than Method U.","Method T","fewer database reads","Method U",q=[Q("comparison","read count","fewer database reads than Method U")],rel=R("Method T","fewer database reads","Method U")),E("Method T kept the same error rate as Method U.","Method T","same error rate","Method U",q=[Q("comparison","error rate","same error rate")],rel=R("Method T","same error rate","Method U"))),
 C("H64","comparison_qualifier","If the input resolution is fixed, the compact model is 6% faster than the full model.", E("The compact model is 6% faster than the full model when input resolution is fixed.","compact model","6% faster","full model","input resolution","fixed",q=[Q("comparison","speed","faster than the full model"),Q("quantity","speed difference","6%"),Q("condition","fixed input resolution","if the input resolution is fixed")],rel=R("compact model","faster","full model"))),
 C("H65","comparison_qualifier","Compared with the 2023 baseline, the updated scheduler made 14% fewer retries under peak load.", E("The updated scheduler made 14% fewer retries than the 2023 baseline under peak load.","updated scheduler","14% fewer retries","2023 baseline","peak load",q=[Q("comparison","retry count","fewer retries than the 2023 baseline") ,Q("quantity","retry reduction","14%"),Q("date","baseline year","2023"),Q("condition","peak load","under peak load")],rel=R("updated scheduler","fewer retries","2023 baseline"))),
 C("H66","comparison_qualifier","Provided that both jobs use identical hardware, job A consumes 2 GB less memory than job B.", E("Job A consumes 2 GB less memory than job B when both jobs use identical hardware.","job A","2 GB less memory","job B","identical hardware",q=[Q("comparison","memory","less memory than job B"),Q("quantity","memory gap","2 GB"),Q("condition","identical hardware","provided that both jobs use identical hardware")],rel=R("job A","less memory","job B"))),
 # 12 mixed high difficulty
 C("H67","mixed_high_difficulty","After the 2025 migration, the team moved the archive to CedarCloud, which cut retrieval cost by 13%.", E("The team moved the archive to CedarCloud after the 2025 migration.","team","moved","archive","CedarCloud","2025",q=[Q("date","year","2025")],rel=R("team","moved","CedarCloud")),E("Moving the archive to CedarCloud cut retrieval cost by 13%.","archive","CedarCloud","cut","retrieval cost","13%",q=[Q("quantity","cost reduction","13%"),Q("causal","migration reduced cost","which cut")]),review="Relative clause 'which' may refer to the migration or archive move; causal attachment needs review."),
 C("H68","mixed_high_difficulty","If the corpus exceeds 4 million passages, the sharded index routes queries to three regions and records the selected shard.", E("If the corpus exceeds 4 million passages, the sharded index routes queries to three regions.","corpus","exceeds","4 million passages","sharded index","routes queries","three regions",q=[Q("condition","large corpus","if the corpus exceeds 4 million passages"),Q("quantity","corpus size","4 million passages"),Q("quantity","region count","three regions")]),E("If the corpus exceeds 4 million passages, the sharded index records the selected shard.","corpus","exceeds","4 million passages","sharded index","records","selected shard",q=[Q("condition","large corpus","if the corpus exceeds 4 million passages"),Q("quantity","corpus size","4 million passages")])),
 C("H69","mixed_high_difficulty","Neither Aurora nor Beacon signed the 2024 license, although Aurora later purchased the codec from Cygnus.", E("Aurora did not sign the 2024 license.","Aurora","sign","2024 license",q=[Q("negation","did not sign","neither","not"),Q("date","year","2024")]),E("Beacon did not sign the 2024 license.","Beacon","sign","2024 license",q=[Q("negation","did not sign","nor","not"),Q("date","year","2024")]),E("Aurora later purchased the codec from Cygnus.","Aurora","purchased","codec","Cygnus",rel=R("Aurora","purchased","Cygnus"))),
 C("H70","mixed_high_difficulty","Because the Mosaic retriever fuses title matches with image embeddings, it returned 26% more relevant items on the gallery set.", E("The Mosaic retriever fuses title matches with image embeddings.","Mosaic retriever","fuses","title matches","image embeddings"),E("Fusing title matches with image embeddings caused the Mosaic retriever to return 26% more relevant items on the gallery set.","Mosaic retriever","returned","26% more","relevant items","gallery set","fuses","title matches","image embeddings",q=[Q("quantity","item increase","26%"),Q("causal","fusion caused more relevant items","because")]),review="Pronoun 'it' refers to Mosaic retriever; event-level causal relation requires review."),
 C("H71","mixed_high_difficulty","While Vega used 1.8 GB less memory than Lyra on the 2024 benchmark, Lyra completed the batch 12 seconds sooner.", E("Vega used 1.8 GB less memory than Lyra on the 2024 benchmark.","Vega","1.8 GB less memory","Lyra","2024 benchmark",q=[Q("comparison","memory","less memory than Lyra"),Q("quantity","memory gap","1.8 GB"),Q("date","benchmark year","2024")],rel=R("Vega","less memory","Lyra")),E("Lyra completed the batch 12 seconds sooner than Vega.","Lyra","completed","batch","12 seconds sooner","Vega",q=[Q("comparison","completion time","sooner than Vega"),Q("quantity","time gap","12 seconds")],rel=R("Lyra","sooner","Vega"))),
 C("H72","mixed_high_difficulty","Unless the signature is valid, the gateway rejects the package; when valid, it stores the manifest in PostgreSQL.", E("The gateway rejects the package unless the signature is valid.","gateway","rejects","package","signature","valid",q=[Q("condition","invalid signature","unless the signature is valid")]),E("When the signature is valid, the gateway stores the manifest in PostgreSQL.","signature","valid","gateway","stores","manifest","PostgreSQL",q=[Q("condition","valid signature","when valid","when the signature is valid")]),review="Elliptical 'when valid' may refer to signature; preserve the condition attachment in adjudication."),
]

REVIEW_CATEGORIES = {"pronoun_anaphora"}

def enrich(case: dict[str, Any]) -> dict[str, Any]:
    d = dict(case)
    d["required_components"] = [g for e in d["expected_claims"] for g in e["required_components"]]
    d["qualifiers"] = [q for e in d["expected_claims"] for q in e["qualifiers"]]
    d["relationships"] = [e["relationship"] for e in d["expected_claims"] if e["relationship"]]
    return d

def run() -> dict[str, Any]:
    if RAW.exists() or SUMMARY.exists():
        raise SystemExit("Refusing to overwrite existing held-out artifacts.")
    if len(DATASET) != 72 or len({c["id"] for c in DATASET}) != len(DATASET):
        raise RuntimeError(f"Dataset integrity check failed: {len(DATASET)} cases")
    category_counts = defaultdict(int)
    for c in DATASET: category_counts[c["category"]] += 1
    if len(category_counts) != 12 or set(category_counts.values()) != {6}:
        raise RuntimeError(f"Expected 12 categories x 6: {dict(category_counts)}")
    available, dependency_reason, parser = frozen.dependency_parser_status()
    methods = [("sentence", frozen.sentence_analyzer), ("rule_based", frozen.rule_based_analyzer),
               ("dependency", None), ("hybrid", frozen.hybrid_analyzer)]
    raw_rows = []
    for original_case in DATASET:
        case = enrich(original_case)
        for method, analyzer in methods:
            start = time.perf_counter()
            if method == "dependency" and not available:
                outputs, runtime = [], time.perf_counter() - start
                ev = frozen.evaluate(case, outputs, available=False)
            else:
                outputs = frozen.dependency_analyzer(case["original_text"], parser) if method == "dependency" else analyzer(case["original_text"])
                runtime = time.perf_counter() - start
                ev = frozen.evaluate(case, outputs, available=True)
            review_reason = None
            # The frozen evaluator is lexical. For hand-designated coreference /
            # attachment ambiguity, report REVIEW when it cannot prove PASS.
            if ev["status"] not in ("PASS", "UNAVAILABLE") and case.get("manual_review_reason"):
                ev["status_before_review"] = ev["status"]
                ev["status"] = "REVIEW"
                review_reason = case["manual_review_reason"]
            elif ev["status"] not in ("PASS", "UNAVAILABLE") and case["category"] in REVIEW_CATEGORIES:
                ev["status_before_review"] = ev["status"]
                ev["status"] = "REVIEW"
                review_reason = "Coreference/anaphora makes lexical-only adjudication uncertain."
            row = {"case_id": case["id"], "category": case["category"], "original_text": case["original_text"],
                   "expected_claim_count": case["expected_claim_count"], "expected_claims": case["expected_claims"],
                   "ground_truth_annotations": {k: case.get(k) for k in ("required_components","dates","quantities","conditions","negations","comparisons","causal_links","relationships","technical_components")},
                   "method": method, "outputs": outputs, "evaluation": ev, "runtime_seconds": runtime,
                   "unavailable_reason": dependency_reason if method == "dependency" and not available else None,
                   "manual_review_reason": review_reason}
            raw_rows.append(row)
    with RAW.open("x", encoding="utf-8") as f:
        for row in raw_rows: f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summaries = {}
    categories = {}
    review_ids = defaultdict(list)
    for method, _ in methods:
        rows = [r for r in raw_rows if r["method"] == method]
        available_rows = [r for r in rows if r["evaluation"]["status"] != "UNAVAILABLE"]
        summaries[method] = frozen.method_summary(raw_rows, method)
        for r in available_rows:
            if r["evaluation"]["status"] == "REVIEW": review_ids[method].append(r["case_id"])
        categories[method] = {}
        for cat in sorted(category_counts):
            subset = [r for r in rows if r["category"] == cat and r["evaluation"]["status"] != "UNAVAILABLE"]
            categories[method][cat] = {
                "cases": len(subset), "pass": sum(r["evaluation"]["status"] == "PASS" for r in subset),
                "fail": sum(r["evaluation"]["status"] == "FAIL" for r in subset),
                "review": sum(r["evaluation"]["status"] == "REVIEW" for r in subset),
                "exact": sum(bool(r["evaluation"].get("exact_structural_accuracy")) for r in subset),
                "under_split_cases": sum(bool(r["evaluation"].get("under_split")) for r in subset),
                "over_split_cases": sum(bool(r["evaluation"].get("over_split")) for r in subset)}
    first_summary = json.loads((ROOT / "baseline_nlp_claim_analysis_summary.json").read_text(encoding="utf-8"))
    # The frozen first benchmark stores per-method metrics under this key.
    first_methods = first_summary.get("method_results", {})
    generalization = {}
    for m in ("sentence", "rule_based", "dependency", "hybrid"):
        fm = first_methods.get(m, {}) if isinstance(first_methods, dict) else {}
        hs = summaries[m]
        generalization[m] = {"first_exact_accuracy": fm.get("exact_structural_accuracy"),
                             "heldout_exact_accuracy": hs.get("exact_structural_accuracy"),
                             "exact_accuracy_delta": hs.get("exact_structural_accuracy") - fm["exact_structural_accuracy"] if isinstance(fm.get("exact_structural_accuracy"), (int, float)) and isinstance(hs.get("exact_structural_accuracy"), (int, float)) else None,
                             "first_component_coverage": fm.get("component_coverage"),
                             "heldout_component_coverage": hs.get("component_coverage"),
                             "component_coverage_delta": hs.get("component_coverage") - fm["component_coverage"] if isinstance(fm.get("component_coverage"), (int, float)) and isinstance(hs.get("component_coverage"), (int, float)) else None,
                             "first_under_split_cases": fm.get("under_split_cases"),
                             "heldout_under_split_cases": hs.get("under_split_cases"),
                             "under_split_delta": hs.get("under_split_cases") - fm["under_split_cases"] if isinstance(fm.get("under_split_cases"), (int, float)) else None,
                             "first_over_split_cases": fm.get("over_split_cases"),
                             "heldout_over_split_cases": hs.get("over_split_cases"),
                             "over_split_delta": hs.get("over_split_cases") - fm["over_split_cases"] if isinstance(fm.get("over_split_cases"), (int, float)) else None,
                             "first_qualifier_preservation": fm.get("qualifier_preservation"),
                             "heldout_qualifier_preservation": hs.get("qualifier_preservation"),
                             "first_relationship_preservation": fm.get("relationship_preservation"),
                             "heldout_relationship_preservation": hs.get("relationship_preservation"),
                             "relationship_preservation_delta": hs.get("relationship_preservation") - fm["relationship_preservation"] if isinstance(fm.get("relationship_preservation"), (int, float)) and isinstance(hs.get("relationship_preservation"), (int, float)) else None,
                             "note": "Descriptive only; datasets and denominators differ. No significance claim."}
    failure_examples = []
    for row in raw_rows:
        ev = row["evaluation"]
        if ev["status"] in {"FAIL", "REVIEW"}:
            failure_examples.append({"case_id": row["case_id"], "category": row["category"],
                "method": row["method"], "status": ev["status"], "original": row["original_text"],
                "expected_claims": row["expected_claims"], "outputs": row["outputs"],
                "under_split": ev.get("under_split"), "over_split": ev.get("over_split"),
                "unmapped_outputs": ev.get("unmapped_outputs"),
                "meaning_preservation_failures": ev.get("meaning_preservation_failures"),
                "manual_review_reason": row.get("manual_review_reason")})
    summary = {"created_utc": datetime.now(timezone.utc).isoformat(), "python": sys.executable,
               "dataset_cases": len(DATASET), "category_counts": dict(category_counts),
               "method_order": [m for m, _ in methods], "dependency_parser_available": available,
               "dependency_parser_status": dependency_reason, "method_summaries": summaries,
               "per_category": categories, "review_case_ids": dict(review_ids),
               "failure_examples": failure_examples,
               "manual_review_reasons": {c["id"]: c["manual_review_reason"] for c in DATASET if c.get("manual_review_reason")},
               "first_vs_heldout": generalization,
               "evaluator_limitations": ["Frozen evaluator uses lexical phrase anchors, not a semantic oracle.",
                   "Hand-flagged coreference/attachment cases become REVIEW only when lexical evaluation did not PASS.",
                   "A PASS means the frozen structural checks matched; it is not proof of full semantic equivalence.",
                   "First-vs-heldout percentages are descriptive and not statistically significant or directly comparable."],
               "raw_results": str(RAW)}
    with SUMMARY.open("x", encoding="utf-8") as f: json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Held-out cases: {len(DATASET)} across {len(category_counts)} categories")
    print(f"Dependency parser: {'AVAILABLE' if available else 'UNAVAILABLE'} — {dependency_reason}")
    print("METHOD          PASS  FAIL  REVIEW  EXACT   COVERAGE  UNDER  OVER  MEAN_MS")
    for m in summaries:
        x = summaries[m]
        if x["cases_available"]:
            print(f"{m:15} {x['pass']:4} {x['fail']:5} {x['review']:7} {x['exact_structural_accuracy']:.3f}   {x['component_coverage']:.3f}   {x['under_split_cases']:5} {x['over_split_cases']:5} {x['mean_runtime_seconds']*1000:.3f}")
        else: print(f"{m:15} UNAVAILABLE")
    print("\nPER CATEGORY (PASS/FAIL/REVIEW)")
    for cat in sorted(category_counts):
        print(cat + ": " + " | ".join(f"{m}={categories[m][cat]['pass']}/{categories[m][cat]['fail']}/{categories[m][cat]['review']}" for m in categories))
    print("\nQUALIFIER PRESERVATION")
    for m, x in summaries.items(): print(m + ": " + json.dumps(x["qualifier_preservation"], sort_keys=True))
    print("\nRELATIONSHIP PRESERVATION")
    for m, x in summaries.items(): print(f"{m}: {x['relationship_links_preserved']}/{x['relationship_links_total']} = {x['relationship_preservation']}")
    print("\nREVIEW CASES")
    for m, ids in review_ids.items(): print(f"{m}: {', '.join(ids) if ids else '(none)'}")
    print(f"\nRaw JSONL: {RAW}\nSummary JSON: {SUMMARY}")
    return summary

if __name__ == "__main__": run()
