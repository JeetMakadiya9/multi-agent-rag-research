"""Controlled, offline comparison of deterministic claim analyzers.

No production modules, LLMs, external APIs, or network services are used.
Ground-truth annotations are hand-authored below. Dependency parsing is only
enabled if an installed parser and its local English model are available.
"""

from __future__ import annotations

import importlib.util
import json
import re
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "baseline_nlp_claim_analysis_results.jsonl"
SUMMARY_PATH = ROOT / "baseline_nlp_claim_analysis_summary.json"
RANDOM_SEED = 20260925


def Q(kind: str, name: str, *alternatives: str) -> dict[str, Any]:
    return {"type": kind, "name": name, "alternatives": list(alternatives)}


def E(text: str, components: list[list[str]], *, qualifiers: list[dict[str, Any]] | None = None,
      relationship: dict[str, list[str]] | None = None) -> dict[str, Any]:
    return {"text": text, "required_components": components, "qualifiers": qualifiers or [], "relationship": relationship}


def REL(subject: str, predicate: str, object_: str) -> dict[str, list[str]]:
    return {"subject": [subject], "predicate": [predicate], "object": [object_]}


DATASET: list[dict[str, Any]] = [
    # A. Simple atomic facts (4)
    {"id":"A01","category":"A_simple_atomic","original_text":"Python is dynamically typed.","expected_claims":[E("Python is dynamically typed.",[["Python"],["dynamically typed","dynamic typing"]])]},
    {"id":"A02","category":"A_simple_atomic","original_text":"FAISS supports vector similarity search.","expected_claims":[E("FAISS supports vector similarity search.",[["FAISS"],["supports","performs"],["vector similarity search","similarity search","vector search"]])]},
    {"id":"A03","category":"A_simple_atomic","original_text":"The model uses BM25.","expected_claims":[E("The model uses BM25.",[["model"],["uses","utilizes"],["BM25"]])]},
    {"id":"A04","category":"A_simple_atomic","original_text":"PostgreSQL stores relational data.","expected_claims":[E("PostgreSQL stores relational data.",[["PostgreSQL"],["stores"],["relational data"]])]},
    # B. Two independent facts (4)
    {"id":"B01","category":"B_two_independent_facts","original_text":"Python is dynamically typed, while Java is statically typed.","expected_claims":[E("Python is dynamically typed.",[["Python"],["dynamically typed","dynamic typing"]]),E("Java is statically typed.",[["Java"],["statically typed","static typing"]])]},
    {"id":"B02","category":"B_two_independent_facts","original_text":"FAISS performs vector search and BM25 performs lexical retrieval.","expected_claims":[E("FAISS performs vector search.",[["FAISS"],["performs"],["vector search","vector retrieval"]]),E("BM25 performs lexical retrieval.",[["BM25"],["performs"],["lexical retrieval","lexical search"]])]},
    {"id":"B03","category":"B_two_independent_facts","original_text":"The API accepts JSON, and the worker stores results in SQLite.","expected_claims":[E("The API accepts JSON.",[["API"],["accepts"],["JSON"]]),E("The worker stores results in SQLite.",[["worker"],["stores"],["results"],["SQLite"]])]},
    {"id":"B04","category":"B_two_independent_facts","original_text":"Model A achieved higher recall than Model B, whereas Model C had lower latency than Model D.","expected_claims":[E("Model A achieved higher recall than Model B.",[["Model A"],["higher recall","greater recall"],["Model B"]],qualifiers=[Q("comparison","recall direction","higher recall than Model B","greater recall than Model B")],relationship=REL("Model A","higher recall","Model B")),E("Model C had lower latency than Model D.",[["Model C"],["lower latency","less latency"],["Model D"]],qualifiers=[Q("comparison","latency direction","lower latency than Model D","less latency than Model D")],relationship=REL("Model C","lower latency","Model D"))]},
    # C. Three components (4)
    {"id":"C01","category":"C_three_components","original_text":"The pipeline uses FAISS, BM25, and a cross-encoder.","expected_claims":[E("The pipeline uses FAISS.",[["pipeline"],["uses"],["FAISS"]]),E("The pipeline uses BM25.",[["pipeline"],["uses"],["BM25"]]),E("The pipeline uses a cross-encoder.",[["pipeline"],["uses"],["cross-encoder","cross encoder"]])]},
    {"id":"C02","category":"C_three_components","original_text":"The service indexes documents, retrieves candidates, and reranks them.","expected_claims":[E("The service indexes documents.",[["service"],["indexes"],["documents"]]),E("The service retrieves candidates.",[["service"],["retrieves"],["candidates"]]),E("The service reranks candidates.",[["service"],["reranks"],["candidates"]])]},
    {"id":"C03","category":"C_three_components","original_text":"The platform accepts PDFs, extracts text, and stores chunks.","expected_claims":[E("The platform accepts PDFs.",[["platform"],["accepts"],["PDFs","PDF"]]),E("The platform extracts text.",[["platform"],["extracts"],["text"]]),E("The platform stores chunks.",[["platform"],["stores"],["chunks"]])]},
    {"id":"C04","category":"C_three_components","original_text":"The search stack combines dense retrieval, lexical retrieval, and reranking.","expected_claims":[E("The search stack combines dense retrieval.",[["search stack"],["combines"],["dense retrieval"]]),E("The search stack combines lexical retrieval.",[["search stack"],["combines"],["lexical retrieval"]]),E("The search stack combines reranking.",[["search stack"],["combines"],["reranking"]])]},
    # D. Relationship plus date (4)
    {"id":"D01","category":"D_relationship_date","original_text":"Company A acquired Company B in 2024.","expected_claims":[E("Company A acquired Company B in 2024.",[["Company A"],["acquired","acquisition"],["Company B"]],qualifiers=[Q("date","year","2024")],relationship=REL("Company A","acquired","Company B"))]},
    {"id":"D02","category":"D_relationship_date","original_text":"Company X partnered with Company Y in March 2025.","expected_claims":[E("Company X partnered with Company Y in March 2025.",[["Company X"],["partnered","partnership"],["Company Y"]],qualifiers=[Q("date","month and year","March 2025","2025-03")],relationship=REL("Company X","partnered","Company Y"))]},
    {"id":"D03","category":"D_relationship_date","original_text":"In June 2022, Lab R released Model S.","expected_claims":[E("Lab R released Model S in June 2022.",[["Lab R"],["released"],["Model S"]],qualifiers=[Q("date","month and year","June 2022","2022-06")],relationship=REL("Lab R","released","Model S"))]},
    {"id":"D04","category":"D_relationship_date","original_text":"The merger of Firm P and Firm Q closed on 15 July 2021.","expected_claims":[E("The merger between Firm P and Firm Q closed on 15 July 2021.",[["Firm P"],["merger"],["Firm Q"],["closed"]],qualifiers=[Q("date","full date","15 July 2021","July 15 2021","2021-07-15")],relationship=REL("Firm P","merger","Firm Q"))]},
    # E. Quantities (4)
    {"id":"E01","category":"E_quantities","original_text":"The model achieved 95% accuracy on 10,000 samples.","expected_claims":[E("The model achieved 95% accuracy on 10,000 samples.",[["model"],["achieved"],["95%","95 percent"],["accuracy"],["10,000 samples","10000 samples"]],qualifiers=[Q("quantity","accuracy percentage","95%","95 percent"),Q("quantity","sample count","10,000 samples","10000 samples")])]},
    {"id":"E02","category":"E_quantities","original_text":"Latency fell by 120 ms, and throughput rose by 18%.","expected_claims":[E("Latency fell by 120 ms.",[["latency"],["fell","decreased","reduced"],["120 ms","120 milliseconds"]],qualifiers=[Q("quantity","latency change","120 ms","120 milliseconds")]),E("Throughput rose by 18%.",[["throughput"],["rose","increased"],["18%","18 percent"]],qualifiers=[Q("quantity","throughput change","18%","18 percent")])]},
    {"id":"E03","category":"E_quantities","original_text":"The index stores 2.4 million vectors.","expected_claims":[E("The index stores 2.4 million vectors.",[["index"],["stores"],["2.4 million","2400000"],["vectors"]],qualifiers=[Q("quantity","vector count","2.4 million","2400000")])]},
    {"id":"E04","category":"E_quantities","original_text":"Model B processed 640 images in 8 seconds.","expected_claims":[E("Model B processed 640 images in 8 seconds.",[["Model B"],["processed"],["640 images"],["8 seconds"]],qualifiers=[Q("quantity","image count","640 images"),Q("quantity","duration","8 seconds")])]},
    # F. Conditions (4)
    {"id":"F01","category":"F_conditions","original_text":"If GPU acceleration is enabled, the system processes images faster.","expected_claims":[E("If GPU acceleration is enabled, the system processes images faster.",[["system"],["processes"],["images"],["faster"]],qualifiers=[Q("condition","GPU enabled","if GPU acceleration is enabled","when GPU acceleration is enabled")])]},
    {"id":"F02","category":"F_conditions","original_text":"When the input contains images, the visual retrieval module is activated.","expected_claims":[E("When the input contains images, the visual retrieval module is activated.",[["input"],["contains"],["images"],["visual retrieval module"],["activated","enabled"]],qualifiers=[Q("condition","image input","when the input contains images","if the input contains images")])]},
    {"id":"F03","category":"F_conditions","original_text":"The system uses BM25 only when dense retrieval is unavailable.","expected_claims":[E("The system uses BM25 only when dense retrieval is unavailable.",[["system"],["uses"],["BM25"]],qualifiers=[Q("condition","dense retrieval unavailable","only when dense retrieval is unavailable","when dense retrieval is unavailable")])]},
    {"id":"F04","category":"F_conditions","original_text":"If confidence falls below 0.6, the pipeline requests human review.","expected_claims":[E("If confidence falls below 0.6, the pipeline requests human review.",[["pipeline"],["requests"],["human review"]],qualifiers=[Q("condition","confidence threshold","if confidence falls below 0.6","when confidence is below 0.6"),Q("quantity","confidence threshold","0.6")])]},
    # G. Negation (4)
    {"id":"G01","category":"G_negation","original_text":"The model does not use BM25.","expected_claims":[E("The model does not use BM25.",[["model"],["BM25"]],qualifiers=[Q("negation","BM25 non-use","does not use BM25","doesn't use BM25","never uses BM25")])]},
    {"id":"G02","category":"G_negation","original_text":"Company A did not acquire Company B.","expected_claims":[E("Company A did not acquire Company B.",[["Company A"],["Company B"]],qualifiers=[Q("negation","acquisition negation","did not acquire","didn't acquire","never acquired")],relationship=REL("Company A","acquire","Company B"))]},
    {"id":"G03","category":"G_negation","original_text":"The system neither stores raw images nor sends them to external services.","expected_claims":[E("The system does not store raw images.",[["system"],["raw images"]],qualifiers=[Q("negation","image storage negation","neither stores","does not store","doesn't store")]),E("The system does not send raw images to external services.",[["system"],["raw images"],["external services"]],qualifiers=[Q("negation","external sending negation","nor sends","does not send","doesn't send")])]},
    {"id":"G04","category":"G_negation","original_text":"No external API is required when local models are enabled.","expected_claims":[E("No external API is required when local models are enabled.",[["external API"],["required"]],qualifiers=[Q("negation","API non-requirement","no external API is required","external API is not required"),Q("condition","local models enabled","when local models are enabled","if local models are enabled")])]},
    # H. Causal (4)
    {"id":"H01","category":"H_causal","original_text":"GPU acceleration reduced inference latency because computation was moved to the GPU.","expected_claims":[E("GPU acceleration reduced inference latency because computation was moved to the GPU.",[["GPU acceleration"],["reduced"],["inference latency"],["computation"],["moved to the GPU","was moved to the GPU"]],qualifiers=[Q("causal","GPU move causes lower latency","because","since","as")])]},
    {"id":"H02","category":"H_causal","original_text":"Caching lowered database load because repeated results were reused.","expected_claims":[E("Caching lowered database load because repeated results were reused.",[["caching"],["lowered","reduced"],["database load"],["repeated results"],["reused"]],qualifiers=[Q("causal","reuse causes lower load","because","since","as")])]},
    {"id":"H03","category":"H_causal","original_text":"The error rate rose because the sensor calibration drifted.","expected_claims":[E("The error rate rose because the sensor calibration drifted.",[["error rate"],["rose","increased"],["sensor calibration"],["drifted"]],qualifiers=[Q("causal","calibration drift causes error increase","because","since","as")])]},
    {"id":"H04","category":"H_causal","original_text":"Higher compression reduced storage use, so the system retained more records.","expected_claims":[E("Higher compression reduced storage use, so the system retained more records.",[["higher compression"],["reduced"],["storage use"],["system"],["retained"],["more records"]],qualifiers=[Q("causal","storage reduction enables more records","so","therefore","as a result")])]},
    # I. Comparison (4)
    {"id":"I01","category":"I_comparison","original_text":"Model A achieved higher accuracy than Model B.","expected_claims":[E("Model A achieved higher accuracy than Model B.",[["Model A"],["higher accuracy","greater accuracy"],["Model B"]],qualifiers=[Q("comparison","accuracy direction","higher accuracy than Model B","greater accuracy than Model B")],relationship=REL("Model A","higher accuracy","Model B"))]},
    {"id":"I02","category":"I_comparison","original_text":"Index A is smaller than Index B but retrieves results more slowly.","expected_claims":[E("Index A is smaller than Index B.",[["Index A"],["smaller than","less large than"],["Index B"]],qualifiers=[Q("comparison","size direction","smaller than Index B")],relationship=REL("Index A","smaller","Index B")),E("Index A retrieves results more slowly than Index B.",[["Index A"],["retrieves"],["results"],["more slowly","slower than"],["Index B"]],qualifiers=[Q("comparison","speed direction","more slowly than Index B","slower than Index B")],relationship=REL("Index A","slower retrieval","Index B"))]},
    {"id":"I03","category":"I_comparison","original_text":"Version 2 uses less memory than Version 1.","expected_claims":[E("Version 2 uses less memory than Version 1.",[["Version 2"],["less memory","lower memory use"],["Version 1"]],qualifiers=[Q("comparison","memory direction","less memory than Version 1","lower memory than Version 1")],relationship=REL("Version 2","less memory","Version 1"))]},
    {"id":"I04","category":"I_comparison","original_text":"The hybrid retriever is faster than BM25-only retrieval while maintaining equal precision.","expected_claims":[E("The hybrid retriever is faster than BM25-only retrieval.",[["hybrid retriever"],["faster than","quicker than"],["BM25-only retrieval","BM25 only retrieval"]],qualifiers=[Q("comparison","speed direction","faster than BM25-only retrieval")],relationship=REL("hybrid retriever","faster","BM25-only retrieval")),E("The hybrid retriever maintains equal precision.",[["hybrid retriever"],["maintaining equal precision","maintains equal precision","equal precision"]])]},
    # J. Conditional + multi-part (4)
    {"id":"J01","category":"J_conditional_multi_part","original_text":"If the dataset is balanced, Model A performs better than Model B.","expected_claims":[E("If the dataset is balanced, Model A performs better than Model B.",[["dataset"],["balanced"],["Model A"],["performs better","better"],["Model B"]],qualifiers=[Q("condition","balanced dataset","if the dataset is balanced","when the dataset is balanced"),Q("comparison","Model A better","better than Model B","performs better than Model B")],relationship=REL("Model A","performs better","Model B"))]},
    {"id":"J02","category":"J_conditional_multi_part","original_text":"When caching is enabled, latency decreases and throughput increases.","expected_claims":[E("When caching is enabled, latency decreases.",[["latency"],["decreases","falls","reduces"]],qualifiers=[Q("condition","cache enabled","when caching is enabled","if caching is enabled")]),E("When caching is enabled, throughput increases.",[["throughput"],["increases","rises"]],qualifiers=[Q("condition","cache enabled","when caching is enabled","if caching is enabled")])]},
    {"id":"J03","category":"J_conditional_multi_part","original_text":"Unless a GPU is available, the service uses the CPU and limits the batch size to 16.","expected_claims":[E("Unless a GPU is available, the service uses the CPU.",[["service"],["uses"],["CPU"]],qualifiers=[Q("condition","GPU unavailable","unless a GPU is available","if a GPU is not available")]),E("Unless a GPU is available, the service limits the batch size to 16.",[["service"],["limits"],["batch size"],["16"]],qualifiers=[Q("condition","GPU unavailable","unless a GPU is available","if a GPU is not available"),Q("quantity","batch size","16")])]},
    {"id":"J04","category":"J_conditional_multi_part","original_text":"If reranking is enabled, precision rises while recall stays constant.","expected_claims":[E("If reranking is enabled, precision rises.",[["precision"],["rises","increases"]],qualifiers=[Q("condition","reranking enabled","if reranking is enabled","when reranking is enabled")]),E("If reranking is enabled, recall stays constant.",[["recall"],["stays constant","remains constant"]],qualifiers=[Q("condition","reranking enabled","if reranking is enabled","when reranking is enabled")])]},
    # K. Multi-entity relationships (4)
    {"id":"K01","category":"K_multi_entity_relationships","original_text":"Company A acquired Company B and partnered with Company C.","expected_claims":[E("Company A acquired Company B.",[["Company A"],["acquired"],["Company B"]],relationship=REL("Company A","acquired","Company B")),E("Company A partnered with Company C.",[["Company A"],["partnered"],["Company C"]],relationship=REL("Company A","partnered","Company C"))]},
    {"id":"K02","category":"K_multi_entity_relationships","original_text":"Researcher A wrote Library X and contributed to Project Y.","expected_claims":[E("Researcher A wrote Library X.",[["Researcher A"],["wrote","authored"],["Library X"]],relationship=REL("Researcher A","wrote","Library X")),E("Researcher A contributed to Project Y.",[["Researcher A"],["contributed to"],["Project Y"]],relationship=REL("Researcher A","contributed to","Project Y"))]},
    {"id":"K03","category":"K_multi_entity_relationships","original_text":"Service X sends embeddings to Index Y and receives document IDs from it.","expected_claims":[E("Service X sends embeddings to Index Y.",[["Service X"],["sends"],["embeddings"],["Index Y"]],relationship=REL("Service X","sends","Index Y")),E("Service X receives document IDs from Index Y.",[["Service X"],["receives"],["document IDs"],["Index Y","it"]],relationship=REL("Service X","receives","Index Y"))]},
    {"id":"K04","category":"K_multi_entity_relationships","original_text":"Team Alpha developed Model Z, while Team Beta evaluated it.","expected_claims":[E("Team Alpha developed Model Z.",[["Team Alpha"],["developed"],["Model Z"]],relationship=REL("Team Alpha","developed","Model Z")),E("Team Beta evaluated Model Z.",[["Team Beta"],["evaluated"],["Model Z","it"]],relationship=REL("Team Beta","evaluated","Model Z"))]},
    # L. Technical retrieval claims (4)
    {"id":"L01","category":"L_technical_retrieval","original_text":"The retrieval system combines FAISS, BM25, and cross-encoder reranking.","expected_claims":[E("The retrieval system combines FAISS.",[["retrieval system"],["combines"],["FAISS"]]),E("The retrieval system combines BM25.",[["retrieval system"],["combines"],["BM25"]]),E("The retrieval system combines cross-encoder reranking.",[["retrieval system"],["combines"],["cross-encoder reranking","cross encoder reranking"]])]},
    {"id":"L02","category":"L_technical_retrieval","original_text":"The system uses dense retrieval for semantic matching and BM25 for lexical matching.","expected_claims":[E("The system uses dense retrieval for semantic matching.",[["system"],["uses"],["dense retrieval"],["semantic matching"]]),E("The system uses BM25 for lexical matching.",[["system"],["uses"],["BM25"],["lexical matching"]])]},
    {"id":"L03","category":"L_technical_retrieval","original_text":"A cross-encoder reranks the top 20 retrieved candidates.","expected_claims":[E("A cross-encoder reranks the top 20 retrieved candidates.",[["cross-encoder","cross encoder"],["reranks","re-ranks"],["top 20"],["retrieved candidates"]],qualifiers=[Q("quantity","candidate limit","top 20","20 candidates")])]},
    {"id":"L04","category":"L_technical_retrieval","original_text":"The hybrid retriever uses vector search for semantic matches and BM25 for lexical matches.","expected_claims":[E("The hybrid retriever uses vector search for semantic matches.",[["hybrid retriever"],["uses"],["vector search"],["semantic matches"]]),E("The hybrid retriever uses BM25 for lexical matches.",[["hybrid retriever"],["uses"],["BM25"],["lexical matches"]])]},
    # M. Qualifier-heavy claims (4)
    {"id":"M01","category":"M_qualifier_heavy","original_text":"On 12 May 2024, Company A acquired Company B for $2.4 billion.","expected_claims":[E("Company A acquired Company B for $2.4 billion on 12 May 2024.",[["Company A"],["acquired"],["Company B"],["$2.4 billion","2.4 billion"]],qualifiers=[Q("date","full date","12 May 2024","May 12 2024","2024-05-12"),Q("quantity","transaction value","$2.4 billion","2.4 billion")],relationship=REL("Company A","acquired","Company B"))]},
    {"id":"M02","category":"M_qualifier_heavy","original_text":"Model Q achieved 93.7% accuracy on 8,500 images after normalization.","expected_claims":[E("Model Q achieved 93.7% accuracy on 8,500 images after normalization.",[["Model Q"],["achieved"],["93.7%","93.7 percent"],["accuracy"],["8,500 images","8500 images"]],qualifiers=[Q("quantity","accuracy","93.7%","93.7 percent"),Q("quantity","image count","8,500 images","8500 images"),Q("condition","normalization","after normalization","when normalized")])]},
    {"id":"M03","category":"M_qualifier_heavy","original_text":"If 4 GPUs are enabled, the system processes 1,200 images per minute with 15% lower latency.","expected_claims":[E("If 4 GPUs are enabled, the system processes 1,200 images per minute with 15% lower latency.",[["system"],["processes"],["1,200 images per minute","1200 images per minute"],["lower latency","reduced latency"]],qualifiers=[Q("condition","four GPUs enabled","if 4 GPUs are enabled","when 4 GPUs are enabled"),Q("quantity","GPU count","4 GPUs"),Q("quantity","image throughput","1,200 images per minute","1200 images per minute"),Q("quantity","latency reduction","15%","15 percent")])]},
    {"id":"M04","category":"M_qualifier_heavy","original_text":"Company C did not acquire Company D in 2023, unlike Company E, which acquired Company F in 2022.","expected_claims":[E("Company C did not acquire Company D in 2023.",[["Company C"],["Company D"]],qualifiers=[Q("negation","non-acquisition","did not acquire","didn't acquire"),Q("date","year","2023")],relationship=REL("Company C","acquire","Company D")),E("Company E acquired Company F in 2022.",[["Company E"],["acquired"],["Company F"]],qualifiers=[Q("date","year","2022")],relationship=REL("Company E","acquired","Company F"))]},
]


VERBS = sorted(set((
    "is are was were does do did has have had uses use used supports support performs perform combines combine stores store accepts accept indexes index retrieves retrieve reranks rerank extracts extract processes process consumes consume limits limit requests request sends send receives receive acquired acquire partnered partners wrote writes contributed contributes released release achieved achieves fell falls decreased decreases reduced reduces rose rises increased increases lowered lowers drifted enabled enables activated activates contains contain requires required moves moved improves improve exposes exposed retained retains maintains maintained rose remains stays used lowered means operates provides finds validates evaluated developed increases applies stores indexes starts computes changed."
    ).split()), key=len, reverse=True)
VERB_RE = re.compile(r"(?i)\b(" + "|".join(re.escape(v) for v in VERBS) + r")\b")
COORD_RE = re.compile(r"(?i)(,\s*(?:and|but|while|whereas)\s+|\s+(?:and|but|while|whereas)\s+|,\s+)")
LEADING_CONDITION_RE = re.compile(r"(?is)^\s*((?:if|when|unless|provided that|assuming|once)\b.*?),\s*(.+)$")
CAUSAL_WORD_RE = re.compile(r"(?i)\b(because|since|so|therefore|as a result)\b")


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9%$]+", " ", text.casefold()).strip()


def phrase_present(text: str, phrase: str) -> bool:
    a, b = f" {normalize(text)} ", f" {normalize(phrase)} "
    return bool(b.strip()) and b in a


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text.strip()) if part.strip()]


def _has_subject_and_predicate(text: str) -> bool:
    match = VERB_RE.search(text)
    return bool(match and normalize(text[:match.start()]).strip())


def _predicate_at_start(text: str) -> bool:
    return bool(VERB_RE.match(text.strip()))


def _subject_before_predicate(text: str) -> str:
    match = VERB_RE.search(text)
    return text[:match.start()].strip(" ,") if match else ""


def _strip_terminal_punctuation(text: str) -> str:
    return re.sub(r"[.;!?]+\s*$", "", text.strip()).strip(" ,")


def _split_causal(text: str) -> list[str] | None:
    match = re.search(r"(?i)\s+(because|since|so|therefore|as a result)\s+", text)
    if not match:
        return None
    left = text[:match.start()].strip(" ,")
    connector = match.group(1)
    right = text[match.end():].strip()
    if connector.casefold() in {"because", "since"}:
        right = connector.capitalize() + " " + right
    elif connector.casefold() in {"so", "therefore", "as a result"}:
        right = connector.capitalize() + " " + right
    return [left, right] if left and right else None


def _safe_clause_split(text: str, *, conservative: bool) -> list[str] | None:
    body = text.strip()
    condition = ""
    guard = LEADING_CONDITION_RE.match(body)
    if guard:
        condition = guard.group(1).strip().rstrip(",") + ", "
        body = guard.group(2).strip()

    # Causal clauses are kept intact by the conservative hybrid. The rule
    # baseline splits them so the evaluator can measure loss of causal linkage.
    if not conservative:
        causal = _split_causal(body)
        if causal:
            return [condition + _strip_terminal_punctuation(part) for part in causal]

    for match in COORD_RE.finditer(body):
        left = body[:match.start()].strip(" ,")
        right = body[match.end():].strip(" ,")
        if not left or not right or not _has_subject_and_predicate(left):
            continue
        inherited = ""
        if _predicate_at_start(right):
            inherited = _subject_before_predicate(left)
            if not inherited:
                continue
            right = inherited + " " + right
        elif not _has_subject_and_predicate(right):
            continue
        # Guard against splitting coordinated list members (FAISS and BM25)
        # that do not introduce a predicate.
        if not _has_subject_and_predicate(right):
            continue
        left_parts = _safe_clause_split(left, conservative=conservative) or [left]
        right_parts = _safe_clause_split(right, conservative=conservative) or [right]
        parts = left_parts + right_parts
        cleaned = [condition + _strip_terminal_punctuation(part) for part in parts]
        return [part for part in cleaned if part.strip()]

    if condition:
        return [condition + _strip_terminal_punctuation(body)]
    return None


def sentence_analyzer(text: str) -> list[str]:
    return split_sentences(text)


def rule_based_analyzer(text: str) -> list[str]:
    outputs: list[str] = []
    for sentence in split_sentences(text):
        outputs.extend(_safe_clause_split(sentence, conservative=False) or [sentence])
    return [item.strip() for item in outputs if item.strip()]


def hybrid_analyzer(text: str) -> list[str]:
    outputs: list[str] = []
    for sentence in split_sentences(text):
        outputs.extend(_safe_clause_split(sentence, conservative=True) or [sentence])
    return [item.strip() for item in outputs if item.strip()]


def dependency_parser_status() -> tuple[bool, str, Any]:
    """Return an installed spaCy dependency parser only when its model exists."""
    if importlib.util.find_spec("spacy") is None:
        return False, "spaCy and its dependency parser are not installed in the project venv.", None
    import spacy  # type: ignore[import-not-found]
    for model in ("en_core_web_sm", "en_core_web_md", "en_core_web_lg"):
        try:
            nlp = spacy.load(model)
            if "parser" in nlp.pipe_names:
                return True, f"Loaded installed spaCy parser model: {model}", nlp
        except OSError:
            continue
    return False, "spaCy is installed but no local English dependency parser model is available.", None


def dependency_analyzer(text: str, nlp: Any) -> list[str]:
    """Conservative parser-backed splitting of conjunctive predicates."""
    doc = nlp(text)
    outputs: list[str] = []
    # The parser method emits one clause per sentence unless it finds a verbal
    # conjunction with a recoverable shared subject. All modifiers stay with
    # their parser subtree text; no production algorithm is reused.
    for sent in doc.sents:
        verbs = [token for token in sent if token.pos_ in {"VERB", "AUX"}]
        root_subjects = [child for child in sent.root.children if child.dep_ in {"nsubj", "nsubjpass", "csubj"}]
        base_subject = " ".join(token.text for token in root_subjects)
        clauses: list[str] = []
        for verb in verbs:
            if verb.dep_ == "conj":
                subject_tokens = [child for child in verb.children if child.dep_ in {"nsubj", "nsubjpass", "csubj"}]
                subject = " ".join(token.text for token in subject_tokens) or base_subject
                if subject:
                    clauses.append(subject + " " + verb.subtree.__iter__().__next__().text if False else subject + " " + " ".join(token.text for token in verb.subtree))
        outputs.extend(clauses or [sent.text])
    return [item.strip() for item in outputs if item.strip()]


METHODS = [
    ("sentence", sentence_analyzer),
    ("rule_based", rule_based_analyzer),
    ("dependency", None),
    ("hybrid", hybrid_analyzer),
]


def phrase_group_present(text: str, group: list[str]) -> bool:
    return any(phrase_present(text, option) for option in group)


def all_groups_present(text: str, groups: list[list[str]]) -> bool:
    return all(phrase_group_present(text, group) for group in groups)


def qualifier_preserved_same_claim(outputs: list[str], claim: dict[str, Any], qualifier: dict[str, Any]) -> bool:
    groups = claim["required_components"]
    return any(all_groups_present(output, groups) and phrase_group_present(output, qualifier["alternatives"]) for output in outputs)


def relation_preserved_same_claim(outputs: list[str], relation: dict[str, list[str]]) -> bool:
    groups = [relation["subject"], relation["predicate"], relation["object"]]
    return any(all_groups_present(output, groups) for output in outputs)


def evaluate(case: dict[str, Any], outputs: list[str], *, available: bool = True) -> dict[str, Any]:
    if not available:
        return {"status": "UNAVAILABLE", "reason": "No installed dependency parser/model.", "exact_structural_accuracy": None}
    claims = case["expected_claims"]
    flattened = "\n".join(outputs)
    output_to_expected: list[list[int]] = []
    expected_to_outputs: dict[int, list[int]] = defaultdict(list)
    for oi, output in enumerate(outputs):
        matched = []
        for ci, claim in enumerate(claims):
            if all_groups_present(output, claim["required_components"]):
                matched.append(ci)
                expected_to_outputs[ci].append(oi)
        output_to_expected.append(matched)

    component_total = 0
    component_covered = 0
    for claim in claims:
        for group in claim["required_components"]:
            component_total += 1
            component_covered += int(phrase_group_present(flattened, group))

    expected_claim_covered = []
    qualifier_records: list[dict[str, Any]] = []
    relation_records: list[dict[str, Any]] = []
    for ci, claim in enumerate(claims):
        full_outputs = []
        for oi, output in enumerate(outputs):
            if all_groups_present(output, claim["required_components"]) and all(phrase_group_present(output, q["alternatives"]) for q in claim["qualifiers"]):
                if claim["relationship"]:
                    rel = claim["relationship"]
                    if not all_groups_present(output, [rel["subject"], rel["predicate"], rel["object"]]):
                        continue
                full_outputs.append(oi)
        expected_claim_covered.append(bool(full_outputs))
        for qualifier in claim["qualifiers"]:
            qualifier_records.append({"type": qualifier["type"], "name": qualifier["name"], "preserved": qualifier_preserved_same_claim(outputs, claim, qualifier), "claim_index": ci})
        if claim["relationship"]:
            relation_records.append({"subject": claim["relationship"]["subject"], "predicate": claim["relationship"]["predicate"], "object": claim["relationship"]["object"], "preserved": relation_preserved_same_claim(outputs, claim["relationship"]), "claim_index": ci})

    merged_outputs = [{"output_index": oi, "expected_claim_indices": targets, "text": outputs[oi]} for oi, targets in enumerate(output_to_expected) if len(targets) > 1]
    unmapped = [{"output_index": oi, "text": output} for oi, (output, targets) in enumerate(zip(outputs, output_to_expected)) if not targets]
    oversplit_claims = []
    for ci, claim in enumerate(claims):
        indexes = expected_to_outputs.get(ci, [])
        complete = expected_claim_covered[ci]
        qualifier_detached = any(not qualifier_preserved_same_claim(outputs, claim, q) and phrase_group_present(flattened, q["alternatives"]) for q in claim["qualifiers"])
        if not complete and (len(set(indexes)) > 1 or qualifier_detached):
            oversplit_claims.append({"claim_index": ci, "output_indices": sorted(set(indexes)), "qualifier_detached": qualifier_detached})
    output_count_delta = len(outputs) - len(claims)
    oversplit = bool(oversplit_claims) or output_count_delta > 0
    undersplit = bool(merged_outputs)

    qualifier_summary: dict[str, dict[str, int]] = {}
    for record in qualifier_records:
        totals = qualifier_summary.setdefault(record["type"], {"total": 0, "preserved": 0})
        totals["total"] += 1
        totals["preserved"] += int(record["preserved"])
    relation_total = len(relation_records)
    relation_preserved = sum(record["preserved"] for record in relation_records)
    missing_claim_indices = [i for i, status in enumerate(expected_claim_covered) if not status]
    schema_valid = bool(outputs) and all(isinstance(value, str) and value.strip() for value in outputs)
    # These analyzers copy source text or reconstruct with source spans. Any
    # lexical novelty is surfaced; semantic changes are evaluated separately.
    source_tokens = set(re.findall(r"[a-z0-9%$]+", normalize(case["original_text"])))
    novel_tokens = sorted({token for output in outputs for token in re.findall(r"[a-z0-9%$]+", normalize(output)) if token not in source_tokens})
    structural_exact = bool(
        schema_valid and len(outputs) == len(claims) and all(expected_claim_covered)
        and not merged_outputs and not unmapped and not oversplit
        and component_covered == component_total and relation_preserved == relation_total
        and all(record["preserved"] for record in qualifier_records) and not novel_tokens
    )
    if structural_exact:
        status = "PASS"
    elif not schema_valid or novel_tokens:
        status = "REVIEW"
    else:
        status = "FAIL"
    return {
        "status": status,
        "schema_valid": schema_valid,
        "expected_claim_count": len(claims),
        "actual_claim_count": len(outputs),
        "exact_structural_accuracy": structural_exact,
        "covered_claims": sum(expected_claim_covered),
        "expected_claims": len(claims),
        "claim_coverage": sum(expected_claim_covered) / len(claims) if claims else 1.0,
        "component_coverage": component_covered / component_total if component_total else 1.0,
        "components_covered": component_covered,
        "components_total": component_total,
        "qualifier_results": qualifier_records,
        "qualifier_summary": qualifier_summary,
        "relationship_results": relation_records,
        "relationship_preservation_rate": relation_preserved / relation_total if relation_total else None,
        "over_split": oversplit,
        "over_split_claims": oversplit_claims,
        "under_split": undersplit,
        "merged_outputs": merged_outputs,
        "unmapped_outputs": unmapped,
        "unmapped_output_count": len(unmapped),
        "novel_lexical_tokens": novel_tokens,
        "missing_claim_indices": missing_claim_indices,
        "meaning_preservation_failures": [r for r in qualifier_records if not r["preserved"]] + [r for r in relation_records if not r["preserved"]],
    }


def run_method(case: dict[str, Any], method: str, analyzer: Any, parser: Any, parser_available: bool, parser_reason: str) -> dict[str, Any]:
    start = time.perf_counter()
    if method == "dependency":
        if parser_available:
            outputs = dependency_analyzer(case["original_text"], parser)
            available = True
            reason = None
        else:
            outputs = []
            available = False
            reason = parser_reason
    else:
        outputs = analyzer(case["original_text"])
        available = True
        reason = None
    runtime = time.perf_counter() - start
    result = evaluate(case, outputs, available=available)
    result.update({"method": method, "runtime_seconds": runtime, "outputs": outputs, "unavailable_reason": reason})
    return result


def method_summary(records: list[dict[str, Any]], method: str) -> dict[str, Any]:
    rows = [r for r in records if r["method"] == method]
    available = [r for r in rows if r["evaluation"]["status"] != "UNAVAILABLE"]
    all_components = sum(r["evaluation"]["components_total"] for r in available)
    covered_components = sum(r["evaluation"]["components_covered"] for r in available)
    qualifiers: dict[str, dict[str, int]] = {}
    for row in available:
        for qtype, stats in row["evaluation"]["qualifier_summary"].items():
            target = qualifiers.setdefault(qtype, {"total": 0, "preserved": 0})
            target["total"] += stats["total"]
            target["preserved"] += stats["preserved"]
    q_rates = {key: {**value, "rate": value["preserved"] / value["total"] if value["total"] else None} for key, value in qualifiers.items()}
    rel_total = sum(len(r["evaluation"]["relationship_results"]) for r in available)
    rel_ok = sum(sum(x["preserved"] for x in r["evaluation"]["relationship_results"]) for r in available)
    latencies = [r["runtime_seconds"] for r in available]
    merged_count = sum(len(r["evaluation"]["merged_outputs"]) for r in available)
    expected_total = sum(r["evaluation"]["expected_claims"] for r in available)
    return {
        "cases_total": len(rows),
        "cases_available": len(available),
        "unavailable_cases": len(rows) - len(available),
        "pass": sum(r["evaluation"]["status"] == "PASS" for r in available),
        "fail": sum(r["evaluation"]["status"] == "FAIL" for r in available),
        "review": sum(r["evaluation"]["status"] == "REVIEW" for r in available),
        "exact_structural_accuracy": sum(r["evaluation"]["exact_structural_accuracy"] for r in available) / len(available) if available else None,
        "component_coverage": covered_components / all_components if all_components else None,
        "components_covered": covered_components,
        "components_total": all_components,
        "qualifier_preservation": q_rates,
        "relationship_preservation": rel_ok / rel_total if rel_total else None,
        "relationship_links_preserved": rel_ok,
        "relationship_links_total": rel_total,
        "over_split_cases": sum(r["evaluation"]["over_split"] for r in available),
        "under_split_cases": sum(r["evaluation"]["under_split"] for r in available),
        "merged_outputs": merged_count,
        "unmapped_output_content_count": sum(r["evaluation"]["unmapped_output_count"] for r in available),
        "novel_lexical_token_count": sum(len(r["evaluation"]["novel_lexical_tokens"]) for r in available),
        "expected_claim_count": expected_total,
        "mean_runtime_seconds": statistics.mean(latencies) if latencies else None,
        "median_runtime_seconds": statistics.median(latencies) if latencies else None,
        "min_runtime_seconds": min(latencies) if latencies else None,
        "max_runtime_seconds": max(latencies) if latencies else None,
    }


def main() -> None:
    if SUMMARY_PATH.exists():
        raise SystemExit(f"Refusing to overwrite benchmark summary: {SUMMARY_PATH}")
    parser_available, parser_reason, parser = dependency_parser_status()
    if RESULTS_PATH.exists():
        print(f"Reading existing raw results without overwriting: {RESULTS_PATH}", flush=True)
        records = [json.loads(line) for line in RESULTS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        records: list[dict[str, Any]] = []
        for case in DATASET:
            case = dict(case)
            case["expected_claim_count"] = len(case["expected_claims"])
            case["required_components"] = [group for claim in case["expected_claims"] for group in claim["required_components"]]
            case["qualifiers"] = [qualifier for claim in case["expected_claims"] for qualifier in claim["qualifiers"]]
            case["relationships"] = [claim["relationship"] for claim in case["expected_claims"] if claim["relationship"]]
            case["conditions"] = [q for q in case["qualifiers"] if q["type"] == "condition"]
            case["negations"] = [q for q in case["qualifiers"] if q["type"] == "negation"]
            case["quantities"] = [q for q in case["qualifiers"] if q["type"] == "quantity"]
            case["dates"] = [q for q in case["qualifiers"] if q["type"] == "date"]
            case["comparisons"] = [q for q in case["qualifiers"] if q["type"] == "comparison"]
            case["causal_links"] = [q for q in case["qualifiers"] if q["type"] == "causal"]
            for method, analyzer in METHODS:
                run = run_method(case, method, analyzer, parser, parser_available, parser_reason)
                record = {"case_id": case["id"], "category": case["category"], "original_text": case["original_text"], "expected_claim_count": case["expected_claim_count"], "expected_claims": case["expected_claims"], "ground_truth_annotations": {key: case[key] for key in ("required_components", "qualifiers", "relationships", "conditions", "negations", "quantities", "dates", "comparisons", "causal_links")}, "method": method, "outputs": run["outputs"], "evaluation": {key: value for key, value in run.items() if key not in {"method", "outputs", "runtime_seconds", "unavailable_reason"}}, "runtime_seconds": run["runtime_seconds"], "unavailable_reason": run["unavailable_reason"]}
                records.append(record)
        # Create once. A subsequent run consumes this file rather than replacing it.
        with RESULTS_PATH.open("x", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    methods = {method: method_summary(records, method) for method, _ in METHODS}
    category_summary: dict[str, Any] = {}
    for category in sorted({item["category"] for item in DATASET}):
        category_summary[category] = {}
        ids = {item["id"] for item in DATASET if item["category"] == category}
        for method, _ in METHODS:
            selected = [r for r in records if r["case_id"] in ids and r["method"] == method and r["evaluation"]["status"] != "UNAVAILABLE"]
            comps_total = sum(r["evaluation"]["components_total"] for r in selected)
            comps_cov = sum(r["evaluation"]["components_covered"] for r in selected)
            category_summary[category][method] = {
                "case_count": len(selected),
                "pass": sum(r["evaluation"]["status"] == "PASS" for r in selected),
                "fail": sum(r["evaluation"]["status"] == "FAIL" for r in selected),
                "review": sum(r["evaluation"]["status"] == "REVIEW" for r in selected),
                "exact_structural_accuracy": sum(r["evaluation"]["exact_structural_accuracy"] for r in selected) / len(selected) if selected else None,
                "component_coverage": comps_cov / comps_total if comps_total else None,
            }

    qualifier_summary: dict[str, Any] = {}
    for method, _ in METHODS:
        qualifier_summary[method] = methods[method]["qualifier_preservation"]

    per_case = []
    for case in DATASET:
        per_case.append({"id": case["id"], "category": case["category"], "original_text": case["original_text"], "expected_claim_count": len(case["expected_claims"]), "expected_claims": case["expected_claims"], "methods": {method: next(r for r in records if r["case_id"] == case["id"] and r["method"] == method) for method, _ in METHODS}})

    failures = []
    for row in records:
        if row["evaluation"]["status"] not in {"PASS", "UNAVAILABLE"}:
            failures.append({"case_id": row["case_id"], "category": row["category"], "method": row["method"], "original": row["original_text"], "expected_structure": row["expected_claims"], "actual_output": row["outputs"], "status": row["evaluation"]["status"], "failure_type": {"missing_claims": row["evaluation"]["missing_claim_indices"], "over_split": row["evaluation"]["over_split"], "under_split": row["evaluation"]["under_split"], "unmapped_output_content": row["evaluation"]["unmapped_outputs"], "qualifier_failures": row["evaluation"]["meaning_preservation_failures"], "novel_lexical_tokens": row["evaluation"]["novel_lexical_tokens"]}})

    previous_llm = {"label": "PREVIOUS LLM-ONLY BASELINE (separate 16-case/48-call benchmark)", "exact_accuracy": 0.583, "component_coverage": 0.736, "omission_rate": 0.264, "unmapped_output_rate": 0.120, "over_split_rate": 0.125, "meaning_preservation_failure_rate": 0.417, "json_parse_success_rate": 0.958, "directly_comparable": False, "reason": "Different dataset, LLM output schema, annotation/evaluation rules, and denominator. Shown for context only; do not rank against this benchmark."}
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "random_seed": RANDOM_SEED,
        "network_used": False,
        "ollama_used": False,
        "external_api_used": False,
        "dependency_parser_available": parser_available,
        "dependency_parser_status": parser_reason,
        "installed_nlp_packages": {name: bool(importlib.util.find_spec(name)) for name in ("spacy", "stanza", "nltk", "benepar", "transformers", "flair", "trankit")},
        "benchmark_cases": len(DATASET),
        "methods": [method for method, _ in METHODS],
        "method_results": methods,
        "category_results": category_summary,
        "qualifier_specific_results": qualifier_summary,
        "per_case_results": per_case,
        "failure_examples": failures,
        "previous_llm_baseline": previous_llm,
        "evaluator_limitations": ["Phrase-anchor matching is conservative and not a semantic oracle.", "Novel lexical tokens and unmapped clauses are flags for review; copied fragments can be unmapped without containing invented facts.", "Dependency-based method is unavailable because no parser library/local parser model was found; it is recorded as unavailable and excluded from method accuracy denominators."],
        "results_jsonl": str(RESULTS_PATH),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Python: {sys.executable} ({sys.version.split()[0]})")
    print(f"Dataset: {len(DATASET)} cases; methods: {', '.join(methods)}; parser available: {parser_available} ({parser_reason})")
    print("METHOD SUMMARY")
    print("method             PASS/available   exact       coverage    over-split   under-split   mean runtime")
    for method, values in methods.items():
        avail = values["cases_available"]
        exact = "N/A" if values["exact_structural_accuracy"] is None else f"{values['exact_structural_accuracy']:.3f}"
        coverage = "N/A" if values["component_coverage"] is None else f"{values['component_coverage']:.3f}"
        mean = "N/A" if values["mean_runtime_seconds"] is None else f"{values['mean_runtime_seconds']:.6f}s"
        print(f"{method:18} {values['pass']:>2}/{avail:<3}          {exact:>7}      {coverage:>7}      {values['over_split_cases']:>3}          {values['under_split_cases']:>3}          {mean}")
    print("QUALIFIER PRESERVATION")
    for method, qtypes in qualifier_summary.items():
        print(method + ": " + json.dumps(qtypes, ensure_ascii=False))
    print(f"Results JSONL: {RESULTS_PATH}")
    print(f"Summary JSON: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
