# ============================================================
# RAG ENGINE PRO
# ============================================================
#
# Advanced Grounded Document Retrieval Engine
#
# Architecture:
#
# PDF
#   ↓
# Structural extraction
#   ↓
# Parent / child chunking
#   ↓
# ┌───────────────────────────────────────────────┐
# │ Dense semantic retrieval                      │
# │ BM25 lexical retrieval                       │
# │ Exact phrase / token retrieval                │
# │ Query expansion                               │
# └───────────────────────────────────────────────┘
#   ↓
# Reciprocal Rank Fusion
#   ↓
# Cross Encoder reranking
#   ↓
# MMR diversity selection
#   ↓
# Adjacent-context expansion
#   ↓
# Evidence scoring
#   ↓
# Grounded context
#
# This file contains NO LLM generation.
# Generation remains the responsibility of app.py.
#
# ============================================================

from __future__ import annotations

import hashlib
import math
import re
import time

from dataclasses import dataclass, field
from collections import Counter, defaultdict
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import faiss
import numpy as np

from rank_bm25 import BM25Okapi
from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder,
)
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

RERANKER_MODEL_NAME = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# ------------------------------------------------------------
# Chunking
# ------------------------------------------------------------

CHILD_CHUNK_SIZE = 900
CHILD_CHUNK_OVERLAP = 150

PARENT_CHUNK_SIZE = 2400
PARENT_CHUNK_OVERLAP = 300

MIN_CHUNK_LENGTH = 40

# ------------------------------------------------------------
# Retrieval
# ------------------------------------------------------------

VECTOR_TOP_K = 20
BM25_TOP_K = 20
EXACT_TOP_K = 15

RRF_K = 60

RERANK_TOP_K = 30
FINAL_TOP_K = 8

# ------------------------------------------------------------
# Adaptive retrieval
# ------------------------------------------------------------

MIN_RETRIEVAL_K = 5
MAX_RETRIEVAL_K = 30

# ------------------------------------------------------------
# Relevance
# ------------------------------------------------------------

# NOTE:
#
# FAISS IndexFlatL2 is being used with normalized embeddings.
# Therefore the squared L2 distance approximately corresponds
# to cosine distance.
#
# Smaller = better.
#
VECTOR_RELEVANCE_THRESHOLD = 1.25

RERANK_MIN_SCORE = -3.0

# ------------------------------------------------------------
# MMR
# ------------------------------------------------------------

MMR_LAMBDA = 0.72

# ------------------------------------------------------------
# Context expansion
# ------------------------------------------------------------

MAX_ADJACENT_CHUNKS = 2

# ------------------------------------------------------------
# Query processing
# ------------------------------------------------------------

MAX_QUERY_VARIANTS = 4

MAX_HISTORY_MESSAGES = 8

# ------------------------------------------------------------
# Evidence
# ------------------------------------------------------------

MIN_EVIDENCE_SCORE = 0.12

# ============================================================
# CONSTANTS
# ============================================================

NO_ANSWER = (
    "I could not find the answer in the provided document."
)

REFERENCE_WORDS = {
    "it",
    "its",
    "they",
    "them",
    "this",
    "that",
    "these",
    "those",
    "above",
    "below",
}

VISUAL_TERMS = {
    "image",
    "images",
    "figure",
    "fig",
    "diagram",
    "diagrams",
    "chart",
    "charts",
    "graph",
    "graphs",
    "plot",
    "plots",
    "table",
    "tables",
    "architecture",
    "flowchart",
    "flow",
    "illustration",
    "visual",
    "shown",
    "show",
    "depict",
    "depicts",
    "draw",
    "drawing",
    "picture",
    "pictures",
    "screenshot",
    "layout",
    "block diagram",
    "what is shown",
    "what does the diagram",
    "what does the figure",
}

COMPARISON_TERMS = {
    "compare",
    "comparison",
    "difference",
    "differences",
    "versus",
    "vs",
    "better",
    "similarity",
    "similarities",
    "contrast",
}

SUMMARY_TERMS = {
    "summarize",
    "summary",
    "overview",
    "main points",
    "key points",
    "briefly explain",
    "in short",
}

DEFINITION_TERMS = {
    "what is",
    "what are",
    "define",
    "definition",
    "meaning of",
    "explain",
}

LIST_TERMS = {
    "list",
    "types",
    "kinds",
    "examples",
    "advantages",
    "disadvantages",
    "applications",
    "features",
    "steps",
    "components",
}

# ============================================================
# DATA CLASSES
# ============================================================


@dataclass
class PageRecord:
    """
    Represents one PDF page.
    """

    page: int
    text: str

    filename: str = ""

    characters: int = 0

    has_text: bool = False

    visual_hint: bool = False

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.text = self.text or ""
        self.characters = len(self.text)
        self.has_text = bool(
            self.text.strip()
        )


@dataclass
class ChunkRecord:
    """
    Represents one searchable child chunk.
    """

    text: str

    filename: str

    page: int

    chunk_id: int

    parent_id: int

    section: str = ""

    position: int = 0

    token_count: int = 0

    character_count: int = 0

    previous_chunk_id: Optional[int] = None

    next_chunk_id: Optional[int] = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.text = self.text.strip()

        self.character_count = len(
            self.text
        )

        self.token_count = len(
            tokenize_text(self.text)
        )


@dataclass
class ParentRecord:
    """
    Larger contextual unit containing child chunks.
    """

    parent_id: int

    text: str

    filename: str

    page: int

    section: str = ""

    child_ids: List[int] = field(
        default_factory=list
    )


@dataclass
class RetrievalResult:
    """
    Unified retrieval result.
    """

    index: int

    chunk: Dict[str, Any]

    rrf_score: float = 0.0

    vector_rank: Optional[int] = None

    vector_distance: Optional[float] = None

    bm25_rank: Optional[int] = None

    bm25_score: Optional[float] = None

    exact_rank: Optional[int] = None

    exact_score: Optional[float] = None

    rerank_score: Optional[float] = None

    lexical_score: float = 0.0

    semantic_score: float = 0.0

    evidence_score: float = 0.0

    diversity_score: float = 0.0

    source_bonus: float = 0.0

    parent_context: str = ""

    retrieval_reasons: List[str] = field(
        default_factory=list
    )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to a normal dictionary.
        """

        result = {
            "index": self.index,
            "chunk": self.chunk,
            "rrf_score": self.rrf_score,
            "vector_rank": self.vector_rank,
            "vector_distance": self.vector_distance,
            "bm25_rank": self.bm25_rank,
            "bm25_score": self.bm25_score,
            "exact_rank": self.exact_rank,
            "exact_score": self.exact_score,
            "rerank_score": self.rerank_score,
            "lexical_score": self.lexical_score,
            "semantic_score": self.semantic_score,
            "evidence_score": self.evidence_score,
            "diversity_score": self.diversity_score,
            "source_bonus": self.source_bonus,
            "parent_context": self.parent_context,
            "retrieval_reasons": self.retrieval_reasons,
        }

        return result


@dataclass
class QueryAnalysis:
    """
    Structured interpretation of a user question.
    """

    original_query: str

    normalized_query: str

    query_type: str = "general"

    is_followup: bool = False

    is_visual: bool = False

    is_comparison: bool = False

    is_summary: bool = False

    is_definition: bool = False

    is_list_request: bool = False

    entities: List[str] = field(
        default_factory=list
    )

    important_terms: List[str] = field(
        default_factory=list
    )

    query_variants: List[str] = field(
        default_factory=list
    )

    confidence: float = 0.0


@dataclass
class RetrievalDiagnostics:
    """
    Debugging information for the retrieval pipeline.
    """

    query: str

    expanded_query: str = ""

    vector_candidates: int = 0

    bm25_candidates: int = 0

    exact_candidates: int = 0

    fused_candidates: int = 0

    reranked_candidates: int = 0

    final_candidates: int = 0

    best_vector_distance: Optional[float] = None

    best_bm25_score: Optional[float] = None

    best_exact_score: Optional[float] = None

    best_rerank_score: Optional[float] = None

    retrieval_time_ms: float = 0.0

    rerank_time_ms: float = 0.0

    total_time_ms: float = 0.0

    rejected: bool = False

    rejection_reason: str = ""


# ============================================================
# TEXT NORMALIZATION
# ============================================================


def normalize_text(text: str) -> str:
    """
    Normalize whitespace while preserving useful punctuation.
    """

    if not text:
        return ""

    text = text.replace(
        "\r\n",
        "\n",
    )

    text = text.replace(
        "\r",
        "\n",
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


def clean_pdf_text(text: str) -> str:
    """
    Clean text extracted from PDF files.

    Important:
    This does NOT aggressively remove punctuation because
    technical documents often contain equations, identifiers,
    model names, versions, and abbreviations.
    """

    if not text:
        return ""

    text = normalize_text(
        text
    )

    # Fix hyphenation caused by PDF line wrapping.
    text = re.sub(
        r"(\w)-\n(\w)",
        r"\1\2",
        text,
    )

    # Convert isolated newlines inside sentences to spaces.
    text = re.sub(
        r"(?<!\n)\n(?!\n)",
        " ",
        text,
    )

    # Normalize spaces again.
    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    return text.strip()


def normalize_for_search(
    text: str,
) -> str:
    """
    Normalized representation used by lexical search.
    """

    if not text:
        return ""

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9_\-\.]+",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def tokenize_text(
    text: str,
) -> List[str]:
    """
    Technical-document-aware tokenizer.
    """

    if not text:
        return []

    normalized = text.lower()

    return re.findall(
        r"[a-zA-Z0-9_]+(?:[-\.][a-zA-Z0-9_]+)*",
        normalized,
    )


def unique_preserve_order(
    values: Iterable[str],
) -> List[str]:
    """
    Deduplicate while preserving order.
    """

    seen: Set[str] = set()

    result: List[str] = []

    for value in values:
        value = value.strip()

        if not value:
            continue

        key = value.lower()

        if key in seen:
            continue

        seen.add(key)

        result.append(value)

    return result


# ============================================================
# HASHING
# ============================================================


def hash_text(
    text: str,
) -> str:
    """
    SHA-256 text identifier.
    """

    return hashlib.sha256(
        text.encode(
            "utf-8",
            errors="ignore",
        )
    ).hexdigest()


# ============================================================
# HEADING / STRUCTURE DETECTION
# ============================================================


def looks_like_heading(
    line: str,
) -> bool:
    """
    Heuristic heading detector.

    The detector intentionally uses multiple signals rather
    than relying on one regex.
    """

    if not line:
        return False

    line = line.strip()

    if len(line) > 140:
        return False

    if len(line) < 2:
        return False

    # Markdown heading.
    if re.match(
        r"^#{1,6}\s+",
        line,
    ):
        return True

    # Numbered heading.
    if re.match(
        r"^\d+(?:\.\d+)*[\.)]?\s+[A-Z]",
        line,
    ):
        return True

    # Roman numeral heading.
    if re.match(
        r"^[IVXLC]+[\.)]?\s+[A-Z]",
        line,
        flags=re.IGNORECASE,
    ):
        return True

    normalized = (
        line.lower()
        .rstrip(":")
        .strip()
    )

    common_headings = {
        "abstract",
        "introduction",
        "background",
        "methodology",
        "method",
        "methods",
        "materials and methods",
        "results",
        "discussion",
        "conclusion",
        "conclusions",
        "references",
        "related work",
        "literature review",
        "future work",
        "applications",
        "advantages",
        "disadvantages",
        "algorithm",
        "implementation",
        "evaluation",
        "overview",
        "architecture",
        "problem statement",
        "objectives",
        "summary",
        "dataset",
        "experiments",
        "experimental setup",
        "system design",
        "proposed system",
        "proposed methodology",
    }

    if normalized in common_headings:
        return True

    # ALL CAPS heading.
    letters = re.sub(
        r"[^A-Za-z]",
        "",
        line,
    )

    if (
        3 <= len(letters) <= 80
        and letters.isupper()
    ):
        return True

    # Short title-like line.
    words = line.split()

    if (
        1 <= len(words) <= 8
        and len(line) <= 80
        and line[-1:] not in ".?!"
    ):
        uppercase_words = sum(
            1
            for word in words
            if word[:1].isupper()
        )

        if uppercase_words >= max(
            1,
            len(words) // 2,
        ):
            return True

    return False


def split_into_sections(
    text: str,
) -> List[Dict[str, str]]:
    """
    Convert a page into structural sections.
    """

    text = clean_pdf_text(
        text
    )

    if not text:
        return []

    blocks = re.split(
        r"\n\s*\n",
        text,
    )

    sections: List[Dict[str, str]] = []

    current_heading = ""

    current_body: List[str] = []

    def flush() -> None:
        nonlocal current_body

        if not current_body:
            return

        body = "\n\n".join(
            current_body
        ).strip()

        if not body:
            current_body = []
            return

        sections.append(
            {
                "heading": current_heading,
                "text": (
                    f"{current_heading}\n\n{body}"
                    if current_heading
                    else body
                ).strip(),
            }
        )

        current_body = []

    for block in blocks:
        block = block.strip()

        if not block:
            continue

        first_line = (
            block.split(
                "\n",
                1,
            )[0].strip()
        )

        if looks_like_heading(
            first_line
        ):
            flush()

            current_heading = first_line

            remaining = block[
                len(first_line):
            ].strip()

            if remaining:
                current_body.append(
                    remaining
                )

        else:
            current_body.append(
                block
            )

    flush()

    if not sections:
        return [
            {
                "heading": "",
                "text": text,
            }
        ]

    return sections


# ============================================================
# PAGE PROCESSING
# ============================================================


def build_page_records(
    pages: Sequence[Dict[str, Any]],
    filename: str = "",
) -> List[PageRecord]:
    """
    Convert raw page dictionaries into PageRecord objects.
    """

    records: List[PageRecord] = []

    for page_data in pages:
        page_number = int(
            page_data.get(
                "page",
                len(records) + 1,
            )
        )

        text = clean_pdf_text(
            page_data.get(
                "text",
                "",
            )
        )

        records.append(
            PageRecord(
                page=page_number,
                text=text,
                filename=filename,
                visual_hint=detect_visual_content(
                    text
                ),
                metadata={
                    key: value
                    for key, value in page_data.items()
                    if key not in {
                        "page",
                        "text",
                    }
                },
            )
        )

    return records


def detect_visual_content(
    text: str,
) -> bool:
    """
    Detect textual references to visual material.

    This does not claim an image exists.
    It only identifies pages likely to contain visual references.
    """

    if not text:
        return False

    normalized = text.lower()

    for term in VISUAL_TERMS:
        if term in normalized:
            return True

    # Common figure/table notation.
    if re.search(
        r"\b(fig(?:ure)?\.?\s*\d+)\b",
        normalized,
    ):
        return True

    if re.search(
        r"\b(table\s*\d+)\b",
        normalized,
    ):
        return True

    return False


# ============================================================
# PARENT / CHILD CHUNKING
# ============================================================


def create_parent_child_chunks(
    pages: Sequence[PageRecord],
    filename: str,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    """
    Create hierarchical chunks.

    Parent chunks preserve broad context.

    Child chunks are optimized for retrieval.
    """

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            "; ",
            ", ",
            " ",
            "",
        ],
    )

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=PARENT_CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            "; ",
            " ",
            "",
        ],
    )

    children: List[Dict[str, Any]] = []

    parents: List[Dict[str, Any]] = []

    next_child_id = 0

    next_parent_id = 0

    for page in pages:
        if not page.text.strip():
            continue

        sections = split_into_sections(
            page.text
        )

        for section_data in sections:
            section_name = section_data.get(
                "heading",
                "",
            ).strip()

            section_text = section_data.get(
                "text",
                "",
            ).strip()

            if not section_text:
                continue

            parent_texts = (
                [section_text]
                if len(section_text)
                <= PARENT_CHUNK_SIZE
                else parent_splitter.split_text(
                    section_text
                )
            )

            for parent_text in parent_texts:
                parent_text = parent_text.strip()

                if len(parent_text) < MIN_CHUNK_LENGTH:
                    continue

                parent_id = next_parent_id

                next_parent_id += 1

                parent_record = {
                    "parent_id": parent_id,
                    "text": parent_text,
                    "filename": filename,
                    "page": page.page,
                    "section": section_name,
                    "child_ids": [],
                }

                parents.append(
                    parent_record
                )

                child_texts = (
                    [parent_text]
                    if len(parent_text)
                    <= CHILD_CHUNK_SIZE
                    else child_splitter.split_text(
                        parent_text
                    )
                )

                for position, child_text in enumerate(
                    child_texts
                ):
                    child_text = child_text.strip()

                    if len(child_text) < MIN_CHUNK_LENGTH:
                        continue

                    child_id = next_child_id

                    next_child_id += 1

                    child_record = {
                        "text": child_text,
                        "filename": filename,
                        "page": page.page,
                        "chunk_id": child_id,
                        "parent_id": parent_id,
                        "section": section_name,
                        "position": position,
                        "token_count": len(
                            tokenize_text(
                                child_text
                            )
                        ),
                        "character_count": len(
                            child_text
                        ),
                        "previous_chunk_id": None,
                        "next_chunk_id": None,
                        "visual_hint": page.visual_hint,
                        "document_hash": hash_text(
                            filename
                        ),
                    }

                    children.append(
                        child_record
                    )

                    parent_record[
                        "child_ids"
                    ].append(
                        child_id
                    )

    # Link neighboring chunks.
    for index, chunk in enumerate(
        children
    ):
        if index > 0:
            previous = children[
                index - 1
            ]

            same_document = (
                previous["filename"]
                == chunk["filename"]
            )

            same_page = (
                previous["page"]
                == chunk["page"]
            )

            if same_document and same_page:
                chunk[
                    "previous_chunk_id"
                ] = previous[
                    "chunk_id"
                ]

        if index < len(children) - 1:
            following = children[
                index + 1
            ]

            same_document = (
                following["filename"]
                == chunk["filename"]
            )

            same_page = (
                following["page"]
                == chunk["page"]
            )

            if same_document and same_page:
                chunk[
                    "next_chunk_id"
                ] = following[
                    "chunk_id"
                ]

    return children, parents


# ============================================================
# LEGACY-COMPATIBLE CHUNK CREATION
# ============================================================


def create_chunks(
    pages: Sequence[Dict[str, Any]],
    filename: str,
) -> List[Dict[str, Any]]:
    """
    Compatibility function matching your previous application.

    It returns child chunks using the new hierarchical chunker.
    """

    page_records = build_page_records(
        pages,
        filename,
    )

    children, _ = create_parent_child_chunks(
        page_records,
        filename,
    )

    return children


# ============================================================
# QUERY ANALYSIS
# ============================================================


def normalize_query(
    question: str,
) -> str:
    """
    Normalize a user query without destroying technical terms.
    """

    question = question or ""

    question = question.strip()

    question = re.sub(
        r"\s+",
        " ",
        question,
    )

    return question


def contains_any_term(
    text: str,
    terms: Iterable[str],
) -> bool:
    """
    Case-insensitive term detection.
    """

    normalized = text.lower()

    return any(
        term.lower() in normalized
        for term in terms
    )


def extract_entities(
    question: str,
) -> List[str]:
    """
    Extract potentially meaningful technical entities.

    This is deliberately heuristic and does not pretend to be
    a full NER model.
    """

    entities: List[str] = []

    # Acronyms.
    entities.extend(
        re.findall(
            r"\b[A-Z]{2,}[A-Za-z0-9]*\b",
            question,
        )
    )

    # Version/model identifiers.
    entities.extend(
        re.findall(
            r"\b[a-zA-Z]+\d+(?:\.\d+)*\b",
            question,
        )
    )

    # Quoted terms.
    entities.extend(
        re.findall(
            r'"([^"]+)"',
            question,
        )
    )

    entities.extend(
        re.findall(
            r"'([^']+)'",
            question,
        )
    )

    return unique_preserve_order(
        entities
    )


def extract_important_terms(
    question: str,
) -> List[str]:
    """
    Extract terms useful for lexical retrieval.
    """

    tokens = tokenize_text(
        question
    )

    stopwords = {
        "what",
        "why",
        "how",
        "when",
        "where",
        "which",
        "who",
        "is",
        "are",
        "was",
        "were",
        "the",
        "a",
        "an",
        "of",
        "to",
        "in",
        "on",
        "for",
        "and",
        "or",
        "with",
        "from",
        "this",
        "that",
        "it",
        "its",
        "they",
        "them",
        "does",
        "do",
        "can",
        "could",
        "would",
        "should",
        "please",
        "tell",
        "me",
    }

    important = [
        token
        for token in tokens
        if token not in stopwords
        and len(token) >= 2
    ]

    # Preserve frequency because repeated technical terms
    # can be meaningful.
    counts = Counter(
        important
    )

    ranked = sorted(
        counts.keys(),
        key=lambda token: (
            counts[token],
            len(token),
        ),
        reverse=True,
    )

    return ranked[:20]


def is_followup_query(
    question: str,
    history: Optional[Sequence[Dict[str, Any]]] = None,
) -> bool:
    """
    Determine whether the question likely depends on previous
    conversational context.
    """

    question_lower = question.lower()

    tokens = tokenize_text(
        question_lower
    )

    if any(
        token in REFERENCE_WORDS
        for token in tokens
    ):
        return True

    reference_patterns = [
        r"\bthe algorithm\b",
        r"\bthe method\b",
        r"\bthe model\b",
        r"\bthe technique\b",
        r"\bthe system\b",
        r"\bthe above\b",
        r"\bmentioned above\b",
        r"\bas mentioned\b",
        r"\bthis approach\b",
        r"\bthat approach\b",
        r"\bprevious answer\b",
        r"\bprevious one\b",
    ]

    if any(
        re.search(
            pattern,
            question_lower,
        )
        for pattern in reference_patterns
    ):
        return True

    if history:
        user_messages = [
            item
            for item in history
            if item.get("role") == "user"
        ]

        if user_messages:
            if len(tokens) <= 6:
                return True

    return False


def analyze_query(
    question: str,
    history: Optional[
        Sequence[Dict[str, Any]]
    ] = None,
) -> QueryAnalysis:
    """
    Produce a structured query representation.
    """

    original = normalize_query(
        question
    )

    normalized = normalize_for_search(
        original
    )

    is_visual = contains_any_term(
        original,
        VISUAL_TERMS,
    )

    is_comparison = contains_any_term(
        original,
        COMPARISON_TERMS,
    )

    is_summary = contains_any_term(
        original,
        SUMMARY_TERMS,
    )

    is_definition = contains_any_term(
        original,
        DEFINITION_TERMS,
    )

    is_list = contains_any_term(
        original,
        LIST_TERMS,
    )

    followup = is_followup_query(
        original,
        history,
    )

    if is_comparison:
        query_type = "comparison"
    elif is_summary:
        query_type = "summary"
    elif is_visual:
        query_type = "visual"
    elif is_definition:
        query_type = "definition"
    elif is_list:
        query_type = "list"
    elif followup:
        query_type = "followup"
    else:
        query_type = "general"

    entities = extract_entities(
        original
    )

    important_terms = extract_important_terms(
        original
    )

    confidence = 0.5

    if entities:
        confidence += 0.1

    if important_terms:
        confidence += 0.1

    if query_type != "general":
        confidence += 0.1

    confidence = min(
        confidence,
        1.0,
    )

    analysis = QueryAnalysis(
        original_query=original,
        normalized_query=normalized,
        query_type=query_type,
        is_followup=followup,
        is_visual=is_visual,
        is_comparison=is_comparison,
        is_summary=is_summary,
        is_definition=is_definition,
        is_list_request=is_list,
        entities=entities,
        important_terms=important_terms,
        confidence=confidence,
    )

    analysis.query_variants = build_query_variants(
        analysis
    )

    return analysis


# ============================================================
# CONVERSATION HANDLING
# ============================================================


def get_recent_user_questions(
    history: Sequence[Dict[str, Any]],
    limit: int = 3,
) -> List[str]:
    """
    Return recent user questions, newest first.
    """

    questions: List[str] = []

    for message in reversed(
        history
    ):
        if message.get("role") != "user":
            continue

        content = message.get(
            "content",
            "",
        )

        if not content:
            continue

        questions.append(
            content
        )

        if len(questions) >= limit:
            break

    return questions


def get_conversation_context(
    history: Sequence[Dict[str, Any]],
    max_messages: int = MAX_HISTORY_MESSAGES,
) -> str:
    """
    Format recent conversation for query resolution.

    This should be used for reference resolution only.
    """

    if not history:
        return ""

    messages = history[
        -max_messages:
    ]

    lines: List[str] = []

    for message in messages:
        role = message.get(
            "role",
            "",
        )

        content = message.get(
            "content",
            "",
        )

        if not content:
            continue

        if role == "user":
            lines.append(
                f"User: {content}"
            )
        elif role == "assistant":
            lines.append(
                f"Assistant: {content}"
            )

    return "\n".join(
        lines
    )


def resolve_followup_query(
    question: str,
    history: Optional[
        Sequence[Dict[str, Any]]
    ] = None,
) -> str:
    """
    Build a retrieval query that resolves obvious follow-up
    references using recent conversation.

    Important:
    This does not invent facts.
    """

    question = normalize_query(
        question
    )

    if not history:
        return question

    analysis = analyze_query(
        question,
        history,
    )

    if not analysis.is_followup:
        return question

    recent_questions = get_recent_user_questions(
        history,
        limit=3,
    )

    previous_question = ""

    if recent_questions:
        if recent_questions[0].strip().lower() != question.lower():
            previous_question = recent_questions[0]
        elif len(recent_questions) >= 2:
            previous_question = recent_questions[1]

    previous_answer = ""

    for message in reversed(
        history
    ):
        if message.get("role") == "assistant":
            previous_answer = message.get(
                "content",
                "",
            )
            break

    pieces = [
        question
    ]

    if previous_question:
        pieces.append(
            previous_question
        )

    if previous_answer:
        pieces.append(
            previous_answer[:1200]
        )

    return " ".join(
        piece
        for piece in pieces
        if piece
    )


# ============================================================
# QUERY VARIANTS
# ============================================================


def build_query_variants(
    analysis: QueryAnalysis,
) -> List[str]:
    """
    Generate retrieval-oriented variants.

    These variants are intentionally conservative.
    """

    query = analysis.original_query

    variants: List[str] = [
        query
    ]

    important = analysis.important_terms

    if important:
        variants.append(
            " ".join(
                important
            )
        )

    if analysis.is_definition:
        variants.append(
            f"definition of {query}"
        )

    if analysis.is_comparison:
        variants.append(
            f"comparison differences {query}"
        )

    if analysis.is_summary:
        variants.append(
            f"main points overview {query}"
        )

    if analysis.is_visual:
        variants.append(
            f"diagram figure table visual {query}"
        )

    # Entity-focused query.
    if analysis.entities:
        variants.append(
            " ".join(
                analysis.entities
            )
        )

    return unique_preserve_order(
        variants
    )[
        :MAX_QUERY_VARIANTS
    ]


# ============================================================
# EMBEDDING UTILITIES
# ============================================================


def normalize_embeddings(
    embeddings: np.ndarray,
) -> np.ndarray:
    """
    L2-normalize embeddings for cosine-style search.
    """

    embeddings = np.asarray(
        embeddings,
        dtype="float32",
    )

    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(
            1,
            -1,
        )

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True,
    )

    norms = np.maximum(
        norms,
        1e-12,
    )

    return (
        embeddings
        / norms
    ).astype(
        "float32"
    )


def encode_texts(
    model: SentenceTransformer,
    texts: Sequence[str],
    batch_size: int = 32,
) -> np.ndarray:
    """
    Encode text and normalize embeddings.
    """

    if not texts:
        return np.empty(
            (0, 0),
            dtype="float32",
        )

    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return normalize_embeddings(
        embeddings
    )


# ============================================================
# INDEX CONSTRUCTION
# ============================================================


def build_faiss_index(
    embeddings: np.ndarray,
) -> Optional[faiss.Index]:
    """
    Build a cosine-similarity-compatible FAISS index.
    """

    if embeddings is None:
        return None

    embeddings = np.asarray(
        embeddings,
        dtype="float32",
    )

    if embeddings.ndim != 2:
        return None

    if len(embeddings) == 0:
        return None

    embeddings = normalize_embeddings(
        embeddings
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    return index


def build_bm25_index(
    chunks: Sequence[Dict[str, Any]],
) -> Optional[BM25Okapi]:
    """
    Build BM25 index.
    """

    if not chunks:
        return None

    tokenized = [
        tokenize_text(
            chunk.get(
                "text",
                "",
            )
        )
        for chunk in chunks
    ]

    if not tokenized:
        return None

    if not any(tokenized):
        return None

    return BM25Okapi(
        tokenized
    )


# ============================================================
# EXACT SEARCH
# ============================================================


def exact_search(
    question: str,
    chunks: Sequence[Dict[str, Any]],
    top_k: int = EXACT_TOP_K,
) -> List[Dict[str, Any]]:
    """
    Exact lexical retrieval.

    This exists specifically to rescue technical queries
    where semantic similarity may be weaker than exact
    terminology matching.
    """

    if not question or not chunks:
        return []

    query_tokens = tokenize_text(
        question
    )

    if not query_tokens:
        return []

    query_set = set(
        query_tokens
    )

    normalized_query = normalize_for_search(
        question
    )

    results: List[
        Dict[str, Any]
    ] = []

    for index, chunk in enumerate(
        chunks
    ):
        text = chunk.get(
            "text",
            "",
        )

        if not text:
            continue

        normalized_text = normalize_for_search(
            text
        )

        tokens = tokenize_text(
            text
        )

        if not tokens:
            continue

        token_set = set(
            tokens
        )

        overlap = len(
            query_set
            & token_set
        )

        if overlap == 0:
            continue

        coverage = (
            overlap
            / max(
                len(query_set),
                1,
            )
        )

        phrase_bonus = 0.0

        if normalized_query in normalized_text:
            phrase_bonus = 1.0

        score = (
            coverage
            + phrase_bonus
        )

        results.append(
            {
                "index": index,
                "score": float(score),
            }
        )

    results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    results = results[
        :top_k
    ]

    for rank, result in enumerate(
        results,
        start=1,
    ):
        result["rank"] = rank

    return results


# ============================================================
# VECTOR SEARCH
# ============================================================


def vector_search(
    question: str,
    chunks: Sequence[Dict[str, Any]],
    embedding_model: SentenceTransformer,
    faiss_index: Optional[faiss.Index],
    top_k: int = VECTOR_TOP_K,
) -> List[Dict[str, Any]]:
    """
    Dense semantic retrieval.

    Uses normalized embeddings + inner product.
    """

    if (
        not question
        or not chunks
        or faiss_index is None
    ):
        return []

    query_embedding = encode_texts(
        embedding_model,
        [question],
        batch_size=1,
    )

    if query_embedding.size == 0:
        return []

    actual_k = min(
        top_k,
        len(chunks),
    )

    scores, indices = faiss_index.search(
        query_embedding,
        actual_k,
    )

    results: List[
        Dict[str, Any]
    ] = []

    for rank, (
        similarity,
        index_number,
    ) in enumerate(
        zip(
            scores[0],
            indices[0],
        ),
        start=1,
    ):
        if index_number < 0:
            continue

        similarity = float(
            similarity
        )

        # Convert cosine similarity to an intuitive distance.
        distance = 1.0 - similarity

        results.append(
            {
                "index": int(
                    index_number
                ),
                "similarity": similarity,
                "distance": distance,
                "rank": rank,
            }
        )

    return results


# ============================================================
# BM25 SEARCH
# ============================================================


def bm25_search(
    question: str,
    chunks: Sequence[Dict[str, Any]],
    bm25_index: Optional[BM25Okapi],
    top_k: int = BM25_TOP_K,
) -> List[Dict[str, Any]]:
    """
    BM25 keyword retrieval.
    """

    if (
        not question
        or not chunks
        or bm25_index is None
    ):
        return []

    tokens = tokenize_text(
        question
    )

    if not tokens:
        return []

    scores = bm25_index.get_scores(
        tokens
    )

    if len(scores) == 0:
        return []

    actual_k = min(
        top_k,
        len(scores),
    )

    top_indices = np.argsort(
        scores
    )[::-1][
        :actual_k
    ]

    results: List[
        Dict[str, Any]
    ] = []

    for rank, index_number in enumerate(
        top_indices,
        start=1,
    ):
        score = float(
            scores[index_number]
        )

        if score <= 0:
            continue

        results.append(
            {
                "index": int(
                    index_number
                ),
                "score": score,
                "rank": rank,
            }
        )

    return results


# ============================================================
# MULTI-QUERY RETRIEVAL
# ============================================================


def retrieve_multiple_queries(
    analysis: QueryAnalysis,
    chunks: Sequence[Dict[str, Any]],
    embedding_model: SentenceTransformer,
    faiss_index: Optional[faiss.Index],
    bm25_index: Optional[BM25Okapi],
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    """
    Run retrieval for multiple query formulations.

    Results are deduplicated before fusion.
    """

    vector_map: Dict[
        int,
        Dict[str, Any],
    ] = {}

    bm25_map: Dict[
        int,
        Dict[str, Any],
    ] = {}

    exact_map: Dict[
        int,
        Dict[str, Any],
    ] = {}

    variants = (
        analysis.query_variants
        or [
            analysis.original_query
        ]
    )

    for variant in variants:
        vector_results = vector_search(
            variant,
            chunks,
            embedding_model,
            faiss_index,
            VECTOR_TOP_K,
        )

        for result in vector_results:
            index = result["index"]

            existing = vector_map.get(
                index
            )

            if (
                existing is None
                or result["rank"]
                < existing["rank"]
            ):
                vector_map[index] = result

        bm25_results = bm25_search(
            variant,
            chunks,
            bm25_index,
            BM25_TOP_K,
        )

        for result in bm25_results:
            index = result["index"]

            existing = bm25_map.get(
                index
            )

            if (
                existing is None
                or result["rank"]
                < existing["rank"]
            ):
                bm25_map[index] = result

        exact_results = exact_search(
            variant,
            chunks,
            EXACT_TOP_K,
        )

        for result in exact_results:
            index = result["index"]

            existing = exact_map.get(
                index
            )

            if (
                existing is None
                or result["rank"]
                < existing["rank"]
            ):
                exact_map[index] = result

    vector_results = sorted(
        vector_map.values(),
        key=lambda item: item["rank"],
    )

    bm25_results = sorted(
        bm25_map.values(),
        key=lambda item: item["rank"],
    )

    exact_results = sorted(
        exact_map.values(),
        key=lambda item: item["rank"],
    )

    return (
        vector_results,
        bm25_results,
        exact_results,
    )


# ============================================================
# RECIPROCAL RANK FUSION
# ============================================================


def reciprocal_rank_fusion(
    vector_results: Sequence[Dict[str, Any]],
    bm25_results: Sequence[Dict[str, Any]],
    exact_results: Sequence[Dict[str, Any]],
    chunks: Sequence[Dict[str, Any]],
    rrf_k: int = RRF_K,
) -> List[RetrievalResult]:
    """
    Fuse dense, BM25 and exact retrieval.

    Exact matching receives a slightly stronger weight because
    technical identifiers are often extremely informative.
    """

    scores: Dict[int, float] = defaultdict(
        float
    )

    metadata: Dict[
        int,
        Dict[str, Any],
    ] = defaultdict(dict)

    # Dense retrieval.
    for result in vector_results:
        index = result["index"]

        rank = result["rank"]

        scores[index] += (
            1.0
            / (
                rrf_k
                + rank
            )
        )

        metadata[index][
            "vector"
        ] = result

    # BM25.
    for result in bm25_results:
        index = result["index"]

        rank = result["rank"]

        scores[index] += (
            1.0
            / (
                rrf_k
                + rank
            )
        )

        metadata[index][
            "bm25"
        ] = result

    # Exact retrieval gets 1.25x contribution.
    for result in exact_results:
        index = result["index"]

        rank = result["rank"]

        scores[index] += (
            1.25
            / (
                rrf_k
                + rank
            )
        )

        metadata[index][
            "exact"
        ] = result

    ranked_indices = sorted(
        scores.keys(),
        key=lambda index: scores[index],
        reverse=True,
    )

    results: List[
        RetrievalResult
    ] = []

    for index in ranked_indices:
        chunk = dict(
            chunks[index]
        )

        vector = metadata[
            index
        ].get(
            "vector"
        )

        bm25 = metadata[
            index
        ].get(
            "bm25"
        )

        exact = metadata[
            index
        ].get(
            "exact"
        )

        result = RetrievalResult(
            index=index,
            chunk=chunk,
            rrf_score=scores[index],
            vector_rank=(
                vector["rank"]
                if vector
                else None
            ),
            vector_distance=(
                vector["distance"]
                if vector
                else None
            ),
            bm25_rank=(
                bm25["rank"]
                if bm25
                else None
            ),
            bm25_score=(
                bm25["score"]
                if bm25
                else None
            ),
            exact_rank=(
                exact["rank"]
                if exact
                else None
            ),
            exact_score=(
                exact["score"]
                if exact
                else None
            ),
        )

        if vector:
            result.retrieval_reasons.append(
                "semantic"
            )

        if bm25:
            result.retrieval_reasons.append(
                "keyword"
            )

        if exact:
            result.retrieval_reasons.append(
                "exact"
            )

        results.append(
            result
        )

    return results


# ============================================================
# SCORE NORMALIZATION
# ============================================================


def min_max_normalize(
    values: Sequence[float],
) -> List[float]:
    """
    Normalize scores to [0, 1].
    """

    if not values:
        return []

    minimum = min(
        values
    )

    maximum = max(
        values
    )

    if math.isclose(
        minimum,
        maximum,
    ):
        return [
            1.0
            for _ in values
        ]

    return [
        (
            value - minimum
        )
        / (
            maximum - minimum
        )
        for value in values
    ]


# ============================================================
# CROSS ENCODER RERANKING
# ============================================================


def rerank_chunks(
    question: str,
    candidates: Sequence[RetrievalResult],
    reranker: CrossEncoder,
) -> List[RetrievalResult]:
    """
    Cross-encoder reranking.

    Unlike embeddings, the cross encoder sees the question and
    candidate passage together.
    """

    if not candidates:
        return []

    pairs = [
        (
            question,
            result.chunk.get(
                "text",
                "",
            ),
        )
        for result in candidates
    ]

    scores = reranker.predict(
        pairs,
        show_progress_bar=False,
    )

    reranked: List[
        RetrievalResult
    ] = []

    for result, score in zip(
        candidates,
        scores,
    ):
        result.rerank_score = float(
            score
        )

        reranked.append(
            result
        )

    reranked.sort(
        key=lambda item: (
            item.rerank_score
            if item.rerank_score is not None
            else -999.0
        ),
        reverse=True,
    )

    return reranked


# ============================================================
# LEXICAL EVIDENCE
# ============================================================


def calculate_lexical_overlap(
    question: str,
    text: str,
) -> float:
    """
    Calculate query-to-chunk token overlap.
    """

    question_tokens = set(
        tokenize_text(
            question
        )
    )

    text_tokens = set(
        tokenize_text(
            text
        )
    )

    if not question_tokens:
        return 0.0

    overlap = len(
        question_tokens
        & text_tokens
    )

    return overlap / len(
        question_tokens
    )


# ============================================================
# SEMANTIC SCORE
# ============================================================


def calculate_semantic_score(
    result: RetrievalResult,
) -> float:
    """
    Convert FAISS distance into an approximate [0, 1] score.
    """

    if result.vector_distance is None:
        return 0.0

    distance = max(
        0.0,
        result.vector_distance,
    )

    return max(
        0.0,
        min(
            1.0,
            1.0 - distance,
        ),
    )


# ============================================================
# EVIDENCE SCORING
# ============================================================


def calculate_evidence_score(
    question: str,
    result: RetrievalResult,
) -> float:
    """
    Combine independent evidence signals.

    This is NOT an LLM confidence probability.
    """

    semantic = calculate_semantic_score(
        result
    )

    lexical = calculate_lexical_overlap(
        question,
        result.chunk.get(
            "text",
            "",
        ),
    )

    exact = (
        min(
            result.exact_score / 2.0,
            1.0,
        )
        if result.exact_score is not None
        else 0.0
    )

    rerank = 0.0

    if result.rerank_score is not None:
        # Smooth unbounded cross-encoder score into [0,1].
        rerank = (
            1.0
            / (
                1.0
                + math.exp(
                    -result.rerank_score
                )
            )
        )

    rrf_component = min(
        result.rrf_score * 100.0,
        1.0,
    )

    score = (
        0.30 * semantic
        + 0.20 * lexical
        + 0.15 * exact
        + 0.25 * rerank
        + 0.10 * rrf_component
    )

    result.semantic_score = semantic

    result.lexical_score = lexical

    result.evidence_score = max(
        0.0,
        min(
            1.0,
            score,
        ),
    )

    return result.evidence_score


# ============================================================
# DIVERSITY / MMR
# ============================================================


def text_similarity(
    text_a: str,
    text_b: str,
) -> float:
    """
    Lightweight token-set similarity for MMR.

    This avoids another embedding call.
    """

    a = set(
        tokenize_text(
            text_a
        )
    )

    b = set(
        tokenize_text(
            text_b
        )
    )

    if not a or not b:
        return 0.0

    intersection = len(
        a & b
    )

    union = len(
        a | b
    )

    if union == 0:
        return 0.0

    return intersection / union


def mmr_select(
    candidates: Sequence[RetrievalResult],
    top_k: int = FINAL_TOP_K,
    lambda_value: float = MMR_LAMBDA,
) -> List[RetrievalResult]:
    """
    Maximal Marginal Relevance.

    Prevents the final context from containing five nearly
    identical chunks.
    """

    if not candidates:
        return []

    remaining = list(
        candidates
    )

    selected: List[
        RetrievalResult
    ] = []

    while (
        remaining
        and len(selected) < top_k
    ):
        best_candidate = None

        best_score = -float(
            "inf"
        )

        for candidate in remaining:
            relevance = (
                candidate.evidence_score
            )

            redundancy = 0.0

            if selected:
                redundancy = max(
                    text_similarity(
                        candidate.chunk.get(
                            "text",
                            "",
                        ),
                        selected_item.chunk.get(
                            "text",
                            "",
                        ),
                    )
                    for selected_item
                    in selected
                )

            mmr_score = (
                lambda_value
                * relevance
                - (
                    1.0
                    - lambda_value
                )
                * redundancy
            )

            if mmr_score > best_score:
                best_score = mmr_score

                best_candidate = candidate

        if best_candidate is None:
            break

        best_candidate.diversity_score = (
            best_score
        )

        selected.append(
            best_candidate
        )

        remaining.remove(
            best_candidate
        )

    return selected


# ============================================================
# ADJACENT CONTEXT
# ============================================================


def build_chunk_lookup(
    chunks: Sequence[Dict[str, Any]],
) -> Dict[int, Dict[str, Any]]:
    """
    Map chunk IDs to chunks.
    """

    return {
        int(
            chunk["chunk_id"]
        ): chunk
        for chunk in chunks
        if "chunk_id" in chunk
    }


def get_adjacent_chunks(
    chunk: Dict[str, Any],
    chunk_lookup: Dict[int, Dict[str, Any]],
    max_adjacent: int = MAX_ADJACENT_CHUNKS,
) -> List[Dict[str, Any]]:
    """
    Retrieve neighboring chunks from the same page.
    """

    results: List[
        Dict[str, Any]
    ] = []

    current = chunk

    for _ in range(
        max_adjacent
    ):
        previous_id = current.get(
            "previous_chunk_id"
        )

        if previous_id is None:
            break

        previous = chunk_lookup.get(
            int(previous_id)
        )

        if previous is None:
            break

        results.insert(
            0,
            previous,
        )

        current = previous

    current = chunk

    for _ in range(
        max_adjacent
    ):
        next_id = current.get(
            "next_chunk_id"
        )

        if next_id is None:
            break

        following = chunk_lookup.get(
            int(next_id)
        )

        if following is None:
            break

        results.append(
            following
        )

        current = following

    return results


def build_parent_context(
    result: RetrievalResult,
    chunks: Sequence[Dict[str, Any]],
    parents: Optional[
        Sequence[Dict[str, Any]]
    ] = None,
) -> str:
    """
    Add parent context to a retrieved child.
    """

    parent_id = result.chunk.get(
        "parent_id"
    )

    if (
        parent_id is None
        or not parents
    ):
        return ""

    for parent in parents:
        if int(
            parent.get(
                "parent_id",
                -1,
            )
        ) == int(parent_id):
            return parent.get(
                "text",
                "",
            )

    return ""


# ============================================================
# SOURCE DIVERSITY
# ============================================================


def apply_source_bonus(
    candidates: Sequence[RetrievalResult],
) -> None:
    """
    Slightly reward results that contribute a new page.

    This is intentionally small so it cannot override strong
    relevance.
    """

    seen_pages: Set[
        Tuple[str, int]
    ] = set()

    for candidate in candidates:
        key = (
            candidate.chunk.get(
                "filename",
                "",
            ),
            int(
                candidate.chunk.get(
                    "page",
                    0,
                )
            ),
        )

        if key not in seen_pages:
            candidate.source_bonus = 0.03

            seen_pages.add(
                key
            )


# ============================================================
# RELEVANCE GATING
# ============================================================


def relevance_gate(
    candidates: Sequence[RetrievalResult],
) -> Tuple[
    bool,
    str,
]:
    """
    Adaptive evidence gate.

    It avoids relying on one arbitrary score.
    """

    if not candidates:
        return (
            False,
            "No retrieval candidates.",
        )

    best = candidates[0]

    best_evidence = (
        best.evidence_score
    )

    best_vector = (
        best.vector_distance
    )

    best_rerank = (
        best.rerank_score
    )

    has_exact = (
        best.exact_score is not None
        and best.exact_score > 0
    )

    has_bm25 = (
        best.bm25_score is not None
        and best.bm25_score > 0
    )

    strong_semantic = (
        best_vector is not None
        and best_vector
        <= VECTOR_RELEVANCE_THRESHOLD
    )

    strong_rerank = (
        best_rerank is not None
        and best_rerank
        >= RERANK_MIN_SCORE
    )

    if (
        best_evidence
        >= MIN_EVIDENCE_SCORE
    ):
        return (
            True,
            "Evidence score passed.",
        )

    if (
        strong_semantic
        or strong_rerank
        or has_exact
        or has_bm25
    ):
        return (
            True,
            "Strong retrieval signal rescued low composite score.",
        )

    return (
        False,
        "No sufficiently strong semantic, lexical, exact, or reranking evidence.",
    )


# ============================================================
# MAIN RETRIEVAL PIPELINE
# ============================================================


def retrieve_chunks_advanced(
    question: str,
    chunks: Sequence[Dict[str, Any]],
    embedding_model: SentenceTransformer,
    reranker: CrossEncoder,
    faiss_index: Optional[faiss.Index],
    bm25_index: Optional[BM25Okapi],
    history: Optional[
        Sequence[Dict[str, Any]]
    ] = None,
    parents: Optional[
        Sequence[Dict[str, Any]]
    ] = None,
    diagnostics: Optional[
        RetrievalDiagnostics
    ] = None,
) -> List[RetrievalResult]:
    """
    Full production retrieval pipeline.

    Pipeline:

        query resolution
        ↓
        query analysis
        ↓
        multi-query retrieval
        ↓
        dense + BM25 + exact
        ↓
        RRF
        ↓
        cross encoder
        ↓
        evidence scoring
        ↓
        source diversity
        ↓
        MMR
        ↓
        parent context
        ↓
        final relevance gate
    """

    start_total = time.perf_counter()

    if diagnostics is None:
        diagnostics = RetrievalDiagnostics(
            query=question
        )

    if not question.strip():
        diagnostics.rejected = True

        diagnostics.rejection_reason = (
            "Empty query."
        )

        return []

    if not chunks:
        diagnostics.rejected = True

        diagnostics.rejection_reason = (
            "No chunks available."
        )

        return []

    resolved_query = resolve_followup_query(
        question,
        history,
    )

    diagnostics.expanded_query = (
        resolved_query
    )

    analysis = analyze_query(
        resolved_query,
        history,
    )

    retrieval_start = time.perf_counter()

    (
        vector_results,
        bm25_results,
        exact_results,
    ) = retrieve_multiple_queries(
        analysis,
        chunks,
        embedding_model,
        faiss_index,
        bm25_index,
    )

    diagnostics.vector_candidates = len(
        vector_results
    )

    diagnostics.bm25_candidates = len(
        bm25_results
    )

    diagnostics.exact_candidates = len(
        exact_results
    )

    if vector_results:
        diagnostics.best_vector_distance = min(
            item["distance"]
            for item in vector_results
        )

    if bm25_results:
        diagnostics.best_bm25_score = max(
            item["score"]
            for item in bm25_results
        )

    if exact_results:
        diagnostics.best_exact_score = max(
            item["score"]
            for item in exact_results
        )

    fused = reciprocal_rank_fusion(
        vector_results,
        bm25_results,
        exact_results,
        chunks,
        RRF_K,
    )

    diagnostics.fused_candidates = len(
        fused
    )

    diagnostics.retrieval_time_ms = (
        time.perf_counter()
        - retrieval_start
    ) * 1000.0

    if not fused:
        diagnostics.rejected = True

        diagnostics.rejection_reason = (
            "All retrieval methods returned no candidates."
        )

        diagnostics.total_time_ms = (
            time.perf_counter()
            - start_total
        ) * 1000.0

        return []

    # Limit expensive cross encoder work.
    candidates = fused[
        :RERANK_TOP_K
    ]

    rerank_start = time.perf_counter()

    reranked = rerank_chunks(
        resolved_query,
        candidates,
        reranker,
    )

    diagnostics.rerank_time_ms = (
        time.perf_counter()
        - rerank_start
    ) * 1000.0

    diagnostics.reranked_candidates = len(
        reranked
    )

    if reranked:
        diagnostics.best_rerank_score = (
            reranked[0].rerank_score
        )

    # Evidence scoring.
    for result in reranked:
        calculate_evidence_score(
            resolved_query,
            result,
        )

    # Give slight diversity bonus.
    apply_source_bonus(
        reranked
    )

    for result in reranked:
        result.evidence_score = min(
            1.0,
            result.evidence_score
            + result.source_bonus,
        )

    # Sort by evidence before MMR.
    reranked.sort(
        key=lambda item: (
            item.evidence_score,
            item.rrf_score,
        ),
        reverse=True,
    )

    passed, reason = relevance_gate(
        reranked
    )

    if not passed:
        diagnostics.rejected = True

        diagnostics.rejection_reason = (
            reason
        )

        diagnostics.total_time_ms = (
            time.perf_counter()
            - start_total
        ) * 1000.0

        return []

    final_results = mmr_select(
        reranked,
        top_k=FINAL_TOP_K,
        lambda_value=MMR_LAMBDA,
    )

    # Attach parent context.
    for result in final_results:
        result.parent_context = (
            build_parent_context(
                result,
                chunks,
                parents,
            )
        )

    diagnostics.final_candidates = len(
        final_results
    )

    diagnostics.total_time_ms = (
        time.perf_counter()
        - start_total
    ) * 1000.0

    return final_results


# ============================================================
# LEGACY-COMPATIBLE RETRIEVAL
# ============================================================


def retrieve_chunks(
    question: str,
    chunks: Sequence[Dict[str, Any]],
    embedding_model: SentenceTransformer,
    reranker: CrossEncoder,
    faiss_index: Optional[faiss.Index],
    bm25_index: Optional[BM25Okapi],
    history: Optional[
        Sequence[Dict[str, Any]]
    ] = None,
    parents: Optional[
        Sequence[Dict[str, Any]]
    ] = None,
) -> List[Dict[str, Any]]:
    """
    Compatibility wrapper.

    Returns dictionaries instead of RetrievalResult objects,
    making migration easier from your existing application.
    """

    results = retrieve_chunks_advanced(
        question=question,
        chunks=chunks,
        embedding_model=embedding_model,
        reranker=reranker,
        faiss_index=faiss_index,
        bm25_index=bm25_index,
        history=history,
        parents=parents,
    )

    return [
        result.to_dict()
        for result in results
    ]


# ============================================================
# CONTEXT BUILDING
# ============================================================


def build_context(
    retrieved_chunks: Sequence[Any],
    include_parent_context: bool = True,
) -> str:
    """
    Build grounded context for the generation model.

    Each evidence unit receives an explicit source identifier.
    """

    if not retrieved_chunks:
        return ""

    context_parts: List[str] = []

    for source_number, item in enumerate(
        retrieved_chunks,
        start=1,
    ):
        if isinstance(
            item,
            RetrievalResult,
        ):
            chunk = item.chunk

            parent_context = (
                item.parent_context
            )

        else:
            chunk = item.get(
                "chunk",
                item,
            )

            parent_context = item.get(
                "parent_context",
                "",
            )

        filename = chunk.get(
            "filename",
            "Unknown",
        )

        page = chunk.get(
            "page",
            "Unknown",
        )

        section = chunk.get(
            "section",
            "",
        )

        text = chunk.get(
            "text",
            "",
        )

        source = (
            f"[SOURCE {source_number}]\n"
            f"File: {filename}\n"
            f"Page: {page}\n"
        )

        if section:
            source += (
                f"Section: {section}\n"
            )

        source += (
            f"Evidence:\n{text}\n"
        )

        if (
            include_parent_context
            and parent_context
            and parent_context.strip()
            and parent_context.strip()
            != text.strip()
        ):
            source += (
                "\nBroader section context:\n"
                f"{parent_context}\n"
            )

        context_parts.append(
            source
        )

    return "\n\n".join(
        context_parts
    )


# ============================================================
# SOURCE EXTRACTION
# ============================================================


def extract_sources(
    retrieved_chunks: Sequence[Any],
) -> List[Dict[str, Any]]:
    """
    Produce unique filename/page source records.
    """

    sources: List[
        Dict[str, Any]
    ] = []

    seen: Set[
        Tuple[str, int]
    ] = set()

    for item in retrieved_chunks:
        if isinstance(
            item,
            RetrievalResult,
        ):
            chunk = item.chunk

        else:
            chunk = item.get(
                "chunk",
                item,
            )

        filename = chunk.get(
            "filename",
            "",
        )

        page = int(
            chunk.get(
                "page",
                0,
            )
        )

        key = (
            filename,
            page,
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        sources.append(
            {
                "filename": filename,
                "page": page,
            }
        )

    return sources


# ============================================================
# VISUAL PAGE SELECTION
# ============================================================


def is_visual_question(
    question: str,
) -> bool:
    """
    Determine whether a query requires visual document
    inspection.
    """

    return contains_any_term(
        question,
        VISUAL_TERMS,
    )


def get_visual_pages(
    question: str,
    retrieved_chunks: Sequence[Any],
    max_pages: int = 3,
) -> List[Dict[str, Any]]:
    """
    Select relevant document pages for multimodal inspection.

    Pages come from retrieved evidence only.
    """

    if not is_visual_question(
        question
    ):
        return []

    candidates: List[
        Dict[str, Any]
    ] = []

    seen: Set[
        Tuple[str, int]
    ] = set()

    for item in retrieved_chunks:
        if isinstance(
            item,
            RetrievalResult,
        ):
            chunk = item.chunk

            score = (
                item.evidence_score
            )

        else:
            chunk = item.get(
                "chunk",
                item,
            )

            score = float(
                item.get(
                    "evidence_score",
                    item.get(
                        "rerank_score",
                        0.0,
                    )
                    or 0.0,
                )
            )

        filename = chunk.get(
            "filename",
            "",
        )

        page = int(
            chunk.get(
                "page",
                0,
            )
        )

        key = (
            filename,
            page,
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        candidates.append(
            {
                "filename": filename,
                "page": page,
                "evidence_score": score,
                "visual_hint": chunk.get(
                    "visual_hint",
                    False,
                ),
            }
        )

    candidates.sort(
        key=lambda item: (
            bool(
                item.get(
                    "visual_hint",
                    False,
                )
            ),
            item.get(
                "evidence_score",
                0.0,
            ),
        ),
        reverse=True,
    )

    return candidates[
        :max_pages
    ]


# ============================================================
# RETRIEVAL DIAGNOSTICS
# ============================================================


def diagnostics_to_dict(
    diagnostics: RetrievalDiagnostics,
) -> Dict[str, Any]:
    """
    Convert diagnostics to JSON-friendly dictionary.
    """

    return {
        "query": diagnostics.query,
        "expanded_query": diagnostics.expanded_query,
        "vector_candidates": diagnostics.vector_candidates,
        "bm25_candidates": diagnostics.bm25_candidates,
        "exact_candidates": diagnostics.exact_candidates,
        "fused_candidates": diagnostics.fused_candidates,
        "reranked_candidates": diagnostics.reranked_candidates,
        "final_candidates": diagnostics.final_candidates,
        "best_vector_distance": diagnostics.best_vector_distance,
        "best_bm25_score": diagnostics.best_bm25_score,
        "best_exact_score": diagnostics.best_exact_score,
        "best_rerank_score": diagnostics.best_rerank_score,
        "retrieval_time_ms": diagnostics.retrieval_time_ms,
        "rerank_time_ms": diagnostics.rerank_time_ms,
        "total_time_ms": diagnostics.total_time_ms,
        "rejected": diagnostics.rejected,
        "rejection_reason": diagnostics.rejection_reason,
    }


# ============================================================
# ANSWER / EVIDENCE UTILITIES
# ============================================================


def get_best_evidence_score(
    retrieved_chunks: Sequence[Any],
) -> float:
    """
    Return best evidence score.
    """

    if not retrieved_chunks:
        return 0.0

    scores: List[float] = []

    for item in retrieved_chunks:
        if isinstance(
            item,
            RetrievalResult,
        ):
            scores.append(
                item.evidence_score
            )

        else:
            scores.append(
                float(
                    item.get(
                        "evidence_score",
                        0.0,
                    )
                )
            )

    return max(
        scores,
        default=0.0,
    )


def evidence_summary(
    retrieved_chunks: Sequence[Any],
) -> Dict[str, Any]:
    """
    Summarize retrieval quality for UI diagnostics.
    """

    if not retrieved_chunks:
        return {
            "count": 0,
            "best_score": 0.0,
            "average_score": 0.0,
            "unique_files": 0,
            "unique_pages": 0,
        }

    scores: List[
        float
    ] = []

    files: Set[
        str
    ] = set()

    pages: Set[
        Tuple[str, int]
    ] = set()

    for item in retrieved_chunks:
        if isinstance(
            item,
            RetrievalResult,
        ):
            chunk = item.chunk

            score = item.evidence_score

        else:
            chunk = item.get(
                "chunk",
                item,
            )

            score = float(
                item.get(
                    "evidence_score",
                    0.0,
                )
            )

        scores.append(
            score
        )

        filename = chunk.get(
            "filename",
            "",
        )

        page = int(
            chunk.get(
                "page",
                0,
            )
        )

        files.add(
            filename
        )

        pages.add(
            (
                filename,
                page,
            )
        )

    return {
        "count": len(
            retrieved_chunks
        ),
        "best_score": max(
            scores,
            default=0.0,
        ),
        "average_score": (
            sum(scores)
            / len(scores)
            if scores
            else 0.0
        ),
        "unique_files": len(
            files
        ),
        "unique_pages": len(
            pages
        ),
    }


# ============================================================
# INDEX VALIDATION
# ============================================================


def validate_search_state(
    chunks: Sequence[Dict[str, Any]],
    embeddings: Optional[np.ndarray],
    faiss_index: Optional[faiss.Index],
    bm25_index: Optional[BM25Okapi],
) -> Tuple[
    bool,
    List[str],
]:
    """
    Validate the consistency of the search database.
    """

    errors: List[
        str
    ] = []

    if not chunks:
        errors.append(
            "No chunks are available."
        )

    if embeddings is None:
        errors.append(
            "Embeddings are missing."
        )

    else:
        if len(embeddings) != len(chunks):
            errors.append(
                "Embedding count does not match chunk count."
            )

        if embeddings.ndim != 2:
            errors.append(
                "Embeddings must be a 2D matrix."
            )

    if faiss_index is None:
        errors.append(
            "FAISS index is missing."
        )

    else:
        if faiss_index.ntotal != len(
            chunks
        ):
            errors.append(
                "FAISS index size does not match chunk count."
            )

    if bm25_index is None:
        errors.append(
            "BM25 index is missing."
        )

    return (
        not errors,
        errors,
    )


# ============================================================
# INDEX BUILDING HELPER
# ============================================================


def build_search_database(
    chunks: Sequence[Dict[str, Any]],
    embedding_model: SentenceTransformer,
) -> Tuple[
    np.ndarray,
    Optional[faiss.Index],
    Optional[BM25Okapi],
]:
    """
    Build all search indexes from scratch.
    """

    texts = [
        chunk.get(
            "text",
            "",
        )
        for chunk in chunks
    ]

    embeddings = encode_texts(
        embedding_model,
        texts,
    )

    faiss_index = build_faiss_index(
        embeddings
    )

    bm25_index = build_bm25_index(
        chunks
    )

    return (
        embeddings,
        faiss_index,
        bm25_index,
    )


# ============================================================
# INCREMENTAL EMBEDDING HELPER
# ============================================================


def append_embeddings(
    existing_embeddings: Optional[np.ndarray],
    new_embeddings: np.ndarray,
) -> np.ndarray:
    """
    Safely append normalized embeddings.
    """

    new_embeddings = normalize_embeddings(
        new_embeddings
    )

    if (
        existing_embeddings is None
        or len(existing_embeddings) == 0
    ):
        return new_embeddings

    existing_embeddings = normalize_embeddings(
        existing_embeddings
    )

    if (
        existing_embeddings.shape[1]
        != new_embeddings.shape[1]
    ):
        raise ValueError(
            "Embedding dimensions do not match."
        )

    return np.vstack(
        [
            existing_embeddings,
            new_embeddings,
        ]
    ).astype(
        "float32"
    )


# ============================================================
# DOCUMENT STATISTICS
# ============================================================


def calculate_document_statistics(
    pages: Sequence[Dict[str, Any]],
    chunks: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Calculate useful document statistics.
    """

    total_characters = sum(
        len(
            page.get(
                "text",
                "",
            )
        )
        for page in pages
    )

    text_pages = sum(
        1
        for page in pages
        if page.get(
            "text",
            "",
        ).strip()
    )

    visual_pages = sum(
        1
        for page in pages
        if detect_visual_content(
            page.get(
                "text",
                "",
            )
        )
    )

    average_chunk_length = (
        sum(
            len(
                chunk.get(
                    "text",
                    "",
                )
            )
            for chunk in chunks
        )
        / len(chunks)
        if chunks
        else 0.0
    )

    return {
        "pages": len(
            pages
        ),
        "text_pages": text_pages,
        "visual_hint_pages": visual_pages,
        "characters": total_characters,
        "chunks": len(
            chunks
        ),
        "average_chunk_length": average_chunk_length,
    }


# ============================================================
# DUPLICATE DETECTION
# ============================================================


def calculate_document_hash(
    file_bytes: bytes,
) -> str:
    """
    Calculate SHA-256 for uploaded document bytes.
    """

    return hashlib.sha256(
        file_bytes
    ).hexdigest()


# ============================================================
# FINAL EXPORT HELPERS
# ============================================================


def prepare_retrieval_for_generation(
    question: str,
    retrieved_chunks: Sequence[Any],
) -> Dict[str, Any]:
    """
    Prepare retrieval output for the generation layer.
    """

    context = build_context(
        retrieved_chunks,
        include_parent_context=True,
    )

    sources = extract_sources(
        retrieved_chunks
    )

    evidence = evidence_summary(
        retrieved_chunks
    )

    visual_pages = get_visual_pages(
        question,
        retrieved_chunks,
    )

    return {
        "question": question,
        "context": context,
        "sources": sources,
        "visual_pages": visual_pages,
        "evidence": evidence,
        "has_evidence": bool(
            retrieved_chunks
        ),
    }


# ============================================================
# PUBLIC API
# ============================================================

__all__ = [
    # Configuration.
    "EMBEDDING_MODEL_NAME",
    "RERANKER_MODEL_NAME",
    "CHILD_CHUNK_SIZE",
    "CHILD_CHUNK_OVERLAP",
    "PARENT_CHUNK_SIZE",
    "PARENT_CHUNK_OVERLAP",
    "VECTOR_TOP_K",
    "BM25_TOP_K",
    "EXACT_TOP_K",
    "RRF_K",
    "RERANK_TOP_K",
    "FINAL_TOP_K",
    "VECTOR_RELEVANCE_THRESHOLD",
    "RERANK_MIN_SCORE",
    "MMR_LAMBDA",
    "NO_ANSWER",

    # Data structures.
    "PageRecord",
    "ChunkRecord",
    "ParentRecord",
    "RetrievalResult",
    "QueryAnalysis",
    "RetrievalDiagnostics",

    # Text.
    "normalize_text",
    "clean_pdf_text",
    "normalize_for_search",
    "tokenize_text",
    "hash_text",

    # Structure.
    "looks_like_heading",
    "split_into_sections",
    "build_page_records",
    "detect_visual_content",

    # Chunking.
    "create_parent_child_chunks",
    "create_chunks",

    # Query.
    "normalize_query",
    "extract_entities",
    "extract_important_terms",
    "is_followup_query",
    "analyze_query",
    "get_recent_user_questions",
    "get_conversation_context",
    "resolve_followup_query",
    "build_query_variants",

    # Embeddings.
    "normalize_embeddings",
    "encode_texts",

    # Indexing.
    "build_faiss_index",
    "build_bm25_index",
    "build_search_database",
    "append_embeddings",

    # Retrieval.
    "exact_search",
    "vector_search",
    "bm25_search",
    "retrieve_multiple_queries",
    "reciprocal_rank_fusion",
    "rerank_chunks",
    "retrieve_chunks_advanced",
    "retrieve_chunks",

    # Ranking.
    "calculate_lexical_overlap",
    "calculate_semantic_score",
    "calculate_evidence_score",
    "mmr_select",
    "relevance_gate",

    # Context.
    "build_chunk_lookup",
    "get_adjacent_chunks",
    "build_parent_context",
    "build_context",

    # Sources / visual.
    "extract_sources",
    "is_visual_question",
    "get_visual_pages",

    # Diagnostics.
    "diagnostics_to_dict",
    "get_best_evidence_score",
    "evidence_summary",
    "validate_search_state",

    # Documents.
    "calculate_document_statistics",
    "calculate_document_hash",

    # Generation preparation.
    "prepare_retrieval_for_generation",
]