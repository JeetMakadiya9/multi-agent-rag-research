"""Content extraction from HTML, PDF and transcripts, using only the standard library, pypdf and (for guessing an
undeclared text encoding) charset_normalizer, which ``requests`` already depends on.

HTML: text without navigation and boilerplate, preferring the page's ``<main>``/``<article>`` content when it is
substantial; title, date, authors, canonical address and language from the markup; pages that look like a login,
paywall or bot check are reported (``gate``) instead of being treated as content. PDF: text per page (page numbers
are kept for citations), with the number of empty pages and whether the file looks scanned (no text layer; there is no
OCR). Transcripts: WebVTT, SubRip and Podcasting 2.0 JSON, kept with their timestamps and speakers.
"""
from __future__ import annotations

import codecs
import html as html_lib
import io
import json
import logging
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Dict, List, Optional, Tuple

SKIP_TAGS = {"script", "style", "noscript", "nav", "header", "footer", "aside", "form", "svg", "iframe", "button",
             "select", "template", "menu"}
BLOCK_TAGS = {"p", "div", "section", "article", "main", "li", "ul", "ol", "br", "tr", "table", "h1", "h2", "h3", "h4",
              "h5", "h6", "blockquote", "pre", "figcaption", "dd", "dt", "td", "th"}
DATE_META = ("citation_publication_date", "citation_date", "citation_online_date", "article:published_time",
             "dc.date", "dc.date.issued", "dcterms.issued", "date", "pubdate", "publication_date")
VOID_TAGS = {"br", "img", "hr", "meta", "link", "input", "source", "wbr", "area", "base", "col", "embed", "param",
             "track"}


AUTHOR_META = ("citation_author", "author", "article:author", "dc.creator", "dcterms.creator", "parsely-author",
               "sailthru.author", "byl")
MAIN_TAGS = {"main", "article"}
MIN_MAIN_CHARS = 400
# Wording of pages that stand in front of the content (logins, paywalls, bot checks, errors). Matched only on short
# pages, so an article that merely mentions "sign in" is not affected.
GATE_PATTERNS = [
    ("bot check", re.compile(r"\b(just a moment|checking your browser|verify you are (a )?human|are you a robot|"
                             r"captcha|access denied|request blocked|unusual traffic)\b", re.I)),
    ("login required", re.compile(r"\b(sign in to (continue|read|view)|log in to (continue|read|view)|please (sign|log) in|"
                                  r"you must be logged in|create an account to (continue|read))\b", re.I)),
    ("paywall", re.compile(r"\b(subscribe to (continue|read)|subscribers only|subscription required|"
                           r"purchase (this )?article|already a subscriber|to continue reading)\b", re.I)),
    ("error page", re.compile(r"\b(page not found|404 not found|this page (does not|doesn't) exist|server error|"
                              r"service unavailable)\b", re.I)),
]
GATE_MAX_CHARS = 1500


@dataclass
class Extracted:
    text: str
    title: str = ""
    published: Optional[str] = None
    pdf_url: Optional[str] = None
    pages: Optional[List[str]] = None       # PDF text per page, when the source is a PDF
    authors: List[str] = field(default_factory=list)
    canonical_url: str = ""
    language: str = ""
    gate: str = ""                          # "login required", "paywall", "bot check", "error page" or ""
    page_count: int = 0                     # PDF pages in the file (``pages`` holds at most ``max_pages`` of them)
    empty_pages: int = 0
    image_only: bool = False                # PDF pages without a text layer (scanned); no OCR is attempted
    truncated: bool = False
    used_main: bool = False                 # the <main>/<article> content was used instead of the whole page


class ExtractionError(Exception):
    """A document that could not be read at all (``code``: pdf_unreadable, transcript_unreadable)."""

    def __init__(self, code: str, message: str = ""):
        self.code = code
        super().__init__(f"{code}: {message}" if message else code)


class _TextParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []
        self.main_parts: List[str] = []
        self.main_depth = 0
        self.skip_depth = 0
        self.in_title = False
        self.title = ""
        self.meta: Dict[str, str] = {}
        self.authors: List[str] = []
        self.canonical = ""
        self.language = ""

    def _emit(self, text: str) -> None:
        self.parts.append(text)
        if self.main_depth:
            self.main_parts.append(text)

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        attributes = {k.lower(): (v or "") for k, v in attrs}
        if tag == "html" and attributes.get("lang"):
            self.language = attributes["lang"].strip()
        if tag == "meta":
            key = (attributes.get("name") or attributes.get("property") or "").lower()
            if key and attributes.get("content"):
                if key in AUTHOR_META:
                    self.authors.append(attributes["content"].strip())
                if key not in self.meta:
                    self.meta[key] = attributes["content"].strip()
            return
        if tag == "link":
            if "canonical" in attributes.get("rel", "").lower().split() and attributes.get("href") and not self.canonical:
                self.canonical = attributes["href"].strip()
            return
        if tag in VOID_TAGS:
            if tag == "br" and not self.skip_depth:
                self._emit("\n")
            return
        if tag in SKIP_TAGS:
            self.skip_depth += 1
        elif tag == "title":
            self.in_title = True
        else:
            if (tag in MAIN_TAGS or attributes.get("role") == "main") and not self.skip_depth:
                self.main_depth += 1
            if tag in BLOCK_TAGS and not self.skip_depth:
                self._emit("\n")

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in SKIP_TAGS and self.skip_depth:
            self.skip_depth -= 1
        elif tag == "title":
            self.in_title = False
        else:
            if tag in BLOCK_TAGS and not self.skip_depth:
                self._emit("\n")
            if tag in MAIN_TAGS and self.main_depth:
                self.main_depth -= 1

    def handle_data(self, data):
        if self.in_title:
            self.title += data
        elif not self.skip_depth:
            self._emit(data)


def clean_text(text: str) -> str:
    """Collapse whitespace and drop short navigation-like lines (no sentence punctuation, under 40 characters)."""
    lines = []
    for raw in text.splitlines():
        line = re.sub(r"\s+", " ", raw).strip()
        if not line:
            continue
        if len(line) < 40 and not re.search(r"[.!?:;]$", line) and len(line.split()) < 6:
            continue
        lines.append(line)
    return "\n".join(lines)


def detect_gate(title: str, text: str) -> str:
    """Why a short page looks like it stands in front of the content, or "" for an ordinary page."""
    if len(text) > GATE_MAX_CHARS:
        return ""
    sample = f"{title}\n{text}"
    return next((reason for reason, pattern in GATE_PATTERNS if pattern.search(sample)), "")


def html_to_text(html: str) -> Extracted:
    parser = _TextParser()
    try:
        parser.feed(html)
        parser.close()
    except Exception:                         # malformed markup: keep whatever was parsed
        pass
    meta = parser.meta
    title = meta.get("citation_title") or meta.get("og:title") or re.sub(r"\s+", " ", parser.title).strip()
    published = next((meta[k] for k in DATE_META if meta.get(k)), None)
    whole = clean_text("".join(parser.parts))
    main = clean_text("".join(parser.main_parts))
    use_main = len(main) >= MIN_MAIN_CHARS and len(main) >= 0.3 * len(whole)
    text = main if use_main else whole
    authors = list(dict.fromkeys(a for a in parser.authors if a and not a.startswith("http")))[:12]
    return Extracted(text=text, title=title, published=published, pdf_url=meta.get("citation_pdf_url"),
                     authors=authors, canonical_url=parser.canonical or meta.get("og:url", ""),
                     language=parser.language, gate=detect_gate(title, text), used_main=use_main)


def decode_text(content: bytes, declared: str = "", sniff_html: bool = False) -> str:
    """Bytes to text: a byte-order mark, then the declared charset, then an HTML <meta charset>, then UTF-8, then
    the best guess of charset_normalizer, then Windows-1252 (which decodes any byte)."""
    for bom, encoding in ((codecs.BOM_UTF8, "utf-8-sig"), (codecs.BOM_UTF16_LE, "utf-16"), (codecs.BOM_UTF16_BE, "utf-16")):
        if content.startswith(bom):
            return content.decode(encoding, errors="replace")
    candidates = [declared.strip().strip('"').lower()] if declared else []
    if sniff_html:
        match = re.search(rb"""<meta[^>]+charset=["']?([A-Za-z0-9_\-:.]+)""", content[:4096], re.I)
        if match:
            candidates.append(match.group(1).decode("ascii", "ignore").lower())
    for encoding in candidates:
        try:
            codecs.lookup(encoding)
            return content.decode(encoding, errors="replace")
        except LookupError:
            continue
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        pass
    try:
        from charset_normalizer import from_bytes
        best = from_bytes(content[:200_000]).best()
        if best is not None and best.encoding:
            return content.decode(best.encoding, errors="replace")
    except Exception:                          # the guesser is optional; fall through to a lossless decode
        pass
    return content.decode("cp1252", errors="replace")


def pdf_to_text(data: bytes, max_pages: int = 30) -> Extracted:
    try:
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError
    except ImportError as exc:                 # pragma: no cover - pypdf is a declared dependency
        raise ExtractionError("pdf_unreadable", f"pypdf is not available ({exc})") from None

    logging.getLogger("pypdf").setLevel(logging.ERROR)     # font-encoding notices would flood the console
    try:
        reader = PdfReader(io.BytesIO(data))
        page_count = len(reader.pages)
    except (PdfReadError, ValueError, KeyError, TypeError, OSError) as exc:
        raise ExtractionError("pdf_unreadable", f"{type(exc).__name__}") from None
    if getattr(reader, "is_encrypted", False):
        raise ExtractionError("pdf_unreadable", "the PDF is encrypted")
    pages = []
    for page in reader.pages[:max_pages]:
        try:
            pages.append(re.sub(r"[ \t]+", " ", page.extract_text() or "").strip())
        except Exception:                     # one unreadable page should not lose the others
            pages.append("")
    title = ""
    try:
        title = str((reader.metadata or {}).get("/Title") or "")
    except Exception:
        pass
    empty = sum(1 for p in pages if len(p) < 20)
    text = "\n\n".join(p for p in pages if p)
    image_only = bool(pages) and empty == len(pages)
    return Extracted(text=text, title=title, pages=pages, page_count=page_count, empty_pages=empty,
                     image_only=image_only, truncated=page_count > max_pages)


# ================================================================================================= transcripts
@dataclass
class Segment:
    start: float                            # seconds from the start of the media
    end: float
    text: str
    speaker: str = ""


_TIME = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})")


def _seconds(stamp: str) -> float:
    match = _TIME.search(stamp)
    if not match:
        raise ValueError(f"bad timestamp {stamp!r}")
    hours, minutes, seconds, fraction = match.groups()
    return int(hours or 0) * 3600 + int(minutes) * 60 + int(seconds) + int(fraction.ljust(3, "0")[:3]) / 1000


def format_timestamp(seconds: float) -> str:
    seconds = int(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def _cue_blocks(text: str) -> List[List[str]]:
    return [[line for line in block.splitlines()] for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n")) if block.strip()]


def parse_vtt(text: str) -> List[Segment]:
    """WebVTT cues; ``<v Speaker>`` voice tags become speaker labels, other markup is removed."""
    if not text.lstrip("﻿").startswith("WEBVTT"):
        raise ExtractionError("transcript_unreadable", "not a WebVTT file")
    segments = []
    for block in _cue_blocks(text):
        timing = next((i for i, line in enumerate(block) if "-->" in line), None)
        if timing is None:
            continue
        start, end = (part.strip().split(" ")[0] for part in block[timing].split("-->", 1))
        body = " ".join(block[timing + 1:]).strip()
        speaker = ""
        voice = re.match(r"<v(?:\.[^ >]+)?\s+([^>]+)>", body)
        if voice:
            speaker = voice.group(1).strip()
        body = html_lib.unescape(re.sub(r"<[^>]+>", "", body)).strip()
        if body:
            segments.append(Segment(_seconds(start), _seconds(end), body, speaker))
    if not segments:
        raise ExtractionError("transcript_unreadable", "no cues")
    return segments


def parse_srt(text: str) -> List[Segment]:
    segments = []
    for block in _cue_blocks(text):
        timing = next((i for i, line in enumerate(block) if "-->" in line), None)
        if timing is None:
            continue
        start, end = (part.strip() for part in block[timing].split("-->", 1))
        body = html_lib.unescape(re.sub(r"<[^>]+>", "", " ".join(block[timing + 1:]))).strip()
        if body:
            segments.append(Segment(_seconds(start), _seconds(end), body))
    if not segments:
        raise ExtractionError("transcript_unreadable", "no subtitles")
    return segments


def parse_transcript_json(text: str) -> List[Segment]:
    """The Podcasting 2.0 JSON transcript format: {"segments": [{"startTime", "endTime", "body", "speaker"}]}."""
    try:
        rows = json.loads(text).get("segments") or []
    except (ValueError, AttributeError):
        raise ExtractionError("transcript_unreadable", "malformed JSON transcript") from None
    segments = [Segment(float(r.get("startTime", 0)), float(r.get("endTime", r.get("startTime", 0))),
                        str(r.get("body", "")).strip(), str(r.get("speaker", "") or "").strip())
                for r in rows if isinstance(r, dict) and str(r.get("body", "")).strip()]
    if not segments:
        raise ExtractionError("transcript_unreadable", "no segments")
    return segments


def parse_transcript(text: str, kind: str = "") -> List[Segment]:
    kind = kind.lower()
    if "vtt" in kind or text.lstrip("﻿").startswith("WEBVTT"):
        return parse_vtt(text)
    if "json" in kind:
        return parse_transcript_json(text)
    if "srt" in kind or "subrip" in kind or re.match(r"\s*\d+\s*\n\s*\d", text):
        return parse_srt(text)
    raise ExtractionError("transcript_unreadable", f"unsupported transcript type {kind or 'unknown'}")


def transcript_pages(segments: List[Segment], window: float = 60.0) -> Tuple[List[str], List[str]]:
    """Group cues into windows of about ``window`` seconds. Each window is one citable "page"; its label is the time
    range it covers, so a passage from it is cited by timestamp, never by an invented page number."""
    pages: List[str] = []
    labels: List[str] = []
    current: List[str] = []
    start = segments[0].start if segments else 0.0
    end = start
    last_speaker = None
    for segment in segments:
        if current and segment.start - start >= window:
            pages.append(" ".join(current))
            labels.append(f"{format_timestamp(start)}–{format_timestamp(end)}")
            current, start, last_speaker = [], segment.start, None
        prefix = f"{segment.speaker}: " if segment.speaker and segment.speaker != last_speaker else ""
        last_speaker = segment.speaker or last_speaker
        current.append(prefix + segment.text)
        end = segment.end
    if current:
        pages.append(" ".join(current))
        labels.append(f"{format_timestamp(start)}–{format_timestamp(end)}")
    return pages, labels


def openalex_abstract(inverted_index: Optional[Dict[str, List[int]]]) -> str:
    """OpenAlex stores abstracts as {word: [positions]}; rebuild the word order."""
    if not inverted_index:
        return ""
    positions = [(p, word) for word, places in inverted_index.items() for p in places]
    return " ".join(word for _, word in sorted(positions))


def strip_tags(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", text or "")).strip()
