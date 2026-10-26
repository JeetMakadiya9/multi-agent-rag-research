"""Load local TXT and PDF documents into page records that keep the document name and page number."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

SUPPORTED_SUFFIXES = (".txt", ".pdf")


@dataclass(frozen=True)
class Page:
    document: str          # file name, used in citations
    page: int              # 1-based; a TXT file is one page unless it contains form feeds
    text: str


@dataclass
class IngestReport:
    pages: List[Page] = field(default_factory=list)
    skipped: List[Tuple[str, str]] = field(default_factory=list)      # (file name, reason)

    @property
    def documents(self) -> List[str]:
        return sorted({p.document for p in self.pages})


def _txt_pages(path: Path) -> List[Page]:
    text = path.read_text(encoding="utf-8-sig")                        # tolerate a UTF-8 byte-order mark
    parts = text.split("\f") if "\f" in text else [text]
    return [Page(path.name, number, part.strip()) for number, part in enumerate(parts, 1) if part.strip()]


def _pdf_pages(path: Path) -> List[Page]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for number, page in enumerate(reader.pages, 1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(Page(path.name, number, text))
    return pages


def load_documents(folder: Path | str) -> IngestReport:
    """Read every supported file in ``folder`` (not recursive). Unreadable, empty or unsupported files are skipped
    and reported, never silently dropped."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Document folder not found: {folder}")
    report = IngestReport()
    for path in sorted(folder.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            report.skipped.append((path.name, "unsupported file type"))
            continue
        try:
            pages = _txt_pages(path) if path.suffix.lower() == ".txt" else _pdf_pages(path)
        except Exception as exc:                                        # a corrupt file must not stop the others
            report.skipped.append((path.name, f"could not be read ({type(exc).__name__})"))
            continue
        if not pages:
            reason = "no extractable text (scanned PDF?)" if path.suffix.lower() == ".pdf" else "empty file"
            report.skipped.append((path.name, reason))
            continue
        report.pages.extend(pages)
    return report
