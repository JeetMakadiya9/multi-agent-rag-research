"""A dependency-free writer for simple text-only PDFs (used to build the controlled test corpus and in tests).

Only printable ASCII is supported; each page is a list of lines set in Helvetica 11 pt.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence


def _escape(line: str) -> str:
    if any(ord(ch) < 32 or ord(ch) > 126 for ch in line):
        raise ValueError("write_simple_pdf supports printable ASCII text only.")
    return line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def write_simple_pdf(path: Path | str, pages: Sequence[Sequence[str]]) -> None:
    if not pages:
        raise ValueError("A PDF needs at least one page.")
    n = len(pages)
    page_ids = [4 + 2 * i for i in range(n)]
    objects: List[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        ("<< /Type /Pages /Kids [" + " ".join(f"{p} 0 R" for p in page_ids) + f"] /Count {n} >>").encode("ascii"),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for i, lines in enumerate(pages):
        body = "BT /F1 11 Tf 14 TL 50 780 Td " + " ".join(f"({_escape(line)}) Tj T*" for line in lines) + " ET"
        stream = body.encode("ascii")
        objects.append((f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 842] /Resources << /Font << /F1 3 0 R >> >>"
                        f" /Contents {page_ids[i] + 1} 0 R >>").encode("ascii"))
        objects.append(b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, obj in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode("ascii") + obj + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("ascii")
    out += b"".join(f"{offset:010d} 00000 n \n".encode("ascii") for offset in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("ascii")
    Path(path).write_bytes(bytes(out))
