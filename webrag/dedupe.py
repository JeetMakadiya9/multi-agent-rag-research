"""Source identity: URL/DOI/arXiv canonical keys, and near-duplicate text detection, so repeated copies of one
report are not counted as independent corroboration."""
from __future__ import annotations

import re
import urllib.parse
from typing import Iterable, List, Optional, Set

TRACKING_PARAMS = re.compile(r"^(utm_.*|fbclid|gclid|mc_cid|mc_eid|ref|ref_src|igshid|spm)$", re.I)
ARXIV_ID = re.compile(r"(\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})(v\d+)?", re.I)


def normalize_doi(doi: Optional[str]) -> Optional[str]:
    if not doi:
        return None
    doi = urllib.parse.unquote(doi.strip()).lower()
    doi = re.sub(r"^(https?://)?(dx\.)?doi\.org/", "", doi)
    doi = re.sub(r"^doi:\s*", "", doi)
    return doi or None


def arxiv_id(text: str) -> Optional[str]:
    """arXiv identifier (without version) from an arXiv URL or an arXiv DOI (10.48550/arXiv.X)."""
    text = text or ""
    match = re.search(r"arxiv\.org/(?:abs|pdf|html)/([^?#\s]+?)(?:\.pdf)?(?:[?#]|$)", text, re.I) or \
        re.search(r"10\.48550/arxiv\.([^\s?#]+)", text, re.I)
    if not match:
        return None
    found = ARXIV_ID.match(match.group(1))
    return found.group(1).lower() if found else None


def normalize_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url.strip())
    host = (parsed.hostname or "").lower()
    host = re.sub(r"^(www\d?|m)\.", "", host)
    host = host.replace(".m.wikipedia.org", ".wikipedia.org")
    query = urllib.parse.urlencode(sorted((k, v) for k, v in urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
                                          if not TRACKING_PARAMS.match(k)))
    path = re.sub(r"/+$", "", parsed.path) or "/"
    port = f":{parsed.port}" if parsed.port and parsed.port not in (80, 443) else ""
    return urllib.parse.urlunsplit(("https", host + port, path, query, ""))


def canonical_key(url: str, doi: Optional[str] = None) -> str:
    """One key per underlying work: arXiv ID, then DOI, then the normalised URL."""
    ident = arxiv_id(url) or arxiv_id(doi or "")
    if ident:
        return f"arxiv:{ident}"
    doi = normalize_doi(doi) or (normalize_doi(url) if re.search(r"doi\.org/", url or "", re.I) else None)
    if doi:
        return f"doi:{doi}"
    return "url:" + normalize_url(url)


def shingles(text: str, size: int = 8) -> Set[int]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {hash(" ".join(words[i:i + size])) for i in range(max(0, len(words) - size + 1))}


def containment(a: Set[int], b: Set[int]) -> float:
    """Share of the smaller shingle set found in the larger one (1.0 = one text is a copy of the other)."""
    if not a or not b:
        return 0.0
    small, large = (a, b) if len(a) <= len(b) else (b, a)
    return len(small & large) / len(small)


def near_duplicate_groups(texts: Iterable[str], threshold: float = 0.6, min_shingles: int = 20) -> List[int]:
    """For each text, the index of the first earlier text it substantially copies (or its own index)."""
    sets = [shingles(t) for t in texts]
    origin = list(range(len(sets)))
    for i, current in enumerate(sets):
        if len(current) < min_shingles:
            continue
        for j in range(i):
            if origin[j] == j and len(sets[j]) >= min_shingles and containment(current, sets[j]) >= threshold:
                origin[i] = j
                break
    return origin
