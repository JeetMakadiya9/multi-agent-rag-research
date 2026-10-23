"""RSS 2.0 and Atom feed parsing with the standard library.

Used by ``search.FeedSearch`` for the feeds the user lists: news sites, official publication feeds and podcasts. For a
podcast, the Podcasting 2.0 ``<podcast:transcript>`` tag (a transcript the publisher makes available) and the audio
enclosure are kept; the audio itself is never downloaded or transcribed here. Only http(s) links are kept.
"""
from __future__ import annotations

import email.utils
import html
import re
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import List, Optional

NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "podcast": "https://podcastindex.org/namespace/1.0",
    "itunes": "http://www.itunes.com/dtds/podcast-1.0.dtd",
    "content": "http://purl.org/rss/1.0/modules/content/",
    "dc": "http://purl.org/dc/elements/1.1/",
}
# Transcript formats that ``extract.parse_transcript`` reads, best first.
TRANSCRIPT_TYPES = ("text/vtt", "application/x-subrip", "application/srt", "text/srt", "application/json")
MAX_ENTRIES = 200


@dataclass
class FeedEntry:
    title: str
    url: str
    summary: str = ""
    published: Optional[str] = None          # ISO date of publication as the feed states it
    authors: List[str] = field(default_factory=list)
    guid: str = ""
    transcript_url: str = ""
    transcript_type: str = ""
    media_url: str = ""
    media_type: str = ""
    duration: str = ""
    feed_title: str = ""


def _text(value: Optional[str]) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", value or ""))).strip()


def _http(url: Optional[str], base: str = "") -> str:
    url = urllib.parse.urljoin(base, (url or "").strip()) if url else ""
    return url if re.match(r"^https?://", url, re.I) else ""


def _rss_date(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    try:
        return email.utils.parsedate_to_datetime(value.strip()).date().isoformat()
    except (TypeError, ValueError, IndexError):
        match = re.match(r"\d{4}-\d{2}-\d{2}", value.strip())
        return match.group(0) if match else None


def _iso_date(value: Optional[str]) -> Optional[str]:
    match = re.match(r"\d{4}-\d{2}-\d{2}", (value or "").strip())
    return match.group(0) if match else None


def _best_transcript(item) -> tuple:
    found = [(t.get("type", "").lower(), t.get("url", "")) for t in item.findall("podcast:transcript", NS)]
    for wanted in TRANSCRIPT_TYPES:
        for kind, url in found:
            if kind == wanted and _http(url):
                return _http(url), kind
    return "", ""


def parse_feed(data: bytes, base_url: str = "") -> List[FeedEntry]:
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise ValueError(f"malformed feed ({exc})") from None
    tag = root.tag.split("}")[-1].lower()
    entries: List[FeedEntry] = []
    if tag in {"rss", "rdf"}:
        channel = root.find("channel")
        feed_title = _text(channel.findtext("title")) if channel is not None else ""
        items = (channel.findall("item") if channel is not None else []) or root.findall("{http://purl.org/rss/1.0/}item")
        for item in items[:MAX_ENTRIES]:
            link = _http(item.findtext("link") or item.findtext("{http://purl.org/rss/1.0/}link"), base_url)
            enclosure = item.find("enclosure")
            media_url = _http(enclosure.get("url"), base_url) if enclosure is not None else ""
            transcript_url, transcript_type = _best_transcript(item)
            title = _text(item.findtext("title") or item.findtext("{http://purl.org/rss/1.0/}title"))
            if not title or not (link or media_url or transcript_url):
                continue
            authors = [_text(a.text) for a in item.findall("dc:creator", NS) + item.findall("author") if _text(a.text)]
            entries.append(FeedEntry(
                title=title, url=link or media_url or transcript_url,
                summary=_text(item.findtext("content:encoded", None, NS) or item.findtext("description")),
                published=_rss_date(item.findtext("pubDate")) or _iso_date(item.findtext("dc:date", None, NS)),
                authors=authors, guid=_text(item.findtext("guid")), transcript_url=transcript_url,
                transcript_type=transcript_type, media_url=media_url,
                media_type=enclosure.get("type", "") if enclosure is not None else "",
                duration=_text(item.findtext("itunes:duration", None, NS)), feed_title=feed_title))
    elif tag == "feed":
        feed_title = _text(root.findtext("atom:title", None, NS))
        for item in root.findall("atom:entry", NS)[:MAX_ENTRIES]:
            links = item.findall("atom:link", NS)
            link = next((_http(l.get("href"), base_url) for l in links if l.get("rel", "alternate") == "alternate"), "")
            title = _text(item.findtext("atom:title", None, NS))
            if not title or not link:
                continue
            authors = [_text(a.findtext("atom:name", None, NS)) for a in item.findall("atom:author", NS)]
            entries.append(FeedEntry(
                title=title, url=link,
                summary=_text(item.findtext("atom:summary", None, NS) or item.findtext("atom:content", None, NS)),
                published=_iso_date(item.findtext("atom:published", None, NS) or item.findtext("atom:updated", None, NS)),
                authors=[a for a in authors if a], guid=_text(item.findtext("atom:id", None, NS)), feed_title=feed_title))
    else:
        raise ValueError("not an RSS or Atom feed")
    return entries
