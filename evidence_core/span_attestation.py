"""Deterministic quote-to-span location (isolated; no effect on any verifier, stance or verdict).

Given a quote returned by a model and the exact passage text it was checked against, find where the quote occurs
in the passage as one contiguous character span. Offsets are half-open ``[start, end)`` into the ORIGINAL passage,
so ``passage[start:end] == span.text`` always holds.

LOCATED is a statement about text position only. It does not mean that the quote, or the passage, is relevant to,
supports or contradicts any claim.

Tiers, tried in order:

* EXACT      the quote occurs character for character in the passage;
* CANONICAL  it occurs after the conservative canonicalisation below is applied to both texts; offsets are mapped
             back to the original passage.

Whitespace and invisible characters (zero-width characters, soft hyphen) at either end of the quote are ignored.

Every occurrence must sit on clean boundaries in the canonical text AND at its offsets in the original passage
(invisible characters are skipped when the neighbours are read). It may not start or end:

* inside a word (``effect`` in ``effective``, ``don`` in ``don't``, ``well`` in ``well-known``) or inside a
  character built with a combining mark (``cafe`` in ``cafe`` + U+0301);
* inside a numeric expression: next to a sign (``-5``, ``+5``, U+2212, U+00B1), a decimal point, thousands
  separator, time, ratio, range, multiplication or power joiner between digits (``1.5``, ``1,500``, ``10:30``,
  ``1990-1995``, ``1.5x10``, ``10^5``, ``1.5`` with a middle dot), a percent, per-mille, degree, temperature or
  prime sign (``5%``, ``5 %``, ``5°C``, ``5 °C``), or a currency symbol (``$5``, ``5 €``);
* next to whitespace that separates it from another digit. Spaces between digits are treated as possible group
  separators (``1 500``), so a quote ending at ``1990`` is rejected in ``in 1990 500 people``; this is deliberately
  conservative.

Exactly one location -> LOCATED; several -> AMBIGUOUS (every span is reported, none is chosen); none -> UNATTESTED;
an empty quote -> EMPTY. Reordered words and non-contiguous word overlap are never located.

Canonicalisation ``canon-v1`` changes only characters that cannot change what the text says:

* a run of whitespace (including no-break and thin spaces, tabs and line breaks) becomes one space;
* typographic single quotes and apostrophes (U+2018, U+2019, U+201A, U+201B) become ``'``; typographic double
  quotes (U+201C to U+201F) become ``"``;
* the Unicode hyphen and non-breaking hyphen (U+2010, U+2011) become ``-``;
* zero-width characters and the soft hyphen are removed;
* Latin ligatures (U+FB00 to U+FB06) are expanded into their letters.

Deliberately unchanged: letter case (folding it would merge unit prefixes such as MW and mW; a quote that matches
only when case is ignored is UNATTESTED with the diagnostic reason CASE_MISMATCH_ONLY); en and em dashes and the
minus sign; digits and digit-like characters (no NFKC, which maps ``10²`` to ``102``); decimal points, thousands
separators and percent signs; every word, including negations; and Unicode composition (no NFC or NFD, so an
accent written as one character does not match the same accent written with a combining mark).

Known limitations: units written as words (``5 km``, ``5 percent``) are ordinary word boundaries, so a quote that
ends at the number is located; scripts written without spaces between words (Chinese, Japanese) never yield a
located substring, because every neighbour is a word character.

Negation cue (a warning, never a status change): ``negation_cue`` is set when the word immediately before the
span, separated from it only by whitespace, is one of ``NEGATION_CUES`` or ends in ``NEGATION_SUFFIX``. It is a
fixed-list heuristic: multi-word cues such as ``no longer`` are missed, a cue followed by punctuation (``No,``) is
missed, and ``not only`` is reported as a cue.

``compare_with_legacy`` reports a caller-supplied legacy check (for example ``docqa.verify.quote_in_passage``) next
to the result without combining the two. This module imports no verifier code.
"""
from __future__ import annotations

import re
import unicodedata
from bisect import bisect_left
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional, Tuple

SPAN_ATTESTATION_VERSION = "span-attest-v1.1"
CANONICALIZATION_VERSION = "canon-v1"


class AttestationStatus(str, Enum):
    """Where the quote text is, never whether it supports anything."""
    LOCATED = "LOCATED"            # exactly one cleanly bounded location in the passage
    AMBIGUOUS = "AMBIGUOUS"        # more than one location; all are reported, none is chosen
    UNATTESTED = "UNATTESTED"      # no contiguous, cleanly bounded location
    EMPTY = "EMPTY"                # nothing is left of the quote after stripping whitespace and invisible characters


class MatchTier(str, Enum):
    EXACT = "EXACT"
    CANONICAL = "CANONICAL"


class AttestationReason(str, Enum):
    UNIQUE_EXACT_MATCH = "UNIQUE_EXACT_MATCH"
    UNIQUE_CANONICAL_MATCH = "UNIQUE_CANONICAL_MATCH"
    MULTIPLE_MATCHES = "MULTIPLE_MATCHES"
    EMPTY_QUOTE = "EMPTY_QUOTE"
    CASE_MISMATCH_ONLY = "CASE_MISMATCH_ONLY"     # diagnostic: a clean location exists only if case is ignored
    BOUNDARY_VIOLATION = "BOUNDARY_VIOLATION"     # the text occurs, but only inside a word or numeric expression
    NO_CONTIGUOUS_MATCH = "NO_CONTIGUOUS_MATCH"


@dataclass(frozen=True)
class CharSpan:
    start: int
    end: int
    text: str                                    # always passage[start:end]
    tier: MatchTier
    negation_cue: Optional[str] = None           # the cue word directly before the span (see NEGATION_CUES)

    @property
    def preceded_by_negation(self) -> bool:
        return self.negation_cue is not None


@dataclass(frozen=True)
class Attestation:
    """The location of a quote in a passage. It carries no stance, support or relevance information."""
    status: AttestationStatus
    tier: Optional[MatchTier]                # best tier among the spans; None when nothing was located
    spans: Tuple[CharSpan, ...]              # sorted by start; one when LOCATED, several when AMBIGUOUS
    reason: AttestationReason
    version: str = SPAN_ATTESTATION_VERSION
    canonicalization: str = CANONICALIZATION_VERSION

    @property
    def located(self) -> bool:
        return self.status is AttestationStatus.LOCATED

    @property
    def span(self) -> Optional[CharSpan]:
        """The single located span; None unless LOCATED."""
        return self.spans[0] if self.located else None


@dataclass(frozen=True)
class LegacyComparison:
    attestation: Attestation
    legacy_accepts: bool                     # the caller-supplied legacy check, reported as returned


# --------------------------------------------------------------------------- canonicalisation
_SINGLE_QUOTES = frozenset("‘’‚‛")
_DOUBLE_QUOTES = frozenset("“”„‟")
_HYPHENS = frozenset("‐‑")
_REMOVED = frozenset("​‌‍⁠﻿­")
_LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st",
              "ﬆ": "st"}


def canonicalize(text: str) -> Tuple[str, Tuple[int, ...]]:
    """Canonical text and, for each canonical character, the index of the original character it came from."""
    kept = [(c, i) for i, c in enumerate(text) if c not in _REMOVED]
    out: List[str] = []
    origin: List[int] = []
    k = 0
    while k < len(kept):
        c, i = kept[k]
        if c.isspace():
            out.append(" ")
            origin.append(i)
            while k < len(kept) and kept[k][0].isspace():
                k += 1
            continue
        if c in _SINGLE_QUOTES:
            replacement = "'"
        elif c in _DOUBLE_QUOTES:
            replacement = '"'
        elif c in _HYPHENS:
            replacement = "-"
        else:
            replacement = _LIGATURES.get(c, c)
        for r in replacement:
            out.append(r)
            origin.append(i)
        k += 1
    return "".join(out), tuple(origin)


def _strip(text: str) -> str:
    """Remove whitespace and invisible characters from both ends."""
    start, end = 0, len(text)
    while start < end and (text[start].isspace() or text[start] in _REMOVED):
        start += 1
    while end > start and (text[end - 1].isspace() or text[end - 1] in _REMOVED):
        end -= 1
    return text[start:end]


def _fold_case(text: str) -> str:
    """Lower case where it keeps the length, so offsets still line up with ``text``."""
    return "".join(c.lower() if len(c.lower()) == 1 else c for c in text)


# --------------------------------------------------------------------------- boundaries
_SIGNS = frozenset("+-−±")                                    # plus, hyphen-minus, minus, plus-minus
_NUMERIC_JOINERS = frozenset(".,:/–—−-×^·")    # decimal, separators, dashes, x, ^, dot
_UNIT_SYMBOLS = frozenset("%‰‱°℃℉′″")  # % per-mille ° ℃ ℉ prime double-prime
_WORD_JOINERS = frozenset("'’-‐‑")                        # inside don't, well-known


def _char(text: str, i: int) -> str:
    return text[i] if 0 <= i < len(text) else ""


def _visible_before(text: str, i: int) -> int:
    """Index of the nearest character before ``i`` that is not invisible; -1 if none."""
    i -= 1
    while i >= 0 and text[i] in _REMOVED:
        i -= 1
    return i


def _visible_after(text: str, i: int) -> int:
    """Index of the nearest character at or after ``i`` that is not invisible; ``len(text)`` if none."""
    while i < len(text) and text[i] in _REMOVED:
        i += 1
    return i


def _is_mark(c: str) -> bool:
    return bool(c) and unicodedata.category(c).startswith("M")


def _is_word(c: str) -> bool:
    return bool(c) and (c.isalnum() or _is_mark(c))


def _is_currency(c: str) -> bool:
    return bool(c) and unicodedata.category(c) == "Sc"


def _starts_cleanly(text: str, start: int) -> bool:
    first = text[start]
    if _is_mark(first):
        return False                                            # starts inside a character
    p = _visible_before(text, start)
    if p < 0:
        return True
    prev = text[p]
    before = _char(text, _visible_before(text, p))
    if _is_word(first):
        if _is_word(prev):
            return False                                        # inside a word or a number
        if prev in _WORD_JOINERS and _is_word(before):
            return False                                        # inside don't / well-known
        if first.isdigit():
            if prev in _SIGNS or _is_currency(prev):
                return False                                    # -5, +5, minus 5, plus-minus 5, 1990-1995, $5
            if prev in _NUMERIC_JOINERS and before.isdigit():
                return False                                    # 1.5, 1,500, 10:30, 1990-1995, 1.5x10, 10^5
            if prev.isspace():
                q = p
                while q >= 0 and (text[q].isspace() or text[q] in _REMOVED):
                    q -= 1
                left = _char(text, q)
                if left.isdigit() or _is_currency(left):
                    return False                                # 1 500 (group separator) / $ 5
    elif prev.isdigit() and (first in _NUMERIC_JOINERS or first in _SIGNS):
        return False                                            # the quote starts inside 1.5 or 1990-1995
    return True


def _ends_cleanly(text: str, end: int) -> bool:
    n = _visible_after(text, end)
    if n >= len(text):
        return True
    last, nxt = text[end - 1], text[n]
    after = _char(text, _visible_after(text, n + 1))
    if _is_mark(nxt):
        return False                                            # ends inside a character
    if _is_word(last):
        if _is_word(nxt):
            return False
        if nxt in _WORD_JOINERS and _is_word(after):
            return False
        if last.isdigit():
            if nxt in _UNIT_SYMBOLS or _is_currency(nxt):
                return False                                    # 5 of 5%, 5°C, 5€
            if (nxt in _NUMERIC_JOINERS or nxt in _SIGNS) and after.isdigit():
                return False                                    # 1 of 1.5, 1,500, 1990-1995, 1.5x10
            if nxt.isspace():
                q = n
                while q < len(text) and (text[q].isspace() or text[q] in _REMOVED):
                    q += 1
                right = _char(text, q)
                if right.isdigit() or right in _UNIT_SYMBOLS or _is_currency(right):
                    return False                                # 1 500 / 5 % / 5 °C / 5 €
    elif nxt.isdigit() and (last in _NUMERIC_JOINERS or last in _SIGNS):
        return False                                            # the quote ends inside 1.5 or 1990-1995
    return True


def _clean(text: str, start: int, end: int) -> bool:
    return _starts_cleanly(text, start) and _ends_cleanly(text, end)


def _occurrences(haystack: str, needle: str) -> List[int]:
    """Every start index of ``needle`` in ``haystack``, overlapping occurrences included."""
    starts, i = [], haystack.find(needle)
    while i != -1:
        starts.append(i)
        i = haystack.find(needle, i + 1)
    return starts


# --------------------------------------------------------------------------- negation cue
NEGATION_CUES = frozenset({"not", "no", "never", "without", "cannot", "neither", "nor", "none", "nothing",
                           "nobody", "nowhere"})
NEGATION_SUFFIX = "n't"
_PRECEDING_WORD = re.compile(r"([^\W\d_][\w'’]*)\s+$")


def _negation_cue(passage: str, start: int) -> Optional[str]:
    window = "".join(c for c in passage[max(0, start - 40):start] if c not in _REMOVED)
    match = _PRECEDING_WORD.search(window)
    if not match:
        return None
    word = match.group(1).lower().replace("’", "'")
    return word if word in NEGATION_CUES or word.endswith(NEGATION_SUFFIX) else None


# --------------------------------------------------------------------------- location
def _span(passage: str, start: int, end: int, tier: MatchTier) -> CharSpan:
    return CharSpan(start, end, passage[start:end], tier, _negation_cue(passage, start))


def _canonical_locations(passage: str, canonical_passage: str, origin: Tuple[int, ...],
                         needle: str) -> List[Tuple[int, int, int]]:
    """(canonical start, original start, original end) of every occurrence of ``needle`` in ``canonical_passage``
    that is clean both in the canonical text and at its original offsets."""
    found = []
    for c_start in _occurrences(canonical_passage, needle):
        c_end = c_start + len(needle)
        start, end = origin[c_start], origin[c_end - 1] + 1
        if _clean(canonical_passage, c_start, c_end) and _clean(passage, start, end):
            found.append((c_start, start, end))
    return found


def attest_quote(quote: str, passage: str) -> Attestation:
    """Locate ``quote`` in ``passage``. Deterministic and side-effect free."""
    if not isinstance(quote, str) or not isinstance(passage, str):
        raise TypeError("quote and passage must be str")
    needle = _strip(quote)
    canonical_quote = canonicalize(needle)[0].strip()
    if not canonical_quote:
        return Attestation(AttestationStatus.EMPTY, None, (), AttestationReason.EMPTY_QUOTE)

    exact_starts = _occurrences(passage, needle)
    exact = [_span(passage, s, s + len(needle), MatchTier.EXACT)
             for s in exact_starts if _clean(passage, s, s + len(needle))]

    canonical_passage, origin = canonicalize(passage)
    # The canonical position of each exact location, so the canonical tier reports only locations not yet found.
    exact_locations = {bisect_left(origin, s.start) for s in exact}
    canonical_starts = _occurrences(canonical_passage, canonical_quote)
    canonical = [_span(passage, start, end, MatchTier.CANONICAL)
                 for c_start, start, end in _canonical_locations(passage, canonical_passage, origin, canonical_quote)
                 if c_start not in exact_locations]

    spans = tuple(sorted(exact + canonical, key=lambda s: (s.start, s.end)))
    if not spans:
        if _canonical_locations(passage, _fold_case(canonical_passage), origin, _fold_case(canonical_quote)):
            reason = AttestationReason.CASE_MISMATCH_ONLY
        elif exact_starts or canonical_starts:
            reason = AttestationReason.BOUNDARY_VIOLATION
        else:
            reason = AttestationReason.NO_CONTIGUOUS_MATCH
        return Attestation(AttestationStatus.UNATTESTED, None, (), reason)
    tier = MatchTier.EXACT if exact else MatchTier.CANONICAL
    if len(spans) > 1:
        return Attestation(AttestationStatus.AMBIGUOUS, tier, spans, AttestationReason.MULTIPLE_MATCHES)
    reason = (AttestationReason.UNIQUE_EXACT_MATCH if tier is MatchTier.EXACT
              else AttestationReason.UNIQUE_CANONICAL_MATCH)
    return Attestation(AttestationStatus.LOCATED, tier, spans, reason)


def compare_with_legacy(quote: str, passage: str, legacy_check: Callable[[str, str], bool]) -> LegacyComparison:
    """The location result and a legacy acceptance check side by side; neither result changes the other."""
    return LegacyComparison(attest_quote(quote, passage), bool(legacy_check(quote, passage)))
