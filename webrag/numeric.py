"""Numeric grounding: which numbers a text states, compared as whole tokens (summary validation, ``agents._statement_ok``).

A generated sentence may only state a number that its cited findings state as the same token. The digits of one
number inside another never count: "5" is not supported by "25", "1.5" not by "21.5", "2" not by "2024".

Token contract (``numbers``), applied identically to the sentence and to the findings:

* a token is a maximal run of ASCII digits with internal separators: ``[0-9]+([.,][0-9]+)*``. "1,500", "1.5" and
  "1,500,000.25" are one token each; a separator not followed by a digit ends the token ("in 2024." is "2024").
* a leading decimal point with no digit before it is kept: ".5" is the token ".5" (not "5", not "0.5").
* tokens are compared as written. This is the grounding contract selected for summary validation, not a claim about
  numeric equivalence: "1,500" and "1500" are the same quantity, but here they are different tokens, as are "1.50"
  and "1.5", "05" and "5", and a decimal comma "1,5" and "1.5". The choice is conservative (a sentence that
  reformats a number is dropped, which removes text but never adds an unsupported figure) and reversible: a later
  contract could normalise separators or trailing zeros, at the cost of deciding locale (is "1,500" 1500 or 1.5?).
* a sign belongs to the token when it directly precedes the digits and is itself not preceded by a letter, digit or
  separator: "-5", "+5" and "±5" are tokens distinct from "5". The Unicode minus (U+2212) and the hyphen-minus are the
  same sign ("−5" == "-5"). A hyphen between two numbers is a range, not a sign ("1990-1995" gives "1990" and
  "1995"); a hyphen after a letter is not a sign ("COVID-19" gives "19").
* units and percent signs are not part of the token: "5%", "5 percent" and "5 km" all state "5". Units are not
  compared here.
* ranges, dates and times are their component tokens: "10–20" gives "10" and "20"; "2024-05-01" gives "2024", "05",
  "01"; "10:30" gives "10" and "30". A date written differently ("1 May 2024") does not match.
* scientific notation is not interpreted: "1.5e-3" gives "1.5" and "3", "1.5×10⁵" gives "1.5" and "10" (superscript
  digits are not ASCII digits). Only non-ASCII decimal digits are ignored entirely.

This is a lexical check, not arithmetic: rounding, unit conversion and recomputation are not recognised.
The other number checks in the application keep their own documented rules and are not changed by this module:
``docqa.verify.numbers_in`` (claim vs passage in the frozen verifier) and ``webrag.deep._numbers`` (the
``number_unsupported`` caution) both drop thousands separators and signs.
"""
from __future__ import annotations

import re
from typing import Iterable, List, Set

NUMERIC_VERSION = "numeric-tokens-v1"

_TOKEN = re.compile(r"(?<![0-9.,])(?:\.[0-9]+|[0-9]+(?:[.,][0-9]+)*)")
_SIGNS = {"-": "-", "−": "-", "+": "+", "±": "±"}


def numbers(text: str) -> List[str]:
    """The numeric tokens of ``text`` in order of appearance, with their sign (see the module contract)."""
    found = []
    for match in _TOKEN.finditer(text or ""):
        token, start = match.group(0), match.start()
        if start > 0 and text[start - 1] in _SIGNS:
            before = text[start - 2] if start > 1 else ""
            if not before or not (before.isalnum() or before in ".,"):
                token = _SIGNS[text[start - 1]] + token
        found.append(token)
    return found


def unsupported_numbers(text: str, sources: Iterable[str]) -> List[str]:
    """The tokens ``text`` states that no source states as the same token (in order, without repeats)."""
    available: Set[str] = set()
    for source in sources:
        available.update(numbers(source))
    missing: List[str] = []
    for token in numbers(text):
        if token not in available and token not in missing:
            missing.append(token)
    return missing
