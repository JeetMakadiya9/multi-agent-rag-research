"""Quote-to-span location: synthetic strings only (no run records, snapshots, datasets or model calls)."""
from __future__ import annotations

import dataclasses

import pytest

from evidence_core.span_attestation import (
    CANONICALIZATION_VERSION, NEGATION_CUES, NEGATION_SUFFIX, SPAN_ATTESTATION_VERSION, Attestation,
    AttestationReason, AttestationStatus, CharSpan, MatchTier, attest_quote, canonicalize, compare_with_legacy,
)

LOCATED, AMBIGUOUS, UNATTESTED, EMPTY = (AttestationStatus.LOCATED, AttestationStatus.AMBIGUOUS,
                                         AttestationStatus.UNATTESTED, AttestationStatus.EMPTY)


def canonical_quote(quote):
    return canonicalize(quote)[0].strip()


def located(result, passage, quote=None):
    """Every reported span must reproduce its passage slice and, when given, canonicalise to the quote."""
    for s in result.spans:
        assert passage[s.start:s.end] == s.text
        if quote is not None:
            assert canonicalize(s.text)[0] == canonical_quote(quote)
    return [(s.start, s.end) for s in result.spans]


# --------------------------------------------------------------------------- exact tier
def test_exact_unique_match_reports_original_offsets():
    passage = "The bridge opened in 1932. It is 503 metres long."
    result = attest_quote("It is 503 metres long.", passage)
    assert result.status is LOCATED and result.tier is MatchTier.EXACT
    assert result.reason is AttestationReason.UNIQUE_EXACT_MATCH
    assert located(result, passage) == [(27, 49)]
    assert result.span.text == "It is 503 metres long."
    assert result.version == SPAN_ATTESTATION_VERSION and result.canonicalization == CANONICALIZATION_VERSION


def test_outer_whitespace_of_the_quote_is_ignored():
    passage = "The bridge opened in 1932."
    result = attest_quote("  \n The bridge opened in 1932.\t ", passage)
    assert result.status is LOCATED and result.span.text == "The bridge opened in 1932."


def test_quote_covering_the_whole_passage():
    result = attest_quote("Sales rose.", "Sales rose.")
    assert result.status is LOCATED and located(result, "Sales rose.") == [(0, 11)]


# --------------------------------------------------------------------------- canonical tier
def test_typographic_apostrophe_and_no_break_space_map_back_to_the_original():
    passage = "Tickets cost 12 euros. The museum isn’t open on Mondays.\nIt opens at 9."
    result = attest_quote("The museum isn't open on Mondays.", passage)
    assert result.status is LOCATED and result.tier is MatchTier.CANONICAL
    assert result.reason is AttestationReason.UNIQUE_CANONICAL_MATCH
    assert result.span.text == "The museum isn’t open on Mondays."
    located(result, passage, "The museum isn't open on Mondays.")


def test_line_break_inside_the_passage_matches_a_space_in_the_quote():
    passage = "The tower was completed\n   in 1889 for the fair."
    result = attest_quote("The tower was completed in 1889", passage)
    assert result.status is LOCATED and result.tier is MatchTier.CANONICAL
    assert result.span.text == "The tower was completed\n   in 1889"


def test_soft_hyphen_zero_width_space_and_ligature_are_transparent():
    passage = "The treatment was effec­tive in a ran​domised trial with a ﬁnal score of 7."
    for quote in ("The treatment was effective", "a randomised trial", "a final score of 7"):
        result = attest_quote(quote, passage)
        assert result.status is LOCATED and result.tier is MatchTier.CANONICAL, quote
        located(result, passage, quote)


def test_typographic_double_quotes_and_unicode_hyphen():
    passage = "He called it “well‐known” at the time."
    result = attest_quote('He called it "well-known" at the time.', passage)
    assert result.status is LOCATED and result.tier is MatchTier.CANONICAL


def test_canonicalize_reports_the_origin_of_every_character():
    text, origin = canonicalize("a  \tb’c­dﬁ")
    assert text == "a b'cdfi"
    assert origin == (0, 1, 4, 5, 6, 8, 9, 9)       # the collapsed space keeps the index of its run's first char


def test_quote_starting_inside_a_ligature_is_not_located():
    result = attest_quote("inal score", "a ﬁnal score")
    assert result.status is UNATTESTED and result.reason is AttestationReason.BOUNDARY_VIOLATION


# --------------------------------------------------------------------------- invariants over many cases
INVARIANT_CASES = [
    ("Sales rose.", "\U0001F600 Sales rose. Sales rose again."),
    ("Sales rose.", "Sales rose. Costs fell. Sales rose."),
    ("ha ha", "ha ha ha"),
    ("It isn't open.", "It isn't open. It isn’t open."),
    ("rose by 5 percent", "Costs\r\nrose by 5 percent."),
    ("a final score of 7", "with a ﬁnal score of 7."),
    ("The museum isn't open on Mondays.", "The museum isn’t open on Mondays."),
    ("​ ha", "x​ ha y"),
    ("the treatment was", "So the treat­ment was effective; the treatment was cheap."),
]


@pytest.mark.parametrize("quote, passage", INVARIANT_CASES)
def test_every_span_is_its_passage_slice_and_canonicalises_to_the_quote(quote, passage):
    result = attest_quote(quote, passage)
    assert result.spans                                   # every case has at least one location
    located(result, passage, quote)
    assert list(result.spans) == sorted(result.spans, key=lambda s: (s.start, s.end))


# --------------------------------------------------------------------------- Unicode at the edges (fixes a, b)
def test_invisible_characters_at_the_quote_ends_are_not_part_of_the_span():
    passage = "x​ ha y"
    result = attest_quote("​ ha", passage)         # was reported twice (one location, two spans)
    assert result.status is LOCATED and result.tier is MatchTier.EXACT
    assert located(result, passage, "​ ha") == [(3, 5)]


def test_exact_location_is_not_reported_again_by_the_canonical_tier():
    passage = "x ​ha ha y"
    result = attest_quote("​ha ha", passage)
    assert result.status is LOCATED and result.tier is MatchTier.EXACT
    assert located(result, passage) == [(3, 8)]


def test_span_may_not_cut_a_character_built_with_a_combining_mark():
    passage = "We met at the café at noon."
    assert attest_quote("the cafe", passage).reason is AttestationReason.BOUNDARY_VIOLATION
    assert attest_quote("́ at noon", passage).reason is AttestationReason.BOUNDARY_VIOLATION
    whole = attest_quote("the café at noon", passage)
    assert whole.status is LOCATED and located(whole, passage) == [(10, 27)]


def test_composed_and_decomposed_accents_do_not_match_each_other():
    result = attest_quote("the café", "We met at the café at noon.")
    assert result.status is UNATTESTED and result.reason is AttestationReason.NO_CONTIGUOUS_MATCH


# --------------------------------------------------------------------------- what canonicalisation must not merge
def test_letter_case_is_not_folded_and_case_only_mismatch_is_diagnosed():
    passage = "The plant produces 5 MW of power."
    for quote in ("The plant produces 5 mW of power.", "the plant produces 5 MW of power."):
        result = attest_quote(quote, passage)
        assert result.status is UNATTESTED and result.spans == ()
        assert result.reason is AttestationReason.CASE_MISMATCH_ONLY


def test_case_mismatch_diagnosis_respects_boundaries():
    result = attest_quote("the", "Theory of X")                 # "the" only inside "Theory", even ignoring case
    assert result.status is UNATTESTED and result.reason is AttestationReason.NO_CONTIGUOUS_MATCH


def test_case_variant_elsewhere_is_not_another_location():
    result = attest_quote("Sales rose.", "Sales rose. sales rose.")
    assert result.status is LOCATED and located(result, "Sales rose. sales rose.") == [(0, 11)]


def test_negation_is_preserved():
    passage = "The drug was not effective. The museum isn’t open."
    assert attest_quote("The drug was effective.", passage).status is UNATTESTED
    assert attest_quote("The museum is open.", passage).status is UNATTESTED


def test_minus_sign_en_dash_and_superscripts_are_not_folded():
    assert attest_quote("a change of -5 points", "a change of −5 points").status is UNATTESTED
    assert attest_quote("between 1990-1995", "between 1990–1995").status is UNATTESTED
    assert attest_quote("an area of 102 km", "an area of 10² km").status is UNATTESTED


# --------------------------------------------------------------------------- boundaries: words and numbers
@pytest.mark.parametrize("quote, passage", [
    ("effect", "The effective dose was low."),                 # inside a longer word
    ("effective dose", "The ineffective dose was low."),       # drops a negating prefix
    ("don", "We don't know."),                                 # inside a contraction
    ("well", "It is well-known."),                             # inside a hyphenated word
    ("effec", "It was effec­tive."),                      # a soft hyphen does not end a word
    ("5 percent", "Costs rose by 15 percent."),                # 5 inside 15
    ("1.5 mg", "A dose of 21.5 mg was used."),                 # inside a larger decimal
    ("A dose of 1.5", "A dose of 1.55 mg was used."),          # prefix of a longer decimal
    ("500 people", "About 1,500 people attended."),            # thousands group
    ("5 points", "a change of -5 points"),                     # sign
    ("5 points", "a change of −5 points"),                # minus sign
    ("5 points", "a change of +5 points"),
    ("5 points", "a change of ±5 points"),
    ("rose by 5", "Costs rose by 5% last year."),              # percent
    ("built in 1990", "It was built in 1990–1995."),      # first half of a range (en dash)
    ("1995.", "It was built in 1990-1995."),                   # second half of a range (hyphen)
    ("costs 1.", "It costs 1.5 euros."),                       # quote ends inside a decimal
    (".5 euros", "It costs 1.5 euros."),                       # quote starts inside a decimal
    ("an area of 10", "an area of 10² km"),               # base of a superscript power
    # Stage 1.1 regressions: invisible characters, spaced suffixes, group separators, currency, units, powers
    ("rose by 5", "Costs rose by 5­% last year."),        # soft hyphen before percent
    ("5 points", "a change of −​5 points"),          # zero-width space after a minus sign
    ("rose by 5", "Costs rose by 5 % last year."),             # percent after a space
    ("rose by 5", "Costs rose by 5 % last year."),        # percent after a no-break space
    ("500 people", "About 1 500 people attended."),       # thin-space group separator
    ("500 people", "About 1 500 people attended."),       # narrow no-break space group separator
    ("500 people", "About 1 500 people attended."),            # plain space between digits (conservative)
    ("in 1990", "in 1990 500 people moved"),                   # plain space between digits, at the end
    ("500 people", "About 1 500 people attended."),  # canonical tier: checked at original offsets
    ("rose by 5", "Costs rose by 5 % last year."),   # canonical tier: spaced percent
    ("5 million", "It cost $5 million."),                      # currency before
    ("5 million", "It cost € 5 million."),                # currency before, after a space
    ("cost 5", "It cost 5€ in total."),                   # currency after
    ("cost 5", "It cost 5 £ in total."),                  # currency after a space
    ("rose by 5", "It rose by 5°C overnight."),           # degree
    ("rose by 5", "It rose by 5 °C overnight."),          # degree after a space
    ("rose by 5", "It rose by 5℃ overnight."),            # degree Celsius sign
    ("a value of 1.5", "a value of 1.5×10⁵ was found"),   # multiplication sign
    ("a value of 10", "a value of 10^5 was found"),            # caret power
    ("a dose of 1", "a dose of 1·5 mg"),                  # middle-dot decimal
])
def test_partial_words_and_partial_numbers_are_boundary_violations(quote, passage):
    result = attest_quote(quote, passage)
    assert result.status is UNATTESTED and result.spans == ()
    assert result.reason is AttestationReason.BOUNDARY_VIOLATION


@pytest.mark.parametrize("quote, passage", [
    ("rose by 5 percent", "Costs rose by 5 percent."),
    ("built in 1990.", "It was built in 1990."),
    ("It costs 1.5 euros", "It costs 1.5 euros."),
    ("About 1,500 people attended.", "About 1,500 people attended."),
    ("a change of -5 points", "a change of -5 points"),
    ("between 1990–1995", "between 1990–1995 it grew"),
    ("costs 5", "It costs 5, or 6 at most."),                   # a comma not followed by a digit ends a number
    ("About 1 500 people", "About 1 500 people attended."),     # the whole spaced number is fine
    ("rose by 5 %", "Costs rose by 5 % last year."),
    ("It cost $5 million.", "It cost $5 million."),
    ("rose by 5°C", "It rose by 5°C overnight."),
])
def test_whole_numbers_signs_and_ranges_are_located(quote, passage):
    assert attest_quote(quote, passage).status is LOCATED


@pytest.mark.parametrize("quote, passage", [
    ("rose by 5", "It rose by 5 km."),
    ("rose by 5", "Costs rose by 5 percent."),
])
def test_units_written_as_words_are_a_documented_limitation(quote, passage):
    assert attest_quote(quote, passage).status is LOCATED       # no unit dictionary: the word is a boundary


# --------------------------------------------------------------------------- order and contiguity
def test_reordered_words_are_not_located():
    passage = "The tower is 300 metres tall and was built in 1889."
    result = attest_quote("The tower is tall, 300 metres", passage)
    assert result.status is UNATTESTED and result.reason is AttestationReason.NO_CONTIGUOUS_MATCH


def test_non_contiguous_word_overlap_is_not_located():
    passage = "The tower is 300 metres tall and was built in 1889."
    assert attest_quote("The tower is 300 tall and was built in 1889.", passage).status is UNATTESTED
    assert attest_quote("The tower was built in 1889.", passage).status is UNATTESTED


# --------------------------------------------------------------------------- ambiguity
def test_repeated_sentence_is_ambiguous_and_no_span_is_chosen():
    passage = "Sales rose. Costs fell. Sales rose."
    result = attest_quote("Sales rose.", passage)
    assert result.status is AMBIGUOUS and result.reason is AttestationReason.MULTIPLE_MATCHES
    assert result.tier is MatchTier.EXACT
    assert located(result, passage) == [(0, 11), (24, 35)]
    assert result.located is False and result.span is None


def test_overlapping_occurrences_are_all_reported():
    passage = "ha ha ha"
    assert located(attest_quote("ha ha", passage), passage) == [(0, 5), (3, 8)]


def test_an_exact_and_a_canonical_occurrence_together_are_ambiguous():
    passage = "It isn't open. It isn’t open."
    result = attest_quote("It isn't open.", passage)
    assert result.status is AMBIGUOUS and result.tier is MatchTier.EXACT
    assert [s.tier for s in result.spans] == [MatchTier.EXACT, MatchTier.CANONICAL]
    assert located(result, passage) == [(0, 14), (15, 29)]


def test_overlapping_exact_and_canonical_occurrences_stay_ambiguous():
    passage = "ha ha ha"
    result = attest_quote("ha ha", passage)
    assert result.status is AMBIGUOUS
    assert located(result, passage) == [(0, 5), (3, 8)]


def test_two_canonical_occurrences_are_ambiguous():
    passage = "ha ha ha"
    result = attest_quote("ha ha", passage)
    assert result.status is AMBIGUOUS and result.tier is MatchTier.CANONICAL
    assert located(result, passage) == [(0, 5), (3, 8)]


def test_one_clean_and_one_partial_occurrence_is_located_once():
    passage = "Costs rose by 5 percent. Costs rose by 15 percent."      # the second "5 percent" is inside "15"
    result = attest_quote("5 percent", passage)
    assert result.status is LOCATED and located(result, passage) == [(14, 23)]


# --------------------------------------------------------------------------- negation cue (fix i)
def test_negation_cue_list_is_published_exactly():
    assert NEGATION_CUES == frozenset({"not", "no", "never", "without", "cannot", "neither", "nor", "none",
                                       "nothing", "nobody", "nowhere"})
    assert NEGATION_SUFFIX == "n't"


def test_matched_negation_cue_is_recorded_but_does_not_change_the_status():
    passage = "The drug was not effective in adults. It doesn’t work in children. It was NOT safe."
    first = attest_quote("effective in adults.", passage)
    second = attest_quote("work in children.", passage)
    third = attest_quote("safe.", passage)
    assert first.status is LOCATED and first.span.negation_cue == "not" and first.span.preceded_by_negation
    assert second.status is LOCATED and second.span.negation_cue == "doesn't"
    assert third.status is LOCATED and third.span.negation_cue == "not"
    plain = attest_quote("effective in adults.", "The drug was effective in adults.")
    assert plain.span.negation_cue is None and plain.span.preceded_by_negation is False


@pytest.mark.parametrize("quote, passage, cue", [
    ("effective.", "It was no longer effective.", None),        # multi-word cue: missed
    ("effective.", "No, effective.", None),                     # cue followed by punctuation: missed
    ("only effective", "It is not only effective.", "not"),     # "not only": reported although not a negation
])
def test_negation_cue_heuristic_limits_are_documented(quote, passage, cue):
    result = attest_quote(quote, passage)
    assert result.status is LOCATED and result.span.negation_cue == cue


# --------------------------------------------------------------------------- location is not support (fix j)
def test_statuses_and_fields_describe_location_only():
    assert {s.value for s in AttestationStatus} == {"LOCATED", "AMBIGUOUS", "UNATTESTED", "EMPTY"}
    assert [f.name for f in dataclasses.fields(Attestation)] == ["status", "tier", "spans", "reason", "version",
                                                                  "canonicalization"]
    assert [f.name for f in dataclasses.fields(CharSpan)] == ["start", "end", "text", "tier", "negation_cue"]


# --------------------------------------------------------------------------- empty and invalid input
@pytest.mark.parametrize("quote", ["", "   ", "\n\t", "​", "­​ "])
def test_empty_quote(quote):
    result = attest_quote(quote, "Some passage.")
    assert result.status is EMPTY and result.reason is AttestationReason.EMPTY_QUOTE and result.spans == ()


def test_empty_passage_is_unattested():
    result = attest_quote("Sales rose.", "")
    assert result.status is UNATTESTED and result.reason is AttestationReason.NO_CONTIGUOUS_MATCH


def test_non_string_input_is_rejected():
    with pytest.raises(TypeError):
        attest_quote(None, "passage")
    with pytest.raises(TypeError):
        attest_quote("quote", b"passage")


def test_results_are_deterministic():
    passage = "It isn't open. It isn’t open. The drug was not effective."
    for quote in ("It isn't open.", "effective.", "missing"):
        assert attest_quote(quote, passage) == attest_quote(quote, passage)


# --------------------------------------------------------------------------- legacy check, reported separately
def test_legacy_result_is_reported_next_to_the_location():
    passage = "The tower is 300 metres tall and was built in 1889."
    calls = []

    def legacy(quote, text):
        calls.append((quote, text))
        return True

    comparison = compare_with_legacy("The tower is tall, 300 metres", passage, legacy)
    assert comparison.legacy_accepts is True
    assert comparison.attestation.status is UNATTESTED
    assert calls == [("The tower is tall, 300 metres", passage)]


def test_existing_quote_in_passage_accepts_word_overlap_that_location_rejects():
    from docqa.verify import quote_in_passage            # frozen helper, called read-only

    passage = "The tower is 300 metres tall and was built in 1889."
    reordered = compare_with_legacy("The tower is tall and 300 metres was built in 1889.", passage, quote_in_passage)
    assert reordered.legacy_accepts is True and reordered.attestation.status is UNATTESTED
    verbatim = compare_with_legacy("The tower is 300 metres tall", passage, quote_in_passage)
    assert verbatim.legacy_accepts is True and verbatim.attestation.status is LOCATED
