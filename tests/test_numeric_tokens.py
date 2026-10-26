"""The numeric token contract (webrag/numeric.py) on its own: no application code, no model, no network."""
import pytest

from webrag.numeric import NUMERIC_VERSION, numbers, unsupported_numbers


# ------------------------------------------------------------------------------------------- token contract
@pytest.mark.parametrize("text, expected", [
    ("Capacity grew 20 percent in 2024.", ["20", "2024"]),
    ("A dose of 1.5 mg and 21.5 mg.", ["1.5", "21.5"]),
    ("About 1,500 people; 1,500,000.25 dollars.", ["1,500", "1,500,000.25"]),
    ("A decimal comma 1,5 and a point 1.5.", ["1,5", "1.5"]),
    ("Trailing zeros 1.50 and leading zeros 05.", ["1.50", "05"]),
    ("It fell by .5 points.", [".5"]),
    ("A change of -5, +5, −5 and ±5.", ["-5", "+5", "-5", "±5"]),
    ("Built 1990-1995, open 10–20 hours.", ["1990", "1995", "10", "20"]),
    ("COVID-19 and SARS-CoV-2 in model x-5.", ["19", "2", "5"]),
    ("On 2024-05-01 at 10:30.", ["2024", "05", "01", "10", "30"]),
    ("Rates of 5% and 5 percent over 5 km.", ["5", "5", "5"]),
    ("Scientific 1.5e-3 and 1.5×10⁵.", ["1.5", "3", "1.5", "10"]),
    ("No numbers here.", []),
    ("", []),
])
def test_tokens_follow_the_documented_contract(text, expected):
    assert numbers(text) == expected


def test_substring_collisions_are_not_support():
    assert unsupported_numbers("Costs rose 5%.", ["Costs rose 25%."]) == ["5"]
    assert unsupported_numbers("A dose of 1.5 mg.", ["A dose of 21.5 mg."]) == ["1.5"]
    assert unsupported_numbers("It took 2 years.", ["It ended in 2024."]) == ["2"]
    assert unsupported_numbers("A ratio of 0.5.", ["A ratio of 10.52."]) == ["0.5"]
    assert unsupported_numbers("May 1.", ["On 2024-05-01."]) == ["1"]


def test_exact_tokens_are_support_in_any_cited_finding():
    assert unsupported_numbers("Capacity rose 20 percent in 2024.", ["Capacity grew 20 percent in 2024."]) == []
    assert unsupported_numbers("1,500 people and 7 sites.", ["1,500 people came.", "There were 7 sites."]) == []
    assert unsupported_numbers("Between 1990 and 1995.", ["Built 1990-1995."]) == []      # range components
    assert unsupported_numbers("Rose 5 percent.", ["Rose 5%."]) == []                      # units are not compared


@pytest.mark.parametrize("summary, finding", [
    ("1500 people", "1,500 people"),        # thousands separator: as written, no equivalence
    ("1,500 people", "1500 people"),
    ("1.5 points", "1,5 points"),           # decimal comma vs point
    ("1.5 mg", "1.50 mg"),                  # trailing zero
    ("5 points", "-5 points"),              # a sign is part of the number
    ("-5 points", "5 points"),
    ("+5 points", "5 points"),
    ("5 May", "05 May"),                    # leading zero
    ("1 May 2024", "2024-05-01"),           # dates are not normalised
])
def test_conservative_choices_reject_reformatted_numbers(summary, finding):
    assert unsupported_numbers(summary, [finding])


def test_unicode_minus_and_hyphen_minus_are_the_same_sign():
    assert unsupported_numbers("A change of −5 points.", ["A change of -5 points."]) == []


def test_missing_numbers_are_reported_once_in_order():
    assert unsupported_numbers("3, 4 and 3 again", ["4"]) == ["3"]
    assert NUMERIC_VERSION == "numeric-tokens-v1"
