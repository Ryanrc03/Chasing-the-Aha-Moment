import pytest

from aha.parsing import (
    SOFT_XML_PATTERN,
    STRICT_XML_PATTERN,
    extract_hash_answer,
    extract_reasoning,
    extract_xml_answer,
    has_answer_tag,
    is_correct,
    normalize_num_like,
    relaxed_equal,
)

GOOD = "<reasoning>\n2 + 2 = 4\n</reasoning>\n<answer>\n4\n</answer>\n"


def test_extract_xml_answer():
    assert extract_xml_answer(GOOD) == "4"
    assert extract_xml_answer("<answer> 42 </answer> trailing") == "42"


def test_extract_xml_answer_without_tag_returns_text():
    assert extract_xml_answer("just 7") == "just 7"
    assert not has_answer_tag("just 7")
    assert has_answer_tag(GOOD)


def test_extract_reasoning():
    assert extract_reasoning(GOOD) == "2 + 2 = 4"
    assert extract_reasoning("no tags") == ""
    assert extract_reasoning("<reasoning>unterminated") == "unterminated"


def test_extract_hash_answer():
    assert extract_hash_answer("steps...\n#### 18") == "18"
    assert extract_hash_answer("no marker") is None


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("18", "18"),
        (" 1,200. ", "1200"),
        ("$18", "18"),
        ("The answer is 42", "42"),
        ("-7", "-7"),
        ("abc", "abc"),
        (None, ""),
    ],
)
def test_normalize_num_like(raw, expected):
    assert normalize_num_like(raw) == expected


def test_relaxed_equal():
    assert relaxed_equal("$1,200", "1200")
    assert relaxed_equal("The answer is 42.", "42")
    assert not relaxed_equal("41", "42")
    assert not relaxed_equal(None, "42")
    assert relaxed_equal("x+1", "x+1")


def test_is_correct_modes():
    assert is_correct("18", "18", "exact")
    assert not is_correct("$18", "18", "exact")
    assert is_correct("$18", "18", "relaxed")
    with pytest.raises(ValueError):
        is_correct("1", "1", "fuzzy")


def test_patterns():
    assert STRICT_XML_PATTERN.match(GOOD)
    assert SOFT_XML_PATTERN.match(GOOD)
    loose = "<reasoning>x</reasoning> <answer>1</answer>"
    assert not STRICT_XML_PATTERN.match(loose)
    assert SOFT_XML_PATTERN.match(loose)
