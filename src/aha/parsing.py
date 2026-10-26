"""Answer extraction and comparison helpers (pure Python, no ML deps)."""

from __future__ import annotations

import re
from typing import Optional

SOFT_XML_PATTERN = re.compile(r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>", re.DOTALL)
STRICT_XML_PATTERN = re.compile(
    r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$", re.DOTALL
)


def extract_xml_answer(text: str) -> str:
    """Text inside the last ``<answer>...</answer>`` block, stripped.

    Mirrors the notebook behaviour: if the tag is missing the whole text
    (or its tail) is returned, so callers that need a presence check should
    use :func:`has_answer_tag`.
    """
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def has_answer_tag(text: str) -> bool:
    return "<answer>" in text and "</answer>" in text


def extract_reasoning(text: str) -> str:
    """Text inside ``<reasoning>...</reasoning>``; empty string if absent."""
    if "<reasoning>" not in text:
        return ""
    body = text.split("<reasoning>", 1)[1]
    return body.split("</reasoning>")[0].strip()


def extract_hash_answer(text: str) -> Optional[str]:
    """GSM8K gold answers end with ``#### 42``; return the part after ``####``."""
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def normalize_num_like(s: Optional[str]) -> str:
    """Normalise a numeric-looking string: strip commas, currency, trailing dots.

    If the string contains integers, the last one is taken as the final answer.
    """
    if s is None:
        return ""
    t = s.strip().rstrip(".").replace(",", "")
    t = t.replace("$", "").replace("USD", "").replace("usd", "")
    tokens = re.findall(r"-?\d+", t)
    if len(tokens) == 1 and t == tokens[0]:
        return str(int(tokens[0]))
    if tokens:
        return str(int(tokens[-1]))
    return t


def relaxed_equal(pred: Optional[str], gold: Optional[str]) -> bool:
    """Numeric-normalised equality, falling back to stripped string equality."""
    if gold is None or pred is None:
        return False
    p = normalize_num_like(pred)
    g = normalize_num_like(gold)
    try:
        return int(p) == int(g)
    except ValueError:
        return p.strip() == g.strip()


def is_correct(pred: Optional[str], gold: Optional[str], mode: str = "exact") -> bool:
    if mode == "exact":
        return pred is not None and gold is not None and pred == gold
    if mode == "relaxed":
        return relaxed_equal(pred, gold)
    raise ValueError(f"unknown match mode: {mode!r} (expected 'exact' or 'relaxed')")
