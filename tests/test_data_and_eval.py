import pytest

from aha.data import (
    EVAL_LOADERS,
    TRAIN_LOADERS,
    extract_boxed,
    format_gsm8k_row,
    format_math_row,
    load_eval_dataset,
    load_train_dataset,
)
from aha.evaluate import Prediction, batched, score_output, summarize
from aha.prompts import SYSTEM_PROMPT


def test_format_gsm8k_row():
    row = {"question": "What is 2+2?", "answer": "2+2=4\n#### 4"}
    out = format_gsm8k_row(row)
    assert out["answer"] == "4"
    assert out["prompt"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert out["prompt"][1] == {"role": "user", "content": "What is 2+2?"}


def test_extract_boxed_nested():
    assert extract_boxed("so $\\boxed{\\frac{1}{2}}$.") == "\\frac{1}{2}"
    assert extract_boxed("first \\boxed{1} then \\boxed{2}") == "2"
    assert extract_boxed("no box") is None
    assert extract_boxed("\\boxed{unterminated") is None


def test_format_math_row():
    out = format_math_row({"problem": "p", "solution": "\\boxed{7}"})
    assert out["answer"] == "7"


def test_unknown_dataset_names():
    with pytest.raises(ValueError):
        load_train_dataset("nope")
    with pytest.raises(ValueError):
        load_eval_dataset("nope")
    assert set(TRAIN_LOADERS) == {"gsm8k", "math"}
    assert set(EVAL_LOADERS) == {"gsm8k_test", "aime2024", "aime2025", "math500"}


def test_batched():
    assert list(batched(range(5), 2)) == [[0, 1], [2, 3], [4]]
    assert list(batched([], 2)) == []


def test_score_output():
    good = "<reasoning>\nx\n</reasoning>\n<answer>\n$1,200\n</answer>\n"
    s = score_output(good, "1200")
    assert s == {"extracted": "$1,200", "xml_ok": True, "soft_format_ok": True, "correct": True}
    s = score_output("1200", "1200")
    assert s["xml_ok"] is False and s["correct"] is False and s["extracted"] is None
    s = score_output("<answer>12</answer>", "1200")
    assert s["xml_ok"] is True and s["soft_format_ok"] is False and s["correct"] is False


def test_summarize():
    preds = [
        Prediction(0, "1", "1", True, True, True, 10, ""),
        Prediction(1, "2", "3", False, True, False, 30, ""),
    ]
    m = summarize("gsm8k_test", preds, [0.5, 1.5], 2.0, None, "m")
    assert m.accuracy == 0.5
    assert m.xml_parse_rate == 1.0
    assert m.soft_format_rate == 0.5
    assert m.avg_latency_s == 1.0
    assert m.avg_completion_tokens == 20.0
    empty = summarize("x", [], [], 0.0, None, "m")
    assert empty.accuracy == 0.0
