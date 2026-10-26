import pytest

from aha.config import LengthRewardConfig, RewardConfig
from aha.rewards import (
    build_reward_funcs,
    count_xml,
    length_reward,
    make_correctness_reward,
    make_int_reward,
    make_length_reward,
    make_soft_format_reward,
    make_strict_format_reward,
    make_xmlcount_reward,
    reasoning_length,
)

STRICT = "<reasoning>\n2 + 2 = 4\n</reasoning>\n<answer>\n4\n</answer>\n"
SOFT = "<reasoning>2 + 2 = 4</reasoning>\n<answer>4</answer>"
WRONG = "<reasoning>\nguess\n</reasoning>\n<answer>\n5\n</answer>\n"
BARE = "The answer is 4"


def completions(*texts):
    return [[{"role": "assistant", "content": t}] for t in texts]


PROMPTS = [[{"role": "system", "content": "s"}, {"role": "user", "content": "q"}]] * 4


class FakeTokenizer:
    """Counts characters as tokens; enough to exercise the tokenizer path."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(text)}


def test_correctness_reward_exact_and_relaxed(capsys):
    comps = completions(STRICT, WRONG, "<answer>$4</answer>", BARE)
    exact = make_correctness_reward(2.0, "exact", log_samples=False)
    assert exact(PROMPTS, comps, answer=["4"] * 4) == [2.0, 0.0, 0.0, 0.0]
    relaxed = make_correctness_reward(2.0, "relaxed", log_samples=True)
    assert relaxed(PROMPTS, comps, answer=["4"] * 4) == [2.0, 0.0, 2.0, 2.0]
    assert "Question:" in capsys.readouterr().out


def test_int_and_format_rewards():
    comps = completions(STRICT, SOFT, WRONG, BARE)
    assert make_int_reward()(comps) == [0.5, 0.5, 0.5, 0.0]
    assert make_strict_format_reward()(comps) == [0.5, 0.0, 0.5, 0.0]
    assert make_soft_format_reward()(comps) == [0.5, 0.5, 0.5, 0.0]


def test_count_xml_matches_notebook_values():
    assert count_xml(STRICT) == pytest.approx(0.5)
    assert count_xml(STRICT + "extra") == pytest.approx(0.5 - 5 * 0.001 - 5 * 0.001)
    assert count_xml(BARE) == 0.0
    assert make_xmlcount_reward(2.0)(completions(STRICT)) == [pytest.approx(1.0)]


def test_reasoning_length_with_and_without_tokenizer():
    assert reasoning_length(STRICT) == 5  # "2 + 2 = 4" has five whitespace tokens
    assert reasoning_length(STRICT, FakeTokenizer()) == len("2 + 2 = 4")
    assert reasoning_length(BARE) == 0


def test_length_reward_sign_and_cap():
    # wrong answer -> alpha * normalised length (bonus)
    assert length_reward(WRONG, "4", alpha=0.5, beta=0.1, l_max=10) == pytest.approx(0.5 * 1 / 10)
    # right answer -> -beta * normalised length (penalty)
    assert length_reward(STRICT, "4", alpha=0.5, beta=0.1, l_max=10) == pytest.approx(-0.1 * 5 / 10)
    # capped at 1.0 when the trace is longer than l_max
    assert length_reward(WRONG, "4", alpha=0.5, beta=0.1, l_max=1) == pytest.approx(0.5)
    # no reasoning segment -> zero either way
    assert length_reward(BARE, "4", alpha=0.5, beta=0.1, l_max=10, match="relaxed") == 0.0


def test_length_reward_func_signature():
    fn = make_length_reward(alpha=1.0, beta=0.5, l_max=5)
    out = fn(completions(WRONG, STRICT), answer=["4", "4"], completion_ids=[[1], [2]])
    assert out == [pytest.approx(1.0 * 1 / 5), pytest.approx(-0.5 * 1.0)]
    assert fn.__name__ == "length_reward_func"


def test_build_reward_funcs_order_and_optional_length():
    base = build_reward_funcs(RewardConfig(log_samples=False))
    assert [f.__name__ for f in base] == [
        "xmlcount_reward_func",
        "soft_format_reward_func",
        "strict_format_reward_func",
        "int_reward_func",
        "correctness_reward_func",
    ]
    shaped = build_reward_funcs(RewardConfig(log_samples=False, length=LengthRewardConfig(enabled=True)))
    assert [f.__name__ for f in shaped][-1] == "length_reward_func"
    assert len(shaped) == 6


def test_total_reward_for_perfect_answer():
    funcs = build_reward_funcs(RewardConfig(log_samples=False))
    # TRL calls reward functions with keyword arguments only
    total = sum(f(prompts=PROMPTS[:1], completions=completions(STRICT), answer=["4"])[0] for f in funcs)
    assert total == pytest.approx(0.5 + 0.5 + 0.5 + 0.5 + 2.0)


def test_strict_format_accepts_eos_without_trailing_newline():
    # vLLM stops at EOS with no final newline; the notebook regex rejected every such output
    no_newline = "<reasoning>\n2 + 2 = 4\n</reasoning>\n<answer>\n4\n</answer>"
    assert make_strict_format_reward()(completions(no_newline)) == [0.5]


def test_reasoning_gate_zeroes_format_rewards_only():
    cfg = RewardConfig(log_samples=False, min_reasoning_words=3)
    funcs = {f.__name__: f for f in build_reward_funcs(cfg)}
    thin = "<reasoning>\nguess\n</reasoning>\n<answer>\n4\n</answer>\n"
    comps = completions(STRICT, thin)
    assert funcs["int_reward_func"](completions=comps) == [0.5, 0.0]
    assert funcs["strict_format_reward_func"](completions=comps) == [0.5, 0.0]
    assert funcs["xmlcount_reward_func"](completions=comps)[1] == 0.0
    # correctness is never gated
    assert funcs["correctness_reward_func"](prompts=PROMPTS[:2], completions=comps, answer=["4", "4"]) == [2.0, 2.0]
