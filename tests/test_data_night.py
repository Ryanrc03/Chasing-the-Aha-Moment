"""Difficulty filter, gate-only format reward and the two data-night configs."""

import json
from pathlib import Path

import pytest

from aha.config import RewardConfig, load_config
from aha.data import load_pass_counts, select_by_pass_count
from aha.rewards import build_reward_funcs, format_intact

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
GOOD = "<reasoning>\n" + "step " * 40 + "\n</reasoning>\n<answer>\n4\n</answer>"
THIN = "<reasoning>\nguess\n</reasoning>\n<answer>\n4\n</answer>"
BROKEN = "The answer is 4"
WRONG = "<reasoning>\n" + "step " * 40 + "\n</reasoning>\n<answer>\n5\n</answer>"


def comps(*texts):
    return [[{"role": "assistant", "content": t}] for t in texts]


PROMPTS = [[{"role": "user", "content": "q"}]] * 4


def test_pass_count_filter(tmp_path):
    f = tmp_path / "p.jsonl"
    f.write_text("\n".join(json.dumps({"idx": i, "passes": p, "n": 6}) for i, p in [(5, 0), (2, 3), (9, 6), (1, 1), (4, 5)]) + "\n")
    passes = load_pass_counts(str(f))
    assert passes == {5: 0, 2: 3, 9: 6, 1: 1, 4: 5}
    assert select_by_pass_count(passes, 1, 5) == [1, 2, 4]
    assert select_by_pass_count(passes, 0, 6) == [1, 2, 4, 5, 9]


def test_format_intact():
    assert format_intact(GOOD, 30)
    assert not format_intact(THIN, 30) and format_intact(THIN, 0)
    assert not format_intact(BROKEN, 0)
    assert not format_intact("<reasoning>x</reasoning><answer>4", 0)  # unclosed answer


def test_gate_mode_rewards():
    funcs = {f.__name__: f for f in build_reward_funcs(RewardConfig(format_mode="gate", match="relaxed", min_reasoning_words=30, log_samples=False))}
    assert set(funcs) == {"format_gate_reward_func", "correctness_reward_func"}
    c = comps(GOOD, THIN, BROKEN, WRONG)
    gate = funcs["format_gate_reward_func"](completions=c)
    corr = funcs["correctness_reward_func"](prompts=PROMPTS, completions=c, answer=["4"] * 4)
    assert gate == [0.25, 0.0, 0.0, 0.25]
    assert corr == [2.0, 0.0, 0.0, 0.0]  # right but thin or broken earns nothing


def test_gate_mode_gives_identical_rewards_to_a_group_of_correct_well_formed_answers():
    funcs = build_reward_funcs(RewardConfig(format_mode="gate", match="relaxed", min_reasoning_words=30, log_samples=False))
    a = GOOD
    b = "<reasoning>\n" + "another way " * 30 + "\n</reasoning>\n<answer>\n4\n</answer>\n\n"  # trailing text differs
    totals = [sum(f(prompts=PROMPTS[:2], completions=comps(a, b), answer=["4", "4"])[i] for f in funcs) for i in range(2)]
    assert totals[0] == totals[1] == 2.25
    # the graded notebook rewards do separate the same two answers
    graded = build_reward_funcs(RewardConfig(match="relaxed", log_samples=False))
    g = [sum(f(prompts=PROMPTS[:2], completions=comps(a, b), answer=["4", "4"])[i] for f in graded) for i in range(2)]
    assert g[0] != g[1]


def test_unknown_format_mode():
    with pytest.raises(ValueError):
        build_reward_funcs(RewardConfig(format_mode="nope"))


def test_data_night_configs():
    base = load_config(CONFIGS / "qwen2.5-3b_3060_fixed.yaml")
    a = load_config(CONFIGS / "qwen2.5-3b_3060_fixed_filtered.yaml")
    b = load_config(CONFIGS / "qwen2.5-3b_3060_fixed_filtered_gate.yaml")
    assert a.data.filter_file == "outputs/presample/gsm8k_instruct_g6.jsonl"
    assert (a.data.filter_min_pass, a.data.filter_max_pass) == (1, 5)
    assert a.reward == base.reward and a.train.learning_rate == base.train.learning_rate
    assert a.train.kl_beta == 0.04 and a.train.loss_type == "dr_grpo"
    assert b.data == a.data and b.reward.format_mode == "gate" and a.reward.format_mode == "graded"
    assert b.run_name == "qwen3b_filtered_gate" and b.train.output_dir == "outputs/qwen3b_filtered_gate"
