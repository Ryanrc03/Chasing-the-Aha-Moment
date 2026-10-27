"""The overnight queue changes exactly one variable per run relative to the fixed config."""

from dataclasses import asdict
from pathlib import Path

import pytest

from aha.config import load_config
from aha.train import grpo_kwargs

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
BOOKKEEPING = {"output_dir", "save_steps"}

QUEUE = {
    "lr2e5": {"learning_rate": 2e-5},
    "noscale": {"scale_rewards": "none"},
    "nokl": {"kl_beta": 0.0},
    "g8": {"num_generations": 8},
}


@pytest.mark.parametrize("tag, change", sorted(QUEUE.items()))
def test_one_variable_per_run(tag, change):
    base = load_config(CONFIGS / "qwen2.5-3b_3060_fixed.yaml")
    run = load_config(CONFIGS / f"qwen2.5-3b_3060_fixed_{tag}.yaml")
    assert run.run_name == f"qwen3b_{tag}"
    assert run.train.output_dir == f"outputs/qwen3b_{tag}"
    # everything outside `train` is identical
    for section in ("model", "data", "reward", "eval"):
        assert asdict(getattr(run, section)) == asdict(getattr(base, section)), section
    diff = {k: v for k, v in asdict(run.train).items() if asdict(base.train)[k] != v and k not in BOOKKEEPING}
    assert diff == change


def test_grpo_kwargs_pass_optional_knobs_only_when_set():
    base = grpo_kwargs(load_config(CONFIGS / "qwen2.5-3b_3060_fixed.yaml"))
    assert base["beta"] == 0.04 and base["loss_type"] == "dr_grpo"
    assert "scale_rewards" not in base
    assert grpo_kwargs(load_config(CONFIGS / "qwen2.5-3b_3060_fixed_noscale.yaml"))["scale_rewards"] == "none"
    # beta = 0.0 is a real value, not "unset": the no-KL ablation must pass it through
    assert grpo_kwargs(load_config(CONFIGS / "qwen2.5-3b_3060_fixed_nokl.yaml"))["beta"] == 0.0
    assert grpo_kwargs(load_config(CONFIGS / "qwen2.5-3b_3060_fixed_lr2e5.yaml"))["learning_rate"] == 2e-5
    assert grpo_kwargs(load_config(CONFIGS / "qwen2.5-3b_3060_fixed_g8.yaml"))["num_generations"] == 8
    # the original run passes neither knob, so TRL's defaults (beta 0, dapo) applied
    orig = grpo_kwargs(load_config(CONFIGS / "qwen2.5-3b_3060.yaml"))
    assert "beta" not in orig and "loss_type" not in orig
