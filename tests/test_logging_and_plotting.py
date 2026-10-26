import json
import math

import pytest

from aha.logging_utils import HISTORY_NAME, JSONL_NAME, read_log
from aha.plotting import detect_kind, rolling_mean

matplotlib = pytest.importorskip("matplotlib", reason="plot tests need matplotlib")


def fake_grpo_records(n=30):
    out = []
    for s in range(1, n + 1):
        out.append({
            "step": s,
            "loss": 0.3 / s,
            "reward": 0.5 + 1.5 * (1 - math.exp(-s / 10)),
            "reward_std": 0.8,
            "kl": 0.001 * s,
            "grad_norm": 0.2,
            "completion_length": 400 - 3 * s,
            "completions/clipped_ratio": 0.1,
            "rewards/correctness_reward_func/mean": 1.0 + s / n,
            "rewards/xmlcount_reward_func/mean": -0.5 + s / n,
            "rewards/soft_format_reward_func/mean": 0.3,
            "rewards/strict_format_reward_func/mean": 0.0,
            "rewards/int_reward_func/mean": 0.2,
        })
    return out


def test_rolling_mean():
    assert rolling_mean([1, 2, 3, 4], 2) == [1, 1.5, 2.5, 3.5]
    assert rolling_mean([1, 2, 3], 1) == [1, 2, 3]


def test_read_log_prefers_history_then_jsonl(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    with pytest.raises(FileNotFoundError):
        read_log(run)
    with open(run / JSONL_NAME, "w") as f:
        f.write(json.dumps({"step": 1, "loss": 1.0}) + "\n")
        f.write(json.dumps({"step": 2, "loss": 0.5}) + "\n")
    assert [r["step"] for r in read_log(run)] == [1, 2]
    with open(run / HISTORY_NAME, "w") as f:
        json.dump([{"step": 1, "loss": 1.0}, {"train_runtime": 3.0}], f)  # summary row has no step
    assert read_log(run) == [{"step": 1, "loss": 1.0}]


def test_detect_kind():
    assert detect_kind(fake_grpo_records(3)) == "grpo"
    assert detect_kind([{"step": 1, "loss": 1.0, "learning_rate": 1e-4}]) == "sft"


def test_plot_run_writes_png_and_pdf(tmp_path):
    from aha.plotting import plot_run

    grpo = tmp_path / "grpo_run"
    grpo.mkdir()
    with open(grpo / HISTORY_NAME, "w") as f:
        json.dump(fake_grpo_records(), f)
    sft = tmp_path / "sft_run"
    sft.mkdir()
    with open(sft / JSONL_NAME, "w") as f:
        for s in range(1, 11):
            f.write(json.dumps({"step": s, "loss": 1 / s, "learning_rate": 2e-4 * (1 - s / 10)}) + "\n")

    written = plot_run(grpo, tmp_path / "fig") + plot_run(sft, tmp_path / "fig")
    names = sorted(p.name for p in written)
    assert names == sorted([
        "grpo_run_grpo.png", "grpo_run_grpo.pdf",
        "grpo_run_grpo_extra.png", "grpo_run_grpo_extra.pdf",
        "sft_run_sft.png", "sft_run_sft.pdf",
    ])
    assert all(p.stat().st_size > 1000 for p in written)
