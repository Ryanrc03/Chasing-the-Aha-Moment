"""Report figures from training logs (matplotlib, static PNG/PDF for the LaTeX report).

Conventions: one y-axis per panel (never dual axes), small multiples for different
measures, thin 2px lines, raw values faint + rolling mean on top, fixed categorical
colour order, recessive grid, text in neutral ink.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

# Fixed categorical order (light surface). Never cycled: series i always gets SERIES[i].
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

REWARD_LABELS = {
    "rewards/correctness_reward_func/mean": "correctness",
    "rewards/int_reward_func/mean": "int",
    "rewards/strict_format_reward_func/mean": "strict format",
    "rewards/soft_format_reward_func/mean": "soft format",
    "rewards/xmlcount_reward_func/mean": "xml count",
    "rewards/length_reward_func/mean": "length (R_len)",
    "rewards/format_gate_reward_func/mean": "format gate",
}


def _series(records: list[dict], key: str) -> tuple[list[int], list[float]]:
    xs, ys = [], []
    for r in records:
        v = r.get(key)
        if isinstance(v, (int, float)):
            xs.append(r["step"])
            ys.append(float(v))
    return xs, ys


def rolling_mean(ys: list[float], window: int) -> list[float]:
    if window <= 1 or len(ys) < 2:
        return list(ys)
    out = []
    for i in range(len(ys)):
        lo = max(0, i - window + 1)
        chunk = ys[lo:i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def _style(ax, title: str, ylabel: str = ""):
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=8)
    ax.set_xlabel("step", color=INK_2, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8)


def _plot_line(ax, xs, ys, color, label=None, window=1):
    if not xs:
        return
    if window > 1 and len(ys) > window:
        ax.plot(xs, ys, color=color, linewidth=1, alpha=0.25)
        ax.plot(xs, rolling_mean(ys, window), color=color, linewidth=2, label=label)
    else:
        ax.plot(xs, ys, color=color, linewidth=2, label=label)


def plot_grpo(records: list[dict], title: str, window: int = 10):
    """2x2 panel: total reward, per-function rewards, loss + KL (separate), completion length."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    fig.suptitle(title, color=INK, fontsize=13, x=0.01, ha="left")

    ax = axes[0, 0]
    _plot_line(ax, *_series(records, "reward"), SERIES[0], window=window)
    _style(ax, "Total reward (mean over group)")

    ax = axes[0, 1]
    i = 0
    for key, label in REWARD_LABELS.items():
        xs, ys = _series(records, key)
        if xs:
            _plot_line(ax, xs, ys, SERIES[i % len(SERIES)], label=label, window=window)
            i += 1
    _style(ax, "Reward components")
    if i:
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)

    ax = axes[1, 0]
    _plot_line(ax, *_series(records, "loss"), SERIES[1], window=window)
    _style(ax, "Loss")

    ax = axes[1, 1]
    _plot_line(ax, *_series(records, "completion_length"), SERIES[2], window=window)
    xs, ys = _series(records, "completions/clipped_ratio")
    _style(ax, "Completion length (tokens)")
    return fig


def plot_grpo_extra(records: list[dict], title: str, window: int = 10):
    """Secondary panel: KL to reference, reward std, clipped-completion ratio, grad norm."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    fig.suptitle(title, color=INK, fontsize=13, x=0.01, ha="left")
    panels = [
        ("kl", "KL to reference policy", SERIES[6]),
        ("reward_std", "Reward std within group", SERIES[0]),
        ("completions/clipped_ratio", "Fraction of completions hitting the length cap", SERIES[7]),
        ("grad_norm", "Gradient norm", SERIES[3]),
    ]
    for ax, (key, label, color) in zip(axes.flat, panels):
        _plot_line(ax, *_series(records, key), color, window=window)
        _style(ax, label)
    return fig


def plot_sft(records: list[dict], title: str, window: int = 5):
    """SFT: loss and learning rate side by side (separate panels, one axis each)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    fig.suptitle(title, color=INK, fontsize=13, x=0.01, ha="left")
    _plot_line(axes[0], *_series(records, "loss"), SERIES[1], window=window)
    _style(axes[0], "Training loss (completion tokens only)")
    _plot_line(axes[1], *_series(records, "learning_rate"), SERIES[0])
    _style(axes[1], "Learning rate")
    return fig


def plot_comparison(runs: dict[str, list[dict]], key: str, title: str, window: int = 10):
    """Overlay one metric across runs (e.g. total reward for base vs shaped GRPO)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for i, (name, records) in enumerate(runs.items()):
        _plot_line(ax, *_series(records, key), SERIES[i % len(SERIES)], label=name, window=window)
    _style(ax, title)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2)
    return fig


def save(fig, out_base: Path, formats: Iterable[str] = ("png", "pdf")) -> list[Path]:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in formats:
        p = out_base.with_suffix(f".{ext}")
        fig.savefig(p, dpi=160 if ext == "png" else None, facecolor="white")
        paths.append(p)
    return paths


def detect_kind(records: list[dict]) -> str:
    return "grpo" if any("reward" in r for r in records) else "sft"


def plot_run(run_dir: Path, out_dir: Path, title: Optional[str] = None,
             window: int = 10) -> list[Path]:
    """Plot everything relevant for one run directory; returns written paths."""
    import matplotlib

    matplotlib.use("Agg")
    from aha.logging_utils import read_log

    records = read_log(run_dir)
    name = title or run_dir.name
    written = []
    if detect_kind(records) == "grpo":
        written += save(plot_grpo(records, f"{name}: GRPO training curves", window),
                        out_dir / f"{run_dir.name}_grpo")
        written += save(plot_grpo_extra(records, f"{name}: GRPO diagnostics", window),
                        out_dir / f"{run_dir.name}_grpo_extra")
    else:
        written += save(plot_sft(records, f"{name}: SFT training curves", max(1, window // 2)),
                        out_dir / f"{run_dir.name}_sft")
    return written
