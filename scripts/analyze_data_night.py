#!/usr/bin/env python
"""Validate data-night runs, perform paired tests and write the final report (CPU only)."""

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aha.data_night import RUNS, analyze  # noqa: E402
from aha.hacking import render_markdown  # noqa: E402


def write_atomic(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)


def figures(data, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 3), constrained_layout=True)
    for y, (label, comp) in enumerate(data["comparisons"].items()):
        delta = 100 * comp["delta"]
        lo, hi = [100 * v for v in comp["ci95"]]
        ax.plot([lo, hi], [y, y], color="#2a78d6", linewidth=2)
        ax.plot(delta, y, "o", color="#2a78d6")
        ax.annotate(f"{delta:+.2f} pp; p={comp['mcnemar_p']:.4f}", (hi, y),
                    xytext=(8, 8), textcoords="offset points", fontsize=8)
    ax.axvline(0, color="#999999", linewidth=1, linestyle="--")
    ax.set_yticks(range(2), list(data["comparisons"]))
    ax.set_xlabel("Accuracy change (percentage points), paired 95% bootstrap CI")
    ax.set_title("Data night: paired GSM8K test comparisons", loc="left")
    ax.margins(x=0.3, y=0.6)
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"data_night_delta_ci.{suffix}", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    for name, run in data["runs"].items():
        windows = run["groups"]["windows"]
        xs = [(w["start"] + w["end"]) / 2 for w in windows]
        axes[0].plot(xs, [100 * w["all_right"] for w in windows], "o-", label=name)
        axes[1].plot(xs, [100 * w["zero_std"] if w["zero_std"] is not None else float("nan")
                          for w in windows], "o-", label=name)
    for ax, title in zip(axes, ("Full correctness reward groups*", "Zero total reward std groups")):
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlabel("Training step (100-step windows)")
        ax.set_ylabel("Groups (%)")
        ax.set_ylim(-3, 103)
        ax.grid(alpha=0.2)
    axes[1].legend(fontsize=7)
    fig.supxlabel("*B requires both correct answers and intact format; A/fixed require correct answers.", fontsize=8)
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"data_night_group_signal.{suffix}", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    status_path = root / "results/data_night_analysis_status.json"
    done = root / "logs/data_night_analysis.DONE"
    # A rerun must not leave an earlier success marker behind if validation fails.
    done.unlink(missing_ok=True)
    stamp = lambda: datetime.now(ZoneInfo("America/Chicago")).isoformat()
    try:
        report, data = analyze(root)
        inputs = [root / "outputs/presample/gsm8k_instruct_g6.jsonl"]
        for run in RUNS:
            inputs += [root / f"results/{run}/gsm8k_test/{name}" for name in ("metrics.json", "predictions.jsonl")]
            inputs += [root / f"outputs/{run}/config.yaml"]
            history = root / f"outputs/{run}/log_history.json"
            inputs += [history if history.exists() else root / f"outputs/{run}/train_log.jsonl"]
        data["input_sha256"] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
        data["generated_at"] = stamp()
        figures(data, root / "report/img/curves/data_night")
        report += ("\n![配对准确率差值](../report/img/curves/data_night/data_night_delta_ci.png)\n\n"
                   "![训练分组信号](../report/img/curves/data_night/data_night_group_signal.png)\n")
        diagnostics = "\n".join(render_markdown(name, run["training"], {"gsm8k_test": run["outputs"]})
                                for name, run in data["runs"].items())
        write_atomic(root / "results/data_night_analysis.json", json.dumps(data, indent=2, ensure_ascii=False))
        write_atomic(root / "results/data_night_hacking.md", diagnostics)
        write_atomic(root / "docs/results_2026-10-28_data_night.md", report)
        write_atomic(status_path, json.dumps({"status": "complete", "at": stamp()}, indent=2))
        write_atomic(done, stamp() + "\n")
    except Exception as exc:
        write_atomic(status_path, json.dumps({"status": "failed", "at": stamp(),
                                             "error": str(exc)}, indent=2, ensure_ascii=False))
        raise
    print(root / "docs/results_2026-10-28_data_night.md")


if __name__ == "__main__":
    main()
