#!/usr/bin/env python
"""Render training curves (loss, reward, reward components, completion length, KL ...) to
PNG + PDF for the report.

    # one or more run directories (each has train_log.jsonl / log_history.json)
    python scripts/plot_curves.py outputs/qwen3b_base outputs/qwen3b_sft --out report/img/curves

    # overlay the total reward of two GRPO runs
    python scripts/plot_curves.py outputs/qwen3b_base outputs/qwen3b_shaped --compare reward --out report/img/curves

Works on any machine with matplotlib; no GPU or training stack needed.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aha.logging_utils import read_log  # noqa: E402
from aha.plotting import plot_comparison, plot_run, save  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="run output directories")
    parser.add_argument("--out", default="report/img/curves", help="output directory for figures")
    parser.add_argument("--window", type=int, default=10, help="rolling-mean window (steps)")
    parser.add_argument("--compare", metavar="METRIC",
                        help="also overlay this logged metric across all given runs (e.g. reward, loss)")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    out = Path(args.out)
    for run in args.runs:
        for p in plot_run(Path(run), out, window=args.window):
            print(p)
    if args.compare:
        runs = {Path(r).name: read_log(Path(r)) for r in args.runs}
        fig = plot_comparison(runs, args.compare, f"{args.compare} across runs", args.window)
        for p in save(fig, out / f"compare_{args.compare.replace('/', '_')}"):
            print(p)


if __name__ == "__main__":
    main()
