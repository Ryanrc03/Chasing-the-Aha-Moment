#!/usr/bin/env python
"""Reward-hacking diagnostics for one or more runs.

    python scripts/analyze_hacking.py qwen3b_base qwen3b_sft --out results/hacking.md

For each run name it reads outputs/<run>/train_log.jsonl (if present) and every
results/<run>/<dataset>/predictions.jsonl, then prints a Markdown report with
first-vs-last reward-component trends and output-level exploit rates.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aha.hacking import load_predictions, output_signals, render_markdown, training_signals  # noqa: E402
from aha.logging_utils import read_log  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--max-tokens", type=int, default=512, help="eval max_tokens (for hit_cap)")
    parser.add_argument("--out", help="write the Markdown report here too")
    args = parser.parse_args()

    sections = []
    for run in args.runs:
        train = None
        run_dir = Path(args.outputs_dir) / run
        try:
            train = training_signals(read_log(run_dir))
        except FileNotFoundError:
            pass
        outputs = {}
        for pred in sorted((Path(args.results_dir) / run).glob("*/predictions.jsonl")):
            outputs[pred.parent.name] = output_signals(load_predictions(pred), args.max_tokens)
        sections.append(render_markdown(run, train, outputs))
    text = "\n".join(sections)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
