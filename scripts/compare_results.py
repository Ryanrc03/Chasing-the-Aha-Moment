#!/usr/bin/env python
"""Tabulate results/<run>/<dataset>/metrics.json files as a Markdown table.

    python scripts/compare_results.py qwen3b_instruct qwen3b_sft qwen3b_base
    python scripts/compare_results.py --datasets gsm8k_test aime2024 -- qwen3b_instruct qwen3b_sft
"""

import argparse
import json
from pathlib import Path

METRICS = [
    ("accuracy", "Accuracy"),
    ("xml_parse_rate", "XML parse rate"),
    ("soft_format_rate", "Format (soft)"),
    ("avg_completion_tokens", "Avg completion tokens"),
    ("avg_latency_s", "Avg latency (s/sample)"),
]


def load(results_dir: Path, run: str, dataset: str) -> dict | None:
    path = results_dir / run / dataset / "metrics.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}" if value < 10 else f"{value:.1f}"
    return str(value)


def table(results_dir: Path, runs: list[str], dataset: str) -> str:
    rows = {run: load(results_dir, run, dataset) for run in runs}
    if not any(rows.values()):
        return f"_{dataset}: no results yet_\n"
    n = next(r["samples"] for r in rows.values() if r)
    out = [f"**{dataset}** ({n} samples)\n", "| Metric | " + " | ".join(runs) + " |",
           "|---|" + "---|" * len(runs)]
    for key, label in METRICS:
        out.append(f"| {label} | " + " | ".join(fmt(rows[r][key]) if rows[r] else "-" for r in runs) + " |")
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="run names (sub-directories of results/)")
    parser.add_argument("--datasets", nargs="*", default=["gsm8k_test", "aime2024", "aime2025"])
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out", help="also write the tables to this Markdown file")
    args = parser.parse_args()

    text = "\n".join(table(Path(args.results_dir), args.runs, d) for d in args.datasets)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
