"""CPU-only, reproducible analysis of the two data-night experiments.

Comparisons are paired by dataset row ID, never file order. Incomplete runs fail
validation before a final report can be written.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from statistics import mean

from aha.hacking import load_predictions, output_signals, training_signals
from aha.logging_utils import read_log
from aha.rewards import format_intact

CORRECTNESS = "rewards/correctness_reward_func/mean"
RUNS = ("qwen3b_fixed", "qwen3b_filtered", "qwen3b_filtered_gate")


def _index(rows: list[dict], expected_samples: int) -> dict[int, dict]:
    indexed = {}
    for row in rows:
        idx = row["idx"]
        if type(idx) is not int or idx in indexed:
            raise ValueError(f"invalid or duplicate prediction ID: {idx}")
        if type(row["correct"]) is not bool or row.get("gold") is None:
            raise ValueError(f"invalid correctness or missing gold at ID {idx}")
        indexed[idx] = row
    if set(indexed) != set(range(expected_samples)):
        raise ValueError(f"expected IDs 0..{expected_samples - 1}; got {len(indexed)} predictions")
    return indexed


def paired_comparison(reference: list[dict], candidate: list[dict], *,
                      expected_samples: int = 1319, bootstrap: int = 4000,
                      seed: int = 42) -> dict:
    if expected_samples < 1 or bootstrap < 1:
        raise ValueError("sample and bootstrap counts must be positive")
    ref, new = _index(reference, expected_samples), _index(candidate, expected_samples)
    for idx in ref:
        if ref[idx]["gold"] != new[idx]["gold"]:
            raise ValueError(f"gold mismatch at ID {idx}")
    deltas = [int(new[i]["correct"]) - int(ref[i]["correct"]) for i in sorted(ref)]
    wins, losses = deltas.count(1), deltas.count(-1)
    discordant = wins + losses
    p = min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(wins, losses) + 1))
            / (2 ** discordant)) if discordant else 1.0
    rng = random.Random(seed)
    draws = sorted(sum(rng.choices(deltas, k=len(deltas))) / len(deltas) for _ in range(bootstrap))

    def quantile(q):
        pos = (len(draws) - 1) * q
        lo, hi = math.floor(pos), math.ceil(pos)
        return draws[lo] + (draws[hi] - draws[lo]) * (pos - lo)

    return {"samples": len(deltas), "reference_accuracy": mean(r["correct"] for r in ref.values()),
            "candidate_accuracy": mean(r["correct"] for r in new.values()),
            "delta": mean(deltas), "ci95": [quantile(0.025), quantile(0.975)],
            "candidate_only": wins, "reference_only": losses, "mcnemar_p": p,
            "bootstrap": bootstrap, "seed": seed}


def group_signals(records: list[dict], *, weight: float = 2.0, window: int = 100) -> dict:
    rows = sorted((r for r in records if CORRECTNESS in r), key=lambda r: r["step"])
    if not rows or window < 1 or weight <= 0:
        raise ValueError("group signals require correctness logs, positive weight and window")
    if len({r["step"] for r in rows}) != len(rows):
        raise ValueError("duplicate training steps")

    def summarize(chunk):
        values = [r[CORRECTNESS] for r in chunk]
        if any(not math.isfinite(v) or v < -1e-6 or v > weight + 1e-6 for v in values):
            raise ValueError("correctness reward outside its expected range")
        right = mean(math.isclose(v, weight, abs_tol=1e-6) for v in values)
        wrong = mean(math.isclose(v, 0, abs_tol=1e-6) for v in values)
        zeros = [r["frac_reward_zero_std"] for r in chunk if "frac_reward_zero_std" in r]
        return {"steps": len(chunk), "all_right": right, "all_wrong": wrong,
                "mixed": 1 - right - wrong, "zero_std": mean(zeros) if len(zeros) == len(chunk) else None,
                "reward_correct_rate": mean(values) / weight,
                "completion_length": mean(r["completion_length"] for r in chunk)}

    out = summarize(rows)
    out["windows"] = []
    for start in range(1, rows[-1]["step"] + 1, window):
        chunk = [r for r in rows if start <= r["step"] < start + window]
        if chunk:
            out["windows"].append({"start": start, "end": start + window - 1, **summarize(chunk)})
    return out


def validate_run(root: Path, name: str, *, expected_samples: int = 1319,
                 expected_steps: int = 500) -> dict:
    import json
    import yaml

    out = root / "outputs" / name
    result = root / "results" / name / "gsm8k_test"
    if not (out / "lora" / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"{name}: final LoRA adapter missing")
    records = read_log(out)
    steps = [r["step"] for r in records if CORRECTNESS in r]
    if sorted(steps) != list(range(1, expected_steps + 1)):
        raise ValueError(f"{name}: expected {expected_steps} unique training steps, got {len(steps)}")
    cfg = yaml.safe_load((out / "config.yaml").read_text(encoding="utf-8"))
    train = cfg["train"]
    if (train["per_device_train_batch_size"], train["gradient_accumulation_steps"]) != (1, 1):
        raise ValueError(f"{name}: cannot infer per-group rates from aggregated multi-prompt logs")
    if train["max_steps"] != expected_steps or train["num_generations"] != 6:
        raise ValueError(f"{name}: unexpected training budget or group size")
    preds = load_predictions(result / "predictions.jsonl")
    _index(preds, expected_samples)
    metrics = json.loads((result / "metrics.json").read_text(encoding="utf-8"))
    if metrics["samples"] != expected_samples or not math.isclose(
            metrics["accuracy"], mean(p["correct"] for p in preds), abs_tol=1e-12):
        raise ValueError(f"{name}: metrics disagree with predictions")
    return {"predictions": preds, "records": records, "metrics": metrics, "config": cfg}


def analyze(root: Path, *, expected_samples: int = 1319, expected_steps: int = 500,
            bootstrap: int = 4000) -> tuple[str, dict]:
    """Validate every input before deriving results; returns Markdown and JSON data."""
    from collections import Counter

    runs = {name: validate_run(root, name, expected_samples=expected_samples,
                              expected_steps=expected_steps) for name in RUNS}
    presample = load_predictions(root / "outputs/presample/gsm8k_instruct_g6.jsonl")
    ids = [p["idx"] for p in presample]
    if len(ids) != len(set(ids)) or any(type(i) is not int or not 0 <= i < 7473 for i in ids):
        raise ValueError("invalid or duplicate pre-sample training IDs")
    if any(p["n"] != 6 or type(p["passes"]) is not int or not 0 <= p["passes"] <= 6 for p in presample):
        raise ValueError("pre-sample must contain six samples per problem and valid pass counts")
    hist = Counter(p["passes"] for p in presample)
    kept = sum(hist[i] for i in range(1, 6))
    if kept < 50:
        raise ValueError("fewer than 50 training problems survived the filter")
    data = {"status": "complete", "expected_steps": expected_steps, "expected_samples": expected_samples,
            "presample": {"total": len(presample), "kept": kept,
                          "histogram": {str(i): hist[i] for i in range(7)}}, "runs": {}, "comparisons": {}}
    for name, run in runs.items():
        cfg = run["config"]
        records = [r for r in run["records"] if CORRECTNESS in r]
        preds = run["predictions"]
        min_words = cfg.get("reward", {}).get("min_reasoning_words", 30)
        data["runs"][name] = {
            "groups": group_signals(records, weight=cfg.get("reward", {}).get("correctness_weight", 2.0)),
            "training": training_signals(records),
            "outputs": output_signals(preds, cfg.get("eval", {}).get("max_tokens", 512)),
            "intact_format_rate": mean(format_intact(p["output"], min_words) for p in preds),
            "metrics": run["metrics"],
        }
    for label, ref, new in (("A_vs_fixed", RUNS[0], RUNS[1]), ("B_vs_A", RUNS[1], RUNS[2])):
        data["comparisons"][label] = paired_comparison(
            runs[ref]["predictions"], runs[new]["predictions"],
            expected_samples=expected_samples, bootstrap=bootstrap)

    lines = ["# 数据之夜结果（2026-10-28）", "",
             f"状态：**已完成输入核验与统计分析**。三组均有 {expected_steps} 个训练步、最终 LoRA，"
             f"以及相同 {expected_samples} 道 GSM8K test 题的逐题预测。", "",
             "## 配对比较", "",
             "| 比较 | 参考准确率 | 新配置准确率 | 差值（百分点） | 95% 配对区间（百分点） | McNemar p | 仅新配置对 / 仅参考对 |",
             "|---|---|---|---|---|---|---|"]
    for label, comp in data["comparisons"].items():
        lo, hi = comp["ci95"]
        lines.append(f"| {label} | {comp['reference_accuracy']:.2%} | {comp['candidate_accuracy']:.2%} | "
                     f"{100 * comp['delta']:+.2f} | [{100 * lo:+.2f}, {100 * hi:+.2f}] | "
                     f"{comp['mcnemar_p']:.5f} | {comp['candidate_only']} / {comp['reference_only']} |")
    lines += ["", f"按题号和 gold 对齐；双侧精确 McNemar，{bootstrap:,} 次配对 bootstrap（seed=42，百分位区间）。"
              "p 为未经多重比较校正的值；仅衡量评测题目抽样不确定性，不涵盖训练噪声。", ""]
    for label, comp in data["comparisons"].items():
        if comp["mcnemar_p"] < 0.05 and comp["ci95"][0] > 0:
            verdict = "本次配对评测显著提升；需要换 seed 复跑，暂不替换基线"
        elif comp["mcnemar_p"] < 0.05 and comp["ci95"][1] < 0:
            verdict = "本次配对评测显著下降；保留 fixed 基线并复核失败样本"
        else:
            verdict = "尚无显著提升证据；保留 fixed 基线"
        lines.append(f"- **{label}**：{verdict}。")
    lines += ["", "## 预采样与训练信号", "",
              f"预采样 {len(presample)} 题，保留 {kept} 题（{kept / len(presample):.1%}，通过 1–5/6 次）。"
              f"每轮 {expected_steps} 步，相当于 {expected_steps / kept:.2f} 个保留题池。", "",
              "| 答对次数 | 0 | 1 | 2 | 3 | 4 | 5 | 6 |", "|---|---|---|---|---|---|---|---|",
              "| 题数 | " + " | ".join(str(hist[i]) for i in range(7)) + " |", "",
              "| 运行 | 全获正确分组 | 全未获正确分组 | 混合组 | 总 reward 零 std | 平均长度 |",
              "|---|---|---|---|---|---|"]

    def pct(value):
        return "缺失" if value is None else f"{value:.1%}"

    for name, sig in data["runs"].items():
        g = sig["groups"]
        lines.append(f"| {name} | {pct(g['all_right'])} | {pct(g['all_wrong'])} | {pct(g['mixed'])} | "
                     f"{pct(g['zero_std'])} | {g['completion_length']:.0f} |")
    fixed_rate = data["runs"][RUNS[0]]["groups"]["all_right"]
    a_rate = data["runs"][RUNS[1]]["groups"]["all_right"]
    lines += ["", f"A 的全对组相对 fixed 改变 {100 * (a_rate - fixed_rate):+.1f} 个百分点。"
              "筛题是否增加有效组，要同时检查混合组比例和分阶段回升情况。", "",
              "B 的 correctness 受格式门控；这里的全获正确分组表示六个回答都正确且格式完整，"
              "全未获正确分组也可能包含答案正确但格式不合格的回答。不能直接当作原始答对组比例。"
              "总 reward 零 std 表示奖励 advantage 为零，KL 正则仍可能贡献梯度。", "",
              "| 步数 | 运行 | 全获正确分组 | 全未获正确分组 | 混合组 | 零 std |",
              "|---|---|---|---|---|---|"]
    for name, sig in data["runs"].items():
        for g in sig["groups"]["windows"]:
            lines.append(f"| {g['start']}–{g['end']} | {name} | {pct(g['all_right'])} | "
                         f"{pct(g['all_wrong'])} | {pct(g['mixed'])} | {pct(g['zero_std'])} |")
    lines += ["", "## 健康检查", "",
              "| 运行 | soft 格式 | 完整格式且推理 ≥30词 | 重复 | 截断 | 平均 token |",
              "|---|---|---|---|---|---|"]
    for name, sig in data["runs"].items():
        o = sig["outputs"]
        lines.append(f"| {name} | {pct(o['rates']['soft_format'])} | {pct(sig['intact_format_rate'])} | "
                     f"{pct(o['rates']['repetitive'])} | {pct(o['rates']['truncated'])} | {o['length']['mean']:.0f} |")
    for name, sig in data["runs"].items():
        flags = sig["training"]["flags"] + sig["outputs"]["flags"]
        lines += ["", f"**{name}**：" + ("；".join(flags) if flags else "现有诊断规则未发现红旗") + "。"]
    a_format, b_format = (data["runs"][r]["intact_format_rate"] for r in RUNS[1:])
    lines += ["", f"B 相对 A 的完整格式率改变 {100 * (b_format - a_format):+.2f} 个百分点。"
              "门控格式率单列检查；现有 analyze_hacking 的格式奖励涨幅规则针对 graded 分项，不能独立判定 gate 健康。", "",
              "## 解释与下一步", "",
              "- 单 seed；统计显著也不证明跨训练随机种子可复现。两项主要比较均报告未校正 p，避免事后挑最好结果。",
              "- A 相对 fixed 同时改变题目池、题量和重复率。缺少同规模随机子集对照，不能把差异完全归因于难度筛选。",
              "- B 相对 A 同时把格式奖励改为常数，并给 correctness 增加格式门槛，不能单独归因于去掉格式噪声。",
              "- 先据本次结果决定是否复跑和加入同规模随机子集对照；MATH 混入、SFT 热启动、动态筛题、pass@8 仍是候选，本次未启动。", "",
              "## 复现与产物", "",
              "`python scripts/analyze_data_night.py`（CPU，无需训练栈）；统计明细 `results/data_night_analysis.json`，"
              "健康诊断 `results/data_night_hacking.md`，执行状态 `results/data_night_analysis_status.json`。", ""]
    return "\n".join(lines), data
