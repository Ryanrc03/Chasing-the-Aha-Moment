# 交接（2026-10-28 00:40）

## 后续接管记录（2026-10-28 00:50，America/Chicago）

用户选择：**接好自动收尾流程，本轮交付进度与文件**，本轮不等夜跑结束。

- 原有队列 PID `2948` 正常预采样，未重启训练、未追加候选实验。
- 已启动独立收尾进程：WSL PID `3456`，Windows 隐藏 `wsl.exe` PID `31948`。
  它等待原队列退出，核对 `logs/data_night.DONE` 后执行 `scripts/analyze_data_night.py`。
- Windows 上用 `Start-Process wsl.exe -WindowStyle Hidden` 保持独立宿主；本次直接从短命 WSL shell
  `nohup ... & disown` 启动的尝试未存活，已改用独立宿主并核验进程和 waiting 状态。
- 统计脚本严格核验 fixed/A/B 各 500 步、最终 LoRA、1,319 道逐题预测及 metrics，一项缺失即失败。
  然后自动生成 A 对 fixed、B 对 A 的精确 McNemar、4,000 次配对 bootstrap、分阶段组信号、健康诊断及 PNG/PDF。
- 报告位置：`docs/results_2026-10-28_data_night.md`。目前明确标为等待，成功后原子替换为最终统计报告。
- **实时状态**：`results/data_night_analysis_status.json`；收尾日志 `logs/data_night_finish.log`，
  错误日志 `logs/data_night_finish.err.log`；**统计全部成功**的标记为 `logs/data_night_analysis.DONE`。
  原队列 DONE 仅表示队列走完，不表示所有实验都成功。
- 新增 `src/aha/data_night.py`、`scripts/analyze_data_night.py`、`scripts/finish_data_night.sh`、
  `tests/test_data_night_analysis.py`。Windows 全套 **107 passed, 2 skipped**；WSL 全套 **109 passed**。
  其中包含模拟完整 500 步 / 1,319 题的报告、图表、状态和失败路径验证。独立代码审查未发现需修复项。
- 用真实旧结果复核：fixed 全对组 45.4%；lr2e5 对 fixed 的分歧题 95/81、McNemar p=0.32714，
  与旧报告一致。A/B 的真实结果尚未产生，不能提前下结论。

下文是 00:40 的原始交接快照；当前状态以以上记录及状态 JSON 为准。

## 一句话现状

Qwen2.5-3B-Instruct 在 RTX 3060 上用 GRPO 训 GSM8K。最好的稳定配置是 `configs/qwen2.5-3b_3060_fixed.yaml`，GSM8K test **77.3%**
（未训练 62.1%，SFT 67.6%）。优化器旋钮（学习率、采样数、KL）已证明推不动这个平台，**今晚在试数据侧**：难度筛题和门控格式分，
预计 06:45 跑完。

## 正在跑的（2026-10-28 00:31 开始）

| 项 | 位置 |
| --- | --- |
| 计划 | `docs/plan_2026-10-28_data_night.md` |
| 启动命令 | `nohup bash scripts/run_data_night.sh > logs/data_night.log 2>&1 &`（WSL 里，已 detach） |
| 进度 | `logs/data_night.log`；完成标记 `logs/data_night.DONE` |
| 结果 | `results/data_night_summary.md`，图在 `report/img/curves/data_night/` |
| 内容 | 预采样 50 分钟 → run A `qwen3b_filtered`（只训 1–5/6 答对的题）→ run B `qwen3b_filtered_gate`（A + 门控格式分），各跑 GSM8K test 全量 |

跑完后要做的：A 对 fixed、B 对 A 做配对检验（McNemar + 配对 bootstrap，写法见 `docs/results_2026-10-27_overnight_queue.md`），
看训练日志里"全对组"比例是否从 45% 降下来，写 `docs/results_2026-10-28_data_night.md`。

## 结果一览

| 模型 | GSM8K test | 说明 |
| --- | --- | --- |
| Instruct（未训练） | 62.1% | |
| SFT 1 epoch | 67.6% | 回答很短，AIME 上 43% 循环重复 |
| GRPO 第一轮 500 步 | 56.9% | reward hacking：格式分涨、正确率掉、回答缩到 90 token |
| GRPO 第一轮 200 步 checkpoint | 74.5% | 塌缩之前的峰值 |
| **GRPO fixed 500 步** | **77.3%** | β 0.04、Dr. GRPO、格式分减半并门控在 ≥ 30 词推理 |
| fixed + lr 2e-5 | 78.3% | +1.1，不显著（p = 0.33） |
| fixed + 8 采样 | 77.8% | +0.5，不显著 |
| fixed − KL | 77.2% | 持平；不塌缩但漂得最远 |
| fixed − std 缩放 | 74.9% | −2.4，显著（p = 0.014），原因不明，单 seed |

AIME 2024/2025 所有模型都接近 0。详细分析：`docs/results_2026-10-26_grpo_vs_sft.md`（reward hacking、修法、性能瓶颈）、
`docs/results_2026-10-27_overnight_queue.md`（四组单变量实验）。

## 已经确认的结论

- reward hacking 的根因：格式分和 correctness 一样重、β = 0 没有 KL、strict 正则从不触发。门控格式分是防塌缩的主力，KL 管漂移速度和多样性。
- 一半训练步是全对组（45%–54%），这些步的 advantage 只来自格式分噪声。这是今晚做数据的理由。
- 150–210 步的准确率低谷是那段训练题更难，所有运行都有，不是某个配置的问题。
- TRL 默认的 `dapo` 损失**没有**偏向短回答（早期文档写错过，已更正）；β = 0 时日志里的 KL 是"没算"不是"没漂"。

## 环境和怎么跑

- WSL2 Ubuntu，venv `~/.venvs/aha`（`scripts/setup_wsl.sh` 建）。版本：vllm 0.10.2、transformers 4.56.2、trl 0.22.2、unsloth 2026.10.8。
- 训练：`python scripts/train.py --config configs/<name>.yaml [--set key=value ...]`，500 步约 1.5–2 小时。
- 评测：`python scripts/eval.py --config ... --set eval.lora_path=outputs/<run>/lora`；`--no-lora` 评未训练模型。GSM8K 全量约 33 分钟。
- 其他：`scripts/run_queue.sh`（多配置串行 + 扫描 + 汇总）、`scripts/presample.py`（难度预采样）、`scripts/analyze_hacking.py`（hacking 诊断）、
  `scripts/plot_curves.py`（曲线）、`scripts/compare_results.py`（对照表）、`scripts/profile_step.py`（GPU profile）。
- 测试：`pytest`（Windows 上也能跑，不需要 GPU），目前 90 个全过。

## 坑

- 从 Windows 调 WSL：`wsl -d Ubuntu -- bash -lc '...'` 里的 `$VAR` 会被外层 Git Bash 展开；裸的 `/mnt/c/...` 参数会被 MSYS 改写。复杂命令写成 `.sh` 再调用。
- 不要在 `run_queue.sh` 运行时改它（bash 边执行边读文件）。
- Windows 上的 `rec` Python 进程有时会占 GPU；`run_data_night.sh` 会先等 GPU 空闲。
- vLLM 评测偶尔会卡死在某个 batch，所有评测都套了 `timeout`。
- 12 GB 显存：8 采样峰值约 10.1 GB；8B 模型放不下。

## 分支和推送状态

| 仓库 | 分支 | 状态 |
| --- | --- | --- |
| Chasing-the-Aha-Moment（本仓库） | `claude/project-refactor-model-selection-5f49e3` | 本地领先 origin 3 个 commit：kernel 计划文档（`523f954`，另一个 session 写的）、数据之夜代码和计划（`9f5ca7c`）、本文档；未推送 |
| Chasing-the-Aha-Moment | `claude/gpu-kernels`（worktree `.claude/worktrees/gpu-kernels`） | kernel 优化，结果见那边的 `docs/results_2026-10-27_kernels.md`；未推送 |
| RLearning（`D:\RLearing`） | `claude/grpo-reward-hacking-kl` | KL 散度和本项目复盘两页笔记；已推送 |

main 在所有仓库都没动过。

## 接下来的候选（按预期收益）

1. 看今晚 A、B 的结果，决定数据侧方向是否有效。
2. 混入 MATH 整数答案子集（4,769 题，现有匹配器能正确判分；其余答案要 `math-verify`）。
3. 用 fixed 模型自己答对的长推理做 SFT 热启动，再在另一半题上 GRPO（不要用金标 SFT：25% 的回答达不到 30 词门槛）。
4. 训练中动态筛题，而不是只在开头筛一次。
5. 用 pass@8 量天花板；复跑 seed 量训练噪声。
