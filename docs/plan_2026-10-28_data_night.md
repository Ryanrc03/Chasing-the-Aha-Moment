# 数据之夜：难度筛选 + 门控格式分（2026-10-28，00:31 – 约 06:45）

## 为什么做这个

夜跑（`docs/results_2026-10-27_overnight_queue.md`）说明，学习率、采样数、KL 都推不动 77%–78% 的平台。训练日志里：

| 运行 | 全对的组 | 全错的组 | 有对有错 |
| --- | --- | --- | --- |
| fixed | 45% | 5% | 50% |
| lr2e5 | 48% | 4% | 48% |
| nokl | 54% | 4% | 41% |

一半的训练步里 correctness 没有任何区分度。这些步的 advantage 全部来自格式分的细微差别，而格式合规率已经约 98%：一半的更新在推格式噪声。
今晚只动数据和奖励，不动优化器，看能不能把这一半无效步变成有效步。

## 今晚做什么

| 顺序 | 内容 | 和谁比 | 回答的问题 |
| --- | --- | --- | --- |
| 0 | 预采样：未训练模型对 GSM8K 训练题各答 6 次，记录答对几次（50 分钟上限，按固定随机顺序，约 900–1,200 题） | — | 训练集里多少题是"有时对"？ |
| A | `qwen2.5-3b_3060_fixed_filtered.yaml`：fixed，只在 1–5/6 答对的题上训练 | fixed 77.3% | 只换数据有没有用？ |
| B | `qwen2.5-3b_3060_fixed_filtered_gate.yaml`：A + 门控格式分 | A | 去掉格式噪声有没有额外收益？ |

**门控格式分**（`reward.format_mode: gate`）：格式完整（soft 结构、`</answer>` 闭合、推理 ≥ 30 词）时给固定 0.25；correctness 只在格式完整时才算 2.0。
这样一组全对、全都格式完整的回答奖励完全相同，advantage 正好是 0，不再被格式细节推着走。

每轮之后：最终 LoRA 在 GSM8K test 全量 1,319 题上评测（为了在 8 点前跑完，今晚不扫 checkpoint），汇总写到 `results/data_night_summary.md`，
图在 `report/img/curves/data_night/`。

冒烟（16 题 + 3 步）已通过：预采样 3.4 s/题，16 题里有对有错 56%、全对 38%、全错 6%，和训练日志一致。

## 怎么读结果

- **主指标**：GSM8K test 全量，同题配对比较（McNemar + 配对 bootstrap）。A 对 fixed、B 对 A。
- **训练信号**：A、B 训练日志里"全对组"的比例应该从 45% 降下来；这是筛题有没有生效的直接证据。
- **健康检查**：`analyze_hacking.py` 不能有红旗；B 的格式合规率不能掉（门控下格式坏了 correctness 也是 0，应该会更稳）。
- **注意**：筛题用的是未训练模型的通过率。训练越往后模型越强，原本"有时对"的题会变成"总是对"，全对组比例会随步数回升，这是预期内的。
  单 seed，±1 到 2 个点以内不下结论。

## 今晚不做、之后做

- MATH 整数答案子集混入（4,769 题，现有匹配器能正确判分）。
- 用 fixed 模型自己答对的长推理做 SFT 热启动，再在另一半题上做 GRPO。
- 训练中动态筛题（DAPO 的 dynamic sampling），而不是只在开头筛一次。
- 用 pass@8 量天花板。

命令：`nohup bash scripts/run_data_night.sh > logs/data_night.log 2>&1 &`

## 自动收尾（00:50 接管补充）

现有队列继续运行；`scripts/finish_data_night.sh 2948` 已由 Windows 隐藏的独立 WSL 宿主启动。
等待队列结束后运行 CPU 分析，不重复训练，也不修改正在执行的 `run_data_night.sh` 或 `run_queue.sh`。

`scripts/analyze_data_night.py` 核验三组完整 500 步、最终适配器和相同 1,319 题后，按 idx/gold 配对，
计算双侧精确 McNemar 与固定 seed=42 的 4,000 次配对 bootstrap，并生成结果页、统计 JSON、健康诊断和图。
每 100 步统计组信号；明确区分 B 的格式门控 correctness 与 A/fixed 的原始 correctness。

输出报告为 `docs/results_2026-10-28_data_night.md`；状态见 `results/data_night_analysis_status.json`，
只有 `logs/data_night_analysis.DONE` 表示收尾全部成功。缺失训练/评测结果会记录 failed 并保留等待页，
不会把缺项统计为零或声称实验成功。

解释时保留单 seed 限制；A 缺少同规模随机子集对照，B 同时改变格式奖励和 correctness 门槛。
主比较仍只有 A 对 fixed、B 对 A，报告未校正 p 值，显著结果需复跑。
