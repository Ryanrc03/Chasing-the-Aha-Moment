# 本地 RTX 3060 训练：模型与数据选型

日期：2026-10-25。硬件：RTX 3060 12 GB（WDDM 已占 ~1.3 GB），WSL2 Ubuntu，默认分配 7.7 GB 内存，主机 16 GB。

## 结论（TL;DR）

| 项目 | 推荐 | 备选 |
|---|---|---|
| 策略模型 | **Qwen2.5-3B-Instruct**（`configs/qwen2.5-3b_3060.yaml`） | Llama-3.2-3B-Instruct（与论文同系列） |
| 训练数据 | **GSM8K train**（7.5k，与论文一致） | MATH-lighteval（更难，`data.train_dataset: math`） |
| 评测集 | GSM8K test + **MATH-500** + AIME 2024/2025 | — |
| 加载方式 | 4-bit + LoRA r=32，vLLM 共置，`gpu_memory_utilization=0.55` | 1.5B 模型可改 16-bit LoRA |
| 步数 | 500 步（3B 比 8B 快约 2 倍，可比论文多跑一倍） | — |

原来的 Llama-3.1-8B 配置在 3060 上**跑不起来**：4-bit 权重约 5.5 GB，加上共置的 vLLM KV cache 和训练激活约 13 GB，超过 12 GB。保留 `configs/llama3.1-8b_t4.yaml` 只是为了在 Colab/T4 上复现论文。

## 模型候选

GSM8K 数字来自各模型官方发布（零样本或 8-shot CoT，不同家族口径略有差异，只用于定性比较）。

| 模型 | 4-bit 权重 | 3060 可行性 | GSM8K（官方） | 评价 |
|---|---|---|---|---|
| Llama-3.1-8B-Instruct | ~5.5 GB | ✗ 需 ~13 GB | ~84 | 论文原配置；只能 T4/Colab |
| **Qwen2.5-3B-Instruct** | ~2.2 GB | ✓ 约 9–10 GB | ~86 | Unsloth 官方 GRPO notebook 用的就是它，最省心；数学强，格式学得快 |
| Llama-3.2-3B-Instruct | ~2.2 GB | ✓ 约 9–10 GB | ~77 | 和论文同一家族，做"8B vs 3B"对照最干净；起点低，RL 提升空间更大 |
| Qwen2.5-1.5B-Instruct | ~1.2 GB | ✓ 约 6 GB | ~73 | 最快；可以把 `num_generations` 提到 8–12、`load_in_4bit: false`。适合快速迭代奖励函数 |
| Qwen2.5-Math-1.5B（base） | ~1.2 GB | ✓ | — | SimpleRL-Zoo 等"小模型 aha 复现"工作的标准底座；起点格式很差，RL 后曲线最陡，最容易观察到"aha"式行为。但是 base 模型，需要更多步数 |
| Qwen3-1.7B / 4B | ~1.3 / 2.8 GB | ✓ | — | **不建议**：自带 thinking 模式，已经是推理模型，会混淆"RL 是否诱导出推理"这个研究问题 |

**为什么首选 Qwen2.5-3B-Instruct：** 显存稳、Unsloth/TRL 路径验证最充分、数学能力足以在 GSM8K 上拿到有意义的正确率奖励（GRPO 需要组内有对有错才有梯度；起点太低会长时间零奖励）。

**需要注意的天花板效应：** 它在 GSM8K 上起点已 ~86%，GRPO 的收益会主要体现在格式一致性和小幅正确率上（论文 8B 也只是 82.6 → 85.0）。如果想看到更大的 RL 效应，有两个方向：换更难的数据（MATH），或换更弱的底座（Llama-3.2-3B 或 Qwen2.5-Math-1.5B base）。

## 数据候选

### 训练

| 数据集 | 规模 | 提示长度 | 评价 |
|---|---|---|---|
| **openai/gsm8k** train | 7,473 | 短（<256 token） | 与论文一致；答案是整数，`int_reward` 和 `relaxed` 匹配都好用 |
| MATH-lighteval train（`math`） | 7,500 | 中等，建议 `max_prompt_length: 512` | 5 个难度等级，比 GSM8K 难一档；答案是 LaTeX 表达式，`int_reward` 基本失效，正确率奖励用 `relaxed` 字符串匹配，较粗 |
| Countdown（TinyZero） | 任意生成 | 短 | 经典"aha"复现任务，但和论文的数学 QA 设定不同，作为后续扩展 |

建议：先 GSM8K 跑通 base / shaped 两组对照（和论文可比），再考虑 MATH 或 GSM8K+MATH 混合。

### 评测

| 数据集 | 规模 | 用途 |
|---|---|---|
| GSM8K test | 1,319 | 分布内主指标 |
| **MATH-500** | 500 | 新增。介于 GSM8K 和 AIME 之间，3B 模型大约 40–60%，能看出区分度；AIME 对 3B 几乎必然是 0 |
| AIME 2024 + 2025 | 60 | 与论文一致，预期仍为 0，保留作格式一致性对照 |

`int_reward` 仅对整数答案有意义；MATH-500 的答案多为表达式，评测用 `relaxed` 比较会偏保守（字符串必须完全相同）。

## 3060 显存预算（Qwen2.5-3B，4-bit，seq 1024，6 个采样）

| 组成 | 估计 |
|---|---|
| vLLM（权重 + KV cache，`gpu_memory_utilization=0.55`） | ~6.6 GB（含 ~2.2 GB 权重） |
| 训练（LoRA 参数 + 8-bit AdamW + 梯度检查点激活） | ~3–4 GB |
| Windows WDDM 占用 | ~1.3 GB |
| 合计 | ~11–12 GB，接近上限 |

OOM 时按顺序调整：`train.num_generations: 4` → `model.gpu_memory_utilization: 0.5` → `model.max_seq_length: 768`。
`UNSLOTH_VLLM_STANDBY=1` 可以让 vLLM 在训练阶段释放显存，但需要 `vllm>=0.11.2`（见下）。

速度：见下面的冒烟实测，3B 在 3060 + WSL 上约 34 s/步（4 个采样）。

## 冒烟测试实测（2026-10-25，`configs/smoke.yaml`）

- Unsloth 自动把 `unsloth/Qwen2.5-3B-Instruct` 解析为 `unsloth/Qwen2.5-3B-Instruct-unsloth-bnb-4bit`，下载 2.3 GB，43 s 下完。
- vLLM 实际占用：权重 2.35 GB，KV cache 3.44 GB（`gpu_memory_utilization=0.55` 被 Unsloth 折算成 0.50，约 6 GB）。
- 2 步训练（4 个采样 / 步，完成长度 ≤768）用时 69 s，约 34 s/步，其中第 1 步含编译预热。
- **速度拆解（6 步基准，2026-10-25）**：采样只占每步约 9 s，4-bit 和 16-bit 一样快（decode 受显存带宽限制），4 条加到 8 条采样几乎不加时间。大头是训练那一半。
  把 `gradient_checkpointing` 从 `unsloth`（激活 offload 到 CPU）换成标准 `true`，每步从 26 s 降到 17 s（4 条采样）；6 条采样约 17–21 s/步，取决于采样长度。**500 步约 2.5–3 小时**，250 步约 1.2–1.5 小时。
  16-bit 加载权重 5.9 GB、采样峰值 8.35 GB，再叠训练会爆 12 GB，而且不比 4-bit 快，不采用。
- vLLM standby 模式（`UNSLOTH_VLLM_STANDBY=1`）在当前 Unsloth（2026.10.x）+ 固定的 vllm 0.10.2 上会被直接拒绝，脚本默认已关闭。如果显存吃紧，升级到 `vllm>=0.11.2` 再开。

## 环境注意事项

1. **WSL 内存**：默认 7.7 GB 偏小（vLLM 启动和数据加载会吃内存）。在 `C:\Users\<user>\.wslconfig` 中加：
   ```ini
   [wsl2]
   memory=12GB
   swap=16GB
   ```
   然后 `wsl --shutdown`。
2. **HF 缓存放在 Linux 文件系统**（如 `~/.cache/huggingface`），不要放 `/mnt/d/...`，跨文件系统 IO 慢几倍。
3. 安装：`bash scripts/setup_wsl.sh`，然后 `python scripts/check_env.py` 确认 torch 能看到 GPU。
4. 先跑 `configs/smoke.yaml`（2 步 + 16 条评测）验证完整链路，再跑正式实验。

## 建议的实验顺序

1. `smoke.yaml` 跑通。
2. 基线：`eval.py --config configs/qwen2.5-3b_3060.yaml --set eval.lora_path=null "eval.datasets=[gsm8k_test, math500]"`（未训练的 Instruct 模型）。
3. `qwen2.5-3b_3060.yaml`（base reward，500 步）→ 评测。
4. `qwen2.5-3b_3060_shaped.yaml`（+ 长度奖励）→ 评测。两组对照即可复现论文 Table 2 的结构。
5. 可选：`llama3.2-3b_3060.yaml` 做家族对照；或 `--set data.train_dataset=math train.max_prompt_length=512` 看更难数据下长度奖励是否更有用；或换 Qwen2.5-Math-1.5B base 观察更明显的"aha"曲线。

长度奖励的 `alpha=0.5, beta=0.1, l_max=512` 是按论文"取较小值、不盖过正确率信号"的描述设的（正确率奖励 2.0，长度项最大 0.5）。论文没有给出具体数值，这三个值是待调的超参数。
