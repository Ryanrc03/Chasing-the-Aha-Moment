# GPU Kernel Optimisation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用独立 microbenchmark 量化 3060 上 GRPO 每一步的主要 kernel，并用一个 Triton NF4 GEMM 把 vLLM decode 的 4-bit 矩阵乘从 30% 带宽利用率提到 ≥ 60%；同时给 lm_head/loss 融合、逐元素 kernel 归因、fp16 累加 GEMM 三项各出一份带数字的结论。

**Architecture:** 新包 `src/aha/kernels/`（roofline 常数、NF4 参考实现、Triton kernel、bnb 补丁），新目录 `bench/`（计时框架和各 benchmark），报告写到 `docs/kernels/`（`results/` 在 .gitignore 里）。所有 GPU 代码在隔离的 WSL venv `~/.venvs/aha-kern` 和 git worktree `.claude/worktrees/gpu-kernels` 里运行；训练配方不改。

**Tech Stack:** torch 2.8.0+cu128, triton 3.4.0, bitsandbytes 0.50.2, vllm 0.10.2, unsloth 2026.10.8, trl 0.22.2, pytest。

**Spec:** `docs/superpowers/specs/2026-10-27-gpu-kernel-optimisation-design.md`

---

## 运行约定

所有 GPU 命令通过 WSL 里的包装脚本执行（Task 1 创建）。从 Windows 侧的 Git Bash 调用：

```bash
wsl -d Ubuntu -- bash /mnt/d/sideProjects/Chasing-the-Aha-Moment/.claude/worktrees/gpu-kernels/scripts/wsl_env.sh <command...>
```

下文把这一长串简写为 `$KERN <command...>`；执行时要写全。不要把 `$VAR` 放进 `wsl -- bash -lc '...'` 的引号里（外层 Git Bash 会先展开），包装脚本就是为了绕开这个问题。

GPU 任务前先确认显卡空闲（Windows 侧的 miniconda `rec` 进程会占 4 GB）：

```bash
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
```

期望 `memory.used` < 2500 MiB 且利用率 < 20%。`bench/harness.py` 的 `assert_gpu_free()` 会再查一次，不满足就拒绝运行。

Git 操作在 worktree 目录 `D:\sideProjects\Chasing-the-Aha-Moment\.claude\worktrees\gpu-kernels` 里做；分支 `claude/gpu-kernels`。共享 stash 栈，不要用裸 `git stash`。

## 文件结构

| 路径 | 职责 |
|---|---|
| `scripts/wsl_env.sh` | 进入隔离 venv、设置 Triton 缓存目录、cd 到 worktree 后执行命令 |
| `src/aha/kernels/__init__.py` | 空 |
| `src/aha/kernels/roofline.py` | GA106 硬件上限、bytes/FLOPs 计算、利用率（无 torch 依赖，CPU 可测） |
| `src/aha/kernels/nf4.py` | NF4 码表、packed 解包、嵌套 absmax 解析、纯 torch 参考反量化 |
| `src/aha/kernels/nf4_gemm.py` | Triton 小 M NF4 GEMM、`nf4_gemm()` 包装、`patch_bitsandbytes()` |
| `src/aha/kernels/fp16acc_gemm.py` | Triton fp16 累加 GEMM（P3，只做 bench） |
| `src/aha/profiling.py` | 修正峰值常数；新增 `attribute_kernels()`（P4） |
| `bench/harness.py` | GPU 空闲检查、CUDA event 计时、L2 冲刷、markdown 表 |
| `bench/bench_gemm_4bit.py` | bnb / dequant+cuBLAS / bf16 / Triton 的 4-bit GEMM 对照 |
| `bench/bench_cublas.py` | 训练形状的 bf16 cuBLAS GEMM 基线 |
| `bench/bench_fp16acc_gemm.py` | P3 的对照与数值评估 |
| `scripts/bench_decode.py` | vLLM decode 端到端 ms/步 + 贪心输出一致性 |
| `scripts/profile_step.py` | 加 `--stack`，输出归因表 |
| `src/aha/config.py`、`src/aha/train.py`、`src/aha/logging_utils.py` | `use_liger_loss` 字段透传；峰值显存写进 jsonl |
| `tests/test_kernels_roofline.py`、`tests/test_nf4_reference.py`、`tests/test_nf4_gemm_gpu.py`、`tests/test_bench_harness.py`、`tests/test_profiling.py`、`tests/test_config.py` | 测试 |
| `docs/kernels/*.md` | 每个 benchmark 一份报告 |

---

### Task 1: 环境隔离（worktree + venv 复制 + 包装脚本）

**Files:**
- Create: `scripts/wsl_env.sh`
- Create: `docs/kernels/README.md`

- [ ] **Step 1: 建分支和 worktree**

```bash
cd /d/sideProjects/Chasing-the-Aha-Moment/.claude/worktrees/project-refactor-model-selection-5f49e3
git worktree add -b claude/gpu-kernels ../gpu-kernels claude/project-refactor-model-selection-5f49e3
cd ../gpu-kernels && git log --oneline -1
```

期望：新目录 `.claude/worktrees/gpu-kernels`，HEAD 与原分支相同。

- [ ] **Step 2: 复制 venv（不重新下载依赖）**

```bash
wsl -d Ubuntu -- bash -lc 'cp -a ~/.venvs/aha ~/.venvs/aha-kern && grep -rlI "/home/rc/.venvs/aha/" ~/.venvs/aha-kern/bin | xargs sed -i "s#/home/rc/.venvs/aha/#/home/rc/.venvs/aha-kern/#g" && mkdir -p ~/.triton-kern && echo copied'
```

`grep -I` 跳过二进制和符号链接，只改 shebang 和 `activate` 里的路径。期望输出 `copied`。

- [ ] **Step 3: 写包装脚本**

`scripts/wsl_env.sh`:

```bash
#!/usr/bin/env bash
# Run a command inside the isolated kernel-work environment (WSL side).
#   wsl -d Ubuntu -- bash /mnt/d/.../gpu-kernels/scripts/wsl_env.sh python -m pytest tests -v
set -euo pipefail
source ~/.venvs/aha-kern/bin/activate
export TRITON_CACHE_DIR="$HOME/.triton-kern"
export UNSLOTH_VLLM_STANDBY=0
cd "$(dirname "$(readlink -f "$0")")/.."
exec "$@"
```

- [ ] **Step 4: 验证隔离环境**

```bash
wsl -d Ubuntu -- bash /mnt/d/sideProjects/Chasing-the-Aha-Moment/.claude/worktrees/gpu-kernels/scripts/wsl_env.sh python -c "import sys,torch,triton,bitsandbytes;print(sys.prefix);print(torch.__version__,triton.__version__,bitsandbytes.__version__)"
```

期望：第一行 `/home/rc/.venvs/aha-kern`，第二行 `2.8.0+cu128 3.4.0 0.50.2`。

- [ ] **Step 5: 现有测试在隔离环境里通过**

```bash
$KERN python -m pytest tests -q
```

期望：全部 PASS（与原分支一致）。

- [ ] **Step 6: 建报告目录并提交**

`docs/kernels/README.md`:

```markdown
# Kernel benchmarks

每个 benchmark 一份报告，由 `bench/` 和 `scripts/bench_decode.py` 生成。硬件 RTX 3060 12 GB（GA106）：
显存带宽 360 GB/s；tensor core fp32 累加 25.6 TFLOPS，fp16 累加 51.2 TFLOPS；CUDA core fp32 12.7 TFLOPS。
环境：`~/.venvs/aha-kern`（见 `scripts/wsl_env.sh`）。
```

```bash
cd /d/sideProjects/Chasing-the-Aha-Moment/.claude/worktrees/gpu-kernels
git add scripts/wsl_env.sh docs/kernels/README.md
git commit -m "Add isolated kernel-work environment wrapper and benchmark report dir"
```

---

### Task 2: Roofline 常数与利用率计算

**Files:**
- Create: `src/aha/kernels/__init__.py`（空文件）
- Create: `src/aha/kernels/roofline.py`
- Modify: `src/aha/profiling.py:14-16`
- Test: `tests/test_kernels_roofline.py`

- [ ] **Step 1: 写失败的测试**

`tests/test_kernels_roofline.py`:

```python
import pytest

from aha.kernels.roofline import RTX3060, achieved, gemm_flops, nf4_weight_bytes, utilisation


def test_constants():
    assert RTX3060["bandwidth_gbs"] == 360.0
    assert RTX3060["tflops_fp32_acc"] == 25.6
    assert RTX3060["tflops_fp16_acc"] == 51.2


def test_bytes_and_flops():
    assert gemm_flops(6, 2048, 2048) == 2 * 6 * 2048 * 2048
    # packed nibbles + one fp16 absmax per 64 weights
    assert nf4_weight_bytes(2048, 2048) == 2048 * 2048 // 2 + 2048 * 2048 // 64 * 2
    assert nf4_weight_bytes(2048, 2048, absmax_bytes=4) == 2048 * 2048 // 2 + 2048 * 2048 // 64 * 4


def test_achieved_and_utilisation():
    a = achieved(bytes_moved=360e9, flops=25.6e12, seconds=1.0)
    assert a["gbs"] == pytest.approx(360.0)
    assert a["tflops"] == pytest.approx(25.6)
    u = utilisation(a)
    assert u["bandwidth"] == pytest.approx(1.0)
    assert u["compute"] == pytest.approx(1.0)
    assert utilisation(a, accumulate="fp16")["compute"] == pytest.approx(0.5)
```

- [ ] **Step 2: 跑测试确认失败**

```bash
$KERN python -m pytest tests/test_kernels_roofline.py -v
```

期望：`ModuleNotFoundError: No module named 'aha.kernels'`。

- [ ] **Step 3: 实现**

`src/aha/kernels/roofline.py`:

```python
"""Hardware ceilings for the RTX 3060 (GA106) and the arithmetic to compare a kernel against them.

GeForce Ampere halves the tensor-core rate when accumulating in fp32, which is what every
bf16 GEMM must do; the 51 TFLOPS figure on spec sheets is fp16-accumulate only.
"""

from __future__ import annotations

RTX3060 = {
    "name": "RTX 3060 12 GB (GA106)",
    "bandwidth_gbs": 360.0,
    "tflops_fp32_acc": 25.6,   # fp16/bf16 tensor core, fp32 accumulate (cuBLAS bf16 path)
    "tflops_fp16_acc": 51.2,   # fp16 tensor core, fp16 accumulate
    "tflops_fp32_cuda": 12.7,  # CUDA cores
    "l2_bytes": 3 * 2**20,
}


def gemm_flops(M: int, N: int, K: int) -> float:
    return 2.0 * M * N * K


def nf4_weight_bytes(N: int, K: int, blocksize: int = 64, absmax_bytes: int = 2) -> int:
    """Bytes streamed per use of an NF4 weight: packed nibbles plus one absmax per block."""
    return N * K // 2 + (N * K // blocksize) * absmax_bytes


def achieved(bytes_moved: float, flops: float, seconds: float) -> dict:
    return {"gbs": bytes_moved / 1e9 / seconds, "tflops": flops / 1e12 / seconds}


def utilisation(ach: dict, gpu: dict = RTX3060, accumulate: str = "fp32") -> dict:
    peak = gpu["tflops_fp32_acc"] if accumulate == "fp32" else gpu["tflops_fp16_acc"]
    return {"bandwidth": ach["gbs"] / gpu["bandwidth_gbs"], "compute": ach["tflops"] / peak}
```

- [ ] **Step 4: 跑测试确认通过**

```bash
$KERN python -m pytest tests/test_kernels_roofline.py -v
```

期望：3 passed。

- [ ] **Step 5: 修正 `profiling.py` 的峰值常数**

把 `src/aha/profiling.py` 第 14–16 行改为：

```python
# RTX 3060 12 GB (GA106): 360 GB/s, ~12.7 TFLOPS FP32 CUDA cores. GeForce Ampere runs the tensor
# cores at half rate with fp32 accumulate, which every bf16 GEMM needs: 25.6 TFLOPS, not the
# 51 TFLOPS (fp16 accumulate) on the spec sheet. See aha.kernels.roofline.
RTX3060 = {"bandwidth_gbs": 360.0, "bf16_tflops": 25.6, "fp32_tflops": 12.7}
```

`tests/test_profiling.py::test_rooflines` 只断言 `achieved_tflops`，不受影响。跑一遍确认：

```bash
$KERN python -m pytest tests/test_profiling.py -q
```

期望：全部 PASS。

- [ ] **Step 6: 提交**

```bash
git add src/aha/kernels/__init__.py src/aha/kernels/roofline.py src/aha/profiling.py tests/test_kernels_roofline.py
git commit -m "Add roofline helpers; correct the 3060 bf16 peak to the fp32-accumulate rate"
```

---

### Task 3: NF4 参考实现（CPU 可测）+ 与 bnb 对拍（GPU）

**Files:**
- Create: `src/aha/kernels/nf4.py`
- Test: `tests/test_nf4_reference.py`

- [ ] **Step 1: 写失败的测试（CPU 部分）**

`tests/test_nf4_reference.py`:

```python
import pytest
import torch

from aha.kernels.nf4 import NF4_CODE, dequant_nf4_reference, pack_nf4_reference, unpack_nf4

CUDA = torch.cuda.is_available()


def test_codebook_is_sorted_and_symmetric_endpoints():
    assert NF4_CODE.shape == (16,)
    assert torch.all(NF4_CODE[1:] > NF4_CODE[:-1])
    assert NF4_CODE[0] == -1.0 and NF4_CODE[-1] == 1.0 and NF4_CODE[7] == 0.0


def test_pack_unpack_roundtrip():
    idx = torch.randint(0, 16, (2 * 64 * 3,), dtype=torch.uint8)
    packed = pack_nf4_reference(idx)
    assert packed.shape == (idx.numel() // 2, 1) and packed.dtype == torch.uint8
    assert packed[0, 0] == (int(idx[0]) << 4) | int(idx[1])  # first element in the high nibble
    assert torch.equal(unpack_nf4(packed, idx.numel()), idx)


def test_dequant_reference_scales_each_block():
    N, K = 2, 128  # 4 blocks of 64
    idx = torch.full((N * K,), 15, dtype=torch.uint8)  # code 1.0 everywhere
    absmax = torch.tensor([0.5, 1.0, 2.0, 4.0])
    W = dequant_nf4_reference(pack_nf4_reference(idx), absmax, N, K)
    assert W.shape == (N, K) and W.dtype == torch.float32
    assert W[0, 0] == 0.5 and W[0, 64] == 1.0 and W[1, 0] == 2.0 and W[1, 127] == 4.0


@pytest.mark.skipif(not CUDA, reason="needs bitsandbytes CUDA")
def test_reference_matches_bitsandbytes():
    import bitsandbytes.functional as F

    from aha.kernels.nf4 import resolve_absmax

    torch.manual_seed(0)
    N, K = 256, 2048
    W = (torch.randn(N, K, device="cuda") * 0.02).to(torch.bfloat16)
    packed, qs = F.quantize_4bit(W, blocksize=64, quant_type="nf4", compress_statistics=True)
    assert qs.nested
    ours = dequant_nf4_reference(packed, resolve_absmax(qs), N, K)
    theirs = F.dequantize_4bit(packed, qs).float()
    assert torch.allclose(ours, theirs, atol=1e-6, rtol=1e-5)
```

- [ ] **Step 2: 跑测试确认失败**

```bash
$KERN python -m pytest tests/test_nf4_reference.py -v
```

期望：`ImportError`。

- [ ] **Step 3: 实现**

`src/aha/kernels/nf4.py`:

```python
"""NF4 (bitsandbytes) layout in plain torch: the oracle every kernel in this package is tested against.

Layout facts (bitsandbytes 0.50): a [N, K] weight is flattened row-major, quantised in blocks of
``blocksize`` (64) values to 4-bit code indices, and packed two per byte with the first value in the
high nibble -> uint8 tensor of shape [(N*K)//2, 1]. One fp32 absmax per block; with
``compress_statistics=True`` the absmax tensor is itself 8-bit block-quantised (blocksize 256,
``quant_state.state2``) with a scalar ``offset`` added back after dequantisation.
"""

from __future__ import annotations

import torch

# bitsandbytes' NF4 codebook (create_normal_map(offset=0.9677083)), ascending.
NF4_CODE = torch.tensor(
    [
        -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
        -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
        0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
        0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
    ],
    dtype=torch.float32,
)


def pack_nf4_reference(idx: torch.Tensor) -> torch.Tensor:
    """4-bit code indices (uint8, even length) -> packed [(n)//2, 1] uint8, first index in the high nibble."""
    idx = idx.reshape(-1)
    assert idx.numel() % 2 == 0
    return ((idx[0::2] << 4) | idx[1::2]).to(torch.uint8).reshape(-1, 1)


def unpack_nf4(packed: torch.Tensor, numel: int) -> torch.Tensor:
    """Packed uint8 -> code indices (uint8) of length ``numel``."""
    b = packed.reshape(-1)
    idx = torch.stack([b >> 4, b & 0xF], dim=-1).reshape(-1)
    return idx[:numel]


def resolve_absmax(quant_state) -> torch.Tensor:
    """fp32 absmax per block, undoing bitsandbytes' nested 8-bit statistics when present."""
    import bitsandbytes.functional as F

    if quant_state.nested:
        absmax = F.dequantize_blockwise(quant_state.absmax, quant_state.state2) + quant_state.offset
    else:
        absmax = quant_state.absmax
    return absmax.float()


def dequant_nf4_reference(packed: torch.Tensor, absmax: torch.Tensor, N: int, K: int,
                          blocksize: int = 64) -> torch.Tensor:
    """fp32 [N, K] weight from packed nibbles and per-block absmax. Slow, obviously correct."""
    idx = unpack_nf4(packed, N * K).long()
    code = NF4_CODE.to(idx.device)
    vals = code[idx].reshape(-1, blocksize) * absmax.float().reshape(-1, 1)
    return vals.reshape(N, K)
```

- [ ] **Step 4: 跑测试确认通过**

```bash
$KERN python -m pytest tests/test_nf4_reference.py -v
```

期望：4 passed（GPU 那条要显卡空闲）。如果 `test_reference_matches_bitsandbytes` 失败且差异是整块对调，说明 nibble 顺序假设错了：把 `pack_nf4_reference` 和 `unpack_nf4` 的高低位互换后重跑，并更新 `test_pack_unpack_roundtrip` 的断言。

- [ ] **Step 5: 提交**

```bash
git add src/aha/kernels/nf4.py tests/test_nf4_reference.py
git commit -m "Add NF4 reference dequantisation validated against bitsandbytes"
```

---

### Task 4: Benchmark 框架

**Files:**
- Create: `bench/__init__.py`（空文件）
- Create: `bench/harness.py`
- Test: `tests/test_bench_harness.py`

- [ ] **Step 1: 写失败的测试（纯逻辑部分）**

`tests/test_bench_harness.py`:

```python
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.harness import GpuBusyError, check_gpu_free_from_query, render_table  # noqa: E402


def test_gpu_free_check_parses_nvidia_smi_line():
    check_gpu_free_from_query("1312, 3")  # idle desktop: fine
    with pytest.raises(GpuBusyError):
        check_gpu_free_from_query("4148, 83")
    with pytest.raises(GpuBusyError):
        check_gpu_free_from_query("1000, 55")
    check_gpu_free_from_query("4148, 83", force=True)


def test_render_table_formats_numbers_and_strings():
    rows = [{"kernel": "bnb", "M": 6, "ms": 0.08312, "bw%": 0.3012}, {"kernel": "ours", "M": 6, "ms": 0.04, "bw%": 0.61}]
    md = render_table(rows, ["kernel", "M", "ms", "bw%"], title="t")
    assert md.splitlines()[0] == "## t"
    assert "| bnb | 6 | 0.0831 | 30.1% |" in md
    assert "| ours | 6 | 0.0400 | 61.0% |" in md
```

- [ ] **Step 2: 跑测试确认失败**

```bash
$KERN python -m pytest tests/test_bench_harness.py -v
```

期望：`ModuleNotFoundError: No module named 'bench'`。

- [ ] **Step 3: 实现**

`bench/harness.py`:

```python
"""Shared microbenchmark plumbing: refuse a busy GPU, time with CUDA events behind an L2 flush,
report as markdown. GPU-free code (parsing, rendering) is unit-tested; timing needs a card.
"""

from __future__ import annotations

import statistics
import subprocess
from pathlib import Path
from typing import Callable, Sequence

USED_MIB_LIMIT = 2500   # WDDM desktop idles around 1.3 GB on this machine
UTIL_PCT_LIMIT = 20


class GpuBusyError(RuntimeError):
    pass


def check_gpu_free_from_query(line: str, force: bool = False) -> None:
    """``line`` is nvidia-smi's ``memory.used, utilization.gpu`` (csv, noheader, nounits)."""
    used, util = (int(x.strip()) for x in line.split(","))
    if (used > USED_MIB_LIMIT or util > UTIL_PCT_LIMIT) and not force:
        raise GpuBusyError(
            f"GPU busy: {used} MiB used, {util}% util (limits {USED_MIB_LIMIT} MiB / {UTIL_PCT_LIMIT}%). "
            "Another process (e.g. the Windows-side miniconda 'rec' python) is on the card; use --force to override."
        )


def assert_gpu_free(force: bool = False) -> None:
    """Call before the first CUDA allocation so the numbers describe other processes only."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()[0]
    check_gpu_free_from_query(out, force=force)


def l2_flush() -> None:
    import torch

    buf = getattr(l2_flush, "_buf", None)
    if buf is None:
        buf = torch.empty(64 * 2**20, dtype=torch.uint8, device="cuda")  # 64 MB > 3 MB L2
        l2_flush._buf = buf
    buf.zero_()


def time_ms(fn: Callable[[int], None], iters: int = 100, warmup: int = 20, flush: bool = True) -> float:
    """Median wall time of ``fn(i)`` in ms, each call behind an L2 flush. ``i`` lets the caller
    rotate between weight copies so the same bytes are not served from cache."""
    import torch

    for i in range(warmup):
        fn(i)
    torch.cuda.synchronize()
    times = []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for i in range(iters):
        if flush:
            l2_flush()
        start.record()
        fn(i)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def rel_fro_error(out, ref) -> float:
    o, r = out.float(), ref.float()
    return ((o - r).norm() / r.norm()).item()


def render_table(rows: Sequence[dict], columns: Sequence[str], title: str) -> str:
    """Markdown table; a float in a column whose name ends with '%' is rendered as a percentage."""
    def fmt(col, v):
        if isinstance(v, float):
            return f"{v:.1%}" if col.endswith("%") else f"{v:.4f}"
        return str(v)

    lines = [f"## {title}", "", "| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for r in rows:
        lines.append("| " + " | ".join(fmt(c, r[c]) for c in columns) + " |")
    return "\n".join(lines) + "\n"


def write_report(path: str | Path, sections: Sequence[str], header: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header.rstrip() + "\n\n" + "\n".join(sections), encoding="utf-8")
    return path
```

- [ ] **Step 4: 跑测试确认通过**

```bash
$KERN python -m pytest tests/test_bench_harness.py -v
```

期望：2 passed。

- [ ] **Step 5: 提交**

```bash
git add bench/__init__.py bench/harness.py tests/test_bench_harness.py
git commit -m "Add microbenchmark harness: GPU-free check, event timing behind L2 flush, markdown tables"
```

---

### Task 5: 4-bit GEMM 基线（P0）

**Files:**
- Create: `bench/bench_gemm_4bit.py`
- Output: `docs/kernels/gemm_4bit.md`

- [ ] **Step 1: 写 benchmark**

`bench/bench_gemm_4bit.py`:

```python
#!/usr/bin/env python
"""Decode-shaped 4-bit GEMM: y[M, N] = x[M, K] @ W_nf4[N, K]^T for the Qwen2.5-3B projections.

Candidates: bitsandbytes' fused kernel (what vLLM calls), bnb dequant + cuBLAS (its fallback),
cuBLAS on a bf16 copy of the weight (the 16-bit bound), and our Triton kernel when present.

    python bench/bench_gemm_4bit.py --out docs/kernels/gemm_4bit.md
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bench.harness import assert_gpu_free, rel_fro_error, render_table, time_ms, write_report  # noqa: E402

# (name, N, K) of the Qwen2.5-3B linear layers (hidden 2048, intermediate 11008, 2 kv heads x 128)
QWEN_SHAPES = [("q_proj", 2048, 2048), ("kv_proj", 256, 2048), ("gate_up", 11008, 2048), ("down", 2048, 11008)]
MS = [6, 8, 16, 24, 32]
ROTATE_BYTES = 48 * 2**20  # copies of each weight so consecutive calls miss L2


def make_weights(N, K, copies, device="cuda"):
    import bitsandbytes.functional as F
    import torch

    from aha.kernels.nf4 import dequant_nf4_reference, resolve_absmax

    W = (torch.randn(N, K, device=device) * 0.02).to(torch.bfloat16)
    packs = []
    for _ in range(copies):
        packed, qs = F.quantize_4bit(W.clone(), blocksize=64, quant_type="nf4", compress_statistics=True)
        packs.append((packed, qs))
    absmax = resolve_absmax(packs[0][1])
    W_ref = dequant_nf4_reference(packs[0][0], absmax, N, K)  # fp32 oracle of the quantised weight
    return packs, W_ref, W_ref.to(torch.bfloat16)


def run(out_path: str, with_triton: bool, force: bool) -> None:
    assert_gpu_free(force)
    import torch
    from bitsandbytes import matmul_4bit
    import bitsandbytes.functional as F

    from aha.kernels.roofline import RTX3060, achieved, gemm_flops, nf4_weight_bytes, utilisation

    if with_triton:
        from aha.kernels.nf4_gemm import absmax_fp16_for, nf4_gemm

    torch.manual_seed(0)
    rows = []
    for name, N, K in QWEN_SHAPES:
        copies = max(1, math.ceil(ROTATE_BYTES / (N * K // 2)))
        packs, W_ref, W16s = make_weights(N, K, copies)
        W16 = [W16s.clone() for _ in range(copies)]
        for M in MS:
            x = (torch.randn(M, K, device="cuda") * 0.5).to(torch.bfloat16)
            ref = x.float() @ W_ref.t()
            nbytes = nf4_weight_bytes(N, K) + x.numel() * 2 + M * N * 2
            flops = gemm_flops(M, N, K)
            cands = {
                "bnb fused": lambda i: matmul_4bit(x, packs[i % copies][0].t(), packs[i % copies][1]),
                "bnb dequant+cuBLAS": lambda i: torch.nn.functional.linear(
                    x, F.dequantize_4bit(packs[i % copies][0], packs[i % copies][1])),
                "cuBLAS bf16 weight": lambda i: torch.nn.functional.linear(x, W16[i % copies]),
            }
            if with_triton:
                am16 = [absmax_fp16_for(qs) for _, qs in packs]
                cands["triton nf4"] = lambda i: nf4_gemm(x, packs[i % copies][0], am16[i % copies], N, K)
            for cname, fn in cands.items():
                with torch.no_grad():
                    out = fn(0)
                    ms = time_ms(fn)
                b = nbytes if "bf16" not in cname else N * K * 2 + x.numel() * 2 + M * N * 2
                ach = achieved(b, flops, ms / 1e3)
                util = utilisation(ach)
                rows.append({"shape": f"{name} {N}x{K}", "M": M, "kernel": cname, "ms": ms,
                             "GB/s": ach["gbs"], "bw%": util["bandwidth"], "TFLOPS": ach["tflops"],
                             "tc%": util["compute"], "rel_err": rel_fro_error(out, ref)})
                print(rows[-1])
    cols = ["shape", "M", "kernel", "ms", "GB/s", "bw%", "TFLOPS", "tc%", "rel_err"]
    header = (f"# 4-bit GEMM, decode shapes ({RTX3060['name']})\n\n"
              "bytes = packed nibbles + fp16 absmax + activations; bw% against 360 GB/s; tc% against 25.6 TFLOPS "
              "(fp32 accumulate). rel_err = ||out - ref||_F / ||ref||_F against an fp32 product of the dequantised weight.")
    write_report(out_path, [render_table(rows, cols, "all shapes")], header)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="docs/kernels/gemm_4bit.md")
    p.add_argument("--triton", action="store_true", help="include aha.kernels.nf4_gemm (Task 7)")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    run(a.out, a.triton, a.force)
```

- [ ] **Step 2: 跑基线**

```bash
$KERN python bench/bench_gemm_4bit.py --out docs/kernels/gemm_4bit.md
```

期望：`bnb fused` 在 `gate_up 11008x2048, M=6` 行的 `bw%` 在 25%–40% 之间（profile 给的是 30%），`cuBLAS bf16 weight` 的 GB/s 明显高于 bnb；`rel_err` 三者都在 1e-2 以下。若 `bnb fused` 的 bw% 高于 60%，说明 L2 轮换失效（copies 太少），把 `ROTATE_BYTES` 提到 96 MB 重跑。

- [ ] **Step 3: 在报告末尾手写两行结论**

在 `docs/kernels/gemm_4bit.md` 末尾追加（用实际数字替换 `<...>`）：

```markdown
## 结论（基线，2026-10-27）

- bnb fused 在 M=6 的大矩阵上达到 <x> GB/s（<y>% 带宽），与 profile 的 30% 一致；dequant+cuBLAS 为 <z>%。
- bf16 权重的 cuBLAS 达到 <w>% 带宽，是 Triton 版应当逼近的上限形状。
```

- [ ] **Step 4: 提交**

```bash
git add bench/bench_gemm_4bit.py docs/kernels/gemm_4bit.md
git commit -m "Benchmark bitsandbytes 4-bit GEMM at decode shapes against cuBLAS"
```

---

### Task 6: 训练形状的 cuBLAS bf16 基线（P0）

**Files:**
- Create: `bench/bench_cublas.py`
- Output: `docs/kernels/cublas_bf16.md`

- [ ] **Step 1: 写 benchmark**

`bench/bench_cublas.py`:

```python
#!/usr/bin/env python
"""Training-shaped GEMMs (M = 6 seq x 400 tokens) through cuBLAS, bf16 and fp16, against the
fp32-accumulate tensor-core peak. Establishes what the fp16-accumulate experiment (P3) must beat.

    python bench/bench_cublas.py --out docs/kernels/cublas_bf16.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bench.harness import assert_gpu_free, rel_fro_error, render_table, time_ms, write_report  # noqa: E402

M = 2400
SHAPES = [("q/o_proj", 2048, 2048), ("kv_proj", 256, 2048), ("gate_up", 11008, 2048), ("down", 2048, 11008),
          ("lora_A", 32, 2048), ("lora_B", 2048, 32), ("lm_head", 151936, 2048)]


def run(out_path: str, force: bool) -> None:
    assert_gpu_free(force)
    import torch

    from aha.kernels.roofline import RTX3060, achieved, gemm_flops, utilisation

    torch.manual_seed(0)
    rows = []
    for name, N, K in SHAPES:
        x32 = torch.randn(M, K, device="cuda") * 0.5
        w32 = torch.randn(N, K, device="cuda") * 0.02
        ref = x32 @ w32.t()
        flops = gemm_flops(M, N, K)
        nbytes = (M * K + N * K + M * N) * 2
        for dt, label in [(torch.bfloat16, "cuBLAS bf16"), (torch.float16, "cuBLAS fp16"), (torch.float32, "cuBLAS fp32 (cuda cores)")]:
            x, w = x32.to(dt), w32.to(dt)
            fn = lambda i: torch.nn.functional.linear(x, w)  # noqa: E731
            out = fn(0)
            ms = time_ms(fn, iters=30, warmup=5)
            ach = achieved(nbytes if dt != torch.float32 else nbytes * 2, flops, ms / 1e3)
            util = utilisation(ach)
            rows.append({"shape": f"{name} {N}x{K}", "kernel": label, "ms": ms, "TFLOPS": ach["tflops"],
                         "tc%": util["compute"], "GB/s": ach["gbs"], "rel_err": rel_fro_error(out, ref)})
            print(rows[-1])
    cols = ["shape", "kernel", "ms", "TFLOPS", "tc%", "GB/s", "rel_err"]
    header = (f"# cuBLAS GEMM at training shapes, M={M} ({RTX3060['name']})\n\n"
              "tc% against 25.6 TFLOPS (fp32 accumulate); fp32 rows are CUDA-core GEMMs (12.7 TFLOPS peak).")
    write_report(out_path, [render_table(rows, cols, "all shapes")], header)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="docs/kernels/cublas_bf16.md")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    run(a.out, a.force)
```

- [ ] **Step 2: 跑**

```bash
$KERN python bench/bench_cublas.py --out docs/kernels/cublas_bf16.md
```

期望：大矩阵的 `cuBLAS bf16` tc% 在 50%–75%；`lora_A`/`lora_B` 的 tc% 很低（< 10%）而 GB/s 接近带宽上限（它们是带宽型，不是算力型）；`lm_head` 一行约 1.5 TFLOP，1 秒量级以内。

- [ ] **Step 3: 追加结论并提交**

在 `docs/kernels/cublas_bf16.md` 末尾追加：

```markdown
## 结论（基线，2026-10-27）

- 底座 GEMM 在 fp32 累加峰值下达到 <x>%，profile 里的 14.4 TFLOPS 不是"算子效率低"，是 GeForce 的累加速率上限。
- LoRA 的两个 GEMM 是带宽型（K 或 N 只有 32），tensor core 利用率 <y>%，融合比调 tile 更有意义。
```

```bash
git add bench/bench_cublas.py docs/kernels/cublas_bf16.md
git commit -m "Benchmark cuBLAS at training shapes against the fp32-accumulate peak"
```

---

### Task 7: Triton 小 M NF4 GEMM（P1 kernel）

**Files:**
- Create: `src/aha/kernels/nf4_gemm.py`
- Test: `tests/test_nf4_gemm_gpu.py`

- [ ] **Step 1: 写失败的测试**

`tests/test_nf4_gemm_gpu.py`:

```python
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA + triton")


def _case(N, K, M, seed=0):
    import bitsandbytes.functional as F

    from aha.kernels.nf4 import dequant_nf4_reference, resolve_absmax

    torch.manual_seed(seed)
    W = (torch.randn(N, K, device="cuda") * 0.02).to(torch.bfloat16)
    packed, qs = F.quantize_4bit(W, blocksize=64, quant_type="nf4", compress_statistics=True)
    x = (torch.randn(M, K, device="cuda") * 0.5).to(torch.bfloat16)
    ref = x.float() @ dequant_nf4_reference(packed, resolve_absmax(qs), N, K).t()
    return x, packed, qs, ref


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@pytest.mark.parametrize("N,K", [(256, 2048), (2048, 2048), (2048, 11008)])
@pytest.mark.parametrize("M", [1, 6, 24, 32])
def test_nf4_gemm_matches_reference_within_bnb_error(N, K, M):
    from bitsandbytes import matmul_4bit

    from aha.kernels.nf4_gemm import absmax_fp16_for, nf4_gemm

    x, packed, qs, ref = _case(N, K, M)
    with torch.no_grad():
        theirs = matmul_4bit(x, packed.t(), qs)
        ours = nf4_gemm(x, packed, absmax_fp16_for(qs), N, K)
    assert ours.shape == (M, N) and ours.dtype == torch.bfloat16
    err_ours, err_bnb = _rel(ours, ref), _rel(theirs, ref)
    assert err_ours <= 1.5 * err_bnb + 1e-4, (err_ours, err_bnb)
    assert err_ours < 1e-2


def test_nf4_gemm_rejects_unsupported_shapes():
    from aha.kernels.nf4_gemm import nf4_gemm

    x = torch.zeros(4, 100, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        nf4_gemm(x, torch.zeros(50, 1, dtype=torch.uint8, device="cuda"), torch.zeros(2, dtype=torch.float16, device="cuda"), 1, 100)
    x = torch.zeros(64, 2048, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        nf4_gemm(x, torch.zeros(2048 * 2048 // 2, 1, dtype=torch.uint8, device="cuda"),
                 torch.zeros(2048 * 2048 // 64, dtype=torch.float16, device="cuda"), 2048, 2048)
```

- [ ] **Step 2: 跑测试确认失败**

```bash
$KERN python -m pytest tests/test_nf4_gemm_gpu.py -v
```

期望：`ImportError`。

- [ ] **Step 3: 实现 kernel**

`src/aha/kernels/nf4_gemm.py`:

```python
"""Small-M NF4 weight-only GEMM in Triton: y[M, N] = x[M, K] @ W[N, K]^T with W packed as bnb NF4.

Why: at decode (M = 6..32) bitsandbytes' fused kernel streams the weight at ~30% of the 3060's
bandwidth. This kernel keeps the weight packed until it is in registers: each program owns a
BLOCK_N row stripe, walks K in blocks of 64 (= the NF4 blocksize, so one absmax per row per
block), loads the packed bytes, splits high/low nibbles, looks both up in the 16-entry codebook,
scales, and feeds two tensor-core dots (even k from the high nibbles, odd k from the low ones)
so no in-register interleave is needed. fp32 accumulate, bf16 out.

Nested absmax statistics are resolved once per weight into an fp16 vector (``absmax_fp16_for``)
and cached by the identity of the quant state's absmax tensor; ~95 MB for the whole 3B model.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from aha.kernels.nf4 import NF4_CODE, resolve_absmax

MAX_M = 32
BLOCKSIZE = 64


@triton.jit
def _nf4_gemm_kernel(a_ptr, w_ptr, absmax_ptr, code_ptr, c_ptr,
                     M, N, K, stride_am, stride_cm,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_n = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    HALF: tl.constexpr = BLOCK_K // 2
    offs_h = tl.arange(0, HALF)                      # byte index inside the K block
    kblocks = K // BLOCK_K
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kb in range(0, kblocks):
        k0 = kb * BLOCK_K
        a_even = tl.load(a_ptr + offs_m[:, None] * stride_am + (k0 + 2 * offs_h)[None, :],
                         mask=mask_m[:, None], other=0.0)
        a_odd = tl.load(a_ptr + offs_m[:, None] * stride_am + (k0 + 2 * offs_h + 1)[None, :],
                        mask=mask_m[:, None], other=0.0)
        w_bytes = tl.load(w_ptr + (offs_n[:, None] * K + k0) // 2 + offs_h[None, :],
                          mask=mask_n[:, None], other=0)              # [BLOCK_N, HALF] uint8
        hi = (w_bytes >> 4).to(tl.int32)
        lo = (w_bytes & 0xF).to(tl.int32)
        scale = tl.load(absmax_ptr + offs_n * kblocks + kb, mask=mask_n, other=0.0).to(tl.float32)
        w_hi = (tl.load(code_ptr + hi) * scale[:, None]).to(tl.bfloat16)   # even k
        w_lo = (tl.load(code_ptr + lo) * scale[:, None]).to(tl.bfloat16)   # odd k
        acc += tl.dot(a_even, tl.trans(w_hi))
        acc += tl.dot(a_odd, tl.trans(w_lo))
    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :], acc.to(tl.bfloat16),
             mask=mask_m[:, None] & mask_n[None, :])


_ABSMAX_CACHE: dict[int, torch.Tensor] = {}
_CODE_CACHE: dict[int, torch.Tensor] = {}


def absmax_fp16_for(quant_state) -> torch.Tensor:
    """fp16 absmax per 64-block for a bnb quant state, cached on the absmax tensor's identity."""
    key = quant_state.absmax.data_ptr()
    hit = _ABSMAX_CACHE.get(key)
    if hit is None:
        hit = resolve_absmax(quant_state).to(torch.float16).contiguous()
        _ABSMAX_CACHE[key] = hit
    return hit


def _code(device) -> torch.Tensor:
    c = _CODE_CACHE.get(device.index)
    if c is None:
        c = NF4_CODE.to(device)
        _CODE_CACHE[device.index] = c
    return c


def nf4_gemm(x: torch.Tensor, packed: torch.Tensor, absmax16: torch.Tensor, N: int, K: int,
             out: torch.Tensor | None = None, block_n: int = 64) -> torch.Tensor:
    """x [M, K] bf16 (M <= 32), packed bnb NF4 weight of logical shape [N, K] -> [M, N] bf16."""
    if x.ndim != 2 or x.dtype != torch.bfloat16:
        raise ValueError(f"x must be 2-D bf16, got {tuple(x.shape)} {x.dtype}")
    M = x.shape[0]
    if M > MAX_M:
        raise ValueError(f"M={M} > {MAX_M}: this kernel is for decode-sized batches")
    if x.shape[1] != K or K % BLOCKSIZE != 0:
        raise ValueError(f"K must equal x.shape[1] and be a multiple of {BLOCKSIZE}, got K={K}, x={tuple(x.shape)}")
    if packed.numel() != N * K // 2 or absmax16.numel() != N * K // BLOCKSIZE:
        raise ValueError("packed/absmax sizes do not match N x K")
    x = x.contiguous()
    if out is None:
        out = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    block_m = 16 if M <= 16 else 32
    grid = (triton.cdiv(N, block_n),)
    _nf4_gemm_kernel[grid](x, packed, absmax16, _code(x.device), out, M, N, K, x.stride(0), out.stride(0),
                           BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=BLOCKSIZE)
    return out
```

- [ ] **Step 4: 跑测试**

```bash
$KERN python -m pytest tests/test_nf4_gemm_gpu.py -v
```

期望：13 passed。若 Triton 报 `tl.dot` 维度错误（HALF 必须 ≥ 16，BLOCK_M ≥ 16）：BLOCK_K=64 给 HALF=32，BLOCK_M 最小 16，检查没有改动这些值。若误差大于 bnb 的 1.5 倍且差异集中在特定列，检查 absmax 的索引 `offs_n * kblocks + kb`（absmax 是按展平行主序每 64 个一块）。

- [ ] **Step 5: 加入 benchmark 并记录数字**

```bash
$KERN python bench/bench_gemm_4bit.py --triton --out docs/kernels/gemm_4bit.md
```

期望：`triton nf4` 在 `gate_up`/`down`、M ∈ {6, 8, 16, 24} 的 bw% ≥ 2 × `bnb fused`，目标 ≥ 55%。`kv_proj`（N=256，只有 4 个 program）允许低。若整体低于 bnb：先试 `block_n=32` 和 `block_n=128`（在 `nf4_gemm` 的签名里改默认值重跑），再加 `num_warps=4/8` 和 `num_stages=3/4` 的 launch 参数（`_nf4_gemm_kernel[grid](..., num_warps=4, num_stages=3)`），把最好的一组写死为默认值。

在 `docs/kernels/gemm_4bit.md` 的结论段追加一行：`- triton nf4（block_n=<n>, warps=<w>, stages=<s>）：gate_up M=6 达到 <x>% 带宽，为 bnb 的 <y> 倍。`

- [ ] **Step 6: 提交**

```bash
git add src/aha/kernels/nf4_gemm.py tests/test_nf4_gemm_gpu.py docs/kernels/gemm_4bit.md
git commit -m "Add Triton small-M NF4 GEMM and benchmark it against bitsandbytes"
```

---

### Task 8: 接入 bitsandbytes（`patch_bitsandbytes`）

**Files:**
- Modify: `src/aha/kernels/nf4_gemm.py`（末尾追加）
- Test: `tests/test_nf4_gemm_gpu.py`（追加）

- [ ] **Step 1: 写失败的测试**

追加到 `tests/test_nf4_gemm_gpu.py`:

```python
def test_patch_routes_small_m_and_delegates_large_m():
    import bitsandbytes

    from aha.kernels import nf4_gemm as mod

    x, packed, qs, ref = _case(2048, 2048, 8)
    with torch.no_grad():
        before = bitsandbytes.matmul_4bit(x, packed.t(), qs)
    mod.patch_bitsandbytes()
    try:
        assert bitsandbytes.matmul_4bit is mod._patched_matmul_4bit
        with torch.no_grad():
            after = bitsandbytes.matmul_4bit(x, packed.t(), qs)
            assert _rel(after, ref) <= 1.5 * _rel(before, ref) + 1e-4
            # 3-D activations (vLLM passes [1, M, K] during prefill-shaped calls) keep their shape
            after3 = bitsandbytes.matmul_4bit(x.unsqueeze(0), packed.t(), qs)
            assert after3.shape == (1, 8, 2048)
            # large M is delegated to the original implementation
            xb = torch.randn(300, 2048, device="cuda").to(torch.bfloat16)
            big = bitsandbytes.matmul_4bit(xb, packed.t(), qs)
            assert torch.equal(big, mod._ORIGINAL_MATMUL_4BIT(xb, packed.t(), qs))
    finally:
        mod.unpatch_bitsandbytes()
    assert bitsandbytes.matmul_4bit is mod._ORIGINAL_MATMUL_4BIT
```

- [ ] **Step 2: 跑测试确认失败**

```bash
$KERN python -m pytest tests/test_nf4_gemm_gpu.py::test_patch_routes_small_m_and_delegates_large_m -v
```

期望：`AttributeError: ... has no attribute 'patch_bitsandbytes'`。

- [ ] **Step 3: 实现补丁**

追加到 `src/aha/kernels/nf4_gemm.py`:

```python
# --- bitsandbytes integration -------------------------------------------------------------
# vLLM's bitsandbytes linear method does ``from bitsandbytes import matmul_4bit`` inside the call,
# so rebinding the package attribute reaches the engine (which Unsloth runs in this process,
# VLLM_ENABLE_V1_MULTIPROCESSING=0). Apply before the engine is built so the warm-up run fills the
# absmax cache before CUDA-graph capture.

_ORIGINAL_MATMUL_4BIT = None


def _patched_matmul_4bit(A, B, quant_state, out=None, bias=None):
    if (
        quant_state is not None
        and quant_state.quant_type == "nf4"
        and quant_state.blocksize == BLOCKSIZE
        and A.dtype == torch.bfloat16
        and not (torch.is_grad_enabled() and A.requires_grad)
        and A.shape[-1] == quant_state.shape[1]
        and quant_state.shape[1] % BLOCKSIZE == 0
        and 1 <= A.numel() // A.shape[-1] <= MAX_M
    ):
        N, K = quant_state.shape
        x2 = A.reshape(-1, K)
        y = nf4_gemm(x2, B.reshape(-1), absmax_fp16_for(quant_state), N, K)
        if bias is not None:
            y = y + bias
        y = y.reshape(*A.shape[:-1], N)
        if out is not None:
            out.copy_(y)
            return out
        return y
    return _ORIGINAL_MATMUL_4BIT(A, B, quant_state, out=out, bias=bias)


def patch_bitsandbytes() -> None:
    global _ORIGINAL_MATMUL_4BIT
    import bitsandbytes

    if bitsandbytes.matmul_4bit is _patched_matmul_4bit:
        return
    _ORIGINAL_MATMUL_4BIT = bitsandbytes.matmul_4bit
    bitsandbytes.matmul_4bit = _patched_matmul_4bit


def unpatch_bitsandbytes() -> None:
    import bitsandbytes

    if _ORIGINAL_MATMUL_4BIT is not None:
        bitsandbytes.matmul_4bit = _ORIGINAL_MATMUL_4BIT
```

- [ ] **Step 4: 跑测试确认通过**

```bash
$KERN python -m pytest tests/test_nf4_gemm_gpu.py -v
```

期望：14 passed。

- [ ] **Step 5: 提交**

```bash
git add src/aha/kernels/nf4_gemm.py tests/test_nf4_gemm_gpu.py
git commit -m "Route decode-sized bitsandbytes matmul_4bit calls to the Triton NF4 GEMM"
```

---

### Task 9: vLLM decode 端到端对照（P1 验收）

**Files:**
- Create: `scripts/bench_decode.py`
- Output: `docs/kernels/decode.md`、`docs/kernels/decode_base.json`、`docs/kernels/decode_patch.json`

- [ ] **Step 1: 写脚本**

`scripts/bench_decode.py`:

```python
#!/usr/bin/env python
"""End-to-end vLLM decode speed with and without the Triton NF4 GEMM, plus greedy-output agreement.

    python scripts/bench_decode.py --config configs/qwen2.5-3b_3060.yaml --out docs/kernels/decode_base.json
    python scripts/bench_decode.py --config configs/qwen2.5-3b_3060.yaml --patch --out docs/kernels/decode_patch.json
    python scripts/bench_decode.py --compare docs/kernels/decode_base.json docs/kernels/decode_patch.json --report docs/kernels/decode.md

Timed runs use temperature 1.0 with ignore_eos so every sequence decodes exactly max_tokens steps;
the greedy run (temperature 0) is saved for the agreement check. The patch must be applied before
the engine is built (CUDA graphs are captured at load).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def measure(args) -> dict:
    from bench.harness import assert_gpu_free

    assert_gpu_free(args.force)
    if args.patch:
        from aha.kernels.nf4_gemm import patch_bitsandbytes

        patch_bitsandbytes()
    import unsloth  # noqa: F401
    import torch
    from vllm import SamplingParams

    from aha.config import load_config
    from aha.data import load_eval_dataset
    from aha.model import load_model_and_tokenizer

    cfg = load_config(args.config, [])
    model, tok = load_model_and_tokenizer(cfg, with_lora=True)
    examples = load_eval_dataset("gsm8k_test", max_samples=max(args.prompts))
    texts = [tok.apply_chat_template(ex.prompt, tokenize=False, add_generation_prompt=True) for ex in examples]

    result = {"patch": args.patch, "max_tokens": args.max_tokens, "n": args.n, "timings": []}
    for n_prompts in args.prompts:
        sp = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=args.max_tokens, n=args.n, seed=0, ignore_eos=True)
        model.fast_generate(texts[:n_prompts], sampling_params=sp)  # warm-up / graph capture
        walls = []
        for _ in range(3):
            torch.cuda.synchronize()
            t0 = time.time()
            model.fast_generate(texts[:n_prompts], sampling_params=sp)
            torch.cuda.synchronize()
            walls.append(time.time() - t0)
        wall = sorted(walls)[1]
        seqs = n_prompts * args.n
        result["timings"].append({"prompts": n_prompts, "sequences": seqs, "wall_s": wall,
                                  "ms_per_step": wall / args.max_tokens * 1e3,
                                  "tok_per_s": seqs * args.max_tokens / wall})
        print(result["timings"][-1])

    greedy = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    outs = model.fast_generate(texts, sampling_params=greedy)
    result["greedy"] = [list(o.outputs[0].token_ids) for o in outs]
    return result


def compare(base_path: str, patch_path: str, report: str) -> None:
    base, patch = json.load(open(base_path)), json.load(open(patch_path))
    lines = ["# vLLM decode: bitsandbytes kernel vs Triton NF4 GEMM", "",
             "| sequences | base ms/step | patch ms/step | speed-up | base tok/s | patch tok/s |", "|---|---|---|---|---|---|"]
    for b, p in zip(base["timings"], patch["timings"]):
        lines.append(f"| {b['sequences']} | {b['ms_per_step']:.1f} | {p['ms_per_step']:.1f} | "
                     f"{b['ms_per_step'] / p['ms_per_step']:.2f}x | {b['tok_per_s']:.0f} | {p['tok_per_s']:.0f} |")
    exact, first_div = 0, []
    for a, c in zip(base["greedy"], patch["greedy"]):
        if a == c:
            exact += 1
        else:
            k = next((i for i, (u, v) in enumerate(zip(a, c)) if u != v), min(len(a), len(c)))
            first_div.append(k)
    n = len(base["greedy"])
    lines += ["", f"Greedy agreement on {n} GSM8K prompts: {exact}/{n} identical token sequences "
              f"({exact / n:.0%}); first divergence at token index {sorted(first_div)} for the rest.", ""]
    Path(report).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/qwen2.5-3b_3060.yaml")
    p.add_argument("--patch", action="store_true")
    p.add_argument("--prompts", type=int, nargs="*", default=[1, 4])
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--out", default=None)
    p.add_argument("--compare", nargs=2, metavar=("BASE", "PATCH"))
    p.add_argument("--report", default="docs/kernels/decode.md")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    if a.compare:
        compare(a.compare[0], a.compare[1], a.report)
    else:
        res = measure(a)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res), encoding="utf-8")
        print(f"wrote {a.out}")
```

- [ ] **Step 2: 跑基线和补丁版（各约 5 分钟，含模型加载）**

```bash
$KERN python scripts/bench_decode.py --out docs/kernels/decode_base.json
$KERN python scripts/bench_decode.py --patch --out docs/kernels/decode_patch.json
$KERN python scripts/bench_decode.py --compare docs/kernels/decode_base.json docs/kernels/decode_patch.json --report docs/kernels/decode.md
```

期望：基线 6 条序列约 25–35 ms/步；补丁版 ms/步下降 ≥ 30%，贪心一致率 ≥ 95%。

若补丁版加载时在 CUDA graph 捕获阶段报错（Triton kernel 在 capture 里首次编译）：在 `measure()` 里打补丁之后、建引擎之前，对每个 (N, K) 形状调用一次 `nf4_gemm` 触发编译：

```python
        import torch
        from aha.kernels.nf4_gemm import nf4_gemm
        for N, K in [(2048, 2048), (256, 2048), (11008, 2048), (2048, 11008)]:
            for M in (16, 32):
                nf4_gemm(torch.zeros(M, K, dtype=torch.bfloat16, device="cuda"),
                         torch.zeros(N * K // 2, 1, dtype=torch.uint8, device="cuda"),
                         torch.zeros(N * K // 64, dtype=torch.float16, device="cuda"), N, K)
```

若一致率 < 95%：先看 `first divergence` 的位置分布。都在很后面（> 100 token）是累加顺序造成的正常分叉；集中在前几个 token 说明某个形状算错了，用 `tests/test_nf4_gemm_gpu.py` 补那个形状（Unsloth 的 dynamic 4-bit 仓库有少数层是 16-bit，不经过补丁；`bnb_shard_offsets` 让 vLLM 对 qkv 合并权重按分片逐个调用，所以形状仍是上面四种）。

- [ ] **Step 3: 提交**

```bash
git add scripts/bench_decode.py docs/kernels/decode.md docs/kernels/decode_base.json docs/kernels/decode_patch.json
git commit -m "Measure vLLM decode with the Triton NF4 GEMM: ms/step and greedy agreement"
```

---

### Task 10: `use_liger_loss` 开关试验（P2）

**Files:**
- Modify: `src/aha/config.py:71-95`（TrainConfig）
- Modify: `src/aha/train.py:45-50`（grpo_kwargs）
- Modify: `src/aha/logging_utils.py:33-38`（on_log）
- Test: `tests/test_config.py`（追加）
- Output: `docs/kernels/liger_loss.md`

- [ ] **Step 1: 写失败的测试**

追加到 `tests/test_config.py`:

```python
def test_use_liger_loss_passthrough():
    from aha.train import grpo_kwargs

    cfg = ExperimentConfig()
    assert "use_liger_loss" not in grpo_kwargs(cfg)
    cfg.train.use_liger_loss = True
    assert grpo_kwargs(cfg)["use_liger_loss"] is True
```

- [ ] **Step 2: 跑测试确认失败**

```bash
$KERN python -m pytest tests/test_config.py::test_use_liger_loss_passthrough -v
```

期望：FAIL（`TrainConfig` 没有 `use_liger_loss`）。

- [ ] **Step 3: 实现**

在 `src/aha/config.py` 的 `TrainConfig` 里，`scale_rewards` 一行之后加：

```python
    use_liger_loss: Optional[bool] = None  # TRL GRPOConfig.use_liger_loss (fused lm_head + loss); None keeps TRL's default
```

在 `src/aha/train.py` 的 `grpo_kwargs` 里，`scale_rewards` 的 if 之后加：

```python
    if t.use_liger_loss is not None:
        kwargs["use_liger_loss"] = t.use_liger_loss
```

在 `src/aha/logging_utils.py` 的 `on_log` 里，把 `record = ...` 一行改为：

```python
            record = {"step": state.global_step, "time": time.time(), **logs}
            try:
                import torch

                if torch.cuda.is_available():
                    record["gpu_peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
            except Exception:  # noqa: BLE001 - logging must never break training
                pass
```

- [ ] **Step 4: 跑测试确认通过**

```bash
$KERN python -m pytest tests/test_config.py tests/test_logging_and_plotting.py -q
```

期望：全部 PASS。

- [ ] **Step 5: 在隔离环境里装 liger-kernel**

```bash
$KERN pip install "liger-kernel>=0.5.5"
$KERN python -c "import liger_kernel, importlib.metadata as m; print(m.version('liger-kernel')); from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss; print('grpo loss ok')"
```

期望：打印版本号和 `grpo loss ok`。记下版本号。

- [ ] **Step 6: 同 seed 跑 10 步 A/B**

```bash
$KERN python scripts/train.py --config configs/smoke.yaml --set train.max_steps=10 train.save_steps=10 train.output_dir=outputs/smoke_liger_off train.use_liger_loss=false
$KERN python scripts/train.py --config configs/smoke.yaml --set train.max_steps=10 train.save_steps=10 train.output_dir=outputs/smoke_liger_on train.use_liger_loss=true
```

两次各约 3–5 分钟。若第二次在构造 trainer 时报错，把完整错误复制进报告，结论为"与 Unsloth 不兼容"，跳到 Step 8。

- [ ] **Step 7: 对比并写报告**

```bash
$KERN python - <<'EOF'
import json
def load(p): return [json.loads(l) for l in open(p) if l.strip()]
off, on = load("outputs/smoke_liger_off/train_log.jsonl"), load("outputs/smoke_liger_on/train_log.jsonl")
rows = []
for a, b in zip(off, on):
    if "loss" in a and "loss" in b:
        rows.append((a["step"], a["loss"], b["loss"], a["time"], b["time"], a.get("gpu_peak_gb"), b.get("gpu_peak_gb")))
dt_off = [rows[i+1][3]-rows[i][3] for i in range(len(rows)-1)]
dt_on = [rows[i+1][4]-rows[i][4] for i in range(len(rows)-1)]
print("| step | loss off | loss on | diff |")
for r in rows: print(f"| {r[0]} | {r[1]:.4f} | {r[2]:.4f} | {abs(r[1]-r[2]):.2e} |")
import statistics as s
print(f"median step s: off {s.median(dt_off):.1f}, on {s.median(dt_on):.1f}")
print(f"peak GB: off {rows[-1][5]}, on {rows[-1][6]}")
EOF
```

把输出整理进 `docs/kernels/liger_loss.md`，格式：

```markdown
# use_liger_loss（TRL 0.22 + Unsloth）

liger-kernel <版本>。smoke 配置，同 seed，10 步。

| step | loss off | loss on | diff |
|---|---|---|---|
...

- 中位步时：off <x> s，on <y> s
- 峰值显存：off <a> GB，on <b> GB
- 是否真的走了 Liger：<在训练日志里 grep "liger" 的结果，以及 loss 曲线是否逐步相同>

## 结论

<三选一：(1) 生效且 loss 差异 < 1e-3，步时/显存收益为 ...，可以进 fixed 配置；
(2) 开关被 Unsloth 的 compute_loss 覆盖、没有生效（loss 完全相同且显存不变），要自写融合 kernel；
(3) 与 Unsloth 冲突报错：<错误摘要>，要自写融合 kernel。>
```

- [ ] **Step 8: 提交**

```bash
git add src/aha/config.py src/aha/train.py src/aha/logging_utils.py tests/test_config.py docs/kernels/liger_loss.md
git commit -m "Try TRL's fused Liger GRPO loss under Unsloth; log peak GPU memory per step"
```

---

### Task 11: 逐元素 kernel 归因（P4）

**Files:**
- Modify: `src/aha/profiling.py`（追加 `attribute_kernels`、`render_attribution`）
- Modify: `scripts/profile_step.py:64-70`（`--stack`）
- Test: `tests/test_profiling.py`（追加）
- Output: `docs/kernels/attribution.md`

- [ ] **Step 1: 写失败的测试**

追加到 `tests/test_profiling.py`:

```python
def test_attribute_kernels_maps_fill_to_python_frame(tmp_path):
    from aha.profiling import attribute_kernels, render_attribution

    trace = {"traceEvents": [
        # python frames (with_stack=True): outer user frame, inner torch frame
        {"ph": "X", "cat": "python_function", "tid": 1, "ts": 0, "dur": 100,
         "name": "site-packages/peft/tuners/lora/layer.py(400): forward"},
        {"ph": "X", "cat": "python_function", "tid": 1, "ts": 10, "dur": 50,
         "name": "site-packages/torch/nn/functional.py(10): linear"},
        {"ph": "X", "cat": "cpu_op", "tid": 1, "ts": 20, "dur": 5, "name": "aten::fill_", "args": {"External id": 7}},
        {"ph": "X", "cat": "cpu_op", "tid": 1, "ts": 60, "dur": 5, "name": "aten::copy_", "args": {"External id": 8}},
        {"ph": "X", "cat": "kernel", "ts": 1000, "dur": 30.0, "name": "void at::native::vectorized_elementwise_kernel<FillFunctor>", "args": {"External id": 7}},
        {"ph": "X", "cat": "kernel", "ts": 1100, "dur": 10.0, "name": "void at::native::vectorized_elementwise_kernel<bfloat16_copy_kernel>", "args": {"External id": 8}},
        {"ph": "X", "cat": "kernel", "ts": 1200, "dur": 500.0, "name": "ampere_bf16_gemm", "args": {"External id": 9}},
    ]}
    p = tmp_path / "t.json"
    p.write_text(json.dumps(trace))
    rows = attribute_kernels(p)
    assert rows[0]["op"] == "aten::fill_" and rows[0]["us"] == 30.0
    assert rows[0]["frame"] == "site-packages/peft/tuners/lora/layer.py(400): forward"
    assert rows[1]["op"] == "aten::copy_" and rows[1]["frame"].endswith("layer.py(400): forward")
    md = render_attribution(rows, total_us=540.0)
    assert "aten::fill_" in md and "5.6%" in md
```

- [ ] **Step 2: 跑测试确认失败**

```bash
$KERN python -m pytest tests/test_profiling.py::test_attribute_kernels_maps_fill_to_python_frame -v
```

期望：`ImportError`。

- [ ] **Step 3: 实现**

追加到 `src/aha/profiling.py`:

```python
ELEMENTWISE_KERNELS = re.compile(r"elementwise|Fill|copy|Memcpy|Memset|cat_|index|zero", re.I)
TORCH_INTERNAL_FRAME = re.compile(r"/torch/|<built-in|torch/_")


def attribute_kernels(path: Path, kernel_pattern: re.Pattern = ELEMENTWISE_KERNELS, depth: int = 3) -> list[dict]:
    """Map memory-traffic kernels back to the aten op and the innermost non-torch Python frame
    that launched them (needs a trace recorded with ``with_stack=True``).

    Links: kernel.args["External id"] == cpu_op.args["External id"]; the op's Python frames are
    the ``python_function`` events on the same thread whose span encloses the op.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    events = data["traceEvents"] if isinstance(data, dict) else data
    ops: dict[int, dict] = {}
    for e in events:
        if e.get("cat") == "cpu_op" and "External id" in e.get("args", {}):
            ext = e["args"]["External id"]
            if ext not in ops or e.get("dur", 0) < ops[ext].get("dur", 0):  # innermost op
                ops[ext] = e
    frames_by_tid: dict[int, list[dict]] = defaultdict(list)
    for e in events:
        if e.get("cat") == "python_function" and "dur" in e:
            frames_by_tid[e.get("tid")].append(e)

    def frames_for(op: dict) -> list[str]:
        s, t = op["ts"], op["ts"] + op.get("dur", 0)
        enclosing = [f for f in frames_by_tid.get(op.get("tid"), []) if f["ts"] <= s and f["ts"] + f["dur"] >= t]
        enclosing.sort(key=lambda f: f["dur"])  # innermost first
        user = [f["name"] for f in enclosing if not TORCH_INTERNAL_FRAME.search(f["name"])]
        return user[:depth] or [f["name"] for f in enclosing[:1]] or ["<no stack>"]

    agg: dict[tuple, dict] = {}
    for e in events:
        cat = str(e.get("cat", "")).lower()
        if cat not in ("kernel", "gpu_memcpy", "gpu_memset") or not kernel_pattern.search(e.get("name", "")):
            continue
        op = ops.get(e.get("args", {}).get("External id"))
        op_name = op["name"] if op else "<no op>"
        frames = frames_for(op) if op else ["<no stack>"]
        key = (op_name, frames[0])
        row = agg.setdefault(key, {"op": op_name, "frame": frames[0], "stack": " <- ".join(frames), "us": 0.0, "launches": 0,
                                   "kernel": e["name"][:60]})
        row["us"] += float(e.get("dur", 0.0))
        row["launches"] += 1
    return sorted(agg.values(), key=lambda r: -r["us"])


def render_attribution(rows: list[dict], total_us: float, top: int = 30) -> str:
    lines = ["## Memory-traffic kernels by launching code", "",
             "| share of step | ms | launches | aten op | frame (innermost non-torch) |", "|---|---|---|---|---|"]
    for r in rows[:top]:
        lines.append(f"| {r['us'] / total_us:.1%} | {r['us'] / 1e3:.1f} | {r['launches']} | `{r['op']}` | `{r['frame']}` |")
    lines += ["", "Full stacks (top 10):", ""]
    for r in rows[:10]:
        lines.append(f"- `{r['op']}` {r['us'] / 1e3:.1f} ms: {r['stack']}")
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: 跑测试确认通过**

```bash
$KERN python -m pytest tests/test_profiling.py -v
```

期望：全部 PASS。

- [ ] **Step 5: 给 `profile_step.py` 加 `--stack`**

在 `scripts/profile_step.py` 里：

1. `profile_training` 的签名加参数 `stack: bool = False`，`profile(...)` 调用改为：

```python
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=stack, with_stack=stack) as prof:
```

2. 在 `prof.export_chrome_trace(...)` 之后加：

```python
    if stack:
        from aha.profiling import attribute_kernels, render_attribution

        rows = attribute_kernels(out / "train_trace.json")
        total = sum(us for _, us in kernels_from_chrome_trace(out / "train_trace.json"))
        attribution_md = render_attribution(rows, total_us=total)
        Path("docs/kernels").mkdir(parents=True, exist_ok=True)
        Path("docs/kernels/attribution.md").write_text(
            "# fill / copy / cast kernels in one training step, attributed to code\n\n" + attribution_md, encoding="utf-8")
        print(attribution_md)
```

3. argparse 加 `p.add_argument("--stack", action="store_true", help="record Python stacks and write docs/kernels/attribution.md")`，并把 `args.stack` 传给 `profile_training`（找到 `main()` 里调用 `profile_training(` 的位置，追加关键字参数 `stack=args.stack`）。

- [ ] **Step 6: 跑归因（需要空闲 GPU，约 5 分钟；trace 会到 100 MB 以上）**

```bash
$KERN python scripts/profile_step.py --config configs/qwen2.5-3b_3060.yaml --out results/profile_stack --stack
```

期望：`docs/kernels/attribution.md` 前几行能把 FillFunctor / bf16 copy / DtoD memcpy 归到具体的 `.py(line): func`，前 10 行覆盖这些 kernel ≥ 80% 的时间。在文件末尾手写 "修复建议" 段：每一行给出"这个 copy/fill 能否消除、怎么消除（LoRA 权重改 bf16 / 预分配缓冲 / 去掉 .contiguous()）、预计省多少 ms"。

- [ ] **Step 7: 提交**

```bash
git add src/aha/profiling.py scripts/profile_step.py tests/test_profiling.py docs/kernels/attribution.md
git commit -m "Attribute fill/copy/cast kernels to launching Python frames"
```

---

### Task 12: fp16 累加 GEMM 的可行性（P3，只做 bench）

**Files:**
- Create: `src/aha/kernels/fp16acc_gemm.py`
- Create: `bench/bench_fp16acc_gemm.py`
- Test: `tests/test_nf4_gemm_gpu.py`（追加一条）
- Output: `docs/kernels/fp16acc_gemm.md`

- [ ] **Step 1: 写失败的测试**

追加到 `tests/test_nf4_gemm_gpu.py`:

```python
@pytest.mark.parametrize("promote_every", [1, 4])
def test_fp16acc_gemm_close_to_fp32(promote_every):
    from aha.kernels.fp16acc_gemm import gemm_fp16acc

    torch.manual_seed(0)
    a = (torch.randn(256, 2048, device="cuda") * 0.5).half()
    b = (torch.randn(2048, 512, device="cuda") * 0.02).half()
    ref = a.float() @ b.float()
    out = gemm_fp16acc(a, b, promote_every=promote_every)
    assert out.dtype == torch.float16
    assert _rel(out, ref) < 5e-3
```

- [ ] **Step 2: 跑测试确认失败**

```bash
$KERN python -m pytest tests/test_nf4_gemm_gpu.py -k fp16acc -v
```

期望：`ImportError`。

- [ ] **Step 3: 实现 kernel**

`src/aha/kernels/fp16acc_gemm.py`:

```python
"""GEMM with fp16 accumulators, promoted to fp32 every ``promote_every`` K-blocks.

GeForce Ampere issues fp16 tensor-core MMAs at twice the rate when the accumulator is fp16.
cuBLAS (through PyTorch) always accumulates in fp32, so the only way to measure the gain is a
kernel of our own. Numerics: an fp16 partial sum over PROMOTE_EVERY*BLOCK_K products of
O(0.5 * 0.02) magnitude stays far from overflow; precision loss is bounded by the chunk length.
This file is benchmark-only: nothing in training calls it.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gemm_chunked_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                         stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                         PROMOTE_EVERY: tl.constexpr, ACC_FP16: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc32 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    CHUNK: tl.constexpr = PROMOTE_EVERY * BLOCK_K
    for kc in range(0, K, CHUNK):
        if ACC_FP16:
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float16)
        else:
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for kk in range(0, CHUNK, BLOCK_K):
            k = kc + kk
            a = tl.load(a_ptrs + k * stride_ak, mask=(offs_m[:, None] < M) & (offs_k[None, :] + k < K), other=0.0)
            b = tl.load(b_ptrs + k * stride_bk, mask=(offs_k[:, None] + k < K) & (offs_n[None, :] < N), other=0.0)
            if ACC_FP16:
                acc = tl.dot(a, b, acc, out_dtype=tl.float16)
            else:
                acc = tl.dot(a, b, acc)
        acc32 += acc.to(tl.float32)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc32.to(tl.float16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def gemm_fp16acc(a: torch.Tensor, b: torch.Tensor, promote_every: int = 4, acc_fp16: bool = True,
                 block_m: int = 128, block_n: int = 128, block_k: int = 32, num_warps: int = 4, num_stages: int = 3) -> torch.Tensor:
    """a [M, K] fp16 @ b [K, N] fp16 -> [M, N] fp16. ``acc_fp16=False`` gives the same kernel with fp32
    accumulation, to separate Triton's codegen gap from the accumulate-rate gain."""
    assert a.dtype == torch.float16 and b.dtype == torch.float16 and a.shape[1] == b.shape[0]
    M, K = a.shape
    _, N = b.shape
    chunk = promote_every * block_k
    if K % chunk != 0:
        raise ValueError(f"K={K} must be a multiple of promote_every*block_k={chunk}")
    c = torch.empty((M, N), dtype=torch.float16, device=a.device)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    _gemm_chunked_kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
                               BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, PROMOTE_EVERY=promote_every,
                               ACC_FP16=acc_fp16, num_warps=num_warps, num_stages=num_stages)
    return c
```

- [ ] **Step 4: 跑测试确认通过**

```bash
$KERN python -m pytest tests/test_nf4_gemm_gpu.py -k fp16acc -v
```

期望：2 passed。若 Triton 3.4 拒绝 `out_dtype=tl.float16`（报 `tl.dot` 不支持的累加类型），在报告里记录这一点，并改用 `tl.dot(a, b).to(tl.float16)` 累加到 fp16 的 `acc += ...` 形式重测（这会失去寄存器级 fp16 累加，收益可能消失，这本身就是结论的一部分）。

- [ ] **Step 5: 写 benchmark**

`bench/bench_fp16acc_gemm.py`:

```python
#!/usr/bin/env python
"""Is fp16 accumulation worth it on a GeForce card? Triton GEMM with fp16 vs fp32 accumulators vs
cuBLAS, at training shapes, with error against an fp32 product.

    python bench/bench_fp16acc_gemm.py --out docs/kernels/fp16acc_gemm.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bench.harness import assert_gpu_free, rel_fro_error, render_table, time_ms, write_report  # noqa: E402

M = 2400
SHAPES = [("q/o_proj", 2048, 2048), ("gate_up", 2048, 11008), ("down", 11008, 2048)]  # (name, K, N)


def run(out_path: str, force: bool) -> None:
    assert_gpu_free(force)
    import torch

    from aha.kernels.fp16acc_gemm import gemm_fp16acc
    from aha.kernels.roofline import achieved, gemm_flops, utilisation

    torch.manual_seed(0)
    rows = []
    for name, K, N in SHAPES:
        a32 = torch.randn(M, K, device="cuda") * 0.5
        b32 = torch.randn(K, N, device="cuda") * 0.02
        ref = a32 @ b32
        a, b = a32.half(), b32.half()
        flops = gemm_flops(M, N, K)
        cands = {
            "cuBLAS fp16 (fp32 acc)": lambda i: a @ b,
            "triton fp32 acc": lambda i: gemm_fp16acc(a, b, acc_fp16=False),
            "triton fp16 acc, promote 1": lambda i: gemm_fp16acc(a, b, promote_every=1),
            "triton fp16 acc, promote 4": lambda i: gemm_fp16acc(a, b, promote_every=4),
            "triton fp16 acc, promote 8": lambda i: gemm_fp16acc(a, b, promote_every=8),
        }
        for cname, fn in cands.items():
            out = fn(0)
            ms = time_ms(fn, iters=30, warmup=5)
            ach = achieved((M * K + K * N + M * N) * 2, flops, ms / 1e3)
            rows.append({"shape": f"{name} K{K} N{N}", "kernel": cname, "ms": ms, "TFLOPS": ach["tflops"],
                         "fp32acc%": utilisation(ach)["compute"], "fp16acc%": utilisation(ach, accumulate="fp16")["compute"],
                         "rel_err": rel_fro_error(out, ref),
                         "cos": torch.nn.functional.cosine_similarity(out.float().flatten(), ref.flatten(), dim=0).item()})
            print(rows[-1])
    cols = ["shape", "kernel", "ms", "TFLOPS", "fp32acc%", "fp16acc%", "rel_err", "cos"]
    header = ("# fp16-accumulate GEMM feasibility (RTX 3060)\n\n"
              "fp32acc% = share of the 25.6 TFLOPS fp32-accumulate peak; fp16acc% = share of the 51.2 TFLOPS fp16-accumulate peak. "
              "cos = cosine similarity to the fp32 product (stands in for gradient direction agreement).")
    write_report(out_path, [render_table(rows, cols, "all shapes")], header)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="docs/kernels/fp16acc_gemm.md")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    run(a.out, a.force)
```

- [ ] **Step 6: 跑并写 go / no-go**

```bash
$KERN python bench/bench_fp16acc_gemm.py --out docs/kernels/fp16acc_gemm.md
```

在 `docs/kernels/fp16acc_gemm.md` 末尾追加：

```markdown
## 结论

- Triton fp32 累加 vs cuBLAS：<x>%（Triton 代码生成本身的差距）。
- fp16 累加 vs 同 kernel 的 fp32 累加：<y> 倍；promote_every=<n> 时 rel_err <e>，cos <c>。
- **Go / No-go：** <go 的条件：fp16 累加 ≥ 1.4× cuBLAS 且 rel_err ≤ 2× cuBLAS 自身误差、cos ≥ 0.9999。满足则下一步计划：
  把底座前向 / 反向的四个大 GEMM 接到这个 kernel 上，模型 dtype 改 fp16（Unsloth `dtype=torch.float16`，带 loss scaling），
  在 smoke 配置上对比 reward 曲线。不满足则记录原因，关闭 P3。>
```

- [ ] **Step 7: 提交**

```bash
git add src/aha/kernels/fp16acc_gemm.py bench/bench_fp16acc_gemm.py tests/test_nf4_gemm_gpu.py docs/kernels/fp16acc_gemm.md
git commit -m "Benchmark fp16-accumulate Triton GEMM against cuBLAS; record go/no-go"
```

---

### Task 13: 汇总报告

**Files:**
- Create: `docs/results_2026-10-27_kernels.md`（日期按实际完成日）
- Modify: `docs/kernels/README.md`

- [ ] **Step 1: 写汇总**

`docs/results_<date>_kernels.md`，中文，结构固定：

```markdown
# 3060 上 GRPO 的 kernel 级优化：结果

## 1. 一句话结论
<decode 提速倍数、训练半步各项的结论>

## 2. 基线（P0）
<引用 docs/kernels/gemm_4bit.md、cublas_bf16.md 的关键行；fp32 累加峰值的更正>

## 3. Triton NF4 GEMM（P1）
<microbench 带宽%、vLLM decode ms/步前后、贪心一致率；还剩什么没做（LoRA epilogue 融合、嵌套 absmax 在 kernel 内解）>

## 4. use_liger_loss（P2）
<三选一的结论>

## 5. 逐元素 kernel 归因（P4）
<前 5 行 + 修复建议>

## 6. fp16 累加 GEMM（P3）
<go / no-go>

## 7. 下一步
<按收益排序>
```

- [ ] **Step 2: 更新 README 索引并提交**

在 `docs/kernels/README.md` 末尾列出每个报告文件和一句话。

```bash
git add docs/results_*_kernels.md docs/kernels/README.md
git commit -m "Summarise kernel-level optimisation results"
```

- [ ] **Step 3: 把补丁默认接进训练（只在 P1 验收通过时）**

若 Task 9 的 decode 提速 ≥ 30% 且一致率 ≥ 95%：在 `scripts/train.py` 和 `scripts/eval.py` 的 `import unsloth` 之前加

```python
if os.environ.get("AHA_TRITON_NF4", "0") == "1":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from aha.kernels.nf4_gemm import patch_bitsandbytes

    patch_bitsandbytes()
```

默认关闭（环境变量开），这样现有实验的可复现性不受影响；在汇总报告"下一步"里写明：下一次完整 500 步训练用 `AHA_TRITON_NF4=1` 跑，和 fixed 比步时和 GSM8K 精度。

```bash
git add scripts/train.py scripts/eval.py
git commit -m "Opt-in AHA_TRITON_NF4=1 to use the Triton NF4 GEMM during training and eval"
```

---

## 自检记录

- Spec 覆盖：P0 → Task 2/4/5/6；P1 → Task 7/8/9；P2 → Task 10；P4 → Task 11；P3 → Task 12；环境隔离 → Task 1；汇总 → Task 13。
- 名字一致性：`nf4_gemm(x, packed, absmax16, N, K)`、`absmax_fp16_for(quant_state)`、`patch_bitsandbytes()` / `unpatch_bitsandbytes()`、`_ORIGINAL_MATMUL_4BIT`、`_patched_matmul_4bit`、`dequant_nf4_reference(packed, absmax, N, K)`、`resolve_absmax(quant_state)`、`gemm_fp16acc(a, b, promote_every, acc_fp16)`、`attribute_kernels(path)` / `render_attribution(rows, total_us)`、`render_table(rows, columns, title)`、`time_ms(fn, iters, warmup)` 在各任务中一致。
- 已知不确定点（执行时按步骤里的分支处理）：nibble 高低位顺序（Task 3 Step 4）、Triton 在 CUDA graph 捕获期编译（Task 9 Step 2）、`tl.dot` 的 fp16 `out_dtype` 支持（Task 12 Step 4）、Unsloth 是否覆盖 `use_liger_loss`（Task 10 Step 7）。
