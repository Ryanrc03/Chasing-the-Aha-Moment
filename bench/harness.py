"""Shared microbenchmark plumbing: refuse a busy GPU, time with CUDA events behind an L2 flush,
report as markdown. GPU-free code (parsing, rendering) is unit-tested; timing needs a card.
"""

from __future__ import annotations

import statistics
import subprocess
from pathlib import Path
from typing import Callable, Sequence

USED_MIB_LIMIT = 2500   # WDDM desktop idles around 1 GB on this machine
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
