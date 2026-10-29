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
