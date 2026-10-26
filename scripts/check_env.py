#!/usr/bin/env python
"""Print GPU / package versions so a training environment can be sanity-checked quickly."""

import importlib
import platform
import shutil
import subprocess

PACKAGES = ["torch", "transformers", "trl", "unsloth", "vllm", "peft", "datasets", "bitsandbytes", "yaml"]


def main() -> None:
    print(f"python   : {platform.python_version()} ({platform.system()} {platform.release()})")
    if shutil.which("nvidia-smi"):
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader"],
            capture_output=True, text=True,
        )
        print(f"gpu      : {out.stdout.strip() or out.stderr.strip()}")
    else:
        print("gpu      : nvidia-smi not found")
    for name in PACKAGES:
        try:
            mod = importlib.import_module(name)
            print(f"{name:<9}: {getattr(mod, '__version__', 'ok')}")
        except Exception as exc:  # noqa: BLE001
            print(f"{name:<9}: MISSING ({type(exc).__name__})")
    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            print(f"cuda     : {torch.version.cuda}, free {free / 2**30:.1f} / {total / 2**30:.1f} GiB")
        else:
            print("cuda     : not available to torch")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
