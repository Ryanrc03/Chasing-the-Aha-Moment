"""Training-curve logging shared by the GRPO and SFT trainers.

Two artefacts per run, both under the run's ``output_dir``:

* ``train_log.jsonl``  - one JSON object per logging step, appended live by a
  TrainerCallback so curves survive a crash or an interrupted run;
* ``log_history.json`` - the trainer's full ``state.log_history`` dumped at the end.

``scripts/plot_curves.py`` turns either file into report figures.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

JSONL_NAME = "train_log.jsonl"
HISTORY_NAME = "log_history.json"


def make_jsonl_callback(output_dir: str | Path):
    """Return a ``transformers.TrainerCallback`` that appends every log dict to JSONL."""
    from transformers import TrainerCallback

    path = Path(output_dir) / JSONL_NAME
    path.parent.mkdir(parents=True, exist_ok=True)

    class JsonlLoggingCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            # a fresh run (not a resume) starts a fresh file
            if state.global_step == 0 and path.exists():
                path.unlink()

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            record = {"step": state.global_step, "time": time.time(), **logs}
            try:
                import torch

                if torch.cuda.is_available():
                    record["gpu_peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
            except Exception:  # noqa: BLE001 - logging must never break training
                pass
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

    return JsonlLoggingCallback()


def save_log_history(trainer, output_dir: str | Path) -> Path:
    path = Path(output_dir) / HISTORY_NAME
    with open(path, "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=1)
    return path


def read_log(run_dir: str | Path) -> list[dict]:
    """Load the curve records of a run, preferring the complete history when present."""
    run_dir = Path(run_dir)
    history = run_dir / HISTORY_NAME
    if history.exists():
        with open(history, encoding="utf-8") as f:
            return [r for r in json.load(f) if "step" in r]
    jsonl = run_dir / JSONL_NAME
    if jsonl.exists():
        with open(jsonl, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    raise FileNotFoundError(f"no {HISTORY_NAME} or {JSONL_NAME} in {run_dir}")
