"""Experiment configuration: dataclasses + YAML loading with inheritance and overrides.

YAML files may contain a ``_base_: other.yaml`` key (path relative to the file);
the base is loaded first and the child is deep-merged on top. Command-line
overrides use dotted keys, e.g. ``train.max_steps=10``; values are parsed as YAML.
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Iterable, Optional

import yaml


@dataclass
class ModelConfig:
    name: str = "unsloth/Qwen2.5-3B-Instruct"
    max_seq_length: int = 1024
    load_in_4bit: bool = True
    fast_inference: bool = True  # vLLM co-located with the trainer
    gpu_memory_utilization: float = 0.55  # vLLM share of GPU memory
    lora_rank: int = 32
    lora_alpha: Optional[int] = None  # defaults to lora_rank
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
        ]
    )
    gradient_checkpointing: str = "unsloth"


@dataclass
class DataConfig:
    train_dataset: str = "gsm8k"  # see aha.data.TRAIN_LOADERS
    train_split: str = "train"
    max_train_samples: Optional[int] = None
    shuffle_seed: Optional[int] = None


@dataclass
class LengthRewardConfig:
    """Length-based reward shaping R_len from the report (Section 3.3)."""

    enabled: bool = False
    alpha: float = 0.5  # bonus weight when the answer is wrong
    beta: float = 0.1  # penalty weight when the answer is right
    l_max: int = 512  # truncation threshold L_max in tokens


@dataclass
class RewardConfig:
    correctness_weight: float = 2.0
    int_weight: float = 0.5
    strict_format_weight: float = 0.5
    soft_format_weight: float = 0.5
    xmlcount_weight: float = 1.0
    match: str = "exact"  # 'exact' (notebook) or 'relaxed' (eval-style)
    log_samples: bool = True  # print one (question, answer, response) per step
    # Format rewards (int / strict / soft / xmlcount) are paid only when the <reasoning>
    # block has at least this many whitespace tokens; 0 disables the gate. Blocks the
    # "empty envelope" exploit where the policy shrinks reasoning to farm format points.
    min_reasoning_words: int = 0
    length: LengthRewardConfig = field(default_factory=LengthRewardConfig)


@dataclass
class TrainConfig:
    output_dir: str = "outputs/run"
    lora_save_dir: Optional[str] = None  # defaults to <output_dir>/lora
    learning_rate: float = 5e-6
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    weight_decay: float = 0.1
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    optim: str = "paged_adamw_8bit"
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    num_generations: int = 6
    max_prompt_length: int = 256
    max_completion_length: Optional[int] = None  # defaults to max_seq_length - max_prompt_length
    max_steps: int = 250
    save_steps: int = 250
    logging_steps: int = 1
    max_grad_norm: float = 0.1
    kl_beta: Optional[float] = None  # TRL GRPOConfig.beta; None keeps TRL's default (0.0 in 0.22!)
    loss_type: Optional[str] = None  # grpo | bnpo | dr_grpo | dapo; None keeps TRL's default (dapo)
    scale_rewards: Optional[str] = None  # group | batch | none; None keeps TRL's default (group = divide by group std)
    use_liger_loss: Optional[bool] = None  # TRL GRPOConfig.use_liger_loss (fused lm_head + loss); None keeps TRL's default
    seed: int = 3407
    report_to: str = "none"
    resume_from_checkpoint: Optional[str] = None


@dataclass
class SFTTrainConfig:
    """Supervised fine-tuning baseline on gold GSM8K solutions rewritten in the XML format."""

    output_dir: str = "outputs/sft"
    lora_save_dir: Optional[str] = None  # defaults to <output_dir>/lora
    learning_rate: float = 2e-4  # usual QLoRA SFT rate; GRPO uses 5e-6
    num_train_epochs: float = 1.0
    max_steps: int = -1  # >0 overrides num_train_epochs
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_length: int = 1024
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    optim: str = "adamw_8bit"
    weight_decay: float = 0.01
    logging_steps: int = 10
    save_steps: int = 0  # 0 = only save the adapter at the end
    seed: int = 3407
    report_to: str = "none"
    completion_only_loss: bool = True  # loss on the assistant turn only


@dataclass
class EvalConfig:
    datasets: list[str] = field(default_factory=lambda: ["gsm8k_test"])
    lora_path: Optional[str] = None  # None evaluates the untouched instruct model
    max_samples: Optional[int] = None
    batch_size: int = 8
    temperature: float = 0.2
    top_p: float = 0.95
    max_tokens: int = 512
    seed: int = 3407
    results_dir: str = "results"
    match: str = "relaxed"
    num_examples_to_keep: int = 5


@dataclass
class ExperimentConfig:
    run_name: str = "run"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    sft: SFTTrainConfig = field(default_factory=SFTTrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # --- derived values -------------------------------------------------
    @property
    def sft_lora_save_dir(self) -> str:
        return self.sft.lora_save_dir or str(Path(self.sft.output_dir) / "lora")

    @property
    def lora_alpha(self) -> int:
        if self.model.lora_alpha is not None:
            return self.model.lora_alpha
        return self.model.lora_rank

    @property
    def max_completion_length(self) -> int:
        if self.train.max_completion_length is not None:
            return self.train.max_completion_length
        return self.model.max_seq_length - self.train.max_prompt_length

    @property
    def lora_save_dir(self) -> str:
        return self.train.lora_save_dir or str(Path(self.train.output_dir) / "lora")

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def deep_merge(base: dict, override: dict) -> dict:
    """Return a new dict with ``override`` merged recursively onto ``base``."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _load_yaml_with_base(path: Path, _seen: Optional[set] = None) -> dict:
    path = path.resolve()
    _seen = _seen or set()
    if path in _seen:
        raise ValueError(f"circular _base_ reference at {path}")
    _seen.add(path)
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a mapping at top level")
    base_ref = raw.pop("_base_", None)
    if base_ref is None:
        return raw
    base = _load_yaml_with_base(path.parent / base_ref, _seen)
    return deep_merge(base, raw)


def parse_overrides(items: Iterable[str]) -> dict:
    """Turn ``["train.max_steps=10", "model.name=foo"]`` into a nested dict."""
    result: dict = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"override must look like key=value, got {item!r}")
        key, raw_value = item.split("=", 1)
        value = yaml.safe_load(raw_value) if raw_value != "" else ""
        node = result
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return result


def _from_dict(cls, data: dict, path: str = ""):
    if not isinstance(data, dict):
        raise TypeError(f"{path or cls.__name__}: expected a mapping, got {type(data).__name__}")
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(f"{path or cls.__name__}: unknown keys {sorted(unknown)}")
    kwargs = {}
    for name, f in known.items():
        if name not in data:
            continue
        value = data[name]
        sub_path = f"{path}.{name}" if path else name
        sub_cls = _NESTED.get(name)
        if sub_cls is not None and isinstance(value, dict):
            kwargs[name] = _from_dict(sub_cls, value, sub_path)
        elif sub_cls is not None:
            raise TypeError(f"{sub_path}: expected a mapping")
        else:
            kwargs[name] = value
    return cls(**kwargs)


# `from __future__ import annotations` turns field types into strings, so nested
# dataclasses are resolved by field name instead of by type.
_NESTED = {
    "model": ModelConfig,
    "data": DataConfig,
    "reward": RewardConfig,
    "train": TrainConfig,
    "sft": SFTTrainConfig,
    "eval": EvalConfig,
    "length": LengthRewardConfig,
}


def config_from_dict(data: dict) -> ExperimentConfig:
    return _from_dict(ExperimentConfig, data)


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> ExperimentConfig:
    """Load a YAML config (with ``_base_`` inheritance) and apply dotted overrides."""
    data = _load_yaml_with_base(Path(path))
    data = deep_merge(data, parse_overrides(overrides))
    return config_from_dict(data)


def dump_config(cfg: ExperimentConfig, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)
