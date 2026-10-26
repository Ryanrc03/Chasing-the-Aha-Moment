from pathlib import Path

import pytest
import yaml

from aha.config import (
    ExperimentConfig,
    config_from_dict,
    deep_merge,
    dump_config,
    load_config,
    parse_overrides,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_defaults_and_derived_values():
    cfg = ExperimentConfig()
    assert cfg.lora_alpha == cfg.model.lora_rank
    assert cfg.max_completion_length == 1024 - 256
    assert cfg.lora_save_dir.replace("\\", "/") == "outputs/run/lora"


def test_parse_overrides_types():
    d = parse_overrides(["train.max_steps=10", "reward.length.enabled=true",
                         "model.name=foo/bar", "eval.lora_path=null", "eval.datasets=[a, b]"])
    assert d["train"]["max_steps"] == 10
    assert d["reward"]["length"]["enabled"] is True
    assert d["model"]["name"] == "foo/bar"
    assert d["eval"]["lora_path"] is None
    assert d["eval"]["datasets"] == ["a", "b"]
    with pytest.raises(ValueError):
        parse_overrides(["no_equals_sign"])


def test_deep_merge_does_not_mutate():
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    out = deep_merge(base, {"a": {"b": 9}})
    assert out == {"a": {"b": 9, "c": 2}, "d": 3}
    assert base["a"]["b"] == 1


def test_unknown_key_rejected():
    with pytest.raises(ValueError, match="unknown keys"):
        config_from_dict({"train": {"max_step": 1}})


@pytest.mark.parametrize("name", sorted(p.name for p in CONFIGS.glob("*.yaml")))
def test_all_shipped_configs_load(name):
    cfg = load_config(CONFIGS / name)
    assert cfg.train.max_prompt_length < cfg.model.max_seq_length
    assert cfg.reward.match in {"exact", "relaxed"}


def test_inheritance_chain_and_overrides():
    cfg = load_config(CONFIGS / "qwen2.5-3b_3060_shaped.yaml", ["train.max_steps=3"])
    assert cfg.run_name == "qwen3b_shaped"
    assert cfg.reward.length.enabled is True
    assert cfg.reward.match == "relaxed"  # inherited from qwen2.5-3b_3060.yaml
    assert cfg.train.learning_rate == 5e-6  # inherited from base.yaml
    assert cfg.train.max_steps == 3
    assert cfg.train.output_dir == "outputs/qwen3b_shaped"


def test_8b_config_reproduces_notebook_settings():
    cfg = load_config(CONFIGS / "llama3.1-8b_t4.yaml")
    assert cfg.model.name == "unsloth/meta-Llama-3.1-8B-Instruct"
    assert cfg.model.gpu_memory_utilization == 0.9
    assert cfg.reward.match == "exact"
    assert cfg.reward.length.enabled is False
    assert cfg.train.num_generations == 6
    assert cfg.max_completion_length == 768


def test_dump_roundtrip(tmp_path):
    cfg = load_config(CONFIGS / "smoke.yaml")
    dump_config(cfg, tmp_path / "out.yaml")
    with open(tmp_path / "out.yaml", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert config_from_dict(data) == cfg
