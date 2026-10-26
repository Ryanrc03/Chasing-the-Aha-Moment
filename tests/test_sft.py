from pathlib import Path

from aha.config import load_config
from aha.data import format_gsm8k_sft_row, gsm8k_solution_to_xml
from aha.parsing import STRICT_XML_PATTERN, extract_xml_answer
from aha.rewards import count_xml

CONFIGS = Path(__file__).resolve().parents[1] / "configs"

GSM8K_ANSWER = (
    "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n"
    "Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n"
    "#### 72"
)


def test_solution_to_xml_strips_annotations_and_matches_strict_format():
    xml = gsm8k_solution_to_xml(GSM8K_ANSWER)
    assert "<<" not in xml and ">>" not in xml
    assert extract_xml_answer(xml) == "72"
    assert "Natalia sold 48/2 = 24 clips in May." in xml
    # the SFT target should earn every format reward GRPO uses
    assert STRICT_XML_PATTERN.match(xml)
    assert count_xml(xml) == 0.5


def test_solution_without_marker_is_dropped():
    assert gsm8k_solution_to_xml("no final answer here") is None
    row = format_gsm8k_sft_row({"question": "q", "answer": "nothing"})
    assert row["completion"] is None


def test_sft_row_shape():
    row = format_gsm8k_sft_row({"question": "How many?", "answer": GSM8K_ANSWER})
    assert [m["role"] for m in row["prompt"]] == ["system", "user"]
    assert row["prompt"][1]["content"] == "How many?"
    assert row["completion"][0]["role"] == "assistant"
    assert row["completion"][0]["content"].endswith("<answer>\n72\n</answer>\n")


def test_sft_config_inherits_model_and_has_own_output():
    cfg = load_config(CONFIGS / "qwen2.5-3b_3060_sft.yaml")
    assert cfg.model.name == "unsloth/Qwen2.5-3B-Instruct"
    assert cfg.sft.learning_rate == 2e-4
    assert cfg.sft_lora_save_dir.replace("\\", "/") == "outputs/qwen3b_sft/lora"
    assert cfg.eval.lora_path == "outputs/qwen3b_sft/lora"
    # GRPO settings untouched by the SFT section
    assert cfg.train.learning_rate == 5e-6
