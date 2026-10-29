import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from aha.data_night import analyze, group_signals, paired_comparison, validate_run


def predictions(values):
    return [{"idx": i, "gold": str(i), "correct": v} for i, v in enumerate(values)]


def test_pairing_uses_ids_and_exact_mcnemar():
    ref = predictions([False] * 5 + [True])
    new = predictions([True] * 5 + [False])
    result = paired_comparison(ref, list(reversed(new)), expected_samples=6, bootstrap=100)
    assert result["candidate_only"] == 5
    assert result["reference_only"] == 1
    assert result["delta"] == pytest.approx(4 / 6)
    assert result["mcnemar_p"] == pytest.approx(14 / 64)
    assert result == paired_comparison(ref, new, expected_samples=6, bootstrap=100)


def test_identical_predictions_have_zero_interval():
    rows = predictions([True, False, True])
    result = paired_comparison(rows, rows, expected_samples=3, bootstrap=30)
    assert result["ci95"] == [0, 0]
    assert result["mcnemar_p"] == 1


@pytest.mark.parametrize("problem", ["missing", "duplicate", "gold", "bool", "ids"])
def test_rejects_invalid_pairing(problem):
    ref = predictions([True, False])
    new = predictions([True, False])
    if problem == "missing":
        new.pop()
    elif problem == "duplicate":
        new[1]["idx"] = 0
    elif problem == "gold":
        new[0]["gold"] = "different"
    elif problem == "bool":
        new[0]["correct"] = "false"
    else:
        ref[1]["idx"] = new[1]["idx"] = 10
    with pytest.raises(ValueError):
        paired_comparison(ref, new, expected_samples=2)


def records():
    return [{"step": i + 1, "rewards/correctness_reward_func/mean": value,
             "frac_reward_zero_std": zero, "completion_length": 200}
            for i, (value, zero) in enumerate([(2, 1), (0, 1), (1, 0), (2, 0)])]


def test_group_rates_ignore_final_trainer_summary_and_use_step_windows():
    result = group_signals(records() + [{"step": 4, "train_runtime": 10}], window=2)
    assert result["all_right"] == 0.5
    assert result["all_wrong"] == 0.25
    assert result["mixed"] == 0.25
    assert result["zero_std"] == 0.5
    assert result["windows"][0]["all_right"] == 0.5
    assert result["windows"][1]["mixed"] == 0.5


def make_run(root):
    train_dir = root / "outputs" / "run"
    eval_dir = root / "results" / "run" / "gsm8k_test"
    train_dir.mkdir(parents=True)
    eval_dir.mkdir(parents=True)
    (train_dir / "log_history.json").write_text(json.dumps(records()))
    (train_dir / "config.yaml").write_text(
        "train:\n  per_device_train_batch_size: 1\n  gradient_accumulation_steps: 1\n"
        "  max_steps: 4\n  num_generations: 6\n"
    )
    (train_dir / "lora").mkdir()
    (train_dir / "lora" / "adapter_model.safetensors").write_bytes(b"test")
    rows = [{**p, "output": "<reasoning>" + "word " * 40 + "</reasoning><answer>1</answer>",
             "completion_tokens": 100} for p in predictions([True, False])]
    (eval_dir / "predictions.jsonl").write_text("\n".join(map(json.dumps, rows)))
    (eval_dir / "metrics.json").write_text(json.dumps({"samples": 2, "accuracy": 0.5}))
    return train_dir, eval_dir


def test_complete_run_is_validated(tmp_path):
    make_run(tmp_path)
    assert len(validate_run(tmp_path, "run", expected_samples=2, expected_steps=4)["predictions"]) == 2


@pytest.mark.parametrize("problem", ["steps", "metrics", "adapter", "batch"])
def test_rejects_partial_or_inconsistent_runs(tmp_path, problem):
    train_dir, eval_dir = make_run(tmp_path)
    if problem == "steps":
        (train_dir / "log_history.json").write_text(json.dumps(records()[:-1]))
    elif problem == "metrics":
        (eval_dir / "metrics.json").write_text(json.dumps({"samples": 2, "accuracy": 1.0}))
    elif problem == "adapter":
        (train_dir / "lora" / "adapter_model.safetensors").unlink()
    else:
        cfg = train_dir / "config.yaml"
        cfg.write_text(cfg.read_text().replace("per_device_train_batch_size: 1", "per_device_train_batch_size: 2"))
    with pytest.raises((ValueError, FileNotFoundError)):
        validate_run(tmp_path, "run", expected_samples=2, expected_steps=4)


def make_night(root):
    import shutil

    train_dir, eval_dir = make_run(root)
    for name in ("qwen3b_fixed", "qwen3b_filtered", "qwen3b_filtered_gate"):
        shutil.copytree(train_dir, root / "outputs" / name)
        shutil.copytree(eval_dir.parent, root / "results" / name)
    presample = root / "outputs" / "presample" / "gsm8k_instruct_g6.jsonl"
    presample.parent.mkdir()
    presample.write_text("\n".join(json.dumps({"idx": i, "passes": i % 7, "n": 6}) for i in range(70)))


def test_final_analysis_contains_both_comparisons_and_limitations(tmp_path):
    make_night(tmp_path)
    report, data = analyze(tmp_path, expected_samples=2, expected_steps=4, bootstrap=30)
    assert set(data["comparisons"]) == {"A_vs_fixed", "B_vs_A"}
    assert data["presample"]["kept"] == 50
    assert "McNemar" in report and "bootstrap" in report
    assert "单 seed" in report and "门控" in report and "尚无显著提升证据" in report
    assert "| 1–100 |" in report


def test_failed_analysis_does_not_return_a_final_report(tmp_path):
    make_night(tmp_path)
    (tmp_path / "results/qwen3b_filtered_gate/gsm8k_test/predictions.jsonl").unlink()
    with pytest.raises(FileNotFoundError):
        analyze(tmp_path, expected_samples=2, expected_steps=4, bootstrap=30)


def make_full_night(root):
    make_night(root)
    for name in ("qwen3b_fixed", "qwen3b_filtered", "qwen3b_filtered_gate"):
        out = root / "outputs" / name
        rows = [{**records()[i % 4], "step": i + 1} for i in range(500)]
        (out / "log_history.json").write_text(json.dumps(rows))
        cfg = out / "config.yaml"
        cfg.write_text(cfg.read_text().replace("max_steps: 4", "max_steps: 500"))
        result = root / "results" / name / "gsm8k_test"
        preds = [{**p, "output": "<reasoning>" + "word " * 40 + "</reasoning><answer>1</answer>",
                  "completion_tokens": 100} for p in predictions([i % 2 == 0 for i in range(1319)])]
        (result / "predictions.jsonl").write_text("\n".join(map(json.dumps, preds)))
        (result / "metrics.json").write_text(json.dumps({"samples": 1319, "accuracy": 660 / 1319}))


def test_cli_writes_verified_report_figures_and_success_marker(tmp_path):
    make_full_night(tmp_path)
    script = Path(__file__).resolve().parents[1] / "scripts/analyze_data_night.py"
    result = subprocess.run([sys.executable, str(script), "--root", str(tmp_path)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    status = json.loads((tmp_path / "results/data_night_analysis_status.json").read_text())
    assert status["status"] == "complete"
    data = json.loads((tmp_path / "results/data_night_analysis.json").read_text())
    assert len(data["input_sha256"]) == 13
    assert (tmp_path / "logs/data_night_analysis.DONE").exists()
    report = (tmp_path / "docs/results_2026-10-28_data_night.md").read_text(encoding="utf-8")
    assert "已完成输入核验" in report and "1–100" in report
    for stem in ("data_night_delta_ci", "data_night_group_signal"):
        for suffix in ("png", "pdf"):
            assert (tmp_path / f"report/img/curves/data_night/{stem}.{suffix}").stat().st_size > 1000


def test_cli_failure_removes_stale_success_and_preserves_existing_report(tmp_path):
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs/data_night_analysis.DONE").touch()
    (tmp_path / "docs").mkdir()
    report = tmp_path / "docs/results_2026-10-28_data_night.md"
    report.write_text("pending")
    script = Path(__file__).resolve().parents[1] / "scripts/analyze_data_night.py"
    result = subprocess.run([sys.executable, str(script), "--root", str(tmp_path)],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode != 0
    assert json.loads((tmp_path / "results/data_night_analysis_status.json").read_text())["status"] == "failed"
    assert not (tmp_path / "logs/data_night_analysis.DONE").exists()
    assert report.read_text() == "pending"


@pytest.mark.skipif(sys.platform != "linux", reason="WSL/Linux queue watcher")
@pytest.mark.parametrize("with_marker", [False, True])
def test_finisher_requires_queue_marker_and_validates_results(tmp_path, with_marker):
    make_full_night(tmp_path)
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    repo = Path(__file__).resolve().parents[1]
    for name in ("finish_data_night.sh", "analyze_data_night.py"):
        shutil.copy2(repo / "scripts" / name, scripts / name)
    (tmp_path / "src").symlink_to(repo / "src", target_is_directory=True)
    (tmp_path / "logs").mkdir()
    if with_marker:
        (tmp_path / "logs/data_night.DONE").touch()
    result = subprocess.run(["bash", str(scripts / "finish_data_night.sh"), "1073741824"],
                            env={**os.environ, "AHA_PYTHON": sys.executable},
                            capture_output=True, text=True, timeout=60)
    assert (result.returncode == 0) == with_marker, result.stdout + result.stderr
    assert (tmp_path / "logs/data_night_analysis.DONE").exists() == with_marker
    status = json.loads((tmp_path / "results/data_night_analysis_status.json").read_text())
    assert status["status"] == ("complete" if with_marker else "failed")
