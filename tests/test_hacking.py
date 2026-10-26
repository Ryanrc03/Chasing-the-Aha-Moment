from aha.hacking import inspect_output, output_signals, render_markdown, repetition_ratio, training_signals

GOOD = "<reasoning>\nTwo plus two is four, so the total is 4.\n</reasoning>\n<answer>\n4\n</answer>\n"
EMPTY = "<reasoning>\n\n</reasoning>\n<answer>\n4\n</answer>\n"
MULTI = "<reasoning>x y z w v</reasoning><answer>3</answer><answer>4</answer>"
TRAIL = GOOD + "Let me double check that. Yes."
TRUNC = "<reasoning>\nstill going on and on"
LOOP = "<reasoning>\n" + "go back and check " * 30 + "</reasoning>\n<answer>\n4\n</answer>\n"


def test_repetition_ratio():
    assert repetition_ratio("a b c d e f g h") == 0.0
    assert repetition_ratio("x y z w " * 20) > 0.8


def test_inspect_output_flags():
    assert inspect_output(GOOD) == {
        "soft_format": True, "empty_reasoning": False, "multi_answer": False, "trailing_text": False,
        "truncated": False, "numeric_answer": True, "repetitive": False, "hit_cap": False,
    }
    assert inspect_output(EMPTY)["empty_reasoning"]
    assert inspect_output(MULTI)["multi_answer"]
    assert inspect_output(TRAIL)["trailing_text"]
    assert inspect_output(TRUNC)["truncated"]
    assert inspect_output(LOOP)["repetitive"]
    assert inspect_output(GOOD, completion_tokens=512, max_tokens=512)["hit_cap"]
    assert not inspect_output("<answer>four</answer>")["numeric_answer"]


def pred(output, correct, tokens):
    return {"output": output, "correct": correct, "completion_tokens": tokens, "gold": "4"}


def test_output_signals_flags_length_exploit_and_empty_reasoning():
    preds = [pred(EMPTY, True, 20)] * 3 + [pred(GOOD, False, 400)] * 3
    sig = output_signals(preds)
    assert sig["samples"] == 6 and sig["accuracy"] == 0.5
    assert sig["rates"]["empty_reasoning"] == 0.5
    assert sig["length"]["mean_correct"] == 20 and sig["length"]["mean_wrong"] == 400
    assert any("empty" in f for f in sig["flags"])
    assert any("longer" in f for f in sig["flags"])
    assert output_signals([])["samples"] == 0


def test_training_signals_detects_format_without_correctness():
    recs = []
    for s in range(1, 101):
        recs.append({
            "step": s,
            "rewards/correctness_reward_func/mean": 1.0,
            "rewards/strict_format_reward_func/mean": 0.0 if s < 50 else 0.5,
            "rewards/soft_format_reward_func/mean": 0.1 if s < 50 else 0.5,
            "rewards/xmlcount_reward_func/mean": -0.5 if s < 50 else 0.4,
            "completion_length": 400 - 3 * s,
            "completions/clipped_ratio": 0.0,
        })
    sig = training_signals(recs)
    assert sig["slice"] == 10
    assert sig["curves"]["rewards/correctness_reward_func/mean"]["delta"] == 0.0
    assert any("envelope" in f for f in sig["flags"])
    assert any("collapsed" in f for f in sig["flags"])


def test_training_signals_clean_run_has_no_flags():
    recs = [{"step": s, "rewards/correctness_reward_func/mean": 0.5 + s / 100,
             "rewards/strict_format_reward_func/mean": s / 200,
             "completion_length": 300 + s, "completions/clipped_ratio": 0.05} for s in range(1, 101)]
    assert training_signals(recs)["flags"] == []


def test_render_markdown_smoke():
    recs = [{"step": s, "reward": s / 10, "completion_length": 300} for s in range(1, 21)]
    md = render_markdown("run", training_signals(recs), {"gsm8k_test": output_signals([pred(GOOD, True, 50)])})
    assert "## run" in md and "| reward |" in md and "gsm8k_test" in md
