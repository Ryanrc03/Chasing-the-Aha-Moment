"""Reward functions for GRPO.

Every reward has the TRL signature ``f(prompts, completions, **kwargs) -> list[float]``
where ``completions`` are chat-style (``[[{"role": "assistant", "content": ...}], ...]``)
and dataset columns (here ``answer``) arrive as keyword arguments.

The five base rewards follow @willccbb's GSM8K recipe used in the notebooks. The
length-based reward implements R_len from the report (Section 3.3)::

    R_len = alpha * 1[wrong] * min(len(z)/L_max, 1) - beta * 1[right] * min(len(z)/L_max, 1)

where ``z`` is the <reasoning> segment measured in tokens.
"""

from __future__ import annotations

from typing import Callable, Optional

from aha.config import RewardConfig
from aha.parsing import (
    SOFT_XML_PATTERN,
    STRICT_XML_PATTERN,
    extract_reasoning,
    extract_xml_answer,
    is_correct,
)

RewardFn = Callable[..., list[float]]


def _contents(completions) -> list[str]:
    return [c[0]["content"] for c in completions]


def _named(fn: RewardFn, name: str) -> RewardFn:
    fn.__name__ = name  # TRL logs each reward under the function's __name__
    return fn


# ---------------------------------------------------------------------------
# Base rewards
# ---------------------------------------------------------------------------


def make_correctness_reward(
    weight: float = 2.0, match: str = "exact", log_samples: bool = True
) -> RewardFn:
    def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
        responses = _contents(completions)
        extracted = [extract_xml_answer(r) for r in responses]
        if log_samples:
            q = prompts[0][-1]["content"]
            print(
                "-" * 20,
                f"Question:\n{q}",
                f"\nAnswer:\n{answer[0]}",
                f"\nResponse:\n{responses[0]}",
                f"\nExtracted:\n{extracted[0]}",
            )
        return [weight if is_correct(r, a, match) else 0.0 for r, a in zip(extracted, answer)]

    return _named(correctness_reward_func, "correctness_reward_func")


def make_int_reward(weight: float = 0.5) -> RewardFn:
    def int_reward_func(completions, **kwargs) -> list[float]:
        return [weight if extract_xml_answer(r).isdigit() else 0.0 for r in _contents(completions)]

    return _named(int_reward_func, "int_reward_func")


def make_strict_format_reward(weight: float = 0.5) -> RewardFn:
    def strict_format_reward_func(completions, **kwargs) -> list[float]:
        return [weight if STRICT_XML_PATTERN.match(r) else 0.0 for r in _contents(completions)]

    return _named(strict_format_reward_func, "strict_format_reward_func")


def make_soft_format_reward(weight: float = 0.5) -> RewardFn:
    def soft_format_reward_func(completions, **kwargs) -> list[float]:
        return [weight if SOFT_XML_PATTERN.match(r) else 0.0 for r in _contents(completions)]

    return _named(soft_format_reward_func, "soft_format_reward_func")


def count_xml(text: str) -> float:
    """0.125 per correctly placed tag, minus 0.001 per trailing character after </answer>."""
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count


def make_xmlcount_reward(weight: float = 1.0) -> RewardFn:
    def xmlcount_reward_func(completions, **kwargs) -> list[float]:
        return [weight * count_xml(c) for c in _contents(completions)]

    return _named(xmlcount_reward_func, "xmlcount_reward_func")


# ---------------------------------------------------------------------------
# Length-based reward shaping (report Section 3.3)
# ---------------------------------------------------------------------------


def reasoning_length(text: str, tokenizer=None) -> int:
    """Token length of the <reasoning> segment (whitespace tokens if no tokenizer)."""
    z = extract_reasoning(text)
    if not z:
        return 0
    if tokenizer is None:
        return len(z.split())
    return len(tokenizer(z, add_special_tokens=False)["input_ids"])


def length_reward(
    text: str,
    gold: Optional[str],
    alpha: float,
    beta: float,
    l_max: int,
    match: str = "exact",
    tokenizer=None,
) -> float:
    norm = min(reasoning_length(text, tokenizer) / float(l_max), 1.0)
    correct = is_correct(extract_xml_answer(text), gold, match)
    return -beta * norm if correct else alpha * norm


def make_length_reward(
    alpha: float, beta: float, l_max: int, match: str = "exact", tokenizer=None
) -> RewardFn:
    def length_reward_func(completions, answer, **kwargs) -> list[float]:
        return [
            length_reward(r, a, alpha, beta, l_max, match, tokenizer)
            for r, a in zip(_contents(completions), answer)
        ]

    return _named(length_reward_func, "length_reward_func")


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def gate_by_reasoning(fn: RewardFn, min_words: int) -> RewardFn:
    """Zero a format reward for completions whose <reasoning> has fewer than ``min_words`` words."""
    if min_words <= 0:
        return fn

    def gated(*args, completions, **kwargs) -> list[float]:
        values = fn(*args, completions=completions, **kwargs)
        ok = [len(extract_reasoning(c).split()) >= min_words for c in _contents(completions)]
        return [v if keep else 0.0 for v, keep in zip(values, ok)]

    return _named(gated, fn.__name__)


def format_intact(text: str, min_words: int = 0) -> bool:
    """Soft <reasoning>/<answer> structure, a closed answer tag, and enough reasoning words."""
    return (
        SOFT_XML_PATTERN.match(text) is not None
        and "</answer>" in text
        and len(extract_reasoning(text).split()) >= min_words
    )


def make_gated_correctness_reward(weight: float = 2.0, match: str = "exact", min_words: int = 0,
                                  log_samples: bool = True) -> RewardFn:
    """Correctness that only counts when the format is intact (logged under the usual name)."""
    plain = make_correctness_reward(weight, match, log_samples)

    def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
        values = plain(prompts=prompts, completions=completions, answer=answer, **kwargs)
        return [v if format_intact(c, min_words) else 0.0 for v, c in zip(values, _contents(completions))]

    return _named(correctness_reward_func, "correctness_reward_func")


def make_format_gate_reward(weight: float = 0.25, min_words: int = 0) -> RewardFn:
    """A flat bonus for an intact format: identical for every well-formed answer, so it never
    separates two well-formed answers inside a group."""
    def format_gate_reward_func(completions, **kwargs) -> list[float]:
        return [weight if format_intact(c, min_words) else 0.0 for c in _contents(completions)]

    return _named(format_gate_reward_func, "format_gate_reward_func")


def build_reward_funcs(cfg: RewardConfig, tokenizer=None) -> list[RewardFn]:
    """Reward list in the same order as the notebooks, plus the optional length term."""
    if cfg.format_mode == "gate":
        funcs = [
            make_format_gate_reward(cfg.format_gate_weight, cfg.min_reasoning_words),
            make_gated_correctness_reward(cfg.correctness_weight, cfg.match, cfg.min_reasoning_words, cfg.log_samples),
        ]
        if cfg.length.enabled:
            funcs.append(make_length_reward(cfg.length.alpha, cfg.length.beta, cfg.length.l_max, cfg.match, tokenizer))
        return funcs
    if cfg.format_mode != "graded":
        raise ValueError(f"unknown reward.format_mode {cfg.format_mode!r} (graded | gate)")
    g = cfg.min_reasoning_words
    funcs = [
        gate_by_reasoning(make_xmlcount_reward(cfg.xmlcount_weight), g),
        gate_by_reasoning(make_soft_format_reward(cfg.soft_format_weight), g),
        gate_by_reasoning(make_strict_format_reward(cfg.strict_format_weight), g),
        gate_by_reasoning(make_int_reward(cfg.int_weight), g),
        make_correctness_reward(cfg.correctness_weight, cfg.match, cfg.log_samples),
    ]
    if cfg.length.enabled:
        funcs.append(
            make_length_reward(
                cfg.length.alpha, cfg.length.beta, cfg.length.l_max, cfg.match, tokenizer
            )
        )
    return funcs
