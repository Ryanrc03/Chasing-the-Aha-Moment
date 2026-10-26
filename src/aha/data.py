"""Dataset loaders.

Training sets return rows with ``prompt`` (chat messages) and ``answer`` (gold string),
which is what TRL's GRPOTrainer expects. Evaluation sets return a list of
:class:`EvalExample`. ``datasets`` is imported lazily.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from aha.parsing import extract_hash_answer
from aha.prompts import SYSTEM_PROMPT, build_chat


@dataclass
class EvalExample:
    prompt: list[dict]
    gold: Optional[str]
    source: str
    idx: int


# ---------------------------------------------------------------------------
# Row formatters (pure functions, unit-testable without `datasets`)
# ---------------------------------------------------------------------------


def format_gsm8k_row(row: dict, system_prompt: str = SYSTEM_PROMPT) -> dict:
    return {
        "prompt": build_chat(row["question"], system_prompt),
        "answer": extract_hash_answer(row["answer"]),
    }


def format_math_row(row: dict, system_prompt: str = SYSTEM_PROMPT) -> dict:
    """MATH-style rows (``problem`` / ``solution`` with a boxed answer)."""
    return {
        "prompt": build_chat(row["problem"], system_prompt),
        "answer": extract_boxed(row["solution"]),
    }


def extract_boxed(solution: str) -> Optional[str]:
    """Content of the last ``\\boxed{...}`` in a MATH solution, handling nested braces."""
    start = solution.rfind("\\boxed{")
    if start == -1:
        return None
    i = start + len("\\boxed{")
    depth = 1
    buf = []
    while i < len(solution):
        ch = solution[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(buf).strip()
        buf.append(ch)
        i += 1
    return None


# ---------------------------------------------------------------------------
# Training sets
# ---------------------------------------------------------------------------


def load_gsm8k_train(split: str = "train", system_prompt: str = SYSTEM_PROMPT):
    from datasets import load_dataset

    data = load_dataset("openai/gsm8k", "main")[split]
    return data.map(
        lambda x: format_gsm8k_row(x, system_prompt), remove_columns=data.column_names
    )


def load_math_train(split: str = "train", system_prompt: str = SYSTEM_PROMPT):
    """MATH (12.5k competition problems) via the ``DigitalLearningGmbH/MATH-lighteval`` copy."""
    from datasets import load_dataset

    data = load_dataset("DigitalLearningGmbH/MATH-lighteval")[split]
    data = data.map(lambda x: format_math_row(x, system_prompt), remove_columns=data.column_names)
    return data.filter(lambda x: x["answer"] is not None)


TRAIN_LOADERS: dict[str, Callable] = {
    "gsm8k": load_gsm8k_train,
    "math": load_math_train,
}


def load_train_dataset(
    name: str,
    split: str = "train",
    max_samples: Optional[int] = None,
    shuffle_seed: Optional[int] = None,
    system_prompt: str = SYSTEM_PROMPT,
):
    if name not in TRAIN_LOADERS:
        raise ValueError(f"unknown train dataset {name!r}; choose from {sorted(TRAIN_LOADERS)}")
    data = TRAIN_LOADERS[name](split=split, system_prompt=system_prompt)
    if shuffle_seed is not None:
        data = data.shuffle(seed=shuffle_seed)
    if max_samples is not None:
        data = data.select(range(min(max_samples, len(data))))
    return data


# ---------------------------------------------------------------------------
# Evaluation sets
# ---------------------------------------------------------------------------


def _examples(
    rows,
    question_key: str,
    gold_fn: Callable[[dict], Optional[str]],
    source: str,
    system_prompt: str,
) -> list[EvalExample]:
    return [
        EvalExample(
            prompt=build_chat(row[question_key], system_prompt),
            gold=gold_fn(row),
            source=source,
            idx=i,
        )
        for i, row in enumerate(rows)
    ]


def load_gsm8k_test(system_prompt: str = SYSTEM_PROMPT) -> list[EvalExample]:
    from datasets import load_dataset

    rows = load_dataset("openai/gsm8k", "main")["test"]  # 1,319 problems
    return _examples(
        rows, "question", lambda r: extract_hash_answer(r["answer"]), "gsm8k_test", system_prompt
    )


def load_aime2024(system_prompt: str = SYSTEM_PROMPT) -> list[EvalExample]:
    from datasets import load_dataset

    rows = load_dataset("HuggingFaceH4/aime_2024")["train"]  # 30 problems
    return _examples(rows, "problem", lambda r: str(r["answer"]).strip(), "aime2024", system_prompt)


def load_aime2025(system_prompt: str = SYSTEM_PROMPT) -> list[EvalExample]:
    from datasets import load_dataset

    rows = load_dataset("opencompass/AIME2025")["train"]  # 30 problems (I + II)
    return _examples(rows, "problem", lambda r: str(r["answer"]).strip(), "aime2025", system_prompt)


def load_math500(system_prompt: str = SYSTEM_PROMPT) -> list[EvalExample]:
    from datasets import load_dataset

    rows = load_dataset("HuggingFaceH4/MATH-500")["test"]  # 500 problems
    return _examples(rows, "problem", lambda r: str(r["answer"]).strip(), "math500", system_prompt)


EVAL_LOADERS: dict[str, Callable[..., list[EvalExample]]] = {
    "gsm8k_test": load_gsm8k_test,
    "aime2024": load_aime2024,
    "aime2025": load_aime2025,
    "math500": load_math500,
}


def load_eval_dataset(
    name: str, max_samples: Optional[int] = None, system_prompt: str = SYSTEM_PROMPT
) -> list[EvalExample]:
    if name not in EVAL_LOADERS:
        raise ValueError(f"unknown eval dataset {name!r}; choose from {sorted(EVAL_LOADERS)}")
    examples = EVAL_LOADERS[name](system_prompt=system_prompt)
    if isinstance(max_samples, int) and max_samples > 0:
        examples = examples[:max_samples]
    return examples
