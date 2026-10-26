"""Batched evaluation with vLLM (through Unsloth's ``fast_generate``).

Metrics mirror the report: accuracy (numeric-normalised), XML parse rate and
soft-format rate, plus average per-sample latency. Results are written to
``<results_dir>/<run_name>/<dataset>/{metrics.json,predictions.jsonl}``.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path
from typing import Iterable, Iterator, Optional

from aha.config import ExperimentConfig
from aha.data import EvalExample, load_eval_dataset
from aha.parsing import SOFT_XML_PATTERN, extract_xml_answer, has_answer_tag, is_correct


@dataclass
class Prediction:
    idx: int
    gold: Optional[str]
    extracted: Optional[str]
    correct: bool
    xml_ok: bool
    soft_format_ok: bool
    completion_tokens: int
    output: str


@dataclass
class Metrics:
    dataset: str
    samples: int
    accuracy: float
    xml_parse_rate: float
    soft_format_rate: float
    avg_latency_s: float
    wall_time_s: float
    avg_completion_tokens: float
    lora_path: Optional[str]
    model: str


def batched(iterable: Iterable, n: int) -> Iterator[list]:
    it = iter(iterable)
    while chunk := list(islice(it, n)):
        yield chunk


def score_output(text: str, gold: Optional[str], match: str = "relaxed") -> dict:
    """Pure scoring of one generation; used by the eval loop and by unit tests."""
    xml_ok = has_answer_tag(text)
    extracted = extract_xml_answer(text) if xml_ok else None
    return {
        "extracted": extracted,
        "xml_ok": xml_ok,
        "soft_format_ok": SOFT_XML_PATTERN.search(text) is not None,
        "correct": xml_ok and is_correct(extracted, gold, match),
    }


def summarize(dataset: str, preds: list[Prediction], latencies: list[float], wall: float,
              lora_path: Optional[str], model_name: str) -> Metrics:
    n = len(preds)
    return Metrics(
        dataset=dataset,
        samples=n,
        accuracy=sum(p.correct for p in preds) / n if n else 0.0,
        xml_parse_rate=sum(p.xml_ok for p in preds) / n if n else 0.0,
        soft_format_rate=sum(p.soft_format_ok for p in preds) / n if n else 0.0,
        avg_latency_s=sum(latencies) / len(latencies) if latencies else float("nan"),
        wall_time_s=wall,
        avg_completion_tokens=sum(p.completion_tokens for p in preds) / n if n else 0.0,
        lora_path=lora_path,
        model=model_name,
    )


def generate_batch(model, texts: list[str], sampling_params, lora_request) -> list[tuple[str, int]]:
    """Return ``(text, n_tokens)`` per prompt via vLLM."""
    outputs = model.fast_generate(texts, sampling_params=sampling_params, lora_request=lora_request)
    return [(o.outputs[0].text, len(o.outputs[0].token_ids)) for o in outputs]


def evaluate_dataset(cfg: ExperimentConfig, model, tokenizer, dataset_name: str,
                     lora_request=None) -> Metrics:
    import torch
    from tqdm import tqdm
    from vllm import SamplingParams

    e = cfg.eval
    examples: list[EvalExample] = load_eval_dataset(dataset_name, max_samples=e.max_samples)
    texts = [
        tokenizer.apply_chat_template(x.prompt, tokenize=False, add_generation_prompt=True)
        for x in examples
    ]
    sampling_params = SamplingParams(
        temperature=e.temperature, top_p=e.top_p, max_tokens=e.max_tokens, seed=e.seed
    )

    preds: list[Prediction] = []
    latencies: list[float] = []
    start = time.time()
    with torch.no_grad():
        pbar = tqdm(total=len(texts), desc=f"Evaluating ({dataset_name})")
        for batch in batched(list(zip(examples, texts)), e.batch_size):
            t0 = time.time()
            generations = generate_batch(model, [t for _, t in batch], sampling_params, lora_request)
            latencies.append((time.time() - t0) / len(batch))
            for (ex, _), (text, n_tok) in zip(batch, generations):
                s = score_output(text, ex.gold, e.match)
                preds.append(Prediction(idx=ex.idx, gold=ex.gold, extracted=s["extracted"],
                                        correct=s["correct"], xml_ok=s["xml_ok"],
                                        soft_format_ok=s["soft_format_ok"],
                                        completion_tokens=n_tok, output=text))
            pbar.update(len(batch))
        pbar.close()
    wall = time.time() - start

    metrics = summarize(dataset_name, preds, latencies, wall, e.lora_path, cfg.model.name)
    out_dir = Path(e.results_dir) / cfg.run_name / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(asdict(metrics), f, indent=2)
    with open(out_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")

    print(f"\n=== {dataset_name} ===")
    for k, v in asdict(metrics).items():
        print(f"{k:>22}: {v}")
    print("\n--- sample predictions ---")
    for p in preds[: e.num_examples_to_keep]:
        print(f"[{p.idx}] gold={p.gold!r} pred={p.extracted!r} correct={p.correct}")
        print("    tail:", p.output[-300:].replace("\n", "\\n"))
    return metrics


def evaluate(cfg: ExperimentConfig) -> dict[str, Metrics]:
    """Evaluate the base model (or base + LoRA if ``cfg.eval.lora_path``) on every eval set."""
    from aha.model import load_model_and_tokenizer

    # Unsloth attaches load_lora / fast_generate to the PEFT wrapper, so a fresh adapter
    # (identity at init) must be attached even when evaluating the base model.
    model, tokenizer = load_model_and_tokenizer(cfg, with_lora=True)
    lora_request = model.load_lora(cfg.eval.lora_path) if cfg.eval.lora_path else None
    results = {}
    for name in cfg.eval.datasets:
        results[name] = evaluate_dataset(cfg, model, tokenizer, name, lora_request)
    return results
