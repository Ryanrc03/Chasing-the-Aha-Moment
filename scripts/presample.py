#!/usr/bin/env python
"""Sample each GSM8K training problem G times with a model and record how many answers are right.

The output feeds the difficulty filter (data.filter_file): problems the model always or never
solves give GRPO no correctness signal, so a run can train only on the "sometimes right" ones.

    python scripts/presample.py --config configs/qwen2.5-3b_3060_fixed.yaml \
        --out outputs/presample/gsm8k_instruct_g6.jsonl --max-minutes 60

Problems are visited in a fixed random order (--seed), so whatever finishes inside the time cap
is a random subset of the training set. Re-running with the same --out resumes where it stopped.
Correctness uses the training reward's definition (answer extracted from <answer>, cfg.reward.match).
"""

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "0")
import unsloth  # noqa: F401,E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vllm import SamplingParams  # noqa: E402

from aha.config import load_config  # noqa: E402
from aha.data import load_gsm8k_train  # noqa: E402
from aha.model import load_model_and_tokenizer  # noqa: E402
from aha.parsing import extract_xml_answer, is_correct  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--samples", type=int, default=6, help="answers per problem (match train.num_generations)")
    p.add_argument("--max-problems", type=int, default=None)
    p.add_argument("--max-minutes", type=float, default=60.0)
    p.add_argument("--chunk", type=int, default=32, help="problems per vLLM call")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--lora", default=None, help="sample with this adapter instead of the base model")
    args = p.parse_args()

    cfg = load_config(args.config)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        with open(out, encoding="utf-8") as f:
            done = {json.loads(line)["idx"] for line in f if line.strip()}

    model, tok = load_model_and_tokenizer(cfg, with_lora=True)
    lora_request = model.load_lora(args.lora) if args.lora else None
    data = load_gsm8k_train()
    order = list(range(len(data)))
    random.Random(args.seed).shuffle(order)
    order = [i for i in order if i not in done]
    if args.max_problems is not None:
        order = order[: max(0, args.max_problems - len(done))]
    sp = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=cfg.max_completion_length, n=args.samples, seed=args.seed)

    start = time.time()
    tokens = 0
    n_new = 0
    hist: Counter = Counter()
    with open(out, "a", encoding="utf-8") as f:
        for c in range(0, len(order), args.chunk):
            if (time.time() - start) / 60 > args.max_minutes:
                print(f"time cap reached after {n_new} problems")
                break
            idxs = order[c : c + args.chunk]
            rows = [data[i] for i in idxs]
            texts = [tok.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True) for r in rows]
            outs = model.fast_generate(texts, sampling_params=sp, lora_request=lora_request)
            for i, r, o in zip(idxs, rows, outs):
                passes = sum(is_correct(extract_xml_answer(s.text), r["answer"], cfg.reward.match) for s in o.outputs)
                lens = [len(s.token_ids) for s in o.outputs]
                tokens += sum(lens)
                hist[passes] += 1
                f.write(json.dumps({"idx": i, "passes": passes, "n": args.samples, "mean_tokens": sum(lens) / len(lens)}) + "\n")
            f.flush()
            n_new += len(idxs)
            el = time.time() - start
            print(f"{n_new} problems, {el / 60:.1f} min, {el / n_new:.2f} s/problem, {tokens / el:.0f} tok/s", flush=True)

    total = sum(hist.values())
    if total:
        print("PASS-HISTOGRAM", {k: hist.get(k, 0) for k in range(args.samples + 1)})
        mixed = sum(hist.get(k, 0) for k in range(1, args.samples))
        print(f"PRESAMPLE {total} new problems: all right {hist.get(args.samples, 0) / total:.0%}, "
              f"all wrong {hist.get(0, 0) / total:.0%}, mixed {mixed / total:.0%}")


if __name__ == "__main__":
    main()
