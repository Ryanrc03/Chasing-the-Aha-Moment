#!/usr/bin/env python
"""End-to-end vLLM decode speed with and without the Triton NF4 GEMM, plus greedy-output agreement.

    python scripts/bench_decode.py --config configs/qwen2.5-3b_3060.yaml --out docs/kernels/decode_base.json
    python scripts/bench_decode.py --config configs/qwen2.5-3b_3060.yaml --patch --out docs/kernels/decode_patch.json
    python scripts/bench_decode.py --compare docs/kernels/decode_base.json docs/kernels/decode_patch.json --report docs/kernels/decode.md

Timed runs use temperature 1.0 with ignore_eos so every sequence decodes exactly max_tokens steps;
the greedy run (temperature 0) is saved for the agreement check. The patch must be applied before
the engine is built (CUDA graphs are captured at load).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

DECODE_SHAPES = [(2048, 2048), (256, 2048), (11008, 2048), (2048, 11008)]


def _precompile_kernels():
    """Compile the Triton kernel for every shape before vLLM captures CUDA graphs."""
    import torch

    from aha.kernels.nf4_gemm import nf4_gemm

    for N, K in DECODE_SHAPES:
        for M in (1, 16, 32):
            nf4_gemm(torch.zeros(M, K, dtype=torch.bfloat16, device="cuda"),
                     torch.zeros(N * K // 2, 1, dtype=torch.uint8, device="cuda"),
                     torch.zeros(N * K // 64, dtype=torch.float16, device="cuda"), N, K)
    torch.cuda.synchronize()


def measure(args) -> dict:
    from bench.harness import assert_gpu_free

    assert_gpu_free(args.force)
    if args.patch:
        from aha.kernels.nf4_gemm import patch_bitsandbytes

        patch_bitsandbytes()
        _precompile_kernels()
    import unsloth  # noqa: F401
    import torch
    from vllm import SamplingParams

    from aha.config import load_config
    from aha.data import load_eval_dataset
    from aha.model import load_model_and_tokenizer

    cfg = load_config(args.config, [])
    model, tok = load_model_and_tokenizer(cfg, with_lora=True)
    examples = load_eval_dataset("gsm8k_test", max_samples=max(max(args.prompts), args.greedy_prompts))
    texts = [tok.apply_chat_template(ex.prompt, tokenize=False, add_generation_prompt=True) for ex in examples]

    result = {"patch": args.patch, "max_tokens": args.max_tokens, "n": args.n, "timings": []}
    for n_prompts in args.prompts:
        sp = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=args.max_tokens, n=args.n, seed=0, ignore_eos=True)
        model.fast_generate(texts[:n_prompts], sampling_params=sp)  # warm-up / graph capture
        walls = []
        for _ in range(3):
            torch.cuda.synchronize()
            t0 = time.time()
            model.fast_generate(texts[:n_prompts], sampling_params=sp)
            torch.cuda.synchronize()
            walls.append(time.time() - t0)
        wall = sorted(walls)[1]
        seqs = n_prompts * args.n
        result["timings"].append({"prompts": n_prompts, "sequences": seqs, "wall_s": wall,
                                  "ms_per_step": wall / args.max_tokens * 1e3,
                                  "tok_per_s": seqs * args.max_tokens / wall})
        print(result["timings"][-1])

    greedy = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    outs = model.fast_generate(texts[:args.greedy_prompts], sampling_params=greedy)
    result["greedy"] = [list(o.outputs[0].token_ids) for o in outs]
    return result


def compare(base_path: str, patch_path: str, report: str) -> None:
    base, patch = json.load(open(base_path)), json.load(open(patch_path))
    lines = ["# vLLM decode: bitsandbytes kernel vs Triton NF4 GEMM", "",
             f"Qwen2.5-3B-Instruct bnb-4bit, {base['max_tokens']} decode steps per run, median of 3.", "",
             "| sequences | base ms/step | patch ms/step | speed-up | base tok/s | patch tok/s |", "|---|---|---|---|---|---|"]
    for b, p in zip(base["timings"], patch["timings"]):
        lines.append(f"| {b['sequences']} | {b['ms_per_step']:.1f} | {p['ms_per_step']:.1f} | "
                     f"{b['ms_per_step'] / p['ms_per_step']:.2f}x | {b['tok_per_s']:.0f} | {p['tok_per_s']:.0f} |")
    exact, first_div = 0, []
    for a, c in zip(base["greedy"], patch["greedy"]):
        if a == c:
            exact += 1
        else:
            k = next((i for i, (u, v) in enumerate(zip(a, c)) if u != v), min(len(a), len(c)))
            first_div.append(k)
    n = len(base["greedy"])
    lines += ["", f"Greedy agreement on {n} GSM8K prompts: {exact}/{n} identical token sequences "
              f"({exact / n:.0%}); first divergence at token index {sorted(first_div)} for the rest.", ""]
    Path(report).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/qwen2.5-3b_3060.yaml")
    p.add_argument("--patch", action="store_true")
    p.add_argument("--prompts", type=int, nargs="*", default=[1, 4])
    p.add_argument("--greedy-prompts", type=int, default=20)
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--out", default=None)
    p.add_argument("--compare", nargs=2, metavar=("BASE", "PATCH"))
    p.add_argument("--report", default="docs/kernels/decode.md")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    if a.compare:
        compare(a.compare[0], a.compare[1], a.report)
    else:
        res = measure(a)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res), encoding="utf-8")
        print(f"wrote {a.out}")
