#!/usr/bin/env python
"""Profile where GPU time goes in one GRPO step on this card.

Part A - training half: a GRPO-shaped forward+backward (num_generations sequences of
prompt+completion tokens, LoRA on a 4-bit base, gradient checkpointing) under
torch.profiler; kernels are grouped into dequant / gemm / attention / ... and a
compute-roofline utilisation is derived.

Part B - sampling half: vLLM decode of the same batch under vLLM's own torch profiler
(VLLM_TORCH_PROFILER_DIR); the chrome trace is aggregated the same way and compared
with the bandwidth floor (weights must be streamed once per decode step).

    python scripts/profile_step.py --config configs/qwen2.5-3b_3060.yaml --out results/profile

Needs a free GPU (do not run while training).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "0")
import unsloth  # noqa: F401,E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aha.config import load_config  # noqa: E402
from aha.data import load_eval_dataset  # noqa: E402
from aha.model import load_model_and_tokenizer  # noqa: E402
from aha.profiling import (  # noqa: E402
    aggregate, categorize, decode_roofline, gpu_busy_fraction, kernels_from_chrome_trace, render, train_roofline,
)


def profile_training(cfg, model, tokenizer, n_seq: int, prompt_len: int, completion_len: int, out: Path,
                     stack: bool = False):
    import torch
    from torch.profiler import ProfilerActivity, profile

    dev = "cuda"
    vocab = model.config.vocab_size
    total_len = prompt_len + completion_len
    g = torch.Generator(device=dev).manual_seed(0)
    input_ids = torch.randint(0, vocab, (n_seq, total_len), device=dev, generator=g)
    attention_mask = torch.ones_like(input_ids)
    model.train()

    def step():
        out_ = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out_.logits[:, prompt_len - 1:-1, :].float()
        logprobs = torch.log_softmax(logits, dim=-1)
        tgt = input_ids[:, prompt_len:]
        token_lp = logprobs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        loss = -(token_lp.mean())  # GRPO-shaped: per-token logprob objective
        loss.backward()
        model.zero_grad(set_to_none=True)

    for _ in range(2):  # warm-up (compile, allocator)
        step()
    torch.cuda.synchronize()
    t0 = time.time()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=stack, with_stack=stack) as prof:
        step()
        torch.cuda.synchronize()
    wall = time.time() - t0

    # key_averages() mixes autograd nodes (CheckpointFunctionBackward, LoRA_MLP, ...) with the
    # kernels they launch, which double counts; take GPU kernel events from the chrome trace.
    prof.export_chrome_trace(str(out / "train_trace.json"))
    leaf = kernels_from_chrome_trace(out / "train_trace.json")
    if stack:
        from aha.profiling import attribute_kernels, render_attribution

        rows = attribute_kernels(out / "train_trace.json")
        attribution_md = render_attribution(rows, total_us=sum(us for _, us in leaf))
        Path("docs/kernels").mkdir(parents=True, exist_ok=True)
        Path("docs/kernels/attribution.md").write_text(
            "# fill / copy / cast kernels in one training step, attributed to code

" + attribution_md, encoding="utf-8")
        print(attribution_md)
    agg = aggregate(leaf)
    params = sum(p.numel() for p in model.parameters())
    roof = train_roofline(params, n_seq * total_len, wall)
    busy = gpu_busy_fraction(leaf, wall * 1e6)
    extra = [
        f"- wall time for one fwd+bwd of {n_seq} x {total_len} tokens: **{wall * 1e3:.0f} ms**",
        f"- GPU busy (sum of kernel time / wall, upper bound): **{busy:.0%}** -> idle/launch overhead ~{1 - busy:.0%}",
        f"- GEMM-only throughput: {roof['flops'] / 1e12 / (sum(us for n, us in leaf if categorize(n) == 'gemm (matmul)') / 1e6):.1f} TFLOPS while a matmul kernel is running",
        f"- params {params / 1e9:.2f} B, ~{roof['flops'] / 1e12:.1f} TFLOP incl. checkpoint recompute -> "
        f"**{roof['achieved_tflops']:.1f} TFLOPS achieved, {roof['compute_utilisation_bf16']:.0%} of BF16 peak**",
    ]
    (out / "train_kernels.json").write_text(json.dumps(agg, indent=1), encoding="utf-8")
    return render(f"Training half: fwd+bwd, {n_seq} seq x ({prompt_len}+{completion_len}) tokens", agg, extra)


def profile_decode(cfg, model, tokenizer, n_gen: int, max_tokens: int, out: Path):
    import torch
    from vllm import SamplingParams

    ex = load_eval_dataset("gsm8k_test", max_samples=1)[0]
    text = tokenizer.apply_chat_template(ex.prompt, tokenize=False, add_generation_prompt=True)
    sp = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=max_tokens, n=n_gen, seed=0,
                        ignore_eos=True)  # fixed length so the roofline is exact
    model.fast_generate([text], sampling_params=sp)  # warm-up
    engine = model.vllm_engine
    trace_dir = out / "vllm_trace"
    trace_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.synchronize()
    t0 = time.time()
    try:
        engine.start_profile()
        outs = model.fast_generate([text], sampling_params=sp)
        engine.stop_profile()
        profiled = True
    except Exception as exc:  # noqa: BLE001
        print(f"vLLM profiler unavailable ({exc}); timing only")
        outs = model.fast_generate([text], sampling_params=sp)
        profiled = False
    torch.cuda.synchronize()
    wall = time.time() - t0
    tokens = sum(len(o.token_ids) for o in outs[0].outputs)
    tps = tokens / wall
    weight_bytes = 2.35 * 2**30  # measured by vLLM at load for the bnb-4bit 3B (see log)
    roof = decode_roofline(weight_bytes, tps, n_gen)
    extra = [
        f"- {n_gen} completions x {max_tokens} tokens in **{wall:.1f} s** = {tps:.0f} tok/s, "
        f"{roof['achieved_ms_per_step']:.1f} ms per decode step (wall time is inflated several-fold "
        f"by CUPTI tracing; use the unprofiled benchmark for absolute speed, this section for shares)",
        f"- bandwidth floor (stream {weight_bytes / 2**30:.2f} GiB of weights per step at 360 GB/s): "
        f"**{roof['bandwidth_floor_ms_per_step']:.1f} ms** -> bandwidth utilisation **{roof['bandwidth_utilisation']:.0%}**",
    ]
    if not profiled:
        return "## Sampling half: vLLM decode\n\n" + "\n".join(extra) + "\n"
    time.sleep(2)  # trace files are written asynchronously
    traces = sorted(trace_dir.rglob("*.json*"))
    kernels = []
    for t in traces:
        if t.suffix == ".gz":
            import gzip

            tmp = t.with_suffix("")
            tmp.write_bytes(gzip.open(t).read())
            t = tmp
        kernels += kernels_from_chrome_trace(t)
    agg = aggregate(kernels)
    busy = gpu_busy_fraction(kernels, wall * 1e6)
    extra.append(f"- GPU busy during decode (kernel time / wall): **{busy:.0%}** (profiler-inflated wall)")
    extra.append(f"- GPU kernel time per decode step: **{agg['total_us'] / 1e3 / max_tokens:.1f} ms** "
                 f"vs bandwidth floor {roof['bandwidth_floor_ms_per_step']:.1f} ms")
    (out / "decode_kernels.json").write_text(json.dumps(agg, indent=1), encoding="utf-8")
    return render(f"Sampling half: vLLM decode, {n_gen} x {max_tokens} tokens", agg, extra)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/qwen2.5-3b_3060.yaml")
    parser.add_argument("--set", nargs="*", default=[])
    parser.add_argument("--out", default="results/profile")
    parser.add_argument("--prompt-len", type=int, default=200)
    parser.add_argument("--completion-len", type=int, default=400)
    parser.add_argument("--decode-tokens", type=int, default=256)
    parser.add_argument("--skip-decode", action="store_true")
    parser.add_argument("--stack", action="store_true", help="record Python stacks and write docs/kernels/attribution.md")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    os.environ["VLLM_TORCH_PROFILER_DIR"] = str((out / "vllm_trace").resolve())
    cfg = load_config(args.config, args.set)
    model, tokenizer = load_model_and_tokenizer(cfg, with_lora=True)

    sections = [f"# GPU profile: {cfg.model.name} on {__import__('torch').cuda.get_device_name()}\n"]
    sections.append(profile_training(cfg, model, tokenizer, cfg.train.num_generations,
                                     args.prompt_len, args.completion_len, out, stack=args.stack))
    if not args.skip_decode and cfg.model.fast_inference:
        sections.append(profile_decode(cfg, model, tokenizer, cfg.train.num_generations,
                                       args.decode_tokens, out))
    text = "\n".join(sections)
    (out / "profile.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
