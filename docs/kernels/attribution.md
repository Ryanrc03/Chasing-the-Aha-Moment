# fill / copy / cast kernels in one training step, attributed to code

## Memory-traffic kernels by launching code

| share of step | ms | launches | aten op | frame (innermost non-torch) |
|---|---|---|---|---|
| 8.7% | 647.9 | 4 | `aten::copy_` | `<no stack>` |
| 5.5% | 412.1 | 4 | `aten::fill_` | `<no stack>` |
| 0.5% | 40.5 | 216 | `aten::copy_` | `peft/tuners/tuners_utils.py(2220): _cast_input_dtype` |
| 0.5% | 38.5 | 612 | `aten::copy_` | `torch/autograd/graph.py(820): _engine_run_backward` |
| 0.4% | 29.0 | 216 | `aten::add_` | `torch/autograd/graph.py(820): _engine_run_backward` |
| 0.4% | 27.1 | 216 | `aten::add` | `peft/tuners/lora/layer.py(1074): forward` |
| 0.3% | 23.6 | 144 | `aten::add` | `unsloth/models/llama.py(1076): LlamaDecoderLayer_fast_forward` |
| 0.3% | 22.0 | 216 | `aten::mul` | `peft/tuners/lora/layer.py(1074): forward` |
| 0.2% | 15.9 | 216 | `aten::copy_` | `peft/tuners/lora/layer.py(1074): forward` |
| 0.1% | 8.7 | 576 | `aten::copy_` | `unsloth/kernels/utils.py(1801): matmul_lora` |
| 0.1% | 8.4 | 108 | `aten::mul` | `torch/autograd/graph.py(820): _engine_run_backward` |
| 0.1% | 7.1 | 1 | `aten::copy_` | `profile_step.py(51): step` |
| 0.1% | 7.0 | 216 | `aten::copy_` | `unsloth/kernels/fast_lora.py(431): backward` |
| 0.1% | 4.9 | 36 | `aten::add` | `torch/autograd/graph.py(820): _engine_run_backward` |
| 0.1% | 4.9 | 72 | `aten::copy_` | `unsloth/utils/attention_dispatch.py(872): _run_attention` |
| 0.0% | 2.1 | 216 | `aten::add` | `bitsandbytes/backends/cuda/ops.py(904): _dequant_linear_fallback` |
| 0.0% | 1.7 | 108 | `aten::add_` | `bitsandbytes/functional.py(992): dequantize_4bit` |
| 0.0% | 1.5 | 1 | `aten::mul_` | `unsloth/models/llama.py(1201): LlamaModel_fast_forward` |
| 0.0% | 0.7 | 36 | `aten::fill_` | `torch/autograd/graph.py(820): _engine_run_backward` |
| 0.0% | 0.2 | 72 | `aten::copy_` | `unsloth/kernels/fast_lora.py(1053): backward` |
| 0.0% | 0.2 | 72 | `aten::fill_` | `xformers/ops/fmha/flash3.py(220): mha_fwd` |
| 0.0% | 0.1 | 144 | `aten::addmm` | `bitsandbytes/backends/cuda/ops.py(904): _dequant_linear_fallback` |
| 0.0% | 0.1 | 72 | `aten::mm` | `unsloth/kernels/utils.py(1801): matmul_lora` |
| 0.0% | 0.1 | 72 | `aten::mm` | `torch/autograd/graph.py(820): _engine_run_backward` |
| 0.0% | 0.0 | 1 | `aten::scatter_add_` | `<no stack>` |
| 0.0% | 0.0 | 36 | `aten::mm` | `unsloth/kernels/fast_lora.py(431): backward` |
| 0.0% | 0.0 | 36 | `aten::addmm_` | `unsloth/kernels/utils.py(1783): addmm_` |
| 0.0% | 0.0 | 1 | `aten::_local_scalar_dense` | `unsloth/models/llama.py(1169): _is_left_padded` |
| 0.0% | 0.0 | 1 | `aten::gather` | `profile_step.py(51): step` |
| 0.0% | 0.0 | 1 | `aten::div` | `<no stack>` |

Full stacks (top 10):

- `aten::copy_` 647.9 ms: <no stack>
- `aten::fill_` 412.1 ms: <no stack>
- `aten::copy_` 40.5 ms: peft/tuners/tuners_utils.py(2220): _cast_input_dtype <- peft/tuners/lora/layer.py(1074): forward <- torch/nn/modules/module.py(1777): _call_impl
- `aten::copy_` 38.5 ms: torch/autograd/graph.py(820): _engine_run_backward <- torch/autograd/__init__.py(243): backward <- torch/utils/checkpoint.py(265): backward
- `aten::add_` 29.0 ms: torch/autograd/graph.py(820): _engine_run_backward <- torch/autograd/__init__.py(243): backward <- torch/utils/checkpoint.py(265): backward
- `aten::add` 27.1 ms: peft/tuners/lora/layer.py(1074): forward <- torch/nn/modules/module.py(1777): _call_impl <- nn.Module: Linear_0
- `aten::add` 23.6 ms: unsloth/models/llama.py(1076): LlamaDecoderLayer_fast_forward <- torch/nn/modules/module.py(1777): _call_impl <- nn.Module: Qwen2DecoderLayer_0
- `aten::mul` 22.0 ms: peft/tuners/lora/layer.py(1074): forward <- torch/nn/modules/module.py(1777): _call_impl <- nn.Module: Linear_0
- `aten::copy_` 15.9 ms: peft/tuners/lora/layer.py(1074): forward <- torch/nn/modules/module.py(1777): _call_impl <- nn.Module: Linear_0
- `aten::copy_` 8.7 ms: unsloth/kernels/utils.py(1801): matmul_lora <- unsloth/kernels/fast_lora.py(1022): forward <- torch/amp/autocast_mode.py(512): decorate_fwd

## Reading the table (2026-10-27)

- The two dominant rows (`aten::copy_` 648 ms and `aten::fill_` 412 ms, 4 launches each) carry no Python stack because
  autograd's backward thread launches them. Their size equals the full [6 x 400 x 151936] fp32 logits: they are the
  `.float()` cast and the gradient zero-fill of the plain `log_softmax` loss inside `scripts/profile_step.py`.
  The real trainer uses Unsloth's chunked `selective_log_softmax` and never materialises this tensor, so this ~14% is a
  profiling artefact, not trainer time.
- Everything attributed to code the trainer does run (peft `_cast_input_dtype` 40 ms, LoRA add/mul ~50 ms, autograd
  copies ~40 ms, Unsloth `matmul_lora` 9 ms) adds up to under 3% of kernel time.
- Conclusion: P4 has no meaningful target left in the training half. The earlier "fill/copy/cast ~18%" figure from
  results/profile was inflated by the same artefact. Wall time in this run (118 s) is inflated by `with_stack`; use it
  for attribution only.
