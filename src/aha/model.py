"""Model loading through Unsloth (4-bit base + LoRA, optional vLLM engine).

``unsloth`` must be imported before ``transformers``/``trl`` for its patches to
apply, so entry-point scripts do ``import unsloth`` as their first line.
"""

from __future__ import annotations

from aha.config import ExperimentConfig


def load_model_and_tokenizer(cfg: ExperimentConfig, with_lora: bool = True):
    """Return ``(model, tokenizer)`` configured per ``cfg.model``.

    With ``with_lora=True`` a fresh LoRA adapter is attached (training). For
    evaluation pass ``with_lora=False`` and load a saved adapter through
    ``model.load_lora(path)`` at generation time.
    """
    from unsloth import FastLanguageModel

    m = cfg.model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=m.name,
        max_seq_length=m.max_seq_length,
        load_in_4bit=m.load_in_4bit,
        fast_inference=m.fast_inference,
        max_lora_rank=m.lora_rank,
        gpu_memory_utilization=m.gpu_memory_utilization,
    )
    if with_lora:
        model = FastLanguageModel.get_peft_model(
            model,
            r=m.lora_rank,
            target_modules=list(m.lora_target_modules),
            lora_alpha=cfg.lora_alpha,
            use_gradient_checkpointing=m.gradient_checkpointing,
            random_state=cfg.train.seed,
        )
    return model, tokenizer
