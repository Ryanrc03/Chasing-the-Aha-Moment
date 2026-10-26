"""Supervised fine-tuning baseline: same model and LoRA as GRPO, trained on gold
GSM8K solutions rewritten into the <reasoning>/<answer> format.

The adapter is saved as a plain PEFT directory, which ``aha.evaluate`` loads through
``model.load_lora`` exactly like a GRPO adapter, so the two can be compared head to head.
"""

from __future__ import annotations

from pathlib import Path

from aha.config import ExperimentConfig, dump_config
from aha.data import load_gsm8k_sft
from aha.logging_utils import make_jsonl_callback, save_log_history
from aha.model import load_model_and_tokenizer


def build_sft_config(cfg: ExperimentConfig):
    from trl import SFTConfig

    s = cfg.sft
    return SFTConfig(
        output_dir=s.output_dir,
        learning_rate=s.learning_rate,
        num_train_epochs=s.num_train_epochs,
        max_steps=s.max_steps,
        per_device_train_batch_size=s.per_device_train_batch_size,
        gradient_accumulation_steps=s.gradient_accumulation_steps,
        max_length=s.max_length,
        warmup_ratio=s.warmup_ratio,
        lr_scheduler_type=s.lr_scheduler_type,
        optim=s.optim,
        weight_decay=s.weight_decay,
        logging_steps=s.logging_steps,
        save_strategy="steps" if s.save_steps > 0 else "no",
        save_steps=s.save_steps if s.save_steps > 0 else 500,
        seed=s.seed,
        report_to=s.report_to,
        run_name=cfg.run_name,
        completion_only_loss=s.completion_only_loss,
    )


def train_sft(cfg: ExperimentConfig) -> str:
    """Run SFT and return the directory the LoRA adapter was saved to."""
    from trl import SFTTrainer

    out_dir = Path(cfg.sft.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, out_dir / "config.yaml")

    # No vLLM engine is needed for SFT; skipping it frees ~4 GB for larger batches.
    cfg.model.fast_inference = False
    model, tokenizer = load_model_and_tokenizer(cfg, with_lora=True)
    dataset = load_gsm8k_sft(
        split=cfg.data.train_split,
        max_samples=cfg.data.max_train_samples,
        shuffle_seed=cfg.data.shuffle_seed,
    )
    print(f"SFT examples: {len(dataset)}")

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=build_sft_config(cfg),
        train_dataset=dataset,
        callbacks=[make_jsonl_callback(out_dir)],
    )
    trainer.train()
    save_log_history(trainer, out_dir)

    lora_dir = cfg.sft_lora_save_dir
    model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)
    print(f"SFT LoRA adapter saved to {lora_dir}")
    return lora_dir
