"""GRPO training loop: config -> model -> dataset -> rewards -> TRL GRPOTrainer."""

from __future__ import annotations

from pathlib import Path

from aha.config import ExperimentConfig, dump_config
from aha.data import load_train_dataset
from aha.logging_utils import make_jsonl_callback, save_log_history
from aha.model import load_model_and_tokenizer
from aha.rewards import build_reward_funcs


def build_grpo_config(cfg: ExperimentConfig):
    from trl import GRPOConfig

    t = cfg.train
    kwargs = dict(
        learning_rate=t.learning_rate,
        adam_beta1=t.adam_beta1,
        adam_beta2=t.adam_beta2,
        weight_decay=t.weight_decay,
        warmup_ratio=t.warmup_ratio,
        lr_scheduler_type=t.lr_scheduler_type,
        optim=t.optim,
        logging_steps=t.logging_steps,
        per_device_train_batch_size=t.per_device_train_batch_size,
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        num_generations=t.num_generations,
        max_prompt_length=t.max_prompt_length,
        max_completion_length=cfg.max_completion_length,
        max_steps=t.max_steps,
        save_steps=t.save_steps,
        max_grad_norm=t.max_grad_norm,
        report_to=t.report_to,
        output_dir=t.output_dir,
        seed=t.seed,
        run_name=cfg.run_name,
    )
    if t.kl_beta is not None:
        kwargs["beta"] = t.kl_beta
    return GRPOConfig(**kwargs)


def train(cfg: ExperimentConfig) -> str:
    """Run GRPO and return the directory the LoRA adapter was saved to."""
    from trl import GRPOTrainer

    out_dir = Path(cfg.train.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, out_dir / "config.yaml")

    model, tokenizer = load_model_and_tokenizer(cfg, with_lora=True)
    dataset = load_train_dataset(
        cfg.data.train_dataset,
        split=cfg.data.train_split,
        max_samples=cfg.data.max_train_samples,
        shuffle_seed=cfg.data.shuffle_seed,
    )
    reward_funcs = build_reward_funcs(cfg.reward, tokenizer=tokenizer)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=build_grpo_config(cfg),
        train_dataset=dataset,
        callbacks=[make_jsonl_callback(out_dir)],
    )
    trainer.train(resume_from_checkpoint=cfg.train.resume_from_checkpoint)
    save_log_history(trainer, out_dir)

    lora_dir = cfg.lora_save_dir
    model.save_lora(lora_dir)
    print(f"LoRA adapter saved to {lora_dir}")
    return lora_dir
