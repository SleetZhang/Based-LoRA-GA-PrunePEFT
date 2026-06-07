"""
Full fine-tuning baseline for Llama-2-7B on MetaMath, evaluated on GSM8K
— ORIGINAL (unmodified) LoRA-GA repo variant.

Adapted from the modified repo's examples/fft_metamath_gsm8k.py. It is fully
self-contained: it only ADDS this file and does NOT edit any existing module, so
the rest of this repo stays pristine for a fair FFT-vs-LoRA/DoRA comparison.

Two deliberate differences make it compatible with THIS repo:

1) Training does NOT call utils.train_text_to_text_model. In this repo that helper
   attaches an EarlyStoppingCallback unconditionally while eval_strategy="no" and
   load_best_model_at_end=False; HF's EarlyStoppingCallback.on_train_begin then
   asserts and crashes before any compute (and eval_strategy is hardcoded there,
   so a caller cannot disable it). We inline an equivalent Trainer setup with the
   SAME TrainingArguments, minus that callback. The training_step override mirrors
   this repo's LogTrainer(do_log=False) loss-normalization convention (it delegates
   to the stock training_step WITHOUT num_items_in_batch) so FFT trains on the same
   footing as this repo's LoRA/DoRA baselines, while tolerating both the 2-arg
   (transformers<=4.45) and 3-arg (>=4.46, num_items_in_batch) signatures so it
   never crashes on a version mismatch.

2) The checkpoint is saved with safe_serialization=False (pytorch_model.bin),
   because this repo's initialize_text_to_text_model forces use_safetensors=False
   on load — so `--stage eval --model_path <dir>` can reload it.

Run from the repo root (single A800-80GB is enough for 7B full fine-tuning):
    CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline python examples/fft_metamath_gsm8k.py
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from typing import Optional

sys.path.append("./peft/src")
sys.path.append(".")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FFTConfig:
    model_id: str = "ckpts/pretrained/Llama-2-7b-hf"
    dataset: str = "meta_math"
    save_path: str = "./save/fft_llama2_metamath_gsm8k_seed42"
    model_path: Optional[str] = None
    stage: str = "all"
    seed: int = 42
    dtype: str = "bf16"
    epochs: int = 1
    learning_rate: float = 1e-5
    per_device_batch_size: int = 1
    real_batch_size: int = 32
    max_length: int = 1024
    logging_steps: int = 10
    eval_epochs: float = 1.0
    early_stopping_patience: int = 3
    gradient_checkpointing: bool = True
    flash_attention: bool = False
    optim: str = "adamw_torch"
    max_steps: int = -1
    max_train_samples: int = -1
    max_valid_samples: int = -1


def parse_args() -> FFTConfig:
    parser = argparse.ArgumentParser(description="FFT Llama-2-7B MetaMath -> GSM8K (original repo)")
    parser.add_argument("--model_id", default=FFTConfig.model_id)
    parser.add_argument("--dataset", default=FFTConfig.dataset)
    parser.add_argument("--save_path", default=FFTConfig.save_path)
    parser.add_argument("--model_path", default=None, help="Full model path for eval-only runs")
    parser.add_argument("--stage", default=FFTConfig.stage, choices=["all", "train", "eval"])
    parser.add_argument("--seed", type=int, default=FFTConfig.seed)
    parser.add_argument("--dtype", default=FFTConfig.dtype, choices=["bf16", "fp32"])
    parser.add_argument("--epochs", type=int, default=FFTConfig.epochs)
    parser.add_argument("--learning_rate", type=float, default=FFTConfig.learning_rate)
    parser.add_argument("--per_device_batch_size", type=int, default=FFTConfig.per_device_batch_size)
    parser.add_argument("--real_batch_size", type=int, default=FFTConfig.real_batch_size)
    parser.add_argument("--max_length", type=int, default=FFTConfig.max_length)
    parser.add_argument("--logging_steps", type=int, default=FFTConfig.logging_steps)
    parser.add_argument("--eval_epochs", type=float, default=FFTConfig.eval_epochs)
    parser.add_argument("--early_stopping_patience", type=int, default=FFTConfig.early_stopping_patience)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--flash_attention", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--optim", default=FFTConfig.optim)
    parser.add_argument("--max_steps", type=int, default=FFTConfig.max_steps)
    parser.add_argument("--max_train_samples", type=int, default=FFTConfig.max_train_samples)
    parser.add_argument("--max_valid_samples", type=int, default=FFTConfig.max_valid_samples)
    return FFTConfig(**vars(parser.parse_args()))


def maybe_select(dataset, max_samples: int):
    if max_samples is None or max_samples <= 0:
        return dataset
    max_samples = min(max_samples, len(dataset))
    if hasattr(dataset, "select"):
        return dataset.select(range(max_samples))
    return dataset[:max_samples]


def prepare_full_finetune_model(model, gradient_checkpointing: bool):
    for param in model.parameters():
        param.requires_grad_(True)
    if gradient_checkpointing and hasattr(model, "config"):
        model.config.use_cache = False
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    logger.info("FFT trainable parameters: %d / %d (%.2f%%)", trainable, total, trainable / total * 100)
    return model


def save_run_config(config: FFTConfig):
    os.makedirs(config.save_path, exist_ok=True)
    config_path = os.path.join(config.save_path, "fft_run_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2, ensure_ascii=False)
    logger.info("Saved FFT run config to %s", config_path)


def _train_full_model(run_name, train_dataset, valid_dataset, model, tokenizer, config: FFTConfig):
    """Crash-free, repo-faithful equivalent of utils.train_text_to_text_model for FFT.

    Reuses this repo's dataset pipeline (preprocess_dataset + transform_dataset =>
    causalLMEncode) and the same Seq2SeqTrainingArguments, but builds the Trainer
    WITHOUT the unconditional EarlyStoppingCallback (which would assert under
    eval_strategy="no").
    """
    from transformers import Seq2SeqTrainingArguments, Trainer

    from examples.utils import preprocess_dataset, transform_dataset

    class _FFTTrainer(Trainer):
        """Matches this repo's LogTrainer(do_log=False): delegate to the stock
        training_step WITHOUT num_items_in_batch (this repo's loss-normalization
        convention), absorbing the optional 3rd arg so it works on both
        transformers<=4.45 (2-arg) and >=4.46 (3-arg) without crashing."""

        def training_step(self, model, inputs, *args, **kwargs):
            return super().training_step(model, inputs)

    per_device_batch_size = config.per_device_batch_size
    real_batch_size = config.real_batch_size
    assert (
        real_batch_size % per_device_batch_size == 0
    ), "real_batch_size must be divisible by per_device_batch_size"
    accu_step = real_batch_size // per_device_batch_size

    train_dataset = preprocess_dataset(train_dataset)
    valid_dataset = preprocess_dataset(valid_dataset)
    train_dataset = transform_dataset("CausalLM", tokenizer, train_dataset, config.max_length)
    valid_dataset = transform_dataset("CausalLM", tokenizer, valid_dataset, config.max_length)

    ta_kwargs = dict(
        output_dir=f"./results/{run_name}/{config.seed}",
        num_train_epochs=config.epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=accu_step,
        logging_dir="./logs",
        logging_steps=config.logging_steps,
        bf16=(config.dtype == "bf16"),
        gradient_checkpointing=config.gradient_checkpointing,
        optim=config.optim,
        eval_strategy="no",
        save_strategy="no",
        save_total_limit=1,
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        learning_rate=config.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        max_grad_norm=1.0,
        remove_unused_columns=False,
        label_names=["labels"],
        seed=config.seed,
        ddp_find_unused_parameters=False,
    )
    if config.max_steps and config.max_steps > 0:
        ta_kwargs["max_steps"] = config.max_steps

    training_args = Seq2SeqTrainingArguments(**ta_kwargs)
    trainer = _FFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=None,
    )
    trainer.train()
    return model


def train_fft(config: FFTConfig):
    from examples.data import DATASET_MAP
    from examples.utils import initialize_text_to_text_model

    if config.dataset not in DATASET_MAP:
        raise ValueError(f"Unsupported dataset: {config.dataset}")

    logger.info("Loading training dataset: %s", config.dataset)
    train_set, val_set, _ = DATASET_MAP[config.dataset]()
    train_set = maybe_select(train_set, config.max_train_samples)
    val_set = maybe_select(val_set, config.max_valid_samples)

    logger.info("Loading base model: %s", config.model_id)
    model, tokenizer = initialize_text_to_text_model(
        config.model_id,
        "CausalLM",
        config.dtype,
        flash_attention=config.flash_attention,
    )
    model = prepare_full_finetune_model(model, config.gradient_checkpointing)

    run_name = f"fft_llama2_{config.dataset}_seed{config.seed}"
    model = _train_full_model(run_name, train_set, val_set, model, tokenizer, config)

    os.makedirs(config.save_path, exist_ok=True)
    logger.info("Saving full fine-tuned model to: %s", config.save_path)
    # safe_serialization=False -> pytorch_model.bin, matching this repo's
    # initialize_text_to_text_model(use_safetensors=False) reload path.
    model.save_pretrained(config.save_path, safe_serialization=False)
    tokenizer.save_pretrained(config.save_path)
    save_run_config(config)
    return model, tokenizer


def eval_gsm8k(config: FFTConfig, model=None, tokenizer=None):
    from accelerate import Accelerator

    from examples.controller import run_gsm8k_evaluation
    from examples.utils import initialize_text_to_text_model

    accelerator = Accelerator()
    if model is None or tokenizer is None:
        eval_path = config.model_path or config.save_path
        logger.info("Loading FFT model for evaluation: %s", eval_path)
        model, tokenizer = initialize_text_to_text_model(
            eval_path,
            "CausalLM",
            config.dtype,
            flash_attention=config.flash_attention,
        )

    model = model.to(accelerator.device)
    accuracy = run_gsm8k_evaluation(model, tokenizer, accelerator)

    if accelerator.is_local_main_process:
        os.makedirs(config.save_path, exist_ok=True)
        result_path = os.path.join(config.save_path, "eval_results.txt")
        with open(result_path, "a", encoding="utf-8") as f:
            f.write("Method: FFT\n")
            f.write("Test Dataset: gsm8k\n")
            f.write(f"Accuracy: {accuracy:.6f}\n")
            f.write("-" * 20 + "\n")
        logger.info("Saved GSM8K eval result to %s", result_path)
    return accuracy


def main():
    config = parse_args()
    import torch

    from examples.utils import seed_everything

    os.environ.setdefault("WANDB_MODE", "offline")
    torch.backends.cuda.matmul.allow_tf32 = True
    seed_everything(config.seed)

    model = None
    tokenizer = None
    if config.stage in ("all", "train"):
        model, tokenizer = train_fft(config)
        torch.cuda.empty_cache()
    if config.stage in ("all", "eval"):
        accuracy = eval_gsm8k(config, model, tokenizer)
        logger.info("FFT GSM8K accuracy: %.6f", accuracy)


if __name__ == "__main__":
    main()
