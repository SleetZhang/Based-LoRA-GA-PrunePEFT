"""
PrunePEFT (LoRA) training script.

Author: zzh
"""

import logging
import os
import torch
from fire import Fire
import wandb
from accelerate import Accelerator

from peft import PeftModel, get_peft_model
from peft.tuners.prunepeft import PrunePEFTConfig
from peft.tuners.prunepeft.model import collect_pruning_process_info, compute_pruning_rankings

from examples.utils import (
    transform_dataset,
    preprocess_dataset,
    initialize_text_to_text_model,
    find_all_linear_modules,
    train_text_to_text_model,
)
from examples.data import DATASET_MAP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_prunepeft_config(model, **kwargs):
    """
    Create PrunePEFT configuration.

    Args:
        model: Base model
        **kwargs: Configuration parameters
    Returns:
        PrunePEFTConfig
    """
    # Get adapter types
    adapter_types_input = kwargs.get("adapter_types", ["lora"])
    if isinstance(adapter_types_input, str):
        adapter_types = [t.strip() for t in adapter_types_input.split(',')]
    else:
        adapter_types = adapter_types_input

    # Get target modules - only use provided target_modules for LoRA-only configs
    target_modules_input = kwargs.get("target_modules", None)
    if target_modules_input and adapter_types == ["lora"]:
        # For LoRA-only, use provided target_modules
        if isinstance(target_modules_input, str):
            target_modules = [module.strip() for module in target_modules_input.split(',')]
        else:
            target_modules = target_modules_input
    else:
        # For bottleneck or combined configs, let the model decide target_modules
        target_modules = None

    # Parse layer selection lists
    adapter_layers_input = kwargs.get("adapter_layers", None)
    lora_layers_input = kwargs.get("lora_layers", None)

    adapter_layers = None
    if adapter_layers_input:
        if isinstance(adapter_layers_input, str):
            if adapter_layers_input.strip():
                adapter_layers = [int(x.strip()) for x in adapter_layers_input.split(',')]
        elif isinstance(adapter_layers_input, (list, tuple)):
            # Fire converts comma-separated strings to tuples
            # Filter out empty strings and convert to int
            adapter_layers = [int(x) for x in adapter_layers_input if str(x).strip()]

    lora_layers = None
    if lora_layers_input:
        if isinstance(lora_layers_input, str):
            if lora_layers_input.strip():
                lora_layers = [int(x.strip()) for x in lora_layers_input.split(',')]
        elif isinstance(lora_layers_input, (list, tuple)):
            # Fire converts comma-separated strings to tuples
            # Filter out empty strings and convert to int
            lora_layers = [int(x) for x in lora_layers_input if str(x).strip()]

    config_kwargs = {
        "task_type": "CAUSAL_LM",
        "adapter_types": adapter_types,
        "target_modules": target_modules,
        "bias": kwargs.get("bias", "none"),
        "adapter_layers": adapter_layers,
        "lora_layers": lora_layers,
    }

    # Add parameters for all adapter types
    if "lora" in adapter_types:
        config_kwargs.update({
            "r": kwargs.get("lora_rank", 8),
            "lora_alpha": kwargs.get("lora_alpha", 16),
            "lora_dropout": kwargs.get("lora_dropout", 0.1),
        })

    if "bottleneck" in adapter_types:
        config_kwargs.update({
            "bottleneck_size": kwargs.get("bottleneck_size", 64),
            "bottleneck_dropout": kwargs.get("bottleneck_dropout", 0.1),
            "init_bottleneck_weights": kwargs.get("init_bottleneck_weights", True),
        })

    return PrunePEFTConfig(**config_kwargs)


def main(
    adapter_types="lora",
    lora_rank=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bottleneck_size=64,
    bottleneck_dropout=0.1,
    init_bottleneck_weights=True,
    adapter_layers="",
    lora_layers="",
    target_modules="q_proj,v_proj,k_proj,o_proj",
    sample_size=128,
    seed=42,
    bias="none",
):
    """
    Main training function for PrunePEFT.

    Args:
        adapter_types: Types of adapters to use (comma-separated string like "lora" or "lora,bottleneck")
        lora_rank: LoRA rank dimension
        lora_alpha: LoRA alpha parameter
        lora_dropout: LoRA dropout rate
        bottleneck_size: Size of bottleneck for bottleneck adapter
        bottleneck_dropout: Dropout for bottleneck adapter
        init_bottleneck_weights: Whether to initialize bottleneck weights
        adapter_layers: Comma-separated layer indices to apply Bottleneck adapter (e.g., "0,1,2"), empty means all layers
        lora_layers: Comma-separated layer indices to apply LoRA adapter (e.g., "0,1,2"), empty means all layers
        target_modules: Target modules to apply adapter
        sample_size: Number of samples
        seed: Random seed
        bias: Bias type (none/all/lora_only)
    """
    accelerator = Accelerator()
    model_id = "/home/autopeft/LoRA-GA/ckpts/pretrained/Llama-2-7b-hf"
    model_type = "CausalLM"
    model_dtype = "bf16"
    dataset_name = "meta_math"

    config = dict(
        model="llama",
        method="prunepeft",
        d=dataset_name,
        lora_r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        s=sample_size,
        sd=seed,
    )

    wandb_name = "_".join([f"{k}={v}" for k, v in config.items()])

    if accelerator.is_local_main_process:
        wandb.init(
            name=wandb_name,
            mode="offline",
            group="prunepeft",
            project="PrunePEFT Methods",
        )

    model, tokenizer = initialize_text_to_text_model(
        model_id, model_type, model_dtype, flash_attention=False
    )

    if accelerator.is_local_main_process:
        logger.info("使用微调方法: PRUNEPEFT")
        logger.info("原始模型结构:")
        logger.info(model)

    logger.info("创建PrunePEFT配置")

    peft_config = create_prunepeft_config(
        model=model,
        adapter_types=adapter_types,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bottleneck_size=bottleneck_size,
        bottleneck_dropout=bottleneck_dropout,
        init_bottleneck_weights=init_bottleneck_weights,
        adapter_layers=adapter_layers,
        lora_layers=lora_layers,
        target_modules=target_modules,
        bias=bias,
    )

    logger.info("PrunePEFT (%s) 配置:", ",".join(adapter_types).upper())
    for adapter_type in adapter_types:
        if adapter_type == "lora":
            logger.info("  LoRA - r: %d, alpha: %d, dropout: %.3f",
                       peft_config.r, peft_config.lora_alpha, peft_config.lora_dropout)
            logger.info("  LoRA layers (raw input): %s", lora_layers)
            logger.info("  LoRA layers (config): %s", peft_config.lora_layers)
        elif adapter_type == "bottleneck":
            logger.info("  Bottleneck - size: %d, dropout: %.3f",
                       peft_config.bottleneck_size, peft_config.bottleneck_dropout)
            logger.info("  Bottleneck adapter layers (raw input): %s", adapter_layers)
            logger.info("  Bottleneck adapter layers (config): %s", peft_config.adapter_layers)
    logger.info("  target_modules: %s", peft_config.target_modules)

    dataset_func = DATASET_MAP[dataset_name]
    train_set, val_set, _ = dataset_func()

    model = get_peft_model(model=model, peft_config=peft_config)

    save_dir = os.path.join("./snapshot", wandb_name)

    if accelerator.is_local_main_process:
        logger.info("PEFT模型配置完成")
        logger.info("PrunePEFT模型结构:")
        logger.info(model)
        model.print_trainable_parameters()

    # Collect pruning process info (gradients and activations) using a small sample
    logger.info("采集剪枝过程所需信息（梯度/激活）")
    # Prepare a small dataloader with encoded samples
    sample_count = min(sample_size, 16)
    encoded_train = transform_dataset(model_type, tokenizer, preprocess_dataset(train_set), max_length=1024)
    from torch.utils.data import DataLoader
    small_loader = DataLoader(encoded_train.select(range(sample_count)), batch_size=1)
    process_info = collect_pruning_process_info(
        model=model,
        data_loader=small_loader,
        device=accelerator.device,
        adapter_name="default",
        opts=("lora", "adapter"),
        num_batches=8,
    )
    logger.info("采集完成: 梯度=%s, 激活=%s", process_info["gradients"], process_info["activations"])

    # 基于过程信息，计算所有策略下的待剪枝排序
    rankings = compute_pruning_rankings(
        model=model,
        adapter_name="default",
        opts=("lora", "adapter"),
        process_info=process_info,
        top_p=3,
    )
    for method, groups in rankings.items():
        logger.info("策略[%s] Top-%d 分组:", method, len(groups))
        for prefix, names, score in groups:
            logger.info("  prefix=%s, score=%.6f, count=%d", prefix, score, len(names))

    model = train_text_to_text_model(
        run_name=os.path.join("peft_test", wandb_name),
        train_dataset=train_set,
        valid_dataset=val_set,
        model=model,
        tokenizer=tokenizer,
        model_type=model_type,
        num_train_epochs=1,
        per_device_batch_size=1,
        real_batch_size=128,
        bf16=(model_dtype == "bf16"),
        eval_epochs=1,
        early_stopping_patience=3,
        max_length=1024,
        logging_steps=10,
        use_loraplus=False,
        loraplus_lr_ratio=None,
        learning_rate=2e-5,
        num_process=accelerator.num_processes,
        gradient_checkpointing=False,
        seed=seed,
        training_args=dict(
            lr_scheduler_type="cosine",
            max_grad_norm=1.0,
            warmup_ratio=0.03,
            weight_decay=0.0,
        ),
    )

    if accelerator.is_local_main_process:
        model.save_pretrained(save_dir)

        model, tokenizer = initialize_text_to_text_model(
            model_id, model_type, model_dtype, flash_attention=False
        )
        model = PeftModel.from_pretrained(model, save_dir)
        logger.info("最终PrunePEFT模型:")
        logger.info(model)


if __name__ == "__main__":
    Fire(main)
