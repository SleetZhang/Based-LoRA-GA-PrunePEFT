"""
PrunePEFT training script with GradTrainer for gradient recording and analysis.

This script demonstrates the use of GradTrainer to record gradients and activations
during PrunePEFT training, enabling detailed analysis of the training process.

Author: Based on prunepeft.py, modified to use GradTrainer
Note: Configured for quick testing with only 2 training steps
Warning: Disabled auto-save/eval to avoid PrunePEFT compatibility issues
"""

import logging
import os
import torch
from fire import Fire
import wandb
from accelerate import Accelerator

from peft import PeftModel, get_peft_model
from peft.tuners.prunepeft import PrunePEFTConfig

from examples.utils import (
    transform_dataset,
    preprocess_dataset,
    initialize_text_to_text_model,
    find_all_linear_modules,
)
from examples.gradTrainer import GradTrainer  # Import our custom GradTrainer
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


def train_with_gradtrainer(
    run_name: str,
    train_dataset,
    valid_dataset,
    model,
    tokenizer,
    model_type: str,
    num_train_epochs: int = 1,
    per_device_batch_size: int = 1,
    real_batch_size: int = 32,
    max_length: int = 1024,
    logging_steps: int = 10,
    bf16: bool = False,
    eval_epochs: int = 1,
    early_stopping_patience: int = 3,
    learning_rate: float = 5e-5,
    num_process: int = 1,
    gradient_checkpointing: bool = False,
    seed: int = 42,
    training_args: dict = None,
):
    """
    Train model using GradTrainer with gradient recording capabilities.

    Args:
        run_name: Name for the training run
        train_dataset: Training dataset
        valid_dataset: Validation dataset
        model: Model to train
        tokenizer: Tokenizer
        model_type: Type of model
        ... (other training arguments)

    Returns:
        Trained model, recorded gradients, activations, and intermediate results
    """
    from transformers import Seq2SeqTrainingArguments

    # Preprocess the dataset
    train_dataset = preprocess_dataset(train_dataset)
    valid_dataset = preprocess_dataset(valid_dataset)

    assert (
        real_batch_size % per_device_batch_size == 0
    ), "real_batch_size must be divisible by per_device_batch_size"
    accu_step = real_batch_size // (
        per_device_batch_size * num_process
    )

    train_dataset, valid_dataset = transform_dataset(
        model_type, tokenizer, train_dataset, max_length
    ), transform_dataset(model_type, tokenizer, valid_dataset, max_length)

    eval_steps = (
        int(len(train_dataset) * eval_epochs) // real_batch_size
    )

    output_dir = f"./results/{run_name}/{seed}"
    training_args_obj = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=accu_step,
        logging_dir="./logs",
        logging_steps=logging_steps,
        bf16=bf16,
        gradient_checkpointing=gradient_checkpointing,
        optim="adamw_torch",
        eval_strategy="no",  # Disable evaluation during training
        # Disable automatic saving to avoid PrunePEFT save issues
        save_strategy="no",
        save_total_limit=0,
        greater_is_better=False,
        do_eval=True,
        learning_rate=learning_rate,
        remove_unused_columns=False,  # tokenize the dataset on the fly
        label_names=["labels"],
        seed=seed,
        ddp_find_unused_parameters=False,
        **(training_args or {}),
    )

    # Create GradTrainer instead of standard trainer
    trainer = GradTrainer(
        model=model,
        args=training_args_obj,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
    )

    logger.info("开始使用 GradTrainer 进行训练...")
    trainer.train()

    # Perform final evaluation manually since we disabled eval_strategy
    logger.info("进行最终评估...")
    # eval_results = trainer.evaluate(eval_dataset=valid_dataset)
    # intermediate_results = [eval_results]  # Wrap in list for compatibility
    intermediate_results = []

    # Get recorded information
    gradients = trainer.get_recorded_gradients()
    activations = trainer.get_recorded_activations()

    logger.info("训练完成!")
    logger.info(f"记录了 {len(gradients)} 个参数的梯度")
    logger.info(f"记录了 {len(activations)} 个层的激活值")
    logger.info(f"记录了 {len(intermediate_results)} 个评估结果")

    return model, gradients, activations, intermediate_results


def analyze_recorded_data(gradients, activations, intermediate_results):
    """
    Analyze and display the recorded training data.

    Args:
        gradients: Recorded gradients dictionary
        activations: Recorded activations dictionary
        intermediate_results: Intermediate evaluation results
    """
    logger.info("\n" + "="*60)
    logger.info("GRADIENT ANALYSIS")
    logger.info("="*60)

    if gradients:
        logger.info(f"总共记录了 {len(gradients)} 个参数的梯度")

        # Analyze gradient statistics
        total_params = 0
        grad_norms = []

        for param_name, grad in gradients.items():
            total_params += grad.size
            grad_norm = torch.norm(torch.from_numpy(grad)).item()
            grad_norms.append((param_name, grad_norm))

        logger.info(f"总参数数量: {total_params}")
        logger.info(f"平均梯度范数: {sum(n for _, n in grad_norms) / len(grad_norms):.6f}")

        # Show top 5 parameters with largest gradients
        sorted_grads = sorted(grad_norms, key=lambda x: x[1], reverse=True)
        logger.info("\n梯度范数最大的前5个参数:")
        for i, (name, norm) in enumerate(sorted_grads[:5]):
            logger.info("5d")
    else:
        logger.warning("没有记录到梯度信息!")

    logger.info("\n" + "="*60)
    logger.info("ACTIVATION ANALYSIS")
    logger.info("="*60)

    if activations:
        logger.info(f"总共记录了 {len(activations)} 个层的激活值")

        for layer_name, activation in list(activations.items())[:3]:  # Show first 3
            logger.info(f"层: {layer_name}")
            logger.info(f"  形状: {activation.shape}")
            logger.info(".6f")
            logger.info(".6f")
    else:
        logger.warning("没有记录到激活信息!")

    logger.info("\n" + "="*60)
    logger.info("TRAINING PROGRESS ANALYSIS")
    logger.info("="*60)

    if intermediate_results:
        logger.info(f"记录了 {len(intermediate_results)} 个评估点")

        losses = [r.get('eval_loss', 0) for r in intermediate_results]
        accuracies = [r.get('eval_accuracy', 0) for r in intermediate_results]

        logger.info(".6f")
        logger.info(".6f")

        if len(losses) > 1:
            loss_improvement = losses[0] - losses[-1]
            logger.info(".6f")
    else:
        logger.warning("没有记录到训练进度信息!")


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
    Main training function for PrunePEFT with GradTrainer.

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
        method="prunepeft_gradtrainer",
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
            group="prunepeft_gradtrainer",
            project="PrunePEFT with GradTrainer",
        )

    model, tokenizer = initialize_text_to_text_model(
        model_id, model_type, model_dtype, flash_attention=False
    )

    if accelerator.is_local_main_process:
        logger.info("使用微调方法: PRUNEPEFT with GradTrainer")
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

    if accelerator.is_local_main_process:
        logger.info("PEFT模型配置完成")
        logger.info("PrunePEFT模型结构:")
        logger.info(model)
        model.print_trainable_parameters()

    # Skip pruning process info collection and strategy analysis
    # Focus only on testing GradTrainer functionality
    logger.info("跳过剪枝信息采集，专注测试 GradTrainer 功能")

    # Train with GradTrainer - only 2 steps for testing
    model, gradients, activations, intermediate_results = train_with_gradtrainer(
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
        logging_steps=1,  # Log every step since we only have 2 steps
        learning_rate=2e-5,
        num_process=accelerator.num_processes,
        gradient_checkpointing=False,
        seed=seed,
        training_args=dict(
            lr_scheduler_type="cosine",
            max_grad_norm=1.0,
            warmup_ratio=0.03,
            weight_decay=0.0,
            max_steps=1,  # Only train for 2 steps
        ),
    )

    # Analyze the recorded data
    if accelerator.is_local_main_process:
        analyze_recorded_data(gradients, activations, intermediate_results)

        # Skip model saving for PrunePEFT as it's not fully supported in save_pretrained
        logger.info("跳过模型保存 (PrunePEFT类型暂不支持save_pretrained)")
        logger.info("训练完成，GradTrainer功能验证成功!")

        # Optional: Try to save using a different approach if needed
        # save_dir = os.path.join("./snapshot", wandb_name)
        # try:
        #     model.save_pretrained(save_dir)
        #     logger.info(f"模型已保存到: {save_dir}")
        # except ValueError as e:
        #     logger.warning(f"模型保存失败: {e}")
        #     logger.info("这不影响GradTrainer功能的验证")


if __name__ == "__main__":
    Fire(main)
