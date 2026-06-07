import torch
from fire import Fire
import os
import wandb
from accelerate import Accelerator

# PEFT imports for different tuning methods
from peft import (
    PeftModel, 
    get_peft_model,
    LoraConfig,
    AdaLoraConfig,
    IA3Config,
    AdaptionPromptConfig,
    LoHaConfig,
    LoKrConfig,
    BOFTConfig,
    OFTConfig,
    LoraGAConfig,
    BottleneckConfig,
)

# LoRA-GA specific imports
from peft.utils.lora_ga_utils import (
    estimate_gradient,
    LoraGAContext,
    save_loraga_model_init,
    save_loraga_model_final,
)

from examples.utils import (
    transform_dataset,
    initialize_text_to_text_model,
    find_all_linear_modules,
    train_text_to_text_model,
)
from examples.data import DATASET_MAP


def get_peft_config(tuning_method, model, **kwargs):
    """
    根据微调方法返回相应的PEFT配置
    
    Args:
        tuning_method: 微调方法名称
        model: 基础模型
        **kwargs: 配置参数
    """
    target_modules = find_all_linear_modules(model=model)
    
    if tuning_method.lower() == "lora":
        return LoraConfig(
            target_modules=target_modules,
            r=kwargs.get("lora_rank", 32),
            lora_alpha=kwargs.get("lora_alpha", 8),
            lora_dropout=kwargs.get("lora_dropout", 0.1),
            bias=kwargs.get("bias", "none"),
            use_rslora=kwargs.get("use_rslora", False),
            init_lora_weights=kwargs.get("init_lora_weights", True),
        )
    
    elif tuning_method.lower() == "dora":
        return LoraConfig(
            target_modules=target_modules,
            r=kwargs.get("lora_rank", 32),
            lora_alpha=kwargs.get("lora_alpha", 8),
            lora_dropout=kwargs.get("lora_dropout", 0.1),
            bias=kwargs.get("bias", "none"),
            use_dora=True,  # 启用DoRA
            init_lora_weights=kwargs.get("init_lora_weights", True),
        )
    
    elif tuning_method.lower() == "adalora":
        return AdaLoraConfig(
            target_modules=target_modules,
            r=kwargs.get("lora_rank", 32),
            lora_alpha=kwargs.get("lora_alpha", 8),
            lora_dropout=kwargs.get("lora_dropout", 0.1),
            target_r=kwargs.get("target_r", 8),
            init_r=kwargs.get("init_r", 12),
            tinit=kwargs.get("tinit", 0),
            tfinal=kwargs.get("tfinal", 1000),
            deltaT=kwargs.get("deltaT", 10),
        )
    
    elif tuning_method.lower() == "ia3":
        return IA3Config(
            target_modules=target_modules,
            feedforward_modules=kwargs.get("feedforward_modules", None),
        )
    
    elif tuning_method.lower() == "adaption_prompt":
        return AdaptionPromptConfig(
            adapter_len=kwargs.get("adapter_len", 10),
            adapter_layers=kwargs.get("adapter_layers", 30),
        )
    
    elif tuning_method.lower() == "loha":
        return LoHaConfig(
            target_modules=target_modules,
            r=kwargs.get("lora_rank", 32),
            alpha=kwargs.get("lora_alpha", 8),
            rank_dropout=kwargs.get("rank_dropout", 0.0),
            module_dropout=kwargs.get("module_dropout", 0.0),
        )
    
    elif tuning_method.lower() == "lokr":
        return LoKrConfig(
            target_modules=target_modules,
            r=kwargs.get("lora_rank", 32),
            alpha=kwargs.get("lora_alpha", 8),
            rank_dropout=kwargs.get("rank_dropout", 0.0),
            module_dropout=kwargs.get("module_dropout", 0.0),
        )
    
    elif tuning_method.lower() == "boft":
        return BOFTConfig(
            target_modules=target_modules,
            boft_block_size=kwargs.get("boft_block_size", 4),
            boft_n_butterfly_factor=kwargs.get("boft_n_butterfly_factor", 1),
            boft_dropout=kwargs.get("boft_dropout", 0.1),
        )
    
    elif tuning_method.lower() == "oft":
        return OFTConfig(
            target_modules=target_modules,
            r=kwargs.get("lora_rank", 32),
            oft_block_size=kwargs.get("oft_block_size", 4),
            oft_dropout=kwargs.get("oft_dropout", 0.1),
        )
    
    elif tuning_method.lower() == "loraga":
        return LoraGAConfig(
            target_modules=target_modules,
            lora_alpha=kwargs.get("lora_alpha", 8),
            r=kwargs.get("lora_rank", 32),
            iters=kwargs.get("sample_size", 128) // 2,
        )
    
    elif tuning_method.lower() == "bottleneck":
        return BottleneckConfig(
            target_modules=target_modules,
            bottleneck_size=kwargs.get("bottleneck_size", 64),
            bottleneck_dropout=kwargs.get("bottleneck_dropout", 0.1),
            bias=kwargs.get("bias", "none"),
            init_bottleneck_weights=kwargs.get("init_bottleneck_weights", True),
        )
    
    else:
        raise ValueError(f"不支持的微调方法: {tuning_method}")


def main(
    tuning_method="lora",  # 微调方法选择
    lora_alpha=8,
    lora_rank=32,
    lora_dropout=0.1,
    sample_size=128,
    seed=31,
    # DoRA特定参数
    use_dora=False,
    # AdaLoRA特定参数
    target_r=8,
    init_r=12,
    tinit=0,
    tfinal=1000,
    deltaT=10,
    # IA3特定参数
    feedforward_modules=None,
    # AdaptionPrompt特定参数
    adapter_len=10,
    adapter_layers=30,
    # BOFT特定参数
    boft_block_size=4,
    boft_n_butterfly_factor=1,
    boft_dropout=0.1,
    # OFT特定参数
    oft_block_size=4,
    oft_dropout=0.1,
    # Bottleneck特定参数
    bottleneck_size=64,
    bottleneck_dropout=0.1,
    init_bottleneck_weights=True,
    # 其他参数
    bias="none",
    use_rslora=False,
    init_lora_weights=True,
):
    """
    主训练函数，支持多种微调方法
    
    Args:
        tuning_method: 微调方法 (lora, dora, adalora, ia3, adaption_prompt, loha, lokr, boft, oft, loraga, bottleneck)
        其他参数: 各种微调方法的特定参数
    """
    accelerator = Accelerator()
    model_id = "/home/autopeft/LoRA-GA/ckpts/pretrained/Llama-2-7b-hf"
    model_type = "CausalLM"
    model_dtype = "bf16"
    dataset_name = "meta_math"
    
    # 配置信息
    config = dict(
        model="llama",
        method=tuning_method,
        d=dataset_name,
        a=lora_alpha,
        r=lora_rank,
        s=sample_size,
        sd=seed,
    )
    
    wandb_name = "_".join([f"{k}={v}" for k, v in config.items()])
    
    if accelerator.is_local_main_process:
        wandb.init(
            name=wandb_name,
            mode="offline",
            group="tuning_methods_comparison",
            project="Multi-PEFT Methods",
        )
    
    # 初始化模型
    model, tokenizer = initialize_text_to_text_model(
        model_id, model_type, model_dtype, flash_attention=False
    )
    
    if accelerator.is_local_main_process:
        print(f"使用微调方法: {tuning_method.upper()}")
        print(model)
    
    # 获取PEFT配置
    peft_config = get_peft_config(
        tuning_method=tuning_method,
        model=model,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        sample_size=sample_size,
        target_r=target_r,
        init_r=init_r,
        tinit=tinit,
        tfinal=tfinal,
        deltaT=deltaT,
        feedforward_modules=feedforward_modules,
        adapter_len=adapter_len,
        adapter_layers=adapter_layers,
        boft_block_size=boft_block_size,
        boft_n_butterfly_factor=boft_n_butterfly_factor,
        boft_dropout=boft_dropout,
        oft_block_size=oft_block_size,
        oft_dropout=oft_dropout,
        bias=bias,
        use_rslora=use_rslora,
        init_lora_weights=init_lora_weights,
    )
    
    # 准备数据集
    dataset_func = DATASET_MAP[dataset_name]
    train_set, val_set, _ = dataset_func()
    
    # LoRA-GA特殊处理
    if tuning_method.lower() == "loraga":
        if isinstance(train_set, list):
            temp_set = train_set[: peft_config.bsz * peft_config.iters]
        else:
            temp_set = train_set.select(range(peft_config.bsz * peft_config.iters))
        
        transform_dataset(
            model_type=model_type,
            dataset=temp_set,
            tokenizer=tokenizer,
            max_length=peft_config.max_length,
        )
        dataloader = torch.utils.data.DataLoader(temp_set, batch_size=peft_config.bsz)
        
        # 估计梯度
        named_grad = estimate_gradient(
            model=model,
            dataloader=dataloader,
            accelerator=accelerator,
            quant_flag=False,
        )
        
        if accelerator.is_local_main_process:
            print(peft_config)
        
        # 使用LoraGAContext
        with LoraGAContext(model=model, named_grad=named_grad):
            model = get_peft_model(model=model, peft_config=peft_config)
    else:
        # 其他微调方法的标准处理
        model = get_peft_model(model=model, peft_config=peft_config)
    
    # 保存目录
    save_dir = os.path.join("./snapshot", wandb_name)
    
    if accelerator.is_local_main_process:
        print(f"PEFT模型配置完成，使用方法: {tuning_method}")
        print(model)
        model.print_trainable_parameters()
        
        # LoRA-GA特殊保存
        if tuning_method.lower() == "loraga":
            save_loraga_model_init(model=model, save_dir=save_dir)
    
    print(f"完成 {tuning_method} 模型配置 ===============================================")
    
    # 训练模型
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
    
    # 保存最终模型
    if accelerator.is_local_main_process:
        if tuning_method.lower() == "loraga":
            save_loraga_model_final(model=model, save_dir=save_dir)
        else:
            model.save_pretrained(save_dir)
        
        # 加载并测试模型
        model, tokenizer = initialize_text_to_text_model(
            model_id, model_type, model_dtype, flash_attention=False
        )
        model = PeftModel.from_pretrained(model, save_dir)
        print(f"最终 {tuning_method} 模型:")
        print(model)


if __name__ == "__main__":
    Fire(main)