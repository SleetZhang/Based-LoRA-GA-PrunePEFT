#!/bin/bash

# 微调方法配置脚本
# 使用方法: examples/baseline.sh [method] [additional_args]
# 示例: examples/baseline.sh lora --lora_rank=16 --lora_alpha=32

# 设置默认参数
DEFAULT_LORA_RANK=32
DEFAULT_LORA_ALPHA=8
DEFAULT_SAMPLE_SIZE=128
DEFAULT_SEED=31

# 获取微调方法参数
TUNING_METHOD=${1:-lora}

echo "=========================================="
echo "启动微调训练"
echo "微调方法: $TUNING_METHOD"
echo "=========================================="

# 根据微调方法设置特定参数
case $TUNING_METHOD in
    "lora")
        echo "使用 LoRA 微调方法"
        python examples/baseline.py \
            --tuning_method=lora \
            --lora_rank=$DEFAULT_LORA_RANK \
            --lora_alpha=$DEFAULT_LORA_ALPHA \
            --lora_dropout=0.1 \
            --sample_size=$DEFAULT_SAMPLE_SIZE \
            --seed=$DEFAULT_SEED \
            "${@:2}"
        ;;
    
    "dora")
        echo "使用 DoRA (Weight-Decomposed LoRA) 微调方法"
        python examples/baseline.py \
            --tuning_method=dora \
            --lora_rank=$DEFAULT_LORA_RANK \
            --lora_alpha=$DEFAULT_LORA_ALPHA \
            --lora_dropout=0.1 \
            --sample_size=$DEFAULT_SAMPLE_SIZE \
            --seed=$DEFAULT_SEED \
            "${@:2}"
        ;;
    
    "adalora")
        echo "使用 AdaLoRA 微调方法"
        python examples/baseline.py \
            --tuning_method=adalora \
            --lora_rank=$DEFAULT_LORA_RANK \
            --lora_alpha=$DEFAULT_LORA_ALPHA \
            --target_r=8 \
            --init_r=12 \
            --tinit=0 \
            --tfinal=1000 \
            --deltaT=10 \
            --sample_size=$DEFAULT_SAMPLE_SIZE \
            --seed=$DEFAULT_SEED \
            "${@:2}"
        ;;
    
    "ia3")
        echo "使用 IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations) 微调方法"
        python examples/baseline.py \
            --tuning_method=ia3 \
            --sample_size=$DEFAULT_SAMPLE_SIZE \
            --seed=$DEFAULT_SEED \
            "${@:2}"
        ;;
    
    "adaption_prompt")
        echo "使用 Adaption Prompt 微调方法"
        python examples/baseline.py \
            --tuning_method=adaption_prompt \
            --adapter_len=10 \
            --adapter_layers=30 \
            --sample_size=$DEFAULT_SAMPLE_SIZE \
            --seed=$DEFAULT_SEED \
            "${@:2}"
        ;;
    
    "loha")
        echo "使用 LoHa (Low-Rank Hadamard Product) 微调方法"
        python examples/baseline.py \
            --tuning_method=loha \
            --lora_rank=$DEFAULT_LORA_RANK \
            --lora_alpha=$DEFAULT_LORA_ALPHA \
            --rank_dropout=0.0 \
            --module_dropout=0.0 \
            --sample_size=$DEFAULT_SAMPLE_SIZE \
            --seed=$DEFAULT_SEED \
            "${@:2}"
        ;;
    
    "lokr")
        echo "使用 LoKr (Low-Rank Kronecker Product) 微调方法"
        python examples/baseline.py \
            --tuning_method=lokr \
            --lora_rank=$DEFAULT_LORA_RANK \
            --lora_alpha=$DEFAULT_LORA_ALPHA \
            --rank_dropout=0.0 \
            --module_dropout=0.0 \
            --sample_size=$DEFAULT_SAMPLE_SIZE \
            --seed=$DEFAULT_SEED \
            "${@:2}"
        ;;
    
    "boft")
        echo "使用 BOFT (Butterfly Factorization) 微调方法"
        python examples/baseline.py \
            --tuning_method=boft \
            --boft_block_size=4 \
            --boft_n_butterfly_factor=1 \
            --boft_dropout=0.1 \
            --sample_size=$DEFAULT_SAMPLE_SIZE \
            --seed=$DEFAULT_SEED \
            "${@:2}"
        ;;
    
    "oft")
        echo "使用 OFT (Orthogonal Fine-Tuning) 微调方法"
        python examples/baseline.py \
            --tuning_method=oft \
            --lora_rank=$DEFAULT_LORA_RANK \
            --oft_block_size=4 \
            --oft_dropout=0.1 \
            --sample_size=$DEFAULT_SAMPLE_SIZE \
            --seed=$DEFAULT_SEED \
            "${@:2}"
        ;;
    
    "loraga")
        echo "使用 LoRA-GA (LoRA with Gradient Approximation) 微调方法"
        python examples/baseline.py \
            --tuning_method=loraga \
            --lora_rank=$DEFAULT_LORA_RANK \
            --lora_alpha=$DEFAULT_LORA_ALPHA \
            --sample_size=$DEFAULT_SAMPLE_SIZE \
            --seed=$DEFAULT_SEED \
            "${@:2}"
        ;;
    
    "bottleneck")
        echo "使用 Bottleneck (传统瓶颈层) 微调方法"
        python examples/baseline.py \
            --tuning_method=bottleneck \
            --bottleneck_size=64 \
            --bottleneck_dropout=0.1 \
            --init_bottleneck_weights=True \
            --sample_size=$DEFAULT_SAMPLE_SIZE \
            --seed=$DEFAULT_SEED \
            "${@:2}"
        ;;
    
    *)
        echo "错误: 不支持的微调方法 '$TUNING_METHOD'"
        echo ""
        echo "支持的微调方法:"
        echo "  lora          - LoRA (Low-Rank Adaptation)"
        echo "  dora          - DoRA (Weight-Decomposed LoRA)"
        echo "  adalora       - AdaLoRA (Adaptive LoRA)"
        echo "  ia3           - IA³ (Infused Adapter)"
        echo "  adaption_prompt - Adaption Prompt"
        echo "  loha          - LoHa (Low-Rank Hadamard Product)"
        echo "  lokr          - LoKr (Low-Rank Kronecker Product)"
        echo "  boft          - BOFT (Butterfly Factorization)"
        echo "  oft           - OFT (Orthogonal Fine-Tuning)"
        echo "  loraga        - LoRA-GA (LoRA with Gradient Approximation)"
        echo "  bottleneck    - Bottleneck (传统瓶颈层)"
        echo ""
        echo "使用示例:"
        echo "  examples/baseline.sh lora --lora_rank=32 --lora_alpha=8"
        echo "  examples/baseline.sh dora --lora_rank=64"
        echo "  examples/baseline.sh adalora --target_r=4 --init_r=8"
        echo "  examples/baseline.sh ia3"
        echo "  examples/baseline.sh loraga --sample_size=256"
        echo "  examples/baseline.sh bottleneck --bottleneck_size=32 --bottleneck_dropout=0.05"
        exit 1
        ;;
esac

echo "=========================================="
echo "微调训练完成"
echo "=========================================="