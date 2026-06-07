#!/bin/bash

# 参数配置
DATASET="meta_math"
TEST_DATASET="gsm8k"
# TEST_DATASET="humaneval"
# TEST_DATASET="mt-bench"
ADAPTER_TYPES="lora"
LORA_RANK=8
LORA_ALPHA=16
LORA_DROPOUT=0.1
BOTTLENECK_SIZE=32
BOTTLENECK_DROPOUT=0.1
INIT_BOTTLENECK_WEIGHTS=true
ADAPTER_LAYERS=""
LORA_LAYERS=""
TARGET_MODULES="q_proj,v_proj,k_proj,o_proj"
SAMPLE_SIZE=128
SEED=42
BIAS="none"
PRUNING_ROUNDS=4
MODULES_PER_ROUND=8
# RANDOM_PRUNING=true
MODEL_PATH=""
ADAPTER_PATH=""
SAVE_PATH="/home/autopeft/LoRA-GA/save/lora_1e4"
STAGE=2,3

# 执行
python examples/controller.py \
    --dataset "$DATASET" \
    --adapter_types "$ADAPTER_TYPES" \
    --lora_rank "$LORA_RANK" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout "$LORA_DROPOUT" \
    --bottleneck_size "$BOTTLENECK_SIZE" \
    --bottleneck_dropout "$BOTTLENECK_DROPOUT" \
    --init_bottleneck_weights "$INIT_BOTTLENECK_WEIGHTS" \
    --adapter_layers "$ADAPTER_LAYERS" \
    --lora_layers "$LORA_LAYERS" \
    --target_modules "$TARGET_MODULES" \
    --sample_size "$SAMPLE_SIZE" \
    --seed "$SEED" \
    --bias "$BIAS" \
    --pruning_rounds "$PRUNING_ROUNDS" \
    --modules_per_round "$MODULES_PER_ROUND" \
    ${RANDOM_PRUNING:+--random_pruning} \
    --test_dataset "$TEST_DATASET" \
    ${MODEL_PATH:+--model_path "$MODEL_PATH"} \
    ${ADAPTER_PATH:+--adapter_path "$ADAPTER_PATH"} \
    ${SAVE_PATH:+--save_path "$SAVE_PATH"} \
    --stage "$STAGE"
