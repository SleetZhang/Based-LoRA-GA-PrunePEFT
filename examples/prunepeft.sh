#!/bin/bash
# PrunePEFT training script
# Author: zzh
# Modify parameters below as needed
export CUDA_VISIBLE_DEVICES=2,3

# Adapter types: comma-separated list like "lora" or "lora,bottleneck"
ADAPTER_TYPES="lora,bottleneck"

# LoRA parameters (used when ADAPTER_TYPES includes "lora")
LORA_RANK=8
LORA_ALPHA=16
LORA_DROPOUT=0.1

# Bottleneck parameters (used when ADAPTER_TYPES includes "bottleneck")
BOTTLENECK_SIZE=64
BOTTLENECK_DROPOUT=0.1
INIT_BOTTLENECK_WEIGHTS=true

# Layer selection: specify which layers to apply adapters (comma-separated integers, e.g., "0,1,2")
# If empty, adapters will be applied to all layers
ADAPTER_LAYERS="0,1,2"  # Layers to apply Bottleneck adapter (block-level), e.g., "0,1,2"
LORA_LAYERS="2,3,4"  # Layers to apply LoRA adapter (linear-level), e.g., "0,1,2"

# Training parameters
TARGET_MODULES="q_proj,v_proj,k_proj,o_proj"
SAMPLE_SIZE=128
SEED=42
BIAS="none"

# Build command
CMD="python examples/prunepeft.py \
    --adapter_types=$ADAPTER_TYPES \
    --lora_rank=$LORA_RANK \
    --lora_alpha=$LORA_ALPHA \
    --lora_dropout=$LORA_DROPOUT \
    --bottleneck_size=$BOTTLENECK_SIZE \
    --bottleneck_dropout=$BOTTLENECK_DROPOUT \
    --init_bottleneck_weights=$INIT_BOTTLENECK_WEIGHTS \
    --adapter_layers=$ADAPTER_LAYERS \
    --lora_layers=$LORA_LAYERS \
    --target_modules=$TARGET_MODULES \
    --sample_size=$SAMPLE_SIZE \
    --seed=$SEED \
    --bias=$BIAS"

# Execute command
eval $CMD
