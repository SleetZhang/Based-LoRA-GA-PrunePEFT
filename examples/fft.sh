#!/bin/bash

# ===== FFT (full fine-tuning) baseline: Llama-2-7b · meta_math -> gsm8k =====
# Self-contained full fine-tuning; a single A800-80GB is enough for 7B FFT.
# Run from the repo root:   bash examples/fft.sh

MODEL_ID="ckpts/pretrained/Llama-2-7b-hf"
DATASET="meta_math"          # train set (GSM-type samples); eval is always gsm8k

# -------------------------------------------------------------------------
# Learning rate: the ONE knob to sweep if FFT trails LoRA/DoRA.
#   1e-5  -> already decided (default for this run)
#   2e-5  -> TRY FIRST if FFT underfits; standard 7B full-SFT LR
#   5e-6  -> TRY if 1e-5/2e-5 looks unstable (loss spikes / eval gets worse)
# Change ONLY this line per run; SAVE_PATH below auto-includes the LR so the
# three runs land in separate folders and never overwrite each other.
LEARNING_RATE=1e-5
# -------------------------------------------------------------------------

SEED=42
DTYPE="bf16"
EPOCHS=1
PER_DEVICE_BATCH_SIZE=1
REAL_BATCH_SIZE=32
MAX_LENGTH=1024
LOGGING_STEPS=10
OPTIM="adamw_torch"
STAGE="all"                  # all | train | eval
MODEL_PATH=""                # set this (and STAGE=eval) to score an existing ckpt

# Per-LR / per-seed save dir so sweeps don't clobber each other's eval_results.txt
SAVE_PATH="./save/fft_${LEARNING_RATE}_seed${SEED}"

CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline python examples/fft_metamath_gsm8k.py \
    --model_id "$MODEL_ID" \
    --dataset "$DATASET" \
    --learning_rate "$LEARNING_RATE" \
    --seed "$SEED" \
    --dtype "$DTYPE" \
    --epochs "$EPOCHS" \
    --per_device_batch_size "$PER_DEVICE_BATCH_SIZE" \
    --real_batch_size "$REAL_BATCH_SIZE" \
    --max_length "$MAX_LENGTH" \
    --logging_steps "$LOGGING_STEPS" \
    --optim "$OPTIM" \
    --gradient_checkpointing \
    --stage "$STAGE" \
    ${MODEL_PATH:+--model_path "$MODEL_PATH"} \
    --save_path "$SAVE_PATH"
