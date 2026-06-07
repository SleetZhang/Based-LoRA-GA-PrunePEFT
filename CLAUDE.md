# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This repository is a fork/custom implementation of Hugging Face's PEFT (Parameter-Efficient Fine-Tuning) library with additional focus on:
1. **LoRA-GA** (Low-Rank Adaptation with Gradient Approximation) - A novel initialization method that aligns gradients of low-rank matrix products with full fine-tuning
2. **PrunePEFT** - Layer-wise pruning functionality for PEFT methods (the main focus of this repository)
3. Various other PEFT methods (LoRA, AdaLoRA, BOFT, etc.)

The repository contains multiple research projects, with the main implementation in the `LoRA-GA` directory. The primary purpose of this repository is to implement and experiment with PrunePEFT methods.

## Installation and Setup

To work with this codebase:

```bash
# Navigate to the main project directory
cd /home/autopeft/LoRA-GA

# Install dependencies
pip install -r requirements.txt

# Install the custom PEFT library in development mode
pip install -e peft
```

## Development Commands

### Testing
```bash
# Run all tests
make test

# Run tests with specific markers
make tests_examples_multi_gpu
make tests_examples_single_gpu
make tests_core_multi_gpu
make tests_core_single_gpu

# Run specific test files
pytest tests/test_lora_megatron.py
pytest tests/test_decoder_models.py

# Run tests with specific patterns
pytest tests/ -k "lora"
pytest tests/ -k "gpu"
```

### Code Quality
```bash
# Check code quality (runs ruff check and format --check)
make quality

# Automatically fix code style issues
make style

# Run individual quality tools
ruff check src tests examples docs scripts docker
ruff format --check src tests examples docs scripts docker
```

### Build and Distribution
```bash
# Build distribution packages
python setup.py bdist_wheel
python setup.py sdist
```

## Key Components and Architecture

### 1. Core PEFT Library Structure (`peft/src/peft/`)
- `__init__.py`: Main entry point exposing all public APIs
- `peft_model.py`: Core PeftModel classes that wrap base models with PEFT functionality
- `config.py`: Base configuration classes
- `mapping.py`: Model type to PEFT model mapping
- `mixed_model.py`: Support for mixed PEFT models
- `auto.py`: Auto classes for model selection

#### Tuners Directory (`peft/src/peft/tuners/`)
Each PEFT method has its own subdirectory:
- `lora/`: Standard LoRA implementation
- `lora_ga/`: LoRA-GA specific implementation
- `adalora/`: Adaptive LoRA
- `boft/`: Block-wise Orthogonal Fine-Tuning
- `prunepeft/`: **Main focus of this repository** - Pruning-enabled PEFT methods
- `ia3/`: IA³ (Injective Adapter with Adaptive Activation Amplitude)
- And many others...

### 2. LoRA-GA Specific Implementation
Located in `peft/src/peft/utils/lora_ga_utils/`:
- `LoraGAConfig`: Extends LoraConfig with gradient approximation initialization
- `estimate_gradient`: Estimates gradients from data for initialization
- `LoraGAContext`: Context manager for gradient handling during model initialization
- `save_loraga_model_init/final`: Specialized saving functions for LoRA-GA models

### 3. PrunePEFT Implementation (Main Focus)
Located in `peft/src/peft/tuners/prunepeft/`:
- `PrunePEFTConfig`: Configuration supporting multiple adapter types with layer selection
- `model.py`: Main PrunePEFT model implementation with pruning capabilities
- `config.py`: PrunePEFT configuration definitions
- `lora_layer.py`: LoRA layer implementation with pruning support
- `combined_layer.py`: Combined adapter layers for mixed configurations
- `collect_pruning_process_info`: Gathers gradient and activation information for ranking
- `compute_pruning_rankings`: Computes layer importance scores for pruning decisions
- Iterative pruning capabilities with configurable rounds

## Key Directories and Their Purpose

### Root Directories
- `/home/autopeft/LoRA-GA/`: Main project directory
- `/home/autopeft/LoRA-GA/peft/`: Custom PEFT library implementation
- `/home/autopeft/LoRA-GA/examples/`: High-level example scripts and controllers
- `/home/autopeft/LoRA-GA/peft/examples/`: PEFT-specific examples for different methods
- `/home/autopeft/LoRA-GA/peft/tests/`: Unit and integration tests
- `/home/autopeft/LoRA-GA/peft/src/peft/`: PEFT source code

### PrunePEFT Specific Implementation (`peft/src/peft/tuners/prunepeft/`)
- `config.py`: PrunePEFT configuration class with support for layer selection
- `model.py`: Core PrunePEFT model implementation with iterative pruning logic
- `lora_layer.py`: Extended LoRA layers with pruning awareness
- `combined_layer.py`: Support for combining multiple adapter types
- `adapter_layer.py`: Base adapter layer implementation
- Other supporting modules for different quantization methods

### Example Implementations
- `peft/examples/lora_ga_finetuning/`: Complete LoRA-GA training pipelines
- `examples/controller.py`: **Primary interface for PrunePEFT experimentation**
- `peft/examples/prunepeft/`: Pruning-focused examples
- `peft/examples/*/`: Examples for each PEFT method (boft, ia3, etc.)

## Common Development Tasks

### Running Example Scripts
```bash
# Main PrunePEFT controller script (primary interface)
python examples/controller.py --adapter_types lora --lora_rank 8
python examples/controller.py --test_dataset gsm8k
python examples/controller.py --pruning_rounds 2 --modules_per_round 1

# LoRA-GA examples
python peft/examples/lora_ga_finetuning/float_llama2-7b_metamath.py
python peft/examples/lora_ga_finetuning/quant_llama-2-7b_metamath.py
```

### Working with Models
```python
# Basic PEFT usage (standard)
from peft import get_peft_model, LoraConfig
peft_config = LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, peft_config)

# LoRA-GA usage
from peft import LoraGAConfig, get_peft_model
from peft.utils.lora_ga_utils import estimate_gradient, LoraGAContext
peft_config = LoraGAConfig(r=8, lora_alpha=32)
named_grad = estimate_gradient(model, dataloader, accelerator)
with LoraGAContext(model=model, named_grad=named_grad):
    model = get_peft_model(model, peft_config)

# PrunePEFT usage (main focus of this repository)
from peft.tuners.prunepeft import PrunePEFTConfig
peft_config = PrunePEFTConfig(
    adapter_types=["lora"],
    r=8,
    lora_alpha=32,
    lora_layers=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
)
model = get_peft_model(model, peft_config)
```

## Making Changes

### Code Quality Standards
- Use `make quality` before committing to ensure code meets standards
- Follow existing docstring conventions
- Maintain type hints where present
- Keep backward compatibility with Hugging Face PEFT APIs

### Adding New PEFT Methods
1. Create new directory in `peft/src/peft/tuners/`
2. Implement `*Config` and `*Model` classes following existing patterns
3. Add conditional imports in `peft/src/peft/tuners/__init__.py`
4. Create configuration mappings in `peft/src/peft/mapping.py`
5. Add tests in `peft/tests/`
6. Provide examples in `peft/examples/`

### Modifying Existing Functionality
- Prefer extending rather than modifying core classes
- Follow adapter pattern used throughout the codebase
- Maintain backward compatibility
- Update documentation when changing APIs

## Repository-Specific Information

### Model Paths
Models are typically referenced with hardcoded paths:
- Base models: `/home/autopeft/LoRA-GA/ckpts/pretrained/Llama-2-7b-hf`
- Saved checkpoints: `./snapshot/` directory with timestamped folders

### Environment Variables
Some scripts reference environment variables for distributed training:
- `CUDA_VISIBLE_DEVICES`: Controls GPU visibility
- Configuration files in `examples/accelerate_config.yaml` for multi-GPU setups

### Common Datasets
Scripts use various datasets through the `DATASET_MAP` in `examples/data.py`:
- `meta_math`: Mathematical reasoning datasets
- `alpaca`: Instruction-following datasets
- `gsm8k`: Grade school math problems
- Others defined in the data module

## Debugging Tips

### Memory Issues
- Use gradient checkpointing for large models
- Monitor with `nvidia-smi` for GPU memory usage
- Enable CPU offloading where supported
- Reduce batch sizes in data loaders

### Test-Specific Commands
```bash
# Run PrunePEFT with specific dataset
python examples/controller.py --dataset meta_math --adapter_types lora

# Run with pruning iterations
python examples/controller.py --pruning_rounds 2 --modules_per_round 1

# Run with evaluation
python examples/controller.py --test_dataset gsm8k

# Run with model loading for testing
python examples/controller.py --test_dataset gsm8k --model_path /path/to/saved/model
```

## Best Practices

1. **Always run tests** before submitting changes
2. **Use existing patterns** when adding new functionality
3. **Keep changes minimal** and focused on specific features
4. **Document new APIs** with clear docstrings
5. **Maintain compatibility** with upstream Hugging Face PEFT where possible

## PrunePEFT Specific Development Guide

### Main Entry Point
The primary interface for experimenting with PrunePEFT is:
`examples/controller.py` - This script handles:
- Training with various adapter configurations
- Iterative pruning workflows
- Evaluation on different benchmarks (GSM8K, HumanEval, MT-Bench)
- Model saving and loading

### Key PrunePEFT Features
1. **Layer Selection**: Configure which layers to apply adapters to
2. **Iterative Pruning**: Remove least important layers over multiple rounds
3. **Mixed Adapter Types**: Combine LoRA, Bottleneck adapters, etc.
4. **Gradient-based Ranking**: Identify layers to prune based on importance

### Configuration Options
In `PrunePEFTConfig`, important parameters include:
- `adapter_types`: List of adapter types to use (e.g., ["lora", "bottleneck"])
- `lora_layers`: Specific layers to apply LoRA adapters
- `adapter_layers`: Specific layers to apply Bottleneck adapters
- `target_modules`: Which model modules to adapt
- `r`, `lora_alpha`: Standard LoRA parameters

### Iterative Pruning Workflow
The controller implements an iterative pruning approach:
1. Train initial model with all adapters
2. Evaluate layer importance using gradients/activations
3. Remove least important layers
4. Retrain pruned model
5. Repeat for specified rounds

This workflow is handled automatically in `examples/controller.py` when `--pruning_rounds` is specified.