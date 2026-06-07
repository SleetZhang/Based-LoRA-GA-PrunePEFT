# PrunePEFT with GradTrainer

这个目录包含使用 GradTrainer 的 PrunePEFT 训练脚本，用于测试和验证 GradTrainer 的梯度记录功能。

## 文件说明

### `gradTrainer.py`
自定义的训练器类，结合了 `LogTrainer` 的架构和 `TrainerWithGrad` 的梯度记录功能。

**主要特性：**
- 继承自 `transformers.Trainer`，保持兼容性
- 自动记录所有可训练参数的梯度
- 通过 forward hooks 记录中间层激活值
- 提供便捷的 API 访问记录数据

### `prunepeft_gradtrainer.py`
基于原始 `prunepeft.py` 修改的脚本，专注测试 GradTrainer 功能。

**主要修改：**
- 使用 `GradTrainer` 替代标准 trainer
- **移除了剪枝策略分析**，专注验证 GradTrainer 功能
- 在训练完成后自动分析记录的梯度和激活数据
- 提供详细的训练过程分析报告

### `prunepeft_gradtrainer.sh`
对应的 shell 脚本，用于运行 PrunePEFT + GradTrainer 训练。

## 使用方法

### 基本运行

```bash
# 运行默认配置 (仅训练2步，用于快速测试)
bash examples/prunepeft_gradtrainer.sh

# 或直接运行 Python 脚本
python examples/prunepeft_gradtrainer.py
```

**注意**: 当前配置为快速测试设置，只训练 **2 步**，用于验证 GradTrainer 功能。
- 禁用了自动评估和保存以避免 PrunePEFT 兼容性问题
- 训练完成后会进行一次手动评估
- 如果需要完整训练，请修改 `max_steps` 参数和相关配置

### 自定义参数

```bash
# 修改适配器类型
ADAPTER_TYPES="lora"  # 只使用 LoRA
# 或
ADAPTER_TYPES="lora,bottleneck"  # 使用 LoRA 和 Bottleneck

# 修改 LoRA 参数
LORA_RANK=8
LORA_ALPHA=16
LORA_DROPOUT=0.1

# 修改层选择
LORA_LAYERS="2,3,4"  # 只在指定层应用 LoRA
ADAPTER_LAYERS="0,1,2"  # 只在指定层应用 Bottleneck
```

## 输出分析

训练完成后，脚本会自动输出以下分析信息：

### 1. 梯度分析
```
============================================================
GRADIENT ANALYSIS
============================================================
总共记录了 156 个参数的梯度
总参数数量: 5242880
平均梯度范数: 0.023456

梯度范数最大的前5个参数:
  1. encoder.block.2.lora_B.weight: norm=1.234567
  2. encoder.block.3.lora_A.weight: norm=1.123456
  ...
```

### 2. 激活分析
```
============================================================
ACTIVATION ANALYSIS
============================================================
总共记录了 8 个层的激活值

层: encoder.block.0.lora_A.weight
  形状: torch.Size([8, 4096])
  均值: 0.012345
  标准差: 0.056789
```

### 3. 训练进度分析
```
============================================================
TRAINING PROGRESS ANALYSIS
============================================================
记录了 10 个评估点
初始损失: 2.345678
最终损失: 1.987654
损失改善: 0.358024
```

### 3. 训练进度分析
```
============================================================
TRAINING PROGRESS ANALYSIS
============================================================
记录了 10 个评估点
初始损失: 2.345678
最终损失: 1.987654
损失改善: 0.358024
```

## GradTrainer API

```python
from examples.gradTrainer import GradTrainer

# 创建 trainer
trainer = GradTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# 训练
trainer.train()

# 获取记录数据
gradients = trainer.get_recorded_gradients()  # Dict[str, np.ndarray]
activations = trainer.get_recorded_activations()  # Dict[str, torch.Tensor]
results = trainer.get_intermediate_results()  # List[Dict]

# 打印摘要
trainer.log_gradients_summary()
trainer.log_activations_summary()

# 重置记录
trainer.reset_recordings()
```

## 与原始脚本的区别

| 特性 | prunepeft.py | prunepeft_gradtrainer.py |
|------|-------------|------------------------|
| Trainer | 标准 Trainer | GradTrainer |
| 梯度记录 | ❌ | ✅ |
| 激活记录 | ❌ | ✅ |
| 训练分析 | 基础 | 详细 |
| 输出格式 | 模型保存 | 模型保存 + 分析报告 |

## 应用场景

1. **GradTrainer 验证**: 测试 GradTrainer 是否正常工作
2. **梯度分析**: 研究不同参数的梯度分布和重要性
3. **训练监控**: 实时监控训练过程中的梯度和激活变化
4. **调试训练**: 分析梯度爆炸、消失等问题
5. **算法验证**: 验证 GradTrainer 的正确性和功能完整性

## 注意事项

- 这个版本**不包含剪枝策略分析**，专注于验证 GradTrainer 功能
- 如果需要剪枝功能，请使用原始的 `prunepeft.py`
- **模型保存**: PrunePEFT类型暂不支持 `save_pretrained()`，脚本会跳过保存步骤
- GradTrainer 会增加内存使用，因为需要记录梯度和激活数据
- 对于大型模型，考虑只记录关键层的梯度
- 记录的数据会保存在内存中，训练完成后可通过 API 访问

## 总结

这个脚本专注于验证 GradTrainer 的功能：
- **移除了剪枝策略分析**，简化了测试流程
- **保留了 GradTrainer** 的完整梯度记录能力
- **提供了详细的训练分析**，帮助验证功能正确性

适合用于：
- **功能测试**: 验证 GradTrainer 是否正常工作
- **梯度研究**: 深入分析训练过程中的梯度变化
- **激活监控**: 观察模型中间层的激活模式
