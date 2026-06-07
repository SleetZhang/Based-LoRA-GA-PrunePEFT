"""
LoRA model for efficient fine-tuning.

Author: zzh
"""

from __future__ import annotations

import re
import warnings
from contextlib import contextmanager
from dataclasses import asdict, replace
from enum import Enum
from itertools import chain
from typing import Optional

import torch
from torch import nn
from torch._subclasses.fake_tensor import log
from tqdm import tqdm
import numpy as np

from peft.config import PeftConfig
from peft.tuners.tuners_utils import (
    BaseTuner,
    BaseTunerLayer,
    check_target_module_exists,
    onload_layer,
)
from peft.utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    _get_submodules,
)

from .config import PrunePEFTConfig
from .lora_layer import Linear as LoraLinear, dispatch_default as lora_dispatch_default
from .adapter_layer import Linear as BottleneckLinear, dispatch_default as bottleneck_dispatch_default
from .block_adapter import BlockWithAdapter, BottleneckBlockAdapter

# --- Utility: collect pruning process information ---
# Global variables to store hooks and collected data
_pruning_hooks = []
_pruning_activations = {}
_pruning_gradients = {}
_pruning_accumulated_gradients = {}
_pruning_trainable_param_names = []

def hook_pruning_process_info(
    model: nn.Module,
    adapter_name: str = "default",
    opts: tuple = ("lora", "adapter"),
):
    """
    Register hooks to collect gradients and activations required by pruning methods.

    This function should be called before training to set up the hooks.
    After training, use get_pruning_process_info to retrieve the collected data.

    Args:
        model: The PEFT-wrapped model (after `get_peft_model`).
        adapter_name: Active adapter name used in parameter paths (e.g., "default").
        opts: Substrings used to identify adapter parameters, e.g. ("lora", "adapter").

    Returns:
        None
    """
    global _pruning_hooks, _pruning_activations, _pruning_gradients, _pruning_trainable_param_names

    # Clear previous data
    _pruning_activations.clear()
    _pruning_gradients.clear()
    _pruning_accumulated_gradients.clear()
    _pruning_trainable_param_names.clear()

    # Remove existing hooks if any
    for h in _pruning_hooks:
        h.remove()
    _pruning_hooks.clear()

    # Map modules to their qualified names for parameter name resolution
    module_name_map = {}
    for mod_name, mod in model.named_modules():
        module_name_map[id(mod)] = mod_name

    # Helper: register hooks on adapter modules
    def maybe_register_hook(module: nn.Module):
        mod_id = id(module)
        mod_name = module_name_map.get(mod_id, None)
        if mod_name is None:
            return
        # Collect trainable param names local to this module
        local_param_names = []
        # recurse=True to include nested adapter params (e.g., lora_A.default.weight)
        for pname, p in module.named_parameters(recurse=True):
            # Build full parameter name to match state_dict keys
            full_name = f"{mod_name}.{pname}" if mod_name else pname
            if p.requires_grad and any(opt in full_name for opt in opts):
                local_param_names.append(full_name)

        if not local_param_names:
            return

        def fwd_hook(mod, args, out):
            # Prefer module input as activation proxy for weight relevance
            act = args[0]
            if isinstance(act, (tuple, list)):
                act = act[0]
            act = act.detach()
            avg_act=act.abs().mean().cpu()

            for pname in local_param_names:
                # One activation per parameter (sufficient for current pruning logic)
                if pname not in _pruning_activations:
                    _pruning_activations[pname] = avg_act
                else:
                    _pruning_activations[pname] += avg_act

        _pruning_hooks.append(module.register_forward_hook(fwd_hook))

        # Register backward hook to accumulate gradients
        def post_backward_hook(module, grad_input, grad_output):
            # 全局变量需提前定义：_pruning_gradients（存储梯度）、local_param_names（模块内参数名列表）

            # Step1: 优先取输出梯度作为梯度代理（类似前向Hook取输入激活）
            # 注：grad_output是元组，通常第一个元素是主输出的梯度；grad_input是输入的梯度（可选）
            grad = grad_output[0] if (isinstance(grad_output, (tuple, list)) and len(grad_output) > 0) else grad_output

            # Step2: 处理梯度格式（兼容tuple/list，过滤None）
            if isinstance(grad, (tuple, list)):
                grad = grad[0] if len(grad) > 0 else None
            if grad is None:
                # 无梯度时跳过（如模块参数未参与计算）
                return

            # Step3: 剥离计算图+移到CPU（与前向Hook的act.detach()/to("cpu")对齐）
            grad = grad.detach()
            avg_grad = grad.abs().mean().cpu()

            # Step4: 为模块下每个参数绑定梯度（与前向Hook的_pruning_activations逻辑完全对齐）
            # local_param_names需在外部定义（同前向Hook，为当前模块的参数名列表）
            for pname in local_param_names:
                # One gradient per parameter (sufficient for current pruning logic)
                # print(pname)
                if pname not in _pruning_gradients:
                    _pruning_gradients[pname] = avg_grad
                else:
                    _pruning_gradients[pname] += avg_grad

        # print("Registering full backward hook")
        _pruning_hooks.append(module.register_full_backward_hook(post_backward_hook))

    # Register hooks for relevant adapter modules
    for m in model.modules():
        if isinstance(m, (LoraLinear, BottleneckLinear, BottleneckBlockAdapter)):
            maybe_register_hook(m)
        elif isinstance(m, BlockWithAdapter):
            # Also hook the inner adapter layer if present
            if m.adapter_layer is not None:
                maybe_register_hook(m.adapter_layer)

    # Store trainable parameter names
    _pruning_trainable_param_names = [
        name for name, p in model.named_parameters()
        if p.requires_grad and any(opt in name for opt in opts)
    ]


def cleanup_pruning_hooks():
    """
    Clean up all registered pruning hooks to prevent "dead module" errors.

    This function should be called after training is complete to remove all hooks
    and prevent errors when the model is deleted or modified.
    """
    global _pruning_hooks

    # Remove all registered hooks
    for h in _pruning_hooks:
        try:
            h.remove()
        except Exception:
            # Ignore errors when removing hooks from dead modules
            pass

    _pruning_hooks.clear()
    # Clear the activations dictionary
    _pruning_activations.clear()
    # Clear the gradients dictionary
    _pruning_gradients.clear()


def get_pruning_process_info(model: nn.Module):
    """
    Retrieve the gradients and activations collected by the hooks.

    This function should be called after training to get the collected data.
    Make sure to call hook_pruning_process_info before training.

    Args:
        model: The PEFT-wrapped model (after `get_peft_model`).

    Returns:
        dict with keys:
            - gradients: dict[str, np.ndarray]
            - activations: dict[str, torch.Tensor]
            - trainable_param_names: list[str]
    """

    # Return collected data
    result = {
        "gradients": _pruning_gradients.copy(),
        "activations": _pruning_activations.copy(),
        "trainable_param_names": _pruning_trainable_param_names.copy(),
    }

    return result


def collect_pruning_process_info(
    model: nn.Module,
    data_loader,
    device: torch.device,
    adapter_name: str = "default",
    opts: tuple = ("lora", "adapter"),
    num_batches: int = 8,
):
    """
    Collect gradients and activations required by pruning methods.

    - Gradients: per-parameter gradient arrays (numpy) for trainable adapter params
    - Activations: per-parameter activation tensors (torch) captured via forward hooks

    Args:
        model: The PEFT-wrapped model (after `get_peft_model`).
        data_loader: Iterable of batches dict with `input_ids`, `attention_mask`, optional `labels`.
        device: Device to run collection on.
        adapter_name: Active adapter name used in parameter paths (e.g., "default").
        opts: Substrings used to identify adapter parameters, e.g. ("lora", "adapter").
        num_batches: Number of batches to use when collecting.

    Returns:
        dict with keys:
            - gradients: dict[str, np.ndarray]
            - activations: dict[str, torch.Tensor]
            - trainable_param_names: list[str]
    """
    import numpy as np
    from collections import defaultdict

    model.train()
    model.to(device)

    # Map modules to their qualified names for parameter name resolution
    module_name_map = {}
    for mod_name, mod in model.named_modules():
        module_name_map[id(mod)] = mod_name

    # Prepare structures
    activations = {}
    gradients = {}
    hooks = []

    # Helper: register hooks on adapter modules
    def maybe_register_hook(module: nn.Module):
        mod_id = id(module)
        mod_name = module_name_map.get(mod_id, None)
        if mod_name is None:
            return
        # Collect trainable param names local to this module
        local_param_names = []
        # recurse=True to include nested adapter params (e.g., lora_A.default.weight)
        for pname, p in module.named_parameters(recurse=True):
            # Build full parameter name to match state_dict keys
            full_name = f"{mod_name}.{pname}" if mod_name else pname
            if p.requires_grad and any(opt in full_name for opt in opts):
                local_param_names.append(full_name)

        if not local_param_names:
            return

        def fwd_hook(mod, args, out):
            # Prefer module input as activation proxy for weight relevance
            act = args[0]
            if isinstance(act, (tuple, list)):
                act = act[0]
            act = act.detach()
            act_cpu = act.to("cpu")
            for pname in local_param_names:
                # One activation per parameter (sufficient for current pruning logic)
                activations[pname] = act_cpu

        hooks.append(module.register_forward_hook(fwd_hook))

    # Register hooks for relevant adapter modules
    for m in model.modules():
        if isinstance(m, (LoraLinear, BottleneckLinear, BottleneckBlockAdapter)):
            maybe_register_hook(m)
        elif isinstance(m, BlockWithAdapter):
            # Also hook the inner adapter layer if present
            if m.adapter_layer is not None:
                maybe_register_hook(m.adapter_layer)

    # Run a few batches to populate activations and gradients
    model.zero_grad(set_to_none=True)
    batches_processed = 0
    for batch in data_loader:
        if batches_processed >= num_batches:
            break
        # Move batch to device
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        # If labels not provided, use language modeling with shifted labels
        if "labels" not in batch:
            labels = batch["input_ids"].clone()
            if "attention_mask" in batch:
                labels[batch["attention_mask"] == 0] = -100
            batch["labels"] = labels

        out = model(**batch)
        loss = out["loss"] if isinstance(out, dict) else out.loss
        loss.mean().backward()
        batches_processed += 1

    # Collect gradients into numpy arrays for pruning methods
    trainable_param_names = []
    for name, p in model.named_parameters():
        if p.requires_grad and any(opt in name for opt in opts):
            trainable_param_names.append(name)
            if p.grad is not None:
                gradients[name] = p.grad.detach().cpu().numpy()

    # Cleanup hooks
    for h in hooks:
        h.remove()

    return {
        "gradients": gradients,
        "activations": activations,
        "trainable_param_names": trainable_param_names,
    }


def _get_trainable_parameter_names(model: nn.Module):
    names = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            names.append(name)
    return names


def _group_parameters_by_prefix(names, adapter_name: str, opts: tuple, model_type: str):
    # # 兼容 roberta 与其它模型的 q/v 路径差异
    # if model_type == 'roberta':
    #     v_name = 'query.'
    #     q_name = 'value.'
    # else:
    #     v_name = 'q_proj.'
    #     q_name = 'v_proj.'

    # filtered = [
    #     name for name in names
    #     if adapter_name in name and 'head' not in name and any(opt in name for opt in opts)
    # ]

    # groups = {}
    # for name in filtered:
    #     prefix = name.split(adapter_name)[0]
    #     prefix = prefix.replace(v_name, '').replace(q_name, '')
    #     groups.setdefault(prefix, []).append(name)
    # return groups

    # remake："lora1":[names, ...]

    import re

    groups = {}
    for name in names:
        layer=re.search(r'layers.(\d+)', name)
        layer_idx = int(layer.group(1))
        peft_type=""
        if 'lora' in name:
            peft_type="lora"
        elif 'adapter' in name:
            peft_type="adapter"
        assert peft_type != ""
        assert layer_idx != ""
        group_name= f"{peft_type}_{layer_idx}"
        if ('lora_B' in name or 'bottleneck_up' in name):
            continue
        groups.setdefault(group_name, []).append(name)
    return groups

def _block_parameters(groups):
    '''
    将参数名字分成8个块：
    ["LoRA 0-20%", "LoRA 20-50%", "LoRA 50-80%", "LoRA 80-100%"]
    ["Adapter 0-20%", "Adapter 20-50%", "Adapter 50-80%", "Adapter 80-100%"]
    '''
    def get_layer_group(layer_idx, total_layers):
        """判断层属于哪个比例分组 (0-3)"""
        if total_layers <= 0:
            return ""
        ratio = layer_idx / total_layers
        if 0 <= ratio < 0.2:
            return "0-20%"  # 0-20%
        elif 0.2 <= ratio < 0.5:
            return "20-50%"  # 20-50%
        elif 0.5 <= ratio < 0.8:
            return "50-80%"  # 50-80%
        else:
            return "80-100%"  # 80-100%

    block_names = ["lora 0-20%", "lora 20-50%", "lora 50-80%", "lora 80-100%"] + ["adapter 0-20%", "adapter 20-50%", "adapter 50-80%", "adapter 80-100%"]
    blocks = {k: {} for k in block_names}
    for group_name, names in groups.items():
        peft_type, layer_idx = group_name.split("_")
        layer_idx = int(layer_idx)
        block_suffix = get_layer_group(layer_idx, 32)
        blocks[f"{peft_type} {block_suffix}"][group_name] = names

    return blocks


def compute_pruning_rankings(
    model: nn.Module,
    adapter_name: str = "default",
    opts: tuple = ("lora", "adapter"),
    process_info: dict | None = None,
    top_p: int = 3,
    threshold: float = 1e-6,
    warmup_results: dict | None = None,
    final_allocation: dict | None = None
):
    """
    基于 pruning_methods_classed.py 的策略，使用过程信息（梯度/激活）计算各分组的剪枝排序。

    Args:
        model: 模型，用于获取状态字典和配置
        adapter_name: 适配器名称
        opts: 适配器类型过滤，如 ("lora", "adapter")
        process_info: 包含梯度和激活信息的字典
        top_p: 每个策略返回的前P个组
        threshold: 用于 values_below_threshold 策略的阈值
        warmup_results: 热身阶段的统计结果，用于calc_block策略

    返回：{ method: [(group_prefix, names_in_group, score), ...] }
    支持的方法：values_below_threshold, gradient, activation, snip, minimum_weight, zeros, block。
    """
    assert process_info is not None, "process_info 不能为空"
    gradients = process_info.get("gradients", {})
    activations = process_info.get("activations", {})

    names = _get_trainable_parameter_names(model)
    model_type = getattr(getattr(model, 'config', None), 'model_type', '')
    groups = _group_parameters_by_prefix(names, adapter_name=adapter_name, opts=opts, model_type=model_type)
    blocks= _block_parameters(groups)

    sd = model.state_dict()

    vals_dict = {}

    def calc_values_below_threshold():
        vals = []
        for group, gn in groups.items():
            # 只计算lora_B和bottleneck_down类型的权重
            target_names = [name for name in gn if ('lora_A' in name or 'bottleneck_down' in name)]
            if not target_names:
                # 如果没有目标权重，则跳过该组
                continue
            total = sum(sd[name].numel() for name in target_names if name in sd)
            below = sum((sd[name].abs() < threshold).sum().item() for name in target_names if name in sd)
            ratio = (below / total) if total > 0 else 0.0
            vals.append((group, target_names, ratio))
        # 与参考实现一致，对比阈值下的占比，按降序
        vals_dict["values_below_threshold"] = vals
        return sorted(vals, key=lambda x: x[2], reverse=True)[:top_p]

    def calc_zeros():
        vals = []
        for group, gn in groups.items():
            # 只计算lora_B和bottleneck_down类型的权重
            target_names = [name for name in gn if ('lora_A' in name or 'bottleneck_down' in name)]
            if not target_names:
                # 如果没有目标权重，则跳过该组
                continue
            total = sum(sd[name].numel() for name in target_names if name in sd)
            num_zeros = sum((sd[name] == 0).sum().item() for name in target_names if name in sd)
            ratio_zeros = (num_zeros / total) if total > 0 else 0.0
            vals.append((group, target_names, ratio_zeros))
        # zeros 策略按零值比例降序
        vals_dict["zeros"] = vals
        return sorted(vals, key=lambda x: x[2], reverse=True)[:top_p]

    def calc_gradient():
        vals = []
        for group, gn in groups.items():
            total_grad=0.0
            for name in gn:
                total_grad += gradients.get(name)
            avg_grad = total_grad / len(gn)
            vals.append((group, gn, avg_grad))
        # 目标是“最小梯度”，按升序
        # print(sorted(vals, key=lambda x: x[2])[:top_p])
        vals_dict["gradient"] = vals
        return sorted(vals, key=lambda x: x[2])[:top_p]

    def calc_activation():
        vals = []
        for group, gn in groups.items():
            total = sum(sd[name].numel() for name in gn if name in sd)
            # 参考实现：取该组中的一个激活作代表
            act_score = 0.0
            for name in gn:
                a = activations.get(name)
                if a is not None:
                    act_score += a.abs().sum().item()
                    break
            avg_act = (act_score / total) if total > 0 else 0.0
            vals.append((group, gn, avg_act))
        # 目标是“最小激活”，按升序
        vals_dict["activation"] = vals
        return sorted(vals, key=lambda x: x[2])[:top_p]

    def calc_snip():
        vals = []
        for group, gn in groups.items():
            total = sum(sd[name].numel() for name in gn if name in sd)
            total_snip = 0.0
            for name in gn:
                g = gradients.get(name)
                if g is None or name not in sd:
                    continue
                w = sd[name]
                if hasattr(g, 'abs'):
                    total_snip += (g.abs() * w).sum().item()
                else:
                    import numpy as np
                    gw = torch.from_numpy(np.asarray(g)).to(w.device)
                    total_snip += (gw.abs() * w).sum().item()
            avg_snip = (total_snip / total) if total > 0 else 0.0
            vals.append((group, gn, avg_snip))
        # SNIP：最小值优先，按升序
        vals_dict["snip"] = vals
        return sorted(vals, key=lambda x: x[2])[:top_p]

    def calc_minimum_weight():
        vals = []
        for group, gn in groups.items():
            # 只计算lora_A和bottleneck_down类型的权重
            target_names = [name for name in gn if ('lora_A' in name or 'bottleneck_down' in name)]
            if not target_names:
                # 如果没有目标权重，则跳过该组
                continue
            total = sum(sd[name].numel() for name in target_names if name in sd)
            if total == 0:
                vals.append((group, target_names, 0.0))
                continue
            min_weight = None
            for name in target_names:
                if name not in sd:
                    continue
                w = sd[name]
                # 参考实现：每个权重张量的均方值，再取该组的最小值
                score = (w.pow(2).sum() / w.numel()).item()
                min_weight = score if min_weight is None else min(min_weight, score)
            avg_min = (min_weight / total) if (min_weight is not None and total > 0) else 0.0
            vals.append((group, target_names, avg_min))
        # minimum_weight：最小值优先，按升序
        vals_dict["minimum_weight"] = vals
        return sorted(vals, key=lambda x: x[2])[:top_p]

    def calc_block():
        nonlocal groups
        assert final_allocation is not None and len(final_allocation) > 0, "final_allocation 不能为空"

        vals=[]

        # 解析 warmup_results 中的配置
        # 兼容两种格式：
        # 1. { "LoRA 0-20%": {"gradient": 10, "snip": 5} } (统计计数)
        # 2. { "LoRA 0-20%": ["gradient", "snip"] } (已分配好的列表)


        # 定义各策略的排序方向 (True: 值越大越容易被剪, False: 值越小越容易被剪)
        sort_reverse = {
            "values_below_threshold": True,
            "zeros": True,
            "gradient": False,
            "activation": False,
            "snip": False,
            "minimum_weight": False
        }

        # 定义需要反转分数的策略（即值越小越好，需要取负以配合Softmax）
        # 这与 sort_reverse 为 False 的策略一致
        reverse_strategies = ["gradient", "activation", "snip", "minimum_weight"]

        for block_name, gs in blocks.items():
            if block_name not in final_allocation:
                continue

            preferred_methods = final_allocation[block_name]

            # 1. 收集该 block 下所有group在各策略下的原始得分
            # param_scores: { param_name: { method: score } }
            group_scores = {name: {} for name in gs.keys()}

            # 标记该 block 是否有有效数据
            has_data = False

            for method in preferred_methods:
                if method not in vals_dict:
                    continue

                # 获取该策略下所有组的评分 (注意：vals_dict 中存储的是未排序的全量列表)
                raw_rankings = [r for r in vals_dict[method] if r[0] in gs.keys()]

                # 根据策略方向进行排序，确定 Rank
                is_reverse = sort_reverse.get(method, False)
                sorted_rankings = sorted(raw_rankings, key=lambda x: x[2], reverse=is_reverse)

                for rank, (group_name, param_names, score) in enumerate(sorted_rankings):
                    # 计算 rank-based weight
                    # rank 从 0 开始，0 表示最应该被剪枝
                    # 公式：5 / (1 + np.exp(rank - 3))
                    w_rank = 5.0 / (1.0 + np.exp(rank - 3.0))


                    group_scores[group_name][method] = {
                        "raw_score": score,
                        "rank": rank,
                        "w_rank": w_rank
                    }
                    has_data = True

            if not has_data:
                continue

            # 2. 计算 Mixed Score
            # S_mixed(L) = sum( W_rank * Softmax(V_m) )

            method_values = {m: [] for m in preferred_methods}
            group_list = [g for g in gs.keys() if group_scores[g]] # 只处理有分数的参数

            if not group_list:
                continue

            for group_name in group_list:
                for method in preferred_methods:
                    if method in group_scores[group_name].keys():
                        val = group_scores[group_name][method]["raw_score"]

                        # 确保 val 是 Python float，避免 BFloat16 Tensor 导致 numpy 转换失败
                        if hasattr(val, "item"):
                             val = val.item()
                        elif hasattr(val, "__float__"):
                             val = float(val)

                        # 如果是反向策略（越小越好），取负
                        if method in reverse_strategies:
                            val = -val
                        method_values[method].append(val)
                    else:
                        # 如果某参数在某策略下没分，填 -inf
                        method_values[method].append(-float('inf'))

            # 计算 Softmax
            method_softmax = {}
            for method in preferred_methods:
                vals_arr = np.array(method_values[method])
                # 数值稳定性处理
                vals_arr = vals_arr - np.max(vals_arr)
                exp_vals = np.exp(vals_arr)
                softmax_vals = exp_vals / np.sum(exp_vals)
                method_softmax[method] = softmax_vals

            # 聚合得分
            for i, g in enumerate(group_list):
                mixed_score = 0.0
                for method in preferred_methods:
                    if method in group_scores[g]:
                        w_rank = group_scores[g][method]["w_rank"]
                        norm_score = method_softmax[method][i]
                        mixed_score += w_rank * norm_score

                # 构造返回项：(prefix, [name], mixed_score)
                # 为了保持接口一致性，每个参数作为一个独立的 group
                #
                vals.append((g, groups[g], mixed_score))


        # 3. 对所有 Block 的所有候选参数进行全局排序，并选出 top_p
        # Mixed Score 越高越应该被剪枝
        return sorted(vals, key=lambda x: x[2], reverse=True)[:top_p]




    rankings = {
        'zeros': calc_zeros(),
        'values_below_threshold': calc_values_below_threshold(),
        'gradient': calc_gradient(),
        'activation': calc_activation(),
        'snip': calc_snip(),
        'minimum_weight': calc_minimum_weight(),
    }

    # 如果提供了warmup_results，添加基于最佳策略的block计算
    if warmup_results is not None:
        rankings['block'] = calc_block()

    return rankings


class PrunePEFTModel(BaseTuner):
    """
    Creates Low Rank Adapter (LoRA) model from a pretrained transformers model.

    Args:
        model: The model to be adapted
        config: The configuration of the LoRA model
        adapter_name: The name of the adapter, defaults to "default"
    """

    prefix: str = "lora_"
    layers_mapping = {
        LoraLinear,
        BottleneckLinear,
        BlockWithAdapter,
        BottleneckBlockAdapter,
    }

    def __init__(self, model, config, adapter_name) -> None:
        peft_config = config[adapter_name]
        self.prunepeft_config = peft_config
        super().__init__(model, config, adapter_name)

        # For combined mode, inject block wrappers after Linear adapters
        # config is a dict {adapter_name: peft_config}

        # Debug: print layer configuration
        if peft_config.lora_layers is not None:
            print(f"DEBUG: LoRA layers configured: {peft_config.lora_layers}")
        if peft_config.adapter_layers is not None:
            print(f"DEBUG: Bottleneck adapter layers configured: {peft_config.adapter_layers}")

        if len(peft_config.adapter_types) > 1:
            self._inject_blocks(peft_config, adapter_name)

        # Explicitly set all adapter and LoRA module parameters to trainable
        self._mark_adapters_as_trainable()

    def _check_new_adapter_config(self, config: PrunePEFTConfig) -> None:
        """
        A helper method to check the config when a new adapter is being added.

        Raise a ValueError if there is something wrong with the config or if it conflicts with existing adapters.
        """
        if (len(self.peft_config) > 1) and (config.bias != "none"):
            raise ValueError(
                f"{self.__class__.__name__} supports only 1 adapter with bias. When using multiple adapters, "
                "set bias to 'none' for all adapters."
            )

    @staticmethod
    def _check_target_module_exists(peft_config: PeftConfig, key: str) -> bool:
        return check_target_module_exists(peft_config, key)

    @staticmethod
    def _extract_layer_index(module_name: str) -> Optional[int]:
        """
        Extract layer index from module name.
        For example, 'model.layers.0.self_attn.q_proj' -> 0
        Also handles 'base_model.model.layers.0.self_attn.q_proj' -> 0
        Returns None if layer index cannot be extracted.
        """
        import re
        # Pattern to match layer indices in common model architectures
        # Supports: layers.0, decoder.layers.0, transformer.h.0, etc.
        # Note: The pattern matches both with and without leading dot, and handles nested paths
        patterns = [
            r'\.layers\.(\d+)\.',  # Standard pattern: .layers.0 (most common)
            r'(?:^|\.)layers\.(\d+)\.',  # Fallback: layers.0 or .layers.0
            r'\.decoder\.layers\.(\d+)\.',  # BART/T5: .decoder.layers.0
            r'\.transformer\.h\.(\d+)\.',  # GPT-2: .transformer.h.0
            r'\.encoder\.layers\.(\d+)\.',  # Encoder: .encoder.layers.0
        ]

        for pattern in patterns:
            match = re.search(pattern, module_name)
            if match:
                return int(match.group(1))

        return None

    def _should_apply_lora(self, prunepeft_config, module_name: str) -> bool:
        """Check if LoRA should be applied to this module based on layer index."""
        if prunepeft_config.lora_layers is None:
            return True  # Apply to all layers if not specified

        layer_index = self._extract_layer_index(module_name)
        if layer_index is None:
            # If layer index cannot be determined, don't apply (to be safe)
            # This prevents applying to modules outside the main transformer layers
            return False

        should_apply = layer_index in prunepeft_config.lora_layers
        return should_apply

    def _should_apply_bottleneck(self, prunepeft_config, module_name: str) -> bool:
        """Check if Bottleneck adapter should be applied to this module based on layer index."""
        if prunepeft_config.adapter_layers is None:
            return False  # Apply to all layers if not specified

        layer_index = self._extract_layer_index(module_name)
        if layer_index is None:
            # If layer index cannot be determined, don't apply (to be safe)
            # This prevents applying to modules outside the main transformer layers
            return False

        should_apply = layer_index in prunepeft_config.adapter_layers
        return should_apply

    def _create_and_replace(
        self,
        prunepeft_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        # Check if this is a block-level module (attention or MLP block)
        module_basename = current_key.split('.')[-1]
        is_block_module = module_basename in ["self_attn", "mlp"]

        prunepeft_config = self.prunepeft_config

        if is_block_module:
            # For combined mode, blocks are handled separately after Linear adapters
            # For single adapter types, process immediately
            if len(prunepeft_config.adapter_types) == 1:
                # Check if bottleneck should be applied to this layer
                if "bottleneck" in prunepeft_config.adapter_types:
                    if self._should_apply_bottleneck(prunepeft_config, current_key):
                        self._create_and_replace_block(prunepeft_config, adapter_name, target, target_name, parent, current_key)
        else:
            # Handle Linear-level modules - use traditional adapter approach
            # Check if LoRA should be applied to this layer
            if "lora" in prunepeft_config.adapter_types:
                if self._should_apply_lora(prunepeft_config, current_key):
                    self._create_and_replace_linear(prunepeft_config, adapter_name, target, target_name, parent, current_key)
                # Debug: log when LoRA is skipped
                elif prunepeft_config.lora_layers is not None:
                    layer_index = self._extract_layer_index(current_key)
                    if layer_index is not None and layer_index not in prunepeft_config.lora_layers:
                        pass  # LoRA skipped for this layer (expected)

    def _inject_blocks(self, config, adapter_name):
        """Inject block wrappers after Linear adapters are processed."""
        # Only wrap blocks if bottleneck is in adapter_types
        if "bottleneck" not in config.adapter_types:
            return

        block_modules = ["self_attn", "mlp"]

        for module_name, module in self.model.named_modules():
            module_basename = module_name.split('.')[-1]
            if module_basename in block_modules:
                # Skip if already wrapped (shouldn't happen in combined mode, but be safe)
                if isinstance(module, BlockWithAdapter):
                    continue

                # Check if bottleneck should be applied to this layer
                layer_index = self._extract_layer_index(module_name)
                should_apply = self._should_apply_bottleneck(config, module_name)

                if not should_apply:
                    continue

                # Find parent and target_name
                parts = module_name.split('.')
                target_name = parts[-1]
                parent_name = '.'.join(parts[:-1])
                parent = self.model.get_submodule(parent_name) if parent_name else self.model

                # Create block wrapper
                self._create_and_replace_block(config, adapter_name, module, target_name, parent, module_name)

    def _create_and_replace_block(
        self,
        prunepeft_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
    ):
        """Handle block-level modules (attention and MLP blocks)"""
        # Get hidden size from model config
        model_config = getattr(self.model, 'config', {})
        hidden_size = getattr(model_config, 'hidden_size', 4096)  # Default to 4096 for LLaMA

        if isinstance(target, BlockWithAdapter):
            # Update existing BlockWithAdapter
            if "bottleneck" in prunepeft_config.adapter_types:
                # Create or update bottleneck adapter for this block
                bottleneck_adapter = BottleneckBlockAdapter(
                    base_layer=target.base_block,  # Use the original block as base
                    adapter_name=adapter_name,
                    bottleneck_size=prunepeft_config.bottleneck_size,
                    bottleneck_dropout=prunepeft_config.bottleneck_dropout,
                    init_bottleneck_weights=prunepeft_config.init_bottleneck_weights,
                    hidden_size=hidden_size,
                )
                target.update_adapter(bottleneck_adapter)
        else:
            # Create new BlockWithAdapter wrapper
            block_with_adapter = BlockWithAdapter(target, adapter_name)

            if "bottleneck" in prunepeft_config.adapter_types:
                # Add bottleneck adapter to the block
                bottleneck_adapter = BottleneckBlockAdapter(
                    base_layer=target,  # The original block
                    adapter_name=adapter_name,
                    bottleneck_size=prunepeft_config.bottleneck_size,
                    bottleneck_dropout=prunepeft_config.bottleneck_dropout,
                    init_bottleneck_weights=prunepeft_config.init_bottleneck_weights,
                    hidden_size=hidden_size,
                )
                block_with_adapter.update_adapter(bottleneck_adapter)

            if adapter_name != self.active_adapter:
                # adding an additional adapter: it is not automatically trainable
                block_with_adapter.requires_grad_(False)
            self._replace_module(parent, target_name, block_with_adapter, target)

    def _create_and_replace_linear(
        self,
        prunepeft_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
    ):
        """Handle Linear-level modules (traditional adapter approach)"""

        # Regexp matching - Find key which matches current target_name in patterns provided
        pattern_keys = list(chain(prunepeft_config.rank_pattern.keys(), prunepeft_config.alpha_pattern.keys()))
        target_name_key = next(filter(lambda key: re.match(rf".*\.{key}$", current_key), pattern_keys), current_key)
        r = prunepeft_config.rank_pattern.get(target_name_key, prunepeft_config.r)
        alpha = prunepeft_config.alpha_pattern.get(target_name_key, prunepeft_config.lora_alpha)

        bias = hasattr(target, "bias") and target.bias is not None

        # Determine which adapter types to apply based on the module name
        effective_adapter_types = prunepeft_config.adapter_types[:]

        if len(prunepeft_config.adapter_types) > 1:
            # For combined mode, only LoRA is applied to Linear layers
            # Bottleneck is handled at block level
            effective_adapter_types = ["lora"]

        # Handle Linear layer adapters
        if isinstance(target, (LoraLinear, BottleneckLinear)):
            # Update existing adapter layer
            if isinstance(target, LoraLinear) and "lora" in effective_adapter_types:
                target.update_layer(
                    adapter_name,
                    r=r,
                    lora_alpha=alpha,
                    lora_dropout=prunepeft_config.lora_dropout,
                    init_lora_weights=prunepeft_config.init_lora_weights,
                    use_rslora=prunepeft_config.use_rslora,
                    use_dora=prunepeft_config.use_dora,
                )
            elif isinstance(target, BottleneckLinear) and "bottleneck" in effective_adapter_types:
                target.update_layer(
                    adapter_name,
                    bottleneck_size=prunepeft_config.bottleneck_size,
                    bottleneck_dropout=prunepeft_config.bottleneck_dropout,
                    init_bottleneck_weights=prunepeft_config.init_bottleneck_weights,
                )
        else:
            # Create new adapter layer based on effective_adapter_types
            new_module = None

            if len(effective_adapter_types) == 1:
                adapter_type = effective_adapter_types[0]
                if adapter_type == "lora":
                    lora_kwargs = {
                        "r": r,
                        "lora_alpha": alpha,
                        "lora_dropout": prunepeft_config.lora_dropout,
                        "fan_in_fan_out": prunepeft_config.fan_in_fan_out,
                        "init_lora_weights": prunepeft_config.init_lora_weights,
                        "use_rslora": prunepeft_config.use_rslora,
                        "use_dora": prunepeft_config.use_dora,
                        "ephemeral_gpu_offload": prunepeft_config.runtime_config.ephemeral_gpu_offload,
                        "bias": bias,
                    }
                    new_module = self._create_new_module_for_type(prunepeft_config, adapter_name, target, "lora", **lora_kwargs)
                elif adapter_type == "bottleneck":
                    bottleneck_kwargs = {
                        "bottleneck_size": prunepeft_config.bottleneck_size,
                        "bottleneck_dropout": prunepeft_config.bottleneck_dropout,
                        "init_bottleneck_weights": prunepeft_config.init_bottleneck_weights,
                        "bias": bias,
                    }
                    new_module = self._create_new_module_for_type(prunepeft_config, adapter_name, target, "bottleneck", **bottleneck_kwargs)

            if new_module is not None:
                if adapter_name != self.active_adapter:
                    # adding an additional adapter: it is not automatically trainable
                    new_module.requires_grad_(False)
                self._replace_module(parent, target_name, new_module, target)


    def _create_new_module_for_type(self, prunepeft_config, adapter_name, target, adapter_type, **kwargs):
        """Create a new module for a specific adapter type."""
        dispatchers = []

        if adapter_type == "lora":
            dispatchers.append(lora_dispatch_default)
        elif adapter_type == "bottleneck":
            dispatchers.append(bottleneck_dispatch_default)

        new_module = None
        for dispatcher in dispatchers:
            # Use different parameter names for different adapter types
            if adapter_type == "lora":
                new_module = dispatcher(target, adapter_name, lora_config=prunepeft_config, **kwargs)
            elif adapter_type == "bottleneck":
                new_module = dispatcher(target, adapter_name, bottleneck_config=prunepeft_config, **kwargs)
            if new_module is not None:  # first match wins
                break

        if new_module is None:
            # no module could be matched
            raise ValueError(
                f"Target module {target} is not supported for {adapter_type}. Currently, only the following modules are supported: "
                "`torch.nn.Linear`, `torch.nn.Embedding`, `torch.nn.Conv2d`, `transformers.pytorch_utils.Conv1D`."
            )

        return new_module

    def _create_new_module(self, prunepeft_config, adapter_name, target, **kwargs):
        # This method is kept for backward compatibility but now uses _create_new_module_for_type
        # Default to lora if adapter_types contains lora, otherwise use the first adapter type
        adapter_type = prunepeft_config.adapter_types[0] if prunepeft_config.adapter_types else "lora"
        return self._create_new_module_for_type(prunepeft_config, adapter_name, target, adapter_type, **kwargs)

    def _replace_module(self, parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)
        # It's not necessary to set requires_grad here, as that is handled by
        # _mark_only_adapters_as_trainable

        # Check if this is a block-level replacement (BlockWithAdapter)
        # For block-level replacements, the original block is already stored in base_block
        # No need to copy any weights or attributes
        if isinstance(new_module, BlockWithAdapter):
            return

        # child layer wraps the original module, unpack it
        if hasattr(child, "base_layer"):
            child = child.base_layer

        if not hasattr(new_module, "base_layer"):
            if hasattr(new_module, "W_q"):  # HQQ
                new_module.W_q = child.W_q
            else:
                new_module.weight = child.weight
            if hasattr(child, "bias"):
                new_module.bias = child.bias

        if getattr(child, "state", None) is not None:
            if hasattr(new_module, "base_layer"):
                new_module.base_layer.state = child.state
            else:
                new_module.state = child.state
            new_module.to(child.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if (self.prefix in name) or ("ranknum" in name):
                weight = (
                    child.qweight
                    if hasattr(child, "qweight")
                    else child.W_q
                    if hasattr(child, "W_q")
                    else child.weight
                    if hasattr(child, "weight")
                    else next(child.parameters())
                )
                module.to(weight.device)

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        """Mark only the adapter layers as trainable."""
        for n, p in model.named_parameters():
            if self.prefix not in n and "bottleneck_down" not in n and "bottleneck_up" not in n:
                p.requires_grad = False

        # No need to handle CombinedLinear layers anymore since we use separate layers

        for active_adapter in self.active_adapters:
            bias = self.peft_config[active_adapter].bias
            if bias == "none":
                continue

            if bias == "all":
                for n, p in model.named_parameters():
                    if "bias" in n:
                        p.requires_grad = True
            elif bias == "lora_only":
                for m in model.modules():
                    if isinstance(m, LoraLinear) and hasattr(m, "bias") and m.bias is not None:
                        m.bias.requires_grad = True
            else:
                raise NotImplementedError(f"Requested bias: {bias}, is not implemented.")

    def _mark_adapters_as_trainable(self):
        """
        Explicitly mark all adapter and LoRA module parameters as trainable.
        This ensures that all adapter parameters have requires_grad=True after initialization.
        """
        for module in self.model.modules():
            # Mark LoRA linear layer parameters as trainable
            # Only mark parameters that are part of the adapter (not base_layer)
            # if isinstance(module, LoraLinear):
            #     for param_name, param in module.named_parameters(recurse=False):
            #         # Only mark adapter parameters (lora_A, lora_B, etc.), not base_layer
            #         if self.prefix in param_name or "lora" in param_name.lower():
            #             param.requires_grad = True
            for param_name, param in module.named_parameters():
                if "lora_A" in param_name or "lora_B" in param_name:
                    param.requires_grad= True

            # Mark Bottleneck linear layer parameters as trainable
            if isinstance(module, BottleneckLinear):
                for param_name, param in module.named_parameters(recurse=False):
                    # Only mark bottleneck adapter parameters
                    if "bottleneck_down" in param_name or "bottleneck_up" in param_name:
                        param.requires_grad = True

            # Mark BlockWithAdapter's BottleneckBlockAdapter parameters as trainable
            if isinstance(module, BlockWithAdapter):
                if module.adapter_layer is not None:
                    for param_name, param in module.adapter_layer.named_parameters():
                        # Mark all parameters in BottleneckBlockAdapter (they are all adapter params)
                        if "bottleneck_down" in param_name or "bottleneck_up" in param_name:
                            param.requires_grad = True

            # Mark BottleneckBlockAdapter parameters directly as trainable
            if isinstance(module, BottleneckBlockAdapter):
                for param_name, param in module.named_parameters():
                    # Mark all bottleneck adapter parameters
                    if "bottleneck_down" in param_name or "bottleneck_up" in param_name:
                        param.requires_grad = True

    def set_adapter(self, adapter_name):
        for module in self.model.modules():
            if isinstance(module, (LoraLinear, BottleneckLinear)):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.set_adapter(adapter_name)
            elif isinstance(module, BlockWithAdapter):
                module.set_adapter(adapter_name)
        self.active_adapter = adapter_name

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                module.enable_adapters(enabled)

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)

    def _prepare_adapter_config(self, peft_config: PrunePEFTConfig, model_config: dict) -> PrunePEFTConfig:
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")

            # For different adapter types, use different target modules
            base_target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_config["model_type"]]

            if peft_config.adapter_types == ["lora"]:
                # LoRA: use standard attention and MLP layers (Linear layers)
                peft_config.target_modules = set(base_target_modules)
            elif peft_config.adapter_types == ["bottleneck"]:
                # Bottleneck: place after self_attn and mlp blocks (block-level modules)
                # Target the attention and MLP blocks directly
                peft_config.target_modules = ["self_attn", "mlp"]
            elif len(peft_config.adapter_types) > 1:
                # Combined LoRA + Bottleneck: LoRA on Linear layers first
                # Blocks will be handled separately after Linear adapters
                peft_config.target_modules = set(base_target_modules)
            else:
                # Default fallback
                peft_config.target_modules = set(base_target_modules)

        return peft_config

    def _unload_and_optionally_merge(
        self,
        merge=True,
        progressbar: bool = False,
        safe_merge: bool = False,
        adapter_names: Optional[list[str]] = None,
    ):
        if merge:
            raise ValueError("PrunePEFT does not support merging. Use unload() instead.")

        key_list = [key for key, _ in self.model.named_modules() if self.prefix not in key]
        desc = "Unloading model"
        for key in tqdm(key_list, disable=not progressbar, desc=desc):
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            with onload_layer(target):
                if hasattr(target, "base_layer"):
                    self._replace_module(parent, target_name, target.get_base_layer(), target)
                elif isinstance(target, ModulesToSaveWrapper):
                    # save any additional trainable modules part of `modules_to_save`
                    new_module = target.modules_to_save[target.active_adapter]
                    if hasattr(new_module, "base_layer"):
                        new_module = new_module.get_base_layer()
                    setattr(parent, target_name, new_module)

        return self.model

    def merge_and_unload(
        self, progressbar: bool = False, safe_merge: bool = False, adapter_names: Optional[list[str]] = None
    ) -> torch.nn.Module:
        """
        PrunePEFT does not support merging. This method is disabled.
        """
        raise ValueError("PrunePEFT does not support merging. Use unload() method instead.")

    def unload(self) -> torch.nn.Module:
        """
        Gets back the base model by removing all the lora modules without merging. This gives back the original base
        model.
        """
        return self._unload_and_optionally_merge(merge=False)

    def get_peft_config_as_dict(self, inference: bool = False):
        config_dict = {}
        for adapter_name, adapter_config in self.peft_config.items():
            config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(adapter_config).items()}
            if inference:
                config["inference_mode"] = True
        config_dict[adapter_name] = config
        return config

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)
