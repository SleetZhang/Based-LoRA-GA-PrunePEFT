# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import warnings
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.other import transpose

from .config import PrunePEFTConfig


class BottleneckLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("bottleneck_down", "bottleneck_up")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("bottleneck_size", "bottleneck_dropout")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.bottleneck_size = {}
        self.bottleneck_dropout = nn.ModuleDict({})
        self.bottleneck_down = nn.ModuleDict({})
        self.bottleneck_up = nn.ModuleDict({})
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, nn.Conv2d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Embedding):
            in_features, out_features = base_layer.num_embeddings, base_layer.embedding_dim
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )
        elif hasattr(base_layer, "infeatures") and hasattr(base_layer, "outfeatures"):
            # QuantLinear
            in_features, out_features = base_layer.infeatures, base_layer.outfeatures
        elif hasattr(base_layer, "input_size") and hasattr(base_layer, "output_size"):
            # Megatron ColumnParallelLinear,RowParallelLinear
            in_features, out_features = base_layer.input_size, base_layer.output_size
        else:
            # possibly support user provided custom layer types using dynamic dispatch
            if hasattr(base_layer, "in_features") and hasattr(base_layer, "out_features"):
                in_features, out_features = base_layer.in_features, base_layer.out_features
            else:
                in_features, out_features = None, None
            warnings.warn(
                f"Unsupported layer type '{type(base_layer)}' encountered, proceed at your own risk.", UserWarning
            )

        self.in_features = in_features
        self.out_features = out_features

    def update_layer(
        self, adapter_name, bottleneck_size, bottleneck_dropout, init_bottleneck_weights
    ):
        if bottleneck_size <= 0:
            raise ValueError(f"`bottleneck_size` should be a positive integer value but the value passed is {bottleneck_size}")

        self.bottleneck_size[adapter_name] = bottleneck_size
        self.bottleneck_dropout[adapter_name] = nn.Dropout(p=bottleneck_dropout)
        
        # Create bottleneck layers: down projection (input -> bottleneck) and up projection (bottleneck -> output)
        self.bottleneck_down[adapter_name] = nn.Linear(self.in_features, bottleneck_size, bias=False)
        self.bottleneck_up[adapter_name] = nn.Linear(bottleneck_size, self.out_features, bias=False)

        if init_bottleneck_weights:
            self.reset_bottleneck_parameters(adapter_name)

        # move to device
        weight = getattr(self.get_base_layer(), "weight", None)
        if weight is not None:
            # the layer is already completely initialized, this is an update
            if weight.dtype.is_floating_point or weight.dtype.is_complex:
                self.bottleneck_down[adapter_name].to(weight.device, dtype=weight.dtype)
                self.bottleneck_up[adapter_name].to(weight.device, dtype=weight.dtype)
            else:
                self.bottleneck_down[adapter_name].to(weight.device)
                self.bottleneck_up[adapter_name].to(weight.device)
        self.set_adapter(self.active_adapters)

    def reset_bottleneck_parameters(self, adapter_name):
        if adapter_name in self.bottleneck_down.keys():
            # Initialize down projection with Xavier uniform
            nn.init.xavier_uniform_(self.bottleneck_down[adapter_name].weight)
            # Initialize up projection with zeros (similar to LoRA initialization)
            nn.init.zeros_(self.bottleneck_up[adapter_name].weight)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.bottleneck_down.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    delta_weight = self.get_delta_weight(active_adapter)
                    new_weight = orig_weights + delta_weight
                    if not torch.isfinite(new_weight).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )
                    base_layer.weight.data = new_weight
                else:
                    delta_weight = self.get_delta_weight(active_adapter)
                    base_layer.weight.data += delta_weight
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged_adapters:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.bottleneck_down.keys():
                delta_weight = self.get_delta_weight(active_adapter)
                self.get_base_layer().weight.data -= delta_weight

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str): The name of the adapter.

        Returns:
            `torch.Tensor`: The delta weight.
        """
        device = self.bottleneck_down[adapter].weight.device
        dtype = self.bottleneck_down[adapter].weight.dtype

        # Create identity input to compute the transformation
        # For bottleneck: output = up(down(input))
        down_weight = self.bottleneck_down[adapter].weight
        up_weight = self.bottleneck_up[adapter].weight
        
        # Compute the equivalent transformation matrix
        delta_weight = torch.mm(up_weight, down_weight)
        
        return delta_weight.to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged_adapters and self.training:
                raise ValueError("Cannot disable adapters when merged adapters are enabled and training is enabled.")
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged_adapters:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.bottleneck_down.keys():
                    continue
                bottleneck_down = self.bottleneck_down[active_adapter]
                bottleneck_up = self.bottleneck_up[active_adapter]
                dropout = self.bottleneck_dropout[active_adapter]
                
                # Apply bottleneck transformation: down -> dropout -> up
                x = x.to(bottleneck_down.weight.dtype)
                bottleneck_result = bottleneck_down(x)
                bottleneck_result = dropout(bottleneck_result)
                bottleneck_result = bottleneck_up(bottleneck_result)
                
                result = result + bottleneck_result.to(torch_result_dtype)

        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "bottleneck." + rep


class Linear(nn.Module, BottleneckLayer):
    # Bottleneck implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        bottleneck_size: int = 64,
        bottleneck_dropout: float = 0.1,
        init_bottleneck_weights: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        BottleneckLayer.__init__(self, base_layer, **kwargs)

        # Freezing the pre-trained weight matrix
        self.get_base_layer().weight.requires_grad = False
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, bottleneck_size, bottleneck_dropout, init_bottleneck_weights)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        return BottleneckLayer.merge(self, safe_merge=safe_merge, adapter_names=adapter_names)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        return BottleneckLayer.unmerge(self)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str): The name of the adapter.

        Returns:
            `torch.Tensor`: The delta weight.
        """
        return BottleneckLayer.get_delta_weight(self, adapter)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        return BottleneckLayer.forward(self, x, *args, **kwargs)

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "bottleneck." + rep


def dispatch_default(
    target: torch.nn.Module,
    adapter_name: str,
    bottleneck_config: PrunePEFTConfig,
    **kwargs,
) -> Optional[torch.nn.Module]:
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if isinstance(target_base_layer, torch.nn.Linear):
        # For bottleneck adapter, only pass relevant parameters
        bottleneck_kwargs = {
            "bottleneck_size": getattr(bottleneck_config, "bottleneck_size", 64),
            "bottleneck_dropout": getattr(bottleneck_config, "bottleneck_dropout", 0.1),
            "init_bottleneck_weights": getattr(bottleneck_config, "init_bottleneck_weights", True),
        }
        kwargs.update(bottleneck_kwargs)
        new_module = Linear(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, Conv1D):
        # For bottleneck adapter, only pass relevant parameters
        bottleneck_kwargs = {
            "bottleneck_size": getattr(bottleneck_config, "bottleneck_size", 64),
            "bottleneck_dropout": getattr(bottleneck_config, "bottleneck_dropout", 0.1),
            "init_bottleneck_weights": getattr(bottleneck_config, "init_bottleneck_weights", True),
        }
        kwargs.update(bottleneck_kwargs)
        new_module = Linear(target, adapter_name, **kwargs)

    return new_module