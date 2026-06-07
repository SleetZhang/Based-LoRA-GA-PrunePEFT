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

import bitsandbytes as bnb
import torch

from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.integrations import dequantize_bnb_weight
from peft.utils.other import transpose

from .layer import BottleneckLayer


if is_bnb_available():

    class Linear8bitLt(torch.nn.Module, BottleneckLayer):
        # Bottleneck implemented in a dense layer with 8-bit quantization
        def __init__(
            self,
            base_layer: torch.nn.Module,
            adapter_name: str,
            bottleneck_size: int = 64,
            bottleneck_dropout: float = 0.1,
            init_bottleneck_weights: bool = True,
            **kwargs,
        ) -> None:
            super().__init__()
            BottleneckLayer.__init__(self, base_layer, **kwargs)
            self.fan_in_fan_out = False

            self._active_adapter = adapter_name
            self.update_layer(
                adapter_name,
                bottleneck_size,
                bottleneck_dropout,
                init_bottleneck_weights,
            )

        def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
            """
            Merge the active adapter weights into the base weights

            Args:
                safe_merge (`bool`, *optional*):
                    If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                    before merging the weights. This is useful if you want to check if the merge operation will produce
                    NaNs. Defaults to `False`.
                adapter_names (`List[str]`, *optional*):
                    The list of adapter names that should be merged. If None, all active adapters will be merged.
                    Defaults to `None`.
            """
            adapter_names = check_adapters_to_merge(self, adapter_names)
            if not adapter_names:
                # no adapter to merge
                return

            for active_adapter in adapter_names:
                if active_adapter in self.bottleneck_A.keys():
                    base_layer = self.get_base_layer()
                    if safe_merge:
                        # Note that safe_merge will be slower than the normal merge
                        # because of the copy operation.
                        orig_weights = base_layer.weight.data.clone()
                        delta_weight = self.get_delta_weight(active_adapter)

                        if not self.fan_in_fan_out:
                            orig_weights += delta_weight
                        else:
                            orig_weights += delta_weight.T

                        if not torch.isfinite(orig_weights).all():
                            raise ValueError(
                                f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                            )

                        base_layer.weight.data = orig_weights
                    else:
                        delta_weight = self.get_delta_weight(active_adapter)
                        if not self.fan_in_fan_out:
                            base_layer.weight.data += delta_weight
                        else:
                            base_layer.weight.data += delta_weight.T

                    self.merged_adapters.append(active_adapter)

        def unmerge(self) -> None:
            """
            This method unmerges all merged adapter layers from the base weights.
            """
            if not self.merged:
                warnings.warn("Already unmerged. Nothing to do.")
                return
            while len(self.merged_adapters) > 0:
                active_adapter = self.merged_adapters.pop()
                if active_adapter in self.bottleneck_A.keys():
                    delta_weight = self.get_delta_weight(active_adapter)
                    base_layer = self.get_base_layer()
                    if not self.fan_in_fan_out:
                        base_layer.weight.data -= delta_weight
                    else:
                        base_layer.weight.data -= delta_weight.T

        def get_delta_weight(self, adapter):
            """
            Compute the delta weight for the given adapter.

            Args:
                adapter (str): The name of the adapter.

            Returns:
                torch.Tensor: The delta weight.
            """
            return (
                transpose(
                    self.bottleneck_B[adapter].weight @ self.bottleneck_A[adapter].weight,
                    self.fan_in_fan_out,
                )
            )

        def _mixed_batch_forward(
            self, x: torch.Tensor, *args: Any, adapter_names: list[str], **kwargs: Any
        ) -> torch.Tensor:
            # This is a simplified version for bottleneck
            # For mixed batch, we'll use the first adapter
            if adapter_names:
                return self.forward(x, *args, **kwargs)
            else:
                return self.base_layer(x, *args, **kwargs)

        def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            # For 8-bit quantized layers, we need to handle the forward pass carefully
            if self.disable_adapters:
                if self.merged:
                    self.unmerge()
                result = self.base_layer(x, *args, **kwargs)
            elif self.merged:
                result = self.base_layer(x, *args, **kwargs)
            else:
                result = self.base_layer(x, *args, **kwargs)
                for active_adapter in self.active_adapters:
                    if active_adapter not in self.bottleneck_A.keys():
                        continue
                    bottleneck_A = self.bottleneck_A[active_adapter]
                    bottleneck_B = self.bottleneck_B[active_adapter]
                    dropout = self.bottleneck_dropout[active_adapter]
                    
                    # Apply bottleneck transformation
                    bottleneck_result = bottleneck_A(dropout(x))
                    bottleneck_result = bottleneck_B(bottleneck_result)
                    result = result + bottleneck_result

            return result

        def __repr__(self) -> str:
            rep = super().__repr__()
            return "bottleneck." + rep


def dispatch_bnb_8bit(target: torch.nn.Module, adapter_name: str, **kwargs):
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    loaded_in_8bit = kwargs.get("loaded_in_8bit", False)
    if loaded_in_8bit and isinstance(target_base_layer, bnb.nn.Linear8bitLt):
        eightbit_kwargs = kwargs.copy()
        eightbit_kwargs.update(
            {
                "has_fp16_weights": target.state.has_fp16_weights,
                "memory_efficient_backward": target.state.memory_efficient_backward,
                "threshold": target.state.threshold,
                "index": target.index,
            }
        )
        new_module = Linear8bitLt(target, adapter_name, **eightbit_kwargs)

    return new_module


if is_bnb_4bit_available():

    class Linear4bit(torch.nn.Module, BottleneckLayer):
        # Bottleneck implemented in a dense layer with 4-bit quantization
        def __init__(
            self,
            base_layer: torch.nn.Module,
            adapter_name: str,
            bottleneck_size: int = 64,
            bottleneck_dropout: float = 0.1,
            init_bottleneck_weights: bool = True,
            **kwargs,
        ) -> None:
            super().__init__()
            BottleneckLayer.__init__(self, base_layer, **kwargs)
            self.fan_in_fan_out = False

            self._active_adapter = adapter_name
            self.update_layer(
                adapter_name,
                bottleneck_size,
                bottleneck_dropout,
                init_bottleneck_weights,
            )

        def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
            """
            Merge the active adapter weights into the base weights

            Args:
                safe_merge (`bool`, *optional*):
                    If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                    before merging the weights. This is useful if you want to check if the merge operation will produce
                    NaNs. Defaults to `False`.
                adapter_names (`List[str]`, *optional*):
                    The list of adapter names that should be merged. If None, all active adapters will be merged.
                    Defaults to `None`.
            """
            adapter_names = check_adapters_to_merge(self, adapter_names)
            if not adapter_names:
                # no adapter to merge
                return

            for active_adapter in adapter_names:
                if active_adapter in self.bottleneck_A.keys():
                    base_layer = self.get_base_layer()
                    if safe_merge:
                        # Note that safe_merge will be slower than the normal merge
                        # because of the copy operation.
                        orig_weights = base_layer.weight.data.clone()
                        delta_weight = self.get_delta_weight(active_adapter)

                        if not self.fan_in_fan_out:
                            orig_weights += delta_weight
                        else:
                            orig_weights += delta_weight.T

                        if not torch.isfinite(orig_weights).all():
                            raise ValueError(
                                f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                            )

                        base_layer.weight.data = orig_weights
                    else:
                        delta_weight = self.get_delta_weight(active_adapter)
                        if not self.fan_in_fan_out:
                            base_layer.weight.data += delta_weight
                        else:
                            base_layer.weight.data += delta_weight.T

                    self.merged_adapters.append(active_adapter)

        def unmerge(self) -> None:
            """
            This method unmerges all merged adapter layers from the base weights.
            """
            if not self.merged:
                warnings.warn("Already unmerged. Nothing to do.")
                return
            while len(self.merged_adapters) > 0:
                active_adapter = self.merged_adapters.pop()
                if active_adapter in self.bottleneck_A.keys():
                    delta_weight = self.get_delta_weight(active_adapter)
                    base_layer = self.get_base_layer()
                    if not self.fan_in_fan_out:
                        base_layer.weight.data -= delta_weight
                    else:
                        base_layer.weight.data -= delta_weight.T

        def get_delta_weight(self, adapter):
            """
            Compute the delta weight for the given adapter.

            Args:
                adapter (str): The name of the adapter.

            Returns:
                torch.Tensor: The delta weight.
            """
            return (
                transpose(
                    self.bottleneck_B[adapter].weight @ self.bottleneck_A[adapter].weight,
                    self.fan_in_fan_out,
                )
            )

        def _mixed_batch_forward(
            self, x: torch.Tensor, *args: Any, adapter_names: list[str], **kwargs: Any
        ) -> torch.Tensor:
            # This is a simplified version for bottleneck
            # For mixed batch, we'll use the first adapter
            if adapter_names:
                return self.forward(x, *args, **kwargs)
            else:
                return self.base_layer(x, *args, **kwargs)

        def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            # For 4-bit quantized layers, we need to handle the forward pass carefully
            if self.disable_adapters:
                if self.merged:
                    self.unmerge()
                result = self.base_layer(x, *args, **kwargs)
            elif self.merged:
                result = self.base_layer(x, *args, **kwargs)
            else:
                result = self.base_layer(x, *args, **kwargs)
                for active_adapter in self.active_adapters:
                    if active_adapter not in self.bottleneck_A.keys():
                        continue
                    bottleneck_A = self.bottleneck_A[active_adapter]
                    bottleneck_B = self.bottleneck_B[active_adapter]
                    dropout = self.bottleneck_dropout[active_adapter]
                    
                    # Apply bottleneck transformation
                    bottleneck_result = bottleneck_A(dropout(x))
                    bottleneck_result = bottleneck_B(bottleneck_result)
                    result = result + bottleneck_result

            return result

        def __repr__(self) -> str:
            rep = super().__repr__()
            return "bottleneck." + rep


def dispatch_bnb_4bit(target: torch.nn.Module, adapter_name: str, **kwargs):
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    loaded_in_4bit = kwargs.get("loaded_in_4bit", False)
    if loaded_in_4bit and is_bnb_4bit_available() and isinstance(target_base_layer, bnb.nn.Linear4bit):
        fourbit_kwargs = kwargs.copy()
        fourbit_kwargs.update(
            {
                "compute_dtype": target_base_layer.compute_dtype,
                "compress_statistics": target_base_layer.weight.compress_statistics,
                "quant_type": target_base_layer.weight.quant_type,
            }
        )
        new_module = Linear4bit(target, adapter_name, **fourbit_kwargs)

    return new_module