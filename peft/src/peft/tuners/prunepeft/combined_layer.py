"""
Combined layer that supports both LoRA and Bottleneck adapters simultaneously.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional

import torch
import torch.nn as nn

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge


class CombinedLinear(nn.Module, BaseTunerLayer):
    """
    A combined layer that can contain both LoRA and Bottleneck adapters.
    The adapters are applied sequentially: base_layer -> bottleneck -> lora -> output
    """

    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("lora_A", "lora_B", "bottleneck_down", "bottleneck_up")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "lora_alpha", "scaling", "lora_dropout", "bottleneck_size", "bottleneck_dropout")

    def __init__(self, base_layer, lora_layer=None, bottleneck_layer=None, **kwargs):
        super().__init__()
        BaseTunerLayer.__init__(self)

        self.base_layer = base_layer
        self.lora_layer = lora_layer
        self.bottleneck_layer = bottleneck_layer

        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self.kwargs = kwargs

        # Get base layer info
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        else:
            # Handle other layer types if needed
            in_features, out_features = getattr(base_layer, 'in_features', None), getattr(base_layer, 'out_features', None)

        self.in_features = in_features
        self.out_features = out_features

    def update_layer(self, adapter_name, **kwargs):
        """Update layer parameters for the given adapter."""
        # Update LoRA layer if present
        if self.lora_layer and hasattr(self.lora_layer, 'update_layer'):
            lora_kwargs = {k: v for k, v in kwargs.items() if k in ['r', 'lora_alpha', 'lora_dropout', 'init_lora_weights', 'use_rslora', 'use_dora']}
            if lora_kwargs:
                self.lora_layer.update_layer(adapter_name, **lora_kwargs)

        # Update Bottleneck layer if present
        if self.bottleneck_layer and hasattr(self.bottleneck_layer, 'update_layer'):
            bottleneck_kwargs = {k: v for k, v in kwargs.items() if k in ['bottleneck_size', 'bottleneck_dropout', 'init_bottleneck_weights']}
            if bottleneck_kwargs:
                self.bottleneck_layer.update_layer(adapter_name, **bottleneck_kwargs)

        self.set_adapter(self.active_adapters)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights.
        This operation is not supported for combined layers as it would be ambiguous
        which adapter's weights to merge.
        """
        raise ValueError("Merging is not supported for combined layers with multiple adapter types. Use unload() instead.")

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        This operation is not supported for combined layers.
        """
        raise ValueError("Unmerging is not supported for combined layers with multiple adapter types.")

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Get delta weight for the given adapter.
        This operation is not supported for combined layers as it would be ambiguous.
        """
        raise ValueError("get_delta_weight is not supported for combined layers with multiple adapter types.")

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Forward pass through the combined layer.
        The order is: base_layer -> bottleneck (if present) -> lora (if present)
        """
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged_adapters and self.training:
                raise ValueError("Cannot disable adapters when merged adapters are enabled and training is enabled.")
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged_adapters:
            result = self.base_layer(x, *args, **kwargs)
        else:
            # Start with base layer
            result = self.base_layer(x, *args, **kwargs)

            # Apply bottleneck adapter if present
            if self.bottleneck_layer and not self.bottleneck_layer.disable_adapters:
                result = self.bottleneck_layer(result, *args, **kwargs)

            # Apply LoRA adapter if present
            if self.lora_layer and not self.lora_layer.disable_adapters:
                result = self.lora_layer(result, *args, **kwargs)

        result = result.to(previous_dtype)
        return result

    def set_adapter(self, adapter_names):
        """Set active adapters for both sub-layers."""
        if self.lora_layer:
            self.lora_layer.set_adapter(adapter_names)
        if self.bottleneck_layer:
            self.bottleneck_layer.set_adapter(adapter_names)

    @property
    def merged(self):
        """Check if any of the sub-layers are merged."""
        lora_merged = self.lora_layer.merged if self.lora_layer else False
        bottleneck_merged = self.bottleneck_layer.merged if self.bottleneck_layer else False
        return lora_merged or bottleneck_merged

    @property
    def disable_adapters(self):
        """Check if adapters are disabled."""
        return self._disable_adapters

    @disable_adapters.setter
    def disable_adapters(self, value):
        """Set disable_adapters for both sub-layers."""
        self._disable_adapters = value
        if self.lora_layer:
            self.lora_layer.disable_adapters = value
        if self.bottleneck_layer:
            self.bottleneck_layer.disable_adapters = value

    def __repr__(self) -> str:
        components = []
        if self.lora_layer:
            components.append("lora")
        if self.bottleneck_layer:
            components.append("bottleneck")
        adapter_str = "+".join(components) if components else "none"
        return f"combined({adapter_str})." + super().__repr__()
