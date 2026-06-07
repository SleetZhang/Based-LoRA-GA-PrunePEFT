"""
Block-level adapter for serial adapter application.

This module provides block-level adapter implementations that can wrap
attention and MLP blocks for serial adapter application.

Author: zzh
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Optional, Any

from peft.tuners.tuners_utils import BaseTunerLayer


class BlockAdapterLayer(nn.Module, BaseTunerLayer):
    """
    Base class for block-level adapters that work serially after blocks.
    """

    def __init__(self, base_layer, **kwargs):
        nn.Module.__init__(self)
        BaseTunerLayer.__init__(self)
        self.base_layer = base_layer
        self._active_adapter = kwargs.get("adapter_name", "default")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the adapter transformation serially.
        This should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement forward method")

    def set_adapter(self, adapter_name: str):
        """Set the active adapter."""
        # Use try-except to handle potential nn.Module restrictions
        try:
            self.active_adapter = adapter_name
        except AttributeError:
            pass  # nn.Module may restrict setting this attribute
        self._active_adapter = adapter_name  # Keep for backward compatibility


class BlockWithAdapter(nn.Module):
    """
    A wrapper class that applies adapters serially after a block (attention or MLP).

    This enables true serial adapter application where the adapter processes
    the output of the original block.
    """

    def __init__(self, base_block: nn.Module, adapter_name: str = "default"):
        super().__init__()
        self.base_block = base_block
        self.adapter_name = adapter_name

        # Adapter components will be set by the tuner
        self.adapter_layer = None
        self._active_adapter = adapter_name

    def set_adapter(self, adapter_name: str):
        """Set the active adapter."""
        self._active_adapter = adapter_name
        if self.adapter_layer and hasattr(self.adapter_layer, 'set_adapter'):
            self.adapter_layer.set_adapter(adapter_name)

    def forward(self, *args, **kwargs):
        # First apply the original block
        block_output = self.base_block(*args, **kwargs)

        # Handle attention block output - it may return a tuple (attn_output, attn_weights)
        if isinstance(block_output, tuple):
            # For attention blocks, only apply adapter to the first element (attn_output)
            attn_output, attn_weights = block_output
            if self.adapter_layer is not None:
                attn_output = self.adapter_layer(attn_output)
            return (attn_output, attn_weights)
        else:
            # For MLP blocks, apply adapter directly to the output
            if self.adapter_layer is not None:
                block_output = self.adapter_layer(block_output)
            return block_output

    def update_adapter(self, adapter_layer):
        """Update the adapter layer."""
        self.adapter_layer = adapter_layer

    def __getattr__(self, name):
        """Forward attribute access to the base_block for submodule access."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            # If the attribute doesn't exist on this wrapper, try to get it from base_block
            if hasattr(self, 'base_block'):
                return getattr(self.base_block, name)
            raise


class BottleneckBlockAdapter(BlockAdapterLayer):
    """
    Bottleneck adapter applied at block level (serial after attention/MLP blocks).
    """

    def __init__(
        self,
        base_layer,
        adapter_name: str = "default",
        bottleneck_size: int = 64,
        bottleneck_dropout: float = 0.1,
        init_bottleneck_weights: bool = True,
        hidden_size: int = None,  # Hidden size for block-level adapters
        **kwargs
    ):
        super().__init__(base_layer, adapter_name=adapter_name, **kwargs)

        # Store configuration for later use
        self.bottleneck_size = bottleneck_size
        self.bottleneck_dropout = bottleneck_dropout
        self.init_bottleneck_weights = init_bottleneck_weights

        # For block-level adapters, we need to know the hidden size
        # Try to infer from base_layer, or use provided hidden_size
        if hidden_size is not None:
            self.hidden_size = hidden_size
        elif hasattr(base_layer, 'out_features'):
            self.hidden_size = base_layer.out_features
        else:
            # For attention/MLP blocks, assume standard hidden size
            # This should be overridden when creating the adapter
            self.hidden_size = 4096  # Default for LLaMA models

        # Initialize adapter components
        self.bottleneck_down = nn.ModuleDict()
        self.bottleneck_up = nn.ModuleDict()
        self.bottleneck_dropout_layers = nn.ModuleDict()

        self.update_layer(adapter_name, bottleneck_size, bottleneck_dropout, init_bottleneck_weights)

        # Ensure active_adapter is set (use _active_adapter to avoid nn.Module restrictions)
        self._active_adapter = adapter_name

    def update_layer(self, adapter_name: str, bottleneck_size: int, bottleneck_dropout: float, init_bottleneck_weights: bool):
        """Update or create bottleneck adapter for the given adapter name."""
        if adapter_name not in self.bottleneck_down:
            # Create bottleneck adapter components
            down_proj = nn.Linear(self.hidden_size, bottleneck_size, bias=False)
            up_proj = nn.Linear(bottleneck_size, self.hidden_size, bias=False)
            dropout_layer = nn.Dropout(bottleneck_dropout)

            if init_bottleneck_weights:
                # Initialize weights (similar to standard bottleneck initialization)
                nn.init.normal_(down_proj.weight, mean=0.0, std=0.02)
                nn.init.normal_(up_proj.weight, mean=0.0, std=0.02)

            self.bottleneck_down[adapter_name] = down_proj
            self.bottleneck_up[adapter_name] = up_proj
            self.bottleneck_dropout_layers[adapter_name] = dropout_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bottleneck transformation serially."""
        if self.disable_adapters:
            return x

        result = x
        previous_dtype = x.dtype

        # Use the active adapter (should be set by set_adapter method)
        # active_adapter should be a string, but active_adapters returns a list
        active_adapters = self.active_adapters
        if active_adapters:
            active_adapter = active_adapters[0]  # Take the first active adapter
        else:
            active_adapter = 'default'

        if active_adapter in self.bottleneck_down:
            bottleneck_down = self.bottleneck_down[active_adapter]
            bottleneck_up = self.bottleneck_up[active_adapter]
            dropout = self.bottleneck_dropout_layers[active_adapter]

            # Apply bottleneck transformation: down -> dropout -> up -> add to input
            x = x.to(bottleneck_down.weight.dtype)
            bottleneck_result = bottleneck_down(x)
            bottleneck_result = dropout(bottleneck_result)
            bottleneck_result = bottleneck_up(bottleneck_result)

            result = result + bottleneck_result.to(previous_dtype)

        return result.to(previous_dtype)

    def __repr__(self) -> str:
        return f"BottleneckBlockAdapter(bottleneck_size={self.bottleneck_size})"
