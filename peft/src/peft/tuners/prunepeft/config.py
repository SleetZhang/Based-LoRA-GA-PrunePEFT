"""
LoRA configuration module.

Author: zzh
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal, Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType
from peft.tuners.lora.config import LoraRuntimeConfig, LoftQConfig


@dataclass
class PrunePEFTConfig(PeftConfig):
    """
    Configuration for PrunePEFT (supports both LoRA and Bottleneck adapters simultaneously).

    Args:
        adapter_types: List of adapter types to use (e.g., ["lora"], ["bottleneck"], ["lora", "bottleneck"])
        r: LoRA attention dimension (the "rank") - used when "lora" in adapter_types
        target_modules: The names of the modules to apply the adapter to
        lora_alpha: The alpha parameter for LoRA scaling - used when "lora" in adapter_types
        lora_dropout: The dropout probability for LoRA layers - used when "lora" in adapter_types
        bias: Bias type for LoRA - used when "lora" in adapter_types
        init_lora_weights: How to initialize the weights of the LoRA layers - used when "lora" in adapter_types
        bottleneck_size: Size of bottleneck for bottleneck adapter - used when "bottleneck" in adapter_types
        bottleneck_dropout: Dropout for bottleneck adapter - used when "bottleneck" in adapter_types
        init_bottleneck_weights: Whether to initialize bottleneck weights - used when "bottleneck" in adapter_types
        layers_to_transform: The layer indices to transform
        layers_pattern: The layer pattern name
    """

    adapter_types: list[str] = field(default_factory=lambda: ["lora"], metadata={"help": "List of adapter types to use (e.g., ['lora'], ['bottleneck'], ['lora', 'bottleneck'])"})
    r: int = field(default=8, metadata={"help": "LoRA attention dimension"})
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with LoRA."
        },
    )
    lora_alpha: int = field(default=8, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.0, metadata={"help": "LoRA dropout"})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    bias: Literal["none", "all", "lora_only"] = field(
        default="none", metadata={"help": "Bias type for LoRA. Can be 'none', 'all' or 'lora_only'"}
    )
    init_lora_weights: Union[bool, str] = field(
        default=True,
        metadata={
            "help": "How to initialize the weights of the LoRA layers."
        },
    )
    bottleneck_size: int = field(default=64, metadata={"help": "Size of bottleneck for bottleneck adapter"})
    bottleneck_dropout: float = field(default=0.1, metadata={"help": "Dropout for bottleneck adapter"})
    init_bottleneck_weights: bool = field(default=True, metadata={"help": "Whether to initialize bottleneck weights"})
    adapter_layers: Optional[list[int]] = field(
        default=None,
        metadata={
            "help": "List of layer indices to apply Bottleneck adapter (block-level). If None, applies to all layers."
        },
    )
    lora_layers: Optional[list[int]] = field(
        default=None,
        metadata={
            "help": "List of layer indices to apply LoRA adapter (linear-level). If None, applies to all layers."
        },
    )
    loftq_config: Union[LoftQConfig, dict] = field(
        default_factory=dict,
        metadata={
            "help": (
                "The configuration of LoftQ. If this is passed, then LoftQ will be used to quantize the backbone "
                "weights and initialize Lora layers. Also set `init_lora_weights='loftq'` in this case."
            )
        },
    )
    use_rslora: bool = field(
        default=False,
        metadata={
            "help": (
                "When set to True, uses Rank-Stabilized LoRA"
                " which sets the adapter scaling factor to `lora_alpha/math.sqrt(r)`, since it"
                " was proven to work better. Otherwise, it will use the original default"
                " value of `lora_alpha/r`."
            )
        },
    )
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
        },
    )
    use_dora: bool = field(
        default=False,
        metadata={
            "help": (
                "Enable 'Weight-Decomposed Low-Rank Adaptation' (DoRA). This technique decomposes the updates of the "
                "weights into two parts, magnitude and direction. Direction is handled by normal LoRA, whereas the "
                "magnitude is handled by a separate learnable parameter. This can improve the performance of LoRA, "
                "especially at low ranks. Right now, DoRA only supports linear and Conv2D layers. DoRA introduces a bigger"
                "overhead than pure LoRA, so it is recommended to merge weights for inference."
            )
        },
    )
    layers_to_transform: Optional[Union[list[int], int]] = field(
        default=None,
        metadata={
            "help": "The layer indexes to transform."
        },
    )
    layers_pattern: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": "The layer pattern name, used only if `layers_to_transform` is different to None."
        },
    )
    rank_pattern: Optional[dict] = field(
        default_factory=dict,
        metadata={
            "help": (
                "The mapping from layer names or regexp expression to ranks which are different from the default rank specified by `r`. "
                "For example, `{model.decoder.layers.0.encoder_attn.k_proj: 8`}"
            )
        },
    )
    alpha_pattern: Optional[dict] = field(
        default_factory=dict,
        metadata={
            "help": (
                "The mapping from layer names or regexp expression to alphas which are different from the default alpha specified by `lora_alpha`. "
                "For example, `{model.decoder.layers.0.encoder_attn.k_proj: 32`}"
            )
        },
    )
    layer_replication: Optional[list[tuple[int, int]]] = field(
        default=None,
        metadata={
            "help": (
                "This enables using LoRA to effectively expand a transformer model to a larger size by repeating some layers. "
                "The transformation handles models (currently Llama, Bert or Falcon compatible architectures) with "
                "a module list in the model which it modifies to expand the number of modules. "
                "Base weights are shared so the memory usage is close to the original model. The intended use is these base weights "
                "remain fixed during finetuning but each layer has a separate LoRA adapter so the layers can be specialed via "
                "the adapter layers fit during fine tuning."
            ),
        },
    )
    runtime_config: LoraRuntimeConfig = field(
        default_factory=LoraRuntimeConfig, metadata={"help": "Runtime configurations"}
    )

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values and `runtime_config` by its values
        (which are not saved or restored).
        """
        rv = asdict(self)
        rv.pop("runtime_config")
        return rv

    def __post_init__(self):
        self.peft_type = PeftType.PRUNEPEFT

        # Validate adapter_types
        valid_adapter_types = ["lora", "bottleneck", "dora"]
        for adapter_type in self.adapter_types:
            if adapter_type not in valid_adapter_types:
                raise ValueError(f"adapter_types must contain only 'lora' or 'bottleneck', got {adapter_type} in {self.adapter_types}")
        if not self.adapter_types:
            raise ValueError("adapter_types cannot be empty")

        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        # if target_modules is a regex expression, then layers_to_transform should be None
        if isinstance(self.target_modules, str) and self.layers_to_transform is not None:
            raise ValueError("`layers_to_transform` cannot be used when `target_modules` is a str.")

        # if target_modules is a regex expression, then layers_pattern should be None
        if isinstance(self.target_modules, str) and self.layers_pattern is not None:
            raise ValueError("`layers_pattern` cannot be used when `target_modules` is a str.")
