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

import re
import warnings
from dataclasses import asdict, replace
from enum import Enum
from itertools import chain
from typing import Optional

import torch
from tqdm import tqdm

from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer, check_target_module_exists, onload_layer
from peft.config import PeftConfig
from peft.utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    TRANSFORMERS_MODELS_TO_BOTTLENECK_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    _get_submodules,
)

from .config import BottleneckConfig
from .layer import BottleneckLayer, dispatch_default


class BottleneckModel(BaseTuner):
    """
    Creates Bottleneck (Bottleneck Adapter) model from a pretrained transformers model.

    The method is described in detail in https://arxiv.org/abs/1902.00751

    Args:
        model ([`~transformers.PreTrainedModel`]): The model to be adapted.
        config ([`BottleneckConfig`]): The configuration of the Bottleneck model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.

    Returns:
        `torch.nn.Module`: The Bottleneck model.

    Example:

        ```py
        >>> from transformers import AutoModelForSeq2SeqLM
        >>> from peft import BottleneckModel, BottleneckConfig

        >>> config = BottleneckConfig(
        ...     peft_type="BOTTLENECK",
        ...     task_type="SEQ_2_SEQ_LM",
        ...     bottleneck_size=64,
        ...     target_modules=["q", "v"],
        ...     bottleneck_dropout=0.1,
        ... )

        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
        >>> bottleneck_model = BottleneckModel(model, config, "default")
        ```

    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`BottleneckConfig`]): The configuration of the Bottleneck model.
    """

    prefix: str = "bottleneck_"
    layers_mapping = {
        BottleneckLayer,
    }

    def __init__(self, model, config, adapter_name) -> None:
        super().__init__(model, config, adapter_name)

    def _check_new_adapter_config(self, config: BottleneckConfig) -> None:
        """
        A helper method to check the config when a new adapter is being added.

        Raise a ValueError if there is something wrong with the config or if it conflicts with existing adapters.

        """
        # TODO: there should be a check if any of the existing adapters actually has bias != "none", or else the check
        # does not correspond to the error message.
        if (len(self.peft_config) > 1) and (config.bias != "none"):
            raise ValueError(
                f"{self.__class__.__name__} supports only 1 adapter with bias. When using multiple adapters, "
                "set bias to 'none' for all adapters."
            )

    @staticmethod
    def _check_target_module_exists(peft_config: PeftConfig, key: str) -> bool:
        return check_target_module_exists(peft_config, key)

    def _create_and_replace(
        self,
        bottleneck_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        # Regexp matching - Find key which matches current target_name in patterns provided
        pattern_keys = list(chain(bottleneck_config.rank_pattern.keys(), bottleneck_config.alpha_pattern.keys()))
        target_name_key = next(filter(lambda key: re.match(rf".*\.{key}$", current_key), pattern_keys), current_key)
        r = bottleneck_config.rank_pattern.get(target_name_key, bottleneck_config.r)
        alpha = bottleneck_config.alpha_pattern.get(target_name_key, bottleneck_config.lora_alpha)

        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "bottleneck_size": bottleneck_config.bottleneck_size,
            "bottleneck_dropout": bottleneck_config.bottleneck_dropout,
            "bias": bias,
            "init_bottleneck_weights": bottleneck_config.init_bottleneck_weights,
        }
        kwargs["loaded_in_8bit"] = getattr(self.model, "is_loaded_in_8bit", False)
        kwargs["loaded_in_4bit"] = getattr(self.model, "is_loaded_in_4bit", False)

        if isinstance(target, BottleneckLayer):
            target.update_layer(
                adapter_name,
                bottleneck_config.bottleneck_size,
                bottleneck_config.bottleneck_dropout,
                bottleneck_config.init_bottleneck_weights,
            )
        else:
            new_module = self._create_new_module(bottleneck_config, adapter_name, target, **kwargs)
            if adapter_name != self.active_adapter:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    def _create_new_module(self, bottleneck_config, adapter_name, target, **kwargs):
        # Collect dispatcher functions to decide what backend to use for the replaced LoRA layer. The order matters,
        # because the first match is always used. Therefore, the default layers should be checked last.
        dispatchers = []

        # avoid eager bnb import
        if is_bnb_available():
            from .bnb import dispatch_bnb_8bit

            dispatchers.append(dispatch_bnb_8bit)

        if is_bnb_4bit_available():
            from .bnb import dispatch_bnb_4bit

            dispatchers.append(dispatch_bnb_4bit)

        dispatchers.extend([dispatch_default])

        new_module = None
        for dispatcher in dispatchers:
            new_module = dispatcher(target, adapter_name, bottleneck_config=bottleneck_config, **kwargs)
            if new_module is not None:  # first match wins
                break

        if new_module is None:
            # no module could be matched
            raise ValueError(
                f"Target module {target} is not supported. Currently, only the following modules are supported: "
                "`torch.nn.Linear`, `torch.nn.Embedding`, `torch.nn.Conv2d`, `transformers.pytorch_utils.Conv1D`."
            )

        return new_module

    def _replace_module(self, parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)
        # It's not necessary to set requires_grad here, as that is handled by
        # _mark_only_adapters_as_trainable

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

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    def get_peft_config_as_dict(self, inference: bool = False):
        config_dict = {}
        for adapter_name, adapter_config in self.peft_config.items():
            config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(adapter_config).items()}
            if inference:
                config["inference_mode"] = True
        config_dict[adapter_name] = config
        return config

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                module.enable_adapters(enabled)

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        """Mark only the adapter layers as trainable."""
        for n, p in model.named_parameters():
            if self.prefix not in n:
                p.requires_grad = False

        for active_adapter in self.active_adapters:
            bias = self.peft_config[active_adapter].bias
            if bias == "none":
                continue

            if bias == "all":
                for n, p in model.named_parameters():
                    if "bias" in n:
                        p.requires_grad = True
            elif bias == "bottleneck_only":
                for m in model.modules():
                    if isinstance(m, BottleneckLayer) and hasattr(m, "bias") and m.bias is not None:
                        m.bias.requires_grad = True
            else:
                raise NotImplementedError(f"Requested bias: {bias}, is not implemented.")

    def set_adapter(self, adapter_name):
        for module in self.model.modules():
            if isinstance(module, BottleneckLayer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.set_adapter(adapter_name)
        self.active_adapter = adapter_name

    def _check_merge_allowed(self):
        """Verify that the configuration supports merging.

        Currently gptq quantization and replicated layers do not support merging.
        """
        if getattr(self.model, "quantization_method", None) == "gptq":
            raise ValueError("Cannot merge BOTTLENECK layers when the model is gptq quantized")
        if self.peft_config.get("layer_replication"):
            raise ValueError("Cannot merge BOTTLENECK layers when base model layers are replicated")

    def _prepare_adapter_config(self, peft_config: PeftConfig, model_config: dict) -> PeftConfig:
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_BOTTLENECK_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = set(
                TRANSFORMERS_MODELS_TO_BOTTLENECK_TARGET_MODULES_MAPPING[model_config["model_type"]]
            )
        return peft_config

    def _unload_and_optionally_merge(
        self,
        merge=True,
        progressbar: bool = False,
        safe_merge: bool = False,
        adapter_names: Optional[list[str]] = None,
    ):
        if merge:
            self._check_merge_allowed()

        key_list = [key for key, _ in self.model.named_modules() if self.prefix not in key]
        desc = "Unloading " + ("and merging " if merge else "") + "model"
        for key in tqdm(key_list, disable=not progressbar, desc=desc):
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            with onload_layer(target):
                if hasattr(target, "base_layer"):
                    if merge:
                        target.merge(safe_merge=safe_merge, adapter_names=adapter_names)
                    self._replace_module(parent, target_name, target.get_base_layer(), target)
                elif isinstance(target, ModulesToSaveWrapper):
                    # save any additional trainable modules part of `modules_to_save`
                    new_module = target.modules_to_save[target.active_adapter]
                    if hasattr(new_module, "base_layer"):
                        # check if the module is itself a tuner layer
                        if merge:
                            new_module.merge(safe_merge=safe_merge, adapter_names=adapter_names)
                        new_module = new_module.get_base_layer()
                    setattr(parent, target_name, new_module)

        return self.model

    def add_weighted_adapter(self, adapters, weights, adapter_name, combination_type="svd", svd_rank=None, svd_clamp=None, svd_full_matrices=True, svd_driver=None):
        """
        This method adds a new adapter by merging the given adapters with the given weights.

        When using the `cat` combination_type you should be aware that rank of the resulting adapter will be equal to
        the sum of all adapters ranks. So it's possible that the mixed adapter may become too big and result in OOM
        errors.

        Args:
            adapters (`list`):
                List of adapter names to be merged.
            weights (`list`):
                List of weights for each adapter.
            adapter_name (`str`):
                Name of the new adapter.
            combination_type (`str`):
                Type of merging. Can be one of [`svd`, `linear`, `cat`]. When using the `cat` combination_type, the rank
                of the resulting adapter is equal to the sum of all adapters ranks (the mixed adapter may be too big and
                result in OOM errors).
            svd_rank (`int`, *optional*):
                Rank of output adapter for svd. If None provided, will use max rank of merging adapters.
            svd_clamp (`float`, *optional*):
                A dynamic clamping factor used to control the range of singular values in SVD, potentially improving the
                stability and performance of the adapter. If None, no clamping is applied. Defaults to None.
            svd_full_matrices (`bool`, *optional*):
                Controls whether to compute the full or reduced SVD, and consequently, the shape of the returned
                tensors U and Vh. Defaults to True.
            svd_driver (`str`, *optional*):
                Name of the cuSOLVER method to be used. This keyword argument only works when merging on CUDA. Can be
                one of [None, `gesvd`, `gesvdj`, `gesvda`]. For more info please refer to `torch.linalg.svd`
                documentation. Defaults to None.
        """

        if adapter_name in list(self.peft_config.keys()):
            return
        for adapter in adapters:
            if adapter not in list(self.peft_config.keys()):
                raise ValueError(f"Adapter {adapter} does not exist")

        # if there is only one adapter, we can only use linear merging
        combination_type = "linear" if len(adapters) == 1 else combination_type

        adapters_ranks = [self.peft_config[adapter].bottleneck_size for adapter in adapters]
        if combination_type == "linear":
            # all adapters need to have the same rank
            if len(set(adapters_ranks)) != 1:
                raise ValueError(
                    "All adapters must have the same bottleneck_size when using `linear` combination_type"
                )

        elif combination_type == "cat":
            # adapters ranks are summed
            bottleneck_size = sum(adapters_ranks)
        elif combination_type == "svd":
            bottleneck_size = svd_rank or max(adapters_ranks)
        else:
            raise ValueError(f"Invalid combination_type: {combination_type}")

        target_module_types = [type(self.peft_config[adapter].target_modules) for adapter in adapters]
        if not target_module_types:
            raise ValueError(f"Found no adapter matching the names in {adapters}")
        if len(set(target_module_types)) > 1:
            raise ValueError(
                "All adapters must target the same module types, "
                f"found {target_module_types[0]} and {target_module_types[1]}"
            )

        if combination_type == "linear":
            bottleneck_size = adapters_ranks[0]
        elif combination_type == "cat":
            bottleneck_size = sum(adapters_ranks)
        elif combination_type == "svd":
            bottleneck_size = svd_rank or max(adapters_ranks)

        # new rank is the max of all ranks
        self.peft_config[adapter_name] = replace(
            self.peft_config[adapters[0]], bottleneck_size=bottleneck_size, lora_alpha=bottleneck_size
        )
        self.inject_adapter(self.model, adapter_name)

        # Do we really need that?
        _freeze_adapter(self.model, adapter_name)

        key_list = [key for key, _ in self.model.named_modules() if self.prefix not in key]
        for key in key_list:
            _, target, _ = _get_submodules(self.model, key)
            if isinstance(target, BottleneckLayer):
                if adapter_name in target.bottleneck_down:
                    target_bottleneck_down = target.bottleneck_down[adapter_name].weight
                    target_bottleneck_up = target.bottleneck_up[adapter_name].weight
                elif adapter_name in target.bottleneck_embedding_A:
                    target_bottleneck_down = target.bottleneck_embedding_A[adapter_name]
                    target_bottleneck_up = target.bottleneck_embedding_B[adapter_name]
                else:
                    continue

                target_bottleneck_down.data = target_bottleneck_down.data * 0
                target_bottleneck_up.data = target_bottleneck_up.data * 0
                if combination_type == "cat":
                    bottlenecks_down, bottlenecks_up = [], []
                    for adapter, weight in zip(adapters, weights):
                        if adapter in target.bottleneck_down:
                            current_adapter_down = target.bottleneck_down[adapter].weight
                            current_adapter_up = target.bottleneck_up[adapter].weight
                        else:
                            current_adapter_down = target.bottleneck_embedding_A[adapter]
                            current_adapter_up = target.bottleneck_embedding_B[adapter]

                        bottlenecks_down.append(current_adapter_down.data * weight * target.scaling[adapter])
                        bottlenecks_up.append(current_adapter_up.data)

                    target_bottleneck_down.data = torch.cat(bottlenecks_down, dim=0)
                    target_bottleneck_up.data = torch.cat(bottlenecks_up, dim=1)
                elif combination_type == "linear":
                    for adapter, weight in zip(adapters, weights):
                        if adapter in target.bottleneck_down:
                            current_adapter_down = target.bottleneck_down[adapter].weight
                            current_adapter_up = target.bottleneck_up[adapter].weight
                        else:
                            current_adapter_down = target.bottleneck_embedding_A[adapter]
                            current_adapter_up = target.bottleneck_embedding_B[adapter]

                        target_bottleneck_down.data += (
                            current_adapter_down.data * weight * target.scaling[adapter]
                        )
                        target_bottleneck_up.data += current_adapter_up.data * weight
                elif combination_type == "svd":
                    bottlenecks_delta_weight = []
                    for adapter, weight in zip(adapters, weights):
                        if adapter in target.bottleneck_down:
                            current_adapter_down = target.bottleneck_down[adapter].weight
                            current_adapter_up = target.bottleneck_up[adapter].weight
                        else:
                            current_adapter_down = target.bottleneck_embedding_A[adapter]
                            current_adapter_up = target.bottleneck_embedding_B[adapter]

                        bottlenecks_delta_weight.append(
                            (current_adapter_up.data @ current_adapter_down.data) * weight * target.scaling[adapter]
                        )

                    conv_filter = isinstance(target_bottleneck_up, torch.nn.Parameter) and len(target_bottleneck_up.shape) == 4
                    if conv_filter:
                        delta_weight = sum(bottlenecks_delta_weight)
                        conv_filter_shape = delta_weight.shape
                        delta_weight = delta_weight.flatten(start_dim=1)
                    else:
                        delta_weight = sum(bottlenecks_delta_weight)

                    if svd_clamp is not None:
                        delta_weight = torch.clamp(delta_weight, -svd_clamp, +svd_clamp)

                    U, S, Vh = torch.linalg.svd(delta_weight, full_matrices=svd_full_matrices, driver=svd_driver)

                    if conv_filter:
                        U = U.reshape(conv_filter_shape[0], -1)
                        U = U[:, : bottleneck_size].reshape(conv_filter_shape[0], bottleneck_size, 1, 1).contiguous()
                        Vh = Vh[: bottleneck_size, :].reshape(bottleneck_size, *conv_filter_shape[1:]).contiguous()

                    target_bottleneck_down.data = Vh[: bottleneck_size, :] * torch.sqrt(S[: bottleneck_size]).unsqueeze(-1)
                    target_bottleneck_up.data = U[:, : bottleneck_size] * torch.sqrt(S[: bottleneck_size]).unsqueeze(0)

    def delete_adapter(self, adapter_name: str) -> None:
        """
        Deletes an existing adapter.

        Args:
            adapter_name (str): Name of the adapter to be deleted.
        """
        if adapter_name not in list(self.peft_config.keys()):
            raise ValueError(f"Adapter {adapter_name} does not exist")
        del self.peft_config[adapter_name]

        key_list = [key for key, _ in self.model.named_modules() if self.prefix not in key]
        new_adapter = None
        for key in key_list:
            _, target, _ = _get_submodules(self.model, key)
            if isinstance(target, BottleneckLayer):
                target.delete_adapter(adapter_name)
                if new_adapter is None:
                    new_adapter = target.active_adapters[:]

        self.active_adapter = new_adapter or []

    def merge_and_unload(
        self, progressbar: bool = False, safe_merge: bool = False, adapter_names: Optional[list[str]] = None
    ) -> torch.nn.Module:
        r"""
        This method merges the Bottleneck layers into the base model. This is needed if someone wants to use the base
        model as a standalone model.

        Args:
            progressbar (`bool`):
                whether to show a progressbar indicating the unload and merge process
            safe_merge (`bool`):
                whether to activate the safe merging check to check if there is any potential Nan in the adapter
                weights
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        Example:

        ```py
        >>> from transformers import AutoModelForCausalLM
        >>> from peft import PeftModel

        >>> base_model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-40b", torch_dtype=torch.bfloat16)
        >>> peft_model_id = "smangrul/falcon-40B-int4-peft-lora-sfttrainer-sample"
        >>> model = PeftModel.from_pretrained(base_model, peft_model_id)
        >>> merged_model = model.merge_and_unload()
        ```
        """
        return self._unload_and_optionally_merge(
            merge=True, progressbar=progressbar, safe_merge=safe_merge, adapter_names=adapter_names
        )

    def unload(self) -> torch.nn.Module:
        """
        Gets back the base model by removing all the bottleneck modules without merging. This gives back the original base
        model.
        """
        return self._unload_and_optionally_merge(merge=False)


def _freeze_adapter(model, adapter_name):
    for n, p in model.named_parameters():
        if adapter_name in n:
            p.requires_grad = False


def _get_submodules(model, key):
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name