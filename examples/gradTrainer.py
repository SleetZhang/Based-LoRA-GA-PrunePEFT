"""
GradTrainer: A trainer that records gradients and activations during training.
Architecture based on LogTrainer but with gradient recording functionality similar to TrainerWithGrad.
"""

from typing import Callable, Dict, List, Optional, Tuple, Union, Any
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import Trainer, Seq2SeqTrainingArguments
from transformers.data.data_collator import DataCollator
from transformers.trainer import (
    EvalPrediction,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
)
from peft.tuners.lora.layer import Linear as LoraLinear


class GradTrainer(Trainer):
    """
    A trainer that records gradients and activations during training.

    This trainer combines the architecture of LogTrainer with the gradient
    recording functionality of TrainerWithGrad, providing detailed monitoring
    of model parameters during training.
    """

    def __init__(
        self,
        model: Union[PreTrainedModel, torch.nn.Module] = None,
        args: Seq2SeqTrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
    ):
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            preprocess_logits_for_metrics,
            optimizers,
        )

        # Initialize gradient and activation recording
        self.recorded_gradients = {}
        self.recorded_activations = {}
        self.intermediate_results = []
        self.forward_hooks = []

        # Check if it's a PEFT model
        self.is_peft = "PeftModel" in type(model).__name__

        # Setup forward hooks for activation recording
        self._setup_activation_hooks()

    def _setup_activation_hooks(self):
        """Setup forward hooks to record activations from trainable layers."""

        def save_activation(name):
            def hook(module, input, output):
                self.recorded_activations[name] = output[0].detach().cpu()
            return hook

        # Register hooks for trainable parameters
        for name, module in self.model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                if param.requires_grad:
                    full_name = f"{name}.{param_name}"
                    hook = module.register_forward_hook(save_activation(full_name))
                    self.forward_hooks.append(hook)
                    break  # Only register one hook per module

    def training_step(
        self, model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], hiddens=None
    ) -> torch.Tensor:
        """
        Perform a training step with gradient recording.

        Args:
            model: The model to train
            inputs: The inputs to the model
            hiddens: Hidden states (unused, kept for compatibility)

        Returns:
            The loss tensor
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        self.accelerator.backward(loss)

        # Record gradients for all parameters
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_cpu = param.grad.detach().cpu().numpy()
                    if name not in self.recorded_gradients:
                        self.recorded_gradients[name] = grad_cpu.copy()
                    else:
                        self.recorded_gradients[name] += grad_cpu

        return loss.detach() / self.args.gradient_accumulation_steps

    def get_recorded_gradients(self) -> Dict[str, np.ndarray]:
        """
        Get the recorded gradients.

        Returns:
            Dictionary mapping parameter names to accumulated gradients
        """
        return self.recorded_gradients.copy()

    def get_recorded_activations(self) -> Dict[str, torch.Tensor]:
        """
        Get the recorded activations.

        Returns:
            Dictionary mapping layer names to activation tensors
        """
        return self.recorded_activations.copy()

    def get_intermediate_results(self) -> List[Dict]:
        """
        Get the intermediate evaluation results.

        Returns:
            List of evaluation results from each epoch
        """
        return self.intermediate_results.copy()

    def reset_recordings(self):
        """Reset all recordings."""
        self.recorded_gradients.clear()
        self.recorded_activations.clear()
        self.intermediate_results.clear()

    def log_gradients_summary(self):
        """Log a summary of recorded gradients."""
        if not self.recorded_gradients:
            print("No gradients recorded yet.")
            return

        print(f"Recorded gradients for {len(self.recorded_gradients)} parameters:")
        for name, grad in self.recorded_gradients.items():
            grad_mean = np.mean(np.abs(grad))
            grad_std = np.std(grad)
            grad_max = np.max(np.abs(grad))
            print("30")

    def log_activations_summary(self):
        """Log a summary of recorded activations."""
        if not self.recorded_activations:
            print("No activations recorded yet.")
            return

        print(f"Recorded activations for {len(self.recorded_activations)} layers:")
        for name, activation in self.recorded_activations.items():
            act_mean = activation.mean().item()
            act_std = activation.std().item()
            act_shape = tuple(activation.shape)
            print("30")

    def __del__(self):
        """Clean up forward hooks when the trainer is deleted."""
        for hook in self.forward_hooks:
            hook.remove()

