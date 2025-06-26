"""
This module defines the BaseCoCoOp training method, a baseline implementation for CoCoOp-based optimization.
It includes metric tracking, data loading, and training step execution using standard cross-entropy loss.
"""
import random
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader

from training_systems.training_methods.TrainingMethod import TrainingMethod

from utils.metrics import AverageMeter
from utils.datasets import ContiguousLabelDataset
from utils.kl import get_kl_loss


class BaseCoCoOp(TrainingMethod):
    """
    BaseCoCoOp implements a simple training routine based on cross-entropy loss without adversarial components.
    This class inherits from TrainingMethod and provides basic training loop functionalities.
    """

    def __init__(
            self,
            model: Any,
            optimizer: Any,
            debug: bool = False
    ) -> None:
        super().__init__(model, optimizer, "Base CoCoOp", debug)

    def get_metrics(self) -> Dict[str, AverageMeter]:
        """
        Initializes and returns the performance metrics used during training.

        Returns:
            Dict[str, AverageMeter]: A dictionary with average meters for loss and accuracy.
        """


        return {
            "loss_metric": AverageMeter(),
            "accuracy_metric": AverageMeter(),
        }

    def get_data_loader(self, dataset: ContiguousLabelDataset, batch_size: int) -> DataLoader:
        """
        Creates and returns a DataLoader for the given dataset.

        Args:
            dataset (ContiguousLabelDataset): Dataset to load samples from.
            batch_size (int): Number of samples per batch.

        Returns:
            DataLoader: PyTorch DataLoader configured with shuffle and multiple workers.
        """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
        )

    def forward_backward(
            self,
            sample,
            batch_idx,
            metrics: Dict[str, AverageMeter],
            dataset: ContiguousLabelDataset
    ) -> Dict[str, float]:
        """
        Executes the forward and backward pass for the BaseCoCoOp training method.

        Args:
            sample (Tuple[Tensor, Tensor]): Batch of input data and targets.
            batch_idx (int): Index of the current batch.
            metrics (Dict[str, AverageMeter]): Dictionary of metrics to be updated.
            dataset (ContiguousLabelDataset): Dataset object (unused in this implementation).

        Returns:
            Dict[str, float]: Dictionary with current loss and accuracy values.
        """
        # Load data into GPU
        inputs, targets = sample
        # === Pseudo-base: cross-entropy ===
        inputs_base = inputs.to(self.device)
        targets_base = targets.to(self.device)

        logits_base, loss_ce = self.model(inputs_base, targets_base)
        # === Combine losses ===
        print("SHAPES LOGITS: ",logits_base.shape, targets_base.shape)
        loss_ce.backward()

        self.optimizer_step()
        batch_size_total = inputs_base.size(0)

        metrics["loss_metric"].update(loss_ce.item(), n=batch_size_total)

        _, predicted = logits_base.max(dim=1)
        correct = (predicted == targets_base).sum().item()
        total = targets_base.size(0)
        metrics["accuracy_metric"].update(correct, n=total, raw=True)

        return {
            "loss": loss_ce.item(),
            "accuracy": correct / targets_base.size(0),
        }

    def debug_metrics_to_pbar_args(self, debug_metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Passes debug metrics directly to be displayed in the progress bar.

        Args:
            debug_metrics (Dict[str, float]): Metrics from the current step.

        Returns:
            Dict[str, float]: Unmodified metrics suitable for display.
        """
        return debug_metrics

    def training_step_return(self, metrics: Dict[str, AverageMeter]) -> [float]:
        """
        Extracts and returns the average loss and accuracy metrics after a training step.

        Args:
            metrics (Dict[str, AverageMeter]): Dictionary containing tracked metrics.

        Returns:
            List[float]: A list containing the average loss and accuracy values.
        """
        return [
            metrics["loss_metric"].avg,
            metrics["accuracy_metric"].avg,
        ]





