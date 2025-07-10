"""
This module defines the KLCoCoOp training method, which combines standard cross-entropy loss with KL divergence
between a model's predictions and a frozen teacher (e.g., CLIP) to enhance generalization to novel classes.
"""
import random
from typing import Dict, Any, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from training_systems.training_methods.MultipleDatasetsTrainingMethod import DoubleDatasetTrainingMethod
from training_systems.training_methods.TrainingMethod import TrainingMethod

from utils.metrics import AverageMeter
from utils.datasets import ContiguousLabelDataset
from utils.kl import get_kl_loss


class KLCoCoOpV2(DoubleDatasetTrainingMethod):
    """
    KLCoCoOp training method combining cross-entropy classification on pseudo-base samples and KL divergence
    loss on pseudo-novel samples. It encourages transferability and generalization by mixing base and novel categories.

    Attributes:
        lambda_kl (float): Weight for the KL divergence loss component.
    """

    def __init__(
            self,
            model: Any,
            optimizer: Any,
            lambda_kl,
            debug: bool = False,
    ) -> None:
        super().__init__(model, optimizer, "Base CoCoOp + KL", debug)
        self.lambda_kl = lambda_kl


    def get_metrics(self) -> Dict[str, AverageMeter]:
        """
        Initializes training metrics including total, cross-entropy, KL divergence losses, and classification accuracy.

        Returns:
            Dict[str, AverageMeter]: Dictionary of metric names mapped to their respective AverageMeter instances.
        """
        return {
            "ce_loss_metric": AverageMeter(),   
            "kl_loss_metric": AverageMeter(),
            "ce_accuracy_metric": AverageMeter(),
        }

    def get_data_loader1(self, pseudo_base: ContiguousLabelDataset, batch_size: int) -> DataLoader:
        """
        Returns a DataLoader that splits each batch into pseudo-base and pseudo-novel subsets.

        Args:
            dataset (ContiguousLabelDataset): The dataset used for training.
            batch_size (int): Number of samples per batch.

        Returns:
            DataLoader: PyTorch DataLoader with a custom collate function to separate base and novel samples.
        """
        
        return DataLoader(
            pseudo_base,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
        )

    def get_data_loader2(self, pseudo_novel_dataset: ContiguousLabelDataset, batch_size: int) -> DataLoader:
        """
        Returns a DataLoader that splits each batch into pseudo-base and pseudo-novel subsets.
        """
        return DataLoader(pseudo_novel_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    
    def forward_backward1(
            self,
            sample,
            batch_idx,
            metrics: Dict[str, AverageMeter],
            dataset: ContiguousLabelDataset
    ) -> Dict[str, float]:
        
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

        metrics["ce_loss_metric"].update(loss_ce.item(), n=batch_size_total)

        _, predicted = logits_base.max(dim=1)
        correct = (predicted == targets_base).sum().item()
        total = targets_base.size(0)
        metrics["ce_accuracy_metric"].update(correct, n=total, raw=True)

        return {
            "ce_loss": loss_ce.item(),
            "accuracy": correct / targets_base.size(0),
        }

    def forward_backward2(
            self,
            sample,
            batch_idx,
            metrics: Dict[str, AverageMeter],
            dataset: ContiguousLabelDataset
    ) -> Dict[str, float]:
      
        # Load data into GPU
        novel_batch = sample

        # === Pseudo-novel: KL divergence with frozen CLIP ===
        inputs_novel = torch.stack([img for img, _ in novel_batch]).to(self.device)
        targets_novel = torch.tensor([lbl for _, lbl in novel_batch]).to(self.device)

        kl_loss = get_kl_loss(self.device, inputs_novel, self.model, targets_novel, dataset)

        # === Combine losses ===
        kl_loss = self.lambda_kl * kl_loss

        kl_loss.backward()

        self.optimizer_step()

        metrics["kl_loss_metric"].update(kl_loss.item(), n=inputs_novel.size(0))

        return {
            "kl_loss": kl_loss.item(),
        }

    def debug_metrics_to_pbar_args1(self, debug_metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Prepares debug metrics for visualization in progress bars or logs.

        Args:
            debug_metrics (Dict[str, float]): Dictionary of debug metrics from the current step.

        Returns:
            Dict[str, float]: Same metrics for direct display.
        """
        return debug_metrics

    def debug_metrics_to_pbar_args2(self, debug_metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Prepares debug metrics for visualization in progress bars or logs.
        """
        return debug_metrics

    def training_step_return(self, metrics: Dict[str, AverageMeter]) -> list[float]:
        """
        Returns the average values of tracked metrics after a training step.

        Args:
            metrics (Dict[str, AverageMeter]): Metric dictionary to extract averages from.

        Returns:
            List[float]: Averages of total loss, accuracy, CE loss, and KL loss.
        """
        return [
            metrics["kl_loss_metric"].avg,
            metrics["ce_loss_metric"].avg,
            metrics["ce_accuracy_metric"].avg,
        ]





