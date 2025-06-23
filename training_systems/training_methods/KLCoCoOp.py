"""
This module defines the KLCoCoOp training method, which combines standard cross-entropy loss with KL divergence
between a model's predictions and a frozen teacher (e.g., CLIP) to enhance generalization to novel classes.
"""
import random
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader

from training_systems.training_methods.TrainingMethod import TrainingMethod

from utils.metrics import AverageMeter
from utils.datasets import ContiguousLabelDataset
from utils.kl import get_kl_loss


class KLCoCoOp(TrainingMethod):
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
            debug: bool = False
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
            "total_loss_metric": AverageMeter(),
            "ce_loss_metric": AverageMeter(),
            "kl_loss_metric": AverageMeter(),
            "accuracy_metric": AverageMeter(),
        }

    def get_data_loader(self, dataset: ContiguousLabelDataset, batch_size: int) -> DataLoader:
        """
        Returns a DataLoader that splits each batch into pseudo-base and pseudo-novel subsets.

        Args:
            dataset (ContiguousLabelDataset): The dataset used for training.
            batch_size (int): Number of samples per batch.

        Returns:
            DataLoader: PyTorch DataLoader with a custom collate function to separate base and novel samples.
        """

        def custom_collate(batch):
            base_samples = []
            novel_samples = []
            targets_in_batch = list(set([target for _, target in batch]))
            random.shuffle(targets_in_batch)
            split_idx = int(0.7 * len(targets_in_batch))
            pseudo_base_ids = targets_in_batch[:split_idx]
            pseudo_novel_ids = targets_in_batch[split_idx:]
            for img, label in batch:
                if label in pseudo_base_ids:
                    base_samples.append((img, label))
                elif label in pseudo_novel_ids:
                    novel_samples.append((img, label))
            return base_samples, novel_samples

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            collate_fn=custom_collate,
        )

    def forward_backward(
            self,
            sample,
            batch_idx,
            metrics: Dict[str, AverageMeter],
            dataset: ContiguousLabelDataset
    ) -> Dict[str, float]:
        """
        Performs forward and backward passes, computing CE loss on pseudo-base and KL loss on pseudo-novel samples.

        Args:
            sample (Tuple[List[Tuple[Tensor, int]], List[Tuple[Tensor, int]]]): Tuple of pseudo-base and pseudo-novel batches.
            batch_idx (int): Current batch index during training.
            metrics (Dict[str, AverageMeter]): Dictionary to update with the training metrics.
            dataset (ContiguousLabelDataset): Dataset object used for KL divergence lookup.

        Returns:
            Dict[str, float]: Scalar values of total loss, CE loss, KL loss, and CE accuracy.
        """
        # Load data into GPU
        base_batch, novel_batch = sample

        if not base_batch or not novel_batch:
            return {
                metric: val
                for metric, val in metrics.items()
            }

        # === Pseudo-base: cross-entropy ===
        inputs_base = torch.stack([img for img, _ in base_batch]).to(self.device)
        targets_base = torch.tensor([lbl for _, lbl in base_batch]).to(self.device)

        logits_base, loss_ce = self.model(inputs_base, targets_base)

        # === Pseudo-novel: KL divergence with frozen CLIP ===
        self.model.eval()  # needed to disable dropout etc.
        inputs_novel = torch.stack([img for img, _ in novel_batch]).to(self.device)
        targets_novel = [lbl for _, lbl in novel_batch]

        kl_loss = get_kl_loss(self.device, inputs_novel, self.model, targets_novel, dataset)

        # === Combine losses ===
        total_loss = loss_ce + self.lambda_kl * kl_loss

        total_loss.backward()

        self.optimizer_step()

        batch_size_total = inputs_base.size(0) + inputs_novel.size(0)

        metrics["total_loss_metric"].update(total_loss.item(), n=batch_size_total)
        metrics["ce_loss_metric"].update(loss_ce.item(), n=inputs_base.size(0))
        metrics["kl_loss_metric"].update(kl_loss.item(), n=inputs_novel.size(0))

        _, predicted = logits_base.max(dim=1)
        correct = (predicted == targets_base).sum().item()
        total = targets_base.size(0)
        metrics["accuracy_metric"].update(correct, n=total, raw=True)

        return {
            "total_loss": total_loss.item(),
            "ce_loss": loss_ce.item(),
            "ce_accuracy": correct / targets_base.size(0),
            "kl_loss": kl_loss.item(),
        }

    def debug_metrics_to_pbar_args(self, debug_metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Prepares debug metrics for visualization in progress bars or logs.

        Args:
            debug_metrics (Dict[str, float]): Dictionary of debug metrics from the current step.

        Returns:
            Dict[str, float]: Same metrics for direct display.
        """
        return debug_metrics

    def training_step_return(self, metrics: Dict[str, AverageMeter]) -> [float]:
        """
        Returns the average values of tracked metrics after a training step.

        Args:
            metrics (Dict[str, AverageMeter]): Metric dictionary to extract averages from.

        Returns:
            List[float]: Averages of total loss, accuracy, CE loss, and KL loss.
        """
        return [
            metrics["total_loss_metric"].avg,
            metrics["accuracy_metric"].avg,
            metrics["ce_loss_metric"].avg,
            metrics["kl_loss_metric"].avg,
        ]





