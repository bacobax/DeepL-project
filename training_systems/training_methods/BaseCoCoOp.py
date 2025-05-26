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
    Adversarial training method.
    """

    def __init__(
            self,
            model: Any,
            optimizer: Any,
            debug: bool = False
    ) -> None:
        super().__init__(model, optimizer, "Base CoCoOp + KL", debug)

    def get_metrics(self) -> Dict[str, AverageMeter]:
        """
        Get the metrics for the adversarial training method.
        """

        return {
            "loss_metric": AverageMeter(),
            "accuracy_metric": AverageMeter(),
        }

    def get_data_loader(self, dataset: ContiguousLabelDataset, batch_size: int) -> DataLoader:
        """
        Get the data loader for the adversarial training method.
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
        # Load data into GPU
        inputs, targets = sample
        # === Pseudo-base: cross-entropy ===
        inputs_base = inputs.to(self.device)
        targets_base = targets.to(self.device)

        logits_base, loss_ce = self.model(inputs_base, targets_base)
        # === Combine losses ===

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
        return debug_metrics

    def training_step_return(self, metrics: Dict[str, AverageMeter]) -> [float]:
        return [
            metrics["loss_metric"].avg,
            metrics["accuracy_metric"].avg,
        ]





