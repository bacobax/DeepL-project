"""
This module implements the Adversarial training method, which incorporates a gradient reversal layer
and an adversarial MLP to encourage domain-invariant feature learning.
"""

from typing import Dict, Any

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from training_systems.core import TrainingMethod
from utils import AverageMeter, ContiguousLabelDataset
from model.cocoop.mlp_adversary import AdversarialMLP, GradientReversalLayer


class OnlyMLP(TrainingMethod):
    """
    Adversarial training method using a Gradient Reversal Layer and an MLP adversary.

    Attributes:
        cls_cluster_dict (Dict[int, Any]): Maps class indices to cluster labels.
        grl (GradientReversalLayer): The gradient reversal layer instance.
        mlp_adversary (AdversarialMLP): The adversarial MLP used to confuse cluster prediction.
        lambda_adv (float): Weight of the adversarial loss term.
        debug (bool): If True, print debug information.
    """

    def __init__(
            self,
            model: Any,
            optimizer: Any,
            cls_cluster_dict: Dict[int, Any],
            mlp_adversary: AdversarialMLP,
            debug: bool = False
    ) -> None:
        """
        Args:
            model (Any): The main model being trained.
            optimizer (Any): Optimizer for updating model parameters.
            cls_cluster_dict (Dict[int, Any]): Mapping from class labels to clusters.
            grl (GradientReversalLayer): The gradient reversal layer.
            mlp_adversary (AdversarialMLP): The adversarial network module.
            lambda_adv (float): Weight for the adversarial loss.
            debug (bool, optional): Enables debug mode. Defaults to False.
        """
        super().__init__(model, optimizer, "Adv.", debug)
        self.cls_cluster_dict = cls_cluster_dict
        self.mlp_adversary = mlp_adversary

    def get_metrics(self) -> Dict[str, AverageMeter]:
        """
        Returns:
            Dict[str, AverageMeter]: Dictionary containing initialized metrics for training.
        """
        return {
            "adv_loss_metric": AverageMeter(),
            "accuracy_metric": AverageMeter(),
        }

    def get_data_loader(self, dataset: ContiguousLabelDataset, batch_size: int) -> DataLoader:
        """
        Args:
            dataset (ContiguousLabelDataset): Dataset to be used.
            batch_size (int): Size of each batch.

        Returns:
            DataLoader: Configured PyTorch DataLoader instance.
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
        Executes the forward and backward pass.

        Args:
            sample (Tuple[Tensor, Tensor]): Batch of input data and targets.
            batch_idx (int): Index of the current batch.
            metrics (Dict[str, AverageMeter]): Dictionary of metrics to be updated.
            dataset (ContiguousLabelDataset): Dataset for cluster label lookup.

        Returns:
            Dict[str, float]: Dictionary with loss and accuracy metrics.
        """
        # Load data into GPU
        inputs, targets = sample
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        targets_real_category = [dataset.idx2cat[c.item()] for c in targets]
        cluster_target = [int(self.cls_cluster_dict[int(tl)]) for tl in targets_real_category]
        cluster_target = torch.tensor(
            cluster_target,
            device=targets.device,
            dtype=torch.float32
        )

        # Forward pass + loss computation dsada
        logits, ce_loss, img_features, ctx, bias = self.model(inputs, targets, get_image_features=True)
        ctx = ctx.detach()  # Detach context to avoid backprop through it
        ctx_shifted = ctx + bias.unsqueeze(1).detach()  # Add bias to context tokens
        ctx_flat = ctx_shifted.reshape(ctx_shifted.size(0), -1).to(dtype=torch.float32)

        cluster_logits = self.mlp_adversary(ctx_flat).squeeze()

        loss_bce = F.binary_cross_entropy_with_logits(cluster_logits, cluster_target)


        # Backward pass
        loss_bce.backward()

        torch.nn.utils.clip_grad_norm_(
            list(self.model.parameters()) + list(self.mlp_adversary.parameters()),
            max_norm=1.0,
            norm_type=2.0,
            error_if_nonfinite=True
        )

        self.optimizer_step()


        # Fetch prediction and loss value
        batch_size = inputs.shape[0]
        metrics["adv_loss_metric"].update(loss_bce.item(), n=batch_size)

        predicted = cluster_logits.gt(0.5).float()  # Convert logits to binary predictions
        correct = predicted.eq(cluster_target).sum().item()
        # Compute training accuracy
        metrics["accuracy_metric"].update(correct, n=batch_size, raw=True)

        return {
            "adv_loss": loss_bce.item(),
            "accuracy": correct / batch_size,
        }

    def debug_metrics_to_pbar_args(self, debug_metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Args:
            debug_metrics (Dict[str, float]): Metrics from current training step.

        Returns:
            Dict[str, float]: Same metrics, passed to progress bar.
        """
        return debug_metrics

    def training_step_return(self, metrics: Dict[str, AverageMeter]) -> list[float]:
        """
        Args:
            metrics (Dict[str, AverageMeter]): Collected training metrics.

        Returns:
            List[float]: Average values of total, accuracy, CE, and adversarial losses.
        """
        return [
            metrics["accuracy_metric"].avg,
            metrics["adv_loss_metric"].avg,
        ]

    def update_lambda_adv(self, lambda_adv) -> None:
        """
        Args:
            lambda_adv (float): New value to set for lambda_adv.
        """
        self.lambda_adv = lambda_adv
