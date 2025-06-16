"""
This module implements the Adversarial training method, which incorporates a gradient reversal layer
and an adversarial MLP to encourage domain-invariant feature learning.
"""

from typing import Dict, Any

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from training_systems.training_methods.TrainingMethod import TrainingMethod
from utils.metrics import AverageMeter
from utils.datasets import ContiguousLabelDataset
from model.cocoop.mlp_adversary import AdversarialMLP, GradientReversalLayer


class Adversarial(TrainingMethod):
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
            grl: GradientReversalLayer,
            mlp_adversary: AdversarialMLP,
            lambda_adv,
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
        self.grl = grl
        self.mlp_adversary = mlp_adversary
        self.lambda_adv = lambda_adv

    def get_metrics(self) -> Dict[str, AverageMeter]:
        """
        Returns:
            Dict[str, AverageMeter]: Dictionary containing initialized metrics for training.
        """
        return {
            "total_loss_metric": AverageMeter(),
            "ce_loss_metric": AverageMeter(),
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
        logits, ce_loss, img_features = self.model(inputs, targets, get_image_features=True)

        concat = torch.cat([img_features, logits], dim=1).to(dtype=torch.float32)
        # === Adversarial loss ===
        reversed_logits = self.grl(concat)
        cluster_logits = self.mlp_adversary(reversed_logits).squeeze()

        loss_bce = F.binary_cross_entropy_with_logits(cluster_logits, cluster_target)
        # Skip adversarial update if prompt learner is frozen
        if any(p.requires_grad for p in self.model.prompt_learner.parameters()):
            ce_grads = self.get_grads(ce_loss)
        else:
            ce_grads = None  # Or torch.zeros_like(...), depending on downstream use

        if batch_idx < 3 and self.debug:
            bce_grads = self.get_grads(loss_bce)
            self.print_grads_norms(bce_grads, ce_grads)

        # === Combine losses ===
        total_loss = ce_loss + self.lambda_adv * loss_bce

        # Backward pass
        total_loss.backward()
        """if batch_idx < 3:
            check_gradients(model)
            check_gradients(mlp_adversary)"""

        torch.nn.utils.clip_grad_norm_(
            list(self.model.parameters()) + list(self.mlp_adversary.parameters()),
            max_norm=1.0,
            norm_type=2.0,
            error_if_nonfinite=True
        )

        self.optimizer_step()

        if batch_idx < 3 and self.debug:  # Log only a few batches for performance
            print(f"[Batch {batch_idx}] CE Loss: {ce_loss.item():.4f} | "
                  f"Adv Loss: {loss_bce.item():.4f} | "
                  f"Total Loss: {total_loss.item():.4f} | "
                  f"lambda_adv: {self.lambda_adv:.4f}")

        # Fetch prediction and loss value
        batch_size = inputs.shape[0]
        metrics["total_loss_metric"].update(total_loss.item(), n=batch_size)
        metrics["ce_loss_metric"].update(ce_loss.item(), n=batch_size)
        metrics["adv_loss_metric"].update(loss_bce.item(), n=batch_size)

        _, predicted = logits.max(dim=1)  # max() returns (maximum_value, index_of_maximum_value)

        # Compute training accuracy
        correct = predicted.eq(targets).sum().item()
        metrics["accuracy_metric"].update(correct, n=batch_size, raw=True)

        return {
            "total_loss": total_loss.item(),
            "ce_loss": ce_loss.item(),
            "adv_loss": loss_bce.item(),
            "accuracy": correct / batch_size,
        }

    def print_grads_norms(self, bce_grads, ce_grads):
        """
        Args:
            bce_grads (Dict[str, Tensor]): Gradients from BCE loss.
            ce_grads (Dict[str, Tensor]): Gradients from CE loss.
        """
        for name in ce_grads:
            ce_norm = ce_grads[name].norm().item()
            bce_norm = bce_grads[name].norm().item()
            print(f"{name}: CE grad norm = {ce_norm:.4e}, BCE grad norm = {bce_norm:.4e}")

    def get_grads(self, loss):
        """
        Args:
            loss (Tensor): Loss tensor to backpropagate.

        Returns:
            Dict[str, Tensor]: Dictionary of gradients.
        """
        loss.backward(retain_graph=True)
        ce_grads = {}
        for name, p in self.model.named_parameters():
            if p.grad is not None and "prompt_learner" in name:
                ce_grads[name] = p.grad.detach().clone()
        # --- Zero gradients ---
        self.optimizer.zero_grad()
        return ce_grads

    def debug_metrics_to_pbar_args(self, debug_metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Args:
            debug_metrics (Dict[str, float]): Metrics from current training step.

        Returns:
            Dict[str, float]: Same metrics, passed to progress bar.
        """
        return debug_metrics

    def training_step_return(self, metrics: Dict[str, AverageMeter]) -> [float]:
        """
        Args:
            metrics (Dict[str, AverageMeter]): Collected training metrics.

        Returns:
            List[float]: Average values of total, accuracy, CE, and adversarial losses.
        """
        return [
            metrics["total_loss_metric"].avg,
            metrics["accuracy_metric"].avg,
            metrics["ce_loss_metric"].avg,
            metrics["adv_loss_metric"].avg,
        ]

    def update_lambda_adv(self, lambda_adv) -> None:
        """
        Args:
            lambda_adv (float): New value to set for lambda_adv.
        """
        self.lambda_adv = lambda_adv
