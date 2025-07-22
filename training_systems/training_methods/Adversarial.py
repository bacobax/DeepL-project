"""
This module implements the Adversarial training method, which incorporates a gradient reversal layer
and an adversarial MLP to encourage domain-invariant feature learning.
"""
from typing import Dict, Any, List

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from training_systems.core import TrainingMethod
from utils import AverageMeter, ContiguousLabelDataset, CLASS_NAMES
from model.cocoop.mlp_adversary import GradientReversalLayer


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
            mlp_adversary,
            lambda_adv,
            tmp_classes: list,
            debug: bool = False,
            gaussian_noise: float = 0.0,
            use_bias_ctx: bool = False
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
        self.tmp_classes = tmp_classes
        self.gaussian_noise = gaussian_noise
        self.use_bias_ctx = use_bias_ctx

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
            dataset: ContiguousLabelDataset,
            accumulation_steps: int = 1,
            step: int = 0
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

        # Use all tmp_classes for adversarial phase
        with self.model.temporary_classnames([CLASS_NAMES[idx] for idx in self.tmp_classes]):
            logits, ce_loss, img_features, _, _, _, _ = self.model(inputs, targets, get_image_features=True, met_net_2=True)
             

            reversed_img_features = self.grl(img_features)
            # print(f"reversed_concat shape: {reversed_concat.shape}")
            cluster_logits, loss_adv = self.mlp_adversary(reversed_img_features, cluster_target.float())


            total_loss = ce_loss + self.lambda_adv * loss_adv

            total_loss = total_loss / accumulation_steps
            total_loss.backward()
            # print(f"step: {step}, total_loss: {total_loss.item():.4f}, accumulation_steps: {accumulation_steps}, ")
            # --- accumulate grads and update ---
            if (step + 1) % accumulation_steps == 0: 
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) + list(self.mlp_adversary.parameters()),
                    max_norm=1.0,
                    norm_type=2.0,
                    error_if_nonfinite=True
                )
                # if step == 1:
                #     for name, param in self.model.named_parameters():
                #         if param.requires_grad:
                #             print(f"prompt_learner Not frozen param: {name}")
                #         else:
                #             print(f"prompt_learner Frozen param: {name}")
                #     for name, param in self.mlp_adversary.named_parameters():
                #         if param.requires_grad:
                #             print(f"Adversary Not Frozen param: {name}")
                #         else:
                #             print(f"Adversary Frozen param: {name}")
                #     for param in self.mlp_adversary.parameters():
                #         if param.requires_grad:
                #             print(f"Adversary param: {param.shape}, requires_grad: {param.requires_grad}")
                #         else:
                #             print(f"Adversary Frozen param: {param.shape}, requires_grad: {param.requires_grad}")
                self.optimizer_step()
                self.optimizer.zero_grad()
            
            if batch_idx < 3 and self.debug:  # Log only a few batches for performance
                print(f"[Batch {batch_idx}] CE Loss: {ce_loss.item():.4f} | "
                      f"Adv Loss: {loss_adv.item():.4f} | "
                      f"Total Loss: {total_loss.item():.4f} | "
                      f"lambda_adv: {self.lambda_adv:.4f}")
            batch_size = inputs.shape[0]
            metrics["total_loss_metric"].update(total_loss.item(), n=batch_size)
            metrics["ce_loss_metric"].update(ce_loss.item(), n=batch_size)
            metrics["adv_loss_metric"].update(loss_adv.item(), n=batch_size)
            _, predicted = logits.max(dim=1)
            correct = predicted.eq(targets).sum().item()
            metrics["accuracy_metric"].update(correct, n=batch_size, raw=True)
            return {
                "total_loss": total_loss.item(),
                "ce_loss": ce_loss.item(),
                "adv_loss": loss_adv.item(),
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

    def training_step_return(self, metrics: Dict[str, AverageMeter]) -> list[float]:
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
