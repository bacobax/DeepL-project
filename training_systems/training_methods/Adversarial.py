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
    Adversarial training method.
    """

    def __init__(
            self,
            model: Any,
            optimizer: Any,
            cls_cluster_dict: Dict[int, Any],
            grl: GradientReversalLayer,
            mlp_adversary: AdversarialMLP,
            lambda_adv
    ) -> None:
        super().__init__(model, optimizer, "Adv.")
        self.cls_cluster_dict = cls_cluster_dict
        self.grl = grl
        self.mlp_adversary = mlp_adversary
        self.lambda_adv = lambda_adv

    def get_metrics(self) -> Dict[str, AverageMeter]:
        """
        Get the metrics for the adversarial training method.
        """
        return {
            "total_loss_metric": AverageMeter(),
            "ce_loss_metric": AverageMeter(),
            "adv_loss_metric": AverageMeter(),
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
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        targets_real_category = [dataset.idx2cat[c.item()] for c in targets]
        cluster_target = [int(self.cls_cluster_dict[int(tl)]) for tl in targets_real_category]
        cluster_target = torch.tensor(
            cluster_target,
            device=targets.device,
            dtype=torch.float16
        )
        # Forward pass + loss computation
        logits, ce_loss, img_features = self.model(inputs, targets, get_image_features=True)
        # === Adversarial loss ===
        reversed_logits = self.grl(torch.cat([img_features, logits], dim=1))
        cluster_logits = self.mlp_adversary(reversed_logits).squeeze()
        loss_bce = F.binary_cross_entropy_with_logits(cluster_logits, cluster_target)

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

        if batch_idx < 3:  # Log only a few batches for performance
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

    def debug_metrics_to_pbar_args(self, debug_metrics: Dict[str, float]) -> Dict[str, float]:
        return debug_metrics

    def training_step_return(self, metrics: Dict[str, AverageMeter]) -> [float]:
        return [
            metrics["total_loss_metric"].avg,
            metrics["accuracy_metric"].avg,
            metrics["ce_loss_metric"].avg,
            metrics["adv_loss_metric"].avg,
        ]

    def update_lambda_adv(self, lambda_adv) -> None:
        """
        Update the lambda_adv parameter.
        """
        self.lambda_adv = lambda_adv

