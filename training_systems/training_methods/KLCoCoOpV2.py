"""
This module defines the KLCoCoOp training method, which combines standard cross-entropy loss with KL divergence
between a model's predictions and a frozen teacher (e.g., CLIP) to enhance generalization to novel classes.
"""
import random
from typing import Dict, Any, Optional

import clip
import torch
from torch.utils.data import DataLoader, Dataset

from training_systems.training_methods.DoubleDatasetTrainingMethod import DoubleDatasetTrainingMethod
from training_systems.training_methods.TrainingMethod import TrainingMethod

from utils.metrics import AverageMeter
from utils.datasets import CLASS_NAMES, ContiguousLabelDataset
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
        if self.debug:
            print(f"[KLCoCoOpV2] Initialized with lambda_kl={self.lambda_kl}, device={self.device}")


    def get_metrics(self) -> Dict[str, AverageMeter]:
        """
        Initializes training metrics including total, cross-entropy, KL divergence losses, and classification accuracy.

        Returns:
            Dict[str, AverageMeter]: Dictionary of metric names mapped to their respective AverageMeter instances.
        """
        if self.debug:
            print("[KLCoCoOpV2] Initializing metrics.")
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
        if self.debug:
            print(f"[KLCoCoOpV2] Creating DataLoader1 for pseudo_base with batch_size={batch_size}.")
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
        if self.debug:
            print(f"[KLCoCoOpV2] Creating DataLoader2 for pseudo_novel with batch_size={batch_size}.")
        return DataLoader(pseudo_novel_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    
    def forward_backward1(
            self,
            sample,
            batch_idx,
            metrics: Dict[str, AverageMeter],
            dataset: ContiguousLabelDataset
    ) -> Dict[str, float]:
        if self.debug and batch_idx < 3:
            print(f"[KLCoCoOpV2] forward_backward1: batch_idx={batch_idx}")
            print(f"[KLCoCoOpV2] Sample type: {type(sample)}")
        # Load data into GPU
        inputs, targets = sample
        # === Pseudo-base: cross-entropy ===
        inputs_base = inputs.to(self.device)
        targets_base = targets.to(self.device)

        logits_base, loss_ce = self.model(inputs_base, targets_base)
        # === Combine losses ===
        if self.debug and batch_idx < 3:
            print(f"[KLCoCoOpV2] LOGITS shape: {logits_base.shape}, TARGETS shape: {targets_base.shape}")
            print(f"[KLCoCoOpV2] CE loss: {loss_ce.item()}")
        loss_ce.backward()

        self.optimizer_step()
        batch_size_total = inputs_base.size(0)

        metrics["ce_loss_metric"].update(loss_ce.item(), n=batch_size_total)

        _, predicted = logits_base.max(dim=1)
        correct = (predicted == targets_base).sum().item()
        total = targets_base.size(0)
        metrics["ce_accuracy_metric"].update(correct, n=total, raw=True)

        if self.debug and batch_idx < 3:
            print(f"[KLCoCoOpV2] Batch accuracy: {correct}/{total} = {correct/total if total > 0 else 0}")
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
        all_pseudo_novel_classes = [item[0] for item in dataset.cat2idx.items()]
        pseudo_novel_class_names = [CLASS_NAMES[c] for c in all_pseudo_novel_classes]
        if self.debug and batch_idx < 3:
            print(f"[KLCoCoOpV2] forward_backward2: batch_idx={batch_idx}")
            print(f"[KLCoCoOpV2] Sample type: {type(sample)}; Sample len: {len(sample) if hasattr(sample, '__len__') else 'N/A'}")
        # Load data into GPU
        inputs_novel, targets_novel = sample
        # === Pseudo-novel: KL divergence with frozen CLIP ===
        inputs_novel = inputs_novel.to(self.device)
        targets_novel = targets_novel.to(self.device)

        with torch.no_grad():
            image_features_clip = self.model.clip_model.encode_image(inputs_novel)
            image_features_clip = image_features_clip / image_features_clip.norm(dim=-1, keepdim=True)

            #category_idxs = [dataset.idx2cat[c.item()] for c in list(set(targets_novel))] # type: ignore

            text_inputs = clip.tokenize(
                [f"a photo of a {CLASS_NAMES[c]}, a type of flower." for c in pseudo_novel_class_names]
            ).to(self.device)

            text_features_clip = self.model.clip_model.encode_text(text_inputs)
            text_features_clip = text_features_clip / text_features_clip.norm(dim=-1, keepdim=True)

            clip_logits = image_features_clip @ text_features_clip.T
        
        print(f"CLIP LOGITS SHAPE: {clip_logits.shape}")

        self.model.train()
        with self.model.temporary_classnames(pseudo_novel_class_names):
            student_logits, student_loss = self.model(inputs_novel, targets_novel)  # [B, num_classes]
            print(f"STUDENT LOGITS SHAPE: {student_logits.shape}")
            kl_loss = torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(student_logits, dim=-1),
                torch.nn.functional.softmax(clip_logits, dim=-1),
                reduction="batchmean"
            )

        # === Combine losses ===
        kl_loss = self.lambda_kl * kl_loss

        if self.debug and batch_idx < 3:
            print(f"[KLCoCoOpV2] KL loss (weighted): {kl_loss.item()}")
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
        if self.debug:
            print(f"[KLCoCoOpV2] debug_metrics_to_pbar_args1: {debug_metrics}")
        return debug_metrics

    def debug_metrics_to_pbar_args2(self, debug_metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Prepares debug metrics for visualization in progress bars or logs.
        """
        if self.debug:
            print(f"[KLCoCoOpV2] debug_metrics_to_pbar_args2: {debug_metrics}")
        return debug_metrics

    def training_step_return(self, metrics: Dict[str, AverageMeter]) -> list[float]:
        """
        Returns the average values of tracked metrics after a training step.

        Args:
            metrics (Dict[str, AverageMeter]): Metric dictionary to extract averages from.

        Returns:
            List[float]: Averages of total loss, accuracy, CE loss, and KL loss.
        """
        if self.debug:
            print(f"[KLCoCoOpV2] training_step_return: KL={metrics['kl_loss_metric'].avg}, CE={metrics['ce_loss_metric'].avg}, Acc={metrics['ce_accuracy_metric'].avg}")
        return [
            metrics["kl_loss_metric"].avg,
            metrics["ce_loss_metric"].avg,
            metrics["ce_accuracy_metric"].avg,
        ]





