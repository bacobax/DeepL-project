"""
 Abstract base class for training methods. it should have forward_backward method to be fulfilled by its children
"""

from abc import ABC, abstractmethod
from typing import Dict, Any

from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.metrics import AverageMeter
from utils.datasets import ContiguousLabelDataset


class TrainingMethod(ABC):
    """
    Abstract base class for training methods. it should have forward_backward method to be fulfilled by its children
    """
    def __init__(self, model: Any, optimizer: Any, title: str) -> None:
        self.model = model
        self.optimizer = optimizer
        self.title = title
        self.device = next(self.model.parameters()).device

    @abstractmethod
    def get_metrics(self) -> Dict[str, AverageMeter]:
        """
        Get the metrics for the training method.
        """
        pass

    @abstractmethod
    def get_data_loader(self, dataset, batch_size) -> DataLoader:
        """
        Get the data loader for the training method.
        """
        pass

    @abstractmethod
    def forward_backward(self, sample, batch_idx, metrics: Dict[str, AverageMeter], dataset: ContiguousLabelDataset) -> Dict[str, float]:
        """
        Forward and backward pass for the training method.
        """
        pass

    @abstractmethod
    def debug_metrics_to_pbar_args(self, debug_metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Convert debug metrics to pbar args.
        """
        pass



    @abstractmethod
    def training_step_return(self, metrics: Dict[str, AverageMeter]) -> [float]:
        """
        Return the training step metrics.
        """
        pass



    def optimizer_step(self) -> None:
        """
        Perform an optimizer step.
        """
        self.optimizer.step()
        self.optimizer.zero_grad()

    def start_training(self) -> None:
        """
        Start training.
        """
        self.model.train()

    def train_step(
            self,
            dataset,
            batch_size,
    ):
        metrics = self.get_metrics()
        self.start_training()
        tmp_dataset = ContiguousLabelDataset(dataset)
        dataloader = self.get_data_loader(tmp_dataset, batch_size)
        pbar = tqdm(dataloader, desc=f"Training-{self.title}", position=1, leave=False)
        for batch_idx, sample in enumerate(dataloader):
            debug_metrics = self.forward_backward(sample, batch_idx, metrics, tmp_dataset)
            pbar.set_postfix(
                self.debug_metrics_to_pbar_args(debug_metrics)
            )
            pbar.update(1)
        return self.training_step_return(metrics)




