"""
 Abstract base class for training methods. it should have forward_backward method to be fulfilled by its children
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Callable

from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.metrics import AverageMeter
from utils.datasets import ContiguousLabelDataset

class DoubleDatasetTrainingMethod:
    """
    Abstract base class for training methods. it should have forward_backward method to be fulfilled by its children

    The TrainingMethod class serves as an abstract interface that standardizes the structure of different training strategies.
    It defines key methods that all training strategies must implement, such as:
     - `get_metrics`: to initialize and return the performance metrics.
     - `get_data_loader`: to prepare the DataLoader tailored for the specific training strategy.
     - `forward_backward`: to implement the forward and backward passes during training.
     - `debug_metrics_to_pbar_args`: to convert debug information for progress bar visualization.
     - `training_step_return`: to return summary metrics after a training epoch.
     It also provides common functionality like `start_training`, `optimizer_step`, and `train_step` to be reused across concrete training methods.
    """

    def __init__(self, model: Any, optimizer: Any, title: str, debug) -> None:
        """
        Initialize the training method.

        Args:
            model (Any): The model to train.
            optimizer (Any): The optimizer to use for training.
            title (str): Title for identification (e.g., for progress display).
            debug (bool): Flag to enable debug mode for extra logging.
        """
        self.model = model
        self.optimizer = optimizer
        self.title = title
        self.device = next(self.model.parameters()).device
        self.debug = debug

    @abstractmethod
    def get_metrics(self) -> Dict[str, AverageMeter]:
        """
        Initialize and return a dictionary of performance metrics.

        Returns:
            Dict[str, AverageMeter]: Dictionary mapping metric names to metric trackers.
        """
        pass

    @abstractmethod
    def get_data_loader1(self, dataset, batch_size) -> DataLoader:
        """
        Create a data loader for the training dataset.

        Args:
            dataset: The training dataset.
            batch_size (int): Number of samples per batch.

        Returns:
            DataLoader: PyTorch data loader instance.
        """
        pass
    @abstractmethod
    def get_data_loader2(self, dataset, batch_size) -> DataLoader:
        """
        Create a data loader for the training dataset.

        Args:
            dataset: The training dataset.
            batch_size (int): Number of samples per batch.

        Returns:
            DataLoader: PyTorch data loader instance.
        """
        pass

    @abstractmethod
    def debug_metrics_to_pbar_args1(self, debug_metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Format debug metrics for display in a progress bar.

        Args:
            debug_metrics (Dict[str, float]): Metrics from the current step.

        Returns:
            Dict[str, float]: Formatted metrics for tqdm display.
        """
        pass

    @abstractmethod
    def debug_metrics_to_pbar_args2(self, debug_metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Format debug metrics for display in a progress bar.
        """
        pass

    @abstractmethod
    def training_step_return(self, metrics: Dict[str, AverageMeter]) -> list[float]:
        """
        Generate summary metrics after completing a training epoch.

        Args:
            metrics (Dict[str, AverageMeter]): Collected training metrics.

        Returns:
            List[float]: List of averaged metric values for reporting.
        """
        pass

    def optimizer_step(self) -> None:
        """
        Apply the optimizer's step and zero the gradients.
        """
        self.optimizer.step()
        self.optimizer.zero_grad()

    def start_training(self) -> None:
        """
        Set the model in training mode.
        """
        self.model.train()

    @abstractmethod
    def forward_backward1(
            self, sample, batch_idx, metrics: Dict[str, AverageMeter], dataset: ContiguousLabelDataset, classes: list[int]
    ) -> Dict[str, float]:
        """
        Execute forward and backward pass, compute loss, and update metrics.

        Args:
            sample: Current batch sample from data loader.
            batch_idx (int): Index of the current batch.
            metrics (Dict[str, AverageMeter]): Metric trackers.
            dataset (ContiguousLabelDataset): Dataset wrapper for label mapping.

        Returns:
            Dict[str, float]: Dictionary of current debug metric values.
        """
        pass

    @abstractmethod
    def forward_backward2(
            self, sample, batch_idx, metrics: Dict[str, AverageMeter], dataset: ContiguousLabelDataset, classes: list[int]
    ) -> Dict[str, float]:
        """
        Execute forward and backward pass, compute loss, and update metrics.
        """

    def double_datasets_train_step(self, dataset1, dataset2, batch_size, names: list[str], classes1, classes2):
        assert len(names) == 2, "Number of names must be 2"
        metrics = self.get_metrics()
        self.start_training()
        tmp_dataset1 = ContiguousLabelDataset(dataset1, classes1)
        tmp_dataset2 = ContiguousLabelDataset(dataset2, classes2)

        dataloader1 = self.get_data_loader1(tmp_dataset1, batch_size)
        dataloader2 = self.get_data_loader2(tmp_dataset2, batch_size)

        pbar = tqdm(dataloader1, desc=f"Training-{self.title}/{names[0]}", position=1, leave=False)
        for batch_idx, sample in enumerate(dataloader1):
            debug_metrics = self.forward_backward1(sample, batch_idx, metrics, tmp_dataset1, classes1)
            pbar.set_postfix(
                self.debug_metrics_to_pbar_args1(debug_metrics)
            )
            pbar.update(1)

        
        pbar = tqdm(dataloader2, desc=f"Training-{self.title}/{names[1]}", position=1, leave=False)
        for batch_idx, sample in enumerate(dataloader2):
            debug_metrics = self.forward_backward2(sample, batch_idx, metrics, tmp_dataset2, classes2)
            pbar.set_postfix(
                self.debug_metrics_to_pbar_args2(debug_metrics)
            )
            pbar.update(1)       

        return self.training_step_return(metrics)





