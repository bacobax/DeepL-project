"""
 Abstract base class for training methods. it should have forward_backward method to be fulfilled by its children
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Callable

from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.metrics import AverageMeter
from utils.datasets import ContiguousLabelDataset


class TrainingMethod(ABC):
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
    def get_data_loader(self, dataset, batch_size) -> DataLoader:
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
    def forward_backward(
            self, sample, batch_idx, metrics: Dict[str, AverageMeter], dataset: ContiguousLabelDataset
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
    def debug_metrics_to_pbar_args(self, debug_metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Format debug metrics for display in a progress bar.

        Args:
            debug_metrics (Dict[str, float]): Metrics from the current step.

        Returns:
            Dict[str, float]: Formatted metrics for tqdm display.
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

    def train_step(self, dataset, batch_size, classnames):
        """
        Perform a complete training epoch, including data loading, training, and metric collection.

        Args:
            dataset: Dataset used for training.
            batch_size (int): Number of samples per training batch.

        Returns:
            List[float]: Averaged values for each tracked metric after the epoch.
        """
        metrics = self.get_metrics()
        self.start_training()
        tmp_dataset = ContiguousLabelDataset(dataset, classnames)
        dataloader = self.get_data_loader(tmp_dataset, batch_size)
        pbar = tqdm(dataloader, desc=f"Training-{self.title}", position=1, leave=False)
        for batch_idx, sample in enumerate(dataloader):
            debug_metrics = self.forward_backward(sample, batch_idx, metrics, tmp_dataset)
            pbar.set_postfix(
                self.debug_metrics_to_pbar_args(debug_metrics)
            )
            pbar.update(1)
        return self.training_step_return(metrics)







