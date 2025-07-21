from abc import ABC, abstractmethod
from typing import Dict, Any
from torch.utils.data import DataLoader
from utils.metrics import AverageMeter


class EvaluationMethod(ABC):
    """
    Abstract base class for evaluation methods. Standardizes evaluation interface for different strategies.
    """

    def __init__(self, model, batch_size: int = 32, device=None):
        """
        Initialize evaluation method.

        Args:
            model: The model to evaluate.
            batch_size (int): Evaluation batch size.
        """
        self.model = model
        self.device = next(self.model.parameters()).device if device is None else device
        self.batch_size = batch_size

    @abstractmethod
    def evaluate(self, dataset, classnames, desc_add="") -> Dict[str, float]:
        """
        Perform the evaluation.

        Args:
            dataset: The dataset to evaluate on.

        Returns:
            Dict[str, float]: Dictionary with evaluation results (e.g., accuracy, loss).
        """
        pass