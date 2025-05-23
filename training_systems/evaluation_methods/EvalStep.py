
from typing import Dict
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import torch

from training_systems.evaluation_methods.EvaluationMethod import EvaluationMethod
from utils.datasets import ContiguousLabelDataset, CLASS_NAMES
from utils.metrics import AverageMeter


class EvalStep(EvaluationMethod):
    """
    Generic evaluation step for models that support temporary class name modification.
    """
    @torch.no_grad()
    def evaluate(self, dataset, new_classnames=None, desc_add="") -> Dict[str, float]:
        """
        Evaluate model performance on the provided dataset.

        Args:
            dataset: Dataset for evaluation.
            new_classnames (list, optional): New class names to apply temporarily.
            desc_add (str): Suffix to append to tqdm description.

        Returns:
            Dict[str, float]: Dictionary containing average loss and accuracy.
        """
        self.model.eval()
        loss_meter = AverageMeter()
        accuracy_meter = AverageMeter()
        tmp_dataset = ContiguousLabelDataset(dataset)
        dataloader = DataLoader(tmp_dataset, batch_size=self.batch_size, shuffle=False, num_workers=1)

        if new_classnames is not None:
            new_classnames = [CLASS_NAMES[c] for c in new_classnames]
            with self.model.temporary_classnames(new_classnames):
                self.walk(loss_meter, accuracy_meter, dataloader, desc_add)
        else:
            self.walk(loss_meter, accuracy_meter, dataloader, desc_add)

        return {"loss": loss_meter.avg, "accuracy": accuracy_meter.avg}

    @torch.no_grad()
    def walk(self, loss_meter, accuracy_meter, dataloader, desc_add=""):
        """
        Perform the evaluation loop over the dataset.

        Args:
            loss_meter: Tracks average loss.
            accuracy_meter: Tracks average accuracy.
            dataloader: DataLoader to iterate over.
            desc_add (str): Additional string to append to tqdm description.
        """
        for images, targets in tqdm(dataloader, desc="Validation" + desc_add, position=1, leave=False):
            images = images.to(self.device)
            targets = targets.to(self.device)
            logits = self.model(images)
            loss = F.cross_entropy(logits, targets)
            predictions = logits.argmax(dim=-1)
            correct = (predictions == targets).sum().item()
            loss_meter.update(loss.item(), n=targets.size(0))
            accuracy_meter.update(correct, n=targets.size(0), raw=True)