from typing import Dict

import clip
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import torch

from training_systems.core import EvaluationMethod
from utils import ContiguousLabelDataset, CLASS_NAMES, AverageMeter


class ZeroShotTestStep(EvaluationMethod):
    """
    Evaluation method for models that have been fine-tuned (e.g., CoCoOp or adversarial models).
    """

    def __init__(self, model, batch_size, categories):
        super().__init__(model, batch_size)
        self.categories = categories
        # here we apply the standard CLIP template used for oxford flowers to all categories
        text_inputs = clip.tokenize(
            [f"a photo of a {CLASS_NAMES[c]}, a type of flower." for c in self.categories]
        ).to(self.device)
        with torch.no_grad():
            # we can encode the text features once as they are shared for all images
            # therefore we do it outside the evaluation loop
            self.text_features = self.model.encode_text(text_inputs)
            # and here we normalize them (standard pratice with CLIP)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def evaluate(self, dataset, desc_add="") -> Dict[str, float]:
        self.model.eval()
        accuracy_meter = AverageMeter()
        tmp_dataset = ContiguousLabelDataset(dataset, self.categories)
        dataloader = DataLoader(tmp_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        # Remap labels into a contiguous set starting from zero
        self.walk(dataloader, accuracy_meter, desc_add)

        return {"accuracy": accuracy_meter.avg}

    @torch.no_grad()
    def walk(self, dataloader, accuracy_meter, desc_add):
        pbar = tqdm(total=len(dataloader), desc="Test (Zero Shots) " + desc_add, position=1, leave=False)
        for image, target in dataloader:
            # base categories range from 0 to 50, while novel ones from 51 to 101
            # therefore we must map categories to the [0, 50], otherwise we will have wrong predictions
            # Map targets in contiguous set starting from zero
            # Labels needs to be .long() in pytorch
            
            image = image.to(self.device)
            target = target.to(self.device)

            # forward image through CLIP image encoder
            image_features = self.model.encode_image(image)
            # and normalize
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # here cosine similarity between image and text features and keep the argmax for every row (every image)
            predicted_class = (image_features @ self.text_features.T).argmax(dim=-1)
            # now we check which are correct, and sum them (False == 0, True == 1)

            correct = (predicted_class == target).sum().item()
            accuracy_meter.update(correct, n=target.size(0), raw=True)
            pbar.set_postfix({
                "accuracy" : accuracy_meter.avg
            })
            pbar.update(1)


class FineTunedTestStep(EvaluationMethod):
    """
    Evaluation method for the f rozen base CLIP model.
    """

    @torch.no_grad()
    def evaluate(self, dataset, classnames: list[int], desc_add="") -> Dict[str, float]:
        self.model.eval()
        accuracy_meter = AverageMeter()
        tmp_dataset = ContiguousLabelDataset(dataset, classnames)
        dataloader = DataLoader(tmp_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        remapped_class_names = [CLASS_NAMES[tmp_dataset.idx2cat[i]] for i in range(len(tmp_dataset.idx2cat))]
        with self.model.temporary_classnames(remapped_class_names):
            pbar = tqdm(total=len(dataloader), desc="Test (Finetuned) " + desc_add, position=1, leave=False)
            for images, targets in dataloader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                logits = self.model(images)
                predictions = logits.argmax(dim=-1)
                correct = (predictions == targets).sum().item()
                accuracy_meter.update(correct, n=targets.size(0), raw=True)
                pbar.set_postfix({
                    "accuracy": accuracy_meter.avg
                })
                pbar.update(1)

        return {"accuracy": accuracy_meter.avg}
