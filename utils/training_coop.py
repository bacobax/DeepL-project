"""
This module provides training, evaluation, and testing functions for the CoOp model and standard CLIP evaluation,
including fine-tuning and zero-shot classification on image datasets.
"""
from clip.model import CLIP
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import clip

from model.coop.custom_clip import CustomCLIPCoOp
from utils.datasets import ContiguousLabelDataset, CLASS_NAMES


@torch.no_grad()
def eval_step(model, dataset, cost_function, new_classnames, batch_size=32, device="cuda"):
    """
    Evaluates the model on a given dataset using cross-entropy loss.

    Args:
        model (nn.Module): The model to evaluate.
        dataset (Dataset): Dataset to evaluate on.
        cost_function (Callable): Loss function to use.
        batch_size (int): Batch size for evaluation.
        device (str): Computation device ("cuda" or "cpu").
        new_classnames (List[int] or None): Optional list of class indices to temporarily substitute for evaluation.

    Returns:
        Tuple[float, float]: Average loss and accuracy.
    """
    model.eval()

    tmp_dataset = ContiguousLabelDataset(dataset, new_classnames)
    dataloader = DataLoader(tmp_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    total_loss = 0.0
    correct = 0
    total = 0
    if new_classnames is not None:
        new_classnames = [CLASS_NAMES[c] for c in new_classnames]
        with model.temporary_classnames(new_classnames):
            correct, total, total_loss = walk_the_dataset(correct, cost_function, dataloader, device, model, total,
                                                          total_loss)

    else:
        correct, total, total_loss = walk_the_dataset(correct, cost_function, dataloader, device, model, total,
                                                      total_loss)
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def walk_the_dataset(correct, cost_function, dataloader, device, model, total, total_loss):
    """
    Iterates over the dataset and computes cumulative loss and accuracy.

    Args:
        correct (int): Running count of correct predictions.
        cost_function (Callable): Loss function used.
        dataloader (DataLoader): DataLoader for the dataset.
        device (str): Computation device.
        model (nn.Module): Model being evaluated.
        total (int): Total number of samples evaluated so far.
        total_loss (float): Accumulated loss value.

    Returns:
        Tuple[int, int, float]: Updated correct, total, and total_loss.
    """
    for images, targets in tqdm(dataloader, desc="Validation", position=1, leave=False):
        images = images.to(device)
        targets = targets.to(device)

        loss, logits = model(images, targets)

        total_loss += loss.item() * targets.size(0)
        predictions = logits.argmax(dim=-1)
        correct += (predictions == targets).sum().item()
        total += targets.size(0)
    return correct, total, total_loss


def training_step(model: CustomCLIPCoOp, dataset, optimizer, batch_size, classnames, device="cuda"):
    """
    Performs one full training epoch for the CoOp model.

    Args:
        model (CustomCLIPCoOp): The model to train.
        dataset (Dataset): Dataset to train on.
        optimizer (Optimizer): Optimizer used for updating model parameters.
        batch_size (int): Batch size for training.
        device (str): Computation device.

    Returns:
        Tuple[float, float]: Average training loss and accuracy.
    """
    samples = 0.0
    cumulative_loss = 0.0
    cumulative_accuracy = 0.0

    # Set the network to training mode

    model.train()

    tmp_dataset = ContiguousLabelDataset(dataset, classnames)
    dataloader = DataLoader(tmp_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    pbar = tqdm(dataloader, desc="Training", position=1, leave=False)
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        # Load data into GPU
        inputs = inputs.to(device)
        targets = targets.to(device)
        # debug if inputs and targets are taken correctly by the dataloader
        #print(inputs.shape)
        #print(targets.shape)
        # Forward pass + loss computation
        loss, logits = model(inputs, targets)

        if torch.isnan(loss):
            print("⚠️ NaN loss encountered!")
            #print("Logits:", logits)
            print("Targets:", targets)

        # Backward pass
        loss.backward()

        # Parameters update
        optimizer.step()

        # Gradients reset
        optimizer.zero_grad()

        # Fetch prediction and loss value
        samples += inputs.shape[0]
        cumulative_loss += loss.item() * inputs.shape[0]
        _, predicted = logits.max(dim=1)  # max() returns (maximum_value, index_of_maximum_value)

        # Compute training accuracy
        cumulative_accuracy += predicted.eq(targets).sum().item()

        pbar.set_postfix(train_loss=loss.item(), train_acc=cumulative_accuracy / samples )
        pbar.update(1)

    return cumulative_loss / samples, cumulative_accuracy / samples


@torch.no_grad()
def test_step(model, dataset, batch_size, device, categories, label="test", base=False):
    """
    Evaluates the model using either fine-tuned or base (zero-shot) strategy.

    Args:
        model (nn.Module): The model to test.
        dataset (Dataset): Dataset for testing.
        batch_size (int): Batch size for testing.
        device (str): Device used for computation.
        label (str): Label for the progress bar.
        base (bool): Whether to use zero-shot CLIP instead of the fine-tuned model.

    Returns:
        float: Accuracy score.
    """
    if not base:
        return finetuned_test_step(model, dataset, batch_size, device, categories, label)
    else:
        return base_test_step(model, dataset, categories, batch_size, device, label)


@torch.no_grad()
def finetuned_test_step(model: CustomCLIPCoOp, dataset, batch_size, device, categories, label="test"):
    """
    Evaluates a fine-tuned CustomCLIPCoOp model on the given dataset.

    Args:
        model (CustomCLIPCoOp): Fine-tuned CoOp model.
        dataset (Dataset): Dataset for evaluation.
        batch_size (int): Batch size.
        device (str): Computation device.
        label (str): Label for progress display.

    Returns:
        float: Accuracy of the model on the dataset.
    """
    model.eval()

    tmp_dataset = ContiguousLabelDataset(dataset, categories)
    dataloader = DataLoader(tmp_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    correct = 0
    total = 0

    for images, targets in tqdm(dataloader, desc=label):
        images = images.to(device)
        targets = targets.to(device)

        logits = model(images)
        predictions = logits.argmax(dim=-1)

        correct += (predictions == targets).sum().item()
        total += targets.size(0)

    accuracy = correct / total
    return accuracy


@torch.no_grad()  # we don't want gradients
def base_test_step(model: CLIP, dataset, categories, batch_size, device, label=""):
    """
    Evaluates a zero-shot CLIP model using cosine similarity between image and text embeddings.

    Args:
        model (CLIP): Pretrained CLIP model.
        dataset (Dataset): Dataset to evaluate.
        categories (List[int]): List of category indices to evaluate.
        batch_size (int): Batch size for evaluation.
        device (str): Computation device.
        label (str): Optional label for progress bar.

    Returns:
        float: Accuracy of zero-shot CLIP classification.
    """
    # let's set the model in evaluation mode
    model.eval()

    # Remap labels into a contiguous set starting from zero
    contig_cat2idx = {cat: idx for idx, cat in enumerate(categories)}

    # here we apply the standard CLIP template used for oxford flowers to all categories
    # and immediately tokenize each sentence (convert natural language into numbers - feel free to print the text input to inspect them)
    text_inputs = clip.tokenize(
        [f"a photo of a {CLASS_NAMES[c]}, a type of flower." for c in categories]
    ).to(device)

    # we can encode the text features once as they are shared for all images
    # therefore we do it outside the evaluation loop
    text_features = model.encode_text(text_inputs)
    # and here we normalize them (standard pratice with CLIP)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # simple dataloader creation
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # here we store the number of correct predictions we will make
    correct_predictions = 0
    for image, target in tqdm(dataloader, desc=label):
        # base categories range from 0 to 50, while novel ones from 51 to 101
        # therefore we must map categories to the [0, 50], otherwise we will have wrong predictions
        # Map targets in contiguous set starting from zero
        # Labels needs to be .long() in pytorch
        target = torch.Tensor([contig_cat2idx[t.item()] for t in target]).long()

        image = image.to(device)
        target = target.to(device)

        # forward image through CLIP image encoder
        image_features = model.encode_image(image)
        # and normalize
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # here cosine similarity between image and text features and keep the argmax for every row (every image)
        predicted_class = (image_features @ text_features.T).argmax(dim=-1)
        # now we check which are correct, and sum them (False == 0, True == 1)
        correct_predictions += (predicted_class == target).sum().item()

    # and now we compute the accuracy
    accuracy = correct_predictions / len(dataset)
    return accuracy