from clip.model import CLIP
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from tqdm import tqdm
import clip
from model.cocoop.custom_clip import CustomCLIP
from utils.datasets import ContiguousLabelDataset, CLASS_NAMES
from utils.metrics import AverageMeter
def check_gradients(model):
    print("=== Checking gradients ===")
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                print(f"{name}: ❌ No gradient")
            elif torch.all(param.grad == 0):
                print(f"{name}: ⚠️ Zero gradient")
            else:
                print(f"{name}: ✅ Gradient flowing")

@torch.no_grad()
def eval_step(model, dataset, cost_function, batch_size=32, device="cuda", new_classnames=None, desc_add=""):
    model.eval()

    tmp_dataset = ContiguousLabelDataset(dataset)
    dataloader = DataLoader(tmp_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()
    if new_classnames is not None:
        new_classnames = [CLASS_NAMES[c] for c in new_classnames]
        with model.temporary_classnames(new_classnames):
            walk_the_dataset(loss_meter, accuracy_meter, dataloader, device, model, desc_add)

    else:
        walk_the_dataset(loss_meter, accuracy_meter, dataloader, device, model, desc_add)

    return loss_meter.avg, accuracy_meter.avg


def walk_the_dataset(loss_meter, accuracy_meter, dataloader, device, model, desc_add=""):
    for images, targets in tqdm(dataloader, desc="Validation"+desc_add, position=1, leave=False):
        images = images.to(device)
        targets = targets.to(device)

        logits = model(images)
        loss = F.cross_entropy(logits, targets)

        predictions = logits.argmax(dim=-1)
        correct = (predictions == targets).sum().item()
        batch_size = targets.size(0)

        loss_meter.update(loss.item(), n=batch_size)
        accuracy_meter.update(correct / batch_size)


@torch.no_grad()
def test_step(model, dataset, new_classnames, batch_size, device, label="test", base=False):
    if not base:
        return finetuned_test_step(model, dataset, new_classnames, batch_size, device, label)
    else:
        return base_test_step(model, dataset, new_classnames, batch_size, device, label)


@torch.no_grad()
def finetuned_test_step(model: CustomCLIP, dataset, new_classnames, batch_size, device, label="test"):
    model.eval()

    tmp_dataset = ContiguousLabelDataset(dataset)
    dataloader = DataLoader(tmp_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    new_classnames = [CLASS_NAMES[c] for c in new_classnames]
    with model.temporary_classnames(new_classnames):
        accuracy_meter = AverageMeter()

        for images, targets in tqdm(dataloader, desc=label):
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            predictions = logits.argmax(dim=-1)

            correct = (predictions == targets).sum().item()
            accuracy_meter.update(correct, n=targets.size(0), raw=True)

    return accuracy_meter.avg


@torch.no_grad()  # we don't want gradients
def base_test_step(model: CLIP, dataset, categories, batch_size, device, label=""):
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
    accuracy_meter = AverageMeter()
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

        correct = (predicted_class == target).sum().item()
        accuracy_meter.update(correct, n=target.size(0), raw=True)

    # and now we compute the accuracy
    return accuracy_meter.avg
