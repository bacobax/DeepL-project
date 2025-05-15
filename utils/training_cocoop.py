from clip.model import CLIP
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import clip
import torch.nn.functional as F
from model.cocoop.custom_clip import CustomCLIP
from utils.datasets import ContiguousLabelDataset, CLASS_NAMES


@torch.no_grad()
def eval_step(model, dataset, cost_function, batch_size=32, device="cuda", new_classnames=None):
    model.eval()

    tmp_dataset = ContiguousLabelDataset(dataset)
    dataloader = DataLoader(tmp_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

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
    for images, targets in tqdm(dataloader, desc="Validation", position=1, leave=False):
        images = images.to(device)
        targets = targets.to(device)

        logits = model(images)
        loss = cost_function(logits, targets)

        total_loss += loss.item() * targets.size(0)
        predictions = logits.argmax(dim=-1)
        correct += (predictions == targets).sum().item()
        total += targets.size(0)
    return correct, total, total_loss


def training_step(model, dataset, optimizer, batch_size, device="cuda"):
    samples = 0.0
    cumulative_loss = 0.0
    cumulative_accuracy = 0.0

    # Set the network to training mode

    model.train()

    tmp_dataset = ContiguousLabelDataset(dataset)
    dataloader = DataLoader(tmp_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    pbar = tqdm(dataloader, desc="Training", position=1, leave=False)
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        # Load data into GPU
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass + loss computation
        logits, loss = model(inputs, targets)

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


def compute_kl_loss(model, clip_model, dataset, pseudo_novel_class_ids, batch_size, device="cuda"):
    model.eval()
    clip_model.eval()

    tmp_dataset = ContiguousLabelDataset(dataset)
    dataloader = DataLoader(tmp_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    kl_losses = []
    total_samples = 0

    for images, _ in tqdm(dataloader, desc="KL Loss Accumulation", position=1, leave=False):
        images = images.to(device)

        # --- CLIP teacher prediction ---
        with torch.no_grad():
            image_features_clip = clip_model.encode_image(images)
            image_features_clip /= image_features_clip.norm(dim=-1, keepdim=True)

            text_inputs = clip.tokenize([
                f"a photo of a {CLASS_NAMES[c]}, a type of flower." for c in pseudo_novel_class_ids
            ]).to(device)

            text_features_clip = clip_model.encode_text(text_inputs)
            text_features_clip /= text_features_clip.norm(dim=-1, keepdim=True)

            clip_logits = image_features_clip @ text_features_clip.T  # [B, |pseudo_novel|]

        # --- Student model prediction ---
        student_logits = model(images)  # [B, |all_classes|]

        # Filter student logits to pseudo_novel subset
        student_logits = student_logits[:, pseudo_novel_class_ids]

        # --- KL Divergence ---
        kl_loss = F.kl_div(
            F.log_softmax(student_logits, dim=-1),
            F.softmax(clip_logits, dim=-1),
            reduction="batchmean"
        )
        kl_losses.append(kl_loss * images.size(0))
        total_samples += images.size(0)

    total_kl = torch.stack(kl_losses).sum() / total_samples
    return total_kl


def compute_ce_loss(model, dataset, batch_size, device="cuda"):
    model.train()

    ce_loss_total = 0.0
    total_samples = 0
    correct = 0

    tmp_dataset = ContiguousLabelDataset(dataset)
    dataloader = DataLoader(tmp_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    all_ce_losses = []

    for inputs, targets in tqdm(dataloader, desc="CE Loss Accumulation", position=1, leave=False):
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits, ce_loss = model(inputs, targets)  # ce_loss is a tensor

        ce_loss.backward()

        # Accumulate loss tensor (weighted by batch size)
        all_ce_losses.append(ce_loss * inputs.size(0))

        # Accuracy tracking (optional)
        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total_samples += inputs.size(0)

    # Mean CE loss (preserving gradient graph)
    ce_loss_total = torch.stack(all_ce_losses).sum() / total_samples
    accuracy = correct / total_samples
    return ce_loss_total, accuracy

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