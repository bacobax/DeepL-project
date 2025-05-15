from clip.model import CLIP
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import clip
import random
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

def training_step_v2(model, dataset, optimizer, batch_size, lambda_kl, device="cuda"):
    samples = 0.0
    kl_samples = 0.0
    ce_samples = 0.0
    cumulative_loss = 0.0
    cumulative_accuracy = 0.0
    cumulative_ce_loss = 0.0
    cumulative_kl_loss = 0.0

    model.train()
    tmp_dataset = ContiguousLabelDataset(dataset)

    def custom_collate(batch):
        base_samples = []
        novel_samples = []
        targets_in_batch = list(set([target for _, target in batch]))
        random.shuffle(targets_in_batch)
        split_idx = int(0.7 * len(targets_in_batch))
        pseudo_base_ids = targets_in_batch[:split_idx]
        pseudo_novel_ids = targets_in_batch[split_idx:]
        for img, label in batch:
            if label in pseudo_base_ids:
                base_samples.append((img, label))
            elif label in pseudo_novel_ids:
                novel_samples.append((img, label))
        return base_samples, novel_samples

    dataloader = DataLoader(tmp_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=custom_collate)
    pbar = tqdm(dataloader, desc="Training", position=1, leave=False)
    for base_batch, novel_batch in dataloader:
        if not base_batch or not novel_batch:
            continue  # skip incomplete batch

        # === Pseudo-base: cross-entropy ===
        inputs_base = torch.stack([img for img, _ in base_batch]).to(device)
        targets_base = torch.tensor([lbl for _, lbl in base_batch]).to(device)
        logits_base, loss_ce = model(inputs_base, targets_base)

        # === Pseudo-novel: KL divergence with frozen CLIP ===
        model.eval()  # needed to disable dropout etc.
        inputs_novel = torch.stack([img for img, _ in novel_batch]).to(device)
        targets_novel = [lbl for _, lbl in novel_batch]

        with torch.no_grad():
            image_features_clip = model.clip_model.encode_image(inputs_novel)
            image_features_clip = image_features_clip / image_features_clip.norm(dim=-1, keepdim=True)

            category_idxs = [tmp_dataset.idx2cat[c] for c in targets_novel]

            text_inputs = clip.tokenize(
                [f"a photo of a {CLASS_NAMES[c]}, a type of flower." for c in category_idxs]
            ).to(device)

            text_features_clip = model.clip_model.encode_text(text_inputs)
            text_features_clip = text_features_clip / text_features_clip.norm(dim=-1, keepdim=True)

            clip_logits = image_features_clip @ text_features_clip.T

        model.train()
        student_logits = model(inputs_novel)  # [B, num_classes]

        kl_loss = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(student_logits, dim=-1),
            torch.nn.functional.softmax(clip_logits, dim=-1),
            reduction="batchmean"
        )

        # === Combine losses ===
        total_loss = loss_ce + lambda_kl * kl_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        batch_size_total = inputs_base.size(0) + inputs_novel.size(0)
        cumulative_loss += total_loss.item() * batch_size_total
        cumulative_ce_loss += loss_ce.item() * inputs_base.size(0)
        cumulative_kl_loss += kl_loss.item() * inputs_novel.size(0)

        samples += batch_size_total
        ce_samples += inputs_base.size(0)
        kl_samples += inputs_novel.size(0)

        _, predicted = logits_base.max(dim=1)
        cumulative_accuracy += predicted.eq(targets_base).sum().item()

        pbar.set_postfix(total_loss=total_loss.item(), train_acc=cumulative_accuracy/samples, loss_ce=loss_ce.item(), kl_loss=kl_loss.item())
        pbar.update(1)

    return (
        cumulative_loss / samples,
        cumulative_accuracy / ce_samples,
        cumulative_ce_loss / ce_samples,
        cumulative_kl_loss / kl_samples,
    )


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