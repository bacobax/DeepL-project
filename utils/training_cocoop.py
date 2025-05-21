from clip.model import CLIP
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from tqdm import tqdm
import clip
import random
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


def adversarial_training_step(
        model,
        dataset,
        optimizer,
        batch_size,
        cls_cluster_dict,
        lambda_adv,
        mlp_adversary,
        grl,
        device="cuda"
):
    total_loss_metric = AverageMeter()
    ce_loss_metric = AverageMeter()
    adv_loss_metric = AverageMeter()
    accuracy_metric = AverageMeter()

    model.train()
    tmp_dataset = ContiguousLabelDataset(dataset)

    dataloader = DataLoader(tmp_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    pbar = tqdm(dataloader, desc="Training", position=1, leave=False)
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        # Load data into GPU
        inputs = inputs.to(device)
        targets = targets.to(device)

        targets_real_category = [tmp_dataset.idx2cat[c.item()] for c in targets]
        cluster_target = [int(cls_cluster_dict[int(tl)]) for tl in targets_real_category]
        cluster_target = torch.tensor(
            cluster_target,
            device=targets.device,
            dtype=torch.float16
        )

        # Forward pass + loss computation
        logits, ce_loss = model(inputs, targets)

        # === Adversarial loss ===
        reversed_logits = grl(logits)
        cluster_logits = mlp_adversary(reversed_logits).squeeze()
        loss_bce = F.binary_cross_entropy(cluster_logits, cluster_target)

        # === Combine losses ===
        total_loss = ce_loss + lambda_adv * loss_bce

        # Backward pass
        total_loss.backward()
        """if batch_idx < 3:
            check_gradients(model)
            check_gradients(mlp_adversary)"""

        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(mlp_adversary.parameters()),
            max_norm=1.0,
            norm_type=2.0,
            error_if_nonfinite=True
        )

        # Parameters update
        optimizer.step()

        # Gradients reset
        optimizer.zero_grad()

        if batch_idx < 3:  # Log only a few batches for performance
            print(f"[Batch {batch_idx}] CE Loss: {ce_loss.item():.4f} | "
                  f"Adv Loss: {loss_bce.item():.4f} | "
                  f"Total Loss: {total_loss.item():.4f} | "
                  f"lambda_adv: {lambda_adv:.4f}")

        # Fetch prediction and loss value
        batch_size = inputs.shape[0]
        total_loss_metric.update(total_loss.item(), n=batch_size)
        ce_loss_metric.update(ce_loss.item(), n=batch_size)
        adv_loss_metric.update(loss_bce.item(), n=batch_size)

        _, predicted = logits.max(dim=1)  # max() returns (maximum_value, index_of_maximum_value)

        # Compute training accuracy
        correct = predicted.eq(targets).sum().item()
        accuracy_metric.update(correct, n=batch_size, raw=True)

        pbar.set_postfix(
            total_train_loss=total_loss_metric.avg,
            ce_loss=ce_loss_metric.avg,
            adv_loss=adv_loss_metric.avg,
            train_acc=accuracy_metric.avg
        )
        pbar.update(1)

    return total_loss_metric.avg, accuracy_metric.avg, ce_loss_metric.avg, adv_loss_metric.avg,


def adversarial_kl_training_step(
        model,
        dataset,
        optimizer,
        batch_size,
        lambda_kl,
        cls_cluster_dict,
        lambda_adv,
        mlp_adversary,
        grl,
        device="cuda"
):

    total_loss_meter = AverageMeter()
    ce_loss_meter = AverageMeter()
    kl_loss_meter = AverageMeter()
    adv_loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()

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

    dataloader = DataLoader(tmp_dataset, batch_size=batch_size, shuffle=True, num_workers=1, collate_fn=custom_collate)
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

        kl_loss = get_kl_loss(device, inputs_novel, model, targets_novel, tmp_dataset)

        reversed_logits = grl(logits_base)
        cluster_logits = mlp_adversary(reversed_logits).squeeze()

        true_labels = [tmp_dataset.idx2cat[c.item()] for c in targets_base]
        cluster_labels = [cls_cluster_dict[int(tl)] for tl in true_labels]

        cluster_labels = torch.tensor(
            cluster_labels,
            device=targets_base.device,
            dtype=torch.float16
        )
        loss_bce = F.binary_cross_entropy(cluster_logits, cluster_labels)

        # === Combine losses ===
        total_loss = loss_ce + lambda_kl * kl_loss + lambda_adv * loss_bce

        optimizer.zero_grad()
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(mlp_adversary.parameters()),
            max_norm=1.0
        )

        optimizer.step()

        batch_size_total = inputs_base.size(0) + inputs_novel.size(0)

        total_loss_meter.update(total_loss.item(), n=batch_size_total)
        ce_loss_meter.update(loss_ce.item(), n=inputs_base.size(0))
        kl_loss_meter.update(kl_loss.item(), n=inputs_novel.size(0))
        adv_loss_meter.update(loss_bce.item(), n=inputs_base.size(0))

        _, predicted = logits_base.max(dim=1)
        correct = (predicted == targets_base).sum().item()
        total = targets_base.size(0)
        accuracy_meter.update(correct, n=total, raw=True)

        pbar.set_postfix(total_loss=total_loss.item(), train_acc=accuracy_meter.avg, loss_ce=loss_ce.item(), kl_loss=kl_loss.item())
        pbar.update(1)

    return (
        total_loss_meter.avg,
        accuracy_meter.avg,
        ce_loss_meter.avg,
        kl_loss_meter.avg,
        adv_loss_meter.avg,
    )


def get_kl_loss(device, inputs_novel, model, targets_novel, tmp_dataset):
    targets_novel_tensor = torch.tensor(targets_novel).to(device)
    categories_novel_tensor = [tmp_dataset.idx2cat[c] for c in list(set(targets_novel))]
    # print(f"input novel shape: {inputs_novel.shape} novel base: {targets_novel_tensor.shape}")
    with torch.no_grad():
        image_features_clip = model.clip_model.encode_image(inputs_novel)
        image_features_clip = image_features_clip / image_features_clip.norm(dim=-1, keepdim=True)

        category_idxs = [tmp_dataset.idx2cat[c] for c in list(set(targets_novel))]

        text_inputs = clip.tokenize(
            [f"a photo of a {CLASS_NAMES[c]}, a type of flower." for c in category_idxs]
        ).to(device)

        text_features_clip = model.clip_model.encode_text(text_inputs)
        text_features_clip = text_features_clip / text_features_clip.norm(dim=-1, keepdim=True)

        clip_logits = image_features_clip @ text_features_clip.T
    model.train()
    student_logits, student_loss = model(inputs_novel, targets_novel_tensor)  # [B, num_classes]
    student_logits_tmp = []
    for img_logits in student_logits:
        student_logits_tmp.append(
            [logit.item() for column_idx, logit in enumerate(img_logits) if column_idx in categories_novel_tensor])
    student_logits = torch.tensor(student_logits_tmp).to(device)
    # print(f"student logits shape: {student_logits.shape}, clip logits shape: {clip_logits.shape}")
    kl_loss = torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(student_logits, dim=-1),
        torch.nn.functional.softmax(clip_logits, dim=-1),
        reduction="batchmean"
    )
    return kl_loss


def training_step_v2(model, dataset, optimizer, batch_size, lambda_kl, device="cuda"):

    cumulative_loss = AverageMeter()
    cumulative_ce_loss = AverageMeter()
    cumulative_kl_loss = AverageMeter()
    cumulative_accuracy = AverageMeter()

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

    dataloader = DataLoader(tmp_dataset, batch_size=batch_size, shuffle=True, num_workers=1, collate_fn=custom_collate)
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
        targets_novel_tensor = torch.tensor(targets_novel).to(device)

        categories_novel_tensor = [tmp_dataset.idx2cat[c] for c in list(set(targets_novel))]

        #print(f"input novel shape: {inputs_novel.shape} novel base: {targets_novel_tensor.shape}")
        with torch.no_grad():
            image_features_clip = model.clip_model.encode_image(inputs_novel)
            image_features_clip = image_features_clip / image_features_clip.norm(dim=-1, keepdim=True)

            category_idxs = [tmp_dataset.idx2cat[c] for c in list(set(targets_novel))]

            text_inputs = clip.tokenize(
                [f"a photo of a {CLASS_NAMES[c]}, a type of flower." for c in category_idxs]
            ).to(device)

            text_features_clip = model.clip_model.encode_text(text_inputs)
            text_features_clip = text_features_clip / text_features_clip.norm(dim=-1, keepdim=True)

            clip_logits = image_features_clip @ text_features_clip.T

        model.train()
        student_logits, student_loss = model(inputs_novel, targets_novel_tensor)  # [B, num_classes]
        student_logits_tmp = []
        for img_logits in student_logits:
            student_logits_tmp.append([logit.item() for column_idx, logit in enumerate(img_logits) if column_idx in categories_novel_tensor])

        student_logits = torch.tensor(student_logits_tmp).to(device)

        #print(f"student logits shape: {student_logits.shape}, clip logits shape: {clip_logits.shape}")

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

        cumulative_loss.update(total_loss.item(), n=batch_size_total)
        cumulative_ce_loss.update(loss_ce.item(), n=inputs_base.size(0))
        cumulative_kl_loss.update(kl_loss.item(), n=inputs_novel.size(0))

        _, predicted = logits_base.max(dim=1)
        correct = (predicted == targets_base).sum().item()
        total = targets_base.size(0)

        cumulative_accuracy.update(correct, n=total, raw=True)

        pbar.set_postfix(total_loss=total_loss.item(), train_acc=cumulative_accuracy.avg, loss_ce=loss_ce.item(), kl_loss=kl_loss.item())
        pbar.update(1)

    return (
        cumulative_loss.avg,
        cumulative_accuracy.avg,
        cumulative_ce_loss.avg,
        cumulative_kl_loss.avg,
    )


def training_step(model, dataset, optimizer, batch_size, device="cuda"):
    # Use AverageMeter for loss and accuracy
    cumulative_loss = AverageMeter()
    cumulative_accuracy = AverageMeter()

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
        batch_size = inputs.shape[0]
        cumulative_loss.update(loss.item(), n=batch_size)
        _, predicted = logits.max(dim=1)  # max() returns (maximum_value, index_of_maximum_value)

        # Compute training accuracy
        correct = predicted.eq(targets).sum().item()
        cumulative_accuracy.update(correct, n=batch_size, raw=True)

        pbar.set_postfix(train_loss=loss.item(), train_acc=cumulative_accuracy.avg)
        pbar.update(1)

    return cumulative_loss.avg, cumulative_accuracy.avg


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
