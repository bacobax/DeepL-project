import clip
import torch

from utils.datasets import CLASS_NAMES


def get_kl_loss(device, inputs_novel, model, targets_novel, tmp_dataset):
    """
    Computes the KL divergence between the student model's predictions and the CLIP model's predictions
    for a batch of novel class images.

    Args:
        device (torch.device): The device (CPU or CUDA) to perform computation on.
        inputs_novel (Tensor): A batch of input images from novel classes.
        model (nn.Module): The student model that includes a CLIP backbone and prompt learner.
        targets_novel (List[int]): Target labels corresponding to the novel class inputs.
        tmp_dataset (ContiguousLabelDataset): Dataset wrapper with label-to-category mappings.

    Returns:
        Tensor: A scalar tensor representing the KL divergence loss.
    """
    targets_novel_tensor = torch.tensor(targets_novel).to(device) if isinstance(targets_novel, list) else targets_novel
    categories_novel_tensor = [tmp_dataset.idx2cat[c.item()] for c in list(set(targets_novel_tensor))]
    # print(f"input novel shape: {inputs_novel.shape} novel base: {targets_novel_tensor.shape}")
    with torch.no_grad():
        image_features_clip = model.clip_model.encode_image(inputs_novel)
        image_features_clip = image_features_clip / image_features_clip.norm(dim=-1, keepdim=True)


        text_inputs = clip.tokenize(
            [f"a photo of a {CLASS_NAMES[c]}, a type of flower." for c in categories_novel_tensor]
        ).to(device)

        text_features_clip = model.clip_model.encode_text(text_inputs)
        text_features_clip = text_features_clip / text_features_clip.norm(dim=-1, keepdim=True)

        clip_logits = image_features_clip @ text_features_clip.T

    remapped_class_names = [ CLASS_NAMES[ c ] for c in categories_novel_tensor ]
    
    # targets_novel_tensor contains contiguous indices (0, 1, 2, ...)
    # categories_novel_tensor should be the original class labels for the current batch
    
    # No need to remap targets; they are already correct
    target_remapping = {cat:idx for idx, cat in enumerate(categories_novel_tensor)}
    target_original = [tmp_dataset.idx2cat[c.item()] for c in targets_novel_tensor]
    target_remapped = torch.tensor([target_remapping[c] for c in target_original]).to(device)
    
    with model.temporary_classnames(remapped_class_names):
        model.train()
        student_logits, student_loss = model(inputs_novel, target_remapped, met_net_2=False)  # [B, num_classes]
        kl_loss = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(student_logits, dim=-1),
            torch.nn.functional.softmax(clip_logits, dim=-1),
            reduction="batchmean"
        )
    return kl_loss
