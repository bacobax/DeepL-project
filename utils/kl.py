import clip
import torch

from utils.datasets import CLASS_NAMES


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
