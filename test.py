import os
import math

import torch
from easydict import EasyDict
from tqdm import tqdm

from model.cocoop.custom_clip import CustomCLIP
from model.cocoop.mlp_adversary import GradientReversalLayer, AdversarialMLP
from utils.datasets import get_data, base_novel_categories, split_data, CLASS_NAMES
from utils.training_cocoop import (
    test_step,
    training_step,
    eval_step,
    training_step_v2,
    adversarial_training_step,
)
import clip
from torch.utils.tensorboard import SummaryWriter
from torch.optim import SGD, Adam, AdamW
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

if __name__ == "__main__":
    clip_model, preprocess = clip.load("ViT-B/32")
    resolution = clip_model.visual.input_resolution

    cfg = EasyDict()
    # Training configuration
    cfg.TRAINER = EasyDict()
    cfg.TRAINER.COCOOP = EasyDict()
    cfg.TRAINER.COCOOP.CTX_LOAD = "./bin/coop/exp1_ctx_only.pth"
    cfg.TRAINER.COCOOP.N_CTX = 4  # Number of context tokens
    cfg.TRAINER.COCOOP.CTX_INIT = (
        ""
    )  # Leave empty for random initialization
    cfg.TRAINER.COCOOP.PREC = "fp16"  # Precision for meta network
    cfg.INPUT = EasyDict()
    cfg.INPUT.SIZE = [
        resolution,
        resolution,
    ]  # Must match CLIP model's input resolution
    train_set, val_set, test_set = get_data(transform=preprocess)

    # split classes into base and novel
    base_classes, novel_classes = base_novel_categories(train_set)

    # split the three datasets
    train_base, _ = split_data(train_set, base_classes)
    val_base, val_novel = split_data(val_set, base_classes)
    test_base, test_novel = split_data(test_set, base_classes)
    # Instantiate the network and move it to the chosen device (GPU)
    my_model = CustomCLIP(
        classnames=[CLASS_NAMES[idx] for idx in base_classes],
        cfg=cfg,
        clip_model=clip_model,
    ).to("cuda")

    path = os.path.join("./runs/CoCoOp", "adv_training_run_20250520_150011", "best_model.pth")
    my_model.load_state_dict(
        torch.load(path)
    )
    my_model.eval()
    print(f"Model loaded from {path}")

    novel_accuracy = test_step(
        my_model,
        test_novel,
        novel_classes,
        10,
        "cuda",
        label="test",
        base=False,
    )

    print(f"Novel accuracy: {novel_accuracy}")
