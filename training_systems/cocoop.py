import os
import math
import random

import torch
from easydict import EasyDict
from tqdm import tqdm

from model.cocoop.custom_clip import CustomCLIP
from utils.datasets import get_data, base_novel_categories, split_data, CLASS_NAMES
from utils.training_cocoop import test_step, training_step, eval_step, compute_ce_loss, compute_kl_loss
import clip
from torch.utils.tensorboard import SummaryWriter
from torch.optim import SGD, Adam, AdamW
from torch import nn
from torch.optim.lr_scheduler import LambdaLR


class CoCoOpSystem:
    def __init__(self,
                 batch_size=16,
                 device="cuda:0",
                 learning_rate=0.002,
                 weight_decay=0.0005,
                 momentum=0.9,
                 epochs=2,
                 run_name="exp1",
                 n_ctx=4,
                 ctx_init="",
                 class_token_position="end",
                 csc=False,
                 lambda_kl=0.5,
                 ):
        self.batch_size = batch_size
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.epochs = epochs
        self.run_name = run_name
        self.n_ctx = n_ctx
        self.ctx_init = ctx_init
        self.class_token_position = class_token_position
        self.csc = csc
        self.lambda_kl = lambda_kl
        self.max_epoch = self.epochs
        self.lr_scheduler_type = "cosine"
        self.warmup_epoch = 1
        self.warmup_type = "constant"
        self.warmup_cons_lr = 1e-5

        # Create a logger for the experiment
        self.writer = SummaryWriter(log_dir=f"runs/CoCoOp/{run_name}")
        self.writer.add_hparams({
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "momentum": self.momentum,
            "epochs": self.epochs,
            "n_ctx": self.n_ctx,
            "ctx_init": self.ctx_init,
            "class_token_position": self.class_token_position,
            "csc": self.csc,
            "max_epoch": self.max_epoch,
            "lr_scheduler_type": self.lr_scheduler_type,
            "warmup_epoch": self.warmup_epoch,
            "warmup_type": self.warmup_type,
            "warmup_cons_lr": self.warmup_cons_lr,
            "lambda_kl": self.lambda_kl,
        }, {})

        # Get dataloaders

        self.clip_model, preprocess = clip.load("ViT-B/32")
        self.train_set, self.val_set, self.test_set = get_data(transform=preprocess)

        # split classes into base and novel
        self.base_classes, self.novel_classes = base_novel_categories(self.train_set)

        # split the three datasets
        self.train_base, _ = split_data(self.train_set, self.base_classes)
        self.val_base, self.val_novel = split_data(self.val_set, self.base_classes)
        self.test_base, self.test_novel = split_data(self.test_set, self.base_classes)

        #self.classnames, _ = embed_dataset_classnames(dataset_name, preprocess=preprocess, model=clip_model)

        resolution = self.clip_model.visual.input_resolution

        cfg = EasyDict()
        # Training configuration
        cfg.TRAINER = EasyDict()
        cfg.TRAINER.COCOOP = EasyDict()
        cfg.TRAINER.COCOOP.CTX_LOAD = "./bin/coop/exp1_ctx_only.pth"
        cfg.TRAINER.COCOOP.N_CTX = self.n_ctx  # Number of context tokens
        cfg.TRAINER.COCOOP.CTX_INIT = self.ctx_init  # Leave empty for random initialization
        cfg.TRAINER.COCOOP.PREC = "fp16"  # Precision for meta network
        cfg.INPUT = EasyDict()
        cfg.INPUT.SIZE = [resolution, resolution]  # Must match CLIP model's input resolution

        # Instantiate the network and move it to the chosen device (GPU)
        self.model = CustomCLIP(
            classnames=[CLASS_NAMES[idx] for idx in self.base_classes],
            cfg=cfg,
            clip_model=self.clip_model,
        ).to(device)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Total trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

        self.optimizer = self.get_optimizer(self.learning_rate, self.weight_decay, self.momentum)
        self.cost_function = nn.CrossEntropyLoss()

    def train(self):
        def lr_lambda(current_epoch):
            if current_epoch < self.warmup_epoch:
                return self.warmup_cons_lr / self.learning_rate
            else:
                cosine_decay = 0.5 * (1 + math.cos(math.pi * (current_epoch - self.warmup_epoch) / (self.max_epoch - self.warmup_epoch)))
                return cosine_decay

        self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda)

        print("Before training:")
        self.compute_evaluation(-1, base=True)
        print("Training the model...")
        print_epoch_interval = 2

        best_novel_accuracy = 0.0
        patience_counter = 0
        patience = 3# adjustable
        best_model_path = os.path.join("runs/CoCoOp", self.run_name, "best_model.pth")
        c = 0
        pbar = tqdm(total=self.max_epoch, desc="OVERALL TRAINING", position=0, leave=True)
        for e in range(self.max_epoch):
            # --- Split base classes into pseudo-base and pseudo-novel ---
            base_class_ids = list(self.base_classes)
            random.shuffle(base_class_ids)
            split_idx = int(0.7 * len(base_class_ids))
            pseudo_base_ids = base_class_ids[:split_idx]
            pseudo_novel_ids = base_class_ids[split_idx:]

            # --- Generate datasets for each split ---
            train_pseudo_base, _ = split_data(self.train_set, pseudo_base_ids)
            train_pseudo_novel, _ = split_data(self.train_set, pseudo_novel_ids)

            # --- Compute CE loss on pseudo-base (with gradient) ---
            ce_loss_total, base_train_accuracy = compute_ce_loss(
                model=self.model,
                dataset=train_pseudo_base,
                batch_size=self.batch_size,
                device=self.device
            )

            # --- Compute KL loss on pseudo-novel (no gradient yet) ---
            kl_loss_total = compute_kl_loss(
                model=self.model,
                clip_model=self.clip_model,
                dataset=train_pseudo_novel,
                pseudo_novel_class_ids=pseudo_novel_ids,
                batch_size=self.batch_size,
                device=self.device
            )

            # --- Total loss and optimization ---
            total_loss = ce_loss_total + self.lambda_kl * kl_loss_total  # lambda_kl = 0.5
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            if e % print_epoch_interval == 0:
                base_val_loss, base_val_accuracy = eval_step(
                    model=self.model,
                    dataset=self.val_base,
                    cost_function=self.cost_function,
                    device=self.device,
                    batch_size=self.batch_size,
                )
                novel_val_loss, novel_val_accuracy = eval_step(
                    model=self.model,
                    dataset=self.val_novel,
                    cost_function=self.cost_function,
                    device=self.device,
                    batch_size=self.batch_size,
                    new_classnames=self.novel_classes,
                )

                self.log_values(e, ce_loss_total.item(), base_train_accuracy, "train_base")
                self.log_values(e, base_val_loss, base_val_accuracy, "validation_base")
                self.log_values(e, novel_val_loss, novel_val_accuracy, "validation_novel")

                pbar.set_postfix(
                    train_acc=base_train_accuracy,
                    val_acc=base_val_accuracy,
                    total_train_acc=total_loss.item()
                )

                if novel_val_accuracy > best_novel_accuracy:
                    best_novel_accuracy = novel_val_accuracy
                    patience_counter = 0
                    torch.save(self.model.state_dict(), best_model_path)
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Stopping early at epoch {e} due to no improvement in novel accuracy.")
                        break

            pbar.update(1)
            self.lr_scheduler.step()
            c += 1

        print("After training:")
        self.model.load_state_dict(torch.load(best_model_path))  # Load best model
        self.compute_evaluation(c)
        self.writer.close()
        self.save_model()

    def save_model(self, path="./bin/cocoop"):
        #create folder if not exist
        if not os.path.exists(path):
            os.makedirs(path)
        # Save the model
        torch.save(self.model.state_dict(), os.path.join(path, f"{self.run_name}.pth"))

    def load_model(self, path="./bin"):
        # Load the model
        self.model.load_state_dict(torch.load(os.path.join(path, f"{self.run_name}.pth")))
        self.model.eval()
        print(f"Model loaded from {path}")

    def compute_evaluation(self, epoch_idx, base=False):
        base_model = self.model if not base else self.clip_model
        base_accuracy = test_step(base_model, self.test_base, self.base_classes, self.batch_size, self.device, label="test", base=base)
        novel_accuracy = test_step(base_model, self.test_novel, self.novel_classes, self.batch_size, self.device, label="test", base=base)
        # Log to TensorBoard
        self.log_value(epoch_idx,  base_accuracy, "base_classes")
        self.log_value(epoch_idx,  novel_accuracy, "novel_classes")

        return base_accuracy, novel_accuracy

    def get_optimizer(self, lr, wd, momentum):
        optimizer = SGD([
            {
                "params": self.model.parameters()
            }
        ], lr=lr, weight_decay=wd, momentum=momentum)

        return optimizer

    def log_value(self, step,  accuracy, prefix):
        self.writer.add_scalar(f"{prefix}/accuracy", accuracy, step)

    def log_values(self, step, loss, accuracy, prefix):
        self.writer.add_scalar(f"{prefix}/loss", loss, step)
        self.writer.add_scalar(f"{prefix}/accuracy", accuracy, step)