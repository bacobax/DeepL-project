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


def harmonic_mean(a, b):
    """returns the harmonic mean of a and b"""
    return 2 * (a * b) / (a + b)


class CoCoOpSystem:
    def __init__(
        self,
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
        lambda_kl=None,
        cls_cluster_dict=None,
        lambda_adv=0.5,
        adv_training_epochs=2,
        cnn_model="ViT-B/32",
        warmup_epoch=1,
        warmup_cons_lr=1e-5
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
        self.cls_cluster_dict = cls_cluster_dict
        self.lambda_adv = lambda_adv
        self.adv_training_epochs = adv_training_epochs
        self.max_epoch = self.epochs
        self.lr_scheduler_type = "cosine"
        self.warmup_epoch = warmup_epoch
        self.warmup_type = "constant"
        self.warmup_cons_lr = warmup_cons_lr

        self.cnn_model = cnn_model

        # Create a logger for the experiment
        self.writer = SummaryWriter(log_dir=f"runs/CoCoOp/{run_name}")

        self.writer.add_hparams(
            {
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
                "lambda_kl_first": self.lambda_kl[0],
                "lambda_kl_second": self.lambda_kl[1],
                "warmup_cons_lr": self.warmup_cons_lr,
                # "cls_cluster_dict": self.cls_cluster_dict,
                "lambda_adv": self.lambda_adv,
                "cnn_model": self.cnn_model,
            },
            {},
        )

        # Get dataloaders

        self.clip_model, preprocess = clip.load(cnn_model)
        self.train_set, self.val_set, self.test_set = get_data(transform=preprocess)

        # split classes into base and novel
        self.base_classes, self.novel_classes = base_novel_categories(self.train_set)

        # split the three datasets
        self.train_base, _ = split_data(self.train_set, self.base_classes)
        self.val_base, self.val_novel = split_data(self.val_set, self.base_classes)
        self.test_base, self.test_novel = split_data(self.test_set, self.base_classes)

        # self.classnames, _ = embed_dataset_classnames(dataset_name, preprocess=preprocess, model=clip_model)

        resolution = self.clip_model.visual.input_resolution

        cfg = EasyDict()
        # Training configuration
        cfg.TRAINER = EasyDict()
        cfg.TRAINER.COCOOP = EasyDict()
        cfg.TRAINER.COCOOP.CTX_LOAD = "./bin/coop/exp1_ctx_only.pth"
        cfg.TRAINER.COCOOP.N_CTX = self.n_ctx  # Number of context tokens
        cfg.TRAINER.COCOOP.CTX_INIT = (
            self.ctx_init
        )  # Leave empty for random initialization
        cfg.TRAINER.COCOOP.PREC = "fp16"  # Precision for meta network
        cfg.INPUT = EasyDict()
        cfg.INPUT.SIZE = [
            resolution,
            resolution,
        ]  # Must match CLIP model's input resolution

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
        print(
            f"Total trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}"
        )

        self.cost_function = nn.CrossEntropyLoss()

        self.grl = GradientReversalLayer(lambda_=1.0)
        self.mlp_adversary = AdversarialMLP(input_dim=len(self.base_classes)).to(device, dtype=torch.float16)
        self.optimizer = self.get_optimizer(
            self.model,
            self.mlp_adversary,
            self.learning_rate,
            self.weight_decay,
            self.momentum,
        )
        for name, param in self.mlp_adversary.named_parameters():
            
            print(f"mlp dtype: {param.dtype}")

    def train(self):
        def lr_lambda(current_epoch):
            if current_epoch < self.warmup_epoch:
                return self.warmup_cons_lr / self.learning_rate
            else:
                cosine_decay = 0.5 * (
                    1
                    + math.cos(
                        math.pi
                        * (current_epoch - self.warmup_epoch)
                        / (self.max_epoch - self.warmup_epoch)
                    )
                )
                return cosine_decay

        self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda)

        print("Before training:")
        self.compute_evaluation(-1, base=True)
        print("Training the model...")
        print_epoch_interval = 2

        best_novel_accuracy = 0.0
        patience_counter = 0
        patience = 4  # adjustable
        best_model_path = os.path.join("runs/CoCoOp", self.run_name, "best_model.pth")
        c = 0
        pbar = tqdm(
            total=self.max_epoch, desc="OVERALL TRAINING", position=0, leave=True
        )
        for e in range(self.max_epoch):
            (
                base_train_total_loss,
                base_train_ce_accuracy,
                base_ce_loss,
                base_kl_loss,
            ) = training_step_v2(
                model=self.model,
                dataset=self.train_base,
                optimizer=self.optimizer,
                batch_size=self.batch_size,
                device=self.device,
                lambda_kl=self.lambda_kl[0],
            )

            # Always log train_base values every epoch
            self.writer.add_scalar(f"train_base/ce_loss", base_ce_loss, e)
            self.writer.add_scalar(f"train_base/ce_accuracy", base_train_ce_accuracy, e)
            self.writer.add_scalar(f"train_base/kl_loss", base_kl_loss, e)
            self.writer.add_scalar(f"train_base/total_loss", base_train_total_loss, e)

            if e % print_epoch_interval == 0:
                base_val_loss, base_val_accuracy = eval_step(
                    model=self.model,
                    dataset=self.val_base,
                    cost_function=self.cost_function,
                    device=self.device,
                    batch_size=self.batch_size,
                    desc_add=" - Base"
                )
                novel_val_loss, novel_val_accuracy = eval_step(
                    model=self.model,
                    dataset=self.val_novel,
                    cost_function=self.cost_function,
                    device=self.device,
                    batch_size=self.batch_size,
                    new_classnames=self.novel_classes,
                    desc_add=" - Novel"
                )

                self.log_values(e, base_val_loss, base_val_accuracy, "validation_base")
                self.log_values(
                    e, novel_val_loss, novel_val_accuracy, "validation_novel"
                )

                pbar.set_postfix(
                    train_acc=base_train_ce_accuracy,
                    val_acc=base_val_accuracy,
                    train_total_loss=base_train_total_loss,
                )

                if novel_val_accuracy > best_novel_accuracy:
                    best_novel_accuracy = novel_val_accuracy
                    patience_counter = 0
                    torch.save(self.model.state_dict(), best_model_path)
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(
                            f"Stopping early at epoch {e} due to no improvement in novel accuracy."
                        )
                        break

            pbar.update(1)
            self.lr_scheduler.step()
            c += 1

        print("After training:")
        self.model.load_state_dict(torch.load(best_model_path))  # Load best model
        base_acc, novel_acc = self.compute_evaluation(c)
        self.writer.add_scalars(
            "Final metrics",
            {
                "Harmonic Mean": harmonic_mean(base_acc, novel_acc),
                "Base Accuracy": base_acc,
                "Novel Accuracy": novel_acc,
            },
            global_step=c,
        )

        last_epoch = self.second_train(c)

        base_acc, novel_acc = self.compute_evaluation(last_epoch)
        self.writer.add_scalars(
            "Final metrics",
            {
                "Harmonic Mean": harmonic_mean(base_acc, novel_acc),
                "Base Accuracy": base_acc,
                "Novel Accuracy": novel_acc,
            },
            global_step=last_epoch + 1,
        )
        self.writer.close()
        self.save_model()

    def second_train(self, start_epoch):

        print("Training the adv  model...")
        print_epoch_interval = 2

        best_novel_accuracy = 0.0
        patience_counter = 0
        patience = 5 # adjustable
        best_model_path = os.path.join("runs/CoCoOp", self.run_name, "best_model.pth")
        at_least_one_improoving = False
        c = start_epoch
        pbar = tqdm(
            total=self.max_epoch,
            desc="OVERALL TRAINING - Adversarial",
            position=0,
            leave=True,
            initial=start_epoch,
        )
        end_adv_training_epoch = start_epoch + self.adv_training_epochs
        # --- Lambda adv warmup ---
        lambda_adv_max = self.lambda_adv
        warmup_epochs = max(1, int(0.2 * self.adv_training_epochs))
        initial_lambda_adv = 0.05
        for e in range(start_epoch, end_adv_training_epoch):
            # Compute current lambda_adv with linear warmup
            progress = (e - start_epoch + 1) / warmup_epochs
            current_lambda_adv = initial_lambda_adv + (lambda_adv_max - initial_lambda_adv) * 0.5 * (1 - math.cos(math.pi * min(progress, 1)))
            (
                base_train_total_loss,
                base_train_ce_accuracy,
                base_ce_loss,
                base_kl_loss,
                base_adv_loss,
            ) = adversarial_training_step(
                model=self.model,
                dataset=self.train_base,
                optimizer=self.optimizer,
                batch_size=self.batch_size,
                cls_cluster_dict=self.cls_cluster_dict,
                lambda_kl=self.lambda_kl[1],
                lambda_adv=current_lambda_adv,
                mlp_adversary=self.mlp_adversary,
                grl=self.grl,
                device=self.device,
            )

            if e % print_epoch_interval == 0:
                base_val_loss, base_val_accuracy = eval_step(
                    model=self.model,
                    dataset=self.val_base,
                    cost_function=self.cost_function,
                    device=self.device,
                    batch_size=self.batch_size,
                    desc_add=" - Base"
                )
                novel_val_loss, novel_val_accuracy = eval_step(
                    model=self.model,
                    dataset=self.val_novel,
                    cost_function=self.cost_function,
                    device=self.device,
                    batch_size=self.batch_size,
                    new_classnames=self.novel_classes,
                    desc_add=" - Novel"
                )
                self.writer.add_scalar("lambda_adv", current_lambda_adv, e)
                self.log_values(e, base_val_loss, base_val_accuracy, "validation_base")
                self.log_values(
                    e, novel_val_loss, novel_val_accuracy, "validation_novel"
                )

                self.writer.add_scalar(f"train_adv/ce_loss", base_ce_loss, e)
                self.writer.add_scalar(
                    f"train_adv/ce_accuracy", base_train_ce_accuracy, e
                )
                self.writer.add_scalar(f"train_adv/kl_loss", base_kl_loss, e)
                self.writer.add_scalar(
                    f"train_adv/total_loss", base_train_total_loss, e
                )
                self.writer.add_scalar(f"train_adv/mlp_loss", base_adv_loss, e)

                pbar.set_postfix(
                    train_acc=base_train_ce_accuracy,
                    val_acc=base_val_accuracy,
                    train_total_loss=base_train_total_loss,
                    train_adv_loss=base_adv_loss,
                )

                if novel_val_accuracy > best_novel_accuracy:
                    best_novel_accuracy = novel_val_accuracy
                    patience_counter = 0
                    torch.save(self.model.state_dict(), best_model_path)
                    at_least_one_improoving = True
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(
                            f"Stopping early at epoch {e} due to no improvement in novel accuracy."
                        )
                        break

            pbar.update(1)
            c += 1

        # print("After training:")  # Remove this print line as requested
        if at_least_one_improoving:
            self.model.load_state_dict(torch.load(best_model_path))
        return c

    def save_model(self, path="./bin/cocoop"):
        # create folder if not exist
        if not os.path.exists(path):
            os.makedirs(path)
        # Save the model
        torch.save(self.model.state_dict(), os.path.join(path, f"{self.run_name}.pth"))

    def load_model(self, path="./bin"):
        # Load the model
        self.model.load_state_dict(
            torch.load(os.path.join(path, f"{self.run_name}.pth"))
        )
        self.model.eval()
        print(f"Model loaded from {path}")

    def compute_evaluation(self, epoch_idx, base=False):
        base_model = self.model if not base else self.clip_model
        base_accuracy = test_step(
            base_model,
            self.test_base,
            self.base_classes,
            self.batch_size,
            self.device,
            label="test",
            base=base,
        )
        novel_accuracy = test_step(
            base_model,
            self.test_novel,
            self.novel_classes,
            self.batch_size,
            self.device,
            label="test",
            base=base,
        )
        # Log to TensorBoard
        self.log_value(epoch_idx, base_accuracy, "base_classes")
        self.log_value(epoch_idx, novel_accuracy, "novel_classes")

        return base_accuracy, novel_accuracy

    def get_optimizer(self, model, mlp_adversary, lr, wd, momentum):
        return SGD(
            [{"params": list(model.parameters()) + list(mlp_adversary.parameters())}],
            lr=lr,
            weight_decay=wd,
            momentum=momentum,
        )

    def log_value(self, step, accuracy, prefix):
        self.writer.add_scalar(f"{prefix}/accuracy", accuracy, step)

    def log_values(self, step, loss, accuracy, prefix):
        self.writer.add_scalar(f"{prefix}/loss", loss, step)
        self.writer.add_scalar(f"{prefix}/accuracy", accuracy, step)

