import os
import math
from copy import deepcopy

import torch
from easydict import EasyDict
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
import clip

from model.cocoop.custom_clip import CustomCLIP
from model.cocoop.mlp_adversary import GradientReversalLayer, AdversarialMLP
from utils.datasets import get_data, base_novel_categories, split_data, CLASS_NAMES
from utils.training_cocoop import (
    test_step,
    eval_step,
)
import hashlib

from utils.tensor_board_logger import TensorboardLogger
from training_systems.training_methods import TrainingMethod, Adversarial, KLAdversarial, KLCoCoOp

def checksum(model):
    with torch.no_grad():
        all_params = torch.cat([p.view(-1).cpu() for p in model.parameters() if p.requires_grad])
        return hashlib.md5(all_params.numpy().tobytes()).hexdigest()

def harmonic_mean(a, b):
    return 2 * (a * b) / (a + b)


class CoCoOpSystem:
    def __init__(self, *, optimizer_configs, **kwargs):
        # Hyperparameters
        self.batch_size = kwargs.get("batch_size", 16)
        self.device = kwargs.get("device", "cuda")
        self.epochs = kwargs.get("epochs", 2)
        self.run_name = kwargs.get("run_name", "exp1")
        self.n_ctx = kwargs.get("n_ctx", 4)
        self.ctx_init = kwargs.get("ctx_init", "")
        self.class_token_position = kwargs.get("class_token_position", "end")
        self.csc = kwargs.get("csc", False)
        self.lambda_kl = kwargs.get("lambda_kl", [0.5, 0.5])
        self.cls_cluster_dict = kwargs.get("cls_cluster_dict", None)
        self.lambda_adv = kwargs.get("lambda_adv", 0.5)
        self.adv_training_epochs = kwargs.get("adv_training_epochs", 2)
        self.cnn_model = kwargs.get("cnn_model", "ViT-B/32")
        self.warmup_epoch = kwargs.get("warmup_epoch", 1)
        self.warmup_cons_lr = kwargs.get("warmup_cons_lr", 1e-5)
        self.using_kl_adv = kwargs.get("using_kl_adv", False)
        self.grl_lambda = kwargs.get("grl_lambda", 1.0)
        self.mlp_opt = kwargs.get("mlp_opt", EasyDict(hidden_dim=512, hidden_layers=2))
        self.skip_tests = kwargs.get("skip_tests", [False, False, False])
        self.max_epoch = self.epochs
        self.optimizer_configs = optimizer_configs

        self.writer = SummaryWriter(log_dir=f"runs/CoCoOp/{self.run_name}")
        self.logger = TensorboardLogger(self.writer)

        self.logger.log_hparams({
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "n_ctx": self.n_ctx,
            "ctx_init": self.ctx_init,
            "class_token_position": self.class_token_position,
            "csc": self.csc,
            "lambda_kl_first": self.lambda_kl[0],
            "lambda_kl_second": self.lambda_kl[1],
            "warmup_epoch": self.warmup_epoch,
            "warmup_cons_lr": self.warmup_cons_lr,
            "lambda_adv": self.lambda_adv,
            "cnn_model": self.cnn_model,
            "grl_lambda" : self.grl_lambda,
            "mlp_hidden_dim": self.mlp_opt.hidden_dim,
            "mlp_hidden_layers": self.mlp_opt.hidden_layers,
        })

        # Load model
        self.clip_model, preprocess = clip.load(self.cnn_model)
        self.clip_model = self.clip_model.to(self.device)

        self.train_set, self.val_set, self.test_set = get_data(transform=preprocess)
        self.base_classes, self.novel_classes = base_novel_categories(self.train_set)
        self.train_base, _ = split_data(self.train_set, self.base_classes)
        self.val_base, self.val_novel = split_data(self.val_set, self.base_classes)
        self.test_base, self.test_novel = split_data(self.test_set, self.base_classes)

        resolution = self.clip_model.visual.input_resolution
        ctx_load = "./bin/coop/exp1_ctx_only.pth" if self.n_ctx == 4 else "./bin/coop/coop_ctx_8.pth"
        cfg = EasyDict({
            "TRAINER": {
                "COCOOP": {"CTX_LOAD": ctx_load, "N_CTX": self.n_ctx, "CTX_INIT": self.ctx_init,
                           "PREC": "fp16"}},
            "INPUT": {"SIZE": [resolution, resolution]}
        })

        self.model = CustomCLIP(
            classnames=[CLASS_NAMES[idx] for idx in self.base_classes],
            cfg=cfg,
            clip_model=self.clip_model
        ).to(self.device)

        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)
            else:
                param.requires_grad_(True)

        self.cost_function = nn.CrossEntropyLoss()
        self.grl = GradientReversalLayer(lambda_=self.grl_lambda)

        clip_dim = self.clip_model.visual.output_dim
        self.mlp_adversary = (AdversarialMLP(
            input_dim=len(self.base_classes)+clip_dim,
            opt=self.mlp_opt
        ).to(self.device))

        print(self.optimizer_configs[1])
        self.optimizer = self.get_optimizer(self.model, None, self.optimizer_configs[0])
        self.lr_scheduler = LambdaLR(self.optimizer, self._lr_lambda)

        self.adversarial_method = Adversarial(
            lambda_adv=0.05,
            model=self.model,
            optimizer=self.optimizer,
            cls_cluster_dict=self.cls_cluster_dict,
            grl=self.grl,
            mlp_adversary=self.mlp_adversary,
        ) if not self.using_kl_adv else KLAdversarial(
            lambda_adv=0.05,
            model=self.model,
            optimizer=self.optimizer,
            cls_cluster_dict=self.cls_cluster_dict,
            grl=self.grl,
            mlp_adversary=self.mlp_adversary,
            lambda_kl=self.lambda_kl[1]
        )

        self.basic_train_method = KLCoCoOp(
            model=self.model,
            optimizer=self.optimizer,
            lambda_kl=self.lambda_kl[0],
        )

    def train(self):

        best_model_path = os.path.join("runs/CoCoOp", self.run_name, "best_model.pth")

        # Base training phase
        base_end_epoch, _ = self._train_base_phase(best_model_path)
        if self.epochs != 0:
            self.model.load_state_dict(torch.load(best_model_path))

        if not self.skip_tests[1]:
            print("Skipping base accuracy test")
            base_acc, novel_acc = self.compute_evaluation(base_end_epoch)
            self._log_final_metrics("Final metrics - After Base Training", base_acc, novel_acc, base_end_epoch)

        self.optimizer = self.get_optimizer(self.model, self.mlp_adversary, self.optimizer_configs[1])
        checksum1 = checksum(self.model)
        # Adversarial phase
        print("Before adv training:", checksum1)
        adv_end_epoch = self._train_adversarial_phase(base_end_epoch, best_model_path)

        checksum2 = checksum(self.model)
        print("After adv training:", checksum2)
        print(f"checksum1: {checksum1}, checksum2: {checksum2}")
        if checksum1 != checksum2:
            print("Model parameters have changed after adversarial training.")

        if not self.skip_tests[2]:
            print("Skipping base accuracy test")
            base_acc, novel_acc = self.compute_evaluation(adv_end_epoch)
            self._log_final_metrics("Final metrics - After Adversarial Training", base_acc, novel_acc, adv_end_epoch)

        self.logger.close()
        self.save_model()

    def _train_base_phase(self, best_model_path):
        best_novel_accuracy = 0.0
        patience = 4
        patience_counter = 0
        c = 0
        method = self.basic_train_method

        pbar = tqdm(total=self.max_epoch, desc="Base Training")

        for e in range(self.max_epoch):

            total_loss, acc, ce_loss, kl_loss = method.train_step(
                self.train_base,
                self.batch_size,
            )

            self.logger.log_training_base(
                e,
                self.optimizer.param_groups[0]["lr"],
                ce_loss,
                acc,
                kl_loss,
                total_loss
            )

            base_val_acc, novel_val_acc = self._evaluate_and_log(e)
            if novel_val_acc > best_novel_accuracy:
                best_novel_accuracy = novel_val_acc
                patience_counter = 0
                torch.save(self.model.state_dict(), best_model_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {e}")
                    break
            self.lr_scheduler.step()
            pbar.set_postfix(
                ce_loss=ce_loss,
                kl_loss=kl_loss,
                base_val_acc=base_val_acc,
                lr=self.optimizer.param_groups[0]["lr"],
                patience_counter=patience_counter,
            )
            pbar.update(1)
            c += 1

        return c, best_novel_accuracy

    def _train_adversarial_phase(self, start_epoch, best_model_path):
        best_novel_accuracy = 0.0
        patience = 5
        patience_counter = 0
        at_least_one_improving = False
        warmup_epochs = max(1, int(0.2 * self.adv_training_epochs))
        lambda_adv_max = self.lambda_adv
        initial_lambda_adv = 0.05
        pbar = tqdm(total=self.adv_training_epochs, desc="Adversarial Training")

        last_model_state = None  # store last model state

        method = self.adversarial_method

        for e in range(start_epoch, start_epoch + self.adv_training_epochs):
            progress = (e - start_epoch + 1) / warmup_epochs
            new_lambda_adv = initial_lambda_adv + (lambda_adv_max - initial_lambda_adv) * 0.5 * (1 - math.cos(math.pi * min(progress, 1)))

            method.update_lambda_adv(new_lambda_adv)

            if self.using_kl_adv:
                total_loss, acc, ce_loss, kl_loss, adv_loss = method.train_step(
                    self.train_base,
                    self.batch_size,
                )
            else:
                total_loss, acc, ce_loss, adv_loss = method.train_step(
                    self.train_base,
                    self.batch_size,
                )
                kl_loss = None

            self.logger.log_training_adv(
                e,
                method.lambda_adv,
                ce_loss,
                acc,
                adv_loss,
                ce_loss + adv_loss + (kl_loss if kl_loss else 0.0),
                kl_loss=kl_loss
            )

            last_model_state = deepcopy(self.model.state_dict())

            base_val_acc, novel_val_acc = self._evaluate_and_log(
                e,
                is_adv=True,
            )

            if novel_val_acc > best_novel_accuracy:
                best_novel_accuracy = novel_val_acc
                torch.save(self.model.state_dict(), best_model_path)
                at_least_one_improving = True
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping adversarial at epoch {e}")
                    break
            pbar.set_postfix(
                ce_loss=ce_loss,
                kl_loss=kl_loss,
                adv_loss=adv_loss,
                base_val_acc=base_val_acc,
                lr=self.optimizer.param_groups[0]["lr"],
                patience_counter=patience_counter,
            )
            pbar.update(1)

        if at_least_one_improving and self.epochs != 0:
            self.model.load_state_dict(torch.load(best_model_path))
            print("Loaded best model from adversarial checkpoint.")

        else:
            print("No improvement during second training. Using model from last adversarial epoch.")
            if last_model_state is not None:
                self.model.load_state_dict(last_model_state)
                print("Loaded last adversarial model state.")

        return start_epoch + self.adv_training_epochs

    def _evaluate_and_log(self, epoch, is_adv=False):
        base_val_loss, base_val_acc = eval_step(
            model=self.model,
            dataset=self.val_base,
            cost_function=self.cost_function,
            device=self.device,
            batch_size=self.batch_size,
            desc_add=" - Base",
        )
        novel_val_loss, novel_val_acc = eval_step(
            model=self.model,
            dataset=self.val_novel,
            cost_function=self.cost_function,
            device=self.device,
            batch_size=self.batch_size,
            new_classnames=self.novel_classes,
            desc_add=" - Novel"
        )

        self.logger.log_validation(epoch, base_val_loss, base_val_acc, novel_val_loss, novel_val_acc, is_adv=is_adv)

        return base_val_acc, novel_val_acc

    def _log_final_metrics(self, tag, base_acc, novel_acc, step):
        self.logger.log_final_metrics(tag, base_acc, novel_acc, step)

    def _lr_lambda(self, current_epoch):
        if current_epoch < self.warmup_epoch:
            return self.warmup_cons_lr / self.optimizer_configs[0].prompt_lr
        return 0.5 * (1 + math.cos(math.pi * (current_epoch - self.warmup_epoch) / (self.max_epoch - self.warmup_epoch + 1e-7)))

    def compute_evaluation(self, epoch_idx, base=False):
        model = self.model if not base else self.clip_model
        base_accuracy = test_step(model, self.test_base, self.base_classes, self.batch_size, self.device, label="test", base=base)
        novel_accuracy = test_step(model, self.test_novel, self.novel_classes, self.batch_size, self.device, label="test", base=base)
        self.logger.log_test_accuracy(epoch_idx, base_accuracy, "base_classes")
        self.logger.log_test_accuracy(epoch_idx, novel_accuracy, "novel_classes")
        return base_accuracy, novel_accuracy

    def save_model(self, path="./bin/cocoop"):
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(path, f"{self.run_name}.pth"))

    def get_optimizer(self, model, mlp_adversary, config):
        params = [
            {
                "params": [p for n, p in model.named_parameters() if "prompt_learner" in n and p.requires_grad],
                "lr": config.prompt_lr,
            }
        ]
        if mlp_adversary is not None:
            params.append({
                "params": mlp_adversary.parameters(),
                "lr": config.mlp_lr,
            })
        return torch.optim.SGD(params, weight_decay=config.weight_decay, momentum=config.momentum)
