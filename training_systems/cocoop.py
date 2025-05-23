"""
Main module for training the CoCoOp system, supporting both base and adversarial training phases.
Includes configuration, data loading, model preparation, and training logic for zero-shot learning with CLIP.
"""
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
import hashlib

from utils.tensor_board_logger import TensorboardLogger
from training_systems.training_methods import TrainingMethod, Adversarial, KLAdversarial, KLCoCoOp
from training_systems.evaluation_methods import BaseTestStep, FineTunedTestStep, EvalStep

def checksum(model):
    """
    Generate an MD5 checksum of the model's parameters to track changes across training.

    Args:
        model (torch.nn.Module): The model to hash.

    Returns:
        str: MD5 hash string.
    """
    with torch.no_grad():
        all_params = torch.cat([p.view(-1).cpu() for p in model.parameters() if p.requires_grad])
        return hashlib.md5(all_params.numpy().tobytes()).hexdigest()


class CoCoOpSystem:
    """
    Manages the full training process of the CoCoOp model, including configuration, training, evaluation,
    checkpointing, and logging. Supports both base and adversarial training.
    """
    def __init__(
            self,
            *,
            batch_size=16,
            device="cuda",
            run_name="exp1",
            cnn_model="ViT-B/32",
            optimizer_configs=None,
            skip_tests=None,
            train_base_checkpoint_path=None,
            debug=False,
            prompt_learner_opt=None,
            kl_loss_opt=None,
            adv_training_opt=None,
            base_training_opt=None,
    ):
        """
        Initialize the CoCoOp system, load data, setup the model, loss functions, optimizers, and logger.

        Args:
            batch_size (int): Batch size for training.
            device (str): Device identifier (e.g., 'cuda' or 'cpu').
            run_name (str): Unique name for the training run.
            cnn_model (str): Backbone CLIP model name.
            optimizer_configs (list): Optimizer settings for base and adversarial training.
            skip_tests (list): Booleans to skip testing after each training stage.
            train_base_checkpoint_path (str): Optional path to a pre-trained base model.
            debug (bool): Enables logging of additional debug information.
            prompt_learner_opt, kl_loss_opt, adv_training_opt, base_training_opt: Configuration dictionaries.
        """
        self.batch_size = batch_size
        self.device = device
        self.epochs = base_training_opt["epochs"]
        self.run_name = run_name
        self.n_ctx = prompt_learner_opt["n_ctx"]
        self.ctx_init = prompt_learner_opt["ctx_init"]
        self.class_token_position = prompt_learner_opt["class_token_position"]
        self.csc = prompt_learner_opt["csc"]
        self.lambda_kl = kl_loss_opt["lambda_kl"]
        self.cls_cluster_dict = adv_training_opt["cls_cluster_dict"]
        self.lambda_adv = adv_training_opt["lambda_adv"]
        self.adv_training_epochs = adv_training_opt["adv_training_epochs"]
        self.cnn_model = cnn_model
        self.warmup_epoch = base_training_opt["warmup_epoch"]
        self.warmup_cons_lr = base_training_opt["warmup_cons_lr"]
        self.using_kl_adv = kl_loss_opt["using_kl_adv"]
        self.grl_lambda = adv_training_opt["grl_lambda"]
        self.mlp_opt = adv_training_opt["mlp_opt"]
        self.skip_tests = skip_tests if skip_tests is not None else [False, False, False]
        self.train_base_checkpoint_path = train_base_checkpoint_path
        self.debug = debug
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

        self.optimizer = self.get_optimizer(self.model, None, self.optimizer_configs[0])
        self.lr_scheduler = LambdaLR(self.optimizer, self._lr_lambda)

        self._set_train_methods()

        self._set_eval_method()

        self._set_test_methods()

    def _set_test_methods(self):
        self.zero_shot_base_classes_test_method = BaseTestStep(
            model=self.clip_model,
            batch_size=self.batch_size,
            categories=self.base_classes,
        )
        self.zero_shot_novel_classes_test_method = BaseTestStep(
            model=self.clip_model,
            batch_size=self.batch_size,
            categories=self.novel_classes,
        )
        self.finetuned_test_method = FineTunedTestStep(
            model=self.model,
            batch_size=self.batch_size,
        )

    def _set_eval_method(self):
        self.eval_method = EvalStep(
            model=self.model,
            batch_size=self.batch_size,
        )

    def _set_train_methods(self):
        self.adversarial_method = Adversarial(
            lambda_adv=0.05,
            model=self.model,
            optimizer=self.optimizer,
            cls_cluster_dict=self.cls_cluster_dict,
            grl=self.grl,
            mlp_adversary=self.mlp_adversary,
            debug=self.debug,
        ) if not self.using_kl_adv else KLAdversarial(
            lambda_adv=0.05,
            model=self.model,
            optimizer=self.optimizer,
            cls_cluster_dict=self.cls_cluster_dict,
            grl=self.grl,
            mlp_adversary=self.mlp_adversary,
            lambda_kl=self.lambda_kl[1],
            debug=self.debug,
        )
        self.basic_train_method = KLCoCoOp(
            model=self.model,
            optimizer=self.optimizer,
            lambda_kl=self.lambda_kl[0],
            debug=self.debug,
        )

    def train(self):
        """
        Execute the full training pipeline: base phase, optionally followed by adversarial training and evaluation.
        """
        best_model_path = os.path.join("runs/CoCoOp", self.run_name, "best_model.pth")
        if self.train_base_checkpoint_path is None:
            # Base training phase
            base_end_epoch, _ = self._train_base_phase(best_model_path)
            if self.epochs != 0:
                self.model.load_state_dict(torch.load(best_model_path))
                self.save_model(path="./bin/cocoop", prefix="after_first_train_")

            if not self.skip_tests[1]:
                print("Skipping base accuracy test")
                base_acc, novel_acc = self.compute_evaluation(base_end_epoch)
                self._log_final_metrics("Final metrics - After Base Training", base_acc, novel_acc, base_end_epoch)
        else:
            base_end_epoch = 0
            print("Skipping base training")
            # Load the model from the checkpoint
            self.model.load_state_dict(torch.load(self.train_base_checkpoint_path))

        self.optimizer = self.get_optimizer(self.model, self.mlp_adversary, self.optimizer_configs[1])

        checksum1 = None
        if self.debug:
            checksum1 = checksum(self.model)
            # Adversarial phase
            print("Before adv training:", checksum1)

        adv_end_epoch = self._train_adversarial_phase(base_end_epoch, best_model_path)

        if self.debug and checksum1:
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
        self.save_model(path="./bin/cocoop", prefix="after_adv_train_")

    def _train_base_phase(self, best_model_path):
        """
        Train the model using KL regularization only (no adversarial objective).

        Args:
            best_model_path (str): Path to store the best base model.

        Returns:
            Tuple[int, float]: Final epoch index and best validation accuracy on novel classes.
        """
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
        """
        Train the model adversarially with dynamic lambda scheduling and early stopping.

        Args:
            start_epoch (int): Starting epoch index.
            best_model_path (str): Path to save the best adversarial model.

        Returns:
            int: Final epoch index after training.
        """
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
        """
        Run validation and log results for both base and novel splits.

        Args:
            epoch (int): Current training epoch.
            is_adv (bool): Whether evaluation is during adversarial training.

        Returns:
            Tuple[float, float]: Accuracy for base and novel classes.
        """
        metrics_base = self.eval_method.evaluate(
            dataset=self.val_base,
            desc_add=" - Base",
        )
        base_val_loss = metrics_base["loss"]
        base_val_acc = metrics_base["accuracy"]

        metrics_novel = self.eval_method.evaluate(
            dataset=self.val_novel,
            new_classnames=self.novel_classes,
            desc_add=" - Novel"
        )
        novel_val_loss = metrics_novel["loss"]
        novel_val_acc = metrics_novel["accuracy"]

        self.logger.log_validation(epoch, base_val_loss, base_val_acc, novel_val_loss, novel_val_acc, is_adv=is_adv)

        return base_val_acc, novel_val_acc

    def _log_final_metrics(self, tag, base_acc, novel_acc, step):
        """
        Log final test results to TensorBoard.

        Args:
            tag (str): Descriptive tag for the log.
            base_acc (float): Accuracy on base classes.
            novel_acc (float): Accuracy on novel classes.
            step (int): Epoch or step index for this log.
        """
        self.logger.log_final_metrics(tag, base_acc, novel_acc, step)

    def _lr_lambda(self, current_epoch):
        """
        Learning rate scheduler with cosine annealing and warm-up.

        Args:
            current_epoch (int): Epoch index.

        Returns:
            float: Learning rate multiplier.
        """
        if current_epoch < self.warmup_epoch:
            return self.warmup_cons_lr / self.optimizer_configs[0].prompt_lr
        return 0.5 * (1 + math.cos(math.pi * (current_epoch - self.warmup_epoch) / (self.max_epoch - self.warmup_epoch + 1e-7)))

    def compute_evaluation(self, epoch_idx, base=False):
        """
        Run evaluation on the test split for both base and novel classes.

        Args:
            epoch_idx (int): Epoch index for logging.
            base (bool): Whether to evaluate the frozen base CLIP model.

        Returns:
            Tuple[float, float]: Base and novel class test accuracy.

        model = self.model if not base else self.clip_model
        base_accuracy = test_step(model, self.test_base, self.base_classes, self.batch_size, self.device, label="test", base=base)
        novel_accuracy = test_step(model, self.test_novel, self.novel_classes, self.batch_size, self.device, label="test", base=base)"""
        if base:
            base_metrics = self.zero_shot_base_classes_test_method.evaluate(
                dataset=self.test_base,
                desc_add=" - Base Zerp Shot",
            )
            novel_metrics = self.zero_shot_novel_classes_test_method.evaluate(
                dataset=self.test_novel,
                desc_add=" - Novel Zero Shot",
            )
        else:
            base_metrics = self.finetuned_test_method.evaluate(
                dataset=self.test_base,
                new_classnames=self.base_classes,
                desc_add=" - Base Fine Tuned",
            )
            novel_metrics = self.finetuned_test_method.evaluate(
                dataset=self.test_novel,
                new_classnames=self.novel_classes,
                desc_add=" - Novel Fine Tuned",
            )

        base_accuracy = base_metrics["accuracy"]
        novel_accuracy = novel_metrics["accuracy"]
        self.logger.log_test_accuracy(epoch_idx, base_accuracy, "base_classes")
        self.logger.log_test_accuracy(epoch_idx, novel_accuracy, "novel_classes")
        return base_accuracy, novel_accuracy

    def save_model(self, path="./bin/cocoop", prefix=""):
        """
        Save model weights to disk.

        Args:
            path (str): Directory to save the model to.
            prefix (str): Filename prefix to distinguish models.
        """
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(path, f"{prefix}{self.run_name}.pth"))

    def get_optimizer(self, model, mlp_adversary, config):
        """
        Build an SGD optimizer with separate learning rates for different parameter groups.

        Args:
            model (torch.nn.Module): Main model.
            mlp_adversary (torch.nn.Module): Optional adversarial MLP.
            config: Optimizer configuration namespace.

        Returns:
            torch.optim.Optimizer: Configured optimizer.
        """
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
