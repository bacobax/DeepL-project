"""
Main module for training the CoCoOp system, supporting both base and adversarial training phases.
Includes configuration, data loading, model preparation, and training logic for zero-shot learning with CLIP.
"""

import os
import math
from copy import deepcopy
from statistics import harmonic_mean
import torch
from easydict import EasyDict
from tqdm import tqdm
from torch.utils.tensorboard.writer import SummaryWriter
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
import clip
import numpy as np
import random
import hashlib

from model.cocoop.custom_clip import CustomCLIP
from model.cocoop.mlp_adversary import GradientReversalLayer, AdversarialMLP
from utils import (
    conditional_clustering,
    random_clustering,
    rotating_cluster_generator_shift,
    get_data,
    base_novel_categories,
    split_data,
    TensorboardLogger,
    CLASS_NAMES
)

from training_systems.training_methods import (
    Adversarial,
    KLCoCoOp,
    BaseCoCoOp,
    KLCoCoOpV2,
)
from training_systems.evaluation_methods import (
    ZeroShotTestStep,
    FineTunedTestStep,
    EvalStep,
)
from training_systems.core import DoubleDatasetTrainingMethod

# --- Add this block for reproducibility ---
def set_global_seed(seed):

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # For CUDA >= 10.2, for full determinism
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # For PyTorch >= 1.8
    if hasattr(torch, 'use_deterministic_algorithms'):
        torch.use_deterministic_algorithms(True)
# --- End reproducibility block ---


def checksum(model):
    """
    Generate an MD5 checksum of the model's parameters to track changes across training.

    Args:
        model (torch.nn.Module): The model to hash.

    Returns:
        str: MD5 hash string.
    """
    with torch.no_grad():
        all_params = torch.cat(
            [p.view(-1).cpu() for p in model.parameters() if p.requires_grad]
        )
        return hashlib.md5(all_params.numpy().tobytes()).hexdigest()


class CoCoOpSystem:
    """
    Manages the full training process of the CoCoOp model, including configuration, training, evaluation,
    checkpointing, and logging. Supports both base and adversarial training.
    """

    def __init__(
        self,
        *,
        test_batch_size=16,
        pseudo_base_ratio=0.7,
        seed=42,
        device="cuda",
        run_name="exp1",
        cnn_model="ViT-B/32",
        hparams_file,
        optimizer_configs=None,
        skip_tests=None,
        train_base_checkpoint_path=None,
        debug=False,
        prompt_learner_opt=None,
        kl_loss_opt=None,
        adv_training_opt=None,
        base_training_opt=None,
        clustering_opt=None,

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
        # --- Set global seed for reproducibility ---
        self.seed = seed if seed is not None else 42
        set_global_seed(self.seed)
        # --- End reproducibility ---
        assert prompt_learner_opt is not None, "prompt_learner_opt must be provided"
        assert kl_loss_opt is not None, "kl_loss_opt must be provided"
        assert adv_training_opt is not None, "adv_training_opt must be provided"
        assert base_training_opt is not None, "base_training_opt must be provided"
        assert clustering_opt is not None, "clustering_opt must be provided"
        assert (
            optimizer_configs is not None and len(optimizer_configs) == 2
        ), "Two optimizer configs must be provided"

        # --- NEW: Pseudo-base/novel split param ---
        self.pseudo_base_ratio = pseudo_base_ratio
        self.pseudo_split_seed = seed

        self.test_batch_size = test_batch_size
        self.device = device
        self.epochs = base_training_opt["epochs"]
        self.run_name = run_name
        self.n_ctx = prompt_learner_opt["n_ctx"]
        self.ctx_init = prompt_learner_opt["ctx_init"]
        self.class_token_position = prompt_learner_opt["class_token_position"]
        self.csc = prompt_learner_opt["csc"]
        self.lambda_kl = kl_loss_opt["lambda_kl"]
        self.double_datasets_kl = kl_loss_opt.get("double_datasets_kl", False)
        self.rotation_period = kl_loss_opt.get("rotation_period", "relative")
        self.lambda_adv = adv_training_opt["lambda_adv"]
        self.adv_training_epochs = adv_training_opt["adv_training_epochs"]
        self.cnn_model = cnn_model
        self.warmup_epoch = base_training_opt["warmup_epoch"]
        self.warmup_cons_lr = base_training_opt["warmup_cons_lr"]
        self.using_kl = kl_loss_opt["using_kl"]
        self.grl_lambda = adv_training_opt["grl_lambda"]
        self.mlp_opt = EasyDict(adv_training_opt["mlp_opt"])
        self.skip_tests = (
            skip_tests if skip_tests is not None else [False, False, False]
        )
        self.train_base_checkpoint_path = train_base_checkpoint_path
        self.debug = debug
        self.max_epoch = self.epochs
        self.optimizer_configs = [EasyDict(conf) for conf in optimizer_configs]
        self.warmup_lambda_adv = adv_training_opt["warmup_lambda_adv"]
        self.base_batch_size = base_training_opt["batch_size"]
        self.adv_batch_size = adv_training_opt["batch_size"]
        self.prompt_learner_warmup_epochs = adv_training_opt["prompt_learner_warmup_epochs"] if "prompt_learner_warmup_epochs" in adv_training_opt else 0
        print(
            "BATCH SIZES: ",
            self.test_batch_size,
            self.base_batch_size,
            self.adv_batch_size,
        )

        self.ignore_no_improvement = adv_training_opt.get("ignore_no_improvement", False)

        self.writer = SummaryWriter(log_dir=f"runs/CoCoOp/{self.run_name}")
        self.writer.add_text("Hparams yaml file", hparams_file)
        self.logger = TensorboardLogger(self.writer)

        self.logger.log_hparams(
            {
                "batch_size_test": self.test_batch_size,
                "base_batch_size": self.base_batch_size,
                "adv_batch_size": self.adv_batch_size,
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
                "grl_lambda": self.grl_lambda,
                "prompt_learner_warmup_epochs" : self.prompt_learner_warmup_epochs,
                "double_datasets_kl": self.double_datasets_kl,
                "pseudo_base_ratio": self.pseudo_base_ratio,
                "pseudo_split_seed": self.seed,
                "rotation_period": self.rotation_period,
            }
        )

        print(self.skip_tests)
        print(self.lambda_kl)

        # Load model
        self.clip_model, preprocess = clip.load(self.cnn_model)
        self.clip_model = self.clip_model.to(self.device)

        self.train_set, self.val_set, self.test_set = get_data(transform=preprocess)
        self.base_classes, self.novel_classes = base_novel_categories(self.train_set)
        # --- NEW: Pseudo-base/novel split ---

        # Helper to split a dataset by class list
        # (moved to method below)

        # Split train_base/val_base into pseudo_base/pseudo_novel
        self.train_base, _ = split_data(self.train_set, self.base_classes)
        self.val_base, self.val_novel = split_data(self.val_set, self.base_classes)
        self.test_base, self.test_novel = split_data(self.test_set, self.base_classes)

        self.rotation_steps = int(len(self.base_classes)*(1-self.pseudo_base_ratio))

        self.cluster_generator = rotating_cluster_generator_shift(
            self.base_classes, 
            self.pseudo_base_ratio, 
            steps=self.rotation_steps, 
            seed=self.seed
        )
        
        _, self.pseudo_base_classes, self.pseudo_novel_classes = next(self.cluster_generator)

        self.train_pseudo_base = self.split_by_classes(self.train_base, self.pseudo_base_classes)
        self.train_pseudo_novel = self.split_by_classes(self.train_base, self.pseudo_novel_classes)
        self.val_pseudo_base = self.split_by_classes(self.val_base, self.pseudo_base_classes)
        self.val_pseudo_novel = self.split_by_classes(self.val_base, self.pseudo_novel_classes)

        # --- Model/classnames: only pseudo_base for first phase ---
        resolution = self.clip_model.visual.input_resolution
        ctx_load = (
            "./bin/coop/exp1_ctx_only.pth"
            if self.n_ctx == 4
            else "./bin/coop/coop_ctx_8.pth"
        )
        cfg = EasyDict(
            {
                "TRAINER": {
                    "COCOOP": {
                        "CTX_LOAD": ctx_load,
                        "N_CTX": self.n_ctx,
                        "CTX_INIT": self.ctx_init,
                        "PREC": "fp16",
                    }
                },
                "INPUT": {"SIZE": [resolution, resolution]},
            }
        )
        self.model = CustomCLIP(

            classnames=[CLASS_NAMES[idx] for idx in self.pseudo_base_classes],
            cfg=cfg,
            clip_model=self.clip_model,
        ).to(self.device)
        print(f"[DEBUG] Model constructed with classnames: {[CLASS_NAMES[idx] for idx in self.pseudo_base_classes]}")

        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)
            else:
                param.requires_grad_(True)

        self.cost_function = nn.CrossEntropyLoss()
        self.grl = GradientReversalLayer(lambda_=self.grl_lambda)

        clip_dim = self.clip_model.visual.output_dim
        print(f"ctx_dim: {self.clip_model.ln_final.weight.shape[0]}, ")

        self.mlp_adversary = AdversarialMLP(
            input_dim=clip_dim+len(self.base_classes), opt=self.mlp_opt, output_dim=clustering_opt["n_clusters"]
        ).to(self.device)
        
        print("mlp adversary struct: ", self.mlp_adversary)
        self.optimizer = self.get_optimizer(self.model, None, self.optimizer_configs[0])
        self.lr_scheduler = LambdaLR(self.optimizer, self._lr_lambda)

        # --- NEW: Cluster dict for adversarial phase ---

        clustering_type = clustering_opt["clustering_type"]

        if clustering_type == "random": 
            # Use random clustering
            self.cls_cluster_dict, _ = random_clustering(
                n_cluster=clustering_opt["n_clusters"],
                seed=self.seed,
                distribution="uniform",
            )
        elif clustering_type == "semantic": 
            # Load clustering information
            self.cls_cluster_dict, _ = conditional_clustering(
                n_cluster=clustering_opt["n_clusters"],
                variance=clustering_opt["variance"],
                cnn=clustering_opt["vision_encoder"],
                device=self.device,
            )
        elif clustering_type == "default":
            # Pseudo_base: cluster 0, pseudo_novel: cluster 1
            self.pseudo_cls_cluster_dict = {c: 0 for c in self.pseudo_base_classes}
            self.pseudo_cls_cluster_dict.update({c: 1 for c in self.pseudo_novel_classes})
            # For adversarial phase, use this dict
            self.cls_cluster_dict = self.pseudo_cls_cluster_dict
        else:
            raise ValueError(f"Unknown clustering type: {clustering_type}")

    def _set_test_methods(self):
        """
        Initializes the zero-shot and fine-tuned test step evaluators for both base and novel class splits.
        """
        self.zero_shot_base_classes_test_method = ZeroShotTestStep(
            model=self.clip_model,
            batch_size=self.test_batch_size,
            categories=self.base_classes,
        )
        self.zero_shot_novel_classes_test_method = ZeroShotTestStep(
            model=self.clip_model,
            batch_size=self.test_batch_size,
            categories=self.novel_classes,
        )
        self.zero_shot_pseudo_base_test_method = ZeroShotTestStep(
            model=self.clip_model,
            batch_size=self.test_batch_size,
            categories=self.pseudo_base_classes,
        )
        self.zero_shot_pseudo_novel_test_method = ZeroShotTestStep(
            model=self.clip_model,
            batch_size=self.test_batch_size,
            categories=self.pseudo_novel_classes,
        )
        self.finetuned_test_method = FineTunedTestStep(
            model=self.model,
            batch_size=self.test_batch_size,
        )

    def _set_eval_method(self):
        """
        Initializes the evaluation method used during validation.
        """
        self.eval_method = EvalStep(
            model=self.model,
            batch_size=self.test_batch_size,
        )

    def _set_train_methods(self):
        """
        Initializes the training method used for both base and adversarial phases, depending on whether KL loss is enabled.
        Chooses between standard and KL-regularized training methods.
        """
        self.adversarial_method = Adversarial(
                lambda_adv=0.05,
                model=self.model,
                optimizer=self.optimizer,
                cls_cluster_dict=self.cls_cluster_dict,
                grl=self.grl,
                mlp_adversary=self.mlp_adversary,
                debug=self.debug,
                tmp_classes=self.base_classes, 
            )
            
        if self.using_kl[0]:
            if self.double_datasets_kl:
                self.basic_train_method = KLCoCoOpV2(
                    model=self.model,
                    optimizer=self.optimizer,
                    debug=self.debug,
                    lambda_kl=self.lambda_kl[0],
                )
            else:
                self.basic_train_method = KLCoCoOp(
                    model=self.model,
                    optimizer=self.optimizer,
                    debug=self.debug,
                    lambda_kl=self.lambda_kl[0],
                )
        else:
            self.basic_train_method = BaseCoCoOp(
                model=self.model,
                optimizer=self.optimizer,
                debug=self.debug,
            )

        

    def train(self):
        """
        Execute the full training pipeline: base phase, optionally followed by adversarial training and evaluation.
        """
        self._set_test_methods()
        self._set_eval_method()
        self._set_train_methods()
        if not self.skip_tests[0]:
            print("Doing base accuracy test")
            base_acc, novel_acc = self.compute_evaluation(-1, base=True)
            self._log_final_metrics(
                "Final metrics - CLIP ZERO SHOT",
                base_acc,
                novel_acc,
                -1,
            )

        best_model_path = os.path.join("runs/CoCoOp", self.run_name, "best_model.pth")
        # Ensure all methods are properly initialized prior to the base training phase
        
        if self.train_base_checkpoint_path is None:
            # Base training phase
            base_end_epoch, _ = self._train_base_phase(best_model_path)
            if self.epochs != 0:
                print(f"[DEBUG] Loading model state dict after base phase from: {best_model_path}")
                self.model.load_state_dict(torch.load(best_model_path))
                print(f"[DEBUG] Loaded model with classnames: {self.model.prompt_learner.n_cls} classes")
                self.save_model(path="./bin/cocoop", prefix="after_first_train_")

            if not self.skip_tests[1]:
                print("Doing base accuracy test")
                base_acc, novel_acc = self.compute_evaluation(base_end_epoch)
                self._log_final_metrics(
                    "Final metrics - After Base Training",
                    base_acc,
                    novel_acc,
                    base_end_epoch,
                )
        else:
            base_end_epoch = 0
            print("Skipping base training")
            print(f"[DEBUG] Loading model state dict from: {self.train_base_checkpoint_path}")
            self.model.load_state_dict(torch.load(self.train_base_checkpoint_path))
            print(f"[DEBUG] Loaded model with classnames: {self.model.prompt_learner.n_cls} classes")

        # Re-initialize test/eval/train methods after loading/training model and before adversarial phase
        self._set_eval_method()
        self._set_test_methods()

        self.optimizer = self.get_optimizer(
            self.model, self.mlp_adversary, self.optimizer_configs[1]
        )
        # After changing optimizer, ensure train methods use the new optimizer
        self._set_train_methods()

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
            print("Doing post-adv. accuracy test")
            base_acc, novel_acc = self.compute_evaluation(adv_end_epoch)
            self._log_final_metrics(
                "Final metrics - After Adversarial Training",
                base_acc,
                novel_acc,
                adv_end_epoch,
            )

        self.logger.close()
        if self.adv_training_epochs != 0:
            self.save_model(path="./bin/cocoop", prefix="after_adv_train_")
            self.save_mlp_adversary()

    def get_next_rotation(self):
        """
        Get the next rotation of the pseudo base and pseudo novel classes.
        """
        _, self.pseudo_base_classes, self.pseudo_novel_classes = next(self.cluster_generator)
        self.train_pseudo_base = self.split_by_classes(self.train_base, self.pseudo_base_classes)
        self.train_pseudo_novel = self.split_by_classes(self.train_base, self.pseudo_novel_classes)
        self.val_pseudo_base = self.split_by_classes(self.val_base, self.pseudo_base_classes)
        self.val_pseudo_novel = self.split_by_classes(self.val_base, self.pseudo_novel_classes)
        return self.pseudo_base_classes, self.pseudo_novel_classes, self.train_pseudo_base, self.train_pseudo_novel, self.val_pseudo_base, self.val_pseudo_novel

    def _train_base_phase(self, best_model_path):
        """
        Train the model using KL regularization only (no adversarial objective).

        Args:
            best_model_path (str): Path to store the best base model.

        Returns:
            Tuple[int, float]: Final epoch index and best validation score.
        """
        best_score = 0.0
        patience = 8
        patience_counter = 0
        c = 0
        method = self.basic_train_method
        rotation_epochs = int(patience * (3/4)) if self.rotation_period == "relative" else self.rotation_period
        pbar = tqdm(total=self.max_epoch, desc="Base Training")

        for e in range(self.max_epoch):

            if self.using_kl[0]:
                if isinstance(method, DoubleDatasetTrainingMethod):
                    if e % rotation_epochs == 0:
                        (
                            self.pseudo_base_classes, 
                            self.pseudo_novel_classes, 
                            self.train_pseudo_base, 
                            self.train_pseudo_novel, 
                            self.val_pseudo_base, 
                            self.val_pseudo_novel
                        ) = self.get_next_rotation()
                    kl_loss, ce_loss, acc = method.double_datasets_train_step(
                        self.train_pseudo_base,
                        self.train_pseudo_novel,
                        self.base_batch_size,
                        ["pseudo_base", "pseudo_novel KL"],
                        self.pseudo_base_classes,
                        self.pseudo_novel_classes,
                    )
                    total_loss = ce_loss + kl_loss
                else:
                    total_loss, acc, ce_loss, kl_loss = method.train_step(
                        self.train_pseudo_base,
                        self.base_batch_size,
                        classnames=self.pseudo_base_classes
                    )
            elif isinstance(method, BaseCoCoOp):
                total_loss, acc = method.train_step(
                    self.train_pseudo_base,
                    self.base_batch_size,
                    classnames=self.pseudo_base_classes
                )
                kl_loss = None
                ce_loss = total_loss

            self.logger.log_training_base(
                e,
                self.optimizer.param_groups[0]["lr"],
                ce_loss,
                acc,
                kl_loss,
                total_loss,
            )

            base_val_acc, novel_val_acc = self._evaluate_and_log(e)

            score = harmonic_mean([base_val_acc, novel_val_acc])

            if score > best_score:
                best_score = score
                patience_counter = 0
                torch.save(self.model.state_dict(), best_model_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {e}")
                    break

            self.lr_scheduler.step()
            pbar.set_postfix(
                PB_val_acc=base_val_acc,
                PN_val_acc=novel_val_acc,
                score=score,
                lr=self.optimizer.param_groups[0]["lr"],
                ce_L=ce_loss,
                kl_L=kl_loss,
                pat_c=patience_counter,
            )
            pbar.update(1)
            c += 1

        return c, best_score

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
        warmup_epochs = self.warmup_lambda_adv
        lambda_adv_max = self.lambda_adv
        initial_lambda_adv = 0.05
        pbar = tqdm(total=self.adv_training_epochs, desc="Adversarial Training")

        last_model_state = None  # store last model state

        method = self.adversarial_method

        for e in range(start_epoch, start_epoch + self.adv_training_epochs):
            progress = (e - start_epoch + 1) / warmup_epochs
            new_lambda_adv = initial_lambda_adv + (
                lambda_adv_max - initial_lambda_adv
            ) * min(progress, 1)

            method.update_lambda_adv(new_lambda_adv)

            if (e-start_epoch) < self.prompt_learner_warmup_epochs:
                for name, param in self.model.named_parameters():
                    if "prompt_learner" in name:
                        param.requires_grad_(False)
            elif (e-start_epoch) == self.prompt_learner_warmup_epochs:
                for name, param in self.model.named_parameters():
                    if "prompt_learner" in name:
                        param.requires_grad_(True)

            if self.using_kl[1]:
                total_loss, acc, ce_loss, kl_loss, adv_loss = method.train_step(
                    self.train_base,
                    self.base_batch_size,
                    classnames=self.base_classes
                )
            else:
                total_loss, acc, ce_loss, adv_loss = method.train_step(
                    self.train_base,
                    self.adv_batch_size,
                    classnames=self.base_classes
                )
                kl_loss = None

            self.logger.log_training_adv(
                e,
                method.lambda_adv,
                ce_loss,
                acc,
                adv_loss,
                ce_loss + adv_loss + (kl_loss if kl_loss else 0.0),
                kl_loss=kl_loss,
            )
            if (e-start_epoch) >= self.prompt_learner_warmup_epochs:

                last_model_state = deepcopy(self.model.state_dict())

                base_val_acc, novel_val_acc = self._evaluate_and_log(
                    e,
                    is_adv=True,
                )
                if novel_val_acc > best_novel_accuracy or self.ignore_no_improvement:
                    best_novel_accuracy = novel_val_acc
                    print(f"[DEBUG] Saving model with classnames: {self.model.prompt_learner.n_cls} classes")
                    torch.save(self.model.state_dict(), best_model_path)
                    at_least_one_improving = True
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping adversarial at epoch {e}")
                        break
                pbar.set_postfix(
                    PB_val_acc=base_val_acc,
                    PN_val_acc=novel_val_acc,
                    ce_L=ce_loss,
                    kl_L=kl_loss,
                    adv_L=adv_loss,
                    lr=self.optimizer.param_groups[0]["lr"],
                    pat_c=patience_counter,
                )
            else:

                pbar.set_postfix(
                    adv_loss=adv_loss,
                )
            pbar.update(1)

        if (at_least_one_improving and self.epochs != 0) or self.ignore_no_improvement:
            print(f"[DEBUG] Loading best model state dict after adversarial phase from: {best_model_path}")
            self.model.load_state_dict(torch.load(best_model_path))
            print(f"[DEBUG] Loaded model with classnames: {self.model.prompt_learner.n_cls} classes")
            print("Loaded best model from adversarial checkpoint (robust, filtered mismatched keys).")
        else:
            print(
                "No improvement during second training. Using model from last adversarial epoch."
            )
            if last_model_state is not None:
                self.model.load_state_dict(self.model.state_dict())
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

        if is_adv:
            metrics_base = self.eval_method.evaluate(
                dataset=self.val_base,
                classnames=self.base_classes,
                desc_add=" - Base",
            )
            base_val_loss = metrics_base["loss"]
            base_val_acc = metrics_base["accuracy"]

            metrics_novel = self.eval_method.evaluate(
                dataset=self.val_novel,
                classnames=self.novel_classes,
                desc_add=" - Novel",
            )
            novel_val_loss = metrics_novel["loss"]
            novel_val_acc = metrics_novel["accuracy"]   
            
        else:

            metrics_base = self.eval_method.evaluate(
                dataset=self.val_pseudo_base,
                classnames=self.pseudo_base_classes,
                desc_add=" - Pseudo Base",
            )
            base_val_loss = metrics_base["loss"]
            base_val_acc = metrics_base["accuracy"]

            metrics_novel = self.eval_method.evaluate(

                dataset=self.val_pseudo_novel,
                classnames=self.pseudo_novel_classes,
                desc_add=" - Pseudo Novel",
            )

            novel_val_loss = metrics_novel["loss"]
            novel_val_acc = metrics_novel["accuracy"]

        self.logger.log_validation(
            epoch,
            base_val_loss,
            base_val_acc,
            novel_val_loss,
            novel_val_acc,
            is_adv=is_adv,
        )

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
            return self.warmup_cons_lr / self.optimizer_configs[0].prompt_lr # type: ignore
        return 0.5 * (
            1
            + math.cos(
                math.pi
                * (current_epoch - self.warmup_epoch)
                / (self.max_epoch - self.warmup_epoch + 1e-7)
            )
        )

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
        novel_accuracy = test_step(model, self.test_novel, self.novel_classes, self.batch_size, self.device, label="test", base=base)
        """
        if base:
            base_metrics = self.zero_shot_pseudo_base_test_method.evaluate(
                dataset=self.test_base,
                desc_add=" - Base Zero Shot",
            )
            novel_metrics = self.zero_shot_novel_classes_test_method.evaluate(
                dataset=self.test_novel,
                desc_add=" - Novel Zero Shot",
            )
        else:
            base_metrics = self.finetuned_test_method.evaluate(
                dataset=self.test_base,
                classnames=self.base_classes,
                desc_add=" - Base Fine Tuned",
            )
            novel_metrics = self.finetuned_test_method.evaluate(
                dataset=self.test_novel,
                classnames=self.novel_classes,
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
        print(f"[DEBUG] Saving model with classnames: {self.model.prompt_learner.n_cls} classes")
        with self.model.temporary_classnames([CLASS_NAMES[idx] for idx in self.base_classes]):
            torch.save(
                self.model.state_dict(), os.path.join(path, f"{prefix}{self.run_name}.pth")
            )

    def save_mlp_adversary(self, path="./bin/cocoop", prefix=""):
        """
        Save the MLP adversary weights to disk.

        Args:
            path (str): Directory to save the MLP adversary model to.
            prefix (str): Filename prefix to distinguish models.
        """
        os.makedirs(path, exist_ok=True)
        torch.save(
            self.mlp_adversary.state_dict(),
            os.path.join(path, f"{prefix}{self.run_name}_mlp_adversary.pth"),
        )

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
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if "prompt_learner" in n and p.requires_grad
                ],
                "lr": config.prompt_lr,
            }
        ]
        if mlp_adversary is not None:
            params.append(
                {
                    "params": mlp_adversary.parameters(),
                    "lr": config.mlp_lr,
                }
            )
        return torch.optim.SGD(
            params, weight_decay=config.weight_decay, momentum=config.momentum
        )

    def split_by_classes(self, dataset, class_list):
        idxs = [i for i, (_, label) in enumerate(dataset) if label in class_list]
        return torch.utils.data.Subset(dataset, idxs)
