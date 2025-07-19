"""
This module defines the CoOpSystem class for training and evaluating a prompt-tuned CLIP model using the CoOp method.
It includes data loading, training with early stopping, evaluation, model saving/loading, and logging to TensorBoard.
"""

import os

import torch
from easydict import EasyDict
from tqdm import tqdm

from model.coop.custom_clip import CustomCLIPCoOp
from utils.datasets import get_data, base_novel_categories, split_data, CLASS_NAMES
from utils.training_coop import test_step, training_step, eval_step
import clip
from torch.utils.tensorboard.writer import SummaryWriter
from torch.optim import SGD, Adam
from torch import nn


class CoOpSystem:
    """
    Implements the CoOp prompt tuning system for training and evaluating CLIP-based models.

    Attributes:
        batch_size (int): Number of samples per training batch.
        device (str): Device identifier (e.g., "cuda:0") to run training on.
        learning_rate (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay used in the optimizer.
        momentum (float): Momentum term (if applicable).
        epochs (int): Number of training epochs.
        run_name (str): Identifier for the experiment run (used for logging and file naming).
        n_ctx (int): Number of context tokens for prompt tuning.
        ctx_init (str): Initialization string for context tokens.
        class_token_position (str): Position of the class token in the prompt.
        csc (bool): Whether to use class-specific context.
    """
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

        # Create a logger for the experiment
        self.writer = SummaryWriter(log_dir=f"runs/CoOp/{run_name}")
        self.writer.add_scalar(f"lr", self.learning_rate, 0)
        self.writer.add_scalar(f"momentum", self.momentum, 0)

        # Get dataloaders

        self.clip_model, _ = clip.load("ViT-B/16", device=self.device)
        self.clip_model = self.clip_model.to(self.device)
        self.clip_model = self.clip_model.float()
        resolution = self.clip_model.visual.input_resolution
        self.train_set, self.val_set, self.test_set = get_data(resolution=resolution)

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
        cfg.TRAINER.COOP = EasyDict()
        cfg.TRAINER.COOP.N_CTX = self.n_ctx  # Number of context tokens
        cfg.TRAINER.COOP.CTX_INIT = self.ctx_init  # Leave empty for random initialization
        cfg.INPUT = EasyDict()
        cfg.INPUT.SIZE = [resolution, resolution]  # Must match CLIP model's input resolution

        # Instantiate the network and move it to the chosen device (GPU)
        self.model = CustomCLIPCoOp(
            classnames=[CLASS_NAMES[idx] for idx in self.base_classes],
            cfg=cfg,
            clip_model=self.clip_model,
        ).to(device)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Total trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

        self.optimizer = self.get_optimizer(self.learning_rate, self.weight_decay, self.momentum)
        self.cost_function = nn.CrossEntropyLoss()

    def train(self):
        """
        Trains the CoOp model on the base classes with early stopping and logs performance metrics.
        Saves the best model and computes evaluation at the end of training.
        """
        print("Before training:")
        print("Training the model...")
        print_epoch_interval = 2
        pbar = tqdm(total=self.epochs, desc="OVERALL TRAINING", position=0, leave=True)

        best_val_acc = 0.0
        patience = 5
        counter = 0
        best_model_state = None

        for e in range(self.epochs):
            base_train_loss, base_train_accuracy = training_step(
                model=self.model,
                dataset=self.train_base,
                optimizer=self.optimizer,
                batch_size=self.batch_size,
                classnames=self.base_classes,
                device=self.device,
            )

            if e % print_epoch_interval == 0:
                base_val_loss, base_val_accuracy = eval_step(
                    model=self.model,
                    dataset=self.val_base,
                    cost_function=self.cost_function,
                    new_classnames=self.base_classes,
                    device=self.device,
                    batch_size=self.batch_size,
                )

                self.log_values(e, base_train_loss, base_train_accuracy, "train_base")
                self.log_values(e, base_val_loss, base_val_accuracy, "validation_base")

                pbar.set_postfix(train_acc=base_train_accuracy, val_acc=base_val_accuracy)

                # Early stopping check
                if base_val_accuracy > best_val_acc:
                    best_val_acc = base_val_accuracy
                    counter = 0
                    best_model_state = self.model.state_dict()
                else:
                    counter += 1
                    if counter >= patience:
                        print(f"Early stopping at epoch {e}, best validation accuracy: {best_val_acc:.4f}")
                        break

            pbar.update(1)

        # Restore best model if early stopped
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        print("After training:")
        self.compute_evaluation(self.epochs)
        self.writer.close()

        self.save_model()
        self.save_prompt_learner()

    def save_model(self, path="./bin/coop"):
        """
        Saves the entire model's state dictionary to disk under the specified path.

        Args:
            path (str): Directory path where the model checkpoint will be saved.
        """
        #create folder if not exist
        if not os.path.exists(path):
            os.makedirs(path)
        # Save the model
        torch.save(self.model.state_dict(), os.path.join(path, f"{self.run_name}.pth"))

    def save_prompt_learner(self, path="./bin/coop"):
        """
        Saves only the prompt learner component of the model to disk.

        Args:
            path (str): Directory path where the prompt learner checkpoint will be saved.
        """
        # Create folder if not exist
        if not os.path.exists(path):
            os.makedirs(path)
        # Save only the self.ctx parameter of the prompt learner
        ctx_state = {"ctx": self.model.prompt_learner.ctx.detach().cpu()}
        torch.save(ctx_state, os.path.join(path, f"{self.run_name}_prompt_learner.pth"))

    def load_model(self, path="./bin"):
        """
        Loads a saved model checkpoint from disk and sets the model to evaluation mode.

        Args:
            path (str): Directory path from which the model checkpoint will be loaded.
        """
        # Load the model
        self.model.load_state_dict(torch.load(os.path.join(path, f"{self.run_name}.pth")))
        self.model.eval()
        print(f"Model loaded from {path}")

    def compute_evaluation(self, epoch_idx, base=False):
        """
        Evaluates the model (or zero-shot CLIP if base=True) on the base test set and logs accuracy.

        Args:
            epoch_idx (int): Index of the current epoch for logging.
            base (bool): If True, use zero-shot CLIP model for evaluation instead of the trained model.

        Returns:
            float: Accuracy on the base test set.
        """
        base_accuracy = test_step(
            self.model if not base else self.clip_model, 
            self.test_base, 
            self.batch_size, 
            self.device, 
            self.base_classes,
            label="test", 
            base=base
        )
        # Log to TensorBoard
        self.log_value(epoch_idx,  base_accuracy, "base_classes")

        return base_accuracy

    def get_optimizer(self, lr, wd, momentum):
        """
        Instantiates and returns the optimizer for the model parameters.

        Args:
            lr (float): Learning rate.
            wd (float): Weight decay.
            momentum (float): Momentum term (unused for Adam optimizer).

        Returns:
            torch.optim.Optimizer: Configured Adam optimizer instance.
        """
        optimizer = Adam([
            {
                "params": self.model.parameters()
            }
        ], lr=lr, weight_decay=wd)

        return optimizer

    def log_value(self, step,  accuracy, prefix):
        """
        Logs a single scalar value (accuracy) to TensorBoard.

        Args:
            step (int): Training step or epoch index.
            accuracy (float): Accuracy value to log.
            prefix (str): Tag prefix to categorize the metric in TensorBoard.
        """
        self.writer.add_scalar(f"{prefix}/accuracy", accuracy, step)

    def log_values(self, step, loss, accuracy, prefix):
        """
        Logs both loss and accuracy values to TensorBoard.

        Args:
            step (int): Training step or epoch index.
            loss (float): Loss value to log.
            accuracy (float): Accuracy value to log.
            prefix (str): Tag prefix to categorize the metrics in TensorBoard.
        """
        self.writer.add_scalar(f"{prefix}/loss", loss, step)
        self.writer.add_scalar(f"{prefix}/accuracy", accuracy, step)