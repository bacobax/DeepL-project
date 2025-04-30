from easydict import EasyDict

from model.custom_clip import CustomCLIP
from utils.datasets import embed_dataset_classnames, get_data, base_novel_categories, split_data
from utils.training import test_step, training_step
import clip
from torch.utils.tensorboard import SummaryWriter
from torch.optim import SGD
from torch import nn
from model.prompt_learner import PromptLearner
class CoCoOpSystem:
    def __init__(self,
                 dataset_name="cifar10",
                 batch_size=16,
                 num_classes=10,
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
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_classes = num_classes
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
        self.writer = SummaryWriter(log_dir=f"runs/{run_name}")

        # Get dataloaders

        clip_model, preprocess = clip.load("RN50")
        self.train_set, self.val_set, self.test_set = get_data(dataset_name, transform=preprocess)


        # split classes into base and novel
        self.base_classes, self.novel_classes = base_novel_categories(self.train_loader)

        # split the three datasets
        self.train_base, _ = split_data(self.train_set, self.base_classes)
        self.val_base, _ = split_data(self.val_set, self.base_classes)
        self.test_base, self.test_novel = split_data(self.test_set, self.base_classes)

        self.classnames, _ = embed_dataset_classnames(dataset_name, preprocess=preprocess, model=clip_model)

        resolution = clip_model.visual.input_resolution

        cfg = EasyDict()
        # Training configuration
        cfg.TRAINER = EasyDict()
        cfg.TRAINER.COCOOP = EasyDict()
        cfg.TRAINER.COCOOP.N_CTX = self.n_ctx  # Number of context tokens
        cfg.TRAINER.COCOOP.CTX_INIT = self.ctx_init  # Leave empty for random initialization
        cfg.TRAINER.COCOOP.PREC = "fp16"  # Precision for meta network
        cfg.INPUT = EasyDict()
        cfg.INPUT.SIZE = [resolution, resolution]  # Must match CLIP model's input resolution

        # Instantiate the network and move it to the chosen device (GPU)
        self.model = CustomCLIP(
            classnames=self.classnames,
            cfg=cfg,
            clip_model=clip_model,
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
        print("Before training:")
        self.compute_evaluation(-1)
        # For each epoch, train the network and then compute evaluation results
        for e in range(self.epochs):
            train_loss, train_accuracy = training_step(self.model, self.train_loader, self.optimizer,
                                                       self.cost_function)
            val_loss, val_accuracy = test_step(self.model, self.val_loader, self.cost_function)

            self.log_values(e, train_loss, train_accuracy, "train")
            self.log_values(e, val_loss, val_accuracy, "validation")

        print("After training:")
        self.compute_evaluation(self.epochs)
        self.writer.close()

    def compute_evaluation(self, epoch_idx):
        train_loss, train_accuracy = test_step(self.model, self.train_loader, self.cost_function)
        val_loss, val_accuracy = test_step(self.model, self.val_loader, self.cost_function)
        test_loss, test_accuracy = test_step(self.model, self.test_loader, self.cost_function)

        # Log to TensorBoard
        self.log_values(epoch_idx, train_loss, train_accuracy, "train")
        self.log_values(epoch_idx, val_loss, val_accuracy, "validation")
        self.log_values(epoch_idx, test_loss, test_accuracy, "test")

        print(f"\tTraining loss {train_loss:.5f}, Training accuracy {train_accuracy:.2f}")
        print(f"\tValidation loss {val_loss:.5f}, Validation accuracy {val_accuracy:.2f}")
        print(f"\tTest loss {test_loss:.5f}, Test accuracy {test_accuracy:.2f}")

        return train_loss, train_accuracy, val_loss, val_accuracy, test_loss, test_accuracy

    def get_optimizer(self, lr, wd, momentum):
        optimizer = SGD([
            {"params": self.model.parameters()}
        ], lr=lr, weight_decay=wd, momentum=momentum)

        return optimizer

    def log_values(self, step, loss, accuracy, prefix):
        self.writer.add_scalar(f"{prefix}/loss", loss, step)
        self.writer.add_scalar(f"{prefix}/accuracy", accuracy, step)
