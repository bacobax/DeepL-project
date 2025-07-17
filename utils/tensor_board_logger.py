import os

from torch.utils.tensorboard.writer import SummaryWriter


def harmonic_mean(a, b):
    return 2 * (a * b) / (a + b)


class CSVLogger:
    """
    A simple logger that writes training metrics to a CSV file.

    Attributes:
        filename (str): The path to the output CSV file.
        file (file object): Open file handle for writing.
    """

    def __init__(self, filename):
        """
        Initializes the CSV logger and creates the output directory and file.

        Args:
            filename (str): Path to the CSV file for logging.
        """
        self.filename = filename
        os.makedirs(
            os.path.dirname(filename), exist_ok=True
        )  # Create folder if it doesn't exist
        self.file = open(filename, "w")
        self.file.write("epoch,base_acc,novel_acc,harmonic_mean\n")

    def log(self, epoch, base_acc, novel_acc):
        """
        Logs a training epoch's results including harmonic mean to the CSV file.

        Args:
            epoch (int): The epoch number.
            base_acc (float): Accuracy on base classes.
            novel_acc (float): Accuracy on novel classes.
        """
        hm = harmonic_mean(base_acc, novel_acc)
        self.file.write(f"{epoch},{base_acc},{novel_acc},{hm}\n")
        self.file.flush()
        print(
            f"Logged: epoch {epoch}, base_acc {base_acc}, novel_acc {novel_acc}, harmonic_mean {hm}"
        )

    def close(self):
        """
        Closes the CSV file.
        """
        self.file.close()


class BaseAndNovelMetrics:
    """
    Tracks base and novel accuracy values and their harmonic mean across epochs.

    Attributes:
        tmp (List[Tuple[int, float, float, float]]): Logged metrics per epoch.
    """

    def __init__(self):
        self.tmp = []

    def update(self, epoch, base_acc, novel_acc):
        """
        Records a new set of metrics for a specific epoch.

        Args:
            epoch (int): Epoch number.
            base_acc (float): Accuracy on base classes.
            novel_acc (float): Accuracy on novel classes.
        """
        self.tmp.append(
            (epoch, base_acc, novel_acc, harmonic_mean(base_acc, novel_acc))
        )

    def get_metrics(self):
        """
        Retrieves the collected metrics.

        Returns:
            List[Tuple[int, float, float, float]] or None: Logged metrics or None if empty.
        """
        if len(self.tmp) == 0:
            return None
        return self.tmp


class TensorboardLogger:
    """
    Handles logging of training and evaluation metrics to TensorBoard and CSV.

    Attributes:
        writer (SummaryWriter): TensorBoard writer instance.
        csv_logger (CSVLogger): CSV logger instance for persistent metric storage.
        hparams (dict): Hyperparameters dictionary.
        base_and_novel_metrics (BaseAndNovelMetrics): Metric tracker.
    """

    def __init__(self, writer: SummaryWriter):
        """
        Initializes the TensorboardLogger.

        Args:
            writer (SummaryWriter): TensorBoard writer instance.
        """
        self.writer = writer
        self.csv_logger = CSVLogger(f"{writer.log_dir}/metrics.csv")
        self.hparams = None
        self.base_and_novel_metrics = BaseAndNovelMetrics()

    def log_hparams(self, hparams: dict):
        """
        Stores hyperparameters for later logging.

        Args:
            hparams (dict): Hyperparameters dictionary.
        """
        self.hparams = hparams

    def log_training_base(self, epoch, lr, ce_loss, acc, kl_loss, total_loss):
        """
        Logs training metrics for the base training phase.

        Args:
            epoch (int): Current epoch.
            lr (float): Learning rate.
            ce_loss (float): Cross entropy loss.
            acc (float): Accuracy.
            kl_loss (float or None): KL divergence loss, optional.
            total_loss (float): Total loss.
        """
        self.writer.add_scalar("learning_rate", lr, epoch)
        self.writer.add_scalar("train_base/ce_loss", ce_loss, epoch)
        self.writer.add_scalar("train_base/ce_accuracy", acc, epoch)
        if kl_loss is not None:
            self.writer.add_scalar("train_base/kl_loss", kl_loss, epoch)
        self.writer.add_scalar("train_base/total_loss", total_loss, epoch)

    def log_training_adv(
        self, epoch, lambda_adv, ce_loss, acc, adv_loss, total_loss, kl_loss=None
    ):
        """
        Logs training metrics for the adversarial training phase.

        Args:
            epoch (int): Current epoch.
            lambda_adv (float): Adversarial loss weight.
            ce_loss (float): Cross entropy loss.
            acc (float): Accuracy.
            adv_loss (float): Adversarial loss.
            total_loss (float): Total loss.
            kl_loss (float or None): KL divergence loss, optional.
        """
        self.writer.add_scalar("lambda_adv", lambda_adv, epoch)
        self.writer.add_scalar("train_adv/ce_loss", ce_loss, epoch)
        self.writer.add_scalar("train_adv/ce_accuracy", acc, epoch)
        self.writer.add_scalar("train_adv/mlp_loss", adv_loss, epoch)
        if kl_loss is not None:
            self.writer.add_scalar("train_adv/kl_loss", kl_loss, epoch)
        self.writer.add_scalar("train_adv/total_loss", total_loss, epoch)

    def log_validation(
        self, epoch, base_loss, base_acc, novel_loss, novel_acc, is_adv=False
    ):
        """
        Logs validation metrics.

        Args:
            epoch (int): Current epoch.
            base_loss (float): Loss on base classes.
            base_acc (float): Accuracy on base classes.
            novel_loss (float): Loss on novel classes.
            novel_acc (float): Accuracy on novel classes.
            is_adv (bool): Whether validation is adversarial.
        """
        self.writer.add_scalar(f"validation_base/loss", base_loss, epoch)
        self.writer.add_scalar(f"validation_base/accuracy", base_acc, epoch)
        self.writer.add_scalar(f"validation_novel/loss", novel_loss, epoch)
        self.writer.add_scalar(f"validation_novel/accuracy", novel_acc, epoch)

        prefix = "validation_adv" if is_adv else "validation_ce"
        self.writer.add_scalar(f"{prefix}_base/loss", base_loss, epoch)
        self.writer.add_scalar(f"{prefix}_base/accuracy", base_acc, epoch)
        self.writer.add_scalar(f"{prefix}_novel/loss", novel_loss, epoch)
        self.writer.add_scalar(f"{prefix}_novel/accuracy", novel_acc, epoch)

    def log_final_metrics(self, tag, base_acc, novel_acc, step):
        """
        Logs final accuracy metrics and updates CSV and metric tracker.

        Args:
            tag (str): Tag name for TensorBoard.
            base_acc (float): Accuracy on base classes.
            novel_acc (float): Accuracy on novel classes.
            step (int): Training step or epoch index.
        """
        harmonic = harmonic_mean(base_acc, novel_acc)
        self.writer.add_scalars(
            tag,
            {
                "Harmonic Mean": harmonic,
                "Base Accuracy": base_acc,
                "Novel Accuracy": novel_acc,
            },
            global_step=step + 1,
        )

        self.csv_logger.log(step + 1, base_acc, novel_acc)

        if self.hparams is not None:
            self.base_and_novel_metrics.update(step + 1, base_acc, novel_acc)

        self.writer.flush()

    def log_test_accuracy(self, step, acc, label):
        """
        Logs test accuracy for a given label.

        Args:
            step (int): Step or epoch index.
            acc (float): Accuracy value.
            label (str): Label name for the accuracy metric.
        """
        self.writer.add_scalar(f"{label}/accuracy", acc, step)

    def close(self):
        """
        Closes the logger, writes hyperparameters and final metrics if available.
        """
        metrics = self.base_and_novel_metrics.get_metrics() or []

        """
        metric_dict = {
            "base_acc_after_base": metrics[0][1] if metrics else 0,
            "novel_acc_after_base": metrics[0][2] if metrics else 0,
            "harmonic_mean_after_base": metrics[0][3] if metrics else 0,
            "base_acc_after_adv": metrics[1][1] if metrics else 0,
            "novel_acc_after_adv": metrics[1][2] if metrics else 0,
            "harmonic_mean_after_adv": metrics[1][3] if metrics else 0,
        }"""

        if self.hparams is not None and metrics:

            tmp = {}
            # Set the prefix based on whether metrics are from base phase (index 0) or adversarial phase (index 1)
            for idx, m in enumerate(metrics):
                prefix = "after_base" if idx == 0 else "after_adv"
                tmp[f"epoch_{prefix}"] = m[0]
                tmp[f"base_acc_{prefix}"] = m[1]
                tmp[f"novel_acc_{prefix}"] = m[2]
                tmp[f"harmonic_mean_{prefix}"] = m[3]

            self.writer.add_hparams(
                hparam_dict=self.hparams,
                metric_dict=tmp,
            )
        self.writer.close()
        self.csv_logger.close()
