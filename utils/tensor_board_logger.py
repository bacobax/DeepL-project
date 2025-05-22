import os

from torch.utils.tensorboard import SummaryWriter


def harmonic_mean(a, b):
    return 2 * (a * b) / (a + b)

class CSVLogger:
    def __init__(self, filename):
        self.filename = filename
        os.makedirs(os.path.dirname(filename), exist_ok=True)  # Create folder if it doesn't exist
        self.file = open(filename, "w")
        self.file.write("epoch,base_acc,novel_acc,harmonic_mean\n")

    def log(self, epoch, base_acc, novel_acc):
        hm = harmonic_mean(base_acc, novel_acc)
        self.file.write(f"{epoch},{base_acc},{novel_acc},{hm}\n")
        print(f"Logged: epoch {epoch}, base_acc {base_acc}, novel_acc {novel_acc}, harmonic_mean {hm}")

    def close(self):
        self.file.close()

class TensorboardLogger:
    def __init__(self, writer: SummaryWriter):
        self.writer = writer
        self.csv_logger = CSVLogger(f"{writer.log_dir}/metrics.csv")

    def log_hparams(self, hparams: dict):
        self.writer.add_hparams(hparam_dict=hparams, metric_dict={})

    def log_training_base(self, epoch, lr, ce_loss, acc, kl_loss, total_loss):
        self.writer.add_scalar("learning_rate", lr, epoch)
        self.writer.add_scalar("train_base/ce_loss", ce_loss, epoch)
        self.writer.add_scalar("train_base/ce_accuracy", acc, epoch)
        self.writer.add_scalar("train_base/kl_loss", kl_loss, epoch)
        self.writer.add_scalar("train_base/total_loss", total_loss, epoch)

    def log_training_adv(self, epoch, lambda_adv, ce_loss, acc, adv_loss, total_loss, kl_loss=None):
        self.writer.add_scalar("lambda_adv", lambda_adv, epoch)
        self.writer.add_scalar("train_adv/ce_loss", ce_loss, epoch)
        self.writer.add_scalar("train_adv/ce_accuracy", acc, epoch)
        self.writer.add_scalar("train_adv/mlp_loss", adv_loss, epoch)
        if kl_loss is not None:
            self.writer.add_scalar("train_adv/kl_loss", kl_loss, epoch)
        self.writer.add_scalar("train_adv/total_loss", total_loss, epoch)

    def log_validation(self, epoch, base_loss, base_acc, novel_loss, novel_acc, is_adv=False):
        prefix = "validation_adv" if is_adv else "validation"
        self.writer.add_scalar(f"{prefix}_base/loss", base_loss, epoch)
        self.writer.add_scalar(f"{prefix}_base/accuracy", base_acc, epoch)
        self.writer.add_scalar(f"{prefix}_novel/loss", novel_loss, epoch)
        self.writer.add_scalar(f"{prefix}_novel/accuracy", novel_acc, epoch)

    def log_final_metrics(self, tag, base_acc, novel_acc, step):
        self.writer.add_scalars(tag, {
            "Harmonic Mean": harmonic_mean(base_acc, novel_acc),
            "Base Accuracy": base_acc,
            "Novel Accuracy": novel_acc,
        }, global_step=step + 1)
        self.csv_logger.log(step + 1, base_acc, novel_acc)

    def log_test_accuracy(self, step, acc, label):
        self.writer.add_scalar(f"{label}/accuracy", acc, step)

    def close(self):
        self.writer.close()
        self.csv_logger.close()
