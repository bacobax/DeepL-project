def harmonic_mean(a, b):
    return 2 * (a * b) / (a + b)


class TensorboardLogger:
    def __init__(self, writer):
        self.writer = writer

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

    def log_test_accuracy(self, step, acc, label):
        self.writer.add_scalar(f"{label}/accuracy", acc, step)
