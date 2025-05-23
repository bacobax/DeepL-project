import torch


def check_gradients(model):
    print("=== Checking gradients ===")
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                print(f"{name}: ❌ No gradient")
            elif torch.all(param.grad == 0):
                print(f"{name}: ⚠️ Zero gradient")
            else:
                print(f"{name}: ✅ Gradient flowing")



