from training_system import CoCoOpSystem
import torch
import os
if __name__ == "__main__":
    #take the device name from DEVICE env variable
    device = os.getenv("DEVICE", "cuda:0")
    print(f"Using device: {device}")

    if torch.backends.mps.is_available():
        print("⚠️ Forcing float32 due to MPS limitations")
        torch.set_default_dtype(torch.float32)
    train_sys = CoCoOpSystem(
        batch_size=16,
        num_classes=10,
        device=device,
        learning_rate=0.01,
        weight_decay=0.0005,
        momentum=0.9,
        epochs=30,
        run_name="exp2",
        n_ctx=4,
        ctx_init="",
        class_token_position="end",
        csc=False,
    )
    
    train_sys.train()
