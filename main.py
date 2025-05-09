from training_systems.cocoop import CoCoOpSystem
from training_systems.coop import CoOpSystem
import torch
import os
from datetime import datetime
if __name__ == "__main__":
    #take the device name from DEVICE env variable
    device = os.getenv("DEVICE", "cuda:0")
    print(f"Using device: {device}")

    if torch.backends.mps.is_available():
        print("⚠️ Forcing float32 due to MPS limitations")
        torch.set_default_dtype(torch.float32)

    use_coop = os.getenv("USING_COOP", "false").lower() in ("1", "true")

    print(f"Using {'CoOp' if use_coop else 'CoCoOp'} for training")

    train_cls = CoOpSystem if use_coop else CoCoOpSystem

    train_sys = train_cls(
        batch_size=16,
        device=device,
        learning_rate=5e-4,
        weight_decay=0.0001,
        momentum=0.9,
        epochs=1,
        run_name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        n_ctx=4,
        ctx_init="",
        class_token_position="end",
        csc=False,
    )
    
    train_sys.train()
