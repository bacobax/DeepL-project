from training_systems.cocoop import CoCoOpSystem
from training_systems.coop import CoOpSystem
import torch
import os
from datetime import datetime
import pickle


if __name__ == "__main__":
    # take the device name from DEVICE env variable
    device = os.getenv("DEVICE", "cuda:0")
    print(f"Using device: {device}")

    if torch.backends.mps.is_available():
        print("⚠️ Forcing float32 due to MPS limitations")
        torch.set_default_dtype(torch.float32)

    use_coop = os.getenv("USING_COOP", "false").lower() in ("1", "true")

    print(f"Using {'CoOp' if use_coop else 'CoCoOp'} for training")

    # load the cluster labels from the file
    CNN = "ViT-B/32"

    CNN_SAFE = CNN.replace("/", "_")
    N_CLUSTERS = 2
    VARIANCE = 0.95

    file_path = (
        f"clustering_split/cluster_labels_{N_CLUSTERS}_{VARIANCE}_{CNN_SAFE}.pkl"
    )

    cls_cluster_dict = None

    with open(file_path, "rb") as f:
        cls_cluster_dict = pickle.load(f)
        print("Loaded cls_cluster_dict:", cls_cluster_dict)

    # dictionary to map class names to cluster IDs
    # cls_cluster_dict = {
    #     "class1": 0,
    #     "class2": 1,
    #     "class3": 2,
    #     # Add more classes and their corresponding cluster IDs
    # }

    if use_coop:
        train_sys = CoOpSystem(
            batch_size=10,
            device=device,
            learning_rate=0.002,
            weight_decay=0.0001,
            momentum=0.9,
            epochs=20,
            run_name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            n_ctx=4,
            ctx_init="",
            class_token_position="end",
            csc=False,
        )
    else:
        train_sys = CoCoOpSystem(
            batch_size=10,
            device=device,
            learning_rate=0.002,
            weight_decay=0.0001,
            momentum=0.9,
            epochs=1,
            run_name=f"adv_training_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            n_ctx=4,
            ctx_init="",
            class_token_position="end",
            csc=False,
            lambda_kl=[0.5, 0.1],
            cls_cluster_dict=cls_cluster_dict,
            lambda_adv=0.5,
            adv_training_epochs=13,
            cnn_model=CNN,
            warmup_epoch=0,
            warmup_cons_lr=1e-5,
            using_kl_adv=False
        )

    train_sys.train()
