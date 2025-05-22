from easydict import EasyDict
import argparse

from training_systems.cocoop import CoCoOpSystem
from training_systems.coop import CoOpSystem
import torch
import os
from datetime import datetime
import pickle
from collections import Counter

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default=os.getenv("DEVICE", "cuda:0"))
    parser.add_argument('--run_name', default=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument('--using_coop', default=os.getenv("USING_COOP", "false").lower() in ("1", "true"), type=lambda x: x.lower() in ("1", "true", "yes", "true"))
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    device = args.device
    run_name = args.run_name

    print(f"Using device: {device}")

    if torch.backends.mps.is_available():
        print("⚠️ Forcing float32 due to MPS limitations")
        torch.set_default_dtype(torch.float32)

    use_coop = args.using_coop

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

    print(Counter(cls_cluster_dict.values()))
    # dictionary to map class names to cluster IDs
    # cls_cluster_dict = {
    #     "class1": 0,
    #     "class2": 1,
    #     "class3": 2,
    #     # Add more classes and their corresponding cluster IDs
    # }

    first_optimizer = EasyDict(prompt_lr=0.002, weight_decay=0.0001, momentum=0.9)  # for base training
    second_optimizer = EasyDict(prompt_lr=0.002, mlp_lr=0.004, weight_decay=0.0005, momentum=0.8)  # for adversarial training
    mlp_opt = EasyDict(hidden_dim=592, hidden_layers=3)
    if use_coop:
        train_sys = CoOpSystem(
            batch_size=10,
            device=device,
            learning_rate=0.002,
            weight_decay=0.0001,
            momentum=0.9,
            epochs=20,
            run_name=run_name,
            n_ctx=4,
            ctx_init="",
            class_token_position="end",
            csc=False,
        )
    else:
        train_sys = CoCoOpSystem(
            batch_size=10,
            device=device,
            epochs=10,
            run_name=run_name,
            n_ctx=8,
            ctx_init="",
            class_token_position="end",
            csc=False,
            lambda_kl=[0.5, 0.1],
            cls_cluster_dict=cls_cluster_dict,
            lambda_adv=0.4,
            adv_training_epochs=10,
            cnn_model=CNN,
            warmup_epoch=2,
            warmup_cons_lr=1e-5,
            using_kl_adv=False,
            optimizer_configs=[first_optimizer, second_optimizer],
            grl_lambda=5,
            mlp_opt=mlp_opt,
            skip_tests=[True, False, False]
        )

    train_sys.train()
