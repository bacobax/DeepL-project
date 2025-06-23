import yaml
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
    parser.add_argument('--using_coop', default=False, type=lambda x: x.lower() in ("1", "true", "yes", "true"))
    parser.add_argument('--config', default="train_config.yaml")
    parser.add_argument('--debug', default=True, type=lambda x: x.lower() in ("1", "true", "yes", "true"))
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command-line arguments
    # Assign parsed arguments to variables
    # Display which device is being used
    # Handle MPS backend by setting default tensor type to float32
    # Indicate whether CoOp or CoCoOp is used for training
    # Load training configuration from YAML file

    # Initialize and train using CoOpSystem if specified in arguments
    # Initialize and train using CoCoOpSystem otherwise
    args = parse_args()

    device = args.device
    run_name = args.run_name
    debug = args.debug
    use_coop = args.using_coop

    print(f"Using device: {device}")

    if torch.backends.mps.is_available():
        print("\u26a0\ufe0f Forcing float32 due to MPS limitations")
        torch.set_default_dtype(torch.float32)

    print(f"Using {'CoOp' if use_coop else 'CoCoOp'} for training")

    # Load hyperparameters from YAML
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    if use_coop:
        coop_cfg = config['coop']
        train_sys = CoOpSystem(
            device=device,
            run_name=run_name,
            **coop_cfg
        )
    else:
        cocoop_cfg = config['cocoop']
        train_sys = CoCoOpSystem(
            device=device,
            run_name=run_name,
            debug=debug,
            hparams_file=args.config,
            **cocoop_cfg
        )

    train_sys.train()
