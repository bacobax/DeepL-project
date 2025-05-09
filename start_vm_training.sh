#!/bin/bash

# Exit script on error
set -e

# === CONFIG ===
ENV_NAME=deepl
ENV_YAML=environment.yml
PYTHON_SCRIPT=main.py  # <-- Replace with your actual entry point
LOG_DIR=logs
export DEVICE="mps"  # or "cpu" if no GPU available
export USING_COOP="true"
# === Activate Conda ===
echo "Activating Conda..."
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

# === Optional: Create log directory ===
mkdir -p $LOG_DIR

# === Run training ===
echo "Starting training on $DEVICE..."
python $PYTHON_SCRIPT --device $DEVICE | tee $LOG_DIR/train_$(date +"%Y%m%d_%H%M%S").log