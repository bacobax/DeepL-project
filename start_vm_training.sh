#!/bin/bash

# Exit script on error
set -e

# === CONFIG ===
ENV_NAME=deepl
ENV_YAML=environment.yml
PYTHON_SCRIPT=main.py  # <-- Replace with your actual entry point
LOG_DIR=logs

DEVICE="cuda"  # or "cpu" if no GPU available
USING_COOP="false"
RUN_PREFIX="NO_KL_ADV_IMG_FT_8_CTX"
HPARAMS_CONF="balanced"
DEBUG="true"

RUN_NAME="${RUN_PREFIX}_$(date +"%Y%m%d_%H%M%S")"
HPARAMS_DIR="hparams_configs"
HPARAMS_FULL_PATH="$HPARAMS_DIR/$HPARAMS_CONF.yaml"

# === Activate Conda ===
echo "Activating Conda..."
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

# === Optional: Create log directory ===
mkdir -p $LOG_DIR

# === Run training ===
echo "Starting training on $DEVICE..."
if python $PYTHON_SCRIPT --debug $DEBUG --config $HPARAMS_FULL_PATH --device $DEVICE --run_name $RUN_NAME --using_coop $USING_COOP | tee $LOG_DIR/train_$(date +"%Y%m%d_%H%M%S").log; then
  # Git operations
  cd "runs/CoCoOp/"
  git add "$RUN_NAME"
  git commit -m "Add logs for $RUN_NAME"
  git push
  cd ../..
fi