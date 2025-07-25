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
RUN_PREFIX="from_yaml"
HPARAMS_CONFS=(
  "base_kl_v2_80_20_kl_03_rot_period_3_4_ctx"
  "base_kl_v2_80_20_kl_03_rot_period_4_4_ctx"
  "base_kl_v2_80_20_kl_03_rot_period_rel_4_ctx"
  "base_kl_v2_80_20_kl_03_rot_period_random_8_ctx"
  "base_kl_v2_80_20_kl_01_rot_period_random_8_ctx"
  "base_kl_v2_80_20_kl_03_rot_period_random_4_ctx"
  "base_kl_v2_80_20_kl_01_rot_period_random_4_ctx"
)
DEBUG="false"

# This will be set inside the loop
HPARAMS_DIR="hparams_configs/report"

# === Activate Conda ===
echo "Activating Conda..."
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

# === Optional: Create log directory ===
mkdir -p $LOG_DIR

# === Run training ===
for HPARAMS_CONF in "${HPARAMS_CONFS[@]}"; do
    HPARAMS_FULL_PATH="$HPARAMS_DIR/$HPARAMS_CONF.yaml"
    RUN_NAME="${RUN_PREFIX}_${HPARAMS_CONF}_$(date +"%Y%m%d_%H%M%S")"
    
    echo "Starting training on $DEVICE with config $HPARAMS_CONF..."
    if python $PYTHON_SCRIPT --debug $DEBUG --config $HPARAMS_FULL_PATH --device $DEVICE --run_name $RUN_NAME --using_coop $USING_COOP | tee $LOG_DIR/train_${HPARAMS_CONF}_$(date +"%Y%m%d_%H%M%S").log; then
        # Git operations
        cd "runs/report/"
        git add "$RUN_NAME"
        git commit -m "Add logs for $RUN_NAME"
        git push
        cd ../..
    fi
done