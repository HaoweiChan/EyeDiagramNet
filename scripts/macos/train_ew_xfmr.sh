#!/bin/bash
# Script to train the Eye-Width Transformer model on a macOS machine.

# Set PYTHONPATH to ensure modules are found
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# The configuration file for this specific training task
CONFIG_FILE="configs/training/train_ew_xfmr.yaml"

# Pass all command line arguments after the script name to the trainer
# Example: ./scripts/macos/train_ew_xfmr.sh fit --trainer.max_epochs 5
ARGS="$@"

echo "========================================="
echo "Eye-Width Transformer Training (macOS)"
echo "========================================="
echo "Config: ${CONFIG_FILE}"
echo "User Args: ${ARGS}"
echo "NOTE: Overriding trainer settings from config for local macOS execution."
echo "========================================="

# Run the training using the generic trainer
# We override trainer settings for local execution on macOS.
# The YAML config is for multi-node/multi-gpu, so we force single-device execution.
python -m ml.trainer \
    ${ARGS} \
    --config ${CONFIG_FILE} \
    --trainer.accelerator=auto \
    --trainer.strategy=auto \
    --trainer.devices=1 \
    --trainer.num_nodes=1

# Check exit status
if [ $? -eq 0 ]; then
    echo "========================================="
    echo "Training completed successfully!"
    echo "========================================="
else
    echo "========================================="
    echo "Training failed!"
    echo "========================================="
    exit 1
fi 