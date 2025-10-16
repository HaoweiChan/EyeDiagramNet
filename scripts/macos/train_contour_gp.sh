#!/bin/bash
# Script to train the Gaussian Process contour prediction model on a macOS machine.

# Set PYTHONPATH to ensure modules are found
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# The configuration file for this specific training task
CONFIG_FILE="configs/training/_train_contour_gp.yaml"

# Pass all command line arguments after the script name to the trainer
# Example: ./scripts/macos/train_contour_gp.sh fit --trainer.max_epochs 1
ARGS="$@"

echo "========================================="
echo "Gaussian Process Contour Training (macOS)"
echo "========================================="
echo "Config: ${CONFIG_FILE}"
echo "User Args: ${ARGS}"
echo "NOTE: GP training uses full-batch CPU optimization."
echo "========================================="

# Run the training using the generic trainer
# GP training is CPU-based and uses full batches
python -m ml.trainer \
    ${ARGS} \
    --config ${CONFIG_FILE} \
    --trainer.accelerator=cpu \
    --trainer.devices=1 \
    --trainer.num_nodes=1

# Check exit status
if [ $? -eq 0 ]; then
    echo "========================================="
    echo "GP Training completed successfully!"
    echo "========================================="
else
    echo "========================================="
    echo "GP Training failed!"
    echo "========================================="
    exit 1
fi

