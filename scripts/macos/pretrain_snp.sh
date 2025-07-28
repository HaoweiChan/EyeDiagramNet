#!/bin/bash
# Script to pretrain SNP encoder using self-supervised learning

# Set PYTHONPATH to ensure modules are found
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# The configuration file for this specific pretraining task
CONFIG_FILE="configs/training/pretrain_snp_ssl.yaml"

# Pass all command line arguments after the script name to the trainer
# Example: ./scripts/macos/pretrain_snp.sh fit --trainer.max_epochs 5
ARGS="$@"

echo "========================================="
echo "SNP Self-Supervised Learning Pretraining"
echo "========================================="
echo "Config: ${CONFIG_FILE}"
echo "Args: ${ARGS}"
echo "========================================="

# Run the pretraining using the generic trainer
python -m ml.trainer ${ARGS} --config ${CONFIG_FILE}

# Check exit status
if [ $? -eq 0 ]; then
    echo "========================================="
    echo "Pretraining completed successfully!"
    echo "========================================="
else
    echo "========================================="
    echo "Pretraining failed!"
    echo "========================================="
    exit 1
fi