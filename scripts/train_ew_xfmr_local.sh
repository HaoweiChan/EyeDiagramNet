#!/bin/bash

# EyeDiagramNet Training Script for bsub (Local Version)
# Usage: bsub -Is -J LongJob -q ML_GPU -app PyTorch -P d_09017 -gpu "num=4" -m GPU_3090_4 < scripts/train_ew_xfmr_local.sh

echo "=========================================="
echo "Starting EyeDiagramNet Training Job"
echo "Time: $(date)"
echo "Host: $(hostname)"
echo "Working Directory: $(pwd)"
echo "=========================================="

# Load required modules
module load LSF/mtkgpu
module load Python3/3.11.8_gpu_torch251
source /proj/siaiadm/ew_predictor/.venv/sipi/bin/activate.csh

# Install requirements if needed
echo "Installing/updating Python requirements..."
pip install -U -r requirements.txt

# Start system monitoring in background
echo "Starting system monitoring..."
python monitor_training.py --interval 10 &
MONITOR_PID=$!

# Function to cleanup background processes
cleanup() {
    echo "Cleaning up background processes..."
    if kill -0 $MONITOR_PID 2>/dev/null; then
        echo "Stopping monitor (PID: $MONITOR_PID)"
        kill -TERM $MONITOR_PID
        wait $MONITOR_PID 2>/dev/null
    fi
    echo "Cleanup complete."
}

# Set trap for cleanup on script exit
trap cleanup EXIT INT TERM

# Wait a moment for monitoring to start
sleep 3

echo "Starting training..."
echo "Command: python trainer.py fit --config configs/train_ew_xfmr.yaml"

# Run the training
python trainer.py fit --config configs/train_ew_xfmr.yaml

# Capture training exit code
TRAIN_EXIT_CODE=$?

echo "=========================================="
echo "Training completed with exit code: $TRAIN_EXIT_CODE"
echo "Time: $(date)"

# The trap will handle cleanup automatically
exit $TRAIN_EXIT_CODE 