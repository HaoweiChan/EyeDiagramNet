#!/bin/tcsh
#
# Train GP contour prediction model
# Usage: ./_train_contour_gp.sh [--debug]
#

# Load modules and setup environment
module load LSF/mtkgpu
module load Python3/3.12.2_gpu_torch270

# Activate virtual environment
source /proj/mtk25738/Documents/EyeDiagramNet/venv/bin/activate.csh

# Performance settings
setenv CUDA_LAUNCH_BLOCKING 0
setenv PYTORCH_CUDA_ALLOC_CONF expandable_segments:True
setenv OMP_NUM_THREADS 8

# Config file
set CONFIG_FILE = "configs/training/_train_contour_gp.yaml"

# Parse arguments
set DEBUG_MODE = 0
if ( $#argv > 0 ) then
    if ( "$argv[1]" == "--debug" ) then
        set DEBUG_MODE = 1
    endif
endif

if ( $DEBUG_MODE == 1 ) then
    # Debug mode: quick test on current node
    echo "Running in DEBUG mode (quick test on current node)"
    python3 -m ml.trainer fit --config $CONFIG_FILE \
        --trainer.fast_dev_run=5 \
        --trainer.accelerator=cpu \
        --trainer.devices=1
else
    # Production mode: submit to queue
    bsub -q ML_GPU -n 8 -R "span[hosts=1]" \
        python3 -m ml.trainer fit --config $CONFIG_FILE \
            --trainer.accelerator=cpu \
            --trainer.devices=1
endif
