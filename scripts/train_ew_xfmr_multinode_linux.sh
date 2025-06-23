#!/bin/tcsh

# 1. Load necessary modules for the cluster environment
module load LSF/mtkgpu
module load openmpi/4.0.3
module load Python3/3.11.8_gpu_torch251

# 2. Activate the project's Python virtual environment
source /proj/siaiadm/ew_predictor/.venv/sipi/bin/activate.csh

# 3. Check for available GPU resources
echo "Checking for available GPU hosts..."
bhosts GPU_3090_4
# bhosts GPU_A6000_8

# 4. Submit the LSF job script
echo "Submitting LSF job script: scripts/run_multinode.lsf"
bsub < scripts/run_multinode.lsf