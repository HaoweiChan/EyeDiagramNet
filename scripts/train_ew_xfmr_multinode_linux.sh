#!/bin/tcsh

# User-configurable variables
set GPU_NUM = 32
set TRAINER_COMMAND = "ml/trainer.py fit --config configs/training/train_ew_xfmr.yaml"

# 1. Load necessary modules for the cluster environment
module load LSF/mtkgpu
module load openmpi/4.0.3
module load Python3/3.12.2_gpu_torch270

# 2. Activate the project's Python virtual environment
source /proj/siaiadm/ew_predictor/.venv/sipi_torch270/bin/activate.csh

# 3. Check for available GPU resources
echo "Checking for available GPU hosts..."
bhosts GPU_3090_4
# bhosts GPU_A6000_8

# 4. Prepare LSF script with substituted values
set lsf_template = scripts/run_multinode.lsf
set lsf_tmp = /tmp/run_multinode_$$.lsf

cp $lsf_template $lsf_tmp

# Substitute placeholders in the temp LSF script
sed -e "s/{{GPU_NUM}}/$GPU_NUM/" -e "s#{{TRAINER_COMMAND}}#$TRAINER_COMMAND#" $lsf_tmp > ${lsf_tmp}.tmp
mv ${lsf_tmp}.tmp $lsf_tmp

# Display first 15 lines of the final LSF script for verification
echo "Generated LSF script preview (first 15 lines):"
head -15 $lsf_tmp

# 5. Submit the LSF job script
echo "Submitting LSF job script: $lsf_tmp"
bsub < $lsf_tmp