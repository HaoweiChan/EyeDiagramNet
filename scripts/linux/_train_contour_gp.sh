#!/bin/tcsh
module load LSF/mtkgpu
module load Python3/3.12.2_gpu_torch270
source /proj/siaiadm/ew_predictor/.venv/sipi_torch270/bin/activate.csh

# GP training uses CPU, so no CUDA settings needed
# Set number of threads for CPU-based GP optimization
setenv OMP_NUM_THREADS 8
setenv MKL_NUM_THREADS 8

echo "Starting Gaussian Process contour training..."
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "MKL_NUM_THREADS: $MKL_NUM_THREADS"
echo "Note: GP training is CPU-based with full-batch optimization"

set python_cmd = ( \
    python3 -m ml.trainer fit --config configs/training/train_contour_gp.yaml \
    --trainer.accelerator cpu \
    --trainer.devices 1 \
    --trainer.num_nodes 1 \
)

if ( "$1" == "--debug" ) then
    set python_cmd = ( \
        $python_cmd \
        --trainer.max_epochs 1 \
    )
endif

# Submit to CPU queue instead of GPU queue
bsub -Is -J GPContour -q normal -P d_09017 $python_cmd

