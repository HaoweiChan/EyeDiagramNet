#!/bin/tcsh
module load LSF/mtkgpu
module load Python3/3.11.8_gpu_torch251
source /proj/siaiadm/ew_predictor/.venv/sipi/bin/activate.csh

# Optimized environment variables for better performance
setenv CUDA_LAUNCH_BLOCKING 0
setenv TORCH_CUDNN_V8_API_ENABLED 1
setenv TORCH_CUDNN_V8_API_DISABLED 0

# Memory optimization
setenv PYTORCH_CUDA_ALLOC_CONF "max_split_size_mb:128"

# Disable some checks for faster execution (use with caution)
setenv TORCH_SHOW_CPP_STACKTRACES 0

# Set number of threads for data loading
setenv OMP_NUM_THREADS 4
setenv MKL_NUM_THREADS 4

echo "Starting optimized training with performance settings..."
echo "CUDA_LAUNCH_BLOCKING: $CUDA_LAUNCH_BLOCKING"
echo "PYTORCH_CUDA_ALLOC_CONF: $PYTORCH_CUDA_ALLOC_CONF"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"

bhosts GPU_3090_4
set python_cmd = ( \
    python3 -m ml.trainer fit --config configs/training/train_ew_xfmr.yaml \
    --trainer.num_nodes 1 \
    # --trainer.devices 1 \
    --trainer.limit_train_batches 1 --trainer.limit_val_batches 1 --trainer.max_epochs 10 \
)

bsub -Is -J LongJob -q ML_GPU -app PyTorch -P d_09017 -gpu "num=4" -m GPU_3090_4 $python_cmd