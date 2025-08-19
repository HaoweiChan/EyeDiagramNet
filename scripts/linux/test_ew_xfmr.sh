#!/bin/tcsh
module load LSF/mtkgpu
module load Python3/3.12.2_gpu_torch270
source /proj/siaiadm/ew_predictor/.venv/sipi_torch270/bin/activate.csh

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

if ($#argv == 0) then
    echo "Usage: $0 <path_to_model_checkpoint>"
    exit 1
endif

echo "Starting model testing with performance settings..."

bhosts GPU_3090_4
set python_cmd = ( \
    python3 -m ml.trainer test --config configs/training/test_ew_xfmr.yaml \
    --model.init_args.ckpt_path $1 \
    --trainer.num_nodes 1 \
)

bsub -Is -J LongJob -q ML_GPU -app PyTorch -P d_09017 -gpu "num=4" -m GPU_3090_4 $python_cmd
