#!/bin/tcsh
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

echo "Starting model testing with performance settings..."

set bsub_flag = 0
set batch_args = "--data.batch_size 120"

# Parse command line arguments
foreach arg ($argv)
    if ("$arg" == "--bsub") then
        set bsub_flag = 1
        break
    endif
end

# Configure GPU and batch size if using bsub
if ($bsub_flag == 1) then
    # Load LSF module for bsub
    module load LSF/mtkgpu
    
    # Check if any GPU 3090 is available
    set gpu_3090_available = `bhosts GPU_3090_4 | awk '$2 == "ok" && $5 == 0 {print $1}' | wc -l`
    echo "Available 3090 GPUs: ${gpu_3090_available}"

    set bsub_script = "bsub -Is -J LongJob -q ML_GPU -app PyTorch -P d_09017"
    if ($gpu_3090_available > 0) then
        set gpu_arg = "-m GPU_3090_4 -gpu num=4"
        echo "Using GPU_3090_4 with batch size 120"
    else
        set gpu_arg = "-m GPU_2080 -gpu num=1"
        set batch_args = "--data.batch_size 60"
        echo "Falling back to GPU_2080 with batch size 60"
    endif
endif

# Build the python command with user config
set user_config = "configs/user/infer_setting.yaml"
set python_cmd = ( \
    python3 -m ml.trainer predict \
    --config configs/training/infer_ew_xfmr.yaml \
    --user_config $user_config \
    --trainer.num_nodes 1 \
    $batch_args \
)

# Execute with or without bsub
if ($bsub_flag == 1) then
    echo "Submitting job with bsub..."
    bhosts GPU_3090_4
    $bsub_script $gpu_arg $python_cmd
else
    echo "Running directly..."
    eval "$python_cmd"
endif