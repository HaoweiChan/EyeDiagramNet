#!/bin/tcsh

# EyeDiagramNet Training Script with embedded bsub command
# Usage: 
#   ./scripts/train_ew_xfmr_local.sh                    # Run without profiling
#   ./scripts/train_ew_xfmr_local.sh --profiling        # Run with profiling enabled

# Parse arguments
set profiling = 0
foreach arg ($argv)
    if ("$arg" == "--profiling" || "$arg" == "-p") then
        set profiling = 1
    endif
end

# Check if we're already in a bsub job (LSB_JOBID is set by LSF)
if ($?LSB_JOBID) then
    # We're already in a bsub job, run the actual training
    echo "=========================================="
    echo "Starting EyeDiagramNet Training Job"
    echo "Time: `date`"
    echo "Host: `hostname`"
    echo "Working Directory: `pwd`"
    echo "Job ID: $LSB_JOBID"
    if ($profiling) then
        echo "Profiling: ENABLED"
    else
        echo "Profiling: DISABLED"
    endif
    echo "=========================================="

    # Load required modules
    module load LSF/mtkgpu
    module load Python3/3.11.8_gpu_torch251
    source /proj/siaiadm/ew_predictor/.venv/sipi/bin/activate.csh

    # Install requirements if needed
    echo "Installing/updating Python requirements..."
    pip install -U -r requirements.txt

    # Configure profiling based on flag
    if ($profiling) then
        echo "Starting system monitoring..."
        python monitor_training.py --interval 10 &
        set monitor_pid = $!
        
        # Create cleanup script for tcsh
        echo "if (\$status == 0) kill -TERM $monitor_pid" > /tmp/cleanup_monitor_$$.csh
        echo "wait" >> /tmp/cleanup_monitor_$$.csh
        
        # Wait a moment for monitoring to start
        sleep 3
        
        set config_file = "configs/train_ew_xfmr.yaml"
    else
        echo "Profiling disabled - using optimized config..."
        # Create a temporary config without profiling
        sed 's/profiler:/# profiler (disabled):/' configs/train_ew_xfmr.yaml > /tmp/train_config_no_profile_$$.yaml
        set config_file = "/tmp/train_config_no_profile_$$.yaml"
    endif

    echo "Starting training..."
    echo "Command: python trainer.py fit --config $config_file"

    # Run the training
    python trainer.py fit --config $config_file
    set train_exit_code = $status

    # Cleanup
    if ($profiling) then
        echo "Cleaning up monitoring process..."
        if (-e /tmp/cleanup_monitor_$$.csh) then
            source /tmp/cleanup_monitor_$$.csh
            rm -f /tmp/cleanup_monitor_$$.csh
        endif
    else
        # Clean up temporary config
        rm -f /tmp/train_config_no_profile_$$.yaml
    endif

    echo "=========================================="
    echo "Training completed with exit code: $train_exit_code"
    echo "Time: `date`"
    echo "=========================================="

    exit $train_exit_code

else
    # We're not in a bsub job yet, submit the job
    echo "Submitting bsub job..."
    if ($profiling) then
        echo "Profiling will be ENABLED in the job"
        bsub -Is -J EyeDiagram_Prof -q ML_GPU -app PyTorch -P d_09017 -gpu "num=4" -m GPU_3090_4 -R "rusage[mem=32000]" "$0 --profiling"
    else
        echo "Profiling will be DISABLED in the job"
        bsub -Is -J EyeDiagram_Fast -q ML_GPU -app PyTorch -P d_09017 -gpu "num=4" -m GPU_3090_4 -R "rusage[mem=24000]" "$0"
    endif
endif 