#!/bin/tcsh

# EyeDiagramNet Training Script with embedded bsub command
# Usage: 
#   tcsh scripts/train_ew_xfmr_local.sh                    # Run without profiling
#   tcsh scripts/train_ew_xfmr_local.sh --profiling        # Run with profiling enabled

# Parse arguments
set profiling = 0
foreach arg ($argv)
    if ("$arg" == "--profiling" || "$arg" == "-p") then
        set profiling = 1
    endif
end

# Load required modules first (needed for bsub to be available)
module load LSF/mtkgpu
module load Python3/3.11.8_gpu_torch251
source /proj/siaiadm/ew_predictor/.venv/sipi/bin/activate.csh

# Check if bsub is available (HPC environment) or if we're on macOS/local environment
set bsub_available = 0
which bsub >& /dev/null
if ($status == 0) then
    set bsub_available = 1
endif

# Check if we're already in a bsub job (LSB_JOBID is set by LSF) OR if bsub is not available (local run)
if ($?LSB_JOBID || $bsub_available == 0) then
    # We're already in a bsub job OR running locally, run the actual training
    echo "=========================================="
    if ($bsub_available == 0) then
        echo "Starting EyeDiagramNet Training Job (Local)"
    else
        echo "Starting EyeDiagramNet Training Job"
        echo "Job ID: $LSB_JOBID"
    endif
    echo "Time: `date`"
    echo "Host: `hostname`"
    echo "Working Directory: `pwd`"
    if ($profiling) then
        echo "Profiling: ENABLED"
    else
        echo "Profiling: DISABLED"
    endif
    echo "=========================================="

    # Load required modules (only if available - HPC environment)
    if ($bsub_available == 1) then
        # Modules already loaded at top of script
        echo "Running on HPC environment with modules loaded"
    else
        # Local environment - check for virtual environment
        if (! $?VIRTUAL_ENV) then
            echo "Warning: No virtual environment detected. Consider activating your Python environment."
        endif
    endif

    # Configure profiling based on flag
    if ($profiling) then
        echo "Starting system monitoring..."
        if (-f "monitor_training.py") then
            python monitor_training.py --interval 10 >& /tmp/monitor_$$.log &
            # tcsh doesn't have $!, so we'll use jobs and ps to manage the process
            set monitor_started = 1
            
            # Wait a moment for monitoring to start
            sleep 3
        else
            echo "Warning: monitor_training.py not found, skipping monitoring"
            set monitor_started = 0
        endif
        
        echo "Profiling ENABLED - adding profiler arguments..."
    else
        echo "Profiling DISABLED - using optimized config..."
        set monitor_started = 0
    endif

    # Base python command
    set python_cmd = ( \
        python3 trainer.py fit --config configs/train_ew_xfmr.yaml \
        --trainer.devices 1 --trainer.num_nodes 1 --trainer.limit_train_batches 1 \
        --trainer.limit_val_batches 1 --trainer.max_epochs 10 \
        --trainer.profiler null \
    )

    # Add profiler arguments if profiling is enabled
    if ($profiling) then
        set python_cmd = ( $python_cmd \
            --trainer.profiler.class_path lightning.pytorch.profilers.PyTorchProfiler \
            --trainer.profiler.init_args.dirpath ./profiler_logs \
            --trainer.profiler.init_args.filename perf_logs \
            --trainer.profiler.init_args.export_to_chrome true \
            --trainer.profiler.init_args.sort_by_key cuda_time_total \
            --trainer.profiler.init_args.group_by_input_shapes true \
            --trainer.profiler.init_args.record_module_names true \
            --trainer.profiler.init_args.row_limit 100 \
        )
    endif

    echo "Starting training..."
    echo "Command: $python_cmd"

    # Run the training
    $python_cmd
    set train_exit_code = $status

    # Cleanup
    if ($profiling == 1 && $monitor_started == 1) then
        echo "Cleaning up monitoring process..."
        # Find and kill the monitoring process using tcsh-compatible method
        set monitor_pids = (`ps -u $USER -o pid,comm | grep monitor_training | awk '{print $1}'`)
        if ($#monitor_pids > 0) then
            foreach pid ($monitor_pids)
                kill -TERM $pid >& /dev/null
            end
            sleep 2
            # Force kill if still running
            set remaining_pids = (`ps -u $USER -o pid,comm | grep monitor_training | awk '{print $1}'`)
            if ($#remaining_pids > 0) then
                foreach pid ($remaining_pids)
                    kill -KILL $pid >& /dev/null
                end
            endif
        endif
        # Clean up log file
        rm -f /tmp/monitor_$$.log
    endif
    
    echo "=========================================="
    echo "Training completed with exit code: $train_exit_code"
    echo "Time: `date`"
    echo "=========================================="

    exit $train_exit_code

else
    # We're not in a bsub job yet and bsub is available, submit the job
    echo "Submitting bsub job..."
    if ($profiling == 1) then
        echo "Profiling will be ENABLED in the job"
        bsub -Is -J EyeDiagram_Prof -q ML_GPU -app PyTorch -P d_09017 -gpu "num=4" -m GPU_3090_4 -R "rusage[mem=32000]" "tcsh $0 --profiling"
    else
        echo "Profiling will be DISABLED in the job"
        bsub -Is -J EyeDiagram_Fast -q ML_GPU -app PyTorch -P d_09017 -gpu "num=4" -m GPU_3090_4 -R "rusage[mem=24000]" "tcsh $0"
    endif
endif 