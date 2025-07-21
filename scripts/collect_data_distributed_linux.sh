#!/bin/tcsh
# Distributed Data Collection Script for Linux (Production)
# Submits ONE bsub job that runs the Python-based distributed collector.

# Load required modules for cluster environment
module load LSF/mtkgpu
module load Python3/3.11.8_gpu_torch251
source /proj/siaiadm/ew_predictor/.venv/sipi/bin/activate.csh

echo "=========================================="
echo "EyeDiagramNet - Distributed Data Collection (Linux)"
echo "Single Machine, Python-Managed Collectors"
echo "=========================================="

# Default configuration
set cfg_file = "${1}"
if ( "$cfg_file" == "" ) then
    set cfg_file = "configs/data/default.yaml"
endif

# Validate config file
if ( ! -f "$cfg_file" ) then
    echo "ERROR: Config file '$cfg_file' not found!"
    echo "Usage: $0 [config_file]"
    exit 1
endif

echo "Configuration: $cfg_file"
echo "Strategy: One bsub job running a Python script to manage all collectors."
echo ""

# Job submission settings
set job_queue = "ML_CPU"
set job_app = "ML_CPU" 
set job_project = "d_09017"
set job_cores = "32" # Request enough cores for Python to manage subprocesses

echo "Job Configuration:"
echo "  Queue: $job_queue"
echo "  App: $job_app"
echo "  Project: $job_project"
echo "  Cores: $job_cores"
echo ""

# Prepare the command that will be executed by bsub
set python_script_path = "scripts/distributed_collector.py"
set job_cmd = "python3 $python_script_path --config $cfg_file"

echo "Submitting distributed collection job to cluster..."
echo "Command to run: $job_cmd"
echo ""

# Submit the single bsub job. Using -Is for interactive output.
# If the job waits in the queue, this script will wait with it.
bsub -Is \
    -J "PythonDistributed" \
    -q "$job_queue" \
    -app "$job_app" \
    -P "$job_project" \
    -n "$job_cores" \
    "$job_cmd"

echo "=========================================="
echo "Distributed collection job finished."
echo "==========================================" 