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

# ---------------------------------------------------------------------------
# Generate a small internal launcher that starts one collector per pattern.
# Each collector runs in the background ( & ) and logs to its own file.
# ---------------------------------------------------------------------------

# Internal launcher path
set internal_launcher = "scripts/internal_distributed_launcher.csh"

# Create/overwrite the internal launcher script
cat > "$internal_launcher" << 'EOF_INT'
#!/bin/tcsh

# ------------------ Internal Distributed Launcher -------------------------
# This script runs INSIDE the allocated bsub node. It launches one
# sequential_collector process per trace pattern, each in the background,
# and waits for them all to finish.
# --------------------------------------------------------------------------

set CONFIG_FILE = "$1"
if ( "$CONFIG_FILE" == "" ) then
    set CONFIG_FILE = "configs/data/default.yaml"
endif

# Dynamically obtain trace patterns from the YAML config using Python.
set patterns = (`python - << PY
import yaml, sys, pathlib
cfg = pathlib.Path(sys.argv[1])
data = yaml.safe_load(cfg.read_text())
patterns = list(data.get('dataset', {}).get('horizontal_dataset', {}).keys())
print(' '.join(patterns))
PY
$CONFIG_FILE`)

if ( $#patterns == 0 ) then
    echo "ERROR: No trace patterns found in $CONFIG_FILE"
    exit 1
endif

echo "Internal launcher will process $#patterns patterns: $patterns"

# Ensure log directory exists
mkdir -p logs/parallel

# Launch each collector in the background
foreach pattern ( $patterns )
    set log_file = "logs/parallel/pattern_${pattern}_`date +%Y%m%d_%H%M%S`.log"
    echo "Launching collector for $pattern -> $log_file"
    python -m simulation.collection.sequential_collector \
        --config "$CONFIG_FILE" \
        --trace_pattern "$pattern" \
        > "$log_file" 2>&1 &
    # Optional small delay to avoid starting all at once
    sleep 2
end

# Wait for all background jobs to complete
wait

echo "All pattern collectors completed."
EOF_INT

# Make the launcher executable
chmod +x "$internal_launcher"

# Command to execute inside bsub
set job_cmd = "tcsh $internal_launcher $cfg_file"

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