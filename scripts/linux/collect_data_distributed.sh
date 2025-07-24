#!/bin/tcsh
# Distributed Data Collection Script (Production)
# Submits ONE bsub job per trace pattern for maximum parallelization.

# Load required modules for cluster environment
module load LSF/mtkgpu
module load Python3/3.12.2_gpu_torch270
source /proj/siaiadm/ew_predictor/.venv/sipi_torch270/bin/activate.csh

echo "=========================================="
echo "EyeDiagramNet - Distributed Data Collection"
echo "One bsub job per trace pattern for maximum parallelization"
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
echo "Strategy: One bsub job per trace pattern for maximum parallelization."
echo ""

# Dynamically obtain trace patterns from the YAML config using Python
set patterns = (`python -c "import yaml, sys, pathlib; cfg = pathlib.Path(sys.argv[1]); data = yaml.safe_load(cfg.read_text()); patterns = list(data.get('dataset', {}).get('horizontal_dataset', {}).keys()); print(' '.join(patterns))" $cfg_file`)

if ( $#patterns == 0 ) then
    echo "ERROR: No trace patterns found in $cfg_file"
    exit 1
endif

echo "Found $#patterns trace patterns: $patterns"
echo ""

# Create log directory for job outputs
mkdir -p logs/parallel

# Submit one bsub job per trace pattern
foreach pattern ( $patterns )
    set log_file = "logs/parallel/${pattern}_`date +%Y%m%d_%H%M%S`.log"
    
    echo "Submitting job for pattern: $pattern"
    echo "Log file: $log_file"
    
    # Submit the job - each job gets its own machine and uses all cores
    set python_cmd = ( python3 -m simulation.collection.sequential_collector --config $cfg_file --trace_pattern $pattern --shuffle)
    bsub -J LongJob \
         -q ML_CPU \
         -app ML_CPU \
         -P d_09017 \
         -o "$log_file" \
         -e "$log_file" \
         $python_cmd
    
    echo "Job submitted for $pattern"
    echo ""
end

echo "=========================================="
echo "All $#patterns jobs submitted successfully!"
echo "Monitor jobs with: bjobs"
echo "View logs in: logs/parallel/"
echo "==========================================" 