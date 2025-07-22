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
# Use python -c for robustness instead of a multi-line heredoc.
set patterns = (`python -c "import yaml, sys, pathlib; cfg = pathlib.Path(sys.argv[1]); data = yaml.safe_load(cfg.read_text()); patterns = list(data.get('dataset', {}).get('horizontal_dataset', {}).keys()); print(' '.join(patterns))" $CONFIG_FILE`)

if ( $#patterns == 0 ) then
    echo "ERROR: No trace patterns found in $CONFIG_FILE"
    exit 1
endif

echo "Internal launcher will process $#patterns patterns: $patterns"

# --- Core Management for Parallel Execution ---
set total_cores = `nproc`
# Determine number of parallel jobs to run (up to 7, one per pattern)
set num_parallel_jobs = $#patterns
if ( $num_parallel_jobs > 7 ) then
    set num_parallel_jobs = 7
endif

# Calculate threads per job, ensuring at least 2 for performance
@ threads_per_job = $total_cores / $num_parallel_jobs
if ( $threads_per_job < 2 ) then
    set threads_per_job = 2
endif

echo "System has $total_cores cores. Distributing among $num_parallel_jobs parallel jobs."
echo "Each collector will be assigned $threads_per_job threads."
# --- End Core Management ---


# Ensure log directory exists
mkdir -p logs/parallel

# Launch each collector in the background
foreach pattern ( $patterns )
    set log_file = "logs/parallel/pattern_${pattern}_`date +%Y%m%d_%H%M%S`.log"
    echo "Launching collector for $pattern -> $log_file with $threads_per_job threads"
    ( python -u -m simulation.collection.sequential_collector \
        --config "$CONFIG_FILE" \
        --trace_pattern "$pattern" \
        --num-threads "$threads_per_job" |& tee "$log_file" ) &
    # Optional small delay to avoid starting all at once
    sleep 2
end

# Wait for all background jobs to complete
wait

echo "All pattern collectors completed." 