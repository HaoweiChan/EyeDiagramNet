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

# Ensure log directory exists
mkdir -p logs/parallel

# Launch each collector in the background
foreach pattern ( $patterns )
    set log_file = "logs/parallel/pattern_${pattern}_`date +%Y%m%d_%H%M%S`.log"
    echo "Launching collector for $pattern -> $log_file"
    # Run collector with unbuffered output (-u) and stream to both stdout and log via tee
    ( python -u -m simulation.collection.sequential_collector \
        --config "$CONFIG_FILE" \
        --trace_pattern "$pattern" |& tee "$log_file" ) &
    # Optional small delay to avoid starting all at once
    sleep 2
end

# Wait for all background jobs to complete
wait

echo "All pattern collectors completed." 