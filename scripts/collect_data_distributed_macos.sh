#!/bin/bash
# Distributed Data Collection Script for macOS (Testing)
# Runs one sequential_collector process per trace pattern for testing.

echo "=========================================="
echo "EyeDiagramNet - Distributed Data Collection (macOS)"
echo "One process per trace pattern for testing"
echo "=========================================="

# Default configuration
CONFIG_FILE="${1:-configs/data/distributed_test.yaml}"

# Validate config file
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file '$CONFIG_FILE' not found!"
    echo "Usage: $0 [config_file]"
    exit 1
fi

echo "Configuration: $CONFIG_FILE"
echo "Strategy: One process per trace pattern for testing."
echo ""

# Dynamically obtain trace patterns from the YAML config using Python
PATTERNS=$(python -c "
import yaml
import sys
from pathlib import Path
cfg = Path(sys.argv[1])
data = yaml.safe_load(cfg.read_text())
patterns = list(data.get('dataset', {}).get('horizontal_dataset', {}).keys())
print(' '.join(patterns))
" "$CONFIG_FILE")

if [ -z "$PATTERNS" ]; then
    echo "ERROR: No trace patterns found in $CONFIG_FILE"
    exit 1
fi

echo "Found patterns: $PATTERNS"
echo ""

# Create log directory for process outputs
mkdir -p logs/parallel

# Run one sequential_collector process per trace pattern
for pattern in $PATTERNS; do
    LOG_FILE="logs/parallel/${pattern}_$(date +%Y%m%d_%H%M%S).log"
    
    echo "Starting collector for pattern: $pattern"
    echo "Log file: $LOG_FILE"
    
    # Run the collector in background with all cores
    python -u -m simulation.collection.sequential_collector \
        --config "$CONFIG_FILE" \
        --trace_pattern "$pattern" \
        > "$LOG_FILE" 2>&1 &
    
    COLLECTOR_PID=$!
    echo "Collector started for $pattern (PID: $COLLECTOR_PID)"
    echo ""
done

echo "=========================================="
echo "All collectors started successfully!"
echo "Monitor processes with: ps aux | grep sequential_collector"
echo "View logs in: logs/parallel/"
echo "=========================================="

# Wait for all background processes to complete
wait

echo "=========================================="
echo "All collectors completed!"
echo "==========================================" 