#!/bin/bash

# Optimized data collection script for EyeDiagramNet (macOS)
# Uses conservative resource limits for development/testing environment
# Target: <50% CPU usage, <20GB RAM usage

# Resource limits for macOS (can be overridden by config)
export BLAS_THREADS=${BLAS_THREADS:-2}  # Conservative BLAS threading
export MAX_WORKERS=${MAX_WORKERS:-4}    # Limit concurrent workers
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}  # Ensure OpenMP follows BLAS limit

# Default config file - use test config for now
CONFIG_FILE=${1:-configs/data/test_local.yaml}

echo "Starting optimized data collection for macOS (conservative mode)..."
echo "Resource limits:"
echo "  BLAS threads: $BLAS_THREADS"
echo "  Max workers: $MAX_WORKERS"
echo "  OpenMP threads: $OMP_NUM_THREADS"
echo "  Config file: $CONFIG_FILE"

# Show system info
echo "System info:"
sysctl -n hw.ncpu | xargs -I {} echo "  CPUs: {}"
sysctl -n hw.memsize | awk '{print "  Memory: " $1/1024/1024/1024 " GB"}'

# Run the optimized data collector
echo "Running data collection with resource limits..."
python -m simulation.collection.parallel_collector --config "$CONFIG_FILE"

echo "Data collection completed." 