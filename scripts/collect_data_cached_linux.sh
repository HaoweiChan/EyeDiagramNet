#!/bin/tcsh
# Cached data collection script for EyeDiagramNet (Linux Production)
# Uses intelligent caching to achieve 5-15x speedup over original collector
# Target: 90-95% CPU usage, full memory utilization

# Load required modules for cluster environment
module load LSF/mtkgpu
module load Python3/3.11.8_gpu_torch251
source /proj/siaiadm/ew_predictor/.venv/sipi/bin/activate.csh

# Aggressive resource settings for Linux production (no limits set - use platform defaults)
# The cached parallel_collector will automatically detect Linux and use:
# - BLAS_THREADS=1 (maximize process parallelism)
# - 95% CPU utilization
# - Minimal memory safety margins
# - Intelligent caching for 5-15x speedup
# Environment variables can still override if needed:
# setenv BLAS_THREADS 1      # (platform default)
# setenv MAX_WORKERS 32      # (optional override)

# Configuration
set cfg_file = "configs/data/d2d_novert.yaml"
set python_cmd = ( python3 -m simulation.collection.parallel_collector_cached --config $cfg_file $argv )

echo "Starting cached data collection for Linux (production mode)..."
echo "Using platform defaults: BLAS_THREADS=1, aggressive worker calculation"
echo "Config file: $cfg_file"
echo "Expected: 5-15x speedup from intelligent caching"

# Submit to cluster with aggressive resource requests
bsub -Is -J LongJob -q ML_CPU -app ML_CPU -P d_09017 $python_cmd