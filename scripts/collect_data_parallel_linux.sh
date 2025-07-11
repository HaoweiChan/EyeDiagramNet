#!/bin/tcsh
# Optimized data collection script for EyeDiagramNet (Linux Production)
# Uses aggressive resource settings to maximize server utilization
# Target: 90-95% CPU usage, full memory utilization

# Load required modules for cluster environment
module load LSF/mtkgpu
module load Python3/3.11.8_gpu_torch251
source /proj/siaiadm/ew_predictor/.venv/sipi/bin/activate.csh

# Aggressive resource settings for Linux production (no limits set - use platform defaults)
# The parallel_collector will automatically detect Linux and use:
# - BLAS_THREADS=1 (maximize process parallelism)
# - 95% CPU utilization
# - Minimal memory safety margins
# Environment variables can still override if needed:
# setenv BLAS_THREADS 1      # (platform default)
# setenv MAX_WORKERS 32      # (optional override)

# Configuration
set cfg_file = "configs/data/d2d_novert_nodir_noind_noctle.yaml"
set python_cmd = ( python3 -m simulation.collection.parallel_collector --config $cfg_file $argv )

echo "Starting aggressive data collection for Linux (production mode)..."
echo "Using platform defaults: BLAS_THREADS=1, aggressive worker calculation"
echo "Config file: $cfg_file"

# Submit to cluster with aggressive resource requests
bsub -Is -J LongJob -q ML_CPU -app ML_CPU -P d_09017 $python_cmd