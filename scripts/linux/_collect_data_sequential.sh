#!/bin/tcsh
# Sequential data collection script for EyeDiagramNet
# Uses single-process execution with optimized BLAS threading
# Target: 75-85% CPU usage with maximum per-operation performance

# Load required modules for cluster environment
module load LSF/mtkgpu
module load Python3/3.12.2_gpu_torch270
source /proj/siaiadm/ew_predictor/.venv/sipi_torch270/bin/activate.csh

# Sequential processing resource settings
# The sequential_collector will automatically detect Linux and use:
# - Optimized BLAS threads (75-85% of available cores)
# - Single-process execution (no multiprocessing overhead)
# - Efficient memory caching for vertical SNPs
# Environment variables can still override if needed:
# setenv OMP_NUM_THREADS 8    # (optional override for BLAS threads)
# setenv BLAS_THREADS 8       # (optional override for BLAS threads)

echo "Sequential processing mode - optimizing BLAS threads for single-process execution"
echo "This mode eliminates multiprocessing overhead and shared memory complexity"

# Configuration
set cfg_file = "configs/data/d2d_der_novert.yaml"
set python_cmd = ( python3 -m simulation.collection.sequential_collector --config $cfg_file --simulator-type "der" $argv )

echo "Starting sequential data collection..."
echo "Using optimized sequential processing: single process with high BLAS thread count"
echo "Config file: $cfg_file"

# Submit to cluster with resource requests optimized for sequential processing
# Sequential processing needs fewer cores but higher per-core performance
bsub -Is -J LongJob -q ML_CPU -app ML_CPU -P d_09017 $python_cmd 