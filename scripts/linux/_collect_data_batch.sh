#!/bin/tcsh
# Batch collector script for EyeDiagramNet
# Runs batch_collector.py to process a single trace file against multiple boundary JSON files
#
# Boundary files are automatically discovered from the directory specified in config:
#   boundary.input_dir: "/path/to/boundary/directory"
# The script randomly samples from all .json files in that directory for each sample.
# No need to specify boundary files as command-line arguments.

# Load required modules for cluster environment
module load LSF/mtkgpu
module load Python3/3.12.2_gpu_torch270
source /proj/siaiadm/ew_predictor/.venv/sipi_torch270/bin/activate.csh

# Configuration
set cfg_file = "configs/data/_sbr_test.yaml"
set python_cmd = ( python3 -m simulation.collection.batch_collector --config $cfg_file $argv )

echo "Starting batch collector..."
echo "Config file: $cfg_file"

# Submit to cluster with resource requests
bsub -Is -J LongJob -q ML_CPU -app ML_CPU -P d_09017 $python_cmd
