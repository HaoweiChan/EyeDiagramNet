#!/bin/tcsh
module load LSF/mtkgpu
module load Python3/3.12.2_gpu_torch270
source /proj/siaiadm/ew_predictor/.venv/sipi_torch270/bin/activate.csh

# Script to run sequence visualizer for contour data
# Usage examples:
#   ./visualize_sequence.sh --demo --save-dir demo_output/
#   ./visualize_sequence.sh tests/data_generation/contour/small_contour --case 0 --save output.png
#   ./visualize_sequence.sh tests/data_generation/contour/small_contour --all-cases --save-dir output/

set python_cmd = ( \
    python3 -m tests.data_analyzer.sequence_visualizer \
    $argv \
)

bsub -Is -J VisualizeSeq -q ML_CPU -app ML_CPU -P d_09017 $python_cmd

