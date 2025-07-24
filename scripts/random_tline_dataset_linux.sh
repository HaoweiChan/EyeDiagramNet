#!/bin/tcsh
module load LSF/mtkgpu
module load Python3/3.12.2_gpu_torch270
source /proj/siaiadm/ew_predictor/.venv/sipi_torch270/bin/activate.csh

set python_cmd = ( python3 -m test_data.random_tline_network --output_dir "test_data/traces" )

bsub -Is -J LongJob -q ML_CPU -app ML_CPU -P d_09017 $python_cmd