#!/bin/tcsh
module load LSF/mtkgpu
module load Python3/3.11.8_gpu_torch251
source /proj/siaiadm/ew_predictor/.venv/sipi/bin/activate.csh

set cfg_file = "configs/data/d2d_novert_nodir_noind_noctle.yaml"
set python_cmd = ( python3 -m simulation.collection.training_data_collector --config $cfg_file $argv )

bsub -Is -J LongJob -q ML_CPU -app ML_CPU -P d_09017 $python_cmd