#!/bin/tcsh
module load Python3/3.12.2_gpu_torch270
source /proj/siaiadm/ew_predictor/.venv/sipi_torch270/bin/activate.csh

set python_cmd = ( \
    python3 -m tests.examine_pickle_data \
    --pickle_dir /proj/siaiadm/ew_predictor/data/add_ind_dir \
    --max_files 20 --max_samples 2 $argv \
)
bsub -J LongJob -q ML_CPU -app ML_CPU -P d_09017 $python_cmd