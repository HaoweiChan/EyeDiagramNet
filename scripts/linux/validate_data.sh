#!/bin/tcsh
module load Python3/3.12.2_gpu_torch270
source /proj/siaiadm/ew_predictor/.venv/sipi_torch270/bin/activate.csh

eval python3 -m tests.examine_pickle_data \
    --pickle_dir /proj/siaiadm/ew_predictor/data/add_ind_dir_cached \
    --max_files 10 --max_samples 2 $argv