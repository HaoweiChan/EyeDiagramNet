#!/bin/tcsh
module load LSF/mtkgpu
module load Python3/3.11.8_gpu_torch251
source /proj/siaiadm/ew_predictor/.venv/sipi/bin/activate.csh

bhosts GPU_3090_4
set python_cmd = ( \
    python3 -m ml.trainer fit --config configs/training/train_ew_xfmr.yaml \
    --trainer.devices 1 --trainer.num_nodes 1 --trainer.limit_train_batches 1 \
    --trainer.limit_val_batches 1 --trainer.max_epochs 10 \
)

bsub -Is -J LongJob -q ML_GPU -app PyTorch -P d_09017 -gpu "num=4" -m GPU_3090_4 $python_cmd