# Scripts Directory

This directory contains utility scripts for training and managing the EyeDiagramNet project.

## Training Scripts

### `train_ew_xfmr_local.sh`
Local training script for bsub job submission systems.

**Usage:**
```bash
# Submit job via bsub
bsub -Is -J LongJob -q ML_GPU -app PyTorch -P d_09017 -gpu "num=4" -m GPU_3090_4 < scripts/train_ew_xfmr_local.sh

# Or run directly (for testing)
bash scripts/train_ew_xfmr_local.sh
```

**Features:**
- System monitoring with background process management
- Proper cleanup and signal handling
- Module loading for bsub environments
- Progress logging and error reporting

**Requirements:**
- Access to bsub/LSF environment
- Python environment with requirements.txt installed
- GPU resources (configured for 4x RTX 3090)

## Related Files

- `../monitor_training.py` - System monitoring script (CPU, GPU, memory, disk I/O)
- `../configs/train_ew_xfmr.yaml` - Training configuration
- `../requirements.txt` - Python dependencies 