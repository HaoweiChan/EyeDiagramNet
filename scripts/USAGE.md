# EyeDiagramNet Training Script Guide

## `train_ew_xfmr_local.sh`
**tcsh-based training script with embedded bsub command and profiling control.**

## Quick Usage

```bash
# Fast training (no profiling) - recommended for production
./scripts/train_ew_xfmr_local.sh

# Training with profiling - for debugging/optimization
./scripts/train_ew_xfmr_local.sh --profiling
# or short form:
./scripts/train_ew_xfmr_local.sh -p
```

## How It Works

### Step 1: Job Submission
The script automatically submits a bsub job with appropriate parameters:

**Fast Mode:**
```bash
bsub -Is -J EyeDiagram_Fast -q ML_GPU -app PyTorch -P d_09017 -gpu "num=4" -m GPU_3090_4 -R "rusage[mem=24000]" [script]
```

**Profiling Mode:**
```bash  
bsub -Is -J EyeDiagram_Prof -q ML_GPU -app PyTorch -P d_09017 -gpu "num=4" -m GPU_3090_4 -R "rusage[mem=32000]" [script] --profiling
```

### Step 2: Inside the Job
Once the job starts, the script:
1. **First execution**: Submits bsub job with appropriate parameters
2. **Inside bsub job**: Runs the actual training with optional profiling
3. Loads modules and activates Python environment
4. Installs/updates requirements
5. Configures profiling (if enabled)
6. Runs training with appropriate config
7. Cleans up temporary files and processes

## Features

- **Smart job submission**: Automatically detects if already in bsub job
- **Profiling control**: `--profiling` flag enables/disables monitoring and PyTorch profiler
- **Memory optimization**: Uses different memory allocations based on profiling mode
- **tcsh compatibility**: Works with tcsh shell environments
- **Automatic cleanup**: Proper process management and temporary file cleanup

## Profiling Modes

| Mode | Flag | Memory | Job Name | Features |
|------|------|---------|----------|----------|
| **Fast** | None | 24GB | `EyeDiagram_Fast` | No profiling, optimized config |
| **Profiled** | `--profiling` | 32GB | `EyeDiagram_Prof` | System monitoring + PyTorch profiler |

### Fast Mode Details
- Job name: `EyeDiagram_Fast`
- Memory: 24GB per GPU
- No system monitoring
- Profiler disabled in config
- Faster execution

### Profiling Mode Details
- Job name: `EyeDiagram_Prof` 
- Memory: 32GB per GPU
- System monitoring active
- PyTorch profiler enabled
- Detailed performance data

## File Outputs

### Fast Mode
- Training logs and checkpoints in `./saved/ew_xfmr/`
- No profiling data

### Profiling Mode  
- Training logs and checkpoints in `./saved/ew_xfmr/`
- System monitoring data in `system_monitor.json`
- PyTorch profiler traces in `./profiler_logs/`

## Requirements

- **tcsh shell environment**
- **Access to bsub/LSF environment**  
- **Python environment with requirements.txt installed**
- **GPU resources (configured for 4x RTX 3090)**

## Related Files

- `../monitor_training.py` - System monitoring script (CPU, GPU, memory, disk I/O)
- `../configs/train_ew_xfmr.yaml` - Training configuration
- `../requirements.txt` - Python dependencies

## Troubleshooting

- **Permission denied**: Run `chmod +x scripts/train_ew_xfmr_local.sh`
- **tcsh not found**: Make sure you're on the Linux cluster with tcsh
- **Module load errors**: Check that LSF modules are available
- **Memory errors**: Use `--profiling` for more memory allocation
- **Job submission fails**: Check LSF queue status and resource availability
- **Training hangs**: Monitor system resources and check for data loading issues 