# EyeDiagramNet: Neural Network Eye Width Prediction System

A comprehensive machine learning system for predicting eye width in high-speed serial interfaces using transformer-based models with uncertainty quantification and physics-based simulation.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage Guide](#usage-guide)
  - [Inference](#inference)
  - [Simulation & Data Collection](#simulation--data-collection)
  - [Training](#training)
- [Configuration](#configuration)
- [Repository Structure](#repository-structure)
- [Changelog](#changelog)
- [Contributing](#contributing)

## Overview

EyeDiagramNet provides:

- **Eye Width Prediction**: Transformer-based models with Laplace uncertainty quantification
- **S-Parameter Analysis**: Self-supervised learning for S-parameter reconstruction
- **Physics Simulation**: SBR and DER simulation engines for data generation
- **Comprehensive Pipeline**: End-to-end workflow from simulation to prediction

## Installation

### Environment Setup

For production usage on server systems:
```bash
# Use pre-configured environment (if available)
source /proj/silaadm/ew_predictor/.venv/sipi/bin/activate

# Or create your own conda environment
conda create -n eyediagramnet python=3.11
conda activate eyediagramnet
pip install -r requirements.txt
```

### Checkpoint Installation

Copy pre-trained checkpoints for immediate inference:
```bash
# Copy from shared location (if available)
cp -r /proj/sipiadm/ew_predictor/checkpoints ./saved/
```

## Usage Guide

### Inference

#### Step 1: Configure Inference Settings

Create or modify your inference configuration file (e.g., `configs/inference/my_inference.yaml`):

```yaml
model:
  class_path: ml.modules.trace_ew_module.TraceEWModule
  init_args:
    model:
      class_path: ml.models.eyewidth_model.EyeWidthRegressor
      # Model parameters...
    ckpt_path: saved/ew_xfmr/checkpoints/best.ckpt

data:
  class_path: ml.data.eyewidth_data.InferenceTraceSeqEWDataloader
  init_args:
    data_dirs:
      my_dataset: /path/to/horizontal/dir/with/AI_input.csv
    drv_snp: /path/to/input_vertical.s96p
    odt_snp: /path/to/output_vertical.s96p
    bound_path: /path/to/boundary.json
    batch_size: 32
    scaler_path: saved/ew_xfmr/scaler.pth

trainer:
  accelerator: gpu
  devices: 1

callbacks:
  - class_path: ml.callbacks.dynamic_threshold.DynamicThresholdOptimizer
    init_args:
      warm_up_epochs: 0
      update_frequency: 1
```

#### Step 2: Configure Boundary Conditions

Create a boundary JSON file (`boundary.json`):

```json
{
  "boundary": {
    "R_tx": 32,
    "R_rx": 1.0e9,
    "C_tx": 4e-13,
    "C_rx": 2e-13,
    "L_tx": 1e-09,
    "L_rx": 1.6e-09,
    "pulse_amplitude": 0.55,
    "bits_per_sec": 1.3e10,
    "vmask": 0.03
  },
  "CTLE": {
    "AC_gain": 0.75,
    "DC_gain": 1.2,
    "fp1": 2e10,
    "fp2": 5e10
  },
  "directions": [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
}
```

**Notes:**
- `CTLE` section is optional - remove if not needed
- `directions`: `1` = TX from lower to higher port number, `0` = higher to lower
- All boundary values should match your specific system requirements

#### Step 3: Run Inference

For production or batch jobs on a Linux server, execute the inference using the provided `tcsh` scripts.

```bash
# Run directly on a Linux machine
tcsh ./scripts/linux/train_ew_xfmr.sh predict --config configs/inference/my_inference.yaml

# For environments with a job scheduler (e.g., LSF), you might use a wrapper.
# Note: 'run_inference.sh' is an example and may need to be created.
tcsh run_inference.sh
```

#### Step 4: View Results

Results will be saved in the configured output directory with:
- Eye width predictions with uncertainty estimates
- Open-eye probability predictions
- Model confidence metrics
- Configuration metadata

### Simulation & Data Collection

#### Sequential Data Collection

For small-scale data generation or testing, use the `tcsh` script:
```bash
tcsh ./scripts/linux/collect_data_sequential.sh
```

#### Parallel Data Collection

For large-scale distributed data generation, use the `tcsh` script:
```bash
tcsh ./scripts/linux/collect_data_distributed.sh
```

#### Batch Simulation

For organized simulation campaigns, configure a batch simulation file and execute it with the `tcsh` script:

```yaml
# Example configs/data/batch_simulation.yaml
dataset:
  horizontal_dataset:
    pattern1_dataset: /path/to/horizontal/data1
    pattern2_dataset: /path/to/horizontal/data2
  vertical_dataset:
    - /path/to/vertical1.s96p
    - /path/to/vertical2.s96p

collections:
  trace_pattern: pattern1_dataset
  horiz_params: MIX_PARAMS,CTLE_PARAMS
  repetition: 10
  nports: 96
  nproc_per_gpu: 8
  output_dir: /path/to/output
```

```bash
# Run batch simulation
python -m simulation.collection.batch_simulator --config configs/data/batch_simulation.yaml
```

### Training

#### Training New Models

**Step 1: Prepare Training Configuration**

```yaml
# configs/training/train_ew_xfmr.yaml
model:
  class_path: ml.modules.trace_ew_module.TraceEWModule
  init_args:
    model:
      class_path: ml.models.eyewidth_model.EyeWidthRegressor
      init_args:
        model_dim: 512
        num_heads: 8
        num_layers: 6
        use_rope: true
        predict_logvar: true
    ew_threshold: 0.3
    use_laplace_on_fit_end: true

data:
  class_path: ml.data.eyewidth_data.TraceSeqEWDataloader
  init_args:
    data_dirs:
      train_data: /path/to/training/data
    label_dir: /path/to/labels
    batch_size: 32
    test_size: 0.2

trainer:
  max_epochs: 100
  accelerator: gpu
  devices: 1
  precision: 16-mixed

callbacks:
  - class_path: ml.callbacks.dynamic_threshold.DynamicThresholdOptimizer
  - class_path: lightning.pytorch.callbacks.ModelCheckpoint
    init_args:
      monitor: val/mae_ew
      mode: min
```

**Step 2: Run Training**

For single-GPU or multi-node training on a Linux server, use the appropriate `tcsh` script:
```bash
# Single-GPU Training
tcsh ./scripts/linux/train_ew_xfmr.sh fit --config configs/training/train_ew_xfmr.yaml

# Multi-Node Training
tcsh ./scripts/linux/train_ew_xfmr_multinode.sh --config configs/training/train_ew_xfmr.yaml
```

#### Pretraining S-Parameter Models

Pretrain models using the `tcsh` script:
```bash
# S-Parameter Pretraining
tcsh ./scripts/linux/pretrain_snp.sh --config configs/training/pretrain_snp_ssl.yaml
```

## Configuration

### Data Configuration

The system uses YAML configuration files organized by purpose:

- `configs/data/default.yaml` - Default data collection settings
- `configs/training/*.yaml` - Training configurations
- `configs/inference/*.yaml` - Inference configurations

### Key Configuration Options

**Model Settings:**
- `model_dim`: Transformer model dimension (256, 512, 1024)
- `num_heads`: Number of attention heads (8, 16, 32)
- `num_layers`: Transformer layers (4, 6, 8, 12)
- `use_rope`: Enable rotary positional embeddings
- `predict_logvar`: Enable uncertainty quantification

**Training Settings:**
- `batch_size`: Batch size (adjust based on GPU memory)
- `max_epochs`: Training epochs
- `precision`: Mixed precision training (16-mixed recommended)
- `use_laplace_on_fit_end`: Enable Laplace uncertainty after training

**Data Settings:**
- `test_size`: Validation split ratio (0.1-0.3)
- `ignore_snp`: Skip S-parameter processing for debugging
- `scaler_path`: Path to data normalization scalers

## Repository Structure

```
EyeDiagramNet/
├── ml/                          # Machine learning pipeline
│   ├── callbacks/               # Training callbacks
│   ├── data/                    # Data loading and processing
│   ├── models/                  # Neural network architectures
│   ├── modules/                 # Lightning training modules
│   ├── utils/                   # ML utilities and tools
│   └── trainer.py               # Main training interface
├── simulation/                  # Physics simulation system
│   ├── engine/                  # SBR and DER simulation engines
│   ├── collection/              # Data collection strategies
│   ├── parameters/              # Parameter management
│   └── io/                      # I/O and configuration utilities
├── common/                      # Shared utilities
├── configs/                     # Configuration files
│   ├── data/                    # Data collection configs
│   └── training/                # Training configs
├── scripts/                     # Execution scripts
│   ├── linux/                   # Production server scripts
│   └── macos/                   # Local development scripts
└── saved/                       # Model outputs and checkpoints
```

For detailed component descriptions, see [Repository Structure Guide](.cursor/rules/repository-structure.mdc).

## Changelog

### Version 0.3.0 (2025-08-04)

#### Added
* Complete repository restructuring with domain-driven design
* Laplace uncertainty quantification for eye width predictions
* S-parameter self-supervised learning capabilities
* Dynamic threshold optimization during training
* Rotary positional embeddings (RoPE) for improved sequence modeling
* Comprehensive callback system for training enhancement
* Multi-platform script support (Linux/macOS)
* Enhanced data processing with semantic feature handling

#### Enhanced
* Transformer architecture with advanced attention mechanisms
* Modular data loading system with caching and optimization
* Flexible simulation engine supporting multiple algorithms
* Configuration management with Lightning CLI integration
* Weight transfer utilities for pretrained model loading

### Version 0.2.2 (2025-04-28)

#### Added
* Config and boundary files can now be relocated to a user-defined folder.

#### Fixed
* Resolved CTLE empty value issue, allowing boundaries without CTLE fields.
* Fixed vertical SNP issue causing memory kills due to too many frequency points; SNP is now interpolated before being fed into the model.

### Version 0.2.1 (2025-04-17)

#### Fixed
* Refactored data collection configuration into a yaml file, users no longer need to modify the code

### Version 0.2.0 (2025-04-15)

#### Added
* Enhanced support for D2D configurations, including cowos and emib for both 8mi and 9mi
* Added CTLE, inductances, and directions for boundary conditions

#### Fixed
* Refactored boundary condition settings to a separate JSON file, improving organization and maintainability
* Removed redundant inference configuration settings, streamlining the overall configuration process

### Version 0.1.0 (2024-11-27)

This is the initial release of the project, featuring:

#### Added
* An inference pipeline for eye width prediction
* Parallel collection for true eye width data
* Support for vertical SNPs and HBM³/UCIe boundary conditions

## Contributing

### Development Workflow

1. **Local Testing**: Use `scripts/macos/` for development and testing
2. **Configuration**: Update `configs/` files for your specific use case
3. **Data Preparation**: Use simulation tools to generate training data
4. **Model Training**: Train models using the Lightning CLI interface
5. **Production Deployment**: Use `scripts/linux/` for server deployment

### Code Organization

- Keep domain separation: `ml/` for ML, `simulation/` for simulation
- Use Lightning CLI for configuration management
- Follow the established import patterns
- Add comprehensive docstrings and type hints
- Update configurations in `configs/` rather than hardcoding values
---

## Authors

**Willy Chan** (mtk25738): willy.chan@mediatek.com
