#!/usr/bin/env python3
"""Training script for EyeDiagramNet models."""

import os
import sys
from pathlib import Path
from lightning.pytorch.cli import LightningCLI

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def cli_main():
    """Main CLI entry point for training."""
    cli = LightningCLI(
        save_config_callback=None,
        auto_configure_optimizers=False,
    )

if __name__ == "__main__":
    cli_main() 