#!/usr/bin/env python3
"""Prediction script for model inference."""

import sys
from pathlib import Path
from lightning.pytorch.cli import LightningCLI

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def cli_main():
    """Main CLI entry point for prediction."""
    cli = LightningCLI(
        subcommand_mode_model=False,
        run=False,
        save_config_callback=None,
        auto_configure_optimizers=False,
    )
    cli.trainer.predict(cli.model, cli.datamodule)

if __name__ == "__main__":
    cli_main() 