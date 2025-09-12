#!/usr/bin/env python3
"""
A comprehensive tool to generate synthetic Touchstone data.

This script is the main entry point for the data generation tool.
It delegates the command-line parsing and execution to the main function
in the data_generation module.

Usage:
    python tests/data_generation/generate_data.py <command> [options]

Commands:
    generate    Generate synthetic Touchstone files.
    index       Create an index for generated .s96p files.

For more detailed help on each command, run:
    python tests/data_generation/generate_data.py <command> --help
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.data_generation.main import main

if __name__ == "__main__":
    main()
