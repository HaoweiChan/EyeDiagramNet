#!/usr/bin/env python3
"""
A comprehensive tool to analyze, clean, and validate training pickle data.

This script is the main entry point for the data analyzer tool.
It delegates the command-line parsing and execution to the main function
in the data_analyzer module.

Usage:
    python tests/analyze_pickle_data.py <pickle_dir> <command> [options]

Commands:
    analyze     Perform a full analysis of the pickle data.
    clean       Clean the pickle files in-place.
    validate    Validate pickle data against simulation.

For more detailed help on each command, run:
    python tests/analyze_pickle_data.py <pickle_dir> <command> --help
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tests.data_analyzer.main import main

if __name__ == "__main__":
    main()
