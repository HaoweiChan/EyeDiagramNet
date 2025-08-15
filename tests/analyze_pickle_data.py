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

from tests.data_analyzer.main import main

if __name__ == "__main__":
    main()
