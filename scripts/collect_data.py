#!/usr/bin/env python3
"""Data collection script for generating training data."""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from simulation.collection.parallel_collector import main as collector_main

if __name__ == "__main__":
    collector_main() 