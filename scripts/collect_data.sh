#!/usr/bin/env bash
# Data collection script for generating training data

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root directory
cd "$PROJECT_ROOT"

# Add project root to Python path and run data collection
PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" python -m simulation.collection.parallel_collector "$@" 