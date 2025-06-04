#!/usr/bin/env bash
# System monitoring script for training processes

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root directory
cd "$PROJECT_ROOT"

# Add project root to Python path and run system monitor
PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" python -m common.monitor_training "$@" 