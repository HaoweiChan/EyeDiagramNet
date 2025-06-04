#!/usr/bin/env bash
# Prediction script for EyeDiagramNet models

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root directory
cd "$PROJECT_ROOT"

# Add project root to Python path and run trainer with predict subcommand
PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" python -m ml.trainer predict "$@" 