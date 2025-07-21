#!/bin/bash
# Distributed Data Collection Script for macOS
# This script is a simple wrapper around the Python-based distributed collector.

echo "=========================================="
echo "EyeDiagramNet - Distributed Data Collection (macOS)"
echo "Python-Managed Collectors"
echo "=========================================="

# Default config file, can be overridden by the first argument
CONFIG_FILE=${1:-configs/data/distributed_test.yaml}

if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ ERROR: Config file '$CONFIG_FILE' not found!"
    echo "Usage: $0 [path_to_config.yaml]"
    exit 1
fi

echo "🔧 Configuration: $CONFIG_FILE"

# Run the main Python distributed collector script
python3 scripts/distributed_collector.py --config "$CONFIG_FILE"

echo ""
echo "=========================================="
echo "macOS distributed collection finished."
echo "==========================================" 