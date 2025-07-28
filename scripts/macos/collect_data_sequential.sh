#!/bin/bash
# Sequential Data Collection Script
# Uses the same configuration file as parallel collector but runs sequentially
# for better performance when multiprocessing overhead is too high

set -e

# Default configuration
CONFIG_FILE="configs/data/default.yaml"
TRACE_PATTERN=""
OUTPUT_DIR=""
PARAM_TYPE=""
MAX_SAMPLES=""
DEBUG=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --trace_pattern)
            TRACE_PATTERN="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --param_type)
            PARAM_TYPE="$2"
            shift 2
            ;;
        --max_samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --debug)
            DEBUG="--debug"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --config FILE         Configuration file (default: configs/data/default.yaml)"
            echo "  --trace_pattern KEY   Trace pattern key to use"
            echo "  --output_dir DIR      Output directory for results"
            echo "  --param_type TYPE     Parameter types (comma-separated)"
            echo "  --max_samples NUM     Maximum samples per trace file"
            echo "  --debug               Enable debug mode"
            echo "  -h, --help            Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --config configs/data/default.yaml --trace_pattern pattern2_cowos_8mi"
            echo ""
            echo "Note: This script uses the same configuration file as the parallel collector"
            echo "      but runs sequentially for better performance in some cases."
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Build command arguments
ARGS=(--config "$CONFIG_FILE")

if [[ -n "$TRACE_PATTERN" ]]; then
    ARGS+=(--trace_pattern "$TRACE_PATTERN")
fi

if [[ -n "$OUTPUT_DIR" ]]; then
    ARGS+=(--output_dir "$OUTPUT_DIR")
fi

if [[ -n "$PARAM_TYPE" ]]; then
    ARGS+=(--param_type "$PARAM_TYPE")
fi

if [[ -n "$MAX_SAMPLES" ]]; then
    ARGS+=(--max_samples "$MAX_SAMPLES")
fi

if [[ -n "$DEBUG" ]]; then
    ARGS+=($DEBUG)
fi

# Set up environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Display configuration
echo "EyeDiagramNet - Sequential Data Collector"
echo "========================================="
echo "Configuration file: $CONFIG_FILE"
echo "Command: python -m simulation.collection.sequential_collector ${ARGS[*]}"
echo ""

# Run the sequential collector
python -m simulation.collection.sequential_collector "${ARGS[@]}" 