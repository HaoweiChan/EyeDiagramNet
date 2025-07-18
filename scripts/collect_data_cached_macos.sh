#!/bin/bash
# Cached Parallel Data Collector for macOS
# High-performance data collection using intelligent caching for massive speedup

set -e

echo "=========================================="
echo "EyeDiagramNet - CACHED Data Collection"
echo "🚀 Expected Performance: 5-15x Speedup"
echo "=========================================="

# Set Python path
export PYTHONPATH="${PYTHONPATH}:."

# Default configuration
CONFIG_FILE="${1:-configs/data/local_test_optimized.yaml}"

# Validate inputs
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ ERROR: Config file '$CONFIG_FILE' not found!"
    echo ""
    echo "Usage: $0 [config_file]"
    echo "Example: $0 configs/data/local_test_optimized.yaml"
    echo ""
    echo "Available configs:"
    find configs/data/ -name "*.yaml" 2>/dev/null | head -5 || echo "  (No config files found in configs/data/)"
    exit 1
fi

echo "🔧 Configuration: $CONFIG_FILE"
echo "🖥️  Platform: macOS (resource-constrained environment)"
echo "💾 Cache Strategy: Network + Test Pattern + Base Processing"
echo "🎯 Target: <50% CPU, <20GB RAM usage"
echo ""

# Create output directory for logs
mkdir -p logs
LOG_FILE="logs/cached_collection_$(date +%Y%m%d_%H%M%S).log"

echo "📋 Test Plan:"
echo "  1. Pre-load vertical SNPs into shared memory cache"
echo "  2. Run cached simulations with intelligent caching"
echo "  3. Monitor cache hit rates and performance gains"
echo "  4. Validate resource usage within macOS limits"
echo "  5. Compare performance against reference times"
echo ""

# Record start time for overall performance
OVERALL_START_TIME=$(date +%s)
echo "⏰ Starting CACHED data collection at $(date)"
echo "📝 Log file: $LOG_FILE"
echo ""

# Run the cached parallel collector with comprehensive monitoring
echo "🚀 Launching CACHED Parallel Collector..."
echo "Command: python -m simulation.collection.parallel_collector_cached --config $CONFIG_FILE"
echo ""

# Capture both stdout and stderr, display in real-time, and save to log
python -m simulation.collection.parallel_collector_cached \
  --config "$CONFIG_FILE" 2>&1 | tee "$LOG_FILE"

# Capture exit code from the Python command
PYTHON_EXIT_CODE=${PIPESTATUS[0]}

# Record end time
OVERALL_END_TIME=$(date +%s)
TOTAL_RUNTIME=$((OVERALL_END_TIME - OVERALL_START_TIME))

echo ""
echo "=========================================="
echo "📊 CACHED Collection Results"
echo "=========================================="
echo "⏱️  Total Runtime: ${TOTAL_RUNTIME}s ($((TOTAL_RUNTIME / 60))m $((TOTAL_RUNTIME % 60))s)"
echo "📁 Log File: $LOG_FILE"

# Analyze results from log file
echo ""
echo "🔍 Performance Analysis:"
echo "------------------------"

# Check exit status
if [ $PYTHON_EXIT_CODE -eq 0 ]; then
    echo "✅ Collection completed successfully"
else
    echo "❌ Collection failed with exit code $PYTHON_EXIT_CODE"
fi

# Cache performance analysis
echo ""
echo "💾 Cache Performance:"
if grep -q "Cache contains.*files" "$LOG_FILE"; then
    CACHE_FILES=$(grep "Cache contains.*files" "$LOG_FILE" | tail -1 | grep -o '[0-9]\+' | head -1)
    echo "  📂 Cached Files: $CACHE_FILES vertical SNP files"
fi

if grep -q "Shared memory allocated:" "$LOG_FILE"; then
    CACHE_MEMORY=$(grep "Shared memory allocated:" "$LOG_FILE" | tail -1 | grep -o '[0-9]\+\.[0-9]\+' | head -1)
    echo "  💾 Cache Memory: ${CACHE_MEMORY}MB shared memory"
fi

if grep -q "Time saved:" "$LOG_FILE"; then
    echo "  ⚡ Cache Benefits:"
    grep "Time saved:\|improvement" "$LOG_FILE" | tail -2 | sed 's/^/    /'
fi

# Worker performance summary
echo ""
echo "👷 Worker Performance:"
if grep -q "CACHED.*completed.*simulations" "$LOG_FILE"; then
    echo "  📈 Completed Tasks:"
    grep "CACHED.*completed.*simulations" "$LOG_FILE" | head -3 | sed 's/^.*Worker /    Worker /' | sed 's/\[.*\] //'
fi

if grep -q "CACHED performance:" "$LOG_FILE"; then
    echo "  ⏱️  Timing Statistics:"
    grep "CACHED performance:" "$LOG_FILE" | head -3 | sed 's/^.*Worker /    Worker /' | sed 's/\[.*\] //'
fi

# Resource usage validation
echo ""
echo "🖥️  Resource Usage:"
if grep -q "macOS resource validation" "$LOG_FILE"; then
    RESOURCE_LINE=$(grep "macOS resource validation" "$LOG_FILE" | tail -1)
    echo "  📊 Final Usage: $RESOURCE_LINE" | sed 's/.*macOS resource validation: //'
fi

# Check for warnings and errors
ERROR_COUNT=$(grep -c "ERROR\|FAILED" "$LOG_FILE" 2>/dev/null || echo "0")
WARNING_COUNT=$(grep -c "WARNING" "$LOG_FILE" 2>/dev/null || echo "0")

echo ""
echo "⚠️  Issues Summary:"
echo "  🔴 Errors: $ERROR_COUNT"
echo "  🟡 Warnings: $WARNING_COUNT"

if [ "$ERROR_COUNT" -gt 0 ]; then
    echo ""
    echo "🔴 Recent Errors:"
    grep "ERROR\|FAILED" "$LOG_FILE" | tail -3 | sed 's/^/  /'
fi

# Speedup analysis (if reference available)
echo ""
echo "🚀 Speedup Analysis:"
REFERENCE_FILE="reference_runtime.txt"
if [ -f "$REFERENCE_FILE" ]; then
    REFERENCE_TIME=$(cat "$REFERENCE_FILE" 2>/dev/null || echo "0")
    if [ "$REFERENCE_TIME" -gt 0 ] 2>/dev/null; then
        # Calculate speedup using bc if available, otherwise use basic arithmetic
        if command -v bc >/dev/null 2>&1; then
            SPEEDUP=$(echo "scale=1; $REFERENCE_TIME / $TOTAL_RUNTIME" | bc -l 2>/dev/null || echo "unknown")
            TIME_SAVED=$((REFERENCE_TIME - TOTAL_RUNTIME))
            
            echo "  📊 Reference Time: ${REFERENCE_TIME}s"
            echo "  ⚡ Achieved Speedup: ${SPEEDUP}x"
            echo "  ⏱️  Time Saved: ${TIME_SAVED}s"
            
            # Speedup evaluation
            if [ "$SPEEDUP" != "unknown" ]; then
                SPEEDUP_INT=$(echo "$SPEEDUP" | cut -d. -f1)
                if [ "$SPEEDUP_INT" -ge 5 ] 2>/dev/null; then
                    echo "  🎉 EXCELLENT: Achieved target speedup (5-15x)!"
                elif [ "$SPEEDUP_INT" -ge 3 ] 2>/dev/null; then
                    echo "  ✅ GOOD: Solid speedup achieved!"
                elif [ "$SPEEDUP_INT" -ge 2 ] 2>/dev/null; then
                    echo "  👍 FAIR: Some improvement, but below target"
                else
                    echo "  ⚠️  LIMITED: Speedup below expectations"
                fi
            fi
        else
            echo "  📊 Reference Time: ${REFERENCE_TIME}s (install 'bc' for speedup calculation)"
        fi
    else
        echo "  ℹ️  Invalid reference time in $REFERENCE_FILE"
    fi
else
    echo "  ℹ️  No reference time available"
    echo "  💡 Run original collector first: ./scripts/collect_data_optimized_macos.sh"
    echo "  💡 Then save runtime to $REFERENCE_FILE for comparison"
fi

# Final status
echo ""
echo "=========================================="
if [ $PYTHON_EXIT_CODE -eq 0 ] && [ "$ERROR_COUNT" -eq 0 ]; then
    echo "🎉 CACHED Collection: SUCCESS"
    echo "✅ No errors detected"
    if [ "$WARNING_COUNT" -gt 0 ]; then
        echo "⚠️  $WARNING_COUNT warnings (check log for details)"
    fi
else
    echo "❌ CACHED Collection: ISSUES DETECTED"
    echo "🔍 Check log file for details: $LOG_FILE"
fi
echo "=========================================="

# Provide next steps
echo ""
echo "📋 Next Steps:"
echo "  1. 📊 Review detailed log: $LOG_FILE"
echo "  2. 🔍 Check output data quality in results directory"
echo "  3. 📈 Compare with original collector performance"
echo "  4. 🧪 Run with --debug for detailed cache statistics"
echo ""
echo "🛠️  Useful Commands:"
echo "  • Debug mode:    $0 $CONFIG_FILE --debug"
echo "  • Original:      python -m simulation.collection.parallel_collector --config $CONFIG_FILE"
echo "  • Cache stats:   grep -A5 'Cache Performance' $LOG_FILE"
echo "  • Timing stats:  grep 'performance:' $LOG_FILE"

# Set exit code based on Python command result
exit $PYTHON_EXIT_CODE 