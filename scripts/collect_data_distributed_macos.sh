#!/bin/bash
# Distributed Data Collection Script for macOS
# Runs multiple sequential collectors in parallel, each handling a different trace pattern
# This avoids Python multiprocessing overhead by using OS-level parallelization

set -e

echo "=========================================="
echo "EyeDiagramNet - Distributed Data Collection (macOS)"
echo "OS-Level Parallelization with Sequential Collectors"
echo "=========================================="

# Set Python path
export PYTHONPATH="${PYTHONPATH}:."

# Default configuration
CONFIG_FILE="${1:-configs/data/default.yaml}"

# Validate inputs
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ ERROR: Config file '$CONFIG_FILE' not found!"
    echo ""
    echo "Usage: $0 [config_file]"
    echo "Example: $0 configs/data/default.yaml"
    exit 1
fi

echo "🔧 Configuration: $CONFIG_FILE"
echo "🖥️  Platform: macOS (testing distributed approach)"
echo "🚀 Strategy: Multiple sequential collectors (OS-level parallelization)"
echo ""

# Define trace patterns based on config file
if [[ "$CONFIG_FILE" == *"distributed_test"* ]]; then
    # Test configuration with fewer patterns
    TRACE_PATTERNS=(
        "pattern_test_1"
        "pattern_test_2"
        "pattern_test_3"
    )
    echo "🧪 Using test patterns for distributed testing"
else
    # Production patterns from default.yaml
    TRACE_PATTERNS=(
        "pattern2_cowos_8mi"
        "pattern2_cowos_9mi"
        "pattern2_emib"
        "pattern2_cowos_8mi_0124"
        "pattern2_cowos_9mi_0124"
        "pattern2_emib_9mi_0124"
        "pattern2_emib_10mi_0124"
    )
    echo "🏭 Using production patterns from default.yaml"
fi

# Create logs directory
mkdir -p logs
MAIN_LOG="logs/distributed_macos_$(date +%Y%m%d_%H%M%S).log"

# Record start time
OVERALL_START_TIME=$(date +%s)
echo "⏰ Starting distributed collection at $(date)"
echo "📝 Main log file: $MAIN_LOG"
echo ""

# Function to check system memory before starting jobs
check_memory_safety() {
    if command -v python3 >/dev/null 2>&1; then
        local memory_info=$(python3 -c "
import psutil
mem = psutil.virtual_memory()
print(f'{mem.percent:.1f},{mem.available/(1024**3):.1f}')
" 2>/dev/null || echo "0,0")
        local mem_percent=$(echo "$memory_info" | cut -d',' -f1)
        local mem_available=$(echo "$memory_info" | cut -d',' -f2)
        
        echo "💾 System Memory: ${mem_percent}% used, ${mem_available}GB available"
        
        # Check if safe to start new jobs
        if (( $(echo "$mem_percent > 80" | bc -l 2>/dev/null || echo "0") )); then
            echo "⚠️  High memory usage detected, consider waiting"
            return 1
        fi
    fi
    return 0
}

# Function to run sequential collector for a specific pattern
run_pattern_collector() {
    local pattern=$1
    local log_file="logs/distributed_${pattern}_$(date +%Y%m%d_%H%M%S).log"
    
    echo "🔄 Starting collector for pattern: $pattern" | tee -a "$MAIN_LOG"
    echo "📝 Pattern log: $log_file" | tee -a "$MAIN_LOG"
    
    # Check memory before starting
    check_memory_safety
    
    # Run sequential collector with specific trace pattern with nice priority for system stability
    nice -n 10 python -m simulation.collection.sequential_collector \
        --config "$CONFIG_FILE" \
        --trace_pattern "$pattern" \
        > "$log_file" 2>&1
    
    local exit_code=$?
    local end_time=$(date)
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ Pattern $pattern completed successfully at $end_time" | tee -a "$MAIN_LOG"
    else
        echo "❌ Pattern $pattern failed with exit code $exit_code at $end_time" | tee -a "$MAIN_LOG"
    fi
    
    return $exit_code
}

# Check how many patterns to run based on macOS resource constraints
# VERY CONSERVATIVE after crash: Reduce concurrent jobs to prevent system overload
MAX_CONCURRENT_JOBS=2  # Further reduced from 3 to 2 to prevent crashes
echo "🎯 Running up to $MAX_CONCURRENT_JOBS concurrent jobs for macOS (reduced after crash prevention)"
echo "📋 Available patterns: ${#TRACE_PATTERNS[@]}"
echo ""

# Track background jobs
declare -a JOB_PIDS=()
declare -a JOB_PATTERNS=()
declare -a JOB_RESULTS=()

# Start initial batch of jobs
JOB_INDEX=0
RUNNING_JOBS=0

start_next_job() {
    if [ $JOB_INDEX -lt ${#TRACE_PATTERNS[@]} ]; then
        local pattern=${TRACE_PATTERNS[$JOB_INDEX]}
        echo "🚀 Starting job $((JOB_INDEX + 1))/${#TRACE_PATTERNS[@]}: $pattern"
        
        # Start background job
        run_pattern_collector "$pattern" &
        local pid=$!
        
        JOB_PIDS+=($pid)
        JOB_PATTERNS+=("$pattern")
        JOB_RESULTS+=("running")
        
        JOB_INDEX=$((JOB_INDEX + 1))
        RUNNING_JOBS=$((RUNNING_JOBS + 1))
        
        echo "👷 Active jobs: $RUNNING_JOBS/$MAX_CONCURRENT_JOBS"
    fi
}

# Start initial jobs with increased delays for system stability
for ((i=0; i<MAX_CONCURRENT_JOBS && i<${#TRACE_PATTERNS[@]}; i++)); do
    start_next_job
    sleep 10  # Longer delay (10s) between job starts to prevent system overload
    
    # Additional safety: Check system load after each job start
    if command -v python3 >/dev/null 2>&1; then
        local load_avg=$(python3 -c "
import psutil
print(psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0)
" 2>/dev/null || echo "0")
        echo "📊 System load: $load_avg"
    fi
done

echo ""
echo "🔄 Monitoring job progress..."
echo "💡 Use Ctrl+C to gracefully stop all jobs"
echo ""

# Monitor jobs and start new ones as they complete
while [ $RUNNING_JOBS -gt 0 ]; do
    # Check each running job
    for i in "${!JOB_PIDS[@]}"; do
        pid=${JOB_PIDS[$i]}
        pattern=${JOB_PATTERNS[$i]}
        status=${JOB_RESULTS[$i]}
        
        if [ "$status" = "running" ]; then
            # Check if job is still running
            if ! kill -0 "$pid" 2>/dev/null; then
                # Job finished, get exit code
                wait "$pid"
                exit_code=$?
                
                if [ $exit_code -eq 0 ]; then
                    JOB_RESULTS[$i]="success"
                    echo "✅ Job completed: $pattern"
                else
                    JOB_RESULTS[$i]="failed"
                    echo "❌ Job failed: $pattern (exit code: $exit_code)"
                fi
                
                RUNNING_JOBS=$((RUNNING_JOBS - 1))
                
                # Start next job if available
                if [ $JOB_INDEX -lt ${#TRACE_PATTERNS[@]} ]; then
                    echo ""
                    start_next_job
                fi
            fi
        fi
    done
    
    # Sleep briefly before next check
    sleep 5
done

# Calculate total time
OVERALL_END_TIME=$(date +%s)
TOTAL_RUNTIME=$((OVERALL_END_TIME - OVERALL_START_TIME))

echo ""
echo "=========================================="
echo "📊 Distributed Collection Results"
echo "=========================================="
echo "⏱️  Total Runtime: ${TOTAL_RUNTIME}s ($((TOTAL_RUNTIME / 60))m $((TOTAL_RUNTIME % 60))s)"
echo "📁 Main Log: $MAIN_LOG"
echo ""

# Count results
SUCCESSFUL_JOBS=0
FAILED_JOBS=0

echo "📋 Job Results:"
for i in "${!JOB_PATTERNS[@]}"; do
    pattern=${JOB_PATTERNS[$i]}
    result=${JOB_RESULTS[$i]}
    
    if [ "$result" = "success" ]; then
        echo "  ✅ $pattern: SUCCESS"
        SUCCESSFUL_JOBS=$((SUCCESSFUL_JOBS + 1))
    elif [ "$result" = "failed" ]; then
        echo "  ❌ $pattern: FAILED"
        FAILED_JOBS=$((FAILED_JOBS + 1))
    else
        echo "  ⚠️  $pattern: UNKNOWN ($result)"
        FAILED_JOBS=$((FAILED_JOBS + 1))
    fi
done

echo ""
echo "📈 Summary:"
echo "  🎯 Total Patterns: ${#TRACE_PATTERNS[@]}"
echo "  ✅ Successful: $SUCCESSFUL_JOBS"
echo "  ❌ Failed: $FAILED_JOBS"
echo "  📊 Success Rate: $((SUCCESSFUL_JOBS * 100 / ${#TRACE_PATTERNS[@]}))%"

# Resource usage summary
echo ""
echo "🖥️  Resource Usage:"
echo "  💾 Peak concurrent jobs: $MAX_CONCURRENT_JOBS (macOS limit)"
echo "  ⚡ OS-level parallelization avoided Python multiprocessing overhead"

# Check individual log files for detailed results
echo ""
echo "📁 Individual Log Files:"
for pattern in "${TRACE_PATTERNS[@]}"; do
    log_pattern="logs/distributed_${pattern}_*.log"
    if ls $log_pattern 1> /dev/null 2>&1; then
        log_file=$(ls -t $log_pattern | head -1)
        file_size=$(wc -l < "$log_file" 2>/dev/null || echo "0")
        echo "  📄 $pattern: $log_file ($file_size lines)"
    else
        echo "  ❓ $pattern: No log file found"
    fi
done

# Final status
echo ""
echo "=========================================="
if [ $FAILED_JOBS -eq 0 ]; then
    echo "🎉 Distributed Collection: COMPLETE SUCCESS"
    echo "✅ All $SUCCESSFUL_JOBS patterns processed successfully"
else
    echo "⚠️  Distributed Collection: PARTIAL SUCCESS"
    echo "✅ $SUCCESSFUL_JOBS successful, ❌ $FAILED_JOBS failed"
fi
echo "=========================================="

echo ""
echo "📋 Next Steps:"
echo "  1. 📊 Review main log: $MAIN_LOG"
echo "  2. 🔍 Check individual pattern logs for detailed results"
echo "  3. 📈 Compare performance with parallel_collector.py"
echo "  4. 🚀 Deploy Linux version for production"
echo ""
echo "🛠️  For Production (Linux):"
echo "  • Use: ./scripts/collect_data_distributed_linux.sh $CONFIG_FILE"
echo "  • Each pattern will run on a separate bsub-allocated machine"

# Exit with appropriate code
if [ $FAILED_JOBS -eq 0 ]; then
    exit 0
else
    exit 1
fi 