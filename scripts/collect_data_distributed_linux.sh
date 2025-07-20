#!/bin/tcsh
# Distributed Data Collection Script for Linux (Production)
# Submits ONE bsub job that runs multiple sequential collectors within the allocated machine
# This avoids Python multiprocessing overhead by using OS-level parallelization within one machine

# Load required modules for cluster environment
module load LSF/mtkgpu
module load Python3/3.11.8_gpu_torch251
source /proj/siaiadm/ew_predictor/.venv/sipi/bin/activate.csh

echo "=========================================="
echo "EyeDiagramNet - Distributed Data Collection (Linux)"
echo "Single Machine with Multiple Sequential Collectors"
echo "=========================================="

# Default configuration
set cfg_file = "${1}"
if ( "$cfg_file" == "" ) then
    set cfg_file = "configs/data/default.yaml"
endif

# Validate config file
if ( ! -f "$cfg_file" ) then
    echo "❌ ERROR: Config file '$cfg_file' not found!"
    echo ""
    echo "Usage: $0 [config_file]"
    echo "Example: $0 configs/data/default.yaml"
    exit 1
endif

echo "🔧 Configuration: $cfg_file"
echo "🖥️  Platform: Linux (production cluster)"
echo "🚀 Strategy: One bsub job running distributed collection internally"
echo ""

# Job submission settings
set job_queue = "ML_CPU"
set job_app = "ML_CPU" 
set job_project = "d_09017"
set job_cores = "32"          # Number of CPU cores to request

echo "📋 Job Configuration:"
echo "  🎯 Queue: $job_queue"
echo "  📱 App: $job_app"
echo "  🏢 Project: $job_project"
echo "  🖥️  Cores: $job_cores"
echo ""

# Create logs directory
mkdir -p logs
set main_log = "logs/distributed_linux_`date +%Y%m%d_%H%M%S`.log"

# Record start time
set start_time = `date +%s`
echo "⏰ Starting distributed collection submission at `date`"
echo "📝 Main log file: $main_log"
echo ""

# Create the distributed collection script that will run inside the bsub job
set internal_script = "scripts/internal_distributed_linux.sh"

# Generate the internal script
cat > "$internal_script" << 'EOF_INTERNAL_SCRIPT'
#!/bin/bash
# Internal distributed collection script that runs within the allocated bsub machine
# This script manages multiple sequential collectors for different trace patterns

set -e

echo "=========================================="
echo "Internal Distributed Collection (Linux)"
echo "Running within allocated bsub machine"
echo "=========================================="

# Get configuration from command line
CONFIG_FILE="$1"
if [ -z "$CONFIG_FILE" ]; then
    CONFIG_FILE="configs/data/default.yaml"
fi

# Set Python path
export PYTHONPATH="${PYTHONPATH}:."

# Define all available trace patterns from default.yaml
TRACE_PATTERNS=(
    "pattern2_cowos_8mi"
    "pattern2_cowos_9mi"
    "pattern2_emib"
    "pattern2_cowos_8mi_0124"
    "pattern2_cowos_9mi_0124"
    "pattern2_emib_9mi_0124"
    "pattern2_emib_10mi_0124"
)

# Get system information
CPU_COUNT=$(nproc)
MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')
echo "🖥️  Allocated Resources: ${CPU_COUNT} CPUs, ${MEMORY_GB}GB RAM"

# Calculate optimal concurrent jobs for Linux server
# Use more aggressive settings since we have a dedicated machine
MAX_CONCURRENT_JOBS=$(echo "scale=0; $CPU_COUNT / 4" | bc -l)  # One job per 4 cores
if [ "$MAX_CONCURRENT_JOBS" -lt 2 ]; then
    MAX_CONCURRENT_JOBS=2
elif [ "$MAX_CONCURRENT_JOBS" -gt 8 ]; then
    MAX_CONCURRENT_JOBS=8  # Cap at 8 concurrent jobs
fi

echo "🎯 Running up to $MAX_CONCURRENT_JOBS concurrent sequential collectors"
echo "📋 Total patterns: ${#TRACE_PATTERNS[@]}"
echo ""

# Create internal logs directory
mkdir -p logs/internal
INTERNAL_MAIN_LOG="logs/internal/distributed_internal_$(date +%Y%m%d_%H%M%S).log"

# Function to run sequential collector for a specific pattern
run_pattern_collector() {
    local pattern=$1
    local log_file="logs/internal/pattern_${pattern}_$(date +%Y%m%d_%H%M%S).log"
    
    echo "🔄 Starting collector for pattern: $pattern" | tee -a "$INTERNAL_MAIN_LOG"
    echo "📝 Pattern log: $log_file" | tee -a "$INTERNAL_MAIN_LOG"
    
    # Run sequential collector with specific trace pattern
    python -m simulation.collection.sequential_collector \
        --config "$CONFIG_FILE" \
        --trace_pattern "$pattern" \
        > "$log_file" 2>&1
    
    local exit_code=$?
    local end_time=$(date)
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ Pattern $pattern completed successfully at $end_time" | tee -a "$INTERNAL_MAIN_LOG"
    else
        echo "❌ Pattern $pattern failed with exit code $exit_code at $end_time" | tee -a "$INTERNAL_MAIN_LOG"
    fi
    
    return $exit_code
}

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

# Start initial jobs
for ((i=0; i<MAX_CONCURRENT_JOBS && i<${#TRACE_PATTERNS[@]}; i++)); do
    start_next_job
    sleep 5  # Small delay between job starts
done

echo ""
echo "🔄 Monitoring internal job progress..."
echo ""

# Monitor jobs and start new ones as they complete
while [ $RUNNING_JOBS -gt 0 ]; do
    # Check each running job
    for i in "${!JOB_PIDS[@]}"; do
        local pid=${JOB_PIDS[$i]}
        local pattern=${JOB_PATTERNS[$i]}
        local status=${JOB_RESULTS[$i]}
        
        if [ "$status" = "running" ]; then
            # Check if job is still running
            if ! kill -0 "$pid" 2>/dev/null; then
                # Job finished, get exit code
                wait "$pid"
                local exit_code=$?
                
                if [ $exit_code -eq 0 ]; then
                    JOB_RESULTS[$i]="success"
                    echo "✅ Internal job completed: $pattern"
                else
                    JOB_RESULTS[$i]="failed"
                    echo "❌ Internal job failed: $pattern (exit code: $exit_code)"
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
    sleep 10
done

# Count results
SUCCESSFUL_JOBS=0
FAILED_JOBS=0

echo ""
echo "📋 Internal Job Results:"
for i in "${!JOB_PATTERNS[@]}"; do
    local pattern=${JOB_PATTERNS[$i]}
    local result=${JOB_RESULTS[$i]}
    
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
echo "📈 Internal Summary:"
echo "  🎯 Total Patterns: ${#TRACE_PATTERNS[@]}"
echo "  ✅ Successful: $SUCCESSFUL_JOBS"
echo "  ❌ Failed: $FAILED_JOBS"
echo "  📊 Success Rate: $((SUCCESSFUL_JOBS * 100 / ${#TRACE_PATTERNS[@]}))%"

# Final status
echo ""
echo "=========================================="
if [ $FAILED_JOBS -eq 0 ]; then
    echo "🎉 Internal Distributed Collection: COMPLETE SUCCESS"
    exit 0
else
    echo "⚠️  Internal Distributed Collection: PARTIAL SUCCESS"
    exit 1
fi
EOF_INTERNAL_SCRIPT

# Make the internal script executable
chmod +x "$internal_script"

echo "📝 Generated internal distributed script: $internal_script"
echo ""

# Prepare the bsub command
set job_cmd = "cd `pwd` && ./$internal_script $cfg_file"

echo "🚀 Submitting distributed collection job to cluster..."
echo "📋 Job will run: $job_cmd"
echo "🔧 Current directory: `pwd`"
echo "📂 Internal script: $internal_script"
echo "⚙️  bsub command details:"
echo "   Queue: $job_queue | App: $job_app | Project: $job_project | Cores: $job_cores"
echo ""

# Submit the single bsub job that will handle all patterns
# Use non-interactive mode for better reliability
set job_output = `bsub \
    -J "DistributedCollection" \
    -q "$job_queue" \
    -app "$job_app" \
    -P "$job_project" \
    -o "$main_log" \
    -e "${main_log}.err" \
    -n "$job_cores" \
    "$job_cmd"`

# Extract job ID from bsub output
set job_id = `echo "$job_output" | grep -o 'Job <[0-9]*>' | grep -o '[0-9]*'`

if ( "$job_id" != "" ) then
    echo "✅ Submitted distributed collection job: $job_id" | tee -a "$main_log"
    echo "📝 Job log: $main_log" | tee -a "$main_log"
    echo "📊 Job will handle all 7 patterns internally" | tee -a "$main_log"
    echo ""
    echo "🔄 Monitoring job progress..."
    echo "💡 Use 'bjobs $job_id' to check job status manually"
    echo "📊 Use 'bjobs -u `whoami`' to see all your jobs"
else
    echo "❌ Failed to submit distributed collection job" | tee -a "$main_log"
    echo "🔍 bsub output: $job_output" | tee -a "$main_log"
    exit 1
endif

# Monitor job status until completion
set job_completed = 0
set check_interval = 60  # Check every minute

while ( $job_completed == 0 )
    # Get job status
    set job_status = `bjobs -noheader $job_id 2>/dev/null | awk '{print $3}' | head -1`
    set current_time = `date`
    
    if ( "$job_status" == "" ) then
        # Job not found, likely completed
        echo "✅ Job $job_id completed (no longer in queue) at $current_time" | tee -a "$main_log"
        set job_completed = 1
    else if ( "$job_status" == "PEND" ) then
        echo "⏳ Job $job_id pending in queue at $current_time" | tee -a "$main_log"
    else if ( "$job_status" == "RUN" ) then
        echo "🔄 Job $job_id running at $current_time" | tee -a "$main_log"
        
        # When job is running, check for live progress in log file
        if ( -f "$main_log" ) then
            set log_lines = `wc -l < "$main_log" 2>/dev/null || echo "0"`
            echo "📝 Log file has $log_lines lines (job is active)"
        endif
    else if ( "$job_status" == "DONE" ) then
        echo "✅ Job $job_id completed successfully at $current_time" | tee -a "$main_log"
        set job_completed = 1
    else if ( "$job_status" == "EXIT" ) then
        echo "❌ Job $job_id failed at $current_time" | tee -a "$main_log"
        set job_completed = 1
    else
        echo "❓ Job $job_id has unknown status: $job_status at $current_time" | tee -a "$main_log"
    endif
    
    # Wait before next check (unless job is completed)
    if ( $job_completed == 0 ) then
        echo "⏰ Next check in $check_interval seconds..."
        sleep $check_interval
    endif
end

echo ""
echo "🔍 Final job status check..."
set final_status = `bjobs -noheader $job_id 2>/dev/null | awk '{print $3}' | head -1`
if ( "$final_status" == "DONE" || "$final_status" == "" ) then
    echo "✅ Job completed successfully"
    set job_success = 1
else
    echo "❌ Job failed with status: $final_status"
    set job_success = 0
endif

# Calculate total time after job completion
set end_time = `date +%s`
set total_runtime = `expr $end_time - $start_time`
set runtime_hours = `expr $total_runtime / 3600`
set runtime_mins = `expr \( $total_runtime % 3600 \) / 60`
set runtime_secs = `expr $total_runtime % 60`

echo ""
echo "=========================================="
if ( $job_success == 1 ) then
    echo "📊 Distributed Collection COMPLETED SUCCESSFULLY"
else
    echo "📊 Distributed Collection FAILED"
endif
echo "=========================================="
echo "⏱️  Total Runtime: ${total_runtime}s (${runtime_hours}h ${runtime_mins}m ${runtime_secs}s)"
echo "📁 Main Log: $main_log"
echo "📁 Internal Logs: logs/internal/"
echo ""

if ( $job_success == 1 ) then
    echo "🖥️  Resource Efficiency:"
    echo "  🏭 Single allocated machine with $job_cores cores"
    echo "  ⚡ OS-level parallelization within dedicated machine"
    echo "  💾 Optimized BLAS threads for single-machine performance"
    
    echo ""
    echo "📋 Next Steps:"
    echo "  1. 📊 Review main log: $main_log"
    echo "  2. 🔍 Check internal logs in logs/internal/ for pattern details"
    echo "  3. 📁 Verify output data in configured output directory"
    echo "  4. 📈 Analyze performance vs parallel_collector.py"
    
    echo ""
    echo "🎉 Distributed collection completed successfully!"
else
    echo "❌ Job Issues:"
    echo "  🔍 Check main log for errors: $main_log"
    echo "  📊 Check job details: bjobs -l $job_id"
    echo "  🧪 Try running manually for debugging"
    
    echo ""
    echo "🛠️  Troubleshooting:"
    echo "  • Check if input files exist and are accessible"
    echo "  • Verify cluster resources and queue status"
    echo "  • Review error log: ${main_log}.err"
    
    echo ""
    echo "❌ Distributed collection failed!"
endif

# Exit with appropriate code
if ( $job_success == 1 ) then
    exit 0
else
    exit 1
endif 