#!/bin/tcsh
# Distributed Data Collection Script for Linux (Production)
# Submits separate bsub jobs for each trace pattern using sequential collectors
# This avoids Python multiprocessing overhead by using cluster-level parallelization

# Load required modules for cluster environment
module load LSF/mtkgpu
module load Python3/3.11.8_gpu_torch251
source /proj/siaiadm/ew_predictor/.venv/sipi/bin/activate.csh

echo "=========================================="
echo "EyeDiagramNet - Distributed Data Collection (Linux)"
echo "Cluster-Level Parallelization with Sequential Collectors"
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
echo "🚀 Strategy: Separate bsub jobs per trace pattern"
echo ""

# Define all available trace patterns from default.yaml
set trace_patterns = ( \
    "pattern2_cowos_8mi" \
    "pattern2_cowos_9mi" \
    "pattern2_emib" \
    "pattern2_cowos_8mi_0124" \
    "pattern2_cowos_9mi_0124" \
    "pattern2_emib_9mi_0124" \
    "pattern2_emib_10mi_0124" \
)

# Create logs directory
mkdir -p logs
set main_log = "logs/distributed_linux_`date +%Y%m%d_%H%M%S`.log"

# Record start time
set start_time = `date +%s`
echo "⏰ Starting distributed collection at `date`"
echo "📝 Main log file: $main_log"
echo ""

# Job submission settings
set job_queue = "ML_CPU"
set job_app = "ML_CPU"
set job_project = "d_09017"
set job_time_limit = "24:00"  # 24 hours per job

echo "📋 Job Configuration:"
echo "  🎯 Queue: $job_queue"
echo "  📱 App: $job_app"
echo "  🏢 Project: $job_project"
echo "  ⏰ Time Limit: $job_time_limit"
echo "  📊 Total Patterns: $#trace_patterns"
echo ""

# Array to track submitted job IDs
set job_ids = ()
set job_patterns = ()

# Submit jobs for each trace pattern
echo "🚀 Submitting jobs to cluster..."
foreach pattern ($trace_patterns)
    echo "📤 Submitting job for pattern: $pattern"
    
    # Create job-specific log file name
    set job_log = "logs/distributed_${pattern}_`date +%Y%m%d_%H%M%S`.log"
    
    # Prepare job command
    set job_cmd = "python3 -m simulation.collection.sequential_collector --config $cfg_file --trace_pattern $pattern"
    
    # Submit job with bsub
    set job_output = `bsub \
        -J LongJob \
        -q "$job_queue" \
        -app "$job_app" \
        -P "$job_project" \
        -W "$job_time_limit" \
        -o "$job_log" \
        -e "${job_log}.err" \
        -R "rusage[mem=250GB]" \
        -n 1 \
        "$job_cmd"`
    
    # Extract job ID from bsub output
    set job_id = `echo "$job_output" | grep -o 'Job <[0-9]*>' | grep -o '[0-9]*'`
    
    if ( "$job_id" != "" ) then
        set job_ids = ($job_ids $job_id)
        set job_patterns = ($job_patterns $pattern)
        echo "✅ Submitted job $job_id for pattern $pattern" | tee -a "$main_log"
        echo "📝 Job log: $job_log" | tee -a "$main_log"
    else
        echo "❌ Failed to submit job for pattern $pattern" | tee -a "$main_log"
        echo "🔍 bsub output: $job_output" | tee -a "$main_log"
    endif
    
    # Small delay between submissions
    sleep 2
end

echo ""
echo "📊 Submission Summary:"
echo "  🎯 Patterns to process: $#trace_patterns"
echo "  ✅ Jobs submitted: $#job_ids"
echo "  ❌ Failed submissions: `expr $#trace_patterns - $#job_ids`"
echo ""

if ( $#job_ids == 0 ) then
    echo "❌ No jobs were submitted successfully!"
    exit 1
endif

# Display submitted jobs
echo "📋 Submitted Jobs:"
@ i = 1
while ( $i <= $#job_ids )
    echo "  🔢 Job $job_ids[$i]: $job_patterns[$i]"
    @ i++
end

echo ""
echo "🔄 Monitoring job progress..."
echo "💡 Use 'bjobs' to check job status manually"
echo "📊 Use 'bjobs -u `whoami`' to see all your jobs"
echo ""

# Function to check job status (tcsh style)
set monitoring = 1
set completed_jobs = 0
set failed_jobs = 0
set check_interval = 300  # Check every 5 minutes

while ( $monitoring )
    # Check status of all submitted jobs
    set running_jobs = 0
    set pending_jobs = 0
    
    @ i = 1
    while ( $i <= $#job_ids )
        set job_id = $job_ids[$i]
        set pattern = $job_patterns[$i]
        
        # Get job status
        set job_status = `bjobs -noheader $job_id 2>/dev/null | awk '{print $3}' | head -1`
        
        if ( "$job_status" == "" ) then
            # Job not found, likely completed
            set completed_jobs = `expr $completed_jobs + 1`
            echo "✅ Job $job_id ($pattern) completed" | tee -a "$main_log"
        else if ( "$job_status" == "RUN" ) then
            set running_jobs = `expr $running_jobs + 1`
        else if ( "$job_status" == "PEND" ) then
            set pending_jobs = `expr $pending_jobs + 1`
        else if ( "$job_status" == "DONE" ) then
            set completed_jobs = `expr $completed_jobs + 1`
            echo "✅ Job $job_id ($pattern) completed successfully" | tee -a "$main_log"
        else if ( "$job_status" == "EXIT" ) then
            set failed_jobs = `expr $failed_jobs + 1`
            echo "❌ Job $job_id ($pattern) failed" | tee -a "$main_log"
        endif
        
        @ i++
    end
    
    # Check if all jobs are done
    set total_finished = `expr $completed_jobs + $failed_jobs`
    if ( $total_finished >= $#job_ids ) then
        set monitoring = 0
    else
        # Report current status
        set current_time = `date`
        echo "📊 Status at $current_time:" | tee -a "$main_log"
        echo "  🔄 Running: $running_jobs" | tee -a "$main_log"
        echo "  ⏳ Pending: $pending_jobs" | tee -a "$main_log"
        echo "  ✅ Completed: $completed_jobs" | tee -a "$main_log"
        echo "  ❌ Failed: $failed_jobs" | tee -a "$main_log"
        echo ""
        
        # Wait before next check
        echo "⏰ Next check in $check_interval seconds..."
        sleep $check_interval
    endif
end

# Calculate total time
set end_time = `date +%s`
set total_runtime = `expr $end_time - $start_time`
set runtime_hours = `expr $total_runtime / 3600`
set runtime_mins = `expr \( $total_runtime % 3600 \) / 60`
set runtime_secs = `expr $total_runtime % 60`

echo ""
echo "=========================================="
echo "📊 Distributed Collection Results"
echo "=========================================="
echo "⏱️  Total Runtime: ${total_runtime}s (${runtime_hours}h ${runtime_mins}m ${runtime_secs}s)"
echo "📁 Main Log: $main_log"
echo ""

# Final status summary
echo "📈 Final Summary:"
echo "  🎯 Total Patterns: $#trace_patterns"
echo "  ✅ Successful: $completed_jobs"
echo "  ❌ Failed: $failed_jobs"
echo "  📊 Success Rate: `expr $completed_jobs \* 100 / $#trace_patterns`%"

# Resource efficiency summary
echo ""
echo "🖥️  Resource Efficiency:"
echo "  🏭 Cluster utilization: $#job_ids separate machines"
echo "  ⚡ OS-level parallelization avoided Python multiprocessing overhead"
echo "  💾 Each job optimized BLAS threads for single-machine performance"

# Check individual log files
echo ""
echo "📁 Individual Job Logs:"
foreach pattern ($trace_patterns)
    set log_pattern = "logs/distributed_${pattern}_*.log"
    set log_files = (`ls $log_pattern 2>/dev/null`)
    if ( $#log_files > 0 ) then
        set log_file = $log_files[1]  # Get the most recent
        set file_size = `wc -l < "$log_file" 2>/dev/null || echo "0"`
        echo "  📄 $pattern: $log_file ($file_size lines)"
        
        # Check for completion markers in log
        if ( `grep -c "Collection completed" "$log_file" 2>/dev/null || echo "0"` > 0 ) then
            echo "    ✅ Collection completed successfully"
        else if ( `grep -c "ERROR\|FAILED" "$log_file" 2>/dev/null || echo "0"` > 0 ) then
            echo "    ❌ Errors detected in log"
        else
            echo "    ⚠️  Status unclear, check log"
        endif
    else
        echo "  ❓ $pattern: No log file found"
    endif
end

# Output directory information
echo ""
echo "📂 Output Locations:"
echo "  📊 Results are distributed by trace pattern in the configured output directory"
echo "  🔍 Check each pattern subdirectory for collected data files"

# Final status
echo ""
echo "=========================================="
if ( $failed_jobs == 0 ) then
    echo "🎉 Distributed Collection: COMPLETE SUCCESS"
    echo "✅ All $completed_jobs patterns processed successfully"
    set exit_code = 0
else
    echo "⚠️  Distributed Collection: PARTIAL SUCCESS"
    echo "✅ $completed_jobs successful, ❌ $failed_jobs failed"
    set exit_code = 1
endif
echo "=========================================="

echo ""
echo "📋 Next Steps:"
echo "  1. 📊 Review main log: $main_log"
echo "  2. 🔍 Check individual job logs for detailed results"
echo "  3. 📁 Verify output data in configured output directory"
echo "  4. 📈 Analyze performance vs parallel_collector.py"
echo ""
echo "🛠️  Useful Commands:"
echo "  • Check all jobs: bjobs -u `whoami`"
echo "  • Job details: bjobs -l <job_id>"
echo "  • System usage: bhosts"

exit $exit_code 