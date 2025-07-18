#!/bin/tcsh
# Cached Parallel Data Collector for Linux Production
# High-performance data collection using intelligent caching for massive speedup
# Target: 90-95% CPU usage, full memory utilization with 5-15x speedup

echo "=========================================="
echo "EyeDiagramNet - CACHED Data Collection"
echo "🚀 Expected Performance: 5-15x Speedup"
echo "🖥️  Platform: Linux Production Cluster"
echo "=========================================="

# Load required modules for cluster environment
module load LSF/mtkgpu
module load Python3/3.11.8_gpu_torch251
source /proj/siaiadm/ew_predictor/.venv/sipi/bin/activate.csh

# Aggressive resource settings for Linux production (no limits set - use platform defaults)
# The cached parallel_collector will automatically detect Linux and use:
# - BLAS_THREADS=1 (maximize process parallelism)
# - 95% CPU utilization
# - Minimal memory safety margins
# - Intelligent caching for 5-15x speedup
# Environment variables can still override if needed:
# setenv BLAS_THREADS 1      # (platform default)
# setenv MAX_WORKERS 32      # (optional override)

# Configuration
set default_cfg = "configs/data/d2d_novert_nodir_noind_noctle.yaml"
set cfg_file = "$1"
if ( "$cfg_file" == "" ) then
    set cfg_file = "$default_cfg"
endif

# Validate configuration file exists
if ( ! -f "$cfg_file" ) then
    echo "❌ ERROR: Config file '$cfg_file' not found!"
    echo ""
    echo "Usage: $0 [config_file]"
    echo "Example: $0 configs/data/d2d_novert_nodir_noind_noctle.yaml"
    echo ""
    echo "Available configs:"
    find configs/data/ -name "*.yaml" | head -5
    exit 1
endif

echo "🔧 Configuration: $cfg_file"
echo "💾 Cache Strategy: Network + Test Pattern + Base Processing"
echo "🎯 Target: 90-95% CPU, aggressive memory usage"
echo ""

# Create output directory for logs
mkdir -p logs
set log_file = "logs/cached_collection_linux_`date +%Y%m%d_%H%M%S`.log"

echo "📋 Cluster Execution Plan:"
echo "  1. Submit job to ML_CPU queue with aggressive resources"
echo "  2. Pre-load vertical SNPs into shared memory cache"
echo "  3. Run cached simulations with intelligent caching"
echo "  4. Monitor cache hit rates and performance gains"
echo "  5. Validate aggressive resource utilization"
echo ""

# Python command using cached parallel collector
set python_cmd = ( python3 -m simulation.collection.parallel_collector_cached --config $cfg_file )

# Add any additional arguments passed to the script
if ( $#argv > 1 ) then
    set python_cmd = ( $python_cmd $argv[2-] )
endif

echo "⏰ Starting CACHED data collection at `date`"
echo "📝 Log file: $log_file"
echo "🚀 Command: $python_cmd"
echo ""

# Submit to cluster with aggressive resource requests and monitoring
echo "📤 Submitting to LSF cluster..."
bsub -Is -J CachedDataCollection -q ML_CPU -app ML_CPU -P d_09017 \
     -o "$log_file" \
     "echo '========================================'; \
      echo 'EyeDiagramNet CACHED Collection Started'; \
      echo 'Timestamp: '`date`; \
      echo 'Config: $cfg_file'; \
      echo 'Expected: 5-15x speedup from caching'; \
      echo '========================================'; \
      echo ''; \
      set start_time = `date +%s`; \
      $python_cmd; \
      set exit_code = \$status; \
      set end_time = `date +%s`; \
      @ total_time = \$end_time - \$start_time; \
      echo ''; \
      echo '========================================'; \
      echo '📊 CACHED Collection Results'; \
      echo '========================================'; \
      echo '⏱️  Total Runtime: '\$total_time's'; \
      echo '📁 Log File: $log_file'; \
      if ( \$exit_code == 0 ) then; \
          echo '✅ Collection completed successfully'; \
      else; \
          echo '❌ Collection failed with exit code '\$exit_code; \
      endif; \
      echo ''; \
      echo '💾 Cache Performance Analysis:'; \
      grep -i 'cache.*files\|shared memory\|time saved\|improvement' $log_file | tail -5; \
      echo ''; \
      echo '👷 Worker Performance Summary:'; \
      grep -i 'cached.*completed\|cached performance' $log_file | head -3; \
      echo ''; \
      echo '⚠️  Issues Summary:'; \
      set error_count = `grep -c 'ERROR\|FAILED' $log_file`; \
      set warning_count = `grep -c 'WARNING' $log_file`; \
      echo '  🔴 Errors: '\$error_count; \
      echo '  🟡 Warnings: '\$warning_count; \
      if ( \$error_count > 0 ) then; \
          echo '🔴 Recent Errors:'; \
          grep 'ERROR\|FAILED' $log_file | tail -3; \
      endif; \
      echo ''; \
      echo '========================================'; \
      if ( \$exit_code == 0 && \$error_count == 0 ) then; \
          echo '🎉 CACHED Collection: SUCCESS'; \
          echo '✅ No errors detected'; \
          if ( \$warning_count > 0 ) then; \
              echo '⚠️  '\$warning_count' warnings (check log for details)'; \
          endif; \
      else; \
          echo '❌ CACHED Collection: ISSUES DETECTED'; \
          echo '🔍 Check log file for details: $log_file'; \
      endif; \
      echo '========================================'; \
      echo ''; \
      echo '📋 Next Steps:'; \
      echo '  1. 📊 Review detailed log: $log_file'; \
      echo '  2. 🔍 Check output data quality in results directory'; \
      echo '  3. 📈 Compare with original collector performance'; \
      echo '  4. 🧪 Run with --debug for detailed cache statistics'; \
      echo ''; \
      exit \$exit_code"

echo ""
echo "🎯 Job submitted to cluster!"
echo "📊 Monitor progress with: bpeek -f"
echo "📁 Results will be logged to: $log_file"
echo ""
echo "🛠️  Useful Commands After Completion:"
echo "  • Check results:     cat $log_file"
echo "  • Cache analysis:    grep -A5 'Cache Performance' $log_file"
echo "  • Timing stats:      grep 'performance:' $log_file"
echo "  • Debug mode:        $0 $cfg_file --debug" 