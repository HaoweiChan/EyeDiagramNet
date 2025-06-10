#!/usr/bin/env python3
"""Optimized data collection runner with resource management"""

import subprocess
import sys
import psutil
import time
import os
from pathlib import Path

def get_optimal_workers():
    """Calculate optimal number of workers based on system resources"""
    # Get system info
    cpu_count = psutil.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    print(f"System: {cpu_count} CPUs, {memory_gb:.1f}GB RAM")
    
    # For large S96P files (~66MB each), be conservative
    # Each worker can use 500MB-1GB during processing
    memory_based_workers = max(1, int(memory_gb / 2))  # 2GB per worker
    cpu_based_workers = min(6, cpu_count)  # Cap at 6 for I/O limits
    
    optimal_workers = min(memory_based_workers, cpu_based_workers)
    print(f"Optimal workers: {optimal_workers} (memory: {memory_based_workers}, cpu: {cpu_based_workers})")
    
    return optimal_workers

def run_collection_with_monitoring(config_file, executor_type="thread", max_samples=10):
    """Run data collection with built-in monitoring"""
    
    # Calculate optimal workers
    optimal_workers = get_optimal_workers()
    
    # Build command
    cmd = [
        sys.executable, "-m", "simulation.collection.training_data_collector",
        "--config", str(config_file),
        "--executor_type", executor_type,
        "--max_workers", str(optimal_workers),
        "--max_samples", str(max_samples)
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    print(f"Starting data collection at {time.strftime('%H:%M:%S')}")
    print("-" * 60)
    
    # Set environment variables to prevent nested parallelism from numpy/scipy
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    
    print("Setting environment variables to prevent nested parallelism:")
    print(f"  OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1")
    
    # Start the process
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
        env=env  # Pass the modified environment
    )
    
    # Monitor the process
    start_time = time.time()
    last_output_time = start_time
    
    try:
        while True:
            # Check if process is still running
            poll = process.poll()
            if poll is not None:
                print(f"Process finished with return code: {poll}")
                break
            
            # Read output with timeout
            try:
                output = process.stdout.readline()
                if output:
                    print(output.strip())
                    last_output_time = time.time()
                else:
                    time.sleep(1)
            except:
                time.sleep(1)
            
            current_time = time.time()
            runtime = current_time - start_time
            
            # Check for hanging (no output for 10 minutes)
            if current_time - last_output_time > 600:
                print(f"\nWARNING: No output for 10 minutes (runtime: {runtime:.0f}s)")
                print("Process may be hanging...")
                
                # Try to get process info
                try:
                    proc = psutil.Process(process.pid)
                    memory_mb = proc.memory_info().rss / (1024 * 1024)
                    cpu_percent = proc.cpu_percent()
                    status = proc.status()
                    
                    print(f"Process status: {status}, {memory_mb:.1f}MB, {cpu_percent:.1f}% CPU")
                    
                    if status in ['zombie', 'stopped'] or (cpu_percent < 0.1 and runtime > 1800):
                        print("Terminating hanging process...")
                        process.terminate()
                        time.sleep(10)
                        if process.poll() is None:
                            print("Force killing process...")
                            process.kill()
                        break
                        
                except psutil.NoSuchProcess:
                    print("Process disappeared...")
                    break
                
                last_output_time = current_time  # Reset to avoid spam
            
            # Print periodic status
            if runtime > 0 and int(runtime) % 300 == 0:  # Every 5 minutes
                print(f"\n--- Status at {runtime:.0f}s ---")
                try:
                    proc = psutil.Process(process.pid)
                    memory_mb = proc.memory_info().rss / (1024 * 1024)
                    cpu_percent = proc.cpu_percent()
                    print(f"Main process: {memory_mb:.1f}MB, {cpu_percent:.1f}% CPU")
                    
                    # Check children
                    children = proc.children(recursive=True)
                    if children:
                        print(f"Child processes: {len(children)}")
                        for child in children[:3]:  # Show first 3
                            try:
                                child_mem = child.memory_info().rss / (1024 * 1024)
                                child_cpu = child.cpu_percent()
                                child_status = child.status()
                                print(f"  PID {child.pid}: {child_status}, {child_mem:.1f}MB, {child_cpu:.1f}% CPU")
                            except:
                                pass
                                
                except psutil.NoSuchProcess:
                    print("Main process disappeared")
                    break
                    
                print("--- End Status ---\n")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        process.terminate()
        time.sleep(5)
        if process.poll() is None:
            process.kill()
    
    # Get final return code
    return_code = process.wait()
    total_time = time.time() - start_time
    
    print(f"\nCollection completed in {total_time:.1f}s with return code: {return_code}")
    return return_code

def create_test_config():
    """Create a minimal test config for S96P files"""
    test_data_dir = Path("test_data")
    if not test_data_dir.exists():
        print("test_data directory not found")
        return None
    
    s96p_files = list(test_data_dir.glob("**/*.s96p"))
    if not s96p_files:
        print("No S96P files found in test_data")
        return None
    
    config_content = f"""
data:
  trace_pattern: "s96p_test"
  output_dir: "test_output_optimized"

dataset:
  horizontal_dataset:
    s96p_test: "{s96p_files[0].parent}"

boundary:
  param_type: "PDN"
  max_samples: 5
  enable_direction: false
  enable_inductance: false

runner:
  max_workers: 4

debug: false
"""
    
    config_file = Path("test_config_optimized.yaml")
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    print(f"Created test config: {config_file}")
    return config_file

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimized data collection runner")
    parser.add_argument("--config", help="Config file path")
    parser.add_argument("--executor", choices=["thread", "process"], default="thread")
    parser.add_argument("--samples", type=int, default=5, help="Max samples per file")
    parser.add_argument("--test", action="store_true", help="Run test with S96P files")
    
    args = parser.parse_args()
    
    if args.test:
        config_file = create_test_config()
        if not config_file:
            return 1
    else:
        config_file = args.config
        if not config_file:
            print("Must provide --config or use --test")
            return 1
    
    return run_collection_with_monitoring(config_file, args.executor, args.samples)

if __name__ == "__main__":
    sys.exit(main()) 