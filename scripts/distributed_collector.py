import os
import sys
import yaml
import time
import psutil
import platform
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

def get_trace_patterns(config: dict) -> list:
    """Extracts trace patterns from the configuration."""
    horizontal_dataset = config.get("dataset", {}).get("horizontal_dataset", {})
    if not horizontal_dataset:
        raise ValueError("Could not find horizontal_dataset in config")
    return list(horizontal_dataset.keys())

def get_max_concurrent_jobs() -> int:
    """Determines the optimal number of concurrent jobs based on platform."""
    cpu_count = psutil.cpu_count()
    system = platform.system()
    
    if system == "Linux":
        # Aggressive for a dedicated server: one job per 4 cores, max 8
        max_jobs = max(2, min(8, cpu_count // 4))
        print(f"Linux detected: Running up to {max_jobs} concurrent jobs.")
    elif system == "Darwin":
        # Conservative for macOS to prevent system crashes
        max_jobs = max(1, min(2, cpu_count // 2))
        print(f"macOS detected: Running up to {max_jobs} concurrent jobs for system stability.")
    else:
        # Generic conservative default
        max_jobs = max(1, min(2, cpu_count // 2))
        print(f"Unknown platform: Running up to {max_jobs} concurrent jobs.")
        
    return max_jobs

def run_collector_for_pattern(pattern: str, config_file: str, log_dir: Path):
    """Launches a sequential collector for a given pattern, with enhanced debugging."""
    log_file_path = log_dir / f"pattern_{pattern}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    command = [
        sys.executable,
        "-m", "simulation.collection.sequential_collector",
        "--config", config_file,
        "--trace_pattern", pattern
    ]
    
    print(f"\nDEBUG: Preparing to run command for '{pattern}':")
    print(f"  Command: {' '.join(command)}")

    log_file_handle = open(log_file_path, 'w')
    
    env = os.environ.copy()
    project_root = Path(__file__).parent.parent.resolve()
    env["PYTHONPATH"] = f"{project_root}{os.pathsep}{env.get('PYTHONPATH', '')}"
    print(f"  PYTHONPATH: {env['PYTHONPATH']}")

    try:
        print(f"DEBUG: Launching subprocess for '{pattern}'...")
        process = subprocess.Popen(
            command, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            env=env, 
            universal_newlines=True, 
            bufsize=1
        )
        print(f"DEBUG: Subprocess for '{pattern}' launched with PID: {process.pid}")
        return process, log_file_handle
    except Exception as e:
        error_msg = f"FATAL ERROR: Failed to launch subprocess for '{pattern}': {e}"
        print(error_msg)
        log_file_handle.write(error_msg + "\n")
        log_file_handle.close()
        return None, None

def main():
    """Main function to run distributed data collection using Python subprocesses."""
    parser = argparse.ArgumentParser(description="Distributed data collector using Python subprocesses.")
    parser.add_argument("--config", default="configs/data/default.yaml", help="Path to the configuration file.")
    args = parser.parse_args()
    
    config_file = Path(args.config)
    if not config_file.is_file():
        print(f"ERROR: Config file not found at '{config_file}'")
        sys.exit(1)
        
    print("==========================================")
    print("EyeDiagramNet - Distributed Data Collector (Python)")
    print("==========================================")
    print(f"Configuration: {config_file}")

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    try:
        trace_patterns = get_trace_patterns(config)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    max_jobs = get_max_concurrent_jobs()
    
    log_dir = Path("logs/distributed_python")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Found {len(trace_patterns)} patterns to process.")
    print(f"Will run up to {max_jobs} jobs concurrently.")
    print(f"Logs will be stored in: {log_dir}")
    print("")

    processes = {}  # {Popen_object: (pattern, log_file_handle)}
    patterns_to_run = trace_patterns[:]
    failed_patterns = []
    
    try:
        while patterns_to_run or processes:
            # Launch new processes if we have capacity
            while len(processes) < max_jobs and patterns_to_run:
                pattern = patterns_to_run.pop(0)
                process, log_file_handle = run_collector_for_pattern(pattern, str(config_file), log_dir)
                if process:
                    # DEBUG: Check if the process terminated immediately
                    time.sleep(0.5) # Give it a moment to potentially crash
                    if process.poll() is not None:
                        print(f"DEBUG: Process for '{pattern}' (PID: {process.pid}) terminated immediately with code {process.returncode}.")
                        # Read any final output
                        if process.stdout:
                            for line in process.stdout.readlines():
                                print(f"[{pattern}] IMMEDIATE EXIT: {line.strip()}")
                                log_file_handle.write(line)
                        log_file_handle.close()
                        failed_patterns.append(pattern)
                    else:
                        print(f"DEBUG: Process for '{pattern}' is running.")
                        processes[process] = (pattern, log_file_handle)
                else:
                    # The launch itself failed, mark as failed
                    failed_patterns.append(pattern)
            
            # Check for completed processes and read output from running processes
            if not processes:
                print("DEBUG: No running processes to monitor. Waiting...")
                time.sleep(5)
                if not patterns_to_run: # If no more to launch, exit
                    break
                continue

            completed_processes = []
            for process, (pattern, log_file) in processes.items():
                # Read any available output from the process
                if process.stdout:
                    for line in iter(process.stdout.readline, ''):
                        if line:
                            output_line = f"[{pattern}] {line.strip()}"
                            print(output_line)
                            log_file.write(line)
                            log_file.flush()
                        else:
                            break # No more output for now
                
                if process.poll() is not None:  # Process has terminated
                    print(f"DEBUG: Detected completed process for '{pattern}' with code {process.returncode}")
                    completed_processes.append(process)
                    log_file.close() # Close the log file handle
                    if process.returncode == 0:
                        print(f"SUCCESS: Collector for '{pattern}' finished.")
                    else:
                        print(f"FAILED: Collector for '{pattern}' finished with exit code {process.returncode}.")
                        failed_patterns.append(pattern)

            for process in completed_processes:
                del processes[process]
            
            # If no patterns are left to run, break the loop once all processes finish
            if not patterns_to_run and not processes:
                break
                
            time.sleep(1) # Shorter sleep for more responsive output
            
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Received Ctrl+C. Terminating child processes...")
        for process, (pattern, log_file) in processes.items():
            print(f"   Terminating process for '{pattern}' (PID: {process.pid})")
            process.terminate()
            log_file.close()
        
        # Wait for processes to terminate
        for process, (pattern, log_file) in processes.items():
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(f"   Forcing kill on process {process.pid}")
                process.kill()
            finally:
                if not log_file.closed:
                    log_file.close()

        print("All child processes terminated.")
        sys.exit(1)
        
    finally:
        # Final check to ensure all file handles are closed
        for process, (pattern, log_file) in processes.items():
            if not log_file.closed:
                log_file.close()

    print("\n==========================================")
    print("Distributed collection finished.")
    if failed_patterns:
        print("There were failures:")
        for pattern in failed_patterns:
            print(f"  - {pattern}")
        sys.exit(1)
    else:
        print("All patterns completed successfully.")
        sys.exit(0)
    print("==========================================")


if __name__ == "__main__":
    main() 