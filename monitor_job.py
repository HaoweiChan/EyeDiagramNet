#!/usr/bin/env python3
"""Job monitoring script to track data collection processes and detect hangs/kills"""

import psutil
import time
import argparse
import sys
from datetime import datetime
from pathlib import Path

def get_data_collection_processes():
    """Find all running data collection processes"""
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info', 'cpu_percent', 'create_time']):
        try:
            if proc.info['cmdline'] and any('training_data_collector' in arg for arg in proc.info['cmdline']):
                processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return processes

def monitor_processes(interval=30, duration=None):
    """Monitor data collection processes"""
    start_time = time.time()
    last_process_count = 0
    last_activity_time = start_time
    
    print(f"Starting job monitoring at {datetime.now()}")
    print(f"Monitoring every {interval} seconds")
    if duration:
        print(f"Will monitor for {duration} seconds")
    print("-" * 60)
    
    while True:
        current_time = time.time()
        
        # Check if we should stop
        if duration and (current_time - start_time) > duration:
            print(f"Monitoring duration {duration}s completed")
            break
        
        # Find running processes
        processes = get_data_collection_processes()
        
        if processes:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Found {len(processes)} data collection processes:")
            
            for proc in processes:
                try:
                    # Get process info
                    pid = proc.info['pid']
                    memory_mb = proc.info['memory_info'].rss / (1024 * 1024)
                    cpu_percent = proc.cpu_percent()
                    runtime = current_time - proc.info['create_time']
                    
                    # Check if process is responsive
                    status = proc.status()
                    
                    print(f"  PID {pid}: {status}, {memory_mb:.1f}MB, {cpu_percent:.1f}% CPU, {runtime:.0f}s runtime")
                    
                    # Check for potential issues
                    if status == 'zombie':
                        print(f"    WARNING: Process {pid} is zombie!")
                    elif status == 'stopped':
                        print(f"    WARNING: Process {pid} is stopped!")
                    elif cpu_percent < 0.1 and runtime > 300:  # Low CPU for 5+ minutes
                        print(f"    WARNING: Process {pid} may be hanging (low CPU usage)")
                    elif memory_mb > 2000:  # High memory usage
                        print(f"    WARNING: Process {pid} using high memory ({memory_mb:.1f}MB)")
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                    print(f"  Process disappeared or access denied: {e}")
            
            # Check for activity
            if len(processes) != last_process_count:
                last_activity_time = current_time
                last_process_count = len(processes)
        else:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] No data collection processes found")
            
            # Check if no processes for too long
            if current_time - last_activity_time > 600:  # 10 minutes
                print("WARNING: No active processes for 10+ minutes - job may have failed")
        
        # Check system resources
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        print(f"System: {cpu_percent:.1f}% CPU, {memory.percent:.1f}% memory ({memory.used/(1024**3):.1f}GB used)")
        
        # Sleep until next check
        time.sleep(interval)

def check_job_status():
    """Quick check of job status"""
    processes = get_data_collection_processes()
    
    if not processes:
        print("No data collection processes running")
        return False
    
    print(f"Found {len(processes)} running processes:")
    for proc in processes:
        try:
            pid = proc.info['pid']
            memory_mb = proc.info['memory_info'].rss / (1024 * 1024)
            runtime = time.time() - proc.info['create_time']
            status = proc.status()
            
            print(f"  PID {pid}: {status}, {memory_mb:.1f}MB, {runtime:.0f}s runtime")
            
            if status in ['zombie', 'stopped']:
                print(f"    Issue detected with PID {pid}")
                return False
                
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    return True

def kill_hanging_processes():
    """Kill potentially hanging processes"""
    processes = get_data_collection_processes()
    
    for proc in processes:
        try:
            pid = proc.info['pid']
            runtime = time.time() - proc.info['create_time']
            status = proc.status()
            cpu_percent = proc.cpu_percent()
            
            # Kill if zombie, stopped, or hanging for too long
            should_kill = (
                status in ['zombie', 'stopped'] or
                (runtime > 3600 and cpu_percent < 0.1)  # 1 hour with no CPU
            )
            
            if should_kill:
                print(f"Killing process PID {pid} (status: {status}, runtime: {runtime:.0f}s)")
                proc.terminate()
                time.sleep(5)
                if proc.is_running():
                    proc.kill()
                    print(f"Force killed PID {pid}")
            
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

def main():
    parser = argparse.ArgumentParser(description="Monitor data collection job")
    parser.add_argument("--interval", type=int, default=30, help="Monitoring interval in seconds")
    parser.add_argument("--duration", type=int, help="Total monitoring duration in seconds")
    parser.add_argument("--check", action="store_true", help="Quick status check")
    parser.add_argument("--kill-hanging", action="store_true", help="Kill hanging processes")
    
    args = parser.parse_args()
    
    if args.check:
        alive = check_job_status()
        sys.exit(0 if alive else 1)
    elif args.kill_hanging:
        kill_hanging_processes()
    else:
        monitor_processes(args.interval, args.duration)

if __name__ == "__main__":
    main() 