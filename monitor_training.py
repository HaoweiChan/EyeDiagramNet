#!/usr/bin/env python3
import psutil
import time
import json
import signal
import sys
from pathlib import Path

# Try to import GPUtil, fallback to alternative GPU monitoring
try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False
    print("Warning: GPUtil not available. GPU monitoring will be limited.")

# Global flag for graceful shutdown
RUNNING = True

def signal_handler(signum, frame):
    global RUNNING
    print(f"\nReceived signal {signum}. Shutting down gracefully...")
    RUNNING = False

def get_gpu_stats():
    """Get GPU stats with fallback options"""
    if HAS_GPUTIL:
        try:
            gpus = GPUtil.getGPUs()
            return [
                {
                    'id': gpu.id,
                    'name': gpu.name,
                    'load': gpu.load * 100,
                    'memory_used': gpu.memoryUsed,
                    'memory_total': gpu.memoryTotal,
                    'temperature': gpu.temperature
                }
                for gpu in gpus
            ]
        except Exception as e:
            print(f"GPUtil error: {e}")
            return []
    else:
        # Alternative: Try nvidia-ml-py if available
        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            gpu_stats = []
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                
                gpu_stats.append({
                    'id': i,
                    'name': name,
                    'load': util.gpu,
                    'memory_used': mem_info.used // 1024**2,  # MB
                    'memory_total': mem_info.total // 1024**2,  # MB
                    'temperature': temp
                })
            return gpu_stats
        except ImportError:
            print("Neither GPUtil nor pynvml available. Install one for GPU monitoring.")
            return []
        except Exception as e:
            print(f"pynvml error: {e}")
            return []

def monitor_system(duration=None, interval=5):
    """Monitor system resources during training"""
    global RUNNING
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    stats = []
    start_time = time.time()
    print(f"Starting system monitoring (interval: {interval}s)...")
    
    if duration:
        print(f"Will run for {duration} seconds")
    else:
        print("Will run until terminated (Ctrl+C or SIGTERM)")
    
    while RUNNING:
        # Check duration if specified
        if duration and (time.time() - start_time) >= duration:
            break
            
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # GPU Stats
            gpu_stats = get_gpu_stats()
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            
            stat = {
                'timestamp': time.time(),
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_gb': memory.used / (1024**3),
                'gpus': gpu_stats,
                'disk_read_mb': disk_io.read_bytes / (1024**2),
                'disk_write_mb': disk_io.write_bytes / (1024**2)
            }
            stats.append(stat)
            
            # Fix the nested f-string issue
            if gpu_stats:
                gpu_loads = [f"{g['load']:.1f}%" for g in gpu_stats]
                print(f"CPU: {cpu_percent:.1f}% | Memory: {memory.percent:.1f}% | "
                      f"GPU Load: {gpu_loads}")
            else:
                print(f"CPU: {cpu_percent:.1f}% | Memory: {memory.percent:.1f}% | "
                      f"GPU: Not available")
            
            time.sleep(interval)
            
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received. Stopping...")
            break
        except Exception as e:
            print(f"Error during monitoring: {e}")
            time.sleep(interval)
    
    # Save stats
    output_file = 'system_monitor.json'
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nMonitoring stopped. Stats saved to {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Monitor system resources')
    parser.add_argument('--duration', type=int, default=None, 
                       help='Duration to monitor in seconds (default: run until stopped)')
    parser.add_argument('--interval', type=int, default=5,
                       help='Monitoring interval in seconds (default: 5)')
    
    args = parser.parse_args()
    monitor_system(duration=args.duration, interval=args.interval) 