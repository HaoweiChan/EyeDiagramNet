#!/usr/bin/env python3
import psutil
import time
import json
from pathlib import Path

# Try to import GPUtil, fallback to alternative GPU monitoring
try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False
    print("Warning: GPUtil not available. GPU monitoring will be limited.")

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

def monitor_system(duration=3600, interval=5):
    """Monitor system resources during training"""
    stats = []
    start_time = time.time()
    
    while time.time() - start_time < duration:
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
    
    # Save stats
    with open('system_monitor.json', 'w') as f:
        json.dump(stats, f, indent=2)

if __name__ == "__main__":
    monitor_system() 