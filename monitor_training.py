#!/usr/bin/env python3
import psutil
import GPUtil
import time
import json
from pathlib import Path

def monitor_system(duration=3600, interval=5):
    """Monitor system resources during training"""
    stats = []
    start_time = time.time()
    
    while time.time() - start_time < duration:
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # GPU Stats
        gpus = GPUtil.getGPUs()
        gpu_stats = []
        for gpu in gpus:
            gpu_stats.append({
                'id': gpu.id,
                'name': gpu.name,
                'load': gpu.load * 100,
                'memory_used': gpu.memoryUsed,
                'memory_total': gpu.memoryTotal,
                'temperature': gpu.temperature
            })
        
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
        gpu_loads = [f"{g['load']:.1f}%" for g in gpu_stats]
        print(f"CPU: {cpu_percent:.1f}% | Memory: {memory.percent:.1f}% | "
              f"GPU Load: {gpu_loads}")
        
        time.sleep(interval)
    
    # Save stats
    with open('system_monitor.json', 'w') as f:
        json.dump(stats, f, indent=2)

if __name__ == "__main__":
    monitor_system() 