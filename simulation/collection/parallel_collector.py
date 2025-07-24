"""Main training data collector orchestrating eye width simulation data collection."""

# -----------------------------------------------------------------------------
# Limit BLAS / OpenMP thread usage BEFORE NumPy/SciPy are imported.
# This must be done at import time to ensure MKL / OpenBLAS obey the limits.
# Can be overridden by config or environment variables.
# -----------------------------------------------------------------------------
import os
import platform

# Platform-aware BLAS thread defaults
def get_platform_blas_default():
    """Get platform-appropriate BLAS thread default"""
    system = platform.system()
    if system == "Linux":
        # Use 2-4 BLAS threads for better CPU utilization on servers
        cpu_count = psutil.cpu_count() if 'psutil' in globals() else os.cpu_count()
        if cpu_count >= 32:
            return "4"  # High-core servers: 4 BLAS threads per worker
        elif cpu_count >= 16:
            return "3"  # Mid-range servers: 3 BLAS threads per worker
        else:
            return "2"  # Smaller servers: 2 BLAS threads per worker
    elif system == "Darwin":  # macOS
        return "2"  # Balanced: Share threads between BLAS and processes

# Allow config to override BLAS thread count, with platform-aware defaults
default_blas_threads = os.environ.get("BLAS_THREADS", get_platform_blas_default())
os.environ.setdefault("OMP_NUM_THREADS", default_blas_threads)
os.environ.setdefault("MKL_NUM_THREADS", default_blas_threads)
os.environ.setdefault("OPENBLAS_NUM_THREADS", default_blas_threads)
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", default_blas_threads)
os.environ.setdefault("NUMEXPR_NUM_THREADS", default_blas_threads)

# After the limits are in place we can safely import heavy numerical libs.

import time
import yaml
import signal
import pickle
import psutil
import threading
import traceback
import numpy as np
import multiprocessing
import concurrent.futures
import multiprocessing.shared_memory as shm
from pathlib import Path
from datetime import datetime

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Fallback tqdm implementation
    class tqdm:
        def __init__(self, iterable=None, total=None, desc=None, **kwargs):
            self.iterable = iterable
            self.total = total or (len(iterable) if iterable else 0)
            self.desc = desc
            self.current = 0
            if desc:
                print(f"{desc}: 0/{self.total}")
        
        def __iter__(self):
            for item in self.iterable:
                yield item
                self.update(1)
        
        def update(self, n=1):
            self.current += n
            if self.current % max(1, self.total // 10) == 0 or self.current == self.total:
                print(f"{self.desc}: {self.current}/{self.total}")
        
        def close(self):
            pass

from common.signal_utils import read_snp
from simulation.parameters.bound_param import PARAM_SETS_MAP
from simulation.engine.sbr_simulator import snp_eyewidth_simulation
from simulation.io.config_utils import load_config, resolve_trace_pattern, resolve_vertical_dirs, build_argparser
from simulation.io.snp_utils import parse_snps, generate_vertical_snp_pairs
from simulation.io.progress_utils import progress_monitor, report_progress
from simulation.parameters.param_utils import parse_param_types, modify_params_for_inductance

# Import performance monitoring functions
try:
    from simulation.engine.network_utils import print_performance_summary, reset_performance_data
    NETWORK_PROFILING_AVAILABLE = True
except ImportError:
    NETWORK_PROFILING_AVAILABLE = False
    def print_performance_summary(worker_id=None): pass
    def reset_performance_data(): pass

# Global profiling state
_profiling_data = threading.local()
# Global monitoring control
_monitor_proc = None
# Global progress queue for inter-process communication
_progress_queue = None
# Global shared memory registry for cleanup
_shared_memory_blocks = []
# Global shutdown control
_shutdown_event = threading.Event()
_executor = None
_progress_thread = None
_shutdown_in_progress = False
_cleanup_done = False

def signal_handler(signum, frame):
    """Handle SIGINT (Ctrl+C) for immediate shutdown"""
    global _shutdown_event, _shutdown_in_progress, _cleanup_done
    
    # Handle double Ctrl+C - force immediate exit
    if _shutdown_in_progress:
        print("\n[FORCE EXIT] Second interrupt signal received. Forcing immediate exit...", flush=True)
        import os
        os._exit(130)  # Immediate exit without any cleanup
    
    print("\n[SHUTDOWN] Received interrupt signal (Ctrl+C). Setting shutdown event...", flush=True)
    print("[SHUTDOWN] Press Ctrl+C again to force immediate exit if shutdown hangs.", flush=True)
    _shutdown_in_progress = True
    _shutdown_event.set()
    
    # Start a timeout thread for hard exit if cleanup takes too long
    def force_exit_after_timeout():
        time.sleep(10.0)  # 10 second hard timeout
        if not _cleanup_done:
            print("\n[FORCE EXIT] Shutdown timeout exceeded. Forcing immediate exit...", flush=True)
            import os
            os._exit(130)
    
    timeout_thread = threading.Thread(target=force_exit_after_timeout, daemon=True)
    timeout_thread.start()

def register_signal_handlers():
    """Register signal handlers for graceful shutdown"""
    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)

def get_worker_id():
    """Get unique worker identifier for profiling"""
    if hasattr(_profiling_data, 'worker_id'):
        return _profiling_data.worker_id
    
    # Create unique worker ID 
    thread_id = threading.get_ident()
    process_id = os.getpid()
    _profiling_data.worker_id = f"P{process_id}-T{thread_id}"
    return _profiling_data.worker_id

def profile_print(message, elapsed_time=None):
    """Thread/process-safe profiling print with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    worker_id = get_worker_id()
    
    if elapsed_time is not None:
        print(f"[{timestamp}] [{worker_id}] {message} ({elapsed_time:.2f}s)", flush=True)
    else:
        print(f"[{timestamp}] [{worker_id}] {message}", flush=True)

def time_block(description):
    """Context manager for timing code blocks"""
    class TimeBlock:
        def __init__(self, desc):
            self.desc = desc
            self.start_time = None
            
        def __enter__(self):
            self.start_time = time.time()
            profile_print(f"Starting {self.desc}")
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            elapsed = time.time() - self.start_time
            if exc_type is None:
                profile_print(f"Completed {self.desc}", elapsed)
            else:
                profile_print(f"Failed {self.desc} with {exc_type.__name__}", elapsed)
    
    return TimeBlock(description)

def monitor_system_resources(interval=10):
    """Background monitoring of system resources in a separate process with responsive shutdown."""
    print(f"[MONITOR] Starting system resource monitoring (interval: {interval}s)", flush=True)
    
    # Set up signal handler for this process to exit gracefully
    import signal
    
    def monitor_signal_handler(signum, frame):
        print(f"[MONITOR] Received signal {signum}, shutting down monitoring...", flush=True)
        import sys
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, monitor_signal_handler)
    signal.signal(signal.SIGINT, monitor_signal_handler)
    
    while True:
        try:
            # CPU usage
            cpu_percent_per_core = psutil.cpu_percent(interval=1, percpu=True)
            cpu_overall = psutil.cpu_percent(interval=0)
            
            # Memory info
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_gb = memory.used / (1024**3)
            memory_total_gb = memory.total / (1024**3)
            
            # Load average
            load_str = ""
            if hasattr(psutil, "getloadavg"):
                load_avg = psutil.getloadavg()
                load_str = f" | Load: {load_avg[0]:.2f}, {load_avg[1]:.2f}, {load_avg[2]:.2f}"
            
            # Format CPU cores string with better utilization info
            active_cores = sum(1 for p in cpu_percent_per_core if p > 5.0)
            if len(cpu_percent_per_core) > 8:
                cores_str = ", ".join([f"{p:.1f}%" for p in cpu_percent_per_core[:4]])
                cores_str += f" ... " + ", ".join([f"{p:.1f}%" for p in cpu_percent_per_core[-4:]])
            else:
                cores_str = ", ".join([f"{p:.1f}%" for p in cpu_percent_per_core])

            timestamp = datetime.now().strftime("%H:%M:%S")
            message = (f"\n[MONITOR {timestamp}] "
                       f"CPU: {cpu_overall:.1f}% ({active_cores}/{len(cpu_percent_per_core)} cores active) | "
                       f"RAM: {memory_used_gb:.1f}/{memory_total_gb:.1f}GB ({memory_percent:.1f}%)"
                       f"{load_str} | Cores: [{cores_str}]")
            print(message, flush=True)
            
            # Sleep in smaller chunks to be more responsive to termination
            remaining_sleep = interval - 1
            while remaining_sleep > 0:
                sleep_chunk = min(remaining_sleep, 2.0)  # Sleep max 2 seconds at a time
                time.sleep(sleep_chunk)
                remaining_sleep -= sleep_chunk
            
        except KeyboardInterrupt:
            print(f"[MONITOR] Keyboard interrupt, shutting down monitoring...", flush=True)
            break
        except Exception as e:
            print(f"[MONITOR] Error: {e}", flush=True)
            # Sleep in chunks even during error recovery
            remaining_sleep = interval
            while remaining_sleep > 0:
                sleep_chunk = min(remaining_sleep, 2.0)
                time.sleep(sleep_chunk)
                remaining_sleep -= sleep_chunk
    
    print(f"[MONITOR] System resource monitoring stopped.", flush=True)

def start_background_monitoring(interval=10):
    """Start background system monitoring in a separate process."""
    global _monitor_proc
    if _monitor_proc is None:
        _monitor_proc = multiprocessing.Process(
            target=monitor_system_resources, 
            args=(interval,),
            daemon=True,
            name="SystemMonitorProc"
        )
        _monitor_proc.start()

def stop_background_monitoring():
    """Stop background system monitoring process with aggressive shutdown."""
    global _monitor_proc
    if _monitor_proc is not None:
        print("[MONITOR] Stopping system resource monitoring...", flush=True)
        try:
            _monitor_proc.terminate()
            _monitor_proc.join(timeout=1.0)  # Reduced from 5 to 1 second
            
            # Force kill if still alive
            if _monitor_proc.is_alive():
                print("[MONITOR] Force killing monitoring process...", flush=True)
                _monitor_proc.kill()
                _monitor_proc.join(timeout=0.5)  # Reduced from 2 to 0.5 seconds
                
                # If still alive after kill, just abandon it
                if _monitor_proc.is_alive():
                    print("[MONITOR] Warning: Could not kill monitoring process, abandoning...", flush=True)
                
        except Exception as e:
            print(f"[MONITOR] Error stopping monitoring process: {e}", flush=True)
        finally:
            _monitor_proc = None
            print("[MONITOR] System resource monitoring stopped.", flush=True)

def get_optimal_workers(config_blas_threads=2):
    """Calculate optimal number of workers based on system resources and platform"""
    # Get system info
    cpu_count = psutil.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    system = platform.system()
    
    print(f"System: {cpu_count} CPUs, {memory_gb:.1f}GB RAM, Platform: {system}")
    
    # Platform-aware resource allocation
    if system == "Darwin":  # macOS - Conservative for development/testing
        print("Using macOS resource allocation (conservative)")
        # Conservative settings for shared development environment
        # Target: <50% CPU usage, <20GB RAM
        max_memory_gb = min(memory_gb, 20.0)  # Cap at 20GB for safety
        memory_per_worker = 1.5  # Conservative memory estimate per worker
        cpu_utilization_target = 0.5  # Use 50% of cores
        
        # Calculate workers based on memory and CPU constraints
        memory_based_workers = max(1, int(max_memory_gb / memory_per_worker))
        cpu_based_workers = max(1, int(cpu_count * cpu_utilization_target // max(1, config_blas_threads)))
        
        # Use minimum to ensure we don't exceed any constraint
        optimal_workers = min(memory_based_workers, cpu_based_workers)
        
    elif system == "Linux":  # Linux - Balanced for production
        print("Using Linux resource allocation (balanced)")
        # Balanced settings to maximize throughput without CPU thrashing
        # Target: 75-85% CPU usage to avoid over-subscription
        memory_per_worker = 0.5  # Reduced memory estimate for better scaling
        cpu_utilization_target = 0.75  # Use 75% of cores to avoid thrashing
        
        # Reserve some memory for system (5% or 2GB, whichever is larger)
        reserved_memory = max(memory_gb * 0.05, 2.0)
        available_memory = memory_gb - reserved_memory
        
        # Calculate workers based on memory and CPU
        memory_based_workers = max(1, int(available_memory / memory_per_worker))
        
        # For CPU calculation, be more conservative to avoid thrashing
        # Each worker should have at least 1 core, preferably 1-2 cores
        # This accounts for the fact that we'll dynamically adjust BLAS threads
        cpu_based_workers = max(1, int(cpu_count * cpu_utilization_target))
        
        # Use minimum but cap at reasonable maximum
        # Cap at 75% of cores to prevent over-subscription
        max_reasonable_workers = min(int(cpu_count * 0.75), 48)  # Cap at 48 workers max
        optimal_workers = min(memory_based_workers, cpu_based_workers, max_reasonable_workers)
        
    else:  # Unknown platform - Use conservative defaults
        print(f"Unknown platform {system}, using conservative defaults")
        memory_per_worker = 1.5
        cpu_utilization_target = 0.5
        
        memory_based_workers = max(1, int(memory_gb * 0.8 / memory_per_worker))
        cpu_based_workers = max(1, int(cpu_count * cpu_utilization_target // max(1, config_blas_threads)))
        optimal_workers = min(memory_based_workers, cpu_based_workers)
    
    print(f"Resource calculation:")
    print(f"  Memory-based workers: {memory_based_workers} ({memory_per_worker}GB per worker, {available_memory:.1f}GB available)")
    if system == "Linux":
        print(f"  CPU-based workers: {cpu_based_workers} (target {cpu_utilization_target*100:.0f}% CPU)")
    else:
        print(f"  CPU-based workers: {cpu_based_workers} (target {cpu_utilization_target*100:.0f}% CPU, {config_blas_threads} BLAS threads)")
    print(f"  Selected workers: {optimal_workers} (constraining factor: {'memory' if optimal_workers == memory_based_workers else 'CPU' if optimal_workers == cpu_based_workers else 'max_limit'})")
    
    # Calculate predicted BLAS thread allocation using the new aggressive strategy
    if optimal_workers <= 8:
        predicted_blas_threads = min(4, max(2, cpu_count // 4))
    elif optimal_workers <= 16:
        predicted_blas_threads = max(2, min(3, cpu_count // 8))
    else:
        # High worker count: Use strategic oversubscription for Numba benefits
        target_total_threads = int(cpu_count * 1.75)  # 175% oversubscription target
        predicted_blas_threads = max(2, min(3, target_total_threads // optimal_workers))
    
    # Ensure minimum 2 threads per worker for meaningful Numba speedup
    predicted_blas_threads = max(2, predicted_blas_threads)
    predicted_blas_threads = min(predicted_blas_threads, 4)
    total_blas_threads = optimal_workers * predicted_blas_threads
    
    # Apply intelligent oversubscription limit
    max_allowed_threads = int(cpu_count * 2.0)
    if total_blas_threads > max_allowed_threads:
        predicted_blas_threads = max(2, max_allowed_threads // optimal_workers)
        total_blas_threads = optimal_workers * predicted_blas_threads
    
    oversubscription_pct = total_blas_threads/cpu_count*100
    print(f"  Predicted BLAS allocation: {optimal_workers} workers × {predicted_blas_threads} BLAS threads = {total_blas_threads} total threads ({oversubscription_pct:.1f}% of {cpu_count} cores)")
    
    # Explain the oversubscription strategy
    if oversubscription_pct > 100:
        print(f"  Oversubscription rationale: Simulation workloads have I/O waiting, cache misses, and memory access delays")
        print(f"  Expected benefit: {oversubscription_pct/100:.1f}x thread utilization enables {predicted_blas_threads}x Numba speedup per worker")
    
    numba_status = "High" if predicted_blas_threads >= 3 else "Good" if predicted_blas_threads >= 2 else "Limited"
    print(f"  Numba optimization: {numba_status} (per-worker BLAS threads: {predicted_blas_threads})")
    
    return optimal_workers

class BufferedPickleWriter:
    """Buffered writer for pickle files to reduce I/O overhead with interrupt safety"""
    
    def __init__(self, pickle_file, batch_size=10):
        self.pickle_file = Path(pickle_file)
        self.batch_size = batch_size
        self.buffer = []
        self.data = self._load_existing_data()
        self._closed = False
        self._last_flush_time = time.time()
        self._flush_interval = 30.0  # Force flush every 30 seconds
        
        # Register atexit handler to ensure cleanup
        import atexit
        atexit.register(self._emergency_cleanup)
    
    def _load_existing_data(self):
        """Load existing pickle data"""
        if self.pickle_file.exists():
            try:
                with open(self.pickle_file, 'rb') as f:
                    return pickle.load(f)
            except:
                pass
        return {
            'configs': [],
            'line_ews': [], 
            'snp_drvs': [],
            'snp_odts': [],
            'directions': [],
            'meta': {}
        }
    
    def add_result(self, result):
        """Add a simulation result to the buffer"""
        if self._closed or _shutdown_event.is_set():
            # If shutdown is in progress, flush immediately
            self.buffer.append(result)
            self.flush()
            return
            
        self.buffer.append(result)
        current_time = time.time()
        
        # Flush if buffer is full or enough time has passed
        if (len(self.buffer) >= self.batch_size or 
            current_time - self._last_flush_time > self._flush_interval):
            self.flush()
    
    def flush(self):
        """Write buffered results to disk with timeout protection"""
        if not self.buffer or self._closed:
            return
        
        # Skip flush if shutdown is in progress to avoid hanging
        if _shutdown_event.is_set():
            return
        
        try:
            # Add all buffered results to data
            for result in self.buffer:
                self.data['configs'].append(result['config_values'])
                self.data['line_ews'].append(result['line_ews'])
                self.data['snp_drvs'].append(result['snp_drv'])
                self.data['snp_odts'].append(result['snp_odt'])
                self.data['directions'].append(result['directions'])
            
            # Update meta if needed
            if self.buffer and not self.data['meta'].get('config_keys'):
                first_result = self.buffer[0]
                self.data['meta']['config_keys'] = first_result['config_keys']
                self.data['meta']['snp_horiz'] = first_result.get('snp_horiz', '')
                self.data['meta']['n_ports'] = first_result.get('n_ports', 0)
                self.data['meta']['param_types'] = first_result.get('param_types', [])
            
            # Write to disk with timeout protection using thread
            def write_with_timeout():
                self.pickle_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self.pickle_file, 'wb') as f:
                    pickle.dump(self.data, f)
            
            write_thread = threading.Thread(target=write_with_timeout, daemon=True)
            write_thread.start()
            write_thread.join(timeout=5.0)  # 5 second timeout for I/O
            
            if write_thread.is_alive():
                # Write is taking too long, abandon it
                print(f"[WARNING] Pickle write timeout for {self.pickle_file.name}, abandoning...", flush=True)
                return
            
            self.buffer.clear()
            self._last_flush_time = time.time()
            
        except Exception as e:
            # If normal write fails, just log and continue
            print(f"[WARNING] Failed to flush pickle data: {e}", flush=True)
            self.buffer.clear()  # Clear buffer to avoid memory issues
    
    def close(self):
        """Flush remaining data and close"""
        if not self._closed:
            # Only flush if not shutting down to avoid hanging
            if not _shutdown_event.is_set():
                self.flush()
            self._closed = True
    
    def _emergency_cleanup(self):
        """Emergency cleanup for atexit"""
        if not self._closed and self.buffer:
            try:
                self.close()
            except:
                pass

class SNPCache:
    """Shared memory cache for vertical SNP files only"""
    
    def __init__(self):
        self.cache = {}  # {snp_path: metadata}
        self.memory_blocks = []
    
    def add_snp(self, snp_path):
        """Load SNP file and store its data in shared memory."""
        snp_key = str(snp_path)
        if snp_key in self.cache:
            return
            
        try:
            ntwk = read_snp(snp_path)
            s_data = ntwk.s
            f_data = ntwk.f
            
            # Validate data before creating shared memory
            if s_data.size == 0 or f_data.size == 0:
                raise ValueError(f"Empty SNP data in {snp_path}")
            
            # Create shared memory for s-parameters
            s_shm = shm.SharedMemory(create=True, size=s_data.nbytes)
            s_shm_array = np.ndarray(s_data.shape, dtype=s_data.dtype, buffer=s_shm.buf)
            s_shm_array[:] = s_data[:]
            
            # Create shared memory for frequencies
            f_shm = shm.SharedMemory(create=True, size=f_data.nbytes)
            f_shm_array = np.ndarray(f_data.shape, dtype=f_data.dtype, buffer=f_shm.buf)
            f_shm_array[:] = f_data[:]
            
            self.cache[snp_key] = {
                's_name': s_shm.name, 's_shape': s_data.shape, 's_dtype': s_data.dtype,
                'f_name': f_shm.name, 'f_shape': f_data.shape, 'f_dtype': f_data.dtype,
                'nports': ntwk.nports, 'z0': ntwk.z0.tolist() if isinstance(ntwk.z0, np.ndarray) else ntwk.z0
            }
            
            self.memory_blocks.extend([s_shm, f_shm])
            _shared_memory_blocks.extend([s_shm, f_shm])
            
        except Exception as e:
            print(f"[ERROR] Failed to cache SNP file {snp_path}: {e}")
            # Don't add to cache if there was an error
            pass
    
    def get_cache_info(self):
        """Get cache information for passing to workers."""
        return self.cache.copy()
    
    def cleanup(self):
        """Clean up shared memory blocks."""
        cleanup_errors = 0
        for block in self.memory_blocks:
            try:
                block.close()
                block.unlink()
            except FileNotFoundError:
                pass # Already unlinked
            except Exception as e:
                cleanup_errors += 1
                if cleanup_errors <= 3:  # Only log first few errors
                    print(f"[CLEANUP] Error cleaning shared memory block {getattr(block, 'name', 'unknown')}: {e}")
        
        if cleanup_errors > 3:
            print(f"[CLEANUP] {cleanup_errors - 3} additional shared memory cleanup errors (suppressed)")
            
        self.memory_blocks.clear()

def init_worker_process(vertical_cache_info, progress_queue, num_workers):
    """Initialize worker process with shared memory access and progress reporting."""
    worker_start_time = time.time()
    worker_id = os.getpid()
    
    profile_print(f"Worker {worker_id} starting initialization...")
    
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Store cache info and progress queue globally in worker
    global _vertical_cache_info, _progress_queue
    _vertical_cache_info = vertical_cache_info
    _progress_queue = progress_queue
    
    profile_print(f"Worker {worker_id} set up globals")
    
    # ADVANCED FIX: Adaptive BLAS thread management for optimal Numba performance
    # Balance between per-worker performance and CPU thrashing prevention
    cpu_count = psutil.cpu_count()
    
    # Calculate optimal BLAS threads with Numba considerations and realistic oversubscription
    # Simulation workloads are NOT pure CPU-bound - they have I/O, memory access, cache misses
    # We can afford some oversubscription for much better Numba performance
    
    if num_workers <= 8:
        # Low worker count: Give maximum threads per worker for best Numba performance
        optimal_blas_threads = min(4, max(2, cpu_count // 4))  # At least 2, preferably 4
    elif num_workers <= 16:
        # Medium worker count: Still prioritize Numba performance with moderate oversubscription
        optimal_blas_threads = max(2, min(3, cpu_count // 8))  # At least 2 threads per worker
    else:
        # High worker count: Use strategic oversubscription for Numba benefits
        # Target 150-200% CPU threads since simulations have I/O wait time
        target_total_threads = int(cpu_count * 1.75)  # 175% oversubscription target
        optimal_blas_threads = max(2, min(3, target_total_threads // num_workers))
    
    # Ensure minimum 2 threads per worker for meaningful Numba speedup
    optimal_blas_threads = max(2, optimal_blas_threads)
    
    # Cap at 4 threads per worker (diminishing returns beyond this)
    optimal_blas_threads = min(optimal_blas_threads, 4)
    
    # Apply intelligent oversubscription limit for simulation workloads
    total_blas_threads = num_workers * optimal_blas_threads
    max_allowed_threads = int(cpu_count * 2.0)  # Allow up to 200% oversubscription
    if total_blas_threads > max_allowed_threads:
        optimal_blas_threads = max(2, max_allowed_threads // num_workers)
    
    # Set BLAS thread limits for this worker process
    os.environ["OMP_NUM_THREADS"] = str(optimal_blas_threads)
    os.environ["MKL_NUM_THREADS"] = str(optimal_blas_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(optimal_blas_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(optimal_blas_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(optimal_blas_threads)
    
    # Also set Numba threading controls for better performance
    os.environ["NUMBA_NUM_THREADS"] = str(optimal_blas_threads)
    
    total_threads_final = num_workers * optimal_blas_threads
    utilization_pct = total_threads_final/cpu_count*100
    profile_print(f"Worker {worker_id} set BLAS threads to {optimal_blas_threads} ({num_workers} workers × {optimal_blas_threads} threads = {total_threads_final} total for {cpu_count} cores, {utilization_pct:.1f}% utilization)")
    
    if utilization_pct > 100:
        profile_print(f"Worker {worker_id} using {utilization_pct:.0f}% oversubscription for Numba performance (simulation workloads have I/O wait time)")
    
    # Set CPU affinity for better CPU utilization (Linux only)
    if platform.system() == "Linux" and hasattr(os, 'sched_setaffinity'):
        cpu_count = psutil.cpu_count()
        
        # Simple round-robin CPU assignment
        if cpu_count > 1:
            assigned_cpu = worker_id % cpu_count
            try:
                os.sched_setaffinity(0, [assigned_cpu])
                profile_print(f"Worker {worker_id} assigned to CPU {assigned_cpu}")
            except:
                # Ignore failures (common in containers/restricted environments)
                profile_print(f"Worker {worker_id} could not set CPU affinity")
    
    init_time = time.time() - worker_start_time
    profile_print(f"Worker {worker_id} initialization completed", init_time)

def get_snp_from_cache(snp_path, cache_info):
    """Load SNP data from shared memory cache or disk if not cached."""
    snp_key = str(snp_path)
    
    # Try to get from shared memory cache first
    if snp_key in cache_info:
        try:
            cache_data = cache_info[snp_key]
            
            # Reconstruct network from shared memory
            s_shm = shm.SharedMemory(name=cache_data['s_name'])
            f_shm = shm.SharedMemory(name=cache_data['f_name'])
            
            try:
                s_array = np.ndarray(cache_data['s_shape'], dtype=cache_data['s_dtype'], buffer=s_shm.buf)
                f_array = np.ndarray(cache_data['f_shape'], dtype=cache_data['f_dtype'], buffer=f_shm.buf)
                
                # Create a simple object that mimics skrf.Network for our use case
                class CachedNetwork:
                    def __init__(self, s, f, nports, z0, original_path=None):
                        self.s = s.copy()  # Copy to avoid shared memory issues
                        self.f = f.copy()
                        self.nports = nports
                        self.z0 = np.array(z0)
                        
                        # Add compatibility attributes for skrf.Network
                        self.number_of_ports = nports
                        self.name = f"cached_network_{nports}port"  # Add name attribute
                        self._original_path = original_path  # Store original path for fallback
                        
                        # Create a minimal frequency object with the required attributes
                        class MinimalFrequency:
                            def __init__(self, f_array):
                                self.f = f_array
                        
                        self.frequency = MinimalFrequency(f.copy())
                    
                    def flip(self):
                        """Flip the network ports for ODT networks (in-place operation)"""
                        n = self.nports
                        if n % 2 != 0:
                            raise ValueError("Cannot flip network with odd number of ports")
                        
                        # Create port reordering: flip each pair of ports
                        port_order = []
                        for i in range(n // 2):
                            port_order.extend([i + n // 2, i])
                        
                        # Reorder S-parameters
                        self.s = self.s[:, port_order, :][:, :, port_order]
                        
                        # Reorder impedance if it's an array
                        if hasattr(self.z0, '__len__') and len(self.z0) > 1:
                            self.z0 = self.z0[port_order]
                    
                    def subnetwork(self, port_index_list):
                        """Create a subnetwork with specified port ordering"""
                        if len(port_index_list) != self.nports:
                            raise ValueError(f"port_index_list must have length {self.nports}")
                        
                        # Create new CachedNetwork with reordered ports
                        new_s = self.s[:, port_index_list, :][:, :, port_index_list]
                        new_z0 = self.z0[port_index_list] if hasattr(self.z0, '__len__') and len(self.z0) > 1 else self.z0
                        
                        return CachedNetwork(new_s, self.f, self.nports, new_z0)
                    
                    def close(self):
                        """Close method for compatibility (no-op for CachedNetwork)"""
                        pass
                
                # Copy the data from shared memory (this is the key fix)
                cached_network = CachedNetwork(s_array, f_array, cache_data['nports'], cache_data['z0'], str(snp_path))
                
                # CRITICAL: Close the SharedMemory objects after copying data
                # This prevents resource leaks and the ".close()" error
                s_shm.close()
                f_shm.close()
                
                return cached_network
                
            except Exception as inner_e:
                # Ensure shared memory is closed even if there's an error
                try:
                    s_shm.close()
                    f_shm.close()
                except:
                    pass
                raise inner_e
            
        except Exception as e:
            # Fall back to disk loading if shared memory fails
            profile_print(f"Shared memory access failed for {snp_path}, loading from disk: {e}")
    
    # Fall back to direct disk loading (for horizontal SNPs or cache misses)
    # Use process-local cache for repeated disk loads to avoid reloading
    if not hasattr(get_snp_from_cache, '_disk_cache'):
        get_snp_from_cache._disk_cache = {}
    
    if snp_key not in get_snp_from_cache._disk_cache:
        get_snp_from_cache._disk_cache[snp_key] = read_snp(snp_path)
    
    return get_snp_from_cache._disk_cache[snp_key]

def _get_valid_block_sizes(n_lines):
    """Finds divisors of n_lines that result in an even number of blocks."""
    divisors = []
    for i in range(1, int(n_lines**0.5) + 1):
        if n_lines % i == 0:
            if (n_lines // i) % 2 == 0:
                divisors.append(i)
            if i * i != n_lines:
                j = n_lines // i
                if (n_lines // j) % 2 == 0:
                    divisors.append(j)
    if not divisors:  # Fallback for odd n_lines
        divisors.append(1)
    return divisors

def format_error_metadata(trace_snp_path, snp_drv_path, snp_odt_path, combined_config, 
                         sim_directions, sample_idx, total_samples, error_msg):
    """
    Format comprehensive error metadata for debugging purposes.
    
    Args:
        trace_snp_path: Path to horizontal trace SNP file
        snp_drv_path: Path to DRV vertical SNP file  
        snp_odt_path: Path to ODT vertical SNP file
        combined_config: Parameter configuration that caused the error
        sim_directions: Directions array used in simulation
        sample_idx: Current sample index (0-based)
        total_samples: Total number of samples for this trace
        error_msg: The actual error message
    
    Returns:
        Formatted error metadata string
    """
    try:
        # Get config values and keys for display
        config_values, config_keys = combined_config.to_list(return_keys=True)
        config_dict = dict(zip(config_keys, config_values))
        
        # Format the comprehensive error report
        error_report = f"""
=== SIMULATION ERROR METADATA ===
Error Message: {error_msg}
Trace File: {trace_snp_path.name} (Full path: {trace_snp_path})
Vertical DRV: {Path(snp_drv_path).name} (Full path: {snp_drv_path})
Vertical ODT: {Path(snp_odt_path).name} (Full path: {snp_odt_path})
Sample: {sample_idx + 1}/{total_samples}
Directions: {sim_directions.tolist() if hasattr(sim_directions, 'tolist') else sim_directions}
Number of Lines: {len(sim_directions) if sim_directions is not None else 'Unknown'}

Parameter Configuration:
"""
        
        # Add parameter details in organized groups
        electrical_params = ['R_drv', 'R_odt', 'C_drv', 'C_odt', 'L_drv', 'L_odt']
        signal_params = ['pulse_amplitude', 'bits_per_sec', 'vmask']
        ctle_params = ['DC_gain', 'AC_gain', 'fp1', 'fp2']
        
        for group_name, param_list in [('Electrical', electrical_params), 
                                     ('Signal', signal_params), 
                                     ('CTLE', ctle_params)]:
            error_report += f"  {group_name} Parameters:\n"
            for param in param_list:
                if param in config_dict:
                    value = config_dict[param]
                    if isinstance(value, float):
                        error_report += f"    {param}: {value:.6e}\n"
                    else:
                        error_report += f"    {param}: {value}\n"
        
        # Add any remaining parameters
        remaining_params = set(config_dict.keys()) - set(electrical_params + signal_params + ctle_params)
        if remaining_params:
            error_report += "  Other Parameters:\n"
            for param in sorted(remaining_params):
                value = config_dict[param]
                if isinstance(value, float):
                    error_report += f"    {param}: {value:.6e}\n"
                else:
                    error_report += f"    {param}: {value}\n"
        
        error_report += "================================="
        
        return error_report
        
    except Exception as meta_error:
        # Fallback if metadata formatting fails
        return f"""
=== SIMULATION ERROR (METADATA FORMATTING FAILED) ===
Error Message: {error_msg}
Trace File: {trace_snp_path}
Vertical DRV: {snp_drv_path}
Vertical ODT: {snp_odt_path}
Sample: {sample_idx + 1}/{total_samples}
Metadata Error: {meta_error}
=====================================================
"""


def collect_trace_simulation_data(trace_snp_file, vertical_pairs_with_counts, combined_params, 
                                pickle_dir, param_type_names, enable_direction=True, 
                                batch_size=10, debug=False, use_optimized=False,
                                simulator_type='sbr'):
    """
    Collect eye width simulation data for a single trace SNP with multiple vertical pairs.
    This is the new coarse-grained task that processes all samples for one trace file.
    
    Args:
        trace_snp_file: Path to horizontal trace SNP file
        vertical_pairs_with_counts: List of (vertical_snp_pair, sample_count) tuples
        combined_params: ParameterSet containing all required parameters  
        pickle_dir: Directory to save pickle files
        param_type_names: List of parameter type names
        enable_direction: Whether to use random directions (True) or all ones (False)
        batch_size: Number of simulations to buffer before writing to disk
        debug: Debug mode flag
        use_optimized: Whether to use optimized Phase 1 functions (default: False)
        simulator_type: Type of simulator to use ('sbr' or 'der')
    """
    task_start_time = time.time()
    worker_id = os.getpid()
    total_simulations = sum(count for _, count in vertical_pairs_with_counts)
    
    # Convert string arguments back to Path objects for processing
    trace_snp_path = Path(trace_snp_file)
    pickle_dir_path = Path(pickle_dir)
    
    profile_print(f"Worker {worker_id} received task: {trace_snp_path.name} ({total_simulations} simulations)")
    
    # Reset performance monitoring for this task
    reset_performance_data()
    
    if not vertical_pairs_with_counts:
        profile_print(f"Worker {worker_id} - no work to do")
        return
    
    # Load horizontal SNP once per trace file
    snp_start_time = time.time()
    profile_print(f"Worker {worker_id} loading horizontal SNP: {trace_snp_path.name}")
    trace_ntwk = read_snp(trace_snp_path)
    snp_load_time = time.time() - snp_start_time
    profile_print(f"Worker {worker_id} loaded horizontal SNP", snp_load_time)
    
    n_ports = trace_ntwk.nports
    n_lines = n_ports // 2
    if n_lines == 0:
        raise ValueError(f"Invalid n_ports={n_ports}, n_lines would be 0")
    
    # Dynamically select the simulation function
    if simulator_type == 'der':
        from simulation.engine.der_simulator import snp_der_simulation
        simulation_func = snp_der_simulation
    else: # Default to sbr
        simulation_func = snp_eyewidth_simulation

    pickle_file = pickle_dir_path / f"{trace_snp_path.stem}.pkl"
    
    total_simulations = sum(count for _, count in vertical_pairs_with_counts)
    
    # Initialize buffered writer
    writer = BufferedPickleWriter(pickle_file, batch_size)
    
    if debug:
        print(f"Processing {len(vertical_pairs_with_counts)} vertical pairs for {trace_snp_path.name}")
        print(f"Detected {n_ports} ports ({n_lines} lines)")
    
    # profile_print(f"Starting {trace_snp_path.name}: {total_simulations} simulations")
    
    total_completed = 0
    
    # Process all vertical pairs and their samples
    for vertical_snp_pair, sample_count in vertical_pairs_with_counts:
        # Check for shutdown between vertical pairs
        if _shutdown_event.is_set():
            profile_print(f"Shutdown detected, stopping {trace_snp_path.name} early")
            break
            
        # Load vertical SNPs from cache
        snp_drv_path, snp_odt_path = vertical_snp_pair
        vertical_start_time = time.time()
        
        # Check if files are in cache for performance monitoring
        drv_cached = str(snp_drv_path) in _vertical_cache_info
        odt_cached = str(snp_odt_path) in _vertical_cache_info
        cache_status = f"DRV:{'cache' if drv_cached else 'disk'}, ODT:{'cache' if odt_cached else 'disk'}"
        
        profile_print(f"Worker {worker_id} loading vertical SNPs ({cache_status}): {Path(snp_drv_path).name}, {Path(snp_odt_path).name}")
        
        drv_ntwk = get_snp_from_cache(snp_drv_path, _vertical_cache_info)
        odt_ntwk = get_snp_from_cache(snp_odt_path, _vertical_cache_info)
        
        vertical_load_time = time.time() - vertical_start_time
        profile_print(f"Worker {worker_id} loaded vertical SNPs ({cache_status})", vertical_load_time)
        
        for sample_idx in range(sample_count):
            # Check for shutdown more frequently - every 10 samples or immediately
            if sample_idx % 10 == 0 and _shutdown_event.is_set():
                profile_print(f"Shutdown detected, stopping {trace_snp_path.name} early")
                break
            
            # Retry logic for failed simulations
            max_retries = 3
            retry_count = 0
            simulation_successful = False
            
            while not simulation_successful and retry_count <= max_retries:
                # Sample parameters (get new config for retries)
                combined_config = combined_params.sample()
                
                # Performance monitoring for simulation
                sim_start_time = time.time()
                
                try:
                    # Set directions
                    if enable_direction:
                        # Generate directions in a block-wise pattern
                        valid_block_sizes = _get_valid_block_sizes(n_lines)
                        block_size = np.random.choice(valid_block_sizes)
                        n_blocks = n_lines // block_size
                        
                        # Create an equal number of 0 and 1 blocks and shuffle them
                        blocks = [0] * (n_blocks // 2) + [1] * (n_blocks // 2)
                        if n_blocks % 2 != 0:
                            blocks.append(np.random.randint(0,2))

                        np.random.shuffle(blocks)
                        
                        # Repeat the blocks to create the final directions array
                        sim_directions = np.repeat(blocks, block_size)
                        # Truncate if n_blocks was odd and we added an extra
                        if len(sim_directions) > n_lines:
                            sim_directions = sim_directions[:n_lines]
                    else:
                        sim_directions = np.ones(n_lines, dtype=int)
                    
                    # Run simulation with comprehensive error handling
                    try:
                        # Prepare simulator-specific arguments
                        sim_kwargs = {
                            'config': combined_config,
                            'snp_files': (trace_ntwk, drv_ntwk, odt_ntwk),
                        }
                        if simulator_type == 'sbr':
                            sim_kwargs['directions'] = sim_directions
                            sim_kwargs['use_optimized'] = use_optimized
                        
                        line_ew = simulation_func(**sim_kwargs)
                        simulation_successful = True
                        
                    except Exception as sim_error:
                        # Handle simulation errors with detailed metadata
                        error_msg = str(sim_error)
                        retry_count += 1
                        
                        # Format comprehensive error metadata
                        metadata_report = format_error_metadata(
                            trace_snp_path, snp_drv_path, snp_odt_path, combined_config,
                            sim_directions, sample_idx, sample_count, error_msg
                        )
                        
                        if retry_count <= max_retries:
                            print(f"[ERROR] Simulation failed (attempt {retry_count}/{max_retries + 1}), will retry with new config:")
                            print(metadata_report)
                            print(f"[RETRY] Retrying simulation for {trace_snp_path.name}, sample {sample_idx + 1}")
                            continue
                        else:
                            print(f"[ERROR] Simulation failed after {max_retries + 1} attempts, giving up:")
                            print(metadata_report)
                            print(f"[SKIP] Skipping simulation for {trace_snp_path.name}, sample {sample_idx + 1}")
                            break
                    
                    if not simulation_successful:
                        continue  # Retry with new config
                
                except Exception as outer_error:
                    # Handle errors in directions generation or other setup
                    error_msg = str(outer_error)
                    retry_count += 1
                    
                    # Format comprehensive error metadata
                    metadata_report = format_error_metadata(
                        trace_snp_path, snp_drv_path, snp_odt_path, combined_config,
                        sim_directions if 'sim_directions' in locals() else None,
                        sample_idx, sample_count, error_msg
                    )
                    
                    if retry_count <= max_retries:
                        print(f"[ERROR] Setup/preprocessing failed (attempt {retry_count}/{max_retries + 1}), will retry:")
                        print(metadata_report)
                        print(f"[RETRY] Retrying setup for {trace_snp_path.name}, sample {sample_idx + 1}")
                        continue
                    else:
                        print(f"[ERROR] Setup/preprocessing failed after {max_retries + 1} attempts:")
                        print(metadata_report)
                        print(f"[SKIP] Skipping simulation for {trace_snp_path.name}, sample {sample_idx + 1}")
                        break
            
            # Always count this sample (successful or failed after retries)
            # If simulation failed after all retries, skip result processing but count the sample
            if not simulation_successful:
                total_completed += 1
                report_progress(1, _progress_queue, _shutdown_event)
                continue
            
            # Handle tuple return
            if isinstance(line_ew, tuple):
                line_ew, actual_directions = line_ew
                sim_directions = actual_directions
            
            # Process results
            line_ew = np.array(line_ew)
            line_ew[line_ew >= 99.9] = -0.1
            
            # Create result
            config_values, config_keys = combined_config.to_list(return_keys=True)
            result = {
                'config_values': config_values,
                'config_keys': config_keys,
                'line_ews': line_ew.tolist(),
                'snp_drv': snp_drv_path.as_posix(),
                'snp_odt': snp_odt_path.as_posix(),
                'directions': sim_directions.tolist(),
                'snp_horiz': str(trace_snp_path),
                'n_ports': n_ports,
                'param_types': param_type_names
            }
            
            # Add to buffered writer
            writer.add_result(result)
            total_completed += 1
            
            # Report progress after each simulation
            report_progress(1, _progress_queue, _shutdown_event)
            
            # Performance monitoring with BLAS/Numba analysis
            sim_time = time.time() - sim_start_time
            if sim_time > 5.0:  # Log slow simulations
                blas_threads = os.environ.get("OMP_NUM_THREADS", "unknown")
                profile_print(f"Worker {worker_id} slow simulation: {sim_time:.1f}s for {trace_snp_path.name} sample {sample_idx+1} (BLAS threads: {blas_threads})")
            
            # Track performance statistics for optimization
            if not hasattr(collect_trace_simulation_data, '_perf_stats'):
                collect_trace_simulation_data._perf_stats = []
            collect_trace_simulation_data._perf_stats.append(sim_time)
            
            if debug:
                print(f"  Completed simulation {sample_idx+1}/{sample_count}: EW={line_ew} ({sim_time:.1f}s)")
                
        # Break outer loop if shutdown detected during inner loop
        if _shutdown_event.is_set():
            break
    
    # Ensure all data is written
    writer.close()
    
    # Final progress report is not needed since we report after each simulation
    task_total_time = time.time() - task_start_time
    
    # Performance analysis for optimization
    if hasattr(collect_trace_simulation_data, '_perf_stats') and collect_trace_simulation_data._perf_stats:
        perf_stats = collect_trace_simulation_data._perf_stats
        avg_sim_time = np.mean(perf_stats)
        max_sim_time = np.max(perf_stats)
        min_sim_time = np.min(perf_stats)
        
        blas_threads = os.environ.get("OMP_NUM_THREADS", "unknown")
        profile_print(f"Worker {worker_id} completed {trace_snp_path.name}: {total_completed}/{total_simulations} simulations", task_total_time)
        profile_print(f"Worker {worker_id} performance: avg={avg_sim_time:.1f}s, min={min_sim_time:.1f}s, max={max_sim_time:.1f}s per simulation (BLAS threads: {blas_threads})")
        
        # Reset stats for next task
        collect_trace_simulation_data._perf_stats = []
    else:
        profile_print(f"Worker {worker_id} completed {trace_snp_path.name}: {total_completed}/{total_simulations} simulations", task_total_time)
    
    # Print network performance summary for this task
    print_performance_summary(worker_id)
    
    if debug:
        print(f"Completed {total_completed} simulations for {trace_snp_path.name}")
    
    return total_completed

def cleanup_shared_memory():
    """Clean up all shared memory blocks with comprehensive error handling"""
    print("[CLEANUP] Cleaning up shared memory blocks...", flush=True)
    cleanup_count = 0
    error_count = 0
    
    for block in _shared_memory_blocks:
        try:
            block.close()
            block.unlink()
            cleanup_count += 1
        except FileNotFoundError:
            # Already unlinked, this is fine
            cleanup_count += 1
        except Exception as e:
            error_count += 1
            print(f"[CLEANUP] Error cleaning up shared memory block: {e}", flush=True)
    
    _shared_memory_blocks.clear()
    
    if cleanup_count > 0 or error_count > 0:
        print(f"[CLEANUP] Shared memory cleanup: {cleanup_count} blocks cleaned, {error_count} errors", flush=True)
    else:
        print("[CLEANUP] No shared memory blocks to clean up", flush=True)



def get_multiprocessing_start_method():
    """Get appropriate multiprocessing start method for the platform"""
    system = platform.system()
    if system == "Darwin":  # macOS
        return "spawn"  # forkserver not available on macOS
    elif system == "Linux":
        return "forkserver"  # Preferred on Linux
    else:
        return "spawn"  # Safe default

def run_with_executor(trace_tasks, combined_params, trace_specific_output_dir, param_types, 
                     enable_direction, num_workers, batch_size, vertical_cache_info=None, use_optimized=False,
                     simulator_type='sbr'):
    """
    Run simulations using ProcessPoolExecutor with optimized task distribution and graceful shutdown.
    This version uses a bounded queue to avoid overwhelming the executor with too many initial tasks.
    
    Args:
        trace_tasks: List of (trace_snp_file, vertical_pairs_with_counts) tuples
        combined_params: Parameter set
        trace_specific_output_dir: Output directory
        param_types: Parameter types
        enable_direction: Direction flag
        num_workers: Number of workers
        batch_size: Batch size for buffered writing
        vertical_cache_info: Shared memory cache info for vertical SNPs
        use_optimized: Whether to use optimized Phase 1 functions (default: False)
        simulator_type: Type of simulator to use ('sbr' or 'der')
    """
    global _executor, _progress_thread
    
    print(f"Using ProcessPoolExecutor with {num_workers} processes...")
    
    # Set appropriate start method
    start_method = get_multiprocessing_start_method()
    print(f"Using multiprocessing start method: {start_method}")
    try:
        multiprocessing.set_start_method(start_method, force=True)
    except RuntimeError:
        # Already set, continue
        pass
    
    # Calculate total expected simulations for progress tracking
    total_expected = sum(
        sum(count for _, count in vertical_pairs_with_counts)
        for _, vertical_pairs_with_counts in trace_tasks
    )
    
    # Create progress queue and start progress monitor
    progress_queue = multiprocessing.Queue()
    progress_thread = threading.Thread(
        target=progress_monitor,
        args=(progress_queue, total_expected, 60, _shutdown_event),
        daemon=True,
        name="ProgressMonitor"
    )
    progress_thread.start()
    _progress_thread = progress_thread
    
    executor_start_time = time.time()
    
    executor = concurrent.futures.ProcessPoolExecutor(
        max_workers=num_workers, 
        initializer=init_worker_process,
        initargs=(vertical_cache_info or {}, progress_queue, num_workers)
    )
    _executor = executor
    
    try:
        task_iterator = iter(trace_tasks)
        active_futures = {}  # {future: (trace_snp_file, vertical_pairs)}
        max_active_futures = num_workers * 4  # Keep the queue size proportional to workers

        def submit_next_task():
            """Helper to submit the next available task from the iterator."""
            try:
                trace_snp_file, vertical_pairs = next(task_iterator)
                # Simplify arguments to reduce serialization overhead
                future = executor.submit(
                    collect_trace_simulation_data,
                    str(trace_snp_file), vertical_pairs, combined_params, 
                    str(trace_specific_output_dir), param_types, enable_direction, batch_size, False, use_optimized,
                    simulator_type
                )
                active_futures[future] = (trace_snp_file, vertical_pairs)
                return True
            except StopIteration:
                return False

        # Submit the initial batch of tasks to fill the queue
        for _ in range(min(len(trace_tasks), max_active_futures)):
            submit_next_task()

        print(f"Submitted initial {len(active_futures)}/{len(trace_tasks)} tasks to ProcessPoolExecutor (max queue: {max_active_futures})")
        print(f"Waiting for workers to start processing tasks...")
        
        # Give workers a moment to initialize
        time.sleep(2.0)

        failed_tasks = []
        completed_tasks = 0
        total_simulations_completed = 0
        last_completion_time = time.time()
        worker_health_timeout = 300.0  # 5 minutes without any completion is concerning
        
        while active_futures:
            # Wait for at least one future to complete with timeout for health monitoring
            done, _ = concurrent.futures.wait(
                active_futures.keys(), return_when=concurrent.futures.FIRST_COMPLETED, timeout=60.0
            )
            
            # Check worker health if no tasks completed recently
            current_time = time.time()
            if current_time - last_completion_time > worker_health_timeout:
                print(f"[WARNING] No tasks completed in {worker_health_timeout/60:.1f} minutes. "
                      f"Active tasks: {len(active_futures)}, Completed: {completed_tasks}", flush=True)
                last_completion_time = current_time  # Reset to avoid spam

            for future in done:
                task_info = active_futures.pop(future)  # Remove from active dict

                if _shutdown_event.is_set():
                    continue

                # Process the result
                exc = future.exception()
                if exc is not None:
                    print(f"\n--- Task FAILED for {task_info[0]} ---")
                    traceback.print_exception(type(exc), exc, exc.__traceback__)
                    failed_tasks.append((task_info, exc))
                else:
                    completed_tasks += 1
                    sim_count = future.result()
                    if sim_count is not None:
                        total_simulations_completed += sim_count
                    last_completion_time = time.time()  # Update health tracking

                # Replenish the queue with a new task
                submit_next_task()
            
            if _shutdown_event.is_set():
                print("[EXECUTOR] Shutdown detected, cancelling remaining futures...")
                for f in active_futures.keys():
                    f.cancel()
                break

        total_time = time.time() - executor_start_time
        
        if _shutdown_event.is_set():
            print(f"ProcessPoolExecutor interrupted after {total_time:.2f}s")
        else:
            print(f"ProcessPoolExecutor completed in {total_time:.2f}s")
            
        print(f"Completed {completed_tasks}/{len(trace_tasks)} tasks")
        print(f"Total simulations completed: {total_simulations_completed}/{total_expected}")

        if failed_tasks:
            print(f"\nWarning: {len(failed_tasks)} trace files failed.")
            
    except KeyboardInterrupt:
        print("[EXECUTOR] Keyboard interrupt detected in executor")
        _shutdown_event.set()
    except Exception as e:
        print(f"[EXECUTOR] Unexpected error: {e}")
        traceback.print_exc()
    finally:
        # Clean shutdown of executor and progress monitor
        global _cleanup_done
        if not _cleanup_done:
            try:
                print("[CLEANUP] Shutting down executor...")
                
                # Force terminate all worker processes immediately
                if hasattr(executor, '_processes') and executor._processes:
                    for p in executor._processes.values():
                        if p.is_alive():
                            p.terminate()
                    
                    # Give processes 1 second to terminate gracefully
                    time.sleep(1.0)
                    
                    # Force kill any remaining processes
                    for p in executor._processes.values():
                        if p.is_alive():
                            p.kill()
                
                executor.shutdown(wait=False, cancel_futures=True)
                
                # Stop progress monitor aggressively
                try:
                    progress_queue.put(('stop', None), timeout=0.1)
                except:
                    pass
                
                if progress_thread.is_alive():
                    progress_thread.join(timeout=0.5)
                    # Don't wait longer - thread will be daemon
                    
                print("[CLEANUP] Executor cleanup completed.")
            except Exception as cleanup_error:
                print(f"[CLEANUP] Error during cleanup: {cleanup_error}")
            finally:
                _executor = None
                _progress_thread = None

def validate_resource_usage():
    """Validate that current resource allocation is within platform-appropriate bounds"""
    system = platform.system()
    cpu_count = psutil.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    # Get current process info
    current_process = psutil.Process()
    current_memory_gb = current_process.memory_info().rss / (1024**3)
    
    # Platform-specific validation
    if system == "Darwin":  # macOS
        # Validate CPU usage (should be <50% sustained)
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 60:  # 10% tolerance
            print(f"WARNING: CPU usage {cpu_percent:.1f}% exceeds macOS target (50%)")
        
        # Validate memory usage (should be <20GB total)
        memory_used = psutil.virtual_memory().used / (1024**3)
        if memory_used > 22:  # 2GB tolerance
            print(f"WARNING: Memory usage {memory_used:.1f}GB exceeds macOS target (20GB)")
            
        print(f"macOS resource validation: CPU {cpu_percent:.1f}%, Memory {memory_used:.1f}GB")
        
    elif system == "Linux":  # Linux
        # For Linux, just log current usage (aggressive usage is expected)
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_used = psutil.virtual_memory().used / (1024**3)
        print(f"Linux resource validation: CPU {cpu_percent:.1f}%, Memory {memory_used:.1f}GB (aggressive usage expected)")
        
    return True

def validate_worker_allocation(num_workers, blas_threads):
    """Validate that worker allocation is appropriate for the platform"""
    system = platform.system()
    cpu_count = psutil.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    total_threads = num_workers * blas_threads
    
    if system == "Darwin":  # macOS
        # Should not exceed 50% of cores
        if total_threads > cpu_count * 0.6:  # 10% tolerance
            print(f"WARNING: Total threads ({total_threads}) may exceed macOS target (50% of {cpu_count} cores)")
            
        # Memory check
        estimated_memory = num_workers * 1.5  # 1.5GB per worker estimate
        if estimated_memory > 22:  # 2GB tolerance above 20GB target
            print(f"WARNING: Estimated memory usage ({estimated_memory:.1f}GB) may exceed macOS target (20GB)")
            
    elif system == "Linux":  # Linux
        # Should use most cores aggressively
        if total_threads < cpu_count * 0.8:
            print(f"INFO: Thread usage ({total_threads}) is conservative for Linux (could use up to 95% of {cpu_count} cores)")
            
    print(f"Worker allocation validation: {num_workers} workers × {blas_threads} BLAS threads = {total_threads} total threads")
    return True

def main():
    """Main function for parallel data collection"""
    # Register signal handlers for graceful shutdown
    register_signal_handlers()
    print("Registered signal handlers for graceful shutdown")
    
    # Start background system monitoring immediately to show baseline system state
    start_background_monitoring(interval=600)
    print("Started system monitoring...")
    
    args = build_argparser().parse_args()
    
    # Add simulator type argument
    parser = build_argparser()
    parser.add_argument('--simulator-type', type=str, default='sbr', choices=['sbr', 'der'],
                        help='Type of simulator to use for data collection.')
    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Config file not found: {args.config}")
        print("Please create the config file or provide command line arguments")
        stop_background_monitoring()
        return
    except Exception as e:
        print(f"Error loading config: {e}")
        stop_background_monitoring()
        return
    
    # Get actual BLAS thread count from environment (already set by platform defaults)
    runner_config = config.get('runner', {})
    config_blas_threads = runner_config.get('blas_threads', None)
    actual_blas_threads = int(os.environ.get("OMP_NUM_THREADS", "2"))
    
    if config_blas_threads and config_blas_threads != actual_blas_threads:
        print(f"Note: BLAS threads set to {actual_blas_threads} in environment, "
              f"config specifies {config_blas_threads}. Environment takes precedence.")
    
    # Use actual environment value for worker calculations
    blas_threads = actual_blas_threads
    
    # Resolve dataset paths
    horizontal_dataset = config.get('dataset', {}).get('horizontal_dataset', {})
    vertical_dataset = config.get('dataset', {}).get('vertical_dataset')
    
    # Override config with command line arguments
    trace_pattern_key = args.trace_pattern if args.trace_pattern else config['data']['trace_pattern']
    trace_pattern = resolve_trace_pattern(trace_pattern_key, horizontal_dataset)
    vertical_dirs = resolve_vertical_dirs(vertical_dataset)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['data']['output_dir'])
    param_type_str = args.param_type if args.param_type else config['boundary']['param_type']
    param_types = parse_param_types(param_type_str)
    max_samples = args.max_samples if args.max_samples else config['boundary']['max_samples']
    
    # Handle enable_direction logic (default to False)
    enable_direction = args.enable_direction or config['boundary'].get('enable_direction', False)
    
    # Handle enable_inductance logic (default to False)  
    enable_inductance = args.enable_inductance or config['boundary'].get('enable_inductance', False)
    
    # Handle simulator type
    simulator_type = args.simulator_type or config.get('runner', {}).get('simulator_type', 'sbr')

    # Handle use_optimized logic (default to False)
    use_optimized = config.get('runner', {}).get('use_optimized', False)
    
    debug = args.debug if args.debug else config.get('debug', False)
    
    # Determine number of workers: Command-line -> Environment -> Config file -> Optimal calculation
    env_workers = os.environ.get('MAX_WORKERS')
    config_workers = runner_config.get('max_workers')
    
    if args.max_workers:
        max_workers = args.max_workers
        print(f"Using max_workers from command line: {max_workers}")
    elif env_workers:
        max_workers = int(env_workers)
        print(f"Using max_workers from environment variable: {max_workers}")
    elif config_workers:
        max_workers = config_workers
        print(f"Using max_workers from config file: {max_workers}")
    else:
        print("max_workers not specified, calculating optimal number...")
        max_workers = get_optimal_workers(blas_threads)
    
    # Get batch size from config with platform-aware defaults (after max_workers is determined)
    system = platform.system()
    if system == "Linux":
        # Use larger batch sizes on servers for better I/O performance
        default_batch_size = min(50, max(20, max_workers * 2))
    else:
        # Conservative batch size for other platforms
        default_batch_size = 10
    
    batch_size = runner_config.get('batch_size', default_batch_size)
    
    print(f"Using configuration:")
    print(f"  Trace pattern: {trace_pattern_key} -> {trace_pattern}")
    print(f"  Vertical dirs: {vertical_dirs}")
    print(f"  Output dir: {output_dir}")
    print(f"  Parameter types: {param_types}")
    print(f"  Max samples: {max_samples}")
    print(f"  Enable direction: {enable_direction}")
    print(f"  Enable inductance: {enable_inductance}")
    print(f"  Use optimized: {use_optimized}")
    print(f"  Debug mode: {debug}")
    print(f"  Max workers: {max_workers}")
    print(f"  Batch size: {batch_size}")
    print(f"  BLAS threads: {blas_threads}")
    print(f"  Platform: {platform.system()}")
    print(f"  Simulator Type: {simulator_type}")
    
    # Validate resource allocation before starting
    print("\n--- Resource Validation ---")
    validate_worker_allocation(max_workers, blas_threads)
    validate_resource_usage()
    print("---------------------------\n")
    
    # Create base output directory and trace-specific subdirectory
    base_output_dir = output_dir
    trace_specific_output_dir = base_output_dir / trace_pattern_key
    trace_specific_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load trace SNP files
    trace_snps = parse_snps(trace_pattern)
    if len(trace_snps) == 0:
        raise ValueError(f"No trace SNP files found in: {trace_pattern}")
    
    print(f"Found {len(trace_snps)} trace SNP files")
    
    # Generate vertical SNP pairs for each trace SNP
    vertical_pairs = generate_vertical_snp_pairs(vertical_dirs, len(trace_snps), trace_snps, base_output_dir, trace_pattern_key)
    if vertical_dirs is None:
        print(f"Generated {len(vertical_pairs)} thru SNP pairs (auto-generated)")
    else:
        print(f"Generated {len(vertical_pairs)} vertical SNP pairs from {len(vertical_dirs)} directories")
    
    # Combine all requested parameter sets
    combined_params = None
    for param_type in param_types:
        param_set = PARAM_SETS_MAP[param_type]
        if combined_params is None:
            combined_params = param_set
        else:
            combined_params = combined_params + param_set
    
    # Apply inductance modification if needed
    combined_params = modify_params_for_inductance(combined_params, enable_inductance)
    
    # Save the collection config to pickle directory for reference
    config_save_path = trace_specific_output_dir / 'collection_config.yaml'
    if not config_save_path.exists():
        try:
            with open(config_save_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            print(f"Saved collection config to: {config_save_path}")
        except Exception as e:
            print(f"Warning: Could not save config file: {e}")
    
    # Build task structure - group trace files for better load balancing
    trace_tasks = []  # [(trace_snp_file, [(vertical_pair, samples_needed)])]
    pending_work = []  # [(trace_snp, vertical_pair, samples_needed)]
    
    # Collect all pending work first
    for trace_snp, vertical_pair in zip(trace_snps, vertical_pairs):
        pickle_file = trace_specific_output_dir / f"{Path(trace_snp).stem}.pkl"
        current_samples = 0
        
        if pickle_file.exists():
            try:
                with open(pickle_file, 'rb') as f:
                    existing_data = pickle.load(f)
                current_samples = len(existing_data.get('configs', []))
            except:
                current_samples = 0
        
        if current_samples < max_samples:
            samples_needed = max_samples - current_samples
            pending_work.append((trace_snp, vertical_pair, samples_needed))
    
    # Group work into larger tasks to reduce overhead
    # Target: 2-4 tasks per worker for good load balancing
    target_tasks = max_workers * 3
    if len(pending_work) > target_tasks:
        # Group multiple trace files per task
        files_per_task = len(pending_work) // target_tasks
        for i in range(0, len(pending_work), files_per_task):
            batch = pending_work[i:i + files_per_task]
            if batch:
                # For simplicity, create one task per trace file but with larger batches
                for trace_snp, vertical_pair, samples_needed in batch:
                    trace_tasks.append((trace_snp, [(vertical_pair, samples_needed)]))
    else:
        # Few files, create one task per file
        for trace_snp, vertical_pair, samples_needed in pending_work:
            trace_tasks.append((trace_snp, [(vertical_pair, samples_needed)]))
    
    total_simulations = sum(
        sum(count for _, count in vertical_pairs_with_counts)
        for _, vertical_pairs_with_counts in trace_tasks
    )
    
    print(f"Created {len(trace_tasks)} tasks for {total_simulations} total simulations")
    print(f"Task distribution: {len(trace_tasks)} tasks, ~{total_simulations/len(trace_tasks) if trace_tasks else 0:.1f} simulations/task")
    print(f"Load balancing: {len(trace_tasks)} tasks for {max_workers} workers (target: {max_workers * 3} tasks)")
    
    if len(trace_tasks) == 0:
        print("All files already have sufficient samples")
        stop_background_monitoring()
        return
    
    # Pre-load vertical SNPs in shared memory cache for massive performance gain
    print("Pre-loading vertical SNP files in shared memory cache...")
    cache_start_time = time.time()
    
    vertical_cache = SNPCache()
    
    # Collect all unique vertical SNP files that workers will need
    unique_vertical_snps = set()
    for _, vertical_pair in zip(trace_snps, vertical_pairs):
        snp_drv_path, snp_odt_path = vertical_pair
        unique_vertical_snps.add(snp_drv_path)
        unique_vertical_snps.add(snp_odt_path)
    
    print(f"Loading {len(unique_vertical_snps)} unique vertical SNP files into shared memory...")
    
    # Load SNPs with tqdm progress bar
    sorted_snps = sorted(unique_vertical_snps)
    with tqdm(sorted_snps, desc="Loading SNPs to cache", unit="file", 
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
        for snp_path in pbar:
            snp_load_start = time.time()
            vertical_cache.add_snp(snp_path)
            snp_load_time = time.time() - snp_load_start
            
            # Update tqdm description with current file and timing
            pbar.set_postfix_str(f"{Path(snp_path).name} ({snp_load_time:.2f}s)")
    
    # Calculate memory usage
    total_memory_mb = sum(block.size for block in vertical_cache.memory_blocks) / (1024 * 1024)
    print(f"Shared memory allocated: {total_memory_mb:.1f} MB ({len(vertical_cache.memory_blocks)} blocks)")
    
    vertical_cache_info = vertical_cache.get_cache_info()
    cache_load_time = time.time() - cache_start_time
    
    print(f"Shared memory cache loaded in {cache_load_time:.2f}s")
    print(f"Cache contains {len(vertical_cache_info)} vertical SNP files")
    
    # Calculate time savings based on actual cache loading performance
    avg_cache_load_time = cache_load_time / len(unique_vertical_snps) if unique_vertical_snps else 0
    estimated_worker_load_time = max(10.0, avg_cache_load_time * 2)  # Workers typically 2x slower due to overhead
    
    # Without cache: each worker loads each vertical SNP independently
    without_cache_time = len(unique_vertical_snps) * estimated_worker_load_time * max_workers
    
    # With cache: one-time cache load + minimal worker access time
    with_cache_time = cache_load_time + (max_workers * 0.1 * len(unique_vertical_snps))  # 0.1s per cache access
    
    time_saved = without_cache_time - with_cache_time
    efficiency_percentage = (time_saved / without_cache_time * 100) if without_cache_time > 0 else 0
    
    print(f"Time analysis:")
    print(f"  Without cache: {without_cache_time:.0f}s ({max_workers} workers × {len(unique_vertical_snps)} files × {estimated_worker_load_time:.1f}s)")
    print(f"  With cache: {with_cache_time:.0f}s ({cache_load_time:.1f}s load + {(with_cache_time - cache_load_time):.1f}s access)")
    print(f"  Time saved: {time_saved:.0f}s ({efficiency_percentage:.1f}% improvement)")
    
    # Run simulations
    try:
        if not debug:
            run_with_executor(trace_tasks, combined_params, trace_specific_output_dir, param_types, 
                             enable_direction, max_workers, batch_size, vertical_cache_info, use_optimized,
                             simulator_type)
        else:
            # Debug mode - run sequentially
            global _vertical_cache_info, _progress_queue
            _vertical_cache_info = vertical_cache_info

            # Create progress queue and start progress monitor for debug mode too
            progress_queue = multiprocessing.Queue()
            progress_thread = threading.Thread(
                target=progress_monitor,
                args=(progress_queue, total_simulations, 5, _shutdown_event),
                daemon=True,
                name="DebugProgressMonitor"
            )
            progress_thread.start()
            _progress_queue = progress_queue
            print(f"Started debug progress monitor for {total_simulations} simulations")
            
            # In debug mode, we're using sequential processing, so BLAS threads should be higher
            actual_blas_threads = int(os.environ.get("OMP_NUM_THREADS", "2"))
            print(f"Debug mode BLAS threads: {actual_blas_threads} (sequential processing)")
            print(f"This should be much faster than parallel mode was (~150s vs ~2000s per task)")

            for i, (trace_snp_file, vertical_pairs_with_counts) in enumerate(trace_tasks):
                print(f"\n--- Debug Task {i+1}/{len(trace_tasks)} ---")
                collect_trace_simulation_data(
                    trace_snp_file, vertical_pairs_with_counts, combined_params, 
                    trace_specific_output_dir, param_types, enable_direction, batch_size, True, use_optimized,
                    simulator_type
                )
            
            # Print overall debug mode performance summary
            print(f"\n--- Debug Mode Overall Performance Summary ---")
            print_performance_summary()
            
            # Signal progress monitor to stop
            try:
                progress_queue.put(('stop', None), timeout=0.5)
            except:
                pass
            
            progress_thread.join(timeout=1.0)  # Reduced from 2 to 1 second
        
        print(f"Data collection completed. Results saved to: {trace_specific_output_dir}")
        
        # Final resource validation
        print("\n--- Final Resource Validation ---")
        validate_resource_usage()
        print("--------------------------------")
        
    except KeyboardInterrupt:
        print("\n[MAIN] Keyboard interrupt received in main function")
        _shutdown_event.set()
    except Exception as e:
        print(f"\n[MAIN] Unexpected error in main function: {e}")
        traceback.print_exc()
        _shutdown_event.set()
    finally:
        # Comprehensive cleanup
        global _cleanup_done, _executor, _progress_thread
        if not _cleanup_done:
            print("\n[CLEANUP] Starting final cleanup...")
            
            # Stop background monitoring
            stop_background_monitoring()
            
            # Clean up shared memory
            try:
                if vertical_cache:
                    print("[CLEANUP] Cleaning up vertical SNP cache...")
                    vertical_cache.cleanup()
                cleanup_shared_memory()
            except Exception as e:
                print(f"[CLEANUP] Error cleaning up cache: {e}")
                pass
            
            # Mark cleanup as done
            _cleanup_done = True
            
            if _shutdown_event.is_set():
                print("[CLEANUP] Final cleanup completed after interrupt.")
                import sys
                sys.exit(130)  # Standard exit code for SIGINT
            else:
                print("[CLEANUP] Final cleanup completed successfully.")

if __name__ == "__main__":
    main()