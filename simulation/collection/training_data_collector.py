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
    else:
        return "2"  # Safe default for unknown platforms

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

from common.signal_utils import read_snp
from simulation.parameters.bound_param import PARAM_SETS_MAP
from simulation.engine.eye_width_simulator import snp_eyewidth_simulation
from simulation.io.config_utils import load_config, resolve_trace_pattern, resolve_vertical_dirs, build_argparser
from simulation.io.snp_utils import parse_snps, generate_vertical_snp_pairs
from simulation.io.progress_utils import progress_monitor, report_progress
from simulation.parameters.param_utils import parse_param_types, modify_params_for_inductance

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

def signal_handler(signum, frame):
    """Handle SIGINT (Ctrl+C) for graceful shutdown with aggressive timeouts"""
    global _shutdown_event, _executor, _progress_thread, _shutdown_in_progress
    
    # Handle double Ctrl+C - force immediate exit
    if _shutdown_in_progress:
        print("\n[FORCE EXIT] Second interrupt signal received. Forcing immediate exit...", flush=True)
        import os
        os._exit(130)  # Immediate exit without any cleanup
    
    print("\n[SHUTDOWN] Received interrupt signal (Ctrl+C). Initiating graceful shutdown...", flush=True)
    print("[SHUTDOWN] Press Ctrl+C again to force immediate exit if shutdown hangs.", flush=True)
    _shutdown_in_progress = True
    _shutdown_event.set()
    
    # Aggressive shutdown with short timeouts
    shutdown_start = time.time()
    
    # Stop progress monitoring first with short timeout
    if _progress_thread and _progress_thread.is_alive():
        print("[SHUTDOWN] Stopping progress monitor...", flush=True)
        try:
            if _progress_queue:
                _progress_queue.put(('stop', None), timeout=0.5)
        except:
            pass
        _progress_thread.join(timeout=1.0)  # Reduced from 3 to 1 second
        if _progress_thread.is_alive():
            print("[SHUTDOWN] Warning: Progress monitor thread still alive, continuing...", flush=True)
        else:
            print("[SHUTDOWN] Progress monitor stopped.", flush=True)
    
    # Shutdown executor with immediate cancellation
    if _executor:
        print("[SHUTDOWN] Shutting down ProcessPoolExecutor...", flush=True)
        _executor.shutdown(wait=False, cancel_futures=True)
        print("[SHUTDOWN] ProcessPoolExecutor shutdown initiated.", flush=True)
    
    # Stop background monitoring with aggressive timeouts
    stop_background_monitoring()
    
    # Clean up resources quickly
    cleanup_shared_memory()
    
    elapsed = time.time() - shutdown_start
    print(f"[SHUTDOWN] Cleanup completed in {elapsed:.1f}s. Exiting...", flush=True)
    
    # Force exit immediately - don't wait for anything else
    import sys
    import os
    
    # If shutdown took too long, force exit more aggressively
    if elapsed > 2.0:
        print("[SHUTDOWN] Shutdown took too long, forcing immediate exit...", flush=True)
        os._exit(130)  # Immediate exit without cleanup
    else:
        sys.exit(130)  # Standard exit code for SIGINT

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
        
    elif system == "Linux":  # Linux - Aggressive for production
        print("Using Linux resource allocation (aggressive)")
        # Aggressive settings to maximize server utilization
        # Target: 90-95% CPU usage, use most available RAM
        memory_per_worker = 0.5  # Reduced memory estimate for better scaling
        cpu_utilization_target = 0.95  # Use 95% of cores
        
        # Reserve some memory for system (5% or 2GB, whichever is larger)
        reserved_memory = max(memory_gb * 0.05, 2.0)
        available_memory = memory_gb - reserved_memory
        
        # Calculate workers based on memory and CPU
        memory_based_workers = max(1, int(available_memory / memory_per_worker))
        
        # For CPU calculation, use a more aggressive approach
        # Each worker can efficiently use 1-2 cores with BLAS threads
        cores_per_worker = max(1, config_blas_threads // 2) if config_blas_threads > 2 else 1
        cpu_based_workers = max(1, int(cpu_count * cpu_utilization_target / cores_per_worker))
        
        # Use minimum but cap at reasonable maximum
        max_reasonable_workers = min(cpu_count, 64)  # Cap at 64 workers max
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
        print(f"  CPU-based workers: {cpu_based_workers} (target {cpu_utilization_target*100:.0f}% CPU, {cores_per_worker} cores per worker, {config_blas_threads} BLAS threads)")
    else:
        print(f"  CPU-based workers: {cpu_based_workers} (target {cpu_utilization_target*100:.0f}% CPU, {config_blas_threads} BLAS threads)")
    print(f"  Selected workers: {optimal_workers} (constraining factor: {'memory' if optimal_workers == memory_based_workers else 'CPU' if optimal_workers == cpu_based_workers else 'max_limit'})")
    
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
            'snp_txs': [],
            'snp_rxs': [],
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
        """Write buffered results to disk"""
        if not self.buffer or self._closed:
            return
        
        try:
            # Add all buffered results to data
            for result in self.buffer:
                self.data['configs'].append(result['config_values'])
                self.data['line_ews'].append(result['line_ews'])
                self.data['snp_txs'].append(result['snp_tx'])
                self.data['snp_rxs'].append(result['snp_rx'])
                self.data['directions'].append(result['directions'])
            
            # Update meta if needed
            if self.buffer and not self.data['meta'].get('config_keys'):
                first_result = self.buffer[0]
                self.data['meta']['config_keys'] = first_result['config_keys']
                self.data['meta']['snp_horiz'] = first_result.get('snp_horiz', '')
                self.data['meta']['n_ports'] = first_result.get('n_ports', 0)
                self.data['meta']['param_types'] = first_result.get('param_types', [])
            
            # Write to disk
            self.pickle_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.pickle_file, 'wb') as f:
                pickle.dump(self.data, f)
            
            self.buffer.clear()
            self._last_flush_time = time.time()
            
        except Exception as e:
            # If normal write fails during shutdown, try emergency save
            if _shutdown_event.is_set():
                try:
                    emergency_file = self.pickle_file.with_suffix('.emergency.pkl')
                    with open(emergency_file, 'wb') as f:
                        pickle.dump({'buffer': self.buffer, 'existing_data': self.data}, f)
                    print(f"[EMERGENCY] Saved buffered data to {emergency_file}", flush=True)
                except:
                    print(f"[ERROR] Failed to save data during shutdown: {e}", flush=True)
            else:
                raise
    
    def close(self):
        """Flush remaining data and close"""
        if not self._closed:
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
        if str(snp_path) in self.cache:
            return
            
        ntwk = read_snp(snp_path)
        s_data = ntwk.s
        f_data = ntwk.f
        
        # Create shared memory for s-parameters
        s_shm = shm.SharedMemory(create=True, size=s_data.nbytes)
        s_shm_array = np.ndarray(s_data.shape, dtype=s_data.dtype, buffer=s_shm.buf)
        s_shm_array[:] = s_data[:]
        
        # Create shared memory for frequencies
        f_shm = shm.SharedMemory(create=True, size=f_data.nbytes)
        f_shm_array = np.ndarray(f_data.shape, dtype=f_data.dtype, buffer=f_shm.buf)
        f_shm_array[:] = f_data[:]
        
        self.cache[str(snp_path)] = {
            's_name': s_shm.name, 's_shape': s_data.shape, 's_dtype': s_data.dtype,
            'f_name': f_shm.name, 'f_shape': f_data.shape, 'f_dtype': f_data.dtype,
            'nports': ntwk.nports, 'z0': ntwk.z0.tolist() if isinstance(ntwk.z0, np.ndarray) else ntwk.z0
        }
        
        self.memory_blocks.extend([s_shm, f_shm])
        _shared_memory_blocks.extend([s_shm, f_shm])
    
    def get_cache_info(self):
        """Get cache information for passing to workers."""
        return self.cache.copy()
    
    def cleanup(self):
        """Clean up shared memory blocks."""
        for block in self.memory_blocks:
            try:
                block.close()
                block.unlink()
            except FileNotFoundError:
                pass # Already unlinked
            except Exception:
                pass
        self.memory_blocks.clear()

def init_worker_process(vertical_cache_info, progress_queue):
    """Initialize worker process with shared memory access and progress reporting."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Store cache info and progress queue globally in worker
    global _vertical_cache_info, _progress_queue
    _vertical_cache_info = vertical_cache_info
    _progress_queue = progress_queue
    
    # Set CPU affinity for better CPU utilization (Linux only)
    try:
        if platform.system() == "Linux" and hasattr(os, 'sched_setaffinity'):
            worker_id = os.getpid()
            cpu_count = psutil.cpu_count()
            
            # Distribute workers across CPU cores
            # Use process ID to determine which cores this worker should use
            cores_per_worker = max(1, cpu_count // psutil.cpu_count(logical=False))  # Physical cores
            
            # Calculate CPU affinity mask
            available_cpus = list(range(cpu_count))
            worker_hash = hash(worker_id) % cpu_count
            
            # Assign cores in round-robin fashion
            worker_cpus = []
            for i in range(cores_per_worker):
                cpu_id = (worker_hash + i) % cpu_count
                worker_cpus.append(cpu_id)
            
            # Set affinity if we have multiple CPUs
            if cpu_count > 1:
                os.sched_setaffinity(0, worker_cpus)
                profile_print(f"Worker {worker_id} set to CPUs: {worker_cpus}")
                
    except Exception as e:
        # Don't fail if CPU affinity setting fails
        profile_print(f"Could not set CPU affinity: {e}")

def get_snp_from_cache(snp_path, cache_info):
    """Load SNP data directly from disk for optimal performance."""
    # Always load directly from disk - faster than shared memory overhead
    return read_snp(snp_path)

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



def collect_trace_simulation_data(trace_snp_file, vertical_pairs_with_counts, combined_params, 
                                pickle_dir, param_type_names, enable_direction=True, 
                                batch_size=10, debug=False):
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
    """
    if not vertical_pairs_with_counts:
        return
    
    # Load horizontal SNP once per trace file
    trace_ntwk = read_snp(trace_snp_file)
    
    n_ports = trace_ntwk.nports
    n_lines = n_ports // 2
    if n_lines == 0:
        raise ValueError(f"Invalid n_ports={n_ports}, n_lines would be 0")
    
    trace_snp_path = Path(trace_snp_file)
    pickle_file = Path(pickle_dir) / f"{trace_snp_path.stem}.pkl"
    
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
        snp_tx_path, snp_rx_path = vertical_snp_pair
        tx_ntwk = get_snp_from_cache(snp_tx_path, _vertical_cache_info)
        rx_ntwk = get_snp_from_cache(snp_rx_path, _vertical_cache_info)
        
        for sample_idx in range(sample_count):
            # Check for shutdown between samples
            if _shutdown_event.is_set():
                profile_print(f"Shutdown detected, stopping {trace_snp_path.name} early")
                break
            # Sample parameters
            combined_config = combined_params.sample()
            
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
                
                # Run simulation
                line_ew = snp_eyewidth_simulation(
                    config=combined_config,
                    snp_files=(trace_ntwk, tx_ntwk, rx_ntwk),
                    directions=sim_directions
                )
                
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
                    'snp_tx': snp_tx_path.as_posix(),
                    'snp_rx': snp_rx_path.as_posix(),
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
                
                if debug:
                    print(f"  Completed simulation {sample_idx+1}/{sample_count}: EW={line_ew}")
                    
            except Exception as e:
                print(f"Error in simulation for {trace_snp_path.name}: {e}")
                if debug:
                    import traceback
                    traceback.print_exc()
                continue
        
        # Break outer loop if shutdown detected during inner loop
        if _shutdown_event.is_set():
            break
    
    # Ensure all data is written
    writer.close()
    
    # Final progress report is not needed since we report after each simulation
    
    # profile_print(f"Completed {trace_snp_path.name}: {total_completed} simulations")
    
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
                     enable_direction, num_workers, batch_size, vertical_cache_info=None):
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
        initargs=(vertical_cache_info or {}, progress_queue)
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
                future = executor.submit(
                    collect_trace_simulation_data,
                    trace_snp_file, vertical_pairs, combined_params, 
                    trace_specific_output_dir, param_types, enable_direction, batch_size, False
                )
                active_futures[future] = (trace_snp_file, vertical_pairs)
                return True
            except StopIteration:
                return False

        # Submit the initial batch of tasks to fill the queue
        for _ in range(min(len(trace_tasks), max_active_futures)):
            submit_next_task()

        print(f"Submitted initial {len(active_futures)}/{len(trace_tasks)} tasks to ProcessPoolExecutor (max queue: {max_active_futures})")

        failed_tasks = []
        completed_tasks = 0
        total_simulations_completed = 0
        
        while active_futures:
            # Wait for at least one future to complete
            done, _ = concurrent.futures.wait(
                active_futures.keys(), return_when=concurrent.futures.FIRST_COMPLETED
            )

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
        try:
            print("[CLEANUP] Shutting down executor...")
            executor.shutdown(wait=False, cancel_futures=True)
            
            # Stop progress monitor with aggressive timeout
            try:
                progress_queue.put(('stop', None), timeout=0.5)
            except:
                pass
            
            if progress_thread.is_alive():
                progress_thread.join(timeout=1.0)
                if progress_thread.is_alive():
                    print("[CLEANUP] Warning: Progress thread still alive, continuing...")
                
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
    
    # Apply BLAS thread configuration from config
    runner_config = config.get('runner', {})
    blas_threads = runner_config.get('blas_threads', 2)
    if blas_threads != int(os.environ.get("OMP_NUM_THREADS", "2")):
        print(f"Note: BLAS threads set to {os.environ.get('OMP_NUM_THREADS')} in environment, "
              f"config specifies {blas_threads}. Environment takes precedence.")
    
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
    
    debug = args.debug if args.debug else config.get('debug', False)
    
    # Get batch size from config with platform-aware defaults
    system = platform.system()
    if system == "Linux":
        # Use larger batch sizes on servers for better I/O performance
        default_batch_size = min(50, max(20, max_workers * 2))
    else:
        # Conservative batch size for other platforms
        default_batch_size = 10
    
    batch_size = runner_config.get('batch_size', default_batch_size)
    
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
    
    print(f"Using configuration:")
    print(f"  Trace pattern: {trace_pattern_key} -> {trace_pattern}")
    print(f"  Vertical dirs: {vertical_dirs}")
    print(f"  Output dir: {output_dir}")
    print(f"  Parameter types: {param_types}")
    print(f"  Max samples: {max_samples}")
    print(f"  Enable direction: {enable_direction}")
    print(f"  Enable inductance: {enable_inductance}")
    print(f"  Debug mode: {debug}")
    print(f"  Max workers: {max_workers}")
    print(f"  Batch size: {batch_size}")
    print(f"  BLAS threads: {blas_threads}")
    print(f"  Platform: {platform.system()}")
    
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
    
    # Build task structure with optimal granularity for better load balancing
    trace_tasks = []  # [(trace_snp_file, [(vertical_pair, samples_needed)])]
    
    # Calculate optimal chunk size based on number of workers and total work
    min_chunk_size = max(1, max_samples // (max_workers * 4))  # 4 chunks per worker minimum
    max_chunk_size = max(min_chunk_size, max_samples // 2)    # Don't make chunks too large
    
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
            
            # Split large tasks into chunks for better load balancing
            if samples_needed > max_chunk_size:
                # Create multiple chunks for this trace file
                remaining_samples = samples_needed
                while remaining_samples > 0:
                    chunk_size = min(remaining_samples, max_chunk_size)
                    trace_tasks.append((trace_snp, [(vertical_pair, chunk_size)]))
                    remaining_samples -= chunk_size
            else:
                # Small task, keep as single chunk
                trace_tasks.append((trace_snp, [(vertical_pair, samples_needed)]))
    
    total_simulations = sum(
        sum(count for _, count in vertical_pairs_with_counts)
        for _, vertical_pairs_with_counts in trace_tasks
    )
    
    print(f"Created {len(trace_tasks)} tasks for {total_simulations} total simulations")
    print(f"Task distribution: {len(trace_tasks)} tasks, ~{total_simulations/len(trace_tasks) if trace_tasks else 0:.1f} simulations/task")
    print(f"Chunking strategy: min_chunk={min_chunk_size}, max_chunk={max_chunk_size} (based on {max_workers} workers)")
    
    if len(trace_tasks) == 0:
        print("All files already have sufficient samples")
        stop_background_monitoring()
        return
    
    # Skip shared memory cache for better performance - let each worker load files directly
    vertical_cache = None
    vertical_cache_info = {}
    
    print("Using direct file loading for better performance...")
    
    # Run simulations
    try:
        if not debug:
            run_with_executor(trace_tasks, combined_params, trace_specific_output_dir, param_types, 
                             enable_direction, max_workers, batch_size, vertical_cache_info)
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

            for i, (trace_snp_file, vertical_pairs_with_counts) in enumerate(trace_tasks):
                print(f"\n--- Debug Task {i+1}/{len(trace_tasks)} ---")
                collect_trace_simulation_data(
                    trace_snp_file, vertical_pairs_with_counts, combined_params, 
                    trace_specific_output_dir, param_types, enable_direction, batch_size, True
                )
            
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
        print("\n[CLEANUP] Starting final cleanup...")
        
        # Stop background monitoring
        stop_background_monitoring()
        
        # Clean up shared memory
        if vertical_cache:
            vertical_cache.cleanup()
        cleanup_shared_memory()
        
        # Ensure executor and progress thread are cleaned up
        global _executor, _progress_thread
        if _executor:
            try:
                _executor.shutdown(wait=False, cancel_futures=True)
            except:
                pass
            _executor = None
            
        if _progress_thread and _progress_thread.is_alive():
            try:
                _progress_thread.join(timeout=1.0)  # Reduced from 2 to 1 second
                if _progress_thread.is_alive():
                    print("[CLEANUP] Warning: Progress thread still alive, abandoning...")
            except:
                pass
            _progress_thread = None
        
        if _shutdown_event.is_set():
            print("[CLEANUP] Final cleanup completed after interrupt.")
            import sys
            sys.exit(130)  # Standard exit code for SIGINT
        else:
            print("[CLEANUP] Final cleanup completed successfully.")

if __name__ == "__main__":
    main()