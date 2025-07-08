"""Main training data collector orchestrating eye width simulation data collection."""

# -----------------------------------------------------------------------------
# Limit BLAS / OpenMP thread usage BEFORE NumPy/SciPy are imported.
# This must be done at import time to ensure MKL / OpenBLAS obey the limits.
# Can be overridden by config or environment variables.
# -----------------------------------------------------------------------------
import os
import platform

# Allow config to override BLAS thread count
default_blas_threads = os.environ.get("BLAS_THREADS", "2")
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
import skrf as rf
import numpy as np
import multiprocessing
import concurrent.futures
import multiprocessing.shared_memory as shm
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from queue import Queue, Empty

from common.signal_utils import read_snp
from simulation.parameters.bound_param import PARAM_SETS_MAP
from simulation.engine.eye_width_simulator import snp_eyewidth_simulation
from simulation.io.config_utils import load_config, resolve_trace_pattern, resolve_vertical_dirs, build_argparser
from simulation.io.snp_utils import parse_snps, generate_vertical_snp_pairs
from simulation.parameters.param_utils import parse_param_types, modify_params_for_inductance

# Global profiling state
_profiling_data = threading.local()
# Global monitoring control
_monitor_proc = None
# Global progress queue for inter-process communication
_progress_queue = None

# Global shared memory registry for cleanup
_shared_memory_blocks = []



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
    """Background monitoring of system resources in a separate process."""
    print(f"[MONITOR] Starting system resource monitoring (interval: {interval}s)", flush=True)
    
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
            
            # Format CPU cores string
            if len(cpu_percent_per_core) > 8:
                cores_str = ", ".join([f"{p:.1f}%" for p in cpu_percent_per_core[:4]])
                cores_str += f" ... " + ", ".join([f"{p:.1f}%" for p in cpu_percent_per_core[-4:]])
            else:
                cores_str = ", ".join([f"{p:.1f}%" for p in cpu_percent_per_core])

            timestamp = datetime.now().strftime("%H:%M:%S")
            message = (f"[MONITOR {timestamp}] "
                       f"CPU: {cpu_overall:.1f}% | "
                       f"RAM: {memory_used_gb:.1f}/{memory_total_gb:.1f}GB ({memory_percent:.1f}%)"
                       f"{load_str} | Cores: [{cores_str}]")
            print(message, flush=True)
            
            time.sleep(interval - 1)
            
        except Exception as e:
            print(f"[MONITOR] Error: {e}", flush=True)
            time.sleep(interval)

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
    """Stop background system monitoring process."""
    global _monitor_proc
    if _monitor_proc is not None:
        print("[MONITOR] Stopping system resource monitoring...", flush=True)
        _monitor_proc.terminate()
        _monitor_proc.join(timeout=2)
        _monitor_proc = None
        print("[MONITOR] System resource monitoring stopped.", flush=True)

def get_optimal_workers(config_blas_threads=2):
    """Calculate optimal number of workers based on system resources"""
    # Get system info
    cpu_count = psutil.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    print(f"System: {cpu_count} CPUs, {memory_gb:.1f}GB RAM")
    
    # Calculate workers based on memory and CPU
    # Each worker uses ~1-2GB during processing
    memory_based_workers = max(1, int(memory_gb / 1.5))
    
    # For CPU-based calculation, consider BLAS threads
    # If using 2-4 BLAS threads per worker, reduce process count accordingly
    cpu_based_workers = max(1, cpu_count // max(1, config_blas_threads))
    
    # Conservative approach: use fewer workers to avoid oversubscription
    optimal_workers = min(memory_based_workers, cpu_based_workers, cpu_count // 2)
    print(f"Optimal workers determined: {optimal_workers} (memory-based: {memory_based_workers}, cpu-based: {cpu_based_workers}, BLAS threads: {config_blas_threads})")
    
    return optimal_workers

class BufferedPickleWriter:
    """Buffered writer for pickle files to reduce I/O overhead"""
    
    def __init__(self, pickle_file, batch_size=10):
        self.pickle_file = Path(pickle_file)
        self.batch_size = batch_size
        self.buffer = []
        self.data = self._load_existing_data()
    
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
        self.buffer.append(result)
        if len(self.buffer) >= self.batch_size:
            self.flush()
    
    def flush(self):
        """Write buffered results to disk"""
        if not self.buffer:
            return
        
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
    
    def close(self):
        """Flush remaining data and close"""
        self.flush()

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

def report_progress(completed_samples):
    """Report progress to main process via queue"""
    global _progress_queue
    if _progress_queue:
        try:
            _progress_queue.put(('progress', completed_samples), timeout=1.0)
        except:
            pass  # Queue full or other error, skip

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
    
    profile_print(f"Starting {trace_snp_path.name}: {total_simulations} simulations")
    
    total_completed = 0
    
    # Process all vertical pairs and their samples
    for vertical_snp_pair, sample_count in vertical_pairs_with_counts:
        # Load vertical SNPs from cache
        snp_tx_path, snp_rx_path = vertical_snp_pair
        tx_ntwk = get_snp_from_cache(snp_tx_path, _vertical_cache_info)
        rx_ntwk = get_snp_from_cache(snp_rx_path, _vertical_cache_info)
        
        for sample_idx in range(sample_count):
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
                
                if debug:
                    print(f"  Completed simulation {sample_idx+1}/{sample_count}: EW={line_ew}")
                    
            except Exception as e:
                print(f"Error in simulation for {trace_snp_path.name}: {e}")
                if debug:
                    import traceback
                    traceback.print_exc()
                continue
    
    # Ensure all data is written
    writer.close()
    
    # Final progress report is not needed since we report after each simulation
    
    profile_print(f"Completed {trace_snp_path.name}: {total_completed} simulations")
    
    if debug:
        print(f"Completed {total_completed} simulations for {trace_snp_path.name}")
    
    return total_completed

def cleanup_shared_memory():
    """Clean up all shared memory blocks"""
    for block in _shared_memory_blocks:
        try:
            block.close()
            block.unlink()
        except:
            pass
    _shared_memory_blocks.clear()

def progress_monitor(progress_queue, total_expected, interval=5):
    """Monitor progress from worker processes"""
    completed = 0
    last_report = time.time()
    start_time = time.time()
    
    while completed < total_expected:
        try:
            msg_type, value = progress_queue.get(timeout=interval)
            if msg_type == 'progress':
                completed += value
                
                now = time.time()
                if now - last_report >= interval:
                    elapsed = now - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (total_expected - completed) / rate if rate > 0 else 0
                    print(f"Progress: {completed}/{total_expected} ({100*completed/total_expected:.1f}%) "
                          f"Rate: {rate:.1f}/sec ETA: {eta:.0f}s", flush=True)
                    last_report = now
                    
        except Empty:
            # Timeout - print current status
            now = time.time()
            elapsed = now - start_time
            rate = completed / elapsed if elapsed > 0 else 0
            print(f"Progress: {completed}/{total_expected} ({100*completed/total_expected:.1f}%) "
                  f"Rate: {rate:.1f}/sec", flush=True)

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
    Run simulations using ProcessPoolExecutor with optimized task distribution
    
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
    
    executor_start_time = time.time()
    
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=num_workers, 
        initializer=init_worker_process,
        initargs=(vertical_cache_info or {}, None)
    ) as executor:
        futures = [
            executor.submit(
                collect_trace_simulation_data,
                trace_snp_file, vertical_pairs_with_counts, combined_params, 
                trace_specific_output_dir, param_types, enable_direction, batch_size, False
            )
            for trace_snp_file, vertical_pairs_with_counts in trace_tasks
        ]
        
        print(f"Submitted {len(futures)} trace files to ProcessPoolExecutor")
        
        # Process results
        failed_tasks = []
        completed_tasks = 0
        total_simulations_completed = 0

        for future in concurrent.futures.as_completed(futures):
            exc = future.exception()
            if exc is not None:
                print("\n--- Task FAILED ------------------------------------")
                print("Exception:")
                traceback.print_exception(type(exc), exc, exc.__traceback__)
                failed_tasks.append(exc)
            else:
                completed_tasks += 1
                sim_count = future.result()
                total_simulations_completed += sim_count
                
                elapsed = time.time() - executor_start_time
                sec_per_sim = elapsed / total_simulations_completed if total_simulations_completed > 0 else 0
                eta = (total_expected - total_simulations_completed) * sec_per_sim if sec_per_sim > 0 else 0
                print(f"Progress: {completed_tasks}/{len(futures)} tasks, "
                      f"{total_simulations_completed}/{total_expected} simulations "
                      f"({100*total_simulations_completed/total_expected:.1f}%) "
                      f"Rate: {sec_per_sim:.2f}s/sim ETA: {eta/60:.1f}min", flush=True)

        total_time = time.time() - executor_start_time
        print(f"ProcessPoolExecutor completed in {total_time:.2f}s")
        print(f"Completed {completed_tasks}/{len(futures)} trace files")

        if failed_tasks:
            print(f"\nWarning: {len(failed_tasks)} trace files failed.")

def main():
    """Main function for parallel data collection"""
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
    
    # Get batch size from config
    batch_size = runner_config.get('batch_size', 10)
    
    # Determine number of workers: Command-line -> Config file -> Optimal calculation
    config_workers = runner_config.get('max_workers')
    if args.max_workers:
        max_workers = args.max_workers
        print(f"Using max_workers from command line: {max_workers}")
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
    
    # Build task structure - each task processes all samples for one trace file
    trace_tasks = []  # [(trace_snp_file, [(vertical_pair, samples_needed)])]
    
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
            trace_tasks.append((trace_snp, [(vertical_pair, samples_needed)]))
    
    total_simulations = sum(
        sum(count for _, count in vertical_pairs_with_counts)
        for _, vertical_pairs_with_counts in trace_tasks
    )
    
    print(f"Created {len(trace_tasks)} tasks for {total_simulations} total simulations")
    print(f"Task distribution: {len(trace_tasks)} tasks, ~{total_simulations/len(trace_tasks) if trace_tasks else 0:.1f} simulations/task")
    
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
            global _vertical_cache_info
            _vertical_cache_info = vertical_cache_info

            for i, (trace_snp_file, vertical_pairs_with_counts) in enumerate(trace_tasks):
                print(f"\n--- Debug Task {i+1}/{len(trace_tasks)} ---")
                collect_trace_simulation_data(
                    trace_snp_file, vertical_pairs_with_counts, combined_params, 
                    trace_specific_output_dir, param_types, enable_direction, batch_size, True
                )
        
        print(f"Data collection completed. Results saved to: {trace_specific_output_dir}")
        
    finally:
        # Stop background monitoring
        stop_background_monitoring()
        
        # Clean up shared memory
        if vertical_cache:
            vertical_cache.cleanup()
        cleanup_shared_memory()

if __name__ == "__main__":
    main()