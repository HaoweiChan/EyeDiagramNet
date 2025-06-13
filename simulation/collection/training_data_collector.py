"""Main training data collector orchestrating eye width simulation data collection."""

# -----------------------------------------------------------------------------
# Limit BLAS / OpenMP thread usage BEFORE NumPy/SciPy are imported.
# This must be done at import time to ensure MKL / OpenBLAS obey the limits.
# -----------------------------------------------------------------------------
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

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

from simulation.parameters.bound_param import PARAM_SETS_MAP
from simulation.engine.eye_width_simulator import snp_eyewidth_simulation
from simulation.io.config_utils import load_config, resolve_trace_pattern, resolve_vertical_dirs, build_argparser
from simulation.io.snp_utils import parse_snps, generate_vertical_snp_pairs
from simulation.parameters.param_utils import parse_param_types, modify_params_for_inductance
from common.signal_utils import read_snp

# Global profiling state
_profiling_data = threading.local()
# Global monitoring control
_monitor_proc = None

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

# Global shared memory registry for cleanup
_shared_memory_blocks = []

def get_optimal_workers():
    """Calculate optimal number of workers based on system resources"""
    # Get system info
    cpu_count = psutil.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    print(f"System: {cpu_count} CPUs, {memory_gb:.1f}GB RAM")
    
    # For large S96P files (~66MB each), be conservative
    # Each worker can use 500MB-1GB during processing
    memory_based_workers = max(1, int(memory_gb / 2))  # 2GB per worker
    cpu_based_workers = min(8, cpu_count)  # Cap at 8 for I/O limits
    
    optimal_workers = min(memory_based_workers, cpu_based_workers)
    print(f"Optimal workers determined: {optimal_workers} (memory-based: {memory_based_workers}, cpu-based: {cpu_based_workers})")
    
    return optimal_workers

class SNPCache:
    """Shared memory cache for SNP files to avoid redundant loading across processes."""
    
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

def init_worker_process(vertical_cache_info):
    """Initialize worker process with shared memory access and signal handling."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Store cache info globally in worker
    global _vertical_cache_info
    _vertical_cache_info = vertical_cache_info

def get_snp_from_cache(snp_path, cache_info):
    """Get SNP data from a shared memory cache, or read from disk as a fallback."""
    if not cache_info or str(snp_path) not in cache_info:
        return read_snp(snp_path) # Fallback for thread mode or uncached files

    shm_info = cache_info[str(snp_path)]

    s_shm = shm.SharedMemory(name=shm_info['s_name'])
    s_array = np.ndarray(shm_info['s_shape'],
                         dtype=shm_info['s_dtype'],
                         buffer=s_shm.buf)

    f_shm = shm.SharedMemory(name=shm_info['f_name'])
    f_array = np.ndarray(shm_info['f_shape'],
                         dtype=shm_info['f_dtype'],
                         buffer=f_shm.buf)

    # Construct Network object without duplicating the underlying data.
    # The S‑parameter and frequency arrays now *share* the same shared‑memory
    # buffers across all worker processes, eliminating the per‑process copy
    # that was previously blowing up RAM usage.
    ntwk = rf.Network()
    ntwk.s = s_array
    ntwk.f = f_array
    ntwk.z0 = np.asarray(shm_info['z0'])

    # Keep shared‑memory segments alive for the lifetime of `ntwk`
    # so they are not freed prematurely.
    ntwk._s_shm = s_shm
    ntwk._f_shm = f_shm

    return ntwk

def collect_snp_batch_simulation_data(task_batch, combined_params, pickle_dir, 
                                    param_type_names, enable_direction=True, debug=False):
    """
    Process a batch of simulations for the same horizontal SNP file.
    This reduces I/O overhead by loading the horizontal file once and writing results once.
    
    Args:
        task_batch: List of (trace_snp_file, vertical_snp_pair, sample_count) tuples for same trace file
        combined_params: ParameterSet containing all required parameters  
        pickle_dir: Directory to save pickle files
        param_type_names: List of parameter type names
        enable_direction: Whether to use random directions (True) or all ones (False)
        debug: Debug mode flag
    """
    if not task_batch:
        return
        
    # All tasks in batch share the same horizontal trace SNP
    trace_snp_file = task_batch[0][0]
    
    # Load horizontal SNP directly from disk (each process handles one horizontal SNP)
    trace_ntwk = read_snp(trace_snp_file)

    n_ports = trace_ntwk.nports
    n_lines = n_ports // 2
    if n_lines == 0:
        raise ValueError(f"Invalid n_ports={n_ports}, n_lines would be 0")
    
    trace_snp_path = Path(trace_snp_file) # Still need path for pickle filename
    pickle_file = Path(pickle_dir) / f"{trace_snp_path.stem}.pkl"
    
    if debug:
        print(f"Processing batch of {len(task_batch)} tasks for {trace_snp_path.name}")
        print(f"Detected {n_ports} ports ({n_lines} lines) from horizontal SNP loaded from disk")
    
    # Load existing pickle data once
    if pickle_file.exists():
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
    else:
        data = {
            'configs': [],
            'line_ews': [], 
            'snp_txs': [],
            'snp_rxs': [],
            'directions': [],
            'meta': {}
        }
    
    # Process all tasks in batch
    batch_results = []
    
    for _, vertical_snp_pair, sample_count in task_batch:
        # Load vertical SNPs from cache
        snp_tx_path, snp_rx_path = vertical_snp_pair
        tx_ntwk = get_snp_from_cache(snp_tx_path, _vertical_cache_info)
        rx_ntwk = get_snp_from_cache(snp_rx_path, _vertical_cache_info)
        
        for _ in range(sample_count):
            # Sample parameters
            combined_config = combined_params.sample()
            
            try:
                # Set directions
                if enable_direction:
                    sim_directions = np.random.randint(0, 2, size=n_lines)
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
                
                # Store result
                config_values, config_keys = combined_config.to_list(return_keys=True)
                batch_results.append({
                    'config_values': config_values,
                    'config_keys': config_keys,
                    'line_ews': line_ew.tolist(),
                    'snp_tx': snp_tx_path.as_posix(),
                    'snp_rx': snp_rx_path.as_posix(),
                    'directions': sim_directions.tolist()
                })
                
                if debug:
                    print(f"  Completed simulation: EW={line_ew}, Dir={sim_directions}")
                    
            except Exception as e:
                print(f"Error in simulation for {trace_snp_path.name}: {e}")
                if debug:
                    import traceback
                    traceback.print_exc()
                continue
    
    # Append all batch results to data
    for result in batch_results:
        data['configs'].append(result['config_values'])
        data['line_ews'].append(result['line_ews'])
        data['snp_txs'].append(result['snp_tx'])
        data['snp_rxs'].append(result['snp_rx'])
        data['directions'].append(result['directions'])
    
    # Update meta once per batch
    if batch_results and not data['meta'].get('config_keys'):
        data['meta']['snp_horiz'] = str(trace_snp_path)
        data['meta']['config_keys'] = batch_results[0]['config_keys']
        data['meta']['n_ports'] = n_ports
        data['meta']['param_types'] = param_type_names
    
    # Write updated data once per batch
    pickle_file.parent.mkdir(parents=True, exist_ok=True)
    with open(pickle_file, 'wb') as f:
        pickle.dump(data, f)
    
    if debug:
        print(f"Batch completed: {len(batch_results)} simulations saved to {pickle_file}")

def collect_snp_simulation_data(trace_snp_file, vertical_snp_pair, params_set, 
                               pickle_dir, param_type_names, enable_direction=True, directions=None, debug=False):
    """
    Collect eye width simulation data for a single trace SNP with vertical SNP pair.
    This is kept for backward compatibility and single-task processing.
    """
    # Use batch function with single task
    task_batch = [(trace_snp_file, vertical_snp_pair, 1)]
    collect_snp_batch_simulation_data(
        task_batch, params_set, pickle_dir, param_type_names, enable_direction, debug
    )

def cleanup_shared_memory():
    """Clean up all shared memory blocks"""
    for block in _shared_memory_blocks:
        try:
            block.close()
            block.unlink()
        except:
            pass
    _shared_memory_blocks.clear()

def run_with_executor(batch_list, combined_params, trace_specific_output_dir, param_types, 
                     enable_direction, num_workers, vertical_cache_info=None):
    """
    Run simulations using ProcessPoolExecutor
    
    Args:
        batch_list: List of task batches
        combined_params: Parameter set
        trace_specific_output_dir: Output directory
        param_types: Parameter types
        enable_direction: Direction flag
        num_workers: Number of workers
        vertical_cache_info: Shared memory cache info for vertical SNPs
    """
    
    print(f"Using ProcessPoolExecutor with {num_workers} processes...")
    multiprocessing.set_start_method("forkserver", force=True)
    
    executor_start_time = time.time()
    
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=num_workers, 
        initializer=init_worker_process,
        initargs=(vertical_cache_info or {},)
    ) as executor:
        futures = [
            executor.submit(
                collect_snp_batch_simulation_data,
                batch_tasks, combined_params, trace_specific_output_dir, 
                param_types, enable_direction, False
            )
            for batch_tasks in batch_list
        ]
        
        print(f"Submitted {len(futures)} batches to ProcessPoolExecutor")
        
        # =================================================================
        # NEW: Print full tracebacks for any failed batch immediately
        failed_batches = []
        completed_batches = 0

        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Processing batches (processes)"
        ):
            exc = future.exception()
            if exc is not None:
                print("\n--- Batch FAILED ------------------------------------")
                print("Exception:")
                traceback.print_exception(type(exc), exc, exc.__traceback__)
                failed_batches.append(exc)
            else:
                completed_batches += 1
                if completed_batches % 10 == 0 or completed_batches <= 5:
                    elapsed = time.time() - executor_start_time
                    rate = completed_batches / elapsed
                    print(f"Completed {completed_batches}/{len(futures)} batches, rate: {rate:.2f} batches/sec")

        total_time = time.time() - executor_start_time
        print(f"ProcessPoolExecutor completed in {total_time:.2f}s, avg rate: {len(futures)/total_time:.2f} batches/sec")

        if failed_batches:
            print(f"\nWarning: {len(failed_batches)} batches failed in total.")

def main():
    """Main function for parallel data collection"""
    # Start background system monitoring immediately to show baseline system state
    start_background_monitoring(interval=15)
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
    
    # Determine number of workers: Command-line -> Config file -> Optimal calculation
    config_workers = config.get('runner', {}).get('max_workers')
    if args.max_workers:
        max_workers = args.max_workers
        print(f"Using max_workers from command line: {max_workers}")
    elif config_workers:
        max_workers = config_workers
        print(f"Using max_workers from config file: {max_workers}")
    else:
        print("max_workers not specified, calculating optimal number...")
        max_workers = get_optimal_workers()
    
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
    
    # Build task batches grouped by trace SNP file
    task_batches = defaultdict(list)  # {trace_snp_file: [(trace_snp, vertical_pair, samples_needed)]}
    
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
            task_batches[trace_snp].append((trace_snp, vertical_pair, samples_needed))
    
    # Convert to list of batches
    batch_list = [batch_tasks for batch_tasks in task_batches.values() if batch_tasks]
    total_tasks = sum(sum(samples for _, _, samples in batch) for batch in batch_list)
    
    print(f"Created {len(batch_list)} batches for {total_tasks} total simulations")
    
    if len(batch_list) == 0:
        print("All files already have sufficient samples")
        stop_background_monitoring()
        return
    
    # Initialize shared memory cache for vertical SNPs only
    # Optimization: Horizontal SNPs don't need shared memory since each batch processes
    # only one horizontal SNP file. Only vertical SNPs are truly shared across processes.
    vertical_cache = None
    vertical_cache_info = {}
    
    if not debug:
        print("Setting up shared memory cache for vertical SNP files...")
        print("Note: Horizontal SNPs are loaded per-process to save memory")
        try:
            # Cache all unique vertical SNPs that will be used across processes.
            # This includes the auto-generated "thru" SNP.
            vertical_cache = SNPCache()
            unique_vertical_snps = set()
            for batch_tasks in batch_list:
                for _, vertical_pair, _ in batch_tasks:
                    unique_vertical_snps.update(vertical_pair)
            
            if unique_vertical_snps:
                for snp_path in tqdm(unique_vertical_snps, desc="Caching vertical SNPs"):
                    vertical_cache.add_snp(snp_path)
                
                vertical_cache_info = vertical_cache.get_cache_info()
                print(f"Cached {len(vertical_cache_info)} vertical SNP files in shared memory.")
            else:
                print("No vertical SNPs found to cache.")
            
        except Exception as e:
            print(f"Warning: Could not set up shared memory cache: {e}")
            if vertical_cache: vertical_cache.cleanup()
            vertical_cache = None
            vertical_cache_info = {}
    
    # Run simulations
    try:
        if not debug:
            run_with_executor(batch_list, combined_params, trace_specific_output_dir, param_types, 
                             enable_direction, max_workers, vertical_cache_info)
        else:
            # Debug mode - run sequentially
            # In debug mode, we are in the main process, so worker globals are not set.
            # We need to manually use the cache info.
            global _vertical_cache_info
            _vertical_cache_info = vertical_cache_info

            for i, batch_tasks in enumerate(tqdm(batch_list, desc="Debug processing batches")):
                print(f"\n--- Batch {i+1}/{len(batch_list)} ---")
                collect_snp_batch_simulation_data(
                    batch_tasks, combined_params, trace_specific_output_dir,
                    param_types, enable_direction, True
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