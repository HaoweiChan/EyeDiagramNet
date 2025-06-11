"""Main training data collector orchestrating eye width simulation data collection."""

import os
import time
import yaml
import pickle
import psutil
import threading
import numpy as np
import multiprocessing
import concurrent.futures
import threadpoolctl
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
_monitoring_stop_event = threading.Event()
_monitoring_thread = None

def get_worker_id():
    """Get unique worker identifier for profiling"""
    if hasattr(_profiling_data, 'worker_id'):
        return _profiling_data.worker_id
    
    # Create unique worker ID 
    import threading
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

class VerticalSNPCache:
    """Shared memory cache for vertical SNP files to avoid redundant loading across processes"""
    
    def __init__(self):
        self.cache = {}  # {snp_path: (shared_memory_name, shape, dtype)}
        self.memory_blocks = []
    
    def add_snp(self, snp_path):
        """Load SNP file and store in shared memory"""
        if str(snp_path) in self.cache:
            return
            
        # Read SNP file
        ntwk = read_snp(snp_path)
        snp_data = ntwk.s  # Complex S-parameter data
        
        # Create shared memory block
        nbytes = snp_data.nbytes
        shm_block = shm.SharedMemory(create=True, size=nbytes)
        
        # Copy data to shared memory
        shm_array = np.ndarray(snp_data.shape, dtype=snp_data.dtype, buffer=shm_block.buf)
        shm_array[:] = snp_data[:]
        
        # Store metadata
        self.cache[str(snp_path)] = {
            'name': shm_block.name,
            'shape': snp_data.shape,
            'dtype': snp_data.dtype,
            'frequencies': ntwk.f.copy()  # Also cache frequencies
        }
        
        self.memory_blocks.append(shm_block)
        _shared_memory_blocks.append(shm_block)
    
    def get_cache_info(self):
        """Get cache information for passing to workers"""
        return self.cache.copy()
    
    def cleanup(self):
        """Clean up shared memory blocks"""
        for block in self.memory_blocks:
            try:
                block.close()
                block.unlink()
            except:
                pass

def init_worker_process(vertical_cache_info):
    """Initialize worker process with shared memory access and signal handling"""
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Store cache info globally in worker
    global _vertical_cache_info
    _vertical_cache_info = vertical_cache_info

def init_worker_thread(vertical_cache_info):
    """Initialize worker thread with shared memory access (threads don't need signal handling)"""
    # Store cache info globally in worker
    global _vertical_cache_info
    _vertical_cache_info = vertical_cache_info

def get_vertical_snp_from_cache(snp_path):
    """Get vertical SNP data from shared memory cache"""
    cache_info = _vertical_cache_info[str(snp_path)]
    
    # Connect to existing shared memory
    shm_block = shm.SharedMemory(name=cache_info['name'])
    
    # Create array view
    snp_array = np.ndarray(
        cache_info['shape'], 
        dtype=cache_info['dtype'], 
        buffer=shm_block.buf
    )
    
    return snp_array.copy(), cache_info['frequencies']  # Return copy to avoid issues when shm_block closes

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
        
    # All tasks in batch should have same trace_snp_file
    trace_snp_file = task_batch[0][0]
    trace_snp_path = Path(trace_snp_file)
    pickle_file = Path(pickle_dir) / f"{trace_snp_path.stem}.pkl"
    
    # Parse n_ports from SNP filename once
    snp_filename = trace_snp_path.name.lower()
    if '.s' in snp_filename and 'p' in snp_filename:
        try:
            extension = snp_filename.split('.s')[-1]
            if not extension.endswith('p'):
                raise ValueError("Extension doesn't end with 'p'")
            port_str = extension.replace('p', '')
            if not port_str.isdigit():
                raise ValueError("No numeric port count found")
            n_ports = int(port_str)
        except (ValueError, IndexError):
            raise ValueError(f"Cannot parse port count from SNP extension in filename: {snp_filename}")
    else:
        import re
        port_match = re.search(r'(\d+)port', snp_filename)
        if port_match:
            n_ports = int(port_match.group(1))
        else:
            raise ValueError(f"Cannot determine number of ports from filename: {snp_filename}")
    
    n_lines = n_ports // 2
    if n_lines == 0:
        raise ValueError(f"Invalid n_ports={n_ports}, n_lines would be 0")
    
    if debug:
        print(f"Processing batch of {len(task_batch)} tasks for {trace_snp_path.name}")
        print(f"Detected {n_ports} ports, {n_lines} lines")
    
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
    
    for trace_snp_file, vertical_snp_pair, sample_count in task_batch:
        snp_tx, snp_rx = vertical_snp_pair
        
        for _ in range(sample_count):
            # Sample parameters
            combined_config = combined_params.sample()
            
            try:
                # Set directions
                if enable_direction:
                    sim_directions = np.random.randint(0, 2, size=n_lines)
                else:
                    sim_directions = np.ones(n_lines, dtype=int)
                
                # Run simulation, limiting threads with threadpoolctl
                if threadpoolctl:
                    with threadpoolctl.threadpool_limits(limits=1):
                        line_ew = snp_eyewidth_simulation(
                            config=combined_config,
                            snp_files=(trace_snp_path, snp_tx, snp_rx),
                            directions=sim_directions
                        )
                else:
                    line_ew = snp_eyewidth_simulation(
                        config=combined_config,
                        snp_files=(trace_snp_path, snp_tx, snp_rx),
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
                    'snp_tx': snp_tx.as_posix(),
                    'snp_rx': snp_rx.as_posix(),
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
                     enable_direction, num_workers, executor_type="process", vertical_cache_info=None):
    """
    Run simulations using either ProcessPoolExecutor or ThreadPoolExecutor
    
    Args:
        batch_list: List of task batches
        combined_params: Parameter set
        trace_specific_output_dir: Output directory
        param_types: Parameter types
        enable_direction: Direction flag
        num_workers: Number of workers
        executor_type: "process" or "thread"
        vertical_cache_info: Shared memory cache info
    """
    
    if executor_type == "thread":
        print(f"Using ThreadPoolExecutor with {num_workers} threads...")
        # For threads, shared memory won't work across processes, so we disable it
        if vertical_cache_info:
            print("Note: Shared memory cache disabled for thread mode")
        
        executor_start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    collect_snp_batch_simulation_data,
                    batch_tasks, combined_params, trace_specific_output_dir, 
                    param_types, enable_direction, False
                )
                for batch_tasks in batch_list
            ]
            
            print(f"Submitted {len(futures)} batches to ThreadPoolExecutor")
            
            try:
                failed_batches = []
                completed_batches = 0
                
                for future in tqdm(
                    concurrent.futures.as_completed(futures), 
                    total=len(futures),
                    desc="Processing batches (threads)"
                ):
                    try:
                        future.result()
                        completed_batches += 1
                        if completed_batches % 10 == 0 or completed_batches <= 5:
                            elapsed = time.time() - executor_start_time
                            rate = completed_batches / elapsed
                            print(f"Completed {completed_batches}/{len(futures)} batches, rate: {rate:.2f} batches/sec")
                    except Exception as e:
                        failed_batches.append(str(e))
                
                total_time = time.time() - executor_start_time
                print(f"ThreadPoolExecutor completed in {total_time:.2f}s, avg rate: {len(futures)/total_time:.2f} batches/sec")
                
                if failed_batches:
                    print(f"\nWarning: {len(failed_batches)} batches failed:")
                    for i, error in enumerate(failed_batches[:5]):
                        print(f"  {i+1}. {error}")
                    if len(failed_batches) > 5:
                        print(f"  ... and {len(failed_batches)-5} more errors")
                        
            except KeyboardInterrupt:
                print("KeyboardInterrupt detected, shutting down threads...")
                # Threads will be terminated when executor exits
                raise
                
    else:  # executor_type == "process"
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
            
            try:
                failed_batches = []
                completed_batches = 0
                
                for future in tqdm(
                    concurrent.futures.as_completed(futures), 
                    total=len(futures),
                    desc="Processing batches (processes)"
                ):
                    try:
                        future.result()
                        completed_batches += 1
                        if completed_batches % 10 == 0 or completed_batches <= 5:
                            elapsed = time.time() - executor_start_time
                            rate = completed_batches / elapsed
                            print(f"Completed {completed_batches}/{len(futures)} batches, rate: {rate:.2f} batches/sec")
                    except Exception as e:
                        failed_batches.append(str(e))
                
                total_time = time.time() - executor_start_time
                print(f"ProcessPoolExecutor completed in {total_time:.2f}s, avg rate: {len(futures)/total_time:.2f} batches/sec")
                
                if failed_batches:
                    print(f"\nWarning: {len(failed_batches)} batches failed:")
                    for i, error in enumerate(failed_batches[:5]):
                        print(f"  {i+1}. {error}")
                    if len(failed_batches) > 5:
                        print(f"  ... and {len(failed_batches)-5} more errors")
                        
            except KeyboardInterrupt:
                print("KeyboardInterrupt detected, shutting down processes...")
                for pid, proc in executor._processes.items():
                    proc.terminate()
                executor.shutdown(wait=False, cancel_futures=True)
                raise

def monitor_system_resources(interval=10):
    """Background monitoring of system resources"""
    print(f"[MONITOR] Starting system resource monitoring (interval: {interval}s)")
    
    while not _monitoring_stop_event.is_set():
        try:
            # Get CPU usage per core
            cpu_percent_per_core = psutil.cpu_percent(interval=1, percpu=True)
            cpu_overall = psutil.cpu_percent(interval=0)
            
            # Get memory info  
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_gb = memory.used / (1024**3)
            memory_total_gb = memory.total / (1024**3)
            
            # Get load average (Unix/Linux only)
            try:
                load_avg = psutil.getloadavg()
                load_str = f", Load: {load_avg[0]:.2f}/{load_avg[1]:.2f}/{load_avg[2]:.2f}"
            except:
                load_str = ""
            
            # Format CPU per-core info (show only first 8 cores if more than 8)
            if len(cpu_percent_per_core) <= 8:
                cpu_cores_str = ", ".join([f"C{i}:{cpu:.1f}%" for i, cpu in enumerate(cpu_percent_per_core)])
            else:
                cpu_cores_str = ", ".join([f"C{i}:{cpu:.1f}%" for i, cpu in enumerate(cpu_percent_per_core[:8])])
                cpu_cores_str += f", ... ({len(cpu_percent_per_core)} cores total)"
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[MONITOR {timestamp}] CPU: {cpu_overall:.1f}% overall | Cores: {cpu_cores_str} | "
                  f"RAM: {memory_used_gb:.1f}/{memory_total_gb:.1f}GB ({memory_percent:.1f}%){load_str}", 
                  flush=True)
            
            # Wait for the remaining interval time or until stop event
            _monitoring_stop_event.wait(interval - 1)  # -1 because cpu_percent already waited 1s
            
        except Exception as e:
            print(f"[MONITOR] Error in monitoring: {e}")
            _monitoring_stop_event.wait(interval)

def start_background_monitoring(interval=10):
    """Start background system monitoring thread"""
    global _monitoring_thread
    if _monitoring_thread is not None:
        return  # Already started
    
    _monitoring_stop_event.clear()
    _monitoring_thread = threading.Thread(
        target=monitor_system_resources, 
        args=(interval,),
        daemon=True,
        name="SystemMonitor"
    )
    _monitoring_thread.start()

def stop_background_monitoring():
    """Stop background system monitoring thread"""
    global _monitoring_thread
    if _monitoring_thread is None:
        return
    
    print("[MONITOR] Stopping system resource monitoring...")
    _monitoring_stop_event.set()
    _monitoring_thread.join(timeout=2)
    _monitoring_thread = None
    print("[MONITOR] System resource monitoring stopped")

def main():
    """Main function for parallel data collection"""
    # Set environment variables to prevent nested parallelism before doing anything else
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    print("Set environment variables to prevent nested parallelism (OMP_NUM_THREADS, etc. = 1)")
    
    args = build_argparser().parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Config file not found: {args.config}")
        print("Please create the config file or provide command line arguments")
        return
    except Exception as e:
        print(f"Error loading config: {e}")
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
    executor_type = args.executor_type
    
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
    print(f"  Executor type: {executor_type}")
    
    # Start background system monitoring
    monitoring_interval = 15  # Monitor every 15 seconds
    start_background_monitoring(monitoring_interval)
    
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
        return
    
    # Initialize shared memory cache for vertical SNPs (only for process mode and not debug)
    vertical_cache = None
    vertical_cache_info = {}
    
    if not debug and executor_type == "process" and vertical_dirs:
        print("Setting up shared memory cache for vertical SNP files...")
        try:
            vertical_cache = VerticalSNPCache()
            
            # Add all unique vertical SNP files to cache
            unique_vertical_snps = set()
            for batch_tasks in batch_list:
                for _, vertical_pair, _ in batch_tasks:
                    unique_vertical_snps.update(vertical_pair)
            
            for snp_path in unique_vertical_snps:
                vertical_cache.add_snp(snp_path)
            
            vertical_cache_info = vertical_cache.get_cache_info()
            print(f"Cached {len(vertical_cache_info)} vertical SNP files in shared memory")
            
        except Exception as e:
            print(f"Warning: Could not set up shared memory cache: {e}")
            if vertical_cache:
                vertical_cache.cleanup()
            vertical_cache = None
            vertical_cache_info = {}
    
    # Run simulations
    try:
        if not debug:
            run_with_executor(batch_list, combined_params, trace_specific_output_dir, param_types, 
                             enable_direction, max_workers, executor_type, vertical_cache_info)
        else:
            # Debug mode - run sequentially
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