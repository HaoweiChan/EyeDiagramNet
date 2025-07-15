"""
Fixed parallel collector that addresses BLAS threading contention issues.

Key changes:
1. Force single-threaded BLAS in all workers (eliminates contention)
2. Reduce worker count to avoid over-subscription 
3. Use larger task batches to reduce process overhead
4. Add performance monitoring to validate improvements
"""

import os
import sys
import time
import platform
import psutil
import multiprocessing
import concurrent.futures
from pathlib import Path

# CRITICAL FIX: Force single-threaded BLAS BEFORE importing numerical libraries
# This eliminates the 20x slowdown caused by BLAS threading contention
def force_single_threaded_blas():
    """Force all BLAS libraries to use single-threaded mode"""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1" 
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["NUMBA_NUM_THREADS"] = "1"
    print("PERFORMANCE FIX: Set all BLAS libraries to single-threaded mode")

force_single_threaded_blas()

# Now import numerical libraries after BLAS is configured
import numpy as np
import pickle
import threading
import signal
import traceback
from datetime import datetime

# Import your existing modules
from common.signal_utils import read_snp
from simulation.parameters.bound_param import PARAM_SETS_MAP
from simulation.engine.eye_width_simulator import snp_eyewidth_simulation
from simulation.io.config_utils import load_config, resolve_trace_pattern, resolve_vertical_dirs, build_argparser
from simulation.io.snp_utils import parse_snps, generate_vertical_snp_pairs
from simulation.parameters.param_utils import parse_param_types, modify_params_for_inductance

# Global shutdown control
_shutdown_event = threading.Event()

def get_optimal_workers_fixed():
    """
    Calculate optimal worker count based on BLAS contention test results.
    
    Key insight: Since each worker uses 1 BLAS thread, we can use more workers
    than cores without severe contention, but need to be conservative.
    """
    cpu_count = psutil.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    system = platform.system()
    
    print(f"System: {cpu_count} CPUs, {memory_gb:.1f}GB RAM, Platform: {system}")
    
    # Conservative worker allocation based on test results
    # Test showed that even with 1 BLAS thread per worker, we get some slowdown
    # So we use fewer workers but larger task batches
    
    if system == "Darwin":  # macOS
        # Very conservative for development
        optimal_workers = min(4, max(2, cpu_count // 2))
        print(f"macOS: Using {optimal_workers} workers (conservative for development)")
        
    elif system == "Linux":  # Linux
        # More aggressive but still conservative based on test results
        # Use 75% of cores since each worker uses 1 BLAS thread
        optimal_workers = max(4, min(cpu_count // 2, int(cpu_count * 0.75)))
        print(f"Linux: Using {optimal_workers} workers (75% of cores with 1 BLAS thread each)")
        
    else:
        # Unknown platform - very conservative
        optimal_workers = min(4, max(2, cpu_count // 3))
        print(f"Unknown platform: Using {optimal_workers} workers (very conservative)")
    
    # Memory check - each worker should have at least 1GB
    memory_limited_workers = max(2, int(memory_gb * 0.8))
    if optimal_workers > memory_limited_workers:
        print(f"Memory constraint: Reducing workers from {optimal_workers} to {memory_limited_workers}")
        optimal_workers = memory_limited_workers
    
    print(f"Final worker count: {optimal_workers} (each using 1 BLAS thread)")
    print(f"Expected thread utilization: {optimal_workers}/{cpu_count} cores ({optimal_workers/cpu_count*100:.1f}%)")
    
    return optimal_workers

def init_worker_process_fixed():
    """Initialize worker process with single-threaded BLAS (already set globally)"""
    worker_id = os.getpid()
    
    # Ignore interrupt signals in workers
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    # Verify BLAS threading is single-threaded
    blas_threads = os.environ.get("OMP_NUM_THREADS", "unknown")
    print(f"[Worker {worker_id}] Initialized with BLAS threads: {blas_threads}")
    
    # Set CPU affinity if possible (Linux only)
    if platform.system() == "Linux":
        try:
            import psutil
            current_process = psutil.Process()
            cpu_count = psutil.cpu_count()
            # Simple round-robin CPU assignment
            assigned_cpu = worker_id % cpu_count
            current_process.cpu_affinity([assigned_cpu])
            print(f"[Worker {worker_id}] Assigned to CPU {assigned_cpu}")
        except:
            pass

def process_trace_batch_fixed(trace_snp_files, vertical_pairs_list, combined_params, 
                             output_dir, param_types, max_samples, enable_direction, 
                             batch_size=20):
    """
    Process multiple trace files in a single worker to reduce process overhead.
    
    This is the key optimization: instead of one process per trace file,
    we process multiple trace files per worker to amortize startup costs.
    """
    worker_id = os.getpid()
    start_time = time.time()
    
    completed_simulations = 0
    total_files = len(trace_snp_files)
    
    print(f"[Worker {worker_id}] Processing {total_files} trace files")
    
    for i, (trace_snp_file, vertical_pairs) in enumerate(zip(trace_snp_files, vertical_pairs_list)):
        if _shutdown_event.is_set():
            print(f"[Worker {worker_id}] Shutdown detected, stopping early")
            break
        
        file_start_time = time.time()
        
        # Load trace SNP
        trace_ntwk = read_snp(trace_snp_file)
        n_ports = trace_ntwk.nports
        n_lines = n_ports // 2
        
        # Check existing samples
        pickle_file = Path(output_dir) / f"{trace_snp_file.stem}.pkl"
        existing_samples = 0
        if pickle_file.exists():
            try:
                with open(pickle_file, 'rb') as f:
                    existing_data = pickle.load(f)
                existing_samples = len(existing_data.get('configs', []))
            except:
                pass
        
        if existing_samples >= max_samples:
            print(f"[Worker {worker_id}] File {i+1}/{total_files}: {trace_snp_file.name} already complete")
            continue
        
        samples_needed = max_samples - existing_samples
        print(f"[Worker {worker_id}] File {i+1}/{total_files}: {trace_snp_file.name} needs {samples_needed} samples")
        
        # Load vertical SNPs
        snp_tx_path, snp_rx_path = vertical_pairs
        tx_ntwk = read_snp(snp_tx_path)
        rx_ntwk = read_snp(snp_rx_path)
        
        # Process samples for this file
        results = []
        for sample_idx in range(samples_needed):
            if _shutdown_event.is_set():
                break
            
            # Sample parameters
            combined_config = combined_params.sample()
            
            # Generate directions
            if enable_direction:
                sim_directions = np.random.choice([0, 1], size=n_lines)
            else:
                sim_directions = np.ones(n_lines, dtype=int)
            
            # Run simulation
            try:
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
                    'snp_tx': str(snp_tx_path),
                    'snp_rx': str(snp_rx_path),
                    'directions': sim_directions.tolist(),
                    'snp_horiz': str(trace_snp_file),
                    'n_ports': n_ports,
                    'param_types': param_types
                }
                
                results.append(result)
                completed_simulations += 1
                
            except Exception as e:
                print(f"[Worker {worker_id}] Simulation failed: {e}")
                continue
        
        # Save results for this file
        if results:
            save_results_to_pickle(pickle_file, results)
            
        file_time = time.time() - file_start_time
        print(f"[Worker {worker_id}] Completed {trace_snp_file.name}: {len(results)} simulations in {file_time:.1f}s")
    
    total_time = time.time() - start_time
    print(f"[Worker {worker_id}] Completed {total_files} files, {completed_simulations} simulations in {total_time:.1f}s")
    
    return completed_simulations

def save_results_to_pickle(pickle_file, new_results):
    """Save results to pickle file"""
    # Load existing data
    existing_data = {'configs': [], 'line_ews': [], 'snp_txs': [], 'snp_rxs': [], 'directions': [], 'meta': {}}
    if pickle_file.exists():
        try:
            with open(pickle_file, 'rb') as f:
                existing_data = pickle.load(f)
        except:
            pass
    
    # Add new results
    for result in new_results:
        existing_data['configs'].append(result['config_values'])
        existing_data['line_ews'].append(result['line_ews'])
        existing_data['snp_txs'].append(result['snp_tx'])
        existing_data['snp_rxs'].append(result['snp_rx'])
        existing_data['directions'].append(result['directions'])
    
    # Update metadata
    if new_results and not existing_data['meta'].get('config_keys'):
        first_result = new_results[0]
        existing_data['meta']['config_keys'] = first_result['config_keys']
        existing_data['meta']['snp_horiz'] = first_result.get('snp_horiz', '')
        existing_data['meta']['n_ports'] = first_result.get('n_ports', 0)
        existing_data['meta']['param_types'] = first_result.get('param_types', [])
    
    # Save to file
    pickle_file.parent.mkdir(parents=True, exist_ok=True)
    with open(pickle_file, 'wb') as f:
        pickle.dump(existing_data, f)

def create_task_batches(trace_snps, vertical_pairs, num_workers):
    """
    Create task batches for workers to reduce process overhead.
    
    Key insight: Instead of many small tasks, use fewer large tasks
    to reduce multiprocessing overhead.
    """
    total_files = len(trace_snps)
    
    # Target 2-3 tasks per worker for good load balancing
    target_tasks = num_workers * 2
    files_per_task = max(1, total_files // target_tasks)
    
    batches = []
    for i in range(0, total_files, files_per_task):
        batch_files = trace_snps[i:i + files_per_task]
        batch_pairs = vertical_pairs[i:i + files_per_task]
        batches.append((batch_files, batch_pairs))
    
    print(f"Created {len(batches)} task batches ({files_per_task} files per batch)")
    return batches

def run_parallel_collection_fixed(trace_snps, vertical_pairs, combined_params, 
                                 output_dir, param_types, max_samples, enable_direction):
    """
    Run parallel collection with fixed BLAS threading and optimized task distribution.
    """
    num_workers = get_optimal_workers_fixed()
    
    # Create task batches
    task_batches = create_task_batches(trace_snps, vertical_pairs, num_workers)
    
    print(f"Starting parallel collection with {num_workers} workers processing {len(task_batches)} batches")
    
    # Set multiprocessing start method
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    start_time = time.time()
    total_completed = 0
    
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=init_worker_process_fixed
    ) as executor:
        
        # Submit all tasks
        futures = []
        for batch_files, batch_pairs in task_batches:
            future = executor.submit(
                process_trace_batch_fixed,
                batch_files, batch_pairs, combined_params,
                output_dir, param_types, max_samples, enable_direction
            )
            futures.append(future)
        
        # Wait for completion
        try:
            for future in concurrent.futures.as_completed(futures):
                if _shutdown_event.is_set():
                    break
                
                try:
                    result = future.result()
                    total_completed += result
                except Exception as e:
                    print(f"Task failed: {e}")
                    traceback.print_exc()
        
        except KeyboardInterrupt:
            print("Received interrupt, shutting down...")
            _shutdown_event.set()
            
            # Cancel remaining futures
            for future in futures:
                future.cancel()
    
    total_time = time.time() - start_time
    print(f"Parallel collection completed: {total_completed} simulations in {total_time:.1f}s")
    return total_completed

def main():
    """Main function with configuration handling that matches default.yaml"""
    print("EyeDiagramNet - Fixed Parallel Data Collector")
    print("=" * 60)
    
    # Parse arguments
    args = build_argparser().parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Config file not found: {args.config}")
        print("Please create the config file or provide command line arguments")
        return 1
    except Exception as e:
        print(f"Error loading config: {e}")
        return 1
    
    # Resolve dataset paths (same as parallel_collector.py)
    horizontal_dataset = config.get('dataset', {}).get('horizontal_dataset', {})
    vertical_dataset = config.get('dataset', {}).get('vertical_dataset')
    
    # Override config with command line arguments (same pattern as parallel_collector.py)
    trace_pattern_key = args.trace_pattern if args.trace_pattern else config['data']['trace_pattern']
    trace_pattern = resolve_trace_pattern(trace_pattern_key, horizontal_dataset)
    vertical_dirs = resolve_vertical_dirs(vertical_dataset)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['data']['output_dir'])
    param_type_str = args.param_type if args.param_type else config['boundary']['param_type']
    param_types = parse_param_types(param_type_str)
    max_samples = args.max_samples if args.max_samples else config['boundary']['max_samples']
    
    # Handle enable_direction logic (default to False) - same as parallel_collector.py
    enable_direction = args.enable_direction or config['boundary'].get('enable_direction', False)
    
    # Handle enable_inductance logic (default to False) - same as parallel_collector.py
    enable_inductance = args.enable_inductance or config['boundary'].get('enable_inductance', False)
    
    debug = args.debug if args.debug else config.get('debug', False)
    
    # Get batch size from config (same as parallel_collector.py)
    runner_config = config.get('runner', {})
    batch_size = runner_config.get('batch_size', 20)
    
    # Display configuration (same style as parallel_collector.py)
    print(f"Using configuration:")
    print(f"  Trace pattern: {trace_pattern_key} -> {trace_pattern}")
    print(f"  Vertical dirs: {vertical_dirs}")
    print(f"  Output dir: {output_dir}")
    print(f"  Parameter types: {param_types}")
    print(f"  Max samples: {max_samples}")
    print(f"  Enable direction: {enable_direction}")
    print(f"  Enable inductance: {enable_inductance}")
    print(f"  Debug mode: {debug}")
    print(f"  Batch size: {batch_size}")
    print(f"  BLAS threads per worker: 1 (PERFORMANCE FIX)")
    
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
    vertical_pairs = generate_vertical_snp_pairs(
        vertical_dirs, len(trace_snps), trace_snps, base_output_dir, trace_pattern_key
    )
    
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
    
    # Run collection
    try:
        total_completed = run_parallel_collection_fixed(
            trace_snps, vertical_pairs, combined_params,
            trace_specific_output_dir, param_types, max_samples, enable_direction
        )
        
        print(f"Collection completed successfully!")
        print(f"Total simulations: {total_completed}")
        print(f"Results saved to: {trace_specific_output_dir}")
        
        return 0
        
    except KeyboardInterrupt:
        print("Collection interrupted by user")
        return 130
    except Exception as e:
        print(f"Collection failed: {e}")
        if debug:
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 