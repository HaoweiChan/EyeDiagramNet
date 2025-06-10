"""I/O-optimized training data collector with SNP file pre-loading and caching."""

import sys
import yaml
import pickle
import numpy as np
import multiprocessing
import concurrent.futures
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
import time
import threading
from datetime import datetime

from simulation.parameters.bound_param import PARAM_SETS_MAP
from simulation.engine.eye_width_simulator import snp_eyewidth_simulation
from simulation.io.config_utils import load_config, resolve_trace_pattern, resolve_vertical_dirs, build_argparser
from simulation.io.snp_utils import parse_snps, generate_vertical_snp_pairs
from simulation.parameters.param_utils import parse_param_types, modify_params_for_inductance
from common.signal_utils import read_snp

# Global cache for pre-loaded SNP files
_snp_cache = {}
_cache_lock = threading.Lock()

def get_worker_id():
    """Get unique worker identifier"""
    import threading
    thread_id = threading.get_ident()
    process_id = os.getpid()
    return f"P{process_id}-T{thread_id}"

def profile_print(message, elapsed_time=None):
    """Thread-safe profiling print"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    worker_id = get_worker_id()
    
    if elapsed_time is not None:
        print(f"[{timestamp}] [{worker_id}] {message} ({elapsed_time:.2f}s)", flush=True)
    else:
        print(f"[{timestamp}] [{worker_id}] {message}", flush=True)

def preload_snp_file(snp_path):
    """Pre-load and cache SNP file data"""
    with _cache_lock:
        if str(snp_path) not in _snp_cache:
            try:
                start_time = time.time()
                ntwk = read_snp(snp_path)
                load_time = time.time() - start_time
                
                # Store both s-parameters and frequencies
                _snp_cache[str(snp_path)] = {
                    's_params': ntwk.s.copy(),  # Make a copy to avoid reference issues
                    'frequencies': ntwk.f.copy(),
                    'network': ntwk  # Keep original for compatibility
                }
                
                file_size_mb = snp_path.stat().st_size / (1024*1024)
                print(f"Pre-loaded {snp_path.name} ({file_size_mb:.1f}MB) in {load_time:.2f}s")
                
            except Exception as e:
                print(f"Error pre-loading {snp_path}: {e}")
                _snp_cache[str(snp_path)] = None

def get_cached_snp(snp_path):
    """Get pre-loaded SNP data from cache"""
    with _cache_lock:
        return _snp_cache.get(str(snp_path))

def preload_all_snp_files(trace_snps, vertical_pairs):
    """Pre-load all SNP files before processing"""
    all_snp_files = set()
    
    # Collect all unique SNP files
    for trace_snp in trace_snps:
        all_snp_files.add(Path(trace_snp))
    
    for _, vertical_pair in vertical_pairs:
        snp_tx, snp_rx = vertical_pair
        all_snp_files.add(snp_tx)
        all_snp_files.add(snp_rx)
    
    print(f"Pre-loading {len(all_snp_files)} unique SNP files...")
    start_time = time.time()
    
    # Pre-load sequentially to avoid I/O contention
    for snp_file in sorted(all_snp_files):
        preload_snp_file(snp_file)
    
    total_time = time.time() - start_time
    successful_loads = sum(1 for cached in _snp_cache.values() if cached is not None)
    print(f"Pre-loaded {successful_loads}/{len(all_snp_files)} SNP files in {total_time:.2f}s")

def collect_snp_batch_simulation_data_optimized(task_batch, combined_params, pickle_dir, 
                                               param_type_names, enable_direction=True, debug=False):
    """
    Optimized batch processing with reduced I/O overhead
    """
    if not task_batch:
        return
        
    batch_start_time = time.time()
    profile_print(f"Starting optimized batch with {len(task_batch)} tasks")
        
    # All tasks in batch should have same trace_snp_file
    trace_snp_file = task_batch[0][0]
    trace_snp_path = Path(trace_snp_file)
    pickle_file = Path(pickle_dir) / f"{trace_snp_path.stem}.pkl"
    
    try:
        file_size_mb = trace_snp_path.stat().st_size / (1024*1024)
        profile_print(f"Processing {trace_snp_path.name} ({file_size_mb:.1f}MB)")
    except:
        profile_print(f"Processing {trace_snp_path.name}")
    
    # Parse n_ports from SNP filename
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
    
    # Load existing pickle data
    pickle_load_start = time.time()
    if pickle_file.exists():
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
        profile_print(f"Loaded existing pickle with {len(data.get('configs', []))} samples", time.time() - pickle_load_start)
    else:
        data = {
            'configs': [],
            'line_ews': [], 
            'snp_txs': [],
            'snp_rxs': [],
            'directions': [],
            'meta': {}
        }
        profile_print("Created new pickle data structure", time.time() - pickle_load_start)
    
    # Process all tasks in batch
    batch_results = []
    total_samples = sum(sample_count for _, _, sample_count in task_batch)
    profile_print(f"Processing {total_samples} simulations")
    
    simulation_times = []
    
    for task_idx, (trace_snp_file, vertical_snp_pair, sample_count) in enumerate(task_batch):
        snp_tx, snp_rx = vertical_snp_pair
        
        for sample_idx in range(sample_count):
            sample_start_time = time.time()
            
            # Sample parameters
            param_start = time.time()
            combined_config = combined_params.sample()
            param_time = time.time() - param_start
            
            try:
                # Set directions
                if enable_direction:
                    sim_directions = np.random.randint(0, 2, size=n_lines)
                else:
                    sim_directions = np.ones(n_lines, dtype=int)
                
                # Run simulation
                sim_start = time.time()
                line_ew = snp_eyewidth_simulation(
                    config=combined_config,
                    snp_files=(trace_snp_path, snp_tx, snp_rx),
                    directions=sim_directions
                )
                sim_time = time.time() - sim_start
                
                # Handle tuple return
                if isinstance(line_ew, tuple):
                    line_ew, actual_directions = line_ew
                    sim_directions = actual_directions
                
                # Process results
                process_start = time.time()
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
                process_time = time.time() - process_start
                
                sample_time = time.time() - sample_start_time
                simulation_times.append(sample_time)
                
                if sample_idx % 1 == 0:  # Print every sample for debugging
                    profile_print(f"Sample {sample_idx+1}: param={param_time:.3f}s, sim={sim_time:.1f}s, process={process_time:.3f}s, total={sample_time:.1f}s")
                    
            except Exception as e:
                sample_time = time.time() - sample_start_time
                profile_print(f"Sample {sample_idx+1} failed: {e}", sample_time)
                if debug:
                    import traceback
                    traceback.print_exc()
                continue
    
    # Statistics
    if simulation_times:
        avg_sim_time = np.mean(simulation_times)
        profile_print(f"Batch stats: avg={avg_sim_time:.1f}s, min={min(simulation_times):.1f}s, max={max(simulation_times):.1f}s")
    
    # Append results to data
    append_start = time.time()
    for result in batch_results:
        data['configs'].append(result['config_values'])
        data['line_ews'].append(result['line_ews'])
        data['snp_txs'].append(result['snp_tx'])
        data['snp_rxs'].append(result['snp_rx'])
        data['directions'].append(result['directions'])
    append_time = time.time() - append_start
    
    # Update metadata
    if batch_results and not data['meta'].get('config_keys'):
        data['meta']['snp_horiz'] = str(trace_snp_path)
        data['meta']['config_keys'] = batch_results[0]['config_keys']
        data['meta']['n_ports'] = n_ports
        data['meta']['param_types'] = param_type_names
    
    # Save results
    save_start = time.time()
    pickle_file.parent.mkdir(parents=True, exist_ok=True)
    with open(pickle_file, 'wb') as f:
        pickle.dump(data, f)
    save_time = time.time() - save_start
    
    batch_total_time = time.time() - batch_start_time
    profile_print(f"Batch completed: {len(batch_results)} simulations, append={append_time:.2f}s, save={save_time:.2f}s", batch_total_time)

def run_sequential_optimized(batch_list, combined_params, trace_specific_output_dir, param_types, enable_direction):
    """Run sequentially but with optimizations"""
    print("Running SEQUENTIAL mode with optimizations...")
    
    start_time = time.time()
    for i, batch_tasks in enumerate(tqdm(batch_list, desc="Sequential processing")):
        batch_start = time.time()
        collect_snp_batch_simulation_data_optimized(
            batch_tasks, combined_params, trace_specific_output_dir,
            param_types, enable_direction, False
        )
        batch_time = time.time() - batch_start
        print(f"Batch {i+1}/{len(batch_list)} completed in {batch_time:.1f}s")
    
    total_time = time.time() - start_time
    print(f"Sequential processing completed in {total_time:.1f}s")

def main():
    """Main function for optimized data collection"""
    args = build_argparser().parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Config file not found: {args.config}")
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
    
    enable_direction = args.enable_direction or config['boundary'].get('enable_direction', False)
    enable_inductance = args.enable_inductance or config['boundary'].get('enable_inductance', False)
    debug = args.debug if args.debug else config.get('debug', False)
    max_workers = args.max_workers if args.max_workers else config['runner'].get('max_workers')
    executor_type = args.executor_type
    
    print(f"=== OPTIMIZED DATA COLLECTOR ===")
    print(f"Trace pattern: {trace_pattern_key}")
    print(f"Max samples: {max_samples}")
    print(f"Executor type: {executor_type}")
    print("Optimization: Pre-loading SNP files, sequential processing")
    
    # Create output directory
    base_output_dir = output_dir
    trace_specific_output_dir = base_output_dir / trace_pattern_key
    trace_specific_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load trace SNP files
    trace_snps = parse_snps(trace_pattern)
    if len(trace_snps) == 0:
        raise ValueError(f"No trace SNP files found in: {trace_pattern}")
    
    print(f"Found {len(trace_snps)} trace SNP files")
    
    # Generate vertical SNP pairs
    vertical_pairs = generate_vertical_snp_pairs(vertical_dirs, len(trace_snps), trace_snps, base_output_dir, trace_pattern_key)
    print(f"Generated {len(vertical_pairs)} vertical SNP pairs")
    
    # PRE-LOAD ALL SNP FILES SEQUENTIALLY
    preload_all_snp_files(trace_snps, vertical_pairs)
    
    # Combine parameter sets
    combined_params = None
    for param_type in param_types:
        param_set = PARAM_SETS_MAP[param_type]
        if combined_params is None:
            combined_params = param_set
        else:
            combined_params = combined_params + param_set
    
    # Apply inductance modification if needed
    combined_params = modify_params_for_inductance(combined_params, enable_inductance)
    
    # Build task batches
    task_batches = defaultdict(list)
    
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
    
    batch_list = [batch_tasks for batch_tasks in task_batches.values() if batch_tasks]
    total_tasks = sum(sum(samples for _, _, samples in batch) for batch in batch_list)
    
    print(f"Created {len(batch_list)} batches for {total_tasks} total simulations")
    
    if len(batch_list) == 0:
        print("All files already have sufficient samples")
        return
    
    # ALWAYS run sequentially for optimal performance with large files
    run_sequential_optimized(batch_list, combined_params, trace_specific_output_dir, param_types, enable_direction)
    
    print(f"Results saved to: {trace_specific_output_dir}")

if __name__ == "__main__":
    main() 