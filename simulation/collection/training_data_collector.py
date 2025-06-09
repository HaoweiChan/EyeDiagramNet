"""Main training data collector orchestrating eye width simulation data collection."""

import sys
import yaml
import pickle
import numpy as np
import multiprocessing
import concurrent.futures
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
import multiprocessing.shared_memory as shm

from simulation.parameters.bound_param import PARAM_SETS_MAP
from simulation.engine.eye_width_simulator import snp_eyewidth_simulation
from simulation.io.config_utils import load_config, resolve_trace_pattern, resolve_vertical_dirs, build_argparser
from simulation.io.snp_utils import parse_snps, generate_vertical_snp_pairs
from simulation.parameters.param_utils import parse_param_types, modify_params_for_inductance
from common.signal_utils import read_snp

# Global shared memory registry for cleanup
_shared_memory_blocks = []

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

def init_worker(vertical_cache_info):
    """Initialize worker process with shared memory access"""
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    
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
                
                # Run simulation - the simulator will load SNP files as needed
                # We could optimize this further by pre-loading the horizontal SNP
                # but that would require more changes to the simulator
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

def main():
    """Main function for parallel data collection"""
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
    max_workers = args.max_workers if args.max_workers else config['runner'].get('max_workers')
    
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
        return
    
    # Initialize shared memory cache for vertical SNPs (if not in debug mode)
    vertical_cache = None
    vertical_cache_info = {}
    
    if not debug and vertical_dirs:
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
            # Use multiprocessing with batched tasks
            num_workers = max_workers or multiprocessing.cpu_count()
            multiprocessing.set_start_method("forkserver", force=True)
            
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=num_workers, 
                initializer=init_worker,
                initargs=(vertical_cache_info,)
            ) as executor:
                futures = [
                    executor.submit(
                        collect_snp_batch_simulation_data,
                        batch_tasks, combined_params, trace_specific_output_dir, 
                        param_types, enable_direction, False
                    )
                    for batch_tasks in batch_list
                ]
                
                try:
                    failed_batches = []
                    for future in tqdm(
                        concurrent.futures.as_completed(futures), 
                        total=len(futures),
                        desc="Processing batches"
                    ):
                        try:
                            future.result()
                        except Exception as e:
                            failed_batches.append(str(e))
                    
                    if failed_batches:
                        print(f"\nWarning: {len(failed_batches)} batches failed:")
                        for i, error in enumerate(failed_batches[:5]):
                            print(f"  {i+1}. {error}")
                        if len(failed_batches) > 5:
                            print(f"  ... and {len(failed_batches)-5} more errors")
                            
                except KeyboardInterrupt:
                    print("KeyboardInterrupt detected, shutting down...")
                    for pid, proc in executor._processes.items():
                        proc.terminate()
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise
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
        # Clean up shared memory
        if vertical_cache:
            vertical_cache.cleanup()
        cleanup_shared_memory()

if __name__ == "__main__":
    main()