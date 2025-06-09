"""Main training data collector orchestrating eye width simulation data collection."""

import sys
import yaml
import pickle
import numpy as np
import multiprocessing
import concurrent.futures
from tqdm import tqdm
from pathlib import Path

from simulation.parameters.bound_param import PARAM_SETS_MAP
from simulation.engine.eye_width_simulator import snp_eyewidth_simulation
from simulation.io.config_utils import load_config, resolve_trace_pattern, resolve_vertical_dirs, build_argparser
from simulation.io.snp_utils import parse_snps, generate_vertical_snp_pairs
from simulation.parameters.param_utils import parse_param_types, modify_params_for_inductance

def init_worker():
    """Initialize worker process - ignore ctrl+c in child workers so only main process sees it"""
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def collect_snp_simulation_data(trace_snp_file, vertical_snp_pair, params_set, 
                               pickle_dir, param_type_names, enable_direction=True, directions=None, debug=False):
    """
    Collect eye width simulation data for a single trace SNP with vertical SNP pair.
    
    Args:
        trace_snp_file: Path to trace SNP file
        vertical_snp_pair: Tuple of (tx_snp, rx_snp) paths
        params_set: ParameterSet containing all required parameters
        pickle_dir: Directory to save pickle files
        param_type_names: List of parameter type names
        enable_direction: Whether to use random directions (True) or all ones (False)
        directions: Optional directions array
        debug: Debug mode flag
    """
    snp_tx, snp_rx = vertical_snp_pair
    trace_snp_path = Path(trace_snp_file)
    pickle_file = Path(pickle_dir) / f"{trace_snp_path.stem}.pkl"
    
    # Parse n_ports from SNP filename (e.g., "trace_8port.s8p" -> 8 ports)
    snp_filename = trace_snp_path.name.lower()
    
    if '.s' in snp_filename and 'p' in snp_filename:
        # Extract number from .sXp extension
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
        # Try to extract from filename pattern
        import re
        port_match = re.search(r'(\d+)port', snp_filename)
        if port_match:
            n_ports = int(port_match.group(1))
        else:
            raise ValueError(f"Cannot determine number of ports from filename: {snp_filename}. "
                           f"Filename must contain .sXp extension (e.g., .s8p) or 'Xport' pattern (e.g., 8port)")
    
    # Calculate number of lines (differential pairs)
    n_lines = n_ports // 2
    
    if n_lines == 0:
        raise ValueError(f"Invalid n_ports={n_ports}, n_lines would be 0. Need at least 2 ports for differential pairs.")
    
    if debug:
        print(f"Detected {n_ports} ports, {n_lines} lines from {snp_filename}")
    
    # Load existing data if pickle file exists
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
            'meta': {}  # Parameter meta
        }
    
    # Sample all parameters from the combined parameter set
    combined_config = params_set.sample()
    
    # Add SNP file paths to config
    config_dict = combined_config.to_dict()
    config_dict['snp_horiz'] = str(trace_snp_path)
    config_dict['snp_tx'] = str(snp_tx)
    config_dict['snp_rx'] = str(snp_rx)
    
    if debug:
        print(f"\nProcessing {trace_snp_path.name}")
        print(f"TX: {snp_tx.name}, RX: {snp_rx.name}")
        print(f"Config: {config_dict}")
    
    try:
        # Set directions based on enable_direction flag
        if directions is None:
            if enable_direction:
                # Generate random directions (0 or 1) for each line
                sim_directions = np.random.randint(0, 2, size=n_lines)
            else:
                # Use all ones for all lines
                sim_directions = np.ones(n_lines, dtype=int)
        else:
            sim_directions = directions
            
        # Run eye width simulation using the correct function
        line_ew = snp_eyewidth_simulation(
            config=combined_config,
            snp_files=(trace_snp_path, snp_tx, snp_rx),
            directions=sim_directions
        )
        
        # Handle case where line_ew might be a tuple (line_ew, directions)
        if isinstance(line_ew, tuple):
            line_ew, actual_directions = line_ew
            sim_directions = actual_directions
        
        # Ensure line_ew is a numpy array and handle closed eyes
        line_ew = np.array(line_ew)
        line_ew[line_ew >= 99.9] = -0.1  # treating 99.9 data as closed eyes
        
        if debug:
            print(f"Eye widths: {line_ew}")
            print(f"Directions: {sim_directions}")
            
    except Exception as e:
        print(f"Error in simulation for {trace_snp_path.name}: {e}")
        raise
    
    # Append new data
    config_values, config_keys = combined_config.to_list(return_keys=True)
    data['configs'].append(config_values)
    data['line_ews'].append(line_ew.tolist())
    data['snp_txs'].append(snp_tx.as_posix())
    data['snp_rxs'].append(snp_rx.as_posix()) 
    data['directions'].append(sim_directions.tolist())
    
    # Update meta with parameter info (store once per file)
    if not data['meta'].get('config_keys'):
        data['meta']['config_keys'] = config_keys
        data['meta']['n_ports'] = n_ports
        data['meta']['param_types'] = param_type_names
    
    # Save updated data
    pickle_file.parent.mkdir(parents=True, exist_ok=True)
    with open(pickle_file, 'wb') as f:
        pickle.dump(data, f)
    
    if debug:
        print(f"Saved data to {pickle_file}")

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
    base_output_dir = output_dir # Save original output_dir as base
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
            # Save the config that was used for this collection
            with open(config_save_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            print(f"Saved collection config to: {config_save_path}")
        except Exception as e:
            print(f"Warning: Could not save config file: {e}")
    
    # Prepare simulation tasks
    simulation_tasks = []
    for trace_snp, vertical_pair in zip(trace_snps, vertical_pairs):
        # Check if we need more samples for this trace SNP
        # Note: pickle_file path is now relative to trace_specific_output_dir
        pickle_file = trace_specific_output_dir / f"{Path(trace_snp).stem}.pkl"
        current_samples = 0
        
        if pickle_file.exists():
            try:
                with open(pickle_file, 'rb') as f:
                    existing_data = pickle.load(f)
                current_samples = len(existing_data.get('configs', []))
            except:
                current_samples = 0
        
        # Only add task if we need more samples
        if current_samples < max_samples:
            samples_needed = max_samples - current_samples
            for _ in range(samples_needed):
                simulation_tasks.append((trace_snp, vertical_pair, combined_params, trace_specific_output_dir, param_types, enable_direction))
    
    print(f"Need to run {len(simulation_tasks)} simulations")
    
    if len(simulation_tasks) == 0:
        print("All files already have sufficient samples")
        return
    
    # Run simulations
    if not debug:
        # Use multiprocessing
        num_workers = max_workers or multiprocessing.cpu_count()
        multiprocessing.set_start_method("forkserver", force=True)
        
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers, 
            initializer=init_worker
        ) as executor:
            futures = [
                executor.submit(
                    collect_snp_simulation_data,
                    trace_snp, vertical_pair, combined_params,
                    trace_specific_output_dir, param_types, enable_direction, None, False
                )
                for trace_snp, vertical_pair, combined_params, trace_specific_output_dir, param_types, enable_direction in simulation_tasks
            ]
            
            try:
                failed_tasks = []
                for future in tqdm(
                    concurrent.futures.as_completed(futures), 
                    total=len(futures),
                    desc="Collecting simulation data"
                ):
                    try:
                        future.result()  # This will raise any exception that occurred
                    except Exception as e:
                        failed_tasks.append(str(e))
                
                if failed_tasks:
                    print(f"\nWarning: {len(failed_tasks)} tasks failed:")
                    for i, error in enumerate(failed_tasks[:5]):  # Show first 5 errors
                        print(f"  {i+1}. {error}")
                    if len(failed_tasks) > 5:
                        print(f"  ... and {len(failed_tasks)-5} more errors")
                        
            except KeyboardInterrupt:
                print("KeyboardInterrupt detected, shutting down...")
                # Kill each process
                for pid, proc in executor._processes.items():
                    proc.terminate()
                # Shutdown the pool
                executor.shutdown(wait=False, cancel_futures=True)
                sys.exit(1)
    else:
        # Debug mode - run sequentially
        for i, (trace_snp, vertical_pair, combined_params, trace_specific_output_dir, param_types, enable_direction) in enumerate(
            tqdm(simulation_tasks, desc="Debug simulation")
        ):
            print(f"\n--- Task {i+1}/{len(simulation_tasks)} ---")
            collect_snp_simulation_data(
                trace_snp, vertical_pair, combined_params,
                trace_specific_output_dir, param_types, enable_direction, None, True
            )
    
    print(f"Data collection completed. Results saved to: {trace_specific_output_dir}")

if __name__ == "__main__":
    main()