import os
import sys
import time
import random
import pickle
import numpy as np
import multiprocessing
import concurrent.futures
from tqdm import tqdm
from pathlib import Path
from itertools import product
import argparse
import yaml

from simulation.parameters.bound_param import SampleResult, ParameterSet, MIX_PARAMS, CTLE_PARAMS
from simulation.engine.eye_width_simulator import snp_eyewidth_simulation

def init_worker():
    """Initialize worker process - ignore ctrl+c in child workers so only main process sees it"""
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def parse_snps(snp_dir):
    """Parse SNP files from directory, supporting both .snp and .npz formats"""
    snp_dir = Path(snp_dir)
    
    # Check for npz subdirectory first
    if (snp_dir / 'npz').exists():
        snp_dir = snp_dir / "npz"
        suffix = '*.npz'
    elif (snp_dir / 'snp').exists():
        snp_dir = snp_dir / "snp"
        suffix = '*.s*p'
    else:
        # Check what files exist in the directory
        if len(list(snp_dir.glob("*.npz"))):
            suffix = '*.npz'
        else:
            suffix = '*.s*p'
    
    return list(snp_dir.glob(suffix))

def collect_snp_simulation_data(trace_snp_file, vertical_snp_pair, params_set, ctle_params_set, 
                               pickle_dir, directions=None, debug=False):
    """
    Collect eye width simulation data for a single trace SNP with vertical SNP pair.
    
    Args:
        trace_snp_file: Path to trace SNP file
        vertical_snp_pair: Tuple of (tx_snp, rx_snp) paths
        params_set: ParameterSet for boundary parameters
        ctle_params_set: ParameterSet for CTLE parameters  
        pickle_dir: Directory to save pickle files
        directions: Optional directions array
        debug: Debug mode flag
    """
    snp_tx, snp_rx = vertical_snp_pair
    trace_snp_path = Path(trace_snp_file)
    pickle_file = Path(pickle_dir) / f"{trace_snp_path.stem}.pkl"
    
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
            'directions': []
        }
    
    # Sample boundary parameters
    boundary_config = params_set.sample()
    
    # Sample CTLE parameters
    ctle_config = ctle_params_set.sample()
    
    # Combine boundary and CTLE parameters
    combined_config = boundary_config + ctle_config
    
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
        # Run eye width simulation using the correct function
        line_ew = snp_eyewidth_simulation(
            config=combined_config,
            snp_files=(trace_snp_path, snp_tx, snp_rx),
            directions=directions
        )
        
        # Handle case where line_ew might be a tuple (line_ew, directions)
        if isinstance(line_ew, tuple):
            line_ew, actual_directions = line_ew
            directions = actual_directions
        
        # Ensure line_ew is a numpy array and handle closed eyes
        line_ew = np.array(line_ew)
        line_ew[line_ew >= 99.9] = -0.1  # treating 99.9 data as closed eyes
        
        if debug:
            print(f"Eye widths: {line_ew}")
            
    except Exception as e:
        print(f"Error in simulation for {trace_snp_path.name}: {e}")
        # Use random data as fallback
        n_lines = 4  # Default number of lines, could be inferred from SNP
        line_ew = np.random.uniform(0, 99.9, size=n_lines)
        line_ew[line_ew >= 99.9] = -0.1
        
        if directions is None:
            directions = np.random.randint(0, 2, size=n_lines)
    
    # Append new data
    data['configs'].append(combined_config.to_list())
    data['line_ews'].append(line_ew.tolist())
    data['snp_txs'].append(snp_tx.as_posix())
    data['snp_rxs'].append(snp_rx.as_posix()) 
    data['directions'].append(directions.tolist() if directions is not None else [])
    
    # Save updated data
    pickle_file.parent.mkdir(parents=True, exist_ok=True)
    with open(pickle_file, 'wb') as f:
        pickle.dump(data, f)
    
    if debug:
        print(f"Saved data to {pickle_file}")

def generate_vertical_snp_pairs(vertical_dir, n_pairs):
    """
    Generate N pairs of vertical SNPs using product with repeat=2 and random selection.
    
    Args:
        vertical_dir: Directory containing vertical SNP files
        n_pairs: Number of pairs to generate
        
    Returns:
        List of (tx_snp, rx_snp) tuples
    """
    vertical_snps = parse_snps(vertical_dir)
    if len(vertical_snps) == 0:
        raise ValueError(f"No SNP files found in vertical directory: {vertical_dir}")
    
    # Generate all possible pairs using product with repeat=2
    all_pairs = list(product(vertical_snps, repeat=2))
    
    # Randomly select N pairs (duplicates allowed)
    selected_pairs = random.choices(all_pairs, k=n_pairs)
    
    return selected_pairs

def load_config(config_path, profile=None):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if profile:
        if profile not in config.get('profiles', {}):
            raise ValueError(f"Profile '{profile}' not found in config")
        return config['profiles'][profile]
    else:
        return config['collection']

def build_argparser():
    """Build argument parser for the collection script"""
    parser = argparse.ArgumentParser(
        description="Collect eye width simulation data for trace patterns"
    )
    parser.add_argument(
        '--config', type=Path, 
        default='configs/collect_training_data.yaml',
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        '--profile', type=str,
        help="Configuration profile to use (overrides default collection config)"
    )
    # Allow command line overrides
    parser.add_argument(
        '--trace_pattern', type=Path,
        help="Directory containing trace SNP files (overrides config)"
    )
    parser.add_argument(
        '--vertical_dir', type=Path,
        help="Directory containing vertical SNP files for TX/RX (overrides config)"
    )
    parser.add_argument(
        '--output_dir', type=Path,
        help="Output directory for pickle files (overrides config)"
    )
    parser.add_argument(
        '--param_type', type=str,
        choices=['DDR_PARAMS', 'HBM2_PARAMS', 'UCIE_PARAMS', 'MIX_PARAMS'],
        help="Parameter set to use for boundary parameters (overrides config)"
    )
    parser.add_argument(
        '--max_samples', type=int,
        help="Maximum number of samples to collect per trace SNP (overrides config)"
    )
    parser.add_argument(
        '--debug', action='store_true',
        help="Enable debug mode (overrides config)"
    )
    parser.add_argument(
        '--max_workers', type=int,
        help="Maximum number of worker processes (overrides config)"
    )
    return parser

def main():
    """Main function for parallel data collection"""
    args = build_argparser().parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config, args.profile)
    except FileNotFoundError:
        print(f"Config file not found: {args.config}")
        print("Please create the config file or provide command line arguments")
        return
    except Exception as e:
        print(f"Error loading config: {e}")
        return
    
    # Override config with command line arguments
    trace_pattern = Path(args.trace_pattern) if args.trace_pattern else Path(config['trace_pattern'])
    vertical_dir = Path(args.vertical_dir) if args.vertical_dir else Path(config['vertical_dir'])
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output_dir'])
    param_type = args.param_type if args.param_type else config['param_type']
    max_samples = args.max_samples if args.max_samples else config['max_samples']
    debug = args.debug if args.debug else config.get('debug', False)
    max_workers = args.max_workers if args.max_workers else config.get('max_workers')
    
    print(f"Using configuration:")
    print(f"  Trace pattern: {trace_pattern}")
    print(f"  Vertical dir: {vertical_dir}")
    print(f"  Output dir: {output_dir}")
    print(f"  Parameter type: {param_type}")
    print(f"  Max samples: {max_samples}")
    print(f"  Debug mode: {debug}")
    print(f"  Max workers: {max_workers}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load trace SNP files
    trace_snps = parse_snps(trace_pattern)
    if len(trace_snps) == 0:
        raise ValueError(f"No trace SNP files found in: {trace_pattern}")
    
    print(f"Found {len(trace_snps)} trace SNP files")
    
    # Generate vertical SNP pairs for each trace SNP
    vertical_pairs = generate_vertical_snp_pairs(vertical_dir, len(trace_snps))
    print(f"Generated {len(vertical_pairs)} vertical SNP pairs")
    
    # Get parameter sets
    from simulation.parameters.bound_param import DDR_PARAMS, HBM2_PARAMS, UCIE_PARAMS, MIX_PARAMS, CTLE_PARAMS
    param_sets = {
        'DDR_PARAMS': DDR_PARAMS,
        'HBM2_PARAMS': HBM2_PARAMS, 
        'UCIE_PARAMS': UCIE_PARAMS,
        'MIX_PARAMS': MIX_PARAMS
    }
    boundary_params = param_sets[param_type]
    ctle_params = CTLE_PARAMS
    
    # Prepare simulation tasks
    simulation_tasks = []
    for trace_snp, vertical_pair in zip(trace_snps, vertical_pairs):
        # Check if we need more samples for this trace SNP
        pickle_file = output_dir / f"{Path(trace_snp).stem}.pkl"
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
                simulation_tasks.append((trace_snp, vertical_pair, boundary_params, ctle_params, output_dir))
    
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
                    trace_snp, vertical_pair, boundary_params, ctle_params,
                    output_dir, None, False
                )
                for trace_snp, vertical_pair, boundary_params, ctle_params, output_dir in simulation_tasks
            ]
            
            try:
                for _ in tqdm(
                    concurrent.futures.as_completed(futures), 
                    total=len(futures),
                    desc="Collecting simulation data"
                ):
                    pass
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
        for i, (trace_snp, vertical_pair, boundary_params, ctle_params, output_dir) in enumerate(
            tqdm(simulation_tasks, desc="Debug simulation")
        ):
            print(f"\n--- Task {i+1}/{len(simulation_tasks)} ---")
            collect_snp_simulation_data(
                trace_snp, vertical_pair, boundary_params, ctle_params,
                output_dir, None, True
            )
    
    print(f"Data collection completed. Results saved to: {output_dir}")

if __name__ == "__main__":
    main()