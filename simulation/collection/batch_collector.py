import yaml
import json
import random
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

from common.signal_utils import read_snp
from common.parameters import ParameterSet, SampleResult
from common.pickle_utils import DataWriter, SimulationResult

from simulation.parameters.bound_param import PARAM_SETS_MAP
from simulation.engine.sbr_simulator import snp_eyewidth_simulation
from simulation.io.config_utils import resolve_trace_pattern
from simulation.io.snp_utils import generate_vertical_snp_pairs
from simulation.io.direction_utils import generate_directions

class EyeWidthSimulatePipeline:
    def __init__(self, config_path, debug=False):
        self.config_path = Path(config_path)
        self.debug = debug
        self.cfg = self._load_config()

    def _load_config(self):
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def _load_boundary_params(self, boundary_path):
        """Load boundary parameters and directions from a JSON file."""
        loaded = json.loads(boundary_path.read_text())
        if "directions" in loaded:
            directions = np.asarray(loaded["directions"], dtype=int)
        else:
            # Default to ones if not provided, based on number of lines in boundary values
            n_lines = len(next(iter(loaded.get("boundary", {}))).values()) // 2
            directions = np.ones(n_lines, dtype=int)
        
        boundary_params = SampleResult(**loaded.get("boundary", {}))
        return directions, boundary_params

    def run_single_trace(self, trace_file_path):
        """
        Process a single trace file by randomly sampling boundary files and vertical SNP pairs.
        """
        trace_file = Path(trace_file_path)
        
        # Resolve paths from the structured config
        output_dir = Path(self.cfg['data']['output_dir'])
        boundary_dir = Path(self.cfg['boundary']['input_dir'])
        vertical_dirs = self.cfg['dataset']['vertical_dataset']
        
        # Get simulation settings from config
        max_samples = self.cfg['boundary']['max_samples']
        enable_direction = self.cfg['boundary'].get('enable_direction', False)
        param_type_key = self.cfg['boundary'].get('param_type')

        if not param_type_key:
            raise ValueError("'param_type' must be specified in the boundary config for validation.")

        master_param_set = PARAM_SETS_MAP.get(param_type_key)
        if not master_param_set:
            raise ValueError(f"param_type '{param_type_key}' not found in PARAM_SETS_MAP.")

        output_dir.mkdir(parents=True, exist_ok=True)

        boundary_files = sorted(list(boundary_dir.glob("*.json")))
        if not boundary_files:
            print(f"No boundary files (.json) found in {boundary_dir}")
            return

        # Discover vertical SNP pairs
        vertical_pairs = generate_vertical_snp_pairs(vertical_dirs, 1, [trace_file], output_dir, trace_file.stem)
        if not vertical_pairs:
            print(f"No vertical SNP pairs found for {trace_file.name}")
            return

        print(f"Processing trace file: {trace_file.name}")
        print(f"Found {len(boundary_files)} boundary files.")
        print(f"Found {len(vertical_pairs)} vertical SNP pairs.")
        print(f"Generating {max_samples} random samples.")

        output_file = output_dir / f"{trace_file.stem}.pkl"
        data_writer = DataWriter(output_file)
        
        trace_ntwk = read_snp(trace_file)
        n_lines = trace_ntwk.nports // 2
        
        # Main sampling loop: for each sample, randomly pick a boundary and a vertical pair
        for i in tqdm(range(max_samples), desc=f"Sampling for {trace_file.stem}"):
            try:
                # Randomly select a boundary file and a vertical pair for each sample
                boundary_path = random.choice(boundary_files)
                drv_path, odt_path = random.choice(vertical_pairs)

                directions, boundary_params = self._load_boundary_params(boundary_path)
                horiz_params = ParameterSet(**boundary_params.to_dict())

                # Sample a configuration from the chosen boundary
                config = horiz_params.sample()
                if config is None:
                    print(f"Warning: Failed to sample a valid config for {boundary_path.name}. Skipping sample {i+1}.")
                    continue

                # --- VALIDATION LOGIC ---
                is_valid = True
                config_dict = config.to_dict()
                
                # 1. Key Validation
                master_keys = set(master_param_set.to_dict().keys())
                sample_keys = set(config_dict.keys())
                if master_keys != sample_keys:
                    print(f"Warning: Key mismatch for sample {i+1} from {boundary_path.name}. Skipping.")
                    print(f"  Missing keys: {master_keys - sample_keys}")
                    print(f"  Extra keys: {sample_keys - master_keys}")
                    is_valid = False
                    continue

                # 2. Range Validation
                for key, param in master_param_set.params.items():
                    sample_value = config_dict[key]
                    if not (param.min <= sample_value <= param.max):
                        print(f"Warning: Value out of range for '{key}' in sample {i+1} from {boundary_path.name}. Skipping.")
                        print(f"  Value: {sample_value}, Range: [{param.min}, {param.max}]")
                        is_valid = False
                        break 
                
                if not is_valid:
                    continue
                # --- END VALIDATION ---

                if self.debug:
                    print(f"\n--- Sample {i+1}/{max_samples} ---")
                    print(f"  Boundary: {boundary_path.name}")
                    print(f"  Vertical Pair: {drv_path.name} / {odt_path.name}")
                
                # Generate directions for each sample
                sim_directions = generate_directions(n_lines, enable_direction)

                # Prepare SNP files tuple for simulation
                snp_files_tuple = (trace_ntwk, read_snp(drv_path), read_snp(odt_path))

                # Simulate eye width
                line_ew = snp_eyewidth_simulation(
                    config,
                    snp_files_tuple,
                    sim_directions,
                )
                line_ew[line_ew >= 99.9] = -0.1

                # Create a structured dataclass instance for the result
                config_keys, config_values = config.to_list(return_keys=True)
                result_to_add = SimulationResult(
                    config_values=config_values,
                    config_keys=config_keys,
                    line_ews=line_ew.tolist(),
                    snp_drv=drv_path.as_posix(),
                    snp_odt=odt_path.as_posix(),
                    directions=sim_directions.tolist(),
                    snp_horiz=str(trace_file),
                    n_ports=n_lines * 2,
                    param_types=self.cfg['boundary'].get('param_type', '').split(',')
                )
                data_writer.add_result(result_to_add)

            except Exception as e:
                # Add more context to the error message
                boundary_name = boundary_path.name if 'boundary_path' in locals() else "N/A"
                vertical_name = f"{drv_path.name}/{odt_path.name}" if 'drv_path' in locals() else "N/A"
                print(f"ERROR: Failed simulation for sample {i+1} with boundary {boundary_name}, vertical pair {vertical_name}")
                print(f"Error details: {e}")
                if self.debug:
                    import traceback
                    traceback.print_exc()

        # Save the final results
        data_writer.save()
        print(f"\nSaved {data_writer.get_sample_count()} total samples to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Run SNP eye-width simulation for a single trace against multiple boundaries.")
    parser.add_argument(
        '--config', type=Path, required=True,
        help="YAML config file with paths for boundary_dir, output_dir, drv_snp, and odt_snp."
    )
    parser.add_argument(
        '--trace-key', type=str,
        help="Key for the trace pattern to use from the horizontal_dataset in the config. Overrides trace_file."
    )
    parser.add_argument(
        '--trace-file', type=Path,
        help="Path to the single trace s-parameter file to process. Used if trace-key is not provided."
    )
    parser.add_argument(
        '--debug', action='store_true',
        help="Enable debug mode for more verbose output."
    )
    args = parser.parse_args()
    
    # Load config to resolve trace file path if a key is provided
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if args.trace_key:
        # Resolve the trace pattern to a single file path string
        trace_patterns = config['dataset']['horizontal_dataset']
        trace_path_str = resolve_trace_pattern(args.trace_key, trace_patterns)
        
        # Since resolve_trace_pattern can return a glob pattern, we find the first match
        trace_file_to_process = next(Path().glob(trace_path_str), None)
        
        if not trace_file_to_process:
             raise FileNotFoundError(f"No file found for trace key '{args.trace_key}' with pattern '{trace_path_str}'")

    elif args.trace_file:
        trace_file_to_process = args.trace_file
    else:
        raise ValueError("Either --trace-key or --trace-file must be provided.")

    simulator = EyeWidthSimulatePipeline(
        args.config,
        debug=args.debug
    )
    simulator.run_single_trace(trace_file_to_process)

if __name__ == "__main__":
    main() 