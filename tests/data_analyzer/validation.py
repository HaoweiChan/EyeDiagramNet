import pickle
import traceback
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from common.pickle_utils import load_pickle_data
from common.param_types import SampleResult as SimulationResult

# Try to import simulation functions for comparison
try:
    from simulation.engine.sbr_simulator import snp_eyewidth_simulation as legacy_snp_eyewidth_simulation
    from common.param_types import SampleResult
    VALIDATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import validation modules: {e}")
    VALIDATION_AVAILABLE = False
    legacy_snp_eyewidth_simulation = None
    SampleResult = dict

def reconstruct_config(data, sample_idx):
    """Reconstruct configuration from pickle data."""
    if 'configs' in data and 'meta' in data and 'config_keys' in data['meta']:
        configs_list = data['configs']
        config_keys = data['meta']['config_keys']
        if sample_idx >= len(configs_list):
            raise ValueError(f"Sample index {sample_idx} out of range.")
        config_values = configs_list[sample_idx]
        if len(config_keys) != len(config_values):
            raise ValueError("Mismatch between config_keys and config_values length.")
        config_dict = dict(zip(config_keys, config_values))
        return SampleResult.from_dict(config_dict) if VALIDATION_AVAILABLE else config_dict
    
    if 'config_dicts' in data and len(data['config_dicts']) > sample_idx:
        config_dict = data['config_dicts'][sample_idx]
        return SampleResult.from_dict(config_dict) if VALIDATION_AVAILABLE else config_dict
    
    raise ValueError(f"No valid config data found for sample {sample_idx}.")

def convert_config_to_legacy_format(config):
    """Convert new config format to legacy format for compatibility."""
    name_mapping = {'L_drv': 'L_tx', 'L_odt': 'L_rx', 'C_drv': 'C_tx', 'C_odt': 'C_rx', 'R_drv': 'R_tx', 'R_odt': 'R_rx'}
    config_dict = config.to_dict() if hasattr(config, 'to_dict') else config
    legacy_dict = {name_mapping.get(k, k): v for k, v in config_dict.items()}
    return SampleResult.from_dict(legacy_dict) if VALIDATION_AVAILABLE and hasattr(config, 'to_dict') else legacy_dict

def get_snp_file_paths(data, sample_idx):
    """Get SNP file paths handling both new and legacy naming conventions."""
    if 'snp_drvs' in data and 'snp_odts' in data:
        snp_drv = Path(data['snp_drvs'][sample_idx])
        snp_odt = Path(data['snp_odts'][sample_idx])
    elif 'snp_txs' in data and 'snp_rxs' in data:
        snp_drv = Path(data['snp_txs'][sample_idx])
        snp_odt = Path(data['snp_rxs'][sample_idx])
    else:
        raise KeyError("No SNP file paths found in data.")
    return snp_drv, snp_odt

def run_validation(pickle_files: list, max_files: int, max_samples: int, output_dir: Path):
    """Runs the validation by comparing pickle data with new simulation results."""
    if not VALIDATION_AVAILABLE:
        print("\nValidation requested but not available due to missing imports.")
        return

    print("\n\n6. VALIDATION: COMPARE PICKLE DATA WITH SIMULATED DATA")
    print("="*40)
    
    num_files_to_validate = min(max_files, len(pickle_files))
    validation_files = random.sample(pickle_files, num_files_to_validate)
    
    detailed_results = []
    for pfile in validation_files:
        print(f"\nValidating {pfile.name}...")
        try:
            results = load_pickle_data(pfile)
            if not results:
                continue

            n_samples_to_validate = min(max_samples, len(results))
            samples_to_validate = random.sample(results, n_samples_to_validate)

            for sample in samples_to_validate:
                try:
                    # Reconstruct config from dataclass
                    config_dict = dict(zip(sample.config_keys, sample.config_values))
                    config = SampleResult.from_dict(config_dict)
                    legacy_config = convert_config_to_legacy_format(config)
                    
                    # Get SNP file paths from dataclass
                    snp_horiz = sample.snp_horiz
                    snp_drv = Path(sample.snp_drv)
                    snp_odt = Path(sample.snp_odt)
                    directions = np.array(sample.directions)
                    pickle_ew = np.array(sample.line_ews)

                    sim_result = legacy_snp_eyewidth_simulation(legacy_config, (snp_horiz, snp_drv, snp_odt), directions)
                    simulated_ew = np.array(sim_result[0] if isinstance(sim_result, tuple) else sim_result)
                    
                    diff = simulated_ew - pickle_ew
                    rel_error = np.abs(diff) / (np.abs(pickle_ew) + 1e-10)
                    
                    detailed_results.append({
                        'file_name': pfile.name,
                        'pickle_ew': pickle_ew,
                        'simulated_ew': simulated_ew,
                        'differences': diff,
                        'relative_errors': rel_error
                    })
                except Exception as e:
                    print(f"    Error in simulation for a sample: {e}")
        except Exception as e:
            print(f"Error processing {pfile.name}: {e}")

    if detailed_results:
        df = pd.DataFrame(detailed_results)
        df.to_csv(output_dir / 'validation_comparison_details.csv', index=False)
        print(f"\nDetailed validation results saved to: {output_dir / 'validation_comparison_details.csv'}")
        # Add plotting function call here if desired
    else:
        print("No validation results generated.")
