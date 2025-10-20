import random
import numpy as np
import pandas as pd
from pathlib import Path

from common.pickle_utils import load_pickle_data


# Try to import simulation functions for comparison
try:
    # First try to import from legacy module, fall back to current module
    try:
        from simulation.engine.sparam_to_ew import snp_eyewidth_simulation as legacy_snp_eyewidth_simulation
        print("Using legacy simulation from sparam_to_ew module")
        USE_LEGACY_FORMAT = True  # Legacy module expects legacy parameter format
    except ImportError:
        from simulation.engine.sbr_simulator import snp_eyewidth_simulation as legacy_snp_eyewidth_simulation
        print("Using current simulation from sbr_simulator module")
        USE_LEGACY_FORMAT = False  # Current module expects new parameter format
    
    from common.parameters import SampleResult
    VALIDATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import validation modules: {e}")
    VALIDATION_AVAILABLE = False
    legacy_snp_eyewidth_simulation = None
    SampleResult = dict
    USE_LEGACY_FORMAT = False


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
    # NOTE: This function is now deprecated as the simulation code expects new format
    # Keeping for backward compatibility but should not be used
    from common.parameters import convert_legacy_param_names
    config_dict = config.to_dict() if hasattr(config, 'to_dict') else config
    legacy_dict = convert_legacy_param_names(config_dict, target_format='legacy')
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
                    
                    # Choose parameter format based on which simulation module is being used
                    if USE_LEGACY_FORMAT:
                        # Convert to legacy format for legacy simulation
                        simulation_config = convert_config_to_legacy_format(config)
                    else:
                        # Use new format for current simulation
                        simulation_config = config
                    
                    # Get SNP file paths from dataclass
                    snp_horiz = sample.snp_horiz
                    snp_drv = Path(sample.snp_drv)
                    snp_odt = Path(sample.snp_odt)
                    directions = np.array(sample.directions)
                    pickle_ew = np.array(sample.line_ews)

                    sim_result = legacy_snp_eyewidth_simulation(simulation_config, (snp_horiz, snp_drv, snp_odt), directions)
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
