import shutil
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict

from common.pickle_utils import DataWriter, load_pickle_data
from common.parameters import SampleResult as SimulationResult
from simulation.parameters.bound_param import PARAM_SETS_MAP


def estimate_block_size(direction_array: np.ndarray) -> int:
    """Estimate the smallest consecutive run length (block size) in a 0/1 array."""
    arr = np.asarray(direction_array).astype(int).flatten()
    if arr.size == 0:
        return 0
    change_idx = np.flatnonzero(np.diff(arr) != 0)
    starts = np.r_[0, change_idx + 1]
    ends = np.r_[change_idx, arr.size - 1]
    run_lengths = ends - starts + 1
    return int(run_lengths.min())


def detect_block_size_1_patterns(directions: list) -> bool:
    """Detect if a direction pattern contains block size 1."""
    if not directions:
        return False
    
    directions = np.array(directions)
    transitions = np.diff(directions)
    change_indices = np.where(transitions != 0)[0]
    all_indices = np.concatenate([[-1], change_indices, [len(directions) - 1]])
    block_lengths = np.diff(all_indices)
    return np.any(block_lengths == 1)

def remove_contaminated_configs(results: List[SimulationResult]) -> List[SimulationResult]:
    """
    Remove samples where config values are actually parameter names (strings).
    These are invalid samples that resulted from the to_list() return order bug.
    """
    if not results:
        return results
    
    valid_results = []
    
    for result in results:
        is_valid = True
        
        # Check if config_values contains strings (should be numeric)
        if result.config_values and isinstance(result.config_values[0], str):
            is_valid = False
        
        # Check if config_values match config_keys (indicating they're swapped)
        if is_valid and result.config_keys and result.config_values:
            if len(result.config_keys) == len(result.config_values):
                # Check if the values are actually the keys
                if all(isinstance(v, str) and v in result.config_keys for v in result.config_values):
                    is_valid = False
        
        if is_valid:
            valid_results.append(result)
    
    return valid_results

def remove_duplicate_configs(results: List[SimulationResult]) -> List[SimulationResult]:
    """Remove samples with duplicate configuration values, keeping only the first occurrence."""
    if not results:
        return results
    
    seen_configs = set()
    unique_results = []
    
    for result in results:
        # Create a tuple of config values for comparison
        config_tuple = tuple(result.config_values)
        
        if config_tuple not in seen_configs:
            seen_configs.add(config_tuple)
            unique_results.append(result)
    
    return unique_results

def validate_boundary_parameters(result: SimulationResult, param_set_names: List[str]) -> tuple[bool, List[str]]:
    """
    Validate that boundary parameters in a sample are within the parameter set ranges.
    
    Args:
        result: SimulationResult dataclass containing config_keys, config_values, and param_types
        param_set_names: List of parameter set names from the sample's param_types
        
    Returns:
        tuple: (is_valid, list of out-of-range parameter descriptions)
    """
    if not param_set_names:
        return True, []
    
    out_of_range_params = []
    
    # Combine all parameter sets for validation
    combined_param_defs = {}
    for param_set_name in param_set_names:
        if param_set_name not in PARAM_SETS_MAP:
            print(f"Warning: Parameter set '{param_set_name}' not found in PARAM_SETS_MAP. Available sets: {list(PARAM_SETS_MAP.keys())}")
            continue
        
        param_set = PARAM_SETS_MAP[param_set_name]
        # Collect parameter definitions from this set
        if hasattr(param_set, 'params'):
            combined_param_defs.update(param_set.params)
        elif hasattr(param_set, '_params'):
            combined_param_defs.update(param_set._params)
    
    # Convert config_keys and config_values to a dict
    config_dict = dict(zip(result.config_keys, result.config_values))
    
    # For each parameter in the config, check if it's within the param_set bounds
    for param_name, param_value in config_dict.items():
        # Skip if parameter is not defined in the combined param sets
        if param_name not in combined_param_defs:
            continue
        
        param_def = combined_param_defs[param_name]
        
        # Get bounds based on parameter type
        if hasattr(param_def, 'low') and hasattr(param_def, 'high'):
            # LinearParameter
            low = param_def.low
            high = param_def.high
            scaler = getattr(param_def, 'scaler', 1.0)
            
            # Apply scaler to bounds for comparison
            low_scaled = low * scaler
            high_scaled = high * scaler
            
            # Check if value is within bounds
            if not (low_scaled <= param_value <= high_scaled):
                out_of_range_params.append(f"{param_name}={param_value:.6e} (range: [{low_scaled:.6e}, {high_scaled:.6e}])")
                
        elif hasattr(param_def, 'values'):
            # DiscreteParameter
            values = param_def.values
            if param_value not in values:
                out_of_range_params.append(f"{param_name}={param_value} (allowed: {values})")
    
    is_valid = len(out_of_range_params) == 0
    return is_valid, out_of_range_params

def clean_pickle_file_inplace(pfile: Path, block_size: int = None, remove_block_size_1: bool = False, 
                             remove_duplicates: bool = False, remove_contaminated: bool = True, 
                             remove_legacy: bool = False) -> tuple[int, int, Dict]:
    """
    Filters samples in a pickle file based on direction block size and/or duplicate configurations.
    The file is overwritten in-place with the cleaned data, preserving the original format.
    A backup of the original file is created with a .bak extension.
    
    Args:
        pfile: Path to the pickle file to clean
        block_size: Only keep samples with this specific direction block size
        remove_block_size_1: Remove samples with block size 1 direction patterns
        remove_duplicates: Remove samples with duplicate configuration values (keeps first occurrence)
        remove_contaminated: Remove samples where config values are strings (DEFAULT: True)
        remove_legacy: Remove samples using legacy parameter naming (snp_tx/snp_rx instead of snp_drv/snp_odt)
    
    Returns:
        tuple: (n_samples_before, num_removed, stats_dict) where stats_dict contains:
            - 'out_of_range_count': number of samples outside param set ranges
            - 'out_of_range_details': list of (sample_idx, out_of_range_params) tuples
            - 'legacy_count': number of legacy format samples removed (if remove_legacy=True)
    
    Note: If the input file uses legacy format (snp_txs/snp_rxs), it will be 
    automatically converted to the new format (snp_drvs/snp_odts) when saved.
    """
    # Check if the original file uses legacy format before loading
    legacy_format_detected = False
    try:
        with open(pfile, 'rb') as f:
            raw_data = pickle.load(f)
        if isinstance(raw_data, dict) and 'snp_txs' in raw_data and 'snp_rxs' in raw_data:
            legacy_format_detected = True
    except Exception:
        pass  # Will be handled by load_pickle_data
    
    # Load data using the standardized dataclass loader
    results = load_pickle_data(pfile)
    n_samples_before = len(results)

    if n_samples_before == 0:
        return 0, 0, {'out_of_range_count': 0, 'out_of_range_details': [], 'legacy_count': 0}

    # Track out-of-range and legacy statistics
    out_of_range_details = []
    legacy_count = 0
    
    # Apply contamination removal first (CRITICAL - these are invalid samples)
    if remove_contaminated:
        results = remove_contaminated_configs(results)
    
    # Apply duplicate removal if requested
    if remove_duplicates:
        results = remove_duplicate_configs(results)
    
    # Remove legacy format samples if requested
    if remove_legacy:
        non_legacy_results = []
        for result in results:
            # Check if this sample uses legacy naming by examining the stored SNP paths
            # Legacy samples will have keys like 'R_tx', 'R_rx' in config_keys
            # or the snp_drv/snp_odt paths might contain 'tx'/'rx' indicators
            is_legacy = False
            if hasattr(result, 'config_keys') and result.config_keys:
                # Check for legacy parameter names
                legacy_params = {'R_tx', 'R_rx', 'C_tx', 'C_rx', 'L_tx', 'L_rx'}
                if any(param in legacy_params for param in result.config_keys):
                    is_legacy = True
            
            if not is_legacy:
                non_legacy_results.append(result)
            else:
                legacy_count += 1
        
        results = non_legacy_results
    
    # Filter the list of dataclasses
    valid_results: List[SimulationResult] = []
    for idx, result in enumerate(results):
        is_valid = True
        
        # Perform checks on the result dataclass
        if remove_block_size_1 and detect_block_size_1_patterns(result.directions):
            is_valid = False
        
        if block_size is not None:
            block_est = estimate_block_size(np.array(result.directions))
            if block_est != block_size:
                is_valid = False
        
        # Validate boundary parameters using param_types from the result itself
        if is_valid and hasattr(result, 'param_types') and result.param_types:
            param_valid, out_of_range_params = validate_boundary_parameters(result, result.param_types)
            if not param_valid:
                is_valid = False
                out_of_range_details.append((idx, result.param_types, out_of_range_params))
        
        if is_valid:
            valid_results.append(result)

    n_samples_after = len(valid_results)
    num_removed = n_samples_before - n_samples_after
    
    stats = {
        'out_of_range_count': len(out_of_range_details),
        'out_of_range_details': out_of_range_details,
        'legacy_count': legacy_count
    }

    if num_removed == 0:
        return n_samples_before, 0, stats

    # Backup the original file
    backup_path = pfile.with_suffix(pfile.suffix + '.bak')
    try:
        shutil.copy2(pfile, backup_path)
    except Exception as e:
        print(f"Warning: Could not create backup for {pfile.name}: {e}")

    # Use the DataWriter to save the cleaned data back in the standard format
    data_writer = DataWriter(pfile)
    # The writer starts with a clean slate by re-initializing from the (now backed-up) file,
    # so we need to clear its internal data before adding our cleaned results.
    data_writer.data = {
        'configs': [], 'line_ews': [], 'snp_drvs': [], 'snp_odts': [],
        'directions': [], 'meta': {}
    }
    
    for result in valid_results:
        data_writer.add_result(result)
    
    data_writer.save()
    
    # Report legacy format conversion if applicable
    if legacy_format_detected:
        print(f"Note: Converted {pfile.name} from legacy format (snp_txs/snp_rxs) to new format (snp_drvs/snp_odts)")

    return n_samples_before, num_removed, stats


