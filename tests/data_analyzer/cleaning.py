import shutil
import pickle
import numpy as np
from pathlib import Path
from typing import List

from common.pickle_utils import DataWriter, load_pickle_data
from common.parameters import SampleResult as SimulationResult


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

def remove_duplicate_configs(results: List[SimulationResult], precision: int = 8) -> List[SimulationResult]:
    """Remove samples with duplicate configuration values, keeping only the first occurrence."""
    if not results:
        return results
    
    seen_configs = set()
    unique_results = []
    
    for result in results:
        try:
            # Round float values to handle precision issues
            rounded_values = [round(v, precision) if isinstance(v, float) else v for v in result.config_values]
            config_tuple = tuple(rounded_values)
            
            if config_tuple not in seen_configs:
                seen_configs.add(config_tuple)
                unique_results.append(result)
        except (TypeError, ValueError) as e:
            # If config values aren't hashable, keep the result (safer than dropping it)
            print(f"Warning: Could not hash config values for duplicate checking: {e}")
            unique_results.append(result)
            
    return unique_results

def clean_pickle_file_inplace(pfile: Path, block_size: int = None, remove_block_size_1: bool = False, remove_duplicates: bool = False) -> tuple[int, int]:
    """
    Filters samples in a pickle file based on direction block size and/or duplicate configurations.
    The file is overwritten in-place with the cleaned data, preserving the original format.
    A backup of the original file is created with a .bak extension.
    
    Args:
        pfile: Path to the pickle file to clean
        block_size: Only keep samples with this specific direction block size
        remove_block_size_1: Remove samples with block size 1 direction patterns
        remove_duplicates: Remove samples with duplicate configuration values (keeps first occurrence)
    
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
        return 0, 0
    
    if block_size is not None:
        print(f"  Will filter for block_size = {block_size}")
    if remove_block_size_1:
        print(f"  Will remove block_size_1 patterns")
    if remove_duplicates:
        print(f"  Will remove duplicate configurations")

    # Apply duplicate removal first if requested
    if remove_duplicates:
        print(f"  Before duplicate removal: {len(results)} samples")
        # Debug: check how many duplicates we detect
        from .analysis import detect_duplicate_configs
        dup_stats = detect_duplicate_configs(results)
        print(f"  Detected duplicates: {dup_stats['duplicate_count']} samples in {len(dup_stats['duplicate_groups'])} groups")
        results = remove_duplicate_configs(results)
        print(f"  After duplicate removal: {len(results)} samples")
    
    # Filter the list of dataclasses
    valid_results: List[SimulationResult] = []
    for result in results:
        is_valid = True
        
        # Perform checks on the result dataclass
        if remove_block_size_1 and detect_block_size_1_patterns(result.directions):
            is_valid = False
        
        if block_size is not None:
            block_est = estimate_block_size(np.array(result.directions))
            if block_est != block_size:
                is_valid = False
        
        if is_valid:
            valid_results.append(result)

    n_samples_after = len(valid_results)
    num_removed = n_samples_before - n_samples_after

    if num_removed == 0:
        return n_samples_before, 0

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

    return n_samples_before, num_removed
