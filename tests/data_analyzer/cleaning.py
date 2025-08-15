import shutil
import pickle
import numpy as np
from pathlib import Path

# Try to import direction_utils, handle potential ImportError
try:
    from simulation.io.direction_utils import get_valid_block_sizes
except ImportError:
    print("Warning: Could not import get_valid_block_sizes from simulation.io.direction_utils.")
    print("Direction validation might not work as expected.")
    def get_valid_block_sizes(n_lines):
        # Fallback function if import fails
        return {1} # Default to a safe value

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

def clean_pickle_file_inplace(pfile: Path, block_size: int = None, remove_block_size_1: bool = False) -> tuple[int, int]:
    """Remove rows with invalid directions, non-numeric eye-widths, or non-matching block sizes.

    Returns: (num_samples_before, num_removed)
    Creates a .bak backup next to the original before overwriting.
    """
    with open(pfile, 'rb') as f:
        data = pickle.load(f)

    # Determine number of samples and check for data consistency
    list_keys = [k for k, v in data.items() if isinstance(v, list)]
    if not list_keys:
        return 0, 0
    
    n_samples = len(data[list_keys[0]])
    if n_samples == 0:
        return 0, 0

    for key in list_keys:
        if len(data[key]) != n_samples:
            print(f"  - Warning: inconsistent list lengths in {pfile.name}, skipping.")
            return n_samples, 0

    invalid_indices = set()

    # Check for invalid direction block sizes and non-matching block sizes
    if 'directions' in data:
        for i, dir_arr in enumerate(data['directions']):
            arr = np.asarray(dir_arr).astype(int).flatten()
            if arr.size == 0:
                invalid_indices.add(i)
                continue
            
            # Check for block size 1 contamination
            if remove_block_size_1 and detect_block_size_1_patterns(dir_arr):
                invalid_indices.add(i)
                continue # Skip other checks if already marked for removal

            block_est = estimate_block_size(arr)
            valid_sizes = get_valid_block_sizes(arr.size)
            if block_est not in valid_sizes:
                invalid_indices.add(i)
            
            if block_size is not None and block_est != block_size:
                invalid_indices.add(i)

    # Check for non-numeric eye width data
    if 'line_ews' in data:
        for i, ew_val in enumerate(data['line_ews']):
            if not isinstance(ew_val, (list, np.ndarray)) or \
               (ew_val and isinstance(ew_val[0], str)):
                invalid_indices.add(i)

    num_removed = len(invalid_indices)
    if num_removed == 0:
        return n_samples, 0

    valid_indices = [i for i in range(n_samples) if i not in invalid_indices]

    # Filter all list-based data
    for key in list_keys:
        data[key] = [data[key][i] for i in valid_indices]

    # Backup and overwrite
    backup_path = pfile.with_suffix(pfile.suffix + '.bak')
    try:
        shutil.copy2(pfile, backup_path)
    except Exception:
        pass
    with open(pfile, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    return n_samples, num_removed
