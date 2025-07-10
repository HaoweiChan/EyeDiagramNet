import json
import torch
import skrf as rf
import numpy as np
from pathlib import Path, PosixPath
from itertools import combinations

def read_snp(snp_file: PosixPath):
    """Read SNP file with robust error handling for file I/O issues."""
    # Convert to Path object to ensure we have proper path handling
    snp_path = Path(snp_file) if not isinstance(snp_file, Path) else snp_file
    
    # Validate file existence and readability
    if not snp_path.exists():
        raise FileNotFoundError(f"SNP file not found: {snp_path}")
    
    if not snp_path.is_file():
        raise ValueError(f"Path is not a file: {snp_path}")
    
    try:
        if snp_path.suffix != '.npz':
            # Use more robust file reading with explicit error handling
            # Convert to absolute path to avoid any relative path issues
            abs_path = snp_path.resolve()
            return rf.Network(str(abs_path))
        
        # Handle .npz files
        data = np.load(snp_path)
        compress_arr = data['compress_arr']
        diag = data['diag']

        decompress_arr = np.repeat(compress_arr, 2, axis=0)
        if diag.shape[0] != compress_arr.shape[0]:
            decompress_arr = np.delete(decompress_arr, -1, axis=0)
        decompress_arr[:,:2] = np.triu(decompress_arr[:,:2]) + np.transpose(np.triu(decompress_arr[:,:2], 1), (0, 2, 1))
        decompress_arr[1::2] = np.tril(decompress_arr[1::2], -1) + np.transpose(np.tril(decompress_arr[1::2], -1), (0, 2, 1))
        decompress_arr[1::2, np.arange(diag.shape[1]), np.arange(diag.shape[1])] = diag

        ntwk = rf.Network()
        ntwk.s = decompress_arr
        if ntwk.s is not None and ntwk.s.ndim > 0:
            npoints = ntwk.s.shape[0]
            ntwk.frequency = rf.Frequency(1, npoints, npoints, unit='GHz')

        return ntwk
    
    except Exception as e:
        # Provide more detailed error information for debugging
        error_msg = f"Error reading SNP file {snp_path}: {str(e)}"
        print(f"[ERROR] {error_msg}")
        print(f"[ERROR] File size: {snp_path.stat().st_size if snp_path.exists() else 'N/A'} bytes")
        print(f"[ERROR] File suffix: {snp_path.suffix}")
        
        # Re-raise with more context
        raise RuntimeError(error_msg) from e

def parse_snps(snp_dir, indices):
    suffix = '*.s*p'
    if len(list(snp_dir.glob("*.npz"))):
        # suffix = '*.npz'
        pass
    # filter using file indices
    all_snp_files = [None] * len(indices)
    for f in snp_dir.glob(suffix):
        idx = int(f.stem.split('_')[-1])
        idx = np.where(indices == idx)[0]
        if len(idx):
            all_snp_files[idx[0]] = f

    missing_idx = [i for i, x in enumerate(all_snp_files) if x is None]
    missing_idx = sorted(missing_idx, reverse=True)
    for i in missing_idx:
        all_snp_files.pop(i)

    return all_snp_files, missing_idx

def flip_snp(s_matrix: torch.Tensor) -> torch.Tensor:
    """
    Invert the ports of a network's scattering parameter matrix (s-matrix), 'flipping' it over left and right.
    In case the network is a 2n-port and n > 1, the 'second' numbering scheme is
    assumed to be consistent with the ** cascade operator.

    Parameters
    ----------
    s_matrix : torch.Tensor
        Scattering parameter matrix. Shape should be `2n x 2n`, or `f x 2n x 2n`.

    Returns
    -------
    flipped_s_matrix : torch.Tensor
        Flipped scattering parameter matrix.

    See Also
    --------
    renumber
    """
    # Clone the input tensor to create a new tensor for the flipped s-matrix
    flipped_s_matrix = s_matrix.clone()

    # Get the number of columns and rows (should be equal and even)
    num_cols = s_matrix.shape[-1]
    num_rows = s_matrix.shape[-2]
    n = num_cols // 2  # Number of ports n (since total ports = 2n)

    # Check if the matrix is square and has an even dimension
    if (num_cols == num_rows) and (num_cols % 2 == 0):
        # Create index tensors for old and new port ordering
        old_indices = torch.arange(0, 2 * n)
        new_indices = torch.cat((torch.arange(n, 2 * n), torch.arange(0, n)))

        if s_matrix.dim() == 2:
            # For 2D tensors (single s-matrix)
            # Renumber rows and columns according to the new indices
            flipped_s_matrix[new_indices, :] = flipped_s_matrix[old_indices, :]  # Renumber rows
            flipped_s_matrix[:, new_indices] = flipped_s_matrix[:, old_indices]  # Renumber columns
        else:
            # For higher-dimensional tensors (e.g., frequency x ports x ports)
            # Use ellipsis to handle any leading dimensions
            flipped_s_matrix[..., new_indices, :] = flipped_s_matrix[..., old_indices, :]  # Renumber rows
            flipped_s_matrix[..., new_indices] = flipped_s_matrix[..., :, old_indices]  # Renumber columns
    else:
        raise IndexError('Matrices should be 2n x 2n, or f x 2n x 2n')

    return flipped_s_matrix

def renumber_snp(s_matrix: torch.Tensor) -> torch.Tensor:
    """
    Renumber the ports of a network's scattering parameter matrix (s-matrix),
    transforming the port order from (1, 2, ..., n) to (1, n/2+1, 2, n/2+2, ..., n/2, n).

    Parameters
    ----------
    s_matrix : torch.Tensor
        Scattering parameter matrix. Shape should be `n x n`, or `f x n x n`.

    Returns
    -------
    renumbered_s_matrix : torch.Tensor
        Renumbered scattering parameter matrix.

    Raises
    ------
    ValueError
        If the number of ports `n` is not even.

    See Also
    --------
    flip
    """
    # Clone the input tensor to create a new tensor for the renumbered s-matrix
    renumbered_s_matrix = s_matrix.clone()

    # Get the number of ports (should be even)
    num_ports = s_matrix.shape[-1]
    num_rows = s_matrix.shape[-2]

    # Check if the matrix is square and has an even number of ports
    if (num_ports == num_rows) and (num_ports % 2 == 0):
        n = num_ports  # Total number of ports
        half_n = n // 2  # Half the number of ports

        # Generate the original and new port indices
        # Original indices: [0, 1, 2, ..., n-1]
        original_indices = torch.arange(n)

        # New indices: [0, half_n, 1, half_n+1, ..., half_n-1, n-1]
        new_indices = torch.empty(n, dtype=torch.long)
        new_indices[0:2] = torch.arange(0, half_n)
        new_indices[1::2] = torch.arange(half_n, n)

        if s_matrix.dim() == 2:
            # For 2D tensors (single s-matrix)
            # Renumber rows and columns according to the new indices
            renumbered_s_matrix[new_indices, :] = renumbered_s_matrix[original_indices, :]  # Renumber rows
            renumbered_s_matrix[:, new_indices] = renumbered_s_matrix[:, original_indices]  # Renumber columns
        else:
            # For higher-dimensional tensors (e.g., frequency x ports x ports)
            # Use ellipsis to handle any leading dimensions
            renumbered_s_matrix[..., new_indices, :] = renumbered_s_matrix[..., original_indices, :]  # Renumber rows
            renumbered_s_matrix[..., new_indices] = renumbered_s_matrix[..., :, original_indices]  # Renumber columns
    else:
        raise ValueError('Number of ports must be even and s-matrix must be square.')

    return renumbered_s_matrix

def greedy_covering_design(total_elements, group_size, window_size=None):
    if window_size is None:
        window_size = total_elements
    cache_folder = Path("common/greedy_covering_design")
    cache_folder.mkdir(parents=True, exist_ok=True)
    cache_file = cache_folder / f"{total_elements}_{group_size}_{window_size}.json"

    if cache_file.exists():
        with cache_file.open('r') as f:
            return np.array(json.load(f))

    def generate_pairs_within_window(total_elements, window_size):
        all_pairs = []
        for i in range(total_elements - window_size + 1):
            lines = list(range(i, i + window_size))
            pairs = list(combinations(lines, 2))
            all_pairs.extend(pairs)
        return set(all_pairs)

    def generate_valid_groups(total_elements, group_size, window_size):
        valid_groups = []
        for i in range(total_elements - window_size + 1):
            for group in combinations(range(i, i + window_size), group_size):
                valid_groups.append(group)
        return valid_groups

    window_size = max(window_size, group_size)
    all_pairs = generate_pairs_within_window(total_elements, window_size)
    valid_groups = generate_valid_groups(total_elements, group_size, window_size)

    selected_groups = []
    covered_pairs = set()

    while all_pairs - covered_pairs:
        best_group = None
        best_covered = set()
        for group in valid_groups:
            covered = {pair for pair in combinations(group, 2) if pair in all_pairs}
            new_covered = covered - covered_pairs
            if len(new_covered) > len(best_covered):
                best_group = group
                best_covered = new_covered

        if not best_group:
            break

        selected_groups.append(best_group)
        covered_pairs.update(best_covered)

    selected_groups = np.array(selected_groups)
    array1 = 2 * selected_groups
    array2 = 2 * selected_groups + 1

    interleaved_groups = np.empty(array1.size + array2.size, dtype=array1.dtype)
    interleaved_groups[0::2] = array1.flatten()
    interleaved_groups[1::2] = array2.flatten()
    interleaved_groups = interleaved_groups.reshape((len(selected_groups), group_size * 2))

    with open(cache_file, 'w') as f:
        json.dump(interleaved_groups.tolist(), f)

    return interleaved_groups

def read_boundary(bound_path):
    """
    Legacy function to read boundary data from JSON file.
    Returns directions and boundary in the format expected by old code.
    """
    import json
    from simulation.parameters.bound_param import SampleResult
    
    with open(bound_path, 'r') as f:
        loaded = json.load(f)
        directions = np.array(loaded['directions']) if 'directions' in loaded else None
        ctle = loaded.get('CTLE', {"AC_gain": np.nan, "DC_gain": np.nan, "fp1": np.nan, "fp2": np.nan})
        boundary = loaded['boundary'] | ctle
        boundary_obj = SampleResult(**boundary)
        
    return directions, boundary_obj 