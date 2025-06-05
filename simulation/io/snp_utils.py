"""SNP file utilities for training data collection."""

import random
import numpy as np
from pathlib import Path
from itertools import product
from skrf.frequency import Frequency
from skrf.network import Network as SkrfNetwork

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

def generate_thru_snp(reference_trace_snp_file, base_output_dir, trace_pattern_key):
    """
    Generates or retrieves an existing auto-generated thru SNP file.
    The thru SNP connects port i to port i + n_ports // 2.
    It is saved in base_output_dir / trace_pattern_key / auto_thru_{N}port.s{N}p.

    Args:
        reference_trace_snp_file: Path to a sample trace SNP file to determine port count.
        base_output_dir: The main output directory.
        trace_pattern_key: The key for the specific trace pattern (used for subdirectory).

    Returns:
        Path to the generated or existing thru SNP file.
    """
    try:
        ref_net = SkrfNetwork(str(reference_trace_snp_file))
        n_ports = ref_net.nports
        if n_ports % 2 != 0:
            # Try to infer from filename if skrf fails for some minimal files
            filename_stem = Path(reference_trace_snp_file).stem
            if 's' in filename_stem and 'p' in filename_stem:
                try:
                    n_ports = int(filename_stem.split('s')[1].split('p')[0])
                    if n_ports % 2 != 0:
                        raise ValueError(
                            f"Number of ports ({n_ports}) derived from filename {filename_stem} must be even for thru connection."
                        )
                except:
                     raise ValueError(
                        f"Could not determine n_ports from {reference_trace_snp_file} and filename {filename_stem} is not standard format (e.g. name.sXp)"
                    )
            else:
                raise ValueError(
                    f"Number of ports ({n_ports}) determined by skrf from {reference_trace_snp_file} must be even for thru connection, and filename {filename_stem} is not standard format."
                )

    except Exception as e:
        print(f"Error reading reference SNP {reference_trace_snp_file} with skrf: {e}")
        # Fallback or re-raise as appropriate for your use case
        # For now, let's try to guess from filename if possible, or assume 4 if not
        print("Attempting to infer port count from filename or defaulting to 4 ports for thru SNP generation.")
        filename_stem = Path(reference_trace_snp_file).stem
        try:
            n_ports = int(filename_stem.split('s')[1].split('p')[0])
            if n_ports % 2 != 0: 
                print(f"Inferred odd n_ports={n_ports} from {filename_stem}, defaulting to 4.")
                n_ports = 4 # Default to 4 if odd number inferred
        except:
            print(f"Could not infer n_ports from {filename_stem}, defaulting to 4.")
            n_ports = 4

    thru_snp_dir = Path(base_output_dir) / trace_pattern_key
    thru_snp_dir.mkdir(parents=True, exist_ok=True)
    thru_snp_filename = f"auto_thru_{n_ports}port.s{n_ports}p"
    thru_snp_path = thru_snp_dir / thru_snp_filename

    if thru_snp_path.exists():
        print(f"Using existing auto-generated thru SNP: {thru_snp_path}")
        return thru_snp_path

    print(f"Generating auto-thru SNP with {n_ports} ports: {thru_snp_path}")

    # A single frequency point is sufficient for a perfect frequency-independent thru
    # Using a common microwave frequency like 1 GHz.
    freq = Frequency(1, 1, 1, unit='ghz') 
    s_matrix = np.zeros((1, n_ports, n_ports), dtype=complex)

    for i in range(n_ports // 2):
        j = i + n_ports // 2
        s_matrix[0, i, j] = 1.0  # Transmission from port i to port j (S_ji)
        s_matrix[0, j, i] = 1.0  # Transmission from port j to port i (S_ij)

    z0_array = np.full(n_ports, 50) # Standard 50 Ohm impedance for all ports

    try:
        thru_net = SkrfNetwork(frequency=freq.f_scaled, s=s_matrix, z0=z0_array) # Use freq.f_scaled for raw numpy array
        thru_net.name = f'auto_thru_{n_ports}port'
        thru_net.write_touchstone(filename=str(thru_snp_path), write_z0=True)
        print(f"Successfully generated auto-thru SNP: {thru_snp_path}")
    except Exception as e:
        print(f"Error writing touchstone file {thru_snp_path}: {e}")
        # Fallback: create a dummy file to prevent repeated generation attempts in one run
        with open(thru_snp_path, 'w') as f:
            f.write(f"# Dummy file due to generation error: {e}\n")
        print(f"Created dummy file at {thru_snp_path} after generation error.")
        # Depending on requirements, you might want to re-raise the exception or handle it differently

    return thru_snp_path

def generate_vertical_snp_pairs(vertical_dirs, n_pairs, trace_snps=None, base_output_dir=None, trace_pattern_key=None):
    """
    Generate N pairs of vertical SNPs using product with repeat=2 and random selection.
    
    Args:
        vertical_dirs: List of directories containing vertical SNP files (None for auto-generated thru SNPs)
        n_pairs: Number of pairs to generate
        trace_snps: List of trace SNP files (needed when vertical_dirs is None)
        base_output_dir: Base output directory (needed for auto-generated thru SNPs)
        trace_pattern_key: Trace pattern key (needed for auto-generated thru SNPs)
        
    Returns:
        List of (tx_snp, rx_snp) tuples
    """
    if vertical_dirs is None:
        if trace_snps is None or len(trace_snps) == 0:
            raise ValueError("trace_snps must be provided and non-empty when vertical_dirs is None")
        if base_output_dir is None or trace_pattern_key is None:
            raise ValueError("base_output_dir and trace_pattern_key must be provided for auto-thru SNP generation")

        # Generate one thru SNP to be used for all pairs
        # Use the first trace_snp as a reference for port count
        thru_snp_path = generate_thru_snp(trace_snps[0], base_output_dir, trace_pattern_key)
        
        # All pairs will use this single thru SNP for both TX and RX
        selected_pairs = [(thru_snp_path, thru_snp_path)] * n_pairs
        
        return selected_pairs
    else:
        # Collect SNPs from all vertical directories
        all_vertical_snps = []
        for vertical_dir in vertical_dirs:
            vertical_snps = parse_snps(vertical_dir)
            if len(vertical_snps) == 0:
                print(f"Warning: No SNP files found in vertical directory: {vertical_dir}")
            else:
                all_vertical_snps.extend(vertical_snps)
        
        if len(all_vertical_snps) == 0:
            raise ValueError(f"No SNP files found in any vertical directories: {vertical_dirs}")
        
        # Generate all possible pairs using product with repeat=2
        all_pairs = list(product(all_vertical_snps, repeat=2))
        
        # Randomly select N pairs (duplicates allowed)
        selected_pairs = random.choices(all_pairs, k=n_pairs)
        
        return selected_pairs 