"""SNP file utilities for training data collection."""

import random
import numpy as np
from pathlib import Path
from itertools import product
from skrf.network import Network

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
    
    Raises:
        ValueError: If the reference SNP file cannot be read or has invalid port count.
    """
    try:
        ref_net = Network(str(reference_trace_snp_file))
        n_ports = ref_net.nports
        if n_ports % 2 != 0:
            raise ValueError(
                f"Number of ports ({n_ports}) from {reference_trace_snp_file} must be even for thru connection."
            )
    except Exception as e:
        raise ValueError(
            f"Error reading reference SNP {reference_trace_snp_file}: {e}"
        )

    thru_snp_dir = Path(base_output_dir) / trace_pattern_key
    thru_snp_dir.mkdir(parents=True, exist_ok=True)
    thru_snp_filename = f"auto_thru_{n_ports}port.s{n_ports}p"
    thru_snp_path = thru_snp_dir / thru_snp_filename

    if thru_snp_path.exists():
        print(f"Using existing auto-generated thru SNP: {thru_snp_path}")
        return thru_snp_path

    print(f"Generating auto-thru SNP with {n_ports} ports: {thru_snp_path}")

    # Extract frequency points from the reference trace SNP to ensure matching frequencies
    try:
        ref_net = Network(str(reference_trace_snp_file))
        freq = ref_net.f
        n_freq_points = len(freq)
        print(f"Using {n_freq_points} frequency points from reference trace SNP ({freq.min()/1e9:.2f} - {freq.max()/1e9:.2f} GHz)")
    except Exception as e:
        raise ValueError(
            f"Could not extract frequency from reference SNP {reference_trace_snp_file}: {e}"
        )
    
    s_matrix = np.zeros((n_freq_points, n_ports, n_ports), dtype=complex)

    # Set thru connections for all frequency points (frequency-independent thru)
    for i in range(n_ports // 2):
        j = i + n_ports // 2
        s_matrix[:, i, j] = 1.0  # Transmission from port i to port j (S_ji)
        s_matrix[:, j, i] = 1.0  # Transmission from port j to port i (S_ij)

    z0_array = np.full(n_ports, 50) # Standard 50 Ohm impedance for all ports

    try:
        thru_net = Network(frequency=freq, s=s_matrix, z0=z0_array)
        thru_net.name = f'auto_thru_{n_ports}port'
        thru_net.write_touchstone(filename=str(thru_snp_path), write_z0=True)
        print(f"Successfully generated auto-thru SNP: {thru_snp_path}")
    except Exception as e:
        raise ValueError(
            f"Error writing touchstone file {thru_snp_path}: {e}"
        )

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
        List of (drv_snp, odt_snp) tuples
    """
    if vertical_dirs is None:
        if trace_snps is None or len(trace_snps) == 0:
            raise ValueError("trace_snps must be provided and non-empty when vertical_dirs is None")
        if base_output_dir is None or trace_pattern_key is None:
            raise ValueError("base_output_dir and trace_pattern_key must be provided for auto-thru SNP generation")

        # Generate one thru SNP to be used for all pairs
        # Use the first trace_snp as a reference for port count
        thru_snp_path = generate_thru_snp(trace_snps[0], base_output_dir, trace_pattern_key)
        
        # All pairs will use this single thru SNP for both DRV and ODT
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

if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Generate all-thru SNP file from reference trace SNP")
    parser.add_argument("reference_snp", help="Path to reference trace SNP file")
    parser.add_argument("-o", "--output-dir", default=".", help="Output directory (default: current directory)")
    parser.add_argument("-k", "--key", default="thru", help="Subdirectory key (default: 'thru')")
    
    args = parser.parse_args()
    
    # Validate input file
    ref_snp_path = Path(args.reference_snp)
    if not ref_snp_path.exists():
        print(f"Error: Reference SNP file {ref_snp_path} does not exist")
        sys.exit(1)
    
    try:
        thru_snp_path = generate_thru_snp(
            reference_trace_snp_file=args.reference_snp,
            base_output_dir=args.output_dir,
            trace_pattern_key=args.key
        )
        print(f"All-thru SNP generated at: {thru_snp_path}")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1) 