#!/usr/bin/env python3
"""Simulation script for eye width calculations."""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from simulation.collection.batch_simulator import EyeWidthSimulatePipeline

def main():
    """Main entry point for simulation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run SNP eye-width simulation")
    parser.add_argument(
        '--infer_yaml', type=Path,
        default="saved/ew_xfmr/inference/example_48p/infer_data.yaml",
        help="YAML with data paths and bound_path"
    )
    parser.add_argument(
        '--device', default="cuda",
        help="Device to run on (cpu|cuda)"
    )
    parser.add_argument(
        '--debug', action='store_true',
        help="Enable debug mode"
    )
    parser.add_argument(
        '--proc_per_gpu', type=int, default=1,
        help="Number of processes per GPU"
    )
    parser.add_argument(
        '--snp_index', type=Path,
        help="Path to snp_index.csv file. If not provided, indices will be sequential."
    )
    args = parser.parse_args()

    simulator = EyeWidthSimulatePipeline(
        args.infer_yaml,
        device=args.device,
        debug=args.debug,
        proc_per_gpu=args.proc_per_gpu
    )
    simulator.process_snp_files()
    simulator.write_results(args.snp_index)

if __name__ == "__main__":
    main() 