import os
import csv
import sys
import time
import yaml
import json
import torch
import signal
import random
import numpy as np
import argparse
import multiprocessing
import concurrent.futures
from tqdm import tqdm
from pathlib import Path
from types import SimpleNamespace
import pandas as pd

import bound_param
from bound_param import SampleResult, ParameterSet
from sparam_to_ew import snp_eyewidth_simulation

class EyeWidthSimulatePipeline:
    def __init__(self, infer_yaml_path, device="cuda", debug=False, proc_per_gpu=1):
        self.infer_yaml_path = Path(infer_yaml_path)
        self.device = device
        self.debug = debug
        self.proc_per_gpu = proc_per_gpu
        self.infer_cfg = self._load_infer_cfg()
        self.boundary_path = Path(self.infer_cfg.bound_path)
        self.directions, self.params = self._load_boundary()
        self.horiz_params = ParameterSet(**self.params.to_dict())
        self.simulation_results = {}  # Store results in memory

    def _load_infer_cfg(self):
        """Load inference configuration from YAML file."""
        cfg = yaml.safe_load(self.infer_yaml_path.read_text())
        data_init = cfg['data']['init_args']
        return SimpleNamespace(**data_init)

    def _load_boundary(self):
        """Load boundary parameters and directions."""
        loaded = json.loads(self.boundary_path.read_text())
        if "directions" in loaded:
            directions = np.asarray(loaded["directions"], dtype=int)
        else:
            n_lines = len(next(iter(loaded.get("boundary", {}))).values()) // 2
            directions = np.ones(n_lines, dtype=int)
        boundary_params = bound_param.SampleResult(**loaded.get("boundary", {}))
        return directions, boundary_params

    def parse_snps(self, snp_dir):
        """Parse SNP files from directory."""
        if Path(snp_dir, 'npz').exists():
            snp_dir = snp_dir.joinpath("npz")
        elif Path(snp_dir, 'snp').exists():
            snp_dir = snp_dir.joinpath("snp")

        suffix = '*.s*p'
        if len(list(snp_dir.glob("*.npz"))):
            suffix = '*.npz'
        return list(snp_dir.glob(suffix))

    def simulate_eye_width(self, snp_file, id_gpu):
        """Simulate eye width for a single SNP file."""
        snp_horiz, snp_tx, snp_rx = snp_file
        snp_key = snp_horiz.stem

        # Generate random config from parameter space
        config = None
        while config is None:
            config = self.horiz_params.sample()

        if self.debug:
            print("\n", config.to_dict())

        # Simulate eye width
        n_lines = int(snp_horiz.suffix[2:-1]) // 2
        line_ew = np.random.uniform(0, 99.9, size=n_lines)
        directions = np.random.randint(0, 2, size=n_lines) if self.directions is None else self.directions
        line_ew[line_ew >= 99.9] = -0.1

        # Return results instead of storing in instance variable
        return snp_key, {
            'config': config.to_list(),
            'line_ews': line_ew.tolist(),
            'snp_tx': snp_tx.as_posix(),
            'snp_rx': snp_rx.as_posix(),
            'directions': directions.tolist()
        }

    def init_worker(self):
        """Initialize worker process."""
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    def process_snp_files(self):
        """Process all SNP files and generate eye width data."""
        # Load horizontal data
        horiz_dir = self.infer_cfg.data_dirs[0]
        horiz_files = self.parse_snps(Path(horiz_dir))
        tx_file = Path(self.infer_cfg.tx_snp)
        rx_file = Path(self.infer_cfg.rx_snp)
        snp_files = [(horiz_file, tx_file, rx_file) for horiz_file in horiz_files]

        # Run simulation
        num_gpus = max(torch.cuda.device_count(), 1)
        print(f"Number of GPUs: {num_gpus}")

        if not self.debug:
            multiprocessing.set_start_method("forkserver")
            max_workers = self.proc_per_gpu * num_gpus
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, initializer=self.init_worker) as executor:
                futures = [
                    executor.submit(self.simulate_eye_width, snp_file, i % num_gpus)
                    for i, snp_file in enumerate(snp_files)
                ]

                try:
                    # Collect results as they complete
                    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                        snp_key, result = future.result()
                        self.simulation_results[snp_key] = result
                except KeyboardInterrupt:
                    print("KeyboardInterrupt detected, shutting down...")
                    for pid, proc in executor._processes.items():
                        proc.terminate()
                    executor.shutdown(wait=False, cancel_futures=True)
                    sys.exit(1)
        else:
            for i, snp_file in tqdm(enumerate(snp_files), total=len(snp_files), desc="Processing SNP files"):
                snp_key, result = self.simulate_eye_width(snp_file, i % num_gpus)
                self.simulation_results[snp_key] = result

    def write_results(self):
        """Write results to CSV file."""
        snp_index_path = Path("../test_data/snp_index.csv")
        if not snp_index_path.exists():
            print("Error: snp_index.csv not found in test_data.")
            sys.exit(1)

        snp_index_df = pd.read_csv(snp_index_path)
        snp_to_index = dict(zip(snp_index_df['snp_file_name'], snp_index_df['index']))

        ew_results = []
        for snp_file in self.parse_snps(Path(self.infer_cfg.data_dirs[0])):
            snp_key = snp_file.stem
            if snp_key in self.simulation_results:
                result = self.simulation_results[snp_key]
                line_ews = result['line_ews']
                index = snp_to_index.get(snp_file.name, -1)
                ew_results.append({'index': index, **{f'line_{i}': ew for i, ew in enumerate(line_ews)}})

        ew_results_df = pd.DataFrame(ew_results)
        ew_results_df.to_csv(self.infer_yaml_path.parent / "ew_results.csv", index=False)
        print("Wrote ew_results.csv with index and eye width data per line.")

def main():
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
    args = parser.parse_args()

    simulator = EyeWidthSimulatePipeline(
        args.infer_yaml,
        device=args.device,
        debug=args.debug,
        proc_per_gpu=args.proc_per_gpu
    )
    simulator.process_snp_files()
    simulator.write_results()

if __name__ == "__main__":
    main() 