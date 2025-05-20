import sys
import json
import yaml
import torch
import signal
import random
import pickle
import logging
import numpy as np
import multiprocessing
from tqdm import tqdm
from pathlib import Path
from types import SimpleNamespace
from typing import List, Dict, Tuple
from concurrent.futures import ProcessPoolExecutor

import bound_param
from bound_param import SampleResult, ParameterSet
from sparam_to_ew import snp_eyewidth_simulation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SNPSimulator:
    def __init__(self, config_path: Path, device: str = "cuda", debug: bool = False):
        self.config_path = config_path
        self.device = device
        self.debug = debug
        self.infer_cfg = self._load_infer_config()
        self.directions, self.params = self._load_boundary()
        self.num_gpus = torch.cuda.device_count() if device == "cuda" else 0
        logger.info(f"Number of GPUs: {self.num_gpus}")

    @staticmethod
    def _load_infer_config(config_path: Path) -> SimpleNamespace:
        """Load inference configuration from YAML file."""
        try:
            cfg = yaml.safe_load(config_path.read_text())
            data_init = cfg["data"]["init_args"]
            return SimpleNamespace(**data_init)
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            raise

    @staticmethod
    def _load_boundary(boundary_path: Path) -> Tuple[np.ndarray, SampleResult]:
        """Load boundary parameters and directions."""
        try:
            loaded = json.loads(boundary_path.read_text())
            if 'directions' in loaded:
                directions = np.array(loaded["directions"], dtype=int)
            else:
                nlines = len(next(iter(loaded.get("boundary", {})).values())) // 2
                directions = np.ones(nlines, dtype=int)
            boundary_params = bound_param.SampleResult(**loaded.get("boundary", {}))
            return directions, boundary_params
        except Exception as e:
            logger.error(f"Failed to load boundary from {boundary_path}: {e}")
            raise

    @staticmethod
    def parse_snps(snp_dir: Path) -> List[Path]:
        """Parse SNP files from directory."""
        if Path(snp_dir, 'npz').exists():
            snp_dir = snp_dir.joinpath('npz')
        elif Path(snp_dir, 'snp').exists():
            snp_dir = snp_dir.joinpath('snp')

        suffix = '*.snp'
        if len(list(snp_dir.glob('*.npz'))) > 0:
            suffix = '*.npz'
        return list(snp_dir.glob(suffix))

    def collect_simulation_data(self, pickle_folder: Path, snp_file: Tuple[Path, Path, Path], 
                              gpu_id: int, max_samples: int = 50) -> None:
        """Collect simulation data for a single SNP file."""
        snp_horiz, snp_tx, snp_rx = snp_file
        pickle_file = pickle_folder.joinpath(f'{snp_horiz.stem}.pkl')
        keys = ['configs', 'line_ews', 'snp_txs', 'snp_rxs', 'directions']
        
        # Load existing data or initialize new
        data = self._load_or_init_data(pickle_file, keys)
        
        if len(data['configs']) >= max_samples:
            return

        # Generate new configuration
        config = self._generate_unique_config(data['configs'])
        if self.debug:
            logger.debug(f"Generated config: {config.to_dict()}")

        # Run simulation
        try:
            line_ew, directions = snp_eyewidth_simulation(
                config, snp_file, device=f'cuda:{gpu_id}'
            )
            line_ew = -0.1 if line_ew >= 99.9 else line_ew

            # Update data
            self._update_simulation_data(data, config, line_ew, snp_tx, snp_rx, directions)
            
            # Save results
            self._save_simulation_data(pickle_file, snp_horiz, data)
        except Exception as e:
            logger.error(f"Simulation failed for {snp_horiz}: {e}")
            raise

    @staticmethod
    def _load_or_init_data(pickle_file: Path, keys: List[str]) -> Dict[str, List]:
        """Load existing data or initialize new data structure."""
        if pickle_file.exists():
            with open(pickle_file, 'rb') as f:
                loaded = pickle.load(f)
                return {key: loaded[key] for key in keys}
        return {key: [] for key in keys}

    def _generate_unique_config(self, existing_configs: List) -> ParameterSet:
        """Generate a unique configuration that hasn't been used before."""
        while True:
            config = self.params.sample()
            if not any(np.array_equal(config.to_array(), prev_config) for prev_config in existing_configs):
                return config

    @staticmethod
    def _update_simulation_data(data: Dict[str, List], config: ParameterSet, 
                              line_ew: float, snp_tx: Path, snp_rx: Path, 
                              directions: np.ndarray) -> None:
        """Update simulation data with new results."""
        data['configs'].append(config.to_list())
        data['line_ews'].append(line_ew.tolist())
        data['snp_txs'].append(snp_tx.as_posix())
        data['snp_rxs'].append(snp_rx.as_posix())
        data['directions'].append(directions.tolist())

    @staticmethod
    def _save_simulation_data(pickle_file: Path, snp_horiz: Path, data: Dict[str, List]) -> None:
        """Save simulation data to pickle file."""
        with open(pickle_file, 'wb') as f:
            pickle.dump({'snp_horiz': snp_horiz.as_posix(), **data}, f)

    def run_simulation(self, proc_per_gpu: int = 1, repetition: int = 1) -> None:
        """Run the main simulation pipeline."""
        # Setup directories and files
        horiz_dir = self.infer_cfg.data_dirs[0]
        horiz_files = self.parse_snps(Path(horiz_dir))
        pickle_folder = Path(self.config_path).parent / "pkl"
        pickle_folder.mkdir(exist_ok=True)

        # Prepare SNP files
        tx_file = Path(self.infer_cfg.tx_snp)
        rx_file = Path(self.infer_cfg.rx_snp)
        snp_files = [(horiz_file, tx_file, rx_file) for horiz_file in horiz_files]
        random.shuffle(snp_files)

        # Run simulation
        if not self.debug:
            self._run_parallel_simulation(snp_files, pickle_folder, proc_per_gpu)
        else:
            self._run_debug_simulation(snp_files, pickle_folder)

    def _run_parallel_simulation(self, snp_files: List[Tuple[Path, Path, Path]], 
                               pickle_folder: Path, proc_per_gpu: int) -> None:
        """Run simulation in parallel using ProcessPoolExecutor."""
        multiprocessing.set_start_method("forkserver")
        max_workers = proc_per_gpu * self.num_gpus
        
        with ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker) as executor:
            futures = [
                executor.submit(
                    self.collect_simulation_data,
                    pickle_folder,
                    snp_file,
                    i % self.num_gpus
                )
                for i, snp_file in enumerate(snp_files)
            ]
            
            try:
                for _ in tqdm(concurrent.futures.as_completed(futures), 
                            total=len(futures), 
                            desc="Running simulations"):
                    pass
            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt detected, shutting down...")
                for pid, proc in executor._processes.items():
                    proc.terminate()
                executor.shutdown(wait=False, cancel_futures=True)
                sys.exit(1)

    def _run_debug_simulation(self, snp_files: List[Tuple[Path, Path, Path]], 
                            pickle_folder: Path) -> None:
        """Run simulation in debug mode."""
        for i, snp_file in enumerate(snp_files):
            self.collect_simulation_data(pickle_folder, snp_file, i % self.num_gpus, debug=True)
            snp_horiz, _, _ = snp_file
            pickle_file = pickle_folder.joinpath(f'{snp_horiz.stem}.pkl')
            with open(pickle_file, 'rb') as f:
                loaded = pickle.load(f)
                logger.debug(f"Debug output for {snp_horiz}: {loaded}")

def init_worker():
    """Initialize worker process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run SNP eye-width simulation")
    parser.add_argument("--infer_yaml", type=Path, required=True, 
                       help="YAML with data paths and bound_path")
    parser.add_argument("--device", default="cuda", 
                       help="Device to run on (cpu/cuda)")
    parser.add_argument("--debug", action='store_true', 
                       help="Enable debug mode")
    parser.add_argument("--proc_per_gpu", type=int, default=1,
                       help="Number of processes per GPU")
    parser.add_argument("--repetition", type=int, default=1,
                       help="Number of repetitions")
    args = parser.parse_args()

    try:
        simulator = SNPSimulator(args.infer_yaml, args.device, args.debug)
        simulator.run_simulation(args.proc_per_gpu, args.repetition)
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
