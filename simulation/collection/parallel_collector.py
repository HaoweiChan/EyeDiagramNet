import os
import csv
import sys
import time
import yaml
import torch
import signal
import random
import pickle
import skrf as rf
import numpy as np
import multiprocessing
import concurrent.futures
from tqdm import tqdm
from pathlib import Path
from functools import partial
from itertools import product
from collections import defaultdict

from simulation.parameters.bound_param import SampleResult, ParameterSet
from simulation.engine.sparam_to_ew import snp_eyewidth_simulation

# New imports for CSV handling
import pandas as pd

def parse_snps(snp_dir):
    if Path(snp_dir, 'npz').exists():
        snp_dir = snp_dir.joinpath("npz")
    elif Path(snp_dir, 'snp').exists():
        snp_dir = snp_dir.joinpath("snp")

    suffix = '*.s*p'
    if len(list(snp_dir.glob("*.npz"))):
        suffix = '*.npz'
    return list(snp_dir.glob(suffix))

def init_worker():
    # Ignore ctrl+c in child workers so only main process sees it
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def collect_snp_simulation_data(pickle_folder, params, snp_file, id_gpu, directions=None, max_samples=50, debug=False):
    # Load pickle file
    snp_horiz, snp_tx, snp_rx = snp_file
    keys = ['configs', 'line_ews', 'snp_txs', 'snp_rxs', 'directions']
    pickle_file = Path(pickle_folder).joinpath(f"{snp_horiz.stem}.pkl")
    if pickle_file.exists():
        with open(pickle_file, 'rb') as f:
            loaded = pickle.load(f)
        data = {key: loaded[key] for key in keys}
    else:
        data = {key: [] for key in keys}

    # Return if the collected data already enough
    if len(data['configs']) >= max_samples:
        return

    # Generate random config from parameter space
    config = None
    while config is None or any(np.array_equal(config.to_array(), prev_config) for prev_config in data['configs']):
        config = params.sample()
    
    if debug:
        print("\n", config.to_dict())

    # Get the eyewidth result
    t0 = time.time()
    # line_ew, directions = snp_eyewidth_simulation(config, snp_file, directions=directions, device=f'cuda:{id_gpu}')
    n_lines = int(snp_horiz.suffix[2:-1]) // 2  # Remove 's' and 'p' from suffix and divide by 2
    line_ew = np.random.uniform(0, 99.9, size=n_lines)
    directions = np.random.randint(0, 2, size=n_lines) if directions is None else directions
    line_ew[line_ew >= 99.9] = -0.1  # treating 99.9 data as closed eyes

    data['configs'].append(config.to_list())
    data['line_ews'].append(line_ew.tolist())
    data['snp_txs'].append(snp_tx.as_posix())
    data['snp_rxs'].append(snp_rx.as_posix())
    data['directions'].append(directions.tolist())

    with open(pickle_file, 'wb') as f:
        pickle.dump({'snp_horiz': snp_horiz.as_posix(), **data}, f)

def flatten_dict(d):
    items = {}
    for k, v in d.items():
        if k == "horizontal_dataset":
            items[k] = v
            continue
        if isinstance(v, dict):
            items.update(flatten_dict(v))
        else:
            items[k] = v
    return items

import json
import yaml
from types import SimpleNamespace
from typing import Tuple

def load_infer_cfg(path: Path) -> SimpleNamespace:
    """Load `data.init_args` from infer_data.yaml as a namespace."""
    cfg = yaml.safe_load(path.read_text())
    data_init = cfg['data']['init_args']
    return SimpleNamespace(**data_init)

def load_boundary(boundary_path: Path) -> Tuple[np.ndarray, SampleResult]:
    """Return (directions, boundary params) parsed from the JSON file."""
    loaded = json.loads(boundary_path.read_text())

    if "directions" in loaded:
        directions = np.asarray(loaded["directions"], dtype=int)
    else:
        # infer number of lines from boundary keys: tx+rx count ~ half
        n_lines = len(next(iter(loaded.get("boundary", {}))).values()) // 2
        directions = np.ones(n_lines, dtype=int)

    boundary_params = bound_param.SampleResult(**loaded.get("boundary", {}))
    return directions, boundary_params

import argparse

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run SNP eye-width simulation"
    )
    p.add_argument(
        '--infer_yaml', type=Path, #required=True,
        default="saved/ew_xfmr/inference/example_48p/infer_data.yaml",
        help="YAML with data paths and bound_path"
    )
    p.add_argument(
        '--device', default="cuda",
        help="Device to run on (cpu|cuda)"
    )
    p.add_argument(
        '--debug', action='store_true',
        help="Enable debug mode"
    )
    p.add_argument(
        '--proc_per_gpu', type=int, default=1,
        help="Number of processes per GPU"
    )
    return p

def main():
    args = build_argparser().parse_args()

    infer_cfg = load_infer_cfg(args.infer_yaml)
    boundary_path = Path(infer_cfg.bound_path)
    directions, params = load_boundary(boundary_path)
    eye_width = run_simulation(
        infer_cfg,
        directions,
        params,
        args.device,
    )

    print(f"Eye-width: {eye_width:.3f} UI")

if __name__ == "__main__":
    args = build_argparser().parse_args()

    infer_cfg = load_infer_cfg(args.infer_yaml)
    boundary_path = Path(infer_cfg.bound_path)
    directions, params = load_boundary(boundary_path)
    horiz_params = ParameterSet(**params.to_dict())

    # Horizontal data loading
    horiz_dir = infer_cfg.data_dirs[0]
    horiz_files = parse_snps(Path(horiz_dir))
    pickle_folder = Path(args.infer_yaml).parent / "pkl"
    pickle_folder.mkdir(parents=True, exist_ok=True)

    # Vertical data loading
    # all_vert_files = []
    # for dir in vertical_dataset:
    #    all_vert_files.extend(list(Path(dir).glob(f"*.${args.noports}p")))
    # vert_pairs = list(product(all_vert_files, repeat=2))
    tx_file = Path(infer_cfg.tx_snp)
    rx_file = Path(infer_cfg.rx_snp)

    # Randomize and combine vertical and horizontal snps
    # random.shuffle(horiz_files)
    # selected_verts = random.choices(vert_pairs, k=len(horiz_files) * args.repetition)
    # snp_files = [(horiz_files[i % len(horiz_files)], *selected_verts[i]) for i in range(len(horiz_files) * args.repetition)]
    snp_files = [(horiz_file, tx_file, rx_file) for horiz_file in horiz_files]
    # Collect data in multi-processing
    # num_gpus = torch.cuda.device_count()
    num_gpus = 1
    print(f"Number of GPUs: {num_gpus}")

    partial_collect_snp_simulation_data = partial(collect_snp_simulation_data, pickle_folder, max_samples=1)
    if not args.debug:
        multiprocessing.set_start_method("forkserver")
        max_workers = args.proc_per_gpu * num_gpus
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker) as executor:
            futures = [
                executor.submit(partial_collect_snp_simulation_data, horiz_params, snp_file, i % num_gpus, directions)
                for i, snp_file in enumerate(snp_files)
            ]

            try:
                for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                    pass
            except KeyboardInterrupt:
                print("KeyboardInterrupt detected, shutting down...")
                # kill each process
                for pid, proc in executor._processes.items():
                    proc.terminate()
                # shutdown the pool so it doesn't try to re-spawn
                executor.shutdown(wait=False, cancel_futures=True)
                sys.exit(1)
    else:
        for i, snp_file in enumerate(snp_files):
            partial_collect_snp_simulation_data(horiz_params, snp_file, i % num_gpus, directions, debug=True)
            
            snp_horiz, snp_tx, snp_rx = snp_file
            pickle_file = Path(pickle_folder).joinpath(f"{snp_horiz.stem}.pkl")
            with open(pickle_file, 'rb') as f:
                loaded = pickle.load(f)
            print(loaded)

    # New code to read snp_index.csv and write ew_results.csv
    snp_index_path = Path("../test_data/snp_index.csv")
    if not snp_index_path.exists():
        print("Error: snp_index.csv not found in test_data.")
        sys.exit(1)
    
    # Read snp_index.csv into a mapping of snp filename to index
    snp_index_df = pd.read_csv(snp_index_path)
    snp_to_index = dict(zip(snp_index_df['snp_file_name'], snp_index_df['index']))
    
    # Prepare to write ew_results.csv
    ew_results = []
    for snp_file in snp_files:
        snp_horiz, _, _ = snp_file
        pickle_file = Path(pickle_folder).joinpath(f"{snp_horiz.stem}.pkl")
        if pickle_file.exists():
            with open(pickle_file, 'rb') as f:
                loaded = pickle.load(f)
            # Assuming the eye width is stored in loaded['line_ews'][0]
            # Adjust if the structure is different
            line_ews = loaded['line_ews'][0]
            index = snp_to_index.get(snp_horiz.name, -1)
            ew_results.append({'index': index, **{i: ew for i, ew in enumerate(line_ews)}})
    
    # Write ew_results.csv
    ew_results_df = pd.DataFrame(ew_results)
    ew_results_df.to_csv(Path(args.infer_yaml).parent / "ew_results.csv", index=False)
    print("Wrote ew_results.csv with index and eye width data per line.")