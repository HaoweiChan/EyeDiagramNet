import json
import psutil
import pickle
import random
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader

from lightning import LightningDataModule
from lightning.pytorch.utilities import CombinedLoader
from lightning.pytorch.utilities.rank_zero import rank_zero_info

from ..utils.scaler import MinMaxScaler
from .processors import CSVProcessor, TraceSequenceProcessor
from simulation.parameters.bound_param import SampleResult, to_new_param_name
from simulation.io.pickle_utils import load_pickle_data
from common.signal_utils import read_snp, flip_snp

def get_loader_from_dataset(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = False
):
    drop_last = shuffle

    # Optimize num_workers based on system capabilities
    import os
    cpu_count = os.cpu_count()
    # Use fewer workers to avoid overwhelming the system
    num_workers = min(4, cpu_count // 2) if cpu_count else 2

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=drop_last,
    )
    return loader

class SharedMemoryCache:
    def __init__(self):
        self.cache = mp.Manager().dict()

    def get(self, key):
        return self.cache.get(key, None)

    def store(self, key, value):
        if isinstance(value, tuple):
            self.cache[key] = tuple(v.share_memory_() for v in value)
        else:
            self.cache[key] = value.share_memory_()

class TraceEWDataset(Dataset):
    def __init__(
        self,
        trace_seqs,
        directions,
        boundaries,
        vert_snps,
        eye_widths,
        metas,
        train=False,
        ignore_snp=False
    ):
        super().__init__()

        self.trace_seqs = torch.from_numpy(trace_seqs.copy()).float()
        self.directions = torch.from_numpy(directions).int()
        self.boundaries = torch.from_numpy(boundaries).float()
        self.config_keys = metas[0]['config_keys']
        self.metas = [{k: v for k, v in d.items() if k != 'config_keys'} for d in metas]

        self.vert_snps = vert_snps
        self.vert_cache = SharedMemoryCache()
        self.ignore_snp = ignore_snp

        self.eye_widths = torch.from_numpy(eye_widths).float()
        self.repetition = self.boundaries.size(1)
        self.train = train
        
        # Create dummy SNP data if ignoring SNPs
        if self.ignore_snp:
            # Create a small dummy SNP tensor: (2, F, P, P) where F=32, P=4 for minimal memory usage
            self.dummy_snp = torch.zeros(2, 32, 4, 4, dtype=torch.complex64)

    def __len__(self):
        return len(self.trace_seqs) * self.repetition

    def __getitem__(self, index):
        seq_index = index // self.repetition
        bnd_index = index % self.repetition

        meta = self.metas[seq_index]
        trace_seq = self.trace_seqs[seq_index]

        if self.ignore_snp:
            # Return dummy SNP data without loading files
            vert_snp = self.dummy_snp.clone()
        else:
            # Retrieve the left and right snps
            drv_vert_file, odt_vert_file = self.vert_snps[seq_index, bnd_index]
            drv_vert_snp = self.vert_snp(drv_vert_file)
            odt_vert_snp = self.vert_snp(odt_vert_file)
            vert_snp = torch.stack((drv_vert_snp, flip_snp(odt_vert_snp)))

        # Retrieve the boundary and eye width values
        direction = self.directions[seq_index, bnd_index]
        boundary = self.boundaries[seq_index, bnd_index]
        eye_width = self.eye_widths[seq_index, bnd_index]

        if self.train and random.random() > 0.5 and not self.ignore_snp:
            trace_seq, direction, eye_width, vert_snp = \
                self.augment(trace_seq, direction, eye_width, vert_snp)
        return trace_seq, direction, boundary, vert_snp, eye_width, meta

    def transform(self, seq_scaler, fix_scaler):
        """Apply scaling transformations using semantic feature access."""
        num = len(self.trace_seqs)

        # Scale only the scalable features (geometry + additional features)
        scalable_feats = TraceSequenceProcessor.get_scalable_features(self.trace_seqs)
        feat_shape = scalable_feats.shape
        scalable_feats_flat = scalable_feats.reshape(-1, feat_shape[-1])
        scaled_feats = seq_scaler.transform(scalable_feats_flat).reshape(feat_shape)
        
        # Update only the scalable portion of the sequence
        self.trace_seqs[:, :, TraceSequenceProcessor.get_scalable_slice()] = scaled_feats.float()

        # Scale boundary features  
        bound_dim = self.boundaries.size(-1)
        scaled_boundary = fix_scaler.transform(self.boundaries).reshape(num, -1, bound_dim)
        self.boundaries = scaled_boundary.float()
        
        return self

    def load_snp(self, snp_file):
        snp_data = self.vert_cache.get(snp_file)
        if snp_data is None:
            snp_data = read_snp(Path(snp_file)).s
            snp_data = torch.from_numpy(snp_data).to(torch.complex64)
            if self.check_memory(snp_data):
                self.vert_cache.store(snp_file, snp_data)
        return snp_data

    def check_memory(self, data):
        data_size = data.element_size() * data.nelement()
        avail_mem = psutil.virtual_memory().available
        return avail_mem > data_size

    def augment(self, trace_seq, direction, eye_width, vert_snp, max_layer_idx=99, max_pos=10000.0):
        seq_len = trace_seq.size(0)
        seq_order = torch.arange(seq_len)
        signal_indices = torch.where(trace_seq[:, 1] == 0)[0]
        signal_order = -torch.ones_like(seq_order)
        signal_order[signal_indices] = torch.arange(signal_indices.size(0))
        pairs = torch.stack((seq_order, signal_order)).T

        # Shuffle the sequence
        pairs = pairs[torch.randperm(pairs.size(0))]
        seq_order = pairs[:, 0]
        signal_order = pairs[:, 1]
        signal_order = signal_order[signal_order >= 0]
        snp_order = torch.cat((signal_order, signal_order + len(signal_order)))

        # Extract the shuffled sequence
        trace_seq = trace_seq[seq_order]
        eye_width = eye_width[signal_order]
        direction = direction[signal_order]
        vert_snp = vert_snp[:, :, :, snp_order]

        # Augment layer values
        max_layer = trace_seq[:, 0].max().int().item()
        trace_seq[:, 0] += torch.randint(0, max_layer_idx - max_layer, (1,)).expand(seq_len)

        # Augment x dim relative position values
        max_spatial_dim = trace_seq[:, -2:].max(0).values
        trace_seq[:, -2:] += (torch.rand((2,)) * (max_pos - max_spatial_dim)).unsqueeze(0).expand(seq_len, 2)

        return trace_seq, direction, eye_width, vert_snp

    def vert_snp(self, snp_file):
        """Load and cache vertical SNP data."""
        return self.load_snp(snp_file)

class InferenceTraceEWDataset(Dataset):
    def __init__(
        self,
        trace_seqs,
        direction,
        boundary,
        drv_snp,
        odt_snp
    ):
        super().__init__()

        self.trace_seqs = torch.from_numpy(trace_seqs.copy()).float()
        self.boundary = torch.from_numpy(boundary).float()
        self.direction = torch.from_numpy(direction).int()

        drv_snp = torch.from_numpy(drv_snp).to(torch.complex64)
        odt_snp = torch.from_numpy(odt_snp).to(torch.complex64)
        self.vert_snp = torch.stack((drv_snp, flip_snp(odt_snp)))
        self.config = {
            'boundary': boundary,
            'directions': direction,
            'snp_drv': drv_snp.name,
            'snp_odt': odt_snp.name
        }

    def __len__(self):
        return len(self.trace_seqs)

    def __getitem__(self, index):
        trace_seq = self.trace_seqs[index]
        return trace_seq, self.direction, self.boundary, self.vert_snp, self.config

    def transform(self, seq_scaler, fix_scaler):
        """Apply scaling transformations using semantic feature access."""
        num = len(self.trace_seqs)
        
        # Scale only the scalable features (geometry + additional features)
        scalable_feats = TraceSequenceProcessor.get_scalable_features(self.trace_seqs)
        feat_shape = scalable_feats.shape
        scalable_feats_flat = scalable_feats.reshape(-1, feat_shape[-1])
        scaled_feats = seq_scaler.transform(scalable_feats_flat).reshape(feat_shape)
        
        # Update only the scalable portion of the sequence
        self.trace_seqs[:, :, TraceSequenceProcessor.get_scalable_slice()] = scaled_feats.float()

        # Scale boundary features  
        bound_dim = self.boundary.size(-1)
        scaled_boundary = fix_scaler.transform(self.boundary).reshape(num, -1, bound_dim)
        self.boundary = scaled_boundary.float()
        
        return self

class TraceSeqEWDataloader(LightningDataModule):
    def __init__(
        self,
        data_dirs: dict[str, str],
        label_dir: str,
        batch_size: int,
        test_size: float = 0.2,
        scaler_path: str | None = None,
        ignore_snp: bool = False,
    ):
        super().__init__()
        self.data_dirs = data_dirs
        self.label_dir = Path(label_dir)
        self.batch_size = batch_size
        self.test_size = test_size
        self.scaler_path = scaler_path
        self.ignore_snp = ignore_snp

        # containers for datasets per "name"
        self.train_dataset: dict[str, TraceEWDataset] = {}
        self.val_dataset: dict[str, TraceEWDataset] = {}

    def setup(self, stage: str | None = None, nan: int = -1):
        # Scalers
        fit_scaler = True
        try:
            self.seq_scaler, self.fix_scaler = torch.load(self.scaler_path)
            rank_zero_info(f"Loaded scalers from {self.scaler_path}")
            fit_scaler = False
        except (FileNotFoundError, AttributeError, EOFError):
            self.seq_scaler = MinMaxScaler(nan=nan)
            self.fix_scaler = MinMaxScaler(nan=nan)
            rank_zero_info("Could not find scalers on disk, creating new ones.")

        # locate every CSV once via processor
        processor = CSVProcessor()
        csv_paths = processor.locate(self.data_dirs) # dict[str, Path]

        # iterate over each named dataset
        for name, csv_path in csv_paths.items():
            case_ids, input_arr = processor.parse(csv_path)

            # Load labels using unified dataclass loading approach
            labels: dict[str, tuple] = {}
            for pkl_file in Path(self.label_dir, name).glob("*.pkl"):
                # Load data as list of SimulationResult dataclasses
                results = load_pickle_data(pkl_file)
                
                if not results:
                    rank_zero_info(f"Skipping malformed or empty pickle: {pkl_file.name}")
                    continue

                # Extract data from the first result to get metadata
                first_result = results[0]
                snp_horiz_path = first_result.snp_horiz

                if not snp_horiz_path:
                    rank_zero_info(f"Skipping malformed pickle: {pkl_file.name} ('snp_horiz' not found).")
                    continue

                # The key must match the case_id from the CSV file
                try:
                    key = int(Path(snp_horiz_path).stem.replace("-", "_").split("_")[-1].split(".")[0])
                except (ValueError, IndexError):
                    rank_zero_info(f"Could not parse case ID from snp_horiz: '{snp_horiz_path}'. "
                                   f"Skipping pickle file: {pkl_file.name}")
                    continue

                # Convert dataclass results back to the format expected by the rest of the code
                configs = []
                directions_list = []
                line_ews_list = []
                snp_drvs = []
                snp_odts = []

                for result in results:
                    # Convert config from keys+values back to dict format for backward compatibility
                    config_dict = dict(zip(result.config_keys, result.config_values))
                    if isinstance(config_dict, dict):
                        config_dict = to_new_param_name(config_dict)
                    configs.append(config_dict)
                    
                    directions_list.append(result.directions)
                    line_ews_list.append(result.line_ews)
                    snp_drvs.append(result.snp_drv)
                    snp_odts.append(result.snp_odt)

                # Handle SNP vertical data based on ignore_snp flag
                if self.ignore_snp:
                    snp_vert = (("dummy_drv.snp", "dummy_odt.snp"),) * len(directions_list)
                else:
                    snp_vert = tuple(zip(snp_drvs, snp_odts))

                # Create metadata dict from the first result
                meta = {
                    'config_keys': first_result.config_keys,
                    'snp_horiz': first_result.snp_horiz,
                    'n_ports': first_result.n_ports,
                    'param_types': first_result.param_types
                }

                labels[key] = (
                    configs,
                    directions_list,
                    line_ews_list,
                    snp_vert,
                    meta
                )

            # keep only indices present in labels
            label_keys = set(labels.keys())
            keep_idx = [i for i, cid in enumerate(case_ids) if cid in label_keys]
            input_arr = input_arr[keep_idx]
            sorted_keys = [case_ids[i] for i in keep_idx]
            sorted_vals = [labels[k] for k in sorted_keys]

            # Align tensors by selecting entries that match the maximum length
            lengths = [len(v[0]) for v in sorted_vals if v and v[0] is not None]
            if not lengths:
                rank_zero_info(f"No valid label entries for {name}; skipping.")
                continue
            max_len = max(lengths)

            keep_indices = [i for i, s in enumerate(sorted_vals) if len(s[0]) == max_len]
            sorted_vals = [sorted_vals[i] for i in keep_indices]
            input_arr = input_arr[keep_indices]

            boundaries_list, directions_list, eye_widths_list, snp_paths_list, metas_list = zip(*sorted_vals)
            boundaries, directions, eye_widths = map(
                np.array, (boundaries_list, directions_list, eye_widths_list)
            )
            snp_paths = np.array(snp_paths_list, dtype=object)
            metas = np.array(metas_list, dtype=object)
            eye_widths[eye_widths < 0] = 0

            rank_zero_info(f"{name}| input_seq {input_arr.shape} | eye_width {eye_widths.shape} | ignore_snp={self.ignore_snp}")

            # train/val split
            indices = np.arange(len(input_arr))
            train_idx, val_idx = train_test_split(
                indices, test_size=self.test_size, shuffle=True, random_state=42
            )

            def _split(arr):
                return arr[train_idx], arr[val_idx]

            x_seq_tr, x_seq_val = _split(input_arr)
            x_tok_tr, x_tok_val = _split(directions)
            x_fix_tr, x_fix_val = _split(boundaries)
            x_vert_tr, x_vert_val = _split(snp_paths)
            y_tr, y_val = _split(eye_widths)
            metas_tr, metas_val = _split(metas)
            
            # fit scalers once on training data
            if fit_scaler:
                # Use semantic processor to get scalable features for fitting
                scalable_feats = TraceSequenceProcessor.get_scalable_features(x_seq_tr)
                seq_feats_flat = scalable_feats.reshape(-1, scalable_feats.shape[-1])
                self.seq_scaler.partial_fit(seq_feats_flat)
                
                fix_feats_flat = x_fix_tr.reshape(-1, x_fix_tr.shape[-1])
                self.fix_scaler.partial_fit(fix_feats_flat)

            # build datasets
            self.train_dataset[name] = TraceEWDataset(
                x_seq_tr, x_tok_tr, x_fix_tr, x_vert_tr, y_tr, metas_tr, train=True, ignore_snp=self.ignore_snp
            )
            self.val_dataset[name] = TraceEWDataset(
                x_seq_val, x_tok_val, x_fix_val, x_vert_val, y_val, metas_val, ignore_snp=self.ignore_snp
            )
        
        # final transform with fitted scalers
        if fit_scaler:
            for name in list(self.train_dataset.keys()):
                self.train_dataset[name] = self.train_dataset[name].transform(
                    self.seq_scaler, self.fix_scaler
                )
                self.val_dataset[name] = self.val_dataset[name].transform(
                    self.seq_scaler, self.fix_scaler
                )

        # persist scalers for future runs
        if fit_scaler and self.trainer and self.trainer.is_global_zero and self.trainer.logger:
            save_path = Path(self.trainer.logger.log_dir) / "scaler.pth"
            torch.save((self.seq_scaler, self.fix_scaler), save_path)
            rank_zero_info(f"Saved scalers to {save_path}")

    def train_dataloader(self):
        per_loader_bs = self.batch_size // max(1, len(self.train_dataset))
        loaders = {
            name: get_loader_from_dataset(ds, batch_size=per_loader_bs, shuffle=True)
            for name, ds in self.train_dataset.items()
        }
        combined_loader = CombinedLoader(loaders, mode="min_size")
        return combined_loader

    def val_dataloader(self):
        per_loader_bs = int(self.batch_size * 1.6 / max(1, len(self.val_dataset)))
        loaders = {
            name: get_loader_from_dataset(ds, batch_size=per_loader_bs, shuffle=False)
            for name, ds in self.val_dataset.items()
        }
        return CombinedLoader(loaders, mode="min_size")

class InferenceTraceSeqEWDataloader(LightningDataModule):
    def __init__(
        self,
        data_dirs: list[str],
        drv_snp: str,
        odt_snp: str,
        batch_size: int,
        bound_path: str = None,
        scaler_path: str = None,
    ):
        super().__init__()
        self.data_dirs = data_dirs
        self.drv_snp = drv_snp
        self.odt_snp = odt_snp
        self.batch_size = batch_size
        self.bound_path = bound_path
        self.scaler_path = scaler_path

    def setup(self, stage=None):
        # Initialize processor and locate CSV files
        processor = CSVProcessor()
        csv_paths = processor.locate(self.data_dirs)

        # Load scaler
        scalers = torch.load(self.scaler_path)
        rank_zero_info(f"Loaded scaler object from {self.scaler_path}")

        drv = read_snp(Path(self.drv_snp))
        odt = read_snp(Path(self.odt_snp))
        assert drv.s.shape[-1] == odt.s.shape[-1], \
            f"DRV {self.drv_snp} and ODT {self.odt_snp} must match ports."

        # Load boundary JSON
        with open(self.bound_path, 'r') as f:
            loaded = json.load(f)
            directions = np.array(loaded['directions']) if 'directions' in loaded else np.ones(drv.s.shape[-1] // 2, dtype=int)
            ctle = loaded.get('CTLE', {"AC_gain": np.nan, "DC_gain": np.nan, "fp1": np.nan, "fp2": np.nan})
            
            # Handle backward compatibility for boundary parameter names
            boundary = loaded['boundary']
            if isinstance(boundary, dict):
                boundary = to_new_param_name(boundary)
            boundary = boundary | ctle
            self.boundary = SampleResult(**boundary)

        self.predict_dataset = []
        for csv_path in csv_paths:
            case_id, input_arr = processor.parse(csv_path)
            rank_zero_info(f"Input array: {input_arr.shape}")
            # Use structured boundary array for the new processor
            ds = InferenceTraceEWDataset(input_arr, directions, self.boundary.to_structured_array(), drv.s, odt.s)
            self.predict_dataset.append(ds.transform(*scalers))

    def predict_dataloader(self):
        return CombinedLoader(
            get_loader_from_dataset(dataset=ds, batch_size=self.batch_size)
            for ds in self.predict_dataset
        )