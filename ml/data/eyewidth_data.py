import os
import psutil
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
from common.signal_utils import read_snp, flip_snp
from common.parameters import (
    convert_configs_to_boundaries, load_scaler_with_config_keys,
    is_enhanced_scaler, save_scaler_with_config_keys
)
from common.pickle_utils import load_pickle_directory


class DummySNP:
    def __init__(self, num_ports=4, num_freqs=32):
        s_matrix = np.zeros((num_freqs, num_ports, num_ports), dtype=complex)
        self.s = s_matrix
        self.name = 'dummy_snp'


def get_loader_from_dataset(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = False
):
    dataset_size = len(dataset)
    
    # Adjust batch_size if it's larger than dataset to avoid issues
    effective_batch_size = min(batch_size, dataset_size)
    
    # Smart drop_last logic: ensure we never get 0 batches when we could get at least 1
    if dataset_size >= effective_batch_size:
        # Can form at least one batch
        if shuffle and dataset_size > effective_batch_size * 2:
            # Only drop last batch for datasets significantly larger than batch size
            drop_last = True
        else:
            # Keep all samples, especially for smaller datasets
            drop_last = False
    else:
        # Dataset smaller than batch size - definitely don't drop
        drop_last = False
    
    # Final safety check: if drop_last would result in 0 batches, force it to False
    potential_batches = dataset_size // effective_batch_size
    if drop_last and potential_batches == 1:
        drop_last = False  # Don't drop the only batch we have

    # Optimize num_workers based on system capabilities
    cpu_count = os.cpu_count()
    num_workers = min(4, cpu_count // 2) if cpu_count else 2

    loader = DataLoader(
        dataset=dataset,
        batch_size=effective_batch_size,
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

        self.trace_seqs = torch.from_numpy(trace_seqs).float()
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
            # Retrieve the drv and odt snps
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

    def vert_snp(self, snp_file):
        """Load and cache vertical SNP data."""
        return self.load_snp(snp_file)
    
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

class InferenceTraceEWDataset(Dataset):
    def __init__(
        self,
        trace_seqs,
        direction,
        boundary,
        drv_snp_obj,
        odt_snp_obj
    ):
        super().__init__()

        self.trace_seqs = torch.from_numpy(trace_seqs.copy()).float()
        self.boundary = torch.from_numpy(boundary).float()
        self.direction = torch.from_numpy(direction).int()

        drv_snp = torch.from_numpy(drv_snp_obj.s).to(torch.complex64)
        odt_snp = torch.from_numpy(odt_snp_obj.s).to(torch.complex64)
        self.vert_snp = torch.stack((drv_snp, flip_snp(odt_snp)))
        
        # Handle SNP names - check if we have real SNP objects or dummy data
        # For real SNP files, try to get the name; for dummy data, use dummy names
        if hasattr(drv_snp_obj, 'name') and drv_snp_obj.name is not None:
            snp_drv_name = drv_snp_obj.name
        else:
            snp_drv_name = 'dummy_drv_snp'
            
        if hasattr(odt_snp_obj, 'name') and odt_snp_obj.name is not None:
            snp_odt_name = odt_snp_obj.name
        else:
            snp_odt_name = 'dummy_odt_snp'
        
        self.config = {
            'boundary': boundary,
            'directions': direction,
            'snp_drv': snp_drv_name,
            'snp_odt': snp_odt_name
        }

    def __len__(self):
        return len(self.trace_seqs)

    def __getitem__(self, index):
        trace_seq = self.trace_seqs[index]
        if self.boundary.shape[0] == len(self.trace_seqs):
            # Transformed
            boundary = self.boundary[index]
        else:
            # Not transformed
            boundary = self.boundary.squeeze(0)
        return trace_seq, self.direction, boundary, self.vert_snp, self.config

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
        expanded_boundary = self.boundary.expand(num, *self.boundary.shape[1:])
        scaled_boundary = fix_scaler.transform(expanded_boundary).reshape(num, -1, bound_dim)
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
        self.test_dataset: dict[str, TraceEWDataset] = {}

    def setup(self, stage: str | None = None, nan: int = -1):
        # Scalers
        fit_scaler = True
        try:
            if self.scaler_path is None:
                raise FileNotFoundError("No scaler path provided")
            # Use weights_only=False for backward compatibility with custom scaler classes
            self.seq_scaler, self.fix_scaler = torch.load(self.scaler_path, weights_only=False)
            rank_zero_info(f"Loaded scalers from {self.scaler_path}")
            fit_scaler = False
        except (FileNotFoundError, AttributeError, EOFError) as e:
            # In test/predict modes, we must have valid scalers - don't create new ones
            if stage in ["test", "predict"] or stage is None:
                error_msg = f"Cannot find or load scaler file for {stage or 'test/predict'} mode"
                if self.scaler_path:
                    error_msg += f" at path: {self.scaler_path}"
                else:
                    error_msg += " (no scaler_path provided)"
                error_msg += f". Original error: {e}"
                rank_zero_info(f"ERROR: {error_msg}")
                raise FileNotFoundError(error_msg)
            
            # Only create new scalers during training
            self.seq_scaler = MinMaxScaler(nan=nan)
            self.fix_scaler = MinMaxScaler(nan=nan)
            rank_zero_info("Could not find scalers on disk, creating new ones for training.")

        # locate every CSV once via processor
        processor = CSVProcessor()
        csv_paths = processor.locate(self.data_dirs) # dict[str, Path]

        # iterate over each named dataset
        for name, csv_path in csv_paths.items():
            case_ids, input_arr = processor.parse(csv_path)

            # Load labels using unified pickle directory loader
            labels = load_pickle_directory(self.label_dir, name)
            
            # Handle SNP vertical data based on ignore_snp flag if needed
            if self.ignore_snp:
                for key in list(labels.keys()):
                    configs, directions_list, line_ews_list, _, meta = labels[key]
                    # Create placeholder paths; they won't be read when ignore_snp=True
                    dummy_snp_vert = (("dummy_drv.snp", "dummy_odt.snp"),) * len(directions_list)
                    labels[key] = (configs, directions_list, line_ews_list, dummy_snp_vert, meta)

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

            configs_list, directions_list, eye_widths_list, snp_paths_list, metas_list = zip(*sorted_vals)
            config_keys = metas_list[0]['config_keys']
            boundaries = convert_configs_to_boundaries(configs_list, config_keys)
            
            directions, eye_widths, snp_paths, metas = map(
                np.array, (directions_list, eye_widths_list, snp_paths_list, metas_list)
            )
            eye_widths[eye_widths < 0] = 0

            rank_zero_info(f"{name}| input_seq {input_arr.shape} | eye_width {eye_widths.shape} | ignore_snp={self.ignore_snp}")

            # train/val/test split
            indices = np.arange(len(input_arr))
            
            # For test stage, use the entire dataset as test set
            if stage == "test":
                test_idx = indices
                x_seq_test = input_arr[test_idx]
                x_tok_test = directions[test_idx]
                x_fix_test = boundaries[test_idx]
                x_vert_test = snp_paths[test_idx]
                y_test = eye_widths[test_idx]
                metas_test = metas[test_idx]
                
                # Create test dataset
                self.test_dataset[name] = TraceEWDataset(
                    x_seq_test, x_tok_test, x_fix_test, x_vert_test, y_test, metas_test, train=False, ignore_snp=self.ignore_snp
                )
                # Apply scaling if scalers are available
                if hasattr(self, 'seq_scaler') and hasattr(self, 'fix_scaler'):
                    self.test_dataset[name] = self.test_dataset[name].transform(
                        self.seq_scaler, self.fix_scaler
                    )
                continue
            
            # Standard train/val split for other stages
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

        # persist scalers for future runs with config_keys metadata
        if fit_scaler and self.trainer and self.trainer.is_global_zero and self.trainer.logger:
            save_path = Path(self.trainer.logger.log_dir) / "scaler.pth"
            
            # Get config_keys from the first dataset for metadata
            first_dataset = next(iter(self.train_dataset.values()))
            config_keys = first_dataset.config_keys
            
            # Use the new enhanced scaler saving with metadata
            save_scaler_with_config_keys((self.seq_scaler, self.fix_scaler), config_keys, save_path)
            rank_zero_info(f"Saved scalers with config_keys metadata to {save_path}: {config_keys}")

    def _calculate_per_loader_batch_size(self, datasets: dict, batch_size_multiplier: float = 1.0) -> dict:
        """
        Calculate dynamic per-loader batch sizes with smart redistribution.
        Small datasets don't waste their sample allocation - unused samples go to larger datasets.
        
        Args:
            datasets: Dictionary of datasets to calculate batch size for
            batch_size_multiplier: Multiplier for batch size (e.g., 1.6 for val/test)
            
        Returns:
            Dictionary mapping dataset names to their optimal batch sizes
        """
        if not datasets:
            return {}
            
        dataset_sizes = {name: len(ds) for name, ds in datasets.items()}
        num_datasets = len(datasets)
        
        # Target total batch size (samples processed simultaneously across all datasets)
        target_total_batch_size = int(self.batch_size * batch_size_multiplier)
        
        # Base allocation per dataset
        base_per_loader_bs = max(1, target_total_batch_size // max(1, num_datasets))
        
        # First pass: allocate based on dataset size constraints
        batch_sizes = {}
        used_samples = 0
        
        for name, size in dataset_sizes.items():
            # Small datasets get their actual size, larger datasets get base allocation
            allocated = min(size, base_per_loader_bs)
            batch_sizes[name] = allocated
            used_samples += allocated
        
        # Second pass: redistribute unused samples to larger datasets
        remaining_samples = target_total_batch_size - used_samples
        
        if remaining_samples > 0:
            # Find datasets that can accept more samples (size > current allocation)
            expandable_datasets = [
                (name, size - batch_sizes[name]) 
                for name, size in dataset_sizes.items() 
                if size > batch_sizes[name]
            ]
            
            # Sort by how much extra capacity they have (largest first)
            expandable_datasets.sort(key=lambda x: x[1], reverse=True)
            
            # Distribute remaining samples
            for name, extra_capacity in expandable_datasets:
                if remaining_samples <= 0:
                    break
                    
                # Give this dataset some extra samples (up to its capacity)
                extra_samples = min(remaining_samples, extra_capacity)
                batch_sizes[name] += extra_samples
                remaining_samples -= extra_samples
        
        return batch_sizes

    def train_dataloader(self):
        batch_sizes = self._calculate_per_loader_batch_size(self.train_dataset, batch_size_multiplier=1.0)
        
        total_batch_size = sum(batch_sizes.values())
        
        rank_zero_info(f"Training batch config: global_bs={self.batch_size}, num_datasets={len(self.train_dataset)}, "
                      f"total_batch_size={total_batch_size}")
        
        loaders = {}
        for name, ds in self.train_dataset.items():
            batch_size = batch_sizes.get(name, 1)
            loader = get_loader_from_dataset(ds, batch_size=batch_size, shuffle=True)
            num_batches = len(loader)
            rank_zero_info(f"Dataset '{name}': size={len(ds)}, batch_size={batch_size}, batches={num_batches}")
            
            # Only include datasets that actually have batches
            if num_batches > 0:
                loaders[name] = loader
            else:
                rank_zero_info(f"WARNING: Skipping dataset '{name}' - no batches created")
        
        if not loaders:
            raise RuntimeError("No datasets have any batches! Check dataset sizes and batch configuration.")
        
        # Use max_size_cycle to ensure training continues even if some datasets are smaller
        combined_loader = CombinedLoader(loaders, mode="max_size_cycle")
        return combined_loader

    def val_dataloader(self):
        batch_sizes = self._calculate_per_loader_batch_size(self.val_dataset, batch_size_multiplier=1.6)
        
        loaders = {
            name: get_loader_from_dataset(ds, batch_size=batch_sizes.get(name, 1), shuffle=False)
            for name, ds in self.val_dataset.items()
        }
        return CombinedLoader(loaders, mode="min_size")
    
    def test_dataloader(self):
        batch_sizes = self._calculate_per_loader_batch_size(self.test_dataset, batch_size_multiplier=1.6)
        
        loaders = {
            name: get_loader_from_dataset(ds, batch_size=batch_sizes.get(name, 1), shuffle=False)
            for name, ds in self.test_dataset.items()
        }
        return CombinedLoader(loaders, mode="min_size")

class InferenceTraceSeqEWDataloader(LightningDataModule):
    def __init__(
        self,
        data_dirs: list[str],
        drv_snp: str = None,
        odt_snp: str = None,
        batch_size: int = 100,
        bound_path: str = None,
        scaler_path: str = None,
        ignore_snp: bool = False,
        training_config_keys: list | None = None,
    ):
        super().__init__()
        self.data_dirs = data_dirs
        self.drv_snp = drv_snp
        self.odt_snp = odt_snp
        self.batch_size = batch_size
        self.bound_path = bound_path
        self.scaler_path = scaler_path
        self.ignore_snp = ignore_snp
        self.training_config_keys = training_config_keys

    def setup(self, stage=None):
        # Initialize processor
        processor = CSVProcessor()

        # Load scaler with training config_keys metadata
        from common.parameters import (
            process_boundary_for_inference,
            validate_boundary_dimensions,
            get_directions_from_boundary_json
        )
        
        scaler_path_obj = Path(self.scaler_path)
        if is_enhanced_scaler(scaler_path_obj):
            scalers, training_config_keys = load_scaler_with_config_keys(scaler_path_obj)
            rank_zero_info(f"Loaded enhanced scaler with training config_keys: {training_config_keys}")
        else:
            rank_zero_info(f"Legacy scaler detected at {self.scaler_path}. Using provided training_config_keys.")
            if not self.training_config_keys:
                raise ValueError(
                    "Legacy scaler file detected, but no 'training_config_keys' were provided "
                    "to the dataloader. Please provide them in the inference config."
                )
            scalers = torch.load(self.scaler_path, weights_only=False)
            training_config_keys = self.training_config_keys
        
        # Process boundary JSON for inference compatibility
        self.boundary, boundary_values, self.config_keys = process_boundary_for_inference(
            self.bound_path, training_config_keys
        )
        rank_zero_info(f"Processed boundary with {len(boundary_values)} parameters: {self.config_keys}")
        
        # Validate dimensions match
        validate_boundary_dimensions(boundary_values, scalers, self.config_keys)
        rank_zero_info(f"✓ Boundary dimensions validated: {len(boundary_values)} parameters")

        # Handle SNP loading based on ignore_snp flag
        if self.ignore_snp:
            rank_zero_info("Using dummy SNP data (ignore_snp=True)")
            # Create dummy SNP data with minimal structure
            drv = DummySNP()
            odt = DummySNP()
            directions = get_directions_from_boundary_json(self.bound_path, default_ports=4)
        else:
            if self.drv_snp is None or self.odt_snp is None:
                raise ValueError("drv_snp and odt_snp must be provided when ignore_snp=False")
            
            drv = read_snp(Path(self.drv_snp))
            odt = read_snp(Path(self.odt_snp))
            assert drv.s.shape[-1] == odt.s.shape[-1], \
                f"DRV {self.drv_snp} and ODT {self.odt_snp} must match ports."
            
            directions = get_directions_from_boundary_json(self.bound_path, default_ports=drv.s.shape[-1])

        # Convert list to dict with directory names as keys for CombinedLoader compatibility
        if isinstance(self.data_dirs, list):
            data_dirs_dict = {Path(dir_path).name: dir_path for dir_path in self.data_dirs}
        else:
            data_dirs_dict = self.data_dirs
            
        csv_paths = processor.locate(data_dirs_dict)
        
        self.predict_dataset = {}
        for name, csv_path in csv_paths.items():
            case_id, input_arr = processor.parse(csv_path)
            rank_zero_info(f"Input array: {input_arr.shape}")
            
            # Create a single configuration array with shape (1, 1, n_parameters) to match training format
            boundary_array = np.array([[boundary_values]], dtype=np.float64)
            
            ds = InferenceTraceEWDataset(input_arr, directions, boundary_array, drv, odt)
            self.predict_dataset[name] = ds.transform(*scalers)

    def _calculate_inference_batch_size(self, datasets: dict) -> int:
        """
        Calculate appropriate batch size for inference datasets.
        
        Args:
            datasets: Dictionary of datasets to calculate batch size for
            
        Returns:
            Appropriate batch size ensuring all datasets can form batches
        """
        if not datasets:
            return 1
            
        dataset_sizes = [len(ds) for ds in datasets.values()]
        min_dataset_size = min(dataset_sizes)
        
        # For inference, we don't need to divide by number of datasets
        # Just ensure batch size doesn't exceed smallest dataset
        effective_batch_size = min(self.batch_size, min_dataset_size)
        effective_batch_size = max(1, effective_batch_size)
        
        # Log if batch size was adjusted
        if effective_batch_size != self.batch_size:
            rank_zero_info(f"Inference batch size adjusted: {self.batch_size} → {effective_batch_size} "
                          f"(limited by smallest dataset: {min_dataset_size} samples)")
        
        return effective_batch_size

    def predict_dataloader(self):
        batch_size = self._calculate_inference_batch_size(self.predict_dataset)
        
        rank_zero_info(f"Inference batch config: requested_bs={self.batch_size}, "
                      f"effective_bs={batch_size}, num_datasets={len(self.predict_dataset)}")
        
        loaders = {}
        for name, ds in self.predict_dataset.items():
            loader = get_loader_from_dataset(dataset=ds, batch_size=batch_size, shuffle=False)
            num_batches = len(loader)
            rank_zero_info(f"Inference dataset '{name}': size={len(ds)}, batches={num_batches}")
            
            # Only include datasets that actually have batches
            if num_batches > 0:
                loaders[name] = loader
            else:
                rank_zero_info(f"WARNING: Skipping inference dataset '{name}' - no batches created")
        
        if not loaders:
            raise RuntimeError("No inference datasets have any batches! Check dataset sizes and batch configuration.")
        
        return CombinedLoader(loaders)