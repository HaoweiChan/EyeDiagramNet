import psutil
import numpy as np
import pandas as pd
from pathlib import Path, PosixPath

import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from lightning import LightningDataModule
from lightning.pytorch.utilities import CombinedLoader
from lightning.pytorch.utilities.rank_zero import rank_zero_info

from ..utils.scaler import MinMaxScaler
from .processors import CSVProcessor
from common.signal_utils import read_snp, parse_snps, greedy_covering_design

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

class TraceDataset(Dataset):
    def __init__(self, trace_seqs, snps, max_ports=8, pairs_per_seq=4):
        super().__init__()
        self.cache = SharedMemoryCache()
        if isinstance(trace_seqs, np.ndarray):
            trace_seqs = torch.from_numpy(trace_seqs)
        self.trace_seqs = trace_seqs.float()

        if isinstance(snps[0], PosixPath):
            self.snps = snps
        else:
            self.snps = torch.tensor(snps, dtype=torch.complex64)
        self.num_ports = (self.trace_seqs[0,:,1] == 0).sum()
        self.max_ports = max_ports
        self.freq_slice = slice(0, 3000, 5)
        self.freq_slice = list(range(0,100)) + list(range(100,3000,5))

        self.pair_combinations = greedy_covering_design(self.num_ports // 2, self.max_ports // 2)
        self.pairs_per_seq = min(pairs_per_seq, len(self.pair_combinations))
        rank_zero_info(f'Comb({self.num_ports}, {self.max_ports}): {len(self.pair_combinations)}. Used pairs per sequence: {self.pairs_per_seq}')

    def __len__(self):
        return len(self.trace_seqs) * self.pairs_per_seq

    def __getitem__(self, index):
        seq_index = index // self.pairs_per_seq
        trace_seq = self.trace_seqs[seq_index]
        snp_file = self.snps[seq_index]

        data = self.cache.get(snp_file)
        if data is None:
            snp = torch.from_numpy(read_snp(snp_file).s).to(torch.complex64)
            snp = snp[self.freq_slice] # (F, P, P)
            snp = self.port_rearrange(snp)
            # Extract snp data up to max ports
            torch.manual_seed(index)
            idx = torch.randint(len(self.pair_combinations), (1,)).item()
            sel_ports = torch.tensor(self.pair_combinations[idx])
            snp = snp[:, sel_ports][:, :, sel_ports]

            data = (snp, sel_ports)
            if self.check_memory(snp):
                self.cache.store(snp_file, data)
        snp, sel_ports = data

        return self.augment(trace_seq, snp, sel_ports)

    def transform(self, scaler, add_noise=False):
        """Apply scaling transformations. Note: TraceDataset excludes spatial features."""
        # For TraceDataset, we scale everything except layer and type (exclude first 2 cols)
        # This dataset doesn't include spatial coordinates
        scalable_feats = self.trace_seqs[:, :, 2:]  # All features except layer and type
        feat_shape = scalable_feats.shape
        scalable_feats_flat = scalable_feats.reshape(-1, feat_shape[-1])
        scaled_feats = scaler.transform(scalable_feats_flat).reshape(feat_shape)
        
        self.trace_seqs[:, :, 2:] = scaled_feats.float()
        
        return self

    def augment(self, trace_seq, snp, sel_ports=None):
        # Extract signal-related indices and features
        signal_indices = torch.where(trace_seq[:, 1] == 0)[0]
        signal_min_idx, signal_max_idx = max(0, sel_ports[0] - 4), min(sel_ports[-1] + 4, len(signal_indices) - 1)
        signal_feats = trace_seq[signal_indices]

        # Extract layer-related information
        layer_count = torch.unique(trace_seq[:, 0], return_counts=True)[1]
        signal_layer_idx = torch.where(layer_count > 2)[0]
        signal_layer_start = layer_count.cumsum(0)[signal_layer_idx - 1][0]
        signal_layer_end = layer_count.cumsum(0)[signal_layer_idx][0]

        # Extract features for non-signal layers
        non_signal_layer_feats_front = trace_seq[:signal_layer_start]
        non_signal_layer_feats_back = trace_seq[signal_layer_end:]

        # Extract features for signal layer
        signal_layer_feats = signal_feats[signal_indices[signal_min_idx]: signal_indices[signal_max_idx] + 1]

        # Calculate new range and update features
        x_dim_min, x_dim_max = signal_feats[signal_min_idx, -3], signal_feats[signal_max_idx, -3]
        x_dim_max = x_dim_max - x_dim_min
        non_signal_layer_feats_front[:, -6] = new_range = x_dim_min
        non_signal_layer_feats_back[:, -6] = new_range

        # Concatenate features and update x dim
        new_trace_seq = torch.cat([non_signal_layer_feats_front, signal_layer_feats, non_signal_layer_feats_back], dim=0)
        widths = new_trace_seq[:, -1]
        layer_change = torch.cat([torch.tensor([1.]), torch.diff(new_trace_seq[:, 0])])
        layer_count = torch.unique(new_trace_seq[:, 0], return_counts=True)[1]
        cum_widths = torch.cat([torch.tensor([0.]), widths.cumsum(0)[:-1]])
        x_dim = cum_widths - cum_widths[layer_change.bool().repeat_interleave(layer_count)]
        new_trace_seq[:, -6] = x_dim.repeat_interleave(2)

        # Subtract min port numbers in sel ports
        sel_ports = sel_ports - signal_min_idx

        if torch.rand(1).item() > 0.5:
            seq_indices = torch.arange(new_trace_seq.size(0))
            rev_seq_indices = seq_indices.flip(0)

            snp_indices = torch.arange(snp.size(1))
            rev_snp_indices = snp_indices.flip(0)

            new_trace_seq = new_trace_seq[rev_seq_indices]
            snp = snp[:, rev_snp_indices][:, :, rev_snp_indices]

            sel_ports = self.num_ports - sel_ports.flip(0) - 1

        return new_trace_seq, snp, sel_ports

    def port_rearrange(self, snp):
        n_ports = snp.size(-1)
        indices = torch.tensor([i // 2 + (i % 2) * (n_ports // 2) for i in range(n_ports)])
        interleave_snp = snp[:, indices][:, :, indices]
        return interleave_snp

    def check_memory(self, data):
        data_size = data.element_size() * data.nelement()
        avail_mem = psutil.virtual_memory().available
        return avail_mem > data_size

class InferenceTraceDataset(Dataset):
    def __init__(self, trace_seqs, max_ports):
        super().__init__()

        if isinstance(trace_seqs, (np.ndarray, np.generic)):
            trace_seqs = torch.from_numpy(trace_seqs)
        self.trace_seqs = trace_seqs.float()
        self.max_ports = max_ports

        num_ports = (self.trace_seqs[0,:,1] == 0).sum()
        self.pair_combinations = greedy_covering_design(num_ports // 2, max_ports // 2)
        rank_zero_info(f'Comb({num_ports}, {max_ports}): {len(self.pair_combinations)}')

    def __len__(self):
        return len(self.trace_seqs) * len(self.pair_combinations)

    def __getitem__(self, index):
        seq_index = index // len(self.pair_combinations)
        trace_seq = self.trace_seqs[seq_index]
        sel_ports = torch.tensor(self.pair_combinations[index % len(self.pair_combinations)])
        return trace_seq, sel_ports

    def transform(self, scaler):
        """Apply scaling transformations. Note: InferenceTraceDataset excludes spatial features."""
        # For InferenceTraceDataset, we scale everything except layer and type
        scalable_feats = self.trace_seqs[:, :, 2:]  # All features except layer and type
        feat_shape = scalable_feats.shape
        scalable_feats_flat = scalable_feats.reshape(-1, feat_shape[-1])
        scaled_feats = scaler.transform(scalable_feats_flat).reshape(feat_shape)
        
        self.trace_seqs[:, :, 2:] = scaled_feats.float()
        return self

class TraceSeqDataLoader(LightningDataModule):
    def __init__(
        self,
        data_dir: dict[str, str],
        batch_size: int,
        test_size: float = 0.2,
        max_ports: int = 8,
        scaler_path: str = None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.test_size = test_size
        self.locate_input_csv_files(data_dir)
        self.max_ports = max_ports
        self.scaler_path = scaler_path
        self.train_dataset = {}
        self.val_dataset = {}

    def locate_input_csv_files(self, data_dir):
        self.input_csv_dir = {}
        for name, csv_dir in data_dir.items():
            rank_zero_info(f"Parsing data from {csv_dir}")
            patterns = ["input for AI*.csv", "*AI input data*.csv"]
            input_csv_dir = [
                d for pattern in patterns for d in Path(csv_dir).glob(pattern)
            ]
            assert len(input_csv_dir) == 1, f"Cannot find input file in {csv_dir}"
            self.input_csv_dir[name] = input_csv_dir[0]

    def replace_zero_length_with_max(self, row):
        max_val = row[row != 0].max()
        row[row == 0] = max_val
        return row

    def parse_input_csv(self, csv_dir):
        # Cast line types to integers
        unique_types = ('S', 'G', 'D')
        df = pd.read_csv(csv_dir, index_col=0, header=0)
        df = df.replace({col: {v: i for i, v in enumerate(unique_types)} for col in df.columns[df.columns.str.contains('Type_')]})

        def replace_zero_with_max(row):
            max_val = row[row != 0].max()
            row[row == 0] = max_val
            return row

        def load_and_preprocess_data(input_csv_dir):
            df = pd.read_csv(input_csv_dir, index_col=0, header=0)
            df = df.replace({col: {v: i for i, v in enumerate(unique_types)} for col in df.columns[df.columns.str.contains('Type_')]})
            return df

        def extract_features(df):
            layer_idx = df.columns.str.contains('Layer_').nonzero()[0]
            feat_dim = layer_idx[1] - layer_idx[0] - 3

            width_idx = df.columns.str.contains('W_').nonzero()[0]
            widths = df.iloc[:, width_idx]

            height_idx = df.columns.str.contains('H_').nonzero()[0]
            heights = df.iloc[:, height_idx]

            length_idx = df.columns.str.contains('L_').nonzero()[0]
            lengths = df.iloc[:, length_idx].apply(replace_zero_with_max, axis=1)

            df.drop(df.columns[np.concatenate((width_idx, height_idx, length_idx))], axis=1, inplace=True)

            return df, widths, heights, lengths, feat_dim

        def calculate_dimensions(df, widths, heights, lengths):
            layer_idx = df.columns.str.contains('Layer_').nonzero()[0]
            layer_change = df.iloc[:, layer_idx].diff(axis=1).fillna(1).values.astype(bool)
            layer_count = np.unique(df.iloc[:, layer_idx], return_counts=True)[1]
            layer_count = layer_count / len(df).astype(int)

            signal_layer_start = layer_count.cumsum()[np.where(layer_count != 1)[0] - 1][0]
            signal_layer_end = layer_count.cumsum()[np.where(layer_count != 1)[0]][0]
            signal_total_widths = widths.values.copy()
            new_widths = signal_total_widths[..., None]
            new_widths[:, signal_layer_end:] = signal_total_widths[..., None]
            new_widths[:, signal_layer_end:] = signal_total_widths[..., None]
            new_widths[:, signal_layer_end:] = signal_total_widths[..., None]
            cum_widths = widths.cumsum(1, axis=1, fill_value=0).values
            x_dim = cum_widths = np.repeat(cum_widths[layer_change].reshape((len(widths), -1)), layer_count, axis=1)

            cum_heights = heights.values[layer_change].reshape((len(heights), -1)).cumsum(1)
            cum_heights = np.roll(cum_heights, shift=1, axis=1)
            cum_heights[:, 0] = 0
            z_dim = np.repeat(cum_heights, layer_count, axis=1)

            y_dim = np.dstack([np.zeros_like(lengths), lengths]).reshape((len(lengths), -1))

            spatial_feats = np.dstack([new_widths, heights, lengths, x_dim, z_dim])

            return spatial_feats, y_dim

        df = load_and_preprocess_data(csv_dir)
        df, widths, heights, lengths, feat_dim = extract_features(df)
        spatial_feats, y_dim = calculate_dimensions(df, widths, heights, lengths)

        new_input_arr = np.concatenate([df.values.reshape(len(df), -1, feat_dim), spatial_feats], axis=2)
        new_input_arr = np.concatenate([np.repeat(new_input_arr, 2, axis=1), y_dim[..., None]], axis=2)

        return new_input_arr, df.index

    def setup(self, stage=None):
        fit_scaler = True
        try:
            self.seq_scaler = torch.load(self.scaler_path)
            rank_zero_info(f'Loaded scaler object from {self.scaler_path}')
            fit_scaler = False
        except (FileNotFoundError, AttributeError):
            self.seq_scaler = MinMaxScaler()
            rank_zero_info('Could not find scaler file in {self.scaler_path}, initiating new scaler object.')

        for name, csv_dir in self.input_csv_dir.items():
            input_arr, sample_idx = self.parse_input_csv(csv_dir)

            # Read output snps
            snp_dir = Path(csv_dir).parent
            if Path(snp_dir, 'npz').exists():
                snp_dir = snp_dir.joinpath("npz")
            elif Path(snp_dir, 'snp').exists():
                snp_dir = snp_dir.joinpath("snp")
            snps, missing_idx = parse_snps(snp_dir, sample_idx.astype(np.float32))
            for i in missing_idx:
                input_arr = np.delete(input_arr, i, axis=0)

            # Get number of ports
            one_snp = read_snp(snps[0])

            indices = np.arange(125)
            snps = [snps[i] for i in indices]
            input_arr = input_arr[indices]
            rank_zero_info(f'{name}| input_arr: {input_arr.shape}, SNP: ({len(snps)}*{one_snp.s.shape})')

            # Split train and validation set
            x_train, x_val, y_train, y_val = train_test_split(input_arr, snps, test_size=self.test_size)
            x_train, x_val = train_test_split(input_arr, snps, test_size=self.test_size, shuffle=False)
            # scale input but ignore layer and type features
            if fit_scaler:
                # For TraceSeqDataLoader, we scale all features except layer and type 
                # (this dataset doesn't include spatial coordinates)
                scalable_feats = input_arr[:, :, 2:]  # Skip layer and type
                seq_feats_flat = scalable_feats.reshape(-1, scalable_feats.shape[-1])
                self.seq_scaler.partial_fit(seq_feats_flat)

            # store dataset
            self.train_dataset[name] = TraceDataset(x_train, y_train, self.max_ports)
            self.val_dataset[name] = TraceDataset(x_val, y_val, self.max_ports)

        rank_zero_info(f'-----------Scaler-----------\n\
samples seen: {self.seq_scaler.n_samples_seen_}\n\
minimum: {self.seq_scaler.min_.round(decimals=2).tolist()}\n\
maximum: {self.seq_scaler.max_.round(decimals=2).tolist()}')
        for key in list(self.train_dataset.keys()):
            self.train_dataset[name] = self.train_dataset[name].transform(self.seq_scaler, add_noise=True)
            self.val_dataset[name] = self.val_dataset[name].transform(self.seq_scaler)

        torch.save(self.seq_scaler, Path(self.trainer.logger.log_dir).joinpath("scaler.pth"))

    def train_dataloader(self):
        return CombinedLoader(
            {
                name: get_loader_from_dataset(
                    dataset=dataset,
                    batch_size=self.batch_size // len(self.train_dataset),
                    shuffle=True,
                )
                for name, dataset in self.train_dataset.items()
            },
            'min_size'
        )

    def val_dataloader(self):
        return CombinedLoader(
            {
                name: get_loader_from_dataset(
                    dataset=dataset,
                    batch_size=int(self.batch_size * 1.6 // len(self.val_dataset)),
                )
                for name, dataset in self.val_dataset.items()
            },
            'min_size'
        )

class InferenceTraceSeqDataLoader(TraceSeqDataLoader):
    def __init__(
        self,
        data_dir: dict[str, str],
        batch_size: int,
        max_ports: int,
        scaler_path: str,
    ):
        super().__init__(data_dir, batch_size, max_ports=max_ports, scaler_path=scaler_path)
        self.predict_dataset = {}

    def setup(self, stage=None):
        self.seq_scaler = torch.load(self.scaler_path)
        rank_zero_info(f'Loaded scaler object from {self.scaler_path}')

        for name, csv_dir in self.input_csv_dir.items():
            input_arr, df = self.parse_input_csv(csv_dir)
            input_arr = input_arr[:100]
            rank_zero_info(f'{name}| input_arr: {input_arr.shape}')

            # store dataset
            self.predict_dataset[name] = InferenceTraceDataset(input_arr, self.max_ports)
        rank_zero_info(f'-----------Scaler-----------\n\
samples seen: {self.seq_scaler.n_samples_seen_}\n\
minimum: {self.seq_scaler.min_.round(decimals=2).tolist()}\n\
maximum: {self.seq_scaler.max_.round(decimals=2).tolist()}')
        for key in list(self.predict_dataset.keys()):
            self.predict_dataset[name] = self.predict_dataset[name].transform(self.seq_scaler)

    def predict_dataloader(self):
        return CombinedLoader(
            {
                name: get_loader_from_dataset(
                    dataset=dataset,
                    batch_size=self.batch_size * 2 // len(self.predict_dataset),
                    shuffle=False,
                )
                for name, dataset in self.predict_dataset.items()
            },
            'sequential'
        )