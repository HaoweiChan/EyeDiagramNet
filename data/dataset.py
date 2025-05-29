import random
import psutil
import itertools
import numpy as np
from pathlib import Path, PosixPath

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader

from common.utils import log_info, read_snp, flip_snp, renumber_snp, greedy_covering_design
from common.trace_processor import TraceSequenceProcessor

def collate_fn(batch, pad_token=-1):
    sequences, labels, sel_ports = zip(*batch)

    max_length = max(len(seq) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        padded_seq = F.pad(seq, (0, 0, 0, max_length - seq.size(0)), value=pad_token)
        padded_sequences.append(padded_seq)

    padded_sequences = torch.stack(padded_sequences, dim=0)
    labels = torch.stack(labels, dim=0)
    sel_ports = torch.stack(sel_ports, dim=0)

    return padded_sequences, labels, sel_ports

def get_loader_from_dataset(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = False
):
    drop_last = shuffle

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        # pin_memory=True,
        num_workers=8,
        persistent_workers=True,
        drop_last=drop_last,
        # collate_fn=collate_fn
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
        train=False
    ):
        super().__init__()

        self.trace_seqs = torch.from_numpy(trace_seqs.copy()).float()
        self.directions = torch.from_numpy(directions).int()
        self.boundaries = torch.from_numpy(boundaries).float()

        self.vert_snps = vert_snps
        self.vert_cache = SharedMemoryCache()

        self.eye_widths = torch.from_numpy(eye_widths).float()
        self.repetition = self.boundaries.size(1)
        self.train = train

    def __len__(self):
        return len(self.trace_seqs) * self.repetition

    def __getitem__(self, index):
        seq_index = index // self.repetition
        bnd_index = index % self.repetition

        trace_seq = self.trace_seqs[seq_index]

        # Retrieve the left and right snps
        tx_vert_file, rx_vert_file = self.vert_snps[seq_index, bnd_index]
        tx_vert_snp = self.vert_snp(tx_vert_file)
        rx_vert_snp = self.vert_snp(rx_vert_file)
        vert_snp = torch.stack((tx_vert_snp, flip_snp(rx_vert_snp)))

        # Retrieve the boundary and eye width values
        direction = self.directions[seq_index, bnd_index]
        boundary = self.boundaries[seq_index, bnd_index]
        eye_width = self.eye_widths[seq_index, bnd_index]

        if self.train and random.random() > 0.5:
            trace_seq, direction, eye_width, vert_snp = \
                self.augment(trace_seq, direction, eye_width, vert_snp)
        return trace_seq, seq_index, direction, boundary, vert_snp, eye_width

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
            snp_data = read_snp(Path(snp_file))
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
        tx_snp,
        rx_snp
    ):
        super().__init__()

        self.trace_seqs = torch.from_numpy(trace_seqs.copy()).float()
        self.boundary = torch.from_numpy(boundary).float()
        self.direction = torch.from_numpy(direction).int()

        tx_snp = torch.from_numpy(tx_snp).to(torch.complex64)
        rx_snp = torch.from_numpy(rx_snp).to(torch.complex64)
        self.vert_snp = torch.stack((tx_snp, flip_snp(rx_snp)))

    def __len__(self):
        return len(self.trace_seqs)

    def __getitem__(self, index):
        trace_seq = self.trace_seqs[index]
        return trace_seq, self.direction, self.boundary, self.vert_snp

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
        log_info(f'Comb({self.num_ports}, {self.max_ports}): {len(self.pair_combinations)}. Used pairs per sequence: {self.pairs_per_seq}')

    def __len__(self):
        return len(self.trace_seqs) * self.pairs_per_seq

    def __getitem__(self, index):
        seq_index = index // self.pairs_per_seq
        trace_seq = self.trace_seqs[seq_index]
        snp_file = self.snps[seq_index]

        data = self.cache.get(snp_file)
        if data is None:
            snp = torch.from_numpy(read_snp(snp_file)).to(torch.complex64)
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
        
        # Optionally add noise for data augmentation
        # if add_noise:
        #     mean, std = 0., 0.1
        #     noise = torch.rand_like(self.trace_seqs[:, :, 2:]) * std + mean
        #     self.trace_seqs[:, :, 2:] += noise
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
        """Rearrange port alignment for easier slicing during training.

        Original port alignment is:
            0 == 2
            1 == 3

        New port alignment will be:
            0 == 1
            2 == 3
        """
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
        log_info(f'Comb({num_ports}, {max_ports}): {len(self.pair_combinations)}')

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