import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from common.signal_utils import log_info, read_snp, flip_snp

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
        # collate_fn=collate_fn,
    )

    return loader

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
        num = len(self.trace_seqs)

        feat_dim = self.trace_seqs.size(-1) - 4
        feat = self.trace_seqs[:, :, 2:-2].reshape(-1, feat_dim)
        self.trace_seqs[:, :, 2:-2] = seq_scaler.transform(feat).reshape(num, -1, feat_dim)

        bound_dim = self.boundary.size(-1)
        boundary = self.boundary.reshape(-1, bound_dim)
        self.boundary = fix_scaler.transform(boundary).reshape(bound_dim).float()

        return self