import os
import torch
import numpy as np
from lightning import LightningDataModule
from typing import Optional, List, Dict
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from glob import glob

class SNPDataset(Dataset):
    """Dataset for loading S-parameter files directly."""

    def __init__(
        self,
        file_paths: List[Path],
        max_freq_points: Optional[int] = None,
        cache_in_memory: bool = False
    ):
        """
        Args:
            file_paths: List of paths to S-parameter files.
            max_freq_points: Maximum number of frequency points to keep.
            cache_in_memory: Whether to cache all data in memory.
        """
        super().__init__()
        self.file_paths = sorted(file_paths) # Sort for determinism
        self.max_freq_points = max_freq_points
        self.cache_in_memory = cache_in_memory
        
        self.cached_data = []
        if self.cache_in_memory:
            print(f"Caching {len(self.file_paths)} SNP files into memory...")
            for file_path in self.file_paths:
                self.cached_data.append(self._read_snp_file(file_path))

    def _read_snp_file(self, file_path: Path) -> torch.Tensor:
        """Reads a single S-parameter file and returns a tensor."""
        # This import is local to avoid circular dependencies
        from common.signal_utils import read_snp
        
        network = read_snp(file_path)
        snp_data = network.s
        
        num_freqs = snp_data.shape[0]
        if self.max_freq_points and num_freqs > self.max_freq_points:
            indices = np.linspace(0, num_freqs - 1, self.max_freq_points, dtype=int)
            snp_data = snp_data[indices]
            
        return torch.tensor(snp_data, dtype=torch.complex64)

    def __len__(self) -> int:
        return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.cache_in_memory:
            # The key 'snp_vert' is maintained for consistency with the SSL module
            return {'snp_vert': self.cached_data[idx]}
        else:
            file_path = self.file_paths[idx]
            snp_tensor = self._read_snp_file(file_path)
            return {'snp_vert': snp_tensor}

class SNPDataModule(LightningDataModule):
    """DataModule for loading SNP files directly from a directory for training."""
    
    def __init__(
        self,
        data_dirs: List[str],
        file_pattern: str = "*.s*p",
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        cache_in_memory: bool = True
    ):
        super().__init__()
        self.save_hyperparameters()
        self.freq_length = None # Will be determined at runtime
    
    def prepare_data(self):
        """
        Inspects the first file to determine the frequency dimension.
        This is done in prepare_data to ensure it's only called on a single process.
        """
        all_file_paths = []
        for data_dir in self.hparams.data_dirs:
            all_file_paths.extend(
                [Path(p) for p in glob(os.path.join(data_dir, self.hparams.file_pattern))]
            )

        if not all_file_paths:
            raise FileNotFoundError(f"No files found matching '{self.hparams.file_pattern}' in {self.hparams.data_dirs}")

        from common.signal_utils import read_snp
        first_file = read_snp(all_file_paths[0])
        self.freq_length = first_file.s.shape[0]
        print(f"Determined frequency length from first file: {self.freq_length}")

    def setup(self, stage: Optional[str] = None):
        """Finds all S-parameter files and assigns them to the training set."""
        all_file_paths = []
        for data_dir in self.hparams.data_dirs:
            all_file_paths.extend(
                [Path(p) for p in glob(os.path.join(data_dir, self.hparams.file_pattern))]
            )
        
        self.train_dataset = SNPDataset(
            all_file_paths,
            max_freq_points=self.freq_length, # Use the determined freq_length
            cache_in_memory=self.hparams.cache_in_memory
        )
    
    def train_dataloader(self):
        """Returns the training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.num_workers > 0
        )


def collate_snp_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Custom collate function for SNP batches with variable port sizes"""
    snp_list = [item['snp_vert'] for item in batch]
    
    # Check if all SNPs have the same shape
    shapes = [snp.shape for snp in snp_list]
    if len(set(shapes)) == 1:
        # Same shape, can stack normally
        return {'snp_vert': torch.stack(snp_list)}
    else:
        # Different shapes, need to pad or handle differently
        # For now, raise an error - in practice you might want to pad
        raise ValueError(f"Inconsistent SNP shapes in batch: {shapes}")


class VariableSizeSNPDataModule(SNPDataModule):
    """DataModule that handles variable-sized SNP matrices"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.collate_fn = collate_snp_batch
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            collate_fn=self.collate_fn
        ) 