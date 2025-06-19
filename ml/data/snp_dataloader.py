import os
import torch
import numpy as np
from lightning import LightningDataModule
from typing import Optional, List, Dict
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from glob import glob
from sklearn.model_selection import train_test_split

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
        freqs = network.f
        snp_data = network.s
        
        num_freqs = len(freqs)
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
    """DataModule for loading SNP files directly from a directory."""
    
    def __init__(
        self,
        data_dir: str,
        file_pattern: str = "*.s*p",
        val_split: float = 0.2,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        max_freq_points: Optional[int] = None,
        cache_in_memory: bool = True,
        seed: int = 42
    ):
        super().__init__()
        self.save_hyperparameters()
    
    def setup(self, stage: Optional[str] = None):
        """Finds and splits the S-parameter files."""
        file_paths = [Path(p) for p in glob(os.path.join(self.hparams.data_dir, self.hparams.file_pattern))]
        
        if not file_paths:
            raise FileNotFoundError(
                f"No files found matching pattern '{self.hparams.file_pattern}' "
                f"in directory '{self.hparams.data_dir}'"
            )
            
        train_paths, val_paths = train_test_split(
            file_paths,
            test_size=self.hparams.val_split,
            random_state=self.hparams.seed,
            shuffle=True
        )
        
        self.train_dataset = SNPDataset(
            train_paths,
            max_freq_points=self.hparams.max_freq_points,
            cache_in_memory=self.hparams.cache_in_memory
        )
        self.val_dataset = SNPDataset(
            val_paths,
            max_freq_points=self.hparams.max_freq_points,
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
    
    def val_dataloader(self):
        """Returns the validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
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