import os
import torch
from glob import glob
from pathlib import Path
from typing import Optional, List, Dict
from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader

class SNPDataset(Dataset):
    """Dataset for loading S-parameter files directly."""

    def __init__(
        self,
        file_paths: List[Path],
        cache_in_memory: bool = False
    ):
        super().__init__()
        self.file_paths = sorted(file_paths)
        self.cache_in_memory = cache_in_memory
        
        self.cached_data = []
        if self.cache_in_memory:
            print(f"Caching {len(self.file_paths)} SNP files into memory...")
            for file_path in self.file_paths:
                self.cached_data.append(self._read_snp_file(file_path))

    def _read_snp_file(self, file_path: Path) -> torch.Tensor:
        from common.signal_utils import read_snp
        network = read_snp(file_path)
        return torch.tensor(network.s, dtype=torch.complex64)

    def __len__(self) -> int:
        return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.cache_in_memory:
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
        cache_in_memory: bool = True,
        augmentation_config: Optional[Dict] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
    
    def setup(self, stage: Optional[str] = None):
        """Finds all S-parameter files and assigns them to the training set."""
        all_file_paths = []
        for data_dir in self.hparams.data_dirs:
            all_file_paths.extend(
                [Path(p) for p in glob(os.path.join(data_dir, self.hparams.file_pattern))]
            )
        
        if not all_file_paths:
            raise FileNotFoundError(f"No files found for pattern '{self.hparams.file_pattern}'")
            
        self.train_dataset = SNPDataset(
            all_file_paths,
            cache_in_memory=self.hparams.cache_in_memory
        )
    
    def train_dataloader(self):
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