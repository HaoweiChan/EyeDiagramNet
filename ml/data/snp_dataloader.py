import os
import torch
import torch.multiprocessing as mp
from pathlib import Path
from typing import Optional, List, Dict
from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch.utilities.rank_zero import rank_zero_info

class SNPDataset(Dataset):
    """Dataset for loading S-parameter files directly with shared memory support."""

    def __init__(
        self,
        file_paths: List[Path]
    ):
        super().__init__()
        self.file_paths = sorted(file_paths)
        
        self.shared_tensors = []
        
        rank_zero_info(f"Caching {len(self.file_paths)} SNP files into shared memory...")
        # Set sharing strategy for complex tensors
        mp.set_sharing_strategy('file_system')
            
        for i, file_path in enumerate(file_paths):
            snp_tensor = self._read_snp_file(file_path)
            # Move tensor to shared memory
            snp_tensor.share_memory_()
            self.shared_tensors.append(snp_tensor)
                
            if (i + 1) % 100 == 0:
                rank_zero_info(f"Cached {i + 1}/{len(self.file_paths)} files...")
                
        rank_zero_info(f"Finished caching all {len(self.file_paths)} SNP files.")

    def _read_snp_file(self, file_path: Path) -> torch.Tensor:
        from common.signal_utils import read_snp
        network = read_snp(file_path)
        return torch.tensor(network.s, dtype=torch.complex64)

    def __len__(self) -> int:
        return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Return a clone to avoid in-place modifications affecting shared memory
        return {'snp_vert': self.shared_tensors[idx].clone()}

class SNPDataModule(LightningDataModule):
    """DataModule for loading SNP files directly from a directory for training."""
    
    def __init__(
        self,
        data_dirs: List[str],
        file_pattern: str = "*.s*p",
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False
    ):
        super().__init__()
        self.save_hyperparameters()
    
    def setup(self, stage: Optional[str] = None):
        """Finds all S-parameter files and assigns them to the training set."""
        # Only load data once in the main process
        if hasattr(self, 'train_dataset'):
            return
            
        all_file_paths = []
        for data_dir in self.hparams.data_dirs:
            # Walk through all subdirectories
            for root, dirs, files in os.walk(data_dir):
                for file in files:
                    # Check if file matches the pattern (case-insensitive)
                    # Convert pattern wildcards to regex
                    import re
                    import fnmatch
                    
                    # Use fnmatch for glob-style pattern matching (case-insensitive)
                    if fnmatch.fnmatch(file.lower(), self.hparams.file_pattern.lower()):
                        all_file_paths.append(Path(root) / file)
        
        if not all_file_paths:
            raise FileNotFoundError(f"No files found for pattern '{self.hparams.file_pattern}'")
            
        rank_zero_info(f"Found {len(all_file_paths)} SNP files matching pattern '{self.hparams.file_pattern}' (case-insensitive)")
            
        self.train_dataset = SNPDataset(all_file_paths)
    
    def train_dataloader(self):
        # Always use num_workers=0 with shared memory
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=False
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