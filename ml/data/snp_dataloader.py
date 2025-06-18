import os
import torch
import pickle
from lightning import LightningDataModule
from typing import Optional, List, Dict, Any
from torch.utils.data import Dataset, DataLoader

class SNPDataset(Dataset):
    """Dataset for SNP self-supervised learning"""
    
    def __init__(
        self,
        data_path: str,
        snp_key: str = 'snp_vert',
        max_samples: Optional[int] = None,
        cache_in_memory: bool = False
    ):
        """
        Args:
            data_path: Path to pickle file or directory containing pickle files
            snp_key: Key to extract SNP data from pickle dictionaries
            max_samples: Maximum number of samples to use
            cache_in_memory: Whether to cache all data in memory
        """
        self.data_path = data_path
        self.snp_key = snp_key
        self.max_samples = max_samples
        self.cache_in_memory = cache_in_memory
        
        # Load data paths
        if os.path.isfile(data_path):
            self.data_files = [data_path]
        else:
            self.data_files = [
                os.path.join(data_path, f) 
                for f in os.listdir(data_path) 
                if f.endswith('.pkl')
            ]
        
        # Load metadata to get total number of samples
        self.sample_indices = []
        self.cached_data = {} if cache_in_memory else None
        
        # First pass - count samples
        for file_idx, file_path in enumerate(self.data_files):
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            if isinstance(data, list):
                for sample_idx in range(len(data)):
                    self.sample_indices.append((file_idx, sample_idx))
            else:
                # Single sample file
                self.sample_indices.append((file_idx, 0))
        
        # Limit samples if requested
        if self.max_samples is not None:
            self.sample_indices = self.sample_indices[:self.max_samples]
        
        # Second pass - cache data if requested
        if self.cache_in_memory:
            for idx, (file_idx, sample_idx) in enumerate(self.sample_indices):
                with open(self.data_files[file_idx], 'rb') as f:
                    data = pickle.load(f)
                
                if isinstance(data, list):
                    sample_data = data[sample_idx]
                else:
                    sample_data = data
                
                self.cached_data[idx] = self._extract_snp(sample_data)
    
    def _extract_snp(self, data: Any) -> torch.Tensor:
        """Extract SNP data from various data formats"""
        if isinstance(data, dict):
            snp = data.get(self.snp_key)
            if snp is None:
                # Try alternative keys
                for key in ['snp_vertical', 'snp', 'vertical_snp']:
                    if key in data:
                        snp = data[key]
                        break
        elif isinstance(data, tuple) or isinstance(data, list):
            # Assume SNP is at a specific index (typically index 3)
            # Based on common data format: (trace_seq, direction, boundary, snp_vert, ...)
            if len(data) > 3:
                snp = data[3]
            else:
                raise ValueError(f"Cannot extract SNP from tuple/list of length {len(data)}")
        else:
            snp = data
        
        if snp is None:
            raise ValueError(f"Could not find SNP data with key '{self.snp_key}'")
        
        # Convert to tensor if needed
        if not isinstance(snp, torch.Tensor):
            snp = torch.tensor(snp, dtype=torch.complex64)
        
        return snp
    
    def __len__(self):
        return len(self.sample_indices)
    
    def __getitem__(self, idx):
        if self.cache_in_memory:
            return {'snp_vert': self.cached_data[idx]}
        
        file_idx, sample_idx = self.sample_indices[idx]
        file_path = self.data_files[file_idx]
        
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, list):
            sample_data = data[sample_idx]
        else:
            sample_data = data
        
        snp = self._extract_snp(sample_data)
        
        return {'snp_vert': snp}

class SNPDataModule(LightningDataModule):
    """Lightning DataModule for SNP self-supervised learning"""
    
    def __init__(
        self,
        data_dir: str,
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        snp_key: str = 'snp_vert',
        max_samples: Optional[int] = None,
        cache_in_memory: bool = False,
        seed: int = 42
    ):
        super().__init__()
        self.data_dir = data_dir
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.snp_key = snp_key
        self.max_samples = max_samples
        self.cache_in_memory = cache_in_memory
        self.seed = seed
        
        # Ensure splits sum to 1
        total_split = train_split + val_split + test_split
        if abs(total_split - 1.0) > 1e-6:
            raise ValueError(f"Splits must sum to 1.0, got {total_split}")
    
    def setup(self, stage: Optional[str] = None):
        """Setup train/val/test datasets"""
        # Check if we have split files
        train_file = os.path.join(self.data_dir, 'snp_train.pkl')
        val_file = os.path.join(self.data_dir, 'snp_val.pkl')
        test_file = os.path.join(self.data_dir, 'snp_test.pkl')
        
        if os.path.exists(train_file) and os.path.exists(val_file) and os.path.exists(test_file):
            # Use pre-split files
            self.train_dataset = SNPDataset(
                train_file,
                snp_key=self.snp_key,
                max_samples=self.max_samples,
                cache_in_memory=self.cache_in_memory
            )
            self.val_dataset = SNPDataset(
                val_file,
                snp_key=self.snp_key,
                max_samples=self.max_samples,
                cache_in_memory=self.cache_in_memory
            )
            self.test_dataset = SNPDataset(
                test_file,
                snp_key=self.snp_key,
                max_samples=self.max_samples,
                cache_in_memory=self.cache_in_memory
            )
            return
        
        # Otherwise, create full dataset and split
        full_dataset = SNPDataset(
            self.data_dir,
            snp_key=self.snp_key,
            max_samples=self.max_samples,
            cache_in_memory=self.cache_in_memory
        )
        
        # Calculate split sizes
        total_size = len(full_dataset)
        train_size = int(total_size * self.train_split)
        val_size = int(total_size * self.val_split)
        test_size = total_size - train_size - val_size
        
        # Split dataset
        generator = torch.Generator().manual_seed(self.seed)
        self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(
            full_dataset, 
            [train_size, val_size, test_size],
            generator=generator
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0
        )
    
    def predict_dataloader(self):
        """Predict dataloader uses full dataset without shuffling"""
        full_dataset = SNPDataset(
            self.data_dir,
            snp_key=self.snp_key,
            max_samples=self.max_samples,
            cache_in_memory=self.cache_in_memory
        )
        
        return DataLoader(
            full_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
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