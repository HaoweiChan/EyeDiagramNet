import random
import torch
import torch.multiprocessing as mp
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from lightning.pytorch.utilities.rank_zero import rank_zero_info

def collate_snp_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Custom collate function for SNP batches with variable port sizes"""
    snp_list = [item['snp_vert'] for item in batch]
    
    # Check if all SNPs have the same shape
    shapes = [snp.shape for snp in snp_list]
    if len(set(shapes)) == 1:
        # Same shape, can stack normally
        return {'snp_vert': torch.stack(snp_list)}
    else:
        # Different shapes - pad to the maximum shape
        # Find maximum dimensions
        max_freq = max(s[0] for s in shapes)
        max_p1 = max(s[1] for s in shapes)
        max_p2 = max(s[2] for s in shapes)
        
        # Pad each tensor to the maximum shape
        padded_snps = []
        for snp in snp_list:
            f, p1, p2 = snp.shape
            # Calculate padding for each dimension (pad_left, pad_right)
            pad_f = (0, max_freq - f)
            pad_p1 = (0, max_p1 - p1)
            pad_p2 = (0, max_p2 - p2)
            
            # Pad the real and imaginary parts separately
            snp_real = torch.nn.functional.pad(snp.real, (pad_p2[0], pad_p2[1], pad_p1[0], pad_p1[1], pad_f[0], pad_f[1]), value=0)
            snp_imag = torch.nn.functional.pad(snp.imag, (pad_p2[0], pad_p2[1], pad_p1[0], pad_p1[1], pad_f[0], pad_f[1]), value=0)
            padded_snp = torch.complex(snp_real, snp_imag)
            padded_snps.append(padded_snp)
        
        return {'snp_vert': torch.stack(padded_snps)}

class SNPDataset(Dataset):
    """Dataset for loading S-parameter files with support for lazy loading and shape grouping."""

    def __init__(
        self,
        file_paths: List[Path],
        cache_mode: str = 'lazy',  # 'all', 'lazy', or 'none'
        cache_size: Optional[int] = 1000,  # Max number of files to cache when using lazy mode
        expected_shape: Optional[Tuple[int, int, int]] = None  # Expected (F, P, P) shape for this dataset
    ):
        super().__init__()
        self.file_paths = sorted(file_paths)
        self.cache_mode = cache_mode
        self.cache_size = cache_size if cache_mode == 'lazy' else None
        self.expected_shape = expected_shape
        
        self.cache = {}
        self.access_count = {}  # Track access frequency for LRU cache
        
        if self.cache_mode == 'all':
            rank_zero_info(f"Pre-caching all {len(self.file_paths)} SNP files into memory...")
            # Set sharing strategy for complex tensors
            mp.set_sharing_strategy('file_system')
            
            for i, file_path in enumerate(file_paths):
                snp_tensor = self._read_snp_file(file_path)
                # Move tensor to shared memory
                snp_tensor.share_memory_()
                self.cache[i] = snp_tensor
                
                if (i + 1) % 100 == 0:
                    rank_zero_info(f"Cached {i + 1}/{len(self.file_paths)} files...")
                    
            rank_zero_info(f"Finished caching all {len(self.file_paths)} SNP files.")

    def _read_snp_file(self, file_path: Path) -> torch.Tensor:
        from common.signal_utils import read_snp
        network = read_snp(file_path)
        return torch.tensor(network.s, dtype=torch.complex64)

    def _manage_cache(self, idx: int, snp_tensor: torch.Tensor):
        """Manage LRU cache for lazy loading mode"""
        if self.cache_mode == 'lazy' and self.cache_size is not None:
            # Add to cache
            self.cache[idx] = snp_tensor
            self.access_count[idx] = self.access_count.get(idx, 0) + 1
            
            # If cache is full, remove least recently used item
            if len(self.cache) > self.cache_size:
                # Find least recently used item (lowest access count)
                lru_idx = min(self.cache.keys(), key=lambda k: self.access_count.get(k, 0))
                if lru_idx != idx:  # Don't remove the item we just added
                    del self.cache[lru_idx]
                    del self.access_count[lru_idx]

    def __len__(self) -> int:
        return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Check if item is in cache
        if idx in self.cache:
            self.access_count[idx] = self.access_count.get(idx, 0) + 1
            if self.cache_mode == 'all':
                # Return a clone for shared memory to avoid in-place modifications
                return {'snp_vert': self.cache[idx].clone()}
            else:
                return {'snp_vert': self.cache[idx]}
        
        # Load from file
        file_path = self.file_paths[idx]
        snp_tensor = self._read_snp_file(file_path)
        
        # Manage cache for lazy loading
        self._manage_cache(idx, snp_tensor)
        
        return {'snp_vert': snp_tensor}

class SNPDataModule(LightningDataModule):
    """DataModule for loading SNP files with support for different shapes across directories."""
    
    def __init__(
        self,
        data_dirs: List[str],
        file_pattern: str = "*.s*p",
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        cache_mode: str = 'lazy',  # 'all', 'lazy', or 'none'
        cache_size: int = 1000,  # For lazy mode
        group_by_shape: bool = True,  # Group files by shape to avoid padding
        sample_files_for_shape: int = 10  # Number of files to sample to determine shape
    ):
        super().__init__()
        self.save_hyperparameters()
    
    def _get_snp_shape(self, file_path: Path) -> Tuple[int, int, int]:
        """Get the shape of an SNP file without loading all data"""
        from common.signal_utils import read_snp
        network = read_snp(file_path)
        return network.s.shape
    
    def setup(self, stage: Optional[str] = None):
        """Finds all S-parameter files and groups them by shape if needed."""
        # Only load data once in the main process
        if hasattr(self, 'train_dataset'):
            return
        
        # Find all files by directory
        files_by_dir = {}
        for data_dir in self.hparams.data_dirs:
            data_path = Path(data_dir)
            if not data_path.exists():
                rank_zero_info(f"Warning: Directory {data_dir} does not exist, skipping...")
                continue
            
            dir_files = []
            # Only look in the specified directory, not subdirectories
            for file_path in data_path.iterdir():
                if file_path.is_file():
                    # Case-insensitive pattern matching
                    import fnmatch
                    if fnmatch.fnmatch(file_path.name.lower(), self.hparams.file_pattern.lower()):
                        dir_files.append(file_path)
            
            if dir_files:
                files_by_dir[data_dir] = dir_files
                rank_zero_info(f"Found {len(dir_files)} files in {data_dir}")
        
        if not files_by_dir:
            raise FileNotFoundError(f"No files found for pattern '{self.hparams.file_pattern}'")
        
        # Group files by shape if enabled
        if self.hparams.group_by_shape:
            shape_to_files = {}
            
            for data_dir, files in files_by_dir.items():
                # Sample a few files to determine the shape for this directory
                sample_size = min(self.hparams.sample_files_for_shape, len(files))
                sample_files = random.sample(files, sample_size)
                
                # Get shapes of sampled files
                shapes = []
                for file_path in sample_files:
                    try:
                        shape = self._get_snp_shape(file_path)
                        shapes.append(shape)
                    except Exception as e:
                        rank_zero_info(f"Error reading shape from {file_path}: {e}")
                
                if shapes:
                    # Check if all sampled files have the same shape
                    if len(set(shapes)) == 1:
                        shape = shapes[0]
                        rank_zero_info(f"Directory {data_dir} has consistent shape: {shape}")
                        
                        if shape not in shape_to_files:
                            shape_to_files[shape] = []
                        shape_to_files[shape].extend(files)
                    else:
                        rank_zero_info(f"Warning: Directory {data_dir} has mixed shapes: {set(shapes)}")
                        # Add each file individually after checking its shape
                        for file_path in files:
                            try:
                                shape = self._get_snp_shape(file_path)
                                if shape not in shape_to_files:
                                    shape_to_files[shape] = []
                                shape_to_files[shape].append(file_path)
                            except Exception as e:
                                rank_zero_info(f"Skipping file {file_path}: {e}")
            
            # Create separate datasets for each shape
            datasets = []
            for shape, files in shape_to_files.items():
                rank_zero_info(f"Creating dataset for shape {shape} with {len(files)} files")
                dataset = SNPDataset(
                    files,
                    cache_mode=self.hparams.cache_mode,
                    cache_size=self.hparams.cache_size,
                    expected_shape=shape
                )
                datasets.append(dataset)
            
            # Concatenate all datasets
            if len(datasets) == 1:
                self.train_dataset = datasets[0]
            else:
                self.train_dataset = ConcatDataset(datasets)
                rank_zero_info(f"Created concatenated dataset with {len(datasets)} shape groups")
        else:
            # Don't group by shape - use all files together
            all_files = []
            for files in files_by_dir.values():
                all_files.extend(files)
            
            rank_zero_info(f"Creating single dataset with {len(all_files)} files (no shape grouping)")
            self.train_dataset = SNPDataset(
                all_files,
                cache_mode=self.hparams.cache_mode,
                cache_size=self.hparams.cache_size
            )
    
    def train_dataloader(self):
        # Use num_workers based on cache mode
        if self.hparams.cache_mode == 'all':
            num_workers = 0  # Shared memory, no workers needed
        else:
            num_workers = self.hparams.num_workers  # Can use workers for lazy/none modes
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=num_workers > 0,
            collate_fn=collate_snp_batch
        )