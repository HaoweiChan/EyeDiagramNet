import os
from torch.utils.data import Dataset, DataLoader


def get_loader_from_dataset(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = False
):
    """Create DataLoader from dataset with optimized settings.
    
    Args:
        dataset: PyTorch Dataset instance
        batch_size: Desired batch size
        shuffle: Whether to shuffle the data
        
    Returns:
        DataLoader with optimized settings for the given dataset
    """
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
