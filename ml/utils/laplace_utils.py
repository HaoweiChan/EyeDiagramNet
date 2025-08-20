import torch
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from lightning.pytorch.utilities.combined_loader import CombinedLoader


class LaplaceDataLoaderWrapper:
    """
    Wrapper for DataLoader to make it compatible with Laplace library.
    
    The Laplace library expects a DataLoader that yields (X, y) tuples where:
    - X is the input to the model (can be a tuple of tensors)
    - y is the target tensor
    
    This wrapper handles both dictionary-based batches (from CombinedLoader) 
    and regular tuple-based batches from single dataloaders.
    """
    
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device
        
        # Expose the dataset attribute for Laplace library
        if hasattr(dataloader, 'dataset'):
            self.dataset = dataloader.dataset
        elif isinstance(dataloader, CombinedLoader):
            total_len = 0
            valid_dataset_found = False
            try:
                # Attempt to access dataloader.loaders
                loaders_collection = dataloader.loaders
                if isinstance(loaders_collection, dict):
                    for _loader in loaders_collection.values():
                        if hasattr(_loader, 'dataset') and _loader.dataset is not None:
                            total_len += len(_loader.dataset)
                            valid_dataset_found = True
                elif isinstance(loaders_collection, (list, tuple)):
                    for _loader in loaders_collection:
                        if hasattr(_loader, 'dataset') and _loader.dataset is not None:
                            total_len += len(_loader.dataset)
                            valid_dataset_found = True
                else:
                    rank_zero_info(f"LaplaceDataLoaderWrapper: CombinedLoader.loaders is of unexpected type: {type(loaders_collection)}")
                    
            except AttributeError:
                rank_zero_info("LaplaceDataLoaderWrapper: CombinedLoader instance does not have 'loaders' attribute as expected.")
                # total_len remains 0, valid_dataset_found remains False

            if valid_dataset_found and total_len > 0:
                class _CombinedDatasetProxy:
                    def __init__(self, length): 
                        self._length = length
                    def __len__(self): 
                        return self._length
                        
                self.dataset = _CombinedDatasetProxy(total_len)
                rank_zero_info(f"LaplaceDataLoaderWrapper: Using CombinedDatasetProxy with total length {total_len} for CombinedLoader.")
            else:
                rank_zero_info("LaplaceDataLoaderWrapper: Could not determine dataset length for CombinedLoader from its sub-loaders. Using len(CombinedLoader) as batch count proxy.")
                # Fallback: Create a proxy whose length is the number of batches in CombinedLoader.
                # This is not N_samples, but might allow Laplace to iterate if N is only for progress bar.
                class _LenProxyDataset:
                    def __init__(self, loader_to_wrap): 
                        self.loader_to_wrap = loader_to_wrap
                    def __len__(self): 
                        return len(self.loader_to_wrap) 
                        
                self.dataset = _LenProxyDataset(dataloader)
        else:
            rank_zero_info("LaplaceDataLoaderWrapper: Wrapped dataloader is not CombinedLoader and has no 'dataset' attribute.")
            self.dataset = None 
        
        # Ensure self.dataset is always set, even if to a dummy one for safety
        if self.dataset is None:
            class _EmptyDataset:
                def __len__(self): 
                    return 0
                    
            self.dataset = _EmptyDataset()
            rank_zero_info("LaplaceDataLoaderWrapper: Fallback to an empty dataset proxy (length 0).")

    def __iter__(self):
        for batch, *_ in self.dataloader:
            if isinstance(batch, dict):
                # Handle dictionary batch (from CombinedLoader or regular training)
                for name, raw_data in batch.items():
                    # raw_data is a tuple of tensors: (trace_seq, direction, boundary, snp_vert, true_ew)
                    # For Laplace _find_last_layer, which expects a single tensor X,
                    # we yield only the primary input (trace_seq) and the target.
                    # raw_data is (trace_seq, direction, boundary, snp_vert, true_ew)
                    inputs = tuple(t.to(self.device) for t in raw_data[:-1])
                    targets = raw_data[-1].to(self.device).squeeze()
                    yield inputs, targets
            else:
                # Handle tuple batch from a single dataloaloader
                # batch is (trace_seq, direction, boundary, snp_vert, true_ew)
                inputs = tuple(t.to(self.device) for t in batch[:-1])
                targets = batch[-1].to(self.device).squeeze()
                yield inputs, targets
    
    def __len__(self):
        return len(self.dataloader)


def setup_laplace_dataloader(datamodule, device):
    """
    Create a Laplace-compatible dataloader from a Lightning datamodule.
    
    Args:
        datamodule: Lightning datamodule with train_dataloader() method
        device: Device to move tensors to
        
    Returns:
        LaplaceDataLoaderWrapper instance ready for Laplace training
    """
    train_loader = datamodule.train_dataloader()
    
    # If train_loader is a dictionary of dataloaders, wrap it with CombinedLoader.
    # Otherwise, use it directly. This makes the logic robust.
    if isinstance(train_loader, dict):
        combined_loader = CombinedLoader(train_loader, mode="max_size_cycle")
        return LaplaceDataLoaderWrapper(combined_loader, device)
    else:
        return LaplaceDataLoaderWrapper(train_loader, device)


def get_sample_inputs_from_datamodule(datamodule, device):
    """
    Extract sample inputs from datamodule for _ForwardWrapper initialization.
    
    Args:
        datamodule: Lightning datamodule
        device: Device to move tensors to
        
    Returns:
        Tuple of (sample_direction, sample_boundary, sample_snp_vert)
    """
    original_train_loader = datamodule.train_dataloader()
    sample_batch_dict, *_ = next(iter(original_train_loader))
    
    # Assuming CombinedLoader, so sample_batch_dict is a dict
    # Take the first available dataset's sample
    sample_raw_data = next(iter(sample_batch_dict.values()))

    # Get single sample (first item of the batch) for each component
    # raw_data is (trace_seq, direction, boundary, snp_vert, true_ew)
    sample_direction = sample_raw_data[1][0:1].to(device)
    sample_boundary = sample_raw_data[2][0:1].to(device)
    sample_snp_vert = sample_raw_data[3][0:1].to(device)
    
    return sample_direction, sample_boundary, sample_snp_vert
