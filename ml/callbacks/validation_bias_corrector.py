"""
Validation Bias Corrector Callback

Automatically computes and applies validation bias correction per dataset to address
systematic prediction bias between training and validation modes.

Usage example:
    from ml.callbacks.validation_bias_corrector import ValidationBiasCorrector
    
    # Add to trainer callbacks
    trainer = pl.Trainer(
        callbacks=[
            ValidationBiasCorrector(
                warm_up_epochs=5,      # Skip first 5 epochs
                update_frequency=1,    # Update every epoch
                ema_alpha=0.1         # Smooth bias updates
            )
        ]
    )
    
    # Enable bias correction in your module's hyperparameters
    module = TraceEWModule(model, enable_bias_correction=True)
    
    # The callback will automatically:
    # - Collect validation predictions and targets per dataset
    # - Compute bias = mean(targets - predictions) per dataset
    # - Store bias values in module state (saved in checkpoint)
    # - Apply correction during validation/inference (not training)
"""

import torch
import traceback
from collections import defaultdict
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_info


class ValidationBiasCorrector(Callback):
    """
    Callback to automatically compute and apply validation bias correction per dataset.
    
    Addresses systematic prediction bias between training and validation modes
    (e.g., due to data augmentation differences).
    
    Args:
        warm_up_epochs: Skip optimization for first N epochs (default: 5)
        update_frequency: Update bias every N epochs (default: 1) 
        ema_alpha: Exponential moving average factor for smoothing bias updates (default: 0.1)
        enable_correction: Whether to apply bias correction during inference (default: True)
    """
    
    def __init__(
        self, 
        warm_up_epochs: int = 5,
        update_frequency: int = 1,
        ema_alpha: float = 0.1,
        enable_correction: bool = True,
    ):
        super().__init__()
        self.warm_up_epochs = warm_up_epochs
        self.update_frequency = update_frequency
        self.ema_alpha = float(max(0.0, min(1.0, ema_alpha)))
        self.enable_correction = enable_correction
        
        # Cache for validation data per dataset
        self.val_cache = defaultdict(lambda: {
            "predictions": [],
            "targets": []
        })
        
        # Track bias evolution per dataset
        self.initial_bias = {}
        self.current_bias = {}
    
    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Initialize bias correction state in the module."""
        if not hasattr(pl_module, 'validation_bias'):
            pl_module.validation_bias = {}
            pl_module.bias_correction_enabled = self.enable_correction
            
        self.initial_bias = pl_module.validation_bias.copy()
        self.current_bias = pl_module.validation_bias.copy()
    
    def on_validation_batch_end(
        self, 
        trainer: Trainer, 
        pl_module: LightningModule, 
        outputs, 
        batch, 
        batch_idx: int, 
        dataloader_idx: int = 0
    ) -> None:
        """
        Collect validation data per dataset for bias computation.
        
        CRITICAL: Uses the same predictions from the main validation step to avoid
        the dual-forward-pass bug that caused exponentially growing validation MAE.
        """
        if not getattr(pl_module, 'bias_correction_enabled', True):
            return
            
        # Use raw predictions stored by the main validation step (same forward pass!)
        if not hasattr(pl_module, '_val_raw_preds'):
            return
            
        for dataset_name, stored_data in pl_module._val_raw_preds.items():
            pred_ew_scaled = stored_data['pred_ew']  # Already scaled
            true_ew_scaled = stored_data['true_ew']  # Already scaled
            
            # Only use samples where eye is open for bias computation  
            true_prob = (true_ew_scaled > 0).float()
            open_eye_mask = true_prob.bool()
            
            if open_eye_mask.any():
                pred_masked = pred_ew_scaled[open_eye_mask]
                true_masked = true_ew_scaled[open_eye_mask]
                
                # Cache for bias computation (detach and move to CPU)
                self.val_cache[dataset_name]["predictions"].append(pred_masked.flatten().detach().cpu())
                self.val_cache[dataset_name]["targets"].append(true_masked.flatten().detach().cpu())
        
        # Clear stored predictions after use to free memory
        if hasattr(pl_module, '_val_raw_preds'):
            del pl_module._val_raw_preds
    
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Compute and update bias correction per dataset."""
        if not getattr(pl_module, 'bias_correction_enabled', True):
            return
            
        # Only run on rank 0 for distributed training
        if trainer.global_rank != 0:
            self._reset_cache()
            return
            
        # Check warm-up period
        if trainer.current_epoch < self.warm_up_epochs:
            rank_zero_info(f"Epoch {trainer.current_epoch}: Skipping bias correction (warm-up)")
            self._reset_cache()
            return
            
        # Check update frequency
        if trainer.current_epoch % self.update_frequency != 0:
            self._reset_cache()
            return
            
        # Compute bias for each dataset
        updated_datasets = []
        for dataset_name in self.val_cache:
            new_bias = self._compute_bias(dataset_name)
            if new_bias is not None:
                # Smooth with EMA to stabilize bias evolution
                current_bias = self.current_bias.get(dataset_name, 0.0)
                smoothed_bias = (1.0 - self.ema_alpha) * current_bias + self.ema_alpha * new_bias
                
                # Update module bias state
                pl_module.validation_bias[dataset_name] = smoothed_bias
                self.current_bias[dataset_name] = smoothed_bias
                updated_datasets.append(dataset_name)
                
                # Log bias per dataset
                pl_module.log(f"val/bias_{dataset_name}", smoothed_bias, prog_bar=False, on_epoch=True, sync_dist=False)
        
        if updated_datasets:
            rank_zero_info(f"Updated validation bias for datasets: {updated_datasets}")
            
        self._reset_cache()
    
    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Print final bias evolution per dataset."""
        if trainer.global_rank == 0 and self.current_bias:
            print(f"\nValidation Bias Correction Summary:")
            for dataset_name, final_bias in self.current_bias.items():
                initial_bias = self.initial_bias.get(dataset_name, 0.0)
                change = final_bias - initial_bias
                print(f"  {dataset_name}: {initial_bias:.3f} → {final_bias:.3f} (Δ={change:+.3f})")
    
    def _compute_bias(self, dataset_name):
        """Compute bias = mean(targets - predictions) for a dataset."""
        if dataset_name not in self.val_cache or not self.val_cache[dataset_name]["predictions"]:
            return None
            
        try:
            # Concatenate cached data for this dataset
            predictions = torch.cat(self.val_cache[dataset_name]["predictions"])
            targets = torch.cat(self.val_cache[dataset_name]["targets"])
            
            if len(predictions) == 0:
                return None
                
            # Compute bias = mean(targets - predictions)
            bias = (targets - predictions).mean().item()
            return bias
            
        except Exception as e:
            rank_zero_info(f"Error computing bias for {dataset_name}: {str(e)}\nFull traceback:\n{traceback.format_exc()}")
            return None
    
    def _reset_cache(self):
        """Reset validation cache for next epoch."""
        self.val_cache = defaultdict(lambda: {
            "predictions": [],
            "targets": []
        })


def apply_validation_bias_correction(pl_module: LightningModule, predictions: torch.Tensor, dataset_name: str = None) -> torch.Tensor:
    """
    Helper function to apply validation bias correction to predictions.
    
    Args:
        pl_module: Lightning module with validation_bias attribute
        predictions: Raw predictions tensor
        dataset_name: Name of dataset (if None, uses first available bias)
        
    Returns:
        Bias-corrected predictions
    """
    if not getattr(pl_module, 'bias_correction_enabled', False):
        return predictions
        
    if not hasattr(pl_module, 'validation_bias') or not pl_module.validation_bias:
        return predictions
        
    # Get bias for specific dataset, or use average bias
    bias = 0.0
    if dataset_name and dataset_name in pl_module.validation_bias:
        # Use specific dataset bias
        bias = pl_module.validation_bias[dataset_name]
    elif pl_module.validation_bias:
        # Use average bias across all datasets when dataset_name is None
        bias = sum(pl_module.validation_bias.values()) / len(pl_module.validation_bias)
    
    # Apply bias correction (only during eval mode)
    if not pl_module.training and bias != 0.0:
        return predictions + bias
    else:
        return predictions
