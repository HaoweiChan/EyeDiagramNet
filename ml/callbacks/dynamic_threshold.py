"""
Dynamic Threshold Optimization Callback

Usage example:
    from ml.callbacks.dynamic_threshold import DynamicThresholdOptimizer
    
    # Add to trainer callbacks
    trainer = pl.Trainer(
        callbacks=[
            DynamicThresholdOptimizer(
                warm_up_epochs=3,      # Skip first 3 epochs
                update_frequency=1,    # Update every epoch
                grid_size=101         # Test 101 threshold values
            )
        ]
    )
    
    # Enable optimization in your module's hyperparameters
    module = TraceEWModule(model, optimize_threshold=True)
    
    # The callback will automatically:
    # - Collect validation predictions during training
    # - Find optimal threshold that maximizes F1 score
    # - Update module.hparams.ew_threshold dynamically
    # - Log val/tau_f1 metric to tensorboard
    # - Print final threshold evolution at training end
"""

import torch
import traceback
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_info


class DynamicThresholdOptimizer(Callback):
    """
    Callback to automatically optimize the ew_threshold during training to maximize F1 score.
    
    Collects validation predictions and labels, then sweeps through threshold candidates
    to find the F1-maximizing value. Only runs after warm-up epochs to avoid noisy
    early-training adjustments.
    
    Args:
        warm_up_epochs: Skip optimization for first N epochs (default: 3)
        update_frequency: Update threshold every N epochs (default: 1) 
        grid_size: Number of threshold candidates to test (default: 101)
    """
    
    def __init__(
        self, 
        warm_up_epochs: int = 3,
        update_frequency: int = 1,
        grid_size: int = 101,
        ema_alpha: float = 0.2,
    ):
        super().__init__()
        self.warm_up_epochs = warm_up_epochs
        self.update_frequency = update_frequency
        self.grid_size = grid_size
        # Exponential moving average factor for smoothing threshold updates
        # new_tau = (1 - ema_alpha) * prev_tau + ema_alpha * best_tau
        self.ema_alpha = float(max(0.0, min(1.0, ema_alpha)))
        
        # Cache for validation data
        self.val_cache = {
            "probs": [],
            "labels": []
        }
        
        # Track threshold evolution
        self.initial_threshold = None
        self.optimal_threshold = None
    
    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Store the initial threshold for comparison."""
        self.initial_threshold = pl_module.hparams.ew_threshold
        self.optimal_threshold = self.initial_threshold
    
    def on_validation_batch_end(
        self, 
        trainer: Trainer, 
        pl_module: LightningModule, 
        outputs, 
        batch, 
        batch_idx: int, 
        dataloader_idx: int = 0
    ) -> None:
        """Collect validation data for threshold optimization."""
        if not getattr(pl_module.hparams, 'optimize_threshold', True):
            return
            
        # Extract data from batch (assuming same structure as TraceEWModule)
        for name, raw in batch.items():
            trace_seq, direction, boundary, snp_vert, true_ew, meta = raw
            
            # Get prediction probabilities 
            with torch.no_grad():
                *_, pred_logits = pl_module(
                    trace_seq, direction, boundary, snp_vert
                )
                pred_prob = torch.sigmoid(pred_logits)
                true_prob = (true_ew > 0).float()
                
                # Cache for threshold optimization (detach and move to CPU)
                self.val_cache["probs"].append(pred_prob.squeeze().detach().cpu())
                self.val_cache["labels"].append(true_prob.squeeze().detach().cpu())
    
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Optimize threshold based on collected validation data."""
        if not getattr(pl_module.hparams, 'optimize_threshold', True):
            return
            
        # Only run on rank 0 for distributed training
        if trainer.global_rank != 0:
            self._reset_cache()
            return
            
        # Check warm-up period
        if trainer.current_epoch < self.warm_up_epochs:
            rank_zero_info(f"Epoch {trainer.current_epoch}: Skipping threshold optimization (warm-up)")
            self._reset_cache()
            return
            
        # Check update frequency
        if trainer.current_epoch % self.update_frequency != 0:
            self._reset_cache()
            return
            
        # Run optimization
        optimal_tau = self._optimize_threshold()
        if optimal_tau is not None:
            # Smooth with EMA to stabilize threshold evolution
            smoothed_tau = (1.0 - self.ema_alpha) * float(self.optimal_threshold) + self.ema_alpha * float(optimal_tau)
            # Update module threshold
            pl_module.hparams.ew_threshold = smoothed_tau
            self.optimal_threshold = smoothed_tau
            
            # Update metrics that use threshold
            self._update_metrics_threshold(pl_module)
            
            # Log only tau_f1
            pl_module.log("val/tau_f1", smoothed_tau, prog_bar=False, on_epoch=True, sync_dist=False)
            
        self._reset_cache()
    
    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Print final threshold evolution."""
        if trainer.global_rank == 0 and self.optimal_threshold is not None:
            improvement = self.optimal_threshold - self.initial_threshold
            print(f"\nThreshold Evolution: {self.initial_threshold:.4f} → {self.optimal_threshold:.4f} (Δ={improvement:+.4f})")
    
    def _optimize_threshold(self):
        """Find optimal threshold by maximizing F1 score."""
        if not self.val_cache["probs"]:
            return None
            
        try:
            # Concatenate cached data
            probs = torch.cat(self.val_cache["probs"])
            labels = torch.cat(self.val_cache["labels"])
            
            if len(probs) == 0:
                return None
                
            # Sweep thresholds
            taus = torch.linspace(0.0, 1.0, self.grid_size)
            best_f1, best_tau = -1.0, None
            
            for t in taus:
                preds = (probs >= t).float()
                tp = (preds * labels).sum()
                fp = (preds * (1 - labels)).sum()
                fn = ((1 - preds) * labels).sum()
                
                precision = tp / (tp + fp + 1e-9)
                recall = tp / (tp + fn + 1e-9)
                f1 = 2 * precision * recall / (precision + recall + 1e-9)
                
                if f1 > best_f1:
                    best_f1, best_tau = f1.item(), t.item()
            
            return best_tau
            
        except Exception as e:
            rank_zero_info(f"Error during threshold optimization: {str(e)}\nFull traceback:\n{traceback.format_exc()}")
            return None
    
    def _update_metrics_threshold(self, pl_module: LightningModule):
        """Update metric thresholds to use the new optimal threshold."""
        for stage_metrics in pl_module.metrics.values():
            for metric in stage_metrics.values():
                if hasattr(metric, 'threshold'):
                    metric.threshold = pl_module.hparams.ew_threshold
    
    def _reset_cache(self):
        """Reset validation cache for next epoch."""
        self.val_cache = {"probs": [], "labels": []}