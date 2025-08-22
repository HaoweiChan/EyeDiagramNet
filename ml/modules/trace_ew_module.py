import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics as tm
from dataclasses import dataclass
from lightning import LightningModule
from lightning.pytorch.utilities.rank_zero import rank_zero_info

from ..models.layers import LearnableLossWeighting
from ..utils import losses
from ..utils.losses import focus_weighted_eye_width_loss, min_focused_loss
from ..utils.init_weights import init_weights
from ..utils.visualization import image_to_buffer, plot_ew_curve

@torch.jit.script
def _augment_sequence_jit(seq: torch.Tensor, insert_frac: int = 10) -> torch.Tensor:
    """
    TorchScript-friendly augmentation used by TraceEWModule.  
    Inserts a small random gap inside each sequence and pads new timesteps with -1.

    Args
    ----
    seq : Tensor[B, L, C]  - original input sequence  
    insert_frac : int      - the divisor controlling maximum insertion length  
                             (default keeps the old "L // 10" behaviour)

    Returns
    -------
    Tensor[B, L+δ, C]  - augmented sequence
    """
    B, L, C = seq.shape
    max_insert_len = L // insert_frac
    if max_insert_len == 0:
        return seq

    insert_len = torch.randint(1, max_insert_len + 1, (1,), device=seq.device).item()
    if insert_len == 0:
        return seq

    new_L: int = int(L + insert_len)

    # TorchScript expects the `size` argument to be a List[int], not a tuple.
    # Explicitly cast each dimension to int to satisfy TorchScript's type checker.
    size: list[int] = [int(B), new_L, int(C)]
    out = torch.full(size, -1.0, dtype=seq.dtype, device=seq.device)

    keep = torch.randperm(new_L, device=seq.device)[:L]
    keep, _ = torch.sort(keep)
    out.index_copy_(1, keep, seq)
    return out

 # ---------------------------------------------------------------------------  
 # Lightweight container so we can keep the main `step` readable
 # ---------------------------------------------------------------------------
@dataclass
class BatchItem:
    trace_seq: torch.Tensor
    direction: torch.Tensor
    boundary: torch.Tensor
    snp_vert: torch.Tensor
    true_ew: torch.Tensor
    meta: dict

class TraceEWModule(LightningModule):
    """
    TraceEWModule: Lightning module for eye width prediction.
    
    This module handles training and inference for eye width prediction models,
    including support for uncertainty quantification and Laplace approximation.
    """
    def __init__(
        self,
        model: nn.Module,
        ckpt_path: str = None,
        strict: bool = False,
        compile_model: bool = False,
        ew_scaler: int = 100,
        ew_threshold: float = 0.3,
        use_laplace_on_fit_end: bool = True,
        ignore_snp: bool = False,
        tau_min: float = 0.1,
        augment_insert_frac: int = 20,  # Less aggressive augmentation
        # Unified loss options
        unified_ew_loss: str = 'focus_weighted',  # 'separate', 'focus_weighted', 'min_focused'
        focus_weight: float = 5.0,  # For focus_weighted loss
        min_focused_alpha: float = 0.7,  # For min_focused loss
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])

        # Set ignore_snp on the model if it supports it
        if hasattr(model, 'ignore_snp'):
            model.ignore_snp = ignore_snp

        self.model = model
        self.train_step_outputs = {}
        self.val_step_outputs = {}
        self.test_step_outputs = {}

        self.metrics = nn.ModuleDict({
            "train_": self.metrics_factory(),
            "val": self.metrics_factory(),
            "test": self.metrics_factory(),
        })
        self.ew_scaler = torch.tensor(self.hparams.ew_scaler)
        # Pre-compute inverse and log for efficiency - avoid repeated divisions
        self.ew_scaler_inv = torch.tensor(1.0 / self.hparams.ew_scaler)
        self.log_ew_scaler = torch.log(self.ew_scaler)
        # Use learnable loss weighting based on unified loss choice
        if self.hparams.unified_ew_loss == 'separate':
            # Original separate losses
            self.weighted_loss = LearnableLossWeighting(['bce', 'min_loss', 'trace_loss'])
        else:
            # Unified eye width loss + BCE
            self.weighted_loss = LearnableLossWeighting(['bce', 'unified_ew_loss'])
        
    def setup(self, stage=None):
        # Warm up the model by performing a dummy forward pass
        if stage in ('fit', None):
            loader = self.trainer.datamodule.train_dataloader()
            # Since train_dataset is a dict, we can grab the first one.
            self.config_keys = next(iter(self.trainer.datamodule.train_dataset.values())).config_keys
        elif stage == 'test':
            loader = self.trainer.datamodule.test_dataloader()
            # Since test_dataset is a dict, we can grab the first one.
            self.config_keys = next(iter(self.trainer.datamodule.test_dataset.values())).config_keys
        elif stage == 'validate':
            loader = self.trainer.datamodule.val_dataloader()
            # Since val_dataset is a dict, we can grab the first one.
            self.config_keys = next(iter(self.trainer.datamodule.val_dataset.values())).config_keys
        else:
            loader = self.trainer.datamodule.predict_dataloader()
            # For prediction/inference, get config_keys from the datamodule
            if hasattr(self.trainer.datamodule, 'config_keys'):
                self.config_keys = self.trainer.datamodule.config_keys
            else:
                # Fallback to boundary keys if config_keys not available
                self.config_keys = list(self.trainer.datamodule.boundary.to_dict().keys())
        
        dummy_batch, *_ = next(iter(loader))
        key = next(iter(dummy_batch.keys()))
        inputs = dummy_batch[key]
        
        # The last element of the tuple from the dataloader is not part of the model's forward pass inputs.
        # For training, it's the target `eye_width`.
        # For prediction, it would be a config dict, which is also not a model input.
        forward_args = inputs[:-2]

        try:
            with torch.no_grad():
                self(*forward_args)
        except (ValueError, RuntimeError) as e:
            rank_zero_info(traceback.format_exc())
            raise
        self.apply(init_weights('xavier'))

        # load model checkpoint
        if self.hparams.ckpt_path is not None:
            rank_zero_info(f'Loading model checkpoint: {self.hparams.ckpt_path}')
            ckpt = torch.load(self.hparams.ckpt_path, map_location=self.device)
            self.load_state_dict(ckpt['state_dict'], strict=self.hparams.strict)
        
        # Compile model for performance optimization after setup
        if stage in ('fit', None) and self.hparams.compile_model:
            # Check if torch.compile is disabled globally
            import os
            if os.environ.get('TORCH_COMPILE_DISABLE') == '1':
                rank_zero_info("torch.compile is disabled globally - using eager mode")
                return
                
            try:
                rank_zero_info("Attempting to compile model with torch.compile...")
                self.model = torch.compile(
                    self.model, 
                    mode="reduce-overhead",
                    dynamic=True,
                    fullgraph=False
                )
                rank_zero_info("Model compilation completed successfully.")
            except Exception as e:
                rank_zero_info(f"Model compilation failed: {str(e)}")
                rank_zero_info("Falling back to eager mode execution.")
                # Ensure model is in eager mode if compilation fails
                if hasattr(self.model, '_orig_mod'):
                    self.model = self.model._orig_mod
                    
                # Disable compilation globally to prevent future attempts
                import os
                os.environ['TORCH_COMPILE_DISABLE'] = '1'
                torch._dynamo.config.suppress_errors = True
                torch._dynamo.config.disable = True
        elif stage in ('fit', None) and not self.hparams.compile_model:
            rank_zero_info("Model compilation disabled by compile_model=False - using eager mode")

    def on_fit_end(self):
        """Fit the last-layer Laplace approximation after the main training."""
        if not self.hparams.use_laplace_on_fit_end or self.trainer.global_rank != 0:
            return
            
        from ..utils.laplace_utils import setup_laplace_dataloader

        # Setup Laplace-compatible dataloader using utility functions
        rank_zero_info("Fitting Laplace approximation on the last layer...")
        laplace_loader = setup_laplace_dataloader(self.trainer.datamodule, self.device)
        
        # Pass the datamodule to fit_laplace
        self.model.fit_laplace(laplace_loader, self.trainer.datamodule)
        rank_zero_info("Laplace approximation fitting complete.")

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    ############################ TRAIN & VALIDATION ############################

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        try:
            return self.step(batch, batch_idx, "train_", dataloader_idx)
        except RuntimeError as e:
            # Check for torch.compile related errors
            if "torch.compile" in str(e) or "aot_autograd" in str(e) or "double backward" in str(e):
                rank_zero_info(f"torch.compile runtime error detected: {e}")
                rank_zero_info("Disabling torch.compile and falling back to eager mode")
                
                # Disable compilation globally
                import os
                os.environ['TORCH_COMPILE_DISABLE'] = '1'
                torch._dynamo.config.suppress_errors = True
                torch._dynamo.config.disable = True
                
                # Reset model to eager mode
                if hasattr(self.model, '_orig_mod'):
                    self.model = self.model._orig_mod
                elif hasattr(self.model, '_orig_module'):
                    self.model = self.model._orig_module
                
                # Retry the step with eager mode
                return self.step(batch, batch_idx, "train_", dataloader_idx)
            else:
                # Re-raise if it's not a torch.compile issue
                raise

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self.step(batch, batch_idx, "val", dataloader_idx)
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.step(batch, batch_idx, "test", dataloader_idx)

    def on_train_epoch_end(self):
        log_metrics = self.compute_metrics("train_")
        if self.current_epoch % self.trainer.check_val_every_n_epoch == 0:
            for dataloader_idx, outputs in self.train_step_outputs.items():
                self.plot_metrics_curve("train_", log_metrics, outputs[0], dataloader_idx)
        self.train_step_outputs.clear()

    def on_validation_epoch_end(self):
        log_metrics = self.compute_metrics("val")
        for dataloader_idx, outputs in self.val_step_outputs.items():
            self.plot_metrics_curve("val", log_metrics, outputs[0], dataloader_idx)
        self.val_step_outputs.clear()
    
    def on_test_epoch_end(self):
        log_metrics = self.compute_metrics("test")
        for dataloader_idx, outputs in self.test_step_outputs.items():
            self.plot_metrics_curve("test", log_metrics, outputs[0], dataloader_idx)
        self.test_step_outputs.clear()

    ############################ INFERENCE ############################

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # Model handles uncertainty method internally
        trace_seq, direction, boundary, snp_vert, _ = batch
        pred_ew, _, _, _, pred_logits = self.model.predict_with_uncertainty(
            trace_seq, direction, boundary, snp_vert
        )
        pred_prob = torch.sigmoid(pred_logits)
        
        pred_ew = pred_ew * self.ew_scaler
        pred_ew[pred_prob < self.hparams.ew_threshold] = -0.1
        return pred_ew

    ############################ PRIVATE METHODS ############################

    def metrics_factory(self):
        metrics = {
            'loss': tm.MeanMetric,
            'mae': tm.MeanAbsoluteError,
            'mape': tm.WeightedMeanAbsolutePercentageError,
            'r2': tm.R2Score,
            'auroc': tm.classification.BinaryAUROC,
            'auprc': tm.classification.BinaryAveragePrecision,
            'accuracy': tm.classification.BinaryAccuracy,
            'f1': tm.classification.BinaryF1Score,
        }
        metrics_dict = nn.ModuleDict()
        for k, metric in metrics.items():
            if 'accuracy' in k or 'f1' in k:
                metrics_dict[k] = metric(threshold=self.hparams.ew_threshold)
            else:
                metrics_dict[k] = metric()
        return metrics_dict

    # -----------------------------------------------------------------------
    # Helper methods introduced by the step-function refactor
    # -----------------------------------------------------------------------
    def _augment(self, seq: torch.Tensor) -> torch.Tensor:
        """Single call-site for sequence augmentation."""
        return self.augment_input_sequence(seq)

    def _to_batch_item(self, raw) -> "BatchItem":
        """Convert the raw tuple coming from DataLoader into a BatchItem and apply EW scaling."""
        trace_seq, direction, boundary, snp_vert, true_ew, meta = raw
        true_ew = true_ew * self.ew_scaler_inv.to(true_ew.device)
        return BatchItem(trace_seq, direction, boundary, snp_vert, true_ew, meta)

    def _run_model(self, item: "BatchItem", stage: str):
        """
        Forward pass wrapper.  
        Returns a dict with:
            * 'train': tensors that carry gradients (used for loss)
            * 'eval': tensors used only for metrics / logging
        """
        # Use the model's internal uncertainty logic (Laplace or MC)
        if stage == "val" and hasattr(self.model, 'predict_with_uncertainty'):
            pred_ew_eval, _, _, _, pred_logits_eval = self.model.predict_with_uncertainty(
                item.trace_seq, item.direction, item.boundary, item.snp_vert
            )
            pred_prob_eval = torch.sigmoid(pred_logits_eval)
        else:
            pred_ew_eval = pred_prob_eval = None

        # Gradient-carrying forward pass (always)
        pred_ew, pred_logits = self(
            item.trace_seq, item.direction, item.boundary, item.snp_vert
        )

        # Fallback to eager outputs for metric display when Laplace not used
        if pred_ew_eval is None:
            pred_prob = torch.sigmoid(pred_logits)
            pred_ew_eval, pred_prob_eval = pred_ew, pred_prob

        return {
            "train": (pred_ew, pred_logits),
            "eval": (pred_ew_eval, pred_prob_eval)
        }

    def _compute_loss(self, item: "BatchItem", forward_out):
        """Calculate composite loss and prepare everything needed for metric updates."""
        pred_ew, pred_logits = forward_out["train"]

        # --- Classification loss -----------------------------------------------------
        true_prob = (item.true_ew > 0).float()
        bce_loss = F.binary_cross_entropy_with_logits(pred_logits, true_prob)

        # --- Eye width regression loss --------------------------------------------
        if self.hparams.unified_ew_loss == 'separate':
            # Original separate losses (now consistent with unified losses)
            tau_m = self.hparams.tau_min
            
            # Min-value prediction loss
            softmin_pred_ew = losses.softmin(pred_ew, tau=tau_m, dim=1)
            min_true_ew = torch.min(item.true_ew, dim=1).values
            min_loss = F.smooth_l1_loss(softmin_pred_ew, min_true_ew)

            # Per-trace prediction loss  
            trace_loss = F.mse_loss(pred_ew, item.true_ew)
            
            loss = self.weighted_loss({
                'bce': bce_loss,
                'min_loss': min_loss,
                'trace_loss': trace_loss
            })
            
        elif self.hparams.unified_ew_loss == 'focus_weighted':
            # Focus-weighted unified loss
            unified_ew_loss = focus_weighted_eye_width_loss(
                pred_ew, item.true_ew, 
                focus_weight=self.hparams.focus_weight
            )
            loss = self.weighted_loss({
                'bce': bce_loss,
                'unified_ew_loss': unified_ew_loss
            })
            
        elif self.hparams.unified_ew_loss == 'min_focused':
            # Min-focused unified loss
            unified_ew_loss = min_focused_loss(
                pred_ew, item.true_ew,
                alpha=self.hparams.min_focused_alpha,
                tau_min=self.hparams.tau_min
            )
            loss = self.weighted_loss({
                'bce': bce_loss,
                'unified_ew_loss': unified_ew_loss
            })
        else:
            raise ValueError(f"Unknown unified_ew_loss: {self.hparams.unified_ew_loss}")

        # Use the eval tensors (may come from MC/Laplace inference) for metrics
        pred_ew_eval, pred_prob_eval = forward_out["eval"]

        # Scale predictions for metrics (no effective eye width gating)
        pred_ew_scaled = pred_ew_eval * self.ew_scaler
        true_ew_scaled = item.true_ew * self.ew_scaler
        
        extras = {
            "pred_ew": pred_ew_scaled,
            "true_ew": true_ew_scaled,
            "pred_prob": pred_prob_eval,
            "true_prob": true_prob,
            "meta": item.meta,
            "boundary": item.boundary,
        }
        return loss, extras

    def _prepare_boundary_meta(self, boundary: torch.Tensor, meta: dict) -> dict:
        """Inverse transforms boundary data and formats it for logging."""
        # Get the scaler from datamodule
        fix_scaler = self.trainer.datamodule.fix_scaler
        
        # Reshape boundary data for inverse transform
        boundary_reshaped = boundary.reshape(-1, boundary.shape[-1])
        
        # Apply inverse transform
        boundary_inverse = fix_scaler.inverse_transform(boundary_reshaped)
        
        # It's safer to check for scaler.nan's type, but assuming it's a float/int
        nan_val = getattr(fix_scaler, 'nan', None)
        if nan_val is not None:
            boundary_inverse[boundary_inverse == nan_val] = torch.nan
        
        # Reshape back to original shape
        boundary_inverse_reshaped = boundary_inverse.reshape(boundary.shape)
        
        # Convert to proper format for meta - create boundary dict for each item in batch
        boundary_dicts = []
        for i in range(len(boundary_inverse_reshaped)):
            boundary_dict = {k: v.item() for k, v in zip(self.config_keys, boundary_inverse_reshaped[i])}
            boundary_dicts.append(boundary_dict)
        
        # Add boundary parameters to metadata for logging
        return {**meta, 'boundary': boundary_dicts}

    def _maybe_collect_samples(self, stage: str, name: str, batch_idx: int, extras: dict):
        """Keep a few random samples around for plotting at epoch-end."""
        if batch_idx in self.get_output_steps(stage):
            # Prepare detailed metadata only for the samples we are saving
            log_extras = extras.copy()
            log_extras["meta"] = self._prepare_boundary_meta(extras["boundary"], extras["meta"])
            del log_extras["boundary"]  # Clean up unprocessed boundary tensor

            if stage == "train_":
                self.train_step_outputs.setdefault(name, []).append(log_extras)
            elif stage == "val":
                self.val_step_outputs.setdefault(name, []).append(log_extras)
            elif stage == "test":
                self.test_step_outputs.setdefault(name, []).append(log_extras)

    def get_output_steps(self, stage):
        if stage == "train_":
            max_batches = self.trainer.num_training_batches
        elif stage == "test":
            max_batches = getattr(self.trainer, 'num_test_batches')[0]
        else:
            max_batches = getattr(self.trainer, f'num_{stage}_batches')[0]
        return (self.current_epoch * max_batches, max_batches - 1)

    def augment_input_sequence(self, seq):
        # TorchScript-accelerated implementation (see _augment_sequence_jit above)
        return _augment_sequence_jit(seq, 10)

    def step(self, batch, batch_idx, stage, dataloader_idx):
        """Refactored step - orchestrates smaller helpers."""
        losses = []

        for name, raw in batch.items():
            # (1) Normalise & pack
            item = self._to_batch_item(raw)

            # (2) Optional augmentation
            if stage == "train_":
                item.trace_seq = self._augment(item.trace_seq)

            # (3) Forward pass(es)
            fwd = self._run_model(item, stage)

            # (4) Loss + metric-ready tensors
            loss, extras = self._compute_loss(item, fwd)
            losses.append(loss)

            # (5) Update metrics & maybe collect samples
            with torch.no_grad():
                self.update_metrics(
                    stage,
                    loss.detach(),
                    extras["pred_ew"],
                    extras["true_ew"],
                    extras["pred_prob"],
                    extras["true_prob"],
                )
                self._maybe_collect_samples(stage, name, batch_idx, extras)

        mean_loss = torch.stack(losses).mean()
        self.log(
            "loss",
            mean_loss,
            on_step=True,
            prog_bar=(stage == "train_"),
            logger=False,
            sync_dist=True
        )
        return {"loss": mean_loss}

    def update_metrics(
        self,
        stage: str,
        loss: torch.Tensor,
        pred_ew: torch.Tensor,
        true_ew: torch.Tensor,
        pred_prob: torch.Tensor,
        true_prob: torch.Tensor,
    ):
        # Pre-compute common values to avoid redundant operations
        mask = true_prob.bool()
        
        # Only compute masked values once for regression metrics
        if mask.any():
            pred_ew_masked = pred_ew[mask]
            true_ew_masked = true_ew[mask]
        else:
            pred_ew_masked = torch.empty(0, device=pred_ew.device, dtype=pred_ew.dtype)
            true_ew_masked = torch.empty(0, device=true_ew.device, dtype=true_ew.dtype)
        
        # Update all metrics efficiently - avoid redundant flattening
        for key, metric in self.metrics[stage].items():
            if 'loss' in key:
                metric.update(loss)
            elif 'f1' in key or 'accuracy' in key or 'auroc' in key or 'auprc' in key:
                metric.update(pred_prob.flatten(), true_prob.flatten().long())
            elif pred_ew_masked.numel() > 0:  # Only update if we have valid masked data
                metric.update(pred_ew_masked, true_ew_masked)

    def compute_metrics(self, stage):
        log_metrics = {}
        for key, metric in self.metrics[stage].items():
            log_metrics[f'{stage}/{key}'] = metric.compute()
            metric.reset()
        if stage in ("train_", "val") and self.logger is not None:
            self.logger.log_metrics(log_metrics, self.current_epoch)

        if stage in ("val", "test"):
            self.log(
                'hp_metric', log_metrics[f'{stage}/mae'],
                prog_bar=True, on_epoch=True, on_step=False, sync_dist=True
            )

        if stage == "test":
            for key, metric in log_metrics.items():
                self.log(key, metric, sync_dist=True)        

        return log_metrics

    def plot_metrics_curve(self, stage, log_metrics, outputs, dataloader_idx):
        fig = plot_ew_curve(outputs, log_metrics, self.hparams.ew_threshold)
        if self.logger:
            tag = "_".join(['sparam', str(dataloader_idx)])
            self.logger.experiment.add_image(f'{stage}/{tag}', image_to_buffer(fig), self.current_epoch)
