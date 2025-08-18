import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics as tm
from dataclasses import dataclass
from lightning import LightningModule
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from lightning.pytorch.utilities.combined_loader import CombinedLoader

from ..models.layers import LearnableLossWeighting
from ..utils import losses
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
        tau_gate: float = 0.1,
        tau_min: float = 0.1,
        min_loss_weight: float = 0.1,
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
        # self.weighted_loss = UncertaintyWeightedLoss(['nll', 'bce'])
        self.weighted_loss = LearnableLossWeighting(['bce', 'min_loss', 'trace_loss'])
        
        # NOTE: GradNormLossBalancer uses second-order gradients which are incompatible 
        # with torch.compile. If you want to use torch.compile, switch to:
        # self.weighted_loss = LearnableLossWeighting(['nll', 'bce'])  # Simple learnable weights
        # or
        # self.weighted_loss = UncertaintyWeightedLoss(['nll', 'bce'])  # Uncertainty weighting

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
            self.config_keys = self.trainer.datamodule.boundary.to_dict().keys()
        
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
            
        rank_zero_info("Fitting Laplace approximation on the last layer...")
        # The dataloader needs to be wrapped to yield (X, y) tuples
        # where X is a tuple of the model's inputs.
        class LaplaceDataLoaderWrapper:
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
                            def __init__(self, length): self._length = length
                            def __len__(self): return self._length
                        self.dataset = _CombinedDatasetProxy(total_len)
                        rank_zero_info(f"LaplaceDataLoaderWrapper: Using CombinedDatasetProxy with total length {total_len} for CombinedLoader.")
                    else:
                        rank_zero_info("LaplaceDataLoaderWrapper: Could not determine dataset length for CombinedLoader from its sub-loaders. Using len(CombinedLoader) as batch count proxy.")
                        # Fallback: Create a proxy whose length is the number of batches in CombinedLoader.
                        # This is not N_samples, but might allow Laplace to iterate if N is only for progress bar.
                        class _LenProxyDataset:
                            def __init__(self, loader_to_wrap): self.loader_to_wrap = loader_to_wrap
                            def __len__(self): return len(self.loader_to_wrap) 
                        self.dataset = _LenProxyDataset(dataloader)
                else:
                    rank_zero_info("LaplaceDataLoaderWrapper: Wrapped dataloader is not CombinedLoader and has no 'dataset' attribute.")
                    self.dataset = None 
                
                # Ensure self.dataset is always set, even if to a dummy one for safety
                if self.dataset is None:
                    class _EmptyDataset:
                        def __len__(self): return 0
                    self.dataset = _EmptyDataset()
                    rank_zero_info("LaplaceDataLoaderWrapper: Fallback to an empty dataset proxy (length 0).")

            def __iter__(self):
                for batch, *_ in self.dataloader:
                    if isinstance(batch, dict):
                        # Handle dictionary batch (from CombinedLoader or regular training)
                        for name, raw_data in batch.items():
                            # raw_data is a tuple of tensors: (trace_seq, direction, boundary, snp_vert, true_ew)
                            # Concatenate inputs into a single tensor
                            # Assuming all input tensors have the same batch size (dim 0)
                            # and can be concatenated along a new dimension or an existing one
                            # This part needs careful consideration of tensor shapes
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

        train_loader = self.trainer.datamodule.train_dataloader()
        
        # If train_loader is a dictionary of dataloaders, wrap it with CombinedLoader.
        # Otherwise, use it directly. This makes the logic robust.
        if isinstance(train_loader, dict):
            combined_loader = CombinedLoader(train_loader, mode="max_size_cycle")
            laplace_loader = LaplaceDataLoaderWrapper(combined_loader, self.device)
        else:
            laplace_loader = LaplaceDataLoaderWrapper(train_loader, self.device)
        
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
        pred_prob = torch.sigmoid(pred_logits)

        # Fallback to eager outputs for metric display when Laplace not used
        if pred_ew_eval is None:
            pred_ew_eval, pred_prob_eval = pred_ew, pred_prob

        return {
            "train": (pred_ew, pred_prob),
            "eval": (pred_ew_eval, pred_prob_eval)
        }

    def _calculate_effective_ew(self, pred_ew: torch.Tensor, pred_prob: torch.Tensor) -> torch.Tensor:
        """Calculates the effective eye-width with a soft gate."""
        t = self.hparams.ew_threshold
        tau_g = self.hparams.tau_gate
        
        # Soft gate based on inference threshold
        s = torch.sigmoid((pred_prob - losses.logit(torch.tensor(t, device=self.device))) / tau_g)
        
        # Effective eye-width with gated closed-eye value
        c_closed = -0.1 / self.ew_scaler.item()
        ew_eff = s * pred_ew + (1 - s) * c_closed
        return ew_eff

    def _compute_loss(self, item: "BatchItem", forward_out):
        """Calculate composite loss and prepare everything needed for metric updates."""
        pred_ew, pred_prob = forward_out["train"]

        # --- classification helpers --------------------------------------------------
        true_prob = (item.true_ew > 0).float()
        
        # --- Soft-min loss for minimum eye-width prediction --------------------------
        tau_m = self.hparams.tau_min
        ew_eff = self._calculate_effective_ew(pred_ew, pred_prob)
        
        # Per-design softmin and true min
        # pred_ew is (B, L), so we reduce along L (dim=1)
        softmin_pred_ew = losses.softmin(ew_eff, tau=tau_m, dim=1)
        min_true_ew = torch.min(item.true_ew, dim=1).values
        
        # Min-value prediction loss
        min_loss = F.smooth_l1_loss(softmin_pred_ew, min_true_ew)

        # Per-trace prediction loss
        trace_loss = F.mse_loss(ew_eff, item.true_ew)
        
        # --- Original classification loss ---------------------------------------------
        bce_loss = F.binary_cross_entropy_with_logits(pred_prob, true_prob)

        # --- Combine losses -----------------------------------------------------------
        loss = self.weighted_loss({
            'bce': bce_loss,
            'min_loss': min_loss,
            'trace_loss': trace_loss
        })

        # Use the eval tensors (may come from MC/Laplace inference) for metrics
        pred_ew_eval, pred_prob_eval = forward_out["eval"]
        
        # Avoid recomputing effective ew if the eval tensors are the same as train tensors
        if pred_ew is pred_ew_eval:
            ew_eff_eval = ew_eff
        else:
            ew_eff_eval = self._calculate_effective_ew(pred_ew_eval, pred_prob_eval)

        # Rescale / post-process for logging
        pred_ew_scaled = ew_eff_eval * self.ew_scaler
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
        # This function is computationally expensive and should only be called when needed for logging.
        # It involves moving data to the CPU for scikit-learn's inverse_transform.
        
        # Ensure boundary tensor is on the CPU
        boundary_cpu = boundary.cpu()
        
        # Get the scaler from datamodule
        fix_scaler = self.trainer.datamodule.fix_scaler
        
        # Reshape boundary data for inverse transform
        boundary_reshaped = boundary_cpu.reshape(-1, boundary_cpu.shape[-1])
        
        # Apply inverse transform (assuming scaler works on numpy)
        boundary_inverse = fix_scaler.inverse_transform(boundary_reshaped.numpy())
        
        # Handle nan values - convert scaler.nan to torch.nan
        # Create a tensor from the numpy array to perform this operation
        boundary_inverse_torch = torch.from_numpy(boundary_inverse)
        
        # It's safer to check for scaler.nan's type, but assuming it's a float/int
        nan_val = getattr(fix_scaler, 'nan', None)
        if nan_val is not None:
            boundary_inverse_torch[boundary_inverse_torch == nan_val] = torch.nan
        
        # Reshape back to original shape
        boundary_inverse_reshaped = boundary_inverse_torch.reshape(boundary_cpu.shape)
        
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

    def on_before_optimizer_step(self, optimizer):
        """Debug: Track which parameters receive gradients after backward pass"""
        # Only run when profiling is enabled to avoid overhead during normal training
        profiling_enabled = (
            hasattr(self.trainer, 'profiler') and 
            self.trainer.profiler is not None and
            self.trainer.profiler.__class__.__name__ not in ['PassThroughProfiler', 'SimpleProfiler']
        )
        
        if profiling_enabled and self.current_epoch == 0 and self.global_step % 100 == 0:
            unused_params = []
            total_params = 0
            
            # When gradient checkpointing is enabled, it's expected that parameters
            # within the checkpointed modules will not have gradients available at this point.
            grad_checkpointing_active = getattr(self.model, 'use_gradient_checkpointing', False)
            checkpointed_modules = ['trace_encoder', 'snp_encoder', 'signal_encoder']

            for name, param in self.named_parameters():
                if param.requires_grad:
                    total_params += 1

                    # If checkpointing is on, skip checking params from modules we know are wrapped.
                    if grad_checkpointing_active and any(mod_name in name for mod_name in checkpointed_modules):
                        continue

                    if param.grad is None:
                        unused_params.append(name)
            
            if unused_params:
                print(f"Step {self.global_step} - Found {len(unused_params)}/{total_params} unused parameters:")
                for param_name in unused_params[:10]:  # Show first 10 to avoid spam
                    print(f"  - {param_name}")
                if len(unused_params) > 10:
                    print(f"  ... and {len(unused_params) - 10} more")
            else:
                print(f"Step {self.global_step} - All {total_params} parameters received gradients")
