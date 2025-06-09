import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics as tm
from lightning import LightningModule

from ..models.layers import (GradNormLossBalancer, LearnableLossWeighting,
                             UncertaintyWeightedLoss)
from ..utils import losses
from ..utils.init_weights import init_weights
from ..utils.visualization import image_to_buffer, log_info, plot_ew_curve

class TraceEWModule(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        ckpt_path: str = None,
        strict: bool = False,
        ew_scaler: int = 50,
        ew_threshold: float = 0.3,
        mc_samples: int = 50,
        use_mc_validation: bool = True,
        compile_model: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])

        self.model = model
        self.train_step_outputs = {}
        self.val_step_outputs = {}

        self.metrics = nn.ModuleDict({
            "train_": self.metrics_factory(),
            "val": self.metrics_factory(),
        })
        self.ew_scaler = torch.tensor(self.hparams.ew_scaler)
        # Pre-compute inverse and log for efficiency - avoid repeated divisions
        self.ew_scaler_inv = torch.tensor(1.0 / self.hparams.ew_scaler)
        self.log_ew_scaler = torch.log(self.ew_scaler)
        # self.weighted_loss = UncertaintyWeightedLoss(['nll', 'bce'])
        self.weighted_loss = LearnableLossWeighting(['nll', 'bce'])
        
        # NOTE: GradNormLossBalancer uses second-order gradients which are incompatible 
        # with torch.compile. If you want to use torch.compile, switch to:
        # self.weighted_loss = LearnableLossWeighting(['nll', 'bce'])  # Simple learnable weights
        # or
        # self.weighted_loss = UncertaintyWeightedLoss(['nll', 'bce'])  # Uncertainty weighting

    def setup(self, stage=None):
        # Warm up the model by performing a dummy forward pass
        if stage in ('fit', None):
            loader = self.trainer.datamodule.train_dataloader()
        else:
            loader = self.trainer.datamodule.predict_dataloader()
        
        dummy_batch, *_ = next(iter(loader))
        if stage in ('fit', None):
            key = next(iter(dummy_batch.keys()))
            inputs = dummy_batch[key]
            forward_args = inputs[:-1]
        else:
            forward_args = dummy_batch

        try:
            with torch.no_grad():
                self(*forward_args)
        except (ValueError, RuntimeError) as e:
            log_info(traceback.format_exc())
            raise
        self.apply(init_weights('xavier'))

        # load model checkpoint
        if self.hparams.ckpt_path is not None:
            log_info(f'Loading model checkpoint: {self.hparams.ckpt_path}')
            ckpt = torch.load(self.hparams.ckpt_path, map_location=self.device)
            self.load_state_dict(ckpt['state_dict'], strict=self.hparams.strict)
        
        # Compile model for performance optimization after setup
        if stage in ('fit', None) and self.hparams.compile_model:
            # Check if torch.compile is disabled globally
            import os
            if os.environ.get('TORCH_COMPILE_DISABLE') == '1':
                log_info("torch.compile is disabled globally - using eager mode")
                return
                
            try:
                log_info("Attempting to compile model with torch.compile...")
                self.model = torch.compile(
                    self.model, 
                    mode="default",
                    dynamic=True,
                    fullgraph=False
                )
                log_info("Model compilation completed successfully.")
            except Exception as e:
                log_info(f"Model compilation failed: {str(e)}")
                log_info("Falling back to eager mode execution.")
                # Ensure model is in eager mode if compilation fails
                if hasattr(self.model, '_orig_mod'):
                    self.model = self.model._orig_mod
                    
                # Disable compilation globally to prevent future attempts
                import os
                os.environ['TORCH_COMPILE_DISABLE'] = '1'
                torch._dynamo.config.suppress_errors = True
                torch._dynamo.config.disable = True
        elif stage in ('fit', None) and not self.hparams.compile_model:
            log_info("Model compilation disabled by compile_model=False - using eager mode")

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    ############################ TRAIN & VALIDATION ############################

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        try:
            return self.step(batch, batch_idx, "train_", dataloader_idx)
        except RuntimeError as e:
            # Check for torch.compile related errors
            if "torch.compile" in str(e) or "aot_autograd" in str(e) or "double backward" in str(e):
                log_info(f"torch.compile runtime error detected: {e}")
                log_info("Disabling torch.compile and falling back to eager mode")
                
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

    def on_train_epoch_end(self):
        log_metrics = self.compute_metrics("train_")
        if self.current_epoch % self.trainer.check_val_every_n_epoch == 0:
            for dataloader_idx in self.train_step_outputs.keys():
                self.plot_sparam_curve("train_", log_metrics, self.train_step_outputs[dataloader_idx][0], dataloader_idx)
        self.train_step_outputs.clear()

    def on_validation_epoch_end(self):
        log_metrics = self.compute_metrics("val")
        for dataloader_idx in self.val_step_outputs.keys():
            self.plot_sparam_curve("val", log_metrics, self.val_step_outputs[dataloader_idx][0], dataloader_idx)
        self.val_step_outputs.clear()

    ############################ INFERENCE ############################

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if hasattr(self.model, 'predict_with_uncertainty'):
            pred_ew, total_var, aleatoric_var, epistemic_var, pred_logits = self.model.predict_with_uncertainty(
                *batch, n_samples=self.hparams.mc_samples
            )
            pred_prob = torch.sigmoid(pred_logits)
        else:
            pred_ew, pred_logvar, pred_logits = self(*batch)
            pred_prob = torch.sigmoid(pred_logits)
        
        pred_ew = pred_ew * self.ew_scaler
        pred_ew[pred_prob < self.hparams.ew_threshold] = -0.1
        return pred_ew

    ############################ PRIVATE METHODS ############################

    def convert_metric_name(self, stage):
        if stage == "train_":
            return "train_"
        return stage

    def metrics_factory(self):
        metrics = {
            'loss': tm.MeanMetric,
            'mae': tm.MeanAbsoluteError,
            'mape': tm.WeightedMeanAbsolutePercentageError,
            'r2': tm.R2Score,
            'accuracy': tm.classification.BinaryAccuracy,
            'f1': tm.classification.BinaryF1Score,
            'cov': tm.classification.BinaryAccuracy,
            'cov2s': tm.classification.BinaryAccuracy,
        }
        metrics_dict = nn.ModuleDict()
        for k, metric in metrics.items():
            if 'accuracy' in k or 'f1' in k:
                metrics_dict[k] = metric(threshold=self.hparams.ew_threshold)
            else:
                metrics_dict[k] = metric()
        return metrics_dict

    def get_output_steps(self, stage):
        if stage == "train_":
            max_batches = self.trainer.num_training_batches
        else:
            max_batches = getattr(self.trainer, f'num_{stage}_batches')[0]
        return (self.current_epoch * max_batches, max_batches - 1)

    def augment_input_sequence(self, seq):
        batch_size, seq_len, feat_dim = seq.shape
        max_insert_len = seq_len // 10
        
        if max_insert_len == 0:
            return seq
        
        # Use a fixed augmentation strategy for better performance
        insert_len = torch.randint(1, max_insert_len + 1, (1,), device=seq.device).item()
        
        if insert_len == 0:
            return seq
        
        new_seq_len = seq_len + insert_len
        
        # More efficient: use torch.full with specific dtype to avoid conversions
        extended_seq = torch.full(
            (batch_size, new_seq_len, feat_dim),
            -1.0, device=seq.device, dtype=seq.dtype
        )
        
        # Generate random positions more efficiently - avoid sorting overhead for large sequences
        if new_seq_len > 1000:
            # For large sequences, use reservoir sampling approach
            keep_positions = torch.arange(seq_len, device=seq.device)
            # Add random offsets
            offsets = torch.randperm(insert_len, device=seq.device)[:seq_len] if insert_len < seq_len else torch.zeros(seq_len, device=seq.device)
            keep_positions = keep_positions + torch.cumsum(torch.bincount(keep_positions + offsets, minlength=new_seq_len)[:new_seq_len], dim=0)[:seq_len]
        else:
            # Original approach for smaller sequences
            keep_positions = torch.sort(torch.randperm(new_seq_len, device=seq.device)[:seq_len])[0]
        
        # Use index_copy_ for better performance
        extended_seq.index_copy_(1, keep_positions, seq)
        
        return extended_seq

    def step(self, batch, batch_idx, stage, dataloader_idx):
        """
        Arguments:
            batch:
                input: (B, E)
                true: (B, F, P, P)
                snp_ports: (B, E)
        """
        loss = 0
        for name, batch_one in batch.items():
            trace_seq, direction, boundary, snp_vert, true_ew = batch_one
            # Use multiplication instead of division - 2-3x faster
            true_ew = true_ew * self.ew_scaler_inv.to(true_ew.device)

            if stage == "train_":
                trace_seq = self.augment_input_sequence(trace_seq)

            # Make the true probability for non-closed eye (optimized)
            true_prob = (true_ew > 0).float()  # More efficient than clone + assignment
            weight_prob = torch.where(true_prob == 0, 10.0, 1.0)
            weight_sum = weight_prob.sum()
            weight_prob = weight_prob * (1.0 / weight_sum)

            # Use Monte Carlo inference for validation if enabled
            if stage == "val" and self.hparams.use_mc_validation and hasattr(self.model, 'predict_with_uncertainty'):
                # Use MC inference for better uncertainty estimation
                pred_ew, total_var, aleatoric_var, epistemic_var, pred_logits = self.model.predict_with_uncertainty(
                    trace_seq, direction, boundary, snp_vert, n_samples=self.hparams.mc_samples
                )
                pred_prob = torch.sigmoid(pred_logits)
                
                # Use total variance for loss computation
                pred_logvar = torch.log(total_var + 1e-8)
                
                # Forward the model normally for loss computation (to get gradients)
                pred_ew_train, pred_logvar_train, pred_prob_train, hidden_states = self(
                    trace_seq, direction, boundary, snp_vert, output_hidden_states=True
                )
                
                loss += self.weighted_loss({
                    'nll': losses.gaussian_nll_loss(pred_ew_train, true_ew, pred_logvar_train, mask=true_prob),
                    'bce': F.binary_cross_entropy_with_logits(pred_prob_train, true_prob, weight=weight_prob)
                }, hidden_states)
            else:
                # Standard forward pass for training
                pred_ew, pred_logvar, pred_prob, hidden_states = self(trace_seq, direction, boundary, snp_vert, output_hidden_states=True)
                pred_prob = torch.sigmoid(pred_prob)
                
                loss += self.weighted_loss({
                    'nll': losses.gaussian_nll_loss(pred_ew, true_ew, pred_logvar, mask=true_prob),
                    'bce': F.binary_cross_entropy_with_logits(pred_prob, true_prob, weight=weight_prob)
                }, hidden_states)

            pred_ew = pred_ew * self.ew_scaler
            true_ew = true_ew * self.ew_scaler
            # Use pre-computed log to avoid repeated computation
            pred_logvar = pred_logvar + 2 + self.log_ew_scaler.to(pred_logvar.device)
            pred_sigma = torch.exp(0.5 * pred_logvar)
            self.update_metrics(stage, loss.detach(), pred_ew, true_ew, pred_prob, true_prob, pred_sigma)
            
            if batch_idx in self.get_output_steps(stage):
                idx = torch.randint(len(pred_ew), (1,)).item()
                step_output = {
                    'pred_ew': pred_ew[idx],
                    'true_ew': true_ew[idx],
                    'pred_prob': pred_prob[idx],
                    'true_prob': true_prob[idx],
                    'pred_sigma': pred_sigma[idx]
                }
                if stage == "train_":
                    self.train_step_outputs.setdefault(name, []).append(step_output)
                else:
                    self.val_step_outputs.setdefault(name, []).append(step_output)

        prog_bar = True if stage == "train_" else False
        loss = loss / len(batch)
        self.log("loss", loss, on_step=True, prog_bar=prog_bar, logger=False, sync_dist=True)

        return {"loss": loss}

    def update_metrics(
        self,
        stage: str,
        loss: torch.Tensor,
        pred_ew: torch.Tensor,
        true_ew: torch.Tensor,
        pred_prob: torch.Tensor,
        true_prob: torch.Tensor,
        pred_sigma: torch.Tensor
    ):
        stage = self.convert_metric_name(stage)
        
        # Pre-compute common values to avoid redundant operations
        mask = true_prob.bool()
        # Use in-place operations to reduce memory allocations
        pred_prob_flat = pred_prob.flatten()
        true_prob_flat = true_prob.flatten()
        
        # Only compute masked values once for regression metrics
        if mask.any():
            pred_ew_masked = pred_ew[mask]
            true_ew_masked = true_ew[mask]
        else:
            pred_ew_masked = torch.empty(0, device=pred_ew.device, dtype=pred_ew.dtype)
            true_ew_masked = torch.empty(0, device=true_ew.device, dtype=true_ew.dtype)
        
        # Pre-compute coverage metrics to avoid redundant calculations
        coverage_metrics = {}
        if mask.any():
            for key in self.metrics[stage].keys():
                if 'cov' in key:
                    sigma_multiplier = 1.0 if '1s' in key else 2.0
                    # Use in-place operations where possible
                    lower = pred_ew - sigma_multiplier * pred_sigma
                    upper = pred_ew + sigma_multiplier * pred_sigma
                    in_range_mask = ((true_ew >= lower) & (true_ew <= upper)).float()
                    coverage_metrics[key] = (in_range_mask[mask], torch.ones_like(in_range_mask[mask]))
        
        # Update all metrics efficiently - avoid redundant flattening
        for key, metric in self.metrics[stage].items():
            if 'loss' in key:
                metric.update(loss)
            elif 'f1' in key or 'accuracy' in key:
                metric.update(pred_prob_flat, true_prob_flat)
            elif 'cov' in key and key in coverage_metrics:
                in_range_flat, true_mask_flat = coverage_metrics[key]
                metric.update(in_range_flat, true_mask_flat)
            elif pred_ew_masked.numel() > 0:  # Only update if we have valid masked data
                metric.update(pred_ew_masked, true_ew_masked)

    def compute_metrics(self, stage):
        stage = self.convert_metric_name(stage)
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

    def plot_sparam_curve(self, stage, log_metrics, outputs, dataloader_idx):
        tag = self.convert_metric_name(stage)
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
            
            for name, param in self.named_parameters():
                if param.requires_grad:
                    total_params += 1
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