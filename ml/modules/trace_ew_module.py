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
from ..utils.graph_smooth import knn_graph, graph_laplacian_penalty
from ..models.token_shift import DeltaTokenModulator

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
        predict_logvar: bool = True,
        # Graph smoothness (optional)
        graph_smooth_enable: bool = False,
        graph_smooth_k: int = 8,
        graph_smooth_sigma: float = 0.25,
        graph_smooth_weight: float = 0.0,
        # In-batch theta pairing (optional)
        pair_enable: bool = False,
        pair_k: int = 2,
        pair_min_delta: float = 0.05,
        pair_theta_keys: list[str] | None = None,
        # Token-shift supervision (optional)
        enable_token_shift: bool = False,
        lambda_tokens: float = 0.0,
        token_mod_hidden: int | None = None,
        token_mod_num_f: int = 6,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])

        # Set ignore_snp on the model if it supports it
        if hasattr(model, 'ignore_snp'):
            model.ignore_snp = ignore_snp
        if hasattr(model, 'predict_logvar'):
            model.predict_logvar = self.hparams.predict_logvar

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
        
        # Token-shift modulator (lazy init when shapes are known)
        self.token_modulator: DeltaTokenModulator | None = None

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

        # Lazy-init token modulator once we know boundary dimension and model dims
        self._maybe_init_token_modulator(sample_boundary=inputs[2])

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
                            trace_seq_input = raw_data[0].to(self.device)
                            targets = raw_data[-1].to(self.device).squeeze()
                            yield trace_seq_input, targets
                    else:
                        # Handle tuple batch from a single dataloader
                        # batch is (trace_seq, direction, boundary, snp_vert, true_ew)
                        trace_seq_input = batch[0].to(self.device)
                        targets = batch[-1].to(self.device).squeeze()
                        yield trace_seq_input, targets
            
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

    # --------------------- One-Shot Theta Sweep Inference ---------------------
    @torch.no_grad()
    def one_shot_sweep(
        self,
        trace_seq: torch.Tensor,
        direction: torch.Tensor,
        snp_vert: torch.Tensor,
        theta_ref: torch.Tensor,
        theta_grid: torch.Tensor,
        batch_size: int = 64,
    ) -> dict:
        """
        Perform one-shot sweep over theta values by reusing encoded tokens and
        modulating them with DeltaTokenModulator, then decoding to predictions.

        Args:
            trace_seq: (B, L, D) trace sequence
            direction: (B, P) directions
            snp_vert: (B, 2, F, P, P)
            theta_ref: (B, T) reference theta for each sample
            theta_grid: (G, T) grid of target theta values
            batch_size: chunk size for processing theta grid

        Returns:
            dict with key 'values' -> (B, G, P) predictions across theta grid
        """
        device = next(self.parameters()).device
        trace_seq = trace_seq.to(device)
        direction = direction.to(device)
        snp_vert = snp_vert.to(device)
        theta_ref = theta_ref.to(device)
        theta_grid = theta_grid.to(device)

        # Encode reference tokens once
        tokens_ref = self.model.encode_trace_tokens(trace_seq)  # (B, P, M)

        # Initialize modulator if needed
        self._maybe_init_token_modulator(sample_boundary=theta_ref)
        if self.token_modulator is None:
            raise RuntimeError("DeltaTokenModulator is not initialized. Enable token-shift or call during training once.")

        B, P, M = tokens_ref.shape
        G = theta_grid.size(0)
        outputs = []

        # Expand reference thetas to batch dimension
        theta_ref_b = theta_ref

        for start in range(0, G, batch_size):
            end = min(start + batch_size, G)
            thetas_batch = theta_grid[start:end]  # (g, T)
            g = thetas_batch.size(0)

            # Repeat for all B samples
            z_ref_rep = tokens_ref.repeat_interleave(g, dim=0)  # (B*g, P, M)
            theta_ref_rep = theta_ref_b.repeat_interleave(g, dim=0)  # (B*g, T)
            theta_tgt_rep = thetas_batch.repeat(B, 1)  # (B*g, T)

            # Predict tokens and decode
            z_hat = self.token_modulator(z_ref_rep, theta_ref_rep, theta_tgt_rep)
            vals, _, _ = self.model.decode_from_tokens(
                tokens=z_hat,
                direction=direction.repeat_interleave(g, dim=0),
                boundary=theta_tgt_rep,  # use theta as boundary subset for decoding path
                snp_vert=snp_vert.repeat_interleave(g, dim=0),
            )
            vals = vals.view(B, g, P)
            outputs.append(vals)

        values = torch.cat(outputs, dim=1)  # (B, G, P)
        return {"values": values}

    ############################ PRIVATE METHODS ############################

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
            * 'train':   tensors that carry gradients (used for loss)
            * 'eval':    tensors used only for metrics / logging
        """
        # Use the model's internal uncertainty logic (Laplace or MC)
        if stage == "val" and hasattr(self.model, 'predict_with_uncertainty'):
            pred_ew_eval, total_var, _, _, pred_logits_eval = self.model.predict_with_uncertainty(
                item.trace_seq, item.direction, item.boundary, item.snp_vert
            )
            pred_prob_eval = torch.sigmoid(pred_logits_eval)
            pred_logvar_eval = torch.log(total_var + 1e-8)
        else:
            pred_ew_eval = pred_logvar_eval = pred_prob_eval = None

        # Gradient-carrying forward pass (always)
        pred_ew, pred_logvar, pred_logits = self(
            item.trace_seq, item.direction, item.boundary, item.snp_vert
        )
        pred_prob = torch.sigmoid(pred_logits)

        # Fallback to eager outputs for metric display when Laplace not used
        if pred_ew_eval is None:
            pred_ew_eval, pred_logvar_eval, pred_prob_eval = pred_ew, pred_logvar, pred_prob

        return {
            "train":   (pred_ew, pred_logvar, pred_prob),
            "eval":    (pred_ew_eval, pred_logvar_eval, pred_prob_eval)
        }

    def _compute_loss(self, item: "BatchItem", forward_out):
        """Calculate composite loss and prepare everything needed for metric updates."""
        pred_ew, pred_logvar, pred_prob = forward_out["train"]

        # --- classification helpers --------------------------------------------------
        true_prob = (item.true_ew > 0).float()
        weight_prob = torch.where(true_prob == 0, 10.0, 1.0)
        weight_prob = weight_prob / weight_prob.sum(dim=-1, keepdim=True)
        # ------------------------------------------------------------------------------

        if self.hparams.predict_logvar:
            loss = self.weighted_loss({
                'nll': losses.gaussian_nll_loss(pred_ew, item.true_ew, pred_logvar, mask=true_prob),
                'bce': F.binary_cross_entropy_with_logits(pred_prob, true_prob, weight=weight_prob)
            })
        else:
            loss = self.weighted_loss({
                'mse': F.mse_loss(pred_ew[true_prob.bool()], item.true_ew[true_prob.bool()]),
                'bce': F.binary_cross_entropy_with_logits(pred_prob, true_prob, weight=weight_prob)
            })

        # Optional graph Laplacian smoothness regularization across batch
        if (
            self.training
            and getattr(self.hparams, 'graph_smooth_enable', False)
            and getattr(self.hparams, 'graph_smooth_weight', 0.0) > 0.0
        ):
            try:
                # Normalize theta (boundary) across batch to unit variance per feature
                theta = item.boundary
                theta_mean = theta.mean(dim=0, keepdim=True)
                theta_std = theta.std(dim=0, keepdim=True)
                theta_norm = (theta - theta_mean) / (theta_std + 1e-6)

                # Build KNN graph over samples in the batch
                idx, w = knn_graph(
                    theta_norm,
                    k=int(getattr(self.hparams, 'graph_smooth_k', 8)),
                    sigma=float(getattr(self.hparams, 'graph_smooth_sigma', 0.25)),
                )

                # Penalize variation of predictions across neighbors (per-port treated as feature dims)
                graph_pen = graph_laplacian_penalty(pred_ew, idx, w)
                weighted_pen = float(self.hparams.graph_smooth_weight) * graph_pen
                loss = loss + weighted_pen

                # Lightweight logging for monitoring
                self.log(
                    'graph_penalty',
                    graph_pen.detach(),
                    on_step=True,
                    logger=False,
                    prog_bar=False,
                    sync_dist=True,
                )
            except Exception as e:
                # Be robust: if anything goes wrong, skip graph penalty for this step
                rank_zero_info(f"Graph smoothness penalty skipped due to error: {e}")

        # Optional: in-batch theta pairing (pairs are used by downstream tasks/losses)
        theta_pairs = None
        if getattr(self.hparams, 'pair_enable', False):
            try:
                theta_pairs = self._find_theta_pairs(item.boundary)
                if theta_pairs is not None:
                    self.log(
                        'num_theta_pairs',
                        float(theta_pairs[0].numel()),
                        on_step=True,
                        logger=False,
                        prog_bar=False,
                        sync_dist=True,
                    )
            except Exception as e:
                rank_zero_info(f"Theta pairing skipped due to error: {e}")

        # Optional token-shift supervision loss
        if getattr(self.hparams, 'enable_token_shift', False):
            try:
                # Ensure modulator is initialized
                self._maybe_init_token_modulator(sample_boundary=item.boundary)

                if theta_pairs is None:
                    token_loss = pred_ew.new_tensor(0.0)
                    num_pairs = 0
                else:
                    src_idx, tgt_idx = theta_pairs
                    num_pairs = int(src_idx.numel())
                    if num_pairs == 0:
                        token_loss = pred_ew.new_tensor(0.0)
                    else:
                        # Encode tokens for all samples: (B, P, M)
                        tokens_all = self.model.encode_trace_tokens(item.trace_seq)
                        # Gather reference/target tokens by pair indices
                        z_ref = tokens_all.index_select(0, src_idx)
                        z_tgt = tokens_all.index_select(0, tgt_idx)

                        # Extract theta vectors with selected dims
                        theta_idx = self._get_theta_indices()
                        if theta_idx is None:
                            theta_ref = item.boundary.index_select(0, src_idx)[:, -1:].to(dtype=z_ref.dtype)
                            theta_tgt = item.boundary.index_select(0, tgt_idx)[:, -1:].to(dtype=z_ref.dtype)
                        else:
                            theta_ref = item.boundary.index_select(0, src_idx)[:, theta_idx].to(dtype=z_ref.dtype)
                            theta_tgt = item.boundary.index_select(0, tgt_idx)[:, theta_idx].to(dtype=z_ref.dtype)

                        # Predict target tokens
                        z_hat = self.token_modulator(z_ref, theta_ref, theta_tgt)
                        token_loss = F.mse_loss(z_hat, z_tgt)

                # Log token loss and number of pairs
                self.log(
                    'token_loss',
                    token_loss.detach(),
                    on_step=True,
                    logger=False,
                    prog_bar=False,
                    sync_dist=True,
                )
                self.log(
                    'num_token_pairs',
                    float(num_pairs),
                    on_step=True,
                    logger=False,
                    prog_bar=False,
                    sync_dist=True,
                )

                # Add to total loss only during training
                if self.training and self.hparams.lambda_tokens > 0.0:
                    loss = loss + float(self.hparams.lambda_tokens) * token_loss
            except Exception as e:
                rank_zero_info(f"Token-shift loss skipped due to error: {e}")


        # Use the eval tensors (may come from MC/Laplace inference) for metrics
        pred_ew_eval, pred_logvar_eval, pred_prob_eval = forward_out["eval"]

        # Rescale / post-process for logging
        pred_ew_eval = pred_ew_eval * self.ew_scaler
        true_ew_scaled = item.true_ew * self.ew_scaler
        
        # Handle uncertainty prediction based on predict_logvar setting
        if self.hparams.predict_logvar:
            pred_logvar_eval = pred_logvar_eval + 2 * self.log_ew_scaler.to(pred_logvar_eval.device)
            pred_sigma = torch.exp(0.5 * pred_logvar_eval)
        else:
            # When predict_logvar is false, sigma should be zero
            pred_sigma = torch.zeros_like(pred_ew_eval)
        
        # Add boundary parameters to metadata for logging (only if trainer/datamodule available)
        # Avoid touching Lightning's `trainer` property when unattached
        if getattr(self, '_trainer', None) is not None and getattr(self._trainer, 'datamodule', None) is not None:
            fix_scaler = self._trainer.datamodule.fix_scaler
            boundary_reshaped = item.boundary.reshape(-1, item.boundary.shape[-1])
            boundary_inverse = fix_scaler.inverse_transform(boundary_reshaped)
            boundary_inverse[boundary_inverse == fix_scaler.nan] = torch.nan
            boundary_inverse = boundary_inverse.reshape(item.boundary.shape)
            boundary_dicts = []
            for i in range(len(boundary_inverse)):
                boundary_dict = {k: v.item() for k, v in zip(self.config_keys, boundary_inverse[i])}
                boundary_dicts.append(boundary_dict)
            meta = {**item.meta, 'boundary': boundary_dicts}
        else:
            meta = item.meta

        extras = {
            "pred_ew": pred_ew_eval,
            "true_ew": true_ew_scaled,
            "pred_prob": pred_prob_eval,
            "true_prob": true_prob,
            "pred_sigma": pred_sigma,
            "meta": meta,
            "theta_pairs": theta_pairs,
        }
        return loss, extras

    # -------------------------------------------------------------------
    # In-batch theta pairing helpers
    # -------------------------------------------------------------------
    def _get_theta_indices(self) -> torch.Tensor | None:
        keys = getattr(self.hparams, 'pair_theta_keys', None)
        if not keys:
            return None
        # Ensure we have config_keys from datamodule
        if not hasattr(self, 'config_keys'):
            return None
        indices = []
        name_to_idx = {name: i for i, name in enumerate(self.config_keys)}
        for k in keys:
            if k in name_to_idx:
                indices.append(name_to_idx[k])
        if not indices:
            return None
        return torch.tensor(indices, dtype=torch.long)

    def _find_theta_pairs(self, boundary: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
        """
        Identify pairs of samples within a batch that have similar geometry but
        sufficiently different theta values.

        Returns:
            (src_idx, tgt_idx) as LongTensors of equal length, or None if none found.
        """
        B, Fb = boundary.shape
        if B <= 1:
            return None

        theta_idx = self._get_theta_indices()
        device = boundary.device
        dtype = boundary.dtype

        # Geometry dims are those not in theta_idx (if provided)
        if theta_idx is not None:
            all_idx = torch.arange(Fb, device=device)
            mask = torch.ones(Fb, dtype=torch.bool, device=device)
            mask[theta_idx] = False
            geom = boundary[:, mask]
            theta_vec = boundary[:, theta_idx]
        else:
            # Fallback: use full boundary as geometry and theta_vec as last column
            geom = boundary
            theta_vec = boundary[:, -1:].to(dtype=dtype)

        # Normalize geometry features per-dimension
        g_mean = geom.mean(dim=0, keepdim=True)
        g_std = geom.std(dim=0, keepdim=True)
        geom_norm = (geom - g_mean) / (g_std + 1e-6)

        # Build KNN graph over geometry
        k = max(1, int(getattr(self.hparams, 'pair_k', 2)))
        idx, _ = knn_graph(geom_norm, k=k, sigma=0.5)

        # For each i, choose first neighbor j with theta difference >= min_delta
        min_delta = float(getattr(self.hparams, 'pair_min_delta', 0.05))
        src_list: list[int] = []
        tgt_list: list[int] = []

        for i in range(B):
            neigh = idx[i] if idx.dim() == 2 else idx[i]
            for j in neigh.tolist():
                if j < 0 or j >= B or j == i:
                    continue
                # Theta difference criterion
                delta = (theta_vec[i] - theta_vec[j]).abs()
                # If multi-dim theta, use L2 norm
                if delta.numel() > 1:
                    delta_val = torch.linalg.vector_norm(delta, ord=2).item()
                else:
                    delta_val = float(delta.item())
                if delta_val >= min_delta:
                    src_list.append(i)
                    tgt_list.append(j)
                    break

        if not src_list:
            return None
        return (
            torch.tensor(src_list, dtype=torch.long, device=device),
            torch.tensor(tgt_list, dtype=torch.long, device=device),
        )

    # -------------------------------------------------------------------
    # Token modulator initialization helper
    # -------------------------------------------------------------------
    def _maybe_init_token_modulator(self, sample_boundary: torch.Tensor | None = None):
        if not getattr(self.hparams, 'enable_token_shift', False):
            return
        if self.token_modulator is not None:
            return
        # Determine theta dimension
        theta_idx = self._get_theta_indices()
        if theta_idx is not None:
            theta_dim = int(theta_idx.numel())
        else:
            # Default to using the last boundary dimension as theta scalar
            theta_dim = 1

        # Token dimension from model
        token_dim = getattr(self.model, 'model_dim', None)
        if token_dim is None:
            # Fallback: infer by running one step of encode if possible
            return
        hidden = int(self.hparams.token_mod_hidden) if self.hparams.token_mod_hidden else int(token_dim)
        num_f = int(self.hparams.token_mod_num_f)
        self.token_modulator = DeltaTokenModulator(token_dim=token_dim, theta_dim=theta_dim, hidden=hidden, num_f=num_f)

    def _maybe_collect_samples(self, stage: str, name: str, batch_idx: int, extras: dict):
        """Keep a few random samples around for plotting at epoch-end."""
        # Skip when not attached to a Trainer (e.g., unit tests calling step directly)
        if getattr(self, '_trainer', None) is None:
            return
        if batch_idx in self.get_output_steps(stage):
            if stage == "train_":
                self.train_step_outputs.setdefault(name, []).append(extras)
            else:
                self.val_step_outputs.setdefault(name, []).append(extras)

    def get_output_steps(self, stage):
        if stage == "train_":
            max_batches = self.trainer.num_training_batches
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
                    extras["pred_sigma"],
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
        pred_sigma: torch.Tensor,
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
        
        # Pre-compute coverage metrics to avoid redundant calculations
        coverage_metrics = {}
        if self.hparams.predict_logvar and mask.any():
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
                metric.update(pred_prob.flatten(), true_prob.flatten())
            elif 'cov' in key:
                if key in coverage_metrics:
                    in_range_flat, true_mask_flat = coverage_metrics[key]
                    metric.update(in_range_flat, true_mask_flat)
            elif pred_ew_masked.numel() > 0:  # Only update if we have valid masked data
                metric.update(pred_ew_masked, true_ew_masked)

    def compute_metrics(self, stage):
        log_metrics = {}
        for key, metric in self.metrics[stage].items():
            if not self.hparams.predict_logvar and 'cov' in key:
                log_metrics[f'{stage}/{key}'] = float('nan')
            else:
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
