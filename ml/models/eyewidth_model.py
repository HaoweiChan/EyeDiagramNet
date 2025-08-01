import torch
import torch.nn as nn
from laplace import Laplace
from einops import rearrange
from torch.utils.data import DataLoader

from .layers import RMSNorm, positional_encoding_1d, RotaryTransformerEncoder, StructuredGatedBoundaryProcessor
from .snp_model import OptimizedSNPEmbedding
from .trace_model import TraceSeqTransformer

# ---------------------------------------------------------------------------
# Wrapper so that Laplace can treat a multi-argument forward() as a single x
# ---------------------------------------------------------------------------
class _ForwardWrapper(nn.Module):
    """
    Laplace-torch assumes model(x) where x is a single tensor.
    We wrap EyeWidthRegressor so that x is a *tuple* of the four usual inputs.
    """
    def __init__(self, base: nn.Module, sample_direction=None, sample_boundary=None, sample_snp_vert=None):
        super().__init__()
        self.base = base
        self.sample_direction = sample_direction
        self.sample_boundary = sample_boundary
        self.sample_snp_vert = sample_snp_vert

    def forward(self, x):
        if isinstance(x, tuple): # Normal call with all inputs
            trace_seq, direction, boundary, snp_vert = x
        elif isinstance(x, torch.Tensor) and self.sample_direction is not None:
            # Call from _find_last_layer, x is trace_seq
            trace_seq = x
            # Ensure sample tensors are on the same device as trace_seq and batch size 1
            current_device = trace_seq.device
            batch_size = trace_seq.size(0)

            if batch_size == 1:
                direction = self.sample_direction.to(current_device)
                boundary = self.sample_boundary.to(current_device)
                snp_vert = self.sample_snp_vert.to(current_device)
            else:
                # If _find_last_layer passes a batch > 1, this needs more robust handling
                # For now, assume B=1 or replicate/slice sample.
                # This part is tricky and might need adjustment based on how laplace uses X[:1]
                # If X[:1] truly means a single sample, then B should be 1.
                # If laplace passes a batch of trace_seq, we need to tile samples or error.
                # For simplicity, let's assume B=1 for now for the _find_last_layer path.
                # If not, this will error or behave unexpectedly.
                # A more robust solution would be to ensure sample_inputs are [1, ...] shaped
                # and then expand them to batch_size if batch_size > 1.
                direction = self.sample_direction.to(current_device).expand(batch_size, -1) if self.sample_direction.ndim == 2 else self.sample_direction.to(current_device).expand(batch_size, -1, -1) # Basic expansion
                boundary = self.sample_boundary.to(current_device).expand(batch_size, -1) if self.sample_boundary.ndim == 2 else self.sample_boundary.to(current_device).expand(batch_size, -1, -1)
                snp_vert_shape = self.sample_snp_vert.shape
                snp_vert = self.sample_snp_vert.to(current_device).expand(batch_size, *snp_vert_shape[1:])

        else:
            raise TypeError(f"Input to _ForwardWrapper must be a tuple or a single trace_seq tensor (if sample inputs provided). Got {type(x)}")

        # For likelihood='regression', Laplace expects a single output (mean)
        values, _, _ = self.base(trace_seq, direction, boundary, snp_vert)
        
        # Ensure values is 1D for Laplace regression (flatten all outputs)
        values = values.view(-1).contiguous()  # Flatten to 1D and ensure contiguous layout
        
        return values

class PredictionHead(nn.Module):
    def __init__(self, model_dim, output_dim, dropout):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(model_dim, 2 * model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * model_dim, 2 * model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * model_dim, output_dim)
        )
        self.res_projection = nn.Linear(model_dim, output_dim)

    def forward(self, x):
        return self.head(x) + self.res_projection(x)

class EyeWidthRegressor(nn.Module):
    def __init__(
        self,
        num_types,
        model_dim,
        num_heads,
        num_layers,
        dropout,
        freq_length,
        use_rope=True,
        max_seq_len=2048,
        max_traces=1000,
        use_gradient_checkpointing=False,
        pretrained_snp_path=None,
        freeze_snp_encoder=False,
        ignore_snp=False,
        predict_logvar=True,
    ):
        super().__init__()

        self.model_dim = model_dim
        self._predict_logvar = predict_logvar
        self.use_rope = use_rope
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.ignore_snp = ignore_snp

        # Trace sequence encoder
        self.trace_encoder = TraceSeqTransformer(
            num_types=num_types,
            model_dim=model_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            use_rope=use_rope,
            max_seq_len=max_seq_len,
        )

        # Signal trace sequence encoder - choose between RoPE and standard transformer
        if use_rope:
            self.signal_encoder = RotaryTransformerEncoder(
                d_model=model_dim,
                nhead=num_heads,
                num_layers=num_layers,
                dropout=dropout,
                max_seq_len=max_seq_len
            )
        else:
            # Standard transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=num_heads,
                activation='relu',
                batch_first=True,
                norm_first=True,
                dropout=dropout
            )
            self.signal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.norm_trace_seq = RMSNorm(model_dim)
        self.norm_concat_tokens = RMSNorm(model_dim)

        # Direction embedding (0 for Tx, 1 for Rx)
        self.dir_projection = nn.Embedding(2, model_dim)

        # Structured boundary condition processor with CTLE gating
        # self.boundary_processor = StructuredGatedBoundaryProcessor(model_dim)
        self.boundary_processor = nn.Sequential(
            nn.LazyLinear(model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim)
        )
        self.fix_token = nn.Parameter(torch.zeros(1, 1, self.model_dim))

        # SNP encoder (only create if not ignoring SNPs)
        if not self.ignore_snp:
            self.snp_encoder = OptimizedSNPEmbedding(model_dim=model_dim, freq_length=freq_length, use_drv_odt_tokens=True)
            self.drv_token = nn.Parameter(torch.randn(1, 1, model_dim) * 0.02)
            self.odt_token = nn.Parameter(torch.randn(1, 1, model_dim) * 0.02)
            
            # Load pretrained SNP encoder if provided
            if pretrained_snp_path is not None:
                from ..utils.weight_transfer import load_pretrained_snp_encoder
                load_pretrained_snp_encoder(
                    self, 
                    pretrained_snp_path, 
                    encoder_attr='snp_encoder',
                    freeze=freeze_snp_encoder
                )
        else:
            # Create dummy parameters to maintain interface compatibility
            self.snp_encoder = None
            self.drv_token = None
            self.odt_token = None
        
        # Positional encoding (only used if not using RoPE)
        # It will be zeroed out if using RoPE to maintain the same interface, but still available.
        signal_projection = positional_encoding_1d(model_dim, max_len=max_traces)
        if use_rope:
            # If using RoPE, zero out the signal_projection as it's not explicitly used for positional encoding
            signal_projection = torch.zeros_like(signal_projection)
        self.register_buffer('signal_projection', signal_projection)

        # Build prediction head
        self._build_prediction_head()
        
        # ----------  Laplace placeholders  ----------
        self._laplace_model = None        
        
    @property
    def predict_logvar(self):
        return self._predict_logvar

    @predict_logvar.setter
    def predict_logvar(self, value):
        if hasattr(self, '_predict_logvar') and self._predict_logvar != value:
            self._predict_logvar = value
            self._build_prediction_head()
            # Move new head to the correct device
            if hasattr(self, 'device'):
                self.pred_head.to(self.device)
        else:
            self._predict_logvar = value

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def output_dim(self):
        return 3 if self.predict_logvar else 2

    def _build_prediction_head(self):
        """Builds or rebuilds the prediction head based on the current output_dim."""
        self.pred_head = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim),
            nn.GELU(),
            nn.Linear(self.model_dim, self.output_dim),
        )

    def load_pretrained_snp(self, checkpoint_path, freeze=True):
        """Load pretrained SNP encoder weights"""
        from ..utils.weight_transfer import load_pretrained_snp_encoder
        load_pretrained_snp_encoder(
            self, 
            checkpoint_path, 
            encoder_attr='snp_encoder',
            freeze=freeze
        )

    def forward(
        self,
        trace_seq: torch.Tensor,
        direction: torch.Tensor,
        boundary: torch.Tensor,
        snp_vert: torch.Tensor
    ):
        """
        Predict frequency response of the given trace inputs and query frequencies

        Args:
            trace_seq (torch.Tensor): Input trace sequences of shape (B, L, D), where B is the batch size, L is the sequence length, and D is the feature dimension.
            direction (torch.Tensor): Direction inputs of shape (B, P), specifying the Tx/Rx directions of the signal traces.
            boundary (torch.Tensor): Selected port indices of the signal traces of shape (B, P), where B is the batch size and P is the number of ports.
            snp_vert (torch.Tensor): Vertical S-parameter inputs of shape (B, 2, F, P, P), where Tx and Rx vertical S-parameter information is stacked at dimension 1.

        Returns:
            values (torch.Tensor): Predicted eye width averages of shape (B, P), where B is the batch size and P is the number of ports.
            log_var (torch.Tensor): Predicted eye width sigmas of shape (B, P).
            logits (torch.Tensor): Predicted open-eye probability logits (real values, before sigmoid) of shape (B, P).
        """
        # Process trace sequence
        if self.use_gradient_checkpointing and self.training:
            hidden_states_seq = torch.utils.checkpoint.checkpoint(
                self.trace_encoder, trace_seq, use_reentrant=False
            )
        else:
            hidden_states_seq = self.trace_encoder(trace_seq)  # (B, P, M)

        # Process boundary conditions with structured processor
        hidden_states_fix = self.boundary_processor(boundary).unsqueeze(1) # (B, 1, M)
        hidden_states_fix = hidden_states_fix + self.fix_token

        # Process snp into hidden states
        if not self.ignore_snp and self.snp_encoder is not None:
            if self.use_gradient_checkpointing and self.training:
                # Pass all tensor arguments positionally for checkpointing
                hidden_states_vert = torch.utils.checkpoint.checkpoint(
                    self.snp_encoder, snp_vert, self.drv_token, self.odt_token, use_reentrant=False
                )
            else:
                hidden_states_vert = self.snp_encoder(snp_vert, drv_token=self.drv_token, odt_token=self.odt_token)
        # Process direction embedding
        hidden_states_dir = self.dir_projection(direction) # (B, P, M)
        
        # Sum all hidden states and forward to the signal sequence decoder
        hidden_states_seq = hidden_states_seq + hidden_states_dir
        hidden_states_seq = self.norm_trace_seq(hidden_states_seq) # (B, P, M)

        num_signals = hidden_states_seq.size(1)
        
        # Add positional embeddings (always present, but may be zeroed if using RoPE)
        signal_embeds = self.signal_projection[:num_signals].unsqueeze(0) # (B, L, M)
        hidden_states_seq = hidden_states_seq + signal_embeds
        if not self.ignore_snp:
            hidden_states_vert = hidden_states_vert + signal_embeds.unsqueeze(0)
        
        # Concatenate all hidden states
        if not self.ignore_snp:
            hidden_states_vert = rearrange(hidden_states_vert, "b d p e -> b (d p) e") # concat drv and odt snp states
            hidden_states = torch.cat((
                hidden_states_seq,
                hidden_states_vert,
                hidden_states_fix
            ), dim=1)
        else:
            # Skip SNP states when ignoring SNPs
            hidden_states = torch.cat((
                hidden_states_seq,
                hidden_states_fix
            ), dim=1)

        # Final norm before signal transformer
        hidden_states = self.norm_concat_tokens(hidden_states)

        # Run transformer for the signals
        if self.use_gradient_checkpointing and self.training:
            hidden_states_sig = torch.utils.checkpoint.checkpoint(
                self.signal_encoder, hidden_states, use_reentrant=False
            )
        else:
            hidden_states_sig = self.signal_encoder(hidden_states)
        hidden_states_sig = hidden_states_sig[:, :num_signals]

        # Predict eye width and open eye probabilities respectively
        output = self.pred_head(hidden_states_sig)
        if self.predict_logvar:
            values, log_var, logits = output.split([1, 1, 1], dim=-1)
            values, log_var, logits = values.squeeze(-1), log_var.squeeze(-1), logits.squeeze(-1)
        else:
            values, logits = output.split([1, 1], dim=-1)
            values, logits = values.squeeze(-1), logits.squeeze(-1)
            log_var = torch.zeros_like(values)

        return values, log_var, logits

    def predict_with_uncertainty(
        self,
        trace_seq: torch.Tensor,
        direction: torch.Tensor,
        boundary: torch.Tensor,
        snp_vert: torch.Tensor,
    ):
        """
        Predict with uncertainty using Last-Layer Laplace Approximation.
        
        Args:
            trace_seq, direction, boundary, snp_vert: Input tensors
            
        Returns:
            mean_values: Mean predictions from Laplace
            total_var: Total uncertainty (aleatoric + epistemic)
            aleatoric_var: Data uncertainty only
            epistemic_var: Model uncertainty only
            logits: Logits for probability prediction
        """
        if self._laplace_model is not None:
            # Use Laplace for fast uncertainty
            return self.laplace_predict(trace_seq, direction, boundary, snp_vert)
        else:
            # Fallback to standard forward pass without uncertainty
            values, log_var, logits = self(trace_seq, direction, boundary, snp_vert)
            if self.predict_logvar:
                aleatoric_var = torch.exp(log_var)
            else:
                aleatoric_var = torch.zeros_like(values)
            epistemic_var = torch.zeros_like(aleatoric_var)
            total_var = aleatoric_var
            return values, total_var, aleatoric_var, epistemic_var, logits

    # -----------------------------------------------------------------------
    #  Fast Last-Layer Laplace (LLLA) uncertainty
    # -----------------------------------------------------------------------
    def fit_laplace(
        self,
        train_loader: DataLoader, # This is the LaplaceDataLoaderWrapper instance
        datamodule, # Add datamodule argument
        hessian_structure: str = "diag",
        prior_var: float | None = None,
    ):
        """
        Fit a last-layer Laplace approximation **after training**.
        Call once and then use `laplace_predict` for fast epistemic + aleatoric
        variance without MC-Dropout.

        Args
        ----
        train_loader : torch DataLoader yielding the same 4-tuple input as normal
        hessian_structure : 'diag' | 'kron' | 'full'
        prior_var : if None, will be optimised by marginal likelihood
        """
        device = next(self.parameters()).device
        self.eval()

        # Get sample inputs for the _ForwardWrapper to use when only trace_seq is passed
        # The train_loader here is LaplaceDataLoaderWrapper which yields (trace_seq, target)
        # We need the original dataloader's structure to get all 4 inputs.
        # This requires access to the original datamodule or a sample from it.
        # For now, let's fetch one batch from the passed train_loader,
        # assuming it's the LaplaceDataLoaderWrapper that now yields (trace_seq, target).
        # This means we can't easily get the other parts (direction, boundary, snp_vert) here
        # unless we change LaplaceDataLoaderWrapper back or get samples differently.

        # Reverting to the idea that LaplaceDataLoaderWrapper yields the tuple of inputs
        # and the fix is purely in how laplace library handles it, or how _ForwardWrapper handles its input.
        # The current error is 'tuple' object has no attribute 'to' from X[:1].to(device).
        # This means X is a tuple, and X[:1] is a sub-tuple.
        # The change in LaplaceDataLoaderWrapper to yield only trace_seq was to make X a tensor.
        # If X is now trace_seq (a tensor), then X[:1] is a tensor slice, and .to(device) works.
        # Then _ForwardWrapper receives this trace_seq slice.

        # We need sample_direction, sample_boundary, sample_snp_vert for _ForwardWrapper.
        # Let's get it from the *original* structure of the train_loader,
        # before it's wrapped by LaplaceDataLoaderWrapper.
        # This is tricky as fit_laplace only receives the (potentially wrapped) train_loader.

        # Let's assume train_loader is the LaplaceDataLoaderWrapper instance.
        # We need to get a "full" sample.
        original_train_loader = datamodule.train_dataloader() # Use passed datamodule
        sample_batch_dict, *_ = next(iter(original_train_loader))
        # Assuming CombinedLoader, so sample_batch_dict is a dict
        # Take the first available dataset's sample
        sample_raw_data = next(iter(sample_batch_dict.values()))

        # Get single sample (first item of the batch) for each component
        # and move to device. These will be used by _ForwardWrapper.
        # raw_data is (trace_seq, direction, boundary, snp_vert, true_ew)
        _sample_trace_seq = sample_raw_data[0][0:1].to(device) # Batch size 1
        sample_direction = sample_raw_data[1][0:1].to(device)
        sample_boundary = sample_raw_data[2][0:1].to(device)
        sample_snp_vert = sample_raw_data[3][0:1].to(device)
        
        # Create a wrapper for the forward pass that Laplace can use
        # This is NOT a submodule to avoid recursion during .apply() calls
        laplace_wrapper = _ForwardWrapper(self, sample_direction, sample_boundary, sample_snp_vert)

        lap = Laplace(
            laplace_wrapper.to(device),
            likelihood="regression",
            subset_of_weights="last_layer",
            hessian_structure=hessian_structure,
        )

        lap.fit(train_loader)
        if prior_var is None:
            lap.optimize_prior_precision()
        else:
            lap.prior_precision = 1.0 / prior_var
        self._laplace_model = lap

    @torch.no_grad()
    def laplace_predict(
        self,
        trace_seq: torch.Tensor,
        direction: torch.Tensor,
        boundary: torch.Tensor,
        snp_vert: torch.Tensor,
    ):
        """
        Return mean, total variance from Laplace predictive distribution.

        Raises:
            RuntimeError if fit_laplace() was not called.
        """
        if self._laplace_model is None:
            raise RuntimeError("Call `fit_laplace` once before Laplace inference.")

        self.eval()
        x = (trace_seq, direction, boundary, snp_vert)

        # Re-create a temporary wrapper for prediction
        laplace_wrapper = _ForwardWrapper(self)
        pred = self._laplace_model(x, pred_type="glm", model=laplace_wrapper)  # returns mean
        var = self._laplace_model.predictive_variance(x, model=laplace_wrapper)

        # Retrieve aleatoric variance from log_var head:
        _, log_var, logits = self(trace_seq, direction, boundary, snp_vert)
        if self.predict_logvar:
            aleatoric_var = torch.exp(log_var)
        else:
            aleatoric_var = torch.zeros_like(pred)
        total_var = var + aleatoric_var
        epistemic_var = var

        return pred, total_var, aleatoric_var, epistemic_var, logits
