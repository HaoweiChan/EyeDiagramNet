import torch
import torch.nn as nn
from einops import rearrange

from .layers import RMSNorm, positional_encoding_1d, MCDropout, RotaryTransformerEncoder, StructuredGatedBoundaryProcessor
from .snp_model import SNPEmbedding
from .trace_model import TraceSeqTransformer

class EyeWidthRegressor(nn.Module):
    def __init__(
        self,
        num_types,
        model_dim,
        output_dim,
        num_heads,
        num_layers,
        dropout,
        freq_length,
        mc_dropout_rate=0.1,
        use_rope=True,
        max_seq_len=2048,
    ):
        super().__init__()

        self.model_dim = model_dim
        self.output_dim = output_dim
        self.mc_dropout_rate = mc_dropout_rate
        self.use_rope = use_rope

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
        
        self.norm = RMSNorm(model_dim)

        # Direction embedding (0 for Tx, 1 for Rx)
        self.dir_projection = nn.Embedding(2, model_dim)

        # Structured boundary condition processor with CTLE gating
        self.boundary_processor = StructuredGatedBoundaryProcessor(model_dim)
        self.fix_token = nn.Parameter(torch.zeros(1, 1, self.model_dim))

        # SNP encoder
        self.snp_encoder = SNPEmbedding(model_dim=model_dim, freq_length=freq_length)
        
        # Positional encoding (only used if not using RoPE)
        if not use_rope:
            signal_projection = positional_encoding_1d(model_dim, max_len=max_seq_len)
            self.register_buffer('signal_projection', signal_projection)
        else:
            # For RoPE, we don't need explicit positional embeddings
            self.signal_projection = None

        # Monte Carlo Dropout layers for epistemic uncertainty
        self.mc_dropout1 = MCDropout(mc_dropout_rate)
        self.mc_dropout2 = MCDropout(mc_dropout_rate)

        # Prediction heads with MC Dropout
        self.pred_head = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            self.mc_dropout1,
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
            self.mc_dropout2,
            nn.GELU(),
            nn.Linear(model_dim, output_dim),
        )

        # Logit conversion
        # self.logit_threshold = nn.Parameter(torch.tensor(0.0))
        # self.threshold = nn.Parameter(torch.tensor(0.0))
        # self.slope = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        trace_seq: torch.Tensor,
        direction: torch.Tensor,
        boundary: torch.Tensor,
        snp_vert: torch.Tensor,
        output_hidden_states: bool = False
    ):
        """
        Predict frequency response of the given trace inputs and query frequencies

        Args:
            trace_seq (torch.Tensor): Input trace sequences of shape (B, L, D), where B is the batch size, L is the sequence length, and D is the feature dimension.
            direction (torch.Tensor): Direction inputs of shape (B, P), specifying the Tx/Rx directions of the signal traces.
            boundary (torch.Tensor): Selected port indices of the signal traces of shape (B, P), where B is the batch size and P is the number of ports.
            snp_vert (torch.Tensor): Vertical S-parameter inputs of shape (B, 2, F, P, P), where Tx and Rx vertical S-parameter information is stacked at dimension 1.
            output_hidden_states (bool, optional): Whether to output shared hidden states. Defaults to False.

        Returns:
            values (torch.Tensor): Predicted eye width averages of shape (B, P), where B is the batch size and P is the number of ports.
            log_var (torch.Tensor): Predicted eye width sigmas of shape (B, P).
            logits (torch.Tensor): Predicted open-eye probability logits (real values, before sigmoid) of shape (B, P).
            hidden_states_sig (torch.Tensor, optional): Hidden states of shared embedding before output head of shape (B, P, D), only returned if output_hidden_states is True.
        """
        # Process trace sequence
        hidden_states_seq = self.trace_encoder(trace_seq)  # (B, P, M)

        # Process boundary conditions with structured processor
        hidden_states_fix = self.boundary_processor(boundary).unsqueeze(1) # (B, 1, M)

        # Process snp into hidden states
        hidden_states_vert = self.snp_encoder(snp_vert) # (B, D, P, M)

        # Process direction embedding
        hidden_states_dir = self.dir_projection(direction) # (B, P, M)

        # Sum all hidden states and forward to the signal sequence decoder
        hidden_states_seq = hidden_states_seq + hidden_states_dir
        hidden_states_seq = self.norm(hidden_states_seq) # (B, P, M)

        num_signals = hidden_states_seq.size(1)
        
        # Add positional embeddings only if not using RoPE
        if not self.use_rope and self.signal_projection is not None:
            signal_embeds = self.signal_projection[:num_signals].unsqueeze(0) # (B, L, M)
            hidden_states_seq = hidden_states_seq + signal_embeds
            hidden_states_vert = hidden_states_vert + signal_embeds.unsqueeze(0)
        
        hidden_states_vert = rearrange(hidden_states_vert, "b d p e -> b (d p) e") # concat tx and rx snp states
        hidden_states = torch.cat([hidden_states_seq, hidden_states_vert, hidden_states_fix + self.fix_token], dim=1)

        # Run transformer for the signals
        hidden_states_sig = self.signal_encoder(hidden_states)
        hidden_states_sig = hidden_states[:, :num_signals] + hidden_states_fix

        # Predict eye width and open eye probabilities respectively
        output = self.pred_head(hidden_states_sig)
        values, log_var, logits = torch.unbind(output, dim=-1)
        # values, log_var = torch.unbind(output, dim=-1)
        # logits = (logits - self.threshold) * self.slope
        # logits = logits - self.logit_threshold

        if output_hidden_states:
            return values, log_var, logits, hidden_states_sig
        return values, log_var, logits

    def predict_with_uncertainty(
        self,
        trace_seq: torch.Tensor,
        direction: torch.Tensor,
        boundary: torch.Tensor,
        snp_vert: torch.Tensor,
        n_samples: int = 50
    ):
        """
        Perform Monte Carlo inference to estimate both aleatoric and epistemic uncertainty
        
        Args:
            trace_seq, direction, boundary, snp_vert: Input tensors
            n_samples: Number of MC samples for epistemic uncertainty estimation
            
        Returns:
            mean_values: Mean predictions across MC samples
            total_var: Total uncertainty (aleatoric + epistemic)
            aleatoric_var: Data uncertainty only
            epistemic_var: Model uncertainty only
            mean_logits: Mean logits for probability prediction
        """
        self.train()  # Enable dropout for MC sampling
        
        predictions = []
        log_vars = []
        logits_list = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                values, log_var, logits = self(trace_seq, direction, boundary, snp_vert)
                predictions.append(values)
                log_vars.append(log_var)
                logits_list.append(logits)
        
        # Stack predictions
        predictions = torch.stack(predictions, dim=0)  # (n_samples, B, P)
        log_vars = torch.stack(log_vars, dim=0)  # (n_samples, B, P)
        logits_list = torch.stack(logits_list, dim=0)  # (n_samples, B, P)
        
        # Compute statistics
        mean_values = predictions.mean(dim=0)  # (B, P)
        epistemic_var = predictions.var(dim=0)  # (B, P) - model uncertainty
        mean_log_var = log_vars.mean(dim=0)  # (B, P)
        aleatoric_var = torch.exp(mean_log_var)  # (B, P) - data uncertainty
        total_var = epistemic_var + aleatoric_var  # Total uncertainty
        mean_logits = logits_list.mean(dim=0)  # (B, P)
        
        return mean_values, total_var, aleatoric_var, epistemic_var, mean_logits